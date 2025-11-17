"""
Graph Neural Networks (GNN) for CAD Structural Analysis
شبکه‌های عصبی گرافی برای تحلیل ساختار نقشه‌های CAD

قابلیت‌ها:
- مدل‌سازی روابط بین المان‌ها (دیوار-ستون، دیوار-در، ...)
- تشخیص ساختار مهندسی (فاصله‌ها، اتصالات، محدودیت‌ها)
- ایده‌آل برای پروژه‌های BIM و تحلیل سازه‌ای
- مشابه Revit Constraints و Autodesk AI
"""

from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass
from enum import Enum
import json

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor
    import numpy as np
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    Tensor = Any
    nn = None
    print("⚠️ PyTorch not available for GNN")


class EdgeType(Enum):
    """انواع یال در گراف CAD"""
    CONNECTED = "connected"  # متصل فیزیکی
    ADJACENT = "adjacent"  # مجاور
    PARALLEL = "parallel"  # موازی
    PERPENDICULAR = "perpendicular"  # عمود
    SUPPORTS = "supports"  # پشتیبانی (ستون-تیر)
    CONTAINS = "contains"  # شامل (اتاق-در)
    ALIGNED = "aligned"  # همترازی
    DISTANCE = "distance"  # با فاصله مشخص


@dataclass
class CADNode:
    """یک گره در گراف CAD"""
    node_id: int
    element_type: str  # 'wall', 'column', 'door', ...
    position: Tuple[float, float, float]  # x, y, z
    dimensions: Optional[Tuple[float, float, float]] = None
    layer: Optional[str] = None
    properties: Optional[Dict[str, Any]] = None
    features: Optional[List[float]] = None  # feature vector


@dataclass
class CADEdge:
    """یک یال در گراف CAD"""
    source_id: int
    target_id: int
    edge_type: EdgeType
    weight: float = 1.0  # وزن یال
    distance: Optional[float] = None
    properties: Optional[Dict[str, Any]] = None


@dataclass
class CADGraph:
    """گراف کامل نقشه CAD"""
    nodes: List[CADNode]
    edges: List[CADEdge]
    metadata: Optional[Dict[str, Any]] = None


class GraphConvolution(nn.Module if TORCH_AVAILABLE else object):
    """لایه Graph Convolution"""
    
    def __init__(self, in_features: int, out_features: int):
        if not TORCH_AVAILABLE:
            return
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.activation = nn.ReLU()
    
    def forward(
        self,
        node_features: Tensor,
        adjacency_matrix: Tensor
    ) -> Tensor:
        """
        Args:
            node_features: (num_nodes, in_features)
            adjacency_matrix: (num_nodes, num_nodes)
        Returns:
            (num_nodes, out_features)
        """
        # محاسبه degree matrix
        degree = adjacency_matrix.sum(dim=1, keepdim=True) + 1e-6
        
        # Normalization
        adjacency_norm = adjacency_matrix / degree
        
        # Message passing
        aggregated = torch.matmul(adjacency_norm, node_features)
        
        # Transform
        output = self.linear(aggregated)
        output = self.activation(output)
        
        return output


class EdgeConvolution(nn.Module if TORCH_AVAILABLE else object):
    """لایه Edge Convolution برای پردازش یال‌ها"""
    
    def __init__(self, in_features: int, out_features: int):
        if not TORCH_AVAILABLE:
            return
        super().__init__()
        # MLP برای پردازش (node_i, node_j, edge_features)
        self.mlp = nn.Sequential(
            nn.Linear(in_features * 2 + 8, out_features),  # +8 for edge features
            nn.ReLU(),
            nn.Linear(out_features, out_features),
            nn.ReLU()
        )
    
    def forward(
        self,
        node_features: Tensor,
        edge_index: Tensor,
        edge_features: Tensor
    ) -> Tensor:
        """
        Args:
            node_features: (num_nodes, in_features)
            edge_index: (2, num_edges) - [source_indices, target_indices]
            edge_features: (num_edges, edge_feature_dim)
        Returns:
            (num_nodes, out_features)
        """
        source_nodes = edge_index[0]
        target_nodes = edge_index[1]
        
        # دریافت features گره‌های مبدا و مقصد
        source_features = node_features[source_nodes]
        target_features = node_features[target_nodes]
        
        # ترکیب features
        combined = torch.cat([source_features, target_features, edge_features], dim=1)
        
        # MLP
        edge_messages = self.mlp(combined)
        
        # Aggregation (sum over edges for each node)
        num_nodes = node_features.size(0)
        output = torch.zeros(num_nodes, edge_messages.size(1), device=node_features.device)
        output.index_add_(0, target_nodes, edge_messages)
        
        return output


class CADGraphNeuralNetwork(nn.Module if TORCH_AVAILABLE else object):
    """
    Graph Neural Network برای تحلیل نقشه‌های CAD
    """
    
    def __init__(
        self,
        node_feature_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 4,
        num_classes: int = 15,
        num_edge_types: int = 8
    ):
        if not TORCH_AVAILABLE:
            return
        super().__init__()
        
        # Node feature encoding
        self.node_encoder = nn.Sequential(
            nn.Linear(node_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Graph Convolution Layers
        self.graph_convs = nn.ModuleList([
            GraphConvolution(hidden_dim, hidden_dim)
            for _ in range(num_layers)
        ])
        
        # Edge Convolution Layers
        self.edge_convs = nn.ModuleList([
            EdgeConvolution(hidden_dim, hidden_dim)
            for _ in range(num_layers)
        ])
        
        # Layer Normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        
        # Node classification head
        self.node_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # Edge classification head
        self.edge_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_edge_types)
        )
        
        # Graph-level prediction (برای پیش‌بینی کلی)
        self.graph_pooling = nn.Linear(hidden_dim, hidden_dim)
        self.graph_classifier = nn.Linear(hidden_dim, 10)  # 10 graph-level classes
    
    def forward(
        self,
        node_features: Tensor,
        adjacency_matrix: Tensor,
        edge_index: Tensor,
        edge_features: Tensor
    ) -> Dict[str, Tensor]:
        """
        Args:
            node_features: (num_nodes, node_feature_dim)
            adjacency_matrix: (num_nodes, num_nodes)
            edge_index: (2, num_edges)
            edge_features: (num_edges, edge_feature_dim)
        Returns:
            Dict with predictions
        """
        # Encode node features
        x = self.node_encoder(node_features)
        
        # Message passing through layers
        for i, (graph_conv, edge_conv, layer_norm) in enumerate(
            zip(self.graph_convs, self.edge_convs, self.layer_norms)
        ):
            # Graph convolution
            x_graph = graph_conv(x, adjacency_matrix)
            
            # Edge convolution
            x_edge = edge_conv(x, edge_index, edge_features)
            
            # Combine and normalize
            x = x_graph + x_edge
            x = layer_norm(x)
            
            # Residual connection (skip first layer)
            if i > 0:
                x = x + node_features if i == 1 else x
        
        # Node-level predictions
        node_logits = self.node_classifier(x)
        
        # Edge-level predictions
        source_features = x[edge_index[0]]
        target_features = x[edge_index[1]]
        edge_combined = torch.cat([source_features, target_features], dim=1)
        edge_logits = self.edge_classifier(edge_combined)
        
        # Graph-level prediction (global pooling)
        graph_embedding = torch.mean(x, dim=0)
        graph_embedding = self.graph_pooling(graph_embedding)
        graph_logits = self.graph_classifier(graph_embedding)
        
        return {
            'node_logits': node_logits,
            'edge_logits': edge_logits,
            'graph_logits': graph_logits,
            'node_embeddings': x
        }


class CADGraphBuilder:
    """ساخت گراف از نقشه CAD"""
    
    def __init__(self, distance_threshold: float = 1000.0):
        """
        Args:
            distance_threshold: حداکثر فاصله برای ایجاد یال
        """
        self.distance_threshold = distance_threshold
    
    def build_graph_from_dxf(self, dxf_path: str) -> CADGraph:
        """ساخت گراف از فایل DXF"""
        import ezdxf
        
        doc = ezdxf.readfile(dxf_path)
        msp = doc.modelspace()
        
        nodes = []
        node_id = 0
        
        # استخراج گره‌ها
        # دیوارها (LWPOLYLINE)
        for polyline in msp.query('LWPOLYLINE'):
            points = list(polyline.get_points('xy'))
            if len(points) >= 2:
                centroid = self._calculate_centroid(points)
                node = CADNode(
                    node_id=node_id,
                    element_type='wall',
                    position=(*centroid, 0),
                    layer=polyline.dxf.layer,
                    properties={'num_points': len(points)}
                )
                nodes.append(node)
                node_id += 1
        
        # ستون‌ها (CIRCLE)
        for circle in msp.query('CIRCLE'):
            if 'COLUMN' in circle.dxf.layer.upper():
                pos = (circle.dxf.center.x, circle.dxf.center.y, 0)
                node = CADNode(
                    node_id=node_id,
                    element_type='column',
                    position=pos,
                    dimensions=(circle.dxf.radius * 2, circle.dxf.radius * 2, 0),
                    layer=circle.dxf.layer
                )
                nodes.append(node)
                node_id += 1
        
        # ساخت یال‌ها
        edges = self._build_edges(nodes)
        
        return CADGraph(nodes=nodes, edges=edges)
    
    def _calculate_centroid(self, points: List[Tuple[float, float]]) -> Tuple[float, float]:
        """محاسبه مرکز چندضلعی"""
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        return (sum(xs) / len(xs), sum(ys) / len(ys))
    
    def _build_edges(self, nodes: List[CADNode]) -> List[CADEdge]:
        """ساخت یال‌ها بر اساس فاصله و نوع"""
        edges = []
        
        for i, node1 in enumerate(nodes):
            for j, node2 in enumerate(nodes[i+1:], start=i+1):
                distance = self._distance_3d(node1.position, node2.position)
                
                if distance < self.distance_threshold:
                    # تعیین نوع یال
                    edge_type = self._determine_edge_type(node1, node2, distance)
                    
                    edge = CADEdge(
                        source_id=node1.node_id,
                        target_id=node2.node_id,
                        edge_type=edge_type,
                        distance=distance,
                        weight=1.0 / (distance + 1)  # وزن معکوس فاصله
                    )
                    edges.append(edge)
        
        return edges
    
    def _distance_3d(self, p1: Tuple[float, float, float], p2: Tuple[float, float, float]) -> float:
        """محاسبه فاصله سه‌بعدی"""
        import math
        return math.sqrt(
            (p2[0] - p1[0])**2 +
            (p2[1] - p1[1])**2 +
            (p2[2] - p1[2])**2
        )
    
    def _determine_edge_type(self, node1: CADNode, node2: CADNode, distance: float) -> EdgeType:
        """تعیین نوع یال"""
        # ستون + دیوار = پشتیبانی
        if (node1.element_type == 'column' and node2.element_type == 'wall') or \
           (node1.element_type == 'wall' and node2.element_type == 'column'):
            return EdgeType.SUPPORTS
        
        # دیوار + دیوار = متصل یا مجاور
        if node1.element_type == 'wall' and node2.element_type == 'wall':
            if distance < 100:  # خیلی نزدیک
                return EdgeType.CONNECTED
            else:
                return EdgeType.ADJACENT
        
        # پیش‌فرض
        return EdgeType.DISTANCE
    
    def to_torch_data(self, graph: CADGraph) -> Dict[str, Tensor]:
        """تبدیل گراف به format PyTorch"""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required")
        
        # Node features (position + type encoding)
        node_features = []
        for node in graph.nodes:
            # One-hot encoding برای نوع
            type_encoding = [0] * 10
            type_map = {'wall': 0, 'column': 1, 'door': 2, 'window': 3}
            type_idx = type_map.get(node.element_type, 9)
            type_encoding[type_idx] = 1
            
            # Position
            features = list(node.position) + type_encoding
            node_features.append(features)
        
        node_features = torch.tensor(node_features, dtype=torch.float32)
        
        # Adjacency matrix
        num_nodes = len(graph.nodes)
        adjacency = torch.zeros(num_nodes, num_nodes)
        
        # Edge index و features
        edge_sources = []
        edge_targets = []
        edge_features_list = []
        
        for edge in graph.edges:
            edge_sources.append(edge.source_id)
            edge_targets.append(edge.target_id)
            
            # Edge features (type one-hot + distance + weight)
            edge_type_encoding = [0] * len(EdgeType)
            edge_type_encoding[list(EdgeType).index(edge.edge_type)] = 1
            edge_feat = edge_type_encoding + [edge.distance or 0, edge.weight]
            edge_features_list.append(edge_feat)
            
            # Fill adjacency
            adjacency[edge.source_id, edge.target_id] = edge.weight
            adjacency[edge.target_id, edge.source_id] = edge.weight  # undirected
        
        edge_index = torch.tensor([edge_sources, edge_targets], dtype=torch.long)
        edge_features = torch.tensor(edge_features_list, dtype=torch.float32)
        
        return {
            'node_features': node_features,
            'adjacency_matrix': adjacency,
            'edge_index': edge_index,
            'edge_features': edge_features
        }


# مثال استفاده
if __name__ == "__main__":
    print("\n" + "="*60)
    print("Graph Neural Networks for CAD")
    print("="*60)
    print("✅ Capabilities:")
    print("   - Relationship modeling between elements")
    print("   - Structural analysis (columns, walls, connections)")
    print("   - Constraint detection and validation")
    print("   - BIM-like intelligence")
    print("\n✅ Edge Types:")
    for edge_type in EdgeType:
        print(f"   - {edge_type.value}")
    print("\n✅ Applications:")
    print("   - Structural integrity checking")
    print("   - Load path analysis")
    print("   - Space connectivity analysis")
    print("   - Code compliance verification")
    print("="*60)
