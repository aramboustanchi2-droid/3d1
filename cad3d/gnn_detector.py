"""
Graph Neural Network (GNN) for CAD Structural Element Detection and Analysis.

This module provides tools to represent Computer-Aided Design (CAD) drawings as
graphs and apply Graph Neural Networks to analyze their structural properties.

Key Capabilities:
- Models spatial and semantic relationships between CAD elements (e.g., walls,
  columns, doors).
- Detects structural patterns, connections, and constraints.
- Ideal for Building Information Modeling (BIM) pre-processing and structural
  analysis tasks.
- Inspired by concepts from Autodesk AI research and Revit's constraint system.
"""

from __future__ import annotations
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import math

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor
    import numpy as np
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # Define dummy classes for type hinting when torch is not available.
    class Tensor: pass
    class nn:
        Module = object
        Sequential = object
        Linear = object
        ReLU = object
        Dropout = object
        LayerNorm = object

    print("Warning: PyTorch is not installed. GNN models will not be available.")


class EdgeType(Enum):
    """Enumerates the types of relationships (edges) between CAD elements."""
    CONNECTED = "connected"          # Physically touching or intersecting.
    ADJACENT = "adjacent"            # Close proximity without touching.
    PARALLEL = "parallel"            # Geometrically parallel.
    PERPENDICULAR = "perpendicular"  # Geometrically perpendicular.
    SUPPORTS = "supports"            # A structural support relationship (e.g., column supporting a beam).
    CONTAINS = "contains"            # One element enclosing another (e.g., room containing a door).
    ALIGNED = "aligned"              # Centerlines or edges are aligned.
    DISTANCE = "distance"            # A generic edge representing a measured distance.


@dataclass
class CADNode:
    """Represents a single element in a CAD drawing as a node in a graph."""
    node_id: int
    element_type: str  # e.g., 'wall', 'column', 'door', 'window'
    position: Tuple[float, float, float]  # Centroid (x, y, z)
    dimensions: Optional[Tuple[float, float, float]] = None  # Bounding box size (length, width, height)
    layer: Optional[str] = None
    properties: Dict[str, Any] = field(default_factory=dict)
    features: Optional[List[float]] = None  # Pre-computed feature vector for the GNN.


@dataclass
class CADEdge:
    """Represents a relationship between two CAD nodes as a directed edge."""
    source_id: int
    target_id: int
    edge_type: EdgeType
    weight: float = 1.0  # Strength or importance of the relationship.
    distance: Optional[float] = None
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CADGraph:
    """Represents a complete CAD drawing as a graph of nodes and edges."""
    nodes: List[CADNode]
    edges: List[CADEdge]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_node_by_id(self, node_id: int) -> Optional[CADNode]:
        """Retrieves a node by its unique ID."""
        # This is inefficient for large graphs; a dict lookup would be better.
        for node in self.nodes:
            if node.node_id == node_id:
                return node
        return None


if TORCH_AVAILABLE:
    class CADGNN(nn.Module):
        """
        A Graph Neural Network designed for analyzing CAD graphs.

        This model uses both node-centric (Graph Convolution) and edge-centric
        (Edge Convolution) message passing to build rich representations of
        CAD elements and their relationships.
        """

        def __init__(
            self,
            node_feature_dim: int,
            edge_feature_dim: int,
            hidden_dim: int = 128,
            num_layers: int = 4,
            num_node_classes: int = 15,
            num_edge_classes: int = len(EdgeType),
        ):
            """
            Initializes the CADGNN model.

            Args:
                node_feature_dim: The dimensionality of the input node features.
                edge_feature_dim: The dimensionality of the input edge features.
                hidden_dim: The dimensionality of hidden layers.
                num_layers: The number of message passing layers.
                num_node_classes: The number of classes for node classification.
                num_edge_classes: The number of classes for edge classification.
            """
            super().__init__()

            self.node_encoder = nn.Linear(node_feature_dim, hidden_dim)
            self.edge_encoder = nn.Linear(edge_feature_dim, hidden_dim)

            self.conv_layers = nn.ModuleList()
            for _ in range(num_layers):
                # Each layer combines node and edge convolutions.
                # Using TransformerConv for attention-based message passing.
                conv = nn.TransformerConv(hidden_dim, hidden_dim, heads=4, dropout=0.1, edge_dim=hidden_dim)
                norm = nn.LayerNorm(hidden_dim * 4) # Output of 4 heads
                self.conv_layers.append(nn.ModuleDict({'conv': conv, 'norm': norm}))

            # Classifier heads
            self.node_classifier = nn.Linear(hidden_dim * 4, num_node_classes)
            self.edge_classifier = nn.Linear(hidden_dim * 4 * 2, num_edge_classes)
            self.graph_classifier = nn.Linear(hidden_dim * 4, 10) # 10 graph-level classes

        def forward(
            self,
            x: Tensor,
            edge_index: Tensor,
            edge_attr: Tensor,
            batch: Optional[Tensor] = None
        ) -> Dict[str, Tensor]:
            """
            Performs the forward pass of the GNN.

            Args:
                x: Node feature matrix of shape [num_nodes, node_feature_dim].
                edge_index: Graph connectivity in COO format, shape [2, num_edges].
                edge_attr: Edge feature matrix, shape [num_edges, edge_feature_dim].
                batch: Batch vector for graph-level outputs, shape [num_nodes].

            Returns:
                A dictionary containing node, edge, and graph-level predictions.
            """
            x = self.node_encoder(x)
            edge_attr = self.edge_encoder(edge_attr)

            for layer in self.conv_layers:
                x_res = x
                x = layer['conv'](x, edge_index, edge_attr)
                x = layer['norm'](x)
                x = F.relu(x)
                # Simple residual connection (assuming dimensions match)
                if x.shape == x_res.shape:
                    x = x + x_res

            # Node-level predictions
            node_logits = self.node_classifier(x)

            # Edge-level predictions
            source_nodes, target_nodes = edge_index
            edge_repr = torch.cat([x[source_nodes], x[target_nodes]], dim=-1)
            edge_logits = self.edge_classifier(edge_repr)

            # Graph-level prediction
            if batch is None:
                batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            
            graph_embedding = torch.mean(x, dim=0) # Simplified global pooling
            graph_logits = self.graph_classifier(graph_embedding)

            return {
                'node_logits': node_logits,
                'edge_logits': edge_logits,
                'graph_logits': graph_logits.unsqueeze(0),
                'node_embeddings': x,
            }
else:
    # Define a placeholder if PyTorch is not available
    class CADGNN:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch must be installed to use CADGNN.")


class CADGraphBuilder:
    """
    Constructs a CADGraph from various sources, like DXF files.
    """
    def __init__(self, distance_threshold: float = 1000.0, angle_tolerance: float = 5.0):
        """
        Args:
            distance_threshold: Maximum distance to consider creating an 'ADJACENT' edge.
            angle_tolerance: Degree tolerance for detecting parallel/perpendicular edges.
        """
        self.distance_threshold = distance_threshold
        self.angle_tolerance = math.radians(angle_tolerance)
        self._node_id_counter = 0

    def build_from_dxf(self, dxf_path: str | Path) -> CADGraph:
        """
        Builds a CADGraph from a DXF file.

        Args:
            dxf_path: Path to the DXF file.

        Returns:
            A CADGraph instance representing the DXF content.
        """
        try:
            import ezdxf
        except ImportError:
            raise ImportError("The 'ezdxf' package is required to build graphs from DXF files.")

        doc = ezdxf.readfile(dxf_path)
        msp = doc.modelspace()
        self._node_id_counter = 0
        nodes: List[CADNode] = []

        # Extract nodes from various entity types
        for polyline in msp.query('LWPOLYLINE[is_closed==True]'):
            points = list(polyline.get_points('xy'))
            if len(points) > 2:
                centroid = self._calculate_centroid(points)
                nodes.append(CADNode(
                    node_id=self._get_next_id(),
                    element_type='wall', # Assumption
                    position=(*centroid, 0),
                    layer=polyline.dxf.layer,
                    properties={'num_vertices': len(points), 'ezdxf_handle': polyline.dxf.handle}
                ))

        for circle in msp.query('CIRCLE'):
            layer_name = circle.dxf.layer.upper()
            el_type = 'column' if 'COL' in layer_name else 'hole'
            nodes.append(CADNode(
                node_id=self._get_next_id(),
                element_type=el_type,
                position=(*circle.dxf.center.xyz,),
                dimensions=(circle.dxf.radius * 2, circle.dxf.radius * 2, 0),
                layer=circle.dxf.layer,
                properties={'radius': circle.dxf.radius, 'ezdxf_handle': circle.dxf.handle}
            ))

        edges = self._build_edges(nodes)
        return CADGraph(nodes=nodes, edges=edges, metadata={'source_file': str(dxf_path)})

    def _get_next_id(self) -> int:
        """Generates a unique node ID."""
        new_id = self._node_id_counter
        self._node_id_counter += 1
        return new_id

    def _calculate_centroid(self, points: List[Tuple[float, float]]) -> Tuple[float, float]:
        """Calculates the centroid of a 2D polygon."""
        if not points: return (0, 0)
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        return (sum(xs) / len(xs), sum(ys) / len(ys))

    def _build_edges(self, nodes: List[CADNode]) -> List[CADEdge]:
        """Generates edges between nodes based on spatial heuristics."""
        edges: List[CADEdge] = []
        for i, node1 in enumerate(nodes):
            for node2 in nodes[i+1:]:
                distance = self._distance_3d(node1.position, node2.position)
                if distance < self.distance_threshold:
                    edge_type = self._determine_edge_type(node1, node2, distance)
                    edges.append(CADEdge(
                        source_id=node1.node_id,
                        target_id=node2.node_id,
                        edge_type=edge_type,
                        distance=distance,
                        weight=1.0 / (1.0 + distance) # Inverse distance weighting
                    ))
        return edges

    def _distance_3d(self, p1: Tuple, p2: Tuple) -> float:
        """Calculates the Euclidean distance between two 3D points."""
        return math.sqrt(sum((c1 - c2) ** 2 for c1, c2 in zip(p1, p2)))

    def _determine_edge_type(self, node1: CADNode, node2: CADNode, distance: float) -> EdgeType:
        """Heuristically determines the edge type between two nodes."""
        type_pair = tuple(sorted((node1.element_type, node2.element_type)))

        if type_pair == ('column', 'wall'):
            return EdgeType.SUPPORTS
        if type_pair == ('wall', 'wall'):
            return EdgeType.CONNECTED if distance < 100.0 else EdgeType.ADJACENT
        
        return EdgeType.DISTANCE # Default fallback

    def to_torch_data(self, graph: CADGraph, device: Optional[str] = None) -> Dict[str, Tensor]:
        """
        Converts a CADGraph into a dictionary of PyTorch tensors for the GNN.

        Args:
            graph: The CADGraph to convert.
            device: The PyTorch device to move tensors to (e.g., 'cuda').

        Returns:
            A dictionary of tensors ('x', 'edge_index', 'edge_attr').
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required to create tensor data.")

        # Create vocabulary mappings
        node_type_vocab = {nt: i for i, nt in enumerate(sorted(list(set(n.element_type for n in graph.nodes))))}
        edge_type_vocab = {et: i for i, et in enumerate(EdgeType)}

        # Node features
        node_features_list = []
        for node in graph.nodes:
            type_one_hot = [0.0] * len(node_type_vocab)
            type_one_hot[node_type_vocab[node.element_type]] = 1.0
            
            pos = list(node.position)
            dims = list(node.dimensions) if node.dimensions else [0.0, 0.0, 0.0]
            features = pos + dims + type_one_hot
            node_features_list.append(features)
        
        x = torch.tensor(node_features_list, dtype=torch.float32, device=device)

        # Edge features and index
        edge_sources, edge_targets = [], []
        edge_features_list = []
        for edge in graph.edges:
            # Create edges in both directions for an undirected graph
            for src, tgt in [(edge.source_id, edge.target_id), (edge.target_id, edge.source_id)]:
                edge_sources.append(src)
                edge_targets.append(tgt)
                
                type_one_hot = [0.0] * len(edge_type_vocab)
                type_one_hot[edge_type_vocab[edge.edge_type]] = 1.0
                
                dist = [edge.distance or 0.0]
                weight = [edge.weight]
                features = type_one_hot + dist + weight
                edge_features_list.append(features)

        edge_index = torch.tensor([edge_sources, edge_targets], dtype=torch.long, device=device)
        edge_attr = torch.tensor(edge_features_list, dtype=torch.float32, device=device)

        return {'x': x, 'edge_index': edge_index, 'edge_attr': edge_attr}


def create_dummy_dxf(path: Path):
    """Creates a simple DXF file for demonstration purposes."""
    try:
        import ezdxf
    except ImportError:
        print("Cannot create dummy DXF: 'ezdxf' is not installed.")
        return

    doc = ezdxf.new()
    msp = doc.modelspace()
    # Add a square room (4 walls)
    msp.add_lwpolyline([(0, 0), (5000, 0), (5000, 5000), (0, 5000)], close=True, dxfattribs={'layer': 'WALLS'})
    # Add a column
    msp.add_circle((4500, 4500), radius=150, dxfattribs={'layer': 'COLUMNS'})
    # Add another room
    msp.add_lwpolyline([(5000, 0), (10000, 0), (10000, 5000), (5000, 5000)], close=True, dxfattribs={'layer': 'WALLS'})
    doc.saveas(path)
    print(f"Created dummy DXF file at: {path}")


if __name__ == "__main__":
    print("="*70)
    print("CAD Graph Neural Network - Demonstration")
    print("="*70)

    if not TORCH_AVAILABLE:
        print("Demonstration skipped: PyTorch is required.")
    else:
        # 1. Setup
        dummy_dxf_path = Path("./dummy_cad_layout.dxf")
        create_dummy_dxf(dummy_dxf_path)

        # 2. Build Graph from DXF
        print("\n--- Building Graph from DXF ---")
        builder = CADGraphBuilder(distance_threshold=5000.0)
        try:
            cad_graph = builder.build_from_dxf(dummy_dxf_path)
            print(f"Graph created with {len(cad_graph.nodes)} nodes and {len(cad_graph.edges)} edges.")
            print(f"Node types found: {set(n.element_type for n in cad_graph.nodes)}")
            print(f"Edge types found: {set(e.edge_type.value for e in cad_graph.edges)}")

            # 3. Convert to PyTorch Tensors
            print("\n--- Converting Graph to Tensors ---")
            data_dict = builder.to_torch_data(cad_graph)
            node_feat_dim = data_dict['x'].shape[1]
            edge_feat_dim = data_dict['edge_attr'].shape[1]
            print(f"Node feature dimension: {node_feat_dim}")
            print(f"Edge feature dimension: {edge_feat_dim}")

            # 4. Initialize and Run GNN Model
            print("\n--- Running GNN Model ---")
            model = CADGNN(
                node_feature_dim=node_feat_dim,
                edge_feature_dim=edge_feat_dim,
                hidden_dim=128,
                num_layers=3,
                num_node_classes=5, # e.g., wall, column, beam, slab, other
                num_edge_classes=len(EdgeType)
            )
            print(f"Model initialized: {model.__class__.__name__}")

            # 5. Forward Pass
            with torch.no_grad():
                model.eval()
                predictions = model(
                    x=data_dict['x'],
                    edge_index=data_dict['edge_index'],
                    edge_attr=data_dict['edge_attr']
                )
            print("\n--- GNN Output Shapes ---")
            for name, tensor in predictions.items():
                print(f"  - {name}: {tensor.shape}")
            
            print("\n✅ Demonstration complete.")

        except ImportError as e:
            print(f"Error during demonstration: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
        finally:
            # Clean up the dummy file
            if dummy_dxf_path.exists():
                dummy_dxf_path.unlink()
