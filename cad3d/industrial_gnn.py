from __future__ import annotations
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path
from enum import Enum
import json

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from torch_geometric.nn import GCNConv, GATConv, SAGEConv, TransformerConv
    from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
    from torch_geometric.data import Data, Batch
    PYTORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    PYTORCH_GEOMETRIC_AVAILABLE = False


class IndustryType(Enum):
    """
    Enumerates the supported industrial domains for specialized GNN models.
    Each industry has unique characteristics that can be leveraged for better
    CAD element analysis.
    """
    BUILDING = "building"          # Building construction (walls, columns, slabs)
    BRIDGE = "bridge"              # Bridge engineering (piers, decks, cables)
    ROAD = "road"                  # Road construction (lanes, signs, barriers)
    DAM = "dam"                    # Dam construction (body, spillway, foundation)
    TUNNEL = "tunnel"              # Tunneling (lining, segments, support)
    FACTORY = "factory"            # Industrial plants (equipment, pipes, structures)
    MACHINERY = "machinery"        # Mechanical engineering (parts, assemblies)
    MEP = "mep"                    # Mechanical, Electrical, Plumbing
    ELECTRICAL = "electrical"      # Electrical systems (conduits, panels)
    PLUMBING = "plumbing"          # Plumbing systems (pipes, fixtures)
    HVAC = "hvac"                  # Heating, Ventilation, Air Conditioning
    RAILWAY = "railway"            # Railway engineering (tracks, signals)
    AIRPORT = "airport"            # Airport infrastructure (runways, terminals)
    SHIPBUILDING = "shipbuilding"  # Naval architecture (hull, decks)
    GENERAL = "general"            # Default for non-specialized cases


if TORCH_AVAILABLE and PYTORCH_GEOMETRIC_AVAILABLE:

    class IndustrySpecificGNN(nn.Module):
        """
        A Graph Neural Network tailored for specific industrial applications.

        This model adapts its architecture and output heads based on the specified
        industry to provide more relevant predictions. For example:
        - In buildings, it focuses on structural roles and element types.
        - In bridges, it predicts stress and deflection.
        - In road design, it analyzes traffic capacity and lane types.
        """
        def __init__(self, industry: IndustryType, node_features: int, edge_features: int, hidden_dim: int = 256, num_layers: int = 4, num_heads: int = 8, dropout: float = 0.1):
            super().__init__()
            self.industry = industry
            self.hidden_dim = hidden_dim

            self.node_proj = nn.Linear(node_features, hidden_dim)
            self.edge_proj = nn.Linear(edge_features, hidden_dim)

            self.gnn_layers = nn.ModuleList()
            for _ in range(num_layers):
                if industry in [IndustryType.BUILDING, IndustryType.BRIDGE, IndustryType.DAM]:
                    # Attention-based GAT is well-suited for complex structural relationships.
                    self.gnn_layers.append(GATConv(hidden_dim, hidden_dim // num_heads, heads=num_heads, dropout=dropout, edge_dim=hidden_dim))
                elif industry in [IndustryType.ROAD, IndustryType.RAILWAY, IndustryType.TUNNEL]:
                    # TransformerConv can capture long-range dependencies in linear structures.
                    self.gnn_layers.append(TransformerConv(hidden_dim, hidden_dim, heads=num_heads, dropout=dropout, edge_dim=hidden_dim))
                else:
                    # GraphSAGE is a robust general-purpose choice.
                    self.gnn_layers.append(SAGEConv(hidden_dim, hidden_dim))
            
            self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
            self._build_industry_heads()

        def _build_industry_heads(self):
            """Creates output layers (heads) specific to the model's industry."""
            if self.industry == IndustryType.BUILDING:
                self.element_classifier = nn.Linear(self.hidden_dim, 20)  # e.g., Wall, Beam, Column
                self.structural_role_classifier = nn.Linear(self.hidden_dim, 5)  # e.g., Load-bearing, Partition
                self.load_capacity_regressor = nn.Linear(self.hidden_dim, 3)  # Axial, Shear, Bending
            elif self.industry == IndustryType.BRIDGE:
                self.component_classifier = nn.Linear(self.hidden_dim, 15) # e.g., Pier, Deck, Cable
                self.stress_regressor = nn.Linear(self.hidden_dim, 6) # Normal, Shear, etc.
                self.deflection_regressor = nn.Linear(self.hidden_dim, 3) # X, Y, Z
            # ... other industries
            else:
                self.general_classifier = nn.Linear(self.hidden_dim, 50)
                self.general_regressor = nn.Linear(self.hidden_dim, 10)

        def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Optional[Tensor] = None, batch: Optional[Tensor] = None) -> Dict[str, Tensor]:
            """
            Performs the forward pass through the GNN.

            Args:
                x: Node feature matrix.
                edge_index: Graph connectivity in COO format.
                edge_attr: Edge feature matrix.
                batch: Batch vector for graph-level outputs.

            Returns:
                A dictionary of output tensors specific to the industry.
            """
            x = self.node_proj(x)
            if edge_attr is not None:
                edge_attr = self.edge_proj(edge_attr)

            for i, (gnn_layer, norm) in enumerate(zip(self.gnn_layers, self.layer_norms)):
                x_residual = x
                if isinstance(gnn_layer, (GATConv, TransformerConv)) and edge_attr is not None:
                    x = gnn_layer(x, edge_index, edge_attr=edge_attr)
                else:
                    x = gnn_layer(x, edge_index)
                x = norm(x + x_residual)
                x = F.relu(x)

            output = {'node_embeddings': x}
            if self.industry == IndustryType.BUILDING:
                output['element_type'] = self.element_classifier(x)
                output['structural_role'] = self.structural_role_classifier(x)
                output['load_capacity'] = self.load_capacity_regressor(x)
            elif self.industry == IndustryType.BRIDGE:
                output['component_type'] = self.component_classifier(x)
                output['stress'] = self.stress_regressor(x)
                output['deflection'] = self.deflection_regressor(x)
            # ... other industries
            else:
                output['classification'] = self.general_classifier(x)
                output['regression'] = self.general_regressor(x)

            if batch is not None:
                output['graph_embedding'] = global_mean_pool(x, batch)
            
            return output

    class HierarchicalGNN(nn.Module):
        """
        A GNN that processes graph data at multiple levels of a hierarchy.

        This is useful for complex systems where elements are nested, such as:
        - Building -> Floor -> Room -> Wall
        - Assembly -> Sub-assembly -> Part -> Feature
        """
        def __init__(self, node_features: int, edge_features: int, hidden_dim: int = 256, num_levels: int = 3, num_layers_per_level: int = 2):
            super().__init__()
            self.num_levels = num_levels
            self.node_proj = nn.Linear(node_features, hidden_dim)
            self.edge_proj = nn.Linear(edge_features, hidden_dim)
            
            self.level_gnns = nn.ModuleList([
                nn.ModuleList([GATConv(hidden_dim, hidden_dim // 8, heads=8, edge_dim=hidden_dim) for _ in range(num_layers_per_level)])
                for _ in range(num_levels)
            ])
            self.pooling_layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_levels - 1)])
            self.output_heads = nn.ModuleList([nn.Linear(hidden_dim, 50) for _ in range(num_levels)])

        def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Optional[Tensor], hierarchy_assignments: List[Tensor]) -> Dict[str, Tensor]:
            """
            Performs a forward pass through the hierarchical GNN.

            Args:
                x: Node features for the finest level.
                edge_index: Edge connectivity for the finest level.
                edge_attr: Edge features for the finest level.
                hierarchy_assignments: A list of tensors mapping nodes from a lower
                    level to a higher level (e.g., assignments[0] maps level 0 nodes to level 1).
            """
            x = self.node_proj(x)
            if edge_attr is not None:
                edge_attr = self.edge_proj(edge_attr)
            
            outputs = {}
            current_x = x

            for level in range(self.num_levels):
                for gnn_layer in self.level_gnns[level]:
                    current_x = gnn_layer(current_x, edge_index, edge_attr=edge_attr)
                    current_x = F.relu(current_x)
                
                outputs[f'level_{level}_logits'] = self.output_heads[level](current_x)
                outputs[f'level_{level}_embeddings'] = current_x

                if level < self.num_levels - 1:
                    assignment = hierarchy_assignments[level]
                    pooled_x = global_mean_pool(current_x, assignment)
                    current_x = self.pooling_layers[level](pooled_x)
                    # In a real model, edge_index and edge_attr would also be coarsened.
            
            return outputs

    class UncertaintyAwareGNN(nn.Module):
        """
        A GNN that can quantify its own prediction uncertainty.

        This is critical for high-stakes applications (e.g., structural analysis)
        where knowing the model's confidence is as important as the prediction itself.
        It uses Monte Carlo (MC) dropout during inference to estimate uncertainty.
        """
        def __init__(self, node_features: int, edge_features: int, hidden_dim: int = 256, num_layers: int = 4, num_classes: int = 50, mc_samples: int = 10):
            super().__init__()
            self.mc_samples = mc_samples
            self.node_proj = nn.Linear(node_features, hidden_dim)
            self.edge_proj = nn.Linear(edge_features, hidden_dim)
            
            self.gnn_layers = nn.ModuleList([GATConv(hidden_dim, hidden_dim // 8, heads=8, edge_dim=hidden_dim) for _ in range(num_layers)])
            self.dropouts = nn.ModuleList([nn.Dropout(0.2) for _ in range(num_layers)])
            
            self.mean_head = nn.Linear(hidden_dim, num_classes)
            self.log_var_head = nn.Linear(hidden_dim, num_classes) # Predict log variance for numerical stability

        def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Optional[Tensor] = None, return_uncertainty: bool = False) -> Dict[str, Tensor]:
            """
            Performs a forward pass, optionally estimating uncertainty.

            Args:
                return_uncertainty: If True, performs multiple forward passes with
                    dropout enabled to compute mean and variance of predictions.
            """
            x = self.node_proj(x)
            if edge_attr is not None:
                edge_attr = self.edge_proj(edge_attr)

            if return_uncertainty:
                self.train() # Enable dropout layers for MC sampling
                predictions = []
                for _ in range(self.mc_samples):
                    x_sample = x
                    for gnn, dropout in zip(self.gnn_layers, self.dropouts):
                        x_sample = F.relu(gnn(x_sample, edge_index, edge_attr=edge_attr))
                        x_sample = dropout(x_sample)
                    predictions.append(self.mean_head(x_sample))
                
                predictions = torch.stack(predictions, dim=0)
                mean = predictions.mean(dim=0)
                variance = predictions.var(dim=0)
                return {
                    'mean': mean,
                    'variance': variance,
                    'confidence': 1.0 / (1.0 + variance.mean(dim=-1))
                }
            else:
                # Standard training forward pass
                for gnn, dropout in zip(self.gnn_layers, self.dropouts):
                    x = F.relu(gnn(x, edge_index, edge_attr=edge_attr))
                    x = dropout(x)
                
                return {
                    'mean': self.mean_head(x),
                    'log_variance': self.log_var_head(x)
                }

else:
    # Define placeholder classes if dependencies are not met
    class IndustrySpecificGNN:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch and PyTorch Geometric must be installed to use GNN models.")
    class HierarchicalGNN:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch and PyTorch Geometric must be installed to use GNN models.")
    class UncertaintyAwareGNN:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch and PyTorch Geometric must be installed to use GNN models.")


def create_industry_gnn(industry: str, node_features: int = 56, edge_features: int = 21, hidden_dim: int = 256) -> 'IndustrySpecificGNN':
    """
    Factory function to create an IndustrySpecificGNN model.

    Args:
        industry: The target industry as a string (e.g., "building", "bridge").
        node_features: The number of features per node.
        edge_features: The number of features per edge.
        hidden_dim: The dimensionality of the hidden layers.

    Returns:
        An instance of IndustrySpecificGNN tailored to the specified industry.
    """
    if not TORCH_AVAILABLE or not PYTORCH_GEOMETRIC_AVAILABLE:
        raise ImportError("Cannot create GNN model: PyTorch and PyTorch Geometric are not installed.")
    
    try:
        industry_type = IndustryType(industry.lower())
    except ValueError:
        print(f"Warning: Unknown industry '{industry}'. Falling back to 'GENERAL' model.")
        industry_type = IndustryType.GENERAL
    
    return IndustrySpecificGNN(
        industry=industry_type,
        node_features=node_features,
        edge_features=edge_features,
        hidden_dim=hidden_dim
    )


def demo_industrial_gnn():
    """Demonstrates the usage of the industrial GNN models."""
    if not TORCH_AVAILABLE or not PYTORCH_GEOMETRIC_AVAILABLE:
        print("Demo skipped: PyTorch and PyTorch Geometric are required.")
        return

    print("="*70)
    print("Industry-Specific GNN Demo")
    print("="*70)

    industries_to_test = [
        ("building", "Building Construction"),
        ("bridge", "Bridge Engineering"),
        ("road", "Road Construction"),
        ("machinery", "Machinery Manufacturing"),
    ]

    for industry_key, industry_name in industries_to_test:
        print(f"\n--- Testing Model for: {industry_name} ({industry_key.upper()}) ---")
        try:
            model = create_industry_gnn(industry=industry_key, node_features=64, edge_features=32)
            
            # Create synthetic data for demonstration
            num_nodes, num_edges = 50, 150
            x = torch.randn(num_nodes, 64)
            edge_index = torch.randint(0, num_nodes, (2, num_edges), dtype=torch.long)
            edge_attr = torch.randn(num_edges, 32)
            
            output = model(x, edge_index, edge_attr)
            
            print(f"  Model created: {model.__class__.__name__}")
            print("  Output heads:")
            for key, value in output.items():
                print(f"    - {key}: Tensor of shape {value.shape}")
        except Exception as e:
            print(f"  Error during model test: {e}")

    print("\n" + "="*70)
    print("Hierarchical GNN Demo")
    print("="*70)
    try:
        h_model = HierarchicalGNN(node_features=64, edge_features=32, num_levels=3)
        
        # Synthetic data for a 3-level hierarchy
        num_nodes_l0 = 100
        num_nodes_l1 = 20
        num_nodes_l2 = 4
        
        x_h = torch.randn(num_nodes_l0, 64)
        edge_index_h = torch.randint(0, num_nodes_l0, (2, 300), dtype=torch.long)
        edge_attr_h = torch.randn(300, 32)
        
        # Define parent assignments for pooling
        assignments = [
            torch.randint(0, num_nodes_l1, (num_nodes_l0,), dtype=torch.long), # Level 0 -> 1
            torch.randint(0, num_nodes_l2, (num_nodes_l1,), dtype=torch.long), # Level 1 -> 2
        ]
        
        h_output = h_model(x_h, edge_index_h, edge_attr_h, assignments)
        print("  Hierarchical model outputs:")
        for key, value in h_output.items():
            print(f"    - {key}: Tensor of shape {value.shape}")
    except Exception as e:
        print(f"  Error during hierarchical model test: {e}")

    print("\n" + "="*70)
    print("Uncertainty-Aware GNN Demo")
    print("="*70)
    try:
        uc_model = UncertaintyAwareGNN(node_features=64, edge_features=32, num_classes=10, mc_samples=15)
        uc_model.eval() # Set to evaluation mode for inference
        
        uc_output = uc_model(x, edge_index, edge_attr, return_uncertainty=True)
        
        print("  Uncertainty model outputs (during inference):")
        for key, value in uc_output.items():
            print(f"    - {key}: Tensor of shape {value.shape}")
        print(f"  Average confidence score: {uc_output['confidence'].mean().item():.4f}")
    except Exception as e:
        print(f"  Error during uncertainty model test: {e}")

    print("\n✅ Demo complete!")


if __name__ == "__main__":
    demo_industrial_gnn()
