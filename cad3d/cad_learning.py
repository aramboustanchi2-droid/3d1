"""
CAD Learning System with Graph Neural Networks (GNNs)

This module implements the machine learning capabilities for the CAD AI. It uses
Graph Neural Networks (GNNs) to learn from the graph-based representation of
CAD drawings provided by `cad_graph.py`.

The primary goal is to enable the AI to understand, analyze, and eventually
generate or modify CAD designs based on learned architectural and engineering
principles.

Key Features:
- GNN model for learning on CAD graphs.
- Training and inference pipelines.
- Node classification task: e.g., predict the structural role of an element.
- Link prediction task: e.g., predict if two elements should be connected.
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader # For batching multiple graphs

from cad_graph import CADGraph, CADComponent, CADDependency, ComponentType, DependencyType
from architectural_analyzer import ArchitecturalAnalyzer, ArchitecturalAnalysis
from hybrid_model import UltimateHybridModel # Import the new ULTIMATE hybrid model
from multimodal_processor import MultiModalProcessor # Import the new multi-modal processor

import random
from pathlib import Path


# ============================================================================
# Original Graph Neural Network Model
# ============================================================================

class CADGNN(torch.nn.Module):
    """
    A Graph Neural Network for learning from CAD data.

    This model uses Graph Convolutional Networks (GCN) to process the graph
    structure of a CAD drawing. It can be used for tasks like node classification
    (e.g., identifying element types or properties) or link prediction.
    """
    def __init__(self, num_node_features: int, num_classes: int, hidden_channels: int = 64):
        super(CADGNN, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_classes)

    def forward(self, data: Data) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            data: A PyTorch Geometric Data object containing the graph.
                  - data.x: Node feature matrix
                  - data.edge_index: Graph connectivity in COO format

        Returns:
            The output logits for each node.
        """
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)

# ============================================================================
# Advanced Graph Neural Network Model (Bigger, Faster, More Powerful)
# ============================================================================

class AdvancedCADGNN(torch.nn.Module):
    """
    An advanced, more powerful Graph Neural Network for CAD data.

    This model uses Graph Attention Networks (GAT) and is deeper and wider.
    GAT allows the model to weigh the importance of different nodes in a
    neighborhood, making it more expressive than GCN.

    Enhancements:
    - Deeper Architecture: More layers to capture complex patterns.
    - Wider Layers: Increased hidden channels (128) for greater capacity.
    - Attention Mechanism: GAT layers focus on the most relevant relationships.
    - Residual Connections: Helps in training deeper networks effectively.
    - Edge Feature Integration: The model now learns from relationship properties.
    """
    def __init__(self, num_node_features: int, num_edge_features: int, num_classes: int, hidden_channels: int = 128, heads: int = 4):
        super().__init__()
        self.conv1 = GATConv(num_node_features, hidden_channels, heads=heads, edge_dim=num_edge_features)
        self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads=heads, edge_dim=num_edge_features)
        self.conv3 = GATConv(hidden_channels * heads, num_classes, heads=1, concat=False, edge_dim=num_edge_features)

    def forward(self, data: Data) -> torch.Tensor:
        """
        Forward pass using Graph Attention layers.

        Args:
            data: A PyTorch Geometric Data object containing the graph.
                  - data.x: Node feature matrix
                  - data.edge_index: Graph connectivity
                  - data.edge_attr: Edge feature matrix

        Returns:
            The output logits for each node.
        """
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # Layer 1
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv1(x, edge_index, edge_attr)
        x = F.elu(x)

        # Layer 2
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index, edge_attr)
        x = F.elu(x)
        
        # Layer 3 (Output)
        x = self.conv3(x, edge_index, edge_attr)

        return F.log_softmax(x, dim=1)

# ============================================================================
# Training and Inference
# ============================================================================

def train(model: torch.nn.Module, data: Data, optimizer: torch.optim.Optimizer, image_embedding: torch.Tensor, audio_embedding: torch.Tensor):
    """
    Performs a single training step with multi-modal data.
    """
    model.train()
    optimizer.zero_grad()
    # Pass the multi-modal embeddings to the model
    out = model(data, image_embedding=image_embedding, audio_embedding=audio_embedding)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad()
def test(model: torch.nn.Module, data: Data, image_embedding: torch.Tensor, audio_embedding: torch.Tensor) -> tuple[float, float]:
    """
    Evaluates the model on the test set with multi-modal data.
    """
    model.eval()
    # Pass the multi-modal embeddings to the model
    out = model(data, image_embedding=image_embedding, audio_embedding=audio_embedding)
    pred = out.argmax(dim=1)
    
    # Accuracy on validation set
    val_correct = pred[data.val_mask] == data.y[data.val_mask]
    val_acc = int(val_correct.sum()) / int(data.val_mask.sum()) if int(data.val_mask.sum()) > 0 else 0
    
    # Accuracy on test set
    test_correct = pred[data.test_mask] == data.y[data.test_mask]
    test_acc = int(test_correct.sum()) / int(data.test_mask.sum()) if int(data.test_mask.sum()) > 0 else 0
    
    return val_acc, test_acc

# ============================================================================
# Example: Structural Usage Prediction
# ============================================================================

def create_prediction_dataset(graph: CADGraph) -> Data:
    """
    Creates a dataset for predicting a property of components.

    This function converts a CADGraph into a PyTorch Geometric Data object,
    with labels for a node classification task.

    Task: Predict if a component is 'load_bearing' or 'non_structural'.
    - Class 0: non_structural
    - Class 1: load_bearing
    - Class 2: unknown/not applicable
    """
    pyg_data = graph.to_pytorch_geometric()
    if pyg_data is None:
        raise ValueError("Could not convert CADGraph to PyTorch Geometric format.")

    num_nodes = pyg_data.num_nodes
    labels = torch.full((num_nodes,), 2, dtype=torch.long)  # Default to 'unknown'
    
    node_map = {comp_id: i for i, comp_id in enumerate(graph.components.keys())}

    for comp_id, component in graph.components.items():
        node_idx = node_map[comp_id]
        # This is an example task. We reuse 'structural_usage' for demonstration.
        # This can be adapted for any other predictable property.
        usage = component.structural_usage
        if usage == "load_bearing":
            labels[node_idx] = 1
        elif usage == "non_structural":
            labels[node_idx] = 0
    
    pyg_data.y = labels

    # Create train/val/test masks
    known_labels_mask = (labels < 2)
    known_indices = torch.where(known_labels_mask)[0]
    
    if len(known_indices) < 3:
        print("⚠️ Warning: Not enough labeled data to create train/val/test splits.")
        pyg_data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        pyg_data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        pyg_data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        return pyg_data

    permuted_indices = known_indices[torch.randperm(len(known_indices))]
    train_size = int(0.6 * len(permuted_indices))
    val_size = int(0.2 * len(permuted_indices))
    
    train_indices = permuted_indices[:train_size]
    val_indices = permuted_indices[train_size : train_size + val_size]
    test_indices = permuted_indices[train_size + val_size:]

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[train_indices] = True
    val_mask[val_indices] = True
    test_mask[test_indices] = True

    pyg_data.train_mask = train_mask
    pyg_data.val_mask = val_mask
    pyg_data.test_mask = test_mask
    
    return pyg_data


def run_universal_prediction_example():
    """
    A full example demonstrating the AI's universal understanding of a
    robotic system, enhanced with multi-modal inputs.
    """
    print("🤖🧠✨ Starting Universal Prediction Example with a Robotic Arm...")

    # Step 1: Initialize the Multi-Modal Processor
    multimodal_embedding_dim = 256
    multimodal_processor = MultiModalProcessor(embedding_dim=multimodal_embedding_dim)
    
    # Step 2: Simulate "seeing" a technical drawing and "hearing" a command
    print("\n👁️ Simulating 'seeing' a robot assembly diagram...")
    image_embedding = multimodal_processor.process_image_input("data/diagrams/robot_arm_assembly.png")
    
    print("\n👂 Simulating 'hearing' a voice command: 'Check motor power connections'...")
    _, audio_embedding = multimodal_processor.process_audio_command("data/audio/check_power.wav")
    
    # Step 3: Create a CADGraph of a robotic arm
    graph = CADGraph("Robotic Arm System")
    
    # Add components
    components = [
        CADComponent(id="base_plate", component_type=ComponentType.FRAME, structural_usage="load_bearing"),
        CADComponent(id="turret", component_type=ComponentType.JOINT, structural_usage="load_bearing"),
        CADComponent(id="main_motor", component_type=ComponentType.MOTOR, structural_usage="non_structural"),
        CADComponent(id="arm1", component_type=ComponentType.LINKAGE, structural_usage="load_bearing"),
        CADComponent(id="elbow_joint", component_type=ComponentType.JOINT, structural_usage="load_bearing"),
        CADComponent(id="arm2", component_type=ComponentType.LINKAGE, structural_usage="load_bearing"),
        CADComponent(id="wrist_motor", component_type=ComponentType.MOTOR, structural_usage="non_structural"),
        CADComponent(id="gripper", component_type=ComponentType.FRAME, structural_usage="non_structural"),
        CADComponent(id="main_controller", component_type=ComponentType.PROCESSOR), # Property to be predicted
        CADComponent(id="power_supply", component_type=ComponentType.BATTERY),      # Property to be predicted
    ]
    for comp in components:
        comp.centroid = (random.random(), random.random(), random.random())
        comp.parameters = {'length': random.random(), 'radius': random.random() * 0.2}
        graph.add_component(comp)

    # Add dependencies
    graph.add_dependency(CADDependency("turret", "base_plate", DependencyType.MOUNTS_TO))
    graph.add_dependency(CADDependency("main_motor", "turret", DependencyType.MOUNTS_TO))
    graph.add_dependency(CADDependency("main_motor", "arm1", DependencyType.ACTUATES))
    graph.add_dependency(CADDependency("arm1", "turret", DependencyType.CONNECTED_TO))
    graph.add_dependency(CADDependency("elbow_joint", "arm1", DependencyType.CONNECTED_TO))
    graph.add_dependency(CADDependency("arm2", "elbow_joint", DependencyType.CONNECTED_TO))
    graph.add_dependency(CADDependency("wrist_motor", "arm2", DependencyType.MOUNTS_TO))
    graph.add_dependency(CADDependency("wrist_motor", "gripper", DependencyType.ACTUATES))
    # Connections to be inferred by the model
    graph.add_dependency(CADDependency("main_controller", "main_motor", DependencyType.CONTROLS))
    graph.add_dependency(CADDependency("main_controller", "wrist_motor", DependencyType.CONTROLS))
    graph.add_dependency(CADDependency("power_supply", "main_motor", DependencyType.POWERED_BY))


    print(f"\n✅ Created a sample CAD graph for a robotic arm with {len(graph.components)} components.")

    # Step 4: Create the dataset for a prediction task
    # Task: Predict if a component is 'load_bearing' (structurally critical)
    try:
        data = create_prediction_dataset(graph)
        if data.train_mask.sum() == 0:
             print("❌ Not enough data to train. Exiting example.")
             return
        print("✅ Converted universal component graph to PyTorch Geometric dataset.")
    except (ValueError, ImportError) as e:
        print(f"❌ Could not create dataset: {e}")
        return

    # Step 5: Initialize and train the ULTIMATE HYBRID model
    num_classes = 3  # (non_structural, load_bearing, unknown)
    model = UltimateHybridModel(
        num_node_features=data.num_node_features,
        num_edge_features=data.num_edge_features,
        num_classes=num_classes,
        hidden_channels=multimodal_embedding_dim
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

    print("\n🚀 Training the ULTIMATE HYBRID model on the robotic system...")
    for epoch in range(1, 151):
        loss = train(model, data, optimizer, image_embedding, audio_embedding)
        if epoch % 15 == 0:
            val_acc, test_acc = test(model, data, image_embedding, audio_embedding)
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}')

    # Step 6: Run inference
    print("\n🔍 Running inference on the robotic system...")
    model.eval()
    with torch.no_grad():
        out = model(data, image_embedding=image_embedding, audio_embedding=audio_embedding)
        predictions = out.argmax(dim=1)

    class_map = {0: "non_structural", 1: "load_bearing", 2: "unknown"}
    
    node_map_inv = {i: comp_id for i, comp_id in enumerate(graph.components.keys())}
    print("\n--- Inference Results (Structural Criticality) ---")
    for i, pred_class in enumerate(predictions):
        comp_id = node_map_inv[i]
        component = graph.components[comp_id]
        if component.structural_usage is None:
            predicted_usage = class_map[pred_class.item()]
            print(f"   - Component '{comp_id}' ({component.component_type.name}):")
            print(f"     Predicted Criticality -> {predicted_usage.upper()}")


if __name__ == "__main__":
    # Note: You need to have PyTorch and PyTorch Geometric installed.
    # pip install torch torch-geometric
    try:
        run_universal_prediction_example()
    except ImportError as e:
        print(f"\n❌ Missing required library: {e}")
        print("Please install the necessary packages to run the learning example:")
        print("pip install torch torch-geometric")
        print("You may also need to install torch-scatter depending on your OS/PyTorch version.")

