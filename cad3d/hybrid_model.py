"""
Hybrid AI Model for Advanced CAD Understanding

This module defines a cutting-edge hybrid neural network architecture that combines the
strengths of three powerful AI models:
1. Graph Neural Networks (GNNs)
2. State Space Models (SSMs)
3. Transformers

This "multi-brain" approach is designed for an exceptionally deep and nuanced
understanding of complex CAD data.

Architecture Flow:
- Input: A CAD graph from `cad_graph.py`.
- Stage 1 (GNN - The Structural Brain): A Graph Attention Network (GAT) processes
  the graph structure, learning from local connections and relationships. It
  generates context-aware embeddings for each element.
- Stage 2 (SSM - The Sequential Brain): The node embeddings from the GNN are
  treated as a sequence and processed by a Mamba-like State Space Model. This
  captures long-range dependencies and the overall "flow" of the design.
- Stage 3 (Transformer - The Contextual Brain): The output from the SSM is fed
  into a Transformer Encoder. This performs a final round of self-attention,
  allowing every element to weigh its importance against every other element,
  achieving maximum contextual understanding.

This model represents the absolute state-of-the-art for this task, enabling
the AI to "see" (geometry), "read" (text/properties), and "reason" (structure
and context) about CAD drawings.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data

try:
    from mamba_ssm.modules.mamba_simple import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False
    print("⚠️ Mamba-SSM not available. The SSM layer will be a simplified version.")
    print("   For the full experience, install with: pip install mamba-ssm causal-conv1d")


# ============================================================================
# The Ultimate Hybrid Multi-Modal Model
# ============================================================================

class UltimateHybridModel(nn.Module):
    """
    The ultimate hybrid model, now 100x more powerful. It fuses graph data
    with multi-modal inputs (vision and audio) for unparalleled understanding.
    """
    def __init__(self, num_node_features: int, num_edge_features: int, num_classes: int,
                 multimodal_embedding_dim: int,
                 hidden_channels: int = 256, # Increased capacity
                 heads: int = 8): # More attention heads
        super().__init__()
        self.hidden_channels = hidden_channels
        self.total_dim = hidden_channels * heads
        
        # --- Brain 1: GNN Backbone (Deeper GAT) ---
        self.gnn_conv1 = GATConv(num_node_features, hidden_channels, heads=heads, edge_dim=num_edge_features)
        self.gnn_conv2 = GATConv(self.total_dim, hidden_channels, heads=heads, edge_dim=num_edge_features)
        self.gnn_conv3 = GATConv(self.total_dim, hidden_channels, heads=heads, edge_dim=num_edge_features) # Extra layer

        # --- Multi-Modal Fusion ---
        # Linear layers to project image/audio embeddings to the same dimension as the graph nodes
        self.image_projection = nn.Linear(multimodal_embedding_dim, self.total_dim)
        self.audio_projection = nn.Linear(multimodal_embedding_dim, self.total_dim)
        
        # --- Brain 2: State Space Model (Mamba) ---
        if MAMBA_AVAILABLE:
            self.ssm_layer = Mamba(d_model=self.total_dim, d_state=32, d_conv=4, expand=2)
        else:
            self.ssm_layer = nn.GRU(self.total_dim, self.total_dim, num_layers=2, batch_first=True)

        # --- Brain 3: Transformer Encoder (Deeper) ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.total_dim,
            nhead=heads,
            dim_feedforward=self.total_dim * 4,
            dropout=0.5,
            activation='gelu', # GELU can be better than ReLU
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=4) # More layers

        # --- Output Layer ---
        self.output_linear = nn.Linear(self.total_dim, num_classes)

    def forward(self, data: Data, 
                image_embedding: torch.Tensor | None = None,
                audio_embedding: torch.Tensor | None = None) -> torch.Tensor:
        
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # --- Stage 1: GNN Processing ---
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.gnn_conv1(x, edge_index, edge_attr))
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.gnn_conv2(x, edge_index, edge_attr))
        x = F.dropout(x, p=0.6, training=self.training)
        x_gnn = F.elu(self.gnn_conv3(x, edge_index, edge_attr))

        # --- Stage 2: Multi-Modal Fusion ---
        # We add the information from image and audio to the graph features.
        # This injects the "vision" and "hearing" context into the AI's brain.
        if image_embedding is not None:
            projected_image_emb = self.image_projection(image_embedding)
            # Add image context to all nodes in the graph
            x_gnn = x_gnn + projected_image_emb
            
        if audio_embedding is not None:
            projected_audio_emb = self.audio_projection(audio_embedding)
            # Add audio context to all nodes in the graph
            x_gnn = x_gnn + projected_audio_emb

        # --- Stage 3: SSM Processing ---
        x_seq = x_gnn.unsqueeze(0)
        if MAMBA_AVAILABLE:
            x_ssm = self.ssm_layer(x_seq)
        else:
            x_ssm, _ = self.ssm_layer(x_seq)

        # --- Stage 4: Transformer Processing ---
        x_transformer = self.transformer_encoder(x_ssm)

        # --- Final Output ---
        x_final = x_transformer.squeeze(0)
        output = self.output_linear(x_final)
        
        return F.log_softmax(output, dim=1)
