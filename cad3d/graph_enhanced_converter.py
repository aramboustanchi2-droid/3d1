"""
Integrated CAD 2D→3D System with Graph Neural Networks
سیستم یکپارچه تبدیل 2D به 3D با شبکه‌های عصبی گرافی

این ماژول تمام سیستم‌ها را یکپارچه می‌کند:
- VAE (Variational Autoencoder)
- Diffusion (برای جزئیات)
- ViT (Vision Transformer)
- GNN (Graph Neural Network)
- Graph-Based Representation

Pipeline کامل:
1. DXF/Image → Graph extraction
2. Graph → GNN → فهم روابط و ساختار
3. GNN features + ViT features → VAE/Diffusion
4. → 3D Model با درک کامل

استفاده برای تمام صنایع:
- ساختمان‌سازی (Buildings)
- پل‌سازی (Bridges)
- تونل‌سازی (Tunnels)
- سدسازی (Dams)
- مکانیک (Mechanical)
- و هر صنعتی با نیاز به نقشه 3D
"""

from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from torch_geometric.data import Data
    PYTORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    PYTORCH_GEOMETRIC_AVAILABLE = False

from .cad_graph import CADGraph, ElementType
from .cad_graph_builder import CADGraphBuilder, IntelligentGraphBuilder
from .cad_gnn import CAD_GAT, CADDepthReconstructionGNN


if TORCH_AVAILABLE and PYTORCH_GEOMETRIC_AVAILABLE:
    
    class GraphEnhancedCAD3DConverter(nn.Module):
        """
        سیستم یکپارچه 2D→3D با قابلیت‌های Graph-Based
        
        Architecture:
        1. Graph Builder: DXF → CAD Graph
        2. GNN: Graph analysis & feature extraction
        3. ViT: Image understanding
        4. Fusion: GNN features + ViT features
        5. VAE/Diffusion: 3D generation
        
        مزایا:
        - درک کامل روابط بین عناصر
        - تحلیل ساختاری دقیق
        - تولید 3D با consistency بالا
        - قابلیت parametric update
        """
        
        def __init__(
            self,
            device: str = "cpu",
            vit_model_path: Optional[Path] = None,
            vae_model_path: Optional[Path] = None,
            diffusion_model_path: Optional[Path] = None,
            num_gnn_layers: int = 4,
            gnn_hidden_dim: int = 256
        ):
            super().__init__()
            
            self.device = device
            
            print("="*70)
            print("Initializing Graph-Enhanced CAD 3D Converter")
            print("="*70)
            
            # ================================================================
            # 1. Graph Builder
            # ================================================================
            print("\n1️⃣ Initializing Graph Builder...")
            self.graph_builder = IntelligentGraphBuilder(
                proximity_threshold=100.0,  # 100mm
                parallel_angle_threshold=5.0
            )
            print("   ✓ Graph Builder ready")
            
            # ================================================================
            # 2. GNN for Graph Analysis
            # ================================================================
            print("\n2️⃣ Initializing Graph Neural Network...")
            
            # Node features: [type_embedding(50), x, y, z, length, width, height] = 56
            # Edge features: [relation_type(20), weight] = 21
            self.gnn = CAD_GAT(
                node_features=56,
                edge_features=21,
                hidden_dim=gnn_hidden_dim,
                num_layers=num_gnn_layers,
                num_heads=8,
                num_classes=len(ElementType)
            ).to(device)
            
            print(f"   ✓ GNN ready ({num_gnn_layers} layers, {gnn_hidden_dim}D)")
            
            # Depth reconstruction GNN
            self.depth_gnn = CADDepthReconstructionGNN(
                node_features=56,
                edge_features=21,
                hidden_dim=gnn_hidden_dim,
                num_gnn_layers=num_gnn_layers
            ).to(device)
            
            print("   ✓ Depth Reconstruction GNN ready")
            
            # ================================================================
            # 3. Vision Transformer (optional)
            # ================================================================
            print("\n3️⃣ Initializing Vision Transformer...")
            
            self.vit = None
            if vit_model_path and vit_model_path.exists():
                try:
                    from .vision_transformer_cad import VisionTransformerCAD, VisionTransformerConfig
                    
                    config = VisionTransformerConfig(
                        image_size=256,
                        patch_size=16,
                        num_classes=len(ElementType),
                        dim=768
                    )
                    self.vit = VisionTransformerCAD(config).to(device)
                    
                    checkpoint = torch.load(vit_model_path, map_location=device)
                    self.vit.load_state_dict(checkpoint['model_state_dict'])
                    self.vit.eval()
                    
                    print(f"   ✓ ViT loaded from {vit_model_path}")
                except Exception as e:
                    print(f"   ⚠️  Could not load ViT: {e}")
            else:
                print("   ℹ️  ViT not loaded (optional)")
            
            # ================================================================
            # 4. Feature Fusion
            # ================================================================
            print("\n4️⃣ Initializing Feature Fusion...")
            
            # Fuse GNN features with ViT features
            vit_dim = 768 if self.vit else 0
            fusion_input_dim = gnn_hidden_dim + vit_dim
            
            self.feature_fusion = nn.Sequential(
                nn.Linear(fusion_input_dim, 512),
                nn.LayerNorm(512),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(512, 512)
            ).to(device)
            
            print(f"   ✓ Fusion layer ready (GNN:{gnn_hidden_dim} + ViT:{vit_dim} → 512)")
            
            # ================================================================
            # 5. VAE Decoder
            # ================================================================
            print("\n5️⃣ Initializing VAE Decoder...")
            
            self.vae = None
            if vae_model_path and vae_model_path.exists():
                try:
                    from .vae_model import create_cad_vae
                    
                    self.vae = create_cad_vae(
                        latent_dim=512,
                        output_points=2048,
                        device=device
                    )
                    
                    checkpoint = torch.load(vae_model_path, map_location=device)
                    self.vae.load_state_dict(checkpoint['state_dict'])
                    self.vae.eval()
                    
                    print(f"   ✓ VAE loaded from {vae_model_path}")
                except Exception as e:
                    print(f"   ⚠️  Could not load VAE: {e}")
            else:
                print("   ℹ️  VAE not loaded (optional)")
            
            # ================================================================
            # 6. Diffusion Model (optional)
            # ================================================================
            print("\n6️⃣ Initializing Diffusion Model...")
            
            self.diffusion = None
            if diffusion_model_path and diffusion_model_path.exists():
                try:
                    from .diffusion_3d_model import create_diffusion_model
                    
                    self.diffusion = create_diffusion_model(
                        num_points=4096,
                        timesteps=1000,
                        device=device
                    )
                    
                    checkpoint = torch.load(diffusion_model_path, map_location=device)
                    self.diffusion.image_encoder.load_state_dict(checkpoint['image_encoder_state'])
                    self.diffusion.unet.load_state_dict(checkpoint['unet_state'])
                    
                    print(f"   ✓ Diffusion loaded from {diffusion_model_path}")
                except Exception as e:
                    print(f"   ⚠️  Could not load Diffusion: {e}")
            else:
                print("   ℹ️  Diffusion not loaded (optional)")
            
            print("\n" + "="*70)
            print("✅ Graph-Enhanced CAD 3D Converter ready!")
            print("="*70)
        
        def convert_dxf_to_3d(
            self,
            dxf_path: Path,
            output_dxf: Path,
            image_path: Optional[Path] = None,
            use_diffusion: bool = False,
            normalize_range: Tuple[float, float] = (0, 1000)
        ) -> Dict[str, Any]:
            """
            تبدیل کامل DXF به 3D با استفاده از تمام قابلیت‌ها
            
            Pipeline:
            1. DXF → Graph
            2. Graph → GNN features
            3. Image → ViT features (if available)
            4. Fusion → VAE/Diffusion → 3D
            5. Save 3D DXF
            
            Args:
                dxf_path: مسیر فایل DXF ورودی
                output_dxf: مسیر خروجی
                image_path: تصویر نقشه (اختیاری)
                use_diffusion: استفاده از Diffusion برای جزئیات بیشتر
                normalize_range: محدوده نرمال‌سازی (mm)
            
            Returns:
                Dict با اطلاعات نتیجه
            """
            print("\n" + "="*70)
            print(f"Converting DXF to 3D: {dxf_path}")
            print("="*70)
            
            result = {
                'input_dxf': str(dxf_path),
                'output_dxf': str(output_dxf),
                'graph_stats': {},
                'gnn_analysis': {},
                '3d_points': None
            }
            
            # ================================================================
            # Step 1: Build Graph from DXF
            # ================================================================
            print("\n📊 Step 1: Building CAD Graph...")
            graph = self.graph_builder.build_from_dxf(dxf_path)
            
            stats = graph.get_statistics()
            result['graph_stats'] = stats
            print(f"   Elements: {stats['total_elements']}")
            print(f"   Relationships: {stats['total_relationships']}")
            
            # ================================================================
            # Step 2: GNN Analysis
            # ================================================================
            print("\n🧠 Step 2: GNN Analysis...")
            
            # Convert to PyTorch Geometric
            graph_data = graph.to_pytorch_geometric()
            if graph_data is None:
                print("   ⚠️  Could not convert to PyTorch Geometric")
                return result
            
            graph_data = graph_data.to(self.device)
            
            # GNN inference
            with torch.no_grad():
                gnn_output = self.gnn(
                    graph_data.x,
                    graph_data.edge_index,
                    graph_data.edge_attr
                )
                
                depth_output = self.depth_gnn(
                    graph_data.x,
                    graph_data.edge_index,
                    graph_data.edge_attr
                )
            
            node_embeddings = gnn_output['node_embeddings']  # [num_nodes, 256]
            depth_per_node = depth_output['depth']  # [num_nodes, 1]
            normals_per_node = depth_output['normals']  # [num_nodes, 3]
            
            print(f"   ✓ Node embeddings: {node_embeddings.shape}")
            print(f"   ✓ Depth predictions: {depth_per_node.shape}")
            print(f"   ✓ Normal predictions: {normals_per_node.shape}")
            
            result['gnn_analysis'] = {
                'num_nodes': node_embeddings.size(0),
                'embedding_dim': node_embeddings.size(1),
                'avg_depth': float(depth_per_node.mean()),
                'depth_std': float(depth_per_node.std())
            }
            
            # ================================================================
            # Step 3: ViT Features (if image available)
            # ================================================================
            vit_features = None
            
            if image_path and image_path.exists() and self.vit is not None:
                print("\n👁️ Step 3: Vision Transformer Analysis...")
                
                try:
                    from PIL import Image
                    import torchvision.transforms as transforms
                    
                    # Load and preprocess image
                    img = Image.open(image_path).convert('RGB')
                    transform = transforms.Compose([
                        transforms.Resize((256, 256)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                           std=[0.229, 0.224, 0.225])
                    ])
                    img_tensor = transform(img).unsqueeze(0).to(self.device)
                    
                    # ViT inference
                    with torch.no_grad():
                        vit_output = self.vit(img_tensor)
                    
                    # Get global features (CLS token)
                    vit_features = vit_output['cls_features']  # [1, 768]
                    
                    print(f"   ✓ ViT features: {vit_features.shape}")
                
                except Exception as e:
                    print(f"   ⚠️  ViT processing failed: {e}")
            else:
                print("\n👁️ Step 3: ViT skipped (no image or model)")
            
            # ================================================================
            # Step 4: Feature Fusion
            # ================================================================
            print("\n🔗 Step 4: Feature Fusion...")
            
            # Global pooling for graph embeddings
            graph_global = node_embeddings.mean(dim=0, keepdim=True)  # [1, 256]
            
            # Concatenate with ViT if available
            if vit_features is not None:
                fusion_input = torch.cat([graph_global, vit_features], dim=-1)
            else:
                # Pad with zeros if no ViT
                padding = torch.zeros(1, 768, device=self.device)
                fusion_input = torch.cat([graph_global, padding], dim=-1)
            
            # Fuse features
            fused_features = self.feature_fusion(fusion_input)  # [1, 512]
            
            print(f"   ✓ Fused features: {fused_features.shape}")
            
            # ================================================================
            # Step 5: 3D Generation
            # ================================================================
            print("\n🎨 Step 5: 3D Generation...")
            
            points_3d = None
            
            if use_diffusion and self.diffusion is not None:
                print("   Using Diffusion model...")
                # Diffusion generation (placeholder - needs proper implementation)
                # points_3d = self.diffusion.generate(fused_features)
                print("   ⚠️  Diffusion generation not yet implemented")
            
            elif self.vae is not None:
                print("   Using VAE decoder...")
                
                with torch.no_grad():
                    # Use fused features as latent
                    decoded = self.vae.decoder.point_decoder(fused_features)  # [1, N, 3]
                    points_3d = decoded[0].cpu().numpy()  # [N, 3]
                
                print(f"   ✓ Generated {points_3d.shape[0]} points")
            
            else:
                # Fallback: use depth from GNN
                print("   Using GNN depth predictions...")
                
                # Reconstruct 3D from graph nodes + depth
                points_3d = []
                for i, elem_id in enumerate(graph.elements.keys()):
                    elem = graph.get_element(elem_id)
                    if elem and elem.centroid:
                        x, y, _ = elem.centroid
                        z = float(depth_per_node[i] * 1000)  # Denormalize
                        points_3d.append([x, y, z])
                
                points_3d = np.array(points_3d)
                print(f"   ✓ Reconstructed {len(points_3d)} points from graph")
            
            # ================================================================
            # Step 6: Normalize and Save
            # ================================================================
            if points_3d is not None and len(points_3d) > 0:
                print("\n💾 Step 6: Saving 3D DXF...")
                
                # Normalize to specified range
                min_mm, max_mm = normalize_range
                pmin = points_3d.min(axis=0)
                pmax = points_3d.max(axis=0)
                prange = pmax - pmin
                prange[prange < 1e-6] = 1.0
                
                points_normalized = (points_3d - pmin) / prange
                points_final = points_normalized * (max_mm - min_mm) + min_mm
                
                # Save to DXF
                import ezdxf
                doc = ezdxf.new('R2010', setup=True)
                msp = doc.modelspace()
                
                # Add points as mesh (simple point cloud for now)
                for pt in points_final:
                    msp.add_point(tuple(pt))
                
                # Could also create more sophisticated mesh
                # ... (add faces, etc.)
                
                doc.saveas(output_dxf)
                
                result['3d_points'] = points_final.tolist()
                result['num_points'] = len(points_final)
                
                print(f"   ✓ Saved to {output_dxf}")
                print(f"   ✓ {len(points_final)} points")
                print(f"   ✓ Range: [{min_mm:.1f}, {max_mm:.1f}] mm")
            else:
                print("   ⚠️  No 3D points generated")
            
            print("\n" + "="*70)
            print("✅ Conversion complete!")
            print("="*70)
            
            return result
        
        def analyze_structure(self, graph: CADGraph) -> Dict[str, Any]:
            """
            تحلیل ساختاری با GNN
            
            قابلیت‌ها:
            - تشخیص نقاط ضعف
            - محاسبه تنش و کرنش (ساده‌سازی شده)
            - بررسی consistency
            """
            print("\n🔬 Structural Analysis...")
            
            graph_data = graph.to_pytorch_geometric()
            if graph_data is None:
                return {}
            
            graph_data = graph_data.to(self.device)
            
            with torch.no_grad():
                gnn_output = self.gnn(
                    graph_data.x,
                    graph_data.edge_index,
                    graph_data.edge_attr
                )
            
            analysis = {
                'total_elements': len(graph.elements),
                'total_connections': len(graph.relationships),
                'structural_analysis': {}
            }
            
            if 'structural_analysis' in gnn_output:
                struct_output = gnn_output['structural_analysis']  # [num_nodes, 3]
                
                # Extract stress, strain, displacement
                stress = struct_output[:, 0].cpu().numpy()
                strain = struct_output[:, 1].cpu().numpy()
                displacement = struct_output[:, 2].cpu().numpy()
                
                analysis['structural_analysis'] = {
                    'max_stress': float(stress.max()),
                    'avg_stress': float(stress.mean()),
                    'max_strain': float(strain.max()),
                    'avg_strain': float(strain.mean()),
                    'max_displacement': float(displacement.max())
                }
                
                print(f"   Max Stress: {stress.max():.4f}")
                print(f"   Max Strain: {strain.max():.4f}")
                print(f"   Max Displacement: {displacement.max():.4f}")
            
            return analysis


else:
    # Placeholder if dependencies not available
    class GraphEnhancedCAD3DConverter:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch and PyTorch Geometric required")


# ============================================================================
# Command Line Interface
# ============================================================================

def main():
    """CLI برای تست سیستم"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Graph-Enhanced CAD 2D→3D Converter")
    parser.add_argument('--input-dxf', type=Path, required=True, help='Input DXF file')
    parser.add_argument('--output-dxf', type=Path, required=True, help='Output 3D DXF file')
    parser.add_argument('--image', type=Path, help='Optional: 2D image of drawing')
    parser.add_argument('--vit-model', type=Path, help='ViT model checkpoint')
    parser.add_argument('--vae-model', type=Path, help='VAE model checkpoint')
    parser.add_argument('--diffusion-model', type=Path, help='Diffusion model checkpoint')
    parser.add_argument('--use-diffusion', action='store_true', help='Use diffusion for detail')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'], help='Device')
    parser.add_argument('--min-mm', type=float, default=0, help='Min coordinate (mm)')
    parser.add_argument('--max-mm', type=float, default=1000, help='Max coordinate (mm)')
    
    args = parser.parse_args()
    
    if not TORCH_AVAILABLE or not PYTORCH_GEOMETRIC_AVAILABLE:
        print("❌ PyTorch and PyTorch Geometric required")
        print("Install: pip install torch torch-geometric")
        return
    
    # Create converter
    converter = GraphEnhancedCAD3DConverter(
        device=args.device,
        vit_model_path=args.vit_model,
        vae_model_path=args.vae_model,
        diffusion_model_path=args.diffusion_model
    )
    
    # Convert
    result = converter.convert_dxf_to_3d(
        dxf_path=args.input_dxf,
        output_dxf=args.output_dxf,
        image_path=args.image,
        use_diffusion=args.use_diffusion,
        normalize_range=(args.min_mm, args.max_mm)
    )
    
    print("\n📊 Result Summary:")
    print(f"   Graph: {result['graph_stats']['total_elements']} elements, "
          f"{result['graph_stats']['total_relationships']} relationships")
    if result.get('num_points'):
        print(f"   3D: {result['num_points']} points generated")


if __name__ == "__main__":
    main()
