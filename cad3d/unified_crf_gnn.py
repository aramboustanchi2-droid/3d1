"""
Unified CRF + GNN System for Industrial CAD
سیستم یکپارچه CRF + GNN برای نقشه‌های صنعتی

این سیستم ترکیب قدرتمندی از:
1. CNN/U-Net: Initial segmentation
2. CRF: Boundary refinement
3. GNN: Relationship analysis & structural understanding

Pipeline کامل:
Image → CNN → CRF (refined segmentation) → Graph Builder → GNN → 3D + Analysis

استفاده برای تمام صنایع:
✅ ساختمان‌سازی (Building) - دیوار، ستون، تیر
✅ پل‌سازی (Bridge) - تیر، تکیه‌گاه، عرشه
✅ جاده‌سازی (Road) - خطوط، مسیر، لاین‌ها
✅ سدسازی (Dam) - بدنه، پی، سرریز
✅ تونل‌سازی (Tunnel) - پوشش، مقطع
✅ کارخانه (Factory) - تجهیزات، ماشین‌آلات
✅ ماشین‌سازی (Machinery) - قطعات، اتصالات
✅ تاسیسات (MEP) - لوله، کانال، کابل
✅ هر صنعتی با نقشه فنی

مزایای این سیستم:
- Segmentation دقیق با CRF
- درک روابط با GNN
- تحلیل ساختاری کامل
- Uncertainty quantification
- Industry-specific intelligence
"""

from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Union
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

# Import our modules
from .crf_segmentation import CRFEnhancedSegmentation, DenseCRF2D, create_simple_unet
from .industrial_gnn import IndustrySpecificGNN, IndustryType
from .cad_graph import CADGraph, CADElement, ElementType, RelationType
from .cad_graph_builder import CADGraphBuilder


if TORCH_AVAILABLE and PYTORCH_GEOMETRIC_AVAILABLE:
    
    class UnifiedCADAnalyzer(nn.Module):
        """
        سیستم یکپارچه تحلیل نقشه‌های CAD
        
        این کلاس همه چیز را یکپارچه می‌کند:
        1. Image → Segmentation (CNN + CRF)
        2. Segmentation → Graph (Graph Builder)
        3. Graph → Analysis (GNN)
        4. Analysis → 3D + Engineering Validation
        
        قابلیت‌ها:
        - تشخیص خودکار عناصر
        - مرزبندی دقیق با CRF
        - تحلیل روابط با GNN
        - محاسبات مهندسی
        - خروجی 3D با کیفیت بالا
        """
        
        def __init__(
            self,
            industry: str = "building",
            num_classes: int = 50,
            hidden_dim: int = 256,
            device: str = "cpu",
            use_crf: bool = True,
            vae_model_path: Optional[Path] = None,
            pretrained_segmentation: Optional[Path] = None
        ):
            """
            Initialize unified system
            
            Args:
                industry: نوع صنعت (building, bridge, road, ...)
                num_classes: تعداد کلاس‌های segmentation
                hidden_dim: dimension مدل GNN
                device: cpu or cuda
                use_crf: استفاده از CRF برای refinement
                vae_model_path: مسیر مدل VAE (برای 3D)
                pretrained_segmentation: مسیر checkpoint مدل segmentation
            """
            super().__init__()
            
            self.industry = industry
            self.num_classes = num_classes
            self.device = device
            
            print("="*70)
            print(f"Initializing Unified CAD Analyzer for {industry.upper()}")
            print("="*70)
            
            # ================================================================
            # 1. Segmentation Module (CNN + CRF)
            # ================================================================
            print("\n1️⃣ Loading Segmentation Module (CNN + CRF)...")
            
            # Create U-Net backbone
            backbone = create_simple_unet(num_classes=num_classes)
            
            # Wrap with CRF
            self.segmentation = CRFEnhancedSegmentation(
                backbone=backbone,
                num_classes=num_classes,
                use_crf=use_crf
            ).to(device)
            
            # Load pretrained if available
            if pretrained_segmentation and pretrained_segmentation.exists():
                checkpoint = torch.load(pretrained_segmentation, map_location=device)
                self.segmentation.load_state_dict(checkpoint['model_state_dict'])
                print(f"   ✓ Loaded pretrained segmentation from {pretrained_segmentation}")
            else:
                print(f"   ✓ Using randomly initialized segmentation")
            
            # ================================================================
            # 2. Graph Builder
            # ================================================================
            print("\n2️⃣ Initializing Graph Builder...")
            
            self.graph_builder = CADGraphBuilder(
                proximity_threshold=100.0,  # 100mm
                parallel_angle_threshold=5.0  # 5 degrees
            )
            
            print("   ✓ Graph Builder ready")
            
            # ================================================================
            # 3. Industry-Specific GNN
            # ================================================================
            print("\n3️⃣ Loading Industry-Specific GNN...")
            
            try:
                industry_type = IndustryType(industry.lower())
            except ValueError:
                print(f"   ⚠️  Unknown industry '{industry}', using GENERAL")
                industry_type = IndustryType.GENERAL
            
            self.gnn = IndustrySpecificGNN(
                industry=industry_type,
                node_features=56,  # Standard node features
                edge_features=21,  # Standard edge features
                hidden_dim=hidden_dim,
                num_layers=4,
                num_heads=8
            ).to(device)
            
            print(f"   ✓ {industry_type.value.upper()} GNN ready")
            
            # ================================================================
            # 4. VAE Decoder (optional, for 3D)
            # ================================================================
            print("\n4️⃣ Loading VAE Decoder (optional)...")
            
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
                print("   ℹ️  VAE not loaded (3D generation unavailable)")
            
            print("\n" + "="*70)
            print("✅ Unified CAD Analyzer ready!")
            print("="*70)
        
        def analyze_image(
            self,
            image: Union[Tensor, np.ndarray],  # [H, W, 3] or [batch, 3, H, W]
            return_intermediate: bool = False,
            generate_3d: bool = False
        ) -> Dict[str, Any]:
            """
            تحلیل کامل نقشه از روی تصویر
            
            Pipeline:
            1. Image → Segmentation (CNN)
            2. Segmentation → Refined (CRF)
            3. Refined → Graph
            4. Graph → GNN Analysis
            5. (Optional) → 3D Generation
            
            Args:
                image: تصویر ورودی
                return_intermediate: برگرداندن نتایج مراحل میانی
                generate_3d: تولید مدل 3D
            
            Returns:
                Dict با نتایج تحلیل
            """
            print("\n" + "="*70)
            print("Analyzing CAD Drawing")
            print("="*70)
            
            result = {}
            
            # Prepare image
            if isinstance(image, np.ndarray):
                if image.ndim == 3:  # [H, W, 3]
                    image_rgb = image.copy()
                    image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                else:  # Already [batch, 3, H, W]
                    image_tensor = torch.from_numpy(image).float()
                    image_rgb = (image_tensor[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            else:
                image_tensor = image
                image_rgb = (image_tensor[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            
            image_tensor = image_tensor.to(self.device)
            
            # ================================================================
            # Step 1: Segmentation with CRF
            # ================================================================
            print("\n📸 Step 1: Image Segmentation (CNN + CRF)...")
            
            with torch.no_grad():
                segmentation_map = self.segmentation.predict(
                    image_tensor,
                    images_rgb=image_rgb[np.newaxis, ...],
                    use_crf=True
                )
            
            segmentation_np = segmentation_map[0].cpu().numpy()
            
            print(f"   ✓ Segmentation complete: {segmentation_np.shape}")
            print(f"   ✓ Detected {len(np.unique(segmentation_np))} classes")
            
            result['segmentation'] = segmentation_np
            
            # ================================================================
            # Step 2: Build Graph from Segmentation
            # ================================================================
            print("\n📊 Step 2: Building CAD Graph from Segmentation...")
            
            graph = self._segmentation_to_graph(segmentation_np)
            
            stats = graph.get_statistics()
            print(f"   ✓ Graph built:")
            print(f"      - Elements: {stats['total_elements']}")
            print(f"      - Relationships: {stats['total_relationships']}")
            
            result['graph'] = graph
            result['graph_stats'] = stats
            
            # ================================================================
            # Step 3: GNN Analysis
            # ================================================================
            print("\n🧠 Step 3: GNN Analysis...")
            
            graph_data = graph.to_pytorch_geometric()
            
            if graph_data is not None:
                graph_data = graph_data.to(self.device)
                
                with torch.no_grad():
                    gnn_output = self.gnn(
                        graph_data.x,
                        graph_data.edge_index,
                        graph_data.edge_attr
                    )
                
                # Parse GNN output based on industry
                analysis = self._parse_gnn_output(gnn_output)
                
                print(f"   ✓ GNN analysis complete")
                
                for key, value in analysis.items():
                    if isinstance(value, (int, float)):
                        print(f"      - {key}: {value:.4f}")
                    elif isinstance(value, np.ndarray):
                        print(f"      - {key}: shape {value.shape}")
                
                result['gnn_analysis'] = analysis
            else:
                print("   ⚠️  Could not convert to PyTorch Geometric")
                result['gnn_analysis'] = {}
            
            # ================================================================
            # Step 4: 3D Generation (optional)
            # ================================================================
            if generate_3d and self.vae is not None:
                print("\n🎨 Step 4: 3D Generation...")
                
                # Use GNN embeddings as latent code
                if 'node_embeddings' in gnn_output:
                    latent = gnn_output['node_embeddings'].mean(dim=0, keepdim=True)  # [1, 256]
                    
                    # Upsample to VAE latent dim if needed
                    if latent.size(-1) != 512:
                        latent_proj = nn.Linear(latent.size(-1), 512).to(self.device)
                        latent = latent_proj(latent)
                    
                    with torch.no_grad():
                        points_3d = self.vae.decoder.point_decoder(latent)  # [1, N, 3]
                    
                    points_3d = points_3d[0].cpu().numpy()
                    
                    print(f"   ✓ Generated {points_3d.shape[0]} 3D points")
                    
                    result['points_3d'] = points_3d
            
            # ================================================================
            # Summary
            # ================================================================
            print("\n" + "="*70)
            print("✅ Analysis Complete!")
            print("="*70)
            print(f"Industry: {self.industry.upper()}")
            print(f"Classes Detected: {len(np.unique(segmentation_np))}")
            print(f"Elements: {stats['total_elements']}")
            print(f"Relationships: {stats['total_relationships']}")
            
            if 'points_3d' in result:
                print(f"3D Points: {len(result['points_3d'])}")
            
            return result
        
        def _segmentation_to_graph(self, segmentation: np.ndarray) -> CADGraph:
            """
            تبدیل segmentation map به CAD graph
            
            برای هر منطقه متصل:
            - یک CADElement بساز
            - روابط فضایی را تشخیص بده
            """
            from scipy import ndimage
            
            graph = CADGraph()
            
            # Find connected components for each class
            unique_classes = np.unique(segmentation)
            unique_classes = unique_classes[unique_classes > 0]  # Skip background
            
            element_id_counter = 0
            
            for class_id in unique_classes:
                # Find connected components
                mask = (segmentation == class_id).astype(int)
                labeled, num_features = ndimage.label(mask)
                
                for i in range(1, num_features + 1):
                    component_mask = (labeled == i)
                    
                    # Compute properties
                    coords = np.argwhere(component_mask)
                    
                    if len(coords) < 5:  # Too small
                        continue
                    
                    # Centroid
                    centroid_y, centroid_x = coords.mean(axis=0)
                    
                    # Bounding box
                    y_min, x_min = coords.min(axis=0)
                    y_max, x_max = coords.max(axis=0)
                    
                    # Create element
                    element = CADElement(
                        id=f"elem_{element_id_counter:04d}",
                        element_type=self._class_to_element_type(int(class_id)),
                        centroid=(float(centroid_x), float(centroid_y), 0.0),
                        properties={
                            'class_id': int(class_id),
                            'area': int(len(coords)),
                            'width': float(x_max - x_min),
                            'height': float(y_max - y_min)
                        },
                        bounding_box=(
                            (float(x_min), float(y_min), 0.0),
                            (float(x_max), float(y_max), 100.0)  # Assume 100mm height
                        )
                    )
                    
                    graph.add_element(element)
                    element_id_counter += 1
            
            # Detect relationships (simplified)
            elements = list(graph.elements.values())
            
            for i, elem1 in enumerate(elements):
                for elem2 in elements[i+1:]:
                    # Check proximity
                    if elem1.centroid and elem2.centroid:
                        dist = np.linalg.norm(
                            np.array(elem1.centroid[:2]) - np.array(elem2.centroid[:2])
                        )
                        
                        if dist < 200:  # Within 200 pixels
                            graph.add_relationship(
                                source_id=elem1.id,
                                target_id=elem2.id,
                                relation_type=RelationType.ADJACENT,
                                weight=1.0 / (1.0 + dist / 100)
                            )
            
            return graph
        
        def _class_to_element_type(self, class_id: int) -> ElementType:
            """Map segmentation class to element type"""
            # This is a simplified mapping
            # In practice, you'd have a proper class→element mapping
            
            mapping = {
                1: ElementType.WALL,
                2: ElementType.COLUMN,
                3: ElementType.BEAM,
                4: ElementType.DOOR,
                5: ElementType.WINDOW,
                6: ElementType.SLAB,
                7: ElementType.STAIR,
                8: ElementType.RAILING,
                # Add more mappings...
            }
            
            return mapping.get(class_id, ElementType.UNKNOWN)
        
        def _parse_gnn_output(self, gnn_output: Dict[str, Tensor]) -> Dict[str, Any]:
            """Parse GNN output into engineering metrics"""
            
            analysis = {}
            
            # Extract embeddings
            if 'node_embeddings' in gnn_output:
                embeddings = gnn_output['node_embeddings'].cpu().numpy()
                analysis['embeddings_mean'] = float(embeddings.mean())
                analysis['embeddings_std'] = float(embeddings.std())
            
            # Industry-specific metrics
            if self.industry == "building":
                if 'load_capacity' in gnn_output:
                    loads = gnn_output['load_capacity'].cpu().numpy()
                    analysis['max_load'] = float(loads.max())
                    analysis['avg_load'] = float(loads.mean())
            
            elif self.industry == "bridge":
                if 'stress' in gnn_output:
                    stress = gnn_output['stress'].cpu().numpy()
                    analysis['max_stress'] = float(stress[:, 0].max())
                    analysis['max_shear'] = float(stress[:, 1].max())
            
            elif self.industry == "road":
                if 'traffic' in gnn_output:
                    traffic = gnn_output['traffic'].cpu().numpy()
                    analysis['avg_capacity'] = float(traffic[:, 0].mean())
            
            # Add more industry-specific parsing...
            
            return analysis


else:
    # Placeholder
    class UnifiedCADAnalyzer:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch and PyTorch Geometric required")


# ============================================================================
# Command Line Interface
# ============================================================================

def main():
    """CLI for testing"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Unified CRF + GNN CAD Analyzer")
    parser.add_argument('--image', type=Path, required=True, help='Input image')
    parser.add_argument('--industry', default='building', help='Industry type')
    parser.add_argument('--output-dir', type=Path, default=Path('.'), help='Output directory')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('--generate-3d', action='store_true', help='Generate 3D model')
    
    args = parser.parse_args()
    
    if not TORCH_AVAILABLE or not PYTORCH_GEOMETRIC_AVAILABLE:
        print("❌ PyTorch and PyTorch Geometric required")
        return
    
    # Load image
    from PIL import Image
    img = Image.open(args.image).convert('RGB')
    img_np = np.array(img)
    
    # Create analyzer
    analyzer = UnifiedCADAnalyzer(
        industry=args.industry,
        device=args.device
    )
    
    # Analyze
    result = analyzer.analyze_image(
        image=img_np,
        generate_3d=args.generate_3d
    )
    
    # Save results
    output_dir = args.output_dir
    output_dir.mkdir(exist_ok=True)
    
    # Save segmentation
    if 'segmentation' in result:
        seg_path = output_dir / "segmentation.npy"
        np.save(seg_path, result['segmentation'])
        print(f"\n💾 Segmentation saved to: {seg_path}")
    
    # Save graph
    if 'graph' in result:
        graph_path = output_dir / "graph.json"
        result['graph'].save_json(graph_path)
        print(f"💾 Graph saved to: {graph_path}")
    
    # Save 3D points
    if 'points_3d' in result:
        points_path = output_dir / "points_3d.npy"
        np.save(points_path, result['points_3d'])
        print(f"💾 3D points saved to: {points_path}")
    
    print("\n✅ All results saved!")


if __name__ == "__main__":
    main()
