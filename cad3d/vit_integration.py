"""
Vision Transformer Integration for Simple Server
ادغام Vision Transformer در سرور ساده

این ماژول امکان استفاده از Vision Transformer را برای تبدیل پیشرفته فراهم می‌کند
"""

from pathlib import Path
import numpy as np
import cv2
from typing import Optional

# Check availability
VIT_AVAILABLE = False
try:
    from .vision_transformer_cad import CADVisionAnalyzer, TORCH_AVAILABLE
    from .advanced_3d_reconstructor import Advanced3DReconstructor
    if TORCH_AVAILABLE:
        VIT_AVAILABLE = True
except ImportError:
    pass


class VITConversionService:
    """
    Service for Vision Transformer-based conversion
    
    Features:
    - Deep semantic understanding
    - Intelligent 3D reconstruction
    - Material and height prediction
    - Auto-scale detection
    """
    
    def __init__(self, model_path: Optional[str] = None, device: str = "cpu"):
        """
        Initialize VIT conversion service
        
        Args:
            model_path: Path to pretrained model (optional)
            device: 'cpu', 'cuda', or 'auto'
        """
        if not VIT_AVAILABLE:
            raise RuntimeError("Vision Transformer not available. Install PyTorch: pip install torch torchvision")
        
        self.reconstructor = Advanced3DReconstructor(
            vit_model_path=model_path,
            device=device
        )
        
        self.enabled = True
        print("✓ Vision Transformer service initialized")
    
    def convert_image_to_3d_dxf(
        self,
        image_path: str,
        output_dxf: str,
        min_confidence: float = 0.5,
        auto_scale: bool = True
    ) -> dict:
        """
        Convert image to 3D DXF using Vision Transformer
        
        Args:
            image_path: Input image file
            output_dxf: Output DXF file
            min_confidence: Minimum confidence for element detection
            auto_scale: Try to detect scale automatically
        
        Returns:
            Statistics dictionary
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Reconstruct
        stats = self.reconstructor.reconstruct_from_image(
            image,
            output_dxf,
            auto_scale=auto_scale,
            min_confidence=min_confidence
        )
        
        return stats
    
    def analyze_image(self, image_path: str) -> dict:
        """
        Analyze image and return detailed results
        
        Args:
            image_path: Input image file
        
        Returns:
            Analysis results dictionary
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Analyze
        analysis = self.reconstructor.analyzer.analyze_image(image)
        
        # Format results
        results = {
            'num_elements': len(analysis['elements']),
            'elements': analysis['elements'],
            'has_depth_map': 'depth_map' in analysis,
            'has_height_map': 'height_map' in analysis,
            'has_material_map': 'material_map' in analysis,
            'confidence_stats': {
                'mean': float(analysis['confidence_map'].mean()),
                'max': float(analysis['confidence_map'].max()),
                'min': float(analysis['confidence_map'].min())
            }
        }
        
        return results
    
    def create_visualization(
        self,
        image_path: str,
        output_path: str
    ):
        """
        Create visualization of Vision Transformer analysis
        
        Args:
            image_path: Input image
            output_path: Output visualization image
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        self.reconstructor.visualize_analysis(image, output_path)


# Global service instance
_vit_service: Optional[VITConversionService] = None


def get_vit_service(
    model_path: Optional[str] = None,
    device: str = "cpu"
) -> Optional[VITConversionService]:
    """
    Get or create VIT service instance
    
    Args:
        model_path: Path to pretrained model
        device: Device to use
    
    Returns:
        VITConversionService instance or None if not available
    """
    global _vit_service
    
    if not VIT_AVAILABLE:
        return None
    
    if _vit_service is None:
        try:
            _vit_service = VITConversionService(
                model_path=model_path,
                device=device
            )
        except Exception as e:
            print(f"Could not initialize VIT service: {e}")
            return None
    
    return _vit_service


def is_vit_available() -> bool:
    """Check if Vision Transformer is available"""
    return VIT_AVAILABLE


# Example usage
if __name__ == "__main__":
    if VIT_AVAILABLE:
        print("Vision Transformer is available!")
        
        # Test service
        service = get_vit_service(device="cpu")
        
        if service:
            print("Service initialized successfully")
            
            # Test with dummy image
            test_img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
            cv2.imwrite("test_input.png", cv2.cvtColor(test_img, cv2.COLOR_RGB2BGR))
            
            try:
                stats = service.convert_image_to_3d_dxf(
                    "test_input.png",
                    "test_output.dxf",
                    auto_scale=True
                )
                print(f"Conversion stats: {stats}")
            except Exception as e:
                print(f"Conversion failed: {e}")
    else:
        print("Vision Transformer not available.")
        print("Install PyTorch: pip install torch torchvision")
