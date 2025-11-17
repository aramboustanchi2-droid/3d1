"""
Tests for Vision Transformer Integration
"""

import pytest


def test_vit_imports():
    """تست: import های ViT"""
    from cad3d.vit_detector import (
        ViTConfig,
        CADVisionTransformer,
        CADViTDetector,
        create_vit_for_cad
    )
    assert ViTConfig is not None
    assert create_vit_for_cad is not None


def test_vit_config():
    """تست: تنظیمات ViT"""
    from cad3d.vit_detector import ViTConfig
    
    config = ViTConfig(
        image_size=512,
        patch_size=16,
        num_classes=15
    )
    
    assert config.image_size == 512
    assert config.patch_size == 16
    assert config.num_classes == 15
    assert config.hidden_size == 768
    assert config.num_attention_heads == 12


def test_vit_model_creation():
    """تست: ساخت مدل ViT"""
    try:
        import torch
        from cad3d.vit_detector import CADVisionTransformer, ViTConfig
        
        config = ViTConfig(image_size=256, patch_size=16, num_classes=15)
        model = CADVisionTransformer(config)
        
        assert model is not None
        
        # تست forward pass
        batch_size = 2
        dummy_input = torch.randn(batch_size, 3, 256, 256)
        outputs = model(dummy_input)
        
        assert 'logits' in outputs
        assert 'bbox_predictions' in outputs
        assert outputs['logits'].shape == (batch_size, 15)
        
        print("\n✅ ViT Model:")
        print(f"   Input: {dummy_input.shape}")
        print(f"   Logits: {outputs['logits'].shape}")
        print(f"   BBox: {outputs['bbox_predictions'].shape}")
        
    except ImportError:
        pytest.skip("PyTorch not available")


def test_vit_detector_creation():
    """تست: ساخت detector"""
    try:
        from cad3d.vit_detector import create_vit_for_cad
        
        detector = create_vit_for_cad(num_classes=15, device='cpu')
        assert detector is not None
        assert len(detector.class_names) == 15
        
        print(f"\n✅ Detector created on {detector.device}")
        
    except ImportError:
        pytest.skip("PyTorch not available")


def test_vit_system_summary():
    """تست: خلاصه سیستم ViT"""
    from cad3d.vit_detector import ViTConfig
    
    config = ViTConfig()
    num_patches = (config.image_size // config.patch_size) ** 2
    
    print("\n" + "="*60)
    print("Vision Transformer (ViT) for CAD")
    print("="*60)
    print(f"✅ Architecture:")
    print(f"   Image Size: {config.image_size}x{config.image_size}")
    print(f"   Patch Size: {config.patch_size}x{config.patch_size}")
    print(f"   Number of Patches: {num_patches}")
    print(f"   Hidden Size: {config.hidden_size}")
    print(f"   Attention Heads: {config.num_attention_heads}")
    print(f"   Transformer Layers: {config.num_hidden_layers}")
    print(f"   Number of Classes: {config.num_classes}")
    print(f"\n✅ Advantages:")
    print(f"   - Global context understanding")
    print(f"   - Long-range dependencies")
    print(f"   - Relationship modeling via attention")
    print(f"   - Better for complex layouts")
    print(f"\n✅ Use Cases:")
    print(f"   - Large architectural plans")
    print(f"   - Complex MEP systems")
    print(f"   - Multi-floor buildings")
    print(f"   - Detailed construction drawings")
    print("="*60)


if __name__ == "__main__":
    test_vit_system_summary()
