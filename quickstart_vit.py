"""
Quick Start Script for Vision Transformer
اسکریپت شروع سریع برای Vision Transformer

این اسکریپت به شما کمک می‌کند Vision Transformer را سریع تست کنید
"""

import sys
import os

def main():
    print("="*60)
    print("Vision Transformer Quick Start")
    print("="*60)
    
    # Step 1: Check Python version
    print("\n1️⃣ Checking Python version...")
    version = sys.version_info
    print(f"   Python {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("   ⚠️ Python 3.8+ required")
        return
    print("   ✓ Python version OK")
    
    # Step 2: Check PyTorch
    print("\n2️⃣ Checking PyTorch...")
    try:
        import torch
        print(f"   ✓ PyTorch {torch.__version__} installed")
        
        # Check CUDA
        if torch.cuda.is_available():
            print(f"   ✓ CUDA available (GPU: {torch.cuda.get_device_name(0)})")
            device = "cuda"
        else:
            print("   ℹ️ CUDA not available (using CPU)")
            device = "cpu"
    except ImportError:
        print("   ✗ PyTorch not installed")
        print("\n📦 Install PyTorch:")
        print("   For CPU:")
        print("   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu")
        print("\n   For GPU (CUDA 11.8):")
        print("   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
        return
    
    # Step 3: Check other dependencies
    print("\n3️⃣ Checking dependencies...")
    deps = {
        'ezdxf': 'DXF file handling',
        'cv2': 'Image processing',
        'matplotlib': 'Visualization',
        'scipy': 'Scientific computing'
    }
    
    missing = []
    for module, desc in deps.items():
        try:
            if module == 'cv2':
                import cv2
            elif module == 'ezdxf':
                import ezdxf
            elif module == 'matplotlib':
                import matplotlib
            elif module == 'scipy':
                import scipy
            print(f"   ✓ {module:12s} ({desc})")
        except ImportError:
            print(f"   ✗ {module:12s} ({desc})")
            missing.append(module if module != 'cv2' else 'opencv-python')
    
    if missing:
        print(f"\n   Install missing: pip install {' '.join(missing)}")
        return
    
    # Step 4: Test Vision Transformer
    print("\n4️⃣ Testing Vision Transformer...")
    
    try:
        from cad3d.vision_transformer_cad import VisionTransformerCAD, VisionTransformerConfig
        print("   ✓ Vision Transformer module imported")
        
        # Create model
        config = VisionTransformerConfig(
            image_size=256,  # Small for quick test
            patch_size=16,
            dim=384,
            depth=6,
            heads=6
        )
        
        model = VisionTransformerCAD(config)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"   ✓ Model created ({total_params:,} parameters)")
        
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return
    
    # Step 5: Test analyzer
    print("\n5️⃣ Testing CAD Analyzer...")
    
    try:
        from cad3d.vision_transformer_cad import CADVisionAnalyzer
        import numpy as np
        
        analyzer = CADVisionAnalyzer(device=device)
        print(f"   ✓ Analyzer initialized (device: {device})")
        
        # Create dummy image
        dummy_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        
        # Analyze (this will be slow on first run)
        print("   ⏳ Running inference (first run may be slow)...")
        results = analyzer.analyze_image(dummy_image)
        
        print(f"   ✓ Analysis complete")
        print(f"      - Detected {len(results['elements'])} elements")
        print(f"      - Semantic map shape: {results['semantic_map'].shape}")
        
    except Exception as e:
        print(f"   ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 6: Test 3D reconstructor
    print("\n6️⃣ Testing 3D Reconstructor...")
    
    try:
        from cad3d.advanced_3d_reconstructor import Advanced3DReconstructor
        import tempfile
        import cv2 as cv
        
        reconstructor = Advanced3DReconstructor(device=device)
        print(f"   ✓ Reconstructor initialized")
        
        # Create test image with shapes
        test_image = np.zeros((256, 256, 3), dtype=np.uint8)
        cv.rectangle(test_image, (50, 50), (200, 200), (200, 200, 200), -1)
        
        # Create temp files
        fd_in, temp_input = tempfile.mkstemp(suffix=".png")
        os.close(fd_in)
        cv.imwrite(temp_input, test_image)
        
        fd_out, temp_output = tempfile.mkstemp(suffix=".dxf")
        os.close(fd_out)
        
        print("   ⏳ Reconstructing 3D...")
        stats = reconstructor.reconstruct_from_image(
            test_image,
            temp_output,
            auto_scale=False,
            min_confidence=0.3
        )
        
        print(f"   ✓ 3D reconstruction complete")
        print(f"      - Generated {stats['total_entities']} entities")
        print(f"      - Created {stats['total_layers']} layers")
        
        # Cleanup
        os.unlink(temp_input)
        os.unlink(temp_output)
        
    except Exception as e:
        print(f"   ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Success!
    print("\n" + "="*60)
    print("✅ All tests passed!")
    print("="*60)
    
    print("\n🎉 Vision Transformer is ready to use!")
    print("\n📖 Next steps:")
    print("   1. Read documentation: cat VISION_TRANSFORMER_GUIDE.md")
    print("   2. Run full demo: python demo_vit.py")
    print("   3. Try with your images:")
    print("      from cad3d.vit_integration import get_vit_service")
    print("      service = get_vit_service()")
    print("      service.convert_image_to_3d_dxf('your_image.jpg', 'output.dxf')")
    
    print("\n💡 For training your own model:")
    print("   python -m cad3d.vit_trainer")
    
    print("\n🚀 Happy coding!")


if __name__ == "__main__":
    main()
