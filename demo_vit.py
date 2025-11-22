"""
Vision Transformer Demo Script
اسکریپت نمایشی Vision Transformer

این اسکریپت قابلیت‌های Vision Transformer را نمایش می‌دهد
"""

import sys
from pathlib import Path
import argparse

def check_dependencies():
    """Check if all dependencies are installed"""
    missing = []
    
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
    except ImportError:
        missing.append("torch")
        print("✗ PyTorch not installed")
    
    try:
        import torchvision
        print(f"✓ torchvision {torchvision.__version__}")
    except ImportError:
        missing.append("torchvision")
        print("✗ torchvision not installed")
    
    try:
        import cv2
        print(f"✓ OpenCV {cv2.__version__}")
    except ImportError:
        missing.append("opencv-python")
        print("✗ OpenCV not installed")
    
    try:
        import ezdxf
        print(f"✓ ezdxf {ezdxf.__version__}")
    except ImportError:
        missing.append("ezdxf")
        print("✗ ezdxf not installed")
    
    try:
        import matplotlib
        print(f"✓ matplotlib {matplotlib.__version__}")
    except ImportError:
        missing.append("matplotlib")
        print("✗ matplotlib not installed")
    
    try:
        import scipy
        print(f"✓ scipy {scipy.__version__}")
    except ImportError:
        missing.append("scipy")
        print("✗ scipy not installed")
    
    if missing:
        print("\n⚠️ Missing dependencies:")
        print("Install with:")
        print(f"pip install {' '.join(missing)}")
        return False
    
    return True


def demo_model_creation():
    """Demo: Create and inspect Vision Transformer model"""
    print("\n" + "="*60)
    print("DEMO 1: Vision Transformer Model")
    print("="*60)
    
    from cad3d.vision_transformer_cad import VisionTransformerCAD, VisionTransformerConfig
    
    # Create model
    config = VisionTransformerConfig(
        image_size=512,
        patch_size=16,
        num_classes=50,
        dim=768,
        depth=12,
        heads=12
    )
    
    model = VisionTransformerCAD(config)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model created")
    print(f"  - Input size: {config.image_size}x{config.image_size}")
    print(f"  - Patch size: {config.patch_size}x{config.patch_size}")
    print(f"  - Number of patches: {(config.image_size // config.patch_size)**2}")
    print(f"  - Embedding dim: {config.dim}")
    print(f"  - Transformer layers: {config.depth}")
    print(f"  - Attention heads: {config.heads}")
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Model size: ~{total_params * 4 / 1024 / 1024:.1f} MB")


def demo_image_analysis():
    """Demo: Analyze image with Vision Transformer"""
    print("\n" + "="*60)
    print("DEMO 2: Image Analysis")
    print("="*60)
    
    from cad3d.vision_transformer_cad import CADVisionAnalyzer
    import numpy as np
    import cv2
    
    # Create analyzer
    analyzer = CADVisionAnalyzer(device="cpu")
    print("✓ Analyzer initialized")
    
    # Create a dummy image with some structure
    image = np.zeros((512, 512, 3), dtype=np.uint8)
    
    # Draw some rectangles (simulating floor plan)
    cv2.rectangle(image, (50, 50), (450, 450), (200, 200, 200), -1)  # Outer wall
    cv2.rectangle(image, (100, 200), (200, 350), (139, 69, 19), -1)  # Door
    cv2.rectangle(image, (250, 150), (350, 250), (135, 206, 250), -1)  # Window
    cv2.rectangle(image, (350, 300), (400, 400), (128, 128, 128), -1)  # Column
    
    print("✓ Test image created (simulated floor plan)")
    
    # Analyze
    print("🔍 Analyzing image...")
    results = analyzer.analyze_image(image)
    
    print(f"✓ Analysis complete")
    print(f"  - Detected elements: {len(results['elements'])}")
    print(f"  - Semantic map shape: {results['semantic_map'].shape}")
    print(f"  - Confidence map shape: {results['confidence_map'].shape}")
    if 'depth_map' in results:
        print(f"  - Depth map shape: {results['depth_map'].shape}")
    if 'height_map' in results:
        print(f"  - Height map shape: {results['height_map'].shape}")
    
    # Show detected elements
    if results['elements']:
        print(f"\n📋 Top detected elements:")
        for i, elem in enumerate(results['elements'][:10], 1):
            print(f"  {i}. {elem['class']} (confidence: {elem['confidence']:.2%})")


def demo_3d_reconstruction():
    """Demo: 3D reconstruction from image"""
    print("\n" + "="*60)
    print("DEMO 3: 3D Reconstruction")
    print("="*60)
    
    from cad3d.advanced_3d_reconstructor import Advanced3DReconstructor
    import numpy as np
    import cv2
    import tempfile
    import os
    
    # Create reconstructor
    reconstructor = Advanced3DReconstructor(device="cpu")
    print("✓ Reconstructor initialized")
    
    # Create test image
    image = np.zeros((512, 512, 3), dtype=np.uint8)
    cv2.rectangle(image, (100, 100), (400, 400), (200, 200, 200), -1)
    cv2.rectangle(image, (150, 250), (200, 350), (139, 69, 19), -1)
    
    # Save to temp file
    fd, temp_image = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    cv2.imwrite(temp_image, image)
    
    # Create output file
    fd, temp_output = tempfile.mkstemp(suffix=".dxf")
    os.close(fd)
    
    print("🏗️ Reconstructing 3D model...")
    
    try:
        stats = reconstructor.reconstruct_from_image(
            cv2.imread(temp_image),
            temp_output,
            auto_scale=False,
            min_confidence=0.3
        )
        
        print(f"✓ 3D reconstruction complete")
        print(f"  - Total entities: {stats['total_entities']}")
        print(f"  - Total layers: {stats['total_layers']}")
        print(f"  - Elements by class:")
        for class_name, count in stats['elements_by_class'].items():
            print(f"      {class_name}: {count}")
        
        # Check output file
        import ezdxf
        doc = ezdxf.readfile(temp_output)
        entities = list(doc.modelspace())
        print(f"  - Verified: {len(entities)} entities in DXF")
        
    except Exception as e:
        print(f"✗ Reconstruction failed: {e}")
    
    finally:
        # Cleanup
        try:
            os.unlink(temp_image)
            os.unlink(temp_output)
        except:
            pass


def demo_training_setup():
    """Demo: Show training setup (without actual training)"""
    print("\n" + "="*60)
    print("DEMO 4: Training Setup")
    print("="*60)
    
    from cad3d.vit_trainer import VisionTransformerTrainer, TrainingConfig, CADDataset
    from cad3d.vision_transformer_cad import VisionTransformerConfig
    
    # Model config
    model_config = VisionTransformerConfig(
        image_size=512,
        patch_size=16,
        num_classes=50,
        dim=768,
        depth=12,
        heads=12
    )
    
    # Training config
    train_config = TrainingConfig(
        train_data_dir="data/train",
        val_data_dir="data/val",
        batch_size=4,
        num_epochs=50,
        learning_rate=1e-4,
        device="cpu"
    )
    
    print("✓ Training configuration:")
    print(f"  - Model parameters: ~86M")
    print(f"  - Batch size: {train_config.batch_size}")
    print(f"  - Learning rate: {train_config.learning_rate}")
    print(f"  - Epochs: {train_config.num_epochs}")
    print(f"  - Optimizer: {train_config.optimizer}")
    print(f"  - Scheduler: {train_config.scheduler}")
    print(f"  - Device: {train_config.device}")
    
    print("\n📁 Expected dataset structure:")
    print("  data/")
    print("    train/")
    print("      images/")
    print("        drawing_001.png")
    print("        drawing_002.png")
    print("      annotations/")
    print("        drawing_001.json")
    print("        drawing_002.json")
    print("    val/")
    print("      images/")
    print("      annotations/")
    
    print("\n💡 To train:")
    print("  1. Prepare dataset in above structure")
    print("  2. Run: python -m cad3d.vit_trainer")
    print("  3. Model saved to: checkpoints/best_model.pth")


def demo_integration():
    """Demo: Show integration with server"""
    print("\n" + "="*60)
    print("DEMO 5: Server Integration")
    print("="*60)
    
    from cad3d.vit_integration import is_vit_available, get_vit_service
    
    print("✓ Checking Vision Transformer availability...")
    
    if is_vit_available():
        print("  ✓ Vision Transformer is available")
        
        # Try to create service
        try:
            service = get_vit_service(device="cpu")
            if service:
                print("  ✓ VIT service initialized successfully")
                print("\n📡 Server integration ready:")
                print("  - Use vit_service.convert_image_to_3d_dxf() for conversion")
                print("  - Use vit_service.analyze_image() for analysis")
                print("  - Use vit_service.create_visualization() for viz")
            else:
                print("  ✗ Could not initialize service")
        except Exception as e:
            print(f"  ✗ Initialization error: {e}")
    else:
        print("  ✗ Vision Transformer not available")
        print("  Install with: pip install torch torchvision")


def main():
    parser = argparse.ArgumentParser(description="Vision Transformer Demo")
    parser.add_argument(
        "--demo",
        type=int,
        choices=[1, 2, 3, 4, 5],
        help="Run specific demo (1-5), or all if not specified"
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("Vision Transformer for CAD - Demo Script")
    print("="*60)
    
    # Check dependencies
    print("\n📦 Checking dependencies...")
    if not check_dependencies():
        print("\n⚠️ Please install missing dependencies first")
        return
    
    print("\n✓ All dependencies installed")
    
    # Run demos
    demos = {
        1: demo_model_creation,
        2: demo_image_analysis,
        3: demo_3d_reconstruction,
        4: demo_training_setup,
        5: demo_integration
    }
    
    if args.demo:
        demos[args.demo]()
    else:
        for demo_func in demos.values():
            demo_func()
    
    print("\n" + "="*60)
    print("✅ Demo complete!")
    print("="*60)
    print("\n💡 Next steps:")
    print("  1. Read VISION_TRANSFORMER_GUIDE.md for full documentation")
    print("  2. Prepare your dataset for training")
    print("  3. Train model: python -m cad3d.vit_trainer")
    print("  4. Use trained model in your application")
    print("\n📖 For more info: cat VISION_TRANSFORMER_GUIDE.md")


if __name__ == "__main__":
    main()
