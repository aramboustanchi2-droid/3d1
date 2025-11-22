"""
Diffusion Model Demo and Testing
آزمایش و نمایش مدل انتشار

Test and demonstrate the 3D Diffusion Model capabilities
"""

import torch
import numpy as np
import cv2
from pathlib import Path
import time

from cad3d.hybrid_vit_diffusion import create_hybrid_converter


def demo_basic_conversion():
    """Basic conversion demo"""
    print("\n" + "="*70)
    print("DEMO 1: Basic Image to 3D Conversion")
    print("="*70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Create converter
    converter = create_hybrid_converter(device=device, enable_learning=False)
    
    # Create a simple test image
    print("\nCreating test CAD drawing...")
    img = np.ones((512, 512, 3), dtype=np.uint8) * 255
    
    # Draw some shapes
    cv2.rectangle(img, (100, 100), (300, 300), (0, 0, 0), 3)
    cv2.circle(img, (400, 400), 80, (0, 0, 0), 3)
    cv2.line(img, (50, 400), (200, 500), (0, 0, 0), 3)
    
    # Save test image
    test_dir = Path("demo_output/diffusion")
    test_dir.mkdir(parents=True, exist_ok=True)
    
    test_image_path = test_dir / "test_cad.png"
    cv2.imwrite(str(test_image_path), img)
    print(f"Test image saved: {test_image_path}")
    
    # Convert to 3D
    output_path = test_dir / "test_cad_3d.dxf"
    
    results = converter.convert_image_to_3d(
        image_path=test_image_path,
        output_path=output_path,
        sampling_steps=20,  # Fast sampling
        learn_from_conversion=False
    )
    
    print("\n📊 Conversion Results:")
    for key, value in results.items():
        print(f"  {key}: {value}")
    
    return results


def demo_batch_conversion():
    """Batch conversion demo"""
    print("\n" + "="*70)
    print("DEMO 2: Batch Conversion with Learning")
    print("="*70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create converter with learning enabled
    converter = create_hybrid_converter(device=device, enable_learning=True)
    
    # Create multiple test images
    test_dir = Path("demo_output/diffusion/batch")
    test_dir.mkdir(parents=True, exist_ok=True)
    
    num_images = 5
    results_list = []
    
    for i in range(num_images):
        # Create unique test image
        img = np.ones((512, 512, 3), dtype=np.uint8) * 255
        
        # Random shapes
        num_shapes = np.random.randint(3, 8)
        for _ in range(num_shapes):
            shape_type = np.random.choice(['rect', 'circle', 'line'])
            
            if shape_type == 'rect':
                x, y = np.random.randint(50, 400, 2)
                w, h = np.random.randint(50, 150, 2)
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 0), 2)
            elif shape_type == 'circle':
                cx, cy = np.random.randint(100, 400, 2)
                r = np.random.randint(30, 80)
                cv2.circle(img, (cx, cy), r, (0, 0, 0), 2)
            else:
                x1, y1 = np.random.randint(50, 450, 2)
                x2, y2 = np.random.randint(50, 450, 2)
                cv2.line(img, (x1, y1), (x2, y2), (0, 0, 0), 2)
        
        # Save test image
        test_image_path = test_dir / f"test_{i:03d}.png"
        cv2.imwrite(str(test_image_path), img)
        
        # Convert
        output_path = test_dir / f"test_{i:03d}_3d.dxf"
        
        results = converter.convert_image_to_3d(
            image_path=test_image_path,
            output_path=output_path,
            sampling_steps=20,
            learn_from_conversion=True  # Enable learning
        )
        
        results_list.append(results)
        
        print(f"\n✅ Converted {i+1}/{num_images}")
    
    # Summary
    print("\n" + "="*70)
    print("📊 Batch Conversion Summary")
    print("="*70)
    
    total_time = sum(r['conversion_time'] for r in results_list)
    avg_time = total_time / len(results_list)
    total_points = sum(r['num_points'] for r in results_list)
    
    print(f"Total conversions: {len(results_list)}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average time: {avg_time:.2f}s")
    print(f"Total points generated: {total_points:,}")
    print(f"Learning updates performed: {results_list[-1]['learning_updates']}")
    
    return results_list


def demo_continuous_learning():
    """Demonstrate continuous learning"""
    print("\n" + "="*70)
    print("DEMO 3: Continuous Learning")
    print("="*70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    converter = create_hybrid_converter(device=device, enable_learning=True)
    
    test_dir = Path("demo_output/diffusion/learning")
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # Create and convert multiple times
    print("\nPerforming multiple conversions to trigger learning...")
    
    for i in range(15):  # 15 conversions to trigger learning update
        # Create test image
        img = np.ones((512, 512, 3), dtype=np.uint8) * 255
        
        # Add progressive complexity
        num_shapes = 3 + i // 3
        for _ in range(num_shapes):
            x, y = np.random.randint(50, 400, 2)
            w, h = np.random.randint(50, 150, 2)
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 0), 2)
        
        test_image_path = test_dir / f"learning_{i:03d}.png"
        cv2.imwrite(str(test_image_path), img)
        
        output_path = test_dir / f"learning_{i:03d}_3d.dxf"
        
        results = converter.convert_image_to_3d(
            image_path=test_image_path,
            output_path=output_path,
            sampling_steps=15,
            learn_from_conversion=True
        )
        
        print(f"Conversion {i+1}/15 | Learning updates: {results['learning_updates']}")
    
    print("\n✅ Continuous learning demonstrated!")
    print(f"Replay buffer size: {len(converter.replay_buffer)}")
    print(f"Total learning updates: {results['learning_updates']}")


def demo_model_comparison():
    """Compare different sampling methods"""
    print("\n" + "="*70)
    print("DEMO 4: Sampling Method Comparison (DDPM vs DDIM)")
    print("="*70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    converter = create_hybrid_converter(device=device, enable_learning=False)
    
    # Create test image
    test_dir = Path("demo_output/diffusion/comparison")
    test_dir.mkdir(parents=True, exist_ok=True)
    
    img = np.ones((512, 512, 3), dtype=np.uint8) * 255
    cv2.rectangle(img, (100, 100), (400, 400), (0, 0, 0), 3)
    cv2.circle(img, (250, 250), 100, (0, 0, 0), 3)
    
    test_image_path = test_dir / "comparison_test.png"
    cv2.imwrite(str(test_image_path), img)
    
    # Test different sampling steps
    steps_list = [10, 20, 50]
    
    print("\nTesting DDIM with different steps:")
    
    for steps in steps_list:
        output_path = test_dir / f"ddim_steps_{steps}.dxf"
        
        start_time = time.time()
        results = converter.convert_image_to_3d(
            image_path=test_image_path,
            output_path=output_path,
            sampling_steps=steps,
            learn_from_conversion=False
        )
        elapsed = time.time() - start_time
        
        print(f"  {steps} steps: {elapsed:.2f}s | {results['num_points']} points")
    
    print("\n✅ Sampling comparison complete!")


def show_model_info():
    """Display model architecture and size"""
    print("\n" + "="*70)
    print("MODEL INFORMATION")
    print("="*70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    from cad3d.diffusion_3d_model import create_diffusion_model
    
    model = create_diffusion_model(num_points=4096, timesteps=1000, device=device)
    
    # Count parameters
    unet_params = sum(p.numel() for p in model.unet.parameters())
    encoder_params = sum(p.numel() for p in model.image_encoder.parameters())
    total_params = unet_params + encoder_params
    
    print("\n📊 Model Architecture:")
    print(f"  • U-Net 3D Diffusion: {unet_params:,} parameters")
    print(f"  • CLIP Image Encoder: {encoder_params:,} parameters")
    print(f"  • Total: {total_params:,} parameters")
    print(f"  • Model size: ~{total_params * 4 / 1024 / 1024:.2f} MB")
    
    print("\n🔧 Capabilities:")
    print("  ✅ DDPM sampling (high quality, slow)")
    print("  ✅ DDIM sampling (fast, 10-50 steps)")
    print("  ✅ PointNet++ 3D understanding")
    print("  ✅ CLIP-guided generation")
    print("  ✅ Vision Transformer integration")
    print("  ✅ Continuous learning from conversions")
    print("  ✅ Experience replay buffer")
    print("  ✅ Semantic-aware 3D generation")
    
    print("\n📈 Performance:")
    print(f"  • Device: {device}")
    
    if device == "cuda":
        print(f"  • GPU: {torch.cuda.get_device_name(0)}")
        print(f"  • CUDA version: {torch.version.cuda}")
        print(f"  • Memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")


def main():
    """Run all demos"""
    print("\n" + "="*70)
    print("🚀 3D DIFFUSION MODEL - COMPLETE DEMO")
    print("="*70)
    print("\nThis demo showcases:")
    print("  1. Basic image to 3D conversion")
    print("  2. Batch conversion with learning")
    print("  3. Continuous learning demonstration")
    print("  4. Sampling method comparison")
    print("  5. Model architecture information")
    
    try:
        # Show model info first
        show_model_info()
        
        # Run demos
        input("\n\nPress Enter to start Demo 1 (Basic Conversion)...")
        demo_basic_conversion()
        
        input("\n\nPress Enter to start Demo 2 (Batch Conversion)...")
        demo_batch_conversion()
        
        input("\n\nPress Enter to start Demo 3 (Continuous Learning)...")
        demo_continuous_learning()
        
        input("\n\nPress Enter to start Demo 4 (Sampling Comparison)...")
        demo_model_comparison()
        
        print("\n" + "="*70)
        print("✅ ALL DEMOS COMPLETE!")
        print("="*70)
        
        print("\n📁 Output files saved to: demo_output/diffusion/")
        print("\n🎯 Next steps:")
        print("  1. Check generated DXF files in demo_output/")
        print("  2. Train model: python -m cad3d.diffusion_trainer")
        print("  3. Use in production: from cad3d.hybrid_vit_diffusion import create_hybrid_converter")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Demo interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
