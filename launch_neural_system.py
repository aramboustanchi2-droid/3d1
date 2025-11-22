"""
Complete Neural System Launcher
راه‌اندازی کامل سیستم عصبی

این اسکریپت:
1. همه مدل‌ها را چک می‌کند
2. آموزش اولیه انجام می‌دهد  
3. سیستم را برای استفاده آماده می‌کند
"""

import sys
import subprocess
import time
from pathlib import Path


def print_header(text: str):
    """چاپ header زیبا"""
    print("\n" + "="*70)
    print(text.center(70))
    print("="*70)


def check_pytorch():
    """بررسی نصب PyTorch"""
    print_header("Checking PyTorch Installation")
    
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"✅ CUDA available")
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA version: {torch.version.cuda}")
        else:
            print("ℹ️  CUDA not available (using CPU)")
        
        return True
    except ImportError:
        print("❌ PyTorch not installed")
        print("\nInstall with:")
        print("  # For CPU:")
        print("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu")
        print("\n  # For GPU (CUDA 11.8):")
        print("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
        return False


def check_dependencies():
    """بررسی وابستگی‌ها"""
    print_header("Checking Dependencies")
    
    deps = {
        'torch': 'PyTorch',
        'torchvision': 'Torchvision',
        'cv2': 'OpenCV (opencv-python)',
        'numpy': 'NumPy',
        'ezdxf': 'ezdxf',
        'matplotlib': 'Matplotlib',
        'scipy': 'SciPy'
    }
    
    missing = []
    installed = []
    
    for module, name in deps.items():
        try:
            if module == 'cv2':
                import cv2
            else:
                __import__(module)
            installed.append(name)
            print(f"✅ {name}")
        except ImportError:
            missing.append(module if module != 'cv2' else 'opencv-python')
            print(f"❌ {name}")
    
    if missing:
        print(f"\n⚠️ Missing: {', '.join(missing)}")
        print(f"\nInstall with:")
        print(f"  pip install {' '.join(missing)}")
        return False
    
    print(f"\n✅ All dependencies installed ({len(installed)} packages)")
    return True


def test_neural_system():
    """تست سیستم عصبی"""
    print_header("Testing Neural System")
    
    try:
        result = subprocess.run(
            [sys.executable, "test_neural_system.py"],
            capture_output=False,
            text=True
        )
        
        if result.returncode == 0:
            print("\n✅ Neural system test passed")
            return True
        else:
            print("\n❌ Neural system test failed")
            return False
            
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        return False


def run_auto_training():
    """اجرای آموزش خودکار"""
    print_header("Running Automatic Training")
    
    try:
        result = subprocess.run(
            [sys.executable, "auto_train_system.py"],
            capture_output=False,
            text=True
        )
        
        if result.returncode == 0:
            print("\n✅ Training completed successfully")
            return True
        else:
            print("\n⚠️ Training had issues (check output above)")
            return False
            
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        return False


def create_training_data_sample():
    """ایجاد داده‌های نمونه برای آموزش"""
    print_header("Creating Sample Training Data")
    
    try:
        import numpy as np
        import cv2
        
        data_dir = Path("training_data")
        data_dir.mkdir(exist_ok=True)
        
        print("Creating synthetic CAD drawings...")
        
        for i in range(10):
            # ایجاد تصویر سینتتیک
            img = np.ones((512, 512, 3), dtype=np.uint8) * 255
            
            # اضافه کردن اشکال
            num_shapes = np.random.randint(3, 8)
            
            for _ in range(num_shapes):
                shape_type = np.random.choice(['rectangle', 'circle', 'line'])
                
                if shape_type == 'rectangle':
                    x1, y1 = np.random.randint(50, 300, 2)
                    w, h = np.random.randint(50, 150, 2)
                    cv2.rectangle(img, (x1, y1), (x1+w, y1+h), (0, 0, 0), 2)
                
                elif shape_type == 'circle':
                    cx, cy = np.random.randint(100, 400, 2)
                    r = np.random.randint(30, 80)
                    cv2.circle(img, (cx, cy), r, (0, 0, 0), 2)
                
                else:  # line
                    x1, y1 = np.random.randint(50, 450, 2)
                    x2, y2 = np.random.randint(50, 450, 2)
                    cv2.line(img, (x1, y1), (x2, y2), (0, 0, 0), 2)
            
            # ذخیره
            filename = data_dir / f"synthetic_cad_{i:03d}.png"
            cv2.imwrite(str(filename), img)
        
        print(f"✅ Created 10 synthetic CAD drawings in {data_dir}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to create training data: {e}")
        return False


def show_model_info():
    """نمایش اطلاعات مدل‌ها"""
    print_header("Model Information")
    
    models_dir = Path("trained_models")
    
    if models_dir.exists():
        model_files = list(models_dir.glob("*.pth"))
        
        if model_files:
            print(f"Found {len(model_files)} trained model(s):")
            for model_file in model_files:
                size_mb = model_file.stat().st_size / 1024 / 1024
                print(f"  • {model_file.name} ({size_mb:.2f} MB)")
        else:
            print("No trained models found yet")
    else:
        print("Models directory not created yet")
    
    # بررسی report
    report_file = models_dir / "training_report.json"
    if report_file.exists():
        print(f"\n✅ Training report available: {report_file}")
        
        try:
            import json
            with open(report_file) as f:
                report = json.load(f)
            
            stats = report.get('statistics', {})
            print(f"\nTraining Statistics:")
            print(f"  - Total samples: {stats.get('total_samples', 0)}")
            print(f"  - Epochs completed: {stats.get('epochs_completed', 0)}")
            print(f"  - Best loss: {stats.get('best_loss', 'N/A')}")
            print(f"  - Training time: {stats.get('training_time', 0):.2f}s")
        except:
            pass


def main():
    """اجرای کامل"""
    print("\n" + "="*70)
    print("🚀 COMPLETE NEURAL SYSTEM LAUNCHER")
    print("="*70)
    print("\nThis script will:")
    print("  1. Check all dependencies")
    print("  2. Test neural models")
    print("  3. Create training data (if needed)")
    print("  4. Run automatic training")
    print("  5. Prepare system for use")
    
    input("\nPress Enter to continue...")
    
    # مرحله 1: بررسی PyTorch
    if not check_pytorch():
        print("\n❌ PyTorch required. Please install and run again.")
        return
    
    # مرحله 2: بررسی وابستگی‌ها
    if not check_dependencies():
        print("\n❌ Missing dependencies. Please install and run again.")
        return
    
    # مرحله 3: تست سیستم عصبی
    print_header("Step 1: Testing Neural System")
    test_passed = test_neural_system()
    
    if not test_passed:
        print("\n⚠️ Some tests failed, but continuing...")
    
    # مرحله 4: ایجاد داده‌های آموزشی
    print_header("Step 2: Preparing Training Data")
    
    data_dir = Path("training_data")
    if not data_dir.exists() or len(list(data_dir.glob("*.png"))) == 0:
        print("No training data found. Creating samples...")
        create_training_data_sample()
    else:
        existing_files = len(list(data_dir.glob("*.png")))
        print(f"✅ Found {existing_files} training images")
    
    # مرحله 5: آموزش خودکار
    print_header("Step 3: Automatic Training")
    print("\n⚠️ This may take several minutes...")
    
    training_success = run_auto_training()
    
    # مرحله 6: نمایش اطلاعات
    print_header("Step 4: Model Information")
    show_model_info()
    
    # نتیجه نهایی
    print("\n" + "="*70)
    if training_success:
        print("✅ SYSTEM READY!")
    else:
        print("⚠️ SYSTEM PARTIALLY READY")
    print("="*70)
    
    print("\n📋 What's Next:")
    print("\n1️⃣ Use Vision Transformer:")
    print("   from cad3d.vit_integration import get_vit_service")
    print("   service = get_vit_service()")
    print("   service.convert_image_to_3d_dxf('input.jpg', 'output.dxf')")
    
    print("\n2️⃣ Run server:")
    print("   python -m uvicorn cad3d.simple_server:app --port 8003")
    
    print("\n3️⃣ Continue training:")
    print("   python auto_train_system.py")
    
    print("\n4️⃣ View demos:")
    print("   python demo_vit.py")
    
    print("\n5️⃣ Read documentation:")
    print("   cat README_VIT.md")
    print("   cat VISION_TRANSFORMER_GUIDE.md")
    
    print("\n" + "="*70)
    print("🎉 Setup Complete!")
    print("="*70)
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
