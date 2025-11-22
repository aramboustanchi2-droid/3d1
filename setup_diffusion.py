"""
Diffusion Model Setup and Installation
نصب و راه‌اندازی مدل انتشار

This script installs all dependencies and prepares the system
"""

import subprocess
import sys
from pathlib import Path


def check_python_version():
    """Check Python version"""
    print("Checking Python version...")
    
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python {version.major}.{version.minor} detected")
        print("   Python 3.8+ required")
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
    return True


def install_pytorch():
    """Install PyTorch"""
    print("\n" + "="*70)
    print("Installing PyTorch...")
    print("="*70)
    
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__} already installed")
        
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print("ℹ️  CUDA not available (using CPU)")
        
        return True
    except ImportError:
        print("PyTorch not found. Installing...")
        
        print("\nChoose installation:")
        print("1. CPU only (smaller, works everywhere)")
        print("2. CUDA 11.8 (for NVIDIA GPU)")
        print("3. CUDA 12.1 (for newer NVIDIA GPU)")
        
        choice = input("\nEnter choice (1-3) [default=1]: ").strip() or "1"
        
        if choice == "1":
            cmd = [
                sys.executable, "-m", "pip", "install",
                "torch", "torchvision",
                "--index-url", "https://download.pytorch.org/whl/cpu"
            ]
        elif choice == "2":
            cmd = [
                sys.executable, "-m", "pip", "install",
                "torch", "torchvision",
                "--index-url", "https://download.pytorch.org/whl/cu118"
            ]
        elif choice == "3":
            cmd = [
                sys.executable, "-m", "pip", "install",
                "torch", "torchvision",
                "--index-url", "https://download.pytorch.org/whl/cu121"
            ]
        else:
            print("Invalid choice, installing CPU version")
            cmd = [
                sys.executable, "-m", "pip", "install",
                "torch", "torchvision",
                "--index-url", "https://download.pytorch.org/whl/cpu"
            ]
        
        print(f"\nRunning: {' '.join(cmd)}")
        result = subprocess.run(cmd)
        
        if result.returncode == 0:
            print("✅ PyTorch installed successfully")
            return True
        else:
            print("❌ PyTorch installation failed")
            return False


def install_dependencies():
    """Install other dependencies"""
    print("\n" + "="*70)
    print("Installing dependencies...")
    print("="*70)
    
    packages = [
        "ezdxf>=1.3.0",
        "opencv-python>=4.8.0",
        "numpy>=1.24.0",
        "scipy>=1.11.0",
        "matplotlib>=3.7.0",
        "pillow>=10.0.0"
    ]
    
    for package in packages:
        print(f"\nInstalling {package}...")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", package],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print(f"✅ {package}")
        else:
            print(f"❌ {package} - {result.stderr}")
            return False
    
    print("\n✅ All dependencies installed")
    return True


def create_directories():
    """Create necessary directories"""
    print("\n" + "="*70)
    print("Creating directories...")
    print("="*70)
    
    dirs = [
        "training_data/diffusion_synthetic/images",
        "training_data/diffusion_synthetic/pointclouds",
        "training_data/real_cad/images",
        "training_data/real_cad/pointclouds",
        "trained_models/diffusion",
        "demo_output/diffusion",
        "output"
    ]
    
    for dir_path in dirs:
        path = Path(dir_path)
        path.mkdir(parents=True, exist_ok=True)
        print(f"✅ {dir_path}")
    
    print("\n✅ All directories created")
    return True


def generate_sample_data():
    """Generate sample training data"""
    print("\n" + "="*70)
    print("Generating sample training data...")
    print("="*70)
    
    try:
        from cad3d.diffusion_trainer import create_synthetic_training_data
        
        data_dir = Path("training_data/diffusion_synthetic")
        
        if len(list((data_dir / "images").glob("*.png"))) > 0:
            print("Sample data already exists")
            return True
        
        create_synthetic_training_data(
            output_dir=data_dir,
            num_samples=50  # Small number for quick setup
        )
        
        print("✅ Sample data generated")
        return True
    except Exception as e:
        print(f"⚠️  Could not generate sample data: {e}")
        print("   You can generate it later with:")
        print("   python -m cad3d.diffusion_trainer")
        return True  # Non-critical


def verify_installation():
    """Verify installation"""
    print("\n" + "="*70)
    print("Verifying installation...")
    print("="*70)
    
    errors = []
    
    # Check PyTorch
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")
    except ImportError as e:
        errors.append(f"PyTorch: {e}")
        print(f"❌ PyTorch")
    
    # Check other packages
    packages = {
        'ezdxf': 'ezdxf',
        'cv2': 'opencv-python',
        'numpy': 'numpy',
        'scipy': 'scipy',
        'matplotlib': 'matplotlib',
        'PIL': 'pillow'
    }
    
    for module, name in packages.items():
        try:
            __import__(module)
            print(f"✅ {name}")
        except ImportError as e:
            errors.append(f"{name}: {e}")
            print(f"❌ {name}")
    
    # Check custom modules
    try:
        from cad3d.diffusion_3d_model import create_diffusion_model
        print(f"✅ diffusion_3d_model")
    except ImportError as e:
        errors.append(f"diffusion_3d_model: {e}")
        print(f"❌ diffusion_3d_model")
    
    try:
        from cad3d.diffusion_trainer import DiffusionTrainer
        print(f"✅ diffusion_trainer")
    except ImportError as e:
        errors.append(f"diffusion_trainer: {e}")
        print(f"❌ diffusion_trainer")
    
    try:
        from cad3d.hybrid_vit_diffusion import create_hybrid_converter
        print(f"✅ hybrid_vit_diffusion")
    except ImportError as e:
        errors.append(f"hybrid_vit_diffusion: {e}")
        print(f"❌ hybrid_vit_diffusion")
    
    if errors:
        print(f"\n❌ {len(errors)} error(s) found:")
        for error in errors:
            print(f"   - {error}")
        return False
    else:
        print("\n✅ All modules verified")
        return True


def show_next_steps():
    """Show next steps"""
    print("\n" + "="*70)
    print("✅ INSTALLATION COMPLETE!")
    print("="*70)
    
    print("\n📚 Next Steps:")
    print("\n1️⃣  Test the installation:")
    print("   python demo_diffusion.py")
    
    print("\n2️⃣  Generate more training data:")
    print("   python -m cad3d.diffusion_trainer")
    
    print("\n3️⃣  Train the model:")
    print("   python")
    print("   >>> from cad3d.diffusion_trainer import *")
    print("   >>> # Follow training guide")
    
    print("\n4️⃣  Use in your code:")
    print("   from cad3d.hybrid_vit_diffusion import create_hybrid_converter")
    print("   converter = create_hybrid_converter(device='cuda')")
    print("   converter.convert_image_to_3d('input.png', 'output.dxf')")
    
    print("\n5️⃣  Read documentation:")
    print("   cat DIFFUSION_MODEL_GUIDE.md")
    
    print("\n" + "="*70)


def main():
    """Main installation process"""
    print("\n" + "="*70)
    print("🚀 DIFFUSION MODEL SETUP")
    print("نصب و راه‌اندازی مدل انتشار")
    print("="*70)
    
    # Step 1: Check Python
    if not check_python_version():
        print("\n❌ Setup failed: Python version too old")
        return False
    
    # Step 2: Install PyTorch
    if not install_pytorch():
        print("\n❌ Setup failed: PyTorch installation failed")
        return False
    
    # Step 3: Install dependencies
    if not install_dependencies():
        print("\n❌ Setup failed: Dependency installation failed")
        return False
    
    # Step 4: Create directories
    if not create_directories():
        print("\n❌ Setup failed: Directory creation failed")
        return False
    
    # Step 5: Generate sample data
    generate_sample_data()
    
    # Step 6: Verify
    if not verify_installation():
        print("\n⚠️  Setup completed with warnings")
        print("   Some modules may not work correctly")
    
    # Step 7: Show next steps
    show_next_steps()
    
    return True


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Installation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Installation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
