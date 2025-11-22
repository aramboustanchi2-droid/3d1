"""
Complete Package Downloader
دانلود بسته کامل نصب

این اسکریپت تمام فایل‌های لازم برای نصب را در یک پوشه جمع می‌کند
"""

import shutil
from pathlib import Path
import zipfile
import json
from datetime import datetime


def create_download_package():
    """Create a complete download package"""
    
    print("="*70)
    print("Creating Download Package")
    print("ساخت بسته دانلود کامل")
    print("="*70)
    
    # Create download directory
    download_dir = Path("cad3d_download_package")
    download_dir.mkdir(exist_ok=True)
    
    print(f"\nCreating package in: {download_dir}")
    
    # List of files to include
    files_to_copy = {
        # Requirements
        "requirements.txt": "نیازمندی‌های اصلی",
        "requirements_diffusion.txt": "نیازمندی‌های Diffusion",
        
        # Setup scripts
        "setup_diffusion.py": "اسکریپت نصب",
        "install_diffusion.bat": "نصب Windows",
        "install_diffusion.sh": "نصب Linux/Mac",
        
        # Documentation
        "INSTALL_DIFFUSION.md": "راهنمای نصب",
        "DIFFUSION_MODEL_GUIDE.md": "راهنمای مدل Diffusion",
        "DOWNLOAD_README.md": "راهنمای دانلود",
        "README.md": "راهنمای اصلی پروژه",
        
        # Demo scripts
        "demo_diffusion.py": "نمایش Diffusion",
        "demo_vit.py": "نمایش Vision Transformer",
        "launch_neural_system.py": "راه‌اندازی سیستم عصبی",
        
        # Main modules (core)
        "cad3d/__init__.py": "ماژول اصلی",
        "cad3d/diffusion_3d_model.py": "مدل Diffusion 3D",
        "cad3d/diffusion_trainer.py": "آموزش Diffusion",
        "cad3d/hybrid_vit_diffusion.py": "ادغام ViT + Diffusion",
        "cad3d/vision_transformer_cad.py": "Vision Transformer",
        "cad3d/vit_integration.py": "ادغام ViT",
    }
    
    # Copy files
    copied = 0
    missing = []
    
    for file_path, description in files_to_copy.items():
        src = Path(file_path)
        dst = download_dir / file_path
        
        if src.exists():
            # Create parent directories
            dst.parent.mkdir(parents=True, exist_ok=True)
            
            # Copy file
            shutil.copy2(src, dst)
            print(f"✅ {file_path:<40} ({description})")
            copied += 1
        else:
            print(f"⚠️  {file_path:<40} (not found)")
            missing.append(file_path)
    
    # Create package info
    package_info = {
        "name": "CAD 2D to 3D Converter with Diffusion Model",
        "name_fa": "تبدیل‌کننده CAD از 2D به 3D با مدل انتشار",
        "version": "1.0.0",
        "created": datetime.now().isoformat(),
        "files_included": copied,
        "files_missing": len(missing),
        "python_version": "3.8+",
        "description": "Complete package for installing CAD 2D→3D conversion system with state-of-the-art Diffusion Model",
        "features": [
            "3D Diffusion Model (DDPM/DDIM)",
            "Vision Transformer",
            "Hybrid ViT + Diffusion",
            "Continuous Learning",
            "Experience Replay",
            "PointNet++ 3D Processing",
            "CLIP-guided Generation"
        ],
        "requirements": {
            "python": ">=3.8",
            "ram": "8GB minimum, 16GB recommended",
            "storage": "5GB minimum, 20GB recommended",
            "gpu": "Optional, NVIDIA GPU with 6GB+ VRAM recommended"
        },
        "installation": {
            "windows": "Run install_diffusion.bat",
            "linux_mac": "Run install_diffusion.sh",
            "manual": "See INSTALL_DIFFUSION.md"
        }
    }
    
    # Save package info
    info_path = download_dir / "package_info.json"
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(package_info, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Package info saved: {info_path}")
    
    # Create quick start guide
    quick_start = """# Quick Start Guide
راهنمای شروع سریع

## Installation (نصب)

### Windows:
```cmd
install_diffusion.bat
```

### Linux/Mac:
```bash
chmod +x install_diffusion.sh
./install_diffusion.sh
```

### Manual:
```cmd
python -m venv .venv
.venv\\Scripts\\activate  # Windows
source .venv/bin/activate  # Linux/Mac
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python setup_diffusion.py
```

## Usage (استفاده)

```python
from cad3d.hybrid_vit_diffusion import create_hybrid_converter

converter = create_hybrid_converter(device="cpu")
converter.convert_image_to_3d("input.png", "output.dxf")
```

## Documentation (مستندات)

- Installation: INSTALL_DIFFUSION.md
- Diffusion Model: DIFFUSION_MODEL_GUIDE.md
- Main README: README.md

## Support (پشتیبانی)

For issues, see INSTALL_DIFFUSION.md → Troubleshooting section
"""
    
    quick_start_path = download_dir / "QUICK_START.md"
    with open(quick_start_path, 'w', encoding='utf-8') as f:
        f.write(quick_start)
    
    print(f"✅ Quick start guide: {quick_start_path}")
    
    # Create ZIP archive
    print("\n" + "="*70)
    print("Creating ZIP archive...")
    print("="*70)
    
    zip_path = Path("cad3d_complete_package.zip")
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file_path in download_dir.rglob('*'):
            if file_path.is_file():
                arcname = file_path.relative_to(download_dir.parent)
                zipf.write(file_path, arcname)
                print(f"📦 {arcname}")
    
    # Get ZIP size
    zip_size_mb = zip_path.stat().st_size / 1024 / 1024
    
    print("\n" + "="*70)
    print("✅ PACKAGE CREATED SUCCESSFULLY!")
    print("="*70)
    
    print(f"\n📁 Directory: {download_dir}")
    print(f"   Files: {copied}")
    if missing:
        print(f"   Missing: {len(missing)}")
    
    print(f"\n📦 ZIP Archive: {zip_path}")
    print(f"   Size: {zip_size_mb:.2f} MB")
    
    print("\n🎯 Next Steps:")
    print("1. Download/share the ZIP file: cad3d_complete_package.zip")
    print("2. Extract it anywhere")
    print("3. Run install_diffusion.bat (Windows) or install_diffusion.sh (Linux/Mac)")
    print("4. Or follow QUICK_START.md")
    
    print("\n" + "="*70)
    
    return download_dir, zip_path


def main():
    """Main function"""
    try:
        download_dir, zip_path = create_download_package()
        
        print("\n✅ Package ready for download!")
        print(f"\nShare this file: {zip_path.absolute()}")
        
    except Exception as e:
        print(f"\n❌ Error creating package: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
