"""
Model Downloader - دانلود مدل‌های پیش‌آموزش‌دیده
Download pre-trained models for immediate use
"""

import urllib.request
import json
from pathlib import Path
import hashlib
import sys


# Model URLs and info
MODELS = {
    'midas_small': {
        'url': 'https://github.com/isl-org/MiDaS/releases/download/v2_1/model-small.onnx',
        'filename': 'midas_v2_small_256.onnx',
        'size_mb': 12.5,
        'description': 'MiDaS v2.1 Small - Depth estimation (256x256)',
        'destination': 'models'
    },
    'example_vit': {
        'url': 'https://huggingface.co/facebook/vit-base-patch16-224/resolve/main/pytorch_model.bin',
        'filename': 'vit_base_pretrained.pth',
        'size_mb': 330,
        'description': 'Vision Transformer Base (pretrained on ImageNet)',
        'destination': 'trained_models',
        'optional': True
    }
}


def compute_file_hash(file_path: Path, algorithm: str = 'sha256') -> str:
    """Compute hash of a file for integrity verification"""
    h = hashlib.new(algorithm)
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


def download_file(url: str, destination: Path, description: str = ""):
    """Download file with progress bar"""
    
    destination.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\nDownloading: {description}")
    print(f"From: {url}")
    print(f"To: {destination}")
    
    def progress_hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = min(downloaded * 100.0 / total_size, 100)
        
        bar_length = 50
        filled = int(bar_length * percent / 100)
        bar = '█' * filled + '-' * (bar_length - filled)
        
        downloaded_mb = downloaded / 1024 / 1024
        total_mb = total_size / 1024 / 1024
        
        print(f"\r[{bar}] {percent:.1f}% ({downloaded_mb:.1f}/{total_mb:.1f} MB)", end='')
        sys.stdout.flush()
    
    try:
        urllib.request.urlretrieve(url, destination, progress_hook)
        print("\n✅ Download complete")
        # Compute hash
        print("Computing SHA256 hash...", end=' ')
        file_hash = compute_file_hash(destination)
        print(f"✅ {file_hash}")
        return file_hash
    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        return None


def create_model_info():
    """Create JSON file with model information"""
    
    info = {
        'models': {},
        'instructions': {
            'ar': 'برای استفاده از مدل‌ها، فایل‌های زیر را دانلود و در پوشه مربوطه قرار دهید',
            'en': 'To use the models, download the following files and place them in the respective folders'
        }
    }
    
    for name, data in MODELS.items():
        info['models'][name] = {
            'url': data['url'],
            'filename': data['filename'],
            'size_mb': data['size_mb'],
            'description': data['description'],
            'destination': data['destination']
        }
    
    info_path = Path('MODELS_DOWNLOAD_INFO.json')
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Model info saved to: {info_path}")
    return info_path


def download_all_models(skip_optional=True):
    """Download all models"""
    
    print("="*70)
    print("MODEL DOWNLOADER - دانلود مدل‌ها")
    print("="*70)
    
    success_count = 0
    fail_count = 0
    skip_count = 0
    integrity_data = {}
    
    for name, data in MODELS.items():
        if skip_optional and data.get('optional', False):
            print(f"\n⏭️  Skipping optional model: {name}")
            skip_count += 1
            continue
        
        destination = Path(data['destination']) / data['filename']
        
        if destination.exists():
            print(f"\n✅ Already exists: {destination}")
            # Compute hash for existing file
            print("Computing SHA256 hash...", end=' ')
            file_hash = compute_file_hash(destination)
            print(f"✅ {file_hash}")
            integrity_data[name] = {'file': str(destination), 'sha256': file_hash}
            success_count += 1
            continue
        
        file_hash = download_file(data['url'], destination, data['description'])
        if file_hash:
            integrity_data[name] = {'file': str(destination), 'sha256': file_hash}
            success_count += 1
        else:
            fail_count += 1
    
    # Save integrity file
    if integrity_data:
        integrity_path = Path('models_integrity.json')
        with open(integrity_path, 'w', encoding='utf-8') as f:
            json.dump(integrity_data, f, indent=2, ensure_ascii=False)
        print(f"\n📄 Integrity hashes saved to: {integrity_path}")
    
    print("\n" + "="*70)
    print(f"✅ Success: {success_count}")
    print(f"❌ Failed: {fail_count}")
    print(f"⏭️  Skipped: {skip_count}")
    print("="*70)


def main():
    """Main function"""
    
    print("\n" + "="*70)
    print("🚀 CAD 3D MODEL DOWNLOADER")
    print("="*70)
    
    print("\nAvailable models:")
    for i, (name, data) in enumerate(MODELS.items(), 1):
        optional = " (optional)" if data.get('optional', False) else ""
        print(f"  {i}. {name}{optional}")
        print(f"     {data['description']}")
        print(f"     Size: {data['size_mb']} MB")
        print()
    
    print("Options:")
    print("  1. Download required models only")
    print("  2. Download all models (including optional)")
    print("  3. Create download info file (for manual download)")
    print("  4. Exit")
    
    choice = input("\nEnter choice (1-4) [default=1]: ").strip() or "1"
    
    if choice == "1":
        download_all_models(skip_optional=True)
    elif choice == "2":
        download_all_models(skip_optional=False)
    elif choice == "3":
        info_path = create_model_info()
        print(f"\n📄 Download info saved to: {info_path}")
        print("You can manually download models using the URLs in this file")
    elif choice == "4":
        print("Exiting...")
        return
    else:
        print("Invalid choice")
        return
    
    print("\n✅ Done!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Cancelled by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
