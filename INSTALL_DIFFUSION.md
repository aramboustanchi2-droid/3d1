# Diffusion Model Installation Guide

# راهنمای نصب مدل انتشار

## 🚀 Quick Install (Windows)

### نصب خودکار (آسان‌ترین روش)

```cmd
install_diffusion.bat
```

این فایل به صورت خودکار:

- ✅ Virtual environment می‌سازد
- ✅ PyTorch نصب می‌کند
- ✅ همه کتابخانه‌ها را نصب می‌کند
- ✅ پوشه‌های لازم را می‌سازد
- ✅ داده‌های نمونه تولید می‌کند

---

## 📦 Manual Install (Windows)

### مرحله 1: Virtual Environment

```cmd
python -m venv .venv
.venv\Scripts\activate
```

### مرحله 2: Install PyTorch

#### CPU only

```cmd
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

#### GPU (CUDA 11.8)

```cmd
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

#### GPU (CUDA 12.1)

```cmd
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### مرحله 3: Install Dependencies

```cmd
pip install -r requirements_diffusion.txt
```

یا نصب دستی:

```cmd
pip install ezdxf>=1.3.0
pip install opencv-python>=4.8.0
pip install numpy>=1.24.0
pip install scipy>=1.11.0
pip install matplotlib>=3.7.0
pip install pillow>=10.0.0
```

### مرحله 4: Run Setup

```cmd
python setup_diffusion.py
```

### مرحله 5: Test

```cmd
python demo_diffusion.py
```

---

## 🐧 Linux/Mac Installation

### Quick Install

```bash
chmod +x install_diffusion.sh
./install_diffusion.sh
```

### Manual Install

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install PyTorch (CPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Or for CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install -r requirements_diffusion.txt

# Run setup
python setup_diffusion.py

# Test
python demo_diffusion.py
```

---

## 🔍 Verification

بعد از نصب، این دستورات را تست کنید:

### Test 1: Check PyTorch

```python
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

### Test 2: Check Diffusion Model

```python
python -c "from cad3d.diffusion_3d_model import create_diffusion_model; print('✅ Diffusion Model OK')"
```

### Test 3: Check Hybrid System

```python
python -c "from cad3d.hybrid_vit_diffusion import create_hybrid_converter; print('✅ Hybrid System OK')"
```

### Test 4: Full Demo

```cmd
python demo_diffusion.py
```

---

## 📁 Directory Structure

بعد از نصب، این ساختار ایجاد می‌شود:

```
3d/
├── cad3d/
│   ├── diffusion_3d_model.py      # مدل اصلی
│   ├── diffusion_trainer.py        # سیستم آموزش
│   └── hybrid_vit_diffusion.py    # ادغام ViT + Diffusion
├── training_data/
│   ├── diffusion_synthetic/        # داده‌های سینتتیک
│   │   ├── images/
│   │   └── pointclouds/
│   └── real_cad/                   # داده‌های واقعی شما
│       ├── images/
│       └── pointclouds/
├── trained_models/
│   └── diffusion/                  # مدل‌های آموزش‌دیده
├── demo_output/
│   └── diffusion/                  # نتایج demo
├── output/                         # خروجی‌های شما
├── setup_diffusion.py              # اسکریپت نصب
├── demo_diffusion.py               # نمایش
├── requirements_diffusion.txt      # نیازمندی‌ها
└── DIFFUSION_MODEL_GUIDE.md       # راهنما
```

---

## ⚠️ Troubleshooting

### مشکل 1: PyTorch import error

```
ImportError: No module named 'torch'
```

**حل:**

```cmd
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### مشکل 2: CUDA not available

```
CUDA: False
```

**حل:**

1. نصب PyTorch با CUDA:

   ```cmd
   pip uninstall torch torchvision
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

2. بررسی NVIDIA Driver:

   ```cmd
   nvidia-smi
   ```

### مشکل 3: Out of memory

```
RuntimeError: CUDA out of memory
```

**حل:**

```python
# استفاده از CPU
converter = create_hybrid_converter(device="cpu")

# یا کاهش تعداد نقاط
model = create_diffusion_model(num_points=1024)  # به جای 4096

# یا کاهش batch size
trainer.train(batch_size=2)  # به جای 8
```

### مشکل 4: Import error for custom modules

```
ModuleNotFoundError: No module named 'cad3d.diffusion_3d_model'
```

**حل:**

```cmd
# مطمئن شوید در root directory هستید
cd c:\Users\aram\Desktop\3d

# و virtual environment فعال است
.venv\Scripts\activate
```

### مشکل 5: Slow performance

**حل:**

- استفاده از GPU (CUDA)
- کاهش `sampling_steps` (مثلاً 20 به جای 50)
- کاهش `num_points` (مثلاً 2048 به جای 4096)

---

## 🎯 Quick Start After Install

### 1. Simple Test

```python
from cad3d.hybrid_vit_diffusion import create_hybrid_converter

converter = create_hybrid_converter(device="cpu")
converter.convert_image_to_3d("input.png", "output.dxf")
```

### 2. With Options

```python
converter = create_hybrid_converter(
    device="cuda",           # استفاده از GPU
    enable_learning=True     # یادگیری مداوم
)

results = converter.convert_image_to_3d(
    image_path="plan.png",
    output_path="plan_3d.dxf",
    sampling_steps=50,       # کیفیت بالا
    learn_from_conversion=True
)

print(f"Generated {results['num_points']} points in {results['conversion_time']:.2f}s")
```

### 3. Training

```python
from cad3d.diffusion_trainer import *

# تولید داده
create_synthetic_training_data("training_data/diffusion_synthetic", 200)

# آموزش
# (دستورات کامل در DIFFUSION_MODEL_GUIDE.md)
```

---

## 📊 System Requirements

### Minimum

- **OS**: Windows 10/11, Linux, macOS
- **Python**: 3.8+
- **RAM**: 8 GB
- **Storage**: 5 GB free space
- **GPU**: Optional (CPU works but slower)

### Recommended

- **OS**: Windows 11, Ubuntu 22.04
- **Python**: 3.10+
- **RAM**: 16 GB+
- **Storage**: 20 GB+ SSD
- **GPU**: NVIDIA GPU with 6GB+ VRAM (RTX 3060+)
- **CUDA**: 11.8 or 12.1

---

## 📚 More Help

- راهنمای کامل: `DIFFUSION_MODEL_GUIDE.md`
- مثال‌های کد: `demo_diffusion.py`
- تست سیستم: `python demo_diffusion.py`

---

## ✅ Installation Checklist

بعد از نصب، این موارد را چک کنید:

- [ ] Python 3.8+ نصب است
- [ ] Virtual environment ساخته شد
- [ ] PyTorch نصب شد
- [ ] همه dependencies نصب شد
- [ ] پوشه‌های لازم ساخته شد
- [ ] داده‌های نمونه تولید شد
- [ ] `python demo_diffusion.py` اجرا می‌شود
- [ ] Import ها کار می‌کند

اگر همه ✅ است، آماده استفاده هستید! 🚀
