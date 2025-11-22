# Complete Installation Package

# بسته نصب کامل

این پوشه شامل تمام فایل‌های نصب سیستم CAD 2D→3D است.

## 📦 فایل‌های موجود برای دانلود

### 1. **requirements.txt** ⭐ (اصلی)

```
نیازمندی‌های کامل سیستم
شامل: PyTorch, OpenCV, ezdxf, FastAPI, و...
```

### 2. **requirements_diffusion.txt**

```
نیازمندی‌های خاص مدل Diffusion
برای سیستم 3D پیشرفته
```

### 3. **setup_diffusion.py**

```
اسکریپت نصب هوشمند
نصب خودکار تمام components
```

### 4. **install_diffusion.bat** (Windows)

```
نصب یک‌کلیکه برای Windows
فقط دابل‌کلیک کنید!
```

### 5. **install_diffusion.sh** (Linux/Mac)

```
نصب یک‌کلیکه برای Linux/Mac
```

---

## 🚀 روش نصب (ساده)

### Windows

```cmd
# 1. دانلود این پوشه کامل
# 2. اجرای:
install_diffusion.bat
```

### Linux/Mac

```bash
# 1. دانلود این پوشه کامل
# 2. اجرای:
chmod +x install_diffusion.sh
./install_diffusion.sh
```

### دستی (همه سیستم‌ها)

```cmd
# 1. ساخت virtual environment
python -m venv .venv

# 2. فعال‌سازی
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# 3. نصب PyTorch
# CPU:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
# GPU (CUDA 11.8):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 4. نصب بقیه
pip install -r requirements.txt

# 5. اجرای setup
python setup_diffusion.py

# 6. تست
python demo_diffusion.py
```

---

## 📥 دانلود فایل‌ها

تمام فایل‌های زیر در این پوشه قرار دارند:

```
download/
├── requirements.txt              ⭐ نیازمندی‌های اصلی
├── requirements_diffusion.txt    مخصوص Diffusion
├── setup_diffusion.py            اسکریپت نصب
├── install_diffusion.bat         نصب Windows
├── install_diffusion.sh          نصب Linux/Mac
├── INSTALL_DIFFUSION.md          راهنمای کامل
└── DOWNLOAD_README.md            این فایل
```

---

## ✅ Checklist نصب

- [ ] Python 3.8+ نصب شده
- [ ] یکی از فایل‌های requirements دانلود شده
- [ ] Virtual environment ساخته شده
- [ ] PyTorch نصب شده (CPU یا CUDA)
- [ ] `pip install -r requirements.txt` اجرا شده
- [ ] `python setup_diffusion.py` اجرا شده (اختیاری)
- [ ] تست: `python -c "import torch; print('OK')"`

---

## 🎯 بعد از نصب

```python
# تست سریع:
from cad3d.hybrid_vit_diffusion import create_hybrid_converter

converter = create_hybrid_converter(device="cpu")
converter.convert_image_to_3d("input.png", "output.dxf")
```

---

## 💡 نکات مهم

1. **PyTorch را جداگانه نصب کنید** (قبل از requirements.txt)
2. برای **GPU** حتماً نسخه CUDA مناسب را انتخاب کنید
3. اگر **خطا** گرفتید، راهنمای INSTALL_DIFFUSION.md را ببینید
4. برای **تست**، `python demo_diffusion.py` اجرا کنید

---

## 📞 در صورت مشکل

مشکلات رایج در **INSTALL_DIFFUSION.md** توضیح داده شده است.

**موفق باشید! 🚀**
