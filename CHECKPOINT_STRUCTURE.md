# 📁 Checkpoint and Directory Structure Documentation

این سند ساختار کامل checkpoint‌ها، دایرکتری‌ها و فایل‌های آموزشی پروژه CAD 3D را مستند می‌کند.

## ✅ تأیید شده (Verified)

تمام ساختارهای زیر با تست جامع `test_checkpoint_structure.py` تأیید شده‌اند.

---

## 🗂️ ساختار کلی پروژه

```
3d/
├── training_data/               # داده‌های آموزشی
│   ├── diffusion_synthetic/     # داده‌های سینتتیک برای Diffusion
│   │   ├── images/              # تصاویر 2D (PNG)
│   │   └── pointclouds/         # ابرهای نقطه 3D (NPY)
│   ├── real_cad/                # داده‌های واقعی CAD
│   │   ├── images/
│   │   └── pointclouds/
│   └── vae_data/                # داده‌های VAE
│       ├── images/
│       └── pointclouds/
│
├── trained_models/              # مدل‌های آموزش‌دیده
│   ├── vae/                     # VAE checkpoints
│   │   ├── vae_best.pth         # بهترین مدل
│   │   ├── vae_epoch_N.pth      # checkpoint هر epoch
│   │   ├── vae_epoch_log.json   # لاگ هر epoch
│   │   └── vae_training_report.json  # گزارش نهایی
│   │
│   ├── diffusion/               # Diffusion checkpoints
│   │   ├── diffusion_best.pth   # بهترین مدل
│   │   ├── diffusion_epoch_N.pth # checkpoint هر N epoch
│   │   └── training_report.json # گزارش آموزش
│   │
│   ├── vit/                     # Vision Transformer checkpoints
│   │   ├── final_model.pth
│   │   ├── checkpoint_epoch_N.pth
│   │   └── training_history.json
│   │
│   └── hybrid/                  # Hybrid model (ViT+Diffusion)
│       └── continuous_learning.pth
│
├── models/                      # مدل‌های دانلود شده
│   ├── midas_v2_small_256.onnx
│   └── example_vit.pth
│
└── outputs/                     # خروجی‌های تولید شده
    └── (تست‌ها و خروجی‌های موقت)
```

---

## 📦 VAE Checkpoint Structure

### فایل‌های تولید شده

```python
trained_models/vae/
├── vae_best.pth                 # بهترین مدل (کمترین val_loss)
├── vae_epoch_1.pth              # Checkpoint epoch 1
├── vae_epoch_2.pth              # Checkpoint epoch 2
├── ...
├── vae_epoch_N.pth              # Checkpoint epoch N
├── vae_epoch_log.json           # لاگ هر epoch (KL weight progression)
└── vae_training_report.json     # گزارش کامل آموزش
```

### محتویات Checkpoint (.pth)

```python
{
    'epoch': int,                 # شماره epoch
    'state_dict': OrderedDict,    # وزن‌های مدل
    'opt': dict,                  # state optimizer
    'scheduler': dict,            # state learning rate scheduler
    'val_loss': float,            # validation loss
    'loss_history': list,         # تاریخچه loss ها
    'last_parts': dict,           # جزئیات loss ها (chamfer, kl, voxel, smooth)
    'kl_weight': float            # وزن فعلی KL divergence
}
```

### محتویات Epoch Log (JSON)

```json
[
    {
        "epoch": 1,
        "train_loss": 0.1352,
        "val_loss": 0.3727,
        "kl_weight": 0.0001
    },
    {
        "epoch": 2,
        "train_loss": 0.0736,
        "val_loss": 0.2344,
        "kl_weight": 0.0002
    }
]
```

### بارگذاری VAE Checkpoint

```python
from cad3d.vae_integration import VAEConverter

# با checkpoint
converter = VAEConverter(
    device='cuda',
    checkpoint='trained_models/vae/vae_best.pth',
    num_points=2048
)

# بدون checkpoint (وزن‌های اولیه)
converter = VAEConverter(device='cuda', num_points=2048)
```

---

## 🌊 Diffusion Checkpoint Structure

### فایل‌های تولید شده

```python
trained_models/diffusion/
├── diffusion_best.pth           # بهترین مدل
├── diffusion_epoch_10.pth       # هر 10 epoch
├── diffusion_epoch_20.pth
├── ...
└── training_report.json         # گزارش آموزش
```

### محتویات Checkpoint (.pth)

```python
{
    'epoch': int,                      # شماره epoch
    'global_step': int,                # تعداد کل batch های پردازش شده
    'image_encoder_state': OrderedDict, # وزن‌های image encoder
    'unet_state': OrderedDict,         # وزن‌های U-Net
    'optimizer_state': dict,           # state optimizer
    'scheduler_state': dict,           # state scheduler
    'loss': float,                     # loss فعلی
    'best_loss': float,                # بهترین loss
    'loss_history': list               # تاریخچه loss ها
}
```

### محتویات Training Report (JSON)

```json
{
    "model": "3D Diffusion Model",
    "architecture": "DDPM with U-Net + PointNet++",
    "total_epochs": 50,
    "total_steps": 1250,
    "best_loss": 0.123456,
    "final_loss": 0.134567,
    "training_time_hours": 2.5,
    "loss_history": [0.9, 0.8, 0.7, ...],
    "replay_buffer_size": 800,
    "device": "cuda",
    "hyperparameters": {
        "timesteps": 1000,
        "num_points": 4096,
        "learning_rate": 0.0001
    }
}
```

### بارگذاری Diffusion Checkpoint

```python
from cad3d.diffusion_3d_model import create_diffusion_model
from cad3d.diffusion_trainer import DiffusionTrainer

model = create_diffusion_model(num_points=4096, device='cuda')
trainer = DiffusionTrainer(model=model, device='cuda')

# بارگذاری checkpoint
trainer.load_checkpoint('trained_models/diffusion/diffusion_best.pth')
```

---

## 🎯 Vision Transformer Checkpoint Structure

### فایل‌های تولید شده

```python
checkpoints/                      # پیش‌فرض: config.checkpoint_dir
├── final_model.pth               # مدل نهایی
├── checkpoint_epoch_5.pth        # هر 5 epoch
├── checkpoint_epoch_10.pth
├── ...
└── training_history.json         # تاریخچه آموزش
```

### محتویات Checkpoint (.pth)

```python
{
    'epoch': int,
    'model_state_dict': OrderedDict,
    'optimizer_state_dict': dict,
    'best_val_loss': float,
    'model_config': dict,            # VisionTransformerConfig
    'train_config': dict,            # TrainingConfig
    'scheduler_state_dict': dict     # اختیاری
}
```

### بارگذاری ViT Checkpoint

```python
from cad3d.vit_trainer import VisionTransformerTrainer
from cad3d.vision_transformer_cad import VisionTransformerConfig

config = VisionTransformerConfig(...)
trainer = VisionTransformerTrainer(config, train_config)
trainer.load_checkpoint('final_model.pth')
```

---

## 🔄 Hybrid Converter (Optional Weights)

### استفاده بدون وزن‌ها (Graceful Degradation)

```python
from cad3d.hybrid_vit_diffusion import HybridCAD3DConverter

# هیچ checkpoint موجود نیست
converter = HybridCAD3DConverter(
    device='cuda',
    vit_model_path=None,
    diffusion_model_path=None,
    enable_learning=True  # یادگیری مستمر فعال
)
# ✅ کار می‌کند با وزن‌های اولیه

# با checkpoint های موجود
converter = HybridCAD3DConverter(
    device='cuda',
    vit_model_path='trained_models/vit/final_model.pth',
    diffusion_model_path='trained_models/diffusion/diffusion_best.pth',
    enable_learning=True
)
# ✅ با وزن‌های آموزش‌دیده کار می‌کند
```

---

## 📊 Dataset Structure

### Synthetic Dataset (Auto-generated)

```python
training_data/diffusion_synthetic/
├── images/
│   ├── synthetic_0000.png       # 256x256 grayscale drawing
│   ├── synthetic_0001.png
│   └── ...
└── pointclouds/
    ├── synthetic_0000.npy       # (N, 3) float32 array
    ├── synthetic_0001.npy
    └── ...
```

### تولید Dataset

```python
from cad3d.diffusion_trainer import create_synthetic_training_data

create_synthetic_training_data(
    output_dir='training_data/diffusion_synthetic',
    num_samples=500
)
```

### استفاده از Dataset

```python
from cad3d.diffusion_trainer import CAD2D3DDataset

dataset = CAD2D3DDataset(
    data_dir='training_data/diffusion_synthetic',
    image_size=256,
    num_points=2048,
    augment=True
)

print(f"Dataset size: {len(dataset)}")
image, pointcloud = dataset[0]  # torch.Tensor (3,256,256), (2048,3)
```

---

## 🧪 تست و تأیید

### اجرای تست جامع

```bash
python cad3d/tests/test_checkpoint_structure.py
```

این تست موارد زیر را بررسی می‌کند:

✅ **VAE Training**

- ایجاد صحیح دایرکتری `trained_models/vae/`
- ذخیره checkpoint برای هر epoch
- ذخیره best checkpoint
- تولید epoch log با KL weight progression
- تولید training report

✅ **Diffusion Training**

- ایجاد صحیح دایرکتری `trained_models/diffusion/`
- ذخیره checkpoint های دوره‌ای
- ذخیره best checkpoint
- تولید training report با تمام hyperparameter ها

✅ **Dataset Generation**

- ایجاد ساختار دایرکتری `images/` و `pointclouds/`
- تولید تعداد صحیح فایل‌ها
- pairing صحیح image-pointcloud

✅ **Optional Weight Loading**

- بارگذاری نرم افزاری با checkpoint های missing
- بارگذاری نرم افزاری با checkpoint=None
- Graceful degradation برای مدل‌های hybrid

---

## 🚀 Best Practices

### 1. Checkpoint Management

```python
# همیشه best model را ذخیره کنید
if val_loss < best_loss:
    save_checkpoint(epoch, val_loss, is_best=True)

# checkpoint های دوره‌ای برای resume training
if epoch % 10 == 0:
    save_checkpoint(epoch, val_loss)
```

### 2. Directory Creation

```python
from pathlib import Path

save_dir = Path('trained_models/vae')
save_dir.mkdir(parents=True, exist_ok=True)  # ایجاد امن
```

### 3. Graceful Loading

```python
# همیشه چک کنید checkpoint موجود است
if checkpoint_path and checkpoint_path.exists():
    load_checkpoint(checkpoint_path)
else:
    print("ℹ️  Using untrained weights")
```

### 4. JSON Logging

```python
# برای plotting و analysis بعدی
epoch_logs = []
for epoch in range(epochs):
    log = {
        'epoch': epoch,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'kl_weight': kl_weight
    }
    epoch_logs.append(log)

with open('epoch_log.json', 'w') as f:
    json.dump(epoch_logs, f, indent=2)
```

---

## 📈 نظارت بر آموزش

### مشاهده KL Weight Progression (VAE)

```python
import json

with open('trained_models/vae/vae_epoch_log.json') as f:
    logs = json.load(f)

for log in logs:
    print(f"Epoch {log['epoch']}: "
          f"train={log['train_loss']:.4f}, "
          f"val={log['val_loss']:.4f}, "
          f"kl_w={log['kl_weight']:.6f}")
```

### مشاهده Loss History (Diffusion)

```python
import json

with open('trained_models/diffusion/training_report.json') as f:
    report = json.load(f)

print(f"Total epochs: {report['total_epochs']}")
print(f"Best loss: {report['best_loss']:.6f}")
print(f"Training time: {report['training_time_hours']:.2f} hours")
```

---

## ⚠️ مشکلات رایج و راه‌حل‌ها

### مشکل 1: Checkpoint files missing

**علت**: مسیر save_dir نادرست یا دسترسی نوشتن ندارد

**راه‌حل**:

```python
save_dir.mkdir(parents=True, exist_ok=True)
```

### مشکل 2: Memory error هنگام load checkpoint

**علت**: GPU memory کافی نیست

**راه‌حل**:

```python
checkpoint = torch.load(path, map_location='cpu')  # ابتدا به CPU
model.load_state_dict(checkpoint['state_dict'])
model = model.to('cuda')  # سپس به GPU
```

### مشکل 3: Dataset pairing mismatch

**علت**: نام فایل‌های image و pointcloud یکسان نیست

**راه‌حل**:

```python
# همیشه از stem یکسان استفاده کنید
image_path = images_dir / f"sample_{i:04d}.png"
pc_path = pointclouds_dir / f"sample_{i:04d}.npy"
```

---

## 📝 خلاصه

✅ تمام ساختارهای checkpoint تست و تأیید شده‌اند  
✅ Optional weight loading به درستی کار می‌کند  
✅ Dataset generation و pairing صحیح است  
✅ JSON logging برای تمام مدل‌ها فعال است  
✅ Graceful degradation برای missing checkpoints پیاده‌سازی شده  

برای مشاهده کد تست کامل: `cad3d/tests/test_checkpoint_structure.py`
