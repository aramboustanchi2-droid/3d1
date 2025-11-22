# 3D Diffusion Model for CAD Conversion

## مدل انتشار سه‌بعدی - قدرتمندترین روش تولید 3D

این سیستم از **Diffusion Models** (مشابه Stable Diffusion 3D, Point-E, DeepFloyd) برای تبدیل نقشه‌های 2D به مدل‌های 3D دقیق استفاده می‌کند.

---

## 🚀 قابلیت‌های اصلی

### 1. **Architecture (معماری)**

- **DDPM** (Denoising Diffusion Probabilistic Models): روش اصلی تولید با کیفیت بالا
- **DDIM Sampling**: تولید سریع (10-50 گام به جای 1000 گام)
- **PointNet++**: درک و پردازش ابرنقطه‌های 3D
- **CLIP Image Encoder**: رمزنگاری ویژگی‌های 2D برای هدایت تولید 3D
- **U-Net 3D**: شبکه اصلی denoising با attention mechanisms

### 2. **Vision Transformer Integration**

- ترکیب ViT با Diffusion برای قدرت بیشتر
- استخراج ویژگی‌های معنایی (semantic)، ارتفاع (height)، عمق (depth)، و مواد (materials)
- Feature Fusion Layer برای ترکیب ویژگی‌های ViT و Diffusion
- نتیجه: **دقت و جزئیات چندین برابر بیشتر**

### 3. **Continuous Learning (یادگیری مداوم)**

- **Experience Replay Buffer**: ذخیره تبدیل‌های اخیر
- یادگیری خودکار از هر تبدیل
- بهبود تدریجی مدل با استفاده
- هر 10 تبدیل → یک بار به‌روزرسانی مدل

### 4. **Multi-Stage Training**

- Pre-training روی داده‌های سینتتیک
- Fine-tuning روی نقشه‌های واقعی CAD
- Progressive resolution training
- Loss scheduling برای آموزش بهینه

---

## 📦 نصب و راه‌اندازی

### نیازمندی‌ها

```bash
pip install torch torchvision
pip install ezdxf opencv-python numpy scipy matplotlib
```

### استفاده ساده

```python
from cad3d.hybrid_vit_diffusion import create_hybrid_converter

# ایجاد converter
converter = create_hybrid_converter(
    device="cuda",  # یا "cpu"
    enable_learning=True  # فعال‌سازی یادگیری مداوم
)

# تبدیل تصویر به 3D
results = converter.convert_image_to_3d(
    image_path="plan.png",
    output_path="plan_3d.dxf",
    sampling_steps=50,  # تعداد گام‌های sampling (کمتر = سریع‌تر)
    learn_from_conversion=True  # یادگیری از این تبدیل
)
```

---

## 🎓 آموزش مدل

### 1. ایجاد داده‌های آموزشی

```python
from cad3d.diffusion_trainer import create_synthetic_training_data

# تولید داده‌های سینتتیک برای شروع
create_synthetic_training_data(
    output_dir="training_data/diffusion_synthetic",
    num_samples=500
)
```

**ساختار داده‌های آموزشی:**

```
training_data/
├── images/
│   ├── drawing_001.png
│   ├── drawing_002.png
│   └── ...
└── pointclouds/
    ├── drawing_001.npy  # (N, 3) numpy array
    ├── drawing_002.npy
    └── ...
```

### 2. آموزش اولیه

```python
from cad3d.diffusion_trainer import DiffusionTrainer, CAD2D3DDataset
from cad3d.diffusion_3d_model import create_diffusion_model

# ایجاد dataset
dataset = CAD2D3DDataset(
    data_dir="training_data/diffusion_synthetic",
    image_size=256,
    num_points=4096,
    augment=True
)

# Split train/val
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(
    dataset, [train_size, val_size]
)

# ایجاد مدل
model = create_diffusion_model(
    num_points=4096,
    timesteps=1000,
    device="cuda"
)

# ایجاد trainer
trainer = DiffusionTrainer(
    model=model,
    device="cuda",
    learning_rate=1e-4
)

# آموزش
trainer.train(
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    epochs=100,
    batch_size=8,
    save_every=10
)
```

### 3. آموزش روی داده‌های واقعی

```python
# استفاده از نقشه‌های CAD واقعی
real_dataset = CAD2D3DDataset(
    data_dir="training_data/real_cad",
    image_size=256,
    num_points=4096,
    augment=True
)

# بارگذاری مدل pre-trained
trainer.load_checkpoint("trained_models/diffusion/diffusion_best.pth")

# Fine-tuning
trainer.train(
    train_dataset=real_dataset,
    epochs=50,
    batch_size=4,
    save_every=5
)
```

---

## 🔬 نمایش و آزمایش

```bash
# اجرای demo کامل
python demo_diffusion.py
```

این demo شامل:

1. ✅ تبدیل ساده تصویر به 3D
2. ✅ تبدیل batch با یادگیری
3. ✅ نمایش یادگیری مداوم
4. ✅ مقایسه روش‌های sampling (DDPM vs DDIM)
5. ✅ اطلاعات معماری مدل

---

## 🎯 Pipeline کامل

```
┌─────────────┐
│ Input Image │ (2D CAD Drawing)
└──────┬──────┘
       │
       ▼
┌─────────────────────┐
│ Vision Transformer  │ Extract rich features:
│ (ViT)               │ - Semantic classes
└──────┬──────────────┘ - Height map
       │                 - Depth map
       │                 - Materials
       ▼
┌─────────────────────┐
│ Feature Fusion      │ Combine ViT + CLIP features
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│ 3D Diffusion Model  │ Generate point cloud:
│ (DDIM Sampling)     │ - Start from noise
└──────┬──────────────┘ - Denoise step by step
       │                 - Guided by 2D features
       │                 - 10-50 steps
       ▼
┌─────────────────────┐
│ Point Cloud         │ (N, 3) 3D coordinates
│ Enhancement         │ + semantic colors
└──────┬──────────────┘ + height information
       │
       ▼
┌─────────────────────┐
│ DXF Mesh Export     │ Convert to CAD format
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│ Output 3D DXF       │ ✅ Ready for CAD software
└─────────────────────┘
       │
       ▼ (if learning enabled)
┌─────────────────────┐
│ Experience Replay   │ Store for learning
│ Buffer              │ Periodic model update
└─────────────────────┘
```

---

## 📊 مقایسه با روش‌های دیگر

| روش | دقت | سرعت | یادگیری | پیچیدگی |
|-----|------|------|---------|----------|
| **Simple Extrusion** | ⭐⭐ | ⭐⭐⭐⭐⭐ | ❌ | ⭐ |
| **Vision Transformer** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ✅ | ⭐⭐⭐ |
| **3D Diffusion** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ✅ | ⭐⭐⭐⭐ |
| **Hybrid (ViT + Diffusion)** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ✅✅ | ⭐⭐⭐⭐ |

### مزایای Diffusion Model

1. ✅ **جزئیات دقیق**: تولید geometry پیچیده
2. ✅ **انعطاف‌پذیری**: قابل آموزش روی هر نوع داده
3. ✅ **Scalability**: از simple تا complex
4. ✅ **State-of-the-art**: بهترین روش فعلی در تحقیقات
5. ✅ **Continuous Learning**: بهبود با استفاده

---

## 🔧 تنظیمات پیشرفته

### Sampling Methods

#### DDPM (کیفیت بالا، کند)

```python
# 1000 steps, maximum quality
point_cloud = diffusion.p_sample_loop(
    shape=(batch_size, 4096, 3),
    condition=features,
    device="cuda",
    progress=True
)
```

#### DDIM (سریع، کیفیت خوب)

```python
# 50 steps, 20x faster
point_cloud = diffusion.ddim_sample(
    shape=(batch_size, 4096, 3),
    condition=features,
    steps=50,
    eta=0.0,  # 0.0 = deterministic, 1.0 = stochastic
    device="cuda"
)
```

### تعداد نقاط

```python
# کم: سریع، کم‌جزئیات
model = create_diffusion_model(num_points=1024)

# متوسط: توازن خوب
model = create_diffusion_model(num_points=2048)

# زیاد: جزئیات بالا، کندتر
model = create_diffusion_model(num_points=8192)
```

### Learning Rate Schedule

```python
# برای fine-tuning
trainer = DiffusionTrainer(
    model=model,
    learning_rate=1e-5  # کمتر برای stability
)

# برای training از صفر
trainer = DiffusionTrainer(
    model=model,
    learning_rate=1e-4  # بیشتر برای learning سریع
)
```

---

## 📈 Training Tips

### 1. Progressive Training

```python
# مرحله 1: Resolution پایین، سریع
train_on_low_res(image_size=128, epochs=20)

# مرحله 2: Resolution متوسط
train_on_medium_res(image_size=256, epochs=30)

# مرحله 3: Resolution بالا
train_on_high_res(image_size=512, epochs=50)
```

### 2. Data Augmentation

```python
dataset = CAD2D3DDataset(
    data_dir="...",
    augment=True  # فعال‌سازی augmentation:
                  # - Horizontal flip
                  # - Brightness/contrast
                  # - Rotation (optional)
)
```

### 3. Monitoring

```python
# بررسی loss history
import matplotlib.pyplot as plt

plt.plot(trainer.loss_history)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Progress')
plt.savefig('training_loss.png')
```

---

## 🎯 Use Cases

### 1. Architectural Floor Plans

```python
converter.convert_image_to_3d(
    image_path="floor_plan.png",
    output_path="building_3d.dxf",
    sampling_steps=50
)
# نتیجه: ساختمان 3D با اتاق‌ها، دیوارها، درها
```

### 2. Mechanical Parts

```python
converter.convert_image_to_3d(
    image_path="part_drawing.png",
    output_path="part_3d.dxf",
    sampling_steps=100  # More steps for precision
)
# نتیجه: قطعه مکانیکی دقیق
```

### 3. Landscape Design

```python
converter.convert_image_to_3d(
    image_path="landscape_plan.png",
    output_path="terrain_3d.dxf",
    sampling_steps=50
)
# نتیجه: توپوگرافی، درختان، مسیرها
```

---

## 🚀 Performance Optimization

### GPU Acceleration

```python
# بررسی CUDA
if torch.cuda.is_available():
    device = "cuda"
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = "cpu"
    print("Using CPU (slower)")

converter = create_hybrid_converter(device=device)
```

### Batch Processing

```python
# پردازش چندین فایل همزمان
image_paths = list(Path("input_images").glob("*.png"))

for img_path in image_paths:
    output_path = Path("output") / f"{img_path.stem}_3d.dxf"
    converter.convert_image_to_3d(img_path, output_path)
```

### Memory Management

```python
# برای GPU با memory کم
model = create_diffusion_model(
    num_points=2048,  # کمتر از 4096
    timesteps=1000,
    device="cuda"
)

# استفاده از batch_size کوچک
trainer.train(batch_size=2)  # به جای 8
```

---

## 📚 References

این implementation الهام‌گرفته از:

1. **DDPM** - Denoising Diffusion Probabilistic Models (Ho et al., 2020)
2. **DDIM** - Denoising Diffusion Implicit Models (Song et al., 2021)
3. **Point-E** - OpenAI's Point Cloud Diffusion (Nichol et al., 2022)
4. **Stable Diffusion** - Stability.ai (Rombach et al., 2022)
5. **PointNet++** - Deep Learning on Point Sets (Qi et al., 2017)
6. **DreamFusion** - Text-to-3D using 2D Diffusion (Poole et al., 2022)

---

## 💡 مثال کامل

```python
# 1. Import
from cad3d.hybrid_vit_diffusion import create_hybrid_converter

# 2. Create converter
converter = create_hybrid_converter(
    device="cuda",
    enable_learning=True
)

# 3. Convert single image
results = converter.convert_image_to_3d(
    image_path="my_plan.png",
    output_path="my_plan_3d.dxf",
    sampling_steps=50,
    learn_from_conversion=True
)

# 4. Check results
print(f"Generated {results['num_points']} points")
print(f"Time: {results['conversion_time']:.2f}s")
print(f"Learning updates: {results['learning_updates']}")

# 5. Open in CAD software
# my_plan_3d.dxf → AutoCAD, FreeCAD, etc.
```

---

## ✅ خلاصه

**3D Diffusion Model** قدرتمندترین و پیشرفته‌ترین روش برای تبدیل نقشه‌های 2D به مدل‌های 3D است.

### ویژگی‌های کلیدی

- 🎯 **دقت بالا**: جزئیات دقیق و realistic
- 🚀 **سرعت قابل قبول**: با DDIM sampling
- 🧠 **یادگیری هوشمند**: بهبود مداوم با استفاده
- 🔧 **انعطاف‌پذیر**: قابل تنظیم برای هر use case
- 📦 **ادغام آسان**: API ساده و واضح

### چرا Diffusion?

- ✅ State-of-the-art در تحقیقات AI
- ✅ استفاده در Stable Diffusion, DALL-E, Midjourney
- ✅ قابلیت تولید جزئیات پیچیده
- ✅ قابل training روی داده‌های سفارشی
- ✅ Continuous improvement با experience replay

**این سیستم قدرت هوش مصنوعی شما را صدها برابر افزایش می‌دهد! 🚀**
