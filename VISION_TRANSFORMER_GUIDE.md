# Vision Transformer System for CAD Conversion

# سیستم Vision Transformer برای تبدیل CAD

این سیستم از **Vision Transformer** برای تحلیل عمیق و تبدیل دقیق نقشه‌های CAD استفاده می‌کند.

## 🎯 قابلیت‌ها

### 1. تحلیل معنایی عمیق

- شناسایی دقیق اجزاء ساختمانی (دیوار، در، پنجره، ستون، تیر، سقف، پله)
- درک روابط بین اجزاء
- تشخیص المان‌های پیچیده در نقشه‌های مهندسی

### 2. پیش‌بینی ارتفاع و عمق

- تخمین خودکار ارتفاع هر المان
- نقشه عمق برای بازسازی سه‌بعدی
- تشخیص مواد و ضخامت‌ها

### 3. بازسازی سه‌بعدی پیشرفته

- تبدیل دقیق 2D به 3D
- حفظ معنای معماری و مهندسی
- لایه‌بندی خودکار بر اساس نوع المان
- رنگ‌بندی هوشمند

### 4. تشخیص مقیاس خودکار

- یافتن ابعاد استاندارد (درها 2100mm)
- خواندن scale bar
- OCR برای متن ابعاد

## 📦 نصب

### پیش‌نیازها

```bash
# نصب PyTorch (CPU version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# برای GPU (اختیاری - سرعت بالاتر)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# وابستگی‌های دیگر
pip install opencv-python matplotlib scipy
pip install ezdxf
```

## 🚀 استفاده سریع

### 1. تبدیل ساده تصویر به 3D DXF

```python
from cad3d.vit_integration import get_vit_service

# سرویس را فعال کنید
service = get_vit_service(device="cpu")  # یا "cuda" برای GPU

if service:
    # تبدیل تصویر به 3D DXF
    stats = service.convert_image_to_3d_dxf(
        image_path="floor_plan.jpg",
        output_dxf="floor_plan_3d.dxf",
        auto_scale=True,
        min_confidence=0.5
    )
    
    print(f"✓ تولید شد {stats['total_entities']} entity")
    print(f"✓ لایه‌ها: {stats['total_layers']}")
    print(f"✓ اجزاء شناسایی شده: {stats['elements_by_class']}")
```

### 2. تحلیل عمیق تصویر

```python
from cad3d.vit_integration import get_vit_service

service = get_vit_service()

# تحلیل بدون تبدیل
analysis = service.analyze_image("drawing.png")

print(f"تعداد اجزاء: {analysis['num_elements']}")
print(f"میانگین اطمینان: {analysis['confidence_stats']['mean']:.2f}")

for elem in analysis['elements'][:10]:  # اولین 10 المان
    print(f"  {elem['class']}: {elem['confidence']:.2%}")
```

### 3. تصویرسازی نتایج

```python
service = get_vit_service()

# ایجاد تصویر تحلیل شامل:
# - نقشه معنایی (semantic segmentation)
# - نقشه اطمینان (confidence map)
# - نقشه ارتفاع (height map)
# - نقشه عمق (depth map)
# - نقشه attention
service.create_visualization(
    image_path="floor_plan.jpg",
    output_path="analysis_visualization.png"
)
```

## 🔧 استفاده پیشرفته

### آموزش مدل روی دیتاست خود

```python
from cad3d.vit_trainer import VisionTransformerTrainer, TrainingConfig, CADDataset
from cad3d.vision_transformer_cad import VisionTransformerConfig

# پیکربندی مدل
model_config = VisionTransformerConfig(
    image_size=512,
    patch_size=16,
    num_classes=50,  # تعداد کلاس‌های شما
    dim=768,
    depth=12,
    heads=12
)

# پیکربندی آموزش
train_config = TrainingConfig(
    train_data_dir="data/train",
    val_data_dir="data/val",
    batch_size=4,
    num_epochs=50,
    learning_rate=1e-4,
    device="cuda"  # یا "cpu"
)

# ایجاد دیتاست
train_dataset = CADDataset("data/train", augment=True)
val_dataset = CADDataset("data/val", augment=False)

# آموزش
trainer = VisionTransformerTrainer(model_config, train_config)
trainer.train(train_dataset, val_dataset)

# مدل در checkpoints/best_model.pth ذخیره می‌شود
```

### ساختار دیتاست

```
data/
  train/
    images/
      drawing_001.png
      drawing_002.png
      ...
    annotations/
      drawing_001.json
      drawing_002.json
      ...
  val/
    images/
    annotations/
```

### فرمت Annotation (JSON)

```json
{
  "semantic_map": [
    [0, 1, 1, 2, 2, ...],
    [0, 1, 1, 2, 2, ...],
    ...
  ],
  "height_map": [
    [0, 3000, 3000, 2100, 2100, ...],
    [0, 3000, 3000, 2100, 2100, ...],
    ...
  ],
  "depth_map": [
    [0, 0.5, 0.5, 0.3, 0.3, ...],
    [0, 0.5, 0.5, 0.3, 0.3, ...],
    ...
  ],
  "material_map": [
    [0, 1, 1, 2, 2, ...],
    [0, 1, 1, 2, 2, ...],
    ...
  ],
  "metadata": {
    "scale": 10.0,
    "drawing_type": "architectural",
    "units": "mm"
  }
}
```

### کلاس‌های پیش‌فرض

```python
# 50 کلاس شامل:
classes = [
    "background",      # 0
    "wall",            # 1
    "door",            # 2
    "window",          # 3
    "column",          # 4
    "beam",            # 5
    "slab",            # 6
    "stair",           # 7
    "railing",         # 8
    "furniture",       # 9
    # و 40 کلاس دیگر برای اجزاء ساختمانی، نمادها، و annotations
]
```

## 📐 ادغام با سرور

```python
from cad3d.vit_integration import get_vit_service, is_vit_available

# در سرور FastAPI
if is_vit_available():
    vit_service = get_vit_service(device="cpu")
    print("✓ Vision Transformer فعال است")
else:
    vit_service = None
    print("⚠️ Vision Transformer غیرفعال است")

# استفاده در endpoint
@app.post("/convert_advanced")
async def convert_advanced(file: UploadFile):
    if vit_service:
        # استفاده از Vision Transformer
        stats = vit_service.convert_image_to_3d_dxf(
            image_path=temp_file,
            output_dxf=output_file,
            auto_scale=True
        )
        return {"status": "success", "stats": stats}
    else:
        # Fallback به روش ساده
        return {"status": "vit_not_available"}
```

## ⚙️ تنظیمات مدل

### اندازه مدل

```python
VisionTransformerConfig(
    image_size=512,    # اندازه ورودی (512x512)
    patch_size=16,     # اندازه patch (16x16) = 32x32 patches
    dim=768,           # بعد embedding
    depth=12,          # تعداد لایه‌های transformer
    heads=12,          # تعداد attention heads
)

# تعداد پارامترها: ~86M (million)
```

### مدل سبک‌تر (برای CPU)

```python
VisionTransformerConfig(
    image_size=256,
    patch_size=16,
    dim=384,
    depth=6,
    heads=6
)

# تعداد پارامترها: ~22M
```

### مدل سنگین‌تر (برای GPU)

```python
VisionTransformerConfig(
    image_size=768,
    patch_size=16,
    dim=1024,
    depth=24,
    heads=16
)

# تعداد پارامترها: ~300M
```

## 📊 ارزیابی عملکرد

```python
from cad3d.vit_trainer import VisionTransformerTrainer

# بارگذاری checkpoint
trainer.load_checkpoint("checkpoints/best_model.pth")

# ارزیابی روی validation set
val_losses = trainer.validate(val_loader)

print(f"Validation Loss: {val_losses['total']:.4f}")
print(f"  Semantic Loss: {val_losses['semantic']:.4f}")
print(f"  Height Loss: {val_losses['height']:.4f}")
print(f"  Depth Loss: {val_losses['depth']:.4f}")
```

## 🎨 مثال کامل: Pipeline پیشرفته

```python
import cv2
from cad3d.vit_integration import get_vit_service

# 1. بارگذاری تصویر
image = cv2.imread("complex_floor_plan.jpg")

# 2. سرویس Vision Transformer
service = get_vit_service(
    model_path="checkpoints/best_model.pth",  # مدل آموزش‌دیده
    device="cuda"  # استفاده از GPU
)

# 3. تحلیل
print("🔍 در حال تحلیل نقشه...")
analysis = service.analyze_image("complex_floor_plan.jpg")

print(f"✓ {analysis['num_elements']} المان شناسایی شد")
print(f"✓ میانگین اطمینان: {analysis['confidence_stats']['mean']:.2%}")

# 4. بازسازی 3D
print("🏗️ در حال بازسازی مدل سه‌بعدی...")
stats = service.convert_image_to_3d_dxf(
    image_path="complex_floor_plan.jpg",
    output_dxf="complex_floor_plan_3d.dxf",
    auto_scale=True,
    min_confidence=0.6
)

print(f"✓ {stats['total_entities']} entity سه‌بعدی تولید شد")
print(f"✓ {stats['total_layers']} لایه ایجاد شد")
print("✓ اجزاء شناسایی شده:")
for class_name, count in stats['elements_by_class'].items():
    print(f"    {class_name}: {count}")

# 5. تصویرسازی نتایج
print("📊 ایجاد تصویرسازی...")
service.create_visualization(
    image_path="complex_floor_plan.jpg",
    output_path="analysis_result.png"
)

print("✅ تمام!")
```

## 🔬 معماری Vision Transformer

```
Input Image (512x512x3)
  ↓
Patch Embedding (32x32 patches × 768 dim)
  ↓
Add Positional Encoding
  ↓
[CLS] Token + Patch Tokens
  ↓
Transformer Encoder (12 layers)
  ├─ Multi-Head Self-Attention (12 heads)
  ├─ Layer Normalization
  ├─ Feed-Forward Network
  └─ Residual Connections
  ↓
Output Embeddings (1024 tokens × 768 dim)
  ↓
Prediction Heads:
  ├─ Semantic Segmentation (50 classes)
  ├─ Height Prediction (mm)
  ├─ Depth Prediction (normalized)
  └─ Material Classification (10 types)
```

## 📈 مزایای Vision Transformer

✅ **درک جامع**: تحلیل تمام بخش‌های نقشه به‌طور همزمان
✅ **روابط فضایی**: attention mechanism روابط بین اجزاء را می‌فهمد
✅ **دقت بالا**: شناسایی المان‌های پیچیده و کوچک
✅ **انعطاف‌پذیری**: قابلیت آموزش روی انواع نقشه‌های مهندسی
✅ **مقیاس‌پذیری**: از مدل سبک تا مدل بسیار بزرگ

## ⚡ نکات بهینه‌سازی

### 1. سرعت (CPU)

```python
# استفاده از مدل کوچک‌تر
service = get_vit_service(device="cpu")
# پیش‌پردازش batch
# کش کردن نتایج
```

### 2. دقت (GPU)

```python
# استفاده از مدل بزرگ‌تر
service = get_vit_service(device="cuda")
# آموزش روی دیتاست سفارشی
# استفاده از ensemble
```

### 3. حافظه

```python
# کاهش batch size
# استفاده از gradient checkpointing
# Mixed precision training
```

## 🐛 عیب‌یابی

### PyTorch نصب نیست

```bash
pip install torch torchvision
```

### Out of Memory (GPU)

```python
# کاهش اندازه مدل
config.dim = 384
config.depth = 6
# یا استفاده از CPU
service = get_vit_service(device="cpu")
```

### دقت پایین

```python
# افزایش confidence threshold
min_confidence=0.7
# آموزش روی دیتاست بیشتر
# استفاده از مدل بزرگ‌تر
```

## 📚 منابع

- [Vision Transformer Paper](https://arxiv.org/abs/2010.11929)
- [DETR (Detection Transformer)](https://arxiv.org/abs/2005.12872)
- [Segment Anything Model (SAM)](https://arxiv.org/abs/2304.02643)

## 📝 License

MIT License - استفاده آزاد برای پروژه‌های تجاری و غیرتجاری

---

**نسخه**: 1.0.0
**تاریخ**: 2025-01-16
**توسعه‌دهنده**: CAD3D Team
