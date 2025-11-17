# 🎓 راهنمای آموزش مدل‌های تشخیص CAD

این راهنما نحوه استفاده از سیستم آموزش (Training System) برای ساخت Dataset و آموزش مدل‌های سفارشی تشخیص CAD را توضیح می‌دهد.

## 📋 فهرست مطالب

1. [نصب Dependencies](#نصب-dependencies)
2. [ساخت Dataset از فایل‌های DXF](#ساخت-dataset-از-فایلهای-dxf)
3. [آموزش مدل](#آموزش-مدل)
4. [استفاده از مدل آموزش‌دیده](#استفاده-از-مدل-آموزشدیده)
5. [نکات و توصیه‌ها](#نکات-و-توصیهها)

---

## نصب Dependencies

```bash
# نصب کتابخانه‌های Neural Network
pip install -r requirements-neural.txt

# نصب PyTorch (برای آموزش)
# CPU version:
pip install torch torchvision torchaudio

# GPU version (CUDA 11.8):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

## ساخت Dataset از فایل‌های DXF

### روش اول: CLI

```bash
# ساخت Dataset با فرمت COCO
python -m cad3d.cli build-dataset \
  --input-dir ./my_dxf_files \
  --output-dir ./training_dataset \
  --format coco \
  --visualize

# ساخت با جستجوی زیرپوشه‌ها و فرمت YOLO
python -m cad3d.cli build-dataset \
  --input-dir ./my_dxf_files \
  --output-dir ./training_dataset \
  --format yolo \
  --recurse \
  --visualize

# ساخت با هر دو فرمت COCO و YOLO
python -m cad3d.cli build-dataset \
  --input-dir ./my_dxf_files \
  --output-dir ./training_dataset \
  --format both \
  --image-size 1024 1024 \
  --recurse \
  --visualize
```

**پارامترها:**

- `--input-dir`: پوشه حاوی فایل‌های DXF
- `--output-dir`: پوشه خروجی برای Dataset
- `--format`: فرمت خروجی (`coco`, `yolo`, `both`)
- `--image-size WIDTH HEIGHT`: اندازه تصاویر (پیش‌فرض: 1024 1024)
- `--recurse`: جستجوی زیرپوشه‌ها
- `--visualize`: ذخیره تصاویر بررسی annotation

### روش دوم: Python API

```python
from cad3d.training_dataset_builder import CADDatasetBuilder

# ساخت builder
builder = CADDatasetBuilder(output_dir="./training_dataset")

# اضافه کردن فایل‌های DXF
builder.add_dxf_to_dataset("floor_plan_1.dxf", image_size=(1024, 1024))
builder.add_dxf_to_dataset("floor_plan_2.dxf", image_size=(1024, 1024))
builder.add_dxf_to_dataset("floor_plan_3.dxf", image_size=(1024, 1024))

# Export به فرمت COCO
builder.export_coco_format()

# Export به فرمت YOLO
builder.export_yolo_format()

# تولید تصاویر بررسی annotation
builder.visualize_annotations()

print(f"✅ Dataset آماده است!")
print(f"   تعداد تصاویر: {len(builder.images)}")
print(f"   تعداد annotation: {len(builder.annotations)}")
```

### ساختار Dataset (COCO Format)

```
training_dataset/
├── images/                 # تصاویر PNG تولید شده از DXF
│   ├── floor_plan_1.png
│   ├── floor_plan_2.png
│   └── ...
├── annotations.json        # فرمت COCO
├── labels/                 # فرمت YOLO (اختیاری)
│   ├── floor_plan_1.txt
│   ├── floor_plan_2.txt
│   └── ...
└── visualizations/         # تصاویر بررسی (اختیاری)
    ├── floor_plan_1_annotated.png
    └── ...
```

---

## آموزش مدل

### روش اول: CLI

```bash
# آموزش با تنظیمات پیش‌فرض
python -m cad3d.cli train \
  --dataset-dir ./training_dataset \
  --output-dir ./models \
  --epochs 50 \
  --batch-size 4 \
  --device cuda

# آموزش با تنظیمات پیشرفته
python -m cad3d.cli train \
  --dataset-dir ./training_dataset \
  --output-dir ./models \
  --epochs 100 \
  --batch-size 8 \
  --lr 0.005 \
  --device cuda \
  --workers 8 \
  --optimizer adam \
  --pretrained

# ادامه آموزش از checkpoint
python -m cad3d.cli train \
  --dataset-dir ./training_dataset \
  --output-dir ./models \
  --epochs 50 \
  --resume ./models/checkpoint_epoch_30.pth \
  --device cuda
```

**پارامترها:**

- `--dataset-dir`: پوشه Dataset (COCO format)
- `--output-dir`: پوشه خروجی checkpoints
- `--epochs`: تعداد epochs (پیش‌فرض: 50)
- `--batch-size`: اندازه batch (پیش‌فرض: 4)
- `--lr`: learning rate (پیش‌فرض: 0.001)
- `--device`: `cuda` یا `cpu` (پیش‌فرض: cuda)
- `--workers`: تعداد data loader workers (پیش‌فرض: 4)
- `--optimizer`: `sgd` یا `adam` (پیش‌فرض: sgd)
- `--resume`: مسیر checkpoint برای ادامه آموزش
- `--pretrained`: استفاده از pre-trained weights

### روش دوم: Python API

```python
from cad3d.training_pipeline import CADDetectionTrainer
import torch

# تنظیم device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ساخت trainer
trainer = CADDetectionTrainer(
    data_dir="./training_dataset",
    output_dir="./models",
    batch_size=4,
    num_workers=4,
    device=device,
    pretrained=True
)

# تنظیم optimizer
trainer.setup_optimizer(
    optimizer_type="sgd",
    learning_rate=0.001
)

# آموزش
trainer.train(num_epochs=50)

print("✅ آموزش تمام شد!")
```

### خروجی‌های آموزش

```
models/
├── best_model.pth              # بهترین مدل (کمترین validation loss)
├── checkpoint_epoch_10.pth     # Checkpoint هر 10 epoch
├── checkpoint_epoch_20.pth
├── checkpoint_epoch_30.pth
└── ...
```

---

## استفاده از مدل آموزش‌دیده

### بارگذاری مدل سفارشی در NeuralCADDetector

```python
from cad3d.neural_cad_detector import NeuralCADDetector
import torch

# ساخت detector با مدل سفارشی
detector = NeuralCADDetector(device="cuda")

# بارگذاری weights آموزش‌دیده
checkpoint = torch.load("./models/best_model.pth")
detector.detection_model.load_state_dict(checkpoint['model_state_dict'])
detector.detection_model.eval()

# استفاده برای تشخیص
image_path = "scanned_floor_plan.jpg"
elements = detector.detect_from_image(image_path, confidence_threshold=0.5)

print(f"✅ {len(elements)} عنصر تشخیص داده شد:")
for elem in elements:
    print(f"   - {elem.element_type}: {elem.confidence:.2%}")
```

### مثال کامل: PDF → DXF با مدل سفارشی

```python
from cad3d.neural_cad_detector import NeuralCADDetector
from cad3d.pdf_processor import PDFToImageConverter, CADPipeline
import torch

# بارگذاری مدل سفارشی
detector = NeuralCADDetector(device="cuda")
checkpoint = torch.load("./models/best_model.pth")
detector.detection_model.load_state_dict(checkpoint['model_state_dict'])
detector.detection_model.eval()

# ساخت pipeline
pdf_converter = PDFToImageConverter(dpi=300)
pipeline = CADPipeline(
    neural_detector=detector,
    pdf_converter=pdf_converter
)

# تبدیل PDF به DXF
pipeline.process_pdf_to_dxf(
    pdf_path="architectural_plan.pdf",
    output_dxf="output_plan.dxf",
    confidence_threshold=0.6
)

print("✅ تبدیل با مدل سفارشی انجام شد!")
```

---

## نکات و توصیه‌ها

### 1️⃣ آماده‌سازی داده‌ها

**✅ کیفیت داده:**

- از فایل‌های DXF تمیز و استاندارد استفاده کنید
- لایه‌ها (Layers) باید نام‌گذاری صحیح داشته باشند:
  - `WALLS`, `WALL`, `دیوار` → wall
  - `DOORS`, `DOOR`, `درب` → door
  - `WINDOWS`, `WINDOW`, `پنجره` → window
  - و غیره...
- حداقل 100-200 فایل DXF متنوع برای آموزش مناسب

**✅ تنوع داده:**

- ساختمان‌های مختلف (مسکونی، تجاری، صنعتی)
- سبک‌های معماری متفاوت
- مقیاس‌های مختلف
- نقشه‌های ساده و پیچیده

### 2️⃣ تنظیمات آموزش

**💡 Batch Size:**

- GPU 6GB: `batch_size=2`
- GPU 8GB: `batch_size=4`
- GPU 12GB+: `batch_size=8`

**💡 Learning Rate:**

- شروع با `lr=0.001` (پیش‌فرض)
- اگر loss خیلی سریع کاهش یافت: `lr=0.005`
- اگر loss بی‌ثبات است: `lr=0.0001`

**💡 Epochs:**

- حداقل 50 epochs برای نتایج خوب
- 100 epochs برای نتایج عالی
- با validation loss بهترین epoch را پیدا کنید

**💡 Pretrained Weights:**

- همیشه از `--pretrained` استفاده کنید
- مدل‌های COCO pretrained بسیار کمک می‌کنند
- زمان آموزش را کاهش می‌دهد

### 3️⃣ نظارت بر آموزش

```python
# در حین آموزش:
# Epoch 1/50: loss=1.234 | val_loss=1.456
# Epoch 2/50: loss=0.987 | val_loss=1.123
# ...

# نشانه‌های خوب:
✅ loss کاهش می‌یابد
✅ val_loss کاهش می‌یابد
✅ تفاوت loss و val_loss کم است

# نشانه‌های مشکل:
❌ loss کاهش نمی‌یابد → learning rate بالاست
❌ val_loss افزایش می‌یابد → overfitting
❌ تفاوت زیاد loss و val_loss → داده کم است
```

### 4️⃣ ارزیابی مدل

```python
from cad3d.training_pipeline import CADDetectionTrainer
import torch

# بارگذاری مدل
trainer = CADDetectionTrainer(
    data_dir="./training_dataset",
    output_dir="./models",
    device=torch.device("cuda")
)

# ارزیابی
val_loss = trainer.validate()
print(f"Validation Loss: {val_loss:.4f}")

# مقایسه با مدل‌های قبلی
# هر چه val_loss کمتر، مدل بهتر
```

### 5️⃣ Fine-tuning برای کاربردهای خاص

```bash
# مثال 1: نقشه‌های مسکونی
python -m cad3d.cli train \
  --dataset-dir ./residential_plans \
  --output-dir ./models/residential \
  --pretrained \
  --epochs 50

# مثال 2: نقشه‌های تاسیسات
python -m cad3d.cli train \
  --dataset-dir ./mep_plans \
  --output-dir ./models/mep \
  --pretrained \
  --epochs 50

# مثال 3: سبک معماری سنتی ایرانی
python -m cad3d.cli train \
  --dataset-dir ./iranian_architecture \
  --output-dir ./models/iranian \
  --pretrained \
  --epochs 100 \
  --lr 0.005
```

### 6️⃣ کلاس‌های تشخیص (15 دسته)

```python
# کلاس‌های CAD پشتیبانی‌شده:
categories = [
    "wall",         # دیوار
    "door",         # درب
    "window",       # پنجره
    "column",       # ستون
    "beam",         # تیر
    "slab",         # سقف
    "hvac",         # تهویه مطبوع
    "plumbing",     # لوله‌کشی
    "electrical",   # برق
    "furniture",    # مبلمان
    "equipment",    # تجهیزات
    "dimension",    # اندازه‌گذاری
    "text",         # متن
    "symbol",       # سمبل
    "grid_line"     # خطوط شبکه
]

# برای افزودن کلاس جدید:
# 1. در training_dataset_builder.py: categories و category_to_id را ویرایش کنید
# 2. در _classify_entity(): منطق classification را اضافه کنید
# 3. Dataset جدید بسازید
# 4. مدل را دوباره آموزش دهید
```

---

## 🎯 مثال‌های کاربردی

### مثال 1: آموزش از صفر

```bash
# 1. جمع‌آوری 200 فایل DXF
mkdir my_cad_library
# ... کپی کردن فایل‌های DXF

# 2. ساخت Dataset
python -m cad3d.cli build-dataset \
  --input-dir ./my_cad_library \
  --output-dir ./dataset \
  --format coco \
  --recurse \
  --visualize

# 3. بررسی تصاویر annotation در dataset/visualizations/

# 4. آموزش مدل
python -m cad3d.cli train \
  --dataset-dir ./dataset \
  --output-dir ./models \
  --epochs 50 \
  --batch-size 4 \
  --device cuda \
  --pretrained

# 5. استفاده از مدل آموزش‌دیده
python -c "
from cad3d.neural_cad_detector import NeuralCADDetector
import torch

detector = NeuralCADDetector(device='cuda')
checkpoint = torch.load('./models/best_model.pth')
detector.detection_model.load_state_dict(checkpoint['model_state_dict'])

elements = detector.detect_from_image('test_image.jpg')
print(f'✅ تشخیص {len(elements)} عنصر')
"
```

### مثال 2: بهبود مدل موجود

```bash
# ادامه آموزش با داده‌های بیشتر
python -m cad3d.cli build-dataset \
  --input-dir ./new_dxf_files \
  --output-dir ./extended_dataset \
  --format coco

# Fine-tune مدل قبلی
python -m cad3d.cli train \
  --dataset-dir ./extended_dataset \
  --output-dir ./models_v2 \
  --resume ./models/best_model.pth \
  --epochs 30 \
  --lr 0.0001 \
  --device cuda
```

### مثال 3: Transfer Learning برای حوزه خاص

```python
"""
آموزش مدل برای نقشه‌های بیمارستانی با Transfer Learning
"""
from cad3d.training_dataset_builder import CADDatasetBuilder
from cad3d.training_pipeline import CADDetectionTrainer
import torch

# 1. ساخت Dataset از نقشه‌های بیمارستانی
builder = CADDatasetBuilder(output_dir="./hospital_dataset")

hospital_plans = [
    "emergency_room.dxf",
    "surgery_room.dxf",
    "patient_room.dxf",
    "icu_ward.dxf",
    # ... 100+ files
]

for plan in hospital_plans:
    builder.add_dxf_to_dataset(plan, image_size=(1024, 1024))

builder.export_coco_format()

# 2. بارگذاری مدل عمومی
device = torch.device("cuda")
trainer = CADDetectionTrainer(
    data_dir="./hospital_dataset",
    output_dir="./models/hospital_specialist",
    batch_size=4,
    device=device,
    pretrained=True
)

# بارگذاری weights از مدل عمومی
general_checkpoint = torch.load("./models/general/best_model.pth")
trainer.model.load_state_dict(general_checkpoint['model_state_dict'])

# 3. Fine-tuning با learning rate کم
trainer.setup_optimizer(
    optimizer_type="adam",
    learning_rate=0.0001  # کم برای حفظ دانش قبلی
)

# 4. آموزش
trainer.train(num_epochs=30)

print("✅ مدل تخصصی بیمارستانی آماده است!")
```

---

## 🐛 عیب‌یابی

### مشکل: Out of Memory (OOM)

```bash
# راه حل 1: کاهش batch size
--batch-size 2

# راه حل 2: کاهش اندازه تصویر
--image-size 512 512

# راه حل 3: کاهش workers
--workers 2
```

### مشکل: Loss کاهش نمی‌یابد

```bash
# راه حل 1: کاهش learning rate
--lr 0.0001

# راه حل 2: استفاده از pretrained weights
--pretrained

# راه حل 3: افزایش epochs
--epochs 100
```

### مشکل: Overfitting

```python
# نشانه: val_loss >> train_loss

# راه حل 1: افزایش داده
# جمع‌آوری DXF بیشتر

# راه حل 2: Data Augmentation
from torchvision import transforms

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.1)
])

# راه حل 3: Early Stopping
# توقف آموزش وقتی val_loss افزایش می‌یابد
```

---

## 📚 منابع بیشتر

- [NEURAL_README.md](NEURAL_README.md) - معماری Neural Network
- [.github/copilot-instructions.md](.github/copilot-instructions.md) - راهنمای توسعه
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [COCO Dataset Format](https://cocodataset.org/#format-data)
- [Faster R-CNN Paper](https://arxiv.org/abs/1506.01497)

---

**✨ آموزش موفق!**
