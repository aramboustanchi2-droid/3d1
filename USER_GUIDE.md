# 📘 راهنمای کامل کاربر - سیستم تشخیص و تبدیل CAD

این راهنما برای کاربران نهایی سیستم نوشته شده و تمام قابلیت‌ها را گام به گام توضیح می‌دهد.

## 📋 فهرست مطالب

1. [نصب و راه‌اندازی](#نصب-و-راهاندازی)
2. [تبدیل DXF دوبعدی به سه‌بعدی](#تبدیل-dxf-دوبعدی-به-سهبعدی)
3. [تبدیل PDF به DXF با هوش مصنوعی](#تبدیل-pdf-به-dxf-با-هوش-مصنوعی)
4. [تبدیل عکس به DXF](#تبدیل-عکس-به-dxf)
5. [آموزش مدل سفارشی](#آموزش-مدل-سفارشی)
6. [بهینه‌سازی مدل](#بهینهسازی-مدل)
7. [استفاده پیشرفته](#استفاده-پیشرفته)
8. [عیب‌یابی](#عیبیابی)

---

## نصب و راه‌اندازی

### نصب پایه (بدون هوش مصنوعی)

برای تبدیل DXF دوبعدی به سه‌بعدی:

```bash
# نصب Python 3.10 یا بالاتر
# دانلود از: https://www.python.org/downloads/

# کلون کردن پروژه
git clone https://github.com/your-repo/cad3d.git
cd cad3d

# ساخت محیط مجازی
python -m venv .venv

# فعال‌سازی (Windows)
.venv\Scripts\activate

# فعال‌سازی (Linux/Mac)
source .venv/bin/activate

# نصب dependencies پایه
pip install -r requirements.txt
```

### نصب کامل (با هوش مصنوعی)

برای استفاده از قابلیت‌های Neural Network:

```bash
# نصب dependencies Neural
pip install -r requirements-neural.txt

# نصب PyTorch
# CPU version:
pip install torch torchvision torchaudio

# GPU version (CUDA 11.8):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### تست نصب

```bash
# تست نصب پایه
python -m cad3d.cli --help

# تست نصب Neural
python -c "import torch; print(f'PyTorch {torch.__version__} installed')"
python -c "import cv2; print(f'OpenCV {cv2.__version__} installed')"
```

---

## تبدیل DXF دوبعدی به سه‌بعدی

### استفاده ساده

```bash
# تبدیل یک فایل
python -m cad3d.cli dxf-extrude \
  --input floor_plan.dxf \
  --output floor_plan_3d.dxf \
  --height 3000
```

**نکات:**

- `--height` به واحد drawing است (معمولاً میلی‌متر)
- فایل خروجی شامل Mesh های سه‌بعدی است
- فقط LWPOLYLINE های بسته extrude می‌شوند

### انتخاب لایه‌های خاص

```bash
# فقط دیوارها و ستون‌ها
python -m cad3d.cli dxf-extrude \
  --input plan.dxf \
  --output plan_3d.dxf \
  --height 3000 \
  --layers WALLS COLUMNS DOORS
```

### بهبود کیفیت (کمان‌ها)

```bash
# کنترل تعداد segments کمان‌ها
python -m cad3d.cli dxf-extrude \
  --input plan.dxf \
  --output plan_3d.dxf \
  --height 3000 \
  --arc-segments 32 \
  --arc-max-seglen 50
```

**نکات:**

- `--arc-segments`: حداکثر تعداد segment ها برای هر کمان
- `--arc-max-seglen`: حداکثر طول segment (واحد drawing)
- عدد بالاتر = دقت بیشتر + حجم فایل بیشتر

### بهینه‌سازی Vertex ها

```bash
# کاهش حجم فایل با حذف vertex های تکراری
python -m cad3d.cli dxf-extrude \
  --input large_plan.dxf \
  --output large_plan_3d.dxf \
  --height 3000 \
  --optimize-vertices
```

### تشخیص اشکال سخت

```bash
# شناسایی و گزارش polyline های مشکل‌دار
python -m cad3d.cli dxf-extrude \
  --input plan.dxf \
  --output plan_3d.dxf \
  --height 3000 \
  --detect-hard-shapes \
  --hard-report-csv hard_shapes.csv
```

**اشکال تشخیص داده می‌شود:**

- Vertex های تکراری
- Edge های با طول صفر
- Polygon های self-intersecting
- مساحت صفر یا خیلی کوچک

### رنگ‌بندی و تفکیک

```bash
# حفظ رنگ entity ها در mesh ها
python -m cad3d.cli dxf-extrude \
  --input plan.dxf \
  --output plan_3d.dxf \
  --height 3000 \
  --colorize \
  --split-by-color \
  --color-report-csv colors.csv
```

### پردازش دسته‌ای

```bash
# تبدیل تمام فایل‌های یک پوشه
python -m cad3d.cli batch-extrude \
  --input-dir ./input_plans \
  --output-dir ./output_3d \
  --out-format DXF \
  --height 3000 \
  --recurse \
  --jobs 4 \
  --report-csv batch_report.csv
```

**نکات:**

- `--recurse`: جستجوی زیرپوشه‌ها
- `--jobs 4`: استفاده از 4 هسته CPU
- `--out-format`: DXF یا DWG

---

## تبدیل PDF به DXF با هوش مصنوعی

### تبدیل ساده PDF

```bash
python -m cad3d.cli pdf-to-dxf \
  --input architectural_plan.pdf \
  --output output_plan.dxf \
  --dpi 300
```

**نکات:**

- هوش مصنوعی عناصر را تشخیص می‌دهد: دیوار، درب، پنجره، ...
- 15 کلاس مختلف CAD پشتیبانی می‌شود
- مناسب برای نقشه‌های اسکن شده یا PDF های تصویری

### تنظیمات کیفیت

```bash
# DPI بالا برای جزئیات بیشتر
python -m cad3d.cli pdf-to-dxf \
  --input plan.pdf \
  --output plan.dxf \
  --dpi 600 \
  --confidence 0.7 \
  --scale 100
```

**پارامترها:**

- `--dpi`: وضوح تصویر (150-600، پیش‌فرض 300)
- `--confidence`: حداقل اطمینان تشخیص (0-1، پیش‌فرض 0.5)
- `--scale`: مقیاس DXF خروجی

### استفاده از GPU

```bash
# اگر GPU دارید، 5-10x سریع‌تر
python -m cad3d.cli pdf-to-dxf \
  --input plan.pdf \
  --output plan.dxf \
  --device cuda
```

### تبدیل PDF به 3D

```bash
# تبدیل مستقیم PDF به DXF سه‌بعدی
python -m cad3d.cli pdf-to-3d \
  --input plan.pdf \
  --output plan_3d.dxf \
  --dpi 300 \
  --intelligent-height
```

**ویژگی `--intelligent-height`:**

- از Machine Learning برای پیش‌بینی ارتفاع استفاده می‌کند
- دیوارها، ستون‌ها، درها ارتفاع‌های مختلف می‌گیرند
- دقت بالاتر از ارتفاع ثابت

---

## تبدیل عکس به DXF

### تبدیل عکس نقشه

```bash
python -m cad3d.cli image-to-dxf \
  --input floor_plan_photo.jpg \
  --output plan.dxf \
  --confidence 0.6
```

### فعال/غیرفعال کردن تشخیص‌ها

```bash
# فقط خطوط و دایره‌ها، بدون OCR
python -m cad3d.cli image-to-dxf \
  --input sketch.jpg \
  --output sketch.dxf \
  --detect-lines \
  --detect-circles \
  --no-detect-text
```

### تشخیص متن فارسی

```bash
# OCR دوزبانه فارسی-انگلیسی
python -m cad3d.cli image-to-dxf \
  --input persian_plan.jpg \
  --output plan.dxf \
  --detect-text
```

---

## آموزش مدل سفارشی

برای بهبود دقت تشخیص روی نقشه‌های خودتان:

### مرحله 1: ساخت Dataset

```bash
# تبدیل فایل‌های DXF به Dataset آموزشی
python -m cad3d.cli build-dataset \
  --input-dir ./my_dxf_library \
  --output-dir ./training_dataset \
  --format coco \
  --recurse \
  --visualize
```

**خروجی:**

- `training_dataset/images/`: تصاویر PNG
- `training_dataset/annotations.json`: Annotation ها (COCO format)
- `training_dataset/visualizations/`: بررسی بصری

**نکات:**

- حداقل 100-200 فایل DXF نیاز است
- تنوع داشته باشد (مسکونی، تجاری، صنعتی)
- لایه‌ها باید نام‌گذاری صحیح داشته باشند

### مرحله 2: آموزش مدل

```bash
# آموزش با تنظیمات پیش‌فرض
python -m cad3d.cli train \
  --dataset-dir ./training_dataset \
  --output-dir ./models \
  --epochs 50 \
  --batch-size 4 \
  --device cuda \
  --pretrained
```

**پارامترهای مهم:**

- `--epochs`: تعداد دوره آموزش (50-100)
- `--batch-size`: بسته به حافظه GPU (2-8)
- `--pretrained`: استفاده از وزن‌های از پیش آموزش‌دیده (توصیه می‌شود)

**نظارت بر آموزش:**

```
Epoch 1/50: loss=1.234 | val_loss=1.456
Epoch 2/50: loss=0.987 | val_loss=1.123
...
Epoch 50/50: loss=0.234 | val_loss=0.289

✅ Training complete!
   Best model: ./models/best_model.pth
```

### مرحله 3: استفاده از مدل

```python
from cad3d.neural_cad_detector import NeuralCADDetector
import torch

# بارگذاری مدل سفارشی
detector = NeuralCADDetector(device="cuda")
checkpoint = torch.load("./models/best_model.pth")
detector.detection_model.load_state_dict(checkpoint['model_state_dict'])

# استفاده برای تشخیص
elements = detector.detect_from_image("test_plan.jpg")
print(f"تشخیص {len(elements)} عنصر")
```

---

## بهینه‌سازی مدل

برای استقرار در محیط تولید:

### ONNX (سازگار با همه سیستم‌ها)

```bash
python -m cad3d.cli optimize-model \
  --model ./models/best_model.pth \
  --output-dir ./optimized \
  --formats onnx \
  --benchmark
```

**مزایا:**

- 1.2-1.5x سریع‌تر
- اجرا روی CPU و GPU
- قابل استفاده در C++, JavaScript, ...

### Quantization (حجم کمتر)

```bash
python -m cad3d.cli optimize-model \
  --model ./models/best_model.pth \
  --output-dir ./optimized \
  --formats quantized \
  --benchmark
```

**مزایا:**

- 4x کوچک‌تر
- 2-3x سریع‌تر
- مناسب برای دستگاه‌های mobile/edge

### TensorRT (GPU های NVIDIA)

```bash
python -m cad3d.cli optimize-model \
  --model ./models/best_model.pth \
  --output-dir ./optimized \
  --formats tensorrt \
  --benchmark
```

**مزایا:**

- 4-8x سریع‌تر روی GPU های NVIDIA
- مناسب برای پردازش realtime

### مقایسه فرمت‌ها

```bash
# بهینه‌سازی و benchmark همه فرمت‌ها
python -m cad3d.cli optimize-model \
  --model ./models/best_model.pth \
  --output-dir ./optimized \
  --formats onnx tensorrt quantized \
  --benchmark
```

**خروجی:**

```
Format          Size (MB)   Time (ms)   Speedup
------------------------------------------------------
PyTorch         150.50      35.20       1.00x
ONNX            148.20      25.30       1.39x
Quantized       37.80       18.50       1.90x
TensorRT        145.60      8.70        4.05x
```

---

## استفاده پیشرفته

### Python API

```python
from cad3d.dxf_extrude import extrude_dxf_closed_polylines

# تبدیل با کنترل کامل
extrude_dxf_closed_polylines(
    input_path="plan.dxf",
    output_path="plan_3d.dxf",
    height=3000,
    layers=["WALLS", "COLUMNS"],
    arc_max_seglen=50,
    optimize=True,
    detect_hard_shapes=True,
    colorize=True
)
```

### پردازش Batch سفارشی

```python
from pathlib import Path
from cad3d.dxf_extrude import extrude_dxf_closed_polylines

input_dir = Path("./input_plans")
output_dir = Path("./output_3d")
output_dir.mkdir(exist_ok=True)

for dxf_file in input_dir.glob("*.dxf"):
    output_file = output_dir / f"{dxf_file.stem}_3d.dxf"
    
    try:
        extrude_dxf_closed_polylines(
            str(dxf_file),
            str(output_file),
            height=3000
        )
        print(f"✅ {dxf_file.name}")
    except Exception as e:
        print(f"❌ {dxf_file.name}: {e}")
```

### یکپارچه‌سازی با AutoCAD

```python
# Script برای اجرا در AutoCAD
import win32com.client

acad = win32com.client.Dispatch("AutoCAD.Application")
doc = acad.ActiveDocument

# باز کردن فایل 3D
doc.Open("c:/path/to/plan_3d.dxf")

# نمایش isometric
acad.ActiveDocument.SetVariable("VIEWDIR", [1, 1, 1])
acad.ZoomExtents()
```

### یکپارچه‌سازی با Revit

```python
# پلاگین Revit (C#)
using Autodesk.Revit.DB;

// Import DXF
Document doc = commandData.Application.ActiveUIDocument.Document;
DWGImportOptions options = new DWGImportOptions();
options.ColorMode = ImportColorMode.Preserved;

doc.Import("C:/path/to/plan.dxf", options, doc.ActiveView);
```

---

## عیب‌یابی

### مشکل: فایل DXF خالی است

**علل:**

- فایل ورودی فقط LINE دارد (نه LWPOLYLINE)
- LWPOLYLINE ها باز هستند
- لایه‌های انتخاب شده اشتباه

**راه حل:**

```bash
# بررسی محتوای فایل
python -c "
import ezdxf
doc = ezdxf.readfile('plan.dxf')
msp = doc.modelspace()
polys = list(msp.query('LWPOLYLINE'))
print(f'{len(polys)} LWPOLYLINE found')
for p in polys[:5]:
    print(f'  Layer: {p.dxf.layer}, Closed: {p.is_closed}')
"
```

### مشکل: Neural Network خیلی کند است

**راه حل 1: استفاده از GPU**

```bash
# بررسی CUDA
python -c "import torch; print(torch.cuda.is_available())"

# استفاده از GPU
python -m cad3d.cli pdf-to-dxf ... --device cuda
```

**راه حل 2: کاهش DPI**

```bash
# DPI کمتر = سریع‌تر (ولی دقت کمتر)
python -m cad3d.cli pdf-to-dxf ... --dpi 150
```

**راه حل 3: بهینه‌سازی مدل**

```bash
# استفاده از ONNX
python -m cad3d.cli optimize-model ...
```

### مشکل: تشخیص دقت کمی دارد

**راه حل 1: تنظیم confidence threshold**

```bash
# افزایش threshold برای دقت بیشتر
python -m cad3d.cli pdf-to-dxf ... --confidence 0.7
```

**راه حل 2: آموزش مدل سفارشی**

```bash
# بهترین راه: آموزش روی داده‌های خودتان
python -m cad3d.cli train ...
```

**راه حل 3: بهبود کیفیت تصویر**

- از PDF های با کیفیت بالا استفاده کنید
- DPI را افزایش دهید (300-600)
- از تصاویر واضح و تمیز استفاده کنید

### مشکل: Out of Memory

**راه حل:**

```bash
# کاهش batch size
python -m cad3d.cli train ... --batch-size 2

# کاهش اندازه تصویر
python -m cad3d.cli build-dataset ... --image-size 512 512

# کاهش workers
python -m cad3d.cli train ... --workers 2
```

### مشکل: Loss کاهش نمی‌یابد

**راه حل:**

```bash
# کاهش learning rate
python -m cad3d.cli train ... --lr 0.0001

# استفاده از pretrained weights
python -m cad3d.cli train ... --pretrained

# افزایش epochs
python -m cad3d.cli train ... --epochs 100
```

---

## 📞 پشتیبانی و منابع

- **مستندات فنی**: [NEURAL_README.md](NEURAL_README.md)
- **راهنمای آموزش**: [TRAINING_GUIDE.md](TRAINING_GUIDE.md)
- **راهنمای استقرار**: [DEPLOYMENT.md](DEPLOYMENT.md)
- **سوالات متداول**: [FAQ.md](FAQ.md)
- **مثال‌های کد**: [examples/](examples/)

---

## 🎯 نکات عملی

### برای معماران

1. ابتدا با فایل‌های DXF ساده شروع کنید
2. از `--visualize` برای بررسی نتایج استفاده کنید
3. برای نقشه‌های پیچیده، `--optimize-vertices` را فعال کنید

### برای توسعه‌دهندگان

1. از Python API برای یکپارچه‌سازی استفاده کنید
2. مدل‌های سفارشی را آموزش دهید
3. با ONNX برای استقرار متقاطع

### برای تیم‌های BIM

1. پردازش batch برای پروژه‌های بزرگ
2. یکپارچه‌سازی با Revit/AutoCAD
3. استانداردسازی نام‌گذاری لایه‌ها

---

**موفق باشید! 🚀**
