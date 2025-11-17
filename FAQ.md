# ❓ سوالات متداول (FAQ)

پاسخ به سوالات رایج کاربران درباره سیستم تبدیل CAD.

---

## 📦 نصب و راه‌اندازی

### آیا باید Python بلد باشم؟

**خیر.** برای استفاده از دستورات CLI نیازی به دانش برنامه‌نویسی نیست. فقط دستورات را در Terminal کپی/پیست کنید.

برای استفاده پیشرفته (Python API) نیاز به دانش اولیه Python دارید.

### کدام نسخه Python نیاز است؟

**Python 3.10 یا بالاتر** توصیه می‌شود. Python 3.8 و 3.9 هم کار می‌کنند ولی برخی ویژگی‌ها ممکن است محدودیت داشته باشند.

### آیا GPU لازم است؟

**خیر، ولی توصیه می‌شود:**

- بدون GPU: تبدیل DXF 2D→3D کار می‌کند
- با GPU: قابلیت‌های Neural Network 5-10x سریع‌تر

### چطور بفهمم GPU کار می‌کند؟

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

اگر `True` نشان داد، GPU در دسترس است.

---

## 🔄 تبدیل DXF

### چرا فایل خروجی خالی است؟

**علت 1: فایل فقط LINE دارد**
این ابزار فقط LWPOLYLINE های بسته را extrude می‌کند. LINE ها پشتیبانی نمی‌شوند.

**راه حل:** در AutoCAD/DraftSight:

1. LINE ها را انتخاب کنید
2. دستور `PEDIT` → Join → همه LINE ها
3. تبدیل به POLYLINE
4. بستن POLYLINE (دستور `CLOSE`)

**علت 2: POLYLINE ها باز هستند**

```bash
# بررسی:
python -c "
import ezdxf
doc = ezdxf.readfile('plan.dxf')
polys = list(doc.modelspace().query('LWPOLYLINE'))
closed = [p for p in polys if p.is_closed]
print(f'{len(closed)}/{len(polys)} closed')
"
```

### چطور ارتفاع مناسب را انتخاب کنم؟

بستگی به **واحد drawing** دارد:

| واحد | ارتفاع معمول دیوار |
|------|-------------------|
| میلی‌متر | 3000 |
| سانتی‌متر | 300 |
| متر | 3 |
| اینچ | 118 |
| فوت | 10 |

**بررسی واحد:**

```bash
python -c "
import ezdxf
doc = ezdxf.readfile('plan.dxf')
units = doc.header['$INSUNITS']
print('Units:', units)
"
```

### کمان‌ها ناهموار هستند. چکار کنم؟

افزایش تعداد segments:

```bash
# پیش‌فرض (16 segments)
python -m cad3d.cli dxf-extrude ... --arc-segments 16

# کیفیت بهتر (32 segments)
python -m cad3d.cli dxf-extrude ... --arc-segments 32

# یا محدودیت طول segment
python -m cad3d.cli dxf-extrude ... --arc-max-seglen 25
```

**نکته:** عدد بالاتر = کیفیت بهتر + حجم فایل بیشتر

### حجم فایل خروجی خیلی زیاد است

```bash
# استفاده از optimize
python -m cad3d.cli dxf-extrude ... --optimize-vertices
```

این گزینه vertex های تکراری را حذف می‌کند (تا 50% کاهش حجم).

---

## 🤖 Neural Network (PDF/Image → DXF)

### چه نوع فایل‌هایی پشتیبانی می‌شوند؟

**تصاویر:**

- JPG, PNG, BMP, TIFF
- حداقل 1024x1024 پیکسل توصیه می‌شود

**PDF:**

- PDF های تصویری (اسکن شده)
- PDF های vector (با text/line)
- Multi-page (هر صفحه جداگانه پردازش می‌شود)

### دقت تشخیص پایین است. چکار کنم؟

**راه حل 1: افزایش کیفیت ورودی**

```bash
# DPI بالاتر برای PDF
python -m cad3d.cli pdf-to-dxf ... --dpi 600

# برای تصاویر: از فایل با وضوح بالا استفاده کنید
```

**راه حل 2: تنظیم confidence threshold**

```bash
# کاهش threshold برای تشخیص بیشتر
python -m cad3d.cli pdf-to-dxf ... --confidence 0.4

# افزایش threshold برای دقت بیشتر
python -m cad3d.cli pdf-to-dxf ... --confidence 0.7
```

**راه حل 3: آموزش مدل سفارشی**

بهترین راه! مدل را روی نقشه‌های خودتان آموزش دهید:

```bash
python -m cad3d.cli build-dataset ...
python -m cad3d.cli train ...
```

### چه المان‌هایی تشخیص داده می‌شوند؟

**15 کلاس CAD:**

1. wall (دیوار)
2. door (درب)
3. window (پنجره)
4. column (ستون)
5. beam (تیر)
6. slab (سقف)
7. hvac (تهویه مطبوع)
8. plumbing (لوله‌کشی)
9. electrical (برق)
10. furniture (مبلمان)
11. equipment (تجهیزات)
12. dimension (اندازه‌گذاری)
13. text (متن)
14. symbol (سمبل)
15. grid_line (خطوط شبکه)

### OCR فارسی کار نمی‌کند

سه موتور OCR پشتیبانی می‌شود:

```bash
# روش 1: EasyOCR (بهترین برای فارسی)
pip install easyocr
python -m cad3d.cli image-to-dxf ... --detect-text

# روش 2: PaddleOCR
pip install paddleocr
```

**نکته:** اولین بار اجرا، مدل‌ها دانلود می‌شوند (~100MB).

### چقدر طول می‌کشد؟

**بدون GPU:**

- PDF یک صفحه: 5-10 ثانیه
- تصویر 1024x1024: 2-3 ثانیه

**با GPU (RTX 3060):**

- PDF یک صفحه: 0.5-1 ثانیه
- تصویر 1024x1024: 0.3 ثانیه

---

## 🎓 آموزش مدل سفارشی

### چند فایل DXF لازم است؟

**حداقل:**

- 50-100 فایل: نتایج متوسط
- 200-500 فایل: نتایج خوب
- 500+ فایل: نتایج عالی

**تنوع مهم‌تر از تعداد است:**

- انواع مختلف ساختمان (مسکونی، تجاری، صنعتی)
- سبک‌های معماری متفاوت
- مقیاس‌های مختلف

### لایه‌های DXF باید چطور نام‌گذاری شوند؟

**نام‌های پشتیبانی شده:**

| کلاس | نام‌های لایه |
|------|--------------|
| wall | WALLS, WALL, دیوار, دیوارها |
| door | DOORS, DOOR, درب, درها |
| window | WINDOWS, WINDOW, پنجره, پنجره‌ها |
| column | COLUMNS, COLUMN, ستون, ستون‌ها |
| ... | ... |

سیستم به‌طور خودکار شناسایی می‌کند (case-insensitive).

### چقدر طول می‌کشد تا مدل آموزش ببیند؟

**بستگی به:**

- تعداد فایل‌ها
- اندازه تصاویر
- GPU

**زمان تخمینی (با GPU):**

- 100 تصویر، 50 epochs: 30-60 دقیقه
- 500 تصویر، 50 epochs: 2-4 ساعت
- 1000 تصویر، 100 epochs: 8-12 ساعت

**بدون GPU:** 5-10x طولانی‌تر (توصیه نمی‌شود)

### چطور بفهمم آموزش خوب پیش می‌رود؟

**نشانه‌های خوب:**

```
Epoch 1/50: loss=1.234 | val_loss=1.456
Epoch 10/50: loss=0.567 | val_loss=0.623
Epoch 20/50: loss=0.345 | val_loss=0.389
Epoch 50/50: loss=0.234 | val_loss=0.289
```

✅ Loss کاهش می‌یابد
✅ val_loss کاهش می‌یابد
✅ تفاوت loss و val_loss کم است (<0.1)

**نشانه‌های بد:**

```
Epoch 1/50: loss=1.234 | val_loss=1.456
Epoch 10/50: loss=1.123 | val_loss=1.478
Epoch 20/50: loss=1.089 | val_loss=1.503
```

❌ Loss کاهش نمی‌یابد → learning rate بالاست
❌ val_loss افزایش می‌یابد → overfitting
❌ تفاوت زیاد loss و val_loss → داده کم است

---

## ⚡ بهینه‌سازی

### کدام فرمت را برای استقرار انتخاب کنم؟

| فرمت | استفاده | مزایا |
|------|----------|-------|
| PyTorch | توسعه، آموزش | انعطاف کامل |
| ONNX | تولید (CPU/GPU) | سازگاری بالا، 1.5x سریع‌تر |
| Quantized | Mobile/Edge | 4x کوچک‌تر، 2x سریع‌تر |
| TensorRT | GPU (NVIDIA) | 4-8x سریع‌تر |

**توصیه:**

- سرور (CPU): ONNX
- سرور (GPU NVIDIA): TensorRT
- موبایل/Embedded: Quantized
- توسعه: PyTorch

### Quantization دقت را کاهش می‌دهد؟

**معمولاً خیلی کم:**

- مدل‌های بزرگ: <1% کاهش دقت
- مدل‌های کوچک: 1-3% کاهش دقت

همیشه benchmark کنید:

```bash
python -m cad3d.cli benchmark ... --model quantized_model.pth
```

### TensorRT نصب نمی‌شود

**محدودیت‌ها:**

- فقط Linux و Windows
- فقط GPU های NVIDIA
- نیاز به CUDA Toolkit

**نصب:**

```bash
# 1. نصب CUDA Toolkit 11.8
# دانلود از: https://developer.nvidia.com/cuda-downloads

# 2. نصب TensorRT
pip install tensorrt

# 3. تست
python -c "import tensorrt; print(tensorrt.__version__)"
```

اگر کار نکرد، از ONNX استفاده کنید (تفاوت سرعت کمتر).

---

## 🐛 خطاها و مشکلات

### `ModuleNotFoundError: No module named 'torch'`

```bash
pip install torch torchvision torchaudio
```

یا برای GPU:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### `CUDA out of memory`

```bash
# کاهش batch size
python -m cad3d.cli train ... --batch-size 2

# کاهش resolution
python -m cad3d.cli build-dataset ... --image-size 512 512
```

### `RuntimeError: CUDA error: no kernel image is available`

نسخه PyTorch با نسخه CUDA سازگار نیست.

**راه حل:**

```bash
# بررسی نسخه CUDA
nvidia-smi

# نصب PyTorch سازگار
# CUDA 11.8:
pip install torch --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1:
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### `ezdxf.DXFStructureError: Invalid DXF file`

فایل DXF خراب است.

**راه حل:**

1. باز کردن در AutoCAD/DraftSight
2. `AUDIT` command
3. `PURGE` command
4. Save As → DXF R2018

### فایل DWG پشتیبانی نمی‌شود

نیاز به ODA File Converter:

**نصب:**

1. دانلود: <https://www.opendesign.com/guestfiles/oda_file_converter>
2. نصب در: `C:\Program Files\ODA\`
3. تنظیم `.env`:

```
ODA_CONVERTER_PATH=C:\Program Files\ODA\ODAFileConverter.exe
```

**استفاده:**

```bash
# تبدیل خودکار DWG → DXF → 3D → DWG
python -m cad3d.cli auto-extrude \
  --input plan.dwg \
  --output plan_3d.dwg \
  --height 3000
```

---

## 📊 عملکرد

### مدل چقدر دقیق است؟

**مدل از پیش آموزش‌دیده (baseline):**

- mAP: 72-75%
- Precision: 78-82%
- Recall: 75-80%

**پس از Fine-tuning روی 500 نقشه:**

- mAP: 85-90%
- Precision: 88-92%
- Recall: 85-89%

**بهترین نتایج:**

- wall, column: >90% دقت
- door, window: 85-90% دقت
- text, dimension: 70-80% دقت (بستگی به کیفیت تصویر)

### چطور دقت را بهبود دهم؟

1. **افزایش DPI:** 300 → 600
2. **Fine-tuning:** آموزش روی داده‌های خودتان
3. **Confidence threshold:** تنظیم برای trade-off precision/recall
4. **کیفیت ورودی:** تصاویر واضح، کنتراست بالا
5. **Data augmentation:** تنوع بیشتر در Dataset آموزشی

---

## 🔗 یکپارچه‌سازی

### چطور با AutoCAD یکپارچه کنم؟

**روش 1: Script (Lisp)**

```lisp
(defun c:IMPORT3D ()
  (command "_.DXFIN" "C:/path/to/plan_3d.dxf")
  (command "_.ZOOM" "_E")
)
```

**روش 2: Python (COM)**

```python
import win32com.client
acad = win32com.client.Dispatch("AutoCAD.Application")
acad.ActiveDocument.Open("C:/path/to/plan_3d.dxf")
```

### چطور با Revit یکپارچه کنم؟

**پلاگین C# (Revit API):**

```csharp
using Autodesk.Revit.DB;

[Transaction(TransactionMode.Manual)]
public class ImportDXFCommand : IExternalCommand
{
    public Result Execute(/* ... */)
    {
        Document doc = commandData.Application.ActiveUIDocument.Document;
        
        DWGImportOptions options = new DWGImportOptions();
        options.ColorMode = ImportColorMode.Preserved;
        
        using (Transaction trans = new Transaction(doc, "Import DXF"))
        {
            trans.Start();
            doc.Import("C:/path/to/plan.dxf", options, doc.ActiveView);
            trans.Commit();
        }
        
        return Result.Succeeded;
    }
}
```

### چطور در Web App استفاده کنم؟

**FastAPI Backend:**

```python
from fastapi import FastAPI, UploadFile
from cad3d.pdf_processor import CADPipeline

app = FastAPI()

@app.post("/convert")
async def convert_pdf(file: UploadFile):
    # Save uploaded file
    with open("temp.pdf", "wb") as f:
        f.write(await file.read())
    
    # Convert
    pipeline = CADPipeline()
    pipeline.process_pdf_to_dxf("temp.pdf", "output.dxf")
    
    # Return DXF file
    return FileResponse("output.dxf")
```

---

## 💡 نکات و ترفندها

### کاهش زمان پردازش Batch

```bash
# استفاده از همه CPU cores
python -m cad3d.cli batch-extrude ... --jobs -1

# 4 cores
python -m cad3d.cli batch-extrude ... --jobs 4
```

### ذخیره Log برای debugging

```bash
# ذخیره خروجی در فایل
python -m cad3d.cli pdf-to-dxf ... 2>&1 | tee log.txt
```

### بررسی سریع کیفیت

```bash
# Visualize annotations قبل از آموزش
python -m cad3d.cli build-dataset ... --visualize

# فولدر visualizations/ را بررسی کنید
```

### استفاده از Configuration File

```bash
# ساخت config.json
{
  "height": 3000,
  "arc_max_seglen": 50,
  "optimize_vertices": true,
  "colorize": true
}

# استفاده (custom script)
import json
with open('config.json') as f:
    config = json.load(f)
    
extrude_dxf_closed_polylines('plan.dxf', 'out.dxf', **config)
```

---

**سوال شما اینجا نیست؟**

- 📧 ایمیل: <support@example.com>
- 💬 Telegram: @cad3d_support
- 🐛 GitHub Issues: github.com/your-repo/issues
