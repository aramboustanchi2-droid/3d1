# راهنمای سریع استفاده

## نصب و راه‌اندازی

### ۱. ساخت محیط مجازی و نصب وابستگی‌ها

```powershell
# ساخت محیط مجازی
python -m venv .venv

# فعال‌سازی محیط مجازی
.\.venv\Scripts\Activate.ps1

# نصب کتابخانه‌ها
pip install ezdxf numpy onnxruntime opencv-python pytest
```

### ۲. تنظیم VS Code

در VS Code، Python interpreter را به `.venv` تنظیم کنید:

- `Ctrl+Shift+P` → `Python: Select Interpreter` → انتخاب `.venv`

## استفاده

### تبدیل DXF دوبعدی به سه‌بعدی

```powershell
# فعال‌سازی محیط مجازی
.\.venv\Scripts\Activate.ps1

# اکستروژن تمام polyline های بسته
python -m cad3d.cli dxf-extrude --input floor_plan_2d.dxf --output floor_plan_3d.dxf --height 3000

# اکستروژن فقط لایه‌های خاص
python -m cad3d.cli dxf-extrude --input plan.dxf --output result.dxf --height 2500 --layers WALLS COLUMNS
```

### تبدیل DXF به DWG (نیاز به ODA File Converter)

```powershell
python -m cad3d.cli dxf-to-dwg --input plan_3d.dxf --output plan_3d.dwg --version ACAD2018
```

### تبدیل عکس به مش سه‌بعدی

ابتدا مدل ONNX را دانلود و در پوشه `models/` قرار دهید:

```powershell
# تنظیم مسیر مدل
$env:MIDAS_ONNX_PATH = "models\midas_v2_small_256.onnx"

# تبدیل عکس به 3D
python -m cad3d.cli img-to-3d --input photo.jpg --output photo_3d.dxf --scale 1000
```

## اجرای تست‌ها

```powershell
.\.venv\Scripts\Activate.ps1
python -m pytest tests/ -v
```

## نتیجه

فایل‌های خروجی DXF حاوی MESH های سه‌بعدی هستند که در AutoCAD، BricsCAD و سایر نرم‌افزارهای CAD قابل مشاهده‌اند.

## نکات مهم

- **ورودی باید DXF باشد**: برای فایل‌های DWG، ابتدا در نرم‌افزار CAD به DXF تبدیل کنید
- **فقط polyline های بسته**: polyline های باز نادیده گرفته می‌شوند
- **قوس‌ها**: با 12 قطعه خط تقریب زده می‌شوند (برای دقت بیشتر کد را تغییر دهید)
- **واحدها**: ارتفاع اکستروژن باید با واحد نقشه همخوانی داشته باشد (معمولاً میلی‌متر)

## مبدل عمومی (Universal Convert)

این فرمان به‌صورت یکپارچه ورودی‌های «تصویر، PDF, DXF, DWG» را می‌پذیرد و خروجی «DXF یا DWG» تولید می‌کند.

```powershell
# فعال‌سازی محیط مجازی
.\.venv\Scripts\Activate.ps1

# DXF → DXF (کپی مستقیم)
python -m cad3d.cli universal-convert --input input.dxf --output outputs\out.dxf

# DXF → DWG (نیاز به ODA File Converter)
python -m cad3d.cli universal-convert --input input.dxf --output outputs\out.dwg --dwg-version ACAD2018

# DWG → DXF (نیاز به ODA File Converter)
python -m cad3d.cli universal-convert --input input.dwg --output outputs\out.dxf --dwg-version ACAD2018

# PDF → DXF (نیاز به PyMuPDF یا pdf2image + poppler و وابستگی‌های neural)
python -m cad3d.cli universal-convert --input input.pdf --output outputs\pdf_out.dxf --dpi 300 --confidence 0.5 --scale 1.0 --device auto

# تصویر → DXF (نیاز به وابستگی‌های neural)
python -m cad3d.cli universal-convert --input input.png --output outputs\img_out.dxf --confidence 0.5 --scale 1.0 --device auto

# اکستروژن خودکار 3D پس از بردارسازی/ورود (پرچم --to-3d)
python -m cad3d.cli universal-convert --input input.dxf --output outputs\out_3d.dxf --to-3d --height 3000
python -m cad3d.cli universal-convert --input input.pdf --output outputs\pdf_out_3d.dwg --to-3d --height 3000 --dpi 300 --device auto
```

نکات:

- مسیر ODA File Converter را طبق فایل `cad3d/config.py` و متغیر محیطی مربوطه پیکربندی کنید تا تبدیل DXF↔DWG فعال شود.
- مسیر‌ها و وابستگی‌های neural (طبق `requirements-neural.txt`) برای PDF/تصویر لازم هستند. برای PDF، نصب یکی از PyMuPDF یا pdf2image (+ poppler) کافی است.
- برای تولید DWG از ورودی‌های تصویر/PDF ابتدا DXF تولید می‌شود، سپس به DWG تبدیل می‌گردد.
