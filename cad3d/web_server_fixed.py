from __future__ import annotations

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from pathlib import Path
import tempfile
import shutil
import traceback
import os

# Internal modules
from .dxf_extrude import extrude_dxf_closed_polylines
from .dwg_io import convert_dxf_to_dwg, convert_dwg_to_dxf

app = FastAPI(title="CAD 2D→3D Converter", version="1.0")


INDEX_HTML = """
<!doctype html>
<html lang="fa" dir="rtl">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>مبدل جامع CAD - تبدیل دوبعدی به سه‌بعدی</title>
  <style>
    body{font-family:Tahoma,Arial,sans-serif;max-width:800px;margin:2rem auto;padding:0 1rem;background:#f5f5f5}
    .card{background:#fff;border:1px solid #ddd;border-radius:12px;padding:1.5rem;box-shadow:0 2px 12px rgba(0,0,0,.06)}
    h1{font-size:1.5rem;margin:0 0 1rem;color:#333;text-align:center}
    .subtitle{text-align:center;color:#666;font-size:.9rem;margin-bottom:1.5rem}
    .info{background:#e7f3ff;border:1px solid #2196F3;padding:1rem;border-radius:8px;margin-bottom:1.5rem;color:#0d47a1}
    label{display:block;margin:.8rem 0 .3rem;color:#333;font-weight:500}
    input[type="file"],input[type="number"],select{width:100%;padding:.6rem;border:1px solid #ccc;border-radius:8px;box-sizing:border-box}
    input[type="checkbox"]{margin-left:.5rem}
    .actions{margin-top:1.5rem;text-align:center}
    button{background:#28a745;color:#fff;border:none;padding:.7rem 2rem;border-radius:8px;cursor:pointer;font-size:1rem}
    button:hover{background:#218838}
    .help{font-size:.85rem;color:#666;margin-top:1rem;text-align:center}
    .advanced{background:#f8f9fa;padding:1rem;border-radius:8px;margin-top:1rem}
    .advanced-toggle{cursor:pointer;color:#007bff;text-align:center;margin-top:1rem}
  </style>
  <script>
    function toggleAdvanced() {
      const adv = document.getElementById('advanced');
      adv.style.display = adv.style.display === 'none' ? 'block' : 'none';
    }
  </script>
</head>
<body>
  <div class="card">
    <h1>🏗️ مبدل جامع CAD</h1>
    <p class="subtitle">تبدیل عکس، PDF، DXF و DWG به فرمت سه‌بعدی</p>
    <div class="info">
      ✅ فرمت‌های ورودی: <strong>JPG, PNG, PDF, DXF, DWG</strong><br/>
      ✅ فرمت‌های خروجی: DXF, DWG (با ODA)
    </div>
    <form method="post" action="/convert" enctype="multipart/form-data">
      <label>📁 فایل ورودی</label>
      <input type="file" name="file" accept=".jpg,.jpeg,.png,.pdf,.dxf,.dwg" required />

      <label>📤 فرمت خروجی</label>
      <select name="out_format">
        <option value="dxf">DXF</option>
        <option value="dwg">DWG (نیاز به ODA)</option>
      </select>

      <label style="margin-top:1.2rem">
        <input type="checkbox" name="to_3d" value="yes" checked />
        اکستروژن سه‌بعدی (بعد از تبدیل)
      </label>

      <label>📏 ارتفاع اکستروژن (میلی‌متر)</label>
      <input type="number" step="1" name="height" value="3000" min="1" />

      <div class="advanced-toggle" onclick="toggleAdvanced()">⚙️ تنظیمات پیشرفته (کلیک کنید)</div>
      
      <div id="advanced" class="advanced" style="display:none">
        <label>DPI (برای PDF)</label>
        <input type="number" step="1" name="dpi" value="300" min="72" max="600" />
        
        <label>Scale (mm per pixel - برای عکس/PDF)</label>
        <input type="number" step="0.1" name="scale" value="1.0" min="0.1" />
        
        <label>Confidence (برای PDF)</label>
        <input type="number" step="0.05" name="confidence" value="0.5" min="0" max="1" />
        
        <label>DWG Version</label>
        <select name="dwg_version">
          <option value="ACAD2018">ACAD2018 (R2018)</option>
          <option value="ACAD2013">ACAD2013 (R2013)</option>
          <option value="ACAD2010">ACAD2010 (R2010)</option>
          <option value="ACAD2007">ACAD2007 (R2007)</option>
        </select>
      </div>

      <div class="actions">
        <button type="submit">🚀 تبدیل و دانلود</button>
      </div>
      <p class="help">
        📌 برای DXF/DWG: LWPOLYLINE بسته لازم است<br/>
        📌 برای خروجی DWG: ODA File Converter باید نصب باشد<br/>
          ✅ همه قابلیت‌ها فعال است!
      </p>
    </form>
  </div>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index():
    return INDEX_HTML


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/convert")
async def convert(
    file: UploadFile = File(...),
    out_format: str = Form("dxf"),
    to_3d: str = Form(None),
    height: float = Form(3000),
    dpi: int = Form(300),
    scale: float = Form(1.0),
    confidence: float = Form(0.5),
    dwg_version: str = Form("ACAD2018"),
):
    temp_input = None
    temp_output = None
    temp_intermediate = None
    
    print(f"=== Convert Request ===", flush=True)
    print(f"File: {file.filename}", flush=True)
    print(f"Out format: {out_format}", flush=True)
    print(f"To 3D: {to_3d}", flush=True)
    print(f"Height: {height}", flush=True)
    
    try:
        # Create temp input file
        suffix = Path(file.filename).suffix.lower()
        stem = Path(file.filename).stem
        
        fd_in, temp_input = tempfile.mkstemp(suffix=suffix, prefix="cad3d_in_")
        os.close(fd_in)
        
        # Save uploaded file
        with open(temp_input, "wb") as f:
            content = await file.read()
            f.write(content)
        
        out_format = out_format.lower()
        out_ext = f".{out_format}"
        
        # Determine final output filename
        suffix_3d = "_3d" if to_3d else ""
        out_filename = f"{stem}{suffix_3d}{out_ext}"
        
        # Create temp output file
        fd_out, temp_output = tempfile.mkstemp(suffix=out_ext, prefix="cad3d_out_")
        os.close(fd_out)
        
        # Process based on input type
        dxf_to_process = None
        
        # IMAGE to DXF
        if suffix in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]:
            try:
                from .neural_cad_detector import NeuralCADDetector
            except Exception:
                return JSONResponse({"error": "Neural pipeline not available. Install: pip install -r requirements-neural.txt"}, status_code=400)
            
            fd_tmp, temp_intermediate = tempfile.mkstemp(suffix=".dxf", prefix="cad3d_vec_")
            os.close(fd_tmp)
            
            detector = NeuralCADDetector(device="auto")
            vectorized = detector.vectorize_drawing(temp_input, scale_mm_per_pixel=float(scale))
            detector.convert_to_dxf(vectorized, temp_intermediate)
            dxf_to_process = temp_intermediate
        
        # PDF to DXF
        elif suffix == ".pdf":
            try:
                from .neural_cad_detector import NeuralCADDetector
                from .pdf_processor import PDFToImageConverter, CADPipeline
            except Exception:
                return JSONResponse({"error": "PDF pipeline not available. Install: pip install -r requirements-neural.txt PyMuPDF pdf2image"}, status_code=400)
            
            fd_tmp, temp_intermediate = tempfile.mkstemp(suffix=".dxf", prefix="cad3d_pdf_")
            os.close(fd_tmp)
            
            detector = NeuralCADDetector(device="auto")
            pdf_conv = PDFToImageConverter(dpi=int(dpi))
            pipe = CADPipeline(neural_detector=detector, pdf_converter=pdf_conv)
            pipe.process_pdf_to_dxf(temp_input, temp_intermediate, confidence_threshold=float(confidence), scale_mm_per_pixel=float(scale))
            dxf_to_process = temp_intermediate
        
        # DXF
        elif suffix == ".dxf":
            dxf_to_process = temp_input
        
        # DWG to DXF
        elif suffix == ".dwg":
            try:
                fd_tmp, temp_intermediate = tempfile.mkstemp(suffix=".dxf", prefix="cad3d_dwg_")
                os.close(fd_tmp)
                convert_dwg_to_dxf(temp_input, temp_intermediate, out_version=dwg_version)
                dxf_to_process = temp_intermediate
            except Exception as e:
                return JSONResponse({"error": f"DWG conversion requires ODA File Converter. Error: {e}"}, status_code=400)
        
        else:
            return JSONResponse({"error": f"Unsupported input format: {suffix}"}, status_code=400)
        
        # Now we have a DXF file in dxf_to_process
        # Apply 3D extrusion if requested
        if to_3d:
            fd_tmp2, temp_extruded = tempfile.mkstemp(suffix=".dxf", prefix="cad3d_3d_")
            os.close(fd_tmp2)
            
            extrude_dxf_closed_polylines(
                dxf_to_process,
                temp_extruded,
                height=float(height),
                layers=None,
                arc_segments=12,
                arc_max_seglen=None,
                optimize=False,
                detect_hard_shapes=False,
                hard_shapes_collector=None,
                colorize=False,
                split_by_color=False,
                color_stats_collector=None,
            )
            dxf_to_process = temp_extruded
        
        # Convert to final output format
        if out_format == "dwg":
            try:
                convert_dxf_to_dwg(dxf_to_process, temp_output, out_version=dwg_version)
            except Exception as e:
                # Fallback to DXF
                shutil.copy2(dxf_to_process, temp_output)
                out_filename = out_filename.replace(".dwg", ".dxf")
                print(f"Warning: DWG conversion failed, returning DXF instead. Error: {e}", flush=True)
        else:
            # Output DXF
            shutil.copy2(dxf_to_process, temp_output)
        
        # Return file
        return FileResponse(
            temp_output,
            filename=out_filename,
            media_type="application/octet-stream",
        )
        
    except Exception as e:
        # Clean up on error
        for tmp_file in [temp_input, temp_output, temp_intermediate]:
            if tmp_file and os.path.exists(tmp_file):
                try:
                    os.unlink(tmp_file)
                except:
                    pass
        
        error_msg = f"Error: {type(e).__name__}: {e}"
        print(error_msg, flush=True)
        traceback.print_exc()
        return JSONResponse({"error": error_msg}, status_code=500)
