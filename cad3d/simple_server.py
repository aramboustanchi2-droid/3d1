"""
Simple CAD Conversion Server - Minimal and Robust
سرور ساده و قوی برای تبدیل CAD
"""

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, FileResponse
import tempfile
import shutil
import os
from pathlib import Path

app = FastAPI(title="Simple CAD Converter")


@app.get("/", response_class=HTMLResponse)
async def index():
    return """
<!doctype html>
<html lang="fa" dir="rtl">
<head>
  <meta charset="utf-8" />
  <title>مبدل ساده CAD</title>
  <style>
    body{font-family:Tahoma;max-width:600px;margin:2rem auto;padding:1rem;background:#f5f5f5}
    .card{background:#fff;border-radius:8px;padding:2rem;box-shadow:0 2px 8px rgba(0,0,0,.1)}
    h1{text-align:center;color:#333}
    label{display:block;margin:1rem 0 .5rem;font-weight:bold}
    input,select{width:100%;padding:.5rem;border:1px solid #ddd;border-radius:4px;box-sizing:border-box}
    button{width:100%;padding:.7rem;background:#28a745;color:#fff;border:none;border-radius:4px;cursor:pointer;margin-top:1rem;font-size:1rem}
    button:hover{background:#218838}
  </style>
</head>
<body>
  <div class="card">
    <h1>🔧 مبدل ساده CAD</h1>
    <form action="/convert" method="post" enctype="multipart/form-data">
      <label>📁 فایل ورودی (عکس، PDF، DXF، DWG):</label>
      <input type="file" name="file" required accept=".jpg,.jpeg,.png,.pdf,.dxf,.dwg" />
      
      <label>📦 فرمت خروجی:</label>
      <select name="out_format">
        <option value="dxf">DXF</option>
        <option value="dwg">DWG</option>
      </select>
      
      <label>
        <input type="checkbox" name="to_3d" value="yes" /> تبدیل به 3D
      </label>
      
      <label>ارتفاع (میلی‌متر):</label>
      <input type="number" name="height" value="3000" step="100" />
      
      <button type="submit">🚀 تبدیل</button>
    </form>
  </div>
</body>
</html>
"""


@app.post("/convert")
async def convert(
    file: UploadFile = File(...),
    out_format: str = Form("dxf"),
    to_3d: str = Form(None),
    height: float = Form(3000),
):
    print(f"\n=== Convert Request ===", flush=True)
    print(f"File: {file.filename}", flush=True)
    print(f"Format: {out_format}, 3D: {to_3d}, Height: {height}", flush=True)
    
    # Save uploaded file
    suffix = Path(file.filename).suffix.lower()
    fd_in, temp_input = tempfile.mkstemp(suffix=suffix)
    os.close(fd_in)
    
    with open(temp_input, 'wb') as f:
        content = await file.read()
        f.write(content)
    
    print(f"Saved to: {temp_input}", flush=True)
    
    # Output file
    out_ext = f".{out_format.lower()}"
    stem = Path(file.filename).stem
    out_filename = f"{stem}_3d{out_ext}" if to_3d else f"{stem}{out_ext}"
    
    fd_out, temp_output = tempfile.mkstemp(suffix=out_ext)
    os.close(fd_out)
    
    try:
        dxf_file = None
        
        # IMAGE to DXF
        if suffix in [".jpg", ".jpeg", ".png", ".bmp"]:
            print("Processing image...", flush=True)
            from .neural_cad_detector import NeuralCADDetector, DetectorConfig
            
            fd_tmp, dxf_file = tempfile.mkstemp(suffix=".dxf")
            os.close(fd_tmp)
            
            cfg = DetectorConfig(quality="high")
            detector = NeuralCADDetector(device="auto", config=cfg)
            print("Vectorizing...", flush=True)
            vectorized = detector.vectorize_drawing(temp_input, scale_mm_per_pixel=1.0)
            print(f"Found {len(vectorized.polygons)} polygons", flush=True)
            detector.convert_to_dxf(vectorized, dxf_file)
            
            # Verify
            import ezdxf
            doc = ezdxf.readfile(dxf_file)
            entities = list(doc.modelspace())
            print(f"DXF has {len(entities)} entities", flush=True)
        
        # PDF to DXF
        elif suffix == ".pdf":
            print("Processing PDF...", flush=True)
            from .neural_cad_detector import NeuralCADDetector, DetectorConfig
            from .pdf_processor import PDFToImageConverter, CADPipeline
            
            fd_tmp, dxf_file = tempfile.mkstemp(suffix=".dxf")
            os.close(fd_tmp)
            
            cfg = DetectorConfig(quality="high")
            detector = NeuralCADDetector(device="auto", config=cfg)
            pdf_conv = PDFToImageConverter(dpi=300)
            pipe = CADPipeline(neural_detector=detector, pdf_converter=pdf_conv)
            print("Converting PDF...", flush=True)
            pipe.process_pdf_to_dxf(temp_input, dxf_file, confidence_threshold=0.5, scale_mm_per_pixel=1.0)
            
            # Verify
            import ezdxf
            doc = ezdxf.readfile(dxf_file)
            entities = list(doc.modelspace())
            print(f"DXF has {len(entities)} entities", flush=True)
        
        # DXF direct
        elif suffix == ".dxf":
            dxf_file = temp_input
            print("Using DXF directly", flush=True)
        
        # DWG to DXF
        elif suffix == ".dwg":
            print("Converting DWG to DXF...", flush=True)
            from .dwg_io import convert_dwg_to_dxf
            
            fd_tmp, dxf_file = tempfile.mkstemp(suffix=".dxf")
            os.close(fd_tmp)
            convert_dwg_to_dxf(temp_input, dxf_file)
            print("DWG converted", flush=True)
        
        else:
            return {"error": f"Unsupported format: {suffix}"}
        
        # 3D conversion
        if to_3d and dxf_file:
            print(f"Applying 3D extrusion (height={height})...", flush=True)
            from .dxf_extrude import extrude_dxf_closed_polylines
            
            fd_3d, dxf_3d = tempfile.mkstemp(suffix=".dxf")
            os.close(fd_3d)
            
            extrude_dxf_closed_polylines(
                dxf_file,
                dxf_3d,
                height=float(height),
                layers=None,
                arc_segments=12,
                optimize=False,
            )
            dxf_file = dxf_3d
            print("3D extrusion complete", flush=True)
            
            # Verify 3D output
            import ezdxf
            doc = ezdxf.readfile(dxf_file)
            entities = list(doc.modelspace())
            print(f"3D DXF has {len(entities)} entities", flush=True)
        
        # Final output
        if out_format.lower() == "dwg":
            print("Converting to DWG...", flush=True)
            from .dwg_io import convert_dxf_to_dwg
            try:
                convert_dxf_to_dwg(dxf_file, temp_output)
                print("DWG conversion complete", flush=True)
            except Exception as e:
                print(f"DWG conversion failed: {e}, using DXF instead", flush=True)
                shutil.copy2(dxf_file, temp_output)
                out_filename = out_filename.replace(".dwg", ".dxf")
        else:
            shutil.copy2(dxf_file, temp_output)
            print("Copied to output", flush=True)
        
        print(f"✅ Success! Returning: {out_filename}", flush=True)
        return FileResponse(temp_output, filename=out_filename, media_type="application/octet-stream")
    
    except Exception as e:
        print(f"❌ Error: {e}", flush=True)
        import traceback
        print(traceback.format_exc(), flush=True)
        return {"error": str(e)}


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)
