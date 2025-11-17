from __future__ import annotations

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse, PlainTextResponse
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional
import shutil
import traceback

# Internal modules
from .dwg_io import convert_dxf_to_dwg, convert_dwg_to_dxf
from .dxf_extrude import extrude_dxf_closed_polylines

app = FastAPI(title="CAD 2D→3D Converter", version="1.0")


INDEX_HTML = """
<!doctype html>
<html lang="fa" dir="rtl">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>تبدیل DXF دوبعدی به سه‌بعدی</title>
  <style>
    body{font-family:Tahoma,Arial,sans-serif;max-width:700px;margin:2rem auto;padding:0 1rem;background:#f5f5f5}
    .card{background:#fff;border:1px solid #ddd;border-radius:12px;padding:1.5rem;box-shadow:0 2px 12px rgba(0,0,0,.06)}
    h1{font-size:1.5rem;margin:0 0 1.5rem;color:#333;text-align:center}
    .alert{background:#fff3cd;border:1px solid #ffc107;padding:1rem;border-radius:8px;margin-bottom:1.5rem;color:#856404}
    label{display:block;margin:.8rem 0 .3rem;color:#333;font-weight:500}
    input[type="file"],input[type="number"]{width:100%;padding:.6rem;border:1px solid #ccc;border-radius:8px;box-sizing:border-box}
    input[type="checkbox"]{margin-left:.5rem}
    .actions{margin-top:1.5rem;text-align:center}
    button{background:#28a745;color:#fff;border:none;padding:.7rem 2rem;border-radius:8px;cursor:pointer;font-size:1rem}
    button:hover{background:#218838}
    .help{font-size:.85rem;color:#666;margin-top:1rem;text-align:center}
  </style>
</head>
<body>
  <div class="card">
    <h1>🏗️ تبدیل DXF دوبعدی به سه‌بعدی</h1>
    <div class="alert">
      ⚠️ فقط فایل‌های DXF پشتیبانی می‌شوند. خروجی: DXF
    </div>
    <form method="post" action="/convert" enctype="multipart/form-data">
      <label>📁 فایل DXF ورودی</label>
      <input type="file" name="file" accept=".dxf" required />

      <label style="margin-top:1.2rem">
        <input type="checkbox" name="to_3d" checked />
        اکستروژن سه‌بعدی
      </label>

      <label>📏 ارتفاع اکستروژن (میلی‌متر)</label>
      <input type="number" step="1" name="height" value="3000" min="1" />

      <input type="hidden" name="out_ext" value="dxf" />

      <div class="actions">
        <button type="submit">🚀 تبدیل و دانلود</button>
      </div>
      <p class="help">
        فایل DXF ورودی باید حاوی LWPOLYLINE بسته باشد.<br/>
        خروجی: فایل DXF سه‌بعدی با MESH
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
    out_ext: str = Form("dxf"),
    dwg_version: str = Form("ACAD2018"),
    to_3d: Optional[str] = Form(None),
    height: float = Form(3000),
    dpi: int = Form(300),
    device: str = Form("auto"),
    confidence: float = Form(0.5),
    scale: float = Form(1.0),
):
    import tempfile
    try:
        out_ext = out_ext.lower().lstrip(".")
        if out_ext not in ("dxf", "dwg"):
            return JSONResponse({"error": "out_ext must be dxf or dwg"}, status_code=400)

        with TemporaryDirectory() as td:
            tdir = Path(td)
            # Save upload
            in_path = tdir / file.filename
            with in_path.open("wb") as f:
                shutil.copyfileobj(file.file, f)

            # Prepare output path
            out_name = in_path.stem + ("_3d" if to_3d else "") + "." + out_ext
            out_path = tdir / out_name

            # Determine type
            ext = in_path.suffix.lower()

            def write_success(p: Path) -> FileResponse:
                # Copy to a persistent temp file before returning
                fd, persistent_path = tempfile.mkstemp(suffix=p.suffix, prefix="cad3d_")
                import os
                os.close(fd)
                shutil.copy2(str(p), persistent_path)
                return FileResponse(persistent_path, filename=p.name, background=None)

            # Helper to extrude to destination
            def extrude_to_destination(in_dxf: Path) -> Path:
                if out_ext == "dwg":
                    tmp_out_dxf = tdir / (in_path.stem + "_tmp_out3d.dxf")
                    extrude_dxf_closed_polylines(
                        str(in_dxf),
                        str(tmp_out_dxf),
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
                    try:
                        convert_dxf_to_dwg(str(tmp_out_dxf), str(out_path), out_version=dwg_version)
                        return out_path
                    except Exception:
                        # ODA not available, fallback to DXF
                        return tmp_out_dxf
                elif out_ext == "dxf":
                    extrude_dxf_closed_polylines(
                        str(in_dxf),
                        str(out_path),
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
                    return out_path
                else:
                    raise RuntimeError("Unsupported output format")

            # IMAGE
            if ext in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"):
                try:
                    from .neural_cad_detector import NeuralCADDetector
                except Exception:
                    return JSONResponse({"error": "Neural pipeline not available. Install requirements-neural.txt"}, status_code=400)
                detector = NeuralCADDetector(device=device)
                vectorized = detector.vectorize_drawing(str(in_path), scale_mm_per_pixel=float(scale))
                tmp_vec_dxf = tdir / (in_path.stem + "_tmp2d.dxf")
                detector.convert_to_dxf(vectorized, str(tmp_vec_dxf))
                if to_3d:
                    final_path = extrude_to_destination(tmp_vec_dxf)
                else:
                    if out_ext == "dwg":
                        try:
                            convert_dxf_to_dwg(str(tmp_vec_dxf), str(out_path), out_version=dwg_version)
                            final_path = out_path
                        except Exception:
                            # ODA not available, return DXF instead
                            final_path = tmp_vec_dxf
                    else:
                        final_path = tmp_vec_dxf
                return write_success(final_path)

            # PDF
            if ext == ".pdf":
                try:
                    from .neural_cad_detector import NeuralCADDetector
                    from .pdf_processor import PDFToImageConverter, CADPipeline
                except Exception:
                    return JSONResponse({"error": "PDF pipeline not available. Install PyMuPDF/pdf2image and requirements-neural.txt"}, status_code=400)
                detector = NeuralCADDetector(device=device)
                pdf_conv = PDFToImageConverter(dpi=int(dpi))
                pipe = CADPipeline(neural_detector=detector, pdf_converter=pdf_conv)
                tmp_vec_dxf = tdir / (in_path.stem + "_tmp2d.dxf")
                pipe.process_pdf_to_dxf(str(in_path), str(tmp_vec_dxf), confidence_threshold=float(confidence), scale_mm_per_pixel=float(scale))
                if to_3d:
                    final_path = extrude_to_destination(tmp_vec_dxf)
                else:
                    if out_ext == "dwg":
                        try:
                            convert_dxf_to_dwg(str(tmp_vec_dxf), str(out_path), out_version=dwg_version)
                            final_path = out_path
                        except Exception:
                            # ODA not available, return DXF instead
                            final_path = tmp_vec_dxf
                    else:
                        final_path = tmp_vec_dxf
                return write_success(final_path)

            # DXF
            if ext == ".dxf":
                if to_3d:
                    final_path = extrude_to_destination(in_path)
                else:
                    if out_ext == "dwg":
                        try:
                            convert_dxf_to_dwg(str(in_path), str(out_path), out_version=dwg_version)
                            final_path = out_path
                        except Exception:
                            # ODA not available, return DXF instead
                            final_path = in_path
                    else:
                        final_path = in_path
                return write_success(final_path)

            # DWG
            if ext == ".dwg":
                if to_3d:
                    try:
                        tmp_in_dxf = tdir / (in_path.stem + "_tmp_in2d.dxf")
                        convert_dwg_to_dxf(str(in_path), str(tmp_in_dxf), out_version=dwg_version)
                        final_path = extrude_to_destination(tmp_in_dxf)
                    except Exception:
                        return JSONResponse({"error": "DWG input requires ODA File Converter for conversion."}, status_code=400)
                else:
                    if out_ext == "dwg":
                        shutil.copyfile(str(in_path), str(out_path))
                        final_path = out_path
                    else:
                        try:
                            convert_dwg_to_dxf(str(in_path), str(out_path), out_version=dwg_version)
                            final_path = out_path
                        except Exception:
                            return JSONResponse({"error": "DWG→DXF conversion requires ODA File Converter."}, status_code=400)
                return write_success(final_path)

            return JSONResponse({"error": "Unsupported input format"}, status_code=400)
    except Exception as e:
        import sys
        error_msg = f"/convert error: {type(e).__name__}: {e}"
        print(error_msg, file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return JSONResponse({"error": error_msg, "traceback": traceback.format_exc()}, status_code=500)
