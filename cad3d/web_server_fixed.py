from __future__ import annotations

from fastapi import FastAPI, UploadFile, File, Form, Request, Response
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path
import tempfile
import shutil
import traceback
import os
import json

# Internal modules
from .dxf_extrude import extrude_dxf_closed_polylines
from .dwg_io import convert_dxf_to_dwg, convert_dwg_to_dxf

BASE_DIR = Path(__file__).parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="CAD 2D→3D Converter", version="1.1", description="Enhanced UI with dynamic theming")

# Mount static files if directory exists
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

_theme_cache: dict | None = None
_current_theme: str | None = None
_theme_config_path = BASE_DIR / "theme_config.json"

def _load_theme_config() -> dict:
    global _theme_cache
    if _theme_cache is not None:
        return _theme_cache
    if not _theme_config_path.exists():
        # Create a default theme config
        default = {
            "default": os.getenv("DEFAULT_AI_THEME", "light"),
            "themes": [
                {"name": "light", "title": "روشن", "accent": "blue"},
                {"name": "dark", "title": "تاریک", "accent": "emerald"},
                {"name": "solar", "title": "سولار", "accent": "amber"},
                {"name": "midnight", "title": "نیمه‌شب", "accent": "violet"}
            ],
            "accents": ["blue", "emerald", "amber", "violet", "rose", "indigo"]
        }
        try:
            with open(_theme_config_path, 'w', encoding='utf-8') as f:
                json.dump(default, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
        _theme_cache = default
        return default
    try:
        with open(_theme_config_path, 'r', encoding='utf-8') as f:
            _theme_cache = json.load(f)
            # Override default from environment if provided
            env_default = os.getenv("DEFAULT_AI_THEME")
            if env_default:
                names = [t["name"] for t in _theme_cache.get("themes", [])]
                if env_default in names:
                    _theme_cache["default"] = env_default
            return _theme_cache
    except Exception:
        _theme_cache = {"default": "light", "themes": [], "accents": []}
        return _theme_cache

def _get_current_theme(request: Request) -> str:
    global _current_theme
    if _current_theme is None:
        cfg = _load_theme_config()
        _current_theme = cfg.get("default", "light")
    cookie_theme = request.cookies.get("theme")
    if cookie_theme:
        # validate
        names = [t["name"] for t in _load_theme_config().get("themes", [])]
        if cookie_theme in names:
            _current_theme = cookie_theme
    return _current_theme

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    theme_cfg = _load_theme_config()
    current = _get_current_theme(request)
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "themes": theme_cfg.get("themes", []),
            "accents": theme_cfg.get("accents", []),
            "current_theme": current,
        }
    )

@app.get("/api/themes")
def list_themes():
    cfg = _load_theme_config()
    return {"current": _current_theme or cfg.get("default", "light"), "themes": cfg.get("themes", []), "accents": cfg.get("accents", [])}

@app.post("/api/themes/select")
def select_theme(theme: str = Form(...), request: Request = None):
    cfg = _load_theme_config()
    names = [t["name"] for t in cfg.get("themes", [])]
    if theme not in names:
        return JSONResponse({"error": "Invalid theme"}, status_code=400)
    global _current_theme
    _current_theme = theme
    resp = RedirectResponse(url="/", status_code=303)
    resp.set_cookie("theme", theme, max_age=60*60*24*30)
    return resp


@app.get("/health")
def health():
  return {"status": "ok", "theme": _current_theme or _load_theme_config().get("default", "light")}


@app.post("/convert")
async def convert(
    file: UploadFile = File(...),
    out_format: str = Form("dxf"),
    to_3d: str = Form(None),
    height: float = Form(3000),
    dpi: int = Form(300),
    scale: float = Form(1.0),
    confidence: float = Form(0.5),
    quality: str = Form("standard"),
    photo_mode: str = Form("off"),
    ref_width_mm: str = Form(""),  # Changed to str to handle empty string
    dwg_version: str = Form("ACAD2018"),
    engineering_profile: str = Form("generic"),
    generate_report: str = Form(None),
):
    temp_input = None
    temp_output = None
    temp_intermediate = None
    
    print(f"=== Convert Request ===", flush=True)
    print(f"File: {file.filename}", flush=True)
    print(f"Out format: {out_format}", flush=True)
    print(f"To 3D: {to_3d}", flush=True)
    print(f"Height: {height}", flush=True)
    print(f"Engineering Profile: {engineering_profile}", flush=True)
    print(f"Generate Report: {generate_report}", flush=True)
    
    # Parse ref_width_mm from string
    ref_width_mm_value = None
    if ref_width_mm and ref_width_mm.strip():
        try:
            ref_width_mm_value = float(ref_width_mm)
        except ValueError:
            pass  # Ignore invalid values
    
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
            from .neural_cad_detector import NeuralCADDetector, DetectorConfig
            from .photo_rectify import process_photo_to_dxf
          except Exception:
            return JSONResponse({"error": "Neural pipeline not available. Install: pip install -r requirements-neural.txt"}, status_code=400)

          fd_tmp, temp_intermediate = tempfile.mkstemp(suffix=".dxf", prefix="cad3d_vec_")
          os.close(fd_tmp)
          if photo_mode == "on":
            # Rectify perspective and optionally auto-scale using reference width
            _vec, used_scale = process_photo_to_dxf(
              temp_input,
              temp_intermediate,
              reference_width_mm=ref_width_mm_value,
              quality=quality,
              fallback_scale_mm_per_pixel=float(scale),
            )
          else:
            cfg = DetectorConfig(quality=quality if quality in ("standard", "high") else "standard")
            detector = NeuralCADDetector(device="auto", config=cfg)
            vectorized = detector.vectorize_drawing(temp_input, scale_mm_per_pixel=float(scale))
            detector.convert_to_dxf(vectorized, temp_intermediate)
          dxf_to_process = temp_intermediate
        
        # PDF to DXF
        elif suffix == ".pdf":
            try:
                from .neural_cad_detector import NeuralCADDetector, DetectorConfig
                from .pdf_processor import PDFToImageConverter, CADPipeline
            except Exception:
                return JSONResponse({"error": "PDF pipeline not available. Install: pip install -r requirements-neural.txt PyMuPDF pdf2image"}, status_code=400)

            fd_tmp, temp_intermediate = tempfile.mkstemp(suffix=".dxf", prefix="cad3d_pdf_")
            os.close(fd_tmp)

            cfg = DetectorConfig(quality=quality if quality in ("standard", "high") else "standard")
            detector = NeuralCADDetector(device="auto", config=cfg)
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
        print(f"DXF to process: {dxf_to_process}", flush=True)
        
        # Verify DXF has content before 3D conversion
        try:
            import ezdxf
            test_doc = ezdxf.readfile(dxf_to_process)
            test_msp = test_doc.modelspace()
            test_entities = list(test_msp)
            print(f"DXF contains {len(test_entities)} entities before 3D conversion", flush=True)
            if len(test_entities) == 0:
                print("WARNING: Input DXF is empty! Vectorization may have failed.", flush=True)
        except Exception as e:
            print(f"Failed to verify DXF: {e}", flush=True)
        
        # Apply 3D conversion if requested
        # TEMPORARILY DISABLED ADVANCED CONVERSION - USE SIMPLE EXTRUSION
        analysis_report_html = None
        conversion_report_html = None
        
        if to_3d:
            # Always use simple extrusion for now until vectorization is fixed
            try:
                print(f"Starting simple 3D extrusion on: {dxf_to_process}", flush=True)
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
                print(f"Simple extrusion complete: {temp_extruded}", flush=True)
                
                # Verify output
                import ezdxf
                verify_doc = ezdxf.readfile(temp_extruded)
                verify_msp = verify_doc.modelspace()
                verify_count = len(list(verify_msp))
                print(f"Extruded DXF contains {verify_count} entities", flush=True)
                
            except Exception as e:
                print(f"Simple extrusion failed: {e}", flush=True)
                print(f"Traceback: {traceback.format_exc()}", flush=True)
                # If even simple extrusion fails, keep 2D
                print("Keeping 2D DXF as output", flush=True)
        
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
        
        # Return file (and optionally report)
        if generate_report and conversion_report_html:
            # Save report as HTML alongside output
            report_path = temp_output + "_report.html"
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(conversion_report_html)
            
            # For now, just return the 3D model; user can request report separately
            # Future: return ZIP with both model and report
        
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
