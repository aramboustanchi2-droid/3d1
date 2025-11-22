# CAD3D: AI-Powered CAD Analysis & 3D Conversion — Copilot Guide

Purpose: Help AI agents work productively on this comprehensive CAD analysis system combining traditional CAD conversion with state-of-the-art AI pipelines.

## Big Picture Architecture

**Core Philosophy**: Dual-path system supporting both traditional CAD geometry operations and advanced AI-driven analysis/conversion:
1. **Traditional Path**: DXF/DWG geometric extrusion (fast, precise, deterministic)
2. **AI Path**: Multi-model neural pipeline (ViT→VAE→Diffusion→GNN) for image/PDF conversion and intelligent analysis

**Key Subsystems** (cad3d/ contains ~50+ modules):
- **CLI Hub** (cli.py): 15+ subcommands organized by function: core (dxf-extrude, img-to-3d), workflow (auto-extrude, batch-extrude), AI (pdf-to-dxf, deep-hybrid), MLOps (train, optimize-model)
- **Geometry Core**: dxf_extrude.py, mesh_utils.py, dwg_io.py — traditional CAD operations
- **Neural Pipeline**: vision_transformer_cad.py, vae_model.py, diffusion_3d_model.py, hybrid_vae_vit_diffusion.py — deep learning stack
- **Graph Analysis**: cad_graph.py, cad_gnn.py, cad_graph_builder.py, crf_segmentation.py — relationship and structural analysis
- **Domain Analyzers**: architectural_analyzer.py, structural_analysis.py, parametric_engine.py — industry-specific intelligence
- **Web Interface**: web_server_fixed.py (FastAPI) — browser-based UI with API endpoints
- **Config**: config.py — DEFAULT_EXTRUDE_HEIGHT, ODA_CONVERTER_PATH, MIDAS_ONNX_PATH from env

**Supported Industries** (14+): Building, Bridge, Road, Dam, Tunnel, Factory, Machinery, MEP, Electrical, Plumbing, HVAC, Railway, Airport, Shipbuilding

## Critical Developer Workflows

**Setup & Dependencies**:
```powershell
# 1. Install PyTorch FIRST (critical order):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118  # or cpu
# 2. Install all other deps:
pip install -r requirements.txt
# 3. Optional: Neural extras (PDF, advanced AI):
pip install -r requirements-neural.txt
# 4. Set environment (create .env file):
DEFAULT_EXTRUDE_HEIGHT=3000.0
ODA_CONVERTER_PATH=C:\path\to\ODAFileConverter.exe  # for DWG support
MIDAS_ONNX_PATH=models\midas_v2_small_256.onnx      # for img-to-3d
```

**Testing** (pytest.ini disables auto-discovery):
```powershell
# Run specific test file (bypass pytest.ini):
python -m pytest -q -c none tests/test_dxf_extrude.py
# Run all tests ignoring config:
python -m pytest -q -c none tests/
# Tests use tmp_path fixture for I/O isolation
```

**Web Server** (development):
```powershell
cd cad3d
python web_server_fixed.py  # Launches FastAPI on http://localhost:8000
# Or: uvicorn cad3d.web_server_fixed:app --reload
```

## Core Patterns & APIs

**ezdxf 1.4+ Mesh Writing** (critical for all 3D output):
```python
with mesh.edit_data() as md:
    md.vertices = vertices_list  # List[Tuple[float, float, float]]
    md.faces = faces_list        # List[Tuple[int, int, int]] or quads
```

**Arc Approximation** (2 modes):
- Fixed: `--arc-segments 16` (simple, predictable)
- Adaptive: `--arc-max-seglen 50` (better quality, drawing-unit-based, segments=ceil(arc_len/max_seglen), capped ≤512)
- Implementation: `_approximate_lwpolyline_points(e, arc_segments, arc_max_seglen)` in dxf_extrude.py

**Face Winding** (build_prism_mesh in mesh_utils.py):
- Bottom: CCW (counter-clockwise)
- Top: reversed (CW)
- Sides: two triangles per edge, careful vertex ordering for normals

**Hard-Shape Detection** (quality control):
```python
issues = detect_polygon_issues(poly)  # Returns: duplicate_vertex, zero_length_edge, tiny_or_zero_area, self_intersection
# CLI: --detect-hard-shapes --hard-report-csv or --hard-report-json (JSON includes area, bbox, centroid)
```

**Color Handling** (--colorize, --split-by-color):
- Source: entity true_color → ACI → layer ACI (fallback chain)
- Target: mesh true_color + optional new layers `<layer>__COLOR_R_G_B`
- Reports: --color-report-csv (aggregated by layer/RGB) or --color-report-json (raw per-mesh entries)

**Deep Hybrid Model** (hybrid_vae_vit_diffusion.py):
```python
converter = create_deep_hybrid_converter(
    device='cuda',           # or 'cpu'
    prior_strength=0.5,      # 0..1: VAE prior weight vs. noise in diffusion init
    normalize_range=(0, 1000),  # mm range for output coords
    ddim_steps=40            # diffusion sampling steps (set 0 to bypass in tests)
)
result = converter.convert(img_path, output_dxf_path)
# Returns dict with 'dxf', 'num_points', 'encode_time_sec', 'decode_time_sec', 'total_time_sec'
```

**Neural Detector Pipeline** (lazy import pattern):
```python
# CLI handlers import heavy deps inside functions to keep startup fast
def handle_pdf_to_dxf(args):
    from .neural_cad_detector import NeuralCADDetector
    from .pdf_processor import PDFToImageConverter, CADPipeline
    detector = NeuralCADDetector(device=args.device)
    pipeline = CADPipeline(neural_detector=detector, pdf_converter=PDFToImageConverter(dpi=args.dpi))
    pipeline.process_pdf_to_dxf(args.input, args.output)
```

## Example Workflows

**Basic Extrusion** (fast, no AI):
```powershell
python -m cad3d.cli dxf-extrude --input plan.dxf --output plan_3d.dxf --height 3000 --layers WALLS COLUMNS --arc-max-seglen 50 --optimize-vertices --detect-hard-shapes --hard-report-csv hard.csv
```

**Auto-Convert** (handles DXF/DWG transparently):
```powershell
# DWG→3D DWG (converts to DXF, extrudes, converts back):
python -m cad3d.cli auto-extrude --input plan.dwg --output plan_3d.dwg --height 3000 --dwg-version ACAD2018 --detect-hard-shapes --hard-report-json hard.json
```

**Batch Processing** (parallel):
```powershell
python -m cad3d.cli batch-extrude --input-dir .\samples --output-dir .\outputs --out-format DXF --height 3000 --recurse --jobs 4 --report-csv report.csv --detect-hard-shapes --hard-report-csv hard.csv --colorize --split-by-color --color-report-csv colors.csv
```

**AI-Driven Conversions**:
```powershell
# Image→3D via depth estimation:
$env:MIDAS_ONNX_PATH="models\midas_v2_small_256.onnx"
python -m cad3d.cli img-to-3d --input photo.jpg --output photo_3d.dxf --scale 1000 --size 256

# PDF→2D DXF via neural detection:
python -m cad3d.cli pdf-to-dxf --input drawing.pdf --output drawing.dxf --dpi 300 --confidence 0.5 --device cuda

# Image→3D via ViT+VAE+Diffusion fusion:
python -m cad3d.cli deep-hybrid --input img.png --output out_3d.dxf --prior-strength 0.6 --ddim-steps 40 --normalize-range 0 1000

# Universal converter (auto-detects input type):
python -m cad3d.cli universal-convert --input plan.pdf --output plan_3d.dwg --to-3d --height 3000 --device cuda
```

## Conventions & Pitfalls

**Geometry**:
- Only closed LWPOLYLINEs are extruded (open polylines ignored)
- Arcs come from bulge values; approximated to line segments
- Units: drawing-defined (often mm); extrusion is +Z from z=0 to z=height
- Self-intersecting/degenerate polys: use --detect-hard-shapes to skip and report

**File I/O**:
- DWG support requires external ODA File Converter (set ODA_CONVERTER_PATH or auto-detect)
- Preferred pattern: create fresh `ezdxf.new(setup=True)` doc for 3D output (avoids 2D/3D mixing)
- Temp files in auto-extrude: cleaned unless --keep-temp flag

**AI Models**:
- MiDaS (img-to-3d): ONNX model, CPU-only, coarse depth (visualization quality, not CAD-precise)
- Neural pipeline: requires requirements-neural.txt (PyTorch, transformers, etc.)
- Device handling: 'auto' picks CUDA if available, else CPU
- Deep-hybrid: most compute-intensive; prior_strength and ddim_steps are key tuning params

**Testing**:
- pytest.ini disables discovery → ALWAYS use: `python -m pytest -q -c none tests/test_*.py`
- Tests use tmp_path for I/O (automatic cleanup)
- For geometry tests, compare mesh vertex counts and bounds (not exact floats)

## When Extending

**New CLI Command**:
1. Add parser in cli.py → `_setup_*_parsers()` function group
2. Create handler function → `handle_your_command(args)`
3. Register in COMMAND_HANDLERS dict
4. Add test in tests/ with tmp_path fixture

**New Neural Model**:
1. Create module in cad3d/ (e.g., new_model.py)
2. Lazy import in CLI handler (keep startup fast)
3. Follow DeepHybridConverter pattern: device param, normalize_range, timing metrics
4. Return dict with 'dxf' path + metadata for sidecar JSON

**New Domain Analyzer**:
1. Inherit from base analyzer pattern (see architectural_analyzer.py)
2. Implement analyze() → returns structured data
3. Add CLI subcommand under _setup_ml_ops_parsers()
4. Support --export-json, --export-csv, --report flags

**Mesh Operations**:
- Reuse build_prism_mesh() for consistent topology
- Call optimize_vertices() for large meshes (dedups verts)
- Use detect_polygon_issues() before extrusion for quality gates
