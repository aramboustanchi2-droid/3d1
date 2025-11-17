# CAD 2D→3D Offline Converter — Copilot Guide

Purpose: Help AI agents work productively on this codebase by capturing project-specific patterns, workflows, and gotchas in ~40–50 lines.

Architecture (cad3d/)
- cli.py: Single entry-point with subcommands: dxf-extrude, img-to-3d, dxf-to-dwg, auto-extrude, batch-extrude.
- dxf_extrude.py: LWPOLYLINE filtering and extrusion; arc handling via bulge; adaptive sampling; optional hard-shape detection and skip with reporting hook.
- mesh_utils.py: ear clipping, prism mesh construction; vertex dedup (optimize_vertices); polygon diagnostics (detect_polygon_issues).
- image_to_depth.py: MiDaS ONNX inference (CPU), grid mesh build; size/scale params.
- dwg_io.py: ODA File Converter wrappers for DXF↔DWG.
- config.py: DEFAULT_EXTRUDE_HEIGHT, ODA_CONVERTER_PATH, MIDAS_ONNX_PATH from env.

Core patterns and APIs
- ezdxf 1.4+: write meshes via Mesh.edit_data():
  with mesh.edit_data() as md: md.vertices = verts; md.faces = faces
- Arc approximation: _approximate_lwpolyline_points(e, arc_segments, arc_max_seglen). If arc_max_seglen is set (drawing units), segments = ceil(arc_len/arc_max_seglen), capped (≤512).
- Face winding: bottom CCW, top reversed; sides as two triangles per edge (see build_prism_mesh).
- Large meshes: enable --optimize-vertices to deduplicate vertices before writing.
- Hard-shape detection: detect_polygon_issues(poly) returns issues like [duplicate_vertex, zero_length_edge, tiny_or_zero_area, self_intersection]. CLI flag --detect-hard-shapes skips such polylines and can write CSV via --hard-report-csv, or JSON via --hard-report-json. JSON includes extra metadata (area, bbox, centroid).
- Color handling: with --colorize, set MESH true_color from entity true_color, ACI, or layer ACI fallback. With --split-by-color, write meshes to `<layer>__COLOR_R_G_B` layers and ensure they exist in the output doc (layer color approximated via rgb2aci). Use --color-report-csv or --color-report-json to generate reports of mesh color usage (CSV aggregates by layer/RGB; JSON includes raw per-mesh entries with source/target layer and RGB values).

Key workflows (examples)
- DXF→3D DXF: python -m cad3d.cli dxf-extrude --input plan.dxf --output plan_3d.dxf --height 3000 --layers WALLS COLUMNS --arc-max-seglen 50 --optimize-vertices --detect-hard-shapes --hard-report-csv hard.csv
- DXF→3D DXF (JSON report): python -m cad3d.cli dxf-extrude --input plan.dxf --output plan_3d.dxf --height 3000 --detect-hard-shapes --hard-report-json hard.json
- DXF→3D DXF (colorize): python -m cad3d.cli dxf-extrude --input plan.dxf --output plan_3d.dxf --height 3000 --colorize --split-by-color
- DXF→3D DXF (color reports): python -m cad3d.cli dxf-extrude --input plan.dxf --output plan_3d.dxf --height 3000 --colorize --split-by-color --color-report-csv colors.csv --color-report-json colors.json
- Image→3D DXF: $env:MIDAS_ONNX_PATH="models\midas_v2_small_256.onnx"; python -m cad3d.cli img-to-3d --input photo.jpg --output photo_3d.dxf --scale 1000 --size 256 --optimize-vertices
- Auto (DXF/DWG in → DXF/DWG out): python -m cad3d.cli auto-extrude --input plan.dwg --output plan_3d.dwg --height 3000 --dwg-version ACAD2018
- Batch: python -m cad3d.cli batch-extrude --input-dir .\in --output-dir .\out --out-format DXF --recurse --jobs 4 --report-csv .\out\report.csv --detect-hard-shapes --hard-report-csv .\out\hard.csv --colorize --split-by-color --color-report-csv .\out\colors.csv

Testing and dev
- Run tests: python -m pytest -q (see tests/ for CLI and mesh utility tests).
- For new geometry ops, add cases under tests/ and use tmp_path for I/O.
- Preferred DXF writing path: create a fresh ezdxf.new(setup=True) doc for 3D results to avoid mixing 2D/3D.

Conventions and pitfalls
- Only closed LWPOLYLINEs are extruded. Arcs come from bulge values. Self-intersecting or degenerate polylines can be detected/skipped with --detect-hard-shapes.
- Units are drawing-defined (often mm). Extrusion is +Z from z=0 to z=height.
- DWG I/O requires ODA File Converter; set ODA_CONVERTER_PATH in .env if not auto-detected.
- MiDaS model path: MIDAS_ONNX_PATH env or --model-path. img-to-3d is for visualization, not CAD-precise.

When extending
- Reuse build_prism_mesh and optimize_vertices for consistent topology/perf.
- New converters: add a module, wire a subcommand in cli.py, write a tmp_path-based test.
- Keep CLI options coherent: --layers, --arc-segments/--arc-max-seglen, --optimize-vertices, version flags for DWG.
