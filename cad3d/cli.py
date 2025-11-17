from __future__ import annotations
import argparse
from pathlib import Path

from .config import settings
from .dxf_extrude import extrude_dxf_closed_polylines
from .dwg_io import convert_dxf_to_dwg
# Lazily import image_to_depth only when needed to avoid heavy deps during CLI import


def _positive_float(x: str) -> float:
    v = float(x)
    if v <= 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return v


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="cad3d", description="Offline 2D->3D CAD tools")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ex = sub.add_parser("dxf-extrude", help="Extrude closed polylines in DXF to 3D MESH")
    p_ex.add_argument("--input", required=True, help="Input DXF path")
    p_ex.add_argument("--output", required=True, help="Output DXF path")
    p_ex.add_argument("--height", type=_positive_float, default=settings.default_height, help="Extrusion height in drawing units")
    p_ex.add_argument("--layers", nargs="*", help="Only extrude polylines on these layers")
    p_ex.add_argument("--optimize-vertices", action="store_true", help="Deduplicate vertices to reduce mesh size")
    p_ex.add_argument("--arc-segments", type=int, default=12, help="Segments to approximate arc bulges (higher=smoother)")
    p_ex.add_argument("--arc-max-seglen", type=float, help="Adaptive arc sampling: max segment length (drawing units)")
    p_ex.add_argument("--detect-hard-shapes", action="store_true", help="Detect and skip problematic polylines (self-intersections, zero-length edges, tiny area, duplicates)")
    p_ex.add_argument("--hard-report-csv", help="Write CSV diagnostics for skipped hard shapes")
    p_ex.add_argument("--hard-report-json", help="Write JSON diagnostics for skipped hard shapes")
    p_ex.add_argument("--colorize", action="store_true", help="Apply mesh colors from source entities or layers")
    p_ex.add_argument("--split-by-color", action="store_true", help="Place meshes on per-color layers like <layer>__COLOR_R_G_B")
    p_ex.add_argument("--color-report-csv", help="Write CSV color summary for produced meshes")
    p_ex.add_argument("--color-report-json", help="Write JSON color summary for produced meshes")

    p_conv = sub.add_parser("dxf-to-dwg", help="Convert DXF to DWG via ODA File Converter")
    p_conv.add_argument("--input", required=True, help="Input DXF path")
    p_conv.add_argument("--output", required=True, help="Output DWG path")
    p_conv.add_argument("--version", default="ACAD2018", help="DWG output version (e.g., ACAD2013, ACAD2018, ACAD2024)")

    p_img = sub.add_parser("img-to-3d", help="Convert single image to 3D DXF mesh using ONNX depth model")
    p_img.add_argument("--input", required=True, help="Input image path")
    p_img.add_argument("--output", required=True, help="Output DXF path")
    p_img.add_argument("--scale", type=_positive_float, default=1000.0, help="Depth scale for Z values")
    p_img.add_argument("--model-path", help="Override ONNX model path (else MIDAS_ONNX_PATH env is used)")
    p_img.add_argument("--size", type=int, default=256, help="ONNX input size (MiDaS small default 256)")
    p_img.add_argument("--optimize-vertices", action="store_true", help="Deduplicate vertices to reduce mesh size")

    # Auto-extrude: accepts DXF or DWG input and produces DXF or DWG output
    p_auto = sub.add_parser("auto-extrude", help="Extrude 2D CAD (DXF/DWG) to 3D and write DXF/DWG")
    p_auto.add_argument("--input", required=True, help="Input DXF or DWG path")
    p_auto.add_argument("--output", required=True, help="Output DXF or DWG path")
    p_auto.add_argument("--height", type=_positive_float, default=settings.default_height, help="Extrusion height in drawing units")
    p_auto.add_argument("--layers", nargs="*", help="Only extrude polylines on these layers")
    p_auto.add_argument("--arc-segments", type=int, default=12, help="Segments to approximate arc bulges (higher=smoother)")
    p_auto.add_argument("--arc-max-seglen", type=float, help="Adaptive arc sampling: max segment length (drawing units)")
    p_auto.add_argument("--keep-temp", action="store_true", help="Keep temporary DXF files for debugging")
    p_auto.add_argument("--dwg-version", default="ACAD2018", help="DWG/DXF version for conversions (ACAD2013/2018/2024)")
    p_auto.add_argument("--optimize-vertices", action="store_true", help="Deduplicate vertices to reduce mesh size")
    p_auto.add_argument("--detect-hard-shapes", action="store_true", help="Detect and skip problematic polylines")
    p_auto.add_argument("--hard-report-csv", help="Write CSV diagnostics for skipped hard shapes")
    p_auto.add_argument("--hard-report-json", help="Write JSON diagnostics for skipped hard shapes")
    p_auto.add_argument("--colorize", action="store_true", help="Apply mesh colors from source entities or layers")
    p_auto.add_argument("--split-by-color", action="store_true", help="Place meshes on per-color layers like <layer>__COLOR_R_G_B")
    p_auto.add_argument("--color-report-csv", help="Write CSV color summary for produced meshes")
    p_auto.add_argument("--color-report-json", help="Write JSON color summary for produced meshes")

    # Batch-extrude: process folder of files
    p_batch = sub.add_parser("batch-extrude", help="Batch extrude a folder of DXF/DWG files")
    p_batch.add_argument("--input-dir", required=True, help="Input folder containing DXF/DWG")
    p_batch.add_argument("--output-dir", required=True, help="Output folder for DXF/DWG results")
    p_batch.add_argument("--out-format", choices=["DXF", "DWG"], default="DXF", help="Output file format")
    p_batch.add_argument("--height", type=_positive_float, default=settings.default_height, help="Extrusion height in drawing units")
    p_batch.add_argument("--layers", nargs="*", help="Only extrude polylines on these layers")
    p_batch.add_argument("--arc-segments", type=int, default=12, help="Segments to approximate arc bulges (higher=smoother)")
    p_batch.add_argument("--arc-max-seglen", type=float, help="Adaptive arc sampling: max segment length (drawing units)")
    p_batch.add_argument("--recurse", action="store_true", help="Recurse into subdirectories")
    p_batch.add_argument("--pattern", nargs="+", default=["*.dxf", "*.dwg"], help="File glob patterns to include")
    p_batch.add_argument("--dwg-version", default="ACAD2018", help="DWG/DXF version for conversions (ACAD2013/2018/2024)")
    p_batch.add_argument("--skip-existing", action="store_true", help="Skip if output already exists")
    p_batch.add_argument("--optimize-vertices", action="store_true", help="Deduplicate vertices to reduce mesh size")
    p_batch.add_argument("--report-csv", help="Write a CSV report of processed files")
    p_batch.add_argument("--report-json", help="Write a JSON report of processed files")
    p_batch.add_argument("--jobs", type=int, default=1, help="Parallel workers (use >1 with care)")
    p_batch.add_argument("--detect-hard-shapes", action="store_true", help="Detect and skip problematic polylines")
    p_batch.add_argument("--hard-report-csv", help="Write CSV diagnostics for skipped hard shapes across the batch")
    p_batch.add_argument("--hard-report-json", help="Write JSON diagnostics for skipped hard shapes across the batch")
    p_batch.add_argument("--colorize", action="store_true", help="Apply mesh colors from source entities or layers")
    p_batch.add_argument("--split-by-color", action="store_true", help="Place meshes on per-color layers like <layer>__COLOR_R_G_B")
    p_batch.add_argument("--color-report-csv", help="Write CSV color summary across the batch")
    p_batch.add_argument("--color-report-json", help="Write JSON color summary across the batch")

    # Architectural analysis command
    p_analyze = sub.add_parser("analyze-architectural", help="Analyze architectural drawings (plans, elevations, sections)")
    p_analyze.add_argument("--input", required=True, help="Input DXF file or folder")
    p_analyze.add_argument("--output-dir", required=True, help="Output directory for analysis results")
    p_analyze.add_argument("--recursive", action="store_true", help="Process folder recursively")
    p_analyze.add_argument("--export-json", action="store_true", help="Export full dataset to JSON")
    p_analyze.add_argument("--export-csv", action="store_true", help="Export rooms data to CSV")
    p_analyze.add_argument("--report", action="store_true", help="Generate text reports for each drawing")

    # Neural AI commands
    p_pdf = sub.add_parser("pdf-to-dxf", help="تبدیل PDF به DXF با شبکه عصبی")
    p_pdf.add_argument("--input", required=True, help="مسیر فایل PDF")
    p_pdf.add_argument("--output", required=True, help="مسیر خروجی DXF")
    p_pdf.add_argument("--dpi", type=int, default=300, help="وضوح تبدیل (300-600)")
    p_pdf.add_argument("--confidence", type=float, default=0.5, help="حداقل اطمینان detection (0-1)")
    p_pdf.add_argument("--scale", type=float, default=1.0, help="مقیاس mm per pixel")
    p_pdf.add_argument("--device", default="auto", help="cpu, cuda, or auto")

    p_img_detect = sub.add_parser("image-to-dxf", help="تبدیل عکس نقشه به DXF با AI")
    p_img_detect.add_argument("--input", required=True, help="مسیر عکس نقشه")
    p_img_detect.add_argument("--output", required=True, help="مسیر خروجی DXF")
    p_img_detect.add_argument("--confidence", type=float, default=0.5, help="حداقل اطمینان detection")
    p_img_detect.add_argument("--scale", type=float, default=1.0, help="مقیاس mm per pixel")
    p_img_detect.add_argument("--detect-lines", action="store_true", default=True, help="تشخیص خطوط")
    p_img_detect.add_argument("--detect-circles", action="store_true", default=True, help="تشخیص دایره‌ها")
    p_img_detect.add_argument("--detect-text", action="store_true", default=True, help="تشخیص متن با OCR")
    p_img_detect.add_argument("--device", default="auto", help="cpu, cuda, or auto")

    p_pdf3d = sub.add_parser("pdf-to-3d", help="تبدیل PDF به 3D DXF با هوش مصنوعی")
    p_pdf3d.add_argument("--input", required=True, help="مسیر فایل PDF")
    p_pdf3d.add_argument("--output", required=True, help="مسیر خروجی 3D DXF")
    p_pdf3d.add_argument("--dpi", type=int, default=300, help="وضوح تبدیل")
    p_pdf3d.add_argument("--intelligent-height", action="store_true", help="استفاده از ML برای پیش‌بینی ارتفاع")
    p_pdf3d.add_argument("--device", default="auto", help="cpu, cuda, or auto")

    # Training commands
    p_build_ds = sub.add_parser("build-dataset", help="ساخت Dataset آموزشی از فایل‌های DXF")
    p_build_ds.add_argument("--input-dir", required=True, help="پوشه حاوی فایل‌های DXF")
    p_build_ds.add_argument("--output-dir", required=True, help="پوشه خروجی Dataset")
    p_build_ds.add_argument("--image-size", type=int, nargs=2, default=[1024, 1024], help="اندازه تصویر (width height)")
    p_build_ds.add_argument("--format", choices=["coco", "yolo", "both"], default="coco", help="فرمت خروجی")
    p_build_ds.add_argument("--visualize", action="store_true", help="ذخیره تصاویر بررسی annotation")
    p_build_ds.add_argument("--recurse", action="store_true", help="جستجوی زیرپوشه‌ها")

    p_train = sub.add_parser("train", help="آموزش مدل تشخیص CAD")
    p_train.add_argument("--dataset-dir", required=True, help="پوشه Dataset (COCO format)")
    p_train.add_argument("--output-dir", required=True, help="پوشه خروجی checkpoints")
    p_train.add_argument("--epochs", type=int, default=50, help="تعداد epochs")
    p_train.add_argument("--batch-size", type=int, default=4, help="اندازه batch")
    p_train.add_argument("--lr", type=float, default=0.001, help="learning rate")
    p_train.add_argument("--device", default="cuda", help="cpu یا cuda")
    p_train.add_argument("--workers", type=int, default=4, help="تعداد data loader workers")
    p_train.add_argument("--optimizer", choices=["sgd", "adam"], default="sgd", help="نوع optimizer")
    p_train.add_argument("--resume", help="مسیر checkpoint برای ادامه آموزش")
    p_train.add_argument("--pretrained", action="store_true", help="استفاده از pre-trained weights")

    # Optimization commands
    p_optimize = sub.add_parser("optimize-model", help="بهینه‌سازی مدل برای استقرار (ONNX/TensorRT/Quantization)")
    p_optimize.add_argument("--model", required=True, help="مسیر مدل PyTorch (.pth)")
    p_optimize.add_argument("--output-dir", required=True, help="پوشه خروجی مدل‌های بهینه‌شده")
    p_optimize.add_argument("--formats", nargs="+", choices=["onnx", "tensorrt", "quantized"], default=["onnx"], help="فرمت‌های خروجی")
    p_optimize.add_argument("--input-size", type=int, nargs=2, default=[1024, 1024], help="اندازه ورودی (height width)")
    p_optimize.add_argument("--benchmark", action="store_true", help="اجرای benchmark")
    p_optimize.add_argument("--device", default="cuda", help="cpu یا cuda")

    p_benchmark = sub.add_parser("benchmark", help="ارزیابی دقت و سرعت مدل")
    p_benchmark.add_argument("--model", required=True, help="مسیر مدل (.pth, .onnx, .trt)")
    p_benchmark.add_argument("--dataset-dir", required=True, help="پوشه Dataset (COCO format)")
    p_benchmark.add_argument("--output-dir", required=True, help="پوشه خروجی نتایج")
    p_benchmark.add_argument("--model-format", choices=["pytorch", "onnx", "tensorrt"], default="pytorch", help="فرمت مدل")
    p_benchmark.add_argument("--batch-size", type=int, default=1, help="اندازه batch")
    p_benchmark.add_argument("--max-samples", type=int, help="حداکثر تعداد نمونه برای ارزیابی")
    p_benchmark.add_argument("--device", default="cuda", help="cpu یا cuda")

    # Universal convert: accept image/PDF/DXF/DWG in, output DXF or DWG
    p_uni = sub.add_parser("universal-convert", help="ورودی تصویر/PDF/DXF/DWG و خروجی DXF/DWG")
    p_uni.add_argument("--input", required=True, help="مسیر ورودی (تصویر/PDF/DXF/DWG)")
    p_uni.add_argument("--output", required=True, help="مسیر خروجی (DXF/DWG)")
    p_uni.add_argument("--dpi", type=int, default=300, help="DPI برای PDF")
    p_uni.add_argument("--confidence", type=float, default=0.5, help="حداقل اطمینان شبکه")
    p_uni.add_argument("--scale", type=float, default=1.0, help="مقیاس mm/pixel برای تصویر")
    p_uni.add_argument("--device", default="auto", help="cpu, cuda, or auto")
    p_uni.add_argument("--dwg-version", default="ACAD2018", help="ورژن خروجی DWG")
    # 3D extrusion options
    p_uni.add_argument("--to-3d", action="store_true", help="پس از بردارسازی/ورود، اکستروژن سه‌بعدی انجام شود")
    p_uni.add_argument("--height", type=_positive_float, default=settings.default_height, help="ارتفاع اکستروژن")
    p_uni.add_argument("--layers", nargs="*", help="فقط لایه‌های مشخص اکستروژن شوند")
    p_uni.add_argument("--arc-segments", type=int, default=12, help="تعداد قطعات تقریب قوس")
    p_uni.add_argument("--arc-max-seglen", type=float, help="طول بیشینه قطعه قوس برای نمونه‌برداری تطبیقی")
    p_uni.add_argument("--optimize-vertices", action="store_true", help="حذف رئوس تکراری برای کاهش حجم")
    p_uni.add_argument("--detect-hard-shapes", action="store_true", help="شناسایی پلی‌لاین‌های مشکل‌دار و رد کردن آنها")

    args = parser.parse_args(argv)

    if args.cmd == "dxf-extrude":
        hard_rows = [] if (args.hard_report_csv or args.hard_report_json) else None
        color_rows = [] if (getattr(args, "color_report_csv", None) or getattr(args, "color_report_json", None)) else None
        extrude_dxf_closed_polylines(
            args.input,
            args.output,
            height=args.height,
            layers=args.layers,
            arc_segments=args.arc_segments,
            arc_max_seglen=args.arc_max_seglen,
            optimize=args.optimize_vertices,
            detect_hard_shapes=args.detect_hard_shapes,
            hard_shapes_collector=hard_rows,
            colorize=getattr(args, "colorize", False),
            split_by_color=getattr(args, "split_by_color", False),
            color_stats_collector=color_rows,
        )
        if args.hard_report_csv and hard_rows:
            import csv
            p = Path(args.hard_report_csv)
            p.parent.mkdir(parents=True, exist_ok=True)
            with p.open("w", newline="", encoding="utf-8") as fp:
                w = csv.writer(fp)
                w.writerow(["layer", "handle", "issues", "vertex_count"]) 
                for r in hard_rows:
                    w.writerow([r.get("layer", ""), r.get("handle", ""), r.get("issues", ""), r.get("vertex_count", 0)])
        if getattr(args, "hard_report_json", None) and hard_rows:
            import json
            p = Path(args.hard_report_json)
            p.parent.mkdir(parents=True, exist_ok=True)
            with p.open("w", encoding="utf-8") as fp:
                json.dump(hard_rows, fp, ensure_ascii=False, indent=2)
        # Color reports
        if getattr(args, "color_report_csv", None) and color_rows:
            import csv
            p = Path(args.color_report_csv)
            p.parent.mkdir(parents=True, exist_ok=True)
            agg = {}
            for r in color_rows:
                key = (r.get("target_layer", ""), int(r.get("r", 0)), int(r.get("g", 0)), int(r.get("b", 0)))
                agg[key] = agg.get(key, 0) + 1
            with p.open("w", newline="", encoding="utf-8") as fp:
                w = csv.writer(fp)
                w.writerow(["target_layer", "r", "g", "b", "count"]) 
                for (layer, rr, gg, bb), cnt in sorted(agg.items()):
                    w.writerow([layer, rr, gg, bb, cnt])
        if getattr(args, "color_report_json", None) and color_rows:
            import json
            p = Path(args.color_report_json)
            p.parent.mkdir(parents=True, exist_ok=True)
            with p.open("w", encoding="utf-8") as fp:
                json.dump(color_rows, fp, ensure_ascii=False, indent=2)
        print(f"Wrote 3D DXF: {args.output}")
    elif args.cmd == "dxf-to-dwg":
        convert_dxf_to_dwg(args.input, args.output, out_version=args.version)
        print(f"Wrote DWG: {args.output}")
    elif args.cmd == "img-to-3d":
        # Lazy import to avoid importing heavy deps when not needed
        from .image_to_depth import image_to_3d_dxf
        image_to_3d_dxf(args.input, args.output, scale=args.scale, model_path=args.model_path, size=args.size, optimize=args.optimize_vertices)
        print(f"Wrote 3D DXF: {args.output}")
    elif args.cmd == "auto-extrude":
        from .dwg_io import convert_dwg_to_dxf
        inp = Path(args.input)
        outp = Path(args.output)
        tmp_in_dxf: Path | None = None
        tmp_out_dxf: Path | None = None

        hard_rows = [] if (getattr(args, "hard_report_csv", None) or getattr(args, "hard_report_json", None)) else None
        color_rows = [] if (getattr(args, "color_report_csv", None) or getattr(args, "color_report_json", None)) else None
        try:
            # Ensure input DXF
            if inp.suffix.lower() == ".dwg":
                tmp_in_dxf = outp.with_suffix("").with_name(outp.stem + "_tmp_in.dxf")
                convert_dwg_to_dxf(str(inp), str(tmp_in_dxf), out_version=args.dwg_version)
                in_dxf = tmp_in_dxf
            elif inp.suffix.lower() == ".dxf":
                in_dxf = inp
            else:
                raise RuntimeError("Unsupported input extension (use .dxf or .dwg)")

            # Always extrude to a DXF first
            if outp.suffix.lower() == ".dwg":
                tmp_out_dxf = outp.with_suffix("").with_name(outp.stem + "_tmp_out.dxf")
                extrude_dxf_closed_polylines(
                    str(in_dxf),
                    str(tmp_out_dxf),
                    height=args.height,
                    layers=args.layers,
                    arc_segments=args.arc_segments,
                    arc_max_seglen=args.arc_max_seglen,
                    optimize=args.optimize_vertices,
                    detect_hard_shapes=args.detect_hard_shapes,
                    hard_shapes_collector=hard_rows,
                    colorize=getattr(args, "colorize", False),
                    split_by_color=getattr(args, "split_by_color", False),
                    color_stats_collector=color_rows,
                )
                convert_dxf_to_dwg(str(tmp_out_dxf), str(outp), out_version=args.dwg_version)
            elif outp.suffix.lower() == ".dxf":
                extrude_dxf_closed_polylines(
                    str(in_dxf),
                    str(outp),
                    height=args.height,
                    layers=args.layers,
                    arc_segments=args.arc_segments,
                    arc_max_seglen=args.arc_max_seglen,
                    optimize=args.optimize_vertices,
                    detect_hard_shapes=args.detect_hard_shapes,
                    hard_shapes_collector=hard_rows,
                    colorize=getattr(args, "colorize", False),
                    split_by_color=getattr(args, "split_by_color", False),
                    color_stats_collector=color_rows,
                )
            else:
                raise RuntimeError("Unsupported output extension (use .dxf or .dwg)")

            print(f"Wrote: {args.output}")
        finally:
            if not getattr(args, "keep_temp", False):
                try:
                    if tmp_in_dxf and tmp_in_dxf.exists():
                        tmp_in_dxf.unlink()
                    if tmp_out_dxf and tmp_out_dxf.exists():
                        tmp_out_dxf.unlink()
                except Exception:
                    pass
            if args.hard_report_csv and hard_rows:
                import csv
                p = Path(args.hard_report_csv)
                p.parent.mkdir(parents=True, exist_ok=True)
                with p.open("w", newline="", encoding="utf-8") as fp:
                    w = csv.writer(fp)
                    w.writerow(["layer", "handle", "issues", "vertex_count"]) 
                    for r in hard_rows:
                        w.writerow([r.get("layer", ""), r.get("handle", ""), r.get("issues", ""), r.get("vertex_count", 0)])
            if getattr(args, "hard_report_json", None) and hard_rows:
                import json
                p = Path(args.hard_report_json)
                p.parent.mkdir(parents=True, exist_ok=True)
                with p.open("w", encoding="utf-8") as fp:
                    json.dump(hard_rows, fp, ensure_ascii=False, indent=2)
            # Color reports
            if getattr(args, "color_report_csv", None) and color_rows:
                import csv
                p = Path(args.color_report_csv)
                p.parent.mkdir(parents=True, exist_ok=True)
                agg = {}
                for r in color_rows:
                    key = (r.get("target_layer", ""), int(r.get("r", 0)), int(r.get("g", 0)), int(r.get("b", 0)))
                    agg[key] = agg.get(key, 0) + 1
                with p.open("w", newline="", encoding="utf-8") as fp:
                    w = csv.writer(fp)
                    w.writerow(["target_layer", "r", "g", "b", "count"]) 
                    for (layer, rr, gg, bb), cnt in sorted(agg.items()):
                        w.writerow([layer, rr, gg, bb, cnt])
            if getattr(args, "color_report_json", None) and color_rows:
                import json
                p = Path(args.color_report_json)
                p.parent.mkdir(parents=True, exist_ok=True)
                with p.open("w", encoding="utf-8") as fp:
                    json.dump(color_rows, fp, ensure_ascii=False, indent=2)
    elif args.cmd == "batch-extrude":
        from .dwg_io import convert_dwg_to_dxf, convert_dxf_to_dwg
        import time, csv, json, traceback, uuid
        from concurrent.futures import ThreadPoolExecutor, as_completed
        in_root = Path(args.input_dir)
        out_root = Path(args.output_dir)
        out_root.mkdir(parents=True, exist_ok=True)

        # Collect files
        files: list[Path] = []
        globs = args.pattern
        if args.recurse:
            for pat in globs:
                files.extend(in_root.rglob(pat))
        else:
            for pat in globs:
                files.extend(in_root.glob(pat))

        processed = 0
        report_rows: list[list[str]] = []
        hard_rows: list[list[str]] = [] if args.hard_report_csv else []
        hard_json: list[dict] = [] if getattr(args, "hard_report_json", None) else []
        failed: list[tuple[Path, str]] = []
        color_rows_all: list[dict] = [] if (getattr(args, "color_report_csv", None) or getattr(args, "color_report_json", None)) else []
        def process_one(f: Path):
            rec = {"file_path": str(f), "output_path": "", "status": "", "message": "", "duration_sec": 0.0}
            try:
                t0 = time.monotonic()
                rel = f.relative_to(in_root)
                target_stem = rel.with_suffix("").name
                out_dir = (out_root / rel.parent)
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / f"{target_stem}.{args.out_format.lower()}"
                if args.skip_existing and out_path.exists():
                    rec.update({"output_path": str(out_path), "status": "skipped", "message": "exists"})
                    return rec

                # Ensure DXF input for extrusion
                tmp_in_dxf: Path | None = None
                tmp_out_dxf: Path | None = None
                try:
                    if f.suffix.lower() == ".dwg":
                        tmp_in_dxf = out_dir / f"{target_stem}.{uuid.uuid4().hex}.tmp_in.dxf"
                        convert_dwg_to_dxf(str(f), str(tmp_in_dxf), out_version=args.dwg_version)
                        in_dxf = tmp_in_dxf
                    else:
                        in_dxf = f

                    if args.out_format == "DWG":
                        tmp_out_dxf = out_dir / f"{target_stem}.{uuid.uuid4().hex}.tmp_out.dxf"
                        local_hard = [] if (args.hard_report_csv or getattr(args, "hard_report_json", None)) else None
                        local_colors = [] if (getattr(args, "color_report_csv", None) or getattr(args, "color_report_json", None)) else None
                        extrude_dxf_closed_polylines(
                            str(in_dxf),
                            str(tmp_out_dxf),
                            height=args.height,
                            layers=args.layers,
                            arc_segments=args.arc_segments,
                            arc_max_seglen=args.arc_max_seglen,
                            optimize=args.optimize_vertices,
                            detect_hard_shapes=args.detect_hard_shapes,
                            hard_shapes_collector=local_hard,
                            colorize=getattr(args, "colorize", False),
                            split_by_color=getattr(args, "split_by_color", False),
                            color_stats_collector=local_colors,
                        )
                        convert_dxf_to_dwg(str(tmp_out_dxf), str(out_path), out_version=args.dwg_version)
                    else:
                        local_hard = [] if (args.hard_report_csv or getattr(args, "hard_report_json", None)) else None
                        local_colors = [] if (getattr(args, "color_report_csv", None) or getattr(args, "color_report_json", None)) else None
                        extrude_dxf_closed_polylines(
                            str(in_dxf),
                            str(out_path),
                            height=args.height,
                            layers=args.layers,
                            arc_segments=args.arc_segments,
                            arc_max_seglen=args.arc_max_seglen,
                            optimize=args.optimize_vertices,
                            detect_hard_shapes=args.detect_hard_shapes,
                            hard_shapes_collector=local_hard,
                            colorize=getattr(args, "colorize", False),
                            split_by_color=getattr(args, "split_by_color", False),
                            color_stats_collector=local_colors,
                        )

                    dt = time.monotonic() - t0
                    rec.update({"output_path": str(out_path), "status": "ok", "duration_sec": round(dt, 3)})
                    if args.hard_report_csv and local_hard:
                        for r in local_hard:
                            hard_rows.append([str(f), r.get("layer", ""), r.get("handle", ""), r.get("issues", ""), str(r.get("vertex_count", 0))])
                    if getattr(args, "hard_report_json", None) and local_hard:
                        for r in local_hard:
                            rr = dict(r)
                            rr["file_path"] = str(f)
                            hard_json.append(rr)
                    if (getattr(args, "color_report_csv", None) or getattr(args, "color_report_json", None)) and local_colors:
                        for r in local_colors:
                            rr = dict(r)
                            rr["file_path"] = str(f)
                            color_rows_all.append(rr)
                    return rec
                finally:
                    if tmp_in_dxf and tmp_in_dxf.exists():
                        try:
                            tmp_in_dxf.unlink()
                        except Exception:
                            pass
                    if tmp_out_dxf and tmp_out_dxf.exists():
                        try:
                            tmp_out_dxf.unlink()
                        except Exception:
                            pass
            except Exception as ex:
                tb = traceback.format_exc(limit=1).strip().replace("\n", " ")
                rec.update({"status": "error", "message": f"{ex} | {tb}"})
                return rec

        if args.jobs > 1 and files:
            with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as ex:
                futs = {ex.submit(process_one, f): f for f in files}
                for fut in as_completed(futs):
                    rec = fut.result()
                    report_rows.append([rec["file_path"], rec["output_path"], rec["status"], rec["message"], str(rec["duration_sec"])])
                    if rec["status"] == "ok":
                        processed += 1
                    elif rec["status"] == "error":
                        failed.append((Path(rec["file_path"]), rec["message"]))
        else:
            for f in files:
                rec = process_one(f)
                report_rows.append([rec["file_path"], rec["output_path"], rec["status"], rec["message"], str(rec["duration_sec"])])
                if rec["status"] == "ok":
                    processed += 1
                elif rec["status"] == "error":
                    failed.append((Path(rec["file_path"]), rec["message"]))

        print(f"Batch done. Processed: {processed}, Failed: {len(failed)}")
        if failed:
            for f, msg in failed[:10]:
                print(f" - {f}: {msg}")
        if args.report_csv:
            csv_path = Path(args.report_csv)
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            with csv_path.open("w", newline="", encoding="utf-8") as fp:
                w = csv.writer(fp)
                w.writerow(["file_path", "output_path", "status", "message", "duration_sec"])
                w.writerows(report_rows)
            print(f"Report written: {csv_path}")
        if args.hard_report_csv and hard_rows:
            csv_path = Path(args.hard_report_csv)
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            with csv_path.open("w", newline="", encoding="utf-8") as fp:
                w = csv.writer(fp)
                w.writerow(["file_path", "layer", "handle", "issues", "vertex_count"])
                w.writerows(hard_rows)
            print(f"Hard-shapes report written: {csv_path}")
        if getattr(args, "hard_report_json", None) and hard_json:
            json_path = Path(args.hard_report_json)
            json_path.parent.mkdir(parents=True, exist_ok=True)
            with json_path.open("w", encoding="utf-8") as fp:
                json.dump(hard_json, fp, ensure_ascii=False, indent=2)
            print(f"Hard-shapes report written: {json_path}")
        if args.report_json:
            json_path = Path(args.report_json)
            json_path.parent.mkdir(parents=True, exist_ok=True)
            records = [
                {"file_path": r[0], "output_path": r[1], "status": r[2], "message": r[3], "duration_sec": float(r[4]) if r[4] else 0.0}
                for r in report_rows
            ]
            with json_path.open("w", encoding="utf-8") as fp:
                json.dump(records, fp, ensure_ascii=False, indent=2)
            print(f"Report written: {json_path}")
        # Batch color reports
        if getattr(args, "color_report_csv", None) and color_rows_all:
            csv_path = Path(args.color_report_csv)
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            agg = {}
            for r in color_rows_all:
                key = (r.get("file_path", ""), r.get("target_layer", ""), int(r.get("r", 0)), int(r.get("g", 0)), int(r.get("b", 0)))
                agg[key] = agg.get(key, 0) + 1
            with csv_path.open("w", newline="", encoding="utf-8") as fp:
                w = csv.writer(fp)
                w.writerow(["file_path", "target_layer", "r", "g", "b", "count"]) 
                for (fpn, layer, rr, gg, bb), cnt in sorted(agg.items()):
                    w.writerow([fpn, layer, rr, gg, bb, cnt])
            print(f"Color report written: {csv_path}")
        if getattr(args, "color_report_json", None) and color_rows_all:
            json_path = Path(args.color_report_json)
            json_path.parent.mkdir(parents=True, exist_ok=True)
            with json_path.open("w", encoding="utf-8") as fp:
                json.dump(color_rows_all, fp, ensure_ascii=False, indent=2)
            print(f"Color report written: {json_path}")
    elif args.cmd == "analyze-architectural":
        from .architectural_analyzer import ArchitecturalAnalyzer, generate_analysis_report
        from .dataset_builder import ArchitecturalDatasetBuilder
        
        input_path = Path(args.input)
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if input_path.is_file():
            # تحلیل یک فایل
            print(f"تحلیل نقشه: {input_path.name}")
            analyzer = ArchitecturalAnalyzer(str(input_path))
            analysis = analyzer.analyze()
            
            # گزارش متنی
            if args.report:
                report = generate_analysis_report(analysis)
                print(report)
                report_path = output_dir / f"{input_path.stem}_analysis.txt"
                report_path.write_text(report, encoding="utf-8")
                print(f"گزارش ذخیره شد: {report_path}")
            
            # JSON خروجی
            if args.export_json:
                import json
                from dataclasses import asdict
                json_path = output_dir / f"{input_path.stem}_analysis.json"
                # ساده‌سازی برای JSON serialization
                simple_data = {
                    "drawing_type": analysis.drawing_type.value,
                    "total_area": analysis.total_area,
                    "num_rooms": len(analysis.rooms),
                    "num_walls": len(analysis.walls),
                    "rooms": [
                        {
                            "name": r.name,
                            "type": r.space_type.value,
                            "area": r.area,
                            "width": r.width,
                            "length": r.length,
                        }
                        for r in analysis.rooms
                    ],
                }
                with json_path.open("w", encoding="utf-8") as f:
                    json.dump(simple_data, f, ensure_ascii=False, indent=2)
                print(f"JSON ذخیره شد: {json_path}")
        
        elif input_path.is_dir():
            # پردازش پوشه
            print(f"پردازش پوشه: {input_path}")
            builder = ArchitecturalDatasetBuilder(str(output_dir))
            builder.process_folder(str(input_path), recursive=args.recursive)
            
            # گزارش‌های متنی
            if args.report:
                for i, analysis in enumerate(builder.analyses, 1):
                    report = generate_analysis_report(analysis)
                    file_name = analysis.metadata.get("file_path", f"drawing_{i}")
                    safe_name = Path(file_name).stem
                    report_path = output_dir / f"{safe_name}_analysis.txt"
                    report_path.write_text(report, encoding="utf-8")
            
            # Export dataset
            if args.export_json:
                builder.export_to_json()
            
            if args.export_csv:
                builder.export_rooms_to_csv()
            
            # آمار و خلاصه
            builder.export_statistics()
            summary = builder.generate_summary_report()
            print("\n" + summary)
            
            summary_path = output_dir / "dataset_summary.txt"
            summary_path.write_text(summary, encoding="utf-8")
            print(f"\nگزارش خلاصه: {summary_path}")
        
        else:
            print(f"❌ مسیر نامعتبر: {input_path}")

    elif args.cmd == "pdf-to-dxf":
        # تبدیل PDF به DXF با شبکه عصبی
        try:
            from .neural_cad_detector import NeuralCADDetector
            from .pdf_processor import PDFToImageConverter, CADPipeline
            
            print("🚀 Neural PDF to DXF Pipeline")
            print(f"   Input: {args.input}")
            print(f"   Output: {args.output}")
            print(f"   DPI: {args.dpi}, Confidence: {args.confidence}, Device: {args.device}")
            
            # ساخت pipeline
            detector = NeuralCADDetector(device=args.device)
            pdf_converter = PDFToImageConverter(dpi=args.dpi)
            pipeline = CADPipeline(
                neural_detector=detector,
                pdf_converter=pdf_converter
            )
            
            # پردازش
            pipeline.process_pdf_to_dxf(
                args.input,
                args.output,
                confidence_threshold=args.confidence,
                scale_mm_per_pixel=args.scale
            )
            
            print(f"\n✅ Success! DXF saved: {args.output}")
        except ImportError as e:
            print(f"❌ Error: {e}")
            print("Install neural dependencies: pip install -r requirements-neural.txt")
        except Exception as e:
            print(f"❌ Error processing PDF: {e}")

    elif args.cmd == "image-to-dxf":
        # تبدیل عکس به DXF با AI
        try:
            from .neural_cad_detector import NeuralCADDetector
            
            print("🚀 Neural Image to DXF")
            print(f"   Input: {args.input}")
            print(f"   Output: {args.output}")
            
            detector = NeuralCADDetector(device=args.device)
            
            # Vectorization
            vectorized = detector.vectorize_drawing(
                args.input,
                scale_mm_per_pixel=args.scale,
                detect_lines=args.detect_lines,
                detect_circles=args.detect_circles,
                detect_text=args.detect_text
            )
            
            # تبدیل به DXF
            detector.convert_to_dxf(vectorized, args.output)
            
            print(f"\n✅ Success! DXF saved: {args.output}")
        except ImportError as e:
            print(f"❌ Error: {e}")
            print("Install neural dependencies: pip install -r requirements-neural.txt")
        except Exception as e:
            print(f"❌ Error processing image: {e}")

    elif args.cmd == "pdf-to-3d":
        # تبدیل PDF به 3D DXF با هوش مصنوعی
        try:
            from .neural_cad_detector import NeuralCADDetector, ImageTo3DExtruder
            from .pdf_processor import PDFToImageConverter, CADPipeline
            
            print("🚀 Neural PDF to 3D DXF Pipeline")
            print(f"   Input: {args.input}")
            print(f"   Output: {args.output}")
            
            detector = NeuralCADDetector(device=args.device)
            pdf_converter = PDFToImageConverter(dpi=args.dpi)
            extruder = ImageTo3DExtruder()
            
            pipeline = CADPipeline(
                neural_detector=detector,
                pdf_converter=pdf_converter,
                extruder_3d=extruder
            )
            
            pipeline.process_pdf_to_3d(
                args.input,
                args.output,
                intelligent_height=args.intelligent_height
            )
            
            print(f"\n✅ Success! 3D DXF saved: {args.output}")
        except ImportError as e:
            print(f"❌ Error: {e}")
            print("Install neural dependencies: pip install -r requirements-neural.txt")
        except Exception as e:
            print(f"❌ Error processing PDF to 3D: {e}")

    elif args.cmd == "build-dataset":
        try:
            from .training_dataset_builder import CADDatasetBuilder
            
            print("📦 Building CAD Training Dataset")
            print(f"   Input DXF: {args.input_dir}")
            print(f"   Output: {args.output_dir}")
            print(f"   Format: {args.format}")
            
            builder = CADDatasetBuilder(output_dir=args.output_dir)
            
            input_path = Path(args.input_dir)
            if args.recurse:
                dxf_files = list(input_path.rglob("*.dxf"))
            else:
                dxf_files = list(input_path.glob("*.dxf"))
            
            print(f"\n🔍 Found {len(dxf_files)} DXF files")
            
            for i, dxf_file in enumerate(dxf_files, 1):
                print(f"   [{i}/{len(dxf_files)}] Processing {dxf_file.name}...", end=" ")
                try:
                    builder.add_dxf_to_dataset(
                        str(dxf_file),
                        image_size=tuple(args.image_size)
                    )
                    print("✅")
                except Exception as e:
                    print(f"❌ {e}")
            
            print(f"\n💾 Exporting annotations...")
            if args.format in ["coco", "both"]:
                builder.export_coco_format()
                print("   ✅ COCO format exported")
            if args.format in ["yolo", "both"]:
                builder.export_yolo_format()
                print("   ✅ YOLO format exported")
            
            if args.visualize:
                print(f"\n🎨 Generating visualization...")
                builder.visualize_annotations()
                print("   ✅ Visualizations saved")
            
            print(f"\n✅ Dataset ready at: {args.output_dir}")
            print(f"   Total images: {len(builder.images)}")
            print(f"   Total annotations: {len(builder.annotations)}")
            
        except ImportError as e:
            print(f"❌ Error: {e}")
            print("Install dependencies: pip install -r requirements-neural.txt")
        except Exception as e:
            print(f"❌ Error building dataset: {e}")
            import traceback
            traceback.print_exc()

    elif args.cmd == "train":
        try:
            from .training_pipeline import CADDetectionTrainer
            import torch
            
            print("🎓 Training CAD Detection Model")
            print(f"   Dataset: {args.dataset_dir}")
            print(f"   Output: {args.output_dir}")
            print(f"   Epochs: {args.epochs}")
            print(f"   Batch Size: {args.batch_size}")
            print(f"   Device: {args.device}")
            
            device = torch.device(args.device if torch.cuda.is_available() else "cpu")
            print(f"   Using device: {device}")
            
            trainer = CADDetectionTrainer(
                data_dir=args.dataset_dir,
                output_dir=args.output_dir,
                batch_size=args.batch_size,
                num_workers=args.workers,
                device=device,
                pretrained=args.pretrained
            )
            
            trainer.setup_optimizer(
                optimizer_type=args.optimizer,
                learning_rate=args.lr
            )
            
            if args.resume:
                print(f"\n📂 Loading checkpoint: {args.resume}")
                trainer.load_checkpoint(args.resume)
            
            print(f"\n🚀 Starting training...")
            trainer.train(num_epochs=args.epochs)
            
            print(f"\n✅ Training complete!")
            print(f"   Best model: {Path(args.output_dir) / 'best_model.pth'}")
            print(f"   Checkpoints: {args.output_dir}")
            
        except ImportError as e:
            print(f"❌ Error: {e}")
            print("Install PyTorch: pip install torch torchvision")
        except Exception as e:
            print(f"❌ Error during training: {e}")
            import traceback
            traceback.print_exc()

    elif args.cmd == "optimize-model":
        try:
            from .model_optimizer import ModelOptimizer
            from .training_pipeline import CADDetectionTrainer
            import torch
            
            print("⚡ Model Optimization Pipeline")
            print(f"   Model: {args.model}")
            print(f"   Output: {args.output_dir}")
            print(f"   Formats: {', '.join(args.formats)}")
            
            device = torch.device(args.device if torch.cuda.is_available() else "cpu")
            print(f"   Device: {device}")
            
            # Load model
            print(f"\n📂 Loading model...")
            trainer = CADDetectionTrainer(
                data_dir=".",  # Not used for optimization
                output_dir=".",
                device=device
            )
            
            checkpoint = torch.load(args.model, map_location=device)
            trainer.model.load_state_dict(checkpoint['model_state_dict'])
            trainer.model.eval()
            
            # Create optimizer
            optimizer = ModelOptimizer(device=str(device))
            
            # Run optimization pipeline
            input_shape = (1, 3, args.input_size[0], args.input_size[1])
            
            results = optimizer.optimize_full_pipeline(
                model=trainer.model,
                output_dir=args.output_dir,
                input_shape=input_shape,
                formats=args.formats,
                benchmark=args.benchmark
            )
            
            print(f"\n✅ Optimization complete!")
            print(f"   Output directory: {args.output_dir}")
            
            if args.benchmark:
                from .model_optimizer import compare_models
                compare_models(results)
            
        except ImportError as e:
            print(f"❌ Error: {e}")
            print("Install dependencies: pip install torch onnx onnxruntime")
        except Exception as e:
            print(f"❌ Error during optimization: {e}")
            import traceback
            traceback.print_exc()

    elif args.cmd == "benchmark":
        try:
            from .benchmark_suite import DetectionBenchmark
            from .training_pipeline import CADDataset, CADDetectionTrainer
            import torch
            from torch.utils.data import DataLoader
            
            print("📊 Model Benchmarking")
            print(f"   Model: {args.model}")
            print(f"   Dataset: {args.dataset_dir}")
            print(f"   Format: {args.model_format}")
            
            device = torch.device(args.device if torch.cuda.is_available() else "cpu")
            print(f"   Device: {device}")
            
            # Load dataset
            print(f"\n📂 Loading dataset...")
            dataset = CADDataset(
                root_dir=args.dataset_dir,
                annotation_file=str(Path(args.dataset_dir) / "annotations.json")
            )
            
            dataloader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=0,
                collate_fn=lambda x: tuple(zip(*x))
            )
            
            print(f"   Dataset size: {len(dataset)} images")
            
            # Load model
            print(f"\n📂 Loading model...")
            if args.model_format == "pytorch":
                trainer = CADDetectionTrainer(
                    data_dir=".",
                    output_dir=".",
                    device=device
                )
                checkpoint = torch.load(args.model, map_location=device)
                trainer.model.load_state_dict(checkpoint['model_state_dict'])
                trainer.model.eval()
                model = trainer.model
            else:
                # ONNX/TensorRT support
                model = args.model
            
            # Run benchmark
            print(f"\n🚀 Running benchmark...")
            benchmark = DetectionBenchmark(model, device=str(device))
            
            overall_metrics, category_metrics = benchmark.evaluate_dataset(
                dataloader,
                max_samples=args.max_samples
            )
            
            # Print report
            benchmark.print_detailed_report(overall_metrics, category_metrics)
            
            # Save results
            output_path = Path(args.output_dir) / "benchmark_results.json"
            benchmark.save_results(str(output_path), overall_metrics, category_metrics)
            
            print(f"\n✅ Benchmark complete!")
            print(f"   Results saved: {output_path}")
            
        except ImportError as e:
            print(f"❌ Error: {e}")
            print("Install PyTorch: pip install torch torchvision")
        except Exception as e:
            print(f"❌ Error during benchmark: {e}")
            import traceback
            traceback.print_exc()

    elif args.cmd == "universal-convert":
        from shutil import copyfile
        from .dwg_io import convert_dxf_to_dwg, convert_dwg_to_dxf
        inp = Path(args.input)
        outp = Path(args.output)
        in_ext = inp.suffix.lower()
        out_ext = outp.suffix.lower()

        def ensure_parent(p: Path):
            p.parent.mkdir(parents=True, exist_ok=True)

        def extrude_to_destination(in_dxf_path: Path) -> None:
            # Extrude DXF and write to desired output (DXF/DWG)
            ensure_parent(outp)
            if out_ext == ".dwg":
                tmp_out_dxf = outp.with_suffix("").with_name(outp.stem + "_tmp_out3d.dxf")
                extrude_dxf_closed_polylines(
                    str(in_dxf_path),
                    str(tmp_out_dxf),
                    height=args.height,
                    layers=args.layers,
                    arc_segments=args.arc_segments,
                    arc_max_seglen=args.arc_max_seglen,
                    optimize=args.optimize_vertices,
                    detect_hard_shapes=args.detect_hard_shapes,
                    hard_shapes_collector=None,
                    colorize=False,
                    split_by_color=False,
                    color_stats_collector=None,
                )
                convert_dxf_to_dwg(str(tmp_out_dxf), str(outp), out_version=args.dwg_version)
                try:
                    tmp_out_dxf.unlink()
                except Exception:
                    pass
            elif out_ext == ".dxf":
                extrude_dxf_closed_polylines(
                    str(in_dxf_path),
                    str(outp),
                    height=args.height,
                    layers=args.layers,
                    arc_segments=args.arc_segments,
                    arc_max_seglen=args.arc_max_seglen,
                    optimize=args.optimize_vertices,
                    detect_hard_shapes=args.detect_hard_shapes,
                    hard_shapes_collector=None,
                    colorize=False,
                    split_by_color=False,
                    color_stats_collector=None,
                )
            else:
                raise RuntimeError("خروجی باید DXF یا DWG باشد")

        try:
            ensure_parent(outp)
            # Image inputs
            if in_ext in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]:
                try:
                    from .neural_cad_detector import NeuralCADDetector
                except ImportError as e:
                    raise RuntimeError("NeuralCADDetector not available. Install requirements-neural.txt") from e
                detector = NeuralCADDetector(device=args.device)
                vectorized = detector.vectorize_drawing(inp, scale_mm_per_pixel=args.scale)
                tmp_vec_dxf = outp.with_suffix("").with_name(outp.stem + "_tmp_vec2d.dxf")
                detector.convert_to_dxf(vectorized, str(tmp_vec_dxf))
                if args.to_3d:
                    extrude_to_destination(tmp_vec_dxf)
                else:
                    if out_ext == ".dwg":
                        convert_dxf_to_dwg(str(tmp_vec_dxf), str(outp), out_version=args.dwg_version)
                    else:
                        ensure_parent(outp)
                        copyfile(str(tmp_vec_dxf), str(outp))
                try:
                    tmp_vec_dxf.unlink()
                except Exception:
                    pass
                print(f"Wrote: {outp}")
                return

            # PDF inputs
            if in_ext == ".pdf":
                try:
                    from .neural_cad_detector import NeuralCADDetector
                    from .pdf_processor import PDFToImageConverter, CADPipeline
                except ImportError as e:
                    raise RuntimeError("PDF pipeline dependencies missing. Install requirements-neural.txt and PyMuPDF/pdf2image") from e
                detector = NeuralCADDetector(device=args.device)
                pdf_conv = PDFToImageConverter(dpi=args.dpi)
                pipe = CADPipeline(neural_detector=detector, pdf_converter=pdf_conv)
                tmp_vec_dxf = outp.with_suffix("").with_name(outp.stem + "_tmp_vec2d.dxf")
                pipe.process_pdf_to_dxf(str(inp), str(tmp_vec_dxf), confidence_threshold=args.confidence, scale_mm_per_pixel=args.scale)
                if args.to_3d:
                    extrude_to_destination(tmp_vec_dxf)
                else:
                    if out_ext == ".dwg":
                        convert_dxf_to_dwg(str(tmp_vec_dxf), str(outp), out_version=args.dwg_version)
                    else:
                        ensure_parent(outp)
                        copyfile(str(tmp_vec_dxf), str(outp))
                try:
                    tmp_vec_dxf.unlink()
                except Exception:
                    pass
                print(f"Wrote: {outp}")
                return

            # DXF inputs
            if in_ext == ".dxf":
                if args.to_3d:
                    extrude_to_destination(inp)
                else:
                    if out_ext == ".dxf":
                        ensure_parent(outp)
                        copyfile(str(inp), str(outp))
                    elif out_ext == ".dwg":
                        convert_dxf_to_dwg(str(inp), str(outp), out_version=args.dwg_version)
                    else:
                        raise RuntimeError("خروجی باید DXF یا DWG باشد")
                print(f"Wrote: {outp}")
                return

            # DWG inputs
            if in_ext == ".dwg":
                if args.to_3d:
                    tmp_in_dxf = outp.with_suffix("").with_name(outp.stem + "_tmp_in2d.dxf")
                    convert_dwg_to_dxf(str(inp), str(tmp_in_dxf), out_version=args.dwg_version)
                    try:
                        extrude_to_destination(tmp_in_dxf)
                    finally:
                        try:
                            tmp_in_dxf.unlink()
                        except Exception:
                            pass
                else:
                    if out_ext == ".dwg":
                        ensure_parent(outp)
                        copyfile(str(inp), str(outp))
                    elif out_ext == ".dxf":
                        convert_dwg_to_dxf(str(inp), str(outp), out_version=args.dwg_version)
                    else:
                        raise RuntimeError("خروجی باید DXF یا DWG باشد")
                print(f"Wrote: {outp}")
                return

            raise RuntimeError("فرمت ورودی پشتیبانی نمی‌شود. از PDF/تصویر/DXF/DWG استفاده کنید.")
        except Exception as e:
            print(f"❌ Error in universal-convert: {e}")


if __name__ == "__main__":
    main()
