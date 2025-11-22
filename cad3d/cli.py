"""
Command-Line Interface for the CAD 2D to 3D Conversion Toolkit.

This module provides a single entry point (`cad3d`) for various offline
conversion and analysis tools, including:
- Extruding 2D DXF polylines into 3D meshes.
- Converting between DXF and DWG formats using the ODA File Converter.
- Generating 3D meshes from single images via depth estimation (MiDaS).
- Batch processing entire directories of CAD files.
- Advanced AI-driven tools for converting PDFs and images to CAD formats.
- Training and optimizing neural network models for CAD object detection.
- A deep-hybrid model combining ViT, VAE, and Diffusion for 3D generation.

The CLI is designed to be extensible, with each major function exposed as a
subcommand. It handles argument parsing, dispatches to the appropriate backend
functions, and manages reporting for operations like batch processing and hard
shape detection.

Examples:
    Extrude a DXF file:
    $ python -m cad3d.cli dxf-extrude --input plan.dxf --output plan_3d.dxf --height 3000

    Convert an image to a 3D mesh:
    $ python -m cad3d.cli img-to-3d --input photo.jpg --output photo_3d.dxf

    Run a batch extrusion job:
    $ python -m cad3d.cli batch-extrude --input-dir ./in --output-dir ./out --jobs 4
"""
from __future__ import annotations
import argparse
import csv
import json
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .config import AppSettings
from .dxf_extrude import extrude_dxf_closed_polylines
from .dwg_io import convert_dxf_to_dwg, convert_dwg_to_dxf

# Lazily import heavy dependencies to keep CLI startup fast
# image_to_depth, neural_cad_detector, etc., are imported inside their handlers.


def _positive_float(x: str) -> float:
    """Argparse type for a positive float."""
    v = float(x)
    if v <= 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return v


def _setup_core_parsers(sub: argparse._SubParsersAction, settings: AppSettings) -> None:
    """
    Set up argparse subparsers for core CAD conversion and extrusion tasks.

    Args:
        sub: The subparser action object from argparse.
        settings: The application settings instance.
    """
    # --- dxf-extrude ---
    p_ex = sub.add_parser("dxf-extrude", help="Extrude closed polylines in a DXF file to 3D MESH entities.")
    p_ex.add_argument("--input", required=True, help="Input DXF path.")
    p_ex.add_argument("--output", required=True, help="Output DXF path.")
    p_ex.add_argument("--height", type=_positive_float, default=settings.default_extrude_height, help="Extrusion height in drawing units.")
    p_ex.add_argument("--layers", nargs="*", help="Only extrude polylines on these specific layers.")
    p_ex.add_argument("--optimize-vertices", action="store_true", help="Deduplicate vertices to reduce final mesh size.")
    p_ex.add_argument("--arc-segments", type=int, default=12, help="Number of segments to approximate arc bulges (higher is smoother).")
    p_ex.add_argument("--arc-max-seglen", type=float, help="Use adaptive arc sampling with a max segment length (in drawing units). Overrides --arc-segments.")
    p_ex.add_argument("--detect-hard-shapes", action="store_true", help="Detect and skip problematic polylines (e.g., self-intersections, zero-area).")
    p_ex.add_argument("--hard-report-csv", help="Path to write a CSV diagnostic report for skipped hard shapes.")
    p_ex.add_argument("--hard-report-json", help="Path to write a JSON diagnostic report for skipped hard shapes.")
    p_ex.add_argument("--colorize", action="store_true", help="Apply colors to meshes from source entities or layers.")
    p_ex.add_argument("--split-by-color", action="store_true", help="Create new layers for each color in the format <layer>__COLOR_R_G_B.")
    p_ex.add_argument("--color-report-csv", help="Path to write a CSV summary of mesh colors.")
    p_ex.add_argument("--color-report-json", help="Path to write a JSON report of mesh colors.")

    # --- dxf-to-dwg ---
    p_conv = sub.add_parser("dxf-to-dwg", help="Convert a DXF file to DWG format using the ODA File Converter.")
    p_conv.add_argument("--input", required=True, help="Input DXF path.")
    p_conv.add_argument("--output", required=True, help="Output DWG path.")
    p_conv.add_argument("--version", default="ACAD2018", help="DWG output version (e.g., ACAD2013, ACAD2018).")

    # --- img-to-3d ---
    p_img = sub.add_parser("img-to-3d", help="Convert a single image to a 3D DXF mesh using a depth estimation model (MiDaS).")
    p_img.add_argument("--input", required=True, help="Input image path.")
    p_img.add_argument("--output", required=True, help="Output DXF path.")
    p_img.add_argument("--scale", type=_positive_float, default=1000.0, help="Depth scale factor for Z-axis values.")
    p_img.add_argument("--model-path", help="Override path to the ONNX model (otherwise, MIDAS_ONNX_PATH environment variable is used).")
    p_img.add_argument("--size", type=int, default=256, help="Input image size for the ONNX model (MiDaS small default is 256).")
    p_img.add_argument("--optimize-vertices", action="store_true", help="Deduplicate vertices to reduce final mesh size.")


def _setup_workflow_parsers(sub: argparse._SubParsersAction, settings: AppSettings) -> None:
    """
    Set up argparse subparsers for automated and batch processing workflows.

    Args:
        sub: The subparser action object from argparse.
        settings: The application settings instance.
    """
    # --- auto-extrude (DXF/DWG in -> DXF/DWG out) ---
    p_auto = sub.add_parser("auto-extrude", help="Automatically extrude a 2D CAD file (DXF/DWG) to a 3D file (DXF/DWG).")
    p_auto.add_argument("--input", required=True, help="Input DXF or DWG path.")
    p_auto.add_argument("--output", required=True, help="Output DXF or DWG path.")
    p_auto.add_argument("--height", type=_positive_float, default=settings.default_extrude_height, help="Extrusion height in drawing units.")
    p_auto.add_argument("--layers", nargs="*", help="Only extrude polylines on these specific layers.")
    p_auto.add_argument("--arc-segments", type=int, default=12, help="Segments for arc approximation.")
    p_auto.add_argument("--arc-max-seglen", type=float, help="Adaptive arc sampling max segment length.")
    p_auto.add_argument("--keep-temp", action="store_true", help="Keep temporary DXF files for debugging purposes.")
    p_auto.add_argument("--dwg-version", default="ACAD2018", help="DWG/DXF version for conversions.")
    p_auto.add_argument("--optimize-vertices", action="store_true", help="Deduplicate vertices to reduce mesh size.")
    p_auto.add_argument("--detect-hard-shapes", action="store_true", help="Detect and skip problematic polylines.")
    p_auto.add_argument("--hard-report-csv", help="Path to write a CSV report for skipped shapes.")
    p_auto.add_argument("--hard-report-json", help="Path to write a JSON report for skipped shapes.")
    p_auto.add_argument("--colorize", action="store_true", help="Apply colors to meshes.")
    p_auto.add_argument("--split-by-color", action="store_true", help="Place meshes on per-color layers.")
    p_auto.add_argument("--color-report-csv", help="Path to write a CSV color summary.")
    p_auto.add_argument("--color-report-json", help="Path to write a JSON color report.")

    # --- batch-extrude ---
    p_batch = sub.add_parser("batch-extrude", help="Batch extrude a folder of DXF/DWG files in parallel.")
    p_batch.add_argument("--input-dir", required=True, help="Input folder containing DXF/DWG files.")
    p_batch.add_argument("--output-dir", required=True, help="Output folder for the results.")
    p_batch.add_argument("--out-format", choices=["DXF", "DWG"], default="DXF", help="Output file format.")
    p_batch.add_argument("--height", type=_positive_float, default=settings.default_extrude_height, help="Extrusion height.")
    p_batch.add_argument("--layers", nargs="*", help="Layers to extrude.")
    p_batch.add_argument("--arc-segments", type=int, default=12, help="Segments for arc approximation.")
    p_batch.add_argument("--arc-max-seglen", type=float, help="Adaptive arc sampling max segment length.")
    p_batch.add_argument("--recurse", action="store_true", help="Recurse into subdirectories.")
    p_batch.add_argument("--pattern", nargs="+", default=["*.dxf", "*.dwg"], help="File glob patterns to include.")
    p_batch.add_argument("--dwg-version", default="ACAD2018", help="DWG/DXF version for conversions.")
    p_batch.add_argument("--skip-existing", action="store_true", help="Skip processing if the output file already exists.")
    p_batch.add_argument("--optimize-vertices", action="store_true", help="Deduplicate vertices.")
    p_batch.add_argument("--report-csv", help="Path to write a CSV report of all processed files.")
    p_batch.add_argument("--report-json", help="Path to write a JSON report of all processed files.")
    p_batch.add_argument("--jobs", type=int, default=1, help="Number of parallel workers to use.")
    p_batch.add_argument("--detect-hard-shapes", action="store_true", help="Detect and skip problematic polylines.")
    p_batch.add_argument("--hard-report-csv", help="Path to write a consolidated CSV report for all skipped shapes.")
    p_batch.add_argument("--hard-report-json", help="Path to write a consolidated JSON report for all skipped shapes.")
    p_batch.add_argument("--colorize", action="store_true", help="Apply colors to meshes.")
    p_batch.add_argument("--split-by-color", action="store_true", help="Place meshes on per-color layers.")
    p_batch.add_argument("--color-report-csv", help="Path to write a consolidated CSV color summary.")
    p_batch.add_argument("--color-report-json", help="Path to write a consolidated JSON color report.")


def _setup_ai_parsers(sub: argparse._SubParsersAction, settings: AppSettings) -> None:
    """
    Set up argparse subparsers for AI-driven conversion and analysis tasks.

    Args:
        sub: The subparser action object from argparse.
        settings: The application settings instance.
    """
    # --- AI-driven 2D Vectorization ---
    p_pdf = sub.add_parser("pdf-to-dxf", help="Convert a PDF to DXF using a neural network for object detection.")
    p_pdf.add_argument("--input", required=True, help="Input PDF file path.")
    p_pdf.add_argument("--output", required=True, help="Output DXF file path.")
    p_pdf.add_argument("--dpi", type=int, default=300, help="Rendering resolution for the PDF (300-600 DPI).")
    p_pdf.add_argument("--confidence", type=float, default=0.5, help="Minimum detection confidence threshold (0-1).")
    p_pdf.add_argument("--scale", type=float, default=1.0, help="Scale factor in millimeters per pixel.")
    p_pdf.add_argument("--device", default="auto", help="Computation device: 'cpu', 'cuda', or 'auto'.")

    p_img_detect = sub.add_parser("image-to-dxf", help="Convert a drawing image to DXF using AI-based detection.")
    p_img_detect.add_argument("--input", required=True, help="Input image path of the drawing.")
    p_img_detect.add_argument("--output", required=True, help="Output DXF file path.")
    p_img_detect.add_argument("--confidence", type=float, default=0.5, help="Minimum detection confidence.")
    p_img_detect.add_argument("--scale", type=float, default=1.0, help="Scale factor in millimeters per pixel.")
    p_img_detect.add_argument("--detect-lines", action="store_true", default=True, help="Enable line detection.")
    p_img_detect.add_argument("--detect-circles", action="store_true", default=True, help="Enable circle detection.")
    p_img_detect.add_argument("--detect-text", action="store_true", default=True, help="Enable text detection via OCR.")
    p_img_detect.add_argument("--device", default="auto", help="Computation device: 'cpu', 'cuda', or 'auto'.")

    # --- AI-driven 3D Conversion ---
    p_pdf3d = sub.add_parser("pdf-to-3d", help="Convert a PDF to a 3D DXF file using AI.")
    p_pdf3d.add_argument("--input", required=True, help="Input PDF file path.")
    p_pdf3d.add_argument("--output", required=True, help="Output 3D DXF file path.")
    p_pdf3d.add_argument("--dpi", type=int, default=300, help="Rendering resolution for the PDF.")
    p_pdf3d.add_argument("--intelligent-height", action="store_true", help="Use an ML model to predict extrusion heights.")
    p_pdf3d.add_argument("--device", default="auto", help="Computation device: 'cpu', 'cuda', or 'auto'.")

    # --- Deep Hybrid Model ---
    p_hybrid = sub.add_parser("deep-hybrid", help="Generate a 3D point cloud from an image using a ViT+VAE+Diffusion tri-fusion model.")
    p_hybrid.add_argument("--input", required=True, help="Input image path.")
    p_hybrid.add_argument("--output", required=True, help="Output DXF path for the point cloud.")
    p_hybrid.add_argument("--prior-strength", type=float, default=0.5, help="Blend ratio for VAE prior vs. noise (0=noise only, 1=prior only).")
    p_hybrid.add_argument("--ddim-steps", type=int, default=40, help="Number of DDIM sampling steps for the diffusion model.")
    p_hybrid.add_argument("--normalize-range", type=float, nargs=2, default=[0.0, 1000.0], help="Min and max for normalizing output CAD units.")
    p_hybrid.add_argument("--optimize-vertices", action="store_true", help="Deduplicate vertices before writing (for future mesh conversion).")

    # --- Universal Converter ---
    p_uni = sub.add_parser("universal-convert", help="Universal converter: input image/PDF/DXF/DWG, output DXF/DWG, with optional 3D extrusion.")
    p_uni.add_argument("--input", required=True, help="Input path (image, PDF, DXF, or DWG).")
    p_uni.add_argument("--output", required=True, help="Output path (DXF or DWG).")
    p_uni.add_argument("--dpi", type=int, default=300, help="DPI for PDF rasterization.")
    p_uni.add_argument("--confidence", type=float, default=0.5, help="Minimum confidence for AI detection.")
    p_uni.add_argument("--scale", type=float, default=1.0, help="Scale in mm/pixel for image inputs.")
    p_uni.add_argument("--device", default="auto", help="Computation device: 'cpu', 'cuda', or 'auto'.")
    p_uni.add_argument("--dwg-version", default="ACAD2018", help="Output version for DWG files.")
    p_uni.add_argument("--to-3d", action="store_true", help="Perform 3D extrusion after vectorization/conversion.")
    p_uni.add_argument("--height", type=_positive_float, default=settings.default_extrude_height, help="Extrusion height for 3D conversion.")
    p_uni.add_argument("--layers", nargs="*", help="Layers to extrude in 3D conversion.")
    p_uni.add_argument("--arc-segments", type=int, default=12, help="Segments for arc approximation.")
    p_uni.add_argument("--arc-max-seglen", type=float, help="Adaptive arc sampling max segment length.")
    p_uni.add_argument("--optimize-vertices", action="store_true", help="Deduplicate vertices to reduce mesh size.")
    p_uni.add_argument("--detect-hard-shapes", action="store_true", help="Detect and skip problematic polylines during extrusion.")


def _setup_ml_ops_parsers(sub: argparse._SubParsersAction) -> None:
    """
    Set up argparse subparsers for Machine Learning Operations (MLOps) tasks like
    dataset building, training, and model optimization.

    Args:
        sub: The subparser action object from argparse.
    """
    # --- Architectural Analysis ---
    p_analyze = sub.add_parser("analyze-architectural", help="Analyze architectural drawings (plans, elevations, sections) to extract structured data.")
    p_analyze.add_argument("--input", required=True, help="Input DXF file or a folder of DXF files.")
    p_analyze.add_argument("--output-dir", required=True, help="Output directory for analysis results.")
    p_analyze.add_argument("--recursive", action="store_true", help="Process folders recursively.")
    p_analyze.add_argument("--export-json", action="store_true", help="Export the full dataset to JSON format.")
    p_analyze.add_argument("--export-csv", action="store_true", help="Export room data to CSV format.")
    p_analyze.add_argument("--report", action="store_true", help="Generate a human-readable text report for each drawing.")

    # --- Dataset Building ---
    p_build_ds = sub.add_parser("build-dataset", help="Build a training dataset from a collection of DXF files.")
    p_build_ds.add_argument("--input-dir", required=True, help="Directory containing the source DXF files.")
    p_build_ds.add_argument("--output-dir", required=True, help="Directory to save the output dataset.")
    p_build_ds.add_argument("--image-size", type=int, nargs=2, default=[1024, 1024], help="Output image size (width height).")
    p_build_ds.add_argument("--format", choices=["coco", "yolo", "both"], default="coco", help="Annotation output format.")
    p_build_ds.add_argument("--visualize", action="store_true", help="Save visualization images with annotations overlaid.")
    p_build_ds.add_argument("--recurse", action="store_true", help="Search for DXF files in subdirectories.")

    # --- Model Training ---
    p_train = sub.add_parser("train", help="Train a CAD object detection model.")
    p_train.add_argument("--dataset-dir", required=True, help="Path to the dataset directory (in COCO format).")
    p_train.add_argument("--output-dir", required=True, help="Directory to save model checkpoints and logs.")
    p_train.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    p_train.add_argument("--batch-size", type=int, default=4, help="Training batch size.")
    p_train.add_argument("--lr", type=float, default=0.001, help="Initial learning rate.")
    p_train.add_argument("--device", default="cuda", help="Computation device: 'cpu' or 'cuda'.")
    p_train.add_argument("--workers", type=int, default=4, help="Number of data loader workers.")
    p_train.add_argument("--optimizer", choices=["sgd", "adam"], default="sgd", help="Optimizer type.")
    p_train.add_argument("--resume", help="Path to a checkpoint file to resume training from.")
    p_train.add_argument("--pretrained", action="store_true", help="Start with pre-trained model weights.")

    # --- Model Optimization ---
    p_optimize = sub.add_parser("optimize-model", help="Optimize a trained model for deployment (ONNX, TensorRT, Quantization).")
    p_optimize.add_argument("--model", required=True, help="Path to the PyTorch model checkpoint (.pth).")
    p_optimize.add_argument("--output-dir", required=True, help="Directory to save the optimized models.")
    p_optimize.add_argument("--formats", nargs="+", choices=["onnx", "tensorrt", "quantized"], default=["onnx"], help="Target optimization formats.")
    p_optimize.add_argument("--input-size", type=int, nargs=2, default=[1024, 1024], help="Model input size (height width).")
    p_optimize.add_argument("--benchmark", action="store_true", help="Run and report benchmarks after optimization.")
    p_optimize.add_argument("--device", default="cuda", help="Device for optimization: 'cpu' or 'cuda'.")

    # --- Benchmarking ---
    p_benchmark = sub.add_parser("benchmark", help="Evaluate the accuracy and speed of a model.")
    p_benchmark.add_argument("--model", required=True, help="Path to the model file (.pth, .onnx, .trt).")
    p_benchmark.add_argument("--dataset-dir", required=True, help="Path to the evaluation dataset (COCO format).")
    p_benchmark.add_argument("--output-dir", required=True, help="Directory to save benchmark results.")
    p_benchmark.add_argument("--model-format", choices=["pytorch", "onnx", "tensorrt"], default="pytorch", help="The format of the model being benchmarked.")
    p_benchmark.add_argument("--batch-size", type=int, default=1, help="Benchmark batch size.")
    p_benchmark.add_argument("--max-samples", type=int, help="Maximum number of samples to evaluate.")
    p_benchmark.add_argument("--device", default="cuda", help="Device for benchmarking: 'cpu' or 'cuda'.")


def main(argv: Optional[List[str]] = None) -> None:
    """
    Main entry point for the `cad3d` command-line interface.

    Parses arguments, sets up subparsers for different commands, and dispatches
    to the appropriate handler function based on the user's input.

    Args:
        argv: A list of command-line arguments, or None to use `sys.argv`.
    """
    # Load settings from environment variables and .env file
    settings = AppSettings()

    parser = argparse.ArgumentParser(prog="cad3d", description="An offline toolkit for 2D-to-3D CAD conversion and analysis.")
    sub = parser.add_subparsers(dest="cmd", required=True, title="Available Commands")

    # Organize parser setup into logical groups
    _setup_core_parsers(sub, settings)
    _setup_workflow_parsers(sub, settings)
    _setup_ai_parsers(sub, settings)
    _setup_ml_ops_parsers(sub)

    args = parser.parse_args(argv)

    # --- Command Dispatch ---
    # A mapping of command names to their handler functions.
    COMMAND_HANDLERS: Dict[str, Callable[[argparse.Namespace], None]] = {
        "dxf-extrude": handle_dxf_extrude,
        "dxf-to-dwg": handle_dxf_to_dwg,
        "img-to-3d": handle_img_to_3d,
        "auto-extrude": handle_auto_extrude,
        "batch-extrude": handle_batch_extrude,
        "analyze-architectural": handle_analyze_architectural,
        "pdf-to-dxf": handle_pdf_to_dxf,
        "image-to-dxf": handle_image_to_dxf,
        "pdf-to-3d": handle_pdf_to_3d,
        "build-dataset": handle_build_dataset,
        "train": handle_train,
        "optimize-model": handle_optimize_model,
        "benchmark": handle_benchmark,
        "universal-convert": handle_universal_convert,
        "deep-hybrid": handle_deep_hybrid,
    }

    handler = COMMAND_HANDLERS.get(args.cmd)
    if handler:
        handler(args)
    else:
        # This case should not be reachable if a command is required.
        parser.print_help()


# --- Reporting Helper Functions ---

def _write_hard_shape_reports(args: argparse.Namespace, hard_rows: Optional[List[Dict[str, Any]]]) -> None:
    """
    Writes diagnostic reports for "hard shapes" that were skipped during processing.

    Args:
        args: The command-line arguments, checked for report path attributes.
        hard_rows: A list of dictionaries, where each entry details a skipped shape.
    """
    if not hard_rows:
        return
    if getattr(args, "hard_report_csv", None):
        _write_csv_report(
            Path(args.hard_report_csv),
            ["layer", "handle", "issues", "vertex_count"],
            [{"layer": r.get("layer", ""), "handle": r.get("handle", ""), "issues": ", ".join(r.get("issues", [])), "vertex_count": r.get("vertex_count", 0)} for r in hard_rows]
        )
    if getattr(args, "hard_report_json", None):
        _write_json_report(Path(args.hard_report_json), hard_rows)


def _write_color_reports(args: argparse.Namespace, color_rows: Optional[List[Dict[str, Any]]]) -> None:
    """
    Writes reports summarizing color statistics for generated meshes.

    Args:
        args: The command-line arguments, checked for report path attributes.
        color_rows: A list of dictionaries with color data for each mesh.
    """
    if not color_rows:
        return
    if getattr(args, "color_report_csv", None):
        _write_aggregated_color_csv_report(Path(args.color_report_csv), color_rows)
    if getattr(args, "color_report_json", None):
        _write_json_report(Path(args.color_report_json), color_rows)


def _write_csv_report(path: Path, headers: List[str], rows: List[Dict[str, Any]]) -> None:
    """
    Writes a generic list of dictionaries to a CSV file.

    Args:
        path: The output file path.
        headers: A list of strings for the CSV header row.
        rows: A list of dictionaries to write as rows.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Report written: {path}")


def _write_json_report(path: Path, data: List[Any]) -> None:
    """
    Writes a list of data to a JSON file.

    Args:
        path: The output file path.
        data: The list of data to serialize.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(data, fp, ensure_ascii=False, indent=2)
    print(f"Report written: {path}")


def _write_aggregated_color_csv_report(path: Path, color_rows: List[Dict[str, Any]]) -> None:
    """
    Aggregates color data and writes a summary CSV report.

    The report counts how many meshes were created for each combination of
    file, target layer, and color.

    Args:
        path: The output file path.
        color_rows: A list of raw color data entries.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    agg: Dict[tuple, int] = {}
    key_fields = ("file_path", "target_layer", "r", "g", "b")
    for r in color_rows:
        key = tuple(r.get(k, "") for k in key_fields)
        agg[key] = agg.get(key, 0) + 1
    
    rows_to_write = []
    for key_tuple, count in sorted(agg.items()):
        row = dict(zip(key_fields, key_tuple))
        row["count"] = count
        rows_to_write.append(row)

    headers = list(key_fields) + ["count"]
    _write_csv_report(path, headers, rows_to_write)


# --- Command Handler Functions ---

def handle_dxf_extrude(args: argparse.Namespace) -> None:
    """
    Handler for the 'dxf-extrude' command.

    Args:
        args: The parsed command-line arguments.
    """
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
    
    _write_hard_shape_reports(args, hard_rows)
    _write_color_reports(args, color_rows)
    print(f"Successfully wrote 3D DXF: {args.output}")


def handle_dxf_to_dwg(args: argparse.Namespace) -> None:
    """
    Handler for the 'dxf-to-dwg' command.

    Args:
        args: The parsed command-line arguments.
    """
    convert_dxf_to_dwg(args.input, args.output, out_version=args.version)
    print(f"Successfully wrote DWG: {args.output}")


def handle_img_to_3d(args: argparse.Namespace) -> None:
    """
    Handler for the 'img-to-3d' command. Lazily imports `image_to_depth`.

    Args:
        args: The parsed command-line arguments.
    """
    from .image_to_depth import image_to_3d_dxf
    image_to_3d_dxf(
        args.input, 
        args.output, 
        scale=args.scale, 
        model_path=args.model_path, 
        size=args.size, 
        optimize=args.optimize_vertices
    )
    print(f"Successfully wrote 3D DXF from image: {args.output}")


def handle_auto_extrude(args: argparse.Namespace) -> None:
    """
    Handler for the 'auto-extrude' command. Manages DWG->DXF->extrude->DWG flow.

    Args:
        args: The parsed command-line arguments.
    """
    inp = Path(args.input)
    outp = Path(args.output)
    tmp_in_dxf: Optional[Path] = None
    tmp_out_dxf: Optional[Path] = None

    hard_rows = [] if (getattr(args, "hard_report_csv", None) or getattr(args, "hard_report_json", None)) else None
    color_rows = [] if (getattr(args, "color_report_csv", None) or getattr(args, "color_report_json", None)) else None
    
    try:
        # Step 1: Convert input to DXF if it's a DWG
        if inp.suffix.lower() == ".dwg":
            tmp_in_dxf = outp.with_suffix("").with_name(f"{outp.stem}_tmp_in.dxf")
            convert_dwg_to_dxf(str(inp), str(tmp_in_dxf), out_version=args.dwg_version)
            in_dxf = tmp_in_dxf
        elif inp.suffix.lower() == ".dxf":
            in_dxf = inp
        else:
            raise ValueError("Unsupported input file type. Use .dxf or .dwg.")

        # Step 2: Prepare arguments and extrude the DXF
        extrude_args = {
            "height": args.height,
            "layers": args.layers,
            "arc_segments": args.arc_segments,
            "arc_max_seglen": args.arc_max_seglen,
            "optimize": args.optimize_vertices,
            "detect_hard_shapes": args.detect_hard_shapes,
            "hard_shapes_collector": hard_rows,
            "colorize": getattr(args, "colorize", False),
            "split_by_color": getattr(args, "split_by_color", False),
            "color_stats_collector": color_rows,
        }

        # Step 3: Convert extruded DXF to output format
        if outp.suffix.lower() == ".dwg":
            tmp_out_dxf = outp.with_suffix("").with_name(f"{outp.stem}_tmp_out.dxf")
            extrude_dxf_closed_polylines(str(in_dxf), str(tmp_out_dxf), **extrude_args)
            convert_dxf_to_dwg(str(tmp_out_dxf), str(outp), out_version=args.dwg_version)
        elif outp.suffix.lower() == ".dxf":
            extrude_dxf_closed_polylines(str(in_dxf), str(outp), **extrude_args)
        else:
            raise ValueError("Unsupported output file type. Use .dxf or .dwg.")

        print(f"Successfully created: {args.output}")
    finally:
        # Step 4: Clean up temporary files
        if not getattr(args, "keep_temp", False):
            if tmp_in_dxf and tmp_in_dxf.exists(): tmp_in_dxf.unlink(missing_ok=True)
            if tmp_out_dxf and tmp_out_dxf.exists(): tmp_out_dxf.unlink(missing_ok=True)
        
        # Step 5: Write any generated reports
        _write_hard_shape_reports(args, hard_rows)
        _write_color_reports(args, color_rows)


def handle_batch_extrude(args: argparse.Namespace) -> None:
    """
    Handler for the 'batch-extrude' command. Processes files in parallel.

    Args:
        args: The parsed command-line arguments.
    """
    in_root = Path(args.input_dir)
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # Gather all files to be processed
    files: List[Path] = []
    for pat in args.pattern:
        files.extend(in_root.rglob(pat) if args.recurse else in_root.glob(pat))

    report_rows: List[Dict[str, Any]] = []
    hard_json_all: List[Dict[str, Any]] = [] if getattr(args, "hard_report_json", None) else []
    color_rows_all: List[Dict[str, Any]] = [] if (getattr(args, "color_report_csv", None) or getattr(args, "color_report_json", None)) else []

    def process_one(f: Path) -> Dict[str, Any]:
        """Processes a single file in the batch."""
        rec = {"file_path": str(f), "output_path": "", "status": "init", "message": "", "duration_sec": 0.0}
        t0 = time.monotonic()
        try:
            rel = f.relative_to(in_root)
            out_dir = out_root / rel.parent
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{rel.stem}.{args.out_format.lower()}"
            
            if args.skip_existing and out_path.exists():
                rec.update({"output_path": str(out_path), "status": "skipped", "message": "Output file already exists."})
                return rec

            tmp_in_dxf, tmp_out_dxf = None, None
            try:
                in_dxf = f
                if f.suffix.lower() == ".dwg":
                    tmp_in_dxf = out_dir / f"{f.stem}.{uuid.uuid4().hex}.tmp_in.dxf"
                    convert_dwg_to_dxf(str(f), str(tmp_in_dxf), out_version=args.dwg_version)
                    in_dxf = tmp_in_dxf

                local_hard: List[Dict] = []
                local_colors: List[Dict] = []
                extrude_args = {
                    "height": args.height, "layers": args.layers, "arc_segments": args.arc_segments,
                    "arc_max_seglen": args.arc_max_seglen, "optimize": args.optimize_vertices,
                    "detect_hard_shapes": args.detect_hard_shapes, "hard_shapes_collector": local_hard,
                    "colorize": args.colorize, "split_by_color": args.split_by_color,
                    "color_stats_collector": local_colors,
                }

                if args.out_format == "DWG":
                    tmp_out_dxf = out_dir / f"{f.stem}.{uuid.uuid4().hex}.tmp_out.dxf"
                    extrude_dxf_closed_polylines(str(in_dxf), str(tmp_out_dxf), **extrude_args)
                    convert_dxf_to_dwg(str(tmp_out_dxf), str(out_path), out_version=args.dwg_version)
                else:
                    extrude_dxf_closed_polylines(str(in_dxf), str(out_path), **extrude_args)

                # Collect results for consolidated reports
                if args.hard_report_json or args.hard_report_csv:
                    for r in local_hard:
                        hard_json_all.append({"file_path": str(f), **r})
                if args.color_report_csv or args.color_report_json:
                    for r in local_colors:
                        color_rows_all.append({"file_path": str(f), **r})
                
                rec.update({"output_path": str(out_path), "status": "ok"})
            finally:
                if tmp_in_dxf: tmp_in_dxf.unlink(missing_ok=True)
                if tmp_out_dxf: tmp_out_dxf.unlink(missing_ok=True)
        except Exception as ex:
            tb = traceback.format_exc(limit=1).strip().replace("\n", " ")
            rec.update({"status": "error", "message": f"{type(ex).__name__}: {ex} | {tb}"})
        
        rec["duration_sec"] = round(time.monotonic() - t0, 3)
        return rec

    # Execute processing, in parallel or sequentially
    if args.jobs > 1 and len(files) > 1:
        with ThreadPoolExecutor(max_workers=args.jobs) as executor:
            futs = {executor.submit(process_one, f): f for f in files}
            for fut in as_completed(futs):
                report_rows.append(fut.result())
    else:
        for f in files:
            report_rows.append(process_one(f))

    # Final summary and reporting
    processed = sum(1 for r in report_rows if r["status"] == "ok")
    skipped = sum(1 for r in report_rows if r["status"] == "skipped")
    failed_rows = [r for r in report_rows if r["status"] == "error"]
    print(f"\nBatch complete. Succeeded: {processed}, Skipped: {skipped}, Failed: {len(failed_rows)}")
    for f_rec in failed_rows[:10]: # Print first 10 errors
        print(f" - ERROR in {Path(f_rec['file_path']).name}: {f_rec['message']}")

    if args.report_csv:
        _write_csv_report(Path(args.report_csv), ["file_path", "output_path", "status", "message", "duration_sec"], report_rows)
    if args.report_json:
        _write_json_report(Path(args.report_json), report_rows)
    if args.hard_report_csv and hard_json_all:
        _write_csv_report(Path(args.hard_report_csv), ["file_path", "layer", "handle", "issues", "vertex_count"], hard_json_all)
    if args.hard_report_json and hard_json_all:
        _write_json_report(Path(args.hard_report_json), hard_json_all)
    if args.color_report_csv and color_rows_all:
        _write_aggregated_color_csv_report(Path(args.color_report_csv), color_rows_all)
    if args.color_report_json and color_rows_all:
        _write_json_report(Path(args.color_report_json), color_rows_all)


def handle_analyze_architectural(args: argparse.Namespace) -> None:
    """
    Handler for the 'analyze-architectural' command.

    Args:
        args: The parsed command-line arguments.
    """
    try:
        from .architectural_analyzer import ArchitecturalAnalyzer, generate_analysis_report
        from .dataset_builder import ArchitecturalDatasetBuilder
    except ImportError as e:
        print(f"❌ Error: {e}\nInstall analysis dependencies: pip install shapely")
        return

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if input_path.is_file():
        print(f"Analyzing single drawing: {input_path.name}")
        analyzer = ArchitecturalAnalyzer(str(input_path))
        analysis = analyzer.analyze()
        
        if args.report:
            report = generate_analysis_report(analysis)
            print(report)
            (output_dir / f"{input_path.stem}_analysis.txt").write_text(report, encoding="utf-8")
        
        if args.export_json:
            json_path = output_dir / f"{input_path.stem}_analysis.json"
            simple_data = {
                "drawing_type": analysis.drawing_type.value, "total_area": analysis.total_area,
                "num_rooms": len(analysis.rooms), "num_walls": len(analysis.walls),
                "rooms": [{"name": r.name, "type": r.space_type.value, "area": r.area, "width": r.width, "length": r.length} for r in analysis.rooms],
            }
            _write_json_report(json_path, [simple_data])
    
    elif input_path.is_dir():
        print(f"Processing architectural dataset from folder: {input_path}")
        builder = ArchitecturalDatasetBuilder(str(output_dir))
        builder.process_folder(str(input_path), recursive=args.recursive)
        
        if args.report:
            for i, analysis in enumerate(builder.analyses, 1):
                report = generate_analysis_report(analysis)
                safe_name = Path(analysis.metadata.get("file_path", f"drawing_{i}")).stem
                (output_dir / f"{safe_name}_analysis.txt").write_text(report, encoding="utf-8")
        
        if args.export_json: builder.export_to_json()
        if args.export_csv: builder.export_rooms_to_csv()
        
        builder.export_statistics()
        summary = builder.generate_summary_report()
        print("\n" + summary)
        (output_dir / "dataset_summary.txt").write_text(summary, encoding="utf-8")
    else:
        print(f"❌ Error: Invalid input path specified: {input_path}")


def _run_neural_pipeline(args: argparse.Namespace, mode: str) -> None:
    """
    Generic dispatcher for neural network-based pipelines.

    This function lazily imports the required heavy modules and runs the
    appropriate conversion or detection pipeline based on the `mode`.

    Args:
        args: The parsed command-line arguments.
        mode: A string identifying the pipeline to run (e.g., 'pdf_to_dxf').
    """
    try:
        from .neural_cad_detector import NeuralCADDetector, ImageTo3DExtruder
        from .pdf_processor import PDFToImageConverter, CADPipeline
        
        print(f"🚀 Initializing Neural Pipeline: {mode.replace('_', ' ').title()}")
        print(f"   Input: {args.input}\n   Output: {args.output}")
        
        detector = NeuralCADDetector(device=args.device)
        pipeline_args: Dict[str, Any] = {'neural_detector': detector}
        
        if 'pdf' in mode:
            pipeline_args['pdf_converter'] = PDFToImageConverter(dpi=args.dpi)
        if '3d' in mode:
            pipeline_args['extruder_3d'] = ImageTo3DExtruder()
            
        pipeline = CADPipeline(**pipeline_args)
        
        if mode == 'pdf_to_dxf':
            pipeline.process_pdf_to_dxf(args.input, args.output, confidence_threshold=args.confidence, scale_mm_per_pixel=args.scale)
        elif mode == 'image_to_dxf':
            vectorized = detector.vectorize_drawing(args.input, scale_mm_per_pixel=args.scale, detect_lines=args.detect_lines, detect_circles=args.detect_circles, detect_text=args.detect_text)
            detector.convert_to_dxf(vectorized, args.output)
        elif mode == 'pdf_to_3d':
            pipeline.process_pdf_to_3d(args.input, args.output, intelligent_height=args.intelligent_height)
            
        print(f"\n✅ Success! Output saved to: {args.output}")
    except ImportError as e:
        print(f"❌ Error: Neural dependencies not found. {e}")
        print("   Please install them with: pip install -r requirements-neural.txt")
    except Exception as e:
        print(f"❌ An unexpected error occurred in the neural pipeline: {e}")
        traceback.print_exc()


def handle_pdf_to_dxf(args: argparse.Namespace) -> None:
    """Handler for 'pdf-to-dxf' command."""
    _run_neural_pipeline(args, 'pdf_to_dxf')

def handle_image_to_dxf(args: argparse.Namespace) -> None:
    """Handler for 'image-to-dxf' command."""
    _run_neural_pipeline(args, 'image_to_dxf')

def handle_pdf_to_3d(args: argparse.Namespace) -> None:
    """Handler for 'pdf-to-3d' command."""
    _run_neural_pipeline(args, 'pdf_to_3d')


def handle_build_dataset(args: argparse.Namespace) -> None:
    """
    Handler for the 'build-dataset' command.

    Args:
        args: The parsed command-line arguments.
    """
    try:
        from .training_dataset_builder import CADDatasetBuilder
        
        print("📦 Building CAD Training Dataset...")
        builder = CADDatasetBuilder(output_dir=args.output_dir)
        input_path = Path(args.input_dir)
        dxf_files = list(input_path.rglob("*.dxf")) if args.recurse else list(input_path.glob("*.dxf"))
        
        if not dxf_files:
            print(f"⚠️ No DXF files found in {input_path}.")
            return
            
        print(f"🔍 Found {len(dxf_files)} DXF files to process.")
        for i, dxf_file in enumerate(dxf_files, 1):
            print(f"   [{i}/{len(dxf_files)}] Processing {dxf_file.name}...", end="", flush=True)
            try:
                builder.add_dxf_to_dataset(str(dxf_file), image_size=tuple(args.image_size))
                print(" ✅")
            except Exception as e:
                print(f" ❌ Error: {e}")
        
        print("\n💾 Exporting annotations...")
        if args.format in ["coco", "both"]: builder.export_coco_format()
        if args.format in ["yolo", "both"]: builder.export_yolo_format()
        if args.visualize: builder.visualize_annotations()
        
        print(f"\n✅ Dataset built successfully at: {args.output_dir}")
        print(f"   Total Images: {len(builder.images)}, Total Annotations: {len(builder.annotations)}")
    except ImportError as e:
        print(f"❌ Error: MLOps dependencies not found. {e}")
        print("   Please install them with: pip install -r requirements-neural.txt")
    except Exception as e:
        print(f"❌ An error occurred while building the dataset: {e}")
        traceback.print_exc()


def handle_train(args: argparse.Namespace) -> None:
    """
    Handler for the 'train' command.

    Args:
        args: The parsed command-line arguments.
    """
    try:
        from .training_pipeline import CADDetectionTrainer
        import torch
        
        print("🎓 Training CAD Detection Model...")
        if not torch.cuda.is_available() and args.device == "cuda":
            print("⚠️ CUDA not available, switching to CPU. Training will be slow.")
            device = torch.device("cpu")
        else:
            device = torch.device(args.device)
        
        print(f"   Using device: {device}")
        
        trainer = CADDetectionTrainer(
            data_dir=args.dataset_dir, output_dir=args.output_dir, batch_size=args.batch_size,
            num_workers=args.workers, device=device, pretrained=args.pretrained
        )
        trainer.setup_optimizer(optimizer_type=args.optimizer, learning_rate=args.lr)
        
        if args.resume:
            print(f"📂 Resuming training from checkpoint: {args.resume}")
            trainer.load_checkpoint(args.resume)
        
        print("🚀 Starting training loop...")
        trainer.train(num_epochs=args.epochs)
        print(f"✅ Training complete! Best model saved to: {Path(args.output_dir) / 'best_model.pth'}")
    except ImportError as e:
        print(f"❌ Error: PyTorch dependencies not found. {e}")
        print("   Please install them with: pip install torch torchvision")
    except Exception as e:
        print(f"❌ An error occurred during training: {e}")
        traceback.print_exc()


def handle_optimize_model(args: argparse.Namespace) -> None:
    """
    Handler for the 'optimize-model' command.

    Args:
        args: The parsed command-line arguments.
    """
    try:
        from .model_optimizer import ModelOptimizer, compare_models
        from .training_pipeline import CADDetectionTrainer
        import torch
        
        print("⚡ Model Optimization Pipeline...")
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        
        # Load the model from the checkpoint
        trainer = CADDetectionTrainer(data_dir=".", output_dir=".", device=device)
        checkpoint = torch.load(args.model, map_location=device)
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        trainer.model.eval()
        
        optimizer = ModelOptimizer(device=str(device))
        input_shape = (1, 3, args.input_size[0], args.input_size[1])
        results = optimizer.optimize_full_pipeline(
            model=trainer.model, output_dir=args.output_dir,
            input_shape=input_shape, formats=args.formats, benchmark=args.benchmark
        )
        
        print(f"✅ Optimization complete! Models saved in: {args.output_dir}")
        if args.benchmark and results:
            compare_models(results)
    except ImportError as e:
        print(f"❌ Error: Optimization dependencies not found. {e}")
        print("   Please install them with: pip install torch onnx onnxruntime")
    except Exception as e:
        print(f"❌ An error occurred during model optimization: {e}")
        traceback.print_exc()


def handle_benchmark(args: argparse.Namespace) -> None:
    """
    Handler for the 'benchmark' command.

    Args:
        args: The parsed command-line arguments.
    """
    try:
        from .benchmark_suite import DetectionBenchmark
        from .training_pipeline import CADDataset, CADDetectionTrainer
        import torch
        from torch.utils.data import DataLoader
        
        print("📊 Running Model Benchmark...")
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        print(f"   Device: {device}")
        
        dataset = CADDataset(root_dir=args.dataset_dir, annotation_file=str(Path(args.dataset_dir) / "annotations.json"))
        # Use collate_fn to handle the dataset's tuple output
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=lambda x: tuple(zip(*x)))
        
        print(f"   Evaluating on {len(dataset)} images.")
        
        model: Any
        if args.model_format == "pytorch":
            trainer = CADDetectionTrainer(data_dir=".", output_dir=".", device=device)
            checkpoint = torch.load(args.model, map_location=device)
            trainer.model.load_state_dict(checkpoint['model_state_dict'])
            trainer.model.eval()
            model = trainer.model
        else:
            model = args.model # Path to ONNX or TRT model
        
        benchmark = DetectionBenchmark(model, device=str(device), model_format=args.model_format)
        overall_metrics, category_metrics = benchmark.evaluate_dataset(dataloader, max_samples=args.max_samples)
        
        benchmark.print_detailed_report(overall_metrics, category_metrics)
        output_path = Path(args.output_dir) / "benchmark_results.json"
        benchmark.save_results(str(output_path), overall_metrics, category_metrics)
        
        print(f"✅ Benchmark complete! Results saved to: {output_path}")
    except ImportError as e:
        print(f"❌ Error: PyTorch/MLOps dependencies not found. {e}")
    except Exception as e:
        print(f"❌ An error occurred during benchmarking: {e}")
        traceback.print_exc()


def handle_universal_convert(args: argparse.Namespace) -> None:
    """
    Handler for the 'universal-convert' command.

    Args:
        args: The parsed command-line arguments.
    """
    from shutil import copyfile
    inp = Path(args.input)
    outp = Path(args.output)
    
    # Create a temporary directory for intermediate files
    temp_dir = outp.parent / f"temp_{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        in_dxf_path_for_processing: Optional[Path] = None

        # --- Step 1: Convert input to a common 2D DXF format ---
        if inp.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]:
            from .neural_cad_detector import NeuralCADDetector
            detector = NeuralCADDetector(device=args.device)
            vectorized = detector.vectorize_drawing(inp, scale_mm_per_pixel=args.scale)
            tmp_vec_dxf = temp_dir / "vectorized.dxf"
            detector.convert_to_dxf(vectorized, str(tmp_vec_dxf))
            in_dxf_path_for_processing = tmp_vec_dxf
        elif inp.suffix.lower() == ".pdf":
            from .neural_cad_detector import NeuralCADDetector
            from .pdf_processor import PDFToImageConverter, CADPipeline
            pipe = CADPipeline(neural_detector=NeuralCADDetector(device=args.device), pdf_converter=PDFToImageConverter(dpi=args.dpi))
            tmp_vec_dxf = temp_dir / "vectorized.dxf"
            pipe.process_pdf_to_dxf(str(inp), str(tmp_vec_dxf), confidence_threshold=args.confidence, scale_mm_per_pixel=args.scale)
            in_dxf_path_for_processing = tmp_vec_dxf
        elif inp.suffix.lower() == ".dxf":
            in_dxf_path_for_processing = inp
        elif inp.suffix.lower() == ".dwg":
            tmp_in_dxf = temp_dir / "input.dxf"
            convert_dwg_to_dxf(str(inp), str(tmp_in_dxf), out_version=args.dwg_version)
            in_dxf_path_for_processing = tmp_in_dxf
        else:
            raise ValueError("Unsupported input format. Use PDF, image, DXF, or DWG.")

        if not in_dxf_path_for_processing:
            raise RuntimeError("Failed to create an intermediate DXF file for processing.")

        # --- Step 2: Process the 2D DXF to the final output ---
        outp.parent.mkdir(parents=True, exist_ok=True)
        if args.to_3d:
            extrude_args = {
                "height": args.height, "layers": args.layers, "arc_segments": args.arc_segments,
                "arc_max_seglen": args.arc_max_seglen, "optimize": args.optimize_vertices,
                "detect_hard_shapes": args.detect_hard_shapes, "hard_shapes_collector": None,
                "colorize": False, "split_by_color": False, "color_stats_collector": None,
            }
            if outp.suffix.lower() == ".dwg":
                tmp_out_dxf = temp_dir / "extruded.dxf"
                extrude_dxf_closed_polylines(str(in_dxf_path_for_processing), str(tmp_out_dxf), **extrude_args)
                convert_dxf_to_dwg(str(tmp_out_dxf), str(outp), out_version=args.dwg_version)
            else: # DXF output
                extrude_dxf_closed_polylines(str(in_dxf_path_for_processing), str(outp), **extrude_args)
        else: # 2D output
            if outp.suffix.lower() == ".dwg":
                convert_dxf_to_dwg(str(in_dxf_path_for_processing), str(outp), out_version=args.dwg_version)
            else: # DXF output
                if in_dxf_path_for_processing != outp:
                    copyfile(str(in_dxf_path_for_processing), str(outp))
        
        print(f"✅ Universal conversion successful. Output: {outp}")
    except Exception as e:
        print(f"❌ Error during universal conversion: {e}")
        traceback.print_exc()
    finally:
        # --- Step 3: Clean up temporary directory ---
        import shutil
        if 'temp_dir' in locals() and temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)


def handle_deep_hybrid(args: argparse.Namespace) -> None:
    """
    Handler for the 'deep-hybrid' command.

    Args:
        args: The parsed command-line arguments.
    """
    try:
        import torch
        from .hybrid_vae_vit_diffusion import create_deep_hybrid_converter
        
        dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Initializing deep-hybrid model on device: {dev}")
        
        converter = create_deep_hybrid_converter(
            device=dev, 
            prior_strength=args.prior_strength, 
            normalize_range=tuple(args.normalize_range), 
            ddim_steps=int(args.ddim_steps)
        )
        
        res = converter.convert(Path(args.input), Path(args.output))
        
        print(f"✅ Deep-hybrid DXF point cloud written to: {res['dxf']}")
        sidecar_path = Path(args.output).with_suffix('.deep_hybrid.json')
        _write_json_report(sidecar_path, [res])
        print(f"📄 Metadata and performance stats saved to: {sidecar_path}")
    except ImportError as e:
        print(f"❌ Error: Deep-hybrid dependencies not found. {e}")
        print("   Please install them with: pip install -r requirements-neural.txt")
    except Exception as e:
        print(f"❌ An error occurred in the deep-hybrid pipeline: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
