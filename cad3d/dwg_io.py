from __future__ import annotations
import os
import shutil
import subprocess
from pathlib import Path

from .config import settings


def _find_oda_converter() -> str | None:
    if settings.oda_converter_path and Path(settings.oda_converter_path).exists():
        return settings.oda_converter_path
    # Common install paths (may vary)
    candidates = [
        r"C:\\Program Files\\ODA\\ODAFileConverter\\ODAFileConverter.exe",
        r"C:\\Program Files (x86)\\ODA\\ODAFileConverter\\ODAFileConverter.exe",
    ]
    for c in candidates:
        if Path(c).exists():
            return c
    return None


def convert_dxf_to_dwg(input_dxf: str, output_dwg: str, out_version: str = "ACAD2018") -> None:
    """
    Convert a DXF file to DWG using ODA File Converter if available.
    out_version examples: ACAD2013, ACAD2018, ACAD2024
    """
    exe = _find_oda_converter()
    if exe is None:
        raise RuntimeError(
            "ODA File Converter not found. Install it or set ODA_CONVERTER_PATH."
        )

    in_path = Path(input_dxf).resolve()
    out_path = Path(output_dwg).resolve()
    out_dir = out_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # ODAFileConverter usage: ODAFileConverter <in_dir> <out_dir> <inVer> <outVer> <recursive> <audit> <saveAllLayouts> <filter>
    # We'll use inVer = "ACAD12" (auto), and filter by file name.
    # Some versions accept: ODAFileConverter <in_dir> <out_dir> <inVer> <outVer> <recursive> <audit> <save> <filter>

    in_dir = str(in_path.parent)
    filter_name = in_path.name

    cmd = [
        exe,
        in_dir,
        str(out_dir),
        "ACAD12",
        out_version,
        "0",
        "0",
        "0",
        filter_name,
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"ODA File Converter failed: {e.stderr.decode(errors='ignore')}"
        )

    # Converted file should be in out_dir with DWG extension
    produced = out_dir / (Path(filter_name).stem + ".dwg")
    if not produced.exists():
        # Some versions keep original ext; try to find any DWG produced
        cand = list(out_dir.glob(Path(filter_name).stem + "*.dwg"))
        if not cand:
            raise RuntimeError("Conversion completed but DWG not found in output directory.")
        produced = cand[0]

    # Move/rename to exact requested path if different
    if produced != out_path:
        shutil.move(str(produced), str(out_path))


def convert_dwg_to_dxf(input_dwg: str, output_dxf: str, out_version: str = "ACAD2018") -> None:
    """
    Convert a DWG file to DXF using ODA File Converter if available.
    out_version examples for DXF: ACAD2013, ACAD2018, ACAD2024
    """
    exe = _find_oda_converter()
    if exe is None:
        raise RuntimeError(
            "ODA File Converter not found. Install it or set ODA_CONVERTER_PATH."
        )

    in_path = Path(input_dwg).resolve()
    out_path = Path(output_dxf).resolve()
    out_dir = out_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    in_dir = str(in_path.parent)
    filter_name = in_path.name

    # For DWG -> DXF, specify output version for DXF
    cmd = [
        exe,
        in_dir,
        str(out_dir),
        "ACAD12",
        out_version,
        "0",
        "0",
        "0",
        filter_name,
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"ODA File Converter failed: {e.stderr.decode(errors='ignore')}"
        )

    produced = out_dir / (Path(filter_name).stem + ".dxf")
    if not produced.exists():
        cand = list(out_dir.glob(Path(filter_name).stem + "*.dxf"))
        if not cand:
            raise RuntimeError("Conversion completed but DXF not found in output directory.")
        produced = cand[0]

    if produced != out_path:
        shutil.move(str(produced), str(out_path))
