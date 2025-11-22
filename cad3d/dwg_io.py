"""
DWG/DXF I/O Operations using ODA File Converter.

This module provides a Python wrapper around the ODA File Converter, a command-line
utility for converting between different versions of DWG and DXF files. It abstracts
the complexities of the command-line interface, providing simple functions to
convert files.

Key Features:
- Automatically locates the ODA File Converter executable in common installation
  directories, the system PATH, or via an environment variable.
- Handles the directory-based input/output requirement of the converter.
- Provides clear error messages if the conversion fails.

Main Functions:
- `convert_dxf_to_dwg`: Converts a DXF file to a specified version of DWG.
- `convert_dwg_to_dxf`: Converts a DWG file to a specified version of DXF.

Requirements:
- The ODA File Converter must be installed. It can be downloaded from the
  Open Design Alliance website.
- For automatic detection, the converter should be in a standard location or
  its path specified via the `ODA_CONVERTER_PATH` environment variable.
"""
from __future__ import annotations
import os
import shutil
import subprocess
from pathlib import Path

import ezdxf
from .config import settings


def _find_oda_converter() -> str | None:
    """
    Finds the ODA File Converter executable by searching in multiple locations.

    The search order is as follows:
    1. The path specified by the `ODA_CONVERTER_PATH` environment variable.
    2. A list of common default installation paths on Windows.
    3. The system's PATH environment variable.

    Returns:
        The absolute path to the executable if found, otherwise None.
    """
    # 1. Check the environment variable from settings.
    if settings.oda_converter_path and Path(settings.oda_converter_path).is_file():
        return settings.oda_converter_path

    # 2. Check common hardcoded installation paths (primarily for Windows).
    common_paths = [
        r"C:\Program Files\ODA\ODAFileConverter\ODAFileConverter.exe",
        r"C:\Program Files (x86)\ODA\ODAFileConverter\ODAFileConverter.exe",
    ]
    for path in common_paths:
        if Path(path).is_file():
            return path
            
    # 3. Check the system's PATH environment variable.
    exe_name = "ODAFileConverter.exe" if os.name == "nt" else "ODAFileConverter"
    path_in_env = shutil.which(exe_name)
    if path_in_env:
        return path_in_env

    return None


def _run_oda_converter(
    input_path: Path,
    output_path: Path,
    output_format: str,  # "DWG" or "DXF"
    output_version: str,
) -> None:
    """
    A generic wrapper to execute the ODA File Converter command-line tool.

    The ODA converter operates on directories rather than individual files. This
    function handles the creation of temporary directories and constructs the
    correct command-line arguments to process a single file.

    Args:
        input_path: The absolute path to the source file.
        output_path: The absolute path for the desired output file.
        output_format: The target format, either "DWG" or "DXF".
        output_version: The target file version string (e.g., "ACAD2018").

    Raises:
        RuntimeError: If the ODA converter executable is not found or if the
                      conversion process fails for any reason.
    """
    exe = _find_oda_converter()
    if not exe:
        raise RuntimeError(
            "ODA File Converter not found. Please install it from the Open Design "
            "Alliance website or set the ODA_CONVERTER_PATH environment variable."
        )

    input_dir = str(input_path.parent)
    output_dir = str(output_path.parent)
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    # Command structure for ODAFileConverter:
    # <exe> <in_dir> <out_dir> <out_ver> <out_type> <recurse> <audit> [<input_filter>]
    # - out_ver: e.g., "ACAD2018"
    # - out_type: "DWG" or "DXF"
    # - recurse: "0" for No, "1" for Yes
    # - audit: "0" for No, "1" for Yes (recovers and fixes errors)
    # - input_filter: A pattern to process specific files, e.g., "MyFile.dxf"
    cmd = [
        exe,
        input_dir,
        output_dir,
        output_version,
        output_format,
        "0",  # Recurse: No
        "1",  # Audit: Yes
        str(input_path.name), # Input filter to process only our target file.
    ]

    try:
        # Execute the command. Capture output to hide it unless an error occurs.
        result = subprocess.run(
            cmd, check=True, capture_output=True, text=True, encoding='utf-8', errors='ignore'
        )
        if "error" in result.stdout.lower() or "fail" in result.stdout.lower():
             raise RuntimeError(f"ODA File Converter reported an issue: {result.stdout}")

    except FileNotFoundError:
        raise RuntimeError(f"ODA Converter executable seems to have disappeared. Path was: {exe}")
    except subprocess.CalledProcessError as e:
        error_message = e.stderr or e.stdout or "An unknown error occurred."
        raise RuntimeError(
            f"ODA File Converter failed with exit code {e.returncode}.\n"
            f"Command: {' '.join(cmd)}\n"
            f"Error: {error_message.strip()}"
        )

    # The converter creates an output file with the same stem as the input.
    # We must locate this file and rename it to the desired output path if they differ.
    expected_output_name = input_path.stem + f".{output_format.lower()}"
    produced_file = output_dir_path / expected_output_name

    if not produced_file.exists():
        # Fallback for cases where the output name might be slightly different.
        candidates = list(output_dir_path.glob(f"{input_path.stem}*.{output_format.lower()}"))
        if not candidates:
            raise RuntimeError(
                f"Conversion process seemed to succeed, but the expected output file "
                f"'{produced_file.name}' was not found in the output directory."
            )
        produced_file = candidates[0]

    # Rename the generated file to match the exact requested output path.
    if produced_file.resolve() != output_path.resolve():
        shutil.move(str(produced_file), str(output_path))


def convert_dxf_to_dwg(input_dxf: str, output_dwg: str, out_version: str = "ACAD2018") -> None:
    """
    Converts a DXF file to a DWG file using the ODA File Converter.

    Args:
        input_dxf: Path to the input DXF file.
        output_dwg: Path where the output DWG file will be saved.
        out_version: The target DWG version (e.g., "ACAD2013", "ACAD2018").
    """
    print(f"Converting DXF '{input_dxf}' to DWG '{output_dwg}' (Version: {out_version})...")
    _run_oda_converter(
        input_path=Path(input_dxf).resolve(),
        output_path=Path(output_dwg).resolve(),
        output_format="DWG",
        output_version=out_version,
    )
    print("✅ Conversion successful.")


def convert_dwg_to_dxf(input_dwg: str, output_dxf: str, out_version: str = "ACAD2018") -> None:
    """
    Converts a DWG file to a DXF file using the ODA File Converter.

    Args:
        input_dwg: Path to the input DWG file.
        output_dxf: Path where the output DXF file will be saved.
        out_version: The target DXF version (e.g., "ACAD2013", "ACAD2018").
    """
    print(f"Converting DWG '{input_dwg}' to DXF '{output_dxf}' (Version: {out_version})...")
    _run_oda_converter(
        input_path=Path(input_dwg).resolve(),
        output_path=Path(output_dxf).resolve(),
        output_format="DXF",
        output_version=out_version,
    )
    print("✅ Conversion successful.")


if __name__ == '__main__':
    print("--- ODA File Converter Wrapper Demonstration ---")
    
    converter_path = _find_oda_converter()
    if not converter_path:
        print("\n⚠️  ODA File Converter not found.")
        print("  This demonstration requires the ODA File Converter to be installed")
        print("  and accessible via the system PATH or ODA_CONVERTER_PATH variable.")
    else:
        print(f"✅ Found ODA File Converter at: {converter_path}")
        
        # Create a temporary directory for the test files.
        demo_dir = Path("demo_output/dwg_io")
        demo_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Create a simple dummy DXF file.
        dummy_dxf_path = demo_dir / "dummy_test.dxf"
        dummy_dwg_path = demo_dir / "dummy_test_converted.dwg"
        dummy_dxf_restored_path = demo_dir / "dummy_test_restored.dxf"
        
        try:
            print(f"\n1. Creating a dummy DXF file: {dummy_dxf_path}")
            doc = ezdxf.new()
            msp = doc.modelspace()
            msp.add_line((0, 0), (10, 10))
            doc.saveas(dummy_dxf_path)
            
            # 2. Convert DXF to DWG.
            print("\n2. Converting DXF to DWG...")
            convert_dxf_to_dwg(str(dummy_dxf_path), str(dummy_dwg_path))
            
            if dummy_dwg_path.exists():
                print(f"  Successfully created DWG file: {dummy_dwg_path}")
            else:
                raise IOError("DWG file was not created.")

            # 3. Convert DWG back to DXF.
            print("\n3. Converting DWG back to DXF...")
            convert_dwg_to_dxf(str(dummy_dwg_path), str(dummy_dxf_restored_path))

            if dummy_dxf_restored_path.exists():
                print(f"  Successfully restored DXF file: {dummy_dxf_restored_path}")
            else:
                raise IOError("Restored DXF file was not created.")

            print("\n--- Demonstration finished successfully! ---")

        except (RuntimeError, IOError) as e:
            print(f"\n[ERROR] An error occurred during the demonstration: {e}")
        except Exception as e:
            print(f"\n[ERROR] An unexpected error occurred: {e}")

