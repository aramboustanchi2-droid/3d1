"""
Image to Depth to 3D DXF Converter.

This module provides functionality to convert a 2D image into a 3D mesh
representation by estimating its depth map. It uses a pre-trained depth
estimation model in the ONNX format (e.g., MiDaS) to infer depth from a
single image. The resulting depth map is then transformed into a 3D
triangular mesh and saved as a DXF file.

Key Functions:
- `image_to_3d_dxf`: The main high-level function that orchestrates the
  entire conversion process from an image file to a 3D DXF file.
- `_infer_depth_onnx`: Performs the core depth estimation using an ONNX model.
- `depth_to_dxf_mesh`: Converts the 2D depth map into a 3D mesh structure.

This tool is primarily intended for creating visual 3D representations from
images and is not suited for applications requiring high metric accuracy.
"""
from __future__ import annotations
import os
from pathlib import Path
from typing import Tuple

import numpy as np
import cv2
import ezdxf
from ezdxf.document import Drawing

try:
    import onnxruntime as ort
except ImportError:
    ort = None

from .config import settings
from .mesh_utils import optimize_vertices as optimize_mesh_vertices


def _load_image(
    path: str, size: int = 256
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Loads and preprocesses an image for the MiDaS depth estimation model.

    The preprocessing steps are specific to the MiDaS model and include
    resizing, normalization to the [0, 1] range, standardization using
    ImageNet statistics, and reformatting to the NCHW tensor format.

    Args:
        path: Path to the input image file.
        size: The square dimension to which the image will be resized.

    Returns:
        A tuple containing:
        - The preprocessed image as a numpy array in NCHW format.
        - A tuple with the original width and height of the image.

    Raises:
        FileNotFoundError: If the image file cannot be found or read.
    """
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image file at: {path}")

    original_h, original_w = img.shape[:2]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (size, size), interpolation=cv2.INTER_CUBIC)

    # Preprocess for MiDaS model
    img_float = img_resized.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    normalized_img = (img_float - mean) / std
    
    # Convert to NCHW (Batch, Channels, Height, Width) format
    input_tensor = np.transpose(normalized_img, (2, 0, 1))[np.newaxis, ...]
    
    return input_tensor, (original_w, original_h)


def _infer_depth_onnx(
    image_path: str, model_path: str, size: int = 256
) -> np.ndarray:
    """
    Performs depth estimation on an image using an ONNX model (e.g., MiDaS).

    Args:
        image_path: Path to the input image.
        model_path: Path to the ONNX model file.
        size: The input size expected by the model.

    Returns:
        A 2D numpy array representing the normalized depth map (values in [0, 1]).

    Raises:
        RuntimeError: If the `onnxruntime` library is not installed.
        FileNotFoundError: If the ONNX model file does not exist.
    """
    if ort is None:
        raise RuntimeError(
            "The 'onnxruntime' library is not installed. "
            "Please install it by running: pip install onnxruntime"
        )
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"ONNX model file not found at: {model_path}")

    try:
        # Prefer CPUExecutionProvider for broader compatibility and simplicity.
        providers = ["CPUExecutionProvider"]
        sess = ort.InferenceSession(model_path, providers=providers)
    except Exception:
        # Fallback to default providers if the CPU provider fails.
        sess = ort.InferenceSession(model_path)

    input_tensor, _ = _load_image(image_path, size=size)
    
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    
    prediction = sess.run([output_name], {input_name: input_tensor})[0]
    
    # Squeeze to remove batch and channel dimensions, resulting in a 2D depth map.
    depth = prediction.squeeze()
    
    # Normalize the depth map to the range [0, 1] for consistent processing.
    min_d, max_d = depth.min(), depth.max()
    if max_d - min_d > 1e-6:
        depth = (depth - min_d) / (max_d - min_d)
    else:
        depth = np.zeros_like(depth)  # Handle flat depth maps.
        
    return depth


def depth_to_dxf_mesh(
    depth: np.ndarray,
    scale: float,
    output_dxf: str,
    optimize: bool = False
) -> None:
    """
    Converts a 2D depth map into a 3D triangular mesh and saves it as a DXF file.

    The mesh is constructed as a grid, where the Z-coordinate of each vertex
    is determined by the corresponding value in the depth map, scaled by the
    `scale` factor.

    Args:
        depth: The 2D numpy array representing the normalized depth map.
        scale: A scaling factor to apply to the depth values, controlling the
               extrusion height of the 3D model.
        output_dxf: The path where the output DXF file will be saved.
        optimize: If True, deduplicates vertices to create a more efficient mesh,
                  reducing file size.
    """
    h, w = depth.shape
    
    # Create a grid of vertices. The Z-coordinate is derived from the depth map.
    x_coords, y_coords = np.meshgrid(np.arange(w), np.arange(h))
    z_coords = depth * float(scale)
    
    # The grid is created "upside down" relative to image coordinates (origin at top-left).
    # We flip the y-axis to align with standard Cartesian coordinates (origin at bottom-left).
    y_coords = h - 1 - y_coords
    
    vertices = np.stack([x_coords, y_coords, z_coords], axis=-1).reshape(-1, 3).tolist()

    # Generate faces by creating two triangles for each cell in the grid.
    faces = []
    for y in range(h - 1):
        for x in range(w - 1):
            i = y * w + x
            i_right = i + 1
            i_down = i + w
            i_diag = i_down + 1
            # Define two triangles (faces) for the quad.
            faces.append((i, i_right, i_diag))
            faces.append((i, i_diag, i_down))

    if optimize:
        vertices, faces = optimize_mesh_vertices(vertices, faces)

    doc: Drawing = ezdxf.new(setup=True)
    msp = doc.modelspace()
    mesh = msp.add_mesh()
    
    with mesh.edit_data() as mesh_data:
        mesh_data.vertices = vertices
        mesh_data.faces = faces
        
    # Ensure the output directory exists.
    Path(output_dxf).parent.mkdir(parents=True, exist_ok=True)
    doc.saveas(output_dxf)


def image_to_3d_dxf(
    image_path: str,
    output_dxf: str,
    scale: float = 1000.0,
    model_path: str | None = None,
    size: int = 256,
    optimize: bool = False
) -> None:
    """
    High-level function to convert an image file to a 3D DXF mesh.

    This function orchestrates the process of loading an image, inferring its
    depth map using an ONNX model, and converting that depth map into a 3D mesh
    saved as a DXF file.

    Args:
        image_path: Path to the input image file.
        output_dxf: Path for the output DXF file.
        scale: Scaling factor for the depth (Z-axis). A larger value results
               in a more pronounced 3D effect.
        model_path: Path to the ONNX depth estimation model. If None, it defaults
                    to the path specified by the `MIDAS_ONNX_PATH` environment
                    variable via the global settings.
        size: The input image size required by the model (e.g., 256 for MiDaS small).
        optimize: If True, optimizes the final mesh to reduce file size.

    Raises:
        FileNotFoundError: If the model path is not specified and cannot be found
                           in the environment variables, or if the path is invalid.
    """
    model = model_path or settings.midas_onnx_path
    if not model or not os.path.exists(model):
        raise FileNotFoundError(
            "MiDaS ONNX model not found. Please set the MIDAS_ONNX_PATH environment "
            "variable or provide the path using the --model-path argument."
        )
        
    print(f"Inferring depth map from '{image_path}' using model '{model}'...")
    depth_map = _infer_depth_onnx(image_path, model, size=size)
    
    print(f"Converting depth map to 3D mesh and saving to '{output_dxf}'...")
    depth_to_dxf_mesh(depth_map, scale, output_dxf, optimize=optimize)
    print("✅ Conversion complete.")


if __name__ == '__main__':
    # This block serves as a demonstration of the module's functionality.
    print("--- Image to 3D DXF Converter Demonstration ---")
    
    # Create a dummy image for testing if one doesn't exist.
    demo_dir = Path("demo_output/image_to_depth")
    demo_dir.mkdir(parents=True, exist_ok=True)
    test_image = demo_dir / "test_gradient.png"

    if not test_image.exists():
        print(f"Creating a synthetic test image at: {test_image}")
        gradient = np.linspace(0, 255, 512, dtype=np.uint8)
        img = np.tile(gradient, (512, 1))
        cv2.imwrite(str(test_image), img)

    # Check for the MiDaS model.
    model_path_str = settings.midas_onnx_path
    if not model_path_str or not os.path.exists(model_path_str):
        print("\n⚠️  Warning: MiDaS ONNX model not found.")
        print("  Please set the 'MIDAS_ONNX_PATH' environment variable to the path of your")
        print("  'midas_v2_small_256.onnx' or similar model file to run this demo.")
    else:
        try:
            output_file = demo_dir / "test_gradient_3d.dxf"
            image_to_3d_dxf(
                image_path=str(test_image),
                output_dxf=str(output_file),
                scale=200.0,  # A smaller scale for the simple gradient.
                model_path=model_path_str,
                size=256,
                optimize=True
            )
            print(f"\nSuccessfully generated 3D DXF file: {output_file}")
        except Exception as e:
            print(f"\nAn error occurred during the demonstration: {e}")
