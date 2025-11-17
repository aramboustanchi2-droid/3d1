from __future__ import annotations
from typing import Tuple
import os

import numpy as np
import cv2
import ezdxf

try:
    import onnxruntime as ort  # type: ignore
except Exception:  # pragma: no cover
    ort = None

from .config import settings


def _load_image(path: str, size: int = 256) -> Tuple[np.ndarray, np.ndarray]:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    h, w = img.shape[:2]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (size, size), interpolation=cv2.INTER_AREA)
    inp = img_resized.astype(np.float32) / 255.0
    # MiDaS small expects NCHW with mean/std approx
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    inp = (inp - mean) / std
    inp = np.transpose(inp, (2, 0, 1))[None, ...]
    return inp, np.array([w, h], dtype=np.int32)


def _infer_depth_onnx(image_path: str, model_path: str, size: int = 256) -> np.ndarray:
    if ort is None:
        raise RuntimeError("onnxruntime is not available. Install it to use image->3D.")
    # Try CPU provider, fallback to default if needed
    try:
        sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])  # type: ignore
    except Exception:
        sess = ort.InferenceSession(model_path)  # type: ignore
    inp, orig_wh = _load_image(image_path, size=size)
    inp_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name
    pred = sess.run([out_name], {inp_name: inp})[0]
    depth = pred[0, 0]
    # Normalize
    depth = (depth - depth.min()) / max(depth.max() - depth.min(), 1e-6)
    return depth


def depth_to_dxf_mesh(depth: np.ndarray, scale: float, output_dxf: str, optimize: bool = False) -> None:
    h, w = depth.shape
    # Create vertices on a grid, z = depth * scale
    xs, ys = np.meshgrid(np.arange(w), np.arange(h))
    zs = depth * float(scale)
    vertices = np.stack([xs, ys, zs], axis=-1).reshape(-1, 3).astype(float).tolist()

    # Faces as two triangles per cell
    faces = []
    for y in range(h - 1):
        for x in range(w - 1):
            i = y * w + x
            i_right = i + 1
            i_down = i + w
            i_diag = i_down + 1
            faces.append((i, i_right, i_diag))
            faces.append((i, i_diag, i_down))

    # Optional vertex optimization can reduce size for near-duplicate vertices
    if optimize:
        try:
            from .mesh_utils import optimize_vertices as _opt
            vertices, faces = _opt(vertices, faces)
        except Exception:
            pass

    doc = ezdxf.new(setup=True)
    msp = doc.modelspace()
    mesh = msp.add_mesh()
    with mesh.edit_data() as mesh_data:
        mesh_data.vertices = vertices
        mesh_data.faces = faces
    doc.saveas(output_dxf)


def image_to_3d_dxf(image_path: str, output_dxf: str, scale: float = 1000.0, model_path: str | None = None, size: int = 256, optimize: bool = False) -> None:
    model = model_path or settings.midas_onnx_path
    if not model or not os.path.exists(model):
        raise FileNotFoundError(
            "MiDaS ONNX model not found. Set MIDAS_ONNX_PATH or pass --model-path."
        )
    depth = _infer_depth_onnx(image_path, model, size=size)
    depth_to_dxf_mesh(depth, scale, output_dxf, optimize=optimize)
