from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
import os

import cv2
import numpy as np


Point = Tuple[float, float]


@dataclass
class VectorizedResult:
    polygons: List[List[Point]]  # list of closed polylines (x, y) in mm
    segments: List[Tuple[Point, Point]]  # line segments in mm


class NeuralCADDetector:
    """
    Minimal, CPU-only detector that vectorizes an image into closed polylines
    using Canny edges + contour approximation. This is a lightweight fallback
    used to enable web workflows without heavy ML models.

    API compatible with the web server expectations:
    - .element_classes (placeholder)
    - vectorize_drawing(image_path, scale_mm_per_pixel)
    - convert_to_dxf(vectorized, out_dxf_path)
    """

    # Placeholder to satisfy older pipelines that inspect classes
    element_classes = ["line", "polyline", "edge"]

    def __init__(self, device: str = "auto") -> None:
        self.device = device  # not used; kept for compatibility

    def _load_image(self, path: str) -> np.ndarray:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Unable to load image: {path}")
        return img

    def vectorize_drawing(self, image_path: str, scale_mm_per_pixel: float = 1.0) -> VectorizedResult:
        img = self._load_image(image_path)
        h, w = img.shape[:2]        

        # Enhanced preprocessing for architectural sketches
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Adaptive histogram equalization for better contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        
        # Denoise while preserving edges
        gray = cv2.bilateralFilter(gray, 9, 75, 75)        
        
        # Multi-scale edge detection for better line capture
        edges1 = cv2.Canny(gray, 30, 90)
        edges2 = cv2.Canny(gray, 50, 150)
        edges3 = cv2.Canny(gray, 80, 200)
        edges = cv2.bitwise_or(cv2.bitwise_or(edges1, edges2), edges3)        
        
        # Morphological operations to connect broken lines and clean noise
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_close, iterations=2)
        
        kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel_open, iterations=1)        
        
        # Detect lines using Hough Transform for better line extraction
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=30, minLineLength=20, maxLineGap=10)
        segments_px: List[Tuple[int, int, int, int]] = []
        
        # Create enhanced edge map with detected lines
        line_img = np.zeros_like(edges)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                segments_px.append((x1, y1, x2, y2))
                cv2.line(line_img, (x1, y1), (x2, y2), 255, 2)

        # Try Line Segment Detector (if available) to capture weak lines
        try:
            lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_ADV)
            lsd_lines = lsd.detect(gray)[0]        
            if lsd_lines is not None:
                for l in lsd_lines:
                    x1, y1, x2, y2 = [int(round(v)) for v in l[0]]
                    segments_px.append((x1, y1, x2, y2))
                    cv2.line(line_img, (x1, y1), (x2, y2), 255, 1)
        except Exception:
            pass
        
        # Combine original edges with Hough lines
        edges = cv2.bitwise_or(edges, line_img)
        
        # Final dilation to ensure connectivity
        kernel_final = np.ones((2, 2), np.uint8)
        edges = cv2.dilate(edges, kernel_final, iterations=1)

        # Contour detection with hierarchical mode for better nested shape detection
        cnts, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        polygons: List[List[Point]] = []
        min_perimeter = 20  # Reduced threshold for smaller details
        min_area = 100  # Filter out tiny noise

        for idx, c in enumerate(cnts):
            peri = cv2.arcLength(c, True)
            area = cv2.contourArea(c)
            
            if peri < min_perimeter or area < min_area:
                continue
            
            # More aggressive approximation for cleaner polylines
            epsilon = 0.005 * peri  # Reduced from 0.01 for more detail
            approx = cv2.approxPolyDP(c, epsilon, True)
            
            if len(approx) < 3:
                continue
                
            pts = approx[:, 0, :]  # Nx2

            # Convert to mm with the given scale
            poly_mm: List[Point] = [(float(x) * scale_mm_per_pixel, float(h - y) * scale_mm_per_pixel) for x, y in pts]

            # Ensure closed (LWPOLYLINE will set close flag, but keep first point last for clarity)
            if poly_mm[0] != poly_mm[-1]:
                poly_mm.append(poly_mm[0])
            polygons.append(poly_mm)

        # Build segments in mm (camera coords -> CAD y-up)
        segments_mm: List[Tuple[Point, Point]] = [
            ((x1 * scale_mm_per_pixel, (h - y1) * scale_mm_per_pixel),
             (x2 * scale_mm_per_pixel, (h - y2) * scale_mm_per_pixel))
            for (x1, y1, x2, y2) in segments_px
        ]

        return VectorizedResult(polygons=polygons, segments=segments_mm)

    def convert_to_dxf(self, vectorized: VectorizedResult, out_dxf_path: str) -> None:
        import ezdxf

        # Create a clean DXF with only 3D-friendly settings
        doc = ezdxf.new(setup=True)
        msp = doc.modelspace()

        for poly in vectorized.polygons:
            # Remove duplicated consecutive points
            cleaned = [poly[0]]
            for p in poly[1:]:
                if p != cleaned[-1]:
                    cleaned.append(p)

            if len(cleaned) < 3:
                continue

            # LWPOLYLINE expects (x, y[, start_width, end_width, bulge])
            msp.add_lwpolyline(cleaned, format="xy", close=True, dxfattribs={"layer": "AI_POLY"})

        # Also write line segments if present
        if getattr(vectorized, 'segments', None):
            if 'AI_LINES' not in doc.layers:
                doc.layers.add('AI_LINES', color=3)
            for (p1, p2) in vectorized.segments:
                msp.add_line(p1, p2, dxfattribs={"layer": "AI_LINES"})

        os.makedirs(os.path.dirname(out_dxf_path) or ".", exist_ok=True)
        doc.saveas(out_dxf_path)
