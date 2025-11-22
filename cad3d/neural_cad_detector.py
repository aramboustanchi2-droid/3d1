from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional
import os

import cv2
import numpy as np


Point = Tuple[float, float]


@dataclass
class VectorizedResult:
    polygons: List[List[Point]]  # list of closed polylines (x, y) in mm
    segments: List[Tuple[Point, Point]]  # line segments in mm


@dataclass
class DetectorConfig:
    quality: str = "standard"  # "standard" | "high"
    min_perimeter_px: int = 20
    min_area_px: int = 100
    close_gaps_px: int = 3
    hough_min_len: int = 30
    hough_max_gap: int = 12
    canny_set: Tuple[Tuple[int,int], ...] = ((30,90),(50,150),(80,200))
    approx_factor: float = 0.005  # fraction of perimeter


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

    def __init__(self, device: str = "auto", config: Optional[DetectorConfig] = None) -> None:
        self.device = device  # not used; kept for compatibility
        self.config = config or DetectorConfig()

    def _load_image(self, path: str) -> np.ndarray:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Unable to load image: {path}")
        return img

    def _binary_ink(self, gray: np.ndarray) -> np.ndarray:
        """Produce a robust binary map of ink strokes from a grayscale page.
        Combines illumination correction, adaptive thresholding, and morphology.
        """
        # Illumination correction
        blur = cv2.medianBlur(gray, 11)
        illum = cv2.subtract(gray, blur)
        norm = cv2.normalize(illum, None, 0, 255, cv2.NORM_MINMAX)
        # Adaptive threshold (dark ink on light paper)
        thr = cv2.adaptiveThreshold(norm, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, 31, 5)
        # Clean noise and bridge gaps
        k = np.ones((3, 3), np.uint8)
        thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, k, iterations=1)
        thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, k, iterations=2)
        return thr

    def _rectify_paper(self, bin_img: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Detect the dominant quadrilateral (paper) and rectify perspective if large enough.
        Returns warped image and transform matrix, or (original, None).
        """
        try:
            cnts, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                return bin_img, None
            h, w = bin_img.shape[:2]
            area_total = h * w
            cnt = max(cnts, key=cv2.contourArea)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if len(approx) != 4 or cv2.contourArea(approx) < 0.25 * area_total:
                return bin_img, None
            quad = approx[:, 0, :].astype(np.float32)
            s = quad.sum(axis=1)
            diff = np.diff(quad, axis=1)[:, 0]
            tl = quad[np.argmin(s)]; br = quad[np.argmax(s)]
            tr = quad[np.argmin(diff)]; bl = quad[np.argmax(diff)]
            W = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
            H = int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))
            dst = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], dtype=np.float32)
            src = np.array([tl, tr, br, bl], dtype=np.float32)
            M = cv2.getPerspectiveTransform(src, dst)
            warped = cv2.warpPerspective(bin_img, M, (W, H))
            return warped, M
        except Exception:
            return bin_img, None

    def _merge_collinear_segments(self, segments: List[Tuple[int, int, int, int]], angle_tol_deg: float = 7.5, dist_tol: float = 12.0) -> List[Tuple[int, int, int, int]]:
        if not segments:
            return []
        def ang(p1, p2):
            v = (p2[0]-p1[0], p2[1]-p1[1])
            return (np.degrees(np.arctan2(v[1], v[0])) % 180.0)
        def near(a, b, tol):
            return np.hypot(a[0]-b[0], a[1]-b[1]) <= tol
        segs = [((x1,y1),(x2,y2)) for (x1,y1,x2,y2) in segments]
        used = [False]*len(segs)
        out: List[Tuple[int,int,int,int]] = []
        for i in range(len(segs)):
            if used[i]:
                continue
            a1,a2 = segs[i]
            a = ang(a1,a2)
            changed=True
            while changed:
                changed=False
                for j in range(i+1, len(segs)):
                    if used[j]:
                        continue
                    b1,b2 = segs[j]
                    b = ang(b1,b2)
                    if min(abs(a-b), 180-abs(a-b))>angle_tol_deg:
                        continue
                    if near(a2,b1,dist_tol):
                        a2=b2; used[j]=True; changed=True
                    elif near(a2,b2,dist_tol):
                        a2=b1; used[j]=True; changed=True
                    elif near(a1,b1,dist_tol):
                        a1=b2; used[j]=True; changed=True
                    elif near(a1,b2,dist_tol):
                        a1=b1; used[j]=True; changed=True
            used[i]=True
            out.append((a1[0],a1[1],a2[0],a2[1]))
        return out

    def vectorize_drawing(self, image_path: str, scale_mm_per_pixel: float = 1.0) -> VectorizedResult:
        img = self._load_image(image_path)
        h, w = img.shape[:2]

        # Enhanced preprocessing for architectural sketches
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Adaptive histogram equalization for better contrast then bilateral denoise
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray_eq = clahe.apply(gray)
        gray_blur = cv2.bilateralFilter(gray_eq, 7, 50, 50)

        # Adaptive threshold (works better across varying illumination)
        thr = cv2.adaptiveThreshold(
            gray_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 35, 5
        )

        # Binary ink map and optional perspective rectification (for desk/table backgrounds)
        ink = self._binary_ink(gray)
        ink, _ = self._rectify_paper(ink)

        # Multi-scale edge detection for better line capture
        # Dynamic multi-threshold Canny based on config
        edges = np.zeros_like(gray)
        for lo, hi in self.config.canny_set:
            edges = cv2.bitwise_or(edges, cv2.Canny(gray, lo, hi))
        # Combine with ink map
        edges = cv2.bitwise_or(edges, ink)

        # Morphological operations to connect broken lines and clean noise
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_close, iterations=2)

        kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel_open, iterations=1)

        # Detect lines using Hough Transform for better line extraction
        lines = cv2.HoughLinesP(
            edges, 1, np.pi/180, threshold=40,
            minLineLength=self.config.hough_min_len,
            maxLineGap=self.config.hough_max_gap
        )
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
        
        # Combine original edges with Hough + LSD lines
        edges = cv2.bitwise_or(edges, line_img)
        
        # Final dilation to ensure connectivity
        kernel_final = np.ones((2, 2), np.uint8)
        edges = cv2.dilate(edges, kernel_final, iterations=1)

        # Contour detection with hierarchical mode for better nested shape detection
        cnts, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        polygons: List[List[Point]] = []
        # Adjust thresholds for high quality mode
        if self.config.quality == "high":
            min_perimeter = max(10, int(0.5 * self.config.min_perimeter_px))
            min_area = max(50, int(0.5 * self.config.min_area_px))
            approx_factor = max(0.003, 0.7 * self.config.approx_factor)
        else:
            min_perimeter = self.config.min_perimeter_px
            min_area = self.config.min_area_px
            approx_factor = self.config.approx_factor

        for idx, c in enumerate(cnts):
            peri = cv2.arcLength(c, True)
            area = cv2.contourArea(c)
            
            if peri < min_perimeter or area < min_area:
                continue
            
            # More aggressive approximation for cleaner polylines
            epsilon = approx_factor * peri
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

        # Merge collinear segments to longer lines
        segments_px = self._merge_collinear_segments(segments_px)

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
