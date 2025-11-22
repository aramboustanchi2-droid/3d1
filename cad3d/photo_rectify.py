from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple
import os

import cv2
import numpy as np

from .neural_cad_detector import NeuralCADDetector, DetectorConfig, VectorizedResult


Point = Tuple[float, float]


@dataclass
class PhotoRectifyConfig:
    quality: str = "high"
    reference_width_mm: Optional[float] = None  # known real-world width of deck/paper/etc.
    target_width_px: Optional[int] = 1200  # width for rectified image


def _detect_trapezoid(gray: np.ndarray) -> Optional[np.ndarray]:
    """Heuristic: find a dominant trapezoid (bridge deck/paper) via long lines.
    Returns 4 points (tl,tr,br,bl) if found, else None.
    """
    h, w = gray.shape[:2]
    edges = cv2.Canny(gray, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=120, minLineLength=min(w, h)//4, maxLineGap=20)
    if lines is None or len(lines) < 4:
        return None

    # Cluster by orientation (two main orientation bins)
    angles = []
    segs = []
    for l in lines:
        x1,y1,x2,y2 = l[0]
        ang = (np.degrees(np.arctan2(y2-y1, x2-x1)) + 180.0) % 180.0
        angles.append(ang)
        segs.append((x1,y1,x2,y2))
    angles = np.array(angles)

    # KMeans 2 clusters on angle
    Z = angles.reshape(-1,1).astype(np.float32)
    compactness, labels, centers = cv2.kmeans(Z, 2, None, (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5), 5, cv2.KMEANS_PP_CENTERS)
    bins = {0: [], 1: []}
    for idx, lab in enumerate(labels.flatten()):
        bins[int(lab)].append(segs[idx])

    if len(bins[0]) < 2 or len(bins[1]) < 2:
        return None

    # Choose extreme lines in each bin to form a trapezoid
    def line_params(seg):
        x1,y1,x2,y2 = seg
        a = y2 - y1
        b = x1 - x2
        c = x2*y1 - x1*y2
        return a,b,c
    def distance_line_point(seg, p):
        a,b,c = line_params(seg)
        x,y = p
        return abs(a*x + b*y + c)/np.hypot(a,b)

    # pick two lines in each bin with max separation by sampling image corners
    corners = [(0,0),(w-1,0),(w-1,h-1),(0,h-1)]
    def pick_two_far(lineset):
        best = None; bestd = -1
        for i in range(len(lineset)):
            for j in range(i+1, len(lineset)):
                # average distance across corners as proxy for separation
                d = np.mean([abs(distance_line_point(lineset[i], c) - distance_line_point(lineset[j], c)) for c in corners])
                if d > bestd:
                    bestd = d; best = (lineset[i], lineset[j])
        return best

    pairA = pick_two_far(bins[0])
    pairB = pick_two_far(bins[1])
    if pairA is None or pairB is None:
        return None

    def intersect(seg1, seg2):
        a1,b1,c1 = line_params(seg1)
        a2,b2,c2 = line_params(seg2)
        det = a1*b2 - a2*b1
        if abs(det) < 1e-6:
            return None
        x = (b1*c2 - b2*c1)/det
        y = (c1*a2 - c2*a1)/det
        return (x,y)

    # Intersections produce quad corners
    tl = intersect(pairA[0], pairB[0])
    tr = intersect(pairA[1], pairB[0])
    br = intersect(pairA[1], pairB[1])
    bl = intersect(pairA[0], pairB[1])
    quad = [tl,tr,br,bl]
    if any(p is None for p in quad):
        return None
    quad = np.array(quad, dtype=np.float32)

    # Validate that all corners lie within some margin of the image bounds
    margin = max(10, int(0.02*min(w,h)))
    if np.any(quad[:,0] < -margin) or np.any(quad[:,0] > w+margin) or np.any(quad[:,1] < -margin) or np.any(quad[:,1] > h+margin):
        # still accept; crop later
        pass
    return quad


def rectify_photo(image_path: str, cfg: PhotoRectifyConfig) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    quad = _detect_trapezoid(gray)
    if quad is None:
        return img, None

    # Order quad as tl,tr,br,bl (assumed from detection order, but re-order by sums/diffs)
    pts = quad
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)[:,0]
    tl = pts[np.argmin(s)]; br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]; bl = pts[np.argmax(diff)]
    ordered = np.array([tl,tr,br,bl], dtype=np.float32)

    # Target rectangle width based on config, aspect by original quad
    W = cfg.target_width_px or 1200
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    tgt_w = float(W)
    tgt_h = float(max(200, int(W * ( (heightA+heightB)/(widthA+widthB+1e-6) ))))
    dst = np.array([[0,0],[tgt_w-1,0],[tgt_w-1,tgt_h-1],[0,tgt_h-1]], dtype=np.float32)
    H = cv2.getPerspectiveTransform(ordered, dst)
    warped = cv2.warpPerspective(img, H, (int(tgt_w), int(tgt_h)))
    return warped, H


def process_photo_to_dxf(
    image_path: str,
    out_dxf_path: str,
    reference_width_mm: Optional[float] = None,
    quality: str = "high",
    fallback_scale_mm_per_pixel: float = 1.0,
) -> Tuple[VectorizedResult, float]:
    """Rectify perspective, estimate scale if reference width given, vectorize, write DXF.
    Returns (vectorized_result, used_mm_per_pixel).
    """
    cfg = PhotoRectifyConfig(quality=quality, reference_width_mm=reference_width_mm)
    rectified, H = rectify_photo(image_path, cfg)

    # Determine scale in mm/px
    scale_mm_per_px = fallback_scale_mm_per_pixel
    if reference_width_mm is not None and H is not None:
        # Measure width in pixels by detecting two strongest vertical edges (post-rectification)
        gry = cv2.cvtColor(rectified, cv2.COLOR_BGR2GRAY)
        ed = cv2.Canny(gry, 50, 150)
        ed = cv2.dilate(ed, np.ones((3,3),np.uint8), iterations=1)
        lines = cv2.HoughLinesP(ed, 1, np.pi/180, threshold=120, minLineLength=gry.shape[1]//3, maxLineGap=20)
        if lines is not None and len(lines) >= 2:
            # cluster near-vertical
            def ang(l):
                x1,y1,x2,y2=l[0]; a=(np.degrees(np.arctan2(y2-y1,x2-x1))+180)%180; return a
            verticals = [l for l in lines if abs(90 - ang(l)) < 20]
            if len(verticals) >= 2:
                # pick two farthest in x
                xs = [ (min(l[0][0], l[0][2]), max(l[0][0], l[0][2])) for l in verticals ]
                # use midpoints
                mids = [ (x0+x1)/2.0 for (x0,x1) in xs ]
                i_min = int(np.argmin(mids)); i_max = int(np.argmax(mids))
                px_gap = abs(mids[i_max] - mids[i_min])
                if px_gap > 5:
                    scale_mm_per_px = float(reference_width_mm) / float(px_gap)

    # Vectorize rectified photo
    nd = NeuralCADDetector(config=DetectorConfig(quality=quality))
    # save temp rectified image
    tmp_path = out_dxf_path + ".rectified.png"
    cv2.imwrite(tmp_path, rectified)
    vec = nd.vectorize_drawing(tmp_path, scale_mm_per_pixel=scale_mm_per_px)
    nd.convert_to_dxf(vec, out_dxf_path)
    try:
        os.remove(tmp_path)
    except OSError:
        pass
    return vec, scale_mm_per_px
