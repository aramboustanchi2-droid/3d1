"""
Fire Protection Analyzer
- Detect sprinklers, hydrants, hose reels, risers, pumps, fire walls/doors, emergency exits and egress paths, fire zones
- Estimate sprinkler coverage vs protected area (simple rule-of-thumb)
- Evaluate egress path lengths within zones (very simplified)
- Export JSON report

Assumptions:
- Units: mm for drawing geometry; converted to meters/m² where needed
- Layer naming drives classification
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Tuple
import math
import json
import ezdxf


class FireElementType(Enum):
    SPRINKLER = "sprinkler"
    HYDRANT = "hydrant"
    HOSE_REEL = "hose_reel"
    RISER = "riser"
    FIRE_PUMP = "fire_pump"
    FIRE_ALARM = "fire_alarm"
    FIRE_WALL = "fire_wall"
    FIRE_DOOR = "fire_door"
    EMERGENCY_EXIT = "emergency_exit"
    EXIT_SIGN = "exit_sign"
    EGRESS_PATH = "egress_path"
    FIRE_ZONE = "fire_zone"


@dataclass
class FireElement:
    element_type: FireElementType
    layer: str
    # geometry
    center: Optional[Tuple[float, float]] = None
    radius_mm: float = 0.0
    boundary: Optional[List[Tuple[float, float]]] = None
    length_mm: float = 0.0

    properties: Dict = field(default_factory=dict)

    @property
    def area_m2(self) -> float:
        if self.boundary and len(self.boundary) >= 3:
            pts = self.boundary
            # drop repeated last point
            if pts[0] == pts[-1]:
                pts = pts[:-1]
            area = 0.0
            for i in range(len(pts)):
                j = (i + 1) % len(pts)
                area += pts[i][0] * pts[j][1]
                area -= pts[j][0] * pts[i][1]
            return abs(area) / 2.0 / 1_000_000.0
        return 0.0


@dataclass
class FireAnalysisResult:
    elements: List[FireElement]
    counts: Dict[str, int]
    area_by_zone_m2: float
    sprinklers: int
    sprinkler_coverage_m2: float
    coverage_ok: bool
    max_egress_length_m: float
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


# coverage parameters
SPRINKLER_COVERAGE_M2_PER_HEAD = 12.0  # light hazard simple assumption
MAX_EGRESS_LENGTH_M = 45.0  # simple limit assumption


class FireProtectionAnalyzer:
    def __init__(self) -> None:
        self.elements: List[FireElement] = []

    def detect(self, doc: ezdxf.document.Drawing) -> List[FireElement]:
        elems: List[FireElement] = []
        msp = doc.modelspace()
        for e in msp:
            layer = e.dxf.layer
            lup = layer.upper()
            if e.dxftype() == "CIRCLE":
                cx, cy = e.dxf.center.x, e.dxf.center.y
                r = e.dxf.radius
                if "SPRINKLER" in lup:
                    elems.append(FireElement(FireElementType.SPRINKLER, layer, center=(cx, cy), radius_mm=r))
                    continue
                if any(k in lup for k in ["HYDRANT", "FHYD"]):
                    elems.append(FireElement(FireElementType.HYDRANT, layer, center=(cx, cy), radius_mm=r))
                    continue
                if any(k in lup for k in ["HOSE", "REEL"]):
                    elems.append(FireElement(FireElementType.HOSE_REEL, layer, center=(cx, cy), radius_mm=r))
                    continue
                if any(k in lup for k in ["RISER", "STANDPIPE"]):
                    elems.append(FireElement(FireElementType.RISER, layer, center=(cx, cy), radius_mm=r))
                    continue
                if any(k in lup for k in ["EXIT", "EMERGENCY_EXIT", "E_EXIT"]):
                    elems.append(FireElement(FireElementType.EMERGENCY_EXIT, layer, center=(cx, cy), radius_mm=r))
                    continue
            if e.dxftype() == "LWPOLYLINE":
                pts = [(p[0], p[1]) for p in e.get_points()]
                closed = getattr(e, "is_closed", False) or (len(pts) >= 3 and pts[0] == pts[-1])
                if closed:
                    if any(k in lup for k in ["FIRE_ZONE", "FZONE", "ZONE_FIRE"]):
                        elems.append(FireElement(FireElementType.FIRE_ZONE, layer, boundary=pts))
                        continue
                    if any(k in lup for k in ["FIRE_WALL", "FIRE_RATED", "F_WALL"]):
                        elems.append(FireElement(FireElementType.FIRE_WALL, layer, boundary=pts))
                        continue
                    if any(k in lup for k in ["FIRE_DOOR", "F_DOOR"]):
                        elems.append(FireElement(FireElementType.FIRE_DOOR, layer, boundary=pts))
                        continue
                else:
                    if any(k in lup for k in ["EGRESS", "EXIT_PATH", "EXIT_ROUTE"]):
                        length = 0.0
                        for i in range(len(pts) - 1):
                            x1, y1 = pts[i]
                            x2, y2 = pts[i + 1]
                            length += math.hypot(x2 - x1, y2 - y1)
                        elems.append(FireElement(FireElementType.EGRESS_PATH, layer, boundary=pts, length_mm=length))
                        continue
            if e.dxftype() == "LINE":
                if any(k in lup for k in ["EGRESS", "EXIT_PATH", "EXIT_ROUTE"]):
                    x1, y1 = e.dxf.start.x, e.dxf.start.y
                    x2, y2 = e.dxf.end.x, e.dxf.end.y
                    length = math.hypot(x2 - x1, y2 - y1)
                    elems.append(FireElement(FireElementType.EGRESS_PATH, layer, length_mm=length))
                    continue
        self.elements = elems
        return elems

    def analyze(self, dxf_path: str) -> FireAnalysisResult:
        doc = ezdxf.readfile(dxf_path)
        elems = self.detect(doc)
        counts: Dict[str, int] = {}
        for el in elems:
            key = el.element_type.value
            counts[key] = counts.get(key, 0) + 1

        zones_area = float(sum(float(el.area_m2) for el in elems if el.element_type == FireElementType.FIRE_ZONE))
        sprinklers = int(counts.get(FireElementType.SPRINKLER.value, 0))
        sprinkler_cov = float(sprinklers * SPRINKLER_COVERAGE_M2_PER_HEAD)
        coverage_ok = bool(sprinkler_cov + 1e-6 >= zones_area)  # allow small epsilon

        # egress lengths
        max_egress_len_m = 0.0
        for el in elems:
            if el.element_type == FireElementType.EGRESS_PATH:
                length_m = float(el.length_mm) / 1000.0
                if length_m > max_egress_len_m:
                    max_egress_len_m = length_m

        warnings: List[str] = []
        recs: List[str] = []
        if not coverage_ok and zones_area > 0:
            warnings.append(
                f"Sprinkler coverage insufficient: heads {sprinklers} -> {sprinkler_cov:.1f} m² < zone area {zones_area:.1f} m²"
            )
            missing = math.ceil(max(0.0, zones_area - sprinkler_cov) / SPRINKLER_COVERAGE_M2_PER_HEAD)
            recs.append(f"Add ~{missing} sprinkler heads to cover the zone")
        if max_egress_len_m > MAX_EGRESS_LENGTH_M:
            warnings.append(
                f"Egress path too long: {max_egress_len_m:.1f} m > {MAX_EGRESS_LENGTH_M:.1f} m"
            )
            recs.append("Add intermediate exits or split zones to reduce egress distance")

        return FireAnalysisResult(
            elements=elems,
            counts=counts,
            area_by_zone_m2=float(zones_area),
            sprinklers=int(sprinklers),
            sprinkler_coverage_m2=float(sprinkler_cov),
            coverage_ok=bool(coverage_ok),
            max_egress_length_m=float(max_egress_len_m),
            warnings=warnings,
            recommendations=recs,
        )

    def export_to_json(self, result: FireAnalysisResult, output_path: str):
        data = {
            "counts": {str(k): int(v) for k, v in result.counts.items()},
            "fire_zone_area_m2": float(result.area_by_zone_m2),
            "sprinklers": int(result.sprinklers),
            "sprinkler_coverage_m2": float(result.sprinkler_coverage_m2),
            "coverage_ok": bool(result.coverage_ok),
            "max_egress_length_m": float(result.max_egress_length_m),
            "warnings": list(result.warnings),
            "recommendations": list(result.recommendations),
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


def create_fire_protection_analyzer() -> FireProtectionAnalyzer:
    return FireProtectionAnalyzer()
