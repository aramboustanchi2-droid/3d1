"""
Landscaping Detector
- Detect trees, lawns, shrub areas, water features, hardscape paths
- Detect irrigation pipes, emitters, and zones
- Compute coverage areas and estimated daily water demand
- Export JSON report

Assumptions:
- Drawing units are millimeters. Areas converted to m². Lengths to meters as needed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple, Dict, Optional
import math
import json
import ezdxf


class LSElementType(Enum):
    TREE = "tree"
    SHRUB_AREA = "shrub_area"
    LAWN = "lawn"
    WATER_FEATURE = "water_feature"
    HARDSCAPE_PATH = "hardscape_path"
    IRRIGATION_PIPE = "irrigation_pipe"
    IRRIGATION_EMITTER = "irrigation_emitter"
    IRRIGATION_ZONE = "irrigation_zone"
    URBAN_GREEN = "urban_green"


@dataclass
class LSElement:
    element_type: LSElementType
    layer: str
    # One of the following populated depending on geometry
    boundary: Optional[List[Tuple[float, float]]] = None
    center: Optional[Tuple[float, float]] = None
    radius_mm: float = 0.0
    length_mm: float = 0.0

    properties: Dict = field(default_factory=dict)

    @property
    def area_m2(self) -> float:
        if self.boundary and len(self.boundary) >= 3:
            return _polygon_area(self.boundary) / 1_000_000.0
        if self.center is not None and self.radius_mm > 0:
            return math.pi * (self.radius_mm / 1000.0) ** 2
        return 0.0


@dataclass
class LSAnalysisResult:
    elements: List[LSElement]
    area_by_type_m2: Dict[str, float]
    counts_by_type: Dict[str, int]
    irrigation_zones: int
    irrigation_emitters: int
    water_demand_lpd: float  # liters per day estimated
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


# Typical daily water demand (liters per day)
WATER_DEMAND_RATES = {
    LSElementType.LAWN: 5.0,       # L/m²/day (high)
    LSElementType.SHRUB_AREA: 3.0, # L/m²/day (medium)
    LSElementType.URBAN_GREEN: 2.0 # L/m²/day (low)
}
TREE_WATER_LPD = 50.0  # per tree per day


class LandscapingDetector:
    def __init__(self) -> None:
        self.elements: List[LSElement] = []

    def detect(self, doc: ezdxf.document.Drawing) -> List[LSElement]:
        elems: List[LSElement] = []
        msp = doc.modelspace()
        for e in msp:
            layer = e.dxf.layer
            lup = layer.upper()

            if e.dxftype() == "CIRCLE":
                cx, cy = e.dxf.center.x, e.dxf.center.y
                r = e.dxf.radius
                # Trees or water feature center
                if "TREE" in lup:
                    elems.append(LSElement(LSElementType.TREE, layer, center=(cx, cy), radius_mm=r))
                    continue
                if any(k in lup for k in ["WATER", "POND", "FOUNTAIN", "POOL"]):
                    elems.append(LSElement(LSElementType.WATER_FEATURE, layer, center=(cx, cy), radius_mm=r))
                    continue
                if any(k in lup for k in ["EMITTER", "SPRINKLER", "DRIP"]):
                    elems.append(LSElement(LSElementType.IRRIGATION_EMITTER, layer, center=(cx, cy), radius_mm=r))
                    continue

            if e.dxftype() == "LWPOLYLINE":
                pts = [(p[0], p[1]) for p in e.get_points()]
                closed = getattr(e, "closed", getattr(e, "is_closed", False)) or (len(pts) >= 3 and pts[0] == pts[-1])
                # classify polygon areas
                if closed:
                    if any(k in lup for k in ["LAWN", "TURF"]):
                        elems.append(LSElement(LSElementType.LAWN, layer, boundary=pts))
                        continue
                    if any(k in lup for k in ["SHRUB", "PLANTER", "BED"]):
                        elems.append(LSElement(LSElementType.SHRUB_AREA, layer, boundary=pts))
                        continue
                    if any(k in lup for k in ["PATH", "WALK", "PAVER"]):
                        elems.append(LSElement(LSElementType.HARDSCAPE_PATH, layer, boundary=pts))
                        continue
                    if any(k in lup for k in ["IRR_ZONE", "IRRIGATION_ZONE", "IRR-ZONE"]):
                        elems.append(LSElement(LSElementType.IRRIGATION_ZONE, layer, boundary=pts))
                        continue
                    if any(k in lup for k in ["LANDSCAPE", "GARDEN", "GREEN"]):
                        elems.append(LSElement(LSElementType.URBAN_GREEN, layer, boundary=pts))
                        continue
                # open polyline -> pipe
                if any(k in lup for k in ["IRR", "IRRIGATION", "DRIP", "SPRINKLER"]) and not closed:
                    length = _polyline_length(pts)
                    elems.append(LSElement(LSElementType.IRRIGATION_PIPE, layer, length_mm=length))
                    continue

            if e.dxftype() == "LINE":
                if any(k in lup for k in ["IRR", "IRRIGATION", "DRIP", "SPRINKLER"]):
                    x1, y1 = e.dxf.start.x, e.dxf.start.y
                    x2, y2 = e.dxf.end.x, e.dxf.end.y
                    length = math.hypot(x2 - x1, y2 - y1)
                    elems.append(LSElement(LSElementType.IRRIGATION_PIPE, layer, length_mm=length))
                    continue

        self.elements = elems
        return elems

    def analyze(self, dxf_path: str) -> LSAnalysisResult:
        doc = ezdxf.readfile(dxf_path)
        elems = self.detect(doc)

        area_by_type: Dict[str, float] = {}
        counts_by_type: Dict[str, int] = {}
        for el in elems:
            key = el.element_type.value
            counts_by_type[key] = counts_by_type.get(key, 0) + 1
            a = el.area_m2
            if a > 0:
                area_by_type[key] = area_by_type.get(key, 0.0) + a

        irrigation_zones = counts_by_type.get(LSElementType.IRRIGATION_ZONE.value, 0)
        irrigation_emitters = counts_by_type.get(LSElementType.IRRIGATION_EMITTER.value, 0)

        # water demand
        total_water = 0.0
        for etype, rate in WATER_DEMAND_RATES.items():
            a = area_by_type.get(etype.value, 0.0)
            total_water += a * rate
        # trees
        tree_count = counts_by_type.get(LSElementType.TREE.value, 0)
        total_water += tree_count * TREE_WATER_LPD

        warnings: List[str] = []
        recs: List[str] = []
        lawn_area = area_by_type.get(LSElementType.LAWN.value, 0.0)
        if lawn_area > 0 and irrigation_emitters == 0:
            warnings.append("Lawn area detected without irrigation emitters")
            recs.append("Add sprinklers or drip emitters for lawn irrigation")
        if irrigation_zones == 0 and (irrigation_emitters > 0 or area_by_type.get(LSElementType.SHRUB_AREA.value, 0.0) > 0):
            warnings.append("Irrigation present without defined zones")
            recs.append("Define irrigation zones for proper control")
        if area_by_type.get(LSElementType.WATER_FEATURE.value, 0.0) > 0:
            recs.append("Consider recirculation systems for water features to reduce consumption")

        return LSAnalysisResult(
            elements=elems,
            area_by_type_m2=area_by_type,
            counts_by_type=counts_by_type,
            irrigation_zones=irrigation_zones,
            irrigation_emitters=irrigation_emitters,
            water_demand_lpd=total_water,
            warnings=warnings,
            recommendations=recs,
        )

    def export_to_json(self, result: LSAnalysisResult, output_path: str):
        data = {
            "areas_m2": result.area_by_type_m2,
            "counts": result.counts_by_type,
            "irrigation": {
                "zones": result.irrigation_zones,
                "emitters": result.irrigation_emitters,
            },
            "water_demand_lpd": result.water_demand_lpd,
            "warnings": result.warnings,
            "recommendations": result.recommendations,
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


# Helpers

def _polyline_length(pts: List[Tuple[float, float]]) -> float:
    length = 0.0
    for i in range(len(pts) - 1):
        x1, y1 = pts[i]
        x2, y2 = pts[i + 1]
        length += math.hypot(x2 - x1, y2 - y1)
    return length


def _polygon_area(points: List[Tuple[float, float]]) -> float:
    if len(points) < 3:
        return 0.0
    # if last point repeats first, ignore last
    if points[0] == points[-1]:
        pts = points[:-1]
    else:
        pts = points
    area = 0.0
    for i in range(len(pts)):
        j = (i + 1) % len(pts)
        area += pts[i][0] * pts[j][1]
        area -= pts[j][0] * pts[i][1]
    return abs(area) / 2.0


def create_landscaping_detector() -> LandscapingDetector:
    return LandscapingDetector()
