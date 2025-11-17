"""
Water, Wastewater & Rainwater Analyzer
- Detect sanitary fixtures, sewage pipes, manholes, pumps
- Detect rainwater pipes, roof drains, tanks/cisterns
- Compute estimated sanitary flow (fixture units -> l/s)
- Compute rainwater peak flow via Rational Method Q = C i A
- Export JSON report

Assumptions:
- Drawing units: mm; areas converted to m²; lengths to meters
- Layer naming conventions used for classification
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Tuple
import math
import json
import ezdxf


class WWRElementType(Enum):
    SANITARY_FIXTURE = "sanitary_fixture"
    SEWER_PIPE = "sewer_pipe"
    MANHOLE = "manhole"
    PUMP = "pump"
    RAIN_PIPE = "rain_pipe"
    ROOF_DRAIN = "roof_drain"
    CISTERN = "cistern"
    RAIN_CATCHMENT = "rain_catchment"  # roof/area polygons


# Typical fixture units mapping (simplified): FU -> l/s factor
FIXTURE_UNIT_TO_LPS = {
    "WC": 0.3,
    "URINAL": 0.15,
    "SINK": 0.1,
    "BASIN": 0.1,
    "SHOWER": 0.15,
    "BATH": 0.2,
}


@dataclass
class WWRItem:
    element_type: WWRElementType
    layer: str
    center: Optional[Tuple[float, float]] = None
    radius_mm: float = 0.0
    boundary: Optional[List[Tuple[float, float]]] = None
    length_mm: float = 0.0
    properties: Dict = field(default_factory=dict)

    @property
    def area_m2(self) -> float:
        if self.boundary and len(self.boundary) >= 3:
            pts = self.boundary
            if pts[0] == pts[-1]:
                pts = pts[:-1]
            area = 0.0
            for i in range(len(pts)):
                j = (i + 1) % len(pts)
                area += pts[i][0] * pts[j][1]
                area -= pts[j][0] * pts[i][1]
            return abs(area) / 2.0 / 1_000_000.0
        if self.center is not None and self.radius_mm > 0:
            return math.pi * (self.radius_mm / 1000.0) ** 2
        return 0.0


@dataclass
class WWRAnalysisResult:
    items: List[WWRItem]
    counts: Dict[str, int]
    sanitary_flow_lps: float
    rain_area_m2: float
    runoffs: Dict[str, float]  # Q_lps by catchment key
    total_rain_q_lps: float
    sewer_pipe_total_length_m: float
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


# Runoff coefficients (typical)
RUNOFF_COEFF = {
    "ROOF": 0.9,
    "PAVED": 0.8,
    "LANDSCAPE": 0.3,
}
# Rainfall intensity (example) i = 100 L/s/ha = 0.01 L/s/m² for selected return period
RAIN_INTENSITY_LPS_PER_M2 = 0.01


class WaterWasteRainAnalyzer:
    def __init__(self) -> None:
        self.items: List[WWRItem] = []

    def detect(self, doc: ezdxf.document.Drawing) -> List[WWRItem]:
        items: List[WWRItem] = []
        msp = doc.modelspace()
        for e in msp:
            layer = e.dxf.layer
            lup = layer.upper()
            if e.dxftype() == "CIRCLE":
                cx, cy = e.dxf.center.x, e.dxf.center.y
                r = e.dxf.radius
                if any(k in lup for k in ["WC", "WATER_CLOSET", "URINAL", "SINK", "BASIN", "SHOWER", "BATH", "SAN_FIX"]):
                    items.append(WWRItem(WWRElementType.SANITARY_FIXTURE, layer, center=(cx, cy), radius_mm=r))
                    continue
                if any(k in lup for k in ["ROOF_DRAIN", "RD", "RAIN_INLET"]):
                    items.append(WWRItem(WWRElementType.ROOF_DRAIN, layer, center=(cx, cy), radius_mm=r))
                    continue
                if any(k in lup for k in ["MANHOLE", "MH"]):
                    items.append(WWRItem(WWRElementType.MANHOLE, layer, center=(cx, cy), radius_mm=r))
                    continue
                if any(k in lup for k in ["PUMP"]):
                    items.append(WWRItem(WWRElementType.PUMP, layer, center=(cx, cy), radius_mm=r))
                    continue
                if any(k in lup for k in ["CISTERN", "TANK", "RAIN_TANK"]):
                    items.append(WWRItem(WWRElementType.CISTERN, layer, center=(cx, cy), radius_mm=r))
                    continue
            if e.dxftype() == "LWPOLYLINE":
                pts = [(p[0], p[1]) for p in e.get_points()]
                closed = getattr(e, "is_closed", False) or (len(pts) >= 3 and pts[0] == pts[-1])
                if closed:
                    if any(k in lup for k in ["ROOF", "RAIN_AREA", "CATCHMENT", "PAVED", "HARDSCAPE", "LANDSCAPE"]):
                        items.append(WWRItem(WWRElementType.RAIN_CATCHMENT, layer, boundary=pts))
                        continue
                else:
                    if any(k in lup for k in ["SEWER", "DRAIN", "WASTE", "RAIN_PIPE"]):
                        length = 0.0
                        for i in range(len(pts) - 1):
                            x1, y1 = pts[i]
                            x2, y2 = pts[i + 1]
                            length += math.hypot(x2 - x1, y2 - y1)
                        dtype = WWRElementType.RAIN_PIPE if "RAIN" in lup else WWRElementType.SEWER_PIPE
                        items.append(WWRItem(dtype, layer, length_mm=length))
                        continue
            if e.dxftype() == "LINE":
                if any(k in lup for k in ["SEWER", "DRAIN", "WASTE", "RAIN_PIPE"]):
                    x1, y1 = e.dxf.start.x, e.dxf.start.y
                    x2, y2 = e.dxf.end.x, e.dxf.end.y
                    length = math.hypot(x2 - x1, y2 - y1)
                    dtype = WWRElementType.RAIN_PIPE if "RAIN" in lup else WWRElementType.SEWER_PIPE
                    items.append(WWRItem(dtype, layer, length_mm=length))
                    continue
        self.items = items
        return items

    def analyze(self, dxf_path: str) -> WWRAnalysisResult:
        doc = ezdxf.readfile(dxf_path)
        items = self.detect(doc)
        counts: Dict[str, int] = {}
        for it in items:
            k = it.element_type.value
            counts[k] = counts.get(k, 0) + 1

        # sanitary flow estimation (very simplified): count fixtures by keyword mapping
        sanitary_lps = 0.0
        for it in items:
            if it.element_type == WWRElementType.SANITARY_FIXTURE:
                lname = it.layer.upper()
                for key, lps in FIXTURE_UNIT_TO_LPS.items():
                    if key in lname:
                        sanitary_lps += lps
                        break

        # rain catchments
        catchments: List[WWRItem] = [it for it in items if it.element_type == WWRElementType.RAIN_CATCHMENT]
        runoffs: Dict[str, float] = {}
        total_q = 0.0
        for idx, c in enumerate(catchments):
            area = float(c.area_m2)
            # choose coefficient by layer
            lup = c.layer.upper()
            if "ROOF" in lup:
                C = RUNOFF_COEFF["ROOF"]
            elif any(k in lup for k in ["PAVED", "HARDSCAPE"]):
                C = RUNOFF_COEFF["PAVED"]
            elif "LANDSCAPE" in lup:
                C = RUNOFF_COEFF["LANDSCAPE"]
            else:
                C = 0.6
            q = float(C) * float(RAIN_INTENSITY_LPS_PER_M2) * area  # L/s
            runoffs[f"catchment_{idx+1}"] = float(q)
            total_q += q

        sewer_len_m = sum(float(it.length_mm) for it in items if it.element_type in (WWRElementType.SEWER_PIPE,)) / 1000.0

        warnings: List[str] = []
        recs: List[str] = []
        # simple checks
        if total_q > 0 and counts.get(WWRElementType.ROOF_DRAIN.value, 0) == 0:
            warnings.append("Rain catchment detected without roof drains")
            recs.append("Add roof drains and route to rain pipes or cistern")

        return WWRAnalysisResult(
            items=items,
            counts=counts,
            sanitary_flow_lps=float(sanitary_lps),
            rain_area_m2=float(sum(float(c.area_m2) for c in catchments)),
            runoffs={k: float(v) for k, v in runoffs.items()},
            total_rain_q_lps=float(total_q),
            sewer_pipe_total_length_m=float(sewer_len_m),
            warnings=warnings,
            recommendations=recs,
        )

    def export_to_json(self, result: WWRAnalysisResult, output_path: str):
        data = {
            "counts": {str(k): int(v) for k, v in result.counts.items()},
            "sanitary_flow_lps": float(result.sanitary_flow_lps),
            "rain_area_m2": float(result.rain_area_m2),
            "runoffs_lps": {str(k): float(v) for k, v in result.runoffs.items()},
            "total_rain_q_lps": float(result.total_rain_q_lps),
            "sewer_pipe_total_length_m": float(result.sewer_pipe_total_length_m),
            "warnings": list(result.warnings),
            "recommendations": list(result.recommendations),
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


def create_wwr_analyzer() -> WaterWasteRainAnalyzer:
    return WaterWasteRainAnalyzer()
