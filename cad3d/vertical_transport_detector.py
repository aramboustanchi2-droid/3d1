"""
Vertical Transportation Detector
- Elevator shafts, cabins, machine rooms, pits
- Escalators and travelators
- Basic compliance checks (minimum dimensions)
- Simple capacity estimation

Assumptions:
- Drawing units in millimeters
- Entities identified primarily by layer naming conventions
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple, Dict, Optional
import math
import json
import ezdxf


class VTElementType(Enum):
    ELEVATOR_SHAFT = "elevator_shaft"
    MACHINE_ROOM = "machine_room"
    PIT = "pit"
    ELEVATOR_CAB = "elevator_cab"
    ESCALATOR = "escalator"
    TRAVELATOR = "travelator"


class Severity(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class VTElement:
    element_type: VTElementType
    layer: str
    bbox: Tuple[float, float, float, float]  # (minx, miny, maxx, maxy)
    length_mm: float = 0.0
    slope_deg: float = 0.0
    properties: Dict = field(default_factory=dict)

    @property
    def width_mm(self) -> float:
        return self.bbox[2] - self.bbox[0]

    @property
    def depth_mm(self) -> float:
        return self.bbox[3] - self.bbox[1]

    @property
    def area_m2(self) -> float:
        return max(0.0, self.width_mm * self.depth_mm) / 1_000_000.0


@dataclass
class ComplianceIssue:
    severity: Severity
    message: str
    element: Optional[VTElement] = None


@dataclass
class VTAnalysisResult:
    elements: List[VTElement]
    issues: List[ComplianceIssue]
    counts: Dict[str, int]
    capacity_pph_estimate: int = 0  # people per hour (rough estimate)


class VerticalTransportDetector:
    # Simplified minimum recommendations (mm)
    MIN_SHAFT_WIDTH = 1600  # typical for 1000kg passenger
    MIN_SHAFT_DEPTH = 1500
    MIN_MACHINE_ROOM_AREA_M2 = 5.0
    MAX_ESCALATOR_SLOPE_DEG = 35.0
    ESCALATOR_MIN_STEP_WIDTH_MM = 600

    def __init__(self):
        self.elements: List[VTElement] = []

    def detect(self, doc: ezdxf.document.Drawing) -> List[VTElement]:
        elems: List[VTElement] = []
        msp = doc.modelspace()
        for e in msp:
            layer = e.dxf.layer
            lup = layer.upper()
            # Bounding box extraction for polylines and circles
            bbox: Optional[Tuple[float, float, float, float]] = None
            if e.dxftype() == "LWPOLYLINE":
                pts = [(p[0], p[1]) for p in e.get_points()]
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                bbox = (min(xs), min(ys), max(xs), max(ys))
            elif e.dxftype() == "CIRCLE":
                cx = e.dxf.center.x
                cy = e.dxf.center.y
                r = e.dxf.radius
                bbox = (cx - r, cy - r, cx + r, cy + r)
            elif e.dxftype() == "LINE":
                x1, y1 = e.dxf.start.x, e.dxf.start.y
                x2, y2 = e.dxf.end.x, e.dxf.end.y
                bbox = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))

            if bbox is None:
                continue

            # Elevator shaft
            if any(k in lup for k in ["ELEV", "LIFT"]) and any(k in lup for k in ["SHAFT", "HOIST", "WELL", "CORE", "CHUTE"]) or lup == "ELEV_SHAFT":
                elems.append(VTElement(VTElementType.ELEVATOR_SHAFT, layer, bbox))
                continue
            # Machine room
            if any(k in lup for k in ["MACHINE", "MOTOR"]) and any(k in lup for k in ["ROOM", "RM"]):
                elems.append(VTElement(VTElementType.MACHINE_ROOM, layer, bbox))
                continue
            # Pit
            if "PIT" in lup and any(k in lup for k in ["ELEV", "LIFT"]):
                elems.append(VTElement(VTElementType.PIT, layer, bbox))
                continue
            # Escalator / Travelator
            if "ESCALATOR" in lup or "ELEV_ESC" in lup:
                length, slope = self._entity_length_and_slope(e)
                elems.append(VTElement(VTElementType.ESCALATOR, layer, bbox, length_mm=length, slope_deg=slope))
                continue
            if "TRAVELATOR" in lup or "MOVING_WALK" in lup:
                length, slope = self._entity_length_and_slope(e)
                elems.append(VTElement(VTElementType.TRAVELATOR, layer, bbox, length_mm=length, slope_deg=slope))
                continue
        self.elements = elems
        return elems

    def analyze(self, dxf_path: str) -> VTAnalysisResult:
        doc = ezdxf.readfile(dxf_path)
        elems = self.detect(doc)
        issues: List[ComplianceIssue] = []

        # compliance checks
        for el in elems:
            if el.element_type == VTElementType.ELEVATOR_SHAFT:
                if el.width_mm < self.MIN_SHAFT_WIDTH:
                    issues.append(ComplianceIssue(Severity.CRITICAL, f"Elevator shaft width {el.width_mm:.0f} < {self.MIN_SHAFT_WIDTH}", el))
                if el.depth_mm < self.MIN_SHAFT_DEPTH:
                    issues.append(ComplianceIssue(Severity.CRITICAL, f"Elevator shaft depth {el.depth_mm:.0f} < {self.MIN_SHAFT_DEPTH}", el))
            elif el.element_type == VTElementType.MACHINE_ROOM:
                if el.area_m2 < self.MIN_MACHINE_ROOM_AREA_M2:
                    issues.append(ComplianceIssue(Severity.WARNING, f"Machine room area {el.area_m2:.1f}m² below recommended {self.MIN_MACHINE_ROOM_AREA_M2}m²", el))
            elif el.element_type == VTElementType.ESCALATOR:
                if el.slope_deg > self.MAX_ESCALATOR_SLOPE_DEG:
                    issues.append(ComplianceIssue(Severity.WARNING, f"Escalator slope {el.slope_deg:.1f}° > {self.MAX_ESCALATOR_SLOPE_DEG}°", el))

        counts: Dict[str, int] = {}
        for t in VTElementType:
            counts[t.value] = sum(1 for el in elems if el.element_type == t)

        # capacity estimation (very rough):
        # passenger elevator ~ 1000kg -> 13 persons, cycle ~30s -> 26 p/min -> 1560 pph per car
        # escalator ~ 90-120 p/min depending on speed/step width; we use 100 p/min per escalator
        cars = counts.get(VTElementType.ELEVATOR_SHAFT.value, 0)
        escalators = counts.get(VTElementType.ESCALATOR.value, 0)
        capacity_pph = int(cars * 1560 + escalators * 6000)  # escalator 100 p/min = 6000 pph both directions not assumed

        return VTAnalysisResult(elements=elems, issues=issues, counts=counts, capacity_pph_estimate=capacity_pph)

    def export_to_json(self, result: VTAnalysisResult, output_path: str):
        data = {
            "counts": result.counts,
            "issues": [
                {
                    "severity": i.severity.value,
                    "message": i.message,
                    "element": {
                        "type": i.element.element_type.value if i.element else None,
                        "layer": i.element.layer if i.element else None,
                        "bbox": i.element.bbox if i.element else None,
                    },
                }
                for i in result.issues
            ],
            "capacity_pph": result.capacity_pph_estimate,
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _entity_length_and_slope(self, e) -> Tuple[float, float]:
        # returns (length_mm, slope_deg)
        if e.dxftype() == "LINE":
            x1, y1 = e.dxf.start.x, e.dxf.start.y
            x2, y2 = e.dxf.end.x, e.dxf.end.y
            dx, dy = x2 - x1, y2 - y1
            length = math.hypot(dx, dy)
            angle = math.degrees(math.atan2(dy, dx))
            slope = abs(angle)
            return length, slope
        elif e.dxftype() == "LWPOLYLINE":
            pts = [(p[0], p[1]) for p in e.get_points()]
            length = 0.0
            for i in range(len(pts) - 1):
                p1, p2 = pts[i], pts[i + 1]
                length += math.hypot(p2[0] - p1[0], p2[1] - p1[1])
            # approximate slope using first segment
            if len(pts) >= 2:
                dx, dy = pts[1][0] - pts[0][0], pts[1][1] - pts[0][1]
                slope = abs(math.degrees(math.atan2(dy, dx)))
            else:
                slope = 0.0
            return length, slope
        else:
            return 0.0, 0.0


def create_vertical_transport_detector() -> VerticalTransportDetector:
    return VerticalTransportDetector()
