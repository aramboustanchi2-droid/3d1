"""
HVAC Advanced Analyzer - تهویه پیشرفته و کنترل کیفیت هوا

Features:
- Detect HVAC spaces (Operating Room, Isolation, Cleanroom, Lab, Patient Room, ICU, Pharmacy, Data Center, Office, Classroom)
- Detect equipments (AHU, FCU, Supply Diffuser, Return Grille, Exhaust Fan, HEPA Filter, Filter Box, UV Unit)
- Detect ducts (Supply / Return / Exhaust)
- Compute required ACH (air changes per hour) vs actual ACH
- Compute pressure regime (positive/negative/neutral)
- HEPA/MERV filtration requirements
- Export JSON report

Notes:
- Units: drawing units assumed to be millimeters for geometry; flow in m3/h.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Tuple, Optional
import json
from pathlib import Path

import ezdxf


class HVACSpaceType(Enum):
    OPERATING_ROOM = "operating_room"
    ISOLATION_ROOM = "isolation_room"
    CLEANROOM = "cleanroom"
    LABORATORY = "laboratory"
    PATIENT_ROOM = "patient_room"
    ICU = "icu"
    PHARMACY = "pharmacy"
    DATA_CENTER = "data_center"
    OFFICE = "office"
    CLASSROOM = "classroom"
    UNKNOWN = "unknown"


class HVACEquipmentType(Enum):
    AHU = "ahu"
    FCU = "fcu"
    SUPPLY_DIFFUSER = "supply_diffuser"
    RETURN_GRILLE = "return_grille"
    EXHAUST_FAN = "exhaust_fan"
    HEPA_FILTER = "hepa_filter"
    FILTER_BOX = "filter_box"
    UV_UNIT = "uv_unit"


class DuctType(Enum):
    SUPPLY = "supply"
    RETURN = "return"
    EXHAUST = "exhaust"


class FilterGrade(Enum):
    MERV8 = "MERV 8"
    MERV13 = "MERV 13"
    MERV14 = "MERV 14"
    HEPA = "HEPA"
    ULPA = "ULPA"


@dataclass
class HVACEquipment:
    equipment_type: HVACEquipmentType
    location: Tuple[float, float]
    layer: str = ""
    flow_cmh: float = 0.0  # cubic meters per hour
    properties: Dict = field(default_factory=dict)


@dataclass
class HVACDuctSegment:
    duct_type: DuctType
    length_mm: float
    layer: str = ""


@dataclass
class HVACSpace:
    space_type: HVACSpaceType
    name: str
    boundary: List[Tuple[float, float]]
    area_m2: float
    height_m: float
    volume_m3: float
    required_ach: float = 0.0
    hepa_required: bool = False
    desired_pressure: str = "neutral"  # positive/negative/neutral

    # measured
    supply_flow_cmh: float = 0.0
    return_flow_cmh: float = 0.0
    exhaust_flow_cmh: float = 0.0

    ach_actual: float = 0.0
    pressure_regime: str = "neutral"
    filtration: FilterGrade = FilterGrade.MERV8

    equipments: List[HVACEquipment] = field(default_factory=list)

    compliance_status: str = "unknown"


@dataclass
class HVACAnalysisResult:
    spaces: List[HVACSpace]
    equipments: List[HVACEquipment]
    ducts: List[HVACDuctSegment]

    total_supply_cmh: float = 0.0
    total_return_cmh: float = 0.0
    total_exhaust_cmh: float = 0.0

    compliant_spaces: int = 0
    non_compliant_spaces: int = 0

    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


class HVACAdvancedAnalyzer:
    # Typical ACH requirements (example values)
    ACH_REQUIREMENTS = {
        HVACSpaceType.OPERATING_ROOM: dict(ach=20.0, hepa=True, pressure="positive"),
        HVACSpaceType.ISOLATION_ROOM: dict(ach=12.0, hepa=True, pressure="negative"),
        HVACSpaceType.CLEANROOM: dict(ach=60.0, hepa=True, pressure="positive"),
        HVACSpaceType.LABORATORY: dict(ach=12.0, hepa=False, pressure="negative"),
        HVACSpaceType.PATIENT_ROOM: dict(ach=6.0, hepa=False, pressure="neutral"),
        HVACSpaceType.ICU: dict(ach=12.0, hepa=False, pressure="neutral"),
        HVACSpaceType.PHARMACY: dict(ach=12.0, hepa=True, pressure="positive"),
        HVACSpaceType.DATA_CENTER: dict(ach=10.0, hepa=False, pressure="neutral"),
        HVACSpaceType.OFFICE: dict(ach=6.0, hepa=False, pressure="neutral"),
        HVACSpaceType.CLASSROOM: dict(ach=6.0, hepa=False, pressure="neutral"),
    }

    # Default flows per equipment (m3/h)
    DEFAULT_FLOWS = {
        HVACEquipmentType.SUPPLY_DIFFUSER: 500.0,
        HVACEquipmentType.RETURN_GRILLE: 500.0,
        HVACEquipmentType.EXHAUST_FAN: 700.0,
        HVACEquipmentType.AHU: 5000.0,
        HVACEquipmentType.FCU: 2000.0,
        HVACEquipmentType.HEPA_FILTER: 0.0,
        HVACEquipmentType.FILTER_BOX: 0.0,
        HVACEquipmentType.UV_UNIT: 0.0,
    }

    def __init__(self):
        self.spaces: List[HVACSpace] = []
        self.equipments: List[HVACEquipment] = []
        self.ducts: List[HVACDuctSegment] = []

    def detect_spaces(self, doc: ezdxf.document.Drawing) -> List[HVACSpace]:
        spaces: List[HVACSpace] = []
        msp = doc.modelspace()

        for e in msp:
            if e.dxftype() not in ("LWPOLYLINE", "POLYLINE"):
                continue
            if not getattr(e, "is_closed", False):
                continue
            layer_upper = e.dxf.layer.upper()
            if not any(k in layer_upper for k in [
                "OPERATING", "ISOLATION", "CLEANROOM", "LAB", "PATIENT", "ICU", "PHARMACY", "DATA", "SERVER", "OFFICE", "CLASSROOM"
            ]):
                continue

            points = []
            if e.dxftype() == "LWPOLYLINE":
                points = [(p[0], p[1]) for p in e.get_points()]
            if len(points) < 3:
                continue

            space_type = self._identify_space_type(layer_upper)
            area = self._polygon_area(points) / 1_000_000.0  # mm^2 -> m^2
            height_m = 3.0
            volume = area * height_m

            req = self.ACH_REQUIREMENTS.get(space_type, dict(ach=6.0, hepa=False, pressure="neutral"))
            space = HVACSpace(
                space_type=space_type,
                name=e.dxf.layer,
                boundary=points,
                area_m2=area,
                height_m=height_m,
                volume_m3=volume,
                required_ach=req["ach"],
                hepa_required=req["hepa"],
                desired_pressure=req["pressure"],
            )
            spaces.append(space)

        self.spaces = spaces
        return spaces

    def detect_equipments(self, doc: ezdxf.document.Drawing) -> List[HVACEquipment]:
        equips: List[HVACEquipment] = []
        msp = doc.modelspace()

        for e in msp:
            layer_upper = e.dxf.layer.upper()
            etype: Optional[HVACEquipmentType] = None
            if any(k in layer_upper for k in ["SUPPLY", "DIFFUSER"]):
                etype = HVACEquipmentType.SUPPLY_DIFFUSER
            elif any(k in layer_upper for k in ["RETURN", "GRILLE"]):
                etype = HVACEquipmentType.RETURN_GRILLE
            elif "EXHAUST" in layer_upper:
                etype = HVACEquipmentType.EXHAUST_FAN
            elif "HEPA" in layer_upper:
                etype = HVACEquipmentType.HEPA_FILTER
            elif "FILTER" in layer_upper:
                etype = HVACEquipmentType.FILTER_BOX
            elif "AHU" in layer_upper:
                etype = HVACEquipmentType.AHU
            elif "FCU" in layer_upper:
                etype = HVACEquipmentType.FCU
            elif "UV" in layer_upper:
                etype = HVACEquipmentType.UV_UNIT

            if etype is None:
                continue

            if e.dxftype() == "CIRCLE":
                loc = (e.dxf.center.x, e.dxf.center.y)
            elif e.dxftype() == "INSERT":
                loc = (e.dxf.insert.x, e.dxf.insert.y)
            elif e.dxftype() in ("POINT",):
                loc = (e.dxf.location.x, e.dxf.location.y)
            elif e.dxftype() in ("LWPOLYLINE",):
                pts = [(p[0], p[1]) for p in e.get_points()]
                loc = pts[0]
            else:
                continue

            flow = self.DEFAULT_FLOWS.get(etype, 0.0)
            equip = HVACEquipment(equipment_type=etype, location=loc, layer=e.dxf.layer, flow_cmh=flow)
            equips.append(equip)

        self.equipments = equips
        return equips

    def detect_ducts(self, doc: ezdxf.document.Drawing) -> List[HVACDuctSegment]:
        ducts: List[HVACDuctSegment] = []
        msp = doc.modelspace()
        for e in msp:
            if e.dxftype() not in ("LINE", "LWPOLYLINE"):
                continue
            layer_upper = e.dxf.layer.upper()
            dtype: Optional[DuctType] = None
            if "SUPPLY" in layer_upper:
                dtype = DuctType.SUPPLY
            elif "RETURN" in layer_upper:
                dtype = DuctType.RETURN
            elif "EXHAUST" in layer_upper:
                dtype = DuctType.EXHAUST
            if dtype is None:
                continue

            length = 0.0
            if e.dxftype() == "LINE":
                p1 = (e.dxf.start.x, e.dxf.start.y)
                p2 = (e.dxf.end.x, e.dxf.end.y)
                length = ((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2) ** 0.5
            elif e.dxftype() == "LWPOLYLINE":
                pts = [(p[0], p[1]) for p in e.get_points()]
                for i in range(len(pts)-1):
                    p1, p2 = pts[i], pts[i+1]
                    length += ((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2) ** 0.5

            ducts.append(HVACDuctSegment(duct_type=dtype, length_mm=length, layer=e.dxf.layer))
        self.ducts = ducts
        return ducts

    def analyze(self, dxf_path: str) -> HVACAnalysisResult:
        doc = ezdxf.readfile(dxf_path)
        spaces = self.detect_spaces(doc)
        equips = self.detect_equipments(doc)
        ducts = self.detect_ducts(doc)

        # assign equipments to spaces
        for sp in spaces:
            sp.equipments = []
        for eq in equips:
            for sp in spaces:
                if self._point_in_polygon(eq.location, sp.boundary):
                    sp.equipments.append(eq)
                    if eq.equipment_type == HVACEquipmentType.SUPPLY_DIFFUSER:
                        sp.supply_flow_cmh += eq.flow_cmh
                    elif eq.equipment_type == HVACEquipmentType.RETURN_GRILLE:
                        sp.return_flow_cmh += eq.flow_cmh
                    elif eq.equipment_type == HVACEquipmentType.EXHAUST_FAN:
                        sp.exhaust_flow_cmh += eq.flow_cmh
                    if eq.equipment_type in (HVACEquipmentType.HEPA_FILTER,):
                        sp.filtration = FilterGrade.HEPA
            
        # compute ACH, pressure, compliance
        compliant = 0
        non_compliant = 0
        warnings: List[str] = []
        recs: List[str] = []
        for sp in spaces:
            sp.ach_actual = (sp.supply_flow_cmh / sp.volume_m3) if sp.volume_m3 > 0 else 0.0
            delta = sp.supply_flow_cmh - (sp.return_flow_cmh + sp.exhaust_flow_cmh)
            if delta > 100:  # threshold
                sp.pressure_regime = "positive"
            elif delta < -100:
                sp.pressure_regime = "negative"
            else:
                sp.pressure_regime = "neutral"

            # filtration
            if sp.filtration != FilterGrade.HEPA and sp.hepa_required:
                warnings.append(f"{sp.name}: HEPA required but not found")
                recs.append(f"{sp.name}: add HEPA filter to meet requirement")

            # compliance on ACH and pressure
            ach_ok = sp.ach_actual >= sp.required_ach
            pr_ok = (sp.desired_pressure == sp.pressure_regime) or (sp.desired_pressure == "neutral" and sp.pressure_regime == "neutral")
            if ach_ok and pr_ok and (not sp.hepa_required or sp.filtration == FilterGrade.HEPA):
                sp.compliance_status = "compliant"
                compliant += 1
            else:
                sp.compliance_status = "non_compliant"
                non_compliant += 1
                if not ach_ok:
                    recs.append(f"{sp.name}: increase supply to reach {sp.required_ach:.1f} ACH (current {sp.ach_actual:.1f})")
                if not pr_ok:
                    recs.append(f"{sp.name}: adjust return/exhaust for {sp.desired_pressure} pressure")

        result = HVACAnalysisResult(
            spaces=spaces,
            equipments=equips,
            ducts=ducts,
            total_supply_cmh=sum(sp.supply_flow_cmh for sp in spaces),
            total_return_cmh=sum(sp.return_flow_cmh for sp in spaces),
            total_exhaust_cmh=sum(sp.exhaust_flow_cmh for sp in spaces),
            compliant_spaces=compliant,
            non_compliant_spaces=non_compliant,
            warnings=warnings,
            recommendations=recs,
        )
        return result

    def export_to_json(self, result: HVACAnalysisResult, output_path: str):
        data = {
            "summary": {
                "spaces": len(result.spaces),
                "total_supply_cmh": result.total_supply_cmh,
                "total_return_cmh": result.total_return_cmh,
                "total_exhaust_cmh": result.total_exhaust_cmh,
                "compliant": result.compliant_spaces,
                "non_compliant": result.non_compliant_spaces,
            },
            "spaces": [
                {
                    "name": sp.name,
                    "type": sp.space_type.value,
                    "area_m2": sp.area_m2,
                    "volume_m3": sp.volume_m3,
                    "required_ach": sp.required_ach,
                    "ach_actual": sp.ach_actual,
                    "desired_pressure": sp.desired_pressure,
                    "pressure_regime": sp.pressure_regime,
                    "supply_cmh": sp.supply_flow_cmh,
                    "return_cmh": sp.return_flow_cmh,
                    "exhaust_cmh": sp.exhaust_flow_cmh,
                    "filtration": sp.filtration.value,
                    "hepa_required": sp.hepa_required,
                    "compliance": sp.compliance_status,
                    "equipments": [eq.equipment_type.value for eq in sp.equipments],
                }
                for sp in result.spaces
            ],
            "warnings": result.warnings,
            "recommendations": result.recommendations,
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    # helpers
    def _identify_space_type(self, layer_upper: str) -> HVACSpaceType:
        if "OPERATING" in layer_upper or "OR" in layer_upper:
            return HVACSpaceType.OPERATING_ROOM
        if "ISOLATION" in layer_upper:
            return HVACSpaceType.ISOLATION_ROOM
        if "CLEANROOM" in layer_upper or "ISO" in layer_upper:
            return HVACSpaceType.CLEANROOM
        if "LAB" in layer_upper:
            return HVACSpaceType.LABORATORY
        if "PATIENT" in layer_upper:
            return HVACSpaceType.PATIENT_ROOM
        if "ICU" in layer_upper:
            return HVACSpaceType.ICU
        if "PHARMACY" in layer_upper:
            return HVACSpaceType.PHARMACY
        if "DATA" in layer_upper or "SERVER" in layer_upper:
            return HVACSpaceType.DATA_CENTER
        if "OFFICE" in layer_upper:
            return HVACSpaceType.OFFICE
        if "CLASSROOM" in layer_upper or "CLASS" in layer_upper:
            return HVACSpaceType.CLASSROOM
        return HVACSpaceType.UNKNOWN

    def _polygon_area(self, pts: List[Tuple[float, float]]) -> float:
        if len(pts) < 3:
            return 0.0
        area = 0.0
        for i in range(len(pts)):
            j = (i + 1) % len(pts)
            area += pts[i][0] * pts[j][1]
            area -= pts[j][0] * pts[i][1]
        return abs(area) / 2.0

    def _point_in_polygon(self, p: Tuple[float, float], poly: List[Tuple[float, float]]) -> bool:
        x, y = p
        inside = False
        n = len(poly)
        p1x, p1y = poly[0]
        for i in range(n+1):
            p2x, p2y = poly[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside


def create_hvac_analyzer() -> HVACAdvancedAnalyzer:
    return HVACAdvancedAnalyzer()
