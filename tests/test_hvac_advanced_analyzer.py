"""
Tests for HVAC Advanced Analyzer
"""
import json
from pathlib import Path

import ezdxf
import pytest

from cad3d.hvac_advanced_analyzer import (
    HVACAdvancedAnalyzer,
    HVACSpaceType,
    HVACEquipmentType,
    DuctType,
    FilterGrade,
    create_hvac_analyzer,
)


def test_imports():
    assert HVACAdvancedAnalyzer is not None
    assert HVACSpaceType is not None
    assert HVACEquipmentType is not None
    assert DuctType is not None
    assert FilterGrade is not None


def build_test_dxf(tmp_path: Path) -> Path:
    doc = ezdxf.new(setup=True)
    msp = doc.modelspace()

    # layers
    for name in [
        "OPERATING_ROOM", "HVAC_SUPPLY", "HVAC_RETURN", "HVAC_EXHAUST", "HEPA_FILTER"
    ]:
        if name not in doc.layers:
            doc.layers.add(name)

    # operating room 10x10 m
    rect = [(0,0), (10000,0), (10000,10000), (0,10000), (0,0)]
    msp.add_lwpolyline(rect, dxfattribs={"layer": "OPERATING_ROOM", "closed": True})

    # supply diffusers (12)
    for i in range(3):
        for j in range(4):
            cx = 1500 + i*3000
            cy = 1500 + j*2000
            msp.add_circle((cx, cy), radius=150, dxfattribs={"layer": "HVAC_SUPPLY"})

    # return grilles (10)
    for i in range(2):
        for j in range(5):
            cx = 7000 + i*2000
            cy = 1000 + j*1800
            msp.add_circle((cx, cy), radius=150, dxfattribs={"layer": "HVAC_RETURN"})

    # exhaust fan (1)
    msp.add_circle((9000, 9000), radius=200, dxfattribs={"layer": "HVAC_EXHAUST"})

    # HEPA filter presence
    msp.add_circle((5000, 5000), radius=100, dxfattribs={"layer": "HEPA_FILTER"})

    # a duct polyline
    msp.add_lwpolyline([(0,500), (10000, 500)], dxfattribs={"layer": "HVAC_SUPPLY"})

    path = tmp_path / "hvac_op_room.dxf"
    doc.saveas(path)
    return path


def test_detect_spaces(tmp_path):
    p = build_test_dxf(tmp_path)
    analyzer = HVACAdvancedAnalyzer()
    doc = ezdxf.readfile(p)
    spaces = analyzer.detect_spaces(doc)
    assert len(spaces) == 1
    sp = spaces[0]
    assert sp.space_type == HVACSpaceType.OPERATING_ROOM
    assert sp.area_m2 == pytest.approx(100.0, rel=1e-3)
    assert sp.volume_m3 == pytest.approx(300.0, rel=1e-3)
    assert sp.required_ach == 20.0
    assert sp.hepa_required is True


def test_detect_equipments(tmp_path):
    p = build_test_dxf(tmp_path)
    analyzer = HVACAdvancedAnalyzer()
    doc = ezdxf.readfile(p)
    eqs = analyzer.detect_equipments(doc)
    # 12 supply + 10 return + 1 exhaust + 1 hepa = 24
    assert len(eqs) >= 24
    # check a few types exist
    types = {e.equipment_type for e in eqs}
    assert HVACEquipmentType.SUPPLY_DIFFUSER in types
    assert HVACEquipmentType.RETURN_GRILLE in types
    assert HVACEquipmentType.EXHAUST_FAN in types
    assert HVACEquipmentType.HEPA_FILTER in types


def test_detect_ducts(tmp_path):
    p = build_test_dxf(tmp_path)
    analyzer = HVACAdvancedAnalyzer()
    doc = ezdxf.readfile(p)
    ducts = analyzer.detect_ducts(doc)
    assert any(d.duct_type == DuctType.SUPPLY for d in ducts)


def test_full_analysis(tmp_path):
    p = build_test_dxf(tmp_path)
    analyzer = HVACAdvancedAnalyzer()
    res = analyzer.analyze(str(p))

    assert len(res.spaces) == 1
    sp = res.spaces[0]

    # supply: 12 * 500 = 6000 cmh -> 6000/300 = 20 ACH
    assert sp.ach_actual == pytest.approx(20.0, rel=1e-3)
    assert sp.pressure_regime == "positive"  # 6000 supply vs ~5000+700 return+exhaust
    assert sp.filtration == FilterGrade.HEPA
    assert sp.compliance_status == "compliant"

    assert res.compliant_spaces == 1
    assert res.non_compliant_spaces == 0


def test_export_json(tmp_path):
    p = build_test_dxf(tmp_path)
    analyzer = HVACAdvancedAnalyzer()
    res = analyzer.analyze(str(p))

    out = tmp_path / "hvac_report.json"
    analyzer.export_to_json(res, str(out))
    assert out.exists() and out.stat().st_size > 10

    with open(out, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert "summary" in data and "spaces" in data
    assert data["summary"]["compliant"] == 1
