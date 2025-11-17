"""
Tests for FireProtectionAnalyzer
"""
from pathlib import Path
import json

import ezdxf
import pytest

from cad3d.fire_protection_analyzer import (
    FireProtectionAnalyzer,
    FireElementType,
    create_fire_protection_analyzer,
)


def test_imports():
    assert FireProtectionAnalyzer is not None
    assert FireElementType is not None


def build_fire_test_dxf(tmp_path: Path) -> Path:
    doc = ezdxf.new(setup=True)
    msp = doc.modelspace()

    # layers
    for name in ["FIRE_ZONE", "SPRINKLER", "EGRESS_PATH", "FIRE_DOOR", "FIRE_WALL"]:
        if name not in doc.layers:
            doc.layers.add(name)

    # Zone 30 x 20 m => 600 m2
    zone = [(0,0), (30000,0), (30000,20000), (0,20000), (0,0)]
    msp.add_lwpolyline(zone, dxfattribs={"layer": "FIRE_ZONE"}, close=True)

    # 50 sprinklers as small circles
    for i in range(10):
        for j in range(5):
            cx = 1500 + i * 2800
            cy = 2000 + j * 3500
            msp.add_circle((cx, cy), radius=100, dxfattribs={"layer": "SPRINKLER"})

    # Egress path ~30 m
    msp.add_line((0, 1000), (30000, 1000), dxfattribs={"layer": "EGRESS_PATH"})

    # Fire door polyline (just presence)
    door = [(10000,0), (11000,0), (11000,500), (10000,500), (10000,0)]
    msp.add_lwpolyline(door, dxfattribs={"layer": "FIRE_DOOR"}, close=True)

    p = tmp_path / "fire_case.dxf"
    doc.saveas(p)
    return p


def test_detect_elements(tmp_path):
    p = build_fire_test_dxf(tmp_path)
    doc = ezdxf.readfile(p)
    ana = FireProtectionAnalyzer()
    elems = ana.detect(doc)
    types = [e.element_type for e in elems]
    assert FireElementType.FIRE_ZONE in types
    assert FireElementType.SPRINKLER in types
    assert FireElementType.EGRESS_PATH in types
    assert FireElementType.FIRE_DOOR in types


def test_analysis_coverage_ok(tmp_path):
    p = build_fire_test_dxf(tmp_path)
    ana = FireProtectionAnalyzer()
    res = ana.analyze(str(p))

    # Zone area 600 m2, sprinklers 50 => coverage 600
    assert res.area_by_zone_m2 == pytest.approx(600.0, rel=1e-3)
    assert res.sprinklers == 50
    assert res.sprinkler_coverage_m2 == pytest.approx(600.0, rel=1e-3)
    assert res.coverage_ok is True

    # Egress ~30 m < 45 m
    assert res.max_egress_length_m == pytest.approx(30.0, rel=1e-3)


def test_export_json(tmp_path):
    p = build_fire_test_dxf(tmp_path)
    ana = FireProtectionAnalyzer()
    res = ana.analyze(str(p))

    out = tmp_path / "fire_report.json"
    ana.export_to_json(res, str(out))

    assert out.exists() and out.stat().st_size > 10
    with open(out, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert "counts" in data and "fire_zone_area_m2" in data and "sprinklers" in data
