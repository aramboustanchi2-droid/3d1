"""
Tests for LandscapingDetector
"""
from pathlib import Path
import json
import math

import ezdxf
import pytest

from cad3d.landscaping_detector import (
    LandscapingDetector,
    LSElementType,
    create_landscaping_detector,
)


def test_imports():
    assert LandscapingDetector is not None
    assert LSElementType is not None


def build_ls_test_dxf(tmp_path: Path) -> Path:
    doc = ezdxf.new(setup=True)
    msp = doc.modelspace()

    # layers
    for name in [
        "LAWN", "SHRUB_BED", "PATH", "POND_WATER", "IRR_SUPPLY", "IRR_ZONE", "SPRINKLER", "TREE"
    ]:
        if name not in doc.layers:
            doc.layers.add(name)

    # Lawn 20m x 10m = 200 m2
    lawn = [(0,0), (20000,0), (20000,10000), (0,10000), (0,0)]
    msp.add_lwpolyline(lawn, dxfattribs={"layer": "LAWN"}, close=True)

    # Shrub area 10m x 10m = 100 m2
    shrub = [(21000,0), (31000,0), (31000,10000), (21000,10000), (21000,0)]
    msp.add_lwpolyline(shrub, dxfattribs={"layer": "SHRUB_BED"}, close=True)

    # Path 20m x 2m = 40 m2
    path = [(0,-3000), (20000,-3000), (20000,-1000), (0,-1000), (0,-3000)]
    msp.add_lwpolyline(path, dxfattribs={"layer": "PATH"}, close=True)

    # Pond - circle r=2000 -> ~12.57 m2
    msp.add_circle((5000, 5000), radius=2000, dxfattribs={"layer": "POND_WATER"})

    # Irrigation pipe
    msp.add_line((0,500), (20000, 500), dxfattribs={"layer": "IRR_SUPPLY"})

    # Irrigation zone 15m x 5m = 75 m2
    irr_zone = [(0, 12000), (15000, 12000), (15000, 17000), (0, 17000), (0, 12000)]
    msp.add_lwpolyline(irr_zone, dxfattribs={"layer": "IRR_ZONE"}, close=True)

    # Emitters (3)
    msp.add_circle((3000, 13000), radius=100, dxfattribs={"layer": "SPRINKLER"})
    msp.add_circle((9000, 15000), radius=100, dxfattribs={"layer": "SPRINKLER"})
    msp.add_circle((13000, 16000), radius=100, dxfattribs={"layer": "SPRINKLER"})

    # Trees (3)
    for cx, cy in [(25000, 2000), (27000, 3000), (29000, 4000)]:
        msp.add_circle((cx, cy), radius=500, dxfattribs={"layer": "TREE"})

    p = tmp_path / "landscape_case.dxf"
    doc.saveas(p)
    return p


def test_detect_elements(tmp_path):
    p = build_ls_test_dxf(tmp_path)
    doc = ezdxf.readfile(p)
    det = LandscapingDetector()
    elems = det.detect(doc)

    types = [e.element_type for e in elems]
    assert LSElementType.LAWN in types
    assert LSElementType.SHRUB_AREA in types
    assert LSElementType.HARDSCAPE_PATH in types
    assert LSElementType.WATER_FEATURE in types
    assert LSElementType.IRRIGATION_PIPE in types
    assert LSElementType.IRRIGATION_ZONE in types
    assert LSElementType.IRRIGATION_EMITTER in types
    assert LSElementType.TREE in types


def test_analyze_and_water_demand(tmp_path):
    p = build_ls_test_dxf(tmp_path)
    det = LandscapingDetector()
    res = det.analyze(str(p))

    # Areas
    assert res.area_by_type_m2.get("lawn", 0.0) == pytest.approx(200.0, rel=1e-3)
    assert res.area_by_type_m2.get("shrub_area", 0.0) == pytest.approx(100.0, rel=1e-3)
    assert res.area_by_type_m2.get("hardscape_path", 0.0) == pytest.approx(40.0, rel=1e-3)

    # Water demand: 200*5 + 100*3 + 3*50 = 1450 L/day
    assert res.water_demand_lpd == pytest.approx(1450.0, rel=1e-3)

    # Irrigation
    assert res.irrigation_zones == 1
    assert res.irrigation_emitters == 3

    # Recommendations should include water feature advice
    recs_text = "\n".join(res.recommendations)
    assert "water features" in recs_text.lower()


def test_export_json(tmp_path):
    p = build_ls_test_dxf(tmp_path)
    det = LandscapingDetector()
    res = det.analyze(str(p))
    out = tmp_path / "landscape_report.json"
    det.export_to_json(res, str(out))

    assert out.exists() and out.stat().st_size > 10
    with open(out, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert "areas_m2" in data and "counts" in data and "irrigation" in data
