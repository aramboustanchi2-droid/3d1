"""
Tests for VerticalTransportDetector
"""
from pathlib import Path
import json

import ezdxf
import pytest

from cad3d.vertical_transport_detector import (
    VerticalTransportDetector,
    VTElementType,
    Severity,
    create_vertical_transport_detector,
)


def test_imports():
    assert VerticalTransportDetector is not None
    assert VTElementType is not None
    assert Severity is not None


def build_vt_test_dxf(tmp_path: Path) -> Path:
    doc = ezdxf.new(setup=True)
    msp = doc.modelspace()

    # layers
    for name in ["ELEV_SHAFT", "MACHINE_ROOM", "ELEV_PIT", "ESCALATOR"]:
        if name not in doc.layers:
            doc.layers.add(name)

    # elevator shaft 1: 1800 x 1700 (compliant)
    shaft = [(0,0), (1800,0), (1800,1700), (0,1700), (0,0)]
    msp.add_lwpolyline(shaft, dxfattribs={"layer": "ELEV_SHAFT"}, close=True)

    # machine room: 1500 x 2500 => 3.75 m2 (will cause WARNING)
    mroom = [(2500,0), (4000,0), (4000,2500), (2500,2500), (2500,0)]
    msp.add_lwpolyline(mroom, dxfattribs={"layer": "MACHINE_ROOM"}, close=True)

    # elevator pit: 1800 x 500
    pit = [(0,-700), (1800,-700), (1800,-200), (0,-200), (0,-700)]
    msp.add_lwpolyline(pit, dxfattribs={"layer": "ELEV_PIT"}, close=True)

    # escalator: a line with ~40 degrees slope to trigger warning
    # start at (0, 3000) end at (2000, 4430) -> slope about 34.7? Actually atan2(1430, 2000)~35.6; let's make steeper
    msp.add_line((0, 3000), (1500, 4500), dxfattribs={"layer": "ESCALATOR"})

    path = tmp_path / "vt_case.dxf"
    doc.saveas(path)
    return path


def test_detect_elements(tmp_path):
    p = build_vt_test_dxf(tmp_path)
    doc = ezdxf.readfile(p)
    det = VerticalTransportDetector()
    elems = det.detect(doc)

    assert any(el.element_type == VTElementType.ELEVATOR_SHAFT for el in elems)
    assert any(el.element_type == VTElementType.MACHINE_ROOM for el in elems)
    assert any(el.element_type == VTElementType.PIT for el in elems)
    assert any(el.element_type == VTElementType.ESCALATOR for el in elems)


def test_analyze_and_issues(tmp_path):
    p = build_vt_test_dxf(tmp_path)
    det = VerticalTransportDetector()
    res = det.analyze(str(p))

    # counts
    assert res.counts.get(VTElementType.ELEVATOR_SHAFT.value, 0) >= 1
    assert res.counts.get(VTElementType.MACHINE_ROOM.value, 0) >= 1
    assert res.counts.get(VTElementType.PIT.value, 0) >= 1
    assert res.counts.get(VTElementType.ESCALATOR.value, 0) >= 1

    # issues should include machine room area warning and escalator slope warning
    messages = "\n".join(i.message for i in res.issues)
    assert "Machine room area" in messages
    assert "Escalator slope" in messages

    # capacity estimate should be > 0 (1 elevator assumed)
    assert res.capacity_pph_estimate > 0


def test_export_json(tmp_path):
    p = build_vt_test_dxf(tmp_path)
    det = VerticalTransportDetector()
    res = det.analyze(str(p))
    out = tmp_path / "vt_report.json"
    det.export_to_json(res, str(out))
    assert out.exists() and out.stat().st_size > 10

    with open(out, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert "counts" in data and "issues" in data
