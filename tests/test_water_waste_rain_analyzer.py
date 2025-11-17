import json
import math
import ezdxf
import pytest

from cad3d.water_waste_rain_analyzer import WaterWasteRainAnalyzer, RAIN_INTENSITY_LPS_PER_M2


def create_square(msp, origin, size_mm, layer, closed=True):
    x, y = origin
    pts = [(x, y), (x + size_mm, y), (x + size_mm, y + size_mm), (x, y + size_mm), (x, y)]
    msp.add_lwpolyline(pts, format="xy", dxfattribs={"layer": layer, "closed": closed})


def create_rect(msp, origin, w_mm, h_mm, layer, closed=True):
    x, y = origin
    pts = [(x, y), (x + w_mm, y), (x + w_mm, y + h_mm), (x, y + h_mm), (x, y)]
    msp.add_lwpolyline(pts, format="xy", dxfattribs={"layer": layer, "closed": closed})


def test_wwr_analysis_basic(tmp_path):
    # Build a synthetic DXF
    doc = ezdxf.new(setup=True)
    msp = doc.modelspace()

    # Catchments
    # Roof: 10m x 10m = 100 m^2, C=0.9
    create_square(msp, (0, 0), 10000, layer="ROOF", closed=True)
    # Paved: 5m x 8m = 40 m^2, C=0.8
    create_rect(msp, (20000, 0), 5000, 8000, layer="PAVED", closed=True)
    # Landscape: 4m x 3m = 12 m^2, C=0.3
    create_rect(msp, (0, 20000), 4000, 3000, layer="LANDSCAPE", closed=True)

    # Sewer pipe: 10m length line
    msp.add_line((0, -10000), (10000, -10000), dxfattribs={"layer": "SEWER"})

    # Roof drain
    msp.add_circle((5000, 5000), radius=100, dxfattribs={"layer": "ROOF_DRAIN"})

    # Sanitary fixtures
    msp.add_circle((30000, 30000), radius=150, dxfattribs={"layer": "WC"})
    msp.add_circle((31000, 30000), radius=100, dxfattribs={"layer": "SINK"})
    msp.add_circle((32000, 30000), radius=100, dxfattribs={"layer": "SHOWER"})

    dxf_path = tmp_path / "wwr_test.dxf"
    doc.saveas(dxf_path)

    analyzer = WaterWasteRainAnalyzer()
    result = analyzer.analyze(str(dxf_path))

    # Counts
    assert result.counts.get("rain_catchment", 0) == 3
    assert result.counts.get("roof_drain", 0) == 1
    assert result.counts.get("sanitary_fixture", 0) == 3

    # Sewer pipe length
    assert math.isclose(result.sewer_pipe_total_length_m, 10.0, rel_tol=1e-6)

    # Sanitary flow: WC 0.3 + SINK 0.1 + SHOWER 0.15 = 0.55 L/s
    assert math.isclose(result.sanitary_flow_lps, 0.55, rel_tol=1e-6)

    # Rain area and total Q
    total_area = 100 + 40 + 12  # m^2
    assert math.isclose(result.rain_area_m2, total_area, rel_tol=1e-6)

    expected_q = 0.9 * RAIN_INTENSITY_LPS_PER_M2 * 100 + 0.8 * RAIN_INTENSITY_LPS_PER_M2 * 40 + 0.3 * RAIN_INTENSITY_LPS_PER_M2 * 12
    assert math.isclose(result.total_rain_q_lps, expected_q, rel_tol=1e-6)


def test_wwr_export_json(tmp_path):
    doc = ezdxf.new(setup=True)
    msp = doc.modelspace()
    create_square(msp, (0, 0), 10000, layer="ROOF", closed=True)
    msp.add_circle((5000, 5000), radius=100, dxfattribs={"layer": "ROOF_DRAIN"})
    dxf_path = tmp_path / "wwr_export.dxf"
    doc.saveas(dxf_path)

    analyzer = WaterWasteRainAnalyzer()
    result = analyzer.analyze(str(dxf_path))

    out_path = tmp_path / "wwr_report.json"
    analyzer.export_to_json(result, str(out_path))

    with open(out_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert "counts" in data and isinstance(data["counts"], dict)
    assert "total_rain_q_lps" in data
    assert data["counts"].get("roof_drain", 0) == 1
    assert data["counts"].get("rain_catchment", 0) == 1
