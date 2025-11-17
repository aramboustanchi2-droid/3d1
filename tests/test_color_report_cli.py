import csv
import json
from pathlib import Path
import ezdxf
from cad3d.cli import main
from ezdxf import colors as ezcolors


def make_poly(path: str, layer: str, aci: int):
    doc = ezdxf.new(setup=True)
    if layer not in doc.layers:
        doc.layers.add(layer, color=aci)
    msp = doc.modelspace()
    pts = [(0,0,0,0),(10,0,0,0),(10,10,0,0),(0,10,0,0)]
    msp.add_lwpolyline(pts, format="xybw", dxfattribs={"closed": True, "layer": layer})
    doc.saveas(path)


def test_dxf_extrude_color_reports(tmp_path):
    src = tmp_path / "src.dxf"
    out = tmp_path / "out.dxf"
    csvp = tmp_path / "colors.csv"
    jsonp = tmp_path / "colors.json"

    # two polylines on different layers with different ACI colors
    doc = ezdxf.new(setup=True)
    l1 = "A"; l2 = "B"
    if l1 not in doc.layers:
        doc.layers.add(l1, color=1)
    if l2 not in doc.layers:
        doc.layers.add(l2, color=3)
    msp = doc.modelspace()
    msp.add_lwpolyline([(0,0,0,0),(10,0,0,0),(10,10,0,0),(0,10,0,0)], format="xybw", dxfattribs={"closed": True, "layer": l1})
    msp.add_lwpolyline([(20,0,0,0),(30,0,0,0),(30,10,0,0),(20,10,0,0)], format="xybw", dxfattribs={"closed": True, "layer": l2})
    doc.saveas(str(src))

    main([
        "dxf-extrude",
        "--input", str(src),
        "--output", str(out),
        "--colorize",
        "--split-by-color",
        "--color-report-csv", str(csvp),
        "--color-report-json", str(jsonp),
    ])

    assert out.exists()
    assert csvp.exists()
    assert jsonp.exists()

    rows = list(csv.reader(csvp.open("r", encoding="utf-8")))
    assert rows and rows[0] == ["target_layer", "r", "g", "b", "count"]
    # two meshes, possibly on two target layers
    total = sum(int(r[-1]) for r in rows[1:])
    assert total == 2

    data = json.load(jsonp.open("r", encoding="utf-8"))
    assert isinstance(data, list) and len(data) == 2
    keys = set()
    for item in data:
        for k in ("source_layer","target_layer","r","g","b"):
            assert k in item
            keys.add(k)
    assert {"source_layer","target_layer","r","g","b"}.issubset(keys)
