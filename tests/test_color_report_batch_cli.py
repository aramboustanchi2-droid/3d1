import csv
import json
from pathlib import Path
import ezdxf
from cad3d.cli import main


def make_src(path: str, layer: str, aci: int):
    doc = ezdxf.new(setup=True)
    if layer not in doc.layers:
        doc.layers.add(layer, color=aci)
    msp = doc.modelspace()
    msp.add_lwpolyline([(0,0,0,0),(10,0,0,0),(10,10,0,0),(0,10,0,0)], format="xybw", dxfattribs={"closed": True, "layer": layer})
    doc.saveas(path)


essential_header = ["file_path", "target_layer", "r", "g", "b", "count"]


def test_batch_color_reports(tmp_path):
    inp = tmp_path / "in"; inp.mkdir()
    out = tmp_path / "out"; out.mkdir()
    csvp = tmp_path / "colors.csv"
    jsonp = tmp_path / "colors.json"

    make_src(str(inp / "one.dxf"), layer="WALLS", aci=1)
    make_src(str(inp / "two.dxf"), layer="COLUMNS", aci=3)

    main([
        "batch-extrude",
        "--input-dir", str(inp),
        "--output-dir", str(out),
        "--out-format", "DXF",
        "--colorize",
        "--split-by-color",
        "--color-report-csv", str(csvp),
        "--color-report-json", str(jsonp),
    ])

    assert csvp.exists() and jsonp.exists()
    rows = list(csv.reader(csvp.open("r", encoding="utf-8")))
    assert rows and rows[0] == essential_header
    # two files -> at least 2 meshes total
    total = sum(int(r[-1]) for r in rows[1:])
    assert total == 2

    data = json.load(jsonp.open("r", encoding="utf-8"))
    assert isinstance(data, list) and len(data) == 2
    for item in data:
        assert "file_path" in item
        for k in ("source_layer","target_layer","r","g","b"):
            assert k in item
