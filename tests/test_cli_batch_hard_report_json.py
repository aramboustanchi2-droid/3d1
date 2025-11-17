import json
from pathlib import Path
import ezdxf
from cad3d.cli import main


def make_bow_tie(path: str):
    doc = ezdxf.new(setup=True)
    msp = doc.modelspace()
    msp.add_lwpolyline([(0, 0), (100, 100), (0, 100), (100, 0)], close=True)
    doc.saveas(path)


def test_batch_hard_report_json(tmp_path):
    inp = tmp_path / "in"; inp.mkdir()
    out = tmp_path / "out"; out.mkdir()
    hard = tmp_path / "hard.json"
    make_bow_tie(str(inp / "bad.dxf"))

    main([
        "batch-extrude",
        "--input-dir", str(inp),
        "--output-dir", str(out),
        "--out-format", "DXF",
        "--detect-hard-shapes",
        "--hard-report-json", str(hard),
    ])

    assert hard.exists()
    data = json.load(hard.open("r", encoding="utf-8"))
    assert isinstance(data, list) and len(data) >= 1
    row = data[0]
    assert "file_path" in row and "issues" in row
    assert "self_intersection" in row["issues"]
