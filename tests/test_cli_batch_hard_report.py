import csv
from pathlib import Path
import ezdxf
from cad3d.cli import main


def make_bow_tie(path: str):
    doc = ezdxf.new(setup=True)
    msp = doc.modelspace()
    msp.add_lwpolyline([(0, 0), (100, 100), (0, 100), (100, 0)], close=True)
    doc.saveas(path)


def test_batch_hard_report_csv(tmp_path):
    inp = tmp_path / "in"; inp.mkdir()
    out = tmp_path / "out"; out.mkdir()
    hard = tmp_path / "hard.csv"
    make_bow_tie(str(inp / "bad.dxf"))

    main([
        "batch-extrude",
        "--input-dir", str(inp),
        "--output-dir", str(out),
        "--out-format", "DXF",
        "--detect-hard-shapes",
        "--hard-report-csv", str(hard),
    ])

    assert hard.exists()
    rows = list(csv.reader(hard.open("r", encoding="utf-8")))
    assert rows and rows[0] == ["file_path", "layer", "handle", "issues", "vertex_count"]
    assert any("self_intersection" in r[3] for r in rows[1:])
