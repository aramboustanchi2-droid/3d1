from pathlib import Path
from cad3d.cli import main
import ezdxf
import csv

def make_rect(path: str):
    doc = ezdxf.new(setup=True)
    msp = doc.modelspace()
    msp.add_lwpolyline([(0, 0), (100, 0), (100, 100), (0, 100)], close=True)
    doc.saveas(path)


def test_batch_extrude_report_csv(tmp_path):
    inp = tmp_path / "in"; inp.mkdir()
    out = tmp_path / "out"; out.mkdir()
    rep = tmp_path / "report.csv"
    make_rect(str(inp / "one.dxf"))
    make_rect(str(inp / "two.dxf"))

    main([
        "batch-extrude",
        "--input-dir", str(inp),
        "--output-dir", str(out),
        "--out-format", "DXF",
        "--report-csv", str(rep),
    ])

    assert rep.exists()
    rows = list(csv.reader(rep.open("r", encoding="utf-8")))
    assert len(rows) >= 3  # header + 2 files
    assert rows[0] == ["file_path", "output_path", "status", "message", "duration_sec"]
