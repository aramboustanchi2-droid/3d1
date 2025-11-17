import csv
import ezdxf
from cad3d.cli import main


def make_bow_tie(path: str):
    doc = ezdxf.new(setup=True)
    msp = doc.modelspace()
    # Self-intersecting bow-tie polygon
    pts = [(0, 0), (100, 100), (0, 100), (100, 0)]
    msp.add_lwpolyline(pts, close=True)
    doc.saveas(path)


def test_cli_dxf_extrude_hard_report(tmp_path):
    src = tmp_path / "bad.dxf"
    out = tmp_path / "out.dxf"
    rep = tmp_path / "hard.csv"
    make_bow_tie(str(src))

    main([
        "dxf-extrude",
        "--input", str(src),
        "--output", str(out),
        "--detect-hard-shapes",
        "--hard-report-csv", str(rep),
    ])

    assert rep.exists()
    rows = list(csv.reader(rep.open("r", encoding="utf-8")))
    assert rows[0] == ["layer", "handle", "issues", "vertex_count"]
    assert len(rows) >= 2
    # Output file should exist but contain no meshes
    assert out.exists()
    doc = ezdxf.readfile(str(out))
    meshes = list(doc.modelspace().query("MESH"))
    assert len(meshes) == 0
