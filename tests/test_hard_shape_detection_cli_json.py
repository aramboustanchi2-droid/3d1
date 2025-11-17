import json
import ezdxf
from cad3d.cli import main


def make_bow_tie(path: str):
    doc = ezdxf.new(setup=True)
    msp = doc.modelspace()
    # Self-intersecting bow-tie polygon
    pts = [(0, 0), (100, 100), (0, 100), (100, 0)]
    msp.add_lwpolyline(pts, close=True)
    doc.saveas(path)


def test_cli_dxf_extrude_hard_report_json(tmp_path):
    src = tmp_path / "bad.dxf"
    out = tmp_path / "out.dxf"
    rep = tmp_path / "hard.json"
    make_bow_tie(str(src))

    main([
        "dxf-extrude",
        "--input", str(src),
        "--output", str(out),
        "--detect-hard-shapes",
        "--hard-report-json", str(rep),
    ])

    assert rep.exists()
    data = json.load(rep.open("r", encoding="utf-8"))
    assert isinstance(data, list) and len(data) >= 1
    assert "issues" in data[0] and "self_intersection" in data[0]["issues"]
    # Output file should exist but contain no meshes
    assert out.exists()
    doc = ezdxf.readfile(str(out))
    meshes = list(doc.modelspace().query("MESH"))
    assert len(meshes) == 0
