import os
from pathlib import Path

import ezdxf

from cad3d.cli import main


def make_rect(path: str):
    doc = ezdxf.new(setup=True)
    msp = doc.modelspace()
    msp.add_lwpolyline([(0, 0), (100, 0), (100, 50), (0, 50)], close=True)
    doc.saveas(path)


def test_batch_extrude_dxfs(tmp_path):
    inp = tmp_path / "in"
    out = tmp_path / "out"
    inp.mkdir()
    (inp / "a").mkdir()
    (inp / "b").mkdir()

    make_rect(str(inp / "a" / "rect1.dxf"))
    make_rect(str(inp / "b" / "rect2.dxf"))

    main([
        "batch-extrude",
        "--input-dir", str(inp),
        "--output-dir", str(out),
        "--out-format", "DXF",
        "--recurse",
    ])

    assert (out / "a" / "rect1.dxf").exists()
    assert (out / "b" / "rect2.dxf").exists()
