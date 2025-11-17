import os
import tempfile

import ezdxf

from cad3d.dxf_extrude import extrude_dxf_closed_polylines


def _make_rect_dxf(path: str) -> None:
    doc = ezdxf.new(setup=True)
    msp = doc.modelspace()
    # Create a closed LWPOLYLINE rectangle
    points = [(0, 0, 0, 0), (100, 0, 0, 0), (100, 50, 0, 0), (0, 50, 0, 0)]
    msp.add_lwpolyline(points, format="xybw", dxfattribs={"closed": True})
    doc.saveas(path)


def test_extrude_rectangle_creates_mesh(tmp_path):
    src = tmp_path / "rect2d.dxf"
    out = tmp_path / "rect3d.dxf"
    _make_rect_dxf(str(src))

    extrude_dxf_closed_polylines(str(src), str(out), height=200.0)

    assert out.exists()
    doc = ezdxf.readfile(str(out))
    meshes = list(doc.modelspace().query("MESH"))
    assert len(meshes) >= 1
