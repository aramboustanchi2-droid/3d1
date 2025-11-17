import ezdxf

from cad3d.dxf_extrude import extrude_dxf_closed_polylines


def make_bow_tie(path: str):
    doc = ezdxf.new(setup=True)
    msp = doc.modelspace()
    # Self-intersecting bow-tie polygon
    pts = [(0, 0), (100, 100), (0, 100), (100, 0)]
    msp.add_lwpolyline(pts, close=True)
    doc.saveas(path)


def test_detects_and_skips_self_intersection(tmp_path):
    src = tmp_path / "bowtie.dxf"
    out = tmp_path / "out.dxf"
    make_bow_tie(str(src))

    collector = []
    extrude_dxf_closed_polylines(
        str(src),
        str(out),
        height=100.0,
        detect_hard_shapes=True,
        hard_shapes_collector=collector,
    )

    assert out.exists()
    doc = ezdxf.readfile(str(out))
    meshes = list(doc.modelspace().query("MESH"))
    assert len(meshes) == 0  # skipped due to self-intersection
    assert len(collector) == 1
    assert "self_intersection" in collector[0].get("issues", "")
