import ezdxf
from ezdxf import colors as ezcolors
from cad3d.dxf_extrude import extrude_dxf_closed_polylines


def make_poly_with_aci(path: str, aci: int, layer: str = "LAYER1"):
    doc = ezdxf.new(setup=True)
    # ensure layer exists with default color
    if layer not in doc.layers:
        doc.layers.add(layer)
    msp = doc.modelspace()
    # set explicit entity color
    pts = [(0,0,0,0),(100,0,0,0),(100,50,0,0),(0,50,0,0)]
    e = msp.add_lwpolyline(pts, format="xybw", dxfattribs={"closed": True, "layer": layer, "color": aci})
    doc.saveas(path)


def make_poly_bylayer(path: str, layer: str, layer_aci: int):
    doc = ezdxf.new(setup=True)
    if layer not in doc.layers:
        doc.layers.add(layer, color=layer_aci)
    msp = doc.modelspace()
    pts = [(0,0,0,0),(50,0,0,0),(50,50,0,0),(0,50,0,0)]
    e = msp.add_lwpolyline(pts, format="xybw", dxfattribs={"closed": True, "layer": layer})
    doc.saveas(path)


def test_colorize_from_entity_aci(tmp_path):
    src = tmp_path / "ent_aci.dxf"
    out = tmp_path / "out.dxf"
    make_poly_with_aci(str(src), aci=1, layer="ENT")
    extrude_dxf_closed_polylines(str(src), str(out), height=100.0, colorize=True)
    doc = ezdxf.readfile(str(out))
    mesh = list(doc.modelspace().query("MESH"))[0]
    r,g,b = ezcolors.aci2rgb(1)
    assert mesh.dxf.get("true_color", None) == ezcolors.rgb2int((r,g,b))
    assert mesh.dxf.layer == "ENT"


def test_colorize_from_layer_aci(tmp_path):
    src = tmp_path / "bylayer.dxf"
    out = tmp_path / "out2.dxf"
    make_poly_bylayer(str(src), layer="WALLS", layer_aci=3)
    extrude_dxf_closed_polylines(str(src), str(out), height=100.0, colorize=True)
    doc = ezdxf.readfile(str(out))
    mesh = list(doc.modelspace().query("MESH"))[0]
    r,g,b = ezcolors.aci2rgb(3)
    assert mesh.dxf.get("true_color", None) == ezcolors.rgb2int((r,g,b))
    assert mesh.dxf.layer == "WALLS"


def test_split_by_color_changes_layer(tmp_path):
    src = tmp_path / "split.dxf"
    out = tmp_path / "out3.dxf"
    make_poly_bylayer(str(src), layer="WALLS", layer_aci=1)
    extrude_dxf_closed_polylines(str(src), str(out), height=100.0, split_by_color=True, colorize=True)
    doc = ezdxf.readfile(str(out))
    mesh = list(doc.modelspace().query("MESH"))[0]
    r,g,b = ezcolors.aci2rgb(1)
    assert mesh.dxf.layer == f"WALLS__COLOR_{r}_{g}_{b}"
