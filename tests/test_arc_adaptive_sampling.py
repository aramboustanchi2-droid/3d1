import math
import ezdxf
from cad3d.dxf_extrude import _approximate_lwpolyline_points


def make_arc_poly():
    # Create a simple polyline with a single arc bulge between (0,0) and (100,0)
    # bulge for 90° arc: tan(θ/4) with θ=pi/2 -> tan(pi/8)
    doc = ezdxf.new(setup=True)
    msp = doc.modelspace()
    bulge = math.tan(math.pi / 8.0)
    e = msp.add_lwpolyline([(0, 0, 0, 0), (100, 0, 0, bulge)], format="xysb", close=False)
    return e


def test_arc_max_seglen_increases_point_count():
    e = make_arc_poly()
    pts_fixed = _approximate_lwpolyline_points(e, arc_segments=4, arc_max_seglen=None)
    pts_adapt = _approximate_lwpolyline_points(e, arc_segments=4, arc_max_seglen=10.0)
    # Adaptive with small max seg length should produce >= fixed points
    assert len(pts_adapt) >= len(pts_fixed)
