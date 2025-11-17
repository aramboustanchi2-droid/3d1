from __future__ import annotations
import math
from typing import List, Tuple

import ezdxf
from ezdxf import colors as ezcolors
from ezdxf.entities import LWPolyline

from .mesh_utils import build_prism_mesh, optimize_vertices
from .mesh_utils import build_prism_mesh, optimize_vertices, detect_polygon_issues, polygon_area


def _approximate_lwpolyline_points(
    e: LWPolyline,
    arc_segments: int = 12,
    arc_max_seglen: float | None = None,
) -> List[Tuple[float, float]]:
    """
    Returns a list of (x, y) points approximating the LWPOLYLINE, including arc segments.
    """
    pts: List[Tuple[float, float]] = []
    if not e.closed and len(e) > 2 and e[0] == e[-1]:
        e.closed = True
    # Iterate through vertices; handle bulge (arc) segments if present
    for i, (x1, y1, *_rest) in enumerate(e):
        x2, y2 = e[(i + 1) % len(e)][0], e[(i + 1) % len(e)][1]
        bulge = e[(i + 1) % len(e)][3] if len(e[(i + 1) % len(e)]) >= 4 else 0.0
        pts.append((x1, y1))
        if bulge and bulge != 0.0:
            # Arc approximation between (x1,y1) -> (x2,y2)
            # bulge = tan(theta/4), theta is included angle
            theta = 4.0 * math.atan(bulge)
            chord = math.hypot(x2 - x1, y2 - y1)
            if chord == 0:
                continue
            r = chord / (2 * math.sin(theta / 2)) if math.sin(theta / 2) != 0 else chord
            # Find center and interpolate points - simplified approach
            # Compute perpendicular bisector for arc center
            mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            dx, dy = (x2 - x1), (y2 - y1)
            len_ch = math.hypot(dx, dy)
            ux, uy = dx / len_ch, dy / len_ch
            # Distance from midpoint to center
            h = math.sqrt(max(r * r - (chord / 2) ** 2, 0.0))
            # Determine side by bulge sign
            cx = mx - uy * h if bulge > 0 else mx + uy * h
            cy = my + ux * h if bulge > 0 else my - ux * h
            # Angles
            a1 = math.atan2(y1 - cy, x1 - cx)
            a2 = math.atan2(y2 - cy, x2 - cx)
            # Normalize sweep direction consistent with bulge sign
            def normalize(a):
                while a <= -math.pi:
                    a += 2 * math.pi
                while a > math.pi:
                    a -= 2 * math.pi
                return a

            a1n = normalize(a1)
            a2n = normalize(a2)
            if bulge > 0 and a2n < a1n:
                a2n += 2 * math.pi
            if bulge < 0 and a2n > a1n:
                a2n -= 2 * math.pi
            # Determine number of segments: fixed or adaptive by arc length
            segs = arc_segments
            if arc_max_seglen is not None and arc_max_seglen > 0:
                arc_len = abs(theta) * abs(r)
                # cap segments to avoid too dense meshes
                segs = max(1, min(512, int(math.ceil(arc_len / arc_max_seglen))))
            for s in range(1, segs):
                t = s / segs
                ang = a1n + t * (a2n - a1n)
                pts.append((cx + r * math.cos(ang), cy + r * math.sin(ang)))
    return pts


def extrude_dxf_closed_polylines(
    input_dxf: str,
    output_dxf: str,
    height: float = 3000.0,
    layers: List[str] | None = None,
    arc_segments: int = 12,
    arc_max_seglen: float | None = None,
    optimize: bool = False,
    detect_hard_shapes: bool = False,
    hard_shapes_collector: list | None = None,
    colorize: bool = False,
    split_by_color: bool = False,
    color_stats_collector: list | None = None,
) -> None:
    """
    Load a 2D DXF, extrude closed LWPOLYLINEs into 3D MESH prisms, and save to a new DXF.
    If `layers` is provided, only polylines on those layers are extruded.
    """
    doc = ezdxf.readfile(input_dxf)
    msp = doc.modelspace()

    # Create a new document for 3D output to avoid mixing
    out = ezdxf.new(setup=True)
    out_msp = out.modelspace()

    def _get_entity_rgb(ent: LWPolyline) -> Tuple[int, int, int]:
        # True color on entity takes precedence
        tc = ent.dxf.get("true_color", None)
        if tc:
            return ezcolors.int2rgb(tc)
        aci = ent.dxf.get("color", None)
        # ByLayer (256) or ByBlock (0) are not explicit colors
        if isinstance(aci, int) and aci not in (0, 256) and 1 <= aci <= 255:
            try:
                return ezcolors.aci2rgb(aci)
            except Exception:
                pass
        # Fallback to layer color
        try:
            layer_name = ent.dxf.layer
            layer = doc.layers.get(layer_name)
            l_aci = getattr(layer.dxf, "color", None)
            if isinstance(l_aci, int) and 1 <= l_aci <= 255:
                return ezcolors.aci2rgb(int(l_aci))
        except Exception:
            pass
        # Try case-insensitive/manual search as a fallback
        try:
            lname = (ent.dxf.layer or "").lower()
            for l in doc.layers:
                if getattr(l.dxf, "name", "").lower() == lname:
                    l_aci = getattr(l.dxf, "color", None)
                    if isinstance(l_aci, int) and 1 <= l_aci <= 255:
                        return ezcolors.aci2rgb(int(l_aci))
                    break
        except Exception:
            pass
        # Default ACI 7 (white/black) if all else fails
        return ezcolors.aci2rgb(7)

    def _ensure_layer(out_doc: ezdxf.EzDxf, name: str, rgb: Tuple[int, int, int]) -> None:
        try:
            _ = out.layers.get(name)
        except Exception:
            # Approximate to nearest ACI for layer color
            try:
                aci = ezcolors.rgb2aci(rgb)
            except Exception:
                aci = 7
            out.layers.add(name, dxfattribs={"color": aci})

    for e in msp.query("LWPOLYLINE"):
        e: LWPolyline
        if not e.closed:
            continue
        if layers and e.dxf.layer not in layers:
            continue
        pts2d = _approximate_lwpolyline_points(e, arc_segments=arc_segments, arc_max_seglen=arc_max_seglen)
        if len(pts2d) < 3:
            continue
        if detect_hard_shapes:
            issues = detect_polygon_issues(pts2d)
            if issues:
                if hard_shapes_collector is not None:
                    # extra diagnostics (JSON consumers can use these)
                    xs = [p[0] for p in pts2d]
                    ys = [p[1] for p in pts2d]
                    area_abs = abs(polygon_area(pts2d))
                    cx = sum(xs) / len(xs)
                    cy = sum(ys) / len(ys)
                    hard_shapes_collector.append({
                        "layer": e.dxf.layer,
                        "handle": getattr(e.dxf, "handle", ""),
                        "issues": ",".join(sorted(set(issues))),
                        "vertex_count": len(pts2d),
                        "area": area_abs,
                        "bbox_min_x": min(xs),
                        "bbox_min_y": min(ys),
                        "bbox_max_x": max(xs),
                        "bbox_max_y": max(ys),
                        "centroid_x": cx,
                        "centroid_y": cy,
                    })
                # Skip problematic shapes
                continue
        verts, faces = build_prism_mesh(pts2d, height)
        if optimize:
            verts, faces = optimize_vertices(verts, faces)
        if not verts:
            continue
        # Determine color/layer
        target_layer = e.dxf.layer
        r, g, b = _get_entity_rgb(e)
        rgb = (r, g, b)
        if split_by_color:
            target_layer = f"{target_layer}__COLOR_{r}_{g}_{b}"
        # Ensure layer exists in output document
        if target_layer:
            _ensure_layer(out, target_layer, rgb or ezcolors.aci2rgb(7))
        mesh = out_msp.add_mesh(dxfattribs={"layer": target_layer} if target_layer else None)
        with mesh.edit_data() as mesh_data:
            mesh_data.vertices = verts
            mesh_data.faces = faces
        # Apply true color if requested
        if colorize and rgb is not None:
            try:
                mesh.dxf.true_color = ezcolors.rgb2int(rgb)
            except Exception:
                # Fallback to setting ACI approximated color
                try:
                    mesh.dxf.color = ezcolors.rgb2aci(rgb)
                except Exception:
                    pass
        # Collect color stats if requested
        if color_stats_collector is not None:
            color_stats_collector.append({
                "source_layer": e.dxf.layer,
                "target_layer": target_layer,
                "r": int(r),
                "g": int(g),
                "b": int(b),
            })

    out.saveas(output_dxf)
