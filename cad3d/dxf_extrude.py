"""
DXF Polyline Extrusion Module.

This module provides the core functionality for converting 2D LWPOLYLINE entities
from a DXF file into 3D MESH entities (prisms). It is designed to handle
complex polylines that include both straight segments and arcs (defined by
bulge values).

Key features include:
- Extrusion of closed LWPOLYLINEs to a specified height.
- Filtering of entities by layer.
- Sophisticated arc approximation using either a fixed number of segments or
  an adaptive method based on maximum segment length.
- Optional detection and reporting of problematic geometries (e.g., self-intersections,
  zero-area polygons) to ensure robust processing.
- Advanced color handling, including applying entity or layer colors to the
  extruded meshes and optionally splitting meshes into new layers based on color.
- Mesh optimization to reduce file size by deduplicating vertices.
"""
from __future__ import annotations
import math
from typing import List, Tuple

import ezdxf
from ezdxf import colors as ezcolors
from ezdxf.document import Drawing
from ezdxf.entities import LWPolyline

from .mesh_utils import (
    build_prism_mesh,
    optimize_vertices,
    detect_polygon_issues,
    polygon_area,
)


def _approximate_lwpolyline_points(
    polyline: LWPolyline,
    arc_segments: int = 12,
    arc_max_seglen: float | None = None,
) -> List[Tuple[float, float]]:
    """
    Approximates a LWPOLYLINE into a list of (x, y) points, including arc segments.

    This function iterates through the polyline's vertices. For segments with a non-zero
    bulge value, it calculates the corresponding arc's geometry (center, radius, angles)
    and interpolates a series of points to approximate the curve.

    Args:
        polyline: The `ezdxf` LWPolyline entity to approximate.
        arc_segments: The default number of segments to use for approximating an arc if
                      `arc_max_seglen` is not specified.
        arc_max_seglen: If provided, this value is used to adaptively determine the
                        number of segments for an arc to ensure that no segment is
                        longer than this maximum length. This provides a more
                        consistent level of detail.

    Returns:
        A list of (x, y) tuples representing the vertices of the approximated polygon.
    """
    points: List[Tuple[float, float]] = []
    
    # Ensure polyline is treated as closed if start and end points are identical.
    if not polyline.closed and len(polyline) > 2 and polyline[0] == polyline[-1]:
        polyline.closed = True

    # Get points with format info to access bulge values
    points_with_info = list(polyline.get_points(format='xyb'))
    
    # Iterate through each segment of the polyline.
    for i in range(len(polyline)):
        p1 = polyline[i]
        p2 = polyline[(i + 1) % len(polyline)]
        # p1/p2 may be Vec2/Vec3 or numpy.ndarray → always index
        x1, y1 = float(p1[0]), float(p1[1])
        x2, y2 = float(p2[0]), float(p2[1])
        # Get bulge from points_with_info
        bulge = points_with_info[i][2] if len(points_with_info[i]) > 2 else 0.0
        
        points.append((x1, y1))

        if bulge != 0.0:
            # This segment is an arc. We need to approximate it.
            # The bulge value is the tangent of 1/4 of the arc's included angle.
            theta = 4.0 * math.atan(bulge)
            chord_len = math.hypot(x2 - x1, y2 - y1)
            
            if chord_len < 1e-9:
                continue  # Skip zero-length segments.

            # Calculate the arc radius from the chord length and angle.
            sin_half_theta = math.sin(theta / 2.0)
            if abs(sin_half_theta) < 1e-9:
                continue # Should not happen if bulge is non-zero.
            radius = abs(chord_len / (2.0 * sin_half_theta))

            # Calculate the arc's center point.
            sagitta = (chord_len / 2.0) * bulge
            midpoint_x, midpoint_y = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            chord_dx, chord_dy = x2 - x1, y2 - y1
            perp_dx, perp_dy = -chord_dy / chord_len, chord_dx / chord_len
            center_x = midpoint_x + sagitta * perp_dx
            center_y = midpoint_y + sagitta * perp_dy

            # Determine start and end angles of the arc.
            start_angle = math.atan2(y1 - center_y, x1 - center_x)
            end_angle = math.atan2(y2 - center_y, y2 - center_x)

            # Ensure the sweep direction is correct based on the bulge sign.
            if bulge < 0:
                start_angle, end_angle = end_angle, start_angle
            
            delta_angle = end_angle - start_angle
            if delta_angle <= 0:
                delta_angle += 2 * math.pi

            # Determine the number of segments for the approximation.
            num_segs = arc_segments
            if arc_max_seglen is not None and arc_max_seglen > 0:
                arc_length = delta_angle * radius
                # Cap segments to prevent excessively dense meshes.
                num_segs = max(2, min(512, int(math.ceil(arc_length / arc_max_seglen))))

            # Interpolate points along the arc.
            for s in range(1, num_segs):
                fraction = s / num_segs
                angle = start_angle + fraction * delta_angle
                px = center_x + radius * math.cos(angle)
                py = center_y + radius * math.sin(angle)
                points.append((px, py))
                
    return points


def extrude_dxf_closed_polylines(
    input_dxf: str,
    output_dxf: str,
    height: float = 3000.0,
    layers: list[str] | None = None,
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
    Loads a 2D DXF, extrudes closed LWPOLYLINEs into 3D MESH prisms, and saves to a new DXF.

    Args:
        input_dxf: Path to the input 2D DXF file.
        output_dxf: Path to save the output 3D DXF file.
        height: The extrusion height for the prisms.
        layers: If provided, only polylines on these layers will be extruded.
        arc_segments: Default number of segments to approximate an arc.
        arc_max_seglen: If set, adaptively calculates arc segments based on this max length.
        optimize: If True, deduplicates vertices in the final mesh to reduce file size.
        detect_hard_shapes: If True, runs diagnostics and skips problematic polygons.
        hard_shapes_collector: An optional list to collect data about skipped hard shapes.
        colorize: If True, applies the source entity's color to the output mesh.
        split_by_color: If True, creates separate layers for each unique color found.
        color_stats_collector: An optional list to collect statistics about colors used.
    """
    try:
        doc = ezdxf.readfile(input_dxf)
        msp = doc.modelspace()
    except IOError:
        print(f"❌ Could not read DXF file: {input_dxf}")
        return
    except ezdxf.DXFStructureError:
        print(f"❌ Invalid or corrupt DXF file: {input_dxf}")
        return

    out_doc = ezdxf.new(setup=True)
    out_msp = out_doc.modelspace()

    for polyline in msp.query("LWPOLYLINE"):
        if not polyline.closed:
            continue
        if layers and polyline.dxf.layer not in layers:
            continue

        # Step 1: Approximate the polyline, converting any arcs (bulges) into straight segments.
        points_2d = _approximate_lwpolyline_points(polyline, arc_segments, arc_max_seglen)
        if len(points_2d) < 3:
            continue

        # Step 2: Optionally detect and skip problematic geometries.
        if detect_hard_shapes:
            issues = detect_polygon_issues(points_2d)
            if issues:
                if hard_shapes_collector is not None:
                    _collect_hard_shape_info(hard_shapes_collector, polyline, points_2d, issues)
                continue  # Skip this problematic shape.

        # Step 3: Build the 3D prism mesh from the 2D polygon.
        verts, faces = build_prism_mesh(points_2d, height)
        if optimize:
            verts, faces = optimize_vertices(verts, faces)
        if not verts:
            continue

        # Step 4: Determine the target layer and color for the new mesh.
        rgb, target_layer = _get_target_color_and_layer(doc, polyline, split_by_color)
        if target_layer:
            _ensure_layer_exists(out_doc, target_layer, rgb)

        # Step 5: Add the generated mesh to the output document.
        mesh = out_msp.add_mesh(dxfattribs={"layer": target_layer})
        with mesh.edit_data() as mesh_data:
            mesh_data.vertices = verts
            mesh_data.faces = faces

        # Step 6: Apply color to the mesh and collect statistics if requested.
        if colorize and rgb:
            mesh.dxf.true_color = ezcolors.rgb2int(rgb)

        if color_stats_collector is not None and rgb:
            color_stats_collector.append({
                "source_layer": polyline.dxf.layer,
                "target_layer": target_layer,
                "r": rgb[0], "g": rgb[1], "b": rgb[2],
            })

    out_doc.saveas(output_dxf)


def _get_target_color_and_layer(
    doc: Drawing, ent: LWPolyline, split_by_color: bool
) -> tuple[tuple[int, int, int] | None, str]:
    """
    Determines the final RGB color and layer name for an extruded entity.

    Color resolution follows a specific order of precedence:
    1. Entity's explicit True Color.
    2. Entity's explicit ACI (AutoCAD Color Index).
    3. Entity's layer's True Color.
    4. Entity's layer's ACI.
    5. Default color (white/black) if no color is found.

    Args:
        doc: The source DXF document.
        ent: The LWPolyline entity being processed.
        split_by_color: If True, the target layer name will be a composite of the
                        original layer and the resolved RGB color.

    Returns:
        A tuple containing the resolved (R, G, B) tuple and the target layer name.
    """
    
    def get_layer_color(layer_name: str) -> tuple[int, int, int] | None:
        """Helper to safely get a layer's color."""
        try:
            layer = doc.layers.get(layer_name)
            if layer.dxf.hasattr("true_color"):
                return ezcolors.int2rgb(layer.dxf.true_color)
            if layer.dxf.hasattr("color"):
                aci = abs(layer.dxf.color)
                if 1 <= aci <= 255:
                    return ezcolors.aci2rgb(aci)
        except (KeyError, AttributeError):
            pass
        return None

    rgb = None
    # 1. Check for entity-level true_color.
    if ent.dxf.hasattr("true_color"):
        rgb = ezcolors.int2rgb(ent.dxf.true_color)
    # 2. Check for entity-level ACI color (if not set to ByLayer).
    elif ent.dxf.hasattr("color") and ent.dxf.color != 256:
        aci = abs(ent.dxf.color)
        if 1 <= aci <= 255:
            rgb = ezcolors.aci2rgb(aci)

    # 3. If no entity color, fall back to the layer's color.
    if rgb is None:
        rgb = get_layer_color(ent.dxf.layer)

    # 4. If still no color, default to white (ACI 7).
    if rgb is None:
        rgb = ezcolors.aci2rgb(7)

    # 5. Determine the target layer name.
    target_layer = ent.dxf.layer
    if split_by_color:
        target_layer = f"{target_layer}__COLOR_{rgb[0]}_{rgb[1]}_{rgb[2]}"
        
    return rgb, target_layer


def _ensure_layer_exists(
    out_doc: Drawing, name: str, rgb: tuple[int, int, int] | None
) -> None:
    """Ensures a layer with the given name and color exists in the output document."""
    if name not in out_doc.layers:
        attribs = {}
        if rgb:
            attribs["true_color"] = ezcolors.rgb2int(rgb)
        out_doc.layers.add(name, dxfattribs=attribs)


def _collect_hard_shape_info(
    collector: list, entity: LWPolyline, pts: list, issues: list[str]
) -> None:
    """Gathers detailed diagnostic information about a problematic polygon."""
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    collector.append({
        "layer": entity.dxf.layer,
        "handle": str(entity.dxf.handle),
        "issues": ",".join(sorted(set(issues))),
        "vertex_count": len(pts),
        "area": abs(polygon_area(pts)),
        "bbox_min_x": min(xs) if xs else 0,
        "bbox_min_y": min(ys) if ys else 0,
        "bbox_max_x": max(xs) if xs else 0,
        "bbox_max_y": max(ys) if ys else 0,
        "centroid_x": sum(xs) / len(pts) if pts else 0,
        "centroid_y": sum(ys) / len(pts) if pts else 0,
    })

