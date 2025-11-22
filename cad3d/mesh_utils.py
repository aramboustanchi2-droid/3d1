"""
Core Mesh and Polygon Utilities.

This module provides a set of fundamental, low-level functions for 2D polygon
manipulation and 3D mesh construction. These utilities form the geometric
backbone of the CAD 2D-to-3D conversion process.

Key functionalities include:
- Polygon diagnostics: Detecting issues like self-intersections, zero-area,
  and duplicate vertices (`detect_polygon_issues`).
- Triangulation: Converting simple 2D polygons into a set of triangles using
  the robust ear-clipping algorithm (`ear_clip_triangulate`).
- Mesh construction: Building a 3D prism mesh from a 2D base polygon by
  extruding it along the Z-axis (`build_prism_mesh`).
- Mesh optimization: Deduplicating vertices to create smaller, cleaner, and
  watertight meshes (`optimize_vertices`).

These functions are designed to be pure and have no dependencies on external
CAD libraries like `ezdxf`, making them easily testable and reusable.
"""
from __future__ import annotations
from typing import List, Sequence, Tuple

Point2D = Tuple[float, float]
Point3D = Tuple[float, float, float]


def polygon_area(poly: Sequence[Point2D]) -> float:
    """Calculates the signed area of a 2D polygon using the shoelace formula.

    The signed area is a standard way to determine the orientation or "winding"
    of a polygon's vertices.

    Args:
        poly: A sequence of (x, y) tuples representing the polygon vertices in order.

    Returns:
        The signed area of the polygon. The area is:
        - Positive for a counter-clockwise (CCW) ordered polygon.
        - Negative for a clockwise (CW) ordered polygon.
        - Zero for a degenerate polygon (e.g., all points are collinear).
    """
    area = 0.0
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]  # Wrap around to the first vertex
        area += (x1 * y2) - (x2 * y1)
    return area / 2.0


def is_clockwise(poly: Sequence[Point2D]) -> bool:
    """Checks if a 2D polygon's vertices are in clockwise (CW) order.

    This is determined by checking if the signed area of the polygon is negative.

    Args:
        poly: A sequence of (x, y) tuples representing the polygon vertices.

    Returns:
        True if the polygon's winding order is clockwise, False otherwise.
    """
    return polygon_area(poly) < 0


def ear_clip_triangulate(poly: Sequence[Point2D]) -> List[Tuple[int, int, int]]:
    """
    Triangulates a simple 2D polygon using the Ear Clipping algorithm.

    This method works for simple polygons (no holes, no self-intersections).
    It iteratively finds "ears" (triangles formed by three consecutive vertices
    that contain no other vertices) and "clips" them from the polygon until
    only one triangle remains.

    The algorithm requires the polygon to have a consistent winding order, which
    is handled internally by ensuring it is counter-clockwise (CCW).

    Args:
        poly: A sequence of (x, y) tuples for the polygon vertices.

    Returns:
        A list of 3-tuples, where each tuple contains the indices of the
        vertices from the original polygon list that form a triangle.
        Returns an empty list if the polygon has fewer than 3 vertices.
    """
    n = len(poly)
    if n < 3:
        return []

    # Create a mutable list of vertex indices to represent the polygon ring.
    verts = list(range(n))

    # Ensure the polygon is in Counter-Clockwise (CCW) order, which is a
    # prerequisite for the convexity and point-in-triangle checks.
    if is_clockwise(poly):
        verts.reverse()

    triangles: List[Tuple[int, int, int]] = []
    max_iter = len(verts) * 2  # Safety break for complex/degenerate cases.

    while len(verts) > 2 and max_iter > 0:
        max_iter -= 1
        ear_found = False
        for i in range(len(verts)):
            # Get indices for the potential ear triangle (previous, current, next).
            i_prev = verts[(i - 1) % len(verts)]
            i_curr = verts[i]
            i_next = verts[(i + 1) % len(verts)]

            p_prev, p_curr, p_next = poly[i_prev], poly[i_curr], poly[i_next]

            # Check 1: Is the vertex convex? For a CCW polygon, the cross product
            # of the vectors (p_curr - p_prev) and (p_next - p_prev) must be positive.
            if _ccw(p_prev, p_curr, p_next) <= 0:
                continue  # This is a reflex or collinear vertex, not an ear tip.

            # Check 2: Does the potential ear triangle contain any other vertices?
            # If it does, it's not a valid ear.
            is_ear = True
            for j_idx in range(len(verts)):
                v_idx = verts[j_idx]
                if v_idx in (i_prev, i_curr, i_next):
                    continue  # Skip the vertices of the ear itself.
                
                # Use barycentric coordinates to check if p_test is inside the triangle.
                p_test = poly[v_idx]
                det = (p_curr[1] - p_next[1]) * (p_prev[0] - p_next[0]) + \
                      (p_next[0] - p_curr[0]) * (p_prev[1] - p_next[1])
                if abs(det) < 1e-12: continue # Collinear, treat as outside.

                l1 = ((p_curr[1] - p_next[1]) * (p_test[0] - p_next[0]) + \
                      (p_next[0] - p_curr[0]) * (p_test[1] - p_next[1])) / det
                l2 = ((p_next[1] - p_prev[1]) * (p_test[0] - p_next[0]) + \
                      (p_prev[0] - p_next[0]) * (p_test[1] - p_next[1])) / det
                l3 = 1.0 - l1 - l2

                # If all barycentric coordinates are strictly between 0 and 1,
                # the point is inside the triangle.
                if 0 < l1 < 1 and 0 < l2 < 1 and 0 < l3 < 1:
                    is_ear = False
                    break
            
            if is_ear:
                # Found a valid ear! Add it to the list and remove its tip from the polygon.
                triangles.append((i_prev, i_curr, i_next))
                verts.pop(i)
                ear_found = True
                break
        
        if not ear_found:
            # If no ear was found in a full pass, the polygon is likely degenerate
            # or complex (e.g., self-intersecting). As a fallback, perform a simple
            # fan triangulation from the first vertex. This is not robust for all
            # shapes but can handle some failure cases gracefully.
            base_v_idx = verts[0]
            for k in range(1, len(verts) - 1):
                triangles.append((base_v_idx, verts[k], verts[k + 1]))
            return triangles

    return triangles


def build_prism_mesh(
    base: Sequence[Point2D], height: float
) -> Tuple[List[Point3D], List[Tuple[int, int, int]]]:
    """
    Builds a triangular mesh for a vertical prism extruded from a 2D base polygon.

    The process involves:
    1. Triangulating the 2D base polygon to form the bottom and top faces.
    2. Creating 3D vertices for the bottom face (at z=0) and top face (z=height).
    3. Generating triangular faces for the bottom, top, and side walls of the prism,
       ensuring correct winding order for outward-facing normals.

    Args:
        base: A sequence of (x, y) tuples for the base polygon. The polygon can be
              open or closed (if closed, the last point is ignored).
        height: The extrusion height along the positive Z-axis.

    Returns:
        A tuple containing:
        - A list of 3D vertex coordinates (x, y, z).
        - A list of 3-tuples, where each tuple contains the indices of the
          vertices forming a single triangular face.
    """
    # Ensure the polygon is open (no duplicate start/end point) for processing.
    if len(base) >= 2 and base[0] == base[-1]:
        base = base[:-1]

    n = len(base)
    if n < 3:
        return [], []

    # Triangulate the base polygon to create the floor/ceiling plan.
    base_triangles = ear_clip_triangulate(base)

    # Build vertices: bottom ring first (indices 0 to n-1), then top ring (n to 2n-1).
    vertices: List[Point3D] = []
    vertices.extend((x, y, 0.0) for x, y in base)
    vertices.extend((x, y, height) for x, y in base)

    faces: List[Tuple[int, int, int]] = []

    # Bottom faces: use the triangulation as is. Assuming the triangulation
    # produces CCW faces, these will correctly point downwards/inwards.
    for a, b, c in base_triangles:
        faces.append((a, b, c))

    # Top faces: offset indices by n and reverse winding order (c, b, a)
    # so the faces point upwards/outwards.
    for a, b, c in base_triangles:
        faces.append((c + n, b + n, a + n))

    # Side faces: create two triangles (a quad) for each edge of the base polygon.
    for i in range(n):
        j = (i + 1) % n
        # Indices for the quad forming the side wall:
        i0, j0 = i, j         # Bottom edge vertices
        i1, j1 = i + n, j + n # Top edge vertices
        
        # Split the quad into two triangles with consistent winding.
        faces.append((i0, j0, j1))
        faces.append((i0, j1, i1))

    return vertices, faces


def optimize_vertices(
    vertices: Sequence[Point3D],
    faces: Sequence[Tuple[int, int, int]],
    tol: float = 1e-6,
) -> Tuple[List[Point3D], List[Tuple[int, int, int]]]:
    """
    Deduplicates vertices within a tolerance and remaps face indices accordingly.

    This is a critical step for creating clean, efficient, and watertight meshes.
    It works by "snapping" vertices to a conceptual grid defined by the tolerance.
    All vertices that fall into the same grid cell are merged into a single vertex.

    Args:
        vertices: The list of input vertex coordinates.
        faces: The list of input face indices.
        tol: The tolerance distance. Vertices closer than this will be merged.
             A smaller tolerance is more precise but may miss merging vertices
             that are only slightly different due to floating-point inaccuracies.

    Returns:
        A tuple containing the new list of unique vertices and the re-indexed faces.
    """
    if not vertices:
        return [], []

    # Maps a quantized vertex coordinate to its new index in the `new_vertices` list.
    vert_key_to_new_index: Dict[int, int] = {}
    # Maps an old vertex index to its new, deduplicated index.
    old_to_new_index_map: Dict[int, int] = {}
    new_vertices: List[Point3D] = []

    for i, v in enumerate(vertices):
        # Create a hashable key by quantizing the vertex coordinates to the grid.
        key = (
            int(round(v[0] / tol)),
            int(round(v[1] / tol)),
            int(round(v[2] / tol)),
        )
        key_hash = hash(key)

        new_idx = vert_key_to_new_index.get(key_hash)
        if new_idx is None:
            # This is the first time we've seen a vertex at this grid location.
            # Add it to our new list and store its index.
            new_idx = len(new_vertices)
            new_vertices.append(v)
            vert_key_to_new_index[key_hash] = new_idx
        
        # Record the mapping from the original index to the new one.
        old_to_new_index_map[i] = new_idx

    # Remap all faces to use the new, deduplicated vertex indices.
    # This also implicitly removes degenerate faces where vertices were merged.
    new_faces = [
        (old_to_new_index_map[a], old_to_new_index_map[b], old_to_new_index_map[c])
        for a, b, c in faces
    ]

    # Optional: Filter out degenerate faces where two or more vertices merged.
    new_faces = [f for f in new_faces if f[0] != f[1] and f[1] != f[2] and f[0] != f[2]]

    return new_vertices, new_faces


def _segments(poly: Sequence[Point2D]) -> List[Tuple[Point2D, Point2D]]:
    """Converts a polygon vertex sequence into a list of edge segments."""
    return [(poly[i], poly[(i + 1) % len(poly)]) for i in range(len(poly))]


def _dist_sq(a: Point2D, b: Point2D) -> float:
    """Calculates the squared Euclidean distance between two 2D points."""
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return dx * dx + dy * dy


def _ccw(a: Point2D, b: Point2D, c: Point2D) -> float:
    """
    Calculates the 2D cross product of vectors (b-a) and (c-a).

    The result determines the orientation of the turn at vertex `b`.
    - > 0: `c` is to the left of the directed line `ab` (a CCW turn).
    - < 0: `c` is to the right of the directed line `ab` (a CW turn).
    - = 0: `a`, `b`, and `c` are collinear.
    """
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _segments_intersect(p1: Point2D, p2: Point2D, q1: Point2D, q2: Point2D, tol: float) -> bool:
    """
    Checks if line segment 'p1p2' properly intersects line segment 'q1q2'.

    A proper intersection means the segments cross each other at a single point
    that is not an endpoint of either segment. Touching at an endpoint or
    collinear overlaps are not considered proper intersections by this function.

    Args:
        p1, p2: Endpoints of the first segment.
        q1, q2: Endpoints of the second segment.
        tol: A small tolerance for floating-point comparisons.

    Returns:
        True if the segments properly intersect, False otherwise.
    """
    # Bounding box check for quick rejection. If the bounding boxes don't
    # overlap, the segments cannot intersect.
    if not (max(p1[0], p2[0]) + tol >= min(q1[0], q2[0]) and
            max(q1[0], q2[0]) + tol >= min(p1[0], p2[0]) and
            max(p1[1], p2[1]) + tol >= min(q1[1], q2[1]) and
            max(q1[1], q2[1]) + tol >= min(p1[1], p2[1])):
        return False

    # Use the orientation test (_ccw) to check for intersection.
    # o1/o2 check if q1 and q2 are on opposite sides of the line defined by p1p2.
    o1 = _ccw(p1, p2, q1)
    o2 = _ccw(p1, p2, q2)
    # o3/o4 check if p1 and p2 are on opposite sides of the line defined by q1q2.
    o3 = _ccw(q1, q2, p1)
    o4 = _ccw(q1, q2, p2)

    # A proper intersection exists if the endpoints of each segment lie on
    # opposite sides of the line defined by the other segment.
    if (o1 > tol and o2 < -tol or o1 < -tol and o2 > tol) and \
       (o3 > tol and o4 < -tol or o3 < -tol and o4 > tol):
        return True

    # Collinear and endpoint-touching cases are not treated as "hard" intersections.
    return False


def detect_polygon_issues(poly: Sequence[Point2D], tol: float = 1e-9) -> List[str]:
    """
    Performs diagnostics on a 2D polygon to detect common validity issues.

    These issues can cause problems during triangulation or extrusion, leading
    to incorrect geometry or algorithm failure.

    Args:
        poly: A sequence of (x, y) tuples for the polygon vertices.
        tol: A small tolerance value for floating-point comparisons.

    Returns:
        A list of strings describing any issues found. An empty list indicates
        the polygon is likely valid for processing. Possible issues include:
        - "too_few_vertices": Fewer than 3 unique vertices.
        - "duplicate_vertex": A vertex appears more than once (non-consecutively).
        - "zero_length_edge": An edge has a length close to zero.
        - "tiny_or_zero_area": The polygon's area is negligible.
        - "self_intersection": Edges of the polygon cross each other.
    """
    issues: List[str] = []
    n = len(poly)
    if n < 3:
        return ["too_few_vertices"]

    # Create a working copy, ensuring it's an open loop for simplicity.
    if _dist_sq(poly[0], poly[-1]) <= tol * tol:
        poly = poly[:-1]
        n -= 1
        if n < 3:
            return ["too_few_vertices"]

    # Check for duplicate (non-consecutive) vertices and zero-length edges.
    seen_keys = set()
    for i in range(n):
        p1 = poly[i]
        p2 = poly[(i + 1) % n]
        
        # Use a quantized key to check for duplicate vertices within tolerance.
        key = (round(p1[0] / tol), round(p1[1] / tol))
        if key in seen_keys:
            if "duplicate_vertex" not in issues:
                issues.append("duplicate_vertex")
        seen_keys.add(key)

        # Check for zero-length edges.
        if _dist_sq(p1, p2) <= tol * tol:
            if "zero_length_edge" not in issues:
                issues.append("zero_length_edge")

    # Check for tiny or zero area.
    if abs(polygon_area(poly)) <= tol:
        if "tiny_or_zero_area" not in issues:
            issues.append("tiny_or_zero_area")

    # Check for self-intersections. This is an O(n^2) check, but it is
    # practical for typical CAD polygons which usually have a modest number of vertices.
    if "self_intersection" not in issues:
        segs = _segments(poly)
        for i in range(n):
            for j in range(i + 2, n):
                # Skip adjacent edges and the edge connecting the last to the first vertex.
                if i == 0 and j == n - 1:
                    continue
                
                p1, p2 = segs[i]
                q1, q2 = segs[j]
                if _segments_intersect(p1, p2, q1, q2, tol=tol):
                    issues.append("self_intersection")
                    break  # Found one, no need to search further.
            if "self_intersection" in issues:
                break

    return issues
