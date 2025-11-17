from __future__ import annotations
from typing import List, Sequence, Tuple

Point2D = Tuple[float, float]
Point3D = Tuple[float, float, float]


def polygon_area(poly: Sequence[Point2D]) -> float:
    area = 0.0
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return area / 2.0


def is_clockwise(poly: Sequence[Point2D]) -> bool:
    return polygon_area(poly) < 0


def ear_clip_triangulate(poly: Sequence[Point2D]) -> List[Tuple[int, int, int]]:
    """
    Basic ear clipping triangulation for simple polygons.
    Returns indices into the input polygon list.
    """
    n = len(poly)
    if n < 3:
        return []
    # Make a working list of vertex indices
    verts = list(range(n))
    triangles: List[Tuple[int, int, int]] = []

    def point_in_triangle(p, a, b, c) -> bool:
        # Barycentric technique
        (x, y) = p
        (x1, y1) = a
        (x2, y2) = b
        (x3, y3) = c
        det = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
        if det == 0:
            return False
        l1 = ((y2 - y3) * (x - x3) + (x3 - x2) * (y - y3)) / det
        l2 = ((y3 - y1) * (x - x3) + (x1 - x3) * (y - y3)) / det
        l3 = 1.0 - l1 - l2
        return 0 < l1 < 1 and 0 < l2 < 1 and 0 < l3 < 1

    def is_convex(a, b, c) -> bool:
        (x1, y1) = a
        (x2, y2) = b
        (x3, y3) = c
        return (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1) > 0

    # Ensure CCW order for ear clipping
    if is_clockwise(poly):
        verts.reverse()

    max_iter = 10000
    while len(verts) > 3 and max_iter > 0:
        max_iter -= 1
        ear_found = False
        for i in range(len(verts)):
            i_prev = verts[(i - 1) % len(verts)]
            i_curr = verts[i]
            i_next = verts[(i + 1) % len(verts)]
            a, b, c = poly[i_prev], poly[i_curr], poly[i_next]
            if not is_convex(a, b, c):
                continue
            # Check if any other point is inside triangle abc
            contains = False
            for j in verts:
                if j in (i_prev, i_curr, i_next):
                    continue
                if point_in_triangle(poly[j], a, b, c):
                    contains = True
                    break
            if contains:
                continue
            # Ear found
            triangles.append((i_prev, i_curr, i_next))
            del verts[i]
            ear_found = True
            break
        if not ear_found:
            # Fallback: create fan from 0
            base = verts[0]
            for k in range(1, len(verts) - 1):
                triangles.append((base, verts[k], verts[k + 1]))
            verts = []
            break

    if len(verts) == 3:
        triangles.append((verts[0], verts[1], verts[2]))

    return triangles


def build_prism_mesh(base: Sequence[Point2D], height: float) -> Tuple[List[Point3D], List[Tuple[int, int, int]]]:
    """
    Build a triangular mesh for a vertical prism extruded from a 2D base polygon.
    Returns (vertices, triangle_faces).
    """
    # Ensure no duplicate last point equal to first
    if len(base) >= 2 and base[0] == base[-1]:
        base = base[:-1]

    n = len(base)
    if n < 3:
        return [], []

    # Triangulate base (CCW enforced in triangulator)
    tri = ear_clip_triangulate(base)

    # Build vertices: bottom then top
    vertices: List[Point3D] = []
    for x, y in base:
        vertices.append((x, y, 0.0))
    for x, y in base:
        vertices.append((x, y, height))

    # Faces: bottom, top, and sides
    faces: List[Tuple[int, int, int]] = []

    # Bottom faces (note: flip winding for bottom to face downward if needed)
    for a, b, c in tri:
        faces.append((a, b, c))

    # Top faces (offset by n, reverse winding to face upward)
    for a, b, c in tri:
        faces.append((c + n, b + n, a + n))

    # Side faces: two triangles per edge
    for i in range(n):
        j = (i + 1) % n
        # bottom i->j and top i->j
        i0, j0 = i, j
        i1, j1 = i + n, j + n
        # Quad split into two triangles
        faces.append((i0, j0, j1))
        faces.append((i0, j1, i1))

    return vertices, faces


def optimize_vertices(
    vertices: Sequence[Point3D],
    faces: Sequence[Tuple[int, int, int]],
    tol: float = 1e-6,
) -> Tuple[List[Point3D], List[Tuple[int, int, int]]]:
    """
    Deduplicate vertices within tolerance and remap faces accordingly.
    The method rounds coordinates to multiples of tol to form a stable key.
    """
    if not vertices:
        return list(vertices), list(faces)

    def key(v: Point3D) -> Tuple[int, int, int]:
        x, y, z = v
        kx = int(round(x / tol))
        ky = int(round(y / tol))
        kz = int(round(z / tol))
        return (kx, ky, kz)

    mapping: dict[int, int] = {}
    index_map: dict[int, int] = {}
    new_vertices: List[Point3D] = []
    for i, v in enumerate(vertices):
        k = key(v)
        j = mapping.get(hash(k))
        if j is None:
            j = len(new_vertices)
            new_vertices.append(v)
            mapping[hash(k)] = j
        index_map[i] = j

    new_faces: List[Tuple[int, int, int]] = []
    for a, b, c in faces:
        new_faces.append((index_map[a], index_map[b], index_map[c]))

    return new_vertices, new_faces


def _segments(poly: Sequence[Point2D]) -> List[Tuple[Point2D, Point2D]]:
    segs: List[Tuple[Point2D, Point2D]] = []
    n = len(poly)
    for i in range(n):
        a = poly[i]
        b = poly[(i + 1) % n]
        segs.append((a, b))
    return segs


def _dist2(a: Point2D, b: Point2D) -> float:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return dx * dx + dy * dy


def _ccw(a: Point2D, b: Point2D, c: Point2D) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _segments_intersect(p1: Point2D, p2: Point2D, q1: Point2D, q2: Point2D, tol: float) -> bool:
    # Bounding box quick reject
    def bbox_overlap():
        return not (
            max(p1[0], p2[0]) + tol < min(q1[0], q2[0]) or
            max(q1[0], q2[0]) + tol < min(p1[0], p2[0]) or
            max(p1[1], p2[1]) + tol < min(q1[1], q2[1]) or
            max(q1[1], q2[1]) + tol < min(p1[1], p2[1])
        )

    if not bbox_overlap():
        return False

    o1 = _ccw(p1, p2, q1)
    o2 = _ccw(p1, p2, q2)
    o3 = _ccw(q1, q2, p1)
    o4 = _ccw(q1, q2, p2)

    # Proper intersection
    if (o1 > tol and o2 < -tol or o1 < -tol and o2 > tol) and (o3 > tol and o4 < -tol or o3 < -tol and o4 > tol):
        return True

    # Near-collinear or touching cases: consider non-problematic unless overlap length > tol
    # For simplicity, we ignore collinear overlaps here (treated as 'hard' only if edges are nearly identical elsewhere)
    return False


def detect_polygon_issues(poly: Sequence[Point2D], tol: float = 1e-9) -> List[str]:
    """
    Basic diagnostics for a single-ring polygon used as extrusion base.
    Returns a list of issue strings; empty list means 'looks OK'.
    Checks: duplicate vertices, zero-length edges, tiny area, and self-intersections.
    """
    issues: List[str] = []
    n = len(poly)
    if n < 3:
        return ["too_few_vertices"]

    # Remove possible duplicate last==first before checks (non-destructive copy)
    if poly[0] == poly[-1]:
        poly = poly[:-1]
        n -= 1
        if n < 3:
            return ["too_few_vertices"]

    # Duplicate (non-consecutive) vertices
    seen = set()
    for i, p in enumerate(poly):
        key = (round(p[0] / tol), round(p[1] / tol))
        if key in seen:
            issues.append("duplicate_vertex")
            break
        seen.add(key)

    # Zero-length edges
    for i in range(n):
        if _dist2(poly[i], poly[(i + 1) % n]) <= tol * tol:
            issues.append("zero_length_edge")
            break

    # Area check
    a = abs(polygon_area(poly))
    if a <= tol:
        issues.append("tiny_or_zero_area")

    # Self-intersections (exclude adjacent edges and first-last adjacency)
    segs = _segments(poly)
    for i in range(len(segs)):
        for j in range(i + 1, len(segs)):
            # Skip adjacent pairs
            if j == i or j == (i + 1) % n or i == (j + 1) % n:
                continue
            p1, p2 = segs[i]
            q1, q2 = segs[j]
            if _segments_intersect(p1, p2, q1, q2, tol=tol):
                issues.append("self_intersection")
                # One is enough to classify as hard
                break
        if "self_intersection" in issues:
            break

    return issues
