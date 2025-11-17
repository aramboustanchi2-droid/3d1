from cad3d.mesh_utils import optimize_vertices

def test_optimize_vertices_merges_duplicates():
    verts = [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (1.0, 1.0, 0.0),
        (0.0, 1.0, 0.0),
        (1e-9, 0.0, 0.0),  # near-duplicate of (0,0,0) with tol=1e-6
    ]
    faces = [(0,1,2), (0,2,3), (4,1,2)]
    nv, nf = optimize_vertices(verts, faces, tol=1e-6)
    assert len(nv) == 4  # duplicate merged
    # Validate faces are reindexed to valid range
    max_index = max(max(a,b,c) for (a,b,c) in nf)
    assert max_index < len(nv)
