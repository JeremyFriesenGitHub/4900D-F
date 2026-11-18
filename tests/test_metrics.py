import numpy as np

import geomproc
from arap import metrics
from arap.meshio import from_geomproc


def test_area_and_volume_of_a_cylinder():
    V, F = from_geomproc(geomproc.create_cylinder(1.0, 2.0, 400, 3, closed=True))
    assert metrics.is_closed(F)
    assert np.isclose(metrics.surface_area(V, F), 2 * np.pi * 2.0 + 2 * np.pi, rtol=1e-3)
    assert np.isclose(metrics.enclosed_volume(V, F), np.pi * 2.0, rtol=1e-3)


def test_rigid_motion_and_scaling_changes():
    V, F = from_geomproc(geomproc.create_cylinder(0.5, 1.0, 30, 5, closed=True))
    Q, _ = np.linalg.qr(np.random.default_rng(0).normal(size=(3, 3)))
    Q *= np.sign(np.linalg.det(Q))  # a rotation, not a reflection
    moved = V @ Q.T + 1.0
    change = metrics.area_volume_change(V, moved, F)
    assert abs(change["area_change"]) < 1e-12 and abs(change["volume_change"]) < 1e-12
    change = metrics.area_volume_change(V, 2.0 * V, F)
    assert np.isclose(change["area_change"], 3.0) and np.isclose(change["volume_change"], 7.0)


def test_open_mesh_has_no_volume(bunny):
    V, F = bunny
    assert not metrics.is_closed(F)
    assert metrics.enclosed_volume(V, F) is None
    assert metrics.area_volume_change(V, V, F)["volume_change"] is None


def test_inside_out_shows_as_a_change_below_minus_100_percent():
    V, F = from_geomproc(geomproc.create_cylinder(0.5, 1.0, 30, 5, closed=True))
    assert metrics.signed_volume(V, F) > 0  # GeomProc orients faces outwards
    mirrored = V * np.array([-1.0, 1.0, 1.0])  # reflection turns it inside out
    assert np.isclose(metrics.enclosed_volume(mirrored, F), metrics.enclosed_volume(V, F))
    assert np.isclose(metrics.area_volume_change(V, mirrored, F)["volume_change"], -2.0)


def test_self_intersections():
    V, F = from_geomproc(geomproc.create_cylinder(0.5, 1.0, 12, 5, closed=True))
    assert metrics.self_intersections(V, F) == 0
    n = V.shape[0]
    apart = metrics.self_intersections(np.vstack([V, V + [2.0, 0.0, 0.0]]), np.vstack([F, F + n]))
    assert apart == 0
    # Two cylinders crossing each other at right angles.
    crossed = np.vstack([V, V[:, [2, 1, 0]] * [1.0, 1.0, -1.0]])
    assert metrics.self_intersections(crossed, np.vstack([F, F + n])) > 0
    # Two triangles that pierce each other like chain links: one edge of each
    # passes through the other, so exactly two pairs.
    P = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0],
                  [0.5, 0.5, -1.0], [0.5, 0.5, 1.0], [3.0, 3.0, 0.0]])
    assert metrics.self_intersections(P, np.array([[0, 1, 2], [3, 4, 5]])) == 2
    # The count does not depend on the unit of length.
    both = np.vstack([F, F + n])
    count = metrics.self_intersections(crossed, both)
    assert metrics.self_intersections(1e-5 * crossed, both) == count
    assert metrics.self_intersections(1e5 * crossed, both) == count


def test_edge_length_error_of_known_stretches():
    V = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    F = np.array([[0, 1, 2]])
    e = metrics.edge_length_error(V, 1.5 * V, F)  # every edge 50 % longer
    assert np.allclose([e["rms_relative"], e["mean_abs_relative"], e["max_abs_relative"]], 0.5)
    stretched = V * [2.0, 1.0, 1.0]  # edges 2, 2, sqrt(8) become 4, 2, sqrt(20)
    rel = np.array([1.0, 0.0, np.sqrt(20.0) / np.sqrt(8.0) - 1.0])
    e = metrics.edge_length_error(V, stretched, F)
    assert np.isclose(e["rms_relative"], np.sqrt(np.mean(rel ** 2)))
    assert np.isclose(e["mean_abs_relative"], rel.mean())
    assert np.isclose(e["max_abs_relative"], 1.0)
    e = metrics.edge_length_error(V, 0.5 * V, F)  # shortening counts as much as stretching
    assert np.allclose([e["rms_relative"], e["mean_abs_relative"], e["max_abs_relative"]], 0.5)


def test_per_vertex_stretch_counts_shortening_as_stretch():
    V = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    F = np.array([[0, 1, 2]])
    moved = V.copy()
    moved[1, 0], moved[2, 1] = 3.0, 1.0  # one edge at vertex 0 grows by half, the other halves
    assert np.isclose(metrics.per_vertex_stretch(V, moved, F)[0], 0.5)


def test_self_intersections_ignores_an_edge_that_stops_short_of_a_triangle():
    # The edge's line goes through the triangle, but the edge ends before it.
    P = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 2.0],
                  [0.5, 0.2, 1.0], [0.5, 0.2, 0.6], [5.0, 5.0, 5.0]])
    F = np.array([[0, 1, 2], [3, 4, 5]])
    assert metrics.self_intersections(P, F) == 0
    P[4] = [0.5, 0.2, -0.5]  # now the edge goes through
    assert metrics.self_intersections(P, F) >= 1


def test_load_mesh_normalises_the_longest_side_to_two():
    from arap.meshio import load_mesh
    from conftest import mesh_path

    V, _ = load_mesh(mesh_path("bunny.obj"))
    assert np.isclose(np.max(V.max(axis=0) - V.min(axis=0)), 2.0)
    raw, _ = load_mesh(mesh_path("bunny.obj"), normalize=False)
    assert not np.isclose(np.max(raw.max(axis=0) - raw.min(axis=0)), 2.0)
