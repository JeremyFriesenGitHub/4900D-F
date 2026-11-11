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
