"""Checks of the measurement helpers and bar meshes used by the experiments."""

import numpy as np

from arap import selection as sel
from experiments import run_all as X
from experiments import scenarios as S


def twisted(V, total_degrees):
    """The bar with each point turned about z in proportion to its height."""
    z0, z1 = V[:, 2].min(), V[:, 2].max()
    a = np.radians(total_degrees) * (V[:, 2] - z0) / (z1 - z0)
    out = V.copy()
    out[:, 0] = np.cos(a) * V[:, 0] - np.sin(a) * V[:, 1]
    out[:, 1] = np.sin(a) * V[:, 0] + np.cos(a) * V[:, 1]
    return out


def test_twist_angle_follows_twists_past_180_degrees():
    V, F = S.bar()
    for degrees in (0.0, 90.0, 250.0, -300.0, 360.0):
        assert np.isclose(X.twist_angle(V, twisted(V, degrees)), degrees, atol=1e-6)


def test_twist_angle_on_a_jittered_bar():
    V, F = S.irregular_bar(jitter=0.02)
    assert [set(r) for r in X.side_rings(V)] == [set(r) for r in X.side_rings(S.bar()[0])]
    for degrees in (90.0, 250.0):
        assert np.isclose(X.twist_angle(V, twisted(V, degrees)), degrees, atol=1e-6)


def test_twist_is_undefined_below_the_coherence_threshold():
    V, F = S.bar()
    ring = X.side_rings(V)[20]
    for turn, defined in ((110.0, True), (130.0, False)):
        # Every other vertex turned: the mean direction has length cos(turn / 2).
        X_ = V.copy()
        X_[ring[::2]] = V[ring[::2]] @ sel.rotation("z", turn).T
        assert np.isnan(X.twist_angle(V, X_)) != defined


def test_twist_is_undefined_across_a_jump_of_more_than_90_degrees():
    V, F = S.bar()
    X_ = V.copy()
    ring = X.side_rings(V)[20]
    X_[ring] = V[ring] @ sel.rotation("z", 80.0).T
    assert np.isclose(X.twist_angle(V, X_), 0.0)
    X_[ring] = V[ring] @ sel.rotation("z", 120.0).T
    assert np.isnan(X.twist_angle(V, X_))


def test_twist_is_undefined_where_a_ring_folds_onto_itself():
    V, F = S.bar()
    X_ = twisted(V, 90.0)
    ring = X.side_rings(V)[20]
    angle = 2.0 * np.arctan2(V[ring, 1], V[ring, 0])  # the ring goes round twice
    X_[ring, 0], X_[ring, 1] = 0.1 * np.cos(angle), 0.1 * np.sin(angle)
    assert np.isnan(X.twist_angle(V, X_))


def test_narrowest_ring_of_a_rigid_motion_and_a_squeeze():
    V, F = S.bar()
    moved = V @ sel.rotation("z", 30.0).T + np.array([0.3, -0.2, 1.0])
    assert np.isclose(X.narrowest_ring(V, moved), 1.0, rtol=0.02)
    squeezed = V.copy()
    ring = X.side_rings(V)[10]
    squeezed[ring, 0] *= 0.25
    assert np.isclose(X.narrowest_ring(V, squeezed), 0.25, rtol=0.03)
    squeezed[ring, 0] = 0.0  # flattened into a line
    assert X.narrowest_ring(V, squeezed) == 0.0


def test_narrowest_ring_adds_the_lobes_of_a_figure_eight():
    V, F = S.bar(sides=200)
    ring = X.side_rings(V)[10]
    t = np.arctan2(V[ring, 1], V[ring, 0])
    eight = V.copy()
    eight[ring, 0] = 0.1 * np.sin(t)
    eight[ring, 1] = 0.1 * np.sin(t) * np.cos(t)
    # Two lobes of 2/3 * 0.1^2 each, against pi * 0.1^2 for the circle; the
    # signed area of a figure eight would be zero.
    assert np.isclose(X.narrowest_ring(V, eight), 4.0 / (3.0 * np.pi), rtol=0.03)


def test_rotation_angle_of_a_rigid_motion():
    P = np.random.default_rng(0).normal(size=(20, 3))
    Q = P @ sel.rotation("x", 65.0).T + np.array([1.0, 2.0, 3.0])
    assert np.isclose(X.rotation_angle(P, Q), 65.0)


def test_bar_surface_points_on_the_rest_mesh():
    V, F = S.bar(rings=41, sides=20)
    samples = np.array([[0.3, 0.1], [-3.1, -0.2], [3.1, 0.4], [np.pi, 0.0]])
    P = X.bar_surface_points(V, F, V, samples)
    assert not np.isnan(P).any()
    assert np.allclose(P[:, 2], samples[:, 1])
    # On a flat side triangle, between the chord and the circle of radius 0.1.
    r = np.hypot(P[:, 0], P[:, 1])
    assert np.all((r <= 0.1 + 1e-12) & (r >= 0.1 * np.cos(np.pi / 20) - 1e-12))
    # A sample at a vertex returns that vertex.
    k = np.flatnonzero((np.hypot(V[:, 0], V[:, 1]) > 0.05) & (np.abs(V[:, 2]) < 0.4))[7]
    at_vertex = X.bar_surface_points(V, F, V, [[np.arctan2(V[k, 1], V[k, 0]), V[k, 2]]])
    assert np.allclose(at_vertex[0], V[k])


def test_bar_surface_points_move_with_the_mesh():
    V, F = S.irregular_bar(rings=41, sides=20)
    samples = np.column_stack([np.linspace(-np.pi, np.pi, 50), np.linspace(-0.45, 0.45, 50)])
    R = sel.rotation("y", 40.0)
    P = X.bar_surface_points(V, F, V, samples)
    Q = X.bar_surface_points(V, F, V @ R.T + 1.0, samples)
    assert not np.isnan(P).any()
    assert np.allclose(Q, P @ R.T + 1.0)


def outward_side_faces(V, F):
    """Fraction of side triangles whose normal points away from the bar's axis."""
    r = np.hypot(V[:, 0], V[:, 1])
    side = F[np.all(r[F] > 0.05, axis=1)]
    n = np.cross(V[side[:, 1]] - V[side[:, 0]], V[side[:, 2]] - V[side[:, 0]])
    c = V[side].mean(axis=1)
    radial = np.column_stack([c[:, 0], c[:, 1], np.zeros(len(c))])
    return np.mean(np.einsum("ij,ij->i", n, radial) > 0)


def test_irregular_bar_keeps_the_shape_and_the_ends():
    V, F = S.bar(rings=41, sides=20)
    W, G = S.irregular_bar(rings=41, sides=20)
    assert np.array_equal(F, G)
    moved = np.linalg.norm(W - V, axis=1) > 1e-12
    assert moved.sum() > 0.8 * V.shape[0]
    r = np.hypot(W[:, 0], W[:, 1])
    assert np.allclose(r[r > 0.05], 0.1)
    ends = (V[:, 2] == V[:, 2].min()) | (V[:, 2] == V[:, 2].max())
    assert not moved[ends].any()


def test_irregular_bar_has_no_folded_triangles():
    V, F = S.bar(rings=41, sides=20)
    assert outward_side_faces(V, F) == 1.0
    for seed in range(5):
        W, G = S.irregular_bar(rings=41, sides=20, seed=seed)
        assert outward_side_faces(W, G) == 1.0
    # Past a quarter of the spacing triangles can fold, which is why the
    # default jitter stays below it.
    W, G = S.irregular_bar(rings=41, sides=20, jitter=0.4)
    assert outward_side_faces(W, G) < 1.0


def test_bend_targets_turn_the_end_towards_minus_y():
    V, F = S.bar()
    everything = np.arange(V.shape[0])
    rings = X.side_rings(V)
    for angle in (30.0, 90.0, 200.0):
        T = S.bend_targets(V, everything, angle)
        axis = T[rings[-1]].mean(axis=0) - T[rings[0]].mean(axis=0)
        theta = np.radians(angle)
        assert np.allclose(axis / np.linalg.norm(axis), [0.0, -np.sin(theta), np.cos(theta)])


def test_bar_ends_take_whole_rings():
    for rings in (25, 41, 61, 81, 121, 161, 241):
        V, F = S.bar(rings=rings, sides=12)
        fixed, handles = S.bar_ends(V)
        z = V[:, 2]
        # Thresholds keep clear of every ring, so jitter cannot split one.
        spacing = 1.0 / (rings - 1)
        for k in range(rings):
            level = -0.5 + k * spacing
            assert min(abs(level - (-0.5 + 0.049)), abs(level - (-0.5 + 0.951))) > 0.02 * spacing
        assert fixed.size == handles.size
        assert np.isclose(z[fixed].max() + z[handles].min(), 0.0)
    jittered = S.bar_twist()
    regular = S.bar_twist(jitter=0.0)
    assert np.array_equal(jittered.fixed, regular.fixed)
    assert np.array_equal(jittered.handles, regular.handles)
    assert not np.array_equal(jittered.vertices, regular.vertices)


def test_flip_diagonals():
    from arap import laplacian as lap
    from arap.metrics import is_closed, signed_volume, unique_edges

    V, F = S.bar(rings=41, sides=20)
    G = S.flip_diagonals(V, F)
    assert is_closed(G) and np.isclose(signed_volume(V, G), signed_volume(V, F))
    old, new = set(map(tuple, unique_edges(F))), set(map(tuple, unique_edges(G)))
    assert len(old - new) == len(new - old) == 20 * 40
    # The diagonal of a rectangle is opposite two right angles, so its
    # cotangent weight is zero and the flip leaves the cotangent Laplacian as
    # it is; the uniform Laplacian changes.
    assert abs(lap.mesh_laplacian(V, F)[0] - lap.mesh_laplacian(V, G)[0]).max() < 1e-12
    assert abs(lap.mesh_laplacian(V, F, "uniform")[0] - lap.mesh_laplacian(V, G, "uniform")[0]).max() == 1.0


def test_centre_line_kink():
    V, F = S.bar(rings=41, sides=20)
    z0 = V[:, 2].min()
    radius = 2.0 / np.pi  # a quarter circle as long as the bar

    def bent(arc_length):
        """The bar bent into a quarter circle; each height h goes to arc length arc_length(h)."""
        a = arc_length(V[:, 2] - z0) / radius
        out = V.copy()
        out[:, 1] = -radius * (1 - np.cos(a)) + V[:, 1] * np.cos(a)
        out[:, 2] = z0 + radius * np.sin(a) + V[:, 1] * np.sin(a)
        return out

    assert np.isclose(X.centre_line_kink(V, bent(lambda h: h)), 1.0, atol=1e-6)
    # The same arc, but most of the turn is crowded towards the top end.
    assert X.centre_line_kink(V, bent(lambda h: h ** 3)) > 2.5
    assert X.centre_line_kink(V, V + 1.0) == 1.0
    W, _ = S.irregular_bar()
    assert X.centre_line_kink(W, W) is None


def test_bend_targets_put_the_end_on_the_arc():
    V, F = S.bar()
    top = np.flatnonzero(V[:, 2] == V[:, 2].max())
    for angle in (90.0, 180.0, 270.0):
        T = S.bend_targets(V, top, angle)
        # The end stays a rigid copy, turned by the bend angle.
        assert np.isclose(X.rotation_angle(V[top], T), angle if angle <= 180 else 360 - angle)
        theta, R = np.radians(angle), 1.0 / np.radians(angle)
        centre = T.mean(axis=0)
        assert np.allclose(centre, [0.0, -R * (1 - np.cos(theta)), -0.5 + R * np.sin(theta)])
