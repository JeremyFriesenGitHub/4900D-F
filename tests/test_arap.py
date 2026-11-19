import numpy as np
import pytest

from arap import laplacian as lap
from arap import selection
from arap.linear import LinearDeformer
from arap.solver import ARAPDeformer

from conftest import bar_bend, random_rotation


def test_rigid_motion_is_reproduced_exactly(bar, rng):
    V, F = bar
    Q = random_rotation(rng)
    moved = V @ Q.T + np.array([0.3, -0.2, 0.5])
    c, _, _, _ = bar_bend(V)
    d = ARAPDeformer(V, F, c, clamp="none")
    R = d.local_step(moved)
    assert np.abs(R - Q).max() < 1e-10
    assert abs(d.energy(moved, R)) < 1e-20
    assert np.abs(d.global_step(R, moved[c]) - moved).max() < 1e-10


@pytest.mark.parametrize("clamp", ["zero", "geomproc"])
def test_rest_pose_is_a_fixed_point_with_nonnegative_weights(bunny, clamp):
    V, F = bunny
    c = selection.by_coordinate(V, "y", below=0.1)
    d = ARAPDeformer(V, F, c, clamp=clamp)
    result = d.deform(V[c])
    assert result.converged
    assert np.abs(result.vertices - V).max() < 1e-12 * d.diagonal
    assert np.abs(result.rotations - np.eye(3)).max() < 1e-8


def test_raw_weights_drift_away_from_rest_pose(bunny):
    # Documents why clamp="zero" is the default: with negative cotangent
    # weights the energy is indefinite and the rest pose is not a minimum.
    V, F = bunny
    c = selection.by_coordinate(V, "y", below=0.1)
    d = ARAPDeformer(V, F, c, clamp="none")
    result = d.deform(V[c], max_iterations=20)
    assert min(result.energies) < 0.0
    assert np.abs(result.vertices - V).max() > 1e-3 * d.diagonal


def test_identity_rotation_step_is_laplacian_editing(bar):
    V, F = bar
    c, t, _, _ = bar_bend(V)
    d = ARAPDeformer(V, F, c)
    identity = np.broadcast_to(np.eye(3), (V.shape[0], 3, 3)).copy()
    membrane = LinearDeformer(V, F, c, method="laplacian").deform(t)
    assert np.abs(d.global_step(identity, t) - membrane).max() < 1e-12
    assert np.abs(d.initial_guess(t, "laplacian") - membrane).max() < 1e-12


@pytest.mark.parametrize("clamp", ["zero", "none"])
def test_gradient_matches_finite_differences(bunny, rng, clamp):
    # The bunny has varying and, unclamped, negative weights (the cube's are all 1).
    V, F = bunny
    d = ARAPDeformer(V, F, [0], clamp=clamp)
    X = V + 0.05 * rng.normal(size=V.shape)
    R = np.stack([random_rotation(rng) for _ in range(V.shape[0])])
    g = d.gradient(X, R)
    h = 1e-6
    for k in rng.choice(V.shape[0], 8, replace=False):
        for a in range(3):
            Xp, Xm = X.copy(), X.copy()
            Xp[k, a] += h
            Xm[k, a] -= h
            fd = (d.energy(Xp, R) - d.energy(Xm, R)) / (2 * h)
            assert abs(fd - g[k, a]) < 1e-6 * max(1.0, abs(g[k, a]))


def test_global_step_minimises_energy_over_free_vertices(bar, rng):
    V, F = bar
    c, t, _, _ = bar_bend(V)
    d = ARAPDeformer(V, F, c)
    R = d.local_step(d.initial_guess(t))
    X = d.global_step(R, t)
    free = np.setdiff1d(np.arange(V.shape[0]), c)
    assert np.abs(d.gradient(X, R)[free]).max() < 1e-10
    for _ in range(5):
        Y = X.copy()
        Y[free] += 1e-3 * rng.normal(size=(free.size, 3))
        assert d.energy(Y, R) > d.energy(X, R)


def test_local_step_minimises_energy_over_rotations(bar, rng):
    V, F = bar
    c, t, _, _ = bar_bend(V)
    d = ARAPDeformer(V, F, c)
    X = d.initial_guess(t)
    best = d.energy(X, d.local_step(X))
    for _ in range(5):
        R = np.stack([random_rotation(rng) for _ in range(V.shape[0])])
        assert d.energy(X, R) > best
    # perturbing the optimal rotations slightly also increases the energy
    R = d.local_step(X)
    small = selection.rotation("z", 0.5)
    assert d.energy(X, R @ small) > best


def test_bar_bend_converges_monotonically_and_meets_constraints(bar):
    V, F = bar
    c, t, _, _ = bar_bend(V)
    d = ARAPDeformer(V, F, c)
    result = d.deform(t, snapshot_iterations=(0, 1, 4))
    E = np.asarray(result.energies)
    assert result.converged
    assert np.all(np.diff(E) <= 1e-12 * E[0])
    assert E[-1] < 0.1 * E[0]
    assert np.array_equal(result.vertices[c], t)
    assert sorted(result.snapshots) == [0, 1, 4]
    assert np.array_equal(result.snapshots[0], result.initial_guess)
    for key in ("factorization", "weights"):
        assert key in d.timings
    for key in ("initial_guess", "initial_rotations", "local_steps", "global_steps",
                "iterations_total", "per_iteration", "total"):
        assert key in result.timings


def test_initial_guesses_reach_the_same_pose(bar):
    V, F = bar
    c, t, _, _ = bar_bend(V)
    d = ARAPDeformer(V, F, c)
    poses = [d.deform(t, init=init, energy_tol=1e-9, step_tol=1e-7).vertices
             for init in ("laplacian", "thin_plate", "rest")]
    for pose in poses[1:]:
        assert np.abs(pose - poses[0]).max() < 1e-4 * d.diagonal


def test_explicit_initial_guess_is_used_with_constraints_enforced(bar):
    V, F = bar
    c, t, _, _ = bar_bend(V)
    d = ARAPDeformer(V, F, c)
    guess = V + 1.0
    result = d.deform(t, init=guess, max_iterations=1)
    assert np.array_equal(result.initial_guess[c], t)
    free = np.setdiff1d(np.arange(V.shape[0]), c)
    assert np.array_equal(result.initial_guess[free], guess[free])


def test_uniform_weights_also_converge(bar):
    V, F = bar
    c, t, _, _ = bar_bend(V)
    result = ARAPDeformer(V, F, c, weights="uniform").deform(t)
    assert result.converged
    assert np.all(np.diff(result.energies) <= 1e-12 * result.energies[0])


def test_component_without_constraints_is_rejected(cube):
    V, F = cube
    n = V.shape[0]
    V2 = np.vstack([V, V + 5.0])
    F2 = np.vstack([F, F + n])
    with pytest.raises(ValueError, match="component"):
        ARAPDeformer(V2, F2, [0])
    ARAPDeformer(V2, F2, [0, n])  # one per component is enough


def test_wrong_target_shape_is_rejected(bar):
    V, F = bar
    c, t, _, _ = bar_bend(V)
    d = ARAPDeformer(V, F, c)
    with pytest.raises(ValueError):
        d.deform(t[:-1])


def test_unknown_initial_guess_is_rejected(bar):
    V, F = bar
    c, t, _, _ = bar_bend(V)
    with pytest.raises(ValueError):
        ARAPDeformer(V, F, c).deform(t, init="magic")


def test_constraint_order_is_up_to_the_caller(bar, rng):
    V, F = bar
    c, t, _, _ = bar_bend(V)
    perm = rng.permutation(c.size)
    ref = ARAPDeformer(V, F, c).deform(t, max_iterations=5)
    shuffled = ARAPDeformer(V, F, c[perm]).deform(t[perm], max_iterations=5)
    assert np.array_equal(shuffled.vertices[c], t)
    assert np.abs(shuffled.vertices - ref.vertices).max() < 1e-10


def test_duplicate_constraints_are_rejected(bar):
    V, F = bar
    with pytest.raises(ValueError, match="unique"):
        ARAPDeformer(V, F, [0, 1, 1])


def test_stops_when_a_rigid_motion_is_reachable(bar):
    # The minimum energy is zero here; the stopping rule must still fire.
    V, F = bar
    c, _, _, _ = bar_bend(V)
    Q = selection.rotation("z", 30.0) @ selection.rotation("x", 20.0)
    moved = V @ Q.T + np.array([0.1, 0.2, -0.1])
    d = ARAPDeformer(V, F, c)
    result = d.deform(moved[c])
    assert result.converged
    assert result.iterations < 500
    assert np.abs(result.vertices - moved).max() < 1e-3 * d.diagonal


def test_unreferenced_vertex_is_reported(cube):
    V, F = cube
    V2 = np.vstack([V, [5.0, 5.0, 5.0]])
    with pytest.raises(ValueError, match="unreferenced"):
        ARAPDeformer(V2, F, [0])


@pytest.mark.parametrize("bad", [np.nan, np.inf])
@pytest.mark.parametrize("weights", ["cotangent", "uniform"])
def test_non_finite_vertices_are_rejected(bar, bad, weights):
    from arap.linear import LinearDeformer

    V, F = bar
    c, t = bar_bend(V)[:2]
    for index in (c[0], np.setdiff1d(np.arange(len(V)), c)[0]):
        broken = V.copy()
        broken[index, 0] = bad
        with pytest.raises(ValueError, match="vertices contain"):
            ARAPDeformer(broken, F, c, weights=weights)
        for method in ("laplacian", "thin_plate"):
            with pytest.raises(ValueError, match="vertices contain"):
                LinearDeformer(broken, F, c, method=method, weights=weights)


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_non_finite_targets_and_initial_guess_are_rejected(bar, bad):
    from arap.linear import LinearDeformer

    V, F = bar
    c, t = bar_bend(V)[:2]
    broken = t.copy()
    broken[0, 0] = bad
    d = ARAPDeformer(V, F, c)
    with pytest.raises(ValueError, match="targets contain"):
        d.deform(broken)
    with pytest.raises(ValueError, match="targets contain"):
        LinearDeformer(V, F, c).deform(broken)
    guess = V.copy()
    guess[np.setdiff1d(np.arange(len(V)), c)[0], 0] = bad
    with pytest.raises(ValueError, match="initial guess contains"):
        d.deform(t, init=guess)


def test_constrained_must_be_integer_indices(bar):
    from arap.linear import LinearDeformer

    V, F = bar
    c = bar_bend(V)[0]
    mask = np.zeros(len(V), dtype=bool)
    mask[c] = True
    for bad in (mask, c.astype(float)):
        with pytest.raises(TypeError, match="integer vertex indices"):
            ARAPDeformer(V, F, bad)
        with pytest.raises(TypeError, match="integer vertex indices"):
            LinearDeformer(V, F, bad)
    # A column of indices is flattened by both deformers.
    column = c.reshape(-1, 1)
    assert ARAPDeformer(V, F, column).constrained.shape == c.shape
    assert LinearDeformer(V, F, column).constrained.shape == c.shape


def test_unknown_clamp_is_rejected_for_uniform_weights_too(bar):
    V, F = bar
    with pytest.raises(ValueError, match="clamp must be one of"):
        ARAPDeformer(V, F, bar_bend(V)[0], weights="uniform", clamp="bogus")


def test_local_step_is_the_weighted_optimum_at_every_vertex(bunny, rng):
    # Paper Eq. 5-6 with a covariance built here from its definition: R_i S_i
    # is symmetric at the optimum, and Tr(R_i S_i) is the largest it can be.
    V, F = bunny
    X = V + 0.05 * rng.normal(size=V.shape)
    d = ARAPDeformer(V, F, [0], clamp="none")
    R = d.local_step(X)
    W = lap.cotangent_weights(V, F, clamp="none").tocoo()
    S = np.zeros((V.shape[0], 3, 3))
    np.add.at(S, W.row, W.data[:, None, None] * (V[W.row] - V[W.col])[:, :, None]
              * (X[W.row] - X[W.col])[:, None, :])
    M = R @ S
    scale = np.abs(S).max()
    assert np.abs(M - np.swapaxes(M, 1, 2)).max() < 1e-9 * scale
    s = np.linalg.svd(S, compute_uv=False)
    best = s[:, 0] + s[:, 1] + np.sign(np.linalg.det(S)) * s[:, 2]
    assert np.any(np.linalg.det(S) < 0.0)  # the reflection fix is exercised
    assert np.abs(np.trace(M, axis1=1, axis2=2) - best).max() < 1e-9 * scale


def test_result_fields_describe_the_returned_pose(bar):
    V, F = bar
    c, t, _, _ = bar_bend(V)
    d = ARAPDeformer(V, F, c)
    r = d.deform(t, max_iterations=7)
    assert r.iterations == 7 and not r.converged and len(r.energies) == 8
    R = d.local_step(r.vertices)
    assert np.abs(r.rotations - R).max() < 1e-12
    assert np.isclose(r.energies[-1], d.energy(r.vertices, R), rtol=1e-12)
    assert np.isclose(r.energies[0], d.energy(r.initial_guess, d.local_step(r.initial_guess)),
                      rtol=1e-12)
    assert np.isclose(d.diagonal, np.linalg.norm(V.max(axis=0) - V.min(axis=0)))


def test_stopping_rule_is_the_documented_one(bar):
    # Each criterion alone stops exactly where its definition says, and the
    # default rule stops at the first iteration that meets both.
    V, F = bar
    c, t, _, _ = bar_bend(V)
    d = ARAPDeformer(V, F, c)
    long = d.deform(t, max_iterations=300, energy_tol=0.0, step_tol=0.0,
                    snapshot_iterations=range(301))
    E = np.asarray(long.energies)
    decrease = (E[:-1] - E[1:]) / E[0]  # of iteration k at index k - 1
    move = np.array([np.linalg.norm(long.snapshots[k] - long.snapshots[k - 1], axis=1).max()
                     for k in range(1, 301)]) / d.diagonal
    first_energy = 1 + int(np.argmax(decrease < 1e-6))
    first_step = 1 + int(np.argmax(move < 1e-5))
    first_both = 1 + int(np.argmax((decrease < 1e-6) & (move < 1e-5)))
    assert 1 < first_energy < first_step < 300
    assert d.deform(t, step_tol=np.inf).iterations == first_energy
    assert d.deform(t, energy_tol=np.inf).iterations == first_step
    assert d.deform(t).iterations == first_both


@pytest.mark.parametrize("rule", [dict(step_tol=np.inf), dict()])
def test_stopping_rule_does_not_depend_on_the_scale_of_the_mesh(bar, rule):
    # 64 is a power of two, so the arithmetic is the same up to exponents.
    V, F = bar
    c, t, _, _ = bar_bend(V)
    a = ARAPDeformer(V, F, c).deform(t, **rule)
    b = ARAPDeformer(64.0 * V, F, c).deform(64.0 * t, **rule)
    assert a.converged and b.converged and a.iterations == b.iterations


def test_rest_guess_moves_only_the_constrained_vertices(bar):
    V, F = bar
    c, t, _, _ = bar_bend(V)
    g = ARAPDeformer(V, F, c).initial_guess(t, "rest")
    free = np.setdiff1d(np.arange(V.shape[0]), c)
    assert np.array_equal(g[c], t) and np.array_equal(g[free], V[free])


@pytest.mark.parametrize("options", [dict(clamp="none"), dict(weights="uniform")])
def test_thin_plate_guess_uses_the_deformers_own_weights(bunny, options):
    V, F = bunny
    ears = selection.by_coordinate(V, "y", above=0.8)
    c, t = selection.combine(V, selection.by_coordinate(V, "y", below=0.15), ears,
                             V[ears] + [-0.45, -0.15, 0.0])
    expected = LinearDeformer(V, F, c, method="thin_plate", **options).deform(t)
    default = LinearDeformer(V, F, c, method="thin_plate").deform(t)
    got = ARAPDeformer(V, F, c, **options).initial_guess(t, "thin_plate")
    assert np.abs(got - expected).max() < 1e-12
    assert np.abs(expected - default).max() > 1e-6  # the options matter on this mesh
