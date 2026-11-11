import numpy as np
import pytest

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
def test_gradient_matches_finite_differences(cube, rng, clamp):
    V, F = cube
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
    for key in ("initial_guess", "local_steps", "global_steps", "per_iteration"):
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


def test_unreferenced_vertex_is_reported(cube):
    V, F = cube
    V2 = np.vstack([V, [5.0, 5.0, 5.0]])
    with pytest.raises(ValueError, match="unreferenced"):
        ARAPDeformer(V2, F, [0])
