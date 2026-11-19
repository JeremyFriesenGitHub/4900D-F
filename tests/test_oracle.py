"""Comparison with libigl's ARAP (spokes energy = Sorkine and Alexa 2007).

libigl fits the 3x3 rotations with a fast single-precision routine, which
limits the end-to-end agreement to roughly 1e-5 of the bounding-box
diagonal on the bar and the bunny. The local step is therefore also
compared with libigl's double-precision fit, and the rest of the iteration
with libigl's own single-precision fit put in place of ours.
"""

import numpy as np
import pytest

from arap.solver import ARAPDeformer

from conftest import bar_bend

try:
    import igl
except ImportError:  # not installed, or installed but unable to load
    pytest.skip("libigl is not available", allow_module_level=True)


def igl_arap(V, F, constrained, targets, initial, iterations):
    data = igl.ARAPData()
    data.energy = igl.ARAP_ENERGY_TYPE_SPOKES
    data.max_iter = iterations
    igl.arap_precomputation(V, F, 3, constrained.astype(np.int32), data)
    return igl.arap_solve(targets, data, initial.copy())


def bunny_ears(V):
    from arap import selection as sel

    fixed = sel.by_coordinate(V, "y", below=0.15)
    ears = sel.by_coordinate(V, "y", above=0.8)
    return sel.combine(V, fixed, ears, V[ears] + [-0.45, -0.15, 0.0])


@pytest.mark.parametrize("iterations", [1, 10])
@pytest.mark.parametrize("name", ["bar", "bunny"])
def test_matches_libigl(request, name, iterations):
    # The bunny has negative cotangent weights, so this also checks the raw
    # (clamp="none") setting that libigl uses.
    V, F = request.getfixturevalue(name)
    c, t = bar_bend(V)[:2] if name == "bar" else bunny_ears(V)
    d = ARAPDeformer(V, F, c, clamp="none")
    guess = d.initial_guess(t)
    ours = d.deform(t, init=guess, max_iterations=iterations, energy_tol=0.0, step_tol=0.0)
    theirs = igl_arap(V, F, c, t, guess, iterations)
    assert np.abs(ours.vertices - theirs).max() < 1e-4 * d.diagonal


def igl_fit(S, single_precision):
    # libigl stacks the covariances row-major by matrix row ((3n, 3): all first
    # rows, then all second rows, ...) and returns the rotations side by side.
    n = S.shape[0]
    stack = np.ascontiguousarray(np.concatenate([S[:, 0, :], S[:, 1, :], S[:, 2, :]]))
    R = igl.fit_rotations(stack, single_precision)
    return np.ascontiguousarray(R.reshape(3, n, 3).transpose(1, 0, 2))


@pytest.mark.parametrize("name", ["bar", "bunny"])
def test_local_step_matches_libigl_double_precision_fit(request, name):
    # Exact oracle for the local step, including the cells where det(S) < 0.
    V, F = request.getfixturevalue(name)
    c, t = bar_bend(V)[:2] if name == "bar" else bunny_ears(V)
    d = ARAPDeformer(V, F, c, clamp="none")
    X = d.initial_guess(t)
    S = d.covariances(X)
    if name == "bunny":
        assert np.any(np.linalg.det(S) < 0.0)
    assert np.abs(d.local_step(X) - igl_fit(S, False)).max() < 1e-10


@pytest.mark.parametrize("name", ["bar", "bunny"])
def test_iteration_matches_libigl_with_its_single_precision_fit(request, name):
    # With libigl's own single-precision rotation fit (and its normalisation
    # of S) in our iteration, the two implementations agree almost exactly,
    # so the rest of the iteration (covariances, right-hand side, solve) is
    # checked to 1e-6 instead of 1e-4.
    V, F = request.getfixturevalue(name)
    c, t = bar_bend(V)[:2] if name == "bar" else bunny_ears(V)
    d = ARAPDeformer(V, F, c, clamp="none")
    guess = d.initial_guess(t)
    X = guess.copy()
    for _ in range(10):
        S = d.covariances(X)
        X = d.global_step(igl_fit(S / np.abs(S).max(), True), t)
    theirs = igl_arap(V, F, c, t, guess, 10)
    assert np.abs(X - theirs).max() < 1e-6 * d.diagonal
