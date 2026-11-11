"""Comparison with libigl's ARAP (spokes energy = Sorkine and Alexa 2007).

libigl computes the 3x3 rotations in single precision, so agreement is
limited to roughly 1e-5 of the bounding-box diagonal.
"""

import numpy as np
import pytest

from arap.solver import ARAPDeformer

from conftest import bar_bend

igl = pytest.importorskip("igl")


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
