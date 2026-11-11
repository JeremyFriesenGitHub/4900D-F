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


@pytest.mark.parametrize("iterations", [1, 10])
def test_matches_libigl(bar, iterations):
    V, F = bar
    c, t, _, _ = bar_bend(V)
    d = ARAPDeformer(V, F, c, clamp="none")
    guess = d.initial_guess(t)
    ours = d.deform(t, init=guess, max_iterations=iterations, energy_tol=0.0, step_tol=0.0)
    theirs = igl_arap(V, F, c, t, guess, iterations)
    assert np.abs(ours.vertices - theirs).max() < 1e-4 * d.diagonal
