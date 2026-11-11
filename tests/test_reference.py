"""The sparse solver agrees with dense solves built on GeomProc's operator."""

import numpy as np
import pytest

from arap import reference
from arap.linear import LinearDeformer
from arap.solver import ARAPDeformer

from conftest import bar_bend


@pytest.fixture(scope="module")
def setup(bar):
    V, F = bar
    c, t, _, _ = bar_bend(V)
    L_gp = reference.geomproc_laplacian(V, F)
    d = ARAPDeformer(V, F, c, clamp="geomproc")
    return V, F, c, t, L_gp, d


def test_geomproc_operator_has_unit_negative_diagonal(setup):
    _, _, _, _, L_gp, _ = setup
    assert np.allclose(np.diag(L_gp), -1.0)
    assert np.allclose(L_gp.sum(axis=1), 0.0)


def test_membrane_matches_geomproc_dense_solve(setup):
    V, F, c, t, L_gp, d = setup
    dense = reference.membrane_deformation(L_gp, V, c, t)
    sparse = LinearDeformer(V, F, c, method="laplacian", clamp="geomproc").deform(t)
    assert np.abs(dense - sparse).max() < 1e-10
    assert np.abs(dense - d.initial_guess(t)).max() < 1e-10


def test_arap_step_matches_geomproc_with_consistent_rhs(setup):
    V, F, c, t, L_gp, d = setup
    R = d.local_step(d.initial_guess(t))
    dense = reference.arap_global_step(L_gp, V, R, c, t)
    assert np.abs(dense - d.global_step(R, t)).max() < 1e-10


def test_inconsistent_rhs_gives_a_different_answer(setup):
    # Pairing GeomProc's row-normalised operator with the paper's
    # unnormalised right-hand side is the mistake the handout warns about.
    V, F, c, t, L_gp, d = setup
    R = d.local_step(d.initial_guess(t))
    wrong = reference.row_replaced_solve(L_gp, -d.rhs(R), c, t)
    right = d.global_step(R, t)
    assert np.abs(wrong - right).max() > 1e-2 * d.diagonal


def test_full_arap_with_geomproc_operator(setup):
    V, F, c, t, L_gp, d = setup
    X = d.initial_guess(t)
    R = d.local_step(X)
    for _ in range(5):
        X = reference.arap_global_step(L_gp, V, R, c, t)
        R = d.local_step(X)
    sparse = d.deform(t, max_iterations=5, energy_tol=0.0, step_tol=0.0)
    assert sparse.iterations == 5
    assert np.abs(sparse.vertices - X).max() < 1e-10


def test_area_weighted_operator_with_consistent_rhs(setup):
    # The handout: if the Voronoi area weights a row of L, it must also
    # weight that row of b. Delta = M^-1 (W - D) with mixed areas M.
    from arap import laplacian as lap

    V, F, c, t, _, _ = setup
    d = ARAPDeformer(V, F, c)  # default clamp="zero"
    M_inv = 1.0 / lap.mixed_areas(V, F)
    L_area = -(M_inv[:, None] * d.L.toarray())
    R = d.local_step(d.initial_guess(t))
    dense = reference.arap_global_step(L_area, V, R, c, t)
    assert np.abs(dense - d.global_step(R, t)).max() < 1e-10
    wrong = reference.row_replaced_solve(L_area, -d.rhs(R), c, t)
    assert np.abs(wrong - dense).max() > 1e-2 * d.diagonal
