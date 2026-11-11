import numpy as np
import pytest
import scipy.sparse as sp

from arap import laplacian as lap
from arap.linear import ConstrainedSolver, LinearDeformer

from conftest import bar_bend


def test_constrained_solver_matches_dense_elimination(sphere, rng):
    V, F = sphere
    L, _ = lap.mesh_laplacian(V, F)
    c = np.array([0, 7, 20])
    rhs = rng.normal(size=(V.shape[0], 3))
    values = rng.normal(size=(3, 3))
    x = ConstrainedSolver(L, c).solve(rhs, values)
    free = np.setdiff1d(np.arange(V.shape[0]), c)
    A = L.toarray()
    expected = np.linalg.solve(A[np.ix_(free, free)], rhs[free] - A[np.ix_(free, c)] @ values)
    assert np.abs(x[free] - expected).max() < 1e-10
    assert np.array_equal(x[c], values)


def test_constrained_solver_needs_a_constraint(sphere):
    V, F = sphere
    L, _ = lap.mesh_laplacian(V, F)
    with pytest.raises(ValueError):
        ConstrainedSolver(L, [])


@pytest.mark.parametrize("method", ["laplacian", "thin_plate"])
def test_zero_displacement_returns_rest_pose(bar, method):
    V, F = bar
    c, _, _, _ = bar_bend(V)
    out = LinearDeformer(V, F, c, method=method).deform(V[c])
    assert np.abs(out - V).max() < 1e-12


@pytest.mark.parametrize("method", ["laplacian", "thin_plate"])
def test_translation_of_all_constraints_translates_the_mesh(bar, method):
    V, F = bar
    c, _, _, _ = bar_bend(V)
    shift = np.array([0.2, -0.1, 0.4])
    out = LinearDeformer(V, F, c, method=method).deform(V[c] + shift)
    assert np.abs(out - (V + shift)).max() < 1e-10


def test_membrane_satisfies_laplace_equation_on_free_vertices(bar):
    V, F = bar
    c, t, _, _ = bar_bend(V)
    out = LinearDeformer(V, F, c, method="laplacian").deform(t)
    L, _ = lap.mesh_laplacian(V, F)
    free = np.setdiff1d(np.arange(V.shape[0]), c)
    assert np.abs((L @ (out - V))[free]).max() < 1e-10


def test_thin_plate_satisfies_bilaplace_equation_on_free_vertices(bar):
    V, F = bar
    c, t, _, _ = bar_bend(V)
    out = LinearDeformer(V, F, c, method="thin_plate").deform(t)
    L, _ = lap.mesh_laplacian(V, F)
    M_inv = sp.diags(1.0 / lap.mixed_areas(V, F))
    free = np.setdiff1d(np.arange(V.shape[0]), c)
    residual = (M_inv @ L @ M_inv @ L @ (out - V))[free]
    assert np.abs(residual).max() < 1e-6 * np.abs(M_inv @ L @ M_inv @ L @ V).max()


def test_unknown_method_rejected(bar):
    V, F = bar
    with pytest.raises(ValueError):
        LinearDeformer(V, F, [0], method="spline")


@pytest.mark.parametrize("method", ["laplacian", "thin_plate"])
def test_constraint_order_is_up_to_the_caller(bar, rng, method):
    V, F = bar
    c, t, _, _ = bar_bend(V)
    perm = rng.permutation(c.size)
    ref = LinearDeformer(V, F, c, method=method).deform(t)
    out = LinearDeformer(V, F, c[perm], method=method).deform(t[perm])
    assert np.array_equal(out[c], t)
    assert np.abs(out - ref).max() < 1e-12


def test_constrained_solver_rejects_duplicates(sphere):
    V, F = sphere
    L, _ = lap.mesh_laplacian(V, F)
    with pytest.raises(ValueError, match="unique"):
        ConstrainedSolver(L, [3, 3])
