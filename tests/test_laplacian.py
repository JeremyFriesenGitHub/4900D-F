import numpy as np
import pytest
import scipy.sparse as sp

from arap import laplacian as lap
from arap.reference import geomproc_laplacian


@pytest.mark.parametrize("clamp", lap.CLAMP_MODES)
def test_weights_symmetric_with_zero_diagonal(bunny, clamp):
    V, F = bunny
    W = lap.cotangent_weights(V, F, clamp=clamp)
    assert abs(W - W.T).max() < 1e-12
    assert np.all(W.diagonal() == 0.0)


@pytest.mark.parametrize("clamp", ["zero", "geomproc"])
def test_clamped_weights_are_nonnegative(bunny, clamp):
    V, F = bunny
    W = lap.cotangent_weights(V, F, clamp=clamp)
    assert W.data.min() >= 0.0


def test_raw_weights_have_negative_entries_on_bunny(bunny):
    # The reason the clamp exists: the scanned bunny has obtuse triangles.
    V, F = bunny
    W = lap.cotangent_weights(V, F, clamp="none")
    assert W.data.min() < 0.0


@pytest.mark.parametrize("weights", ["cotangent", "uniform"])
def test_laplacian_rows_sum_to_zero_and_is_psd(sphere, weights):
    V, F = sphere
    L, _ = lap.mesh_laplacian(V, F, weights=weights)
    assert np.abs(np.asarray(L.sum(axis=1))).max() < 1e-12
    eig = np.linalg.eigvalsh(L.toarray())
    assert eig.min() > -1e-10
    # one connected component -> exactly one zero eigenvalue
    assert np.sum(eig < 1e-9) == 1


def test_laplacian_annihilates_linear_functions_on_flat_patch():
    # Interior rows of the cotangent Laplacian vanish on linear functions.
    x, y = np.meshgrid(np.linspace(0, 1, 6), np.linspace(0, 1, 6))
    rng = np.random.default_rng(0)
    V = np.column_stack([x.ravel(), y.ravel(), np.zeros(36)])
    V[:, :2] += 0.03 * rng.normal(size=(36, 2))
    F = []
    for r in range(5):
        for c in range(5):
            a, b, d, e = r * 6 + c, r * 6 + c + 1, (r + 1) * 6 + c, (r + 1) * 6 + c + 1
            F += [[a, b, e], [a, e, d]]
    F = np.array(F)
    L, _ = lap.mesh_laplacian(V, F, clamp="none")
    interior = [r * 6 + c for r in range(1, 5) for c in range(1, 5)]
    f = 2.0 * V[:, 0] - 3.0 * V[:, 1] + 1.0
    assert np.abs((L @ f)[interior]).max() < 1e-12


def test_raw_cotangent_matches_libigl(bunny):
    igl = pytest.importorskip("igl")
    V, F = bunny
    L, _ = lap.mesh_laplacian(V, F, clamp="none")
    C = igl.cotmatrix(V, F)
    # libigl's cotmatrix is the negative semidefinite W - D.
    assert abs(L + C).max() < 1e-10


@pytest.mark.parametrize("name", ["bunny", "sphere", "cube"])
def test_geomproc_mode_reproduces_geomproc_operator(request, name):
    V, F = request.getfixturevalue(name)
    W = lap.cotangent_weights(V, F, clamp="geomproc")
    d = np.asarray(W.sum(axis=1)).ravel()
    ours = (sp.diags(1.0 / d) @ W - sp.eye(V.shape[0])).toarray()
    theirs = geomproc_laplacian(V, F)
    assert np.abs(ours - theirs).max() < 1e-10


def test_mixed_areas_sum_to_surface_area(bunny):
    V, F = bunny
    A = lap.mixed_areas(V, F)
    assert np.all(A > 0.0)
    assert np.isclose(A.sum(), lap.face_areas(V, F).sum(), rtol=1e-12)


def test_mixed_areas_equal_voronoi_on_acute_mesh():
    # Regular tetrahedron: every corner is 60 degrees, so each vertex gets a
    # third of each incident face.
    V = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], dtype=float)
    F = np.array([[0, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2]])
    A = lap.mixed_areas(V, F)
    area = lap.face_areas(V, F)
    assert np.allclose(A, area.sum() / 4.0)


def test_uniform_weights_count_neighbours(cube):
    V, F = cube
    W = lap.uniform_weights(F, V.shape[0])
    edges = np.unique(np.sort(np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1), axis=0)
    assert W.nnz == 2 * edges.shape[0]
    assert np.all(W.data == 1.0)


def test_degenerate_face_is_ignored():
    V = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [2, 0, 0]], dtype=float)
    F = np.array([[0, 1, 2], [0, 3, 1]])  # second face is collinear
    cot = lap.corner_cotangents(V, F)
    assert np.all(cot[1] == 0.0)
    assert np.all(np.isfinite(cot))


def test_bad_clamp_rejected(cube):
    V, F = cube
    with pytest.raises(ValueError):
        lap.cotangent_weights(V, F, clamp="sometimes")
