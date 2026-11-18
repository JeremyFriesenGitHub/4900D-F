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
    try:
        import igl
    except ImportError:  # not installed, or installed but unable to load
        pytest.skip("libigl is not available")
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


def test_mixed_areas_of_an_obtuse_triangle():
    # The corner at vertex 0 is obtuse: it gets half the area, the others a quarter.
    V = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [-0.5, 0.2, 0.0]])
    F = np.array([[0, 1, 2]])
    A = lap.mixed_areas(V, F)
    area = lap.face_areas(V, F)[0]
    assert np.allclose(A, [area / 2.0, area / 4.0, area / 4.0])


def test_deformers_clamp_negative_weights_by_default(bunny):
    from arap.solver import ARAPDeformer
    from arap.linear import LinearDeformer

    V, F = bunny
    d = ARAPDeformer(V, F, [0])
    assert d.clamp == "zero"
    assert d.W.data.min() >= 0.0
    assert np.abs(d.W - lap.cotangent_weights(V, F, clamp="zero")).max() == 0.0
    linear = LinearDeformer(V, F, [0])
    L_zero, _ = lap.mesh_laplacian(V, F, clamp="zero")
    assert np.abs(linear.system_matrix - L_zero).max() == 0.0


def test_zero_clamp_acts_on_the_summed_edge_weight(bunny):
    V, F = bunny
    raw = lap.cotangent_weights(V, F, clamp="none")
    zero = lap.cotangent_weights(V, F, clamp="zero")
    assert abs(zero - raw.maximum(0)).max() == 0.0


def test_zero_clamp_on_two_triangles_with_known_cotangents():
    # Two triangles sharing the edge (0, 1). At vertex 2 the angle is acute
    # (cot = 1.875); at vertex 3 it is obtuse (cot = -0.75).
    V = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 2.0, 0.0], [0.5, -0.25, 0.0]])
    F = np.array([[0, 1, 2], [1, 0, 3]])
    cot = lap.corner_cotangents(V, F)
    assert cot[0, 2] == pytest.approx(1.875)
    assert cot[1, 2] == pytest.approx(-0.75)
    # The edge weight is clamped after the two corners are added: it stays
    # 1/2 (1.875 - 0.75), where clamping each corner first would give 0.9375.
    W = lap.cotangent_weights(V, F, clamp="zero")
    assert W[0, 1] == pytest.approx(0.5625)
    # A boundary edge opposite an obtuse corner alone gets weight 0.
    V2 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, -0.25, 0.0]])
    W2 = lap.cotangent_weights(V2, np.array([[1, 0, 2]]), clamp="zero")
    assert W2[0, 1] == 0.0
    assert lap.cotangent_weights(V2, np.array([[1, 0, 2]]), clamp="none")[0, 1] == pytest.approx(-0.375)


def test_cotangent_weight_of_an_equilateral_triangle():
    # w_ij = 1/2 cot 60 degrees for the single angle opposite each edge.
    V = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, np.sqrt(3.0) / 2.0, 0.0]])
    W = lap.cotangent_weights(V, np.array([[0, 1, 2]]), clamp="none")
    expected = 0.5 / np.sqrt(3.0)
    assert np.allclose(W.toarray()[np.triu_indices(3, 1)], expected)
