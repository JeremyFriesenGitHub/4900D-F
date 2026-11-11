"""Sparse discrete Laplace operators for triangle meshes.

All matrices use the sign convention of the ARAP paper (Sorkine and Alexa
2007, Eq. 9)::

    L = D - W,    D = diag(sum_j w_ij)

so ``L`` is symmetric positive semidefinite. The course slides and GeomProc
write the operator with the opposite sign (negative diagonal), which is
``-L``; the solutions of the linear systems used here do not depend on that
choice as long as the right-hand side uses the same sign.

Edge weights
------------
``cotangent_weights`` returns the symmetric matrix ``W`` with

    w_ij = 1/2 (cot alpha_ij + cot beta_ij)

where alpha_ij and beta_ij are the angles opposite edge (i, j). A boundary
edge has a single opposite angle. Three ways of treating negative values are
available through ``clamp``:

``"none"``
    The raw weights, exactly as in the paper.
``"zero"``
    Edge weights below zero are set to zero. This keeps the rest pose a
    fixed point of the ARAP iterations on meshes with obtuse triangles (see
    the README for the measurements that motivated it).
``"geomproc"``
    Reproduces GeomProc's ``mesh.geometric_laplacian``: a negative
    cotangent at a corner is replaced by half of the triangle area before
    the two corners of an edge are added.
"""

import numpy as np
import scipy.sparse as sp

CLAMP_MODES = ("none", "zero", "geomproc")


def _as_arrays(vertices, faces):
    V = np.asarray(vertices, dtype=np.float64)
    F = np.asarray(faces, dtype=np.int64)
    if V.ndim != 2 or V.shape[1] != 3:
        raise ValueError("vertices must have shape (n, 3)")
    if F.ndim != 2 or F.shape[1] != 3:
        raise ValueError("faces must have shape (m, 3)")
    return V, F


def corner_cotangents(vertices, faces):
    """Cotangent of the interior angle at every face corner.

    Returns an array ``C`` of shape (m, 3), where ``C[f, k]`` is the cotangent
    of the angle at vertex ``faces[f, k]``. That angle is opposite the edge
    ``(faces[f, k+1], faces[f, k+2])`` (indices modulo 3). Corners of
    degenerate (zero-area) faces get a cotangent of 0, so those faces do not
    contribute to the operator.
    """
    V, F = _as_arrays(vertices, faces)
    cot = np.zeros(F.shape, dtype=np.float64)
    for k in range(3):
        a = V[F[:, k]]
        u = V[F[:, (k + 1) % 3]] - a
        v = V[F[:, (k + 2) % 3]] - a
        dot = np.einsum("ij,ij->i", u, v)
        cross = np.linalg.norm(np.cross(u, v), axis=1)
        scale = np.einsum("ij,ij->i", u, u) + np.einsum("ij,ij->i", v, v)
        ok = cross > 1e-12 * scale
        cot[ok, k] = dot[ok] / cross[ok]
    return cot


def face_areas(vertices, faces):
    """Area of every triangle, shape (m,)."""
    V, F = _as_arrays(vertices, faces)
    e1 = V[F[:, 1]] - V[F[:, 0]]
    e2 = V[F[:, 2]] - V[F[:, 0]]
    return 0.5 * np.linalg.norm(np.cross(e1, e2), axis=1)


def edge_index_pairs(faces):
    """Directed edges opposite each corner, as ``(I, J, corner)`` arrays.

    Corner ``k`` of face ``f`` is opposite the edge ``(I, J)`` with
    ``I = faces[f, (k+1) % 3]`` and ``J = faces[f, (k+2) % 3]``. The arrays are
    ordered corner-major: all faces for corner 0, then corner 1, then 2.
    """
    F = np.asarray(faces, dtype=np.int64)
    I = np.concatenate([F[:, 1], F[:, 2], F[:, 0]])
    J = np.concatenate([F[:, 2], F[:, 0], F[:, 1]])
    return I, J


def cotangent_weights(vertices, faces, clamp="zero"):
    """Symmetric sparse matrix of cotangent edge weights (zero diagonal).

    Parameters
    ----------
    vertices : (n, 3) array_like
    faces : (m, 3) array_like of int
    clamp : {"zero", "none", "geomproc"}
        How negative weights are treated; see the module docstring.

    Returns
    -------
    W : scipy.sparse.csr_matrix, shape (n, n)
    """
    if clamp not in CLAMP_MODES:
        raise ValueError(f"clamp must be one of {CLAMP_MODES}, got {clamp!r}")
    V, F = _as_arrays(vertices, faces)
    n = V.shape[0]
    cot = corner_cotangents(V, F)
    if clamp == "geomproc":
        # GeomProc replaces an obtuse corner's cotangent with
        # |(vi - vk) x (vj - vk)| / 4, i.e. half of the triangle area.
        half_area = 0.5 * face_areas(V, F)
        cot = np.where(cot < 0.0, half_area[:, None], cot)
    I, J = edge_index_pairs(F)
    vals = 0.5 * np.concatenate([cot[:, 0], cot[:, 1], cot[:, 2]])
    half = sp.coo_matrix((vals, (I, J)), shape=(n, n))
    W = (half + half.T).tocsr()
    W.sum_duplicates()
    if clamp == "zero":
        W.data = np.maximum(W.data, 0.0)
    W.eliminate_zeros()
    return W


def uniform_weights(faces, n_vertices):
    """Symmetric sparse matrix with w_ij = 1 for every mesh edge."""
    F = np.asarray(faces, dtype=np.int64)
    W = adjacency_matrix(F, n_vertices).astype(np.float64)
    W.data[:] = 1.0
    return W


def adjacency_matrix(faces, n_vertices):
    """Symmetric 0/1 vertex adjacency of the mesh (no self loops)."""
    F = np.asarray(faces, dtype=np.int64)
    I, J = edge_index_pairs(F)
    A = sp.coo_matrix((np.ones(I.shape[0]), (I, J)), shape=(n_vertices, n_vertices))
    A = (A + A.T).tocsr()
    A.data[:] = 1.0
    return A


def edge_weights(vertices, faces, weights="cotangent", clamp="zero"):
    """Dispatch to ``cotangent_weights`` or ``uniform_weights``."""
    if weights == "cotangent":
        return cotangent_weights(vertices, faces, clamp=clamp)
    if weights == "uniform":
        return uniform_weights(faces, np.asarray(vertices).shape[0])
    raise ValueError(f"weights must be 'cotangent' or 'uniform', got {weights!r}")


def laplacian_from_weights(W):
    """``L = D - W`` (paper sign, positive semidefinite)."""
    W = sp.csr_matrix(W)
    d = np.asarray(W.sum(axis=1)).ravel()
    return (sp.diags(d) - W).tocsr()


def mixed_areas(vertices, faces):
    """Mixed Voronoi area per vertex (Meyer et al. 2003).

    This is the ``A_i`` of the course's cotangent formula
    ``Delta f(v_i) = 1/(2 A_i) sum_j (cot a_ij + cot b_ij)(f_j - f_i)``.
    For a non-obtuse triangle each corner receives its Voronoi region; for an
    obtuse triangle the obtuse corner receives half of the area and the other
    two a quarter each. The areas sum to the total surface area.
    """
    V, F = _as_arrays(vertices, faces)
    n = V.shape[0]
    cot = corner_cotangents(V, F)
    area = face_areas(V, F)
    obtuse = np.any(cot < 0.0, axis=1)
    A = np.zeros(n)
    for k in range(3):
        k1, k2 = (k + 1) % 3, (k + 2) % 3
        p = V[F[:, k]]
        l1 = np.sum((V[F[:, k1]] - p) ** 2, axis=1)  # edge opposite corner k2
        l2 = np.sum((V[F[:, k2]] - p) ** 2, axis=1)  # edge opposite corner k1
        voronoi = (l1 * cot[:, k2] + l2 * cot[:, k1]) / 8.0
        part = np.where(obtuse, np.where(cot[:, k] < 0.0, area / 2.0, area / 4.0), voronoi)
        A += np.bincount(F[:, k], weights=part, minlength=n)
    return A


def mesh_laplacian(vertices, faces, weights="cotangent", clamp="zero"):
    """Convenience wrapper returning ``(L, W)``."""
    W = edge_weights(vertices, faces, weights=weights, clamp=clamp)
    return laplacian_from_weights(W), W
