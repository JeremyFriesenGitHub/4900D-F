"""Dense reference solves with GeomProc's own Laplacian.

GeomProc's ``mesh.geometric_laplacian`` returns ``L_gp = D^-1 W - I``: each
row of cotangent weights is divided by its sum and the diagonal is -1. The
project handout warns that the Laplacian must be used "consistently". With
``L_gp`` on the left-hand side, the ARAP right-hand side must be built from
the same normalised entries::

    b_i = sum_j L_gp[i, j] * 1/2 (R_i + R_j) (p_j - p_i)

These functions solve the systems the way the GeomProc demos do (dense
matrix, constrained rows replaced by identity rows, ``numpy.linalg.solve``).
They are only meant for small meshes and for checking that the sparse
solver in ``arap.solver`` gives the same answer.
"""

import numpy as np

from .meshio import to_geomproc


def geomproc_laplacian(vertices, faces):
    """Dense ``L_gp`` from ``geomproc.mesh.geometric_laplacian``."""
    tm = to_geomproc(vertices, faces)
    tm.compute_connectivity()
    L, _negative, _boundary = tm.geometric_laplacian()
    return np.asarray(L, dtype=np.float64)


def row_replaced_solve(L, rhs, constrained, targets):
    """Solve ``L x = rhs`` with rows of constrained vertices replaced (GeomProc idiom)."""
    A = np.array(L, dtype=np.float64)
    b = np.array(rhs, dtype=np.float64)
    c = np.asarray(constrained, dtype=np.int64)
    A[c, :] = 0.0
    A[c, c] = 1.0
    b[c] = targets
    return np.linalg.solve(A, b)


def _off_diagonal(L):
    i, j = np.nonzero(L)
    keep = i != j
    return i[keep], j[keep], L[i[keep], j[keep]]


def rhs_from_rows(L, vertices, rotations):
    """``b_i = sum_j L[i, j] * 1/2 (R_i + R_j)(p_j - p_i)`` for any row-scaled operator."""
    V = np.asarray(vertices, dtype=np.float64)
    i, j, w = _off_diagonal(np.asarray(L))
    R_sum = rotations[i] + rotations[j]
    contrib = 0.5 * w[:, None] * np.einsum("hab,hb->ha", R_sum, V[j] - V[i])
    b = np.zeros_like(V)
    np.add.at(b, i, contrib)
    return b


def membrane_deformation(L, vertices, constrained, targets):
    """Delta d = 0 with GeomProc's operator: ``L_gp p' = L_gp p`` with replaced rows."""
    V = np.asarray(vertices, dtype=np.float64)
    return row_replaced_solve(L, L @ V, constrained, targets)


def arap_global_step(L, vertices, rotations, constrained, targets):
    """One ARAP global step with GeomProc's operator and a consistent right-hand side."""
    b = rhs_from_rows(L, vertices, rotations)
    return row_replaced_solve(L, b, constrained, targets)
