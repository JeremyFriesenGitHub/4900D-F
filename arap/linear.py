"""Linear surface deformation from the course slides (Topic 7).

Both methods solve for per-vertex displacements ``d`` with hard constraints:
handle vertices get their prescribed displacement and fixed vertices get
zero. Free vertices satisfy

* membrane / "Laplacian deformation":  ``Delta d_i = 0``
* thin plate:                           ``Delta^2 d_i = 0``

with the cotangent operator ``Delta = M^{-1} (W - D) = -M^{-1} L`` and ``M``
the diagonal of mixed areas. For the membrane system the per-row area
factor cancels, so ``Delta d = 0`` is the same as ``L d = 0``, i.e.
``L p' = L p``. That is naive Laplacian editing, the initial guess the ARAP
paper uses in its Fig. 3.

Constraints are imposed by elimination: the constrained columns move to the
right-hand side and the free block stays symmetric positive definite, so it
is factored once and reused for every solve.
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.sparse.csgraph import connected_components

from . import laplacian as lap


class ConstrainedSolver:
    """Factorised solver for ``K x = rhs`` with ``x[constrained]`` prescribed.

    ``K`` must be symmetric and its free block positive definite (true for a
    mesh Laplacian when every connected component has a constrained vertex).
    The indices may come in any order; ``values`` passed to ``solve`` must
    follow the same order.
    """

    def __init__(self, K, constrained):
        K = sp.csr_matrix(K)
        n = K.shape[0]
        constrained = np.asarray(constrained, dtype=np.int64).ravel()
        if constrained.size == 0:
            raise ValueError("at least one constrained vertex is required")
        if constrained.min() < 0 or constrained.max() >= n:
            raise ValueError("constrained vertex index out of range")
        if np.unique(constrained).size != constrained.size:
            raise ValueError("constrained indices must be unique")
        # A part of the matrix's graph with no constrained vertex makes the
        # free block singular; floating-point LU need not notice, so check.
        graph = abs(K)
        graph.eliminate_zeros()
        _, labels = connected_components(graph, directed=False)
        if np.setdiff1d(labels, labels[constrained]).size:
            raise ValueError(
                "the constrained system is singular: make sure every connected "
                "component of the mesh contains a constrained vertex"
            )
        mask = np.ones(n, dtype=bool)
        mask[constrained] = False
        self.n = n
        self.constrained = constrained
        self.free = np.flatnonzero(mask)
        K_free_rows = K[self.free]
        self._K_fc = K_free_rows[:, constrained].tocsr()
        K_ff = K_free_rows[:, self.free].tocsc()
        try:
            self._lu = spla.splu(
                K_ff,
                permc_spec="MMD_AT_PLUS_A",
                diag_pivot_thresh=0.0,
                options={"SymmetricMode": True},
            )
        except RuntimeError as exc:  # singular factor
            raise ValueError(
                "the constrained system is singular: make sure every connected "
                "component of the mesh contains a constrained vertex"
            ) from exc

    def solve(self, rhs, values):
        """Solve with full-length ``rhs`` (n, d) and constrained ``values`` (k, d)."""
        rhs = np.asarray(rhs, dtype=np.float64)
        values = np.asarray(values, dtype=np.float64)
        x = np.empty((self.n,) + rhs.shape[1:], dtype=np.float64)
        x[self.constrained] = values
        if self.free.size:
            x[self.free] = self._lu.solve(rhs[self.free] - self._K_fc @ values)
        return x


def _check_targets(vertices, constrained, targets):
    constrained = np.asarray(constrained, dtype=np.int64)
    targets = np.asarray(targets, dtype=np.float64)
    if targets.shape != (constrained.size, 3):
        raise ValueError(
            f"targets must have shape ({constrained.size}, 3), got {targets.shape}"
        )
    if np.unique(constrained).size != constrained.size:
        raise ValueError("constrained indices must be unique")
    return constrained, targets


class LinearDeformer:
    """Membrane or thin-plate deformation with a reusable factorisation.

    Parameters
    ----------
    vertices, faces : rest mesh.
    constrained : indices of handle and fixed vertices.
    method : {"laplacian", "thin_plate"}
    weights, clamp : passed to ``laplacian.edge_weights``.
    """

    def __init__(self, vertices, faces, constrained, method="laplacian",
                 weights="cotangent", clamp="zero"):
        self.vertices = np.asarray(vertices, dtype=np.float64)
        self.faces = np.asarray(faces, dtype=np.int64)
        self.constrained = np.asarray(constrained, dtype=np.int64)
        self.method = method
        L, _ = lap.mesh_laplacian(self.vertices, self.faces, weights=weights, clamp=clamp)
        if method == "laplacian":
            K = L
        elif method == "thin_plate":
            area = lap.mixed_areas(self.vertices, self.faces)
            if np.any(area <= 0.0):
                raise ValueError("thin-plate deformation needs positive vertex areas")
            K = L @ sp.diags(1.0 / area) @ L
        else:
            raise ValueError(f"unknown method {method!r}")
        self.system_matrix = sp.csr_matrix(K)
        self._solver = ConstrainedSolver(self.system_matrix, self.constrained)

    def deform(self, targets):
        """Deformed vertex positions for the given constrained targets."""
        constrained, targets = _check_targets(self.vertices, self.constrained, targets)
        displacement = targets - self.vertices[constrained]
        zero = np.zeros_like(self.vertices)
        d = self._solver.solve(zero, displacement)
        return self.vertices + d


def laplacian_deformation(vertices, faces, constrained, targets, **kwargs):
    """One-shot membrane deformation (``Delta d = 0``)."""
    return LinearDeformer(vertices, faces, constrained, method="laplacian", **kwargs).deform(targets)


def thin_plate_deformation(vertices, faces, constrained, targets, **kwargs):
    """One-shot thin-plate deformation (``Delta^2 d = 0``)."""
    return LinearDeformer(vertices, faces, constrained, method="thin_plate", **kwargs).deform(targets)
