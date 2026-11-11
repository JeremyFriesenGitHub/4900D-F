"""As-rigid-as-possible surface deformation (Sorkine and Alexa 2007).

The energy (paper Eq. 7, with cell weights w_i = 1) is

    E(p') = sum_i sum_{j in N(i)} w_ij || (p'_i - p'_j) - R_i (p_i - p_j) ||^2

and is minimised by alternating two exact steps:

local step (Eq. 5-6)
    For fixed p', each R_i is the rotation closest to the covariance
    S_i = sum_j w_ij e_ij e'_ij^T, with e_ij = p_i - p_j (rest) and
    e'_ij = p'_i - p'_j (current).
global step (Eq. 8-9)
    For fixed rotations, p' solves L p' = b with
    b_i = sum_j (w_ij / 2) (R_i + R_j) (p_i - p_j),
    with the constrained vertices eliminated (Eq. 10).

``L`` depends only on the rest mesh and the constrained set, so it is
factored once; each iteration costs one batched SVD and three
back-substitutions.
"""

import time
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

from . import laplacian as lap
from .linear import ConstrainedSolver, LinearDeformer, _check_targets
from .rotations import fit_rotations


@dataclass
class ARAPResult:
    """Output of ``ARAPDeformer.deform``."""

    vertices: np.ndarray
    rotations: np.ndarray
    energies: list
    iterations: int
    converged: bool
    initial_guess: np.ndarray
    snapshots: dict = field(default_factory=dict)
    timings: dict = field(default_factory=dict)


class ARAPDeformer:
    """ARAP deformation of a triangle mesh with a fixed set of constraints.

    Parameters
    ----------
    vertices : (n, 3) array_like
        Rest positions p.
    faces : (m, 3) array_like of int
    constrained : array_like of int
        Indices of all constrained vertices (fixed and handle vertices), in
        any order. Targets passed to ``deform`` follow the same order.
    weights : {"cotangent", "uniform"}
        Edge weights w_ij. The paper uses cotangent weights.
    clamp : {"zero", "none", "geomproc"}
        Treatment of negative cotangent weights (see ``arap.laplacian``).
        The same weights are used in L, b, the covariances and the energy.
    """

    def __init__(self, vertices, faces, constrained, weights="cotangent", clamp="zero"):
        start = time.perf_counter()
        self.vertices = np.array(vertices, dtype=np.float64)
        self.faces = np.array(faces, dtype=np.int64)
        self.constrained = np.asarray(constrained, dtype=np.int64).ravel()
        if np.unique(self.constrained).size != self.constrained.size:
            raise ValueError("constrained indices must be unique")
        self.weights = weights
        self.clamp = clamp
        n = self.vertices.shape[0]
        self.n = n
        self.diagonal = float(np.linalg.norm(self.vertices.max(0) - self.vertices.min(0)))
        self._check_components()

        self.W = lap.edge_weights(self.vertices, self.faces, weights=weights, clamp=clamp)
        self.L = lap.laplacian_from_weights(self.W)

        # Half-edge arrays: one entry per ordered pair (i, j) with w_ij != 0.
        coo = self.W.tocoo()
        self._i = coo.row.astype(np.int64)
        self._j = coo.col.astype(np.int64)
        self._w = coo.data.astype(np.float64)
        self._rest_edges = self.vertices[self._i] - self.vertices[self._j]
        h = self._i.size
        # Sums half-edge quantities into their owning vertex i.
        self._gather = sp.csr_matrix((np.ones(h), (self._i, np.arange(h))), shape=(n, h))
        self.timings = {"weights": time.perf_counter() - start}

        start = time.perf_counter()
        self._solver = ConstrainedSolver(self.L, self.constrained)
        self.timings["factorization"] = time.perf_counter() - start
        self._thin_plate = None

    # ------------------------------------------------------------------ setup
    def _check_components(self):
        if self.constrained.size and (self.constrained.min() < 0
                                      or self.constrained.max() >= self.n):
            raise ValueError("constrained vertex index out of range")
        A = lap.adjacency_matrix(self.faces, self.n)
        _, labels = connected_components(A, directed=False)
        missing = set(np.unique(labels)) - set(np.unique(labels[self.constrained]))
        if missing:
            raise ValueError(
                f"{len(missing)} connected component(s) have no constrained vertex; "
                "add at least one constraint per component (a vertex that no face "
                "uses counts as its own component, so remove unreferenced vertices)"
            )

    # ------------------------------------------------------------- the steps
    def covariances(self, deformed):
        """Per-vertex covariance S_i = sum_j w_ij e_ij e'_ij^T, shape (n, 3, 3)."""
        e_def = deformed[self._i] - deformed[self._j]
        outer = (self._w[:, None, None] * self._rest_edges[:, :, None] * e_def[:, None, :])
        return (self._gather @ outer.reshape(-1, 9)).reshape(self.n, 3, 3)

    def local_step(self, deformed):
        """Optimal rotations R_i for the current positions (paper Eq. 5-6)."""
        return fit_rotations(self.covariances(deformed))

    def rhs(self, rotations):
        """Right-hand side b of L p' = b (paper Eq. 8)."""
        R_sum = rotations[self._i] + rotations[self._j]
        contrib = 0.5 * self._w[:, None] * np.einsum("hab,hb->ha", R_sum, self._rest_edges)
        return self._gather @ contrib

    def global_step(self, rotations, targets):
        """Positions minimising the energy for fixed rotations (Eq. 9-10)."""
        return self._solver.solve(self.rhs(rotations), targets)

    def energy(self, deformed, rotations):
        """ARAP energy (paper Eq. 7 with w_i = 1)."""
        e_def = deformed[self._i] - deformed[self._j]
        r = e_def - np.einsum("hab,hb->ha", rotations[self._i], self._rest_edges)
        return float(np.sum(self._w * np.einsum("ha,ha->h", r, r)))

    def gradient(self, deformed, rotations):
        """Gradient of the energy for fixed rotations: 4 (L p' - b)."""
        return 4.0 * (self.L @ deformed - self.rhs(rotations))

    # -------------------------------------------------------- initial guesses
    def initial_guess(self, targets, method="laplacian"):
        """Starting positions for the iterations.

        ``"laplacian"``  naive Laplacian editing, Delta d = 0 (course slides);
                         identical to one global step with all R_i = I.
        ``"thin_plate"`` Delta^2 d = 0 from the course slides.
        ``"rest"``       the rest pose with the constrained vertices moved.
        """
        constrained, targets = self._targets(targets)
        if method == "laplacian":
            return self._solver.solve(self.L @ self.vertices, targets)
        if method == "thin_plate":
            if self._thin_plate is None:
                self._thin_plate = LinearDeformer(
                    self.vertices, self.faces, self.constrained, method="thin_plate",
                    weights=self.weights, clamp=self.clamp,
                )
            return self._thin_plate.deform(targets)
        if method == "rest":
            guess = self.vertices.copy()
            guess[constrained] = targets
            return guess
        raise ValueError(f"unknown initial guess {method!r}")

    def _targets(self, targets):
        return _check_targets(self.vertices, self.constrained, targets)

    # ------------------------------------------------------------- iteration
    def deform(self, targets, init="laplacian", max_iterations=500,
               energy_tol=1e-6, step_tol=1e-5, snapshot_iterations=()):
        """Run the local/global iterations until convergence.

        Parameters
        ----------
        targets : (k, 3) array
            Positions of the constrained vertices, in the order of
            ``self.constrained`` (the order given to the constructor).
        init : str or (n, 3) array
            Initial guess method (see ``initial_guess``) or explicit positions.
        max_iterations : int
        energy_tol : float
            Stop when the relative energy decrease of an iteration is below
            this value ...
        step_tol : float
            ... and the largest vertex move is below ``step_tol`` times the
            bounding-box diagonal.
        snapshot_iterations : iterable of int
            Iterations whose positions are stored in ``result.snapshots``
            (0 is the initial guess).
        """
        constrained, targets = self._targets(targets)
        timings = {}
        start = time.perf_counter()
        if isinstance(init, str):
            current = self.initial_guess(targets, method=init)
        else:
            current = np.array(init, dtype=np.float64)
            if current.shape != self.vertices.shape:
                raise ValueError("explicit initial guess must have shape (n, 3)")
            current[constrained] = targets
        guess = current.copy()
        timings["initial_guess"] = time.perf_counter() - start

        wanted = set(int(k) for k in snapshot_iterations)
        snapshots = {0: guess.copy()} if 0 in wanted else {}

        local_time = global_time = 0.0
        t = time.perf_counter()
        rotations = self.local_step(current)
        local_time += time.perf_counter() - t
        energy = self.energy(current, rotations)
        energies = [energy]
        converged = False
        iterations = 0
        for iterations in range(1, max_iterations + 1):
            t = time.perf_counter()
            updated = self.global_step(rotations, targets)
            global_time += time.perf_counter() - t
            t = time.perf_counter()
            rotations = self.local_step(updated)
            local_time += time.perf_counter() - t
            new_energy = self.energy(updated, rotations)
            step = float(np.max(np.linalg.norm(updated - current, axis=1)))
            decrease = (energy - new_energy) / max(abs(energy), 1e-300)
            current, energy = updated, new_energy
            energies.append(energy)
            if iterations in wanted:
                snapshots[iterations] = current.copy()
            if decrease < energy_tol and step < step_tol * self.diagonal:
                converged = True
                break

        timings["local_steps"] = local_time
        timings["global_steps"] = global_time
        timings["iterations_total"] = local_time + global_time
        timings["per_iteration"] = (local_time + global_time) / max(iterations, 1)
        return ARAPResult(
            vertices=current,
            rotations=rotations,
            energies=energies,
            iterations=iterations,
            converged=converged,
            initial_guess=guess,
            snapshots=snapshots,
            timings=timings,
        )
