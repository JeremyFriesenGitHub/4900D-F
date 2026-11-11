"""Helpers to pick constrained vertices and build their target positions."""

import numpy as np

import geomproc

_AXES = {"x": 0, "y": 1, "z": 2}


def _axis_index(axis):
    if isinstance(axis, str):
        return _AXES[axis.lower()]
    return int(axis)


def by_coordinate(vertices, axis, below=None, above=None):
    """Vertices in a slab along one axis, with thresholds relative to the extent.

    ``below=0.1`` keeps vertices whose coordinate is under
    ``min + 0.1 * (max - min)``; ``above=0.9`` keeps those over
    ``min + 0.9 * (max - min)``. Giving both keeps the union.
    """
    V = np.asarray(vertices, dtype=np.float64)
    c = V[:, _axis_index(axis)]
    lo, hi = c.min(), c.max()
    mask = np.zeros(V.shape[0], dtype=bool)
    if below is not None:
        mask |= c < lo + below * (hi - lo)
    if above is not None:
        mask |= c > lo + above * (hi - lo)
    return np.flatnonzero(mask)


def within_radius(vertices, center, radius):
    """Vertices within ``radius`` of ``center``."""
    V = np.asarray(vertices, dtype=np.float64)
    return np.flatnonzero(np.linalg.norm(V - np.asarray(center, dtype=np.float64), axis=1) <= radius)


def nearest(vertices, point):
    """Index of the vertex closest to ``point``."""
    V = np.asarray(vertices, dtype=np.float64)
    return int(np.argmin(np.linalg.norm(V - np.asarray(point, dtype=np.float64), axis=1)))


def rotation(axis, degrees):
    """3x3 rotation about a coordinate axis, using GeomProc's transforms."""
    angle = np.radians(degrees)
    fn = {"x": geomproc.rotation_x, "y": geomproc.rotation_y, "z": geomproc.rotation_z}
    return np.asarray(fn[axis.lower()](angle), dtype=np.float64)


def rigid_targets(points, rotation_matrix=None, pivot=None, translation=None):
    """Apply ``p -> R (p - c) + c + t`` to a set of points."""
    P = np.asarray(points, dtype=np.float64)
    R = np.eye(3) if rotation_matrix is None else np.asarray(rotation_matrix, dtype=np.float64)
    c = P.mean(axis=0) if pivot is None else np.asarray(pivot, dtype=np.float64)
    t = np.zeros(3) if translation is None else np.asarray(translation, dtype=np.float64)
    return (P - c) @ R.T + c + t


def combine(vertices, fixed, handles, handle_targets):
    """Merge fixed and handle constraints into sorted ``(indices, targets)``.

    Fixed vertices keep their rest position. ``handle_targets`` must follow
    the order of ``handles``. The result is sorted by vertex index, which is
    the order ``ARAPDeformer`` expects.
    """
    V = np.asarray(vertices, dtype=np.float64)
    fixed = np.asarray(fixed, dtype=np.int64)
    handles = np.asarray(handles, dtype=np.int64)
    if np.intersect1d(fixed, handles).size:
        raise ValueError("a vertex cannot be both fixed and a handle")
    idx = np.concatenate([fixed, handles])
    tgt = np.vstack([V[fixed], np.asarray(handle_targets, dtype=np.float64).reshape(-1, 3)])
    order = np.argsort(idx)
    return idx[order], tgt[order]


def save_indices(path, indices):
    """Write vertex indices, one per line (same format as GeomProc's .bnd files)."""
    np.savetxt(path, np.asarray(indices, dtype=np.int64), fmt="%d")


def load_indices(path):
    """Read vertex indices written by ``save_indices``."""
    return np.atleast_1d(np.loadtxt(path, dtype=np.int64))
