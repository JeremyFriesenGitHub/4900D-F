"""Quality measures for deformations."""

import numpy as np


def unique_edges(faces):
    """Undirected edges of a triangle mesh, shape (e, 2)."""
    F = np.asarray(faces, dtype=np.int64)
    E = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    return np.unique(np.sort(E, axis=1), axis=0)


def edge_lengths(vertices, edges):
    V = np.asarray(vertices, dtype=np.float64)
    return np.linalg.norm(V[edges[:, 0]] - V[edges[:, 1]], axis=1)


def edge_length_error(rest, deformed, faces):
    """Relative edge-length change between two poses.

    Returns a dict with the RMS, mean and maximum of
    ``(|e'| - |e|) / |e|`` over all edges. The ARAP paper reports a
    "relative RMS edge-length error" (Table 1) without defining it; this is
    the definition used in this project.
    """
    E = unique_edges(faces)
    l0 = edge_lengths(rest, E)
    l1 = edge_lengths(deformed, E)
    rel = (l1 - l0) / l0
    return {
        "rms_relative": float(np.sqrt(np.mean(rel ** 2))),
        "mean_abs_relative": float(np.mean(np.abs(rel))),
        "max_abs_relative": float(np.max(np.abs(rel))),
    }


def per_vertex_stretch(rest, deformed, faces):
    """Mean absolute relative edge-length change around each vertex."""
    V = np.asarray(rest, dtype=np.float64)
    E = unique_edges(faces)
    rel = np.abs(edge_lengths(deformed, E) - edge_lengths(V, E)) / edge_lengths(V, E)
    n = V.shape[0]
    total = np.bincount(E[:, 0], rel, n) + np.bincount(E[:, 1], rel, n)
    count = np.bincount(E[:, 0], minlength=n) + np.bincount(E[:, 1], minlength=n)
    return total / np.maximum(count, 1)


def bbox_diagonal(vertices):
    V = np.asarray(vertices, dtype=np.float64)
    return float(np.linalg.norm(V.max(axis=0) - V.min(axis=0)))
