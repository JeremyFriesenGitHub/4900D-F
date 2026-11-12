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


def surface_area(vertices, faces):
    """Total triangle area."""
    V = np.asarray(vertices, dtype=np.float64)
    F = np.asarray(faces, dtype=np.int64)
    return float(0.5 * np.linalg.norm(np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]),
                                      axis=1).sum())


def is_closed(faces):
    """True if every edge is shared by exactly two faces."""
    F = np.asarray(faces, dtype=np.int64)
    E = np.sort(np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1)
    _, counts = np.unique(E, axis=0, return_counts=True)
    return bool(np.all(counts == 2))


def signed_volume(vertices, faces):
    """Signed volume of a closed mesh by the divergence theorem.

    Positive when the faces are oriented outwards. Returns ``None`` for a
    mesh with boundary edges, where the volume is not defined. Where the
    surface passes through itself, regions turned inside out count
    negatively, so the value can drop far below the true enclosed volume.
    """
    if not is_closed(faces):
        return None
    V = np.asarray(vertices, dtype=np.float64)
    F = np.asarray(faces, dtype=np.int64)
    return float(np.einsum("ij,ij->i", V[F[:, 0]], np.cross(V[F[:, 1]], V[F[:, 2]])).sum() / 6.0)


def enclosed_volume(vertices, faces):
    """Volume enclosed by a closed mesh, whatever the face orientation (``None`` if open)."""
    v = signed_volume(vertices, faces)
    return None if v is None else abs(v)


def area_volume_change(rest, deformed, faces):
    """Relative change of surface area and enclosed volume, e.g. -0.05 = 5 % smaller.

    The volume change compares signed volumes, so a result that turns
    (partly) inside out shows up as a change below -100 % instead of being
    hidden by the absolute value.
    """
    a0, a1 = surface_area(rest, faces), surface_area(deformed, faces)
    v0, v1 = signed_volume(rest, faces), signed_volume(deformed, faces)
    return {
        "area_change": a1 / a0 - 1.0,
        "volume_change": None if v0 is None else v1 / v0 - 1.0,
    }


def self_intersections(vertices, faces, chunk=256):
    """Number of places where the surface passes through itself.

    Counts pairs of an edge and a triangle that share no vertex and where the
    edge crosses the inside of the triangle (Moller-Trumbore test). Two
    sheets of surface that cross along a curve give a count that grows with
    the length of that curve, so 0 means the surface does not cross itself
    and the size of a nonzero count is only a rough measure.
    """
    V = np.asarray(vertices, dtype=np.float64)
    F = np.asarray(faces, dtype=np.int64)
    E = unique_edges(F)
    A, B, C = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    e1, e2 = B - A, C - A
    f_lo, f_hi = np.minimum(np.minimum(A, B), C), np.maximum(np.maximum(A, B), C)
    eps = 1e-12
    count = 0
    for start in range(0, len(E), chunk):
        edges = E[start:start + chunk]
        P, Q = V[edges[:, 0]], V[edges[:, 1]]
        lo, hi = np.minimum(P, Q), np.maximum(P, Q)
        # Only pairs whose bounding boxes overlap and that share no vertex.
        overlap = np.all((lo[:, None] <= f_hi[None]) & (hi[:, None] >= f_lo[None]), axis=2)
        shares = (edges[:, None, :, None] == F[None, :, None, :]).any(axis=(2, 3))
        i, j = np.nonzero(overlap & ~shares)
        if i.size == 0:
            continue
        D = Q[i] - P[i]
        h = np.cross(D, e2[j])
        det = np.einsum("ij,ij->i", e1[j], h)
        ok = np.abs(det) > eps
        inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
        s = P[i] - A[j]
        u = inv * np.einsum("ij,ij->i", s, h)
        q = np.cross(s, e1[j])
        v = inv * np.einsum("ij,ij->i", D, q)
        t = inv * np.einsum("ij,ij->i", e2[j], q)
        hit = ok & (u > eps) & (v > eps) & (u + v < 1.0 - eps) & (t > eps) & (t < 1.0 - eps)
        count += int(np.count_nonzero(hit))
    return count
