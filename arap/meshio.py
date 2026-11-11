"""Mesh input/output through GeomProc.

GeomProc stores vertices as float32 in some code paths; everything returned
here is converted to float64 vertices and int64 faces so the numerical code
works in double precision.
"""

import os

import numpy as np

import geomproc

FIXED_COLOR = (0.85, 0.10, 0.10)   # red, as in the paper's figures
HANDLE_COLOR = (0.95, 0.80, 0.10)  # yellow
FREE_COLOR = (0.75, 0.75, 0.75)


def load_mesh(path, normalize=True, merge_tolerance=None):
    """Load a triangle mesh with GeomProc.

    Parameters
    ----------
    path : str
        OBJ, OFF or PLY file.
    normalize : bool
        Scale uniformly so the longest bounding-box side spans [-1, 1]
        (``geomproc.mesh.normalize``).
    merge_tolerance : float or None
        If given, merge vertices closer than this distance first
        (``geomproc.mesh.remove_duplicated_vertices``). Needed for meshes cut
        open along texture seams.

    Returns
    -------
    vertices : (n, 3) float64 array
    faces : (m, 3) int64 array
    """
    tm = geomproc.load(path)
    if merge_tolerance is not None:
        tm.remove_duplicated_vertices(merge_tolerance)
    if normalize:
        tm.normalize()
    return np.array(tm.vertex, dtype=np.float64), np.array(tm.face, dtype=np.int64)


def from_geomproc(tm):
    """Vertices and faces of a GeomProc mesh as float64 / int64 arrays."""
    return np.array(tm.vertex, dtype=np.float64), np.array(tm.face, dtype=np.int64)


def to_geomproc(vertices, faces, colors=None):
    """Build a GeomProc mesh from arrays."""
    tm = geomproc.mesh()
    tm.vertex = np.array(vertices, dtype=np.float64)
    tm.face = np.array(faces, dtype=np.int64)
    if colors is not None:
        tm.vcolor = np.array(colors, dtype=np.float64)
    return tm


def constraint_colors(n_vertices, fixed=(), handles=()):
    """Per-vertex colours: fixed vertices red, handle vertices yellow."""
    colors = np.tile(np.array(FREE_COLOR), (n_vertices, 1))
    colors[np.asarray(fixed, dtype=np.int64)] = FIXED_COLOR
    colors[np.asarray(handles, dtype=np.int64)] = HANDLE_COLOR
    return colors


def save_mesh(path, vertices, faces, colors=None):
    """Save a mesh with GeomProc, optionally with per-vertex colours."""
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    tm = to_geomproc(vertices, faces, colors)
    wo = geomproc.write_options()
    wo.write_vertex_colors = colors is not None
    tm.save(path, wo)
