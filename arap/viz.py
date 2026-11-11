"""Off-screen figures with matplotlib (no OpenGL needed)."""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402
import numpy as np  # noqa: E402

BASE_COLOR = np.array([0.62, 0.72, 0.86])
# Cyclic axis permutations that bring the chosen axis to the plotted z axis
# (cyclic, so orientation is kept).
_UP = {"z": [0, 1, 2], "y": [2, 0, 1], "x": [1, 2, 0]}
LIGHT = np.array([0.35, -0.55, 0.75])


def _face_colors(vertices, faces, vertex_colors=None):
    V = np.asarray(vertices)
    F = np.asarray(faces)
    n = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-300)
    light = LIGHT / np.linalg.norm(LIGHT)
    shade = 0.35 + 0.65 * np.abs(n @ light)
    if vertex_colors is None:
        base = np.tile(BASE_COLOR, (F.shape[0], 1))
    else:
        base = np.asarray(vertex_colors)[F].mean(axis=1)
    return np.clip(base * shade[:, None], 0.0, 1.0)


def render_panels(panels, path, view=(20, -60), title=None, panel_size=3.2,
                  vertical_axis="z"):
    """Render meshes side by side with a shared scale.

    ``panels`` is a list of dicts with keys ``vertices``, ``faces`` and
    optional ``title``, ``colors`` (per-vertex RGB) and ``view``. ``view`` is
    ``(elevation, azimuth)`` in degrees and ``vertical_axis`` the axis drawn
    upwards ("y" for most scanned models).
    """
    k = len(panels)
    fig = plt.figure(figsize=(panel_size * k, panel_size + (0.4 if title else 0.0)))
    order = _UP[vertical_axis]
    allv = np.vstack([np.asarray(p["vertices"])[:, order] for p in panels])
    center = 0.5 * (allv.max(axis=0) + allv.min(axis=0))
    half = 0.5 * np.max(allv.max(axis=0) - allv.min(axis=0))
    for idx, p in enumerate(panels):
        ax = fig.add_subplot(1, k, idx + 1, projection="3d")
        V = np.asarray(p["vertices"])[:, order]
        F = np.asarray(p["faces"])
        coll = Poly3DCollection(V[F], facecolors=_face_colors(V, F, p.get("colors")),
                                edgecolors="none", linewidths=0.0)
        ax.add_collection3d(coll)
        for setter, c in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), center):
            setter(c - half, c + half)
        ax.set_box_aspect((1, 1, 1))
        elev, azim = p.get("view", view)
        ax.view_init(elev=elev, azim=azim)
        ax.set_axis_off()
        if p.get("title"):
            ax.set_title(p["title"], fontsize=10)
    if title:
        fig.suptitle(title, fontsize=11)
    # Room at the top in inches: 0.3 for the panel titles, 0.4 more for the
    # overall title, so short figures do not overlap them.
    height = fig.get_size_inches()[1]
    fig.subplots_adjust(left=0, right=1, bottom=0, wspace=0,
                        top=1.0 - (0.3 + (0.4 if title else 0.0)) / height)
    _save(fig, path)


def plot_energy(curves, path, title="ARAP energy per iteration"):
    """Energy against iteration on a log scale; ``curves`` maps label -> list."""
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    for label, values in curves.items():
        values = np.asarray(values, dtype=float)
        ax.semilogy(np.arange(values.size), np.maximum(values, 1e-300), label=label)
    ax.set_xlabel("iteration")
    ax.set_ylabel("energy")
    ax.set_title(title, fontsize=10)
    ax.grid(True, which="both", alpha=0.3)
    if len(curves) > 1:
        ax.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, path)


def plot_scaling(n_vertices, series, path, ylabel="seconds", title=None, logy=True):
    """Line plot of several timing series against mesh size."""
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    for label, values in series.items():
        ax.plot(n_vertices, values, marker="o", label=label)
    ax.set_xlabel("number of vertices")
    ax.set_ylabel(ylabel)
    if logy:
        ax.set_yscale("log")
    if title:
        ax.set_title(title, fontsize=10)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, path)


def plot_lines(x, series, path, xlabel, ylabel, title=None, reference=None):
    """Line plot of ``series`` (label -> values) against ``x``.

    ``reference`` is an optional ``(label, x, y)`` drawn as a thin dashed grey
    line, for example the value a perfect method would reach.
    """
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    if reference is not None:
        label, rx, ry = reference
        ax.plot(rx, ry, color="0.55", linewidth=1.0, linestyle="--", label=label)
    for label, values in series.items():
        ax.plot(x, values, linewidth=2.0, marker="o", markersize=3.5, label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, path)


def _save(fig, path):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
