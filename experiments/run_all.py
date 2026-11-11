"""Run every experiment and write figures, meshes and metrics to ``output/``.

Usage (from the repository root)::

    python -m experiments.run_all                 # everything
    python -m experiments.run_all --only bar_bend scaling
    python -m experiments.run_all --update-docs   # also refresh docs/figures

Each experiment writes into ``output/<experiment>/``. All numbers end up in
``output/metrics.json`` and the tables in ``output/summary.md``.
"""

import argparse
import json
import os
import shutil
import time

import numpy as np

from arap import laplacian as lap
from arap import reference
from arap import selection as sel
from arap.meshio import constraint_colors, load_mesh, save_mesh
from arap.metrics import (area_volume_change, edge_length_error, per_vertex_stretch,
                          self_intersections)
from arap.solver import ARAPDeformer
from arap.viz import plot_energy, plot_lines, plot_scaling, render_panels

from . import scenarios as S

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAX_ITERATIONS = 1000
# Stopping rule for the large rotations: tighter and with a higher cap than
# the default, because near the point where the even twist stops being a
# minimum the energy is almost flat and the default rule stops early.
LARGE_ROTATION_RULE = dict(max_iterations=10000, energy_tol=1e-8, step_tol=1e-6)
# Stopping rule for the stability check: much tighter than the default.
NUDGE_RULE = dict(max_iterations=5000, energy_tol=1e-10, step_tol=1e-8)


# ---------------------------------------------------------------- helpers
def stripe_colors(vertices, bands=8):
    """Alternating colours by angle around the z axis, so twists are visible."""
    angle = np.arctan2(vertices[:, 1], vertices[:, 0])
    band = np.floor((angle + np.pi) / (2 * np.pi) * bands).astype(int) % 2
    light, dark = np.array([0.80, 0.84, 0.92]), np.array([0.25, 0.38, 0.62])
    return np.where(band[:, None] == 1, dark, light)


def stretch_colors(rest, deformed, faces, vmax=0.3):
    """White (no stretch) to red (``vmax`` relative edge change) per vertex."""
    s = np.clip(per_vertex_stretch(rest, deformed, faces) / vmax, 0.0, 1.0)[:, None]
    return (1.0 - s) * np.array([0.92, 0.92, 0.92]) + s * np.array([0.80, 0.10, 0.10])


def iterations_to_within(energies, fraction=0.01):
    """First iteration whose energy is at most ``(1 + fraction)`` times the final value."""
    E = np.asarray(energies)
    return int(np.argmax(E <= (1.0 + fraction) * E[-1]))


def optional_igl():
    """libigl if it is installed (it is only used as an outside reference)."""
    try:
        import igl
    except ImportError:
        return None
    return igl


def median_time(fn, repeats=5, warmup=True):
    """Median wall-clock time of ``fn()`` over ``repeats`` calls.

    One untimed call first lets BLAS threads and caches settle; a single
    sample on a shared machine can be off by two orders of magnitude.
    """
    if warmup:
        fn()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - start)
    return float(np.median(samples))


def side_rings(rest):
    """Vertex indices of each ring of side vertices of a bar along z, bottom to top.

    The two end rings are exactly level (also on ``scenarios.irregular_bar``),
    so their size gives the number of sides; every other side vertex goes to
    the ring whose height is nearest, which allows jittered rings.
    """
    r = np.hypot(rest[:, 0], rest[:, 1])
    side = np.flatnonzero(r > 0.5 * r.max())
    z = rest[side, 2]
    z0, z1 = z.min(), z.max()
    sides = np.count_nonzero(np.isclose(z, z0))
    rings = side.size // sides
    level = np.rint((z - z0) / (z1 - z0) * (rings - 1)).astype(int)
    return [side[level == k] for k in range(rings)]


def twist_angle(rest, deformed, min_coherence=0.5):
    """Twist of a bar along z in degrees: top ring angle minus bottom ring angle.

    Each ring of side vertices gets the mean angle by which its vertices
    turned about the ring's centroid; the ring angles are unwrapped from the
    bottom up, so a twist past 180 degrees is followed as long as
    neighbouring rings differ by less than 180 degrees. When a ring is
    pinched or folded onto itself, its vertices' turn angles no longer agree
    (the length of their mean unit vector drops below ``min_coherence``), the
    ring has no angle, and the twist is undefined: NaN is returned. NaN is
    also returned when two neighbouring rings differ by more than 90 degrees,
    since the direction of such a jump (and so the sign of the twist) would
    be decided by rounding. This is a threshold on how well the vertices
    agree, not a test for folded triangles.
    """
    ring_angles = []
    for ring in side_rings(rest):
        c = deformed[ring].mean(axis=0)
        turned = (np.arctan2(deformed[ring, 1] - c[1], deformed[ring, 0] - c[0])
                  - np.arctan2(rest[ring, 1], rest[ring, 0]))
        mean = np.mean(np.exp(1j * turned))
        if abs(mean) < min_coherence:
            return float("nan")
        ring_angles.append(np.angle(mean))
    unwrapped = np.unwrap(ring_angles)
    if np.any(np.abs(np.diff(unwrapped)) > 0.5 * np.pi):
        return float("nan")
    return float(np.degrees(unwrapped[-1] - unwrapped[0]))


def enclosed_area_2d(points, grid=120):
    """Area a closed 2D polygon winds around (nonzero winding rule), on a grid.

    Unlike the signed shoelace sum, the lobes of a polygon that crosses
    itself add up instead of cancelling.
    """
    lo, hi = points.min(axis=0), points.max(axis=0)
    step = (hi - lo) / grid
    if np.any(step <= 0.0):
        return 0.0
    centres = [lo[k] + step[k] * (np.arange(grid) + 0.5) for k in range(2)]
    X, Y = np.meshgrid(*centres)
    angle = np.arctan2(points[None, :, 1] - Y.ravel()[:, None],
                       points[None, :, 0] - X.ravel()[:, None])
    turn = np.diff(np.concatenate([angle, angle[:, :1]], axis=1), axis=1)
    winding = np.rint(np.sum((turn + np.pi) % (2.0 * np.pi) - np.pi, axis=1) / (2.0 * np.pi))
    return float(np.count_nonzero(winding) * step[0] * step[1])


def narrowest_ring(rest, deformed):
    """Smallest cross-section area along a bar along z, as a fraction of its rest area.

    Each ring of side vertices, in its rest order, is projected onto the xy
    plane (so this suits a bar that stays along z, as in a twist) and the
    area it winds around is measured with ``enclosed_area_2d``.
    """
    ratios = []
    for ring in side_rings(rest):
        order = ring[np.argsort(np.arctan2(rest[ring, 1], rest[ring, 0]))]
        ratios.append(enclosed_area_2d(deformed[order, :2]) / enclosed_area_2d(rest[order, :2]))
    return float(min(ratios))


def rotation_angle(rest_points, deformed_points):
    """Angle in degrees of the best-fit rotation taking one point set to the other."""
    A = rest_points - rest_points.mean(axis=0)
    B = deformed_points - deformed_points.mean(axis=0)
    U, _, Vt = np.linalg.svd(A.T @ B)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T
    return float(np.degrees(np.arccos(np.clip(0.5 * (np.trace(R) - 1.0), -1.0, 1.0))))


def stepped(deformer, targets_at, angles, **rule):
    """Solve for each handle angle in turn, starting from the previous result.

    This is the paper's interactive setting, where the previous frame is the
    initial guess. ``rule`` is passed to ``deform`` (default: at most
    ``MAX_ITERATIONS`` iterations). Yields ``(angle, result, iterations so far)``.
    """
    rule = dict(dict(max_iterations=MAX_ITERATIONS), **rule)
    current, total = "laplacian", 0
    for angle in angles:
        result = deformer.deform(targets_at(angle), init=current, **rule)
        current, total = result.vertices, total + result.iterations
        yield angle, result, total


def bar_surface_points(rest, faces, deformed, samples):
    """Deformed positions of points given as (angle, height) on the side of a bar along z.

    Each sample is located in a side triangle of the rest mesh, in (angle,
    height) coordinates, and carried to the deformed mesh with its barycentric
    coordinates. Different triangulations of the same bar can then be
    compared point by point.
    """
    angle = np.arctan2(rest[:, 1], rest[:, 0])
    radius = np.hypot(rest[:, 0], rest[:, 1])
    side = faces[np.all(radius[faces] > 0.5 * radius.max(), axis=1)]
    A, Z = angle[side], rest[side, 2]
    # Triangles across the seam at +-pi: lift their negative angles by 2 pi.
    seam = A.max(axis=1) - A.min(axis=1) > np.pi
    A[seam] = np.where(A[seam] < 0.0, A[seam] + 2.0 * np.pi, A[seam])
    a0, z0 = A[:, 0], Z[:, 0]
    e1 = np.stack([A[:, 1] - a0, Z[:, 1] - z0], axis=1)
    e2 = np.stack([A[:, 2] - a0, Z[:, 2] - z0], axis=1)
    det = e1[:, 0] * e2[:, 1] - e2[:, 0] * e1[:, 1]
    points = np.full((len(samples), 3), np.nan)
    for k, (sa, sz) in enumerate(samples):
        for a in (sa, sa + 2.0 * np.pi):
            da, dz = a - a0, sz - z0
            l1 = (da * e2[:, 1] - e2[:, 0] * dz) / det
            l2 = (e1[:, 0] * dz - da * e1[:, 1]) / det
            hit = np.flatnonzero((l1 >= -1e-9) & (l2 >= -1e-9) & (l1 + l2 <= 1.0 + 1e-9))
            if hit.size:
                i, b1, b2 = side[hit[0]], l1[hit[0]], l2[hit[0]]
                points[k] = (1.0 - b1 - b2) * deformed[i[0]] + b1 * deformed[i[1]] + b2 * deformed[i[2]]
                break
    return points


def summarize(result, rest, faces):
    E = np.asarray(result.energies)
    return {
        "energy_monotone": bool(np.all(np.diff(E) <= 1e-12 * abs(E[0]))),
        "iterations": result.iterations,
        "converged": result.converged,
        "iterations_to_1pct": iterations_to_within(result.energies),
        "energy_initial": result.energies[0],
        "energy_final": result.energies[-1],
        "edge_error_initial": edge_length_error(rest, result.initial_guess, faces),
        "edge_error_final": edge_length_error(rest, result.vertices, faces),
        "shape_change_initial": area_volume_change(rest, result.initial_guess, faces),
        "shape_change_final": area_volume_change(rest, result.vertices, faces),
        "timings": result.timings,
    }


def markdown_table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for row in rows:
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(lines)


def twist_cell(row, key="twist_achieved"):
    """A measured twist for the tables: "-" if not measured, "folded" if undefined."""
    if key not in row:
        return "-"
    x = row[key]
    return "folded" if x is None or np.isnan(x) else str(int(round(x)))


def pct(x):
    """A relative change as a signed percentage, or "-" when undefined."""
    return "-" if x is None else f"{100 * x:+.1f}%"


def fmt(x, digits=3):
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    return f"{x:.{digits}g}"


# ------------------------------------------------------------ experiments
def showcase(out, scenario_name):
    """Rest pose, Laplacian editing (initial guess) and converged ARAP."""
    sc = S.ALL[scenario_name]()
    V, F = sc.vertices, sc.faces
    c, t = sc.constraints
    d = ARAPDeformer(V, F, c)
    snaps = (0, 1, 2, 4, 10) if scenario_name == "bar_bend" else ()
    result = d.deform(t, max_iterations=MAX_ITERATIONS, snapshot_iterations=snaps + (100,))
    colors = constraint_colors(V.shape[0], sc.fixed, sc.handles)
    if scenario_name == "bar_twist":
        # Stripes make the twist visible; constrained vertices keep their colour.
        free = np.setdiff1d(np.arange(V.shape[0]), c)
        colors[free] = stripe_colors(V)[free]
    folder = os.path.join(out, scenario_name)
    render_panels(
        [dict(vertices=V, faces=F, colors=colors, title="rest pose"),
         dict(vertices=result.initial_guess, faces=F, colors=colors, title="Laplacian editing"),
         dict(vertices=result.vertices, faces=F, colors=colors,
              title=f"ARAP ({result.iterations} it.)")],
        os.path.join(folder, "comparison.png"), view=sc.view, vertical_axis=sc.up,
    )
    render_panels(
        [dict(vertices=result.initial_guess, faces=F, title="Laplacian editing",
              colors=stretch_colors(V, result.initial_guess, F)),
         dict(vertices=result.vertices, faces=F, title="ARAP",
              colors=stretch_colors(V, result.vertices, F))],
        os.path.join(folder, "stretch.png"), view=sc.view, vertical_axis=sc.up,
        title="edge-length change (white 0%, red 30% or more)",
    )
    if snaps:
        panels = [dict(vertices=V, faces=F, colors=colors, title="rest")]
        for k in snaps:
            label = "initial guess" if k == 0 else f"iteration {k}"
            panels.append(dict(vertices=result.snapshots[k], faces=F, colors=colors, title=label))
        panels.append(dict(vertices=result.vertices, faces=F, colors=colors,
                           title=f"converged ({result.iterations})"))
        render_panels(panels, os.path.join(folder, "iterations.png"), view=sc.view,
                      vertical_axis=sc.up, panel_size=2.4)
    plot_energy({scenario_name: result.energies}, os.path.join(folder, "energy.png"))
    save_mesh(os.path.join(folder, "rest.obj"), V, F, colors)
    save_mesh(os.path.join(folder, "laplacian.obj"), result.initial_guess, F, colors)
    save_mesh(os.path.join(folder, "arap.obj"), result.vertices, F, colors)
    metrics = summarize(result, V, F)
    metrics.update(vertices=int(V.shape[0]), faces=int(F.shape[0]), constrained=int(c.size),
                   setup=d.timings, description=sc.description)
    # How much the shape still changes after the first hundred iterations.
    if 100 in result.snapshots:
        metrics["moved_after_iteration_100"] = float(
            np.max(np.linalg.norm(result.vertices - result.snapshots[100], axis=1)) / d.diagonal)
    if scenario_name == "hinge":
        # The top block should end up as a rigid copy of itself.
        # Faces of the top block, without those inside the handle (which the
        # constraints move rigidly for both methods).
        top = np.flatnonzero(V[:, 2] > 0.45)
        keep = np.all(np.isin(F, top), axis=1) & ~np.all(np.isin(F, sc.handles), axis=1)
        metrics["top_block_edge_error"] = {
            "laplacian": edge_length_error(V, result.initial_guess, F[keep]),
            "arap": edge_length_error(V, result.vertices, F[keep]),
        }
    return metrics


def weights(out):
    """Cotangent against uniform edge weights (paper Section 3).

    The scanned meshes have obtuse triangles, so the cotangent rows also
    depend on the clamp; "cotangent, raw" uses the unclamped weights.
    """
    rows = {}
    variants = {"cotangent": ("cotangent", "zero"), "uniform": ("uniform", "zero"),
                "cotangent, raw": ("cotangent", "none")}
    for name in ("bunny_ears", "camel_drink"):
        sc = S.ALL[name]()
        V, F = sc.vertices, sc.faces
        c, t = sc.constraints
        panels, rows[name] = [], {}
        for label, (w, clamp_mode) in variants.items():
            r = ARAPDeformer(V, F, c, weights=w, clamp=clamp_mode).deform(
                t, max_iterations=MAX_ITERATIONS)
            rows[name][label] = summarize(r, V, F)
            if clamp_mode == "none":
                continue
            panels.append(dict(vertices=r.vertices, faces=F, title=f"{w} weights",
                               colors=stretch_colors(V, r.vertices, F)))
        render_panels(panels, os.path.join(out, "weights", f"{name}.png"), view=sc.view,
                      vertical_axis=sc.up, title="edge-length change (white 0%, red 30% or more)")
    return rows


def clamp(out):
    """Zero displacement: does the rest pose stay put for each weight treatment?"""
    igl = optional_igl()
    rows = {}
    for name, axis in (("bunny", "y"), ("camel", "y"), ("kid", "z")):
        V, F = load_mesh(os.path.join(S.MESH_DIR, f"{name}.obj"))
        c = sel.by_coordinate(V, axis, below=0.1)
        negative = float(np.mean(lap.cotangent_weights(V, F, clamp="none").data < 0.0))
        rows[name] = {}
        for mode in ("none", "zero", "geomproc"):
            d = ARAPDeformer(V, F, c, clamp=mode)
            r = d.deform(V[c], max_iterations=50, energy_tol=0.0, step_tol=0.0)
            rows[name][mode] = {
                "max_drift_over_diagonal": float(np.max(np.linalg.norm(r.vertices - V, axis=1))
                                                 / d.diagonal),
                "min_energy": float(min(r.energies)),
                "negative_weight_fraction": negative,
            }
            if mode == "none" and igl is not None:
                data = igl.ARAPData()
                data.energy = igl.ARAP_ENERGY_TYPE_SPOKES
                data.max_iter = 50
                igl.arap_precomputation(V, F, 3, c.astype(np.int32), data)
                U = igl.arap_solve(V[c], data, V.copy())
                rows[name][mode]["libigl_max_drift_over_diagonal"] = float(
                    np.max(np.linalg.norm(U - V, axis=1)) / d.diagonal)
            if name == "bunny" and mode == "none":
                render_panels(
                    [dict(vertices=V, faces=F, title="rest pose"),
                     dict(vertices=r.vertices, faces=F, title="raw weights, 50 it.",
                          colors=stretch_colors(V, r.vertices, F, vmax=0.1))],
                    os.path.join(out, "clamp", "bunny_raw_drift.png"), view=(10, 0),
                    vertical_axis="y",
                    title="no handle moved; red marks 10% edge change or more",
                )
    return rows


def initial_guess(out):
    """Laplacian (slides, Delta d = 0), thin plate (Delta^2 d = 0) and rest pose."""
    rows = {}
    for name in ("bar_bend", "hinge", "camel_drink"):
        sc = S.ALL[name]()
        V, F = sc.vertices, sc.faces
        c, t = sc.constraints
        d = ARAPDeformer(V, F, c)
        rows[name], curves, finals = {}, {}, {}
        for init in ("laplacian", "thin_plate", "rest"):
            start = time.perf_counter()
            r = d.deform(t, init=init, max_iterations=MAX_ITERATIONS)
            total = time.perf_counter() - start
            rows[name][init] = summarize(r, V, F)
            rows[name][init]["total_seconds"] = total
            curves[init] = r.energies
            finals[init] = r.vertices
        for init in ("thin_plate", "rest"):
            rows[name][init]["max_distance_to_laplacian_result"] = float(
                np.max(np.linalg.norm(finals[init] - finals["laplacian"], axis=1)) / d.diagonal)
        plot_energy(curves, os.path.join(out, "initial_guess", f"{name}_energy.png"),
                    title=f"{name}: energy for each initial guess")
    return rows


def scaling(out):
    """Time per iteration against mesh size, sparse factorised vs dense solves.

    Timings are medians of repeated runs; run on an otherwise idle machine.
    """
    sizes = [(12, 25), (20, 41), (32, 81), (48, 121), (64, 161), (96, 241)]
    dense_limit = 6000
    rows = []
    for sides, rings in sizes:
        sc = S.bar_bend(rings=rings, sides=sides)
        V, F = sc.vertices, sc.faces
        c, t = sc.constraints
        n = V.shape[0]
        # Three repeats of setup and of a 20-iteration run; keep the medians.
        runs = []
        for _ in range(3):
            d = ARAPDeformer(V, F, c)
            r = d.deform(t, max_iterations=20, energy_tol=0.0, step_tol=0.0)
            runs.append({
                "weights_seconds": d.timings["weights"],
                "factorization_seconds": d.timings["factorization"],
                "local_step_seconds": r.timings["local_steps"] / r.iterations,
                "global_step_seconds": r.timings["global_steps"] / r.iterations,
                "iteration_seconds": r.timings["per_iteration"],
            })
        row = {"vertices": int(n)}
        row.update({k: float(np.median([run[k] for run in runs])) for k in runs[0]})
        if n <= dense_limit:
            row["geomproc_laplacian_seconds"] = median_time(
                lambda: reference.geomproc_laplacian(V, F), repeats=3, warmup=False)
            L_gp = reference.geomproc_laplacian(V, F)
            R = r.rotations
            row["dense_global_step_seconds"] = median_time(
                lambda: reference.arap_global_step(L_gp, V, R, c, t))
        rows.append(row)
    n = [row["vertices"] for row in rows]
    nd = [row["vertices"] for row in rows if "dense_global_step_seconds" in row]
    plot_scaling(n, {
        "sparse: setup (weights + factorisation)": [r["weights_seconds"] + r["factorization_seconds"] for r in rows],
        "sparse: one iteration": [r["iteration_seconds"] for r in rows],
    }, os.path.join(out, "scaling", "sparse.png"), title="sparse solver")
    plot_scaling(nd, {
        "sparse: global step": [r["global_step_seconds"] for r in rows][: len(nd)],
        "dense (GeomProc idiom): global step": [r["dense_global_step_seconds"] for r in rows[: len(nd)]],
        "GeomProc geometric_laplacian build": [r["geomproc_laplacian_seconds"] for r in rows[: len(nd)]],
    }, os.path.join(out, "scaling", "dense_vs_sparse.png"), title="dense vs sparse")
    return rows


def geomproc_consistency(out):
    """The sparse solver against dense solves with row-scaled operators.

    GeomProc's ``geometric_laplacian`` is ``D^-1 W - I``; the area-weighted
    operator is ``M^-1 (W - D)`` with mixed areas. Each is paired with a
    right-hand side scaled the same way (consistent) and with the paper's
    unscaled one (inconsistent).
    """
    rows = {}
    igl = optional_igl()

    def dist(a, b):
        return float(np.max(np.linalg.norm(a - b, axis=1)) / diag)

    for name in ("bar_bend", "bunny_ears"):
        sc = S.ALL[name]()
        V, F = sc.vertices, sc.faces
        c, t = sc.constraints
        L_gp = reference.geomproc_laplacian(V, F)
        d = ARAPDeformer(V, F, c, clamp="geomproc")
        diag = d.diagonal
        guess = d.initial_guess(t)
        R = d.local_step(guess)
        dense_membrane = reference.membrane_deformation(L_gp, V, c, t)
        dense_step = reference.arap_global_step(L_gp, V, R, c, t)
        wrong_step = reference.row_replaced_solve(L_gp, -d.rhs(R), c, t)
        X, Rk = guess, R
        for _ in range(10):
            X = reference.arap_global_step(L_gp, V, Rk, c, t)
            Rk = d.local_step(X)
        sparse10 = d.deform(t, max_iterations=10, energy_tol=0.0, step_tol=0.0).vertices
        row = {
            "membrane": dist(dense_membrane, guess),
            "one_step_consistent_rhs": dist(dense_step, d.global_step(R, t)),
            "ten_iterations": dist(X, sparse10),
            "one_step_inconsistent_rhs": dist(wrong_step, d.global_step(R, t)),
        }
        # Area-weighted operator with the default (clamped) weights.
        dz = ARAPDeformer(V, F, c)
        Rz = dz.local_step(dz.initial_guess(t))
        L_area = -(1.0 / lap.mixed_areas(V, F))[:, None] * dz.L.toarray()
        expected = dz.global_step(Rz, t)
        row["area_weighted_consistent_rhs"] = dist(
            reference.arap_global_step(L_area, V, Rz, c, t), expected)
        row["area_weighted_inconsistent_rhs"] = dist(
            reference.row_replaced_solve(L_area, -dz.rhs(Rz), c, t), expected)
        if igl is not None:
            raw = ARAPDeformer(V, F, c, clamp="none")
            g = raw.initial_guess(t)
            ours = raw.deform(t, init=g, max_iterations=10, energy_tol=0.0, step_tol=0.0).vertices
            data = igl.ARAPData()
            data.energy = igl.ARAP_ENERGY_TYPE_SPOKES
            data.max_iter = 10
            igl.arap_precomputation(V, F, 3, c.astype(np.int32), data)
            theirs = igl.arap_solve(t, data, g.copy())
            row["libigl_ten_iterations_raw_weights"] = dist(ours, theirs)
        rows[name] = row
    return rows


def large_rotations(out):
    """Bends and twists far past the default scenarios, with three ways to start.

    ``laplacian``: one solve from the Laplacian-editing guess (the default).
    ``thin_plate``: one solve from the thin-plate guess.
    ``steps``: the handle moves 10 degrees at a time and each solve starts
    from the previous result, as in the paper's interactive use.
    Bends use a thinner bar (radius 0.05), since bending the default bar by
    270 degrees would compress its inner side by about half. Twists use the
    default (slightly jittered) twist bar, see ``scenarios.bar_twist``. Every
    solve uses ``LARGE_ROTATION_RULE``.
    """
    step = 10
    rule = LARGE_ROTATION_RULE
    rows = {"bend": {}, "twist": {}}
    for kind, angles in (("bend", (90, 180, 270)), ("twist", (90, 180, 270, 360))):
        if kind == "bend":
            sc = S.bar_bend(rings=61, sides=20, radius=0.05)
            handle_targets = lambda a: S.bend_targets(sc.vertices, sc.handles, a)
        else:
            sc = S.bar_twist()
            handle_targets = lambda a: sel.rigid_targets(
                sc.vertices[sc.handles], sel.rotation("z", a), pivot=np.zeros(3))
        V, F = sc.vertices, sc.faces
        targets_at = lambda a: sel.combine(V, sc.fixed, sc.handles, handle_targets(a))[1]
        c, _ = sc.constraints
        d = ARAPDeformer(V, F, c)
        colors = constraint_colors(V.shape[0], sc.fixed, sc.handles)
        if kind == "twist":
            free = np.setdiff1d(np.arange(V.shape[0]), c)
            colors[free] = stripe_colors(V)[free]
        # One stepped run up to the largest angle passes through the smaller ones.
        steps, trace, previous = {}, [], 0
        for a, r, total in stepped(d, targets_at, range(step, max(angles) + 1, step), **rule):
            if kind == "twist":
                trace.append((a, twist_angle(V, r.vertices), narrowest_ring(V, r.vertices),
                              self_intersections(r.vertices, F), r.energies[-1],
                              total - previous, r.converged))
            previous = total
            if a in angles:
                steps[a] = (r, total)
        for angle in angles:
            t = targets_at(angle)
            results = {init: (d.deform(t, init=init, **rule), None)
                       for init in ("laplacian", "thin_plate")}
            results["steps"] = steps[angle]
            row = {}
            for how, (r, total) in results.items():
                m = summarize(r, V, F)
                if total is not None:
                    m["iterations"] = total
                m["self_intersections"] = self_intersections(r.vertices, F)
                if kind == "twist":
                    m["twist_achieved"] = twist_angle(V, r.vertices)
                    m["narrowest_ring"] = narrowest_ring(V, r.vertices)
                row[how] = m
            rows[kind][str(angle)] = row
            if (kind, angle) in (("bend", 270), ("twist", 180), ("twist", 270)):
                panels = [dict(vertices=results["laplacian"][0].initial_guess, faces=F,
                               colors=colors, title="Laplacian editing")]
                for how, label in (("laplacian", "ARAP, Laplacian start"),
                                   ("thin_plate", "ARAP, thin-plate start"),
                                   ("steps", f"ARAP, {step}-degree steps")):
                    panels.append(dict(vertices=results[how][0].vertices, faces=F,
                                       colors=colors, title=label))
                render_panels(panels, os.path.join(out, "large_rotations", f"{kind}_{angle}.png"),
                              view=sc.view, vertical_axis=sc.up,
                              title=f"{kind} by {angle} degrees")
        if trace:
            a, reached, narrowest, crossings, energy, iterations, converged = map(
                np.array, zip(*trace))
            # Gaps (NaN) where a ring has folded over itself and has no twist angle.
            plot_lines(a, {"twist of the result": reached}, os.path.join(
                out, "large_rotations", "twist_steps.png"),
                xlabel="handle rotation (degrees)", ylabel="twist reached (degrees)",
                title=f"twist in {step}-degree steps (gap: a ring folded, no twist angle)",
                reference=("handle rotation", a, a))
            plot_lines(a, {"narrowest cross-section": narrowest}, os.path.join(
                out, "large_rotations", "twist_narrowest.png"),
                xlabel="handle rotation (degrees)", ylabel="fraction of rest area",
                title=f"narrowest cross-section, {step}-degree steps")
            rows["twist_steps"] = {"handle": a.tolist(), "reached": reached.tolist(),
                                   "narrowest_ring": narrowest.tolist(),
                                   "self_intersections": crossings.tolist(),
                                   "energy": energy.tolist(), "iterations": iterations.tolist(),
                                   "converged": converged.tolist()}
    return rows


def centre_line_kink(rest, deformed):
    """Sharpest turn of a bar's centre line divided by its mean turn.

    The centre line joins the centroids of the rings of side vertices; the
    turn at a ring is the angle between the segments before and after it.
    An even arc gives 1; a bend concentrated in a few rings gives much more.
    Returns None when the rings are not level (an irregular mesh), since
    the uneven spacing of their centroids would then dominate.
    """
    rings = side_rings(rest)
    if any(np.ptp(rest[ring, 2]) > 1e-9 for ring in rings):
        return None
    c = np.array([deformed[ring].mean(axis=0) for ring in rings])
    d = np.diff(c, axis=0)
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    turn = np.arccos(np.clip(np.einsum("ij,ij->i", d[:-1], d[1:]), -1.0, 1.0))
    if turn.mean() < 1e-12:
        return 1.0  # a straight line: evenly bent by nothing
    return float(turn.max() / turn.mean())


def triangulation(out):
    """The 90-degree bar bend on different meshes of the same bar.

    Only the end caps are constrained, so the constraints mean the same on
    every mesh. Each result is compared at 800 fixed points of the bar's side
    (``bar_surface_points``) with the 20 x 41 result, and with its own mirror
    image: the bend is symmetric about the plane x = 0, and so is a regular
    bar as a set of points, but its triangles are not, because every quad
    has its diagonal the same way round.
    """
    regular = S.bar(rings=41, sides=20)
    meshes = {
        "12 x 25": S.bar(rings=25, sides=12),
        "20 x 41": regular,
        "20 x 41, diagonals flipped": (regular[0], S.flip_diagonals(*regular)),
        "20 x 41 irregular": S.irregular_bar(rings=41, sides=20),
        "16 x 81 (long, thin triangles)": S.bar(rings=81, sides=16),
        "12 x 81": S.bar(rings=81, sides=12),
        "20 x 81": S.bar(rings=81, sides=20),
        "12 x 161": S.bar(rings=161, sides=12),
        "32 x 41": S.bar(rings=41, sides=32),
        "32 x 81": S.bar(rings=81, sides=32),
        "48 x 77": S.bar(rings=77, sides=48),
    }
    rng = np.random.default_rng(1)
    samples = np.column_stack([rng.uniform(-np.pi, np.pi, 800), rng.uniform(-0.45, 0.45, 800)])
    # The same points mirrored in x = 0: angle t becomes pi - t.
    mirrored = samples.copy()
    mirrored[:, 0] = (np.pi - samples[:, 0] + np.pi) % (2.0 * np.pi) - np.pi
    flip_x = np.array([-1.0, 1.0, 1.0])
    reference = ("cotangent", "20 x 41")
    rows, points, panels, rest_points, results = {}, {}, {}, {}, {}
    for w in ("cotangent", "uniform"):
        rows[w], panels[w] = {}, []
        for name, (V, F) in meshes.items():
            z0, z1 = V[:, 2].min(), V[:, 2].max()
            fixed = np.flatnonzero(V[:, 2] <= z0 + 1e-9)
            handles = np.flatnonzero(V[:, 2] >= z1 - 1e-9)
            c, t = sel.combine(V, fixed, handles, S.bend_targets(V, handles, 90.0))
            d = ARAPDeformer(V, F, c, weights=w)
            r = d.deform(t, max_iterations=MAX_ITERATIONS)
            results[w, name] = r.vertices
            points[w, name] = bar_surface_points(V, F, r.vertices, samples)
            rest_points[name] = bar_surface_points(V, F, V, samples)
            mirror = bar_surface_points(V, F, r.vertices, mirrored) * flip_x
            asym = np.linalg.norm(points[w, name] - mirror, axis=1) / d.diagonal
            m = summarize(r, V, F)
            m.update(vertices=int(V.shape[0]), diagonal=d.diagonal,
                     mirror_asymmetry_rms=float(np.sqrt(np.mean(asym ** 2))),
                     mirror_asymmetry_max=float(np.max(asym)),
                     kink=centre_line_kink(V, r.vertices))
            rows[w][name] = m
            if name in ("12 x 25", "20 x 41", "16 x 81 (long, thin triangles)", "20 x 81",
                        "12 x 161", "32 x 41", "48 x 77"):
                panels[w].append(dict(vertices=r.vertices, faces=F, title=name.split(" (")[0],
                                      colors=stretch_colors(V, r.vertices, F)))
    for (w, name), p in points.items():
        diagonal = rows[w][name].pop("diagonal")
        gap = np.linalg.norm(p - points[reference], axis=1) / diagonal
        # Triangulation dependence alone: against the same weighting's 20 x 41 result.
        same = np.linalg.norm(p - points[w, reference[1]], axis=1) / diagonal
        # How far apart the meshes' sample points are before any deformation.
        floor = np.linalg.norm(rest_points[name] - rest_points[reference[1]], axis=1) / diagonal
        rows[w][name].update(distance_rms=float(np.sqrt(np.mean(gap ** 2))),
                             distance_max=float(np.max(gap)),
                             distance_same_weights_rms=float(np.sqrt(np.mean(same ** 2))),
                             distance_at_rest_rms=float(np.sqrt(np.mean(floor ** 2))))
    # Vertex by vertex on the regular 20 x 41 mesh, free of any interpolation:
    # the vertex at (-x, y, z) of each vertex, and the effect of the flip.
    V = regular[0]
    mirror = np.array([np.argmin(np.linalg.norm(V - v * flip_x, axis=1)) for v in V])
    assert np.allclose(V[mirror], V * flip_x)
    diagonal = float(np.linalg.norm(np.ptp(V, axis=0)))
    vertices = {}
    for w in rows:
        X = results[w, "20 x 41"]
        asym = np.linalg.norm(X - X[mirror] * flip_x, axis=1) / diagonal
        flip = np.linalg.norm(X - results[w, "20 x 41, diagonals flipped"], axis=1) / diagonal
        vertices[w] = {"mirror_asymmetry_max": float(asym.max()),
                       "flip_moves_rms": float(np.sqrt(np.mean(flip ** 2))),
                       "flip_moves_max": float(flip.max())}
    for w, row in panels.items():
        render_panels(row, os.path.join(out, "triangulation", f"{w}.png"), view=(10, 0),
                      panel_size=2.4, title=f"{w} weights (sides x rings)")
    return {"meshes": rows, "regular_20x41_vertices": vertices}


def point_handle(out):
    """A single handle vertex against a handle patch, for the 90-degree bar bend.

    With one vertex the handle has no orientation, so ARAP chooses how the
    end turns; a translated patch keeps its orientation, so the bar has to
    bend back to meet it.
    """
    sc = S.bar_bend()
    V, F = sc.vertices, sc.faces
    z1 = V[:, 2].max()
    top = np.array([0.0, 0.0, z1])
    pole = int(np.argmin(np.linalg.norm(V - top, axis=1)))
    turned = S.bend_targets(V, sc.handles, 90.0)
    shift = turned[sc.handles == pole][0] - V[pole]
    cases = {
        "one vertex": (np.array([pole]), V[[pole]] + shift),
        "end patch moved": (sc.handles, V[sc.handles] + shift),
        "end patch moved and turned": (sc.handles, turned),
    }
    rows, panels = {}, []
    for name, (handles, targets) in cases.items():
        c, t = sel.combine(V, sc.fixed, handles, targets)
        d = ARAPDeformer(V, F, c)
        r = d.deform(t, max_iterations=MAX_ITERATIONS)
        m = summarize(r, V, F)
        # The top end is the same patch in every case, also when only its
        # centre vertex is constrained.
        m["end_rotation_degrees"] = rotation_angle(V[sc.handles], r.vertices[sc.handles])
        m["handle_vertices"] = int(handles.size)
        rows[name] = m
        panels.append(dict(vertices=r.vertices, faces=F, title=name,
                           colors=constraint_colors(V.shape[0], sc.fixed, handles)))
    render_panels(panels, os.path.join(out, "point_handle", "bar.png"), view=sc.view,
                  vertical_axis=sc.up, title="the same end position, three kinds of handle")
    return rows


def stability(out):
    """Nudge a result and solve again: is it a minimum or a balanced unstable state?

    The stopping rule cannot tell a minimum from a balanced state that is not
    one (a saddle), and a perfectly symmetric start can stay on such a state.
    Each free vertex is moved at random by up to 0.1% of the diagonal and the
    solve is repeated with a much tighter rule (``NUDGE_RULE``). From a
    minimum the energy returns to the same value; from a saddle it ends lower.
    """
    rng = np.random.default_rng(3)
    rule = LARGE_ROTATION_RULE

    def twist(angle, jitter=0.02):
        sc = S.bar_twist(angle=angle, jitter=jitter)
        d = ARAPDeformer(sc.vertices, sc.faces, sc.constraints[0])
        return d, sc.constraints[1], d.deform(sc.constraints[1], max_iterations=MAX_ITERATIONS)

    def thin_bend(how):
        sc = S.bar_bend(rings=61, sides=20, radius=0.05)
        V = sc.vertices
        d = ARAPDeformer(V, sc.faces, sc.constraints[0])
        targets_at = lambda a: sel.combine(V, sc.fixed, sc.handles,
                                           S.bend_targets(V, sc.handles, a))[1]
        if how == "laplacian":
            return d, targets_at(270), d.deform(targets_at(270), **rule)
        *_, (_, r, _) = stepped(d, targets_at, range(10, 271, 10), **rule)
        return d, targets_at(270), r

    def twist_steps(angle):
        sc = S.bar_twist()
        V = sc.vertices
        d = ARAPDeformer(V, sc.faces, sc.constraints[0])
        targets_at = lambda a: sel.combine(V, sc.fixed, sc.handles, sel.rigid_targets(
            V[sc.handles], sel.rotation("z", a), pivot=np.zeros(3)))[1]
        *_, (_, r, _) = stepped(d, targets_at, range(10, angle + 1, 10), **rule)
        return d, targets_at(angle), r

    def default(name):
        sc = S.ALL[name]()
        d = ARAPDeformer(sc.vertices, sc.faces, sc.constraints[0])
        return d, sc.constraints[1], d.deform(sc.constraints[1], max_iterations=MAX_ITERATIONS)

    def end_caps(sides, rings):
        V, F = S.bar(rings=rings, sides=sides)
        fixed = np.flatnonzero(V[:, 2] <= V[:, 2].min() + 1e-9)
        handles = np.flatnonzero(V[:, 2] >= V[:, 2].max() - 1e-9)
        c, t = sel.combine(V, fixed, handles, S.bend_targets(V, handles, 90.0))
        d = ARAPDeformer(V, F, c)
        return d, t, d.deform(t, max_iterations=MAX_ITERATIONS)

    def one_vertex():
        sc = S.bar_bend()
        V = sc.vertices
        pole = int(np.argmin(np.linalg.norm(V - [0.0, 0.0, V[:, 2].max()], axis=1)))
        shift = S.bend_targets(V, sc.handles, 90.0)[sc.handles == pole][0] - V[pole]
        c, t = sel.combine(V, sc.fixed, np.array([pole]), V[[pole]] + shift)
        d = ARAPDeformer(V, sc.faces, c)
        return d, t, d.deform(t, max_iterations=MAX_ITERATIONS)

    cases = {
        "bar twist 100, regular bar": lambda: twist(100, jitter=0.0),
        "bar twist 105, regular bar": lambda: twist(105, jitter=0.0),
        "bar twist 110, regular bar": lambda: twist(110, jitter=0.0),
        "bar twist 135, regular bar": lambda: twist(135, jitter=0.0),
        "bar twist 135 (default, jittered bar)": lambda: twist(135),
        "bar bend 90 (default)": lambda: default("bar_bend"),
        "bend 270, thin bar, Laplacian start": lambda: thin_bend("laplacian"),
        "bend 270, thin bar, 10-degree steps": lambda: thin_bend("steps"),
        "twist 180, 10-degree steps": lambda: twist_steps(180),
        "bend 90, one handle vertex": one_vertex,
        "bend 90, 32 x 81, end caps only": lambda: end_caps(32, 81),
    }
    rows = {}
    for name, build in cases.items():
        d, t, r = build()
        X = r.vertices
        noise = rng.uniform(-1.0, 1.0, X.shape)
        noise *= 1e-3 * d.diagonal / np.linalg.norm(noise, axis=1).max()
        noise[d.constrained] = 0.0
        again = d.deform(t, init=X + noise, **NUDGE_RULE)
        rows[name] = {
            "energy": r.energies[-1],
            "energy_after_nudge": again.energies[-1],
            "energy_change": again.energies[-1] / r.energies[-1] - 1.0,
            "moved_over_diagonal": float(np.max(np.linalg.norm(again.vertices - X, axis=1))
                                         / d.diagonal),
            "iterations_after_nudge": again.iterations,
            "converged_after_nudge": again.converged,
        }
    return rows


# ----------------------------------------------------------------- report
def write_summary(metrics, path):
    parts = ["# Experiment summary", "",
             "Produced by `python -m experiments.run_all`. Edge error is the RMS of the "
             "relative edge-length change; distances are divided by the bounding-box diagonal. "
             "\"Iterations to within 1%\" counts the iterations until the energy is within 1% "
             "of its final value.", ""]
    shows = [k for k in S.ALL if k in metrics]
    if shows:
        rows = []
        for k in shows:
            m = metrics[k]
            rows.append([k, m["vertices"], m["constrained"], m["iterations"], fmt(m["converged"]),
                         fmt(m["energy_monotone"]),
                         m["iterations_to_1pct"],
                         fmt(m["edge_error_initial"]["rms_relative"]),
                         fmt(m["edge_error_final"]["rms_relative"]),
                         fmt(m["moved_after_iteration_100"], 2) if "moved_after_iteration_100" in m else "-",
                         fmt(m["setup"]["weights"] + m["setup"]["factorization"]),
                         fmt(1000 * m["timings"]["per_iteration"])])
        parts += ["## Deformations", "", markdown_table(
            ["scenario", "vertices", "constrained", "iterations", "converged",
             "energy never rose", "iterations to within 1%", "edge error Laplacian", "edge error ARAP",
             "largest move after iteration 100", "setup (s)", "ms / iteration"], rows), ""]
        rows = []
        for k in shows:
            m = metrics[k]
            if "shape_change_final" not in m:
                continue
            a, b = m["shape_change_initial"], m["shape_change_final"]
            rows.append([k, pct(a["area_change"]), pct(b["area_change"]),
                         pct(a["volume_change"]), pct(b["volume_change"])])
        if rows:
            parts += ["Change of surface area and enclosed volume (\"-\": the mesh is open, "
                      "so it encloses no volume).", "", markdown_table(
                ["scenario", "area Laplacian", "area ARAP", "volume Laplacian", "volume ARAP"],
                rows), ""]
        if "hinge" in metrics:
            h = metrics["hinge"]["top_block_edge_error"]
            parts += [f"Hinge, edge error inside the top block: Laplacian "
                      f"{fmt(h['laplacian']['rms_relative'])}, ARAP {fmt(h['arap']['rms_relative'])}.", ""]
    if "weights" in metrics:
        rows = [[name, w, fmt(m["edge_error_final"]["rms_relative"]),
                 fmt(m["edge_error_final"]["max_abs_relative"]), m["iterations"],
                 fmt(m["energy_final"])]
                for name, by_w in metrics["weights"].items() for w, m in by_w.items()]
        parts += ["## Edge weights", "", "\"cotangent, raw\": negative cotangent weights "
                  "are kept (`clamp=\"none\"`); the other rows clamp them to zero.", "",
                  markdown_table(["scenario", "weights", "edge error (RMS)", "edge error (max)",
                                  "iterations", "energy"], rows), ""]
    if "clamp" in metrics:
        rows = [[name, mode, fmt(m["negative_weight_fraction"]), fmt(m["max_drift_over_diagonal"]),
                 fmt(m["min_energy"]),
                 fmt(m["libigl_max_drift_over_diagonal"]) if "libigl_max_drift_over_diagonal" in m else "-"]
                for name, by_mode in metrics["clamp"].items() for mode, m in by_mode.items()]
        parts += ["## Rest pose with no handle moved (50 iterations)", "", markdown_table(
            ["mesh", "clamp", "negative raw weights", "max drift", "lowest energy",
             "libigl max drift"], rows), ""]
    if "initial_guess" in metrics:
        rows = []
        for name, by_init in metrics["initial_guess"].items():
            for init, m in by_init.items():
                rows.append([name, init, m["iterations"], m["iterations_to_1pct"],
                             fmt(m["energy_initial"]), fmt(m["energy_final"]),
                             fmt(m.get("max_distance_to_laplacian_result", 0.0)),
                             fmt(m["total_seconds"])])
        parts += ["## Initial guess", "", markdown_table(
            ["scenario", "initial guess", "iterations", "iterations to within 1%", "energy start",
             "energy end", "distance to Laplacian-start result", "seconds"], rows), ""]
    if "scaling" in metrics:
        rows = [[r["vertices"], fmt(r["weights_seconds"] + r["factorization_seconds"]),
                 fmt(1000 * r["local_step_seconds"]), fmt(1000 * r["global_step_seconds"]),
                 fmt(1000 * r["dense_global_step_seconds"]) if "dense_global_step_seconds" in r else "-",
                 fmt(r["geomproc_laplacian_seconds"]) if "geomproc_laplacian_seconds" in r else "-"]
                for r in metrics["scaling"]]
        parts += ["## Timing (bar bend, 20 iterations)", "",
                  "Medians of repeated runs. The dense global step is one row-replaced "
                  "`numpy.linalg.solve` with GeomProc's matrix, as in the class demos.", "",
                  markdown_table(
            ["vertices", "setup (s)", "local step (ms)", "global step (ms)",
             "dense global step (ms)", "GeomProc Laplacian build (s)"], rows), ""]
    if "geomproc_consistency" in metrics:
        keys = ("membrane", "one_step_consistent_rhs", "ten_iterations",
                "one_step_inconsistent_rhs", "area_weighted_consistent_rhs",
                "area_weighted_inconsistent_rhs", "libigl_ten_iterations_raw_weights")
        rows = [[name] + [fmt(m[k], 2) if k in m else "-" for k in keys]
                for name, m in metrics["geomproc_consistency"].items()]
        parts += ["## Agreement with GeomProc's operator and with libigl", "",
                  "Largest distance between corresponding vertices, divided by the "
                  "bounding-box diagonal. The libigl column is empty when libigl is not "
                  "installed.", "",
                  markdown_table(["scenario", "membrane, GeomProc operator",
                                  "one ARAP step, GeomProc operator", "ten ARAP iterations, GeomProc operator",
                                  "one step, GeomProc operator, inconsistent b",
                                  "one step, area-weighted, consistent b",
                                  "one step, area-weighted, inconsistent b",
                                  "libigl, ten iterations"], rows), ""]
    if "large_rotations" in metrics:
        rows = []
        for kind in ("bend", "twist"):
            by_angle = metrics["large_rotations"][kind]
            for angle, by_how in by_angle.items():
                for how, m in by_how.items():
                    rows.append([kind, angle, how, m["iterations"], fmt(m["converged"]),
                                 fmt(m["energy_final"]),
                                 fmt(m["edge_error_final"]["rms_relative"]),
                                 pct(m["shape_change_final"]["volume_change"]),
                                 m["self_intersections"],
                                 twist_cell(m),
                                 fmt(m["narrowest_ring"], 2) if "narrowest_ring" in m else "-"])
        parts += ["## Large rotations", "",
                  f"Bends use a bar of radius 0.05, twists the default (slightly jittered) bar. "
                  f"Every solve stops when an iteration lowers the energy by less than "
                  f"{LARGE_ROTATION_RULE['energy_tol']:g} of the starting energy and moves no vertex "
                  f"more than {LARGE_ROTATION_RULE['step_tol']:g} of the diagonal, or after "
                  f"{LARGE_ROTATION_RULE['max_iterations']} iterations. \"steps\" moves the "
                  "handle 10 degrees at a time and starts each solve from the previous result; "
                  "its iteration count is the total over all steps and \"converged\" refers to "
                  "the last step. \"crossings\" counts the pairs of an edge and a triangle that "
                  "pass through each other (0: the surface does not cross itself). The twist column "
                  "is the twist of the result, measured ring by ring along the bar (\"folded\": "
                  "the vertices of some ring turned by angles so different that the length of "
                  "their mean direction is below 0.5, so that ring has no angle and the twist is "
                  "undefined). \"narrowest\" is the smallest area that a ring of side vertices "
                  "winds around when it is projected along the bar's axis, as a fraction of its "
                  "rest area; once rings tilt this projected area is smaller than a cut "
                  "across the bar. Volume changes are of the signed volume: where the surface "
                  "crosses itself, parts that are turned inside out count negatively, and below "
                  "-100 % the signed volume is negative.", "",
                  markdown_table(["deformation", "degrees", "start", "iterations", "converged",
                                  "energy", "edge error", "volume change", "crossings",
                                  "twist reached", "narrowest"], rows), ""]
        steps = metrics["large_rotations"].get("twist_steps")
        if steps:
            rows = [[fmt(int(a)), twist_cell({"t": t}, "t"), fmt(n, 2), k, fmt(e), i,
                     fmt(ok)]
                    for a, t, n, k, e, i, ok in zip(
                        steps["handle"], steps["reached"], steps["narrowest_ring"],
                        steps["self_intersections"], steps["energy"], steps["iterations"],
                        steps["converged"])]
            parts += ["Twist in 10-degree steps, after each step:", "",
                      markdown_table(["handle rotation", "twist reached", "narrowest",
                                      "crossings", "energy", "iterations", "converged"], rows), ""]
    if "triangulation" in metrics:
        tri = metrics["triangulation"]["meshes"]
        share = lambda x: f"{100 * x:.2f}%"
        rows = [[w, name, m["vertices"], m["iterations"], fmt(m["converged"]),
                 fmt(m["energy_final"]), fmt(m["edge_error_final"]["rms_relative"]),
                 share(m["distance_rms"]), share(m["distance_same_weights_rms"]),
                 share(m["mirror_asymmetry_rms"]), share(m["mirror_asymmetry_max"]),
                 "-" if m["kink"] is None else fmt(m["kink"], 2),
                 pct(m["shape_change_final"]["volume_change"])]
                for w, by_mesh in tri.items() for name, m in by_mesh.items()]
        floor = max(m["distance_at_rest_rms"] for m in tri["cotangent"].values())
        vert = metrics["triangulation"]["regular_20x41_vertices"]
        parts += ["## Triangulation and resolution (bar bent by 90 degrees)", "",
                  "Meshes are named sides x rings. Only the end caps are constrained. "
                  "Distances are measured at 800 fixed points of the bar's side, as a share of the "
                  "bounding-box diagonal, to the 20 x 41 cotangent result and to the 20 x 41 "
                  "result with the same weights. Before any deformation these points already "
                  f"differ by up to {share(floor)} (RMS) between meshes, since each side is a "
                  "polygon with its own number of sides. The bend is symmetric about the plane "
                  "x = 0; \"mirror asymmetry\" is the distance between the result and its mirror "
                  "image at the same points. On a regular mesh the points and their mirror images "
                  "lie in triangles whose diagonals run the other way, so an exactly symmetric "
                  "result still shows a few hundredths of a percent there. \"kink\" is the "
                  "sharpest turn of the centre line from one ring to the next divided by its mean "
                  "turn (1 for an even arc; \"-\": the rings of the irregular mesh are not level). "
                  "Cotangent energies of different meshes are comparable on the regular meshes "
                  "only: the irregular mesh has negative cotangent weights, which are clamped "
                  "to zero.", "",
                  markdown_table(["weights", "mesh", "vertices", "iterations", "converged",
                                  "energy", "edge error", "distance to cotangent 20 x 41 (RMS)",
                                  "distance to same weights' 20 x 41 (RMS)",
                                  "mirror asymmetry (RMS)", "mirror asymmetry (max)", "kink",
                                  "volume change"], rows), "",
                  "Vertex by vertex on the regular 20 x 41 mesh, with no interpolation:", "",
                  markdown_table(["weights", "mirror asymmetry (max)",
                                  "flipping the diagonals moves the result (RMS)",
                                  "flipping the diagonals moves the result (max)"],
                                 [[w, share(m["mirror_asymmetry_max"]), share(m["flip_moves_rms"]),
                                   share(m["flip_moves_max"])] for w, m in vert.items()]), ""]
    if "point_handle" in metrics:
        rows = [[name, m["handle_vertices"], m["iterations"], fmt(m["energy_final"]),
                 fmt(m["edge_error_final"]["rms_relative"]),
                 fmt(m["end_rotation_degrees"], 3), pct(m["shape_change_final"]["volume_change"])]
                for name, m in metrics["point_handle"].items()]
        parts += ["## One handle vertex against a handle patch (bar, 90-degree bend)", "",
                  "The end is moved to the same place in every case. \"end rotation\" is the "
                  "angle by which the top end of the bar turned.", "",
                  markdown_table(["handle", "handle vertices", "iterations", "energy",
                                  "edge error", "end rotation (degrees)", "volume change"],
                                 rows), ""]
    if "stability" in metrics:
        rows = [[name, fmt(m["energy"], 5), fmt(m["energy_after_nudge"], 5),
                 f"{100 * m['energy_change']:+.3f}%", fmt(m["moved_over_diagonal"], 2),
                 m["iterations_after_nudge"], fmt(m["converged_after_nudge"])]
                for name, m in metrics["stability"].items()]
        parts += ["## Is the result a minimum? (nudge and solve again)", "",
                  "Every free vertex of the result is moved at random by up to 0.1% of the "
                  "diagonal, then the solve is repeated with a tight stopping rule "
                  f"(energy decrease below {NUDGE_RULE['energy_tol']:g} of the start, step "
                  f"below {NUDGE_RULE['step_tol']:g} of the diagonal, at most "
                  f"{NUDGE_RULE['max_iterations']} iterations). From a minimum the energy comes "
                  "back to the same value; a clearly lower energy means the result was a "
                  "balanced state that is not a minimum.", "",
                  markdown_table(["result", "energy", "energy after nudge", "change",
                                  "largest move / diagonal", "iterations", "converged"], rows), ""]
    with open(path, "w") as f:
        f.write("\n".join(parts))


EXPERIMENTS = {name: (lambda out, n=name: showcase(out, n)) for name in S.ALL}
EXPERIMENTS.update({
    "weights": weights,
    "clamp": clamp,
    "initial_guess": initial_guess,
    "scaling": scaling,
    "geomproc_consistency": geomproc_consistency,
    "large_rotations": large_rotations,
    "point_handle": point_handle,
    "triangulation": triangulation,
    "stability": stability,
})

DOC_FIGURES = [
    "bar_bend/iterations.png",
    "bar_bend/stretch.png",
    "bar_twist/comparison.png",
    "hinge/comparison.png",
    "kid_arms/comparison.png",
    "camel_drink/comparison.png",
    "bunny_ears/comparison.png",
    "clamp/bunny_raw_drift.png",
    "weights/camel_drink.png",
    "initial_guess/hinge_energy.png",
    "scaling/dense_vs_sparse.png",
    "large_rotations/bend_270.png",
    "large_rotations/twist_180.png",
    "large_rotations/twist_270.png",
    "large_rotations/twist_steps.png",
    "triangulation/cotangent.png",
    "triangulation/uniform.png",
    "point_handle/bar.png",
]


def to_builtin(x):
    """Plain Python types for JSON; NaN (an undefined measurement) becomes null."""
    if isinstance(x, float) and np.isnan(x):
        return None
    if isinstance(x, dict):
        return {k: to_builtin(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_builtin(v) for v in x]
    if isinstance(x, np.generic):
        return to_builtin(x.item())
    return x


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", nargs="+", choices=sorted(EXPERIMENTS), help="experiments to run")
    parser.add_argument("--out", default=os.path.join(ROOT, "output"))
    parser.add_argument("--update-docs", action="store_true",
                        help="copy the key figures and the summary into docs/")
    args = parser.parse_args(argv)
    names = args.only or list(EXPERIMENTS)
    if args.update_docs and set(names) != set(EXPERIMENTS):
        parser.error("--update-docs copies every table and figure into docs/, so it needs "
                     "a full run: leave out --only")
    os.makedirs(args.out, exist_ok=True)
    metrics_path = os.path.join(args.out, "metrics.json")
    metrics = {}
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics = json.load(f)
    for name in names:
        start = time.perf_counter()
        metrics[name] = to_builtin(EXPERIMENTS[name](args.out))
        print(f"{name}: {time.perf_counter() - start:.1f} s", flush=True)
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2, allow_nan=False)
    write_summary(metrics, os.path.join(args.out, "summary.md"))
    if args.update_docs:
        docs = os.path.join(ROOT, "docs")
        for rel in DOC_FIGURES:
            src = os.path.join(args.out, rel)
            if os.path.exists(src):
                dst = os.path.join(docs, "figures", rel.replace("/", "_"))
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copyfile(src, dst)
        shutil.copyfile(os.path.join(args.out, "summary.md"), os.path.join(docs, "results.md"))


if __name__ == "__main__":
    main()
