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
from arap.metrics import area_volume_change, edge_length_error, per_vertex_stretch
from arap.solver import ARAPDeformer
from arap.viz import plot_energy, plot_lines, plot_scaling, render_panels

from . import scenarios as S

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAX_ITERATIONS = 1000


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
    """Vertex indices of each ring of side vertices of a regular bar along z, bottom to top."""
    r = np.hypot(rest[:, 0], rest[:, 1])
    side = np.flatnonzero(r > 0.5 * r.max())
    levels = np.unique(np.round(rest[side, 2], 9))
    return [side[np.isclose(rest[side, 2], z)] for z in levels]


def twist_angle(rest, deformed, min_coherence=0.5):
    """Twist of a bar along z in degrees: top ring angle minus bottom ring angle.

    Each ring of side vertices gets the mean angle by which its vertices
    turned about the ring's centroid; the ring angles are unwrapped from the
    bottom up, so a twist past 180 degrees is followed as long as
    neighbouring rings differ by less than 180 degrees. When a ring is
    pinched or folded onto itself, its vertices' turn angles no longer agree
    (the length of their mean unit vector drops below ``min_coherence``), the
    ring has no angle, and the twist is undefined: NaN is returned.
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


def stepped(deformer, targets_at, angles):
    """Solve for each handle angle in turn, starting from the previous result.

    This is the paper's interactive setting, where the previous frame is the
    initial guess. Yields ``(angle, result, iterations so far)``.
    """
    current, total = "laplacian", 0
    for angle in angles:
        result = deformer.deform(targets_at(angle), init=current, max_iterations=MAX_ITERATIONS)
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
    result = d.deform(t, max_iterations=MAX_ITERATIONS, snapshot_iterations=snaps)
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
    """Cotangent against uniform edge weights (paper Section 3)."""
    rows = {}
    for name in ("bunny_ears", "camel_drink"):
        sc = S.ALL[name]()
        V, F = sc.vertices, sc.faces
        c, t = sc.constraints
        panels, rows[name] = [], {}
        for w in ("cotangent", "uniform"):
            r = ARAPDeformer(V, F, c, weights=w).deform(t, max_iterations=MAX_ITERATIONS)
            rows[name][w] = summarize(r, V, F)
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
    270 degrees would compress its inner side by about half.
    """
    step = 10
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
        steps, trace = {}, []
        for a, r, total in stepped(d, targets_at, range(step, max(angles) + 1, step)):
            if kind == "twist":
                trace.append((a, twist_angle(V, r.vertices), narrowest_ring(V, r.vertices),
                              r.energies[-1]))
            if a in angles:
                steps[a] = (r, total)
        for angle in angles:
            t = targets_at(angle)
            results = {init: (d.deform(t, init=init, max_iterations=MAX_ITERATIONS), None)
                       for init in ("laplacian", "thin_plate")}
            results["steps"] = steps[angle]
            row = {}
            for how, (r, total) in results.items():
                m = summarize(r, V, F)
                if total is not None:
                    m["iterations"] = total
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
            a, reached, narrowest, energy = np.array(trace).T
            # Gaps (NaN) where a ring has folded over itself and has no twist angle.
            plot_lines(a, {"twist of the result": reached}, os.path.join(
                out, "large_rotations", "twist_steps.png"),
                xlabel="handle rotation (degrees)", ylabel="twist reached (degrees)",
                title=f"twist in {step}-degree steps (gap: a ring folded, no twist angle)",
                reference=("handle rotation", a, a))
            rows["twist_steps"] = {"handle": a.tolist(), "reached": reached.tolist(),
                                   "narrowest_ring": narrowest.tolist(),
                                   "energy": energy.tolist()}
    return rows


def triangulation(out):
    """The 90-degree bar bend on different meshes of the same bar.

    Only the end caps are constrained, so the constraints mean the same on
    every mesh. Each result is compared with the 20 x 41 cotangent result at
    800 fixed points of the bar's side (``bar_surface_points``).
    """
    meshes = {
        "12 x 25": S.bar(rings=25, sides=12),
        "20 x 41": S.bar(rings=41, sides=20),
        "16 x 81 (long, thin triangles)": S.bar(rings=81, sides=16),
        "20 x 41 irregular": S.irregular_bar(rings=41, sides=20),
        "32 x 41": S.bar(rings=41, sides=32),
        "32 x 81": S.bar(rings=81, sides=32),
        "48 x 77": S.bar(rings=77, sides=48),
    }
    rng = np.random.default_rng(1)
    samples = np.column_stack([rng.uniform(-np.pi, np.pi, 800), rng.uniform(-0.45, 0.45, 800)])
    reference = ("cotangent", "20 x 41")
    rows, points, panels = {}, {}, {}
    for w in ("cotangent", "uniform"):
        rows[w], panels[w] = {}, []
        for name, (V, F) in meshes.items():
            z0, z1 = V[:, 2].min(), V[:, 2].max()
            fixed = np.flatnonzero(V[:, 2] <= z0 + 1e-9)
            handles = np.flatnonzero(V[:, 2] >= z1 - 1e-9)
            c, t = sel.combine(V, fixed, handles, S.bend_targets(V, handles, 90.0))
            d = ARAPDeformer(V, F, c, weights=w)
            r = d.deform(t, max_iterations=MAX_ITERATIONS)
            points[w, name] = bar_surface_points(V, F, r.vertices, samples)
            m = summarize(r, V, F)
            m.update(vertices=int(V.shape[0]), diagonal=d.diagonal)
            rows[w][name] = m
            panels[w].append(dict(vertices=r.vertices, faces=F, title=name.split(" (")[0],
                                  colors=stretch_colors(V, r.vertices, F)))
    for (w, name), p in points.items():
        diagonal = rows[w][name].pop("diagonal")
        gap = np.linalg.norm(p - points[reference], axis=1) / diagonal
        # Triangulation dependence alone: against the same weighting's 20 x 41 result.
        same = np.linalg.norm(p - points[w, reference[1]], axis=1) / diagonal
        rows[w][name].update(distance_rms=float(np.sqrt(np.mean(gap ** 2))),
                             distance_max=float(np.max(gap)),
                             distance_same_weights_rms=float(np.sqrt(np.mean(same ** 2))))
    for w, row in panels.items():
        render_panels(row, os.path.join(out, "triangulation", f"{w}.png"), view=(10, 0),
                      panel_size=2.4, title=f"{w} weights (sides x rings)")
    return rows


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
                         fmt(m["setup"]["weights"] + m["setup"]["factorization"]),
                         fmt(1000 * m["timings"]["per_iteration"])])
        parts += ["## Deformations", "", markdown_table(
            ["scenario", "vertices", "constrained", "iterations", "converged",
             "energy never rose", "iterations to within 1%", "edge error Laplacian", "edge error ARAP",
             "setup (s)", "ms / iteration"], rows), ""]
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
                 fmt(m["edge_error_final"]["max_abs_relative"]), m["iterations"]]
                for name, by_w in metrics["weights"].items() for w, m in by_w.items()]
        parts += ["## Edge weights", "", markdown_table(
            ["scenario", "weights", "edge error (RMS)", "edge error (max)", "iterations"], rows), ""]
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
                                 twist_cell(m),
                                 fmt(m["narrowest_ring"], 2) if "narrowest_ring" in m else "-"])
        parts += ["## Large rotations", "",
                  "Bends use a bar of radius 0.05, twists the default bar. \"steps\" moves the "
                  "handle 10 degrees at a time and starts each solve from the previous result; "
                  "its iteration count is the total over all steps and \"converged\" refers to "
                  "the last step. The twist column is the twist of the result, measured "
                  "ring by ring along the bar (\"folded\": some ring folds over itself, so its "
                  "vertices no longer agree on an angle and the twist is undefined). \"narrowest\" is the smallest cross-section area "
                  "along the bar as a fraction of its rest area (the area each ring winds around, seen "
                  "along the bar). A volume change below -100 % "
                  "means the surface passes through itself and parts of it are inside out.", "",
                  markdown_table(["deformation", "degrees", "start", "iterations", "converged",
                                  "energy", "edge error", "volume change", "twist reached",
                                  "narrowest"], rows), ""]
        steps = metrics["large_rotations"].get("twist_steps")
        if steps:
            rows = [[fmt(int(a)), twist_cell({"t": t}, "t"), fmt(n, 2), fmt(e)]
                    for a, t, n, e in zip(steps["handle"], steps["reached"],
                                          steps["narrowest_ring"], steps["energy"])]
            parts += ["Twist in 10-degree steps, after each step:", "",
                      markdown_table(["handle rotation", "twist reached", "narrowest", "energy"],
                                     rows), ""]
    if "triangulation" in metrics:
        rows = [[w, name, m["vertices"], m["iterations"], fmt(m["converged"]),
                 fmt(m["energy_final"]), fmt(m["edge_error_final"]["rms_relative"]),
                 fmt(m["distance_rms"], 2), fmt(m["distance_max"], 2),
                 fmt(m["distance_same_weights_rms"], 2),
                 pct(m["shape_change_final"]["volume_change"])]
                for w, by_mesh in metrics["triangulation"].items() for name, m in by_mesh.items()]
        parts += ["## Triangulation and resolution (bar bent by 90 degrees)", "",
                  "Meshes are named sides x rings. Only the end caps are constrained. "
                  "Distances are to the 20 x 41 cotangent result at 800 fixed points of the "
                  "bar's side; the last distance column is to the 20 x 41 result with the "
                  "same weights. Cotangent energies of different meshes are comparable on "
                  "the regular meshes only: the irregular mesh has negative cotangent weights, "
                  "which are clamped to zero.", "",
                  markdown_table(["weights", "mesh", "vertices", "iterations", "converged",
                                  "energy", "edge error", "distance (RMS)", "distance (max)",
                                  "distance to same weights' 20 x 41 (RMS)", "volume change"],
                                 rows), ""]
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
