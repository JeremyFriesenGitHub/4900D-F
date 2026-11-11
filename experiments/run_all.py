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
from arap.metrics import edge_length_error, per_vertex_stretch
from arap.solver import ARAPDeformer
from arap.viz import plot_energy, plot_scaling, render_panels

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
        "timings": result.timings,
    }


def markdown_table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for row in rows:
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(lines)


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
    with open(path, "w") as f:
        f.write("\n".join(parts))


EXPERIMENTS = {name: (lambda out, n=name: showcase(out, n)) for name in S.ALL}
EXPERIMENTS.update({
    "weights": weights,
    "clamp": clamp,
    "initial_guess": initial_guess,
    "scaling": scaling,
    "geomproc_consistency": geomproc_consistency,
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
]


def to_builtin(x):
    if isinstance(x, dict):
        return {k: to_builtin(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_builtin(v) for v in x]
    if isinstance(x, np.generic):
        return x.item()
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
            json.dump(metrics, f, indent=2)
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
