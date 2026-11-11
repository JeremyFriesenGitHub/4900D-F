# As-rigid-as-possible surface deformation

COMP 4900D project 2.11: an implementation of "As-Rigid-As-Possible Surface
Modeling" (Sorkine and Alexa, SGP 2007) built on the course's GeomProc
library.

![Bar bent by 90 degrees: initial guess and ARAP iterations](docs/figures/bar_bend_iterations.png)

## Setup

Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt        # numpy, scipy, matplotlib
pip install -r requirements-dev.txt    # adds pytest, and libigl for the comparisons in the tests and experiments
```

GeomProc is not on PyPI, so version 1.9 (commit `894415c` of
<https://github.com/ovankaic/GeomProc>) is included unchanged in `geomproc/`
together with its licence. Its sample meshes are in `meshes/`. Run every
command from the repository root.

## Usage

```python
import numpy as np
from arap.meshio import load_mesh, save_mesh
from arap import selection as sel
from arap.solver import ARAPDeformer

V, F = load_mesh("meshes/bunny.obj")                  # GeomProc load + normalize
fixed = sel.by_coordinate(V, "y", below=0.15)          # base stays put
ears = sel.by_coordinate(V, "y", above=0.8)            # handle
constrained, targets = sel.combine(V, fixed, ears, V[ears] + [-0.45, -0.15, 0.0])

deformer = ARAPDeformer(V, F, constrained)             # factorises L once
result = deformer.deform(targets)                      # Laplacian guess, then iterate
print(result.iterations, result.converged, result.energies[-1])
save_mesh("output/bunny_arap.obj", result.vertices, F)
```

`ARAPDeformer` can be reused for any number of target positions as long as the
set of constrained vertices stays the same. Options:

| argument | values | default |
|---|---|---|
| `weights` | `"cotangent"`, `"uniform"` | `"cotangent"` |
| `clamp` (negative cotangent weights) | `"zero"`, `"none"` (paper), `"geomproc"` | `"zero"` |
| `init` in `deform` | `"laplacian"` (slides), `"thin_plate"`, `"rest"`, or an (n, 3) array | `"laplacian"` |
| `max_iterations`, `energy_tol`, `step_tol` | stopping rule, see below | 1000 in the experiments, 500 otherwise |

Experiments and tests:

```bash
python -m experiments.run_all                      # all experiments, writes output/
python -m experiments.run_all --only bar_bend hinge
python -m experiments.run_all --update-docs        # also refreshes docs/figures and docs/results.md
python -m pytest
```

`output/` receives, for every scenario, the rest pose, the Laplacian editing
result and the ARAP result as OBJ files with the constrained vertices coloured
(fixed red, handles yellow) for viewing in MeshLab, plus the figures, the
energy curve, `metrics.json` and `summary.md`. A full run takes about eight
minutes.

## Code layout

| module | contents |
|---|---|
| `arap/laplacian.py` | sparse cotangent and uniform weights, `L = D - W`, mixed (Voronoi) areas |
| `arap/rotations.py` | batched best-fit rotations with the reflection fix |
| `arap/linear.py` | constrained sparse solver (factorised once), membrane and thin-plate deformation from the slides |
| `arap/solver.py` | `ARAPDeformer`: local step, global step, energy, gradient, iteration |
| `arap/reference.py` | dense solves with GeomProc's own `geometric_laplacian`, the way the class demos solve systems |
| `arap/selection.py` | choosing fixed and handle vertices, rigid targets (uses GeomProc's rotation helpers) |
| `arap/metrics.py` | edge-length distortion, surface area and enclosed volume |
| `arap/meshio.py` | loading, normalising and saving through GeomProc |
| `arap/viz.py` | off-screen figures with matplotlib |
| `experiments/` | the scenarios and `run_all.py` |
| `tests/` | unit tests, GeomProc consistency checks, comparison with libigl |

## Method

With rest positions `p`, deformed positions `p'`, edge weights `w_ij` and one
rotation `R_i` per vertex, ARAP minimises (paper Eq. 7, with `w_i = 1`)

```
E(p', R) = sum_i sum_{j in N(i)} w_ij || (p'_i - p'_j) - R_i (p_i - p_j) ||^2
```

by alternating two steps, each of which can only lower the energy:

1. **Local step** (Section 2.1). With `p'` fixed, `R_i` is the rotation that
   best maps the rest edges around vertex `i` onto the deformed ones. With
   `S_i = sum_j w_ij e_ij e'_ij^T = U Σ V^T`, `R_i = V U^T`. If that has
   determinant -1, the column of `U` for the smallest singular value is
   negated. All vertices are done in one batched `numpy.linalg.svd` call.
2. **Global step** (Eq. 8 and 9). With the rotations fixed, `p'` solves
   `L p' = b`, where `L = D - W` and
   `b_i = sum_j (w_ij / 2) (R_i + R_j) (p_i - p_j)`.
   Constrained vertices are eliminated (Eq. 10): their columns move to the
   right-hand side and the remaining block stays symmetric positive definite.
   That block depends only on the rest mesh and the constrained set, so it is
   factored once with `scipy.sparse.linalg.splu`. Each iteration then costs
   one batched SVD and three back-substitutions.

**Initial guess.** The course slides (Topic 7) define Laplacian deformation as
solving `Δd = 0` for the displacements `d` of the free vertices, with the
handles and fixed vertices prescribed. With the cotangent operator
`Δ = -M^{-1} L` the per-vertex area factor cancels, so this is `L p' = L p`:
naive Laplacian editing, which is also the global step with every `R_i = I`.
This is the default initial guess, the same one used in the paper's Fig. 3.
The thin-plate version from the slides, `Δ²d = 0` (solved as
`L M^{-1} L d = 0`), is available as an alternative.

**Stopping rule.** Iterate until the energy decrease of an iteration is below
`1e-6` times the energy of the initial guess and no vertex moved more than
`1e-5` of the bounding-box diagonal, or the iteration limit is reached. The
decrease is compared with the initial energy, not the current one: when the
constraints allow a rigid motion the minimum energy is zero, and a decrease
relative to the current energy then never becomes small.

## Design decisions

**Sparse operator instead of GeomProc's dense one.** GeomProc's
`mesh.geometric_laplacian` builds a dense `n × n` matrix in Python loops, and
the class demos solve with `numpy.linalg.solve` on every call. That is fine for
the bunny but too slow to run hundreds of iterations on the 8 000 to 10 000
vertex meshes (see the timing table). The operator here is built with numpy and
`scipy.sparse`, which the handout allows for linear algebra. With
`clamp="geomproc"` it reproduces GeomProc's operator to rounding error, and
`tests/test_reference.py` checks that the sparse solver and a dense solve on
GeomProc's own matrix give the same deformation.

**A consistent Laplacian.** GeomProc's operator is row-normalised:
`L_gp = D^{-1} W - I`. The handout warns that the Laplacian must be used
consistently. Scaling a row of `L` changes the solution unless the same row of
`b` is scaled by the same factor. With `L_gp` the right-hand side must be
`b_i = sum_j L_gp[i, j] · 1/2 (R_i + R_j)(p_j - p_i)`, and likewise for an
area-weighted operator `M^{-1}(W - D)`. After a single step, pairing `L_gp`
with the paper's unscaled `b` moves vertices by 34 % (bar) to 68 % (bunny) of
the bounding-box diagonal, and pairing the area-weighted operator with it
moves them by 11 % to 24 % (consistency table below; the area-weighted error
also depends on the mesh's scale, since `M^{-1}` scales as one over length
squared). The code uses the unscaled symmetric `L` with the unscaled `b`, and
the tests and the experiments check that both scaled variants give the same
answer when their `b` is scaled to match.

**Negative cotangent weights are clamped to zero.** This is the one deviation
from the paper. The scanned meshes have many obtuse triangles (12 % of the
camel's edges have a negative weight). With negative weights the energy can
become negative and the rest pose is no longer a minimum, so the iterations
walk away from it even when no handle has moved: after 50 iterations the
camel has drifted by 15 % of its diagonal (table below). Clamping makes the energy non-negative and the rest
pose an exact fixed point. GeomProc's operator is also non-negative, because
its mixed-area treatment replaces a negative cotangent by half the triangle
area. The raw weights are still available with `clamp="none"`, which is what
libigl uses, and the tests compare that setting against libigl.

**Elimination instead of row replacement.** The class demos impose
constraints by replacing rows of the matrix with identity rows. That gives the
same solution, but the matrix stops being symmetric. Elimination keeps the
system symmetric positive definite.

## Results

All numbers come from one run of `python -m experiments.run_all` (about eight
minutes) on a shared 4-core 2.1 GHz Xeon virtual machine with Python 3.13,
NumPy 2.5 and SciPy 1.18. On that machine timings can differ by more than
50 % between runs of the same code; every other number is the same from run
to run.
The full tables are in [docs/results.md](docs/results.md). Edge error is the
RMS of the relative edge-length change `(|e'| - |e|) / |e|` over all edges;
distances are divided by the bounding-box diagonal. The loaded meshes are
normalised with GeomProc so their longest side spans [-1, 1]; the synthetic
bar (GeomProc's `create_cylinder`) has radius 0.1 and length 1. The Timing
table gives medians of repeated runs. "ms per iteration" in the Deformations
table is the mean over the iterations of a single run and includes the energy
and the convergence test, and the camel's run time is also from one run.

### Deformations

| scenario | vertices | constrained | iterations | within 1 % of final energy after | edge error, Laplacian editing | edge error, ARAP | ms per iteration |
|---|---|---|---|---|---|---|---|
| bar bend, 90° | 822 | 82 | 128 | 31 | 0.191 | 0.065 | 3.1 |
| bar twist, 135° | 822 | 82 | 162 | 70 | 0.175 | 0.076 | 3.3 |
| two-part cylinder hinge, 60° | 6 002 | 1 202 | 132 | 108 | 0.669 | 0.011 | 24 |
| bunny ears | 2 503 | 881 | 155 | 22 | 0.068 | 0.036 | 11 |
| kid raising its arms | 8 441 | 724 | 862 | 90 | 0.201 | 0.041 | 39 |
| camel lowering its head | 9 770 | 3 381 | 797 | 96 | 0.100 | 0.017 | 43 |

ARAP also keeps the size of the shape much better, except for the bunny's
area, which changes slightly more than with Laplacian editing. Change of
surface area and enclosed volume (the bunny is an open mesh, so it encloses no
volume):

| scenario | area, Laplacian editing | area, ARAP | volume, Laplacian editing | volume, ARAP |
|---|---|---|---|---|
| bar bend | −18.5 % | −0.4 % | −33.8 % | −2.6 % |
| bar twist | −7.5 % | −2.9 % | −17.0 % | −6.1 % |
| hinge | −4.7 % | −0.0 % | −16.0 % | −0.1 % |
| bunny ears | +1.7 % | +2.0 % | - | - |
| kid | +20.4 % | +3.0 % | +27.2 % | −4.0 % |
| camel | −4.0 % | −0.3 % | −2.5 % | −0.6 % |

ARAP has no volume term. The losses that remain (2.6 % for the bar bend,
6.1 % for the twist) most likely come from the cross-sections shrinking: a
bent bar's cross-section flattens a little in the bending plane, which eases
both the compression of the inner side and the stretch of the outer side, and
a twisted tube whose ends are held at the same height narrows.

Fixed vertices are red and handles yellow (the twisted bar's other vertices
are striped so the twist shows). Each row shows the rest pose, Laplacian
editing (the initial guess) and the converged ARAP result.

![Twisted bar](docs/figures/bar_twist_comparison.png)
![Two-part cylinder used as a hinge](docs/figures/hinge_comparison.png)
![Kid raising its arms](docs/figures/kid_arms_comparison.png)
![Camel lowering its head](docs/figures/camel_drink_comparison.png)
![Bunny ears pulled forward](docs/figures/bunny_ears_comparison.png)

- **Laplacian editing does not rotate local detail.** It only interpolates
  the handle displacements. The bent bar shears into a slanted tube, the
  twisted bar collapses towards its axis near the handle, the kid's arms
  stretch into thin strands, and the camel's body sinks while its legs bend.
  ARAP lowers the edge error by a factor of about 2 (bunny) to 60 (hinge).
- **Rigid parts stay rigid.** In the hinge, the top block should turn as a
  rigid body around the thin neck. Inside the block, leaving out the handle
  itself, the edge error is 0.33 for Laplacian editing and 0.0002 for ARAP.
- **Some stretch is unavoidable.** If a bar of radius 0.1 is bent into a
  quarter circle of radius 2/π ≈ 0.64 and its cross-section stays round, the
  outside of the bend stretches by about 16 % and the inside compresses by as
  much. Flattening the cross-section reduces that but cannot remove it, so no
  deformation that meets these constraints can have zero edge error.

![Edge-length change in the bent bar](docs/figures/bar_bend_stretch.png)

**Convergence.** The energy decreased at every iteration in every run, as the
two exact steps guarantee. Most of the change happens early: the energy is
within 1 % of its final value after 22 to 108 iterations. The stopping rule
then keeps iterating while vertices still creep by more than `1e-5` of the
diagonal, which takes 128 to 862 iterations in all, with little visible change
after the first hundred.

**Why this is far more than the paper's 2 to 3 iterations.** The paper's
figure is for interactive editing at 10 to 30 frames per second: the handle
moves a little between frames, each frame starts from the previous frame's
result (the paper's default initial guess), and a few iterations per frame are
enough to keep up while the user drags. Here each scenario is one large edit
solved from scratch, starting from Laplacian editing, which is far from the
answer for bends and twists, and the run continues until the strict stopping
rule holds rather than until the shape looks right. The "within 1 %" column is
the closer comparison. The paper also notes that finer meshes need more
iterations, because the Laplacian's condition number grows with the mesh size;
the triangulation experiment below shows the same.

### Timing

| vertices | setup (weights + factorisation) | local step | global step (sparse) | global step (dense, GeomProc operator) | building GeomProc's operator |
|---|---|---|---|---|---|
| 822 | 5 ms | 2.6 ms | 0.4 ms | 11 ms | 0.32 s |
| 2 594 | 9 ms | 7.6 ms | 1.1 ms | 155 ms | 1.0 s |
| 5 810 | 19 ms | 17 ms | 2.5 ms | 1 040 ms | 2.3 s |
| 23 138 | 90 ms | 67 ms | 12 ms | not run | not run |

![Dense against sparse solves](docs/figures/scaling_dense_vs_sparse.png)

Times are for the bar bend on cylinders of increasing resolution. Each
iteration is dominated by the local step (the covariances and the batched
SVD). Because the factorisation is reused, the global step is six to seven
times cheaper than the local step, and about 400 times cheaper than a dense
`numpy.linalg.solve` on GeomProc's matrix at 5 810 vertices. A full ARAP run
on the 10 000-vertex camel takes about 33 seconds, most of it in the final
creeping iterations.

### Negative cotangent weights

| mesh | edges with a negative raw weight | drift after 50 iterations, raw weights | same with libigl | drift, clamped at zero |
|---|---|---|---|---|
| bunny | 7.2 % | 0.072 | 0.072 | 4e-14 |
| camel | 11.9 % | 0.148 | 0.148 | 1e-13 |
| kid | 5.2 % | 0.028 | 0.028 | 1e-13 |

No handle is moved in this test, so the rest pose is the correct answer. With
raw weights the energy becomes negative and the surface drifts away from it.
libigl, which uses raw weights, drifts by the same amount.

![Bunny drifting with raw weights](docs/figures/clamp_bunny_raw_drift.png)

### Edge weights

| scenario | weights | edge error (RMS) | edge error (max) | iterations |
|---|---|---|---|---|
| bunny ears | cotangent | 0.036 | 0.25 | 155 |
| bunny ears | uniform | 0.037 | 0.31 | 123 |
| camel | cotangent | 0.017 | 0.20 | 797 |
| camel | uniform | 0.014 | 0.22 | 1 000 (limit) |

The two weightings give similar edge errors on these meshes. Cotangent weights
have the smaller worst-case error and converge on the camel, where uniform
weights hit the iteration limit. The paper's argument for cotangent weights is
that they make the result depend less on how the surface is triangulated,
which edge error alone does not measure.

### Initial guess

| scenario | Laplacian editing (slides) | thin plate (slides) | rest pose |
|---|---|---|---|
| bar bend | 128 iterations | 118 | 142 |
| hinge | 132 | 93 | 227 |
| camel | 797 | 781 | 798 |

![Energy for each initial guess on the hinge](docs/figures/initial_guess_hinge_energy.png)

All three starting points reach the same pose to within 0.1 % of the
diagonal. The thin-plate guess starts at a lower energy and saves iterations
on the bar and the hinge. Starting from the rest pose costs the most
iterations.

### Agreement with GeomProc and libigl

Largest distance between corresponding vertices, divided by the bounding-box
diagonal.

| | bar bend | bunny ears |
|---|---|---|
| membrane solve, GeomProc operator | 7e-15 | 1e-15 |
| one ARAP step, GeomProc operator, consistent `b` | 7e-15 | 2e-15 |
| ten ARAP iterations, GeomProc operator | 2e-14 | 6e-15 |
| one step, GeomProc operator, inconsistent `b` | **0.34** | **0.68** |
| one step, area-weighted operator, consistent `b` | 2e-12 | 5e-13 |
| one step, area-weighted operator, inconsistent `b` | **0.11** | **0.24** |
| libigl, ten iterations, raw weights | 9e-6 | 6e-5 |

The GeomProc rows compare the sparse solver (with `clamp="geomproc"`)
against dense solves on GeomProc's own `geometric_laplacian` with rows
replaced for the constraints, as in the class demos. The area-weighted rows
use `M^{-1}(W - D)` with the mixed areas. "Inconsistent" pairs a row-scaled
matrix with the paper's unscaled right-hand side, which is the mistake the
handout warns about. The last row compares the raw-weight setting with
libigl's ARAP (spokes energy, the paper's energy). libigl fits the rotations
in single precision, which most likely explains the remaining difference.

### Large rotations

The default scenarios bend the bar by 90° and twist it by 135°. Here the
bends go to 270° and the twists to 360°. Bends use a thinner bar (radius
0.05, 61 rings): bending the default bar by 270° would compress its inner side
by about half. Each edit is solved three ways: from the Laplacian-editing
guess (the default), from the thin-plate guess, and in 10° steps where each
solve starts from the previous result, as in the paper's interactive use.
Final energies, with the full table in [docs/results.md](docs/results.md):

| | Laplacian start | thin-plate start | 10° steps |
|---|---|---|---|
| bend 90° | 0.0022 | 0.0022 | 0.0022 |
| bend 180° | 0.0081 | 0.0081 | 0.0081 |
| bend 270° | 0.050, passes through itself | 0.015 (iteration limit) | 0.015 |
| twist 90° | 0.016 | 0.016 | 0.016 |
| twist 180° | 0.030 (iteration limit), buckled and folded | 0.095, stopped early, pinched next to the handle | 0.051, narrowed in the middle |
| twist 270° | 0.016, a 90° twist the other way | 0.016, the same | 0.068, folded through itself |
| twist 360° | 0, the rest pose | 0, the rest pose | 1e-7, the rest pose |

![Bar bent by 270 degrees](docs/figures/large_rotations_bend_270.png)

- **Bends: the thin-plate start and the steps fix the failure.** Up to 180°
  all three starts reach the same shape. At 270° Laplacian editing shortens
  and shears the bar into a straight tube between its ends. ARAP started there
  ends with three times the energy and the bar passing through itself: parts
  of the surface are turned inside out, so the signed volume even changes sign
  (a volume change of −108 % in the table). The thin-plate guess and the steps
  both reach a loop with energy 0.015. The two loops are similar but not the
  same: they differ by up to 10 % of the diagonal, and the thin-plate run
  stopped at the iteration limit.
- **Twists: the handle positions do not say how many turns were made.** After
  a 270° turn the handle vertices sit exactly where a −90° turn puts them,
  and after 360° exactly where they started. A single solve therefore returns
  a 90° twist the other way for 270°, and the untwisted bar for 360°, from
  either start.
- **The steps carry the twist only to 180°.** Up to 170° the stepped twist is
  even along the bar and every cross-section keeps at least 88 % of its area.
  At 180° the tube snaps: the middle narrows to 17 % of its rest
  cross-section, about half of the twist bunches into the middle fifth of the
  bar, and the energy drops from 0.054 to 0.051 although the handle turned
  further. From 190° to 270° rings in the middle fold over themselves, with
  the walls passing through each other, and the narrowest cross-section is
  4 % to 14 % of its rest area. A folded ring has no defined twist angle,
  hence the gap in the plot. At 280° the fold opens and the bar springs to an
  even −80° twist with the same energy as the +80° step, which then unwinds
  evenly to the rest pose at 360°.
- **Past about 170° an even twist is not what ARAP prefers.** At 180° the
  Laplacian start buckles and folds into a shape with energy 0.030, below the
  even twist's 0.054 at 170°. ARAP has no explicit bending term and no volume
  term, so a hollow tube can lower its energy by buckling, folding or
  narrowing instead of twisting evenly. The thin-plate start shows a weakness
  of the stopping rule: it stops after 19 iterations on a plateau at 0.095,
  pinched next to the handle (narrowest cross-section 3 % of its rest area),
  although more iterations bring it down to about 0.030, like the Laplacian
  start.

![Bar twisted by 180 degrees](docs/figures/large_rotations_twist_180.png)
![Bar twisted by 270 degrees](docs/figures/large_rotations_twist_270.png)
![Twist reached in 10-degree steps](docs/figures/large_rotations_twist_steps.png)

The volume column in [docs/results.md](docs/results.md) follows the same
pattern: the bends that work lose 0.3 % (90°), 3.5 % (180°) and 14 % (270°)
of their volume, the 180° twists lose 9 % to 23 % however they are solved,
and the folded stepped 270° twist 22 %, against 2.7 % for the 90° twist
(and 3.1 % for the single 270° solves, which are 90° twists). The steps are
not cheap with this stopping rule either: each 10° step still takes 70 to
200 iterations on average, so the stepped 270° bend costs about 4 000 in
total.

### One handle vertex instead of a patch

The paper's dinosaur (Fig. 1) and cactus (Fig. 7) are edited by moving a
single vertex. Here the end of the default bar is moved to the end of the 90°
arc in three ways:

| handle | handle vertices | energy | edge error | end rotation | volume change |
|---|---|---|---|---|---|
| centre vertex of the top cap, moved | 1 | 0.011 | 0.048 | 65° | −5.2 % |
| top end, moved | 41 | 0.031 | 0.072 | 0° | −10.9 % |
| top end, moved and turned by 90° (the bar bend scenario) | 41 | 0.016 | 0.065 | 90° | −2.6 % |

![Three kinds of handle](docs/figures/point_handle_bar.png)

A single vertex has no orientation, so ARAP decides how the end turns. It
turns by 65° on its own: ARAP produces rotations from a translation alone,
which is the point of the paper's cactus example and something Laplacian
editing cannot do. Because the end's orientation is free, this has the lowest
energy of the three. No spike forms at the moved vertex; its neighbours follow
it, and most of the bending gathers in a kink near the fixed end. A moved
patch keeps its orientation, so the bar has to bend back to meet it in an
S shape, with nearly three times the energy of the single vertex (and twice
that of the turned patch).

### Triangulation and resolution

The 90° bar bend is repeated on seven meshes of the same bar, named
sides × rings: coarser and finer regular meshes, long thin triangles
(16 × 81), and the 20 × 41 mesh with every side vertex between the end rings
moved randomly along the surface by up to 20 % of the spacing ("irregular";
up to 25 % no triangle can fold over). Only the end caps are constrained, so
the constraints mean the same thing on every mesh. Each result is compared at
800 fixed points of the bar's side, found by their barycentric coordinates in
the rest mesh, with the 20 × 41 result that uses the same weights:

| mesh | vertices | cotangent: iterations | cotangent: distance (RMS) | uniform: iterations | uniform: distance (RMS) |
|---|---|---|---|---|---|
| 12 × 25 | 302 | 62 | 0.4 % | 84 | 0.3 % |
| 20 × 41 | 822 | 124 | 0 (reference) | 102 | 0 (reference) |
| 16 × 81, long thin triangles | 1 298 | 251 | 0.3 % | 198 | 0.4 % |
| 20 × 41, irregular | 822 | 114 | 0.1 % | 108 | 0.1 % |
| 32 × 41 | 1 314 | 350 | **1.5 %** | 328 | **1.9 %** |
| 32 × 81 | 2 594 | 1 000 (limit) | **3.2 %** | 198 | 0.2 % |
| 48 × 77 | 3 698 | 1 000 (limit) | **3.5 %** | 1 000 (limit) | **2.7 %** |

![Cotangent weights on seven meshes](docs/figures/triangulation_cotangent.png)
![Uniform weights on seven meshes](docs/figures/triangulation_uniform.png)

- **On coarse and medium meshes the triangulation hardly matters, with either
  weighting.** Up to 20 sides, every result is within 0.5 % of the diagonal
  of its 20 × 41 reference, including the irregular mesh and the long thin
  triangles. The two weightings differ from each other by more than that:
  uniform results are 0.6 % to 0.7 % from the cotangent reference. So this
  experiment does not show the advantage the paper claims for cotangent
  weights. The irregular mesh is a mild test: jittering makes almost half of
  its triangles obtuse (largest angle 141°) and gives 16 % of its edges a
  negative cotangent weight, which the default clamp sets to zero, so its
  cotangent run does not use the paper's raw weights either. A mesh whose
  triangle sizes vary strongly would be a better test.
- **On finer meshes the bar kinks instead of bending evenly.** With cotangent
  weights from 32 sides on, the tube creases near its ends and is straighter
  in between, most clearly on 32 × 81 and 48 × 77. The energy is lower than
  the even arc's and keeps falling as the mesh gets finer (0.0148 at 20 × 41,
  0.0134 at 32 × 41, 0.0113 at 32 × 81, 0.0093 at 48 × 77), and the volume
  loss grows from about 3 % to 8 % to 14 %. Uniform weights kink on 48 × 77
  as well and dent on 32 × 41, but keep the arc on 32 × 81. We read this as a
  property of the energy, not a numerical error. ARAP has no explicit bending
  term: it penalises each vertex neighbourhood for any change other than a
  rotation, and a neighbourhood that only bends with the surface costs a
  little, but that cost falls as the triangles get smaller (roughly with the
  square of the edge length in a side check on a rolled-up strip). The inner
  side of an even bend with a round cross-section has to compress by about
  16 %; creasing avoids much of that compression at the price of sharp
  bending, which a finer mesh pays less for. Coarse meshes are too stiff to
  crease, so they give the even arc. A side check outside `run_all`, with a
  much tighter tolerance and
  up to 20 000 iterations, gave the same picture: 24 sides still bend evenly,
  28 sides start to kink, and the energy was still falling at 48 sides. For
  thin hollow shapes the ARAP result therefore depends on the resolution,
  and refining the mesh does not settle on one answer.
- **Iterations grow with resolution**, as the paper says: on the regular
  meshes that keep the even arc, cotangent weights need 62, 124 and 251
  iterations at 302, 822 and 1 298 vertices. The 1 000-iteration runs on the
  two finest meshes are slow mainly because the creases are still forming.
  Cotangent energies can be compared between the regular
  meshes; on the irregular mesh the clamped weights change the energy. With
  uniform weights the energy also depends on the shape of the triangles
  (0.0075 on the long thin triangles against 0.019 on 20 × 41).

### Limitations

- The constraints are hard, and the set of constrained vertices is fixed
  when the solver is created. A new set means a new factorisation, which
  takes about 0.1 s or less here.
- Convergence is linear, so a tight tolerance costs many iterations on large
  meshes. The paper's interactive setting runs only a few iterations per
  frame.
- The energy has no explicit bending term and no volume term, and nothing
  stops the surface from passing through itself. Hollow shapes such as the
  bar can buckle, kink, fold or pinch instead of bending or twisting evenly,
  under large edits and, on fine meshes, under moderate ones.
- Positional constraints do not record how many turns a handle has made: a
  270° twist is solved as a 90° twist the other way, and 360° as no twist.
  Moving the handle in 10° steps carried the twist only up to 180° on the
  bar; past that the tube folded through itself and later unwound.
- The stopping rule can be met on a plateau that is not a minimum: the
  thin-plate start of the 180° twist stopped after 19 iterations at three
  times the energy that further iterations reach.
- Clamping negative weights changes the energy on meshes with obtuse
  triangles. An intrinsic Delaunay Laplacian would avoid negative weights
  without changing the energy, but it is beyond the scope of this project.

## Libraries and references

- O. Sorkine and M. Alexa. As-Rigid-As-Possible Surface Modeling. *Symposium
  on Geometry Processing*, 2007.
- M. Meyer, M. Desbrun, P. Schröder and A. H. Barr. Discrete
  Differential-Geometry Operators for Triangulated 2-Manifolds. In
  *Visualization and Mathematics III*, Springer, 2003 (mixed Voronoi areas).
- COMP 4900D course slides, Topic 7: Deformation.
- [GeomProc](https://github.com/ovankaic/GeomProc) 1.9: mesh loading,
  normalisation and saving, rotation helpers, sample meshes, and the reference
  `geometric_laplacian`.
- [NumPy](https://numpy.org/): arrays and the batched SVD.
- [SciPy](https://scipy.org/): sparse matrices and the sparse LU factorisation.
- [Matplotlib](https://matplotlib.org/): figures.
- [libigl](https://libigl.github.io/) Python bindings: an independent ARAP
  implementation used only for comparison, in the tests and in two experiment
  columns (rest-pose drift with raw weights, and the libigl column of the
  agreement table). The `arap/` package does not use it; without it the tests
  that need it are skipped and those columns are left empty.
