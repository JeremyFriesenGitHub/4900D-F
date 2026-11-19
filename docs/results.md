# Experiment summary

Produced by `python -m experiments.run_all`. Edge error is the RMS of the relative edge-length change; distances are divided by the bounding-box diagonal. "Iterations to within 1%" counts the iterations until the energy is within 1% of its final value.

## Deformations

| scenario | vertices | constrained | iterations | converged | energy never rose | iterations to within 1% | edge error Laplacian | edge error ARAP | largest move after iteration 100 | setup (s) | ms / iteration |
|---|---|---|---|---|---|---|---|---|---|---|---|
| bar_bend | 822 | 82 | 128 | yes | yes | 31 | 0.191 | 0.065 | 0.00047 | 0.00741 | 3.27 |
| bar_twist | 822 | 82 | 440 | yes | yes | 131 | 0.174 | 0.0668 | 0.028 | 0.00402 | 3.24 |
| hinge | 6002 | 1202 | 132 | yes | yes | 108 | 0.669 | 0.011 | 0.00056 | 0.0234 | 21.4 |
| bunny_ears | 2503 | 881 | 155 | yes | yes | 22 | 0.0679 | 0.0362 | 0.0013 | 0.0143 | 10.3 |
| kid_arms | 8441 | 724 | 862 | yes | yes | 90 | 0.201 | 0.0406 | 0.042 | 0.0455 | 36.6 |
| camel_drink | 9770 | 3381 | 797 | yes | yes | 96 | 0.1 | 0.0168 | 0.02 | 0.048 | 40.2 |

Change of surface area and enclosed volume ("-": the mesh is open, so it encloses no volume). "crossings" counts the pairs of an edge and a triangle that pass through each other in the rest pose and in the ARAP result; where it is not zero the surface passes through itself and the volume change is of the signed volume, which counts the overlap twice.

| scenario | area Laplacian | area ARAP | volume Laplacian | volume ARAP | crossings (rest / ARAP) |
|---|---|---|---|---|---|
| bar_bend | -18.5% | -0.4% | -33.8% | -2.6% | 0 / 0 |
| bar_twist | -7.4% | -1.0% | -16.9% | -11.5% | 0 / 0 |
| hinge | -4.7% | 0.0% | -16.0% | -0.1% | 0 / 326 |
| bunny_ears | +1.7% | +2.0% | - | - | 0 / 0 |
| kid_arms | +20.4% | +3.0% | +27.2% | -4.0% | 0 / 0 |
| camel_drink | -4.0% | -0.3% | -2.5% | -0.6% | 20 / 300 |

Where the bars' volume goes (volume = length of the centre line times the mean cross-section along it): bar_bend: centre line +0.6%, mean cross-section -3.2%; bar_twist: centre line 0.0%, mean cross-section -11.5%, narrowest cross-section 0.74 of its rest area.

Hinge, edge error inside the top block: Laplacian 0.331, ARAP 0.000221.

## Edge weights

"cotangent, raw": negative cotangent weights are kept (`clamp="none"`); the other rows clamp them to zero.

| scenario | weights | edge error (RMS) | edge error (max) | iterations | energy |
|---|---|---|---|---|---|
| bunny_ears | cotangent | 0.0362 | 0.246 | 155 | 0.105 |
| bunny_ears | uniform | 0.037 | 0.312 | 123 | 0.184 |
| bunny_ears | cotangent, raw | 0.0715 | 1.53 | 165 | -2.18 |
| camel_drink | cotangent | 0.0168 | 0.201 | 797 | 0.0102 |
| camel_drink | uniform | 0.0144 | 0.22 | 1000 | 0.023 |
| camel_drink | cotangent, raw | 0.306 | 11.1 | 1000 | -3.98 |

## Rest pose with no handle moved (50 iterations)

| mesh | clamp | negative raw weights | max drift | lowest energy | libigl max drift |
|---|---|---|---|---|---|
| bunny | none | 0.0715 | 0.0723 | -3.09 | 0.0723 |
| bunny | zero | 0.0715 | 4.28e-14 | 7.9e-28 | - |
| bunny | geomproc | 0.0715 | 6.19e-14 | 9.1e-28 | - |
| camel | none | 0.119 | 0.148 | -4.61 | 0.148 |
| camel | zero | 0.119 | 1.38e-13 | 7.26e-27 | - |
| camel | geomproc | 0.119 | 2.76e-13 | 1.36e-26 | - |
| kid | none | 0.0517 | 0.0277 | -0.0704 | 0.0277 |
| kid | zero | 0.0517 | 1.39e-13 | 8.58e-28 | - |
| kid | geomproc | 0.0517 | 9.2e-14 | 1.33e-27 | - |

## Initial guess

| scenario | initial guess | iterations | iterations to within 1% | energy start | energy end | distance to Laplacian-start result | seconds |
|---|---|---|---|---|---|---|---|
| bar_bend | laplacian | 128 | 31 | 0.216 | 0.0156 | 0 | 0.381 |
| bar_bend | thin_plate | 118 | 14 | 0.114 | 0.0156 | 0.000422 | 0.353 |
| bar_bend | rest | 142 | 38 | 22.5 | 0.0156 | 5.02e-05 | 0.478 |
| hinge | laplacian | 132 | 108 | 2.79 | 0.000599 | 0 | 2.87 |
| hinge | thin_plate | 93 | 65 | 0.474 | 0.000596 | 0.000586 | 1.97 |
| hinge | rest | 227 | 201 | 335 | 0.000598 | 0.000438 | 5.29 |
| camel_drink | laplacian | 797 | 96 | 0.178 | 0.0102 | 0 | 31.6 |
| camel_drink | thin_plate | 781 | 146 | 0.165 | 0.0102 | 0.000802 | 30.1 |
| camel_drink | rest | 798 | 97 | 23.1 | 0.0102 | 2.78e-06 | 33.2 |

## Timing (bar bend, 20 iterations)

Medians of repeated runs. The dense global step is one row-replaced `numpy.linalg.solve` with GeomProc's matrix, as in the class demos.

| vertices | setup (s) | local step (ms) | global step (ms) | dense global step (ms) | GeomProc Laplacian build (s) |
|---|---|---|---|---|---|
| 302 | 0.00428 | 1.64 | 0.247 | 1.48 | 0.103 |
| 822 | 0.00401 | 2.45 | 0.355 | 10.7 | 0.277 |
| 2594 | 0.00907 | 7.9 | 1.13 | 177 | 0.939 |
| 5810 | 0.0185 | 16.8 | 2.63 | 1228 | 2.38 |
| 10306 | 0.0348 | 29.1 | 4.77 | - | - |
| 23138 | 0.0899 | 67.4 | 11.9 | - | - |

## Agreement with GeomProc's operator and with libigl

Largest distance between corresponding vertices, divided by the bounding-box diagonal. The libigl columns are empty when libigl is not installed. The last column repeats the ten iterations with libigl's own single-precision rotation fit in place of the one here. The entries at rounding level (1e-12 and below) come from dense solves and depend on the number of threads the linear algebra library uses.

| scenario | membrane, GeomProc operator | one ARAP step, GeomProc operator | ten ARAP iterations, GeomProc operator | one step, GeomProc operator, inconsistent b | one step, area-weighted, consistent b | one step, area-weighted, inconsistent b | libigl, ten iterations | libigl, ten iterations, libigl's rotation fit |
|---|---|---|---|---|---|---|---|---|
| bar_bend | 7e-15 | 6.9e-15 | 2.1e-14 | 0.34 | 2.1e-12 | 0.11 | 9.2e-06 | 5.1e-08 |
| bunny_ears | 1.1e-15 | 1.5e-15 | 5.5e-15 | 0.68 | 4.5e-13 | 0.24 | 5.8e-05 | 2.1e-15 |

## Large rotations

Bends use a bar of radius 0.05, twists the default (slightly jittered) bar. Every solve stops when an iteration lowers the energy by less than 1e-08 of the starting energy and moves no vertex more than 1e-06 of the diagonal, or after 10000 iterations. "steps" moves the handle 10 degrees at a time and starts each solve from the previous result; its iteration count is the total over all steps and "converged" refers to the last step. "crossings" counts the pairs of an edge and a triangle that pass through each other (0: the surface does not cross itself). The twist column is the twist of the result, measured ring by ring along the bar ("folded": the vertices of some ring turned by angles so different that the length of their mean direction is below 0.5, or two neighbouring rings differ by more than 90 degrees, so the twist is undefined). "narrowest" is the smallest area that a ring of side vertices winds around when it is projected along the bar's axis, as a fraction of its rest area; once rings tilt this projected area is smaller than a cut across the bar. Volume changes are of the signed volume: where the surface crosses itself, parts that are turned inside out count negatively, and below -100 % the signed volume is negative.

| deformation | degrees | start | iterations | converged | energy | edge error | volume change | crossings | twist reached | narrowest |
|---|---|---|---|---|---|---|---|---|---|---|
| bend | 90 | laplacian | 144 | yes | 0.00217 | 0.0367 | -0.3% | 0 | - | - |
| bend | 90 | thin_plate | 121 | yes | 0.00217 | 0.0367 | -0.3% | 0 | - | - |
| bend | 90 | steps | 1147 | yes | 0.00217 | 0.0367 | -0.3% | 0 | - | - |
| bend | 180 | laplacian | 289 | yes | 0.00814 | 0.0685 | -3.5% | 0 | - | - |
| bend | 180 | thin_plate | 461 | yes | 0.00814 | 0.0685 | -3.5% | 0 | - | - |
| bend | 180 | steps | 2145 | yes | 0.00814 | 0.0685 | -3.5% | 0 | - | - |
| bend | 270 | laplacian | 914 | yes | 0.0495 | 0.0631 | -108.1% | 215 | - | - |
| bend | 270 | thin_plate | 8405 | yes | 0.0149 | 0.0698 | -14.1% | 0 | - | - |
| bend | 270 | steps | 12738 | yes | 0.0149 | 0.0698 | -14.1% | 0 | - | - |
| twist | 90 | laplacian | 181 | yes | 0.0161 | 0.0487 | -2.7% | 0 | 90 | 0.97 |
| twist | 90 | thin_plate | 177 | yes | 0.0161 | 0.0487 | -2.7% | 0 | 90 | 0.97 |
| twist | 90 | steps | 1176 | yes | 0.0161 | 0.0487 | -2.7% | 0 | 90 | 0.97 |
| twist | 180 | laplacian | 9509 | yes | 0.03 | 0.0655 | -16.1% | 126 | folded | 0.21 |
| twist | 180 | thin_plate | 10000 | no | 0.0299 | 0.0664 | -16.1% | 122 | folded | 0.21 |
| twist | 180 | steps | 12131 | yes | 0.0516 | 0.0788 | -23.3% | 40 | 180 | 0.17 |
| twist | 270 | laplacian | 184 | yes | 0.0161 | 0.0406 | -3.2% | 0 | -90 | 0.96 |
| twist | 270 | thin_plate | 179 | yes | 0.0161 | 0.0406 | -3.2% | 0 | -90 | 0.96 |
| twist | 270 | steps | 33639 | yes | 0.0254 | 0.0537 | -7.5% | 46 | -90 | 0.58 |
| twist | 360 | laplacian | 1 | yes | 2.23e-29 | 1.96e-15 | 0.0% | 0 | 0 | 1 |
| twist | 360 | thin_plate | 1 | yes | 1.97e-29 | 1.86e-15 | 0.0% | 0 | 0 | 1 |
| twist | 360 | steps | 35045 | yes | 1.01e-09 | 1.09e-05 | 0.0% | 0 | 0 | 1 |

Twist in 10-degree steps, after each step:

| handle rotation | twist reached | narrowest | crossings | energy | iterations | converged |
|---|---|---|---|---|---|---|
| 10 | 10 | 1 | 0 | 0.000202 | 148 | yes |
| 20 | 20 | 1 | 0 | 0.000807 | 126 | yes |
| 30 | 30 | 1 | 0 | 0.00181 | 126 | yes |
| 40 | 40 | 0.99 | 0 | 0.00322 | 126 | yes |
| 50 | 50 | 0.99 | 0 | 0.00502 | 127 | yes |
| 60 | 60 | 0.99 | 0 | 0.00721 | 128 | yes |
| 70 | 70 | 0.98 | 0 | 0.00979 | 130 | yes |
| 80 | 80 | 0.97 | 0 | 0.0127 | 131 | yes |
| 90 | 90 | 0.97 | 0 | 0.0161 | 134 | yes |
| 100 | 100 | 0.96 | 0 | 0.0198 | 137 | yes |
| 110 | 110 | 0.95 | 0 | 0.0238 | 144 | yes |
| 120 | 120 | 0.87 | 0 | 0.0279 | 3237 | yes |
| 130 | 130 | 0.78 | 0 | 0.0321 | 490 | yes |
| 140 | 140 | 0.69 | 0 | 0.0362 | 600 | yes |
| 150 | 150 | 0.57 | 0 | 0.0403 | 813 | yes |
| 160 | 160 | 0.44 | 0 | 0.0443 | 1150 | yes |
| 170 | 170 | 0.29 | 0 | 0.0481 | 1672 | yes |
| 180 | 180 | 0.17 | 40 | 0.0516 | 2712 | yes |
| 190 | folded | 0.14 | 62 | 0.0549 | 3025 | yes |
| 200 | folded | 0.1 | 70 | 0.0577 | 3665 | yes |
| 210 | folded | 0.072 | 82 | 0.06 | 1024 | yes |
| 220 | folded | 0.048 | 86 | 0.062 | 6216 | yes |
| 230 | folded | 0.03 | 92 | 0.0637 | 1839 | yes |
| 240 | folded | 0.078 | 100 | 0.0652 | 3748 | yes |
| 250 | -110 | 0.45 | 59 | 0.0275 | 974 | yes |
| 260 | -100 | 0.51 | 54 | 0.0266 | 456 | yes |
| 270 | -90 | 0.58 | 46 | 0.0254 | 561 | yes |
| 280 | -80 | 0.97 | 0 | 0.0127 | 391 | yes |
| 290 | -70 | 0.98 | 0 | 0.00978 | 130 | yes |
| 300 | -60 | 0.98 | 0 | 0.00721 | 128 | yes |
| 310 | -50 | 0.99 | 0 | 0.00502 | 127 | yes |
| 320 | -40 | 0.99 | 0 | 0.00322 | 126 | yes |
| 330 | -30 | 1 | 0 | 0.00181 | 126 | yes |
| 340 | -20 | 1 | 0 | 0.000807 | 126 | yes |
| 350 | -10 | 1 | 0 | 0.000202 | 126 | yes |
| 360 | 0 | 1 | 0 | 1.01e-09 | 126 | yes |

Twist in 10-degree steps on a bar three times as long (same radius, ring spacing and end slabs; 2422 vertices, 82 constrained), every third step:

| handle rotation | twist reached | narrowest | crossings | energy | iterations | converged |
|---|---|---|---|---|---|---|
| 30 | 30 | 1 | 0 | 0.000581 | 844 | yes |
| 60 | 60 | 1 | 0 | 0.00232 | 834 | yes |
| 90 | 90 | 1 | 0 | 0.00522 | 821 | yes |
| 120 | 120 | 0.99 | 0 | 0.00926 | 806 | yes |
| 150 | 150 | 0.99 | 0 | 0.0144 | 793 | yes |
| 180 | 180 | 0.99 | 0 | 0.0208 | 780 | yes |
| 210 | 210 | 0.98 | 0 | 0.0282 | 770 | yes |
| 240 | 240 | 0.98 | 0 | 0.0367 | 762 | yes |
| 270 | 270 | 0.97 | 0 | 0.0463 | 756 | yes |

The long bar's 270-degree twist in the nudge test of the last section: energy 0.046254, continued 0.046254, nudged 0.046254 (+0.000% against continued), the two shapes 0.00% of the diagonal apart, converged yes / yes.

## Triangulation and resolution (bar bent by 90 degrees)

Meshes are named sides x rings. Only the end caps are constrained. Distances are measured at 800 fixed points of the bar's side, as a share of the bounding-box diagonal, to the 20 x 41 cotangent result and to the 20 x 41 result with the same weights. Before any deformation these points already lie up to 0.18% (RMS) from those of the 20 x 41 mesh, since each side is a polygon with its own number of sides. The bend is symmetric about the plane x = 0; "mirror asymmetry" is the distance between the result and its mirror image at the same points. On a regular mesh the points and their mirror images lie in triangles whose diagonals run the other way, so an exactly symmetric result still shows a few hundredths of a percent there, and the irregular mesh shows 0.05% (RMS) before any deformation. "kink" is the sharpest turn of the centre line from one ring to the next divided by its mean turn (1 for an even arc; "-": the rings of the irregular mesh are not level). Cotangent energies of different meshes are comparable on the regular meshes only: on the irregular mesh 400 of 2460 edges have a negative cotangent weight, which is clamped to zero. "converged" means that the default stopping rule fired, not that the shape has stopped moving. A finer mesh samples a crease of the same width with more rings, so kink values are comparable between meshes with the same number of rings only.

| weights | mesh | vertices | iterations | converged | energy | edge error | distance to cotangent 20 x 41 (RMS) | distance to same weights' 20 x 41 (RMS) | mirror asymmetry (RMS) | mirror asymmetry (max) | kink | volume change |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cotangent | 12 x 25 | 302 | 62 | yes | 0.0161 | 0.0653 | 0.43% | 0.43% | 0.05% | 0.14% | 1.0 | -2.3% |
| cotangent | 20 x 41 | 822 | 124 | yes | 0.0148 | 0.0635 | 0.00% | 0.00% | 0.02% | 0.06% | 1.1 | -2.9% |
| cotangent | 20 x 41, diagonals flipped | 822 | 124 | yes | 0.0148 | 0.0635 | 0.02% | 0.02% | 0.02% | 0.06% | 1.1 | -2.9% |
| cotangent | 20 x 41 irregular | 822 | 114 | yes | 0.0156 | 0.0611 | 0.09% | 0.09% | 0.18% | 0.36% | - | -2.8% |
| cotangent | 16 x 81 (long, thin triangles) | 1298 | 251 | yes | 0.0147 | 0.0613 | 0.25% | 0.25% | 0.01% | 0.03% | 1.6 | -2.5% |
| cotangent | 12 x 81 | 974 | 114 | yes | 0.0149 | 0.0623 | 0.42% | 0.42% | 0.01% | 0.04% | 1.2 | -2.3% |
| cotangent | 20 x 81 | 1622 | 1000 | no | 0.0139 | 0.053 | 1.84% | 1.84% | 0.02% | 0.14% | 5.1 | -4.5% |
| cotangent | 12 x 161 | 1934 | 1000 | no | 0.0145 | 0.056 | 1.45% | 1.45% | 0.01% | 0.11% | 7.6 | -3.1% |
| cotangent | 32 x 41 | 1314 | 350 | yes | 0.0134 | 0.0574 | 1.48% | 1.48% | 0.02% | 0.11% | 2.5 | -8.4% |
| cotangent | 32 x 81 | 2594 | 1000 | no | 0.0113 | 0.0442 | 3.16% | 3.16% | 0.01% | 0.09% | 4.9 | -8.8% |
| cotangent | 48 x 77 | 3698 | 1000 | no | 0.00925 | 0.0411 | 3.52% | 3.52% | 0.01% | 0.06% | 4.5 | -14.4% |
| uniform | 12 x 25 | 302 | 84 | yes | 0.0209 | 0.0635 | 0.73% | 0.28% | 0.98% | 1.55% | 1.0 | -1.1% |
| uniform | 20 x 41 | 822 | 102 | yes | 0.0191 | 0.0629 | 0.62% | 0.00% | 1.04% | 1.56% | 1.0 | -1.2% |
| uniform | 20 x 41, diagonals flipped | 822 | 102 | yes | 0.0191 | 0.0629 | 0.63% | 1.04% | 1.04% | 1.56% | 1.0 | -1.2% |
| uniform | 20 x 41 irregular | 822 | 108 | yes | 0.019 | 0.0629 | 0.61% | 0.06% | 1.03% | 1.55% | - | -1.2% |
| uniform | 16 x 81 (long, thin triangles) | 1298 | 198 | yes | 0.00754 | 0.0636 | 0.71% | 0.43% | 0.54% | 0.81% | 1.0 | -0.1% |
| uniform | 12 x 81 | 974 | 280 | yes | 0.00566 | 0.0637 | 0.71% | 0.49% | 0.45% | 0.68% | 1.0 | -0.1% |
| uniform | 20 x 81 | 1622 | 157 | yes | 0.00938 | 0.0634 | 0.68% | 0.36% | 0.62% | 0.91% | 1.0 | -0.2% |
| uniform | 12 x 161 | 1934 | 784 | yes | 0.00279 | 0.0639 | 0.72% | 0.58% | 0.27% | 0.40% | 1.0 | -0.1% |
| uniform | 32 x 41 | 1314 | 328 | yes | 0.0258 | 0.053 | 1.83% | 1.89% | 2.23% | 4.49% | 1.6 | -10.9% |
| uniform | 32 x 81 | 2594 | 198 | yes | 0.0144 | 0.0615 | 0.50% | 0.21% | 0.91% | 1.30% | 1.0 | -1.1% |
| uniform | 48 x 77 | 3698 | 1000 | no | 0.0168 | 0.0428 | 2.63% | 2.71% | 2.22% | 5.22% | 3.7 | -11.8% |

Vertex by vertex on the regular 20 x 41 mesh, with no interpolation:

| weights | mirror asymmetry (max) | flipping the diagonals moves the result (RMS) | flipping the diagonals moves the result (max) |
|---|---|---|---|
| cotangent | 0.00% | 0.00% | 0.00% |
| uniform | 1.53% | 0.99% | 1.53% |

The irregular mesh with its negative cotangent weights left in (no clamp): 588 iterations, converged yes, energy 0.0144, mirror asymmetry 0.35% (RMS) and 1.53% (max).

The irregular mesh is one random draw (seed 0). Mirror asymmetry over 24 draws (seeds 0 to 23):

| weights | RMS | max | converged |
|---|---|---|---|
| cotangent | 0.18% to 0.23% (median 0.20%) | 0.35% to 0.43% (median 0.39%) | 24 of 24 |
| uniform | 1.03% to 1.06% (median 1.05%) | 1.55% to 1.63% (median 1.58%) | 24 of 24 |
| cotangent, raw | 0.18% to 1.48% (median 0.48%) | 0.52% to 5.45% (median 1.91%) | 24 of 24 |

## One handle vertex against a handle patch (bar, 90-degree bend)

The end is moved to the same place in every case. "end rotation" is the angle by which the top end of the bar turned.

| handle | handle vertices | iterations | energy | edge error | end rotation (degrees) | volume change |
|---|---|---|---|---|---|---|
| one vertex | 1 | 541 | 0.0113 | 0.048 | 65 | -5.2% |
| end patch moved | 41 | 196 | 0.0309 | 0.0718 | 9.09e-16 | -10.9% |
| end patch moved and turned | 41 | 128 | 0.0156 | 0.065 | 90 | -2.6% |

## Is the result a minimum? (nudge and solve again)

Each result is continued twice with a tight stopping rule (energy decrease below 1e-10 of the start, step below 1e-08 of the diagonal, at most 5000 iterations): as it is ("continued"), and after every free vertex is moved at random by up to 0.1% of the diagonal ("nudged"). "nudged against continued" compares the two final energies and "apart" is the largest distance between the two final shapes as a share of the diagonal. If the nudged run ends clearly lower, the result was a balanced state that is not a minimum. If both end at the same energy below the result's, the result had only not finished converging; "moved on" is how far the continued run ends from the result it started from, so it shows how far short the result had stopped. Two runs that are both unconverged and agree say only that they creep the same way. The last two columns are the edge error and the volume change of the result itself.

| result | energy | continued | nudged | nudged against continued | apart | moved on | iterations (continued / nudged) | converged (continued / nudged) | edge error | volume change |
|---|---|---|---|---|---|---|---|---|---|---|
| bar twist 100, regular bar | 0.019603 | 0.019603 | 0.019603 | +0.000% | 0.00% | 0.02% | 152 / 579 | yes / yes | 0.0548 | -3.4% |
| bar twist 105, regular bar | 0.021564 | 0.021564 | 0.021564 | +0.000% | 0.00% | 0.02% | 153 / 1379 | yes / yes | 0.0579 | -3.7% |
| bar twist 106, regular bar | 0.021966 | 0.021966 | 0.021966 | +0.000% | 0.00% | 0.02% | 153 / 2495 | yes / yes | 0.0585 | -3.8% |
| bar twist 108, regular bar | 0.022782 | 0.022782 | 0.022778 | -0.016% | 0.72% | 0.02% | 154 / 5000 | yes / no | 0.0597 | -4.0% |
| bar twist 110, regular bar | 0.023611 | 0.023611 | 0.023592 | -0.081% | 1.14% | 0.02% | 155 / 2901 | yes / yes | 0.0609 | -4.1% |
| bar twist 135, regular bar | 0.035097 | 0.035097 | 0.03383 | -3.610% | 3.98% | 0.02% | 167 / 816 | yes / yes | 0.0764 | -6.1% |
| bar twist 135 (default, jittered bar) | 0.034175 | 0.034173 | 0.034173 | +0.000% | 0.00% | 0.54% | 5000 / 5000 | no / no | 0.0668 | -11.5% |
| bar bend 90 (default) | 0.015611 | 0.015611 | 0.015611 | +0.000% | 0.00% | 0.03% | 197 / 195 | yes / yes | 0.065 | -2.6% |
| bend 270, thin bar, Laplacian start | 0.049544 | 0.049544 | 0.049544 | +0.000% | 0.00% | 0.01% | 625 / 661 | yes / yes | 0.0631 | -108.1% |
| bend 270, thin bar, 10-degree steps | 0.014873 | 0.014873 | 0.014873 | +0.000% | 0.00% | 0.20% | 5000 / 5000 | no / no | 0.0698 | -14.1% |
| twist 180, 10-degree steps | 0.051647 | 0.051647 | 0.051647 | -0.000% | 0.00% | 0.21% | 5000 / 5000 | no / no | 0.0788 | -23.3% |
| bend 90, one handle vertex | 0.011314 | 0.011314 | 0.011314 | +0.000% | 0.00% | 0.15% | 1021 / 1033 | yes / yes | 0.048 | -5.2% |
| bend 90, 32 x 81, end caps only | 0.011338 | 0.011331 | 0.011331 | -0.000% | 0.00% | 1.13% | 4073 / 4075 | yes / yes | 0.0442 | -8.8% |
| bend 90, 20 x 41, end caps only | 0.014823 | 0.014823 | 0.014823 | -0.000% | 0.00% | 0.03% | 191 / 193 | yes / yes | 0.0635 | -2.9% |
| bend 90, 16 x 81, end caps only | 0.014706 | 0.014704 | 0.014704 | -0.000% | 0.00% | 0.28% | 2464 / 2474 | yes / yes | 0.0613 | -2.5% |
| bend 90, 32 x 81, end caps only, uniform weights | 0.014448 | 0.014447 | 0.014447 | +0.000% | 0.00% | 0.04% | 348 / 350 | yes / yes | 0.0615 | -1.1% |

## A few iterations per frame (bar bend dragged to 90 degrees)

Every frame starts from the previous frame's shape with the handle moved on. The last column is the largest distance, over all frames and vertices, to the converged shape for the same handle position, as a share of the diagonal.

| handle rotation per frame (degrees) | iterations per frame | largest distance to the converged shape |
|---|---|---|
| 1 | 3 | 0.58% |
| 3 | 3 | 1.54% |
