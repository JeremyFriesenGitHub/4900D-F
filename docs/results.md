# Experiment summary

Produced by `python -m experiments.run_all`. Edge error is the RMS of the relative edge-length change; distances are divided by the bounding-box diagonal. "Iterations to within 1%" counts the iterations until the energy is within 1% of its final value.

## Deformations

| scenario | vertices | constrained | iterations | converged | energy never rose | iterations to within 1% | edge error Laplacian | edge error ARAP | setup (s) | ms / iteration |
|---|---|---|---|---|---|---|---|---|---|---|
| bar_bend | 822 | 82 | 128 | yes | yes | 31 | 0.191 | 0.065 | 0.0052 | 3.12 |
| bar_twist | 822 | 82 | 162 | yes | yes | 70 | 0.175 | 0.0764 | 0.00426 | 3.27 |
| hinge | 6002 | 1202 | 132 | yes | yes | 108 | 0.669 | 0.011 | 0.0245 | 24.4 |
| bunny_ears | 2503 | 881 | 155 | yes | yes | 22 | 0.0679 | 0.0362 | 0.0154 | 11.3 |
| kid_arms | 8441 | 724 | 862 | yes | yes | 90 | 0.201 | 0.0406 | 0.0446 | 38.5 |
| camel_drink | 9770 | 3381 | 797 | yes | yes | 96 | 0.1 | 0.0168 | 0.0358 | 43.4 |

Change of surface area and enclosed volume ("-": the mesh is open, so it encloses no volume).

| scenario | area Laplacian | area ARAP | volume Laplacian | volume ARAP |
|---|---|---|---|---|
| bar_bend | -18.5% | -0.4% | -33.8% | -2.6% |
| bar_twist | -7.5% | -2.9% | -17.0% | -6.1% |
| hinge | -4.7% | -0.0% | -16.0% | -0.1% |
| bunny_ears | +1.7% | +2.0% | - | - |
| kid_arms | +20.4% | +3.0% | +27.2% | -4.0% |
| camel_drink | -4.0% | -0.3% | -2.5% | -0.6% |

Hinge, edge error inside the top block: Laplacian 0.331, ARAP 0.000221.

## Edge weights

| scenario | weights | edge error (RMS) | edge error (max) | iterations |
|---|---|---|---|---|
| bunny_ears | cotangent | 0.0362 | 0.246 | 155 |
| bunny_ears | uniform | 0.037 | 0.312 | 123 |
| camel_drink | cotangent | 0.0168 | 0.201 | 797 |
| camel_drink | uniform | 0.0144 | 0.22 | 1000 |

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
| bar_bend | laplacian | 128 | 31 | 0.216 | 0.0156 | 0 | 0.413 |
| bar_bend | thin_plate | 118 | 14 | 0.114 | 0.0156 | 0.000422 | 0.41 |
| bar_bend | rest | 142 | 38 | 22.5 | 0.0156 | 5.02e-05 | 0.444 |
| hinge | laplacian | 132 | 108 | 2.79 | 0.000599 | 0 | 3.02 |
| hinge | thin_plate | 93 | 65 | 0.474 | 0.000596 | 0.000586 | 2.19 |
| hinge | rest | 227 | 201 | 335 | 0.000598 | 0.000438 | 5.04 |
| camel_drink | laplacian | 797 | 96 | 0.178 | 0.0102 | 0 | 33.3 |
| camel_drink | thin_plate | 781 | 146 | 0.165 | 0.0102 | 0.000802 | 33.4 |
| camel_drink | rest | 798 | 97 | 23.1 | 0.0102 | 2.78e-06 | 34.2 |

## Timing (bar bend, 20 iterations)

Medians of repeated runs. The dense global step is one row-replaced `numpy.linalg.solve` with GeomProc's matrix, as in the class demos.

| vertices | setup (s) | local step (ms) | global step (ms) | dense global step (ms) | GeomProc Laplacian build (s) |
|---|---|---|---|---|---|
| 302 | 0.00242 | 0.947 | 0.15 | 1.85 | 0.119 |
| 822 | 0.00469 | 2.63 | 0.375 | 10.8 | 0.318 |
| 2594 | 0.00887 | 7.6 | 1.08 | 155 | 0.995 |
| 5810 | 0.0187 | 16.5 | 2.52 | 1.04e+03 | 2.27 |
| 10306 | 0.0364 | 29.7 | 5.06 | - | - |
| 23138 | 0.0898 | 67.3 | 11.8 | - | - |

## Agreement with GeomProc's operator and with libigl

Largest distance between corresponding vertices, divided by the bounding-box diagonal. The libigl column is empty when libigl is not installed.

| scenario | membrane, GeomProc operator | one ARAP step, GeomProc operator | ten ARAP iterations, GeomProc operator | one step, GeomProc operator, inconsistent b | one step, area-weighted, consistent b | one step, area-weighted, inconsistent b | libigl, ten iterations |
|---|---|---|---|---|---|---|---|
| bar_bend | 7e-15 | 6.9e-15 | 2.1e-14 | 0.34 | 2.1e-12 | 0.11 | 9.2e-06 |
| bunny_ears | 1.1e-15 | 1.5e-15 | 5.5e-15 | 0.68 | 4.5e-13 | 0.24 | 5.8e-05 |

## Large rotations

Bends use a bar of radius 0.05, twists the default bar. "steps" moves the handle 10 degrees at a time and starts each solve from the previous result; its iteration count is the total over all steps and "converged" refers to the last step. The twist column is the twist of the result, measured ring by ring along the bar ("folded": some ring folds over itself, so its vertices no longer agree on an angle and the twist is undefined). "narrowest" is the smallest cross-section area along the bar as a fraction of its rest area (the area each ring winds around, seen along the bar). A volume change below -100 % means the surface passes through itself and parts of it are inside out.

| deformation | degrees | start | iterations | converged | energy | edge error | volume change | twist reached | narrowest |
|---|---|---|---|---|---|---|---|---|---|
| bend | 90 | laplacian | 108 | yes | 0.00217 | 0.0367 | -0.3% | - | - |
| bend | 90 | thin_plate | 82 | yes | 0.00217 | 0.0367 | -0.3% | - | - |
| bend | 90 | steps | 748 | yes | 0.00217 | 0.0367 | -0.3% | - | - |
| bend | 180 | laplacian | 213 | yes | 0.00814 | 0.0684 | -3.5% | - | - |
| bend | 180 | thin_plate | 367 | yes | 0.00814 | 0.0685 | -3.5% | - | - |
| bend | 180 | steps | 1286 | yes | 0.00814 | 0.0685 | -3.5% | - | - |
| bend | 270 | laplacian | 670 | yes | 0.0495 | 0.0631 | -108.2% | - | - |
| bend | 270 | thin_plate | 1000 | no | 0.015 | 0.0706 | -13.8% | - | - |
| bend | 270 | steps | 4093 | yes | 0.0149 | 0.0699 | -14.1% | - | - |
| twist | 90 | laplacian | 132 | yes | 0.0159 | 0.0489 | -2.7% | 90 | 0.97 |
| twist | 90 | thin_plate | 127 | yes | 0.0159 | 0.0489 | -2.7% | 90 | 0.97 |
| twist | 90 | steps | 757 | yes | 0.0159 | 0.0489 | -2.7% | 90 | 0.97 |
| twist | 180 | laplacian | 1000 | no | 0.0298 | 0.067 | -16.4% | folded | 0.21 |
| twist | 180 | thin_plate | 19 | yes | 0.0946 | 0.156 | -9.0% | -180 | 0.031 |
| twist | 180 | steps | 1989 | yes | 0.0512 | 0.079 | -23.4% | 180 | 0.17 |
| twist | 270 | laplacian | 132 | yes | 0.0159 | 0.0407 | -3.1% | -90 | 0.97 |
| twist | 270 | thin_plate | 127 | yes | 0.0159 | 0.0407 | -3.1% | -90 | 0.97 |
| twist | 270 | steps | 5478 | yes | 0.0684 | 0.0935 | -21.9% | folded | 0.071 |
| twist | 360 | laplacian | 1 | yes | 1.76e-28 | 6.89e-15 | +0.0% | 0 | 1 |
| twist | 360 | thin_plate | 1 | yes | 1.77e-28 | 6.9e-15 | +0.0% | 0 | 1 |
| twist | 360 | steps | 6396 | yes | 1.02e-07 | 0.00011 | -0.0% | 0 | 1 |

Twist in 10-degree steps, after each step:

| handle rotation | twist reached | narrowest | energy |
|---|---|---|---|
| 10 | 10 | 1 | 0.0002 |
| 20 | 20 | 1 | 0.000801 |
| 30 | 30 | 1 | 0.0018 |
| 40 | 40 | 0.99 | 0.00319 |
| 50 | 50 | 0.99 | 0.00498 |
| 60 | 60 | 0.99 | 0.00716 |
| 70 | 70 | 0.98 | 0.00971 |
| 80 | 80 | 0.97 | 0.0126 |
| 90 | 90 | 0.97 | 0.0159 |
| 100 | 100 | 0.96 | 0.0196 |
| 110 | 110 | 0.95 | 0.0236 |
| 120 | 120 | 0.94 | 0.028 |
| 130 | 130 | 0.93 | 0.0326 |
| 140 | 140 | 0.92 | 0.0376 |
| 150 | 150 | 0.91 | 0.0429 |
| 160 | 160 | 0.9 | 0.0485 |
| 170 | 170 | 0.88 | 0.0544 |
| 180 | 180 | 0.17 | 0.0512 |
| 190 | folded | 0.14 | 0.0544 |
| 200 | folded | 0.11 | 0.0572 |
| 210 | folded | 0.072 | 0.0595 |
| 220 | folded | 0.058 | 0.0615 |
| 230 | folded | 0.04 | 0.0632 |
| 240 | folded | 0.037 | 0.0647 |
| 250 | folded | 0.078 | 0.0661 |
| 260 | folded | 0.089 | 0.0673 |
| 270 | folded | 0.071 | 0.0684 |
| 280 | -80 | 0.97 | 0.0126 |
| 290 | -70 | 0.98 | 0.00971 |
| 300 | -60 | 0.99 | 0.00716 |
| 310 | -50 | 0.99 | 0.00498 |
| 320 | -40 | 0.99 | 0.00319 |
| 330 | -30 | 1 | 0.0018 |
| 340 | -20 | 1 | 0.000801 |
| 350 | -10 | 1 | 0.0002 |
| 360 | 0 | 1 | 1.02e-07 |

## Triangulation and resolution (bar bent by 90 degrees)

Meshes are named sides x rings. Only the end caps are constrained. Distances are to the 20 x 41 cotangent result at 800 fixed points of the bar's side; the last distance column is to the 20 x 41 result with the same weights. Cotangent energies of different meshes are comparable on the regular meshes only: the irregular mesh has negative cotangent weights, which are clamped to zero.

| weights | mesh | vertices | iterations | converged | energy | edge error | distance (RMS) | distance (max) | distance to same weights' 20 x 41 (RMS) | volume change |
|---|---|---|---|---|---|---|---|---|---|---|
| cotangent | 12 x 25 | 302 | 62 | yes | 0.0161 | 0.0653 | 0.0043 | 0.0083 | 0.0043 | -2.3% |
| cotangent | 20 x 41 | 822 | 124 | yes | 0.0148 | 0.0635 | 0 | 0 | 0 | -2.9% |
| cotangent | 16 x 81 (long, thin triangles) | 1298 | 251 | yes | 0.0147 | 0.0613 | 0.0025 | 0.0064 | 0.0025 | -2.5% |
| cotangent | 20 x 41 irregular | 822 | 114 | yes | 0.0156 | 0.0611 | 0.00092 | 0.0022 | 0.00092 | -2.8% |
| cotangent | 32 x 41 | 1314 | 350 | yes | 0.0134 | 0.0574 | 0.015 | 0.043 | 0.015 | -8.4% |
| cotangent | 32 x 81 | 2594 | 1000 | no | 0.0113 | 0.0442 | 0.032 | 0.055 | 0.032 | -8.8% |
| cotangent | 48 x 77 | 3698 | 1000 | no | 0.00925 | 0.0411 | 0.035 | 0.063 | 0.035 | -14.4% |
| uniform | 12 x 25 | 302 | 84 | yes | 0.0209 | 0.0635 | 0.0073 | 0.013 | 0.0028 | -1.1% |
| uniform | 20 x 41 | 822 | 102 | yes | 0.0191 | 0.0629 | 0.0062 | 0.0098 | 0 | -1.2% |
| uniform | 16 x 81 (long, thin triangles) | 1298 | 198 | yes | 0.00754 | 0.0636 | 0.0071 | 0.013 | 0.0043 | -0.1% |
| uniform | 20 x 41 irregular | 822 | 108 | yes | 0.019 | 0.0629 | 0.0061 | 0.01 | 0.00056 | -1.2% |
| uniform | 32 x 41 | 1314 | 328 | yes | 0.0258 | 0.053 | 0.018 | 0.043 | 0.019 | -10.9% |
| uniform | 32 x 81 | 2594 | 198 | yes | 0.0144 | 0.0615 | 0.005 | 0.0083 | 0.0021 | -1.1% |
| uniform | 48 x 77 | 3698 | 1000 | no | 0.0168 | 0.0428 | 0.026 | 0.064 | 0.027 | -11.8% |

## One handle vertex against a handle patch (bar, 90-degree bend)

The end is moved to the same place in every case. "end rotation" is the angle by which the top end of the bar turned.

| handle | handle vertices | iterations | energy | edge error | end rotation (degrees) | volume change |
|---|---|---|---|---|---|---|
| one vertex | 1 | 541 | 0.0113 | 0.048 | 65 | -5.2% |
| end patch moved | 41 | 196 | 0.0309 | 0.0718 | 1.21e-06 | -10.9% |
| end patch moved and turned | 41 | 128 | 0.0156 | 0.065 | 90 | -2.6% |
