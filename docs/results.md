# Experiment summary

Produced by `python -m experiments.run_all`. Edge error is the RMS of the relative edge-length change; distances are divided by the bounding-box diagonal. "Iterations to within 1%" counts the iterations until the energy is within 1% of its final value.

## Deformations

| scenario | vertices | constrained | iterations | converged | energy never rose | iterations to within 1% | edge error Laplacian | edge error ARAP | setup (s) | ms / iteration |
|---|---|---|---|---|---|---|---|---|---|---|
| bar_bend | 822 | 82 | 128 | yes | yes | 31 | 0.191 | 0.065 | 0.00537 | 3.18 |
| bar_twist | 822 | 82 | 162 | yes | yes | 70 | 0.175 | 0.0764 | 0.00475 | 3.79 |
| hinge | 6002 | 1202 | 132 | yes | yes | 108 | 0.669 | 0.011 | 0.0239 | 23.4 |
| bunny_ears | 2503 | 881 | 155 | yes | yes | 22 | 0.0679 | 0.0362 | 0.0107 | 10.4 |
| kid_arms | 8441 | 724 | 862 | yes | yes | 90 | 0.201 | 0.0406 | 0.0453 | 37.5 |
| camel_drink | 9770 | 3381 | 797 | yes | yes | 96 | 0.1 | 0.0168 | 0.0516 | 45.2 |

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
| bar_bend | laplacian | 128 | 31 | 0.216 | 0.0156 | 0 | 0.409 |
| bar_bend | thin_plate | 118 | 14 | 0.114 | 0.0156 | 0.000422 | 0.423 |
| bar_bend | rest | 142 | 38 | 22.5 | 0.0156 | 5.02e-05 | 0.47 |
| hinge | laplacian | 132 | 108 | 2.79 | 0.000599 | 0 | 3.5 |
| hinge | thin_plate | 93 | 65 | 0.474 | 0.000596 | 0.000586 | 2.36 |
| hinge | rest | 227 | 201 | 335 | 0.000598 | 0.000438 | 5.44 |
| camel_drink | laplacian | 797 | 96 | 0.178 | 0.0102 | 0 | 34.3 |
| camel_drink | thin_plate | 781 | 146 | 0.165 | 0.0102 | 0.000802 | 34.7 |
| camel_drink | rest | 798 | 97 | 23.1 | 0.0102 | 2.78e-06 | 34.3 |

## Timing (bar bend, 20 iterations)

Medians of repeated runs. The dense global step is one row-replaced `numpy.linalg.solve` with GeomProc's matrix, as in the class demos.

| vertices | setup (s) | local step (ms) | global step (ms) | dense global step (ms) | GeomProc Laplacian build (s) |
|---|---|---|---|---|---|
| 302 | 0.00242 | 0.936 | 0.151 | 1.91 | 0.117 |
| 822 | 0.00442 | 2.81 | 0.429 | 12.6 | 0.343 |
| 2594 | 0.00948 | 8.06 | 1.23 | 181 | 1.05 |
| 5810 | 0.0186 | 17.5 | 2.74 | 1.18e+03 | 2.54 |
| 10306 | 0.0477 | 34.6 | 5.86 | - | - |
| 23138 | 0.101 | 73.7 | 13.1 | - | - |

## Agreement with GeomProc's operator and with libigl

Largest distance between corresponding vertices, divided by the bounding-box diagonal. The libigl column is empty when libigl is not installed.

| scenario | membrane, GeomProc operator | one ARAP step, GeomProc operator | ten ARAP iterations, GeomProc operator | one step, GeomProc operator, inconsistent b | one step, area-weighted, consistent b | one step, area-weighted, inconsistent b | libigl, ten iterations |
|---|---|---|---|---|---|---|---|
| bar_bend | 7e-15 | 6.9e-15 | 2.1e-14 | 0.34 | 2.1e-12 | 0.11 | 9.2e-06 |
| bunny_ears | 1.1e-15 | 1.5e-15 | 5.5e-15 | 0.68 | 4.5e-13 | 0.24 | 5.8e-05 |
