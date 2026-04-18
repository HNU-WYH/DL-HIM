# Speedup Analysis (baseline: HINTS-Fixed (Jacobi))

## Error-based Speedup

| Threshold | HINTS-Fixed (Jacobi) | HINTS-AA (Jacobi) | HINTS-PAAA (Jacobi) | HINTS-ELS (Jacobi) |
|---|---|---|---|---|
| 1e+00 | 1.00x | 1.51x | 1.18x | 0.79x |
| 1e-01 | 1.00x | 1.14x | 0.96x | 0.69x |
| 1e-02 | 1.00x | 0.79x | 0.39x | 0.65x |
| 1e-03 | 1.00x | 1.44x | 1.00x | 0.60x |
| 1e-04 | 1.00x | 2.07x | 1.54x | 0.68x |
| 1e-05 | 1.00x | 2.31x | 1.90x | 0.78x |
| 1e-06 | — | — | — | — |

## Residual-based Speedup

| Threshold | HINTS-Fixed (Jacobi) | HINTS-AA (Jacobi) | HINTS-PAAA (Jacobi) | HINTS-ELS (Jacobi) |
|---|---|---|---|---|
| 1e+00 | 1.00x | 1.85x | 2.38x | 0.88x |
| 1e-01 | 1.00x | 2.26x | 2.63x | 0.92x |
| 1e-02 | — | — | — | — |
| 1e-03 | — | — | — | — |
| 1e-04 | — | — | — | — |
| 1e-05 | — | — | — | — |
| 1e-06 | — | — | — | — |
