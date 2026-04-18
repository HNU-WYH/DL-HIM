# Speedup Analysis (baseline: HINTS-Fixed (Jacobi))

## Error-based Speedup

| Threshold | HINTS-Fixed (Jacobi) | HINTS-PAAA (Jacobi) | HINTS-Fixed (GS) | HINTS-PAAA (GS) |
|---|---|---|---|---|
| 1e+00 | 1.00x | 1.42x | 0.75x | 0.73x |
| 1e-01 | 1.00x | 1.22x | 0.60x | 0.61x |
| 1e-02 | 1.00x | 0.73x | 0.51x | 0.39x |
| 1e-03 | 1.00x | 1.56x | 0.59x | 0.89x |
| 1e-04 | 1.00x | 2.54x | 0.68x | 1.41x |
| 1e-05 | 1.00x | 3.05x | 0.69x | 1.74x |
| 1e-06 | 1.00x | 3.43x | 0.69x | 1.99x |

## Residual-based Speedup

| Threshold | HINTS-Fixed (Jacobi) | HINTS-PAAA (Jacobi) | HINTS-Fixed (GS) | HINTS-PAAA (GS) |
|---|---|---|---|---|
| 1e+00 | 1.00x | 3.18x | 0.76x | 1.72x |
| 1e-01 | 1.00x | 3.74x | 0.77x | 2.14x |
| 1e-02 | 1.00x | 4.04x | 0.75x | 2.40x |
| 1e-03 | 1.00x | 4.29x | 0.74x | 2.60x |
| 1e-04 | 1.00x | 4.37x | 0.74x | 2.66x |
| 1e-05 | 1.00x | 4.35x | 0.73x | 2.60x |
| 1e-06 | 1.00x | 4.38x | 0.73x | 2.54x |
