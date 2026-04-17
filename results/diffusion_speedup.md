# Speedup Analysis (baseline: Jacobi)

## Error-based Speedup

| Threshold | Jacobi | Gauss-Seidel | HINTS-Fixed (Jacobi) | HINTS-PAAA (Jacobi) | HINTS-Fixed (GS) | HINTS-PAAA (GS) |
|---|---|---|---|---|---|---|
| 1e+00 | 1.00x | 0.11x | 13.73x | 23.04x | 4.52x | 4.15x |
| 1e-01 | 1.00x | 0.11x | 189.06x | 317.11x | 62.22x | 57.19x |
| 1e-02 | 1.00x | 0.11x | 124.52x | 110.35x | 39.76x | 29.89x |
| 1e-03 | 1.00x | 0.11x | 103.94x | 115.50x | 30.28x | 29.09x |
| 1e-04 | 1.00x | 0.11x | 89.28x | 139.97x | 27.44x | 33.59x |
| 1e-05 | 1.00x | 0.11x | 80.87x | 176.07x | 25.99x | 37.06x |
| 1e-06 | — | — | — | — | — | — |

## Residual-based Speedup

| Threshold | Jacobi | Gauss-Seidel | HINTS-Fixed (Jacobi) | HINTS-PAAA (Jacobi) | HINTS-Fixed (GS) | HINTS-PAAA (GS) |
|---|---|---|---|---|---|---|
| 1e+00 | 1.00x | 0.11x | 37.47x | 60.59x | 13.12x | 16.41x |
| 1e-01 | 1.00x | 0.11x | 43.16x | 80.33x | 15.29x | 21.35x |
| 1e-02 | 1.00x | 0.11x | 46.72x | 107.28x | 16.32x | 25.75x |
| 1e-03 | 1.00x | 0.11x | 51.10x | 123.56x | 17.01x | 29.88x |
| 1e-04 | 1.00x | 0.11x | 52.75x | 122.22x | 17.90x | 33.24x |
| 1e-05 | — | — | — | — | — | — |
| 1e-06 | — | — | — | — | — | — |
