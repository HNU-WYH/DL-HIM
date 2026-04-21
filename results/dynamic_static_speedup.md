# Dynamic vs Static — Iterations to Threshold (baseline: Static)

## Error-based

| Threshold | Static | Dynamic (K=10) |
|---|---|---|
| 1e+00 | 20 iters | 20 iters (1.00x) |
| 1e-01 | 40 iters | 20 iters (2.00x) |
| 1e-02 | 180 iters | 180 iters (1.00x) |
| 1e-03 | 480 iters | 380 iters (1.26x) |
| 1e-04 | 820 iters | 580 iters (1.41x) |
| 1e-05 | 920 iters | 780 iters (1.18x) |
| 1e-06 | 1140 iters | 980 iters (1.16x) |

## Residual-based

| Threshold | Static | Dynamic (K=10) |
|---|---|---|
| 1e+00 | 220 iters | 200 iters (1.10x) |
| 1e-01 | 520 iters | 395 iters (1.32x) |
| 1e-02 | 780 iters | 600 iters (1.30x) |
| 1e-03 | 975 iters | 804 iters (1.21x) |
| 1e-04 | 1160 iters | 1000 iters (1.16x) |
| 1e-05 | 1417 iters | 1200 iters (1.18x) |
| 1e-06 | 1660 iters | 1413 iters (1.17x) |
