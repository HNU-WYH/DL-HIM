# Dynamic vs Static — Iterations to Threshold (baseline: Static)

## Error-based

| Threshold | Static | Dynamic (K=10) |
|---|---|---|
| 1e+00 | 20 iters | 20 iters (1.00x) |
| 1e-01 | 40 iters | 40 iters (1.00x) |
| 1e-02 | 200 iters | 200 iters (1.00x) |
| 1e-03 | 260 iters | 240 iters (1.08x) |
| 1e-04 | 320 iters | 260 iters (1.23x) |
| 1e-05 | 380 iters | 320 iters (1.19x) |
| 1e-06 | 460 iters | 380 iters (1.21x) |

## Residual-based

| Threshold | Static | Dynamic (K=10) |
|---|---|---|
| 1e+00 | 207 iters | 180 iters (1.15x) |
| 1e-01 | 280 iters | 228 iters (1.23x) |
| 1e-02 | 360 iters | 280 iters (1.29x) |
| 1e-03 | 437 iters | 340 iters (1.29x) |
| 1e-04 | 540 iters | 400 iters (1.35x) |
| 1e-05 | 628 iters | 442 iters (1.42x) |
| 1e-06 | 716 iters | 500 iters (1.43x) |

## Console Info

================================================================================
Start training: trainer=static, loss_type=error, loss_norm=l2.
================================================================================
Epoch [0/101], Train Loss:  1.0001e+00, Val Loss:  3.3412e-03
[Checkpoint] saved → ./checkpoints/fns_diffusion1d/static_error_l2\diffusion_1D_Grid31_ep100.pt  (epoch=100, best_val=inf)
Epoch [100/101], Train Loss:  8.5328e-02, Val Loss:  9.8424e-06
==================================================
(Static) Time and Memory Usage Report
Total Wall-clock Time  : 13.9259 sec
Avg Time per Epoch     : 0.1379 sec
 Peak GPU Memory       : 110.73 MB
==================================================
Finished training.
Final train loss: 8.5328e-02
Final val loss:   9.8424e-06
Model successfully saved to ./checkpoints/fns_diffusion1d/static_error_l2\diffusion_1D_Grid31_ep101.pt
Model and Loss saved to ./checkpoints/fns_diffusion1d/static_error_l2
Cleaning up memory for trainer type: static...
Static Trainer Memory reset complete.
Memory cleanup finished.


================================================================================
Start training: trainer=dynamic, loss_type=error, loss_norm=l2.
================================================================================
Epoch [0/101], Train Loss:  1.0000e+00, Val Loss:  3.3412e-03, Horizon: 1
[Checkpoint] saved → ./checkpoints/fns_diffusion1d/dynamic_error_l2\diffusion_1D_Grid31_ep100.pt  (epoch=100, best_val=inf)
Epoch [100/101], Train Loss:  1.0905e-02, Val Loss:  1.4390e-05, Horizon: 10
==================================================
 (Dynamic) Time and Memory Usage Report
 Horizon Setting       : K=10
 Total Wall-clock Time : 83.0404 sec
 Avg Time per Epoch    : 0.8222 sec
 Peak GPU Memory       : 944.50 MB
==================================================
Finished training.
Final train loss: 1.0905e-02
Final val loss:   1.4390e-05
Model successfully saved to ./checkpoints/fns_diffusion1d/dynamic_error_l2\diffusion_1D_Grid31_ep101.pt
Model and Loss saved to ./checkpoints/fns_diffusion1d/dynamic_error_l2
Cleaning up memory for trainer type: dynamic...
Dynamic Trainer Memory reset complete.
Memory cleanup finished.

