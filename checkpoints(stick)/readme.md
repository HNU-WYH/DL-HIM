# Model Checkpoints

This folder contains trained model checkpoints for the 1D diffusion and helmholtz equations.

## File naming convention

Files are named using the following pattern:

```
<architecture>_<equation>_<nDim>d/<framework>_<loss>_<norm>/<equation>_<nDim>D_Grid<XX>_Ep<epochs>_<time>.pt
```

- **<architecture>**: architecture of neural operators
- **<equation>**: type of PDE (`helmholtz` or `diffusion`)
- **<nDim>**: spatial dimensionality of PDEs (e.g., `1`)
- **<framework>**: training framework of neural operators(e.g. `dynamic`, `static`)
- **<loss>**: training target of neural operators (e.g. `error`, `residual`)
- **<norm>**: norm of loss (e.g. `l1`, `l2`, `h1`)
- **Grid<XX>**: number of grid points in the FDM mesh (e.g., `Grid31`)
- **Ep<epochs>**: number of training epochs (e.g., `Ep20000`)

## Examples

- `diffusion_1D_Grid31_Ep20000.pt`  → Diffusion equation, 1D, 31 grid points, trained for **20,000** epochs
- `helmholtz_1D_Grid31_Ep20000.pt`  → Helmholtz equation, 1D, 31 grid points, trained for **20,000** epochs