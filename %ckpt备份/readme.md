# Model Checkpoints

This folder contains trained model checkpoints for the diffusion and convection-diffusion equations in 1D and 2D.

## File naming convention

Files are named using the following pattern:

```
<equation>_<nDim>D_Grid<XX>_Ep<epochs>.pt
```

- **<equation>**: type of PDE (`convdiff` or `diffusion`)
- **<nDim>D**: spatial dimensionality (e.g., `1D`)
- **Grid<XX>**: number of grid points in the FDM mesh (e.g., `Grid31`)
- **Ep<epochs>**: number of training epochs (e.g., `Ep20000`)

## Examples

- `diffusion_1D_Grid31_Ep20000.pt`  → Diffusion equation, 1D, 31 grid points, trained for **20,000** epochs
- `convdiff_1D_Grid31_Ep20000.pt`  → Convection-diffusion equation, 1D, 31 grid points, trained for **20,000** epochs