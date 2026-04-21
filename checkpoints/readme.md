# Model Checkpoints

This folder contains trained model checkpoints for the 1D diffusion and helmholtz equations.

## Directory and File Convention

Checkpoints are organized in the following hierarchy:

```
<model_name>/<framework>_<loss>_<norm>/<equation>_<nDim>D_Grid<XX>_ep<epochs>.pt
```

- **<model_name>**: The model architecture and problem identifier (e.g., `deeponet_diffusion1d`, `fns_diffusion1d_fno`, `fns_diffusion1d_unet`).
- **<framework>**: Training framework (`dynamic` or `static`).
- **<loss>**: Training target (`error` or `residual`).
- **<norm>**: Norm used for loss (`l1`, `l2`, or `h1`).
- **<equation>**: Type of PDE (`diffusion` or `helmholtz`).
- **<nDim>**: Spatial dimensionality (`1`).
- **Grid<XX>**: Number of grid points in the FDM mesh (e.g., `Grid31`).
- **ep<epochs>**: Number of training epochs completed (e.g., `ep20000`).

## Examples

- `deeponet_diffusion1d/dynamic_error_h1/diffusion_1D_Grid31_ep20000.pt`
- `fns_diffusion1d_fno/static_residual_l2/diffusion_1D_Grid31_ep101.pt`
- `deeponet_helmholtz1d/dynamic_error_l2/helmholtz_1D_Grid31_ep20000.pt`