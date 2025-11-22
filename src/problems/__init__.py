from .diffusion1d import Diffusion1D
from .helmholtz1d import Helmholtz1D
from .convdiff1d import ConvectionDiffusion1D


# ===============================================
# PDE Problem
# Define different PDE Problem
# ===============================================
Problems = {
    "poisson":      Diffusion1D,
    "diffusion":    Diffusion1D,
    "helmholtz":    Helmholtz1D,
    "convdiff":     ConvectionDiffusion1D,
}


def create_problem(name):
    if name not in Problems:
        raise ValueError(f"Unknown problem type: {name}")
    return Problems[name]