from src.problems.diffusion1d import Diffusion1D
from src.problems.diffusion2d import Diffusion2D
from src.problems.helmholtz1d import Helmholtz1D
from src.problems.convdiff1d import ConvectionDiffusion1D


# ===============================================
# PDE Problem
# Define different PDE Problem
# ===============================================
Problems = {
    "poisson1d":      Diffusion1D,
    "diffusion1d":    Diffusion1D,
    "diffusion2d":    Diffusion2D,
    "helmholtz1d":    Helmholtz1D,
    "convdiff1d":     ConvectionDiffusion1D,
}


def create_problem(name: str = "Poisson", n_dim: int = 1):
    problem_name = name.lower()
    if problem_name.endswith("1d") or problem_name.endswith("2d"):
        lookup_name = problem_name
    else:
        lookup_name = f"{problem_name}{n_dim}d"

    if lookup_name not in Problems:
        raise ValueError(f"Unknown problem type: {lookup_name}")
    return Problems[lookup_name]
