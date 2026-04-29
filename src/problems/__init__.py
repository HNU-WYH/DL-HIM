from src.problems.diffusion1d import Diffusion1D
from src.problems.diffusion2d import Diffusion2D
from src.problems.helmholtz1d import Helmholtz1D
from src.problems.convdiff1d import ConvectionDiffusion1D


# ===============================================
# PDE Problem
# Define different PDE Problem
# ===============================================
Problems = {
    "poisson":      Diffusion1D,
    "diffusion":    Diffusion1D,
    "diffusion2d":  Diffusion2D,
    "helmholtz":    Helmholtz1D,
    "convdiff":     ConvectionDiffusion1D,
}


def create_problem(name: str = "Poisson"):
    problem_name = name.lower()
    if problem_name not in Problems:
        raise ValueError(f"Unknown problem type: {problem_name}")
    return Problems[problem_name]
