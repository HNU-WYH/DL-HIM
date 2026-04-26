import numpy as np
from warnings import warn
from src.utils.fdm_utils import *


class Diffusion2D:
    """
    Solve the diffusion equation
        -d [k(x) du(x)] = f(x) in [0,1]
    with homogeneous boundary conditions:
        u(0) = u(1) = 0
    """

    def __init__(self, x_nodes, k_x, f_x, mat_type="sparse", **kwargs):
        """
        Args:
            x_nodes: uniform grid in [0,1]^2  shape: (n,2)
            k_x: parameter function                       shape: (n,)
            f_x: rhs function                             shape: (n,)
            mat_type: "sparse" or "dense"
        """
        self.x = np.asarray(x_nodes, dtype=float)
        self.f = np.asarray(f_x, dtype=float)
        self.mat_type = mat_type.lower()

        if np.isscalar(k_x) or len(k_x) == 1:
            self.k = np.ones_like(x_nodes) * k_x
        else:
            self.k = np.asarray(k_x, dtype=float)

        assert self.x.shape == self.f.shape == self.k.shape

        # Compute the global matrix A_global
        self.A_global = build_diffusion_matrix_1d(self.x, self.k, mat_type=self.mat_type)

        # Initialize variables
        self.A_ii, self.u_left, self.u_right = None, None, None
        self.f_inner, self.u_inner, self.du_dx = None, None, None

    def build_system(self, u_left=0.0, u_right=0.0, method="fdm"):
        """
            -d [k(x) du(x)] = f(x)
        """
        if method.lower() != 'fdm':
            raise NotImplementedError(f"{method} has not been implemented yet")

        # Apply dirichlet B.C.
        self.A_ii, self.f_inner, inner, bc_idx = apply_dirichlet_bc_1d(
            A=self.A_global, f=self.f, u_left=u_left, u_right=u_right, mat_type=self.mat_type
        )

        # Restoring the boundary conditions
        self.u_left, self.u_right = u_left, u_right

        return self.A_ii, self.f_inner, inner, bc_idx

    def solve(self, u_left=0.0, u_right=0.0, method="fdm"):
        """
            -d [k(x) du(x)] = f(x)
        """
        if method.lower() != 'fdm':
            raise NotImplementedError(f"{method} has not been implemented yet")

        if (self.A_ii is not None) and (u_left == self.u_left) and (u_right == self.u_right):
            # solver the inner system directly
            self.u_inner = direct_solve(self.A_ii, self.f_inner, mat_type=self.mat_type)
            inner = slice(1, -1)

        else:
            # solve the system with dirichlet b.c.
            self.u_inner, self.A_ii, self.f_inner, inner, bc_idx = solve_dirichlet_system_1d(
                self.A_global, self.f, u_left=u_left, u_right=u_right, mat_type=self.mat_type
            )

            # Restoring the boundary conditions
            self.u_left, self.u_right = u_left, u_right

        # compute the derivative
        self.du_dx = numerical_derivative_1d(expand_solution(self.u_inner, u_left, u_right), self.x)[inner]

        return {"u_inner":  self.u_inner,           # (n-2,)
                "du_inner": self.du_dx,             # (n-2)
                "A_inner":  self.A_ii,              # (n-2, n-2)
                "f_inner":  self.f_inner,           # (n-2,)
                "k_x":      self.k,                 # (n,)
                "x_inner":  self.x[inner],          # (n-2)
                }
