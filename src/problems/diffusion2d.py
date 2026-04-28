import numpy as np
from warnings import warn
from src.utils.fdm2d_utils import *


class Diffusion2D:
    """
    Solve the diffusion equation
        -d [k(x) du(x)] = f(x) in [0,1]
    with homogeneous boundary conditions:
        u(0) = u(1) = 0
    """

    def __init__(self, x_nodes, bc_idx, inner_idx, k_x, f_x, mat_type="sparse", **kwargs):
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
        self.k = np.asarray(k_x, dtype=float)
        self.bc_idx = bc_idx
        self.inner_idx = inner_idx

        assert self.x.shape[0] == self.f.shape[0] == self.k.shape[0]
        assert len(self.bc_idx) + len(self.inner_idx)== self.x.shape[0]

        # Compute the global matrix A_global
        self.A_global = build_diffusion_matrix_2d(self.x, self.k, mat_type=self.mat_type)

        # Initialize variables
        self.A_ii, self.u_bc = None, None
        self.f_inner, self.u_inner, self.du_dx = None, None, None

    def build_system(self, u_bc=0.0, method="fdm"):
        """
            -d [k(x) du(x)] = f(x)
        """
        if method.lower() != 'fdm':
            raise NotImplementedError(f"{method} has not been implemented yet")

        # Apply dirichlet B.C.
        self.A_ii, self.f_inner = apply_dirichlet_bc_2d(
            A=self.A_global, f=self.f, bc_idx=self.bc_idx, inner_idx=self.inner_idx, 
            u_bc=u_bc, mat_type=self.mat_type
        )

        # Restoring the boundary conditions
        self.u_bc = u_bc

        return self.A_ii, self.f_inner, self.inner_idx, self.bc_idx

    def solve(self, u_bc=0.0, method="fdm"):
        """
            -d [k(x) du(x)] = f(x)
        """
        if method.lower() != 'fdm':
            raise NotImplementedError(f"{method} has not been implemented yet")

        if (self.A_ii is not None) and np.all(u_bc == self.u_bc):
            # solver the inner system directly
            self.u_inner = direct_solve(self.A_ii, self.f_inner, mat_type=self.mat_type)

        else:
            # solve the system with dirichlet b.c.
            self.u_inner, self.A_ii, self.f_inner  = solve_dirichlet_system_2d(
                self.A_global, self.f, bc_idx = self.bc_idx, inner_idx = self.inner_idx,
                u_bc = u_bc, mat_type=self.mat_type
            )

            # Restoring the boundary conditions
            self.u_bc = u_bc 

        # compute the derivative
        self.du_dx = numerical_derivative_2d(
            expand_solution(
                self.u_inner, u_bc, self.inner_idx, self.bc_idx
            ), 
            self.x)[self.inner_idx, :]

        return {"u_inner":  self.u_inner,           # (n-2,)
                "du_inner": self.du_dx,             # (n-2)
                "A_inner":  self.A_ii,              # (n-2, n-2)
                "f_inner":  self.f_inner,           # (n-2,)
                "k_x":      self.k,                 # (n,)
                "x_inner":  self.x[self.inner_idx],          # (n-2)
                }
