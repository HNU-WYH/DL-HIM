import numpy as np

from src.utils.fdm_utils import (
    build_diffusion_matrix_2d,
    apply_dirichlet_bc_2d,
    solve_dirichlet_system_2d,
)


class Diffusion2D:
    """
    Solve the 2D diffusion equation
        -∇·(k(x,y) ∇u) = f(x,y)  in [0,1]×[0,1]
    with homogeneous Dirichlet boundary conditions on all four sides.
    """

    def __init__(self, x_nodes, y_nodes, k_xy, f_xy, mat_type="sparse", **kwargs):
        """
        Args:
            x_nodes: (Nx,)
            y_nodes: (Ny,)
            k_xy:    (Nx, Ny)
            f_xy:    (Nx, Ny)
            mat_type: "sparse" or "dense"
        """
        self.x = np.asarray(x_nodes, dtype=float)
        self.y = np.asarray(y_nodes, dtype=float)
        self.k = np.asarray(k_xy, dtype=float)
        self.f = np.asarray(f_xy, dtype=float)
        self.mat_type = mat_type.lower()

        assert self.k.shape == self.f.shape == (len(self.x), len(self.y))

        self.Nx, self.Ny = len(self.x), len(self.y)
        self.A_global = build_diffusion_matrix_2d(self.x, self.y, self.k, mat_type=self.mat_type)

        self.A_ii = None
        self.f_inner = None
        self.u_inner = None

    def build_system(self, u_left=0.0, u_right=0.0, method="fdm"):
        """
        Apply Dirichlet BC and return the interior system.
        (u_left/u_right are placeholders for interface compatibility; 2D uses zero on all sides.)
        """
        if method.lower() != "fdm":
            raise NotImplementedError(f"{method} has not been implemented yet")

        self.A_ii, self.f_inner, inner_idx, bc_idx = apply_dirichlet_bc_2d(
            self.A_global, self.f.ravel(), mat_type=self.mat_type, shape=(self.Nx, self.Ny)
        )
        return self.A_ii, self.f_inner, inner_idx, bc_idx

    def solve(self, u_left=0.0, u_right=0.0, method="fdm"):
        """
        Solve the 2D diffusion system.

        Returns:
            dict with keys:
                u_inner:  (Mx, My)        interior solution
                A_inner:  sparse matrix   (Mx*My, Mx*My)
                f_inner:  (Mx*My,)        interior RHS
                k_x:      (Nx, Ny)        diffusion coefficient
        """
        if method.lower() != "fdm":
            raise NotImplementedError(f"{method} has not been implemented yet")

        self.u_inner, self.A_ii, self.f_inner, inner_idx, bc_idx = solve_dirichlet_system_2d(
            self.A_global, self.f.ravel(), mat_type=self.mat_type, shape=(self.Nx, self.Ny)
        )

        Mx, My = self.Nx - 2, self.Ny - 2
        u_inner_2d = self.u_inner.reshape(Mx, My)

        return {
            "u_inner": u_inner_2d,
            "A_inner": self.A_ii,
            "f_inner": self.f_inner,
            "k_x": self.k,
        }
