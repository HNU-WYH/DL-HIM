import numpy as np
from src.utils.gen1d_util import generate_x_nodes_1d


# =============================================================================
# 2-D Generation Utilities
# =============================================================================

def generate_xy_nodes(grid_type, num_points_x, num_points_y):
    """
    Generate two 1D node vectors for the x and y directions.

    Returns:
        x: (num_points_x, )
        y: (num_points_y, )
    """
    x = generate_x_nodes_1d(grid_type, num_points_x)
    y = generate_x_nodes_1d(grid_type, num_points_y)
    return x, y


def grf_generate_2d(x_nodes, y_nodes, func_num, sigma, l0, mean=0.0, minimal=None, **kwargs):
    """
    Gaussian Random Field on a 2D tensor-product grid using a separable kernel.

        C_{x,y} = σ^2 * K_x ⊗ K_y
                = σ^2 * (Lx @ Lx^T) ⊗ (Ly @ Ly^T)
                = σ^2 * (Ly ⊗ Lx) @ (Ly ⊗ Lx)^T

        ⇒ f = u + (Ly ⊗ Lx)z = Lx @ Z @ Ly^T where z = Vec(Z)

    Args:
        x_nodes:  (Nx, ); the x-axis coordinates.
        y_nodes:  (Ny, ); the y-axis coordinates.
        func_num: scalar; Number of independent random fields to generate.
        sigma:    list; Standard deviations for multi-scale superposition.
        l0:       list; Length scales for multi-scale superposition.
        mean:     float; Mean value of the field. Defaults to 0.0.
        minimal:  float; Minimum value threshold to filter samples.

    Returns:
        grfs: (func_num, Nx, Ny)
    """
    # Compute Kernel X & Y
    Nx, Ny = len(x_nodes), len(y_nodes)
    rx2 = (x_nodes - x_nodes[:, None]) ** 2
    ry2 = (y_nodes - y_nodes[:, None]) ** 2

    eps = 1e-10 # For Robust Decomposition
    sample_multiplier = 2 if minimal is not None else 1 # For Redundancy

    grfs_list, collected = [], 0
    while collected < func_num:
        sample_size = sample_multiplier * (func_num - collected)
        samples = np.zeros((sample_size, Nx, Ny), dtype=float)

        for s, l in zip(sigma, l0):
            Kx = np.exp(-rx2 / (2 * l ** 2))
            Ky = np.exp(-ry2 / (2 * l ** 2))
            Lx = np.linalg.cholesky(Kx + eps * np.eye(Nx))
            Ly = np.linalg.cholesky(Ky + eps * np.eye(Ny))

            for n in range(sample_size):
                Z = np.random.randn(Nx, Ny)
                samples[n] += float(s) * (Lx @ Z @ Ly.T)

        samples += float(mean)

        if minimal is not None:
            samples = samples[np.min(samples, axis=(1, 2)) > minimal]
            # nothing to append
            if len(samples) == 0:
                continue

        grfs_list.append(samples)
        collected += len(samples)

    return np.vstack(grfs_list)[:func_num]


def fixed_generate_2d(x_nodes, y_nodes, func_num, value=1.0, **kwargs):
    """Constant function on a 2D grid."""
    Nx, Ny = len(x_nodes), len(y_nodes)
    return np.full((func_num, Nx, Ny), float(value))


def gaussian_generate_2d(x_nodes, y_nodes, func_num, standard=True, mean=0.0, std=1.0, **kwargs):
    """
    IID Gaussian field on a 2D grid.

    Returns:
        samples: (func_num, Nx, Ny)
    """
    Nx, Ny = len(x_nodes), len(y_nodes)
    if standard:
        mean = 0.0
        std = 1.0
    return np.random.normal(loc=float(mean), scale=float(std), size=(func_num, Nx, Ny))


def fourier_generate_2d(x_nodes, y_nodes, func_num, freq_num=3, freq_upper=10, amplitude=0.5, **kwargs):
    """
    Random Fourier series on a 2D tensor-product grid.

    Returns:
        samples: (func_num, Nx, Ny)
    """
    Nx, Ny = len(x_nodes), len(y_nodes)
    XX, YY = np.meshgrid(x_nodes, y_nodes, indexing="ij")
    func = []

    for _ in range(func_num):
        f_xy = np.zeros((Nx, Ny))
        for _ in range(freq_num):
            n = np.random.randint(1, freq_upper)
            m = np.random.randint(1, freq_upper)
            a_nm = np.random.normal(0, amplitude)
            b_nm = np.random.normal(0, amplitude)
            f_xy += a_nm * np.cos(np.pi * n * XX) * np.cos(np.pi * m * YY)
            f_xy += b_nm * np.sin(np.pi * n * XX) * np.cos(np.pi * m * YY)
        func.append(f_xy)

    return np.array(func)
