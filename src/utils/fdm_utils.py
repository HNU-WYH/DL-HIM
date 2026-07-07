import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


# =============================================================================
# 1-D FDM Utilities
# =============================================================================

def build_diffusion_matrix_1d(x, k_x, mat_type="sparse"):
    """
    Build the FDM matrix of a diffusion operator -d(d k(x) u(x)), including boundary rows/cols.
    For each inner grid point i, we have

        -d( k(i) d u(i)) = (F_{i+1/2} - F_{i-1/2}) / (x_{i+1/2} - x_{i-1/2)

    where:
        - F_{i+1/2} = k_{i+1/2}(u_{i+1}-u_i)/(x_{i+1}-x_i)
        - F_{i-1/2} = k_{i-1/2}(u_i-u_{i-1})/(x_i-x_{i-1})
        - k_{i+1/2} = (k_{i}+k_{i+1})/2
        - k_{i-1/2} = (k_{i}+k_{i-1})/2
    Args:
        x: (n,); coordinates
        k_x: scalar, (n,) or (n-1,); parameter function
        mat_type: "sparse" or "dense"

    Returns:
        A (n, n)
    """
    n = len(x)
    if np.isscalar(k_x) or len(k_x) == 1:
        k_x = np.ones(n - 1) * k_x
    elif len(k_x) == n:
        k_x = 0.5 * (k_x[:-1] + k_x[1:])
    elif len(k_x) != n - 1:
        raise ValueError("k_x must be scalar, or have length len(x) or len(x)-1")

    if mat_type.lower() == "sparse":
        Ap = sp.lil_matrix((n, n))  # (n, n)
    else:
        Ap = np.zeros(shape=(n, n), dtype=float)  # (n, n)

    # whether the mesh is uniformly spaced
    h = np.diff(x)  # (n-1,)
    if np.allclose(h, h[0]):
        h = h[0]
        for i in range(1, n - 1):
            Ap[i, i - 1] = -k_x[i - 1] / h ** 2
            Ap[i, i + 1] = -k_x[i] / h ** 2
            Ap[i, i] = -Ap[i, i - 1] - Ap[i, i + 1]

    else:
        for i in range(1, n - 1):
            # Leftmost (0) & Rightmost (n-1) is the boundary
            h_mean = 0.5 * (h[i - 1] + h[i])

            # 3-point stencil
            c_left = k_x[i - 1] / (h_mean * h[i - 1])
            c_right = k_x[i] / (h_mean * h[i])

            Ap[i, i - 1] = -c_left
            Ap[i, i + 1] = -c_right
            Ap[i, i] = c_left + c_right

    return Ap


def build_convection_matrix_1d(x, k_x, mat_type="sparse"):
    """
        Build the FDM matrix of a convection-diffusion operator k(x)·du(x).
        The matrix includes boundary rows/cols.

        using the upwind form, we have:
        - if k(i)>0,  k(i)·du(x) =k[i] * (u[i] - u[i-1])/(x[i] - x[i-1])
        - if k(i)<0,  k(i)·du(x) =k[i] * (u[i+1] - u[i])/(x[i+1] - x[i])

        Args:
            x: (n,)
            k_x: scalar or (n,)
            mat_type: "sparse" or "dense"

        Returns:
            A (n, n)
    """
    n = len(x)
    if np.isscalar(k_x) or len(k_x) == 1:
        k_x = np.ones(n) * k_x
    elif n != len(k_x):
        raise ValueError("b_x must be scalar, or have length len(x)")

    if mat_type.lower() == "sparse":
        Ap = sp.lil_matrix((n, n))
    else:
        Ap = np.zeros((n, n), dtype=float)

    h = np.diff(x)
    if np.allclose(h, h[0]):
        h = h[0]
        for i in range(1, n - 1):
            if k_x[i] >= 0:
                Ap[i, i] += k_x[i] / h
                Ap[i, i - 1] += -k_x[i] / h
            else:
                Ap[i, i + 1] += k_x[i] / h
                Ap[i, i] += -k_x[i] / h
    else:
        for i in range(1, n - 1):
            if k_x[i] >= 0:
                Ap[i, i] += k_x[i] / h[i - 1]
                Ap[i, i - 1] += -k_x[i] / h[i - 1]
            else:
                Ap[i, i + 1] += k_x[i] / h[i]
                Ap[i, i] += -k_x[i] / h[i]

    return Ap


def build_reaction_matrix_1d(x, k_x, mat_type="sparse"):
    """
    build the diagonal matrix for zero-order term k(x)^2 u(x)
    Args:
        x: (n,)
        k_x: scalar or (n,)
        mat_type: "sparse" or "dense"

    Returns:
        A (n, n)

    """
    n = len(x)
    if np.isscalar(k_x) or len(k_x) == 1:
        k_x = np.ones(n) * k_x
    elif n != len(k_x):
        raise ValueError("k_x must be scalar, or have length len(x)")

    if mat_type.lower() == "sparse":
        Ap = sp.diags(np.power(k_x, 2), offsets=0, shape=(n, n), format="lil")
    else:
        Ap = np.diag(np.power(k_x, 2))
    return Ap


def numerical_derivative_1d(u, x):
    """
    use central differences to compute du/dx。
    Args:
        u: [n,] or [batch, n]
        x: [n,]
    """
    if u.ndim == 2:
        return np.stack([np.gradient(row, x) for row in u], axis=0)
    elif u.ndim == 1:
        return np.gradient(u, x)
    else:
        raise ValueError("Unsupported dimension for numerical_derivative_1d")


def solve_dirichlet_system_1d(A, f, u_left=0.0, u_right=0.0, mat_type="sparse"):
    """
    solve the system use dense solver or sparse solver
    1. Apply the boundary conditions
    2. Use a direct solver to solve it
    """
    # for robustness
    mat_type = mat_type.lower()

    # apply the boundary condition
    A_ii, f_inner, inner, bc_idx = apply_dirichlet_bc_1d(A, f, u_left, u_right, mat_type)

    # direct solving
    u_inner = direct_solve(A_ii, f_inner, mat_type)

    return u_inner, A_ii, f_inner, inner, bc_idx


def direct_solve(A, b, mat_type="sparse"):
    """
    Directly solve Ax = b
    Args:
        A: (N, N) matrix
        b: (N,) vector
        mat_type: "sparse" or "dense"
    Returns:
        x: (N,) vector
    """
    if mat_type.lower() == "sparse":
        return spla.spsolve(A, b)
    else:
        return np.linalg.solve(A, b)


def apply_dirichlet_bc_1d(A, f, u_left=0.0, u_right=0.0, mat_type="sparse"):
    """
    Applying dirichlet boundary conditions on the matrix A and the right-hand side f
    with u(0) = u_left, u(1) = u_right,

        [ A_ii,  A_ib ]    [ u_inner]   [ f_inner ]
        [ A_bi,  A_bb ]  · [  u_bc  ] = [   f_bc  ]

    Then we have:  A_ii @ u_inner = f_full - A_ib @ u_bc

    Args:
        A:        (n, n)
        f:        (n, )
        u_left:   scalar
        u_right:  scalar
        mat_type: "sparse" or "dense"

    Returns:
        the revised (A, f) and the inner slice and boundary indices
    """
    n = A.shape[0]
    inner = slice(1, -1)
    bc_idx = (0, n-1)

    f = f.copy()
    A_bc = A.copy()
    u_bc = np.asarray([u_left, u_right], dtype=float)

    if mat_type.lower() == "sparse":
        A_csr = A_bc.tocsr()
        A_ii = A_csr[inner, inner]   # matrix of inner points
        A_ib = A_csr[inner, bc_idx]  # matrix of inner points and boundary points
    else:
        A_ii = A_bc[inner, inner]  # matrix of inner points
        A_ib = A_bc[inner, bc_idx]  # matrix of inner points and boundary points

    # f = A_ii u_ii + A_ib u_bc
    f_inner = f[inner] - A_ib @ u_bc

    return A_ii, f_inner, inner, bc_idx


def expand_solution(u_inner, u_left=0.0, u_right=0.0):
    """
    expand the solution with given boundary values
    """
    n = len(u_inner) + 2
    u_expand = np.zeros(n)

    u_expand[1:-1] = u_inner
    u_expand[0], u_expand[-1] = u_left, u_right
    return u_expand


# =============================================================================
# 2-D FDM Utilities
# =============================================================================

def build_diffusion_matrix_2d(x, y, k_xy, mat_type="sparse"):
    """
    Build the FDM matrix for the 2D diffusion operator -∇·(k·∇u) via a 5-point stencil.
    Boundary rows/cols are included in the generated matrix, which are left as zeros.
    Node ordering: row-major, idx = i * Ny + j.

    The PDE is split into x and y directions:
        -∇·(k ∇u) = - [∂/∂x(k ∂u/∂x) + ∂/∂y(k ∂u/∂y)] = 0

    For a grid point (i, j), we have :
        -∇·(k·∇u)_{i,j} = - [ (F_{i+1/2, j} - F_{i-1/2, j}) / Δx_{i±1/2}
                             + (G_{i, j+1/2} - G_{i, j-1/2}) / Δy_{j±1/2} ]
        Where:
            1. F_{i+1/2, j} = k_{i+1/2, j} * (u_{i+1, j} - u_{i, j}) / (x_{i+1} - x_i)
            2. F_{i-1/2, j} = k_{i-1/2, j} * (u_{i, j} - u_{i-1, j}) / (x_i - x_{i-1})
            3. Δx_{i±1/2}   = x_{i+1/2} - x_{i-1/2} = (x_{i+1} - x_{i-1}) / 2
            4. G_{i, j+1/2} = k_{i, j+1/2} * (u_{i, j+1} - u_{i, j}) / (y_{j+1} - y_j)
            5. G_{i, j-1/2} = k_{i, j-1/2} * (u_{i, j} - u_{i, j-1}) / (y_j - y_{j-1})
            6. Δy_{j±1/2}   = y_{j+1/2} - y_{j-1/2} = (y_{j+1} - y_{j-1}) / 2

    Which yields the following 5-point stencil weights for node (i, j):

                 [       0                -C_{i, j+1}              0       ]
       Stencil = [  -C_{i-1, j}             C_{i, j}          -C_{i+1, j}  ]
                 [       0                -C_{i, j-1}              0       ]

       Where the coefficients (C) extracted from the fluxes are:
            - C_{i+1, j} = k_{i+1/2, j} / [ (x_{i+1} - x_i) * Δx_{i±1/2} ]  (Right)
            - C_{i-1, j} = k_{i-1/2, j} / [ (x_i - x_{i-1}) * Δx_{i±1/2} ]  (Left)
            - C_{i, j+1} = k_{i, j+1/2} / [ (y_{j+1} - y_j) * Δy_{j±1/2} ]  (Top)
            - C_{i, j-1} = k_{i, j-1/2} / [ (y_j - y_{j-1}) * Δy_{j±1/2} ]  (Bottom)
            - C_{i, j}   = C_{i+1, j} + C_{i-1, j} + C_{i, j+1} + C_{i, j-1} (Center)

    Args:
        x: (Nx,) the nodes of x
        y: (Ny,) the nodes of y
        k_xy: (Nx, Ny) values of k(x,y)
        mat_type: "sparse" or "dense"

    Returns:
        A: (Nx*Ny, Nx*Ny)
    """
    # Initialization
    Nx, Ny = len(x), len(y)

    N = Nx * Ny
    if mat_type.lower() == "sparse":
        A = sp.lil_matrix((N, N))
    else:
        A = np.zeros((N, N))


    # compute the grid size
    dx = np.diff(x)
    dy = np.diff(y)
    dx_uniform = np.allclose(dx, dx[0])
    dy_uniform = np.allclose(dy, dy[0])
    hx = dx[0] if dx_uniform else dx
    hy = dy[0] if dy_uniform else dy

    # compute the coefficient function
    k = np.asarray(k_xy)
    if k.shape != (Nx, Ny):
        raise ValueError(f"k_xy shape {k.shape} does not match grid ({Nx}, {Ny})")

    # Half-point k values
    k_ip = 0.5 * (k[:-1, :] + k[1:, :])   # (Nx-1, Ny), x-direction interfaces
    k_jp = 0.5 * (k[:, :-1] + k[:, 1:])   # (Nx, Ny-1), y-direction interfaces

    # Apply the five point stencil
    for i in range(1, Nx - 1):
        for j in range(1, Ny - 1):
            idx = i * Ny + j
            if dx_uniform and dy_uniform:
                cx_plus  = k_ip[i, j] / (hx ** 2)
                cx_minus = k_ip[i - 1, j] / (hx ** 2)
                cy_plus  = k_jp[i, j] / (hy ** 2)
                cy_minus = k_jp[i, j - 1] / (hy ** 2)
            else:
                hx_mean  = 0.5 * (hx[i - 1] + hx[i])
                hy_mean  = 0.5 * (hy[j - 1] + hy[j])
                cx_plus  = k_ip[i, j] / (hx_mean * hx[i])
                cx_minus = k_ip[i - 1, j] / (hx_mean * hx[i - 1])
                cy_plus  = k_jp[i, j] / (hy_mean * hy[j])
                cy_minus = k_jp[i, j - 1] / (hy_mean * hy[j - 1])

            A[idx, idx] = cx_plus + cx_minus + cy_plus + cy_minus
            A[idx, (i + 1) * Ny + j] = -cx_plus
            A[idx, (i - 1) * Ny + j] = -cx_minus
            A[idx, i * Ny + (j + 1)] = -cy_plus
            A[idx, i * Ny + (j - 1)] = -cy_minus

    return A


def apply_dirichlet_bc_2d(A, f, mat_type="sparse", shape=None):
    """
    Apply homogeneous Dirichlet BC on all four sides.

    Args:
        A: (Nx*Ny, Nx*Ny) the full size matrix
        f: (Nx*Ny,) the full size rhs vector
        mat_type: "sparse" or "dense"
        shape: (Nx, Ny). If None, will try to infer from sqrt(N).

    Returns:
        A_ii, f_inner, inner_idx, bc_idx
    """
    # Shape set-up
    N = A.shape[0]
    if shape is None:
        Ny = int(round(np.sqrt(N)))
        Nx = N // Ny
        if Nx * Ny != N:
            raise ValueError("Cannot infer 2D grid shape from flat length. Please pass shape=(Nx, Ny).")
    else:
        Nx, Ny = shape

    # set the indices of boundary and inner grid points
    f = np.asarray(f).copy()
    inner_idx = [i * Ny + j for i in range(1, Nx - 1) for j in range(1, Ny - 1)]
    bc_idx = [idx for idx in range(N) if idx not in inner_idx]

    # set the reduced matrix A_ii
    if mat_type.lower() == "sparse":
        A_csr = A.tocsr()
        A_ii = A_csr[inner_idx, :][:, inner_idx]
        # A_ib = A_csr[inner_idx, :][:, bc_idx]
    else:
        A_ii = A[np.ix_(inner_idx, inner_idx)]
        # A_ib = A[np.ix_(inner_idx, bc_idx)]

    # Homogeneous BC: u_bc = 0
    # f_inner = f[inner] - A_ib @ 0 = f[inner]
    f_inner = f[inner_idx]
    return A_ii, f_inner, inner_idx, bc_idx


def solve_dirichlet_system_2d(A, f, mat_type="sparse", shape=None):
    """
    Convenience wrapper: build + apply_bc + direct_solve.
    """
    A_ii, f_inner, inner_idx, bc_idx = apply_dirichlet_bc_2d(A, f, mat_type=mat_type, shape=shape)
    u_inner = direct_solve(A_ii, f_inner, mat_type=mat_type)
    return u_inner, A_ii, f_inner, inner_idx, bc_idx


def expand_solution_2d(u_inner, Nx, Ny):
    """
    Expand interior solution back to full (Nx, Ny) grid with zero boundaries.

    Args:
        u_inner: ((Nx-2)*(Ny-2),) or (Nx-2, Ny-2)
        Nx, Ny: full grid sizes

    Returns:
        u_expand: (Nx, Ny)
    """
    u_expand = np.zeros((Nx, Ny))
    u_arr = np.asarray(u_inner)
    if u_arr.ndim == 1:
        u_arr = u_arr.reshape(Nx - 2, Ny - 2)
    u_expand[1:-1, 1:-1] = u_arr
    return u_expand

