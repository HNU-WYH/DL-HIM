import numpy as np
import warnings
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt

debug = False

def build_diffusion_matrix_2d(x, k_x, mat_type="sparse"):
    """
    Build the FDM matrix of a diffusion operator -d(d k(x) u(x)).
    The matrix includes boundary rows/cols.

    for each inner grid point p = n_grid * i + j, we have
    -d( k(p)d u(p)) = (F_{p+1/2} - F_{p-1/2}) / (x_{p+1/2} - x_{p-1/2)
    where:  
        - F_{p+1/2} = k_{p+1/2}(u_{p+1}-u_p)/(x_{p+1}-x_p)
        - F_{p-1/2} = k_{p-1/2}(u_p-u_{p-1})/(x_p-x_{p-1})
        - k_{p+1/2} = (k_{p}+k_{p+1})/2
        - k_{p-1/2} = (k_{p}+k_{p-1})/2
    Args:
        x: (n, 2)
        k_x: (n, ) 
        mat_type: "sparse" or "dense"

    Returns:
        A (n, n)
    """
    n = x.shape[0]
    n_grid = int(np.sqrt(n))
    h = np.diff(x[1, :2])[0]
    x = np.asarray(x, dtype=float).reshape(n_grid, n_grid, 2)
    k_x = np.asarray(k_x, dtype=float).reshape(n_grid, n_grid)

    if mat_type.lower() == "sparse":
        #Ap = sp.lil_matrix((n, n))  # (n, n)
        Dx = sp.lil_matrix((n, n))  # (n, n)
        Dy = sp.lil_matrix((n, n))  # (n, n)
    else:
        Dx = np.zeros(shape=(n, n), dtype=float)  # (n, n)
        Dy = np.zeros(shape=(n, n), dtype=float)  # (n, n)

        
    # build dy(k dy)[ ]
    for i in range(1, n_grid-1):
        y_cur = x[i, :, 1]
        k_cur = k_x[i, :]
        k_cur = 0.5 * (k_cur[:-1] + k_cur[1:])
        for j in range(1, n_grid - 1):
            q = i * n_grid + j
            Dy[q,q-1] = -k_cur[j-1] / h**2
            Dy[q,q + 1] = -k_cur[j] / h**2
            Dy[q,q] = -Dy[q,q-1] - Dy[q,q+1]

    # build dx(k dx)[ ]
    for j in range(1, n_grid-1):
        x_cur = x[:, j, 0]
        k_cur = k_x[:, j]
        k_cur = 0.5 * (k_cur[:-1] + k_cur[1:])
        for i in range(1, n_grid - 1):
            q = i * n_grid + j
            Dx[q,q-n_grid] = -k_cur[i-1] / h**2
            Dx[q,q + n_grid] = -k_cur[i] / h**2
            Dx[q,q] = -Dx[q,q-n_grid] - Dx[q,q+n_grid]

    Ap = Dx + Dy

    """if debug:
        print('1/h^2', 1/(h**2))
        Ap_plot = Ap
        if mat_type.lower() == "sparse":
            Ap_plot = np.asarray(Ap.todense()) 
        plt.imshow(np.abs(Ap_plot), cmap='jet')
        plt.colorbar()
        plt.show()
        exit()"""

    return Ap


def numerical_derivative_2d(u, x):
    """
    use central differences to compute du/dx。
    Args:
        u: [n] 
        x: [n,2]
    """
    n_grid = int(np.sqrt(u.shape[0]))
    u = u.reshape(n_grid, n_grid)
    h = np.diff(x[1, :2])[0]
    result = np.gradient(u, h)
    result = np.hstack((result[0].reshape(-1,1), result[1].reshape(-1,1)))
    return result  


def solve_dirichlet_system_2d(A, f, bc_idx, inner_idx, u_bc=0.0, mat_type="sparse"):
    """
    solve the system use dense solver or sparse solver
    1. Apply the boundary conditions
    2. Use a direct solver to solve it
    """
    # for robustness
    mat_type = mat_type.lower()

    # apply the boundary condition
    A_ii, f_inner= apply_dirichlet_bc_2d(A, f, bc_idx, inner_idx, u_bc, mat_type)

    # direct solving
    u_inner = direct_solve(A_ii, f_inner, mat_type)

    return u_inner, A_ii, f_inner


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


def apply_dirichlet_bc_2d(A, f, bc_idx, inner_idx, u_bc=0.0, mat_type="sparse"):
    """
    Applying dirichlet boundary conditions on the matrix A and the right-hand side f
    with u(0) = u_left, u(1) = u_right
    return the revised (A, f) and the inner slice and boundary indices

    equivalent to define a lifting vector u_bc = [u_left, 0, ..., 0, u_right]
    the inhomogeneous solution u = u_0 + u_bc
    where u[inner] = u_0[inner], u[boundary] = u_bc[boundary]

    Args:
        A:
        f:
        u_left:
        u_right:
        mat_type:

    Returns:

    """
    f = f.copy()
    A_bc = A.copy()

    if type(u_bc) == float:
        u_bc = u_bc * np.ones(len(bc_idx))
    else:
        assert len(u_bc) == len(bc_idx)
        u_bc = np.asarray(u_bc, dtype=float)

    if mat_type.lower() == "sparse":
        A_csr = A_bc.tocsr()
        A_ii = A_csr[inner_idx.reshape(-1, 1), inner_idx.reshape(1, -1)]   # matrix of inner points
        A_ib = A_csr[inner_idx.reshape(-1, 1), bc_idx.reshape(1, -1)]  # matrix of inner points and boundary points
    else:
        A_ii = A_bc[inner_idx.reshape(-1,1), inner_idx.reshape(1,-1)]  # matrix of inner points
        A_ib = A_bc[inner_idx.reshape(-1,1), bc_idx.reshape(1,-1)]  # matrix of inner points and boundary points

    # f = A_ii u_ii + A_ib u_bc
    f_inner = f[inner_idx] - A_ib @ u_bc

    return A_ii, f_inner


def expand_solution(u_inner, u_bc, inner_idx, bc_idx):
    n = len(inner_idx) + len(bc_idx) 
    u_expand = np.zeros(n)

    u_expand[inner_idx] = u_inner
    u_expand[bc_idx] = u_bc
    return u_expand
