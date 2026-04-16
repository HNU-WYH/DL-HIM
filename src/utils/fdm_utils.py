import numpy as np
import warnings
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def build_diffusion_matrix_1d(x, k_x, mat_type="sparse"):
    """
    Build the FDM matrix of a diffusion operator -d(d k(x) u(x)).
    The matrix includes boundary rows/cols.

    for each inner grid point i, we have
    -d(d k(i) u(i)) = (F_{i+1/2} - F_{i-1/2}) / (x_{i+1/2} - x_{i-1/2)
    where:
        - F_{i+1/2} = k_{i+1/2}(u_{i+1}-u_i)/(x_{i+1}-x_i)
        - F_{i-1/2} = k_{i-1/2}(u_i-u_{i-1})/(x_i-x_{i-1})
        - k_{i+1/2} = (k_{i}+k_{i+1})/2
        - k_{i-1/2} = (k_{i}+k_{i-1})/2
    Args:
        x: (n,)
        k_x: (n,) or (1,) or (n-1,)
        mat_type: "sparse" or "dense"

    Returns:
        A (n, n)
    """
    n = len(x)
    if np.isscalar(k_x) or len(k_x) == 1:
        k_x = np.ones(n - 1) * k_x
    elif len(k_x) == n:
        k_x = 0.5 * (k_x[:-1] + k_x[1:])
        # warnings.warn("convert k_x from (n,) to (n-1,)")
    elif len(k_x) == n - 1:
        pass
    else:
        raise ValueError("k_x must be scalar, or have length len(x) or len(x)-1")

    if mat_type.lower() == "sparse":
        Ap = sp.lil_matrix((n, n))  # (n, n)
    else:
        Ap = np.zeros(shape=(n, n), dtype=float)  # (n, n)

    # whether the mesh is uniformly spaced
    h = np.diff(x)  # (n-1,)
    uniform = np.allclose(h, h[0])
    if uniform:
        h = h[0]
        for i in range(1, n - 1):
            Ap[i, i - 1] = -k_x[i - 1] / h ** 2
            Ap[i, i + 1] = -k_x[i] / h ** 2
            Ap[i, i] = -Ap[i, i - 1] - Ap[i, i + 1]

    else:
        for i in range(1, n - 1):
            # (0) & (n-1) is the boundary
            h_mean = 0.5 * (h[i - 1] + h[i])

            c_left = k_x[i - 1] / (h_mean * h[i - 1])
            c_right = k_x[i] / (h_mean * h[i])

            Ap[i, i - 1] = -c_left
            Ap[i, i + 1] = -c_right
            Ap[i, i] = c_left + c_right

    return Ap


def build_convection_matrix_1d(x, b_x, mat_type="sparse"):
    """
        Build the FDM matrix of a convection-diffusion operator b(x)·du(x).
        The matrix includes boundary rows/cols.

        using the upwind form, we have:
        - if b(i)>0,  b(i)·du(x) =b[i] * (u[i] - u[i-1])/(x[i] - x[i-1])
        - if b(i)<0,  b(i)·du(x) =b[i] * (u[i+1] - u[i])/(x[i+1] - x[i])

        Args:
            x: (n,)
            b_x: (n,)
            mat_type: "sparse" or "dense"

        Returns:
            A (n, n)
    """
    n = len(x)
    if np.isscalar(b_x) or len(b_x) == 1:
        b_x = np.ones(n) * b_x
    elif n == len(b_x):
        pass
    else:
        raise ValueError("b_x must be scalar, or have length len(x)")

    if mat_type.lower() == "sparse":
        Ap = sp.lil_matrix((n, n))
    else:
        Ap = np.zeros((n, n), dtype=float)

    h = np.diff(x)
    if np.allclose(h, h[0]):
        h = h[0]
        for i in range(1, n - 1):
            if b_x[i] >= 0:
                Ap[i, i] += b_x[i] / h
                Ap[i, i - 1] += -b_x[i] / h
            else:
                Ap[i, i + 1] += b_x[i] / h
                Ap[i, i] += -b_x[i] / h
    else:
        for i in range(1, n - 1):
            if b_x[i] >= 0:
                Ap[i, i] += b_x[i] / h[i - 1]
                Ap[i, i - 1] += -b_x[i] / h[i - 1]
            else:
                Ap[i, i + 1] += b_x[i] / h[i]
                Ap[i, i] += -b_x[i] / h[i]

    return Ap


def build_reaction_matrix_1d(x, k_x, mat_type="sparse"):
    """
    build the diagonal matrix for zero-order term k(x)^2 u(x)
    Args:
        x:
        k_x:
        mat_type:

    Returns:

    """
    n = len(x)
    if np.isscalar(k_x) or len(k_x) == 1:
        k_x = np.ones(n) * k_x
    elif n == len(k_x):
        pass
    else:
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
    n = len(u_inner) + 2
    u_expand = np.zeros(n)

    u_expand[1:-1] = u_inner
    u_expand[0], u_expand[-1] = u_left, u_right
    return u_expand

