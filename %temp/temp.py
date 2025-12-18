import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt


import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import time


# ---------------- Anderson Acceleration（单向量版本，和你的一样逻辑） ----------------
class AndersonAcceleration:
    """
    Anderson acceleration for u_{k+1} = G(u_k)
    """
    def __init__(self, m=5, reg=1e-10):
        self.m = m
        self.reg = reg
        self.hist_u = []
        self.hist_f = []
        self.hist_diff_u = []
        self.hist_diff_f = []

    def reset(self):
        self.hist_u.clear()
        self.hist_f.clear()
        self.hist_diff_u.clear()
        self.hist_diff_f.clear()

    def compute(self, u_k: np.ndarray, G_u_k: np.ndarray):
        """
        输入: u_k, G(u_k)，形状可以是一维或多维，返回 update direction p_k
             使得: u_{k+1} = u_k + p_k
        """
        ori_size = u_k.shape
        f_k_raw = G_u_k - u_k

        # 统一成 (B, F)
        if u_k.ndim == 1:
            u_k = u_k[None, :]
            f_k = f_k_raw[None, :]
        else:
            B = u_k.shape[0]
            u_k = u_k.reshape(B, -1)
            f_k = f_k_raw.reshape(B, -1)

        if len(self.hist_u) == 0:
            self.hist_u.append(u_k)
            self.hist_f.append(f_k)
            return f_k_raw

        delta_u = u_k - self.hist_u[-1]
        delta_f = f_k - self.hist_f[-1]

        self.hist_u.append(u_k)
        self.hist_f.append(f_k)
        self.hist_diff_u.append(delta_u)
        self.hist_diff_f.append(delta_f)

        current_m = len(self.hist_diff_f)
        if current_m > self.m:
            self.hist_u.pop(0)
            self.hist_f.pop(0)
            self.hist_diff_u.pop(0)
            self.hist_diff_f.pop(0)
            current_m -= 1

        Mat_F = np.stack(self.hist_diff_f, axis=-1)     # [B, F, m]
        Mat_U = np.stack(self.hist_diff_u, axis=-1)     # [B, F, m]
        Mat_F_T = Mat_F.transpose(0, 2, 1)              # [B, m, F]

        H = Mat_F_T @ Mat_F + self.reg * np.eye(current_m)[None, :, :]
        rhs = Mat_F_T @ f_k[..., None]                  # [B, m, 1]

        gamma = np.linalg.solve(H, rhs)                 # [B, m, 1]
        temp = (Mat_U + Mat_F) @ gamma                  # [B, F, 1]
        p_k = f_k - temp.squeeze(-1)                    # [B, F]

        return p_k.reshape(ori_size)


# ---------------- 1D Poisson 系统构造 ----------------
def get_poisson_system(n):
    """
    生成 1D Poisson: -u'' = f, x in (0,1), u(0)=u(1)=0
    使用均匀网格，n = 内点个数
    """
    h = 1.0 / (n + 1)
    main = 2.0 * np.ones(n) / h**2
    off = -1.0 * np.ones(n - 1) / h**2
    A = np.diag(main) + np.diag(off, 1) + np.diag(off, -1)

    # 造一个混合频率的真解，确保低频成分存在
    b = np.ones(n)
    u_true = np.linalg.solve(A, b)
    return A, b, u_true


# ---------------- Jacobi & Jacobi+AA ----------------
def jacobi_solve(A, b, x0=None, max_iter=1000, tol=1e-10, omega=2/3):
    """
    经典加权 Jacobi：x^{k+1} = x^k + omega * D^{-1}(b - A x^k)
    返回: x, residual_history（||b - A x_k||_2）
    """
    n = A.shape[0]
    if x0 is None:
        x = np.zeros(n)
    else:
        x = x0.copy()

    if sp.issparse(A):
        D = A.diagonal()
        def matvec_A(v): return A @ v
    else:
        A = np.asarray(A)
        D = np.diag(A)
        def matvec_A(v): return A.dot(v)

    D_inv = np.reciprocal(D, where=np.abs(D) > 1e-14)

    res_hist = []
    r = b - matvec_A(x)
    res_hist.append(np.linalg.norm(r))

    for k in range(max_iter):
        if res_hist[-1] < tol:
            break
        r = b - matvec_A(x)
        x = x + omega * (D_inv * r)
        r_new = b - matvec_A(x)
        res_hist.append(np.linalg.norm(r_new))

    return x, res_hist


def jacobi_aa_solve(A, b, x0=None, max_iter=1000, tol=1e-10,
                    omega=2/3, m_aa=20, reg=1e-10):
    """
    Jacobi + Anderson Acceleration：
        G(x) = x + omega * D^{-1}(b - A x)
        每一步: x_{k+1} = x_k + p_k,  p_k = AA.compute(x_k, G(x_k))
    """
    n = A.shape[0]
    if x0 is None:
        x = np.zeros(n)
    else:
        x = x0.copy()

    if sp.issparse(A):
        D = A.diagonal()
        def matvec_A(v): return A @ v
    else:
        A = np.asarray(A)
        D = np.diag(A)
        def matvec_A(v): return A.dot(v)

    D_inv = np.reciprocal(D, where=np.abs(D) > 1e-14)

    def G(x_vec):
        r = b - matvec_A(x_vec)
        return x_vec + omega * (D_inv * r)

    aa = AndersonAcceleration(m=m_aa, reg=reg)

    res_hist = []
    r = b - matvec_A(x)
    res_hist.append(np.linalg.norm(r))

    for k in range(max_iter):
        if res_hist[-1] < tol:
            break

        Gx = G(x)
        p_k = aa.compute(x, Gx)
        x = x + p_k

        r_new = b - matvec_A(x)
        res_hist.append(np.linalg.norm(r_new))

    return x, res_hist


# ---------------- GMRES（统一 true residual） ----------------
def solve_gmres_flexible(A, b, x0=None, restart=20, max_iter=1000,
                         tol=1e-10, use_jacobi=False):
    N = A.shape[0]

    # A 封装成 LinearOperator
    if sp.issparse(A):
        def matvec_A(v): return A @ v
    else:
        A = np.asarray(A)
        def matvec_A(v): return A.dot(v)

    A_op = spla.LinearOperator((N, N), matvec=matvec_A)

    # Jacobi 预条件 M^{-1}
    M_op = None
    if use_jacobi:
        if sp.issparse(A):
            diag = A.diagonal()
        else:
            diag = np.diag(A)
        diag_inv = np.reciprocal(diag, where=np.abs(diag) > 1e-14)

        def matvec_M(v): return diag_inv * v
        M_op = spla.LinearOperator((N, N), matvec=matvec_M)

    # callback: 拿到当前 x_k，算 ||b - A x_k||
    res_hist = []

    def callback_x(xk):
        rk = b - matvec_A(xk)
        res_hist.append(np.linalg.norm(rk))

    # restart / max_iter 解释：max_iter = restart 循环次数
    if restart is None or restart > N:
        restart_val = N
    else:
        restart_val = restart
    max_restart_cycles = max_iter

    # 初始解
    if x0 is None:
        x0 = np.zeros_like(b)
    r0 = b - matvec_A(x0)
    res_hist.append(np.linalg.norm(r0))

    start_t = time.time()
    x, info = spla.gmres(
        A_op, b, x0=x0,
        restart=restart_val,
        maxiter=max_restart_cycles,
        rtol=tol, atol=0.0,
        M=M_op,
        callback=callback_x,
        callback_type='x',    # callback 收到 x_k
    )
    elapsed = time.time() - start_t

    print(f"GMRES(restart={restart_val}, Jacobi={use_jacobi}): "
          f"Final ||b-Ax||={res_hist[-1]:.2e}, "
          f"Steps(recorded)={len(res_hist)-1}, info={info}, "
          f"Time={elapsed:.4f}s")

    return x, res_hist


# ---------------- CG（同样用 true residual） ----------------
def solve_cg(A, b, x0=None, max_iter=1000, tol=1e-10, use_jacobi=False):
    N = A.shape[0]

    if sp.issparse(A):
        def matvec_A(v): return A @ v
    else:
        A = np.asarray(A)
        def matvec_A(v): return A.dot(v)

    A_op = spla.LinearOperator((N, N), matvec=matvec_A)

    M_op = None
    if use_jacobi:
        if sp.issparse(A):
            diag = A.diagonal()
        else:
            diag = np.diag(A)
        diag_inv = np.reciprocal(diag, where=np.abs(diag) > 1e-14)

        def matvec_M(v): return diag_inv * v
        M_op = spla.LinearOperator((N, N), matvec=matvec_M)

    res_hist = []

    # 初始解
    if x0 is None:
        x0 = np.zeros_like(b)
    r0 = b - matvec_A(x0)
    res_hist.append(np.linalg.norm(r0))

    def callback_x(xk):
        rk = b - matvec_A(xk)
        res_hist.append(np.linalg.norm(rk))

    start_t = time.time()
    x, info = spla.cg(
        A_op, b, x0=x0,
        rtol=tol, atol=0.0,
        maxiter=max_iter,
        M=M_op,
        callback=callback_x
    )
    elapsed = time.time() - start_t

    print(f"CG(Jacobi={use_jacobi}): "
          f"Final ||b-Ax||={res_hist[-1]:.2e}, "
          f"Steps(recorded)={len(res_hist)-1}, info={info}, "
          f"Time={elapsed:.4f}s")

    return x, res_hist


# ---------------- 主程序：跑一遍对比 ----------------
if __name__ == "__main__":
    N = 199
    M = 99
    MAX_ITER = 200


    A, b, u_true = get_poisson_system(N)
    x0 = np.zeros_like(b)

    # 1. 纯 Jacobi
    x_jac, res_jac = jacobi_solve(A, b, x0=x0, max_iter=MAX_ITER, omega=2/3)

    # 2. Jacobi + AA(m=M)
    x_aa, res_aa = jacobi_aa_solve(A, b, x0=x0, max_iter=MAX_ITER,
                                   omega=2/3, m_aa=M, reg=1e-15)

    # 3. CG + Jacobi
    x_cg, res_cg = solve_cg(A, b, x0=x0, max_iter=MAX_ITER,
                            tol=1e-15, use_jacobi=True)

    # 4. GMRES(restart=M) + Jacobi
    x_gm, res_gm = solve_gmres_flexible(A, b, x0=x0,
                                        restart=M, max_iter=MAX_ITER,
                                        tol=1e-15, use_jacobi=True)

    plt.figure(figsize=(10, 6))
    plt.semilogy(res_jac, label="Jacobi (ω=2/3)")
    plt.semilogy(res_cg, label="CG + Jacobi")
    plt.semilogy(res_aa, label=f"Jacobi(ω=2/3) + AA(m={M})")
    plt.semilogy(res_gm, label=f"GMRES({M}) + Jacobi")

    plt.title(f"1D Poisson, N={N}: Residual History (||b - A x_k||_2)")
    plt.xlabel("Iteration (or restart index)")
    plt.ylabel("Residual L2 Norm")
    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()
    plt.show()
