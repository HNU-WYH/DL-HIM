import numpy as np


class AndersonAcceleration:
    """
    Anderson acceleration for u_{k+1} = G(u_k)
    """
    def __init__(self, m=5, reg=1e-10):
        self.m = m
        self.reg = reg

        # history of solutions and residuals
        # f_k = G(u_k) - u_k
        self.hist_u = []
        self.hist_f = []

        # difference of [u] and [f]
        # diff_u_k = u_k - u_{k-1}
        # diff_f_k = f_k - f_{k-1}
        self.hist_diff_u = []
        self.hist_diff_f = []

    def compute(self, u_k: np.ndarray, G_u_k: np.ndarray):
        """
        Input: the current u_k and G(u_k) with the shape compatibility:
            - Single input: (n,), (n, 1), (n, m, ...)
            - Batch input: (B, n), (B, n, 1), (B, H, W), ...
        Output: the update direction p_k where u_{k+1} = u_k + (alpha_k) · p_k
        """
        # keep the original input
        ori_size = u_k.shape

        # compute the residual
        f_k_raw = G_u_k - u_k

        # for shape compatibility
        # convert to [batch_size (B), flatten features (F)]
        if u_k.ndim == 1:
            # (n,) -> (1, n)
            u_k = u_k[None, :]
            f_k = f_k_raw[None, :]
        else:
            # Case: (B, n), (B, n, 1), (B, H, W) -> (B, flattened_features)
            batch_size = u_k.shape[0]
            u_k = u_k.reshape(batch_size, -1)
            f_k = f_k_raw.reshape(batch_size, -1)

        # for no prior information
        # return the update f_k = G(u_k) - u_k directly
        if len(self.hist_u) == 0:
            self.hist_u.append(u_k)
            self.hist_f.append(f_k)
            return f_k_raw

        # compute the difference
        delta_u = u_k - self.hist_u[-1]
        delta_f = f_k - self.hist_f[-1]

        # store the history
        self.hist_u.append(u_k)
        self.hist_f.append(f_k)
        self.hist_diff_u.append(delta_u)
        self.hist_diff_f.append(delta_f)

        # update the history when size > m+1
        # determine the future update direction based on (m+1) G(u)
        # namely, based on m past difference information
        current_m = len(self.hist_diff_f)
        if current_m > self.m:
            self.hist_u.pop(0)
            self.hist_f.pop(0)
            self.hist_diff_u.pop(0)
            self.hist_diff_f.pop(0)

        Mat_F = np.stack(self.hist_diff_f, axis=-1)                 # m of [B, F]s -> [B, F, m]
        Mat_U = np.stack(self.hist_diff_u, axis=-1)                 # m of [B, F]s -> [B, F, m]
        Mat_F_T = Mat_F.transpose(0, 2, 1)                          # [B, F, m] -> [B, m, F]

        H = Mat_F_T @ Mat_F + self.reg * np.eye(current_m)          # [B, m, F] @ [B, F, m] + [m, m] -> [B, m, m]
        rhs = Mat_F_T @ f_k[..., None]                              # [B, m, F] @ [B, F, 1] -> [B, m, 1]

        gamma = np.linalg.solve(H, rhs)                             # [B, m, 1]

        temp = (Mat_U + Mat_F) @ gamma                              # [B, F, m] @ [B, m, 1] -> [B, F, 1]
        p_k = f_k - temp.squeeze(-1)                                # [B, F] - [B, F] -> [B, F]
        return p_k.reshape(ori_size)                                # [B, F] -> original size

    def reset(self):
        self.hist_u = []
        self.hist_f = []
        self.hist_diff_u = []
        self.hist_diff_f = []

    def __bool__(self):
        return True

def adaptive_step_size_cg(a_mat: np.ndarray, p_k: np.ndarray, r_k: np.ndarray, eps: float = 1e-16):
    """
    Computes optimal step size for CG: alpha = (p.T @ r) / (p.T @ A @ p)

    Compatible Shapes:
    ------------------
    a_mat: [B, N, N] (Batched) OR [N, N] (Shared/Single)
    p_k:   [B, N, 1] / [B, N] (Batched) OR [N, 1] / [N,] (Single)
    r_k:   [B, N, 1] / [B, N] (Batched) OR [N, 1] / [N,] (Single)

    Returns:
    --------
    alpha_k: [B,] (if batched) OR Scalar (if single)
    """

    # 1. Define Helper Function
    def normalize_vec(vec):
        """
        Normalize all vectors to [B, N, 1]
        return the normalized vector and whether the input is single or not
        """
        # Case 1: [N] -> Single -> [1, N, 1]
        if vec.ndim == 1:
            return vec[None, :, None], True

        # Case 2: [N, 1] -> Single -> [1, N, 1]
        if vec.ndim == 2 and vec.shape[-1] == 1:
            return vec[None, ...], True

        # Case 3: [B, N] -> Batched -> [B, N, 1]
        # 判断依据：维度是2，但最后一维是 N (不是1)
        if vec.ndim == 2:
            return vec[..., None], False

        # Case 4: [B, N, 1] -> Batched -> Keep as is
        return vec, False

    # 2. normalize the input
    p_vec, is_single = normalize_vec(p_k)     # [B, N, 1]
    r_vec, _ = normalize_vec(r_k)               # [B, N, 1]
    p_vec_T = p_vec.transpose(0, 2, 1)          # [B, 1, N]

    # 3. Batch Matrix Multiplication for optimal step size
    # denominator: p^T A p
    denom = p_vec_T @ a_mat @ p_vec             # [B, 1, N] @ [(B,) N, N] @ [B, 1, N] -> [B, 1, 1]

    # numerator: p^T r
    numer = p_vec_T @ r_vec                     # [B, 1, N] @ [B, N, 1] -> [B, 1, 1]

    # compute Alpha
    alpha_k = numer / (denom + eps)             # [B, 1, 1]

    # 4. 还原输出形状
    if is_single:
        return alpha_k.item()                   # return scalar
    else:
        return alpha_k.squeeze()                # return array [B,]
