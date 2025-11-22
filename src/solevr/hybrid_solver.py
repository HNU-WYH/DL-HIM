import time
import torch
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from box import Box
from tqdm import tqdm
import matplotlib.pyplot as plt


from src.utils.fdm_utils import build_diffusion_matrix_1d, solve_dirichlet_system_1d, expand_solution
from src.utils.stepin_utils import AndersonAcceleration, adaptive_step_size_cg
from src.neural_operator.base import NeuralOperatorBase
from src.neural_operator import create_no
from problems import create_problem


class HybridSolver:
    def __init__(self, cfg: Box, k_x: np.ndarray = None, f_x: np.ndarray = None,
                 x_nodes: np.ndarray = None, model: NeuralOperatorBase = None, cp_path: str = None):
        """
        Args:
            cfg: 配置 Box
            model: the class of corresponding neural operator
            k_x：
            f_x:
            cp_path: the checkpoint for the neural operator
        """
        self.config = cfg

        self.k_x = np.asarray(k_x)
        self.f_x = np.asarray(f_x)

        self.solver_type = cfg.solver.get("type", "Hybrid")
        self.relax_factor = cfg.solver.numerical.get("relaxation_factor", 0.66)
        self.smoother_type = cfg.solver.numerical.get("method", "Jacobi").title()

        # 1. initialize model with given checkpoints
        if cfg.solver.type in ["Hybrid", "Neural"]:
            self.operator_type = cfg.training.get("operator_type", "DeepONet")
            self._neural_init(model, cp_path)

            if self.solver_type == "Hybrid":
                self.hybrid_ratio = config.solver.hybrid.get("update_ratio", 20)
                self.neural_update_type = config.solver.hybrid.get("neural_update", "fixed")

        elif cfg.solver.type == "Numerical":
            print(f"Numerical Solver ({self.smoother_type} method) ")

        else:
            raise ValueError(f"Unsupported solver type: {self.solver_type}")

        self._assemble_system()

    def _neural_init(self, model: NeuralOperatorBase = None, cp_path = None):
        print(f"{self.solver_type.title()} Solver ({self.operator_type}) ")

        self.model = create_no(config) if model is None else model
        self.cp_path = self.config["load_path"] if cp_path is None else cp_path
        model.load_model(self.cp_path)

    def _assemble_system(self):




        self.x_nodes = np.linspace(0, 1, config.data.mesh.grid_num)
        self.k_x = np.asarray(model, dtype=np.float32)
        self.f_x = np.asarray(f_x, dtype=np.float32)

        # 2. 初始化数值部分 (矩阵组装)
        self._assemble_system()

        # 3. 初始化 DeepONet 部分
        self.model = None
        if model_path is not None:
            self._load_deeponet(model_path)

        # 4. 迭代器配置
        self.numerical_method = config.solver.numerical.get("method", "jacobi").lower()
        self.relax_factor = config.solver.numerical.get("relaxation_factor", 0.66)  # specifically for Jacobi
        self.hybrid_ratio = config.solver.hybrid.get("update_ratio", 20)

    def _assemble_system(self):
        """根据 Problem Type 组装 A 和 f (内部点)"""
        problem_type = self.config.problem.type.lower()

        if problem_type == "poisson" or problem_type == "diffusion":
            # - d/dx (k du/dx) = f
            self.A = build_diffusion_matrix_1d(self.x_nodes, self.k_x, mat_type="sparse")
        else:
            # 可以在这里扩展 Helmholtz 等
            raise NotImplementedError(f"Problem {problem_type} not supported yet in this simple solver")

        # 应用边界条件 (Dirichlet u=0) 并提取内部矩阵
        # 注意：solve_dirichlet_system_1d 会返回 A_inner, f_inner
        # 这里我们为了迭代，需要显式持有 A_inner
        # 这里的 hack 是传入一个 dummy u_left/right 来利用 fdm_utils 的切片逻辑
        _, self.A_inner, self.f_inner, self.inner_slice, _ = solve_dirichlet_system_1d(
            self.A, self.f_x, u_left=0.0, u_right=0.0, mat_type="sparse"
        )

        # 预计算 Jacobi 需要的 D_inv
        if sparse.isspmatrix(self.A_inner):
            diag = self.A_inner.diagonal()
        else:
            diag = np.diag(self.A_inner)

        with np.errstate(divide='ignore'):
            self.D_inv = 1.0 / diag

    def _load_deeponet(self, model_path):
        """加载 DeepONet"""
        self.model = DeepONet1d(self.config)

        self.model.eval()  # Set to evaluation mode

    def compute_residual(self, u_inner):
        """计算残差 r = f - A u"""
        return self.f_inner - self.A_inner @ u_inner

    def _numerical_step(self, u_prev):
        """
        执行一步数值迭代 (Damped Jacobi)
        u_{k+1} = u_k + omega * D^{-1} * (f - A u_k)
        """
        if self.numerical_method == "jacobi":
            residual = self.compute_residual(u_prev)
            delta = self.relax_factor * (self.D_inv * residual)
            return u_prev + delta
        else:
            raise NotImplementedError("Currently only 'jacobi' is optimized.")

    def _neural_step(self, u_prev, step_size=1.0):
        """
        执行一步 Neural 修正
        1. 计算残差 r = f - A u
        2. DeepONet 预测 delta_u = G(k, r)
        3. u_{k+1} = u_k + step_size * delta_u
        """
        if self.model is None:
            raise ValueError("DeepONet model not loaded!")

        residual = self.compute_residual(u_prev)

        # DeepONet 预测
        # 注意: DeepONet.predict 需要 (Batch, N) 的输入
        # k_x 需要全网格, residual 是内部网格 (N-2)
        # 这里的逻辑对应: L(e) = r, DeepONet 预测 e

        delta_u = self.model.predict(
            k_x=self.k_x,  # (N,) -> internally handles batch
            f_x=residual,  # (N-2,) -> treated as the "f" input for Green's function
            trunk_input=None  # default inner nodes
        )

        # delta_u 也是 (N-2,)
        return u_prev + step_size * delta_u

    def solve(self,
              mode: str = "hybrid",
              u_init: np.ndarray = None,
              max_iter: int = 1000,
              tol: float = 1e-6,
              neural_step_size: float = 1.0,
              use_anderson: bool = False,
              anderson_m: int = 5):
        """
        统一求解入口

        Args:
            mode: "numerical", "deeponet", "hybrid"
            u_init: 初始猜测 (inner points only, shape N-2)
            max_iter: 最大迭代次数
            tol: 收敛容差
            neural_step_size: DeepONet 更新的步长 (0 < eta <= 1)
            use_anderson: 是否启用 Anderson 加速
            anderson_m: Anderson 历史步数

        Returns:
            u_final (full grid), metrics dict
        """

        # 初始化 u
        num_inner = self.f_inner.shape[0]
        if u_init is None:
            u_curr = np.zeros(num_inner)
        else:
            # 确保输入是 inner points
            if len(u_init) == len(self.x_nodes):
                u_curr = u_init[self.inner_slice]
            else:
                u_curr = u_init.copy()

        # 初始化 Anderson
        aa = AndersonAcceleration(m=anderson_m) if use_anderson else None

        # 记录指标
        history = {
            "residual_norm": [],
            "time": []
        }

        start_time = time.time()
        pbar = tqdm(range(max_iter), desc=f"Solving ({mode})")

        for iter_idx in pbar:
            u_old = u_curr.copy()

            # === 1. 策略选择 (Strategy) ===
            if mode == "numerical":
                u_next = self._numerical_step(u_curr)

            elif mode == "deeponet":
                # 纯 DeepONet 迭代通常等价于 Richardson 迭代: u = u + G(r)
                u_next = self._neural_step(u_curr, step_size=neural_step_size)

            elif mode == "hybrid":
                # 策略: 每 N 步做一次 Neural，其余做 Numerical
                # 这里的逻辑是: 0, ratio, 2*ratio ... 时做 Neural
                if (iter_idx % self.hybrid_ratio) == 0:
                    u_next = self._neural_step(u_curr, step_size=neural_step_size)
                else:
                    u_next = self._numerical_step(u_curr)
            else:
                raise ValueError(f"Unknown mode: {mode}")

            # === 2. Anderson Acceleration (Optional) ===
            # u_next 视为 G(u_curr), 我们希望加速不动点 u = G(u)
            if use_anderson and iter_idx > 0:
                u_next = aa.compute(u_curr, u_next)

            # === 3. 检查收敛性 ===
            # 计算真实残差模长
            current_res = self.compute_residual(u_next)
            res_norm = np.linalg.norm(current_res, ord=np.inf)

            history["residual_norm"].append(res_norm)
            history["time"].append(time.time() - start_time)

            pbar.set_postfix({"Res": f"{res_norm:.2e}"})

            if res_norm < tol:
                u_curr = u_next
                print(f"Converged at iter {iter_idx} with residual {res_norm:.2e}")
                break

            u_curr = u_next

        # 还原全网格解
        u_final = expand_solution(u_curr, u_left=0.0, u_right=0.0)
        return u_final, history


# ==========================================
# 使用示例
# ==========================================
if __name__ == "__main__":
    from utils.cfg_util import load_config
    from src.utils.vis_util import select_test_sample

    # 1. 加载配置
    config = load_config("*poi*")
    model_path = config["newest_model_path"]  # 或者指定具体的 .pt 路径

    # 2. 获取测试数据
    # 注意: select_test_sample 返回的 k_x, f_x 已经是 numpy 数组
    k_x, f_x, u_true = select_test_sample(config, 33)

    # 3. 实例化求解器
    solver = HybridSolver(config, k_x, model_path=model_path)

    # 4. 运行不同模式的对比

    # (A) 纯数值 (Jacobi)
    print("\n--- Running Numerical Solver ---")
    u_num, hist_num = solver.solve(mode="numerical", max_iter=2000, tol=1e-6)

    # (B) 纯 DeepONet 迭代 (带步长控制)
    print("\n--- Running DeepONet Iterative ---")
    u_don, hist_don = solver.solve(mode="deeponet", max_iter=50, tol=1e-6, neural_step_size=0.8)

    # (C) Hybrid (带 Anderson 加速)
    print("\n--- Running Hybrid + Anderson ---")
    u_hyb, hist_hyb = solver.solve(mode="hybrid", max_iter=500, tol=1e-6,
                                   use_anderson=True, anderson_m=5)

    # 5. 画图对比
    plt.figure(figsize=(10, 5))
    plt.semilogy(hist_num["residual_norm"], label="Numerical (Jacobi)")
    plt.semilogy(hist_don["residual_norm"], label="DeepONet Iterative")
    plt.semilogy(hist_hyb["residual_norm"], label="Hybrid + Anderson")
    plt.xlabel("Iteration")
    plt.ylabel("Residual Norm (Log)")
    plt.legend()
    plt.title("Solver Convergence Comparison")
    plt.grid(True, which="both", linestyle='--')
    plt.show()

    # 6. 解的对比
    plt.figure(figsize=(10, 5))
    x = np.linspace(0, 1, len(u_true))
    plt.plot(x, u_true, 'k-', label="Ground Truth")
    plt.plot(x, u_hyb, 'r--', label="Hybrid Solution")
    plt.legend()
    plt.title("Solution Accuracy")
    plt.show()