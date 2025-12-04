import time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from box import Box
from tqdm import tqdm
from typing import Optional

from src.problems import create_problem
from src.utils.gen1d_util import generate_x_nodes
from src.neural_operator import NeuralOperatorBase, create_no
from src.utils.stepin_utils import AndersonAcceleration, adaptive_step_size_cg


class HybridSolver:
    def __init__(self, cfg: Box,
                 k_x: np.ndarray, f_x: np.ndarray, eps=1.0,
                 don_x_nodes: Optional[np.ndarray] = None,
                 prob_x_nodes: Optional[np.ndarray] = None,
                 model: Optional[NeuralOperatorBase] = None,
                 cp_path: Optional[str] = None
                 ):
        """
            Hybrid solver combining classical iterations and neural operators.
        """
        self.config = cfg

        self.eps = float(eps)
        self.k_x = np.asarray(k_x, dtype=np.float32)
        self.f_x = np.asarray(f_x, dtype=np.float32)
        self.solver_type = cfg.solver.get("type", "Hybrid").lower()

        self.don_x_nodes = np.asarray(don_x_nodes) if don_x_nodes is not None \
            else generate_x_nodes(cfg.data.mesh.grid_type, cfg.data.mesh.grid_num)
        self.prob_x_nodes = np.asarray(prob_x_nodes) if prob_x_nodes is not None \
            else np.linspace(0, 1, k_x.shape[-1])

        # Numerical Configuration
        numerical_cfg = cfg.solver.get("numerical", {})
        self.relax_factor = numerical_cfg.get("relaxation_factor", 0.66)
        self.smoother_type = numerical_cfg.get("method", "Jacobi").lower()

        # Hybrid Configuration
        hybrid_cfg = cfg.solver.get("hybrid", {})
        self.hybrid_ratio = hybrid_cfg.get("update_ratio", 20)
        self.neural_update_type = hybrid_cfg.get("neural_update", "fixed").lower()

        # Neural Configuration
        self.model = model
        self.cp_path = cp_path

        if self.solver_type in ["hybrid", "neural"]:
            self.operator_type = cfg.training.get("operator_type", "DeepONet").lower()
            self._neural_init()

        self._assemble_system()

    def _neural_init(self) -> None:
        print(f"{self.solver_type.title()} Solver ({self.operator_type}) ")

        if self.model is None:
            self.model = create_no(self.config)

        if self.cp_path is None:
            self.cp_path = self.config["model_load_path"]

        self.model.load_model(self.cp_path)

    def _assemble_system(self) -> None:

        problem_cls = create_problem(self.config.problem.type)
        self.problem = problem_cls(x_nodes=self.prob_x_nodes, k_x=self.k_x, f_x=self.f_x,
                                   eps=self.eps, mat_type="sparse")

        self.A_inner, self.f_inner, self.inner_slice, self.bc_idx = self.problem.build_system(
            u_left=0.0, u_right=0.0, method=self.config.problem.method
        )

        if sp.issparse(self.A_inner):
            self.A_inner = self.A_inner.toarray()

        if self.smoother_type == "jacobi":
            self.M = self.A_inner.diagonal()

        elif self.smoother_type in ["gauss-seidel", "gauss_seidel", "gs", "g-s"]:
            self.M = np.tril(self.A_inner)

        else:
            raise ValueError(f"{self.smoother_type.title()} smoother is not supported.")

    def compute_residual(self, u_inner: np.ndarray) -> np.ndarray:
        return self.f_inner - self.A_inner @ u_inner

    def _numerical_step(self, u_inner: np.ndarray) -> np.ndarray:
        """
            u_{k+1} = u_{k} + omega * M^{-1} * (f-A u_k)
        """

        residual = self.compute_residual(u_inner)

        if self.smoother_type == "jacobi":
            D_inv = 1.0 / self.M
            delta = D_inv * residual

        elif self.smoother_type in ["gauss-seidel", "gauss_seidel", "gs", "g-s"]:
            delta = spla.spsolve_triangular(A=self.M, b=residual, lower=True)

        else:
            raise ValueError(f"{self.smoother_type.title()} smoother is not supported.")

        return u_inner + self.relax_factor * delta

    def _neural_step(self, u_inner: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("Neural Operator model is not initialized")

        residual = self.compute_residual(u_inner)
        delta_u = self.model.predict(k_x=self.k_x, f_x=residual, x_k=self.prob_x_nodes,
                                     x_f=self.prob_x_nodes[self.inner_slice],
                                     query_points=self.prob_x_nodes[self.inner_slice])

        if self.neural_update_type == "cg":
            alpha = adaptive_step_size_cg(self.A_inner, delta_u, residual)
        else:
            alpha = 1.0

        return u_inner + alpha * delta_u

    def solve(self, u_init: Optional[np.ndarray] = None,
              max_iter: Optional[int] = None,
              tol: Optional[float] = None,
              aa_m: Optional[int] = None,
              mode: Optional[str] = None):
        """
        统一求解入口

        Args:
            u_init: 初始猜测 (inner points only, shape N-2)
            max_iter: 最大迭代次数
            tol: 收敛容差
            aa_m: Anderson 历史步数
            mode: "numerical", "deeponet", "hybrid"

        Returns:
            u_final (full grid), metrics dict
        """

        mode = (mode or self.solver_type).lower()
        max_iter = max_iter or self.config.problem.get("iteration", 1000)
        tol = tol or self.config.problem.get("tolerance", 1e-10)

        # Initialize solution
        if u_init is None:
            u_curr = np.zeros(len(self.prob_x_nodes) - 2, dtype=np.float32)            # (new grid point - 2,)

        else:
            # error checking (if include boundary)
            if len(u_init) == len(self.prob_x_nodes):
                u_curr = np.asarray(u_init, dtype=np.float32)[self.inner_slice]               # (delete boundary condition)
            elif len(u_init) == len(self.prob_x_nodes) - 2:
                u_curr = np.asarray(u_init, dtype=np.float32)
            else:
                raise ValueError("The shape of initial solution is not compatible")

        # Initialize Anderson Class
        aa = AndersonAcceleration(m=aa_m) if self.neural_update_type == "aa" else None

        # For visualization and recording the process of solving PDEs
        history = {"residual_norm": [], "time": [], "iter": []}

        # Solving the PDE with DL-HIM
        start_time = time.time()
        pbar = tqdm(range(max_iter), desc=f"Solving ({mode})")

        for iter_idx in pbar:
            numerical_update = (iter_idx+1) % self.hybrid_ratio

            # 1. Choose the Strategy
            if mode == "numerical":
                u_next = self._numerical_step(u_curr)

            elif mode == "deeponet":
                u_next = self._neural_step(u_curr)

            elif mode == "hybrid":
                if numerical_update:
                    u_next = self._numerical_step(u_curr)
                else:
                    u_next = self._neural_step(u_curr)
            else:
                raise ValueError(f"Unknown mode: {mode}")

            # 2. Anderson Acceleration (Optional)
            # u_next = G(u_curr)
            if aa and not numerical_update:
                u_next = u_curr + aa.compute(u_curr, u_next)

            # 3. Convergence Checking
            current_res = self.compute_residual(u_next)
            res_norm = np.linalg.norm(current_res, ord=np.inf)

            history["residual_norm"].append(res_norm)
            history["time"].append(time.time() - start_time)
            history["iter"].append(iter_idx)

            pbar.set_postfix({"Res": f"{res_norm: .2e}"})

            u_curr = u_next
            if res_norm < tol:
                print(f"Converged at iter {iter_idx} with residual {res_norm:.2e}")
                break

        return u_curr, history
