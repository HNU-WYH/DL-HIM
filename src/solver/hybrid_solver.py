import time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch
import torch.nn.functional as F

from box import Box
from tqdm import tqdm
from typing import Optional

from src.problems import create_problem
from src.utils.gen1d_util import generate_x_nodes
from src.neural_operator import NeuralOperatorBase, create_no
from src.utils.stepin_utils import AndersonAcceleration, AndersonAccelerationGPU, adaptive_step_size_cg


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
        self.k_x = np.asarray(k_x, dtype=np.float64)
        self.f_x = np.asarray(f_x, dtype=np.float64)
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

        self.gpu_pipeline: bool = bool(cfg.solver.get("gpu_pipeline", False))

        if self.solver_type in ["hybrid", "neural"]:
            self.operator_type = cfg.training.get("operator_type", "DeepONet").lower()
            self._neural_init()

        self._assemble_system()

        if self.gpu_pipeline:
            self._assemble_system_gpu()

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

        if self.smoother_type == "jacobi":
            self.M = self.A_inner.diagonal()

        elif self.smoother_type in ["gauss-seidel", "gauss_seidel", "gs", "g-s"]:
            self.M = sp.tril(self.A_inner, format='csr')

        else:
            raise ValueError(f"{self.smoother_type.title()} smoother is not supported.")

    def compute_residual(self, u_inner: np.ndarray) -> np.ndarray:
        return self.f_inner - self.A_inner @ u_inner

    def _numerical_step(self, u_inner: np.ndarray, residual: Optional[np.ndarray] = None) -> np.ndarray:
        """
            u_{k+1} = u_{k} + omega * M^{-1} * (f-A u_k)
        """
        if residual is None:
            residual = self.compute_residual(u_inner)

        if self.smoother_type == "jacobi":
            D_inv = 1.0 / self.M
            delta = D_inv * residual

        elif self.smoother_type in ["gauss-seidel", "gauss_seidel", "gs", "g-s"]:
            delta = spla.spsolve_triangular(A=self.M, b=residual, lower=True)

        else:
            raise ValueError(f"{self.smoother_type.title()} smoother is not supported.")

        return u_inner + self.relax_factor * delta

    def _neural_step(self, u_inner: np.ndarray, residual: Optional[np.ndarray] = None) -> np.ndarray:
        if self.model is None:
            raise ValueError("Neural Operator model is not initialized")

        if residual is None:
            residual = self.compute_residual(u_inner)
        delta_u = self.model.predict(k_x=self.k_x, f_x=residual, x_k=self.prob_x_nodes,
                                     x_f=self.prob_x_nodes[self.inner_slice],
                                     query_points=self.prob_x_nodes[self.inner_slice])

        if self.neural_update_type == "cg":
            a_mat = self.A_inner.toarray() if sp.issparse(self.A_inner) else self.A_inner
            alpha = adaptive_step_size_cg(a_mat, delta_u, residual)
        else:
            alpha = 1.0

        return u_inner + alpha * delta_u

    # ------------------------------------------------------------------
    # GPU pipeline
    # ------------------------------------------------------------------

    def _assemble_system_gpu(self) -> None:
        """
        Pre-converts system matrices and solver-grid tensors to GPU once at
        init time so the iteration loop never needs to touch the CPU.

        Precision: float64 throughout to preserve solver convergence behaviour.
        The neural-step boundary casts float64↔float32 inside GPU memory only.
        """
        device: torch.device = (
            self.model.device if self.model is not None
            else torch.device(self.config.get("device", "cuda"))
        )
        self._gpu_device = device
        dtype = torch.float64

        # --- system matrix (sparse CSR on GPU) ---
        csr = self.A_inner.tocsr()
        self.A_gpu = torch.sparse_csr_tensor(
            torch.from_numpy(csr.indptr.astype(np.int64)),
            torch.from_numpy(csr.indices.astype(np.int64)),
            torch.from_numpy(csr.data.astype(np.float64)),
            size=csr.shape,
            dtype=dtype,
            device=device,
        )
        self.f_gpu = torch.as_tensor(self.f_inner.astype(np.float64), dtype=dtype, device=device)

        # --- preconditioner on GPU ---
        if self.smoother_type == "jacobi":
            self.D_inv_gpu = torch.as_tensor(
                (1.0 / self.A_inner.diagonal()).astype(np.float64), dtype=dtype, device=device
            )
        elif self.smoother_type in ["gauss-seidel", "gauss_seidel", "gs", "g-s"]:
            # Dense lower triangular; solve_triangular is O(N²) but avoids PCIe round-trip
            L_dense = np.tril(self.A_inner.toarray()).astype(np.float64)
            self.L_gpu = torch.as_tensor(L_dense, dtype=dtype, device=device)

        # --- neural-step precomputations (only when a model is present) ---
        if self.model is None:
            return

        model_grid_size: Optional[int] = getattr(
            self.model, "fns_num_x_nodes",
            getattr(self.model, "don_num_x_nodes", None)
        )
        model_x_nodes_np = getattr(
            self.model, "fns_x_nodes_np",
            getattr(self.model, "don_x_nodes_np", None)
        )

        # k_x interpolated once to model grid [1, N_model] float32
        k_np = self.k_x.astype(np.float32)
        if model_grid_size is not None and len(k_np) != model_grid_size and model_x_nodes_np is not None:
            k_interp = np.interp(model_x_nodes_np, self.prob_x_nodes, k_np).astype(np.float32)
        else:
            k_interp = k_np
        self.k_x_gpu = torch.as_tensor(k_interp[None, :], dtype=torch.float32, device=device)

        # query points for neural model = solver's inner grid coords [N_inner, 1] float32
        self.query_points_gpu = torch.as_tensor(
            self.prob_x_nodes[self.inner_slice, None].astype(np.float32),
            dtype=torch.float32, device=device,
        )

        # whether residual must be interpolated to model input size
        n_solver_inner = int(self.prob_x_nodes[self.inner_slice].shape[0])
        n_model_inner = (model_grid_size - 2) if model_grid_size is not None else n_solver_inner
        self._f_interp_needed: bool = (n_solver_inner != n_model_inner)
        self._f_interp_size: int = n_model_inner

    # ---- GPU residual / smoother / neural step ----

    def compute_residual_gpu(self, u_inner: torch.Tensor) -> torch.Tensor:
        return self.f_gpu - torch.sparse.mm(self.A_gpu, u_inner.unsqueeze(1)).squeeze(1)

    def _numerical_step_gpu(self, u_inner: torch.Tensor,
                             residual: Optional[torch.Tensor] = None) -> torch.Tensor:
        if residual is None:
            residual = self.compute_residual_gpu(u_inner)

        if self.smoother_type == "jacobi":
            delta = self.D_inv_gpu * residual
        elif self.smoother_type in ["gauss-seidel", "gauss_seidel", "gs", "g-s"]:
            delta = torch.linalg.solve_triangular(
                self.L_gpu, residual.unsqueeze(1), upper=False
            ).squeeze(1)
        else:
            raise ValueError(f"Unsupported smoother: {self.smoother_type}")

        return u_inner + self.relax_factor * delta

    def _neural_step_gpu(self, u_inner: torch.Tensor,
                          residual: Optional[torch.Tensor] = None) -> torch.Tensor:
        if residual is None:
            residual = self.compute_residual_gpu(u_inner)

        # residual is float64; cast to float32 for the neural model (in-GPU, zero PCIe cost)
        f_x = residual.to(torch.float32).unsqueeze(0)  # [1, N_inner]

        if self._f_interp_needed:
            # 1-D linear interpolation from solver inner grid to model inner grid
            f_x = F.interpolate(
                f_x.unsqueeze(1), size=self._f_interp_size, mode="linear", align_corners=True
            ).squeeze(1)  # [1, N_model_inner]

        delta_u = self.model.predict_tensor(
            k_x=self.k_x_gpu,
            f_x=f_x,
            query_points=self.query_points_gpu,
        ).squeeze(0)  # [N_inner]  (query_points are solver inner nodes → output at solver resolution)

        # cast back to float64 for the numerical update
        delta_f64 = delta_u.to(u_inner.dtype)

        if self.neural_update_type == "cg":
            # optimal step size via sparse matvec — stays fully on GPU
            Ap = torch.sparse.mm(self.A_gpu, delta_f64.unsqueeze(1)).squeeze(1)
            num = torch.dot(delta_f64, residual)
            den = torch.dot(delta_f64, Ap)
            alpha = (num / (den + 1e-18)).item()
        else:
            alpha = 1.0

        return u_inner + alpha * delta_f64

    # ---- main GPU solve loop ----

    def _solve_gpu(self, u_init, max_iter: int, tol: float,
                   aa_m: Optional[int], mode: str):
        device = self._gpu_device
        dtype = torch.float64

        # initialise solution on GPU
        n_inner = len(self.prob_x_nodes) - 2
        if u_init is None:
            u_curr = torch.zeros(n_inner, dtype=dtype, device=device)
        else:
            arr = np.asarray(u_init, dtype=np.float64)
            if arr.shape[0] == len(self.prob_x_nodes):
                arr = arr[self.inner_slice]
            u_curr = torch.as_tensor(arr, dtype=dtype, device=device)

        aa = (
            AndersonAccelerationGPU(m=aa_m, device=str(device))
            if self.neural_update_type == "aa" else None
        )

        history: dict = {"residual_norm": [], "time": [], "iter": []}
        start_time = time.time()
        r_curr = self.compute_residual_gpu(u_curr)
        pbar = tqdm(range(max_iter), desc=f"Solving ({mode}, GPU)")

        self.model.eval() if self.model is not None else None
        with torch.no_grad():
            for iter_idx in pbar:
                numerical_update = (iter_idx + 1) % self.hybrid_ratio

                if mode == "numerical":
                    u_next = self._numerical_step_gpu(u_curr, residual=r_curr)
                    if aa is not None:
                        u_next = u_curr + aa.compute(u_curr, u_next)

                elif mode == "deeponet":
                    u_next = self._neural_step_gpu(u_curr, residual=r_curr)

                elif mode == "hybrid":
                    if numerical_update:
                        u_next = self._numerical_step_gpu(u_curr, residual=r_curr)
                    else:
                        u_next = self._neural_step_gpu(u_curr, residual=r_curr)
                        if aa:
                            u_next = u_curr + aa.compute(u_curr, u_next)
                else:
                    raise ValueError(f"Unknown mode: {mode}")

                r_curr = self.compute_residual_gpu(u_next)
                res_norm = torch.linalg.vector_norm(r_curr, ord=float("inf")).item()

                history["residual_norm"].append(res_norm)
                history["time"].append(time.time() - start_time)
                history["iter"].append(iter_idx)
                pbar.set_postfix({"Res": f"{res_norm:.2e}"})

                u_curr = u_next
                if res_norm < tol:
                    print(f"Converged at iter {iter_idx} with residual {res_norm:.2e}")
                    break

        # only here do we copy back to host — once, after the loop
        return u_curr.cpu().numpy(), history

    # ------------------------------------------------------------------

    def solve(self, u_init: Optional[np.ndarray] = None,
              max_iter: Optional[int] = None,
              tol: Optional[float] = None,
              aa_m: Optional[int] = None,
              mode: Optional[str] = None):
        """
        Args:
            u_init: (inner points only, shape N-2)
            max_iter:
            tol:
            aa_m: Anderson history size
            mode: "numerical", "deeponet", "hybrid"

        Returns:
            u_final (full grid), metrics dict
        """

        mode = (mode or self.solver_type).lower()
        max_iter = max_iter or self.config.problem.get("iteration", 1000)
        tol = tol or self.config.problem.get("tolerance", 1e-10)

        if self.gpu_pipeline:
            return self._solve_gpu(u_init, max_iter, tol, aa_m, mode)

        # Initialize solution
        if u_init is None:
            u_curr = np.zeros(len(self.prob_x_nodes) - 2, dtype=np.float64)            # (new grid point - 2,)

        else:
            # error checking (if include boundary)
            if len(u_init) == len(self.prob_x_nodes):
                u_curr = np.asarray(u_init, dtype=np.float64)[self.inner_slice]               # (delete boundary condition)
            elif len(u_init) == len(self.prob_x_nodes) - 2:
                u_curr = np.asarray(u_init, dtype=np.float64)
            else:
                raise ValueError("The shape of initial solution is not compatible")

        # Initialize Anderson Class
        aa = AndersonAcceleration(m=aa_m) if self.neural_update_type == "aa" else None

        # For visualization and recording the process of solving PDEs
        history = {"residual_norm": [], "time": [], "iter": []}

        # Solving the PDE with DL-HIM
        start_time = time.time()
        r_curr = self.compute_residual(u_curr)
        pbar = tqdm(range(max_iter), desc=f"Solving ({mode})")

        for iter_idx in pbar:
            numerical_update = (iter_idx+1) % self.hybrid_ratio

            # 1. Choose the Strategy
            if mode == "numerical":
                u_next = self._numerical_step(u_curr, residual=r_curr)
                if aa is not None:
                    u_next = u_curr + aa.compute(u_curr, u_next)

            elif mode == "deeponet":
                u_next = self._neural_step(u_curr, residual=r_curr)

            elif mode == "hybrid":
                if numerical_update:
                    u_next = self._numerical_step(u_curr, residual=r_curr)
                    # if aa:
                    #     u_next = u_curr + aa.compute(u_curr, u_next)
                else:
                    u_next = self._neural_step(u_curr, residual=r_curr)
                    if aa:
                        u_next = u_curr + aa.compute(u_curr, u_next)
            else:
                raise ValueError(f"Unknown mode: {mode}")

            # 2. Convergence Checking
            r_curr = self.compute_residual(u_next)
            res_norm = np.linalg.norm(r_curr, ord=np.inf)

            history["residual_norm"].append(res_norm)
            history["time"].append(time.time() - start_time)
            history["iter"].append(iter_idx)

            pbar.set_postfix({"Res": f"{res_norm: .2e}"})

            u_curr = u_next
            if res_norm < tol:
                print(f"Converged at iter {iter_idx} with residual {res_norm: .2e}")
                break

        return u_curr, history
