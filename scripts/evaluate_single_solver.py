import os

from torchgen.executorch.api.et_cpp import return_type

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import copy
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from box import Box
from tqdm import tqdm

from src.solver.hybrid_solver import HybridSolver
from src.utils.fdm_utils import expand_solution
from src.utils.cfg_util import load_config

# Pre-registered config for unresolved problem and testing data
CONFIG_WILDCARD = "diffusion1d*"           # config filename wildcard
DATASET_PATH: Optional[str] = None         # path to .npz dataset; None -> config default
TEST_GRID_NUM: Optional[int] = None        # use dataset grid by default
TEST_DATASET_PATH: Optional[str] = None    # path to .npz test dataset; None -> derive from the dataset path
SAMPLE_INDICES: Sequence[int] = None       # validation indices to evaluate

# Pre-registered model checkpoints (use the keys inside CASES)
# If Default is None, will raise an error
MODEL_PATHS: Dict[str, Optional[str]] = {"Default": "checkpoints/diffusion1d/static_error_h1/diffusion_1D_Grid31_Ep20000_2025-12-05"}

# Evaluation plan: each dict describes one curve on the plot
CASES: List[Dict] = [
    {"label": "Pure-DeepONet", "mode": "neural",    "model": "Default", "one_shot": True},

    {"label": "Gauss-Seidel",  "mode": "numerical", "model": None,  "numerical_method": "gauss-seidel"},

    {"label": "Jacobi",        "mode": "numerical", "model": None,      "numerical_method": "jacobi"},

    {"label": "Jacobi-AA",     "mode": "numerical", "model": None,      "numerical_method": "jacobi", "neural_update": "aa", "aa_m": 15},

    {"label": "Hybrid-fixed",  "mode": "hybrid",    "model": "Default", "numerical_method": "jacobi", "hybrid_ratio": 20, "neural_update": "fixed"},

    {"label": "Hybrid-AA",     "mode": "hybrid",    "model": "Default", "numerical_method": "jacobi", "hybrid_ratio": 20, "neural_update": "aa", "aa_m": 15},

    {"label": "Hybrid-CG",     "mode": "hybrid",    "model": "Default", "numerical_method": "jacobi", "hybrid_ratio": 20, "neural_update": "cg"},
]

# Iteration / tolerance overrides applied to every case unless the case itself
# provides its own values otherwise using the value in the yaml file
MAX_ITER: Optional[int] = None
TOL: Optional[float] = None


# =====================
# Helper functions
# =====================
def load_evaluation_dataset(cfg: Box) -> Tuple[np.lib.npyio.NpzFile, bool]:
    """Load the preferred evaluation dataset.

    Returns
    -------
    dataset: np.lib.npyio.NpzFile
        The loaded dataset (test preferred, otherwise validation split).
    use_test_dataset: bool
        Whether the returned dataset is the dedicated test set.
    """
    dataset_path = DATASET_PATH or cfg["dataset_path"]

    test_path = TEST_DATASET_PATH
    if test_path is None:
        base, ext = os.path.splitext(dataset_path)
        test_path = f"{base}_test{ext}"

    if os.path.exists(test_path):
        return np.load(test_path), True

    return np.load(dataset_path), False

def select_test_sample(grid_num: Optional[int], dataset,
                       test_sample_indices: List[int] = None,
                       use_test_dataset: bool = False):
    if not isinstance(test_sample_indices, list):
        return_list = False
        test_sample_indices = [test_sample_indices]
    else:
        return_list = True

    x_before = dataset["x_data"]
    k_key = "k_data" if use_test_dataset else "k_data_val"
    f_key = "f_data" if use_test_dataset else "f_data_val"
    u_key = "u_data" if use_test_dataset else "u_data_val"

    k = dataset[k_key][test_sample_indices]
    f = dataset[f_key][test_sample_indices]
    u = dataset[u_key][test_sample_indices]

    x_after = np.linspace(0, 1, grid_num) if grid_num else x_before

    if grid_num and len(x_before) != len(x_after):
        k = [np.interp(x_after, x_before, k[i]) for i in range(k.shape[0])]
        f = [np.interp(x_after[1:-1], x_before[1:-1], f[i]) for i in range(f.shape[0])]
        u = [np.interp(x_after[1:-1], x_before[1:-1], u[i]) for i in range(u.shape[0])]

    if return_list:
        return np.array(k), np.array(f), x_after, np.array(u)
    else:
        return k[0], f[0], x_after, u[0]

def resolve_model_path(model_key: Optional[str]) -> Optional[str]:
    if model_key is None:
        return None
    if model_key not in MODEL_PATHS:
        raise KeyError(f"Model '{model_key}' not registered in MODEL_PATHS")
    return MODEL_PATHS[model_key]

def apply_global_overrides(cfg: Box) -> Box:
    cfg = copy.deepcopy(cfg)
    if DATASET_PATH:
        cfg["dataset_path"] = DATASET_PATH
    return cfg

def apply_case_overrides(base_cfg: Box, case: Dict,
                         tol = TOL, max_iter = MAX_ITER) -> Box:
    cfg = copy.deepcopy(base_cfg)

    # Ensure hybrid initialization for neural runs
    cfg.solver.type = case.get("mode", cfg.solver.get("type", "hybrid")).lower()

    # override the tol and max_iter
    if tol is not None:
        cfg.problem.tolerance = tol
    if max_iter is not None:
        cfg.problem.max_iter = max_iter

    numerical_method = case.get("numerical_method")
    if numerical_method is not None:
        cfg.solver.numerical["method"] = numerical_method

    hybrid_ratio = case.get("hybrid_ratio")
    if hybrid_ratio is not None:
        cfg.solver.hybrid["update_ratio"] = hybrid_ratio

    neural_update = case.get("neural_update")
    if neural_update is not None:
        cfg.solver.hybrid["neural_update"] = neural_update

    aa_m = case.get("aa_m")
    if aa_m is not None:
        cfg.solver.hybrid["aa_m"] = aa_m

    model_path = case.get("model_path") or resolve_model_path(case.get("model"))
    if model_path is not None:
        cfg["model_load_path"] = model_path

    return cfg


def pad_series(values: List[float], target_len: int) -> np.ndarray:
    """pad the history of residual or error"""
    if not values:
        return np.zeros(target_len)
    if len(values) >= target_len:
        return np.asarray(values[:target_len], dtype=np.float32)
    pad_val = values[-1]
    padded = np.full(target_len, pad_val, dtype=np.float32)
    padded[: len(values)] = values
    return padded


#TODO: 这里改为直接用cfg会不会更好? 每一次迭代都用override的cfg来工作
#TODO: 这里我一会儿又要expand,一会儿又不要,这里也要改
def collect_history(solver: HybridSolver,
                    u_ref_inner: np.ndarray,
                    max_iter: int, tol: float,
                    mode: str, one_shot: bool = False,
                    aa_m: Optional[int] = None,
                    ) -> Tuple[np.ndarray, np.ndarray]:
    mode = mode.lower()
    u_curr = np.zeros_like(u_ref_inner, dtype=np.float32)

    if one_shot:
        u_pred = solver._neural_step(u_curr)
        residual = solver.compute_residual(u_pred)
        res_norm = float(np.linalg.norm(residual, ord=2))
        err_norm = float(np.linalg.norm(u_pred - u_ref_inner, ord=2) / np.linalg.norm(u_ref_inner, ord=2))
        return np.full(max_iter, res_norm, dtype=np.float32) , np.full(max_iter, err_norm, dtype=np.float32),

    # Initialize Anderson Acceleration if requested
    aa = None
    if solver.neural_update_type == "aa":
        from src.utils.stepin_utils import AndersonAcceleration

        history_size = aa_m or solver.config.solver.hybrid.get("aa_m", 0)
        aa = AndersonAcceleration(m=history_size) if history_size else None

    errors: List[float] = []
    residuals: List[float] = []

    for iter_idx in range(max_iter):
        numerical_update = (iter_idx + 1) % solver.hybrid_ratio

        if mode == "numerical":
            u_next = solver._numerical_step(u_curr)
            if aa:
                u_next = u_curr + aa.compute(u_curr, u_next)
        elif mode == "deeponet":
            u_next = solver._neural_step(u_curr)
            if aa:
                u_next = u_curr + aa.compute(u_curr, u_next)
        elif mode == "hybrid":
            if numerical_update:
                u_next = solver._numerical_step(u_curr)
            else:
                u_next = solver._neural_step(u_curr)
                if aa:
                    u_next = u_curr + aa.compute(u_curr, u_next)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        res_vec = solver.compute_residual(u_next)
        res_norm = float(np.linalg.norm(res_vec, ord=2))
        err_norm = float(np.linalg.norm(u_next - u_ref_inner, ord=2)/ np.linalg.norm(u_ref_inner, ord=2))

        residuals.append(res_norm)
        errors.append(err_norm)
        u_curr = u_next

        if res_norm < tol:
            break

    return pad_series(residuals, max_iter) , pad_series(errors, max_iter),


def evaluate_case_on_sample(
    base_cfg: Box,
    case: Dict,
    k_x: np.ndarray,
    f_inner: np.ndarray,
    u_gt_inner: np.ndarray,
    x_nodes: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    cfg = apply_case_overrides(base_cfg, case)

    max_iter = case.get("max_iter") or MAX_ITER or cfg.problem.get("iteration", 200)
    tol = case.get("tol") or TOL or cfg.problem.get("tolerance", 1e-10)

    solver = HybridSolver(
        cfg,
        k_x=k_x,
        f_x=expand_solution(f_inner),
        eps=cfg.data.get("eps", 1.0),
        prob_x_nodes=x_nodes,
        cp_path=cfg.get("model_load_path"),
    )

    err_hist, res_hist = collect_history(
        solver,
        u_gt_inner,
        mode=case.get("mode", "hybrid"),
        max_iter=max_iter,
        tol=tol,
        one_shot=case.get("one_shot", False),
        aa_m=case.get("aa_m"),
    )
    return err_hist, res_hist


def average_histories(histories: Iterable[np.ndarray]) -> np.ndarray:
    stacked = np.stack(list(histories), axis=0)
    return stacked.mean(axis=0)


def run_evaluation():
    cfg = apply_global_overrides(load_config(CONFIG_WILDCARD))
    data, use_test_dataset = load_evaluation_dataset(cfg)

    available_indices = list(range(len(data["k_data"] if use_test_dataset else data["k_data_val"])))
    sample_indices = SAMPLE_INDICES if SAMPLE_INDICES is not None else available_indices
    if any(idx not in available_indices for idx in sample_indices):
        raise IndexError("Sample indices exceed available evaluation data")

    k_val, f_val, x_val, u_val = select_test_sample(TEST_GRID_NUM or len(data["x_data"]), data,
                                                    sample_indices, use_test_dataset=use_test_dataset)
    case_results = {case["label"]: {"errors": [], "residuals": [], "iters": None} for case in CASES}

    for sample_idx, _ in enumerate(tqdm(sample_indices, desc="Samples")):
        for case in CASES:
            err_hist, res_hist = evaluate_case_on_sample(
                cfg,
                case,
                k_val[sample_idx],
                f_val[sample_idx],
                u_val[sample_idx],
                x_val,
            )

            case_results[case["label"]]["errors"].append(err_hist)
            case_results[case["label"]]["residuals"].append(res_hist)
            case_results[case["label"]]["iters"] = np.arange(1, len(err_hist) + 1)

    avg_results = {}
    for label, store in case_results.items():
        avg_err = average_histories(store["errors"])
        avg_res = average_histories(store["residuals"])
        avg_results[label] = {"iter": store["iters"], "error": avg_err, "residual": avg_res}

    # Plotting
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    for label, vals in avg_results.items():
        plt.semilogy(vals["iter"], vals["error"], label=label)
    plt.title("Average Relative Error vs Iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Relative L2 Error")
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    for label, vals in avg_results.items():
        plt.semilogy(vals["iter"], vals["residual"], label=label)
    plt.title("Average Residual vs Iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Residual (L-inf)")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_evaluation()