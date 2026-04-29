"""
Evaluate 2D DL-HIM solvers: plot Error / Residual vs Iteration.
This script is dedicated to diffusion2d-style datasets and hybrid solvers.
"""
import os
import time
import copy
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from box import Box
from tqdm import tqdm
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from scipy.interpolate import RegularGridInterpolator

from src.utils.cfg_util import load_config
from src.utils.fdm_utils import expand_solution_2d
from src.solver.hybrid_solver import HybridSolver
from src.utils.stepin_utils import AndersonAcceleration, PAAA

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# =============================================================================
# Configuration
# =============================================================================
CONFIG_WILDCARD = "diffusion2d*"
TEST_GRID_SHAPE: Optional[Tuple[int, int]] = None
TEST_DATASET_PATH: Optional[str] = None
SAMPLE_INDICES: Optional[Sequence[int]] = [0, 1]
MAX_ITER: Optional[int] = 500
TOL: Optional[float] = None
OUTPUT_PATH: Optional[str] = None

MODEL_PATHS: Dict[str, Optional[str]] = {
    "Default": None,
}

CASES: List[Dict] = [
    {"label": "Jacobi", "mode": "numerical", "model": None, "numerical_method": "jacobi"},
    {"label": "HINTS-Fixed (Jacobi)", "mode": "hybrid", "model": "Default", "numerical_method": "jacobi",
     "relaxation_factor": 0.6, "hybrid_ratio": 20, "neural_update": "fixed"},
    {"label": "HINTS-AA (Jacobi)", "mode": "hybrid", "model": "Default", "numerical_method": "jacobi",
     "relaxation_factor": 0.6, "hybrid_ratio": 20, "neural_update": "aa", "aa_m": 10},
    {"label": "HINTS-PAAA (Jacobi)", "mode": "hybrid", "model": "Default", "numerical_method": "jacobi",
     "relaxation_factor": 0.6, "hybrid_ratio": 20, "neural_update": "paaa", "aa_m": 10},
    {"label": "HINTS-ELS (Jacobi)", "mode": "hybrid", "model": "Default", "numerical_method": "jacobi",
     "relaxation_factor": 0.6, "hybrid_ratio": 20, "neural_update": "cg"},
]


# =============================================================================
# Data loading
# =============================================================================
def load_evaluation_dataset(cfg: Box, test_path=TEST_DATASET_PATH) -> Tuple[np.lib.npyio.NpzFile, bool]:
    dataset_path = cfg["dataset_path"]
    if test_path is None:
        base, ext = os.path.splitext(dataset_path)
        test_path = f"{base}_test{ext}"
    if os.path.exists(test_path):
        return np.load(test_path), True
    return np.load(dataset_path), False


def _interp_grid_2d(values: np.ndarray,
                    x_src: np.ndarray, y_src: np.ndarray,
                    x_tgt: np.ndarray, y_tgt: np.ndarray) -> np.ndarray:
    fn = RegularGridInterpolator((x_src, y_src), values, method="linear")
    XX, YY = np.meshgrid(x_tgt, y_tgt, indexing="ij")
    pts = np.stack([XX.ravel(), YY.ravel()], axis=-1)
    return fn(pts).reshape(len(x_tgt), len(y_tgt))


def select_test_sample(grid_shape: Optional[Tuple[int, int]],
                       dataset,
                       test_sample_indices: List[int] = None,
                       use_test_dataset: bool = False):
    if np.isscalar(test_sample_indices):
        return_list = False
        test_sample_indices = [test_sample_indices]
    else:
        return_list = True

    x_before = dataset["x_data"]
    y_before = dataset["y_data"]
    k_key = "k_data" if use_test_dataset else "k_data_val"
    f_key = "f_data" if use_test_dataset else "f_data_val"

    k = dataset[k_key][test_sample_indices]
    f = dataset[f_key][test_sample_indices]

    if grid_shape is None:
        x_after, y_after = x_before, y_before
    else:
        Nx, Ny = grid_shape
        x_after = np.linspace(0, 1, Nx)
        y_after = np.linspace(0, 1, Ny)

    if len(x_before) != len(x_after) or len(y_before) != len(y_after):
        k = np.stack([
            _interp_grid_2d(k[i], x_before, y_before, x_after, y_after)
            for i in range(k.shape[0])
        ], axis=0)
        f = np.stack([
            _interp_grid_2d(f[i], x_before[1:-1], y_before[1:-1], x_after[1:-1], y_after[1:-1])
            for i in range(f.shape[0])
        ], axis=0)

    if return_list:
        return np.asarray(k), np.asarray(f), x_after, y_after
    return k[0], f[0], x_after, y_after


# =============================================================================
# Config helpers
# =============================================================================
def resolve_model_path(model_key: Optional[str], model_paths=MODEL_PATHS) -> Optional[str]:
    if model_key is None:
        return None
    if model_key not in model_paths:
        raise KeyError(f"Model '{model_key}' not in MODEL_PATHS")
    return model_paths[model_key]


def normalize_mode(mode: str) -> str:
    mode = mode.lower()
    if mode == "neural":
        return "deeponet"
    return mode


def apply_case_overrides(base_cfg: Box, case: Dict, tol=TOL, max_iter=MAX_ITER) -> Box:
    cfg = copy.deepcopy(base_cfg)
    cfg.solver.type = normalize_mode(case.get("mode", cfg.solver.get("type", "hybrid")))

    if tol is not None:
        cfg.problem.tolerance = tol
    if max_iter is not None:
        cfg.problem.max_iter = max_iter

    if (v := case.get("numerical_method")) is not None:
        cfg.solver.numerical["method"] = v
    if (v := case.get("relaxation_factor")) is not None:
        cfg.solver.numerical["relaxation_factor"] = v
    if (v := case.get("hybrid_ratio")) is not None:
        cfg.solver.hybrid["update_ratio"] = v
    if (v := case.get("neural_update")) is not None:
        cfg.solver.hybrid["neural_update"] = v
    if (v := case.get("aa_m")) is not None:
        cfg.solver.hybrid["aa_m"] = v

    model_path = case.get("model_path") or resolve_model_path(case.get("model"))
    if model_path is not None:
        cfg["model_load_path"] = model_path

    return cfg


# =============================================================================
# History collection
# =============================================================================
def pad_series(values: List[float], target_len: int) -> np.ndarray:
    if not values:
        return np.zeros(target_len)
    if len(values) >= target_len:
        return np.asarray(values[:target_len], dtype=np.float64)
    padded = np.full(target_len, values[-1], dtype=np.float64)
    padded[: len(values)] = values
    return padded


def collect_history(solver: HybridSolver, u_ref_inner: np.ndarray,
                    max_iter: int, tol: float, mode: str,
                    one_shot: bool = False, aa_m: Optional[int] = None,
                    use_cache_residual: bool = True,
                    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mode = normalize_mode(mode)
    u_curr = np.zeros_like(u_ref_inner, dtype=np.float64)

    if one_shot:
        t0 = time.perf_counter()
        u_pred = solver._neural_step(u_curr)
        residual = solver.compute_residual(u_pred)
        res_norm = float(np.linalg.norm(residual, ord=2))
        err_norm = float(np.linalg.norm(u_pred - u_ref_inner, ord=2))
        elapsed = time.perf_counter() - t0
        return (
            np.full(max_iter, res_norm, dtype=np.float64),
            np.full(max_iter, err_norm, dtype=np.float64),
            np.full(max_iter, elapsed, dtype=np.float64),
            u_pred,
        )

    if solver.neural_update_type == "aa":
        history_size = aa_m or solver.config.solver.hybrid.get("aa_m", 10)
        aa = AndersonAcceleration(m=history_size, reg=1e-20)
    elif solver.neural_update_type == "paaa":
        history_size = aa_m or solver.config.solver.hybrid.get("aa_m", 10)
        aa = PAAA(m=history_size, reg=1e-20)
    else:
        aa = None

    errors: List[float] = []
    residuals: List[float] = []
    times: List[float] = []

    r_curr = solver.compute_residual(u_curr) if use_cache_residual else None
    t0 = time.perf_counter()

    for iter_idx in range(max_iter):
        numerical_update = (iter_idx + 1) % solver.hybrid_ratio
        apply_anderson = False

        if mode == "numerical":
            g_k = solver._numerical_step(u_curr, residual=r_curr)
            apply_anderson = aa is not None
        elif mode == "deeponet":
            g_k = solver._neural_step(u_curr, residual=r_curr)
        elif mode == "hybrid":
            if numerical_update:
                g_k = solver._numerical_step(u_curr, residual=r_curr)
            else:
                g_k = solver._neural_step(u_curr, residual=r_curr)
                apply_anderson = aa is not None
        else:
            raise ValueError(f"Unknown mode: {mode}")

        if apply_anderson and solver.neural_update_type == "paaa":
            r_gk = solver.compute_residual(g_k)
            u_next, r_next, _ = aa.compute(g_k, r_gk)
        elif apply_anderson and solver.neural_update_type == "aa":
            u_next = u_curr + aa.compute(u_curr, g_k)
            r_next = solver.compute_residual(u_next)
        else:
            u_next = g_k
            r_next = solver.compute_residual(u_next)

        res_norm = float(np.linalg.norm(r_next, ord=2))
        err_norm = float(np.linalg.norm(u_next - u_ref_inner, ord=2))
        residuals.append(res_norm)
        errors.append(err_norm)
        times.append(time.perf_counter() - t0)

        u_curr = u_next
        r_curr = r_next

        if res_norm < tol:
            break

    return pad_series(residuals, max_iter), pad_series(errors, max_iter), pad_series(times, max_iter), u_curr


def evaluate_case_on_sample(base_cfg: Box, case: Dict,
                            k_xy: np.ndarray, f_inner: np.ndarray,
                            x_nodes: np.ndarray, y_nodes: np.ndarray,
                            max_iter: int = MAX_ITER, tol: float = TOL,
                            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cfg = apply_case_overrides(base_cfg, case)
    max_iter = max_iter or case.get("max_iter") or cfg.problem.get("iteration", 200)
    tol = tol or case.get("tol") or cfg.problem.get("tolerance", 1e-10)

    solver = HybridSolver(
        cfg,
        k_x=k_xy,
        f_x=expand_solution_2d(f_inner, len(x_nodes), len(y_nodes)),
        eps=cfg.testing.get("eps", 1.0),
        prob_x_nodes=(x_nodes, y_nodes),
        cp_path=cfg.get("model_load_path"),
    )

    if sp.issparse(solver.A_inner):
        u_gt_inner = spla.spsolve(solver.A_inner, solver.f_inner)
    else:
        u_gt_inner = np.linalg.solve(solver.A_inner, solver.f_inner)

    res_hist, err_hist, time_hist, u_curr = collect_history(
        solver, u_gt_inner,
        mode=case.get("mode", "hybrid"),
        max_iter=max_iter, tol=tol,
        one_shot=case.get("one_shot", False),
        aa_m=case.get("aa_m"),
    )
    return err_hist, res_hist, time_hist, u_curr, u_gt_inner


def average_histories(histories: Iterable[np.ndarray]) -> np.ndarray:
    return np.stack(list(histories), axis=0).mean(axis=0)


# =============================================================================
# Main evaluation loop
# =============================================================================
def run_evaluation(config_wildcard: str = CONFIG_WILDCARD,
                   test_dataset_path: str = TEST_DATASET_PATH,
                   sample_indices: Sequence[int] = SAMPLE_INDICES,
                   test_grid_shape: Optional[Tuple[int, int]] = TEST_GRID_SHAPE,
                   cases=CASES):
    cfg = load_config(config_wildcard)
    data, use_test_dataset = load_evaluation_dataset(cfg, test_dataset_path)

    available_indices = list(range(len(data["k_data"] if use_test_dataset else data["k_data_val"])))
    sample_indices = sample_indices if sample_indices is not None else available_indices
    if any(idx not in available_indices for idx in sample_indices):
        raise IndexError("Sample indices exceed available evaluation data")

    k_val, f_val, x_val, y_val = select_test_sample(
        grid_shape=test_grid_shape,
        dataset=data,
        test_sample_indices=sample_indices,
        use_test_dataset=use_test_dataset,
    )

    case_results = {case["label"]: {"errors": [], "residuals": [], "times": [], "iters": None} for case in cases}

    for sample_idx, _ in enumerate(tqdm(sample_indices, desc="Samples")):
        for case in cases:
            err_hist, res_hist, time_hist, _, _ = evaluate_case_on_sample(
                cfg, case, k_val[sample_idx], f_val[sample_idx], x_val, y_val,
            )
            case_results[case["label"]]["errors"].append(err_hist)
            case_results[case["label"]]["residuals"].append(res_hist)
            case_results[case["label"]]["times"].append(time_hist)
            case_results[case["label"]]["iters"] = np.arange(1, len(err_hist) + 1)

    avg_results = {}
    for label, store in case_results.items():
        avg_results[label] = {
            "iter": store["iters"],
            "error": average_histories(store["errors"]),
            "residual": average_histories(store["residuals"]),
            "time": average_histories(store["times"]),
        }

    return avg_results


# =============================================================================
# Plotting
# =============================================================================
METHOD_COLORS = {
    "Fixed": "#1f77b4",
    "ELS":   "#ff7f0e",
    "AA":    "#2ca02c",
    "PAAA":  "#d62728",
    "Jacobi": "#9467bd",
}


def _get_method_name(label: str) -> str:
    if label == "Jacobi":
        return "Jacobi"
    if "PAAA" in label:
        return "PAAA"
    if "AA" in label:
        return "AA"
    if "ELS" in label:
        return "ELS"
    return "Fixed"


if __name__ == "__main__":
    avg_results = run_evaluation(
        config_wildcard=CONFIG_WILDCARD,
        test_dataset_path=TEST_DATASET_PATH,
        sample_indices=SAMPLE_INDICES,
        test_grid_shape=TEST_GRID_SHAPE,
    )

    fig = plt.figure(figsize=(7, 7))
    ax1 = plt.subplot(2, 1, 1)
    ax2 = plt.subplot(2, 1, 2)

    for label, vals in avg_results.items():
        method = _get_method_name(label)
        color = METHOD_COLORS[method]
        ax1.semilogy(vals["iter"], vals["error"], color=color, linewidth=1.5, label=label)
        ax2.semilogy(vals["iter"], vals["residual"], color=color, linewidth=1.5, label=label)

    ax1.set_title("Average Error Norm vs Iteration (2D)")
    ax1.set_ylabel(r"$\ell_2$ Norm of Error")
    ax1.grid(True)

    ax2.set_title("Average Residual Norm vs Iteration (2D)")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel(r"$\ell_2$ Norm of Residual")
    ax2.grid(True)

    legend_methods = [Line2D([0], [0], color=METHOD_COLORS[_get_method_name(label)], lw=2, label=label)
                      for label in avg_results.keys()]

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.16)
    fig.legend(handles=legend_methods, loc="lower center", bbox_to_anchor=(0.5, 0.01),
               ncol=2, frameon=True)

    if OUTPUT_PATH:
        if os.path.dirname(OUTPUT_PATH):
            os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
        plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
        print(f"Figure saved to: {OUTPUT_PATH}")
        data_dict = {}
        for label, vals in avg_results.items():
            key = label.replace(" ", "_").replace("(", "").replace(")", "")
            for k in ("iter", "error", "residual", "time"):
                data_dict[f"{key}__{k}"] = vals[k]
        np.savez(os.path.splitext(OUTPUT_PATH)[0] + "_data.npz", **data_dict)
    else:
        plt.show()
