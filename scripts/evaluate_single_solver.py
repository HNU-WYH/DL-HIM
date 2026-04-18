"""
Evaluate DL-HIM solvers: plot Error / Residual vs Iteration.
Configure all settings via the uppercase constants below.
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

from src.utils.cfg_util import load_config
from src.utils.fdm_utils import expand_solution
from src.solver.hybrid_solver import HybridSolver
from src.utils.visualization import plot_case_predictions
from src.utils.stepin_utils import AndersonAcceleration, PAAA

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# =============================================================================
# Configuration
# =============================================================================
CONFIG_WILDCARD = "diffusion*"
TEST_GRID_NUM: Optional[int] = 1201
TEST_DATASET_PATH: Optional[str] = None
SAMPLE_INDICES: Optional[Sequence[int]] = [1,2]
PLOT_SAMPLE_INDICES: Optional[Sequence[int]] = None
MAX_ITER: Optional[int] = 1000
TOL: Optional[float] = None
OUTPUT_PATH: Optional[str] = None                        # e.g. "results/diffusion_iter.pdf"; None → show

MODEL_PATHS: Dict[str, Optional[str]] = {
    "Default": "./checkpoints/fns_diffusion1d_fno/dynamic_error_l2/diffusion_1D_Grid31_Ep101_2026-04-17.pt",
    # "Default": "./checkpoints/deeponet_diffusion1d/dynamic_residual_l2/diffusion_1D_Grid31_Ep20000_2026-01-26.pt",
    # "Default": "checkpoints/deeponet_helmholtz1d/dynamic_residual_l2/helmholtz_1D_Grid31_Ep20000_2026-01-26.pt",
}

CASES: List[Dict] = [
    # {"label": "Pure-DeepONet", "mode": "neural", "model": "Default", "one_shot": True},

    # {"label": "Jacobi", "mode": "numerical", "model": None, "numerical_method": "jacobi"},

    # {"label": "Jacobi-AA", "mode": "numerical", "model": None, "numerical_method": "jacobi",
    #  "numerical_update": "aa", "aa_m": 10},

    {"label": "HINTS-Fixed (Jacobi)", "mode": "hybrid", "model": "Default", "numerical_method": "jacobi",
     "relaxation_factor": 0.66, "hybrid_ratio": 20, "neural_update": "fixed"},

    {"label": "HINTS-AA (Jacobi)", "mode": "hybrid", "model": "Default", "numerical_method": "jacobi",
     "relaxation_factor": 0.66, "hybrid_ratio": 20, "neural_update": "aa", "aa_m": 10},

    {"label": "HINTS-PAAA (Jacobi)", "mode": "hybrid", "model": "Default", "numerical_method": "jacobi",
     "relaxation_factor": 0.66, "hybrid_ratio": 20, "neural_update": "paaa", "aa_m": 10},

    {"label": "HINTS-ELS (Jacobi)", "mode": "hybrid", "model": "Default", "numerical_method": "jacobi",
     "relaxation_factor": 0.66, "hybrid_ratio": 20, "neural_update": "cg"},

    # {"label": "Gauss-Seidel", "mode": "numerical", "model": None, "numerical_method": "gauss-seidel"},

    {"label": "HINTS-Fixed (GS)", "mode": "hybrid", "model": "Default", "numerical_method": "gauss-seidel",
     "relaxation_factor": 1.0, "hybrid_ratio": 20, "neural_update": "fixed"},

    {"label": "HINTS-AA (GS)", "mode": "hybrid", "model": "Default", "numerical_method": "gauss-seidel",
     "relaxation_factor": 1.0, "hybrid_ratio": 20, "neural_update": "aa", "aa_m": 10},

    {"label": "HINTS-PAAA (GS)", "mode": "hybrid", "model": "Default", "numerical_method": "gauss-seidel",
     "relaxation_factor": 1.0, "hybrid_ratio": 20, "neural_update": "paaa", "aa_m": 10},

    {"label": "HINTS-ELS (GS)", "mode": "hybrid", "model": "Default", "numerical_method": "gauss-seidel",
     "relaxation_factor": 1.0, "hybrid_ratio": 20, "neural_update": "cg"},
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


def select_test_sample(grid_num: Optional[int], dataset,
                       test_sample_indices: List[int] = None,
                       use_test_dataset: bool = False):
    if np.isscalar(test_sample_indices):
        return_list = False
        test_sample_indices = [test_sample_indices]
    else:
        return_list = True

    x_before = dataset["x_data"]
    k_key = "k_data" if use_test_dataset else "k_data_val"
    f_key = "f_data" if use_test_dataset else "f_data_val"

    k = dataset[k_key][test_sample_indices]
    f = dataset[f_key][test_sample_indices]

    x_after = np.linspace(0, 1, grid_num) if grid_num is not None else x_before

    if grid_num and len(x_before) != len(x_after):
        k = [np.interp(x_after, x_before, k[i]) for i in range(k.shape[0])]
        f = [np.interp(x_after[1:-1], x_before[1:-1], f[i]) for i in range(f.shape[0])]

    if return_list:
        return np.array(k), np.array(f), x_after
    return k[0], f[0], x_after


# =============================================================================
# Config helpers
# =============================================================================
def resolve_model_path(model_key: Optional[str], model_paths=MODEL_PATHS) -> Optional[str]:
    if model_key is None:
        return None
    if model_key not in model_paths:
        raise KeyError(f"Model '{model_key}' not in MODEL_PATHS")
    return model_paths[model_key]


def infer_hard_constraints_from_path(model_path: Optional[str]) -> Optional[bool]:
    if not model_path:
        return None
    norm_path = os.path.normpath(model_path)
    if f"{os.sep}nocons{os.sep}" in norm_path:
        return False
    if f"{os.sep}cons{os.sep}" in norm_path:
        return True
    return None


def apply_case_overrides(base_cfg: Box, case: Dict, tol=TOL, max_iter=MAX_ITER) -> Box:
    cfg = copy.deepcopy(base_cfg)
    cfg.solver.type = case.get("mode", cfg.solver.get("type", "hybrid")).lower()

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
    mode = mode.lower()
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
                            k_x: np.ndarray, f_inner: np.ndarray, x_nodes: np.ndarray,
                            max_iter: int = MAX_ITER, tol: float = TOL,
                            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cfg = apply_case_overrides(base_cfg, case)
    max_iter = max_iter or case.get("max_iter") or cfg.problem.get("iteration", 200)
    tol = tol or case.get("tol") or cfg.problem.get("tolerance", 1e-10)

    solver = HybridSolver(
        cfg,
        k_x=k_x,
        f_x=expand_solution(f_inner),
        eps=cfg.testing.get("eps", 1.0),
        prob_x_nodes=x_nodes,
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
def run_evaluation(plot_indices: Optional[Sequence[int]] = None,
                   config_wildcard: str = CONFIG_WILDCARD,
                   test_dataset_path: str = TEST_DATASET_PATH,
                   sample_indices: Sequence[int] = SAMPLE_INDICES,
                   test_grid_num: int = TEST_GRID_NUM,
                   cases=CASES):
    cfg = load_config(config_wildcard)
    data, use_test_dataset = load_evaluation_dataset(cfg, test_dataset_path)

    available_indices = list(range(len(data["k_data"] if use_test_dataset else data["k_data_val"])))
    sample_indices = sample_indices if sample_indices is not None else available_indices
    if any(idx not in available_indices for idx in sample_indices):
        raise IndexError("Sample indices exceed available evaluation data")

    plot_indices = list(plot_indices) if plot_indices is not None else []

    k_val, f_val, x_val = select_test_sample(grid_num=test_grid_num, dataset=data,
                                             test_sample_indices=sample_indices,
                                             use_test_dataset=use_test_dataset)

    case_results = {case["label"]: {"errors": [], "residuals": [], "times": [], "iters": None} for case in cases}
    plot_predictions = {idx: {} for idx in plot_indices}

    for sample_idx, _ in enumerate(tqdm(sample_indices, desc="Samples")):
        for case in cases:
            err_hist, res_hist, time_hist, u_curr, u_true = evaluate_case_on_sample(
                cfg, case, k_val[sample_idx], f_val[sample_idx], x_val,
            )
            case_results[case["label"]]["errors"].append(err_hist)
            case_results[case["label"]]["residuals"].append(res_hist)
            case_results[case["label"]]["times"].append(time_hist)
            case_results[case["label"]]["iters"] = np.arange(1, len(err_hist) + 1)

            if sample_idx in plot_predictions:
                plot_predictions[sample_idx][case["label"]] = {
                    "u_true": u_true, "u_pred": u_curr,
                    "x_nodes": x_val[1:-1], "error": err_hist[-1],
                }

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
}
SOLVER_STYLES = {
    "Jacobi": "-",
    "GS":     "--",
}


def _get_method_name(label: str) -> str:
    if "PAAA" in label:
        return "PAAA"
    if "AA" in label:
        return "AA"
    if "ELS" in label:
        return "ELS"
    return "Fixed"


if __name__ == "__main__":
    avg_results = run_evaluation(
        plot_indices=PLOT_SAMPLE_INDICES,
        config_wildcard=CONFIG_WILDCARD,
        test_grid_num=TEST_GRID_NUM,
    )

    fig = plt.figure(figsize=(7, 7))
    ax1 = plt.subplot(2, 1, 1)
    ax2 = plt.subplot(2, 1, 2)

    for label, vals in avg_results.items():
        method = _get_method_name(label)
        solver = "GS" if "GS" in label else "Jacobi"
        color = METHOD_COLORS[method]
        ls = SOLVER_STYLES[solver]
        ax1.semilogy(vals["iter"], vals["error"], color=color, linestyle=ls, linewidth=1.5)
        ax2.semilogy(vals["iter"], vals["residual"], color=color, linestyle=ls, linewidth=1.5)

    ax1.set_title("Average Error Norm vs Iteration")
    ax1.set_ylabel(r"$\ell_2$ Norm of Error")
    ax1.set_ylim(top=1e2)
    ax1.grid(True)

    ax2.set_title("Average Residual Norm vs Iteration")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel(r"$\ell_2$ Norm of Residual")
    ax2.set_ylim(top=1e2)
    ax2.grid(True)

    legend_methods = [Line2D([0], [0], color=c, lw=2, label=f"HINTS-{m}") for m, c in METHOD_COLORS.items()]
    legend_solvers = [Line2D([0], [0], color="gray", linestyle=s, lw=2, label=n) for n, s in SOLVER_STYLES.items()]

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.18)
    fig.legend(handles=legend_methods, title="Methods (Colors)",
               loc="lower center", bbox_to_anchor=(0.35, 0.005), ncol=2, frameon=True)
    fig.legend(handles=legend_solvers, title="Solvers (Line Styles)",
               loc="lower center", bbox_to_anchor=(0.78, 0.005), ncol=1, frameon=True)

    if OUTPUT_PATH:
        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True) if os.path.dirname(OUTPUT_PATH) else None
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
