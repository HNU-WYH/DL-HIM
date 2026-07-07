"""
Compute / Cost Analysis Script for DL-HIM
==========================================

Responsibilities
----------------
- Error / Residual vs Wall-clock Time plots (time-to-solution curves)
- Speedup tables relative to a chosen baseline (e.g. pure Jacobi or GS)
- Raw data export (.npz) for reproducible replotting

Self-contained: does not import from evaluate_single_solver.py.
Configure all settings via the uppercase constants below.
"""
import os
import time
import copy
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
from box import Box
from tqdm import tqdm
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from src.utils.cfg_util import load_config
from src.utils.fdm_utils import expand_solution
from src.solver.hybrid_solver import HybridSolver
from src.utils.stepin_utils import AndersonAcceleration, PAAA

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# =============================================================================
# Configuration
# =============================================================================
CONFIG_WILDCARD = "diffusion1d*"
TEST_GRID_NUM: Optional[int] = 801
TEST_DATASET_PATH: Optional[str] = None
SAMPLE_INDICES: Optional[Sequence[int]] = None
MAX_ITER: Optional[int] = 4000
TOL: Optional[float] = 1e-6  # early-stop threshold for all cases

MODEL_PATHS: Dict[str, Optional[str]] = {
    "Default": "checkpoints/fns_diffusion1d_fno/dynamic_error_l2/diffusion_1D_Grid31_ep100.pt",
}

BASELINE: str = "HINTS-Fixed (Jacobi)"                               # label of the speedup reference case
OUTPUT_PATH: Optional[str] = "results/diffusion_cost.pdf"            # e.g. "results/diffusion_cost.pdf"; None → show
SAVE_TABLE_PATH: Optional[str] = "results/diffusion_speedup.md"      # e.g. "results/diffusion_speedup.md"; None → skip

CASES: List[Dict] = [
    {"label": "HINTS-Fixed (Jacobi)", "mode": "hybrid", "model": "Default", "numerical_method": "jacobi",
     "relaxation_factor": 0.66, "hybrid_ratio": 20, "neural_update": "fixed"},

    {"label": "HINTS-PAAA (Jacobi)", "mode": "hybrid", "model": "Default", "numerical_method": "jacobi",
     "relaxation_factor": 0.66, "hybrid_ratio": 20, "neural_update": "paaa", "aa_m": 10},

    {"label": "HINTS-Fixed (GS)", "mode": "hybrid", "model": "Default", "numerical_method": "GS",
     "relaxation_factor": 1.0, "hybrid_ratio": 20, "neural_update": "fixed"},

    {"label": "HINTS-PAAA (GS)", "mode": "hybrid", "model": "Default", "numerical_method": "GS",
     "relaxation_factor": 1.0, "hybrid_ratio": 20, "neural_update": "paaa", "aa_m": 10},
]

# =============================================================================
# Time-to-solution thresholds
# =============================================================================
ERROR_THRESHOLDS = [1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
RESIDUAL_THRESHOLDS = [1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]


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

        if res_norm < tol and err_norm < tol:
            break

    return pad_series(residuals, max_iter), pad_series(errors, max_iter), pad_series(times, max_iter), u_curr


def evaluate_case_on_sample(base_cfg: Box, case: Dict,
                            k_x: np.ndarray, f_inner: np.ndarray, x_nodes: np.ndarray,
                            max_iter: int = MAX_ITER, tol: float = TOL,
                            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cfg = apply_case_overrides(base_cfg, case)
    # Priority: case-level max_iter > global MAX_ITER > cfg default
    if case.get("max_iter") is not None:
        max_iter = case.get("max_iter")
    elif max_iter is None:
        max_iter = cfg.problem.get("iteration", 700)
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
    histories = list(histories)
    if not histories:
        return np.array([])
    max_len = max(h.shape[0] for h in histories)
    padded = [pad_series(h.tolist(), max_len) if len(h) < max_len else h for h in histories]
    return np.stack(padded, axis=0).mean(axis=0)


# =============================================================================
# Main evaluation loop (time-focused, no iteration plots)
# =============================================================================
def run_cost_evaluation(config_wildcard=CONFIG_WILDCARD,
                        test_dataset_path=TEST_DATASET_PATH,
                        sample_indices=SAMPLE_INDICES,
                        test_grid_num=TEST_GRID_NUM,
                        cases=CASES) -> Dict[str, Dict]:
    cfg = load_config(config_wildcard)
    data, use_test_dataset = load_evaluation_dataset(cfg, test_dataset_path)

    available_indices = list(range(len(data["k_data"] if use_test_dataset else data["k_data_val"])))
    sample_indices = sample_indices if sample_indices is not None else available_indices
    if any(idx not in available_indices for idx in sample_indices):
        raise IndexError("Sample indices exceed available evaluation data")

    k_val, f_val, x_val = select_test_sample(grid_num=test_grid_num, dataset=data,
                                             test_sample_indices=sample_indices,
                                             use_test_dataset=use_test_dataset)

    case_results = {case["label"]: {"errors": [], "residuals": [], "times": [], "iters": None} for case in cases}

    for sample_idx, _ in enumerate(tqdm(sample_indices, desc="Samples")):
        for case in cases:
            err_hist, res_hist, time_hist, _, _ = evaluate_case_on_sample(
                cfg, case, k_val[sample_idx], f_val[sample_idx], x_val,
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
# Cost-analysis specific functions
# =============================================================================
def find_first_time(values: np.ndarray, times: np.ndarray, threshold: float) -> Optional[float]:
    mask = values <= threshold
    if not np.any(mask):
        return None
    return float(times[np.argmax(mask)])


def _safe_semilogy(ax, x, y, label, linewidth=1.5):
    """Plot log-y curve, skipping non-positive points."""
    x = np.asarray(x)
    y = np.asarray(y)
    mask = (x > 0) & (y > 0)
    if not np.any(mask):
        return
    ax.semilogy(x[mask], y[mask], label=label, linewidth=linewidth)


def plot_cost_curves(avg_results: Dict[str, Dict], output_path: Optional[str] = None):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 7))

    for label, vals in avg_results.items():
        _safe_semilogy(ax1, vals["time"], vals["error"], label=label)
    ax1.set_title("Average Error Norm vs Wall-clock Time")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel(r"$\ell_2$ Norm of Error")
    ax1.set_ylim(top=1e2)
    ax1.grid(True)

    for label, vals in avg_results.items():
        _safe_semilogy(ax2, vals["time"], vals["residual"], label=label)
    ax2.set_title("Average Residual Norm vs Wall-clock Time")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel(r"$\ell_2$ Norm of Residual")
    ax2.set_ylim(top=1e2)
    ax2.grid(True)
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=3, frameon=True)

    plt.tight_layout()
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True) if os.path.dirname(output_path) else None
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Cost analysis figure saved to: {output_path}")
    else:
        plt.show()


def build_speedup_table(avg_results: Dict[str, Dict], baseline_label: str,
                        thresholds: List[float], metric: str = "error") -> str:
    if baseline_label not in avg_results:
        raise KeyError(f"Baseline '{baseline_label}' not found. Available: {list(avg_results.keys())}")

    baseline_vals = avg_results[baseline_label]
    header = "| Threshold | " + " | ".join(avg_results.keys()) + " |"
    sep    = "|" + "|".join(["---"] * (len(avg_results) + 1)) + "|"
    lines  = [header, sep]

    for thr in thresholds:
        baseline_time = find_first_time(baseline_vals[metric], baseline_vals["time"], thr)
        row = [f"{thr:.0e}"]
        for label, vals in avg_results.items():
            t = find_first_time(vals[metric], vals["time"], thr)
            if t is None or baseline_time is None:
                row.append("—")
            else:
                speedup = baseline_time / t if t > 0 else float("inf")
                row.append(f"{speedup:.2f}x")
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def save_raw_data(avg_results: Dict[str, Dict], output_path: str):
    data_dict = {}
    for label, vals in avg_results.items():
        key = label.replace(" ", "_").replace("(", "").replace(")", "")
        for k in ("iter", "error", "residual", "time"):
            data_dict[f"{key}__{k}"] = vals[k]
    data_path = os.path.splitext(output_path)[0] + "_data.npz"
    np.savez(data_path, **data_dict)
    print(f"Raw data saved to: {data_path}")


if __name__ == "__main__":
    avg_results = run_cost_evaluation()

    plot_cost_curves(avg_results, output_path=OUTPUT_PATH)

    print(f"\nBaseline for speedup: {BASELINE}\n")

    print("=" * 80)
    print("Speedup Table — Error threshold → Time to reach")
    print("=" * 80)
    error_table = build_speedup_table(avg_results, BASELINE, ERROR_THRESHOLDS, metric="error")
    print(error_table)

    print("\n" + "=" * 80)
    print("Speedup Table — Residual threshold → Time to reach")
    print("=" * 80)
    residual_table = build_speedup_table(avg_results, BASELINE, RESIDUAL_THRESHOLDS, metric="residual")
    print(residual_table)

    if SAVE_TABLE_PATH:
        os.makedirs(os.path.dirname(SAVE_TABLE_PATH), exist_ok=True) if os.path.dirname(SAVE_TABLE_PATH) else None
        with open(SAVE_TABLE_PATH, "w", encoding="utf-8") as f:
            f.write(f"# Speedup Analysis (baseline: {BASELINE})\n\n")
            f.write("## Error-based Speedup\n\n")
            f.write(error_table + "\n\n")
            f.write("## Residual-based Speedup\n\n")
            f.write(residual_table + "\n")
        print(f"\nSpeedup table saved to: {SAVE_TABLE_PATH}")

    if OUTPUT_PATH:
        save_raw_data(avg_results, OUTPUT_PATH)
