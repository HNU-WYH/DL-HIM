import os
import copy
import numpy as np
import matplotlib.pyplot as plt

from box import Box
from tqdm import tqdm
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from src.utils.cfg_util import load_config
from src.utils.fdm_utils import expand_solution
from src.solver.hybrid_solver import HybridSolver
from src.utils.visualization import plot_case_predictions
from src.utils.stepin_utils import AndersonAcceleration, AndersonMixing

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# In[]:
# Pre-registered config for unresolved problem and testing data
CONFIG_WILDCARD = "diffusion*"                       # config filename wildcard

TEST_GRID_NUM: Optional[int] = None                    # use dataset grid by default
TEST_DATASET_PATH: Optional[str] = None                # default: cfg["dataset_path"] + "_test.npz"

SAMPLE_INDICES: Optional[Sequence[int]] = np.arange(10)                  # validation indices to evaluate
PLOT_SAMPLE_INDICES: Optional[Sequence[int]] = [8]  # indices in the testing data to visualize

MAX_ITER: Optional[int] = 700                         # Iteration / tolerance applied to every case
TOL: Optional[float] = None                            # by default using the value in the yaml file

# Pre-registered model checkpoints (use the keys inside CASES)
# If Default is None, will raise an error
MODEL_PATHS: Dict[str, Optional[str]] = {
    "Default": "checkpoints/deeponet_diffusion1d/cons/static_error_l2/diffusion_1D_Grid31_Ep20000_2026-01-25.pt",
}

# Evaluation plan: each dict describes one curve on the plot
CASES: List[Dict] = [
    # {"label": "Pure-DeepONet", "mode": "neural",    "model": "Default", "one_shot": True},

    # {"label": "Gauss-Seidel", "mode": "numerical", "model": None, "numerical_method": "gauss-seidel"},

    # {"label": "Jacobi", "mode": "numerical", "model": None, "numerical_method": "jacobi"},

    {"label": "Jacobi-AA", "mode": "numerical", "model": None, "numerical_method": "jacobi",
     "numerical_update": "aa", "aa_m": 20},

    {"label": "Hybrid-fixed", "mode": "hybrid", "model": "Default", "numerical_method": "jacobi",
    "hybrid_ratio": 20, "neural_update": "fixed"},

    {"label": "Hybrid-AA", "mode": "hybrid", "model": "Default", "numerical_method": "jacobi",
     "hybrid_ratio": 20, "neural_update": "aa", "aa_m": 20},

    {"label": "Hybrid-PAAA", "mode": "hybrid", "model": "Default", "numerical_method": "jacobi",
     "hybrid_ratio": 20, "neural_update": "am", "aa_m": 20},

    # {"label": "Hybrid-Adaptive", "mode": "hybrid", "model": "Default", "numerical_method": "jacobi",
    # "hybrid_ratio": 20, "neural_update": "cg"},
]


# In[]:
# =====================
# Helper functions
# =====================
def load_evaluation_dataset(cfg: Box, test_path=TEST_DATASET_PATH) -> Tuple[np.lib.npyio.NpzFile, bool]:
    """Load the preferred evaluation dataset.

    Returns
    -------
    dataset: np.lib.npyio.NpzFile
        The loaded dataset (test preferred, otherwise validation split).
    use_test_dataset: bool
        Whether the returned dataset is the dedicated test set.
    """
    dataset_path = cfg["dataset_path"]

    if test_path is None:
        base, ext = os.path.splitext(dataset_path)
        test_path = f"{base}_test{ext}"

    if os.path.exists(test_path):
        return np.load(test_path), True

    return np.load(dataset_path), False


# 这个进行了大概记得改回来
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
    u_key = "u_data" if use_test_dataset else "u_data_val"

    k = dataset[k_key][test_sample_indices]
    f = dataset[f_key][test_sample_indices]
    u = dataset[u_key][test_sample_indices]
    # a_mats = dataset["a_mats"][test_sample_indices]
    # res = np.mean(f - (a_mats @ u[...,None]).squeeze(-1))

    if grid_num is None:
        x_after = x_before
    else:
        x_after = np.linspace(0, 1, grid_num)

    if grid_num and len(x_before) != len(x_after):
        k = [np.interp(x_after, x_before, k[i]) for i in range(k.shape[0])]
        f = [np.interp(x_after[1:-1], x_before[1:-1], f[i]) for i in range(f.shape[0])]
        u = [np.interp(x_after[1:-1], x_before[1:-1], u[i]) for i in range(u.shape[0])]

    if return_list:
        return np.array(k), np.array(f), x_after, np.array(u)
    else:
        return k[0], f[0], x_after, u[0]


def resolve_model_path(model_key: Optional[str], model_paths=MODEL_PATHS) -> Optional[str]:
    if model_key is None:
        return None
    if model_key not in model_paths:
        raise KeyError(f"Model '{model_key}' not in registered MODEL_PATHS")
    return model_paths[model_key]


def infer_hard_constraints_from_path(model_path: Optional[str]) -> Optional[bool]:
    if not model_path:
        return None
    norm_path = os.path.normpath(model_path)
    cons_token = f"{os.sep}cons{os.sep}"
    nocons_token = f"{os.sep}nocons{os.sep}"
    if nocons_token in norm_path:
        return False
    if cons_token in norm_path:
        return True
    return None


def apply_case_overrides(base_cfg: Box, case: Dict,
                         tol=TOL, max_iter=MAX_ITER) -> Box:
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
        use_hard_cons = case.get("use_hard_cons")
        if use_hard_cons is None:
            use_hard_cons = infer_hard_constraints_from_path(model_path)
        if use_hard_cons is not None:
            cfg.training.don_setting.hard_cons = use_hard_cons
            cfg.training.fns_setting.hard_cons = use_hard_cons

    return cfg


def pad_series(values: List[float], target_len: int) -> np.ndarray:
    """pad the history of residual or error"""
    if not values:
        return np.zeros(target_len)
    if len(values) >= target_len:
        return np.asarray(values[:target_len], dtype=np.float64)
    pad_val = values[-1]
    padded = np.full(target_len, pad_val, dtype=np.float64)
    padded[: len(values)] = values
    return padded


# TODO: 这里改为直接用cfg会不会更好? 每一次迭代都用override的cfg来工作
# TODO: 这里我一会儿又要expand,一会儿又不要,这里也要改
def collect_history(solver: HybridSolver,
                    u_ref_inner: np.ndarray,
                    max_iter: int, tol: float,
                    mode: str, one_shot: bool = False,
                    aa_m: Optional[int] = None,
                    use_cache_residual: bool = True,
                    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run a single case and collect (residual, error) histories.
    """
    mode = mode.lower()
    # u_ref_inner = solver.u_inner
    u_curr = np.zeros_like(u_ref_inner, dtype=np.float64)

    if one_shot:
        u_pred = solver._neural_step(u_curr)
        residual = solver.compute_residual(u_pred)
        res_norm = float(np.linalg.norm(residual, ord=2))
        err_norm = float(np.linalg.norm(u_pred - u_ref_inner, ord=2))
        return np.full(max_iter, res_norm, dtype=np.float64), np.full(max_iter, err_norm, dtype=np.float64), u_pred

    # Initialize Anderson Acceleration if requested
    if solver.neural_update_type == "aa":
        history_size = aa_m or solver.config.solver.hybrid.get("aa_m", 10)
        aa = AndersonAcceleration(m=history_size, reg=1e-20)
    elif solver.neural_update_type == "am":
        history_size = aa_m or solver.config.solver.hybrid.get("aa_m", 10)
        aa = AndersonMixing(m=history_size, reg=1e-20)
    else:
        aa = None

    # --------------------------
    # Recording History
    # --------------------------
    errors: List[float] = []
    residuals: List[float] = []

    # Use cached residual to avoid computing the residual redundantly inside steps
    r_curr = solver.compute_residual(u_curr) if use_cache_residual else None

    for iter_idx in range(max_iter):
        numerical_update = (iter_idx + 1) % solver.hybrid_ratio

        # 1) Produce proposal g_k from the base solver
        apply_anderson = False
        if mode == "numerical":
            g_k = solver._numerical_step(u_curr, residual=r_curr)
            apply_anderson = aa is not None

        elif mode == "deeponet":
            g_k = solver._neural_step(u_curr, residual=r_curr)

        elif mode == "hybrid":
            if numerical_update:
                g_k = solver._numerical_step(u_curr, residual=r_curr)
                apply_anderson = False
            else:
                g_k = solver._neural_step(u_curr, residual=r_curr)
                apply_anderson = aa is not None

        else:
            raise ValueError(f"Unknown mode: {mode}")

        # 2) Apply Anderson with at most ONE matvec per iteration
        if apply_anderson and solver.neural_update_type == "am":
            r_gk = solver.compute_residual(g_k)
            u_next, r_next, _ = aa.compute(g_k, r_gk)

        elif apply_anderson and solver.neural_update_type == "aa":
            u_next = u_curr + aa.compute(u_curr, g_k)
            r_next = solver.compute_residual(u_next)
        else:
            u_next = g_k
            r_next = solver.compute_residual(u_next)

        # 3) Record metrics and advance
        res_norm = float(np.linalg.norm(r_next, ord=2))
        err_norm = float(np.linalg.norm(u_next - u_ref_inner, ord=2))

        residuals.append(res_norm)
        errors.append(err_norm)

        u_curr = u_next
        r_curr = r_next

        if res_norm < tol:
            break

    return pad_series(residuals, max_iter), pad_series(errors, max_iter), u_curr


def evaluate_case_on_sample(base_cfg: Box, case: Dict,
                            k_x: np.ndarray, f_inner: np.ndarray,
                            u_gt_inner: np.ndarray, x_nodes: np.ndarray,
                            max_iter: int = MAX_ITER, tol: float = TOL
                            ) -> Tuple[np.ndarray, np.ndarray]:

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

    res_hist, err_hist, u_curr = collect_history(solver, u_gt_inner,
                                                 mode=case.get("mode", "hybrid"),
                                                 max_iter=max_iter, tol=tol,
                                                 one_shot=case.get("one_shot", False),
                                                 aa_m=case.get("aa_m"),
                                                 )
    return err_hist, res_hist, u_curr


def average_histories(histories: Iterable[np.ndarray]) -> np.ndarray:
    stacked = np.stack(list(histories), axis=0)
    return stacked.mean(axis=0)


def run_evaluation(plot_indices: Optional[Sequence[int]] = None,
                   config_wildcard: str = CONFIG_WILDCARD,
                   test_dataset_path: str = TEST_DATASET_PATH,
                   sample_indices: Sequence[int] = SAMPLE_INDICES,
                   test_grid_num: int = TEST_GRID_NUM, cases=CASES,
                   ):
    # load config and overrides the validation dataset (act as testing if testing data is not provided)
    cfg = load_config(config_wildcard)

    # load data, if testing data path (training_path + "_test") exist, load testing data
    data, use_test_dataset = load_evaluation_dataset(cfg, test_dataset_path)

    # If not providing the sample indices, using all data in validation/testing
    available_indices = list(range(len(data["k_data"] if use_test_dataset else data["k_data_val"])))
    sample_indices = sample_indices if sample_indices is not None else available_indices
    if any(idx not in available_indices for idx in sample_indices):
        raise IndexError("Sample indices exceed available evaluation data")

    # If not providing the plotting indices, do not plotting any data
    plot_indices = list(plot_indices) if plot_indices is not None else []
    if len(sample_indices) < len(plot_indices):
        raise IndexError("Plot sample indices must be smaller than the validation indices")

    # get the unresolved testing data and corresponding reference solution
    k_val, f_val, x_val, u_val = select_test_sample(grid_num=test_grid_num,
                                                    dataset=data, test_sample_indices=sample_indices,
                                                    use_test_dataset=use_test_dataset)

    # Initialize the result for loss, error and iters
    case_results = {case["label"]: {"errors": [], "residuals": [], "iters": None} for case in cases}

    # Initialize the prediction for plotting the results of selected indices under difference cases
    plot_predictions = {idx: {} for idx in plot_indices}

    # Evaluate every sample in the sampled indices
    for sample_idx, _ in enumerate(tqdm(sample_indices, desc="Samples")):
        for case in cases:
            err_hist, res_hist, u_curr = evaluate_case_on_sample(cfg, case,
                                                                 k_val[sample_idx], f_val[sample_idx],
                                                                 u_val[sample_idx], x_val,
            )

            case_results[case["label"]]["errors"].append(err_hist)
            case_results[case["label"]]["residuals"].append(res_hist)
            case_results[case["label"]]["iters"] = np.arange(1, len(err_hist) + 1)

            if sample_idx in plot_predictions:
                plot_predictions[sample_idx][case["label"]] = {"u_true": u_val[sample_idx], "u_pred": u_curr,
                                                               "x_nodes": x_val[1:-1], "error": err_hist[-1]}

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
    plt.title("Average Error vs Iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Error L2 Norm")
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    for label, vals in avg_results.items():
        plt.semilogy(vals["iter"], vals["residual"], label=label)
    plt.title("Average Residual vs Iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Residual L-2 Norm")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

    if plot_predictions:
        plot_case_predictions(plot_predictions)

    return avg_results


if __name__ == "__main__":
    avg_results3 = run_evaluation(plot_indices=PLOT_SAMPLE_INDICES)
    #
    plt.figure(figsize=(6, 3.8))
    for label, vals in avg_results3.items():
        if label == "Hybrid-CG":
            label = "Hybrid-Adaptive"
        plt.semilogy(vals["iter"], vals["residual"], label=label)
    plt.title("Average Residual Norm vs Iteration")
    plt.xlabel("DL-HIM Iteration")
    plt.ylabel("$\ell_2$ Norm of Residual")
    plt.grid(True)
    plt.legend(
        loc='upper center',  # 图例自己的对齐点（上边缘居中）
        bbox_to_anchor=(0.5, -0.2),  # 图例相对于坐标轴的位置 (x=0.5居中, y=-0.15在轴下方)
        ncol=3,  # 设置列数，建议设为3或5，让图例横向排列更美观
        frameon=True  # 是否显示图例边框
    )

    plt.tight_layout()
    plt.show()