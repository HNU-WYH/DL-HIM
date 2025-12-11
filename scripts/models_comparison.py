import os
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm
from typing import Optional

from src.utils.cfg_util import load_config
from scripts.evaluate_single_solver import evaluate_case_on_sample, load_evaluation_dataset, select_test_sample


# In[]:
CONFIG_WILDCARD = "diffusion1d*"
CHECKPOINT_ROOT = "checkpoints/diffusion1d"

TEST_GRID_NUM: Optional[int] = None                    # if not equal, interpolate to uniformly spaced TEST_GRID_NUM
TEST_DATASET_PATH: Optional[str] = None                # path to .npz test dataset; None -> derive from the dataset path

SAMPLE_INDICES: Optional[Iterable[int]] = np.arange(10)
PLOT_SAMPLE_INDICES: Optional[Iterable[int]] = [3, 5]

MAX_ITER: Optional[int] = None                         # Iteration / tolerance applied to every case
TOL: Optional[float] = None                            # by default using the value in the yaml file


# In[]:
def discover_checkpoints(root: str = CHECKPOINT_ROOT) -> List[Tuple[str, str]]:
    """Return (label, path) pairs for checkpoints under the root directory."""
    checkpoint_pairs: List[Tuple[str, str]] = []
    if not os.path.isdir(root):
        raise FileNotFoundError(f"Checkpoint root '{root}' does not exist")

    for family in sorted(os.listdir(root)):
        family_path = os.path.join(root, family)
        if not os.path.isdir(family_path):
            continue

        for candidate in sorted(os.listdir(family_path)):
            candidate_path = os.path.join(family_path, candidate)
            if candidate_path.endswith(".pt"):
                # label = os.path.join(family, candidate)
                checkpoint_pairs.append((family, candidate_path))
    return checkpoint_pairs


def build_case(label: str, model_path: str) -> Dict:
    return {
        "label": label,
        "mode": "hybrid",
        "numerical_method": "jacobi",
        "hybrid_ratio": 20,
        "neural_update": "fixed",
        "aa_m": 15,
        "model_path": model_path,
    }


def average_histories(histories: Iterable[np.ndarray]) -> np.ndarray:
    stacked = np.stack(list(histories), axis=0)
    return stacked.mean(axis=0)


def compare_checkpoints(plot_indices):
    # load config and data
    cfg = load_config(CONFIG_WILDCARD)
    data, use_test_dataset = load_evaluation_dataset(cfg, TEST_DATASET_PATH)

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
    k_samples, f_samples, x_nodes, u_samples = select_test_sample(
        TEST_GRID_NUM or len(data["x_data"]), data, sample_indices, use_test_dataset=use_test_dataset
    )

    # 后面我不知道该怎么改了, codex你来改吧
    checkpoint_cases = [build_case(label, path) for label, path in discover_checkpoints()]
    if not checkpoint_cases:
        raise RuntimeError(f"No checkpoints discovered under '{CHECKPOINT_ROOT}'")

    case_results = {case["label"]: {"errors": [], "residuals": [], "iters": None} for case in checkpoint_cases}
    plot_predictions = {idx: {} for idx in plot_indices}

    for case in tqdm(checkpoint_cases, desc="Checkpoints"):
        for k_sample, f_sample, u_sample in zip(k_samples, f_samples, u_samples):
            err_hist, res_hist, u_curr = evaluate_case_on_sample(cfg, case, k_sample, f_sample, u_sample, x_nodes,)

            case_results[case["label"]]["errors"].append(err_hist)
            case_results[case["label"]]["residuals"].append(res_hist)
            case_results[case["label"]]["iters"] = np.arange(1, len(err_hist) + 1)

    avg_results = {}
    for label, store in case_results.items():
        avg_err = average_histories(store["errors"])
        avg_res = average_histories(store["residuals"])
        avg_results[label] = {"iter": store["iters"], "error": avg_err, "residual": avg_res}

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    for label, vals in avg_results.items():
        plt.semilogy(vals["iter"], vals["error"], label=label)
    plt.title("Relative Error vs Iteration (dataset average)")
    plt.xlabel("Iteration")
    plt.ylabel("Relative L2 Error")
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    for label, vals in avg_results.items():
        plt.semilogy(vals["iter"], vals["residual"], label=label)
    plt.title("Residual vs Iteration (single sample)")
    plt.xlabel("Iteration")
    plt.ylabel("Residual L-2 Norm")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    compare_checkpoints(plot_indices=PLOT_SAMPLE_INDICES)