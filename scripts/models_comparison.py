import os
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm
from typing import Optional

from scripts.evaluate_single_solver import (
    CONFIG_WILDCARD,
    TEST_GRID_NUM,
    apply_global_overrides,
    evaluate_case_on_sample,
    load_config,
    load_evaluation_dataset,
    select_test_sample,
)

CHECKPOINT_ROOT = "checkpoints/diffusion1d"
SAMPLE_INDICES: Optional[Iterable[int]] = np.arange(10)

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
        "neural_update": "cg",
        "aa_m": 15,
        "model_path": model_path,
    }


def average_histories(histories: Iterable[np.ndarray]) -> np.ndarray:
    stacked = np.stack(list(histories), axis=0)
    return stacked.mean(axis=0)


def compare_checkpoints():
    cfg = apply_global_overrides(load_config(CONFIG_WILDCARD))
    data, use_test_dataset = load_evaluation_dataset(cfg)

    k_key = "k_data" if use_test_dataset else "k_data_val"
    sample_indices = list(SAMPLE_INDICES) if SAMPLE_INDICES is not None else list(range(len(data[k_key])))

    k_samples, f_samples, x_nodes, u_samples = select_test_sample(
        TEST_GRID_NUM or len(data["x_data"]), data, sample_indices, use_test_dataset=use_test_dataset
    )

    checkpoint_cases = [build_case(label, path) for label, path in discover_checkpoints()]

    if not checkpoint_cases:
        raise RuntimeError(f"No checkpoints discovered under '{CHECKPOINT_ROOT}'")

    case_results = {case["label"]: {"errors": [], "residuals": [], "iters": None} for case in checkpoint_cases}

    for case in tqdm(checkpoint_cases, desc="Checkpoints"):
        for k_sample, f_sample, u_sample in zip(k_samples, f_samples, u_samples):
            err_hist, res_hist = evaluate_case_on_sample(
                cfg,
                case,
                k_sample,
                f_sample,
                u_sample,
                x_nodes,
            )

            case_results[case["label"]]["errors"].append(err_hist)
            case_results[case["label"]]["residuals"].append(res_hist)
            case_results[case["label"]]["iters"] = np.arange(1, len(err_hist) + 1)

    avg_results = {}
    for label, store in case_results.items():
        avg_err = average_histories(store["errors"])
        avg_res = average_histories(store["residuals"])
        avg_results[label] = {"iter": store["iters"], "error": avg_err, "residual": avg_res}

    plt.figure(figsize=(6, 5))
    # plt.subplot(1, 2, 1)
    for label, vals in avg_results.items():
        plt.semilogy(vals["iter"], vals["error"], label=label)
    plt.title("Relative Error vs Iteration (dataset average)")
    plt.xlabel("Iteration")
    plt.ylabel("Relative L2 Error")
    plt.grid(True)
    plt.legend()

    # plt.subplot(1, 2, 2)
    # for label, vals in avg_results.items():
    #     plt.semilogy(vals["iter"], vals["residual"], label=label)
    # plt.title("Residual vs Iteration (single sample)")
    # plt.xlabel("Iteration")
    # plt.ylabel("Residual (L-inf)")
    # plt.grid(True)
    # plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    compare_checkpoints()