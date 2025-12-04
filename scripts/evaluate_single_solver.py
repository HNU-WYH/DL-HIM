import argparse
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm

from src.solver.hybrid_solver import HybridSolver
from src.utils.cfg_util import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a single solver mode")
    parser.add_argument(
        "--config", default="diffusion1d*", help="Wildcard for the config file name."
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Optional path to the dataset (.npz). Overrides the path in the config.",
    )
    parser.add_argument(
        "--mode",
        default=None,
        choices=["numerical", "deeponet", "hybrid"],
        help="Solver mode to evaluate. Defaults to the value in the config.",
    )
    parser.add_argument(
        "--operator-type",
        default=None,
        help="Override the operator type (e.g., DeepONet, FNS).",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Override the neural operator checkpoint path.",
    )
    parser.add_argument(
        "--step-method",
        default=None,
        choices=["jacobi", "gauss-seidel", "gauss_seidel", "gs", "g-s"],
        help="Override the numerical smoother used in the solver.",
    )
    parser.add_argument(
        "--hybrid-ratio",
        type=int,
        default=None,
        help="Override the ratio of numerical to neural updates in hybrid mode.",
    )
    parser.add_argument(
        "--neural-update",
        default=None,
        choices=["fixed", "cg", "aa"],
        help="Override the neural update strategy.",
    )
    parser.add_argument(
        "--max-iter", type=int, default=None, help="Maximum iterations for the solver."
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=None,
        help="Residual tolerance for convergence.",
    )
    parser.add_argument(
        "--aa-m",
        type=int,
        default=None,
        help="Anderson acceleration history size (used when neural-update is 'aa').",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Number of validation samples to evaluate. Defaults to the entire split.",
    )
    return parser.parse_args()


def pad_rhs(f_inner: np.ndarray) -> np.ndarray:
    """Pad inner RHS values with zero Dirichlet boundaries to full length."""
    f_full = np.zeros(len(f_inner) + 2, dtype=np.float32)
    f_full[1:-1] = f_inner
    return f_full


def build_solver(
    cfg,
    k_x: np.ndarray,
    f_inner: np.ndarray,
    x_nodes: np.ndarray,
    model_path: str = None,
) -> HybridSolver:
    f_full = pad_rhs(f_inner)
    return HybridSolver(
        cfg,
        k_x=k_x,
        f_x=f_full,
        eps=cfg.data.get("eps", 1.0),
        prob_x_nodes=x_nodes,
        cp_path=model_path,
    )


def evaluate_mode(
    cfg,
    mode: str,
    dataset: Dict[str, np.ndarray],
    x_nodes: np.ndarray,
    model_path: str,
    max_iter: int = None,
    tol: float = None,
    aa_m: int = None,
    limit: int = None,
) -> Tuple[List[float], List[float]]:
    errors, residuals = [], []
    u_gt = dataset["u_data_val"]
    k_batch = dataset["k_data_val"]
    f_batch = dataset["f_data_val"]

    total = len(u_gt) if limit is None else min(limit, len(u_gt))
    for idx in tqdm(range(total), desc=f"Evaluating ({mode})"):
        solver = build_solver(cfg, k_batch[idx], f_batch[idx], x_nodes, model_path)
        u_pred, history = solver.solve(
            max_iter=max_iter,
            tol=tol,
            aa_m=aa_m,
            mode=mode,
        )
        err = np.linalg.norm(u_pred - u_gt[idx], ord=2) / np.linalg.norm(u_gt[idx], ord=2)
        errors.append(err)
        if history["residual_norm"]:
            residuals.append(history["residual_norm"][-1])
    return errors, residuals


def main():
    args = parse_args()

    cfg = load_config(args.config)
    if args.dataset:
        cfg["dataset_path"] = args.dataset
    if args.operator_type:
        cfg.training.operator_type = args.operator_type
    if args.step_method:
        cfg.solver.numerical["method"] = args.step_method
    if args.hybrid_ratio:
        cfg.solver.hybrid["update_ratio"] = args.hybrid_ratio
    if args.neural_update:
        cfg.solver.hybrid["neural_update"] = args.neural_update
    if args.mode:
        cfg.solver["type"] = args.mode
    if args.model_path:
        cfg["model_load_path"] = args.model_path

    data = np.load(cfg["dataset_path"])
    x_nodes = data["x_data"]

    target_mode = cfg.solver.get("type", "numerical").lower()
    selected_modes = {"target": target_mode, "baseline": "numerical"}

    print("\nEvaluation settings:")
    print(f"  Mode: {target_mode}")
    print(f"  Operator: {cfg.training.operator_type}")
    print(f"  Model path: {cfg['model_load_path'] if args.model_path else 'from config'}")
    print(f"  Dataset: {cfg['dataset_path']}")

    aa_m = args.aa_m or cfg.solver.hybrid.get("aa_m", None)
    max_iter = args.max_iter or cfg.problem.get("iteration", None)
    tol = args.tol or cfg.problem.get("tolerance", None)

    results = {}
    for label, mode in selected_modes.items():
        errs, res = evaluate_mode(
            cfg,
            mode,
            data,
            x_nodes,
            cfg.get("model_load_path"),
            max_iter=max_iter,
            tol=tol,
            aa_m=aa_m,
            limit=args.num_samples,
        )
        results[label] = {
            "mean_error": float(np.mean(errs)) if errs else float("nan"),
            "std_error": float(np.std(errs)) if errs else float("nan"),
            "mean_residual": float(np.mean(res)) if res else float("nan"),
            "sample_size": len(errs),
        }

    print("\nSummary (relative L2 error):")
    for label, stats in results.items():
        print(
            f"  {label.title():<10} | samples: {stats['sample_size']:<4d} "
            f"| mean: {stats['mean_error']:.4e} | std: {stats['std_error']:.4e} "
            f"| final residual: {stats['mean_residual']:.4e}"
        )


if __name__ == "__main__":
    main()