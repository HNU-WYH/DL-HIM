"""
================================================================================
Compute / Cost Analysis Script for DL-HIM
================================================================================

作用：
    评估各求解器（HINTS-Fixed / AA / PAAA / ELS × Jacobi / GS）的
    时间-精度权衡（time-to-solution）与加速比。

推荐运行方式（从项目根目录执行）：
    python scripts/compute_cost_analysis.py \
        --config helmholtz* \
        --output supplement/costs_mem_study/helmholtz_cost.pdf \
        --save-table supplement/costs_mem_study/helmholtz_speedup.md

    如果想对 diffusion 做同样分析，把 helmholtz* 换成 diffusion* 即可。

    也可以指定 baseline（默认使用 CASES 里的第一个 case 作为 baseline）：
    python scripts/compute_cost_analysis.py \
        --config helmholtz* \
        --baseline "HINTS-Fixed (Jacobi)" \
        --output supplement/costs_mem_study/helmholtz_cost.pdf \
        --save-table supplement/costs_mem_study/helmholtz_speedup.md

生成的文件：
    1. <output> 指定的图片（如 helmholtz_cost.pdf）
       - 上下两张子图：Error vs Time、Residual vs Time
    2. <output 前缀>_data.npz（如 helmholtz_cost_data.npz）
       - 可复现绘图的原始数据：iter、error、residual、time
    3. --save-table 指定的 markdown 文件（如 helmholtz_speedup.md）
       - 基于 baseline 的 speedup 表格（error / residual 双表）

终端输出：
    - 运行进度（evaluate_single_solver 自带的 tqdm）
    - Baseline 信息
    - Error-based Speedup 表
    - Residual-based Speedup 表
    - 各保存文件的路径

依赖：
    - 会自动调用 scripts.evaluate_single_solver.run_evaluation() 重新跑所有
      CASES 中的 case，并收集 wall-clock time。因此无需事先手动跑
      evaluate_single_solver.py。
================================================================================
"""
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional

from scripts.evaluate_single_solver import run_evaluation, CASES

# =============================================================================
# Analysis thresholds for time-to-solution
# =============================================================================
ERROR_THRESHOLDS = [1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
RESIDUAL_THRESHOLDS = [1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]


def find_first_time(values: np.ndarray, times: np.ndarray, threshold: float) -> Optional[float]:
    """Return the first time point where values <= threshold."""
    mask = values <= threshold
    if not np.any(mask):
        return None
    return float(times[np.argmax(mask)])


def plot_cost_curves(avg_results: Dict[str, Dict], output_path: Optional[str] = None):
    """Plot averaged Error and Residual against wall-clock time."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 7))

    for label, vals in avg_results.items():
        ax1.semilogy(vals["time"], vals["error"], label=label, linewidth=1.5)
    ax1.set_title("Average Error Norm vs Wall-clock Time")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel(r"$\ell_2$ Norm of Error")
    ax1.set_ylim(top=1e2)
    ax1.grid(True)

    for label, vals in avg_results.items():
        ax2.semilogy(vals["time"], vals["residual"], label=label, linewidth=1.5)
    ax2.set_title("Average Residual Norm vs Wall-clock Time")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel(r"$\ell_2$ Norm of Residual")
    ax2.set_ylim(top=1e2)
    ax2.grid(True)
    ax2.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, -0.20),
        ncol=4,
        frameon=True
    )

    plt.tight_layout()
    if output_path:
        if os.path.dirname(output_path):
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Cost analysis figure saved to: {output_path}")
    else:
        plt.show()


def build_speedup_table(
    avg_results: Dict[str, Dict],
    baseline_label: str,
    thresholds: List[float],
    metric: str = "error",
) -> str:
    """Build a markdown-style speedup table relative to a baseline case."""
    if baseline_label not in avg_results:
        raise KeyError(f"Baseline '{baseline_label}' not found in results. Available: {list(avg_results.keys())}")

    baseline_vals = avg_results[baseline_label]
    lines = []
    header = f"| Threshold | {' | '.join(avg_results.keys())} |"
    sep = "|" + "|".join(["---"] * (len(avg_results) + 1)) + "|"
    lines.append(header)
    lines.append(sep)

    for thr in thresholds:
        baseline_time = find_first_time(baseline_vals[metric], baseline_vals["time"], thr)
        row = [f"{thr:.0e}"]
        for label, vals in avg_results.items():
            t = find_first_time(vals[metric], vals["time"], thr)
            if t is None or baseline_time is None:
                row.append("—")
            else:
                speedup = baseline_time / t if t > 0 else float('inf')
                row.append(f"{speedup:.2f}x")
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def save_raw_data(avg_results: Dict[str, Dict], output_path: str):
    """Save averaged curves to an .npz file for later replotting."""
    data_dict = {}
    for label, vals in avg_results.items():
        safe_label = label.replace(" ", "_").replace("(", "").replace(")", "")
        data_dict[f"{safe_label}__iter"] = vals["iter"]
        data_dict[f"{safe_label}__error"] = vals["error"]
        data_dict[f"{safe_label}__residual"] = vals["residual"]
        data_dict[f"{safe_label}__time"] = vals["time"]
    data_path = os.path.splitext(output_path)[0] + "_data.npz"
    np.savez(data_path, **data_dict)
    print(f"Raw data saved to: {data_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute and visualize cost (time-to-solution) analysis for DL-HIM."
    )
    parser.add_argument("--config", type=str, default=None,
                        help="Config filename wildcard, e.g. 'diffusion*' or 'helmholtz*'.")
    parser.add_argument("--grid", type=int, default=None,
                        help="Override test grid resolution.")
    parser.add_argument("--output", type=str, default=None,
                        help="Output figure path, e.g. 'results/cost_analysis.pdf'.")
    parser.add_argument("--baseline", type=str, default=None,
                        help="Label of the baseline case for speedup calculation. "
                             "If omitted, the first case in CASES is used.")
    parser.add_argument("--save-table", type=str, default=None,
                        help="Optional path to save the speedup markdown table, e.g. 'results/speedup.md'.")
    args = parser.parse_args()

    # -------------------------------------------------------------------------
    # Run evaluation (this collects averaged error/residual/time histories)
    # -------------------------------------------------------------------------
    avg_results = run_evaluation(
        plot_indices=None,  # no per-sample prediction plots needed for cost analysis
        config_wildcard=args.config,
        test_grid_num=args.grid,
    )

    # -------------------------------------------------------------------------
    # Plot Error / Residual vs Wall-clock Time
    # -------------------------------------------------------------------------
    plot_cost_curves(avg_results, output_path=args.output)

    # -------------------------------------------------------------------------
    # Speedup table (relative to baseline)
    # -------------------------------------------------------------------------
    baseline = args.baseline or list(avg_results.keys())[0]
    print(f"\nBaseline for speedup: {baseline}\n")

    print("=" * 80)
    print("Speedup Table (Error threshold -> Time to reach)")
    print("=" * 80)
    error_table = build_speedup_table(avg_results, baseline, ERROR_THRESHOLDS, metric="error")
    print(error_table)

    print("\n" + "=" * 80)
    print("Speedup Table (Residual threshold -> Time to reach)")
    print("=" * 80)
    residual_table = build_speedup_table(avg_results, baseline, RESIDUAL_THRESHOLDS, metric="residual")
    print(residual_table)

    # -------------------------------------------------------------------------
    # Optional: save table and raw data
    # -------------------------------------------------------------------------
    if args.save_table:
        if os.path.dirname(args.save_table):
            os.makedirs(os.path.dirname(args.save_table), exist_ok=True)
        with open(args.save_table, "w", encoding="utf-8") as f:
            f.write(f"# Speedup Analysis (baseline: {baseline})\n\n")
            f.write("## Error-based Speedup\n\n")
            f.write(error_table + "\n\n")
            f.write("## Residual-based Speedup\n\n")
            f.write(residual_table + "\n")
        print(f"\nSpeedup table saved to: {args.save_table}")

    if args.output:
        save_raw_data(avg_results, args.output)


if __name__ == "__main__":
    main()
