import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from src.utils.cfg_util import load_config
from src.utils.fdm_utils import expand_solution
from src.solver.hybrid_solver import HybridSolver

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# =====================
# Configuration
# =====================
CONFIG_WILDCARD = "diffusion*"

MODEL_TYPE = "DeepONet"
MODEL_PATH = "checkpoints/deeponet_diffusion1d/cons/static_error_l2/diffusion_1D_Grid31_Ep20000_2026-01-25.pt"

TEST_GRID_NUM = 801  #
SAMPLE_INDEX = 8
MAX_ITER_CYCLES = 36


def run_analysis():
    cfg = load_config(CONFIG_WILDCARD)
    cfg["model_load_path"] = MODEL_PATH
    cfg.training.operator_type = MODEL_TYPE

    base, ext = os.path.splitext(cfg["dataset_path"])
    test_path = f"{base}_test{ext}"
    data = np.load(test_path if os.path.exists(test_path) else cfg["dataset_path"])

    x_orig = data["x_data"]
    x_fine = np.linspace(0, 1, TEST_GRID_NUM)

    u_key = "u_data" if os.path.exists(test_path) else "u_data_val"
    k_key = "k_data" if os.path.exists(test_path) else "k_data_val"
    f_key = "f_data" if os.path.exists(test_path) else "f_data_val"

    k_x = np.interp(x_fine, x_orig, data[k_key][SAMPLE_INDEX])
    f_inner = np.interp(x_fine[1:-1], x_orig[1:-1], data[f_key][SAMPLE_INDEX])
    u_ref = np.interp(x_fine[1:-1], x_orig[1:-1], data[u_key][SAMPLE_INDEX])

    cycle_ratio = cfg.solver.hybrid.update_ratio

    solver = HybridSolver(cfg, k_x=k_x, f_x=expand_solution(f_inner),
                          prob_x_nodes=x_fine, cp_path=MODEL_PATH)

    # ---------------------------------------------------------
    # DL-HIM (Hybrid Loop)
    # ---------------------------------------------------------
    u_hybrid = np.zeros_like(u_ref)
    hybrid_delta = []
    hybrid_res = []
    # hybrid_err = []

    print(f"Running Hybrid cycles (Cycle = 1 Neural + {cycle_ratio - 1} Jacobi)...")
    for _ in tqdm(range(MAX_ITER_CYCLES), desc="Hybrid Cycles"):
        # A. 记录当前残差与误差 (物理层面)
        res = np.linalg.norm(solver.compute_residual(u_hybrid), ord=2)
        # err = np.linalg.norm(u_ref - u_hybrid, ord=2)
        hybrid_res.append(res)
        # hybrid_err.append(err)

        u_next = solver.solve(u_hybrid, max_iter=cycle_ratio)[0]

        delta = np.linalg.norm(u_next - u_hybrid, ord=2)
        hybrid_delta.append(delta)
        u_hybrid = u_next


    # u_jacobi = np.zeros_like(u_ref)
    # jacobi_res = []
    #
    # print("Running Pure Jacobi baseline...")
    # for i in tqdm(range(MAX_ITER_CYCLES * cycle_ratio), desc="Jacobi Steps"):
    #     if i % cycle_ratio == 0:
    #         jacobi_res.append(np.linalg.norm(solver.compute_residual(u_jacobi), ord=2))
    #
    #     u_jacobi = solver._numerical_step(u_jacobi)

    iters = np.arange(MAX_ITER_CYCLES)
    fig, ax1 = plt.subplots(figsize=(6, 4))

    color_delta = 'tab:blue'
    ax1.set_xlabel('Iteration') #, fontsize=16)
    ax1.set_ylabel(r'Hybrid Update Norm $\|\delta_k\| = \|u_{k+1} - u_k\|_2$', color=color_delta) #, fontsize=16)
    ax1.semilogy(20*iters, hybrid_delta, '-', color=color_delta, label='Update Norm ($\|\delta_k\|$)') #, alpha=0.6)
    ax1.tick_params(axis='y', labelcolor=color_delta)
    ax1.tick_params(axis='both') #, labelsize=12)
    ax1.grid(True)

    ax2 = ax1.twinx()
    ax2.set_ylabel('Residual Norms $\|r_k\|=\|f-Au_k\|$', color='r') #, fontsize=16)
    ax2.set_ylim([0.5*1e-3, 0.9*1e2])
    ax2.semilogy(20*iters, hybrid_res, 'r-', label='Physical Residual ($\|r_k\|$)', alpha=0.8) #, linewidth=2.0)
    # ax2.semilogy(iters, jacobi_res, 'r:', linewidth=2.0, label='Pure Jacobi Residual')
    # ax2.semilogy(iters, hybrid_err, color='gray', linestyle='-.', linewidth=1.5, label='Hybrid Error to Ref', alpha=0.5)
    ax2.tick_params(axis='y', labelcolor='r') #, labelsize=12)

    plt.title(f'the False Fixed Point of DL-HIMs') #, fontsize=18)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right') #, fontsize=16)

    fig.tight_layout()
    os.makedirs("figure", exist_ok=True)
    save_path = "figure/false_fixed_point.png"
    plt.savefig(save_path, dpi=300)
    print(f"Comparison plot saved to {save_path}")
    plt.show()


if __name__ == "__main__":
    run_analysis()