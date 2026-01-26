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
# 使用 DeepONet 模型，它在处理高分辨率网格时的频谱偏差最容易产生伪不动点
MODEL_PATH = "checkpoints/deeponet_diffusion1d/cons/static_error_l2/diffusion_1D_Grid31_Ep20000_2026-01-25.pt"

TEST_GRID_NUM = 801  # 高分辨率网格以放大频谱鸿沟
SAMPLE_INDEX = 8  # 选取典型样本
MAX_ITER_CYCLES = 100  # 混合周期的总数


def run_analysis():
    # 1. 加载配置与数据
    cfg = load_config(CONFIG_WILDCARD)
    cfg["model_load_path"] = MODEL_PATH

    base, ext = os.path.splitext(cfg["dataset_path"])
    test_path = f"{base}_test{ext}"
    data = np.load(test_path if os.path.exists(test_path) else cfg["dataset_path"])

    # 2. 数据准备与插值 (参考单算例评估逻辑)
    x_orig = data["x_data"]
    x_fine = np.linspace(0, 1, TEST_GRID_NUM)

    # 获取 Key
    u_key = "u_data" if os.path.exists(test_path) else "u_data_val"
    k_key = "k_data" if os.path.exists(test_path) else "k_data_val"
    f_key = "f_data" if os.path.exists(test_path) else "f_data_val"

    # 插值到细网格
    k_x = np.interp(x_fine, x_orig, data[k_key][SAMPLE_INDEX])
    f_inner = np.interp(x_fine[1:-1], x_orig[1:-1], data[f_key][SAMPLE_INDEX])
    u_ref = np.interp(x_fine[1:-1], x_orig[1:-1], data[u_key][SAMPLE_INDEX])

    # 周期比例 (如 19 步 Jacobi + 1 步 Neural)
    cycle_ratio = cfg.solver.hybrid.update_ratio

    # 3. 初始化求解器
    solver = HybridSolver(cfg, k_x=k_x, f_x=expand_solution(f_inner),
                          prob_x_nodes=x_fine, cp_path=MODEL_PATH)

    # ---------------------------------------------------------
    # 4. DL-HIM 混合迭代回路 (Hybrid Loop)
    # ---------------------------------------------------------
    u_hybrid = np.zeros_like(u_ref)
    hybrid_delta = []
    hybrid_res = []
    hybrid_err = []

    print(f"Running Hybrid cycles (Cycle = 1 Neural + {cycle_ratio - 1} Jacobi)...")
    for _ in tqdm(range(MAX_ITER_CYCLES), desc="Hybrid Cycles"):
        # A. 记录当前残差与误差 (物理层面)
        res = np.linalg.norm(solver.compute_residual(u_hybrid), ord=2)
        # err = np.linalg.norm(u_ref - u_hybrid, ord=2)
        hybrid_res.append(res)
        # hybrid_err.append(err)

        # B. 执行一个完整周期: u_{k+1} = G(u_k)
        # 根据你的反馈，solve(max_iter=ratio) 会跑完一个完整的 Jacobi + Neural 周期
        u_next = solver.solve(u_hybrid, max_iter=cycle_ratio)[0]

        # C. 记录更新量 (数学层面: 不动点残差)
        delta = np.linalg.norm(u_next - u_hybrid, ord=2)
        hybrid_delta.append(delta)
        u_hybrid = u_next

    # # ---------------------------------------------------------
    # # 5. 纯 Jacobi 迭代回路 (Baseline)
    # # ---------------------------------------------------------
    # u_jacobi = np.zeros_like(u_ref)
    # jacobi_res = []
    #
    # print("Running Pure Jacobi baseline...")
    # # 为了在图中对齐，Jacobi 同样按照周期的频率记录残差
    # for i in tqdm(range(MAX_ITER_CYCLES * cycle_ratio), desc="Jacobi Steps"):
    #     if i % cycle_ratio == 0:
    #         jacobi_res.append(np.linalg.norm(solver.compute_residual(u_jacobi), ord=2))
    #
    #     # 仅执行单步 Jacobi 物理平滑
    #     u_jacobi = solver._numerical_step(u_jacobi)

    # ---------------------------------------------------------
    # 6. 绘图 (双 Y 轴对比)
    # ---------------------------------------------------------
    iters = np.arange(MAX_ITER_CYCLES)
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # 左轴: 混合迭代的更新量 Delta
    # 展示 $\|\delta_k\| \to 0$ 诱导的数学收敛假象
    color_delta = 'tab:blue'
    ax1.set_xlabel('DL-HIM Cycle', fontsize=16)
    ax1.set_ylabel(r'Hybrid Update Norm $\|\delta_k\| = \|u_{k+1} - u_k\|_2$', color=color_delta, fontsize=16)
    ax1.semilogy(iters, hybrid_delta, 'o-', color=color_delta, label='Update Norm ($\|\delta_k\|$)', alpha=0.6)
    ax1.tick_params(axis='y', labelcolor=color_delta)
    ax1.tick_params(axis='both', labelsize=12)
    ax1.grid(True, which="both", ls="-", alpha=0.2)

    # 右轴: 物理残差对比与真实误差
    # 展示即使更新量消失，物理误差依然很大且高于 Jacobi 某些阶段
    ax2 = ax1.twinx()
    ax2.set_ylabel('Residual Norms $\|r_k\|=\|f-Au_k\|$', color='r', fontsize=16)
    ax2.set_ylim([0.5*1e-3, 0.9*1e2])
    ax2.semilogy(iters, hybrid_res, 'r-', linewidth=2.0, label='Physical Residual ($\|r_k\|$)', alpha=0.8)
    # ax2.semilogy(iters, jacobi_res, 'r:', linewidth=2.0, label='Pure Jacobi Residual')
    # ax2.semilogy(iters, hybrid_err, color='gray', linestyle='-.', linewidth=1.5, label='Hybrid Error to Ref', alpha=0.5)
    ax2.tick_params(axis='y', labelcolor='r', labelsize=12)

    plt.title(f'the False Fixed Point of DL-HIMs on 1D Diffusion Equation)',
              fontsize=18)

    # 图例合并
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=16)

    fig.tight_layout()
    os.makedirs("figure", exist_ok=True)
    save_path = "figure/false_fixed_point_vs_jacobi.png"
    plt.savefig(save_path, dpi=300)
    print(f"Comparison plot saved to {save_path}")
    plt.show()


if __name__ == "__main__":
    run_analysis()