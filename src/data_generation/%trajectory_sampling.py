import os
import numpy as np
from tqdm import trange

from src.utils.cfg_util import load_config
from utils.vis_util import compute_residual
from src.solver.HINTSolver import HintsSolver
from src.data_generation.generator_1D import DataGenerator


def collect_non_pre_data(config,
                         num_samples: int,
                         num_residual: int,
                         num_iters: int = None,
                         method: str = "NON",
                         save_dir: str = None):
    """
    离线收集 (residual, increment) 对，用于 NON 或 PRE 方案训练 DeepONet。

    Args:
      config:        配置字典，包含 dataset_path, grid_num, iteration_method 等。
      num_samples:   随机样本数量（使用 Gaussian RF 生成的 k/f 数据）。
      num_residual:  每个样本上采集的 Jacobi 迭代步数。
      method:        "NON" 或 "PRE", PRE 就是使用deeponet来获得HINTS trajectory, NON就是用stationary method来获得HINTS trajectory。
      save_dir:      数据保存目录，若 None 则放在 config['dataset_path'] 同级下。

    Returns:
      residuals:  np.ndarray, shape=(num_samples*num_iters, grid_num)
      increments: np.ndarray, same shape, 对应的增量 (u_{i+1}-u_i) 或 D^{-1}r
    """

    # 1) 准备数据：先保证有 k_data_train/f_data_train
    if not os.path.exists(config["dataset_path"]):
        DataGenerator(config).save()
    data = np.load(config["dataset_path"])
    k_all = data["k_data_train"]  # shape = (N_total, grid_num)
    f_all = data["f_data_train"]
    u_all = data["u_data_train"]
    # a_all = data["a_mats_train"]
    # du_all = data["du_data_train"]

    # 只取前 num_samples
    random_indices = np.random.permutation(k_all.shape[0])
    k_all = k_all[random_indices][:num_samples]
    f_all = f_all[random_indices][:num_samples]
    u_all = u_all[random_indices][:num_samples]
    # a_all = a_all[random_indices][:num_samples]
    # du_all = du_all[random_indices][:num_samples]

    # 2) 选择model_path
    model_path = config["newest_model_path"]

    if num_iters is None:
        num_iters = config.get('NUMERICAL_TO_DON_RATIO', 15)
    config['iteration'] = num_iters

    # 2) 预分配列表
    residuals = []
    increments = []
    parameters = []

    # 3) 对每个样本跑 Jacobi 迭代，采集残差和增量
    for samp in trange(num_samples, desc=f"Collecting {method} data"):
        u_true = u_all[samp]
        k_x = k_all[samp]
        f_x = f_all[samp]

        # 用 HintsSolver 构建系统矩阵 A、f_p、边界 idx
        u = np.zeros(config["don_grid_num"])
        solver = HintsSolver(config, k_x, f_x, model_path, u_init=u)
        r = compute_residual(solver.A, solver.f_p, u, solver.non_bc)

        increments.append(u_true - u)
        residuals.append(r)
        parameters.append(k_x)

        for it in range(num_residual - 1):
            if method.upper() == "PRE":
                solver.solve(iteration_method="Hybrid", u_init=u, verbose=False)
                u = solver.u_approx
                r = compute_residual(solver.A, solver.f_p, u, solver.non_bc)

            elif method.upper() == "NON":
                solver.solve(iteration_method="Numerical", u_init=u, verbose=False)
                u = solver.u_approx
                r = compute_residual(solver.A, solver.f_p, u, solver.non_bc)

            else:
                raise NotImplementedError("Only PRE and NON methods are supported")

            increments.append(u_true - u)
            residuals.append(r)
            parameters.append(k_x)

    # 4) 打包为数组
    residuals = np.stack(residuals, axis=0)
    increments = np.stack(increments, axis=0)
    parameters = np.stack(parameters, axis=0)

    func_num = residuals.shape[0]
    training_sample = int(config["train_ratio"] * func_num)
    random_indices = np.random.permutation(residuals.shape[0])

    # 5) 保存到 npz
    if save_dir is None:
        save_dir = os.path.dirname(config["dataset_path"])
    os.makedirs(save_dir, exist_ok=True)
    fn = os.path.join(save_dir,
                      f"{config['problem']}_{config['n_dim']}D_Grid{config['don_grid_num']}_{method.lower()}.npz")
    np.savez(fn,
             x_data=data['x_data'],
             f_data_train=residuals[random_indices][:training_sample],
             k_data_train=parameters[random_indices][:training_sample],
             u_data_train=increments[random_indices][:training_sample],
             f_data_test=residuals[random_indices][training_sample:func_num],
             k_data_test=parameters[random_indices][training_sample:func_num],
             u_data_test=increments[random_indices][training_sample:func_num])
    print(f"Saved {method} data to {fn}")


def smooth_dataset(k_all, f_all, u_all, config, num_smooth):
    residuals, increments, parameters = [], [], []

    for samp in trange(k_all.shape[0], desc="Smoothing"):
        k_x, f_x, u_true = k_all[samp], f_all[samp], u_all[samp]

        u = np.zeros(config["don_grid_num"])
        solver = HintsSolver(config, k_x, f_x, model_path=None, u_init=u, numerical_only=True)
        solver.solve(iteration_method="Numerical", u_init=u, verbose=False)
        u_smooth = solver.u_approx
        r = compute_residual(solver.A, solver.f_p, u_smooth, solver.non_bc)

        residuals.append(r)
        increments.append(u_true - u_smooth)
        parameters.append(k_x)

    return np.stack(residuals), np.stack(increments), np.stack(parameters)


def collect_presmooth_data(config,
                           num_smooth: int = None,
                           save_dir: str = None,
                           dataset_path: str = None):
    """
    对data进行jacobi presmooth，然后得到presmooth之后的dataset

    Args:
      config:            配置字典，包含 dataset_path, grid_num, iteration_method 等。
      num_smooth:        Jacobi Smooth的次数.
      save_dir:          数据保存目录，若 None 则放在 config['dataset_path'] 同级下。
      dataset_path:      dataset的path.

    Returns:
      residuals:  np.ndarray, shape=(num_samples*num_iters, grid_num)
      increments: np.ndarray, same shape, 对应的增量 (u_{i+1}-u_i) 或 D^{-1}r
    """

    # 1) 准备数据：先保证有 k_data_train/f_data_train
    if dataset_path is not None:
        config["dataset_path"] = dataset_path

    if not os.path.exists(config["dataset_path"]):
        DataGenerator(config).save()

    data = np.load(config["dataset_path"])

    # 2). 准备原始数据, 用于切割
    k_train, f_train, u_train = data["k_data_train"], data["f_data_train"], data["u_data_train"]
    k_test, f_test, u_test = data["k_data_test"], data["f_data_test"], data["u_data_test"]

    if num_smooth is None:
        num_smooth = config.get('NUMERICAL_TO_DON_RATIO', 20)
    config['iteration'] = num_smooth

    print("Smoothing training set...")
    f_train, u_train, k_train = smooth_dataset(k_train, f_train, u_train, config, num_smooth)

    print("Smoothing test set...")
    f_val, u_val, k_val = smooth_dataset(k_test, f_test, u_test, config, num_smooth)

    # 3) 保存到 npz
    if save_dir is None:
        save_dir = os.path.dirname(config["dataset_path"])
    os.makedirs(save_dir, exist_ok=True)

    filename = os.path.splitext(config["dataset_path"])
    filename = "".join([filename[0], "_smooth", filename[1]])
    np.savez(filename,
             x_data=data['x_data'],
             f_data_train=f_train,
             k_data_train=k_train,
             u_data_train=u_train,
             f_data_val=f_val,
             k_data_val=k_val,
             u_data_val=u_val,
             f_data_test=f_test,
             k_data_test=k_test,
             u_data_test=u_test)
    print(f"Saved the pre-smoothed data to {filename}")


# In[]:
if __name__ == "__main__":
    config = load_config("poi*")
    config["dataset_path"] = "dataset/helmholtz_1D_Grid31_50p.npz"
    config["newest_model_path"] = "models/baseline/helmholtz_1D_Grid_31_20000_50p.pt"

    # 使用deeponet record trajectory
    collect_non_pre_data(config, num_samples=2000, num_residual=5, method="PRE")

    # # 使用jacobi record trajectory, 由于jacobi速度太慢, 故设置 200 iteration 的间隔
    # collect_non_pre_data(config, num_samples=2000, num_residual=5, num_iters=200, method="NON")

    # collect_presmooth_data(config, dataset_path="./dataset/poisson_1D_Grid31.npz")

