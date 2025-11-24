import os
import numpy as np
from tqdm import tqdm
from box import Box

from src.problems import create_problem
from src.utils.fdm_utils import expand_solution
from src.utils.gen1d_util import generate_x_nodes, function_generators

# import matplotlib.pyplot as plt
# from scipy.integrate import solve_bvp

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class DataGenerator1d:
    def __init__(self, config: Box):
        self.config = config
        self.problem = config.problem.type.lower()
        self.solver = config.problem.method.lower()

        self.n_dim = config.problem.n_dim
        self.func_num = config.data.func_num
        self.split_ratio = config.data.train_ratio

        self.dataset = None
        self.eps = config.data.get("eps", 1.0)
        self.x_nodes = generate_x_nodes(config.data.mesh.grid_type, config.data.mesh.grid_num)

        if self.problem not in ["poisson", "diffusion", "helmholtz", "convdiff"]:
            raise NotImplementedError(f"{self.problem} problem is not supported")
        self.u_generator = create_problem(self.problem)

        # for helmholtz equations, ill-conditioned system matrices need to be filtered
        if self.problem == "helmholtz":
            self.threshold = config.data.get("threshold", 0.8)
            self.redundancy = max(1.0/self.threshold, config.data.get("redundancy", 1.6))
            self.redundant_func_num = int(self.redundancy * self.func_num)
        else:
            self.redundant_func_num = self.func_num

    def __init_data(self):
        num_points = self.config.data.mesh.grid_num                       # n
        u = np.zeros(shape=(self.redundant_func_num, num_points-2))       # (b, n-2) with inner points only
        du = np.zeros_like(u)                                             # (b, n-2) with inner points only

        a_mats = []                   # empty for appending with final size (b, n-2, n-2)
        f_inner_data = []             # empty for appending with final size (b,
        cond_num = np.zeros(self.redundant_func_num)

        k_data = self.__generate_kf("k_x")                                # (b, n)
        f_data = self.__generate_kf("f_x")                                # (b, n)
        return u, du, a_mats, k_data, f_data, f_inner_data, cond_num

    def __generate_kf(self, func_name="f_x"):
        gen_cfg = getattr(self.config.data, func_name)
        gen_name = getattr(gen_cfg, "generator").lower()
        gen_setting = getattr(gen_cfg, gen_name + "_setting")
        return function_generators[gen_name](self.x_nodes, self.redundant_func_num, **gen_setting)

    def generate_data(self, force_gen=False, seed=None):
        if os.path.exists(self.config['dataset_path']) and not force_gen:
            print("Dataset already exists, no need to generate data")
            self.dataset = np.load(self.config['dataset_path'])
        else:
            if seed is not None:
                np.random.seed(seed)

            u, du, a_mats, k_data, f_data, f_inner_data, cond_num = self.__init_data()

            idx = 0
            pbar = tqdm(total=self.redundant_func_num)
            while idx < self.redundant_func_num:
                k, f = k_data[idx], f_data[idx]
                u_gen = self.u_generator(self.x_nodes, k, f, eps=self.eps)
                solution = u_gen.solve(u_left=0.0, u_right=0.0, method=self.solver)

                A_inner = solution['A_inner']
                if hasattr(A_inner, "toarray"):
                    A_inner = A_inner.toarray()
                # if self.self.problem == 'helmholtz':
                cond_num[idx] = np.linalg.cond(A_inner)

                a_mats.append(A_inner)
                u[idx] = solution['u_inner']
                du[idx] = solution['du_inner']
                f_inner_data.append(solution["f_inner"])

                idx += 1
                pbar.update(1)
            pbar.close()

            a_mats = np.asarray(a_mats)
            f_inner_data = np.asarray(f_inner_data)
            # check for numerical instability
            if self.problem == 'helmholtz':
                threshold = np.quantile(cond_num, self.threshold)  # threshold
                mask_idx = cond_num < threshold

                u = u[mask_idx]
                du = du[mask_idx]
                k_data = k_data[mask_idx]
                a_mats = a_mats[mask_idx]
                cond_num = cond_num[mask_idx]
                f_inner_data = f_inner_data[mask_idx]

            training_sample = int(self.split_ratio * self.func_num)
            self.dataset = {
                "x_data": self.x_nodes,
                "u_data_train": u[:training_sample],
                "du_data_train": du[:training_sample],
                "f_data_train": f_inner_data[:training_sample],
                "k_data_train": k_data[:training_sample],
                "a_mats_train": a_mats[:training_sample],
                "cond_num_train": cond_num[:training_sample],
                "u_data_val": u[training_sample:self.func_num],
                "du_data_val": du[training_sample:self.func_num],
                "f_data_val": f_inner_data[training_sample:self.func_num],
                "k_data_val": k_data[training_sample:self.func_num],
                "a_mats_val": a_mats[training_sample:self.func_num],
                "cond_num_val": cond_num[training_sample:self.func_num],
            }

    def save(self, force_save=False, save_path=None):
        if save_path is None:
            save_path = self.config['dataset_path']

        save_flag = False
        if force_save:
            save_flag = True
        else:
            if os.path.exists(save_path):
                print("Dataset already exists")
            else:
                save_flag = True

        if save_flag:
            if self.dataset is None:
                self.generate_data(True)

            k_mean = np.mean(self.dataset["k_data_train"])
            k_std = np.std(self.dataset["k_data_train"])
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            np.savez_compressed(save_path, **self.dataset, k_mean=k_mean, k_std=k_std)
            print(f"Successfully save Dataset to {save_path}")

    @staticmethod
    def solve(x_data, f_data, k_data, eps=1.0, u_left=0.0, u_right=0.0, method="fdm", problem="poisson"):
        """
        Static method for outer function to compute solutions of a batch of PDEs numerically
        """
        if f_data.ndim == 1:
            f_data = f_data[None, :]
        if k_data.ndim == 1:
            k_data = k_data[None, :]

        method, problem == method.lower(), problem.lower()
        func_num, points_num = f_data.shape[0], f_data.shape[1]
        if func_num != k_data.shape[0] or len(x_data) != points_num:
            raise ValueError("The number of parameter functions kx and rhs functions fx is not compatible")

        if problem not in ["poisson", "diffusion", "helmholtz", "convdiff"]:
            raise NotImplementedError(f"{problem} problem is not supported")
        pde_solver = create_problem(problem)

        u = np.zeros(shape=(func_num, points_num))                           # (n,)
        du = np.zeros(shape=(func_num, points_num))                          # (n,)
        a_mats = np.zeros(shape=(func_num, points_num - 2, points_num - 2))  # (n-2, n-2)

        for idx in tqdm(range(func_num)):
            k, f = k_data[idx], f_data[idx]
            u_gen = pde_solver(x_data, k, f, eps=eps)
            solution = u_gen.solve(u_left=u_left, u_right=u_right, method=method)

            u[idx] = expand_solution(solution["u_inner"], u_left, u_right)   # (n,)
            du[idx] = np.gradient(u[idx], x_data)                            # (n,)

            A_inner = solution['A_inner']
            if hasattr(A_inner, "toarray"):
                A_inner = A_inner.toarray()
            a_mats[idx] = A_inner                                            # (n-2, n-2)
        return u, du, a_mats


class TestDataGenerator(DataGenerator1d):
    def __init__(self, config: Box, test_num=100):
        super().__init__(config)
        self.func_num = test_num
        if self.problem == "helmholtz":
            self.threshold = config.data.get("threshold", 0.8)
            self.redundancy = max(1.0 / self.threshold, config.data.get("redundancy", 1.6))
            self.redundant_func_num = int(self.redundancy * self.func_num)
        else:
            self.redundant_func_num = self.func_num

    # def __f_generator(self):
    #     """
    #     Four types of rhs for testing the DL-HIM, instead of gaussian generator only
    #     1. f = Au
    #     2.
    #     3.
    #     4.
    #     """
    #     pass
    #
    #
    # def __init_data(self):
    #     num_points = self.config.data.mesh.grid_num                    # n
    #     u = np.zeros(shape=(self.redundant_func_num, num_points - 2))  # (b, n-2) with inner points only
    #     du = np.zeros_like(u)                                          # (b, n-2) with inner points only
    #
    #     a_mats = []                # empty for appending with final size (b, n-2, n-2)
    #     cond_num = np.zeros(self.redundant_func_num)
    #     k_data = self.__generate_kf("k_x")      # (b, n)
    #     return u, du, a_mats, k_data, f_data, cond_num


# class DemoGenerator:
#     def __init__(self, config):
#         self.problem = config.get("problem", "poisson")
#         self.num_points = config.get("don_grid_num", 80)
#         self.x = np.linspace(0, 1, self.num_points)
#
#         self.f_high = lambda x: 2 * np.sin(5 * np.pi * x)
#         self.k_high = lambda x: np.ones_like(x)
#         self.dk_high_dx = lambda x: np.zeros_like(x)
#
#         self.f_low = lambda x: 1 + 0.5 * x
#         self.k_low = lambda x: np.ones_like(x)
#         self.dk_low_dx = lambda x: np.zeros_like(x)
#
#     def solve_poisson(self, k_func, dk_func, f_func):
#         """
#             用 solve_bvp 求解 1D Poisson:
#                 - d/dx [k(x) du/dx] = f(x), 边界: u(0)=0, u(1)=0
#             返回 sol 对象(可提取 u, du)
#         """
#
#         def poisson_eq(x, u):
#             du_dx = u[1]
#             d2u_dx2 = (-f_func(x) - dk_func(x) * du_dx) / k_func(x)
#             return np.vstack((du_dx, d2u_dx2))
#
#         def bc(u_a, u_b):
#             return np.array([u_a[0], u_b[0]])
#
#         u_guess = np.zeros((2, self.num_points))
#         sol = solve_bvp(poisson_eq, bc, self.x, u_guess)
#
#         return sol
#
#     def solve_helmholtz(self, k_func, dk_func, f_func):
#         """
#         用 solve_bvp 求解 1D Helmholtz 方程:
#             u''(x) + k(x)^2 * u(x) = f(x), 边界条件: u(0)=0, u(1)=0
#         参数:
#             k_func: 表示 k(x) 的函数
#             dk_func: 此处不需要, 保留参数保持接口一致
#             f_func: 表示 f(x) 的函数
#         返回:
#             sol 对象 (可提取解 u 和 u')
#         """
#
#         def helmholtz_eq(x, u):
#             # u[0] 表示 u, u[1] 表示 u'
#             # 根据方程: u''(x) = f(x) - k(x)^2 * u(x)
#             return np.vstack((u[1], f_func(x) - k_func(x) ** 2 * u[0]))
#
#         def bc(u_a, u_b):
#             return np.array([u_a[0], u_b[0]])
#
#         u_guess = np.zeros((2, self.num_points))
#         sol = solve_bvp(helmholtz_eq, bc, self.x, u_guess)
#         return sol
#
#     def generate_demo_dataset(self, solver="fdm"):
#         """
#         生成包含两个样本（高频和低频）的示例数据集.
#         solver:
#         "scipy" -> scipy.integrate.solve_bvp
#         "fdm" -> utils.data_gen_util.poisson(x, k, f).fdm_solve()
#         """
#
#         x_vis = self.x.copy()
#
#         dataset = {
#             "x_data": self.x,
#             "f_data": [],
#             "k_data": [],
#             "u_data": [],
#             "du_data": []
#         }
#
#         samples = [
#             {"f_func": self.f_high, "k_func": self.k_high, "dk_func": self.dk_high_dx, "label": "High-Frequency"},
#             {"f_func": self.f_low, "k_func": self.k_low, "dk_func": self.dk_low_dx, "label": "Low-Frequency"}
#         ]
#
#         for s in samples:
#             if solver == "scipy":
#                 if self.problem == "poisson":
#                     sol = self.solve_poisson(s["k_func"], s["dk_func"], s["f_func"])
#                 elif self.problem == "helmholtz":
#                     sol = self.solve_helmholtz(s["k_func"], s["dk_func"], s["f_func"])
#                 if sol.status == 0:
#                     # 在可视化网格 x_vis 上插值
#                     u_sol = sol.sol(x_vis)[0]  # u(x)
#                     du_sol = sol.sol(x_vis)[1]  # du/dx
#                     # 计算 k,f 在同一 x_vis 上
#                     k_val = s["k_func"](x_vis)
#                     f_val = s["f_func"](x_vis)
#                     # 存入 dataset
#                     dataset["f_data"].append(f_val)
#                     dataset["k_data"].append(k_val)
#                     dataset["u_data"].append(u_sol)
#                     dataset["du_data"].append(du_sol)
#                 else:
#                     print(f"{s['label']} solve failed: ", sol.message)
#
#             elif solver == "fdm":
#                 k_val = s["k_func"](x_vis)
#                 f_val = s["f_func"](x_vis)
#
#                 if self.problem == "poisson":
#                     u_gen = poisson1d(x_vis, k_val, f_val)
#                 elif self.problem == "helmholtz":
#                     u_gen = helmholtz1d(x_vis, k_val, f_val)
#
#                 solution = u_gen.fdm_solve()
#                 u_sol = solution['u']
#                 du_sol = solution['du_dx']
#                 # 存入 dataset
#                 dataset["f_data"].append(f_val)
#                 dataset["k_data"].append(k_val)
#                 dataset["u_data"].append(u_sol)
#                 dataset["du_data"].append(du_sol)
#             else:
#                 raise ValueError(f"Solver {solver} not supported")
#
#         dataset["f_data"] = np.array(dataset["f_data"])
#         dataset["k_data"] = np.array(dataset["k_data"])
#         dataset["u_data"] = np.array(dataset["u_data"])
#         dataset["du_data"] = np.array(dataset["du_data"])
#
#         return dataset
#
#
# if __name__ == '__main__':
#     # In[]:
#     gen_config = load_config("helm*")
#     generator = DataGenerator(gen_config)
#     generator.save(True)
#
#     x_nodes = generator.dataset["x_data"]
#     u_data = generator.dataset["u_data"]
#
#     plot_sol(x_nodes, u_data, 10)
#
#     # In[]:
#     demo = DemoGenerator(gen_config)
#
#     # 1. 生成 dataset
#     dataset_scipy = demo.generate_demo_dataset("scipy")
#     dataset_fdm = demo.generate_demo_dataset("fdm")
#
#     # 2. 说明
#     # dataset["u_data"] = shape (2, num_points)
#     # index=0 => High-Frequency
#     # index=1 => Low-Frequency
#     # dataset["du_data"] similarly
#     # dataset["x_data"] shape (num_points, )
#
#     x_data_scipy = dataset_scipy["x_data"]  # shape (num_points,)
#     x_data_fdm = dataset_fdm["x_data"]
#
#     # (若 x_data_scipy != x_data_fdm 可以视需要进行插值，但目前看来是一致的self.x)
#     #  -> 0: High-Freq, 1: Low-Freq
#     u_scipy_high = dataset_scipy["u_data"][0]
#     u_fdm_high = dataset_fdm["u_data"][0]
#     # du_scipy_high = dataset_scipy["du_data"][0]
#     # du_fdm_high = dataset_fdm["du_data"][0]
#
#     u_scipy_low = dataset_scipy["u_data"][1]
#     u_fdm_low = dataset_fdm["u_data"][1]
#     # du_scipy_low = dataset_scipy["du_data"][1]
#     # du_fdm_low = dataset_fdm["du_data"][1]
#
#     # 3. 画图对比
#     fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
#
#     # -- 左图: 高频 --
#     ax1 = axs[0]
#     ax1.plot(x_data_scipy, u_scipy_high, 'b-', label='u_scipy high')
#     ax1.plot(x_data_scipy, u_fdm_high, 'r--', label='u_fdm high')
#     # ax1.plot(x_data_scipy, du_scipy_high, 'g-', label='du_scipy high')
#     # ax1.plot(x_data_scipy, du_fdm_high, 'y--', label='du_fdm high')
#
#     ax1.set_title('High-Frequency: SciPy vs FDM')
#     ax1.set_xlabel('x')
#     ax1.set_ylabel('value')
#     ax1.grid(True)
#     ax1.legend()
#
#     # -- 右图: 低频 --
#     ax2 = axs[1]
#     ax2.plot(x_data_scipy, u_scipy_low, 'b-', label='u_scipy low')
#     ax2.plot(x_data_scipy, u_fdm_low, 'r--', label='u_fdm low')
#     # ax2.plot(x_data_scipy, du_scipy_low, 'g-', label='du_scipy low')
#     # ax2.plot(x_data_scipy, du_fdm_low, 'y--', label='du_fdm low')
#     ax2.set_title('Low-Frequency: SciPy vs FDM')
#     ax2.set_xlabel('x')
#     ax2.set_ylabel('value')
#     ax2.grid(True)
#     ax2.legend()
#
#     plt.tight_layout()
#     plt.show()
