import os
import numpy as np

from tqdm import tqdm
from box import Box

from src.problems import create_problem
from src.utils.fdm2d_utils import expand_solution
from src.utils.gen2d_util import generate_x_nodes, function_generators, get_boundary_inner_idx

# import matplotlib.pyplot as plt
# from scipy.integrate import solve_bvp

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class DataGenerator2d:
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
        self.bc_idx, self.inner_idx = get_boundary_inner_idx(self.x_nodes)
        if self.problem not in ["diffusion2d"]:
            raise NotImplementedError(f"{self.problem} problem is not supported")
        self.u_generator = create_problem(self.problem)

        # for helmholtz equations, ill-conditioned system matrices need to be filtered
        if self.problem == "helmholtz":
            self.threshold = config.data.get("threshold", 0.6)
            self.redundancy = max(1.0/self.threshold, config.data.get("redundancy", 1.6))
            self.redundant_func_num = int(self.redundancy * self.func_num)
        else:
            self.redundant_func_num = self.func_num

    def __init_data(self):
        num_points = (self.config.data.mesh.grid_num - 2)**2                       # n
        u = np.zeros(shape=(self.redundant_func_num, num_points))       # (b, (n-2)^2) with inner points only
        du = np.zeros(shape=(self.redundant_func_num, num_points, 2))   #                                             # (b, (n-2)^2) with inner points only

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
                u_gen = self.u_generator(self.x_nodes, self.bc_idx, self.inner_idx, k, f, eps=self.eps)
                solution = u_gen.solve(u_bc=0.0, method=self.solver)

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
                print(f"Dataset already exists at {save_path}")
            else:
                save_flag = True

        if save_flag:
            if self.dataset is None:
                #remove seed later
                self.generate_data(True, seed=42)

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


class TestDataGenerator(DataGenerator2d):
    def __init__(self, config: Box, test_num = None):
        self.config = config
        self.testing_cfg = config.get("testing", Box())

        self.problem = config.problem.type.lower()
        self.solver = config.problem.method.lower()
        self.u_generator = create_problem(self.problem)

        self.dataset = None
        self.eps = self.testing_cfg.get("eps", 1.0)
        self.func_num = self.testing_cfg.func_num if test_num is None else test_num

        if self.problem == "helmholtz":
            self.threshold = self.testing_cfg.get("threshold", 0.5)
            self.redundancy = max(1.0 / self.threshold, self.testing_cfg.get("redundancy", 2.0))
            self.redundant_func_num = int(self.redundancy * self.func_num)
        else:
            self.redundant_func_num = self.func_num

        rhs_types = self.testing_cfg.f_x.get("generator", ["gaussian"])
        if isinstance(rhs_types, str): rhs_types = [rhs_types]

        self.rhs_types = [m.lower() for m in rhs_types]
        self.rhs_type_num = len(self.rhs_types)

        self.grid_num = self.testing_cfg.mesh.grid_num
        self.grid_type = self.testing_cfg.mesh.grid_type
        self.x_nodes = generate_x_nodes(self.grid_type, self.grid_num)
        self.bc_idx, self.inner_idx = get_boundary_inner_idx(self.x_nodes)

    def __generate_k_base(self):
        """
        Generate a base set of k(x) samples for reuse.
        """
        gen_cfg = self.testing_cfg.k_x
        gen_name = getattr(gen_cfg, "generator").lower()
        gen_setting = getattr(gen_cfg, gen_name+"_setting")

        # return: [B, N]
        return function_generators[gen_name](x_nodes = self.x_nodes, func_num = self.redundant_func_num, **gen_setting)

    def generate_data(self, force_gen = False, seed = None):
        if seed is not None:
            np.random.seed(seed)

        # 0. Initialize the data
        all_u, all_du, all_f, all_k, all_a, all_cond = [], [], [], [], [], []

        # 1. generate k_base for testing
        k_base = self.__generate_k_base() # (B, N)

        # 2. generate different types of f(x) based on k_base
        # shape of f(x) -> (4B, N)
        for rtype in self.rhs_types:
            k_list, u_list, du_list, f_list, a_list, cond_list = [], [], [], [], [], []

            if rtype in ["fixed", "gaussian", "grf"]:
                f_cfg = self.testing_cfg.f_x
                f_setting = getattr(f_cfg, rtype + "_setting", {})
                f_batch = function_generators[rtype](self.x_nodes, self.redundant_func_num, **f_setting)

                for idx in tqdm(range(self.redundant_func_num), desc=f"Solve {rtype}"):
                    k_curr = k_base[idx]
                    f_curr = f_batch[idx]

                    # Solving the system
                    solver_inst = self.u_generator(self.x_nodes, self.bc_idx, self.inner_idx, k_curr, f_curr, eps=self.eps)
                    solution = solver_inst.solve(u_bc=0.0, method=self.solver)

                    # Assign the solution
                    A_inner = solution['A_inner']                                   # [N-2, N-2]
                    if hasattr(A_inner, "toarray"): A_inner = A_inner.toarray()     # [N-2, N-2]

                    u_list.append(solution['u_inner'])                              # [N-2, ]
                    du_list.append(solution['du_inner'])                            # [N-2, 2]
                    f_list.append(solution['f_inner'])                              # [N-2, ]
                    k_list.append(k_curr)                                           # [N, ]
                    a_list.append(A_inner)                                          # [N-2, N-2]
                    cond_list.append(np.linalg.cond(A_inner))

            else:
                raise ValueError(f"Unsupported rhs type: {rtype}, only gaussian, fixed, grf are supported")

            all_k.append(np.stack(k_list))
            all_u.append(np.stack(u_list))
            all_du.append(np.stack(du_list))
            all_f.append(np.stack(f_list))
            all_a.append(np.stack(a_list))
            all_cond.append(np.array(cond_list))

        k_final = np.concatenate(all_k, axis=0)
        u_final = np.concatenate(all_u, axis=0)
        du_final = np.concatenate(all_du, axis=0)
        f_final = np.concatenate(all_f, axis=0)
        a_final = np.concatenate(all_a, axis=0)
        cond_final = np.concatenate(all_cond, axis=0)

        # Filtering the samples based on condition numbers
        if self.problem == 'helmholtz':
            threshold_val = np.quantile(cond_final, self.threshold)
            mask = cond_final < threshold_val

            k_final = k_final[mask]
            u_final = u_final[mask]
            du_final = du_final[mask]
            f_final = f_final[mask]
            a_final = a_final[mask]
            cond_final = cond_final[mask]

        # Truncating the redundant samples
        total_target = self.func_num * len(self.rhs_types)
        limit = min(k_final.shape[0], total_target)

        self.dataset = {
            "x_data": self.x_nodes,
            "u_data": u_final[:limit],
            "du_data": du_final[:limit],
            "f_data": f_final[:limit],
            "k_data": k_final[:limit],
            "a_mats": a_final[:limit],
            "cond_num": cond_final[:limit]
        }

    def save(self, force_save=False, save_path=None):
        if save_path is None:
            save_path = self.config['dataset_path']
            save_path = save_path[:-4] + "_test.npz"

        save_flag = False
        if force_save:
            save_flag = True
        else:
            if os.path.exists(save_path):
                print(f"Dataset already exists at {save_path}")
            else:
                save_flag = True

        if save_flag:
            if self.dataset is None:
                self.generate_data(True)

            k_data = self.dataset["k_data"]
            k_mean = np.mean(k_data)
            k_std = np.std(k_data)

            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            np.savez_compressed(save_path,
                                **self.dataset,
                                k_mean=k_mean,
                                k_std=k_std,
                                rhs_types=self.rhs_types,
                                n_per_type=self.redundant_func_num
                                )
            print(f"Successfully save Dataset to {save_path}")
