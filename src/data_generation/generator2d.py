import os
import warnings

from tqdm import tqdm
from box import Box

from src.problems import create_problem
from src.utils.gen2d_util import *

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class DataGenerator2d:
    """Generate 2D training & validation data for neural operator."""

    def __init__(self, config: Box):
        self.config = config
        self.problem = config.problem.type.lower()
        self.solver = config.problem.method.lower()
        self.n_dim = config.problem.n_dim
        self.func_num = config.data.func_num
        self.split_ratio = config.data.train_ratio
        self.dataset = None
        self.eps = config.data.get("eps", 1.0)

        mesh_cfg = config.data.mesh
        self.x_nodes, self.y_nodes = generate_xy_nodes(
            mesh_cfg.grid_type, mesh_cfg.grid_nx, mesh_cfg.grid_ny
        )

        if self.problem not in ["diffusion2d"]:
            raise NotImplementedError(f"{self.problem} problem is not supported by DataGenerator2d")
        self.u_generator = create_problem(self.problem, self.n_dim)

    def __init_data(self):
        Nx, Ny = len(self.x_nodes), len(self.y_nodes)
        u = np.zeros((self.func_num, Nx - 2, Ny - 2))
        k_data = self.__generate_kf("k_x")
        f_data = self.__generate_kf("f_x")
        return u, k_data, f_data

    def __generate_kf(self, func_name="f_x"):
        gen_cfg = getattr(self.config.data, func_name)
        gen_name = getattr(gen_cfg, "generator")
        if isinstance(gen_name, list):
            gen_name = gen_name[0]
        gen_name = gen_name.lower()
        gen_setting = getattr(gen_cfg, gen_name + "_setting", {})

        if gen_name == "grf":
            return grf_generate_2d(self.x_nodes, self.y_nodes, self.func_num, **gen_setting)
        elif gen_name == "fixed":
            return fixed_generate_2d(self.x_nodes, self.y_nodes, self.func_num, **gen_setting)
        elif gen_name == "gaussian":
            return gaussian_generate_2d(self.x_nodes, self.y_nodes, self.func_num, **gen_setting)
        else:
            raise NotImplementedError(
                f"2D generator '{gen_name}' not implemented. Use 'grf', 'fixed', or 'gaussian'."
            )

    def generate_data(self, force_gen=False, seed=None):
        if os.path.exists(self.config["dataset_path"]) and not force_gen:
            print("Dataset already exists, no need to generate data")
            self.dataset = np.load(self.config["dataset_path"])
        else:
            if seed is not None:
                np.random.seed(seed)

            u, k_data, f_data = self.__init_data()
            f_inner_list = []

            idx = 0
            pbar = tqdm(total=self.func_num)
            while idx < self.func_num:
                k, f = k_data[idx], f_data[idx]
                u_gen = self.u_generator(self.x_nodes, self.y_nodes, k, f, eps=self.eps)
                solution = u_gen.solve(method=self.solver)

                u[idx] = solution["u_inner"]
                f_inner_list.append(solution["f_inner"])

                idx += 1
                pbar.update(1)
            pbar.close()

            f_inner_arr = np.asarray(f_inner_list)
            Mx, My = u.shape[1], u.shape[2]
            f_inner_2d = f_inner_arr.reshape(self.func_num, Mx, My)

            training_sample = int(self.split_ratio * self.func_num)
            self.dataset = {
                "x_data": self.x_nodes,
                "y_data": self.y_nodes,
                "u_data_train": u[:training_sample],
                "f_data_train": f_inner_2d[:training_sample],
                "k_data_train": k_data[:training_sample],
                "u_data_val": u[training_sample:self.func_num],
                "f_data_val": f_inner_2d[training_sample:self.func_num],
                "k_data_val": k_data[training_sample:self.func_num],
                "a_mats_train": np.array([]),
                "a_mats_val": np.array([]),
            }

    def save(self, force_save=False, save_path=None):
        if save_path is None:
            save_path = self.config["dataset_path"]

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

            k_mean = np.mean(self.dataset["k_data_train"])
            k_std = np.std(self.dataset["k_data_train"])
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            warnings.warn(
                "a_mats are not saved in the 2D diffusion dataset "
                "(not required for error+l2 training). "
                "Use residual loss if you need them.",
                UserWarning,
            )

            np.savez_compressed(save_path, **self.dataset, k_mean=k_mean, k_std=k_std)
            print(f"Successfully save Dataset to {save_path}")


class TestDataGenerator2d:
    """Generate 2D testing data (can mix multiple rhs types)."""

    def __init__(self, config: Box, test_num=None):
        self.config = config
        self.testing_cfg = config.get("testing", Box())
        self.problem = config.problem.type.lower()
        self.solver = config.problem.method.lower()
        self.u_generator = create_problem(self.problem, n_dim=2)
        self.dataset = None
        self.eps = self.testing_cfg.get("eps", 1.0)
        self.func_num = self.testing_cfg.func_num if test_num is None else test_num

        rhs_types = self.testing_cfg.f_x.get("generator", ["grf"])
        if isinstance(rhs_types, str):
            rhs_types = [rhs_types]
        self.rhs_types = [m.lower() for m in rhs_types]
        self.rhs_type_num = len(self.rhs_types)

        mesh_cfg = self.testing_cfg.mesh
        self.x_nodes, self.y_nodes = generate_xy_nodes(
            mesh_cfg.grid_type, mesh_cfg.grid_nx, mesh_cfg.grid_ny
        )

    def __generate_k_base(self):
        gen_cfg = self.testing_cfg.k_x
        gen_name = getattr(gen_cfg, "generator").lower()
        gen_setting = getattr(gen_cfg, gen_name + "_setting", {})
        if gen_name == "grf":
            return grf_generate_2d(self.x_nodes, self.y_nodes, self.func_num, **gen_setting)
        elif gen_name == "fixed":
            return fixed_generate_2d(self.x_nodes, self.y_nodes, self.func_num, **gen_setting)
        elif gen_name == "gaussian":
            return gaussian_generate_2d(self.x_nodes, self.y_nodes, self.func_num, **gen_setting)
        else:
            raise NotImplementedError(
                f"2D generator '{gen_name}' not implemented. Use 'grf', 'fixed', or 'gaussian'."
            )

    def generate_data(self, force_gen=False, seed=None):
        if seed is not None:
            np.random.seed(seed)

        all_u, all_f, all_k = [], [], []
        k_base = self.__generate_k_base()

        for rtype in self.rhs_types:
            k_list, u_list, f_list = [], [], []

            if rtype in ["fixed", "grf", "gaussian"]:
                f_cfg = self.testing_cfg.f_x
                f_setting = getattr(f_cfg, rtype + "_setting", {})
                if rtype == "grf":
                    f_batch = grf_generate_2d(self.x_nodes, self.y_nodes, self.func_num, **f_setting)
                elif rtype == "gaussian":
                    f_batch = gaussian_generate_2d(self.x_nodes, self.y_nodes, self.func_num, **f_setting)
                else:
                    f_batch = fixed_generate_2d(self.x_nodes, self.y_nodes, self.func_num, **f_setting)

                for idx in tqdm(range(self.func_num), desc=f"Solve {rtype}"):
                    k_curr = k_base[idx]
                    f_curr = f_batch[idx]
                    solver_inst = self.u_generator(self.x_nodes, self.y_nodes, k_curr, f_curr, eps=self.eps)
                    solution = solver_inst.solve(method=self.solver)

                    u_list.append(solution["u_inner"])
                    f_list.append(solution["f_inner"])
                    k_list.append(k_curr)
            else:
                raise ValueError(f"Unsupported rhs type: {rtype}")

            all_k.append(np.stack(k_list))
            all_u.append(np.stack(u_list))
            all_f.append(np.stack(f_list))

        k_final = np.concatenate(all_k, axis=0)
        u_final = np.concatenate(all_u, axis=0)
        f_final = np.concatenate(all_f, axis=0)

        Mx, My = u_final.shape[1], u_final.shape[2]
        f_final = f_final.reshape(-1, Mx, My)

        self.dataset = {
            "x_data": self.x_nodes,
            "y_data": self.y_nodes,
            "u_data": u_final,
            "f_data": f_final,
            "k_data": k_final,
            "a_mats": np.array([]),
        }

    def save(self, force_save=False, save_path=None):
        if save_path is None:
            save_path = self.config["dataset_path"]
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

            warnings.warn(
                "a_mats are not saved in the 2D diffusion dataset "
                "(not required for error+l2 training).",
                UserWarning,
            )

            np.savez_compressed(
                save_path,
                **self.dataset,
                k_mean=k_mean,
                k_std=k_std,
                rhs_types=self.rhs_types,
                n_per_type=self.func_num,
            )
            print(f"Successfully save Dataset to {save_path}")
