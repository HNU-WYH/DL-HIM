import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import pickle


def load_pkl_file(dataset_path):
    if dataset_path.endswith('.pkl'):
        # 加载 pkl 文件
        with open(dataset_path, 'rb') as f:
            data = pickle.load(f)
        # 重映射为 npz 的 key
        new_data = {
            'x_data': data.get('x_nodes'),
            'f_data_train': data.get('f_train'),
            'k_data_train': data.get('k_train'),
            'u_data_train': data.get('u_train'),
            'du_data_train': data.get('du_dx_train'),
            'f_data_test': data.get('f_test'),
            'k_data_test': data.get('k_test'),
            'u_data_test': data.get('u_test'),
            'du_data_test': data.get('du_dx_test')
        }
        return new_data

    else:
        raise ValueError("Unsupported file format. Only .pkl files are supported.")


def generate_uniform_grid(num_points):
    x_uniform = np.linspace(0.0, 1.0, num_points)
    return x_uniform


def generate_power_law_grid(num_points, power=3.0):
    """
    在[0, 1]区间内生成一个非均匀网格 (幂律分布)。
    """
    y = np.linspace(0.0, 1.0, num_points)
    x_non_uniform = y ** power
    return x_non_uniform


def generate_tanh_clustered_grid(num_points, strength=3.2):
    """
    在[0, 1]区间内生成一个两端密集、中间稀疏的非均匀网格。
    """
    y = np.linspace(-1.0, 1.0, num_points)
    x_warped = np.tanh(strength * y)
    x_min, x_max = np.tanh(-strength), np.tanh(strength)
    x_normalized = (x_warped - x_min) / (x_max - x_min)
    return x_normalized


def generate_x_nodes(grid_type, num_points):
    grid_type = grid_type.lower()

    if grid_type == 'uniform':
        return generate_uniform_grid(num_points)
    elif grid_type == 'power':
        return generate_power_law_grid(num_points)
    elif grid_type == 'tanh':
        return generate_tanh_clustered_grid(num_points)
    else:
        raise ValueError(f"Unknown grid type: {grid_type}")


def grf_generate(x_nodes, func_num, sigma, l0, mean=0.0, minimal=None, **kwargs):
    """
    这里补充一个latex公式
    Args:
        x_nodes:
        func_num:
        sigma:
        l0:
        mean:
        minimal:

    Returns:

    """
    x_num = len(x_nodes)                          # (x_num,)
    r_square = (x_nodes - x_nodes[:, None])**2    # (x_num, x_num)
    covariance_matrix = np.zeros((x_num, x_num))  # (x_num, x_num)

    if minimal is not None:
        func_num = 6 * func_num
    # if not isinstance(sigma_0, list):
    #     sigma_0 = [sigma_0]
    # if not isinstance(l_0, list):
    #     l_0 = [l_0]

    assert len(sigma) == len(l0)
    if sigma == [0]:
        # if sigma_0 = 0，then returns all 0
        return np.zeros((func_num, x_num))        # (func_num, x_num)

    for idx, l in enumerate(l0):
        s = sigma[idx]
        covariance_matrix += (s ** 2) * np.exp(- r_square / (2 * l ** 2))     # (x_num, x_num)

    mean = mean * np.ones_like(x_nodes)  # (x_num,)
    grfs = np.random.multivariate_normal(mean, covariance_matrix, func_num)   # (func_num, x_num)
    if minimal is not None:
        grfs = grfs[np.where(np.min(grfs, axis=1) > minimal)][:func_num]

    return grfs


def fourier_generate(x_nodes, func_num, freq_num=3, freq_upper=10, amplitude=0.5, **kwargs):
    """
    这里补充一个latex公式
    Args:
        x_nodes
        func_num:
        freq_num: the number for overlapping sine and cosine functions
        freq_upper:
        amplitude:

    Returns:

    """
    func = []                                                 # (func, x_num)
    for _ in range(func_num):
        f_x = np.zeros_like(x_nodes)                          # the function
        for i in range(freq_num):
            n = np.random.randint(1, freq_upper)              # random frequency
            a_n = np.random.normal(0, amplitude)              # random amplitude for cosine function
            f_x += a_n * np.cos(np.pi * n * x_nodes)          # cosine function

            b_n = np.random.normal(0, amplitude)              # random amplitude for sine function
            f_x += b_n * np.sin(np.pi * n * x_nodes)          # sine function
        func.append(f_x)
    return np.array(func)                                     # (func_num, x_len)


def gaussian_generate(x_nodes, func_num, **kwargs):
    """
    生成 f ~ N(0, 1) 作为一个随机向量 (高斯白噪声)。
    每个点的值都是从标准正态分布中独立采样的。
    """
    x_num = len(x_nodes)                                      # size of x_nodes
    size = (func_num, x_num)                                  # size of f_x
    f_x = np.random.normal(loc=0.0, scale=1.0, size=size)     # sampling from N(0, I)
    return f_x


def fixed_generate(x_nodes, func_num, value=1.0, **kwargs):
    x_num = len(x_nodes)
    f_x = np.ones(shape=(x_num, func_num))
    f_x = value * f_x
    return f_x


function_generators = {
    "fixed": fixed_generate,
    "fourier": fourier_generate,
    "grf": grf_generate,
    "gaussian": gaussian_generate
}

# class k_f_generator:
#     def __init__(self, x, func_num):
#         self.x = x
#         self.func_num = func_num
#
#     def default_k_generate(self, sigma_0 = None, l_0 = None,
#                           k_0 = None, k_min = None, method = "GRF"):
#         """
#         The following is the default case in paper for poisson 1D equation
#         """
#         if method == "GRF":
#             if sigma_0 is None:
#                 sigma_0 = [0.3,]
#
#             if l_0 is None:
#                 l_0 = [0.1,]
#
#             if k_0 is None:
#                 k_0 = [1.0,]
#
#             if k_min is None:
#                 k_min = [0.3,]
#
#             num_of_samples = 3 * self.func_num
#             grf = self.GRF_generator(0, sigma_0, l_0, num_of_samples) + k_0
#             valid_examples = grf[np.where(np.min(grf, axis=1) > k_min)]
#             if valid_examples.shape[0] >= self.func_num:
#                 return valid_examples[:self.func_num]
#             else:
#                 num_of_samples = 6 * self.func_num
#                 grf = self.GRF_generator(0, sigma_0, l_0, num_of_samples) + k_0
#                 return grf[np.where(np.min(grf, axis=1) > k_min)][:self.func_num]
#
#         elif method == "Fixed":
#             k = np.ones_like(self.x)
#             k_copied = np.tile(k, (self.func_num, 1))
#             return k_copied
#
#         else:
#             raise KeyError("Invalid method")
#
#     def default_f_generate(self, sigma_0 = None, l_0 = None, method = "GRF"):
#         """
#         The following is the default case in paper for poisson 1D equation
#         """
#         if method == "GRF":
#             if sigma_0 is None:
#                 sigma_0 = [1.0, ]
#             if l_0 is None:
#                 l_0 = [0.1, ]
#
#             f_data_all = self.GRF_generator(0, sigma_0, l_0)
#
#         elif method == "Fourier":
#             f_data_all = self.Fourier_generator()
#
#         elif method == "Gaussian":
#             f_data_all = self.Gaussian_generator()
#
#         else:
#             raise KeyError(f"Invalid method for f generation: {method}")
#
#         return f_data_all
#
#     def GRF_generator(self, mu = 0, sigma_0=None, l_0=None, func_num = None):
#         if func_num is None:
#             func_num = self.func_num
#         if sigma_0 is None:
#             sigma_0 = [1.0, 0.5, 0.2]
#         if l_0 is None:
#             l_0 = [0.1, 0.2, 0.4]
#
#         if sigma_0 == [0]:  # 如果 sigma_0 为 0，直接返回全零
#             return np.zeros((func_num, len(self.x)))
#
#         assert len(sigma_0) == len(l_0)
#
#         # compute the matrix for (x_i - x_j)^2
#         r_square = (self.x - self.x[:, None])**2
#         covariance_matrix = np.zeros((len(self.x), len(self.x)))
#
#         for idx, l in enumerate(l_0):
#             s = sigma_0[idx]
#             covariance_matrix += (s ** 2) * np.exp(- r_square / (2 * l**2))
#         mu = mu * np.ones_like(self.x)
#
#         return np.random.multivariate_normal(mu, covariance_matrix, func_num) # shape = (func_num, x_len)
#
#     def Fourier_generator(self, N = 10, amplitude = 0.5):
#         """ Not used in our project"""
#         func = []
#         for _ in range(self.func_num):
#             # a0 = np.random.normal(0, amplitude)
#             # k = a0 * np.zeros_like(self.x)
#
#             # for n in range(1, N + 1):
#                 # a_n = np.random.normal(0, amplitude)
#                 # b_n = np.random.normal(0, amplitude)
#                 # k += a_n * np.cos(2 * np.pi * n * self.x) + b_n * np.sin(2 * np.pi * n * self.x)
#
#             num = np.random.randint(1, 3)
#             k = np.zeros_like(self.x)
#             for i in range(num):
#                 n = np.random.randint(1, N)
#                 a_n = np.random.normal(0, amplitude)
#                 b_n = np.random.normal(0, amplitude)
#                 k += a_n * np.cos(np.pi * n * self.x) + b_n * np.sin(np.pi * n * self.x)
#
#             func.append(k)
#
#         return np.array(func) # shape = (func_num, x_len)
#
#     def Gaussian_generator(self):
#         """
#         生成 f ~ N(0, 1) 作为一个随机向量 (高斯白噪声)。
#         每个点的值都是从标准正态分布中独立采样的。
#         """
#         num_points = len(self.x)
#         size = (self.func_num, num_points)
#
#         # 从均值为0，标准差为1的正态分布中采样
#         return np.random.normal(loc=0.0, scale=1.0, size=size)
#
# class poisson1d:
#     """
#     Solving the following poisson equation with homogenous dirichlet B.C.:
#         - d/dx [k(x) du/dx] = f(x), with u(0)=0, u(1)=0
#     """
#     def __init__(self, x, k, f):
#         # self.h = x[1] - x[0]
#         self.x = x
#         self.k = k
#         self.f = f
#
#     def fdm_solve(self, mat_type ="sparse"):
#         """
#         Solving the poisson equation with homogeneous dirichlet B.C
#         """
#         # calculate LHS matrix
#         n = len(self.x)
#         if mat_type == "sparse":
#             a_mat = sp.lil_matrix((n, n))
#         else:
#             a_mat = np.zeros((n, n))
#         u = np.zeros_like(self.x)
#
#         h_diffs = np.diff(self.x)
#         is_uniform = np.allclose(h_diffs, h_diffs[0])
#
#         if is_uniform:
#             h = h_diffs[0]
#             for index in range(1, n-1):
#                 a_mat[index, index] = (self.k[index-1] + 2 * self.k[index] + self.k[index+1]) / (2 * h**2)
#                 a_mat[index, index - 1] = - (self.k[index] + self.k[index - 1]) / (2 * h**2)
#                 a_mat[index, index + 1] = - (self.k[index] + self.k[index + 1]) / (2 * h**2)
#
#         else:
#             for index in range(1, n - 1):
#                 h_left = self.x[index] - self.x[index - 1]
#                 h_right = self.x[index + 1] - self.x[index]
#
#                 k_mid_left = 0.5 * (self.k[index - 1] + self.k[index])
#                 k_mid_right = 0.5 * (self.k[index] + self.k[index + 1])  # 修正了之前注释中的笔误
#
#                 C_left = k_mid_left / h_left
#                 C_right = k_mid_right / h_right
#
#                 a_mat[index, index] = C_left + C_right
#                 a_mat[index, index - 1] = -C_left
#                 a_mat[index, index + 1] = -C_right
#
#         # 应用 Dirichlet 边界条件
#         a_mat[0, 0] = 1.0
#         a_mat[-1, -1] = 1.0
#         self.f[0] = 0.0
#         self.f[-1] = 0.0
#
#         if mat_type == "sparse":
#             a_mat = a_mat.tocsr()
#             u[1:-1] = spla.spsolve(a_mat[1:-1, 1:-1], self.f[1:-1])
#         else:
#             u[1:-1] = np.linalg.solve(a_mat[1:-1, 1:-1], self.f[1:-1])
#
#         du_dx = self.numerical_derivative(u, self.x)
#
#         return {'u': u,
#                 'du_dx': du_dx,
#                 'a_mat': a_mat[1:-1, 1:-1],
#                 'lhs': self.f[1:-1],
#                 'inner': slice(1, -1),
#                 'b.c.': [0, -1],
#                 'x': self.x
#                 }
#
#     @staticmethod
#     def numerical_derivative(u, x):
#         if u.ndim == 2:
#             du_dx_list = [np.gradient(u_row, x) for u_row in u]
#             return np.array(du_dx_list)
#         elif u.ndim == 1:
#             return np.gradient(u, x)
#         else:
#             raise ValueError("Unsupported dimension")
#
#     # @staticmethod
#     # def error_compute(u, x, k, f):
#     #     """
#     #     No need to implement this, as we solve it by direct solver
#     #     """
#     #     du_dx = poisson1d.numerical_derivative(u, x)
#     #     rhs = poisson1d.numerical_derivative(k * du_dx, x)
#     #     error = np.abs(rhs - f)
#     #
#     #     if error.ndim == 1:
#     #         internal_error = error[1:-1]
#     #         mean_error = np.mean(internal_error)
#     #     elif error.ndim == 2:
#     #         internal_error = error[:, 1:-1]
#     #         mean_error = np.mean(internal_error)
#     #     else:
#     #         raise ValueError("Unsupported dimension for error computation")
#     #     return mean_error
#
#     def fem_solve(self):
#         raise NotImplementedError(" FEM is not used")
#
# class helmholtz1d:
#     """
#     Solving the following helmholtz equation with homogenous dirichlet B.C.:
#         u''(x) + k(x)^2u(x) = f(x), with u(0)=0, u(1)=0
#     """
#
#     def __init__(self, x, k, f):
#         self.h = x[1] - x[0]
#         self.x = x
#         self.k = k
#         self.f = f
#         self.new_x = None
#         self.fine_grid_checking()
#
#     def fine_grid_checking(self):
#         """usually make sure 10 grids points per wavelength """
#         required_h = 2 * np.pi / (10 * self.k.max())
#         if self.h >  required_h:
#             required_grid = 1 // required_h + 1
#             raise ValueError(f"k(x) is too large for the current grid, need at Least {required_grid} grid points, or reduce k0 in configs")
#
#
#
#     def fdm_solve(self, mat_type ="sparse"):
#         """
#         Solving the poisson equation with homogeneous dirichlet B.C
#         """
#         # calculate LHS matrix
#         n = len(self.x)
#         if mat_type == "sparse":
#             a_mat = sp.lil_matrix((n, n))
#         else:
#             a_mat = np.zeros((n, n))
#         u = np.zeros_like(self.x)
#
#         for index in range(1, n-1):
#             a_mat[index, index] = self.k[index]**2 - 2 / (self.h**2)
#             a_mat[index, index - 1] = 1 / (self.h**2)
#             a_mat[index, index + 1] = 1 / (self.h**2)
#
#         # 应用 Dirichlet 边界条件
#         a_mat[0, 0] = 1.0
#         a_mat[-1, -1] = 1.0
#
#         if mat_type == "sparse":
#             a_mat = a_mat.tocsr()
#             u[1:-1] = spla.spsolve(a_mat[1:-1, 1:-1], self.f[1:-1])
#         else:
#             u[1:-1] = np.linalg.solve(a_mat[1:-1, 1:-1], self.f[1:-1])
#
#         du_dx = self.numerical_derivative(u, self.x)
#
#         return {
#         'u': u,
#         'du_dx': du_dx,
#         'a_mat': a_mat[1:-1, 1:-1],
#         'lhs': self.f[1:-1],
#         'inner': slice(1,-1),
#         'b.c.': [0,-1],
#         }
#
#     @staticmethod
#     def numerical_derivative(u, x):
#         h = x[1] - x[0]
#         du_dx = np.zeros_like(u)
#         if u.ndim == 2:
#             du_dx[:,0] = (u[:,1] - u[:,0])/h
#             du_dx[:,-1] = (u[:,-1] - u[:,-2])/h
#             du_dx[:,1:-1] = (u[:,2:] - u[:,:-2]) / (2 * h)
#
#         elif u.ndim == 1:
#             du_dx[0] = (u[1] - u[0]) / h
#             du_dx[-1] = (u[-1] - u[-2]) / h
#             du_dx[1:-1] = (u[2:] - u[:-2]) / (2 * h)
#
#         return du_dx
#
#     @staticmethod
#     def error_compute(u, x, k, f):
#         """
#         No need to implement this, as we solve it by direct solver
#         """
#         du_dx = helmholtz1d.numerical_derivative(u, x)
#         ddu = helmholtz1d.numerical_derivative(du_dx, x)
#         rhs = ddu + k**2 * u
#         error = np.abs(rhs - f)
#
#         if error.ndim == 1:
#             internal_error = error[1:-1]
#             mean_error = np.mean(internal_error)
#         elif error.ndim == 2:
#             internal_error = error[:, 1:-1]
#             mean_error = np.mean(internal_error)
#         else:
#             raise ValueError("Unsupported dimension for error computation")
#         return mean_error
#
#     def fem_solve(self):
#         raise NotImplementedError(" FEM is not used")