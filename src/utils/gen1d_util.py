import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import pickle


def load_pkl_file(dataset_path):
    if dataset_path.endswith('.pkl'):
        # Load pkl File
        with open(dataset_path, 'rb') as f:
            data = pickle.load(f)

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
    Generate Non-uniform Grid on [0, 1]。
    """
    y = np.linspace(0.0, 1.0, num_points)
    x_non_uniform = y ** power
    return x_non_uniform


def generate_tanh_clustered_grid(num_points, strength=3.2):
    """
    Generate Non-uniform Grid on [0, 1]。
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
    Guassian Random Field
    """
    x_num = len(x_nodes)                          # (x_num,)
    r_square = (x_nodes - x_nodes[:, None])**2    # (x_num, x_num)
    covariance_matrix = np.zeros((x_num, x_num))  # (x_num, x_num)

    target_num = func_num  # cache desired batch size
    sample_multiplier = 2 if minimal is not None else 1

    assert len(sigma) == len(l0)
    if sigma == [0]:
        # if sigma_0 = 0，then returns all 0
        return np.zeros((target_num, x_num))[:func_num]                                 # (func_num, x_num)

    for idx, l in enumerate(l0):
        s = sigma[idx]
        covariance_matrix += (s ** 2) * np.exp(- r_square / (2 * l ** 2))               # (x_num, x_num)

    mean = mean * np.ones_like(x_nodes)  # (x_num,)

    grfs_list = []
    collected = 0
    while collected < target_num:
        sample_size = sample_multiplier * (target_num - collected)
        samples = np.random.multivariate_normal(mean, covariance_matrix, sample_size)   # (sample_size, x_num)
        if minimal is not None:
            samples = samples[np.min(samples, axis=1) > minimal]
        if len(samples) == 0:
            continue

        grfs_list.append(samples)
        collected += len(samples)

    grfs = np.vstack(grfs_list)[:target_num]
    return grfs


def fourier_generate(x_nodes, func_num, freq_num=3, freq_upper=10, amplitude=0.5, **kwargs):
    """
    Fourier basis Overlapping
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
    f ~ N(0, 1)
    """
    x_num = len(x_nodes)                                      # size of x_nodes
    size = (func_num, x_num)                                  # size of f_x
    f_x = np.random.normal(loc=0.0, scale=1.0, size=size)     # sampling from N(0, I)
    return f_x


def fixed_generate(x_nodes, func_num, value=1.0, **kwargs):
    x_num = len(x_nodes)
    f_x = np.ones(shape=(func_num, x_num))
    f_x = value * f_x
    return f_x


function_generators = {
    "fixed": fixed_generate,
    "fourier": fourier_generate,
    "grf": grf_generate,
    "gaussian": gaussian_generate
}