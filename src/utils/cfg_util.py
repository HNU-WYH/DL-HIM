from datetime import datetime
import os
import yaml
import torch
import numpy as np
from box import Box
from glob import glob


# from utils.dl_util import find_available_model
# from utils.data_gen_util import generate_power_law_grid, generate_tanh_clustered_grid


def get_project_root():
    script_dir = os.path.abspath(__file__)
    project_dir = script_dir.split("\\src")[0]
    return project_dir


def find_available_model(name_rule):
    """
    find the latest available model directory based on given name_rule
    (such as r'./checkpoints/Poisson_1D_*').
    """
    model_list = glob(name_rule)
    if not model_list:
        return None
    latest_model = max(model_list, key=os.path.getmtime)
    return latest_model


def convert2list(value):
    """
    ensures that some values in the config are lists
    """
    if value is None:
        return []
    elif isinstance(value, list):
        return value
    else:
        Warning("Transform to List automatically")
        return [value]


def ensure_list_in_field(config, filed_paths=None):
    if filed_paths is None:
        filed_paths = [
            'data.k_x.grf_setting.sigma',
            'data.k_x.grf_setting.l0',
            'data.f_x.grf_setting.sigma',
            'data.f_x.grf_setting.l0',
            'solver.eigen_tracking.modes_of_interest'
        ]

    for field_path in filed_paths:
        keys = field_path.split(".")
        last_key = keys[-1]
        try:
            current = config
            for key in keys[:-1]:
                current = getattr(current, key)
            current[last_key] = convert2list(current[last_key])
        except AttributeError:
            raise ValueError(f"Required field missing: {field_path}")
    return config


def load_config(config_wildcard: str = "diffusion1d*",
                dataset_name: str = None,
                model_save_name: str = None,
                model_load_name: str = None):
    """
    Args:
        config_wildcard: a wildcard of the config name (only name, no path here);
        dataset_name: the name of dataset to be loaded (only name, no path here);
        model_save_name: the name of model to be saved (only name, no path here);
        model_load_name: the name of model to be loaded (only name, no path here);
    Returns:
        config: the config of problem, dataset, model, and solver
    """
    # find the path of config
    dir_root = get_project_root()
    config_path = glob(os.path.join(dir_root, "configs", config_wildcard))
    if len(config_path) == 0:
        raise FileNotFoundError("No file can be found, please input the correct wildcard of config")
    elif len(config_path) != 1:
        print(config_path)
        raise "more than one config files are founded"
    else:
        config_path = config_path[0]

    # open the config file
    with open(config_path, 'r', encoding="utf-8") as f:
        config = Box(yaml.safe_load(f))

    # add the device information
    config['device'] = "cuda" if torch.cuda.is_available() else "cpu"

    # add the path of model to be loaded
    if model_load_name is not None:
        config["model_load_path"] = find_available_model(os.path.join(dir_root, "checkpoints", model_load_name))
    else:
        config["model_load_path"] = find_available_model(os.path.join(dir_root, "checkpoints", "baseline",
                                                                      config.problem.type + "_" +
                                                                      str(config.problem.n_dim) + "D_Grid" +
                                                                      str(config.data.mesh.grid_num) + "_Ep" +
                                                                      str(config.training.num_epoch) + "*"))
    if model_save_name is not None:
        config["model_save_path"] = os.path.join(dir_root, "checkpoints", model_save_name)
    else:
        config["model_save_path"] = os.path.join(dir_root, "checkpoints", "uncategorized",
                                                 config.problem.type + "_" +
                                                 str(config.problem.n_dim) + "D_Grid" +
                                                 str(config.data.mesh.grid_num) + "_Ep" +
                                                 str(config.training.num_epoch) + "_" +
                                                 str(datetime.now())[:10])

    # add the training & validation data information
    if dataset_name is not None:
        config["dataset_path"] = os.path.join(dir_root, "dataset", dataset_name)
    else:
        config["dataset_path"] = os.path.join(dir_root, "dataset",
                                              config.problem.type + "_" +
                                              str(config.problem.n_dim) + "D_Grid" +
                                              str(config.data.mesh.grid_num) + ".npz")

    config = ensure_list_in_field(config)
    return config


if __name__ == "__main__":
    cfg = load_config()

