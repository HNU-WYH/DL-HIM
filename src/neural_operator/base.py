import os
import torch
import numpy as np

from box import Box
from torch import nn
from warnings import warn
from src.utils.gen1d_util import generate_x_nodes


class NeuralOperatorBase(torch.nn.Module):
    def __init__(self, config: Box):
        # basic configs
        super().__init__()
        self.config = config
        self.device = torch.device(config.device)

        # input and output dimension
        self.n_dim = config.problem.n_dim
        self.output_dim = config.problem.output_dim

        # Required in the design of loss computation
        self.loss_type = config.training.loss.type.lower()
        self.loss_norm = config.training.loss.norm.lower()
        self.use_autograd = config.training.loss.get("use_autograd", False)

        # whether to compute gradient of solution
        if self.loss_type not in {"error", "residual"}:
            raise ValueError(f"Unsupported loss type: {self.loss_type}")

    def save_model(self, save_path=None):
        if save_path is None:
            save_path = self.config["model_save_path"]

        directory = os.path.dirname(save_path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        try:
            torch.save(self.state_dict(), save_path)
            print(f"Model successfully saved to {save_path}")
        except Exception as e:
            print(f"Failed to save model: {e}")
            raise RuntimeError

    def load_model(self, load_path=None):
        if load_path is None:
            load_path = self.config["model_load_path"]

        if not os.path.exists(load_path):
            raise FileNotFoundError(
                f"Model file not found: {load_path}. \n"
                "Please check the path or train a new model first."
            )

        try:
            state_dict = torch.load(load_path, map_location=self.device)
            self.load_state_dict(state_dict, strict=True)
            print(f"Model successfully loaded from {load_path}")

        except Exception as e:
            print(f"Failed to load model: {e}")
            raise RuntimeError

    def forward(self, **kwargs):
        pass

    def predict(self, **kwargs):
        pass

    def predict_tensor(self, k_x, f_x, query_points):
        """GPU-native inference. Inputs and output are torch.Tensors on self.device."""
        raise NotImplementedError
