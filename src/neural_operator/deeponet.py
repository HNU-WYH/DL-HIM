import os
import torch
import numpy as np

from box import Box
from torch import nn

from .base import NeuralOperatorBase
from src.utils.gen1d_util import generate_x_nodes


class DeepONet1d(NeuralOperatorBase):
    def __init__(self, config: Box):  # hard_constraints=None):
        super().__init__(config)

        # Get grid properties and function dimensions
        self.num_x_nodes = config.data.mesh.grid_num
        self.x_nodes_np = generate_x_nodes(config.data.mesh.grid_type, self.num_x_nodes)
        self.register_buffer("x_nodes_torch", torch.tensor(self.x_nodes_np, dtype=torch.float32))

        # can be lambda x: x.T * (x.T-1.0), but not necessary
        # if provided, apply hard constraints
        # self.hard_constraints = hard_constraints

        # for normalizing the parameter function k(x)
        # giving value when training
        self.register_buffer("k_mean", torch.zeros(1, dtype=torch.float32))
        self.register_buffer("k_sigma", torch.zeros(1, dtype=torch.float32))

        # Define the dimension of the output of subnetworks
        self.f_dim = config.training.don_setting.f_dim

        # Initialize the trunk network and the branch network
        self.trunk_net = self._build_trunk()
        self.branch_net = self._build_branch(input_size=2 * self.num_x_nodes - 2)

        # =================================
        # Implemented in the base class
        # =================================
        # self.requires_res, self.require_du, self.requires_dres = False, False, False
        # # whether to compute gradient of solution
        # if config.training.loss.type == "Error":
        #     if config.training.loss.norm == "h1":
        #         warn("when compute gradient, no hard constraints should be applied")
        #         self.require_du = True
        #
        # # whether to compute residual or gradient of residual
        # elif config.training.loss.type == "Residual":
        #     self.requires_res = True
        #     if config.training.loss.norm == "h1":
        #         warn("when compute gradient, no hard constraints should be applied")
        #         self.requires_dres = True
        #
        # else:
        #     raise ValueError(f"Unsupported loss type: {config.training.loss.type}")

        self.to(self.device)

    def _build_branch(self, input_size):
        """
        Each branch network will take k_x or f_x as input and output f_dim features.
        """
        if self.n_dim == 1:
            return nn.Sequential(
                nn.Linear(input_size, 60),
                nn.ReLU(),
                nn.Linear(60, 60),
                nn.ReLU(),
                nn.Linear(60, self.f_dim),
            )
        else:
            raise NotImplementedError

    def _build_trunk(self):
        """
        Trunk network will take x as input and output f_dim features.
        """
        return nn.Sequential(
            nn.Linear(self.n_dim, 80),
            nn.Tanh(),
            nn.Linear(80, 80),
            nn.Tanh(),
            nn.Linear(80, self.f_dim),
        )

    def _normalize(self, k, f):
        if torch.all(self.k_sigma == 0):
            raise ValueError("k_mean and k_sigma must be set before normalization.")
        k_norm = (k - self.k_mean) / self.k_sigma

        f_norm_factor = torch.sqrt(torch.mean(f ** 2, dim=1, keepdim=True))
        f_norm = f / f_norm_factor

        branch_in = torch.concat([k_norm, f_norm], dim=1)
        return branch_in, f_norm_factor

    def forward(self, k_x, f_x, a_mats=None, trunk_input=None, **kwargs):
        """
            :param k_x: e.g. shape = [func num, grid points]
            :param f_x: e.g. shape = [func num, grid points - 2]
            :param trunk_input: e.g. shape = [query points, n dim]
            :param a_mats: e.g. shape = [func num, grid points - 2, grid points - 2]
            :return: u_pred, res, du_pred, dres
        """
        if k_x.ndim == self.n_dim:
            k_x = k_x[None, :]
            f_x = f_x[None, :]
        assert k_x.size(0) == f_x.size(0)
        assert k_x.size(1) == self.num_x_nodes
        assert f_x.size(1) == self.num_x_nodes - 2

        k_x = k_x.to(self.device)
        f_x = f_x.to(self.device)

        if trunk_input is None:
            trunk_input = self.x_nodes_torch[1:-1][:, None]
        trunk_input = trunk_input.to(self.device)  # (num_x-2, 1)

        # Normalize the inputs
        branch_in, f_norm = self._normalize(k_x, f_x)  # (B, 2num_x-2), (B, 1)

        # Get outputs from branch networks
        branch_out = self.branch_net(branch_in)  # (B, f_dim)

        # Get trunk output
        trunk_out = self.trunk_net(trunk_input)  # (num_x-2, f_dim)

        # Final output computation
        u_pred = branch_out @ trunk_out.T  # (B, num_x-2)

        # Apply normalization
        u_pred = u_pred * f_norm  # (B, num_x-2) * (B, 1) -> (B, num_x-2)

        # # Apply hard constraints if needed
        # if self.hard_constraints:
        #     H_x = self.hard_constraints(trunk_input).squeeze()  # (num_x-2)
        #     u_pred = u_pred * H_x  # (B, num_x-2) * (num_x-2) -> (B, num_x-2)

        res, du_pred, dres = None, None, None
        if self.requires_res and a_mats is not None:
            res = self.compute_residual(a_mats, f_x, u_pred)

        if self.require_du:
            jvp = self._compute_jvp(trunk_input)  # (num_x-2, f_dim)
            du_pred = (branch_out @ jvp.T) * f_norm  # (B, num_x-2)
            # Warning: "If a hard constraint is applied, the gradient is: du = du_pred * H + u_pred * dH"

            if self.requires_dres and a_mats is not None:
                dres = -a_mats @ du_pred  # (B, num_x-2, num_x-2) @ (B, num_x-2)

        return {
            "u_pred": u_pred,
            "res": res,
            "du_pred": du_pred,
            "dres": dres
        }

    def predict(self, k_x: np.ndarray, f_x: np.ndarray, trunk_input=None,
                x_k: np.ndarray = None, x_f: np.ndarray = None, **kwargs):
        """
            :param k_x: e.g. shape = [func num, grid points]
            :param x_k: e.g. shape = [grid points, ]
            :param f_x: e.g. shape = [func num, grid points - 2]
            :param x_f: e.g. shape = [grid points - 2, ]
            :param trunk_input: e.g. shape = [query points, n dim]
            :return: e.g. shape => [func num, query points]
        """
        if k_x.ndim == 1:
            k_x = k_x[None, :]
        if f_x.ndim == 1:
            f_x = f_x[None, :]
        assert k_x.shape[0] == f_x.shape[0]
        batch_size = f_x.shape[0]

        # If trunk_input is None, use the internal x_nodes for predictions
        if trunk_input is None:
            trunk_input = self.x_nodes_torch[1:-1][:, None]
        trunk_input = trunk_input.to(self.device)  # (num_x-2, 1)

        # Interpolate if needed
        k_x, f_x = self._preprocess_input(k_x, x_k, f_x, x_f, batch_size)

        with torch.no_grad():
            # Normalize inputs and make prediction
            branch_in, f_norm = self._normalize(k_x, f_x)  # (B, 2num_x-2), (B, 1)
            branch_out = self.branch_net(branch_in)  # (B, f_dim)
            trunk_out = self.trunk_net(trunk_input)  # (num_x-2, f_dim)

            u = branch_out @ trunk_out.T  # (B, num_x-2)
            u = u * f_norm  # Apply normalization

            # if self.hard_constraints:
            #     H_x = self.hard_constraints(trunk_input).squeeze()  # (num_x-2)
            #     u = u * H_x  # Apply hard constraints

            if batch_size == 1:
                u = u.squeeze()

        return u.cpu().detach().numpy()

    def _preprocess_input(self, k_x, x_k, f_x, x_f, batch_size=None):
        if batch_size is None:
            batch_size = f_x.shape[0]

        if k_x.shape[-1] != self.num_x_nodes:
            if isinstance(k_x, torch.Tensor):
                k_x = k_x.detach().cpu().numpy()

            if x_k is None:
                x_k = np.linspace(0, 1, k_x.shape[-1])

            k_x_interp = np.stack([np.interp(self.x_nodes_np, x_k, k_x[i]) for i in range(batch_size)], axis=0)
            k_x = torch.as_tensor(k_x_interp, dtype=torch.float32, device=self.device)
        else:
            k_x = torch.as_tensor(k_x, dtype=torch.float32, device=self.device)

        if f_x.shape[-1] != self.num_x_nodes - 2:
            if isinstance(f_x, torch.Tensor):
                f_x = f_x.detach().cpu().numpy()

            if x_f is None:
                x_f = np.linspace(0, 1, f_x.shape[-1] + 2)[1:-1]

            f_x_interp = np.stack([np.interp(self.x_nodes_np[1:-1], x_f, f_x[i]) for i in range(batch_size)], axis=0)
            f_x = torch.as_tensor(f_x_interp, dtype=torch.float32, device=self.device)
        else:
            f_x = torch.as_tensor(f_x, dtype=torch.float32, device=self.device)

        return k_x, f_x

    @staticmethod
    def compute_residual(a_mats, f_x, u_pred):
        return f_x - a_mats @ u_pred

    def _compute_jvp(self, trunk_input):
        seed = torch.ones_like(trunk_input)  # (B,1)
        _, jvp = torch.autograd.functional.jvp(  # each row of Jacobi is a gradient
            lambda x: self.trunk_net(x),  # Jacobi of trunk net (B, f_dim, 1)
            (trunk_input,),  # seed (B, 1)
            (seed,),  # jvp: Jacobi @ seed -> (B, f_dim)
            create_graph=True)  # create graph for backpropagation
        return jvp  # (B, f_dim)
