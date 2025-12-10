import os
import torch
import numpy as np

from box import Box
from torch import nn

from .base import NeuralOperatorBase
from src.utils.gen1d_util import generate_x_nodes


class DeepONet1d(NeuralOperatorBase):
    def __init__(self, config: Box):
        super().__init__(config)

        # Get grid properties and function dimensions
        self.don_num_x_nodes = config.data.mesh.grid_num
        self.don_x_nodes_np = generate_x_nodes(config.data.mesh.grid_type, self.don_num_x_nodes)
        self.register_buffer("don_x_nodes_torch", torch.tensor(self.don_x_nodes_np[1:-1, None], dtype=torch.float32))

        # define a hard constraint H(x) = x * (x - 1.0)
        # This forces the solution to be 0 at x=0 and x=1
        self.hard_constraints = lambda x: x * (1.0 - x)

        # for normalizing the parameter function k(x)
        self.register_buffer("k_mean", torch.zeros(1, dtype=torch.float32))
        self.register_buffer("k_sigma", torch.zeros(1, dtype=torch.float32))

        # Define the dimension of the output of subnetworks
        self.f_dim = config.training.don_setting.f_dim

        # Initialize the trunk network and the branch network
        self.trunk_net = self._build_trunk()
        self.branch_net = self._build_branch(input_size=2 * self.don_num_x_nodes - 2)

        self.to(self.device)

    def _build_branch(self, input_size):
        """
        Each branch network will take k_x and f_x as input and output f_dim features.
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
            :param k_x: e.g. shape = [func num, don grid points]
            :param f_x: e.g. shape = [func num, don grid points - 2]
            :param trunk_input: e.g. shape = [query points, n dim] (query points = don grid points - 2)
            :param a_mats: e.g. shape = [func num, don grid points - 2, don grid points - 2]
            :return: u_pred, res, du_pred, dres
        """
        if k_x.ndim == self.n_dim:
            k_x = k_x[None, :]
            f_x = f_x[None, :]
        assert k_x.size(0) == f_x.size(0)
        assert k_x.size(1) == self.don_num_x_nodes
        assert f_x.size(1) == self.don_num_x_nodes - 2

        k_x = k_x.to(self.device)
        f_x = f_x.to(self.device)

        if trunk_input is None:
            trunk_input = self.don_x_nodes_torch
        trunk_input = trunk_input.to(self.device)  # (num_x-2, 1)

        # Normalize the inputs
        branch_in, f_norm = self._normalize(k_x, f_x)  # (B, 2num_x-2), (B, 1)

        # Get outputs from branch networks
        branch_out = self.branch_net(branch_in)  # (B, f_dim)

        # Get trunk output
        trunk_out = self.trunk_net(trunk_input)  # (num_x-2, f_dim)

        # Intermediate output computation
        u_net = branch_out @ trunk_out.T  # (B, num_x-2)

        # Apply f normalization
        # (B, num_x-2) * (B, 1) -> (B, num_x-2)
        u_norm = u_net * f_norm

        # Apply hard constraints
        H_x = self.hard_constraints(trunk_input).squeeze(-1)  # (num_x-2)
        u_pred = u_norm * H_x  # (B, num_x-2) * (num_x-2) -> (B, num_x-2)

        res, du_pred, dres = None, None, None
        if self.require_res and a_mats is not None:
            res = self.compute_residual(a_mats, f_x, u_pred)

        if self.require_du:
            jvp = self._compute_jvp(trunk_input)      # (num_x-2, f_dim)
            du_norm = (branch_out @ jvp.T) * f_norm   # (B, num_x-2)

            x_coords = trunk_input.squeeze(-1)        # (numx-2, )
            dH_x = 1.0 - 2.0 * x_coords               # (numx-2, )
            du_pred = du_norm * H_x + u_norm * dH_x   # (B, num_x-2)

            if self.require_dres and a_mats is not None:
                dres = (-a_mats @ du_pred[..., None]).squeeze(-1)  # (B, num_x-2, num_x-2) @ (B, num_x-2, 1)

        return {
            "u_pred": u_pred,
            "res": res,
            "du_pred": du_pred,
            "dres": dres
        }

    def predict(self, k_x: np.ndarray, f_x: np.ndarray,
                x_k: np.ndarray = None, x_f: np.ndarray = None,
                query_points: np.ndarray = None, **kwargs):
        """
            Inference stage for solving PDEs in other mesh sizes, different from the mesh used in DeepONet training
            For notation:
            - new grid points = the size of new mesh (G)
            - don grid points = the size of mesh in training (D)
            - query points = the inner points of the new mesh (G-2)

            :param k_x: e.g. shape = [func num, grid points]
            :param x_k: e.g. shape = [grid points, ]
            :param f_x: e.g. shape = [func num, grid points - 2]
            :param x_f: e.g. shape = [grid points - 2, ]
            :param query_points: e.g. shape = [query points, n dim] (query_points = grid points - 2)
            :return: e.g. shape => [func num, query points]
        """
        if k_x.ndim == 1:
            k_x = k_x[None, :]  # (batch, new grid points) = (B, G)
        if f_x.ndim == 1:
            f_x = f_x[None, :]  # (batch, new grid points - 2) = (B, G-2)
        assert k_x.shape[0] == f_x.shape[0]
        batch_size = f_x.shape[0]

        # Interpolate if needed
        k_x, f_x, x_f = self._preprocess_input(k_x, x_k, f_x, x_f, batch_size)   # (D,), (D-2,), (G-2, 1)

        # If query_points is None, use the internal x_nodes for predictions
        if query_points is None:
            query_points = x_f
        if query_points.ndim == 1:
            query_points = query_points[:, None]
        query_points = torch.as_tensor(query_points, dtype=torch.float32, device=self.device)  # (G-2, 1)

        with torch.no_grad():
            # Normalize inputs and make prediction
            branch_in, f_norm = self._normalize(k_x, f_x)                   # (B, 2D-2), (B, 1)
            branch_out = self.branch_net(branch_in)                         # (B, f_dim)
            trunk_out = self.trunk_net(query_points)                        # (G-2, f_dim)

            u_net = branch_out @ trunk_out.T                                # (B, G-2)
            u_norm = u_net * f_norm                                          # Apply normalization
            u = u_norm

            if self.hard_constraints:
                H_x = self.hard_constraints(query_points).squeeze(-1)       # (G-2, )
                u = u_norm * H_x                                             # Apply hard constraints

            if batch_size == 1:
                u = u.squeeze()

        return u.cpu().detach().numpy()  # (B, G-2)

    def _preprocess_input(self, k_x, x_k, f_x, x_f, batch_size=None):
        if batch_size is None:
            batch_size = f_x.shape[0]

        if k_x.shape[-1] != self.don_num_x_nodes:
            if isinstance(k_x, torch.Tensor):
                k_x = k_x.detach().cpu().numpy()                                            # (G, )

            if x_k is None:
                x_k = np.linspace(0, 1, k_x.shape[-1])                                      # (G, )

            k_x_interp = np.stack([np.interp(self.don_x_nodes_np, x_k, k_x[i]) for i in range(batch_size)], axis=0)
            k_x = torch.as_tensor(k_x_interp, dtype=torch.float32, device=self.device)
        else:
            k_x = torch.as_tensor(k_x, dtype=torch.float32, device=self.device)             # (D,)

        if f_x.shape[-1] != self.don_num_x_nodes - 2:
            if isinstance(f_x, torch.Tensor):
                f_x = f_x.detach().cpu().numpy()                                            # (G - 2,)

            if x_f is None:
                x_f = np.linspace(0, 1, f_x.shape[-1] + 2)[1:-1]                            # (G - 2,)

            f_x_interp = np.stack([np.interp(self.don_x_nodes_np[1:-1], x_f, f_x[i]) for i in range(batch_size)], axis=0)
            f_x = torch.as_tensor(f_x_interp, dtype=torch.float32, device=self.device)
        else:
            f_x = torch.as_tensor(f_x, dtype=torch.float32, device=self.device)             # (D - 2,)

        return k_x, f_x, x_f[:, None]     # (D,), (D - 2,), (G - 2, 1)

    @staticmethod
    def compute_residual(a_mats, f_x, u_pred):
        return (f_x[..., None] - a_mats @ u_pred[..., None]).squeeze(-1)

    def _compute_jvp(self, trunk_input):
        seed = torch.ones_like(trunk_input)      # seed = (B, 1)

        # return the outputs and gradient of trunk_net, only jvp is required here
        _, jvp = torch.autograd.functional.jvp(  # Each row of Jacobi is a gradient
            func=lambda x: self.trunk_net(x),    # Jacobi of trunk net: (B, 1) -> (B, F) is a vector (B * F, B)
            inputs=(trunk_input,),               # 由于batch之间彼此独立, Jacobi 是一个分块对角矩阵. seed = (B, 1)
            v=(seed,),                           # jvp: Jacobi @ seed -> (B*F)
            create_graph=True)                   # create computational graph for backpropagation

        return jvp                               # jvp会自动reshape为trunk_net的outputs的shape: (B*F) -> (B, F)
