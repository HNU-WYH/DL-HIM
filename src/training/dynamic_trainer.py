import torch
import numpy as np

from typing import Optional
from functorch import vmap

from src.neural_operator import NeuralOperatorBase

class DynamicTrainer:
    def __init__(self, model: NeuralOperatorBase, print_interval: int = 100):
        # Initialization
        self.model = model
        self.device = self.model.device

        # # the a_mats and residual computing are required in the dynamic setting
        # self.model.require_res = True

        # training hyperparameters
        self.epochs = self.model.config.training.num_epoch
        self.batch_size = self.model.config.training.batch_size

        self.alpha = self.model.config.training.loss.grad_alpha
        self.loss_type = self.model.config.training.loss.type.lower()
        self.loss_norm = self.model.config.training.loss.norm.lower()

        # dynamic training setting
        dyn_cfg = self.model.config.training.dynamic
        self.max_horizon = dyn_cfg.get("horizon_size", 5)
        self.curriculum_enabled = dyn_cfg.curriculum.get("enabled", False)
        self.curriculum_interval = dyn_cfg.curriculum.get("update_interval", 100)
        self.current_horizon = 1 if self.curriculum_enabled else self.max_horizon

        # dynamic solver setting
        solver_cfg = self.model.config.solver
        self.update_ratio = solver_cfg.hybrid.get("update_ratio", 20)
        self.relax_factor = solver_cfg.numerical.get("relaxation_factor", 0.6)

        # print setting
        self.print_interval = print_interval

        # dataset initialization
        self.train_losses, self.val_losses = [], []
        self.k_train = self.f_train = self.u_train = self.du_train = self.a_mats_train = None
        self.x_nodes = self.k_val = self.f_val = self.u_val = self.du_val = self.a_mats_val = None

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.model.config.training.lr,
            weight_decay=self.model.config.training.weight_decay
        )

    def load_dataset(self, dataset):
        # load coordinates of x
        self.x_nodes = torch.tensor(dataset["x_data"], dtype=torch.float32, device=self.device)

        # training data
        self.k_train = torch.tensor(dataset["k_data_train"], dtype=torch.float32, device=self.device)
        self.f_train = torch.tensor(dataset["f_data_train"], dtype=torch.float32, device=self.device)
        self.u_train = torch.tensor(dataset["u_data_train"], dtype=torch.float32, device=self.device)
        self.a_mats_train = torch.tensor(dataset["a_mats_train"], dtype=torch.float32, device=self.device)

        # validation data
        self.k_val = torch.tensor(dataset["k_data_val"], dtype=torch.float32, device=self.device)
        self.f_val = torch.tensor(dataset["f_data_val"], dtype=torch.float32, device=self.device)
        self.u_val = torch.tensor(dataset["u_data_val"], dtype=torch.float32, device=self.device)
        self.a_mats_val = torch.tensor(dataset["a_mats_val"], dtype=torch.float32, device=self.device)

        # Update normalization factors (k_mean and k_sigma)
        k_mean_val = torch.as_tensor(dataset["k_mean"], device=self.device, dtype=torch.float32)
        k_std_val = torch.as_tensor(dataset["k_std"], device=self.device, dtype=torch.float32)
        self.model.k_mean.fill_(k_mean_val.item())
        self.model.k_sigma.fill_(k_std_val.item())

        if self.model.require_du:
            self.du_val = torch.tensor(dataset["du_data_val"], dtype=torch.float32, device=self.device)
            self.du_train = torch.tensor(dataset["du_data_train"], dtype=torch.float32, device=self.device)
        else:
            self.du_train, self.du_val = None, None

    @staticmethod
    def _residual_compute(A_batch:torch.Tensor, f_batch:torch.Tensor, u_batch:torch.Tensor):
        """
        params:
            A_batch: [B, N-2, N-2];
            f_batch: [B, N-2];
            u_batch: [B, N-2];
        """
        return f_batch - torch.einsum("bnm,bm->bn", A_batch, u_batch)

    def _jacobi_step(self, D_inv: torch.Tensor, residual: torch.Tensor):
        return self.relax_factor * D_inv * residual

    def _hybrid_rollout(self, A_batch:torch.Tensor, f_batch:torch.Tensor, k_batch: torch.Tensor,
                        u_curr: Optional[torch.Tensor] = None, horizon: int = 5, **kwargs):
        # Initialization
        if u_curr is not None:
            u_curr = torch.as_tensor(u_curr, dtype=torch.float32, device=self.device)
        else:
            u_curr = torch.zeros_like(f_batch, dtype=torch.float32, device=self.device)

        # compute the initial residual
        r_batch = self._residual_compute(A_batch=A_batch, f_batch=f_batch, u_batch=u_curr)

        # compute the diagonal and its inverse to avoid redundant computation
        D_batch = A_batch.diagonal(dim1=-2, dim2=-1)
        D_inv =torch.reciprocal(D_batch)

        # storing the intermediate solution and residual
        u_seq, res_seq = [], []
        total_steps = horizon * self.update_ratio

        for step in range(total_steps):
            #TODO: this is a problem that might leading to mismatch
            if (step+1) % self.update_ratio == 0:
                pred_dict = self.model(k_x=k_batch, f_x=r_batch, a_mats=A_batch)
                delta_u = pred_dict["u_pred"]
            else:
                delta_u = self._jacobi_step(D_inv=D_inv, residual=r_batch)

            u_curr = u_curr + delta_u
            r_batch = self._residual_compute(A_batch=A_batch, f_batch=f_batch, u_batch=u_curr)

            if (step+1) % self.update_ratio == 0:
                u_seq.append(u_curr)
                res_seq.append(r_batch)

            if len(u_seq) >= horizon:
                break

        return (torch.as_tensor(u_seq, dtype=torch.float32, device=self.device),
                torch.as_tensor(res_seq, dtype=torch.float32, device=self.device))

    def _compute_du(self, batch_func: torch.Tensor, x_nodes: Optional[torch.Tensor] = None):
        """
        params:
            batch_func: [B, N-2,];
            x_nodes: [N-2,];
        """
        if x_nodes is not None:
            x_nodes = torch.as_tensor(x_nodes, dtype=torch.float32, device=self.device)     # [N-2, ]
        else:
            x_nodes = self.x_nodes[1:-1]                                                    # [N-2, ]

        def gradient_compute(single_func, x_data):
            """
            params:
                func_value: [N-2,];
                x_nodes: [N-2,];
            return:
                grad: [N-2,]
            """
            grad_tuple = torch.gradient(input=single_func, spacing=(x_data,), dim=0)
            return grad_tuple[0]

        batch_gradient = vmap(gradient_compute, in_dims=(0, None))
        gradient_value = batch_gradient(batch_func, x_nodes)                # [B, N-2]
        return gradient_value

    def compute_loss(self, u_seq: torch.Tensor, res_seq: torch.Tensor,
                     u_true: Optional[torch.Tensor]=None, du_true: Optional[torch.Tensor]=None):
        """
        params:
            u_seq: [B, N-2];
            res_seq: [B, N-2];
            u_true: [N-2,];
            du_true: [N-2,];
        return:
            loss: scalar
        """
        if self.loss_type == "error":
            if self.loss_norm == "l2":
                loss = torch.nn.functional.mse_loss(u_seq, u_true)
            elif self.loss_norm == "l1":
                loss = torch.nn.functional.l1_loss(u_seq, u_true)
            elif self.loss_norm == "h1":
                du_seq = self._compute_du(u_seq)
                loss = torch.nn.functional.mse_loss(u_seq, u_true) + \
                    self.alpha * torch.nn.functional.mse_loss(du_seq, du_true)
            else:
                raise NotImplementedError("Unknown norm type for the error loss.")

        elif self.loss_type == "residual":
            if self.loss_norm == "l2":
                loss = torch.mean(torch.pow(res_seq, 2))
            elif self.loss_norm == "l1":
                loss = torch.mean(torch.abs(res_seq))
            elif self.loss_norm == "h1":
                dres_seq = self._compute_du(res_seq)
                loss = torch.mean(torch.pow(res_seq, 2)) + self.alpha * torch.mean(torch.pow(dres_seq, 2))
            else:
                raise NotImplementedError("Unknown norm type for the residual loss.")

        else:
            raise NotImplementedError("Unknown loss type.")

        return torch.mean(loss)

    def train_epoch(self):
        #TODO
        self.k_train.shape[0]
        perm = np.random.permutation(data_size)

        for i in range(0, len(perm), self.batch_size):
            end_idx = min(i + self.batch_size, data_size)
            batch_idx = perm[i:end_idx]

            k_batch = self.k_train[batch_idx]
            A_batch = self.a_mats_train[batch_idx]

            f_batch = torch.randn((k_batch.size(0), k_batch.size(1) - 2), device=self.device)
            u_true, du_true = self._compute_true_solution(A_batch, f_batch)

            u_seq, res_seq = self._hybrid_rollout(A_batch, f_batch, k_batch, self.current_horizon)
            du_seq = self._compute_du(u_seq) if self.loss_norm == "h1" else None
            dres_seq = [-A_batch @ du for du in du_seq] if (
                        self.loss_type == "residual" and self.loss_norm == "h1") else None

            loss = self.compute_loss(u_seq=u_seq, res_seq=res_seq, du_seq=du_seq, dres_seq=dres_seq,
                                     u_true=u_true, du_true=du_true)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item() * len(batch_idx)

        return epoch_loss / data_size

    def val_epoch(self):
        self.model.eval()

        # 这个逻辑绝对不对, 对于dynamic而言, 我的可以直接算出来d_res, 不需要du
        # TODO 建议不要把model.require_du之类的放在model里面, 应该作为一个变量放在forward里面,
        #  TODO trainer根据需要确定require_du的大小之类的
        du_val = self.du_val if self.model.require_du else None
        A_val = self.a_mats_val

        with torch.no_grad():
            u_seq, res_seq = self._hybrid_rollout(A_val, self.f_val, self.k_val, self.current_horizon)
            du_seq = self._compute_du(u_seq) if self.loss_norm == "h1" else None
            dres_seq = [-A_val @ du for du in du_seq] if (self.loss_type == "residual" and self.loss_norm == "h1") else None

            val_loss = self.compute_loss(u_seq=u_seq, res_seq=res_seq, du_seq=du_seq, dres_seq=dres_seq,
                                         u_true=self.u_val, du_true=du_val)
        return val_loss.item()

    def _update_curriculum(self, epoch: int):
        if self.curriculum_enabled and (epoch + 1) % self.curriculum_interval == 0:
            self.current_horizon = min(self.current_horizon + 1, self.max_horizon)

    def train(self):
        """
        return training & validation losses
        """
        for epoch in range(self.epochs):
            train_loss = self.train_epoch()
            val_loss = self.val_epoch()
            self._update_curriculum(epoch)

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            if epoch % self.print_interval == 0 or epoch == self.epochs - 1:
                print(f"Epoch [{epoch}/{self.epochs}], Train Loss: {train_loss: .4e}, Val Loss: {val_loss: .4e}, Horizon: {self.current_horizon}")

        return np.array(self.train_losses), np.array(self.val_losses)




