import torch
import numpy as np

from typing import Optional
from functorch import vmap

from src.neural_operator import NeuralOperatorBase


class DynamicTrainer:
    def __init__(self, model: NeuralOperatorBase, print_interval: int = 1000):
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
            # If same as the inference: Jacobi first and then Neural Operator
            # NO are trained on smoothed dataset, which lead to mismatch
            # as the smoothing effects varies much between coarse grid (training) and fine grid (inference)
            if step % self.update_ratio == 0:
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
        #TODO: 把是否当场生成rhs data做成一个选项放在config里面
        # 如果选择当场生成, 再使用不同的逻辑
        # 我现在这个逻辑是针对于static dataset的
        self.model.train()

        epoch_loss = 0.0
        data_size = self.u_train.shape[0]
        perm = np.random.permutation(data_size)

        for i in range(0, data_size, self.batch_size):
            end_idx = min(i + self.batch_size, data_size)
            batch_idx = perm[i:end_idx]

            f_batch = self.f_train[batch_idx]
            k_batch = self.k_train[batch_idx]
            A_batch = self.a_mats_train[batch_idx]

            u_true = self.u_train[batch_idx]
            du_true = self.du_train[batch_idx]

            u_seq, res_seq = self._hybrid_rollout(A_batch=A_batch, f_batch=f_batch, k_batch=k_batch,
                                                  u_curr=None, horizon=self.current_horizon)
            loss = self.compute_loss(u_seq=u_seq, res_seq=res_seq, u_true=u_true, du_true=du_true)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item() * len(batch_idx)

        return epoch_loss / data_size

    def val_epoch(self):
        #TODO: 这个逻辑有一点奇怪, 没法跟static保持一致
        # 对于dynamic而言, 我们用autograd算出来的 du, 是对于输入给NO的residual对应的correction e_i的梯度
        # 这个梯度和外界solver的梯度不是完全一致的. d( u_i + e_i - u^\star)
        # 我们不能用autograd算梯度, 或者至少应该做一下处理, 比如 de_i (autograd) + du_i(previous step) - du_val 才合理
        # 在这个程序实现里, 我们使用了中心差分来算grad, 所以我们的对于require_du, require_res, require_dres的判断标准发生了很大的改变
        # 至少我们算dres不需要a_mats了. 直接用res中心差分算就完了

        #TODO: 由于上面的问题, 我认为不要把model.require_du之类的放在model里面
        # 应该在trainer里面自己来判断需要load哪种data, 以及怎么样train
        # 对于NO, 那么则是设置bool 变量, 你要告诉no, 我到底需不需要你autograd出来的du和dres
        # 而不是根据require_du, require_dres什么的来判断, 我认为这个逻辑需要放在trainer里面
        # 同理static trainer也需要更改,来保持统一

        # 对于validation而言, 我们就不做dynamic 展开了, 直接预测
        self.model.eval()
        with torch.no_grad():
            pred_dict = self.model(k_x=self.k_val, f_x=self.f_val, a_mats=self.a_mats_val)
            val_loss = torch.nn.functional.mse_loss(pred_dict["u_pred"], self.u_val)
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




