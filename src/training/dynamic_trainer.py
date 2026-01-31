import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import gc
import time
import torch
import warnings
import numpy as np

import matplotlib.pyplot as plt
from typing import Optional
from torch import vmap

from src.neural_operator import NeuralOperatorBase
from src.utils.visualization import plot_test_samples


class DynamicTrainer:
    def __init__(self, model: NeuralOperatorBase,
                 print_interval: int = 1000,
                 plot_interval: int = 1000,
                 plot_save_dir: Optional[str] = None,
                 ):

        # Initialization
        self.model = model
        self.device = self.model.device
        self.use_autograd = self.model.config.training.loss.get("use_autograd", False)

        # training hyperparameters
        self.epochs = self.model.config.training.num_epoch
        self.batch_size = self.model.config.training.batch_size

        self.alpha = self.model.config.training.loss.grad_alpha
        self.loss_type = self.model.config.training.loss.type.lower()
        self.loss_norm = self.model.config.training.loss.norm.lower()
        self.relative_loss = self.model.config.training.loss.get("relative", True)
        self.relative_eps = self.model.config.training.loss.get("relative_eps", 1e-12)

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

        # whether to load the reference gradient for solution
        self.need_du_true = self.loss_type == "error" and self.loss_norm == "h1"

        # print and visualization setting
        self.print_interval = print_interval
        self.plot_interval = plot_interval
        self.plot_save_dir = plot_save_dir
        self.sole_plot_dir = os.path.join(plot_save_dir, "sole_operator") if plot_save_dir else None
        self.rollout_plot_dir = os.path.join(plot_save_dir, "hybrid_rollout") if plot_save_dir else None

        # dataset initialization
        self.train_losses, self.val_losses = [], []
        self.k_train = self.f_train = self.u_train = self.du_train = self.a_mats_train = None
        self.x_nodes = self.k_val = self.f_val = self.u_val = self.du_val = self.a_mats_val = None
        self._f_true_current = self._df_true_current = None

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

        if self.need_du_true:
            self.du_val = torch.tensor(dataset["du_data_val"], dtype=torch.float32, device=self.device)
            self.du_train = torch.tensor(dataset["du_data_train"], dtype=torch.float32, device=self.device)
        else:
            self.du_val = self.du_train = None

    @staticmethod
    def _residual_compute(A_batch: torch.Tensor, f_batch: torch.Tensor, u_batch: torch.Tensor):
        """
        params:
            A_batch: [B, N-2, N-2];
            f_batch: [B, N-2];
            u_batch: [B, N-2];
        """
        return f_batch - torch.einsum("bnm,bm->bn", A_batch, u_batch)

    def _jacobi_step(self, D_inv: torch.Tensor, residual: torch.Tensor):
        return self.relax_factor * D_inv * residual

    def _hybrid_rollout(self, A_batch: torch.Tensor, f_batch: torch.Tensor, k_batch: torch.Tensor,
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
        D_inv = torch.reciprocal(D_batch)

        # storing the intermediate solution and residual
        u_seq, res_seq = [], []
        total_steps = horizon * self.update_ratio

        for step in range(total_steps):
            # neural operator first in training, as when applied in fine grid
            # smoother has little impact on solutions
            # training on initial residual will be more helpful

            if step % self.update_ratio == 0:
                pred_dict = self.model(k_x=k_batch, f_x=r_batch, a_mats=A_batch)
                delta_u = pred_dict["u_pred"]
            else:
                delta_u = self._jacobi_step(D_inv=D_inv, residual=r_batch)

            u_curr = u_curr + delta_u
            r_batch = self._residual_compute(A_batch=A_batch, f_batch=f_batch, u_batch=u_curr)

            if step % self.update_ratio == 0:
                u_seq.append(u_curr)
                res_seq.append(r_batch)

            if len(u_seq) >= horizon:
                break

        return torch.stack(u_seq), torch.stack(res_seq)

    def _grad_fd(self, batch_func: torch.Tensor, x_nodes: Optional[torch.Tensor] = None):
        """
        params:
            batch_func: [S, B, N-2,] or list/tuple of [B, N-2] tensors;
            x_nodes: [N-2,];
        """
        if x_nodes is not None:
            x_nodes = torch.as_tensor(x_nodes, dtype=torch.float32, device=self.device)  # [N-2, ]
        else:
            x_nodes = self.x_nodes[1:-1]  # [N-2, ]

        def gradient_compute(single_func, x_data):
            """
            params:
                single_func: [B, N-2,];
                x_nodes: [N-2,];
            return:
                grad: [B, N-2]
            """
            grad_tuple = torch.gradient(input=single_func, spacing=(x_data,), dim=1)
            return grad_tuple[0]

        batch_gradient = vmap(gradient_compute, in_dims=(0, None))
        return batch_gradient(batch_func, x_nodes)

    def compute_loss(self, u_seq: torch.Tensor, res_seq: torch.Tensor,
                     u_true: Optional[torch.Tensor] = None,
                     du_true: Optional[torch.Tensor] = None):
        """
        params:
            u_seq:  list of [B, N-2];
            res_seq: list of [B, N-2];
            u_true: [B, N-2,];
            du_true: [B, N-2,];
        return:
            loss: scalar
        """
        if self.loss_type == "error":

            warnings.filterwarnings("ignore", message="Using a target size *")

            if self.loss_norm == "l2":
                loss = self._relative_error(u_seq, u_true, ord_val=2)
            elif self.loss_norm == "l1":
                loss = self._relative_error(u_seq, u_true, ord_val=1)
            elif self.loss_norm == "h1":
                du_seq = self._grad_fd(u_seq, x_nodes=self.x_nodes[1:-1])
                loss = self._relative_error(u_seq, u_true, ord_val=2) + \
                       self.alpha * self._relative_error(du_seq, du_true, ord_val=2)
            else:
                raise NotImplementedError("Unknown norm type for the error loss.")

        elif self.loss_type == "residual":
            if self.loss_norm == "l2":
                loss = self._relative_residual(res_seq, f_true=self._f_true_current, ord_val=2)
            elif self.loss_norm == "l1":
                loss = self._relative_residual(res_seq, f_true=self._f_true_current, ord_val=1)
            elif self.loss_norm == "h1":
                dres_seq = self._grad_fd(res_seq, x_nodes=self.x_nodes[1:-1])
                self._df_true_current = torch.gradient(self._f_true_current, spacing=(self.x_nodes[1:-1],), dim=-1)[0]

                loss = self._relative_residual(res_seq, f_true=self._f_true_current, ord_val=2) + \
                       self.alpha * self._relative_residual(dres_seq, f_true=self._df_true_current, ord_val=2)
            else:
                raise NotImplementedError("Unknown norm type for the residual loss.")

        else:
            raise NotImplementedError("Unknown loss type.")

        return torch.mean(loss)

    def train_epoch(self):
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
            self._f_true_current = f_batch

            u_true = self.u_train[batch_idx]
            du_true = self.du_train[batch_idx] if self.need_du_true else None

            u_seq, res_seq = self._hybrid_rollout(A_batch=A_batch, f_batch=f_batch, k_batch=k_batch,
                                                  u_curr=None, horizon=self.current_horizon)
            self._f_true_current = f_batch
            loss = self.compute_loss(u_seq=u_seq, res_seq=res_seq, u_true=u_true, du_true=du_true)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item() * len(batch_idx)

        return epoch_loss / data_size

    def val_epoch(self):
        # For validation, we only check the performance of a sole operator
        self.model.eval()
        with torch.no_grad():
            pred_dict = self.model(k_x=self.k_val, f_x=self.f_val, a_mats=self.a_mats_val)

            sole_pred = pred_dict["u_pred"]
            val_loss = torch.nn.functional.mse_loss(sole_pred, self.u_val)
        return val_loss.item(), sole_pred

    def _relative_error(self, u_pred: torch.Tensor, u_true: torch.Tensor, ord_val: int=2):
        if not self.relative_loss:
            if ord_val == 1:
                return torch.nn.functional.l1_loss(u_pred, u_true)
            elif ord_val == 2:
                return torch.nn.functional.mse_loss(u_pred, u_true)
            else:
                raise ValueError("only l1 and l2 norm are supported")

        else:
            if ord_val == 1:
                diff_norm = torch.linalg.vector_norm(u_pred - u_true, ord=1, dim=-1)
                ref_norm = torch.linalg.vector_norm(u_true, ord=1, dim=-1)
                return torch.mean(diff_norm / torch.clamp(ref_norm, min=self.relative_eps))
            elif ord_val == 2:
                diff_norm = torch.linalg.vector_norm(u_pred - u_true, ord=2, dim=-1)
                ref_norm = torch.linalg.vector_norm(u_true, ord=2, dim=-1)
                return torch.mean(diff_norm / torch.clamp(ref_norm, min=self.relative_eps))

    def _relative_residual(self, res_seq: torch.Tensor, ord_val: int = 2,
                           f_true: Optional[torch.Tensor] = None) -> torch.Tensor:
        if not self.relative_loss:
            if ord_val == 1:
                return torch.mean(torch.abs(res_seq))
            if ord_val == 2:
                return torch.mean(torch.pow(res_seq, 2))
        else:
            if f_true is None:
                raise ValueError("f_true is required for relative residual loss.")

            if ord_val == 1:
                res_norm = torch.linalg.vector_norm(res_seq, ord=1, dim=-1)
                f_norm = torch.linalg.vector_norm(f_true, ord=1, dim=-1)
                return torch.mean(res_norm / torch.clamp(f_norm, min=self.relative_eps))
            elif ord_val == 2:
                res_norm = torch.linalg.vector_norm(res_seq, ord=2, dim=-1)
                f_norm = torch.linalg.vector_norm(f_true, ord=2, dim=-1)
                return torch.mean(res_norm / torch.clamp(f_norm, min=self.relative_eps))

    def _update_curriculum(self, epoch: int):
        if self.curriculum_enabled and (epoch + 1) % self.curriculum_interval == 0:
            self.current_horizon = min(self.current_horizon + 1, self.max_horizon)

    def train(self):
        """
        return training & validation losses
        """
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)
            torch.cuda.synchronize(self.device)

        start_time = time.time()
        for epoch in range(self.epochs):
            train_loss = self.train_epoch()
            val_loss, val_pred = self.val_epoch()
            self._update_curriculum(epoch)

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            if epoch % self.print_interval == 0 or epoch == self.epochs - 1:
                print(f"Epoch [{epoch}/{self.epochs}], Train Loss: {train_loss: .4e}, "
                      f"Val Loss: {val_loss: .4e}, Horizon: {self.current_horizon}")

            self._maybe_sole_validation(epoch, val_pred)
            self._maybe_rollout_validation(epoch)

        if torch.cuda.is_available():
            torch.cuda.synchronize(self.device)
        end_time = time.time()
        total_time = end_time - start_time

        print("\n" + "=" * 50)
        print(f" (Dynamic) Time and Memory Usage Report")
        print(f" Horizon Setting       : K={self.max_horizon}")
        print(f" Total Wall-clock Time : {total_time:.4f} sec")
        print(f" Avg Time per Epoch    : {total_time / self.epochs:.4f} sec")

        if torch.cuda.is_available():
            peak_bytes = torch.cuda.max_memory_allocated(self.device)
            peak_mb = peak_bytes / (1024 ** 2)
            print(f" Peak GPU Memory       : {peak_mb:.2f} MB")
        print("=" * 50 + "\n")


        self.train_losses, self.val_losses = np.array(self.train_losses), np.array(self.val_losses)
        self._plot_loss_history(out_path=self.plot_save_dir)
        return self.train_losses, self.val_losses

    def _maybe_sole_validation(self, epoch: int, val_pred: torch.Tensor):
        if self.plot_interval is None:
            return

        if self.sole_plot_dir is None:
            return

        should_plot = (epoch % self.plot_interval == 0) or (epoch == self.epochs - 1)
        if not should_plot:
            return

        x_nodes = self.x_nodes[1:-1]
        u_true = self.u_val.detach().cpu().numpy()

        plot_test_samples(epoch_index=epoch + 1,
                          x_nodes=x_nodes,
                          u_test_pred=val_pred.detach().cpu().numpy(),
                          u_test=u_true,
                          out_dir=self.sole_plot_dir)

    def _maybe_rollout_validation(self, epoch):
        if self.plot_interval is None:
            return

        if self.rollout_plot_dir is None:
            return

        should_plot = (epoch % self.plot_interval == 0) or (epoch == self.epochs - 1)
        if not should_plot:
            return

        x_nodes = self.x_nodes[1:-1]
        u_true = self.u_val.detach().cpu().numpy()

        self.model.eval()
        with torch.no_grad():
            rollout_seq, _ = self._hybrid_rollout(A_batch=self.a_mats_val,
                                                  f_batch=self.f_val,
                                                  k_batch=self.k_val,
                                                  u_curr=None,
                                                  horizon=self.max_horizon)
            rollout_pred = rollout_seq[-1]

        plot_test_samples(epoch_index=epoch + 1,
                          x_nodes=x_nodes,
                          u_test_pred=rollout_pred.detach().cpu().numpy(),
                          u_test=u_true,
                          out_dir=self.rollout_plot_dir)

    def _plot_loss_history(self, out_path: str):
        if len(self.train_losses) == 0:
            return

        os.makedirs(out_path, exist_ok=True)
        fig_path = os.path.join(out_path, "loss_curve.png")
        epochs = np.arange(1, len(self.train_losses) + 1)

        plt.figure(figsize=(8, 5))
        plt.plot(epochs, np.array(self.train_losses), label="Train Loss")
        plt.plot(epochs, np.array(self.val_losses), label="Validation Loss")
        plt.yscale("log")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(fig_path, dpi=150)
        plt.close()

    def reset_memory(self):
        """
        Clean Memory
        """
        # 1. Delete data
        del self.k_train, self.f_train, self.u_train, self.du_train, self.a_mats_train
        del self.x_nodes, self.k_val, self.f_val, self.u_val, self.du_val, self.a_mats_val

        # 2. Delete Optimizer
        del self.optimizer

        # 3. Delete Model
        self.model = None

        # 4. GC Clean
        gc.collect()

        # 5. Clean PyTorch Cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("Dynamic Trainer Memory reset complete.")