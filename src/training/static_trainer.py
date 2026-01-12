import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import numpy as np

from typing import Optional
import matplotlib.pyplot as plt

from src.neural_operator import NeuralOperatorBase
from src.utils.visualization import plot_test_samples


class StaticTrainer:
    def __init__(self, model: NeuralOperatorBase,
                 print_interval: int = 1000,
                 plot_interval: Optional[int] = 1000,
                 plot_save_dir: Optional[str] = None,):

        self.model = model
        self.device = self.model.device
        self.epochs = self.model.config.training.num_epoch
        self.batch_size = self.model.config.training.batch_size

        self.alpha = self.model.config.training.loss.grad_alpha
        self.loss_type = self.model.config.training.loss.type.lower()
        self.loss_norm = self.model.config.training.loss.norm.lower()

        self.relative_loss = self.model.config.training.loss.get("relative", False)
        self.relative_eps = self.model.config.training.loss.get("relative_eps", 1e-12)

        # whether A_mats/du_red is required in trainer
        self.need_A = self.loss_type == "residual"
        self.need_du_true = self.loss_type == "error" and self.loss_norm == "h1"

        # whether compute gradient of solution inside the model by autograd
        self.use_autograd = getattr(self.model, "use_autograd", False)
        self.compute_autograd = self.need_du_true and self.use_autograd

        self.print_interval = print_interval
        self.plot_interval = plot_interval
        self.plot_save_dir = plot_save_dir

        self.train_losses, self.val_losses = [], []
        self.k_train = self.f_train = self.u_train = self.du_train = self.a_mats_train = None
        self.x_nodes = self.k_val = self.f_val = self.u_val = self.du_val = self.a_mats_val = None

        self._f_true_current = None
        self._du_f_true_current = None

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.model.config.training.lr,
            weight_decay=self.model.config.training.weight_decay
        )

    def load_dataset(self, dataset):
        # load coordinates of x
        self.x_nodes = torch.tensor(dataset["x_data"], dtype=torch.float32, device=self.device)

        self.k_train = torch.tensor(dataset["k_data_train"], dtype=torch.float32, device=self.device)
        self.f_train = torch.tensor(dataset["f_data_train"], dtype=torch.float32, device=self.device)
        self.u_train = torch.tensor(dataset["u_data_train"], dtype=torch.float32, device=self.device)

        self.k_val = torch.tensor(dataset["k_data_val"], dtype=torch.float32, device=self.device)
        self.f_val = torch.tensor(dataset["f_data_val"], dtype=torch.float32, device=self.device)
        self.u_val = torch.tensor(dataset["u_data_val"], dtype=torch.float32, device=self.device)

        # Update k_mean and k_sigma
        k_mean_val = torch.as_tensor(dataset["k_mean"], device=self.device, dtype=torch.float32)
        k_std_val = torch.as_tensor(dataset["k_std"], device=self.device, dtype=torch.float32)
        self.model.k_mean.fill_(k_mean_val.item())
        self.model.k_sigma.fill_(k_std_val.item())

        if self.need_A:
            self.a_mats_train = torch.tensor(dataset["a_mats_train"], dtype=torch.float32, device=self.device)
            self.a_mats_val = torch.tensor(dataset["a_mats_val"], dtype=torch.float32, device=self.device)
        else:
            self.a_mats_train, self.a_mats_val = None, None

        if self.need_du_true:
            self.du_val = torch.tensor(dataset["du_data_val"], dtype=torch.float32, device=self.device)
            self.du_train = torch.tensor(dataset["du_data_train"], dtype=torch.float32, device=self.device)
        else:
            self.du_train, self.du_val = None, None

    def compute_loss(self, u_pred=None, u_true=None, du_pred=None, du_true=None, res=None, dres=None, **kwargs):
        if self.loss_type == "error":
            if self.loss_norm == "l2":
                return self._relative_error(u_pred, u_true, ord_val=2)
            elif self.loss_norm == "l1":
                return self._relative_error(u_pred, u_true, ord_val=1)
            elif self.loss_norm == "h1":
                return self._relative_error(u_pred, u_true, ord_val=2) + \
                    self.alpha * self._relative_error(du_pred, du_true, ord_val=2)
            else:
                raise NotImplementedError("Unknown norm type for the error loss.")

        elif self.loss_type == "residual":
            if self.loss_norm == "l2":
                return self._relative_residual(res, f_true=self._f_true_current, ord_val=2)
            elif self.loss_norm == "l1":
                return self._relative_residual(res, f_true=self._f_true_current, ord_val=1)
            elif self.loss_norm == "h1":
                self._du_f_true_current = self._grad_fd(self._f_true_current, x_nodes=self.x_nodes[1:-1])
                return self._relative_residual(res, f_true=self._f_true_current, ord_val=2) + \
                    self.alpha * self._relative_residual(dres, f_true=self._df_true_current, ord_val=2)
            else:
                raise NotImplementedError("Unknown norm type for the residual loss.")
        else:
            raise NotImplementedError("Unknown loss type.")

    def train_epoch(self):
        """
        :return: average training loss
        """
        self.model.train()

        epoch_loss = 0.0
        data_size = self.k_train.shape[0]
        perm = np.random.permutation(data_size)

        for i in range(0, len(perm), self.batch_size):
            end_idx = min(i + self.batch_size, data_size)

            batch_idx = perm[i:end_idx]
            k_batch = self.k_train[batch_idx]
            f_batch = self.f_train[batch_idx]
            u_batch = self.u_train[batch_idx]
            self._f_true_current = f_batch

            du_batch = self.du_train[batch_idx] if self.need_du_true else None
            a_mats_batch = self.a_mats_train[batch_idx] if self.need_A else None

            # 前向传播
            pred_dict = self.model(k_x=k_batch, f_x=f_batch, a_mats=a_mats_batch,
                                   compute_du=self.compute_autograd)
            u_pred = pred_dict["u_pred"]
            du_pred = self._get_du_pred(u_pred, pred_dict, self.x_nodes[1:-1])

            res, dres = None, None
            if self.loss_type == "residual":
                res = self._compute_residual(a_mats=a_mats_batch, f_batch=f_batch, u_pred=u_pred)
                if self.loss_norm == "h1":
                    dres = self._grad_fd(res, x_nodes=self.x_nodes[1:-1])

            loss = self.compute_loss(u_pred=u_pred, u_true=u_batch,
                                     du_pred=du_pred, du_true=du_batch,
                                     res=res, dres=dres)

            # 反向传播并优化
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item() * len(batch_idx)

        return epoch_loss / data_size

    def val_epoch(self):
        """
        average validation loss
        """
        self.model.eval()
        with torch.no_grad():
            pred_dict = self.model(k_x=self.k_val, f_x=self.f_val, a_mats=self.a_mats_val)
            val_loss = torch.nn.functional.mse_loss(pred_dict["u_pred"], self.u_val)
        return val_loss.item(), pred_dict["u_pred"]

    def train(self):
        """
        training losses and validation losses
        """
        for epoch in range(self.epochs):
            train_loss = self.train_epoch()
            val_loss, val_pred = self.val_epoch()

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            if epoch % self.print_interval == 0 or epoch == self.epochs - 1:
                print(f"Epoch [{epoch}/{self.epochs}], Train Loss: {train_loss: .4e}, Val Loss: {val_loss: .4e}")

            self._maybe_plot_validation(epoch, val_pred)

        self.train_losses, self.val_losses = np.array(self.train_losses), np.array(self.val_losses)
        self._plot_loss_history(out_path=self.plot_save_dir)
        return self.train_losses, self.val_losses

    def _maybe_plot_validation(self, epoch, val_pred):
        if self.plot_interval is None:
            return

        if self.plot_save_dir is None:
            return

        if (epoch % self.plot_interval == 0) or (epoch == self.epochs - 1):
            val_pred = val_pred.detach().cpu().numpy()
            plot_test_samples(epoch_index=epoch + 1,
                              x_nodes=self.x_nodes[1:-1],
                              u_test_pred=val_pred,
                              u_test=self.u_val.detach().cpu().numpy(),
                              out_dir=os.path.join(self.plot_save_dir, "sole_operator"),
                              )
        else:
            return

    def _plot_loss_history(self, out_path: str):
        if len(self.train_losses) == 0:
            return

        os.makedirs(out_path, exist_ok=True)
        fig_path = os.path.join(out_path, "loss_curve.png")
        epochs = np.arange(1, len(self.train_losses) + 1)

        plt.figure(figsize=(8, 5))
        plt.plot(epochs, self.train_losses, label="Train Loss")
        plt.plot(epochs, self.val_losses, label="Validation Loss")
        plt.yscale("log")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(fig_path, dpi=150)
        plt.close()

    def _get_du_pred(self, u_pred: torch.Tensor, pred_dict: dict, x_nodes: Optional[torch.Tensor] = None) -> \
    Optional[torch.Tensor]:
        """
        Choose du_pred from model output when available and requested; otherwise fall back to FD.
        """
        if self.loss_type != "error" or self.loss_norm != "h1":
            return None

        if self.use_autograd:
            model_du = pred_dict.get("du_pred", None)
            if model_du is not None:
                return model_du

        return self._grad_fd(u_pred, x_nodes=x_nodes)

    @staticmethod
    def _compute_residual(a_mats: torch.Tensor, f_batch: torch.Tensor, u_pred: torch.Tensor) -> torch.Tensor:
        """
        params:
            a_mats: [B, N-2, N-2];
            f_batch: [B, N-2];
            u_pred: [B, N-2];
        """
        if a_mats is None:
            raise ValueError("a_mats is required to compute residual in residual loss mode.")
        return f_batch - torch.einsum("bnm,bm->bn", a_mats, u_pred)

    def _grad_fd(self, func: torch.Tensor, x_nodes: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Central difference gradient using torch.gradient support batches.
        params:
         func: [B, N-2]
         x_nodes: [N-2,] (interior nodes)
        """
        if x_nodes is not None:
            x_nodes = torch.as_tensor(x_nodes, dtype=func.dtype, device=self.device)
        else:
            x_nodes = self.x_nodes[1:-1]

        return torch.gradient(input=func, spacing=(x_nodes,), dim=1)[0]

    def _relative_error(self, u_pred: torch.Tensor, u_true: torch.Tensor, ord_val: int = 2):
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

