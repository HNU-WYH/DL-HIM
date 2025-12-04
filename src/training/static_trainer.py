import torch
import numpy as np

from src.neural_operator import NeuralOperatorBase


class StaticTrainer:
    def __init__(self, model: NeuralOperatorBase, print_interval: int = 100):
        self.model = model
        self.device = self.model.config.device
        self.epochs = self.model.config.training.num_epoch
        self.batch_size = self.model.config.training.batch_size

        self.alpha = self.model.config.training.loss.grad_alpha
        self.loss_type = self.model.config.training.loss.type.lower()
        self.loss_norm = self.model.config.training.loss.norm.lower()

        self.print_interval = print_interval

        self.train_losses, self.val_losses = [], []
        self.k_val, self.f_val, self.u_val, self.du_val, self.a_mats_val = None, None, None, None, None
        self.k_train, self.f_train, self.u_train, self.du_train, self.a_mats_train = None, None, None, None, None

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.model.config.training.lr,
            weight_decay=self.model.config.training.weight_decay
        )

    def load_dataset(self, dataset):
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

        if self.model.require_res:
            self.a_mats_train = torch.tensor(dataset["a_mats_train"], dtype=torch.float32, device=self.device)
            self.a_mats_val = torch.tensor(dataset["a_mats_val"], dtype=torch.float32, device=self.device)
        else:
            self.a_mats_train, self.a_mats_val = None, None

        if self.model.require_du:
            self.du_val = torch.tensor(dataset["du_data_val"], dtype=torch.float32, device=self.device)
            self.du_train = torch.tensor(dataset["du_data_train"], dtype=torch.float32, device=self.device)
        else:
            self.du_train, self.du_val = None, None

    def compute_loss(self, u_pred=None, u_true=None, du_pred=None, du_true=None, res=None, dres=None, **kwargs):
        if self.loss_type == "error":
            if self.loss_norm == "l2":
                return torch.nn.functional.mse_loss(u_pred, u_true)
            elif self.loss_norm == "l1":
                return torch.nn.functional.l1_loss(u_pred, u_true)
            elif self.loss_norm == "h1":
                return torch.nn.functional.mse_loss(u_pred, u_true) + \
                    self.alpha * torch.nn.functional.mse_loss(du_pred, du_true)
            else:
                raise NotImplementedError("Unknown norm type for the error loss.")

        elif self.loss_type == "residual":
            if self.loss_norm == "l2":
                return torch.mean(torch.pow(res, 2))
            elif self.loss_norm == "l1":
                return torch.mean(torch.abs(res))
            elif self.loss_norm == "h1":
                return torch.mean(torch.pow(res, 2)) + self.alpha * torch.mean(torch.pow(dres, 2))
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

            du_batch = self.du_train[batch_idx] if self.model.require_du else None
            a_mats_batch = self.a_mats_train[batch_idx] if self.model.require_res else None

            # 前向传播
            pred_dict = self.model(k_x=k_batch, f_x=f_batch, a_mats=a_mats_batch)
            loss = self.compute_loss(u_true=u_batch, du_true=du_batch, **pred_dict)

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

        du_val = self.du_val if self.model.require_du else None
        a_mats_val = self.a_mats_val if self.model.require_res else None

        with torch.no_grad():
            pred_dict = self.model(k_x=self.k_val, f_x=self.f_val, a_mats=a_mats_val)
            val_loss = self.compute_loss(u_true=self.u_val, du_true=du_val, **pred_dict)
        return val_loss.item()

    def train(self):
        """
        training losses and validation losses
        """
        for epoch in range(self.epochs):
            train_loss = self.train_epoch()
            val_loss = self.val_epoch()

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            if epoch % self.print_interval == 0 or epoch == self.epochs - 1:
                print(f"Epoch [{epoch}/{self.epochs}], Train Loss: {train_loss: .4e}, Val Loss: {val_loss: .4e}")

        return np.array(self.train_losses), np.array(self.val_losses)
