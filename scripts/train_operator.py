import gc
import os
import torch

import numpy as np
from box import Box

from src.neural_operator import create_no
from src.training import StaticTrainer, DynamicTrainer
from src.utils.cfg_util import load_config

CONFIG_WILDCARD = "diffusion1d*"    # config filename wildcard
TRAINER_TYPE = "static"           # "static" or "dynamic"
DATASET_PATH = None               # path to .npz dataset; None -> config default
LOSS_NORM = None                  # "l1", "l2", or "h1"; None -> config default
LOSS_TYPE = None                  # "error" or "residual"; None -> config default
SAVE_AFTER_TRAIN = True           # save final checkpoint after training
PRINT_INTERVAL = 50              # Interval of printing; DeepONet recommend 1000, FNS recommend 100
PLOT_INTERVAL = None              # Interval of plotting validation samples; None = disable

# --- Checkpoint saving -------------------------------------------------------
SAVE_INTERVAL = 100               # static only: save every N epochs (None = disabled)
SAVE_BEST = False                  # save best-val-loss checkpoint (both trainers)


def apply_training_overrides(cfg: Box,
                             loss_type=LOSS_TYPE,
                             loss_norm=LOSS_NORM,
                             dataset_path=DATASET_PATH,
                             trainer_type=TRAINER_TYPE,
                             ):
    if trainer_type is not None:
        cfg.training.mode = trainer_type

    if dataset_path is not None:
        cfg["dataset_path"] = dataset_path

    if loss_type is not None:
        cfg.training.loss.type = loss_type

    if loss_norm is not None:
        cfg.training.loss.norm = loss_norm

    return cfg


def select_trainer(model, trainer_type: str = TRAINER_TYPE, save_path=None,
                   print_interval=PRINT_INTERVAL, plot_interval=PLOT_INTERVAL,
                   checkpoint_name="model",
                   save_interval=SAVE_INTERVAL, save_best=SAVE_BEST):
    """Return the proper trainer object based on the configuration."""
    trainer_type = trainer_type.lower()

    if trainer_type == "dynamic":
        return DynamicTrainer(model, plot_save_dir=save_path, print_interval=print_interval,
                              plot_interval=plot_interval,
                              checkpoint_dir=save_path, checkpoint_name=checkpoint_name,
                              save_interval=save_interval, save_best=save_best)

    if trainer_type == "static":
        return StaticTrainer(model, plot_save_dir=save_path, print_interval=print_interval,
                             plot_interval=plot_interval,
                             checkpoint_dir=save_path, checkpoint_name=checkpoint_name,
                             save_interval=save_interval, save_best=save_best)

    raise ValueError(f"Trainer type {trainer_type} not supported")


def ckpt_dir(cfg: Box):
    n_dim = cfg.problem.n_dim
    operator_type = cfg.training.operator_type.lower()
    problem_name, trainer_type = cfg.problem.type.lower(), cfg.training.mode.lower()
    loss_type, loss_norm = cfg.training.loss.type.lower(), cfg.training.loss.norm.lower()

    ckpt_base = f"./checkpoints/{operator_type}_{problem_name}{n_dim}d/{trainer_type}_{loss_type}_{loss_norm}"
    return ckpt_base


def train_operator(config_wildcard=CONFIG_WILDCARD,
                   loss_type=LOSS_TYPE,
                   loss_norm=LOSS_NORM,
                   dataset_path=DATASET_PATH,
                   trainer_type=TRAINER_TYPE,
                   save_after_train=SAVE_AFTER_TRAIN,
                   print_interval=PRINT_INTERVAL,
                   plot_interval=PLOT_INTERVAL,
                   save_interval=SAVE_INTERVAL,
                   save_best=SAVE_BEST,
                   ):
    """Train a neural operator according to the given configuration."""
    cfg = load_config(config_wildcard)
    cfg = apply_training_overrides(cfg, loss_type, loss_norm, dataset_path, trainer_type)

    # define the save path
    ckpt_base = ckpt_dir(cfg)
    os.makedirs(ckpt_base, exist_ok=True)

    # checkpoint filename prefix: e.g. "diffusion_1D_Grid31" or "diffusion2d_2D_Grid31x31"
    problem = cfg.problem.type.lower()
    n_dim = cfg.problem.n_dim
    mesh_cfg = cfg.data.mesh
    if n_dim == 1:
        grid_label = mesh_cfg.grid_num
    else:
        grid_label = f"{mesh_cfg.grid_nx}x{mesh_cfg.grid_ny}"
    ckpt_name = f"{problem}_{n_dim}D_Grid{grid_label}"

    model = create_no(cfg)
    trainer = select_trainer(model, trainer_type, save_path=ckpt_base,
                             print_interval=print_interval, plot_interval=plot_interval,
                             checkpoint_name=ckpt_name,
                             save_interval=save_interval, save_best=save_best)
    dataset = np.load(cfg.dataset_path, allow_pickle=True)

    trainer.load_dataset(dataset)
    train_loss, val_loss = trainer.train()

    print("\nFinished training.")
    print(f"Final train loss: {train_loss[-1]:.4e}")
    print(f"Final val loss:   {val_loss[-1]:.4e}")

    if save_after_train:
        file_name = os.path.basename(cfg.model_save_path)
        cfg.model_save_path = os.path.join(ckpt_base, file_name + ".pt")
        cfg.loss_save_path = os.path.join(ckpt_base, file_name + "_loss.npz")

        # save the checkpoint
        model.save_model(cfg.model_save_path)

        # save the loss to the npz file
        np.savez_compressed(cfg.loss_save_path, train_loss=train_loss, val_loss=val_loss)

        # print the saving information
        print(f"Model and Loss saved to {os.path.dirname(cfg.model_save_path)}")

    # ===============================================================
    print(f"Cleaning up memory for trainer type: {trainer_type}...")

    if hasattr(trainer, "reset_memory"):
        trainer.reset_memory()

    del trainer
    del model
    del dataset

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("Memory cleanup finished.\n")
    # ==============================================================


def batch_train(trainer_types=("static", "dynamic", ),
                loss_types=("error", "residual"),
                loss_norms=("l2", "l1", "h1"),
                config_wildcard=CONFIG_WILDCARD,
                dataset_path=DATASET_PATH,
                save_after_train=SAVE_AFTER_TRAIN,
                print_interval=PRINT_INTERVAL,
                plot_interval=PLOT_INTERVAL,
                save_interval=SAVE_INTERVAL,
                save_best=SAVE_BEST,
                ):
    """
    Checkpoints saved under:
    ./checkpoints/{operator_type}_{problem_name}{dimension}/{trainer_type}_{loss_type}_{loss_norm}/

    Within each run:
      {name}_ep{N}.pt   — periodic snapshot (static only, every save_interval epochs)
      {name}_best.pt    — best validation loss so far (both trainers)
      {name}.pt         — final checkpoint (if save_after_train=True)
    """
    for trainer_type in trainer_types:
        for loss_type in loss_types:
            for loss_norm in loss_norms:
                print("=" * 80)
                print(f"Start training: trainer={trainer_type}, loss_type={loss_type}, loss_norm={loss_norm}.")
                print("=" * 80)

                try:
                    train_operator(config_wildcard=config_wildcard,
                                   loss_type=loss_type,
                                   loss_norm=loss_norm,
                                   dataset_path=dataset_path,
                                   trainer_type=trainer_type,
                                   save_after_train=save_after_train,
                                   print_interval=print_interval,
                                   plot_interval=plot_interval,
                                   save_interval=save_interval,
                                   save_best=save_best,
                    )
                except Exception as e:
                    print(f"FAILED: trainer={trainer_type}, loss_type={loss_type}, loss_norm={loss_norm} -> {e}")
                    continue


if __name__ == "__main__":
    batch_train()
