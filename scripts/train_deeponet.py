import os
import argparse
import numpy as np

from typing import Optional
from src.training import StaticTrainer, DynamicTrainer
from src.utils.cfg_util import load_config
from src.neural_operator import create_no


def _load_dataset(dataset_path: str):
    """Load the cached dataset from disk.

    Parameters
    ----------
    dataset_path: str
        Path to the ``.npz`` dataset file.
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"Dataset not found at {dataset_path}. Please generate it with scripts/generate_dataset.py first."
        )

    return np.load(dataset_path)


def _select_trainer(model, mode: str):
    """Return the proper trainer class based on the configuration."""
    if mode.lower() == "dynamic":
        return DynamicTrainer(model)
    return StaticTrainer(model)


def train_deeponet(
    config_wildcard: str = "diffusion1d*",
    dataset_name: Optional[str] = None,
    model_save_name: Optional[str] = None,
):
    """Train a DeepONet model according to the given configuration.

    Parameters
    ----------
    config_wildcard: str
        Wildcard for locating the YAML configuration file under ``configs/``.
    dataset_name: str | None
        Optional dataset filename (located under ``dataset/``) to override the default.
    model_save_name: str | None
        Optional checkpoint filename (under ``checkpoints/``) to override the default.
    """

    config = load_config(
        config_wildcard=config_wildcard,
        dataset_name=dataset_name,
        model_save_name=model_save_name,
    )

    dataset = _load_dataset(config.dataset_path)
    model = create_no(config)

    trainer = _select_trainer(model, config.training.mode)
    trainer.load_dataset(dataset)

    train_loss, val_loss = trainer.train()
    model.save_model(config.model_save_path)

    loss_save_path = os.path.splitext(config.model_save_path)[0] + "_loss.npz"
    np.savez_compressed(loss_save_path, train_loss=train_loss, val_loss=val_loss)
    print(f"Training finished. Loss curves saved to {loss_save_path}")


def main():
    parser = argparse.ArgumentParser(description="Train DeepONet with configured settings")
    parser.add_argument("--config", default="diffusion1d*", help="Config wildcard under configs/")
    parser.add_argument("--dataset", default=None, help="Dataset filename under dataset/")
    parser.add_argument("--save-name", default=None, help="Checkpoint filename under checkpoints/")
    args = parser.parse_args()

    train_deeponet(args.config, args.dataset, args.save_name)


if __name__ == "__main__":
    main()