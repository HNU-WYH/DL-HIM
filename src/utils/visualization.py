import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import numpy as np
import matplotlib.pyplot as plt

from typing import Dict, Optional, Sequence, Tuple, Union

def _prepare_x_nodes(x_nodes: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """Ensure ``x_nodes`` is a 1D numpy array."""

    if isinstance(x_nodes, torch.Tensor):
        x_nodes = x_nodes.detach().cpu().numpy()

    if x_nodes.ndim == 2 and x_nodes.shape[1] == 1:
        x_nodes = x_nodes.squeeze(1)

    return x_nodes


def _subplot_indices(index: int, samples_per_row: int) -> Tuple[int, int]:
    row_idx = index // samples_per_row
    col_base = (index % samples_per_row) * 2
    return row_idx, col_base


def plot_test_samples(
    epoch_index: int,
    x_nodes: Union[np.ndarray, torch.Tensor],
    u_test_pred: np.ndarray,
    u_test: np.ndarray,
    out_dir: str = "results",
    num_of_test_plots: int = 16,
    u_val_pred: Optional[np.ndarray] = None,
    labels: Sequence[str] = ("Pred", "Pred2"),
):
    """Plot predictions against ground truth for selected validation samples.

    Args:
        epoch_index: Epoch counter used in the figure title/filename.
        x_nodes: Spatial grid points (1D array-like).
        u_test_pred: Primary prediction array with shape ``[num_samples, grid]``.
        u_test: Ground-truth solutions with the same shape as ``u_test_pred``.
        out_dir: Base directory to store the figures.
        num_of_test_plots: Maximum number of samples to visualize.
        u_val_pred: Optional secondary predictions to compare.
        labels: Legend labels for the prediction curves.
    """

    os.makedirs(out_dir, exist_ok=True)

    x_nodes = _prepare_x_nodes(x_nodes)

    num_of_samples = u_test_pred.shape[0]
    num_to_plot = min(num_of_test_plots, num_of_samples)

    samples_per_row = 2
    num_rows = (num_to_plot - 1) // samples_per_row + 1

    fig_width_per_sample = 6.0
    fig_height_per_row = 4.0
    fig = plt.figure(
        figsize=(fig_width_per_sample * samples_per_row, fig_height_per_row * num_rows)
    )

    for i in range(num_to_plot):
        row_idx, col_base = _subplot_indices(i, samples_per_row)

        ax1 = plt.subplot(num_rows, samples_per_row * 2, row_idx * (samples_per_row * 2) + col_base + 1)
        ax1.plot(x_nodes, u_test_pred[i], "-b", label=labels[0])
        ax1.plot(x_nodes, u_test[i], "-r", label="True")
        if u_val_pred is not None:
            ax1.plot(x_nodes, u_val_pred[i], "-g", label=labels[1])
        ax1.set_title(f"Test#{i + 1} Pred vs True")
        ax1.legend()
        ax1.set_xlabel("x")
        ax1.set_ylabel("u")

        ax2 = plt.subplot(num_rows, samples_per_row * 2, row_idx * (samples_per_row * 2) + col_base + 2)
        ax2.plot(x_nodes, u_test_pred[i] - u_test[i], "-b", label=labels[0])
        if u_val_pred is not None:
            ax2.plot(x_nodes, u_val_pred[i] - u_test[i], "-g", label=labels[1])
        ax2.set_title(f"Test#{i + 1} Error")
        ax2.legend()
        ax2.set_xlabel("x")
        ax2.set_ylabel("Error")

    l2_error = np.sqrt(np.mean((u_test_pred - u_test) ** 2)) * 1e4
    fig.suptitle(f"[Epoch {epoch_index}] L2 Err: {l2_error:.4f} × 10⁻⁴", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    test_fig_path = os.path.join(out_dir, f"Fig_Test_epoch_{epoch_index}.png")
    plt.savefig(test_fig_path, dpi=150)
    plt.close(fig)


def plot_loss_history(
    loss_files: Union[str, Sequence[str]],
    labels: Optional[Sequence[str]] = None,
    save_path: Optional[str] = None,
    title: str = "Training vs Validation Loss",
):
    """Plot train/validation loss curves from one or more ``npz`` files.

    Args:
        loss_files: Path or list of paths to ``npz`` files containing
            ``train_loss`` and ``val_loss`` arrays.
        labels: Optional custom labels matching ``loss_files``. Defaults to the
            basename of each file.
        save_path: If provided, the figure is saved to this path instead of
            shown interactively.
        title: Figure title.
    """

    if isinstance(loss_files, str):
        loss_files = [loss_files]

    if labels is None:
        labels = [os.path.basename(os.path.dirname(path)) for path in loss_files]
    elif len(labels) != len(loss_files):
        raise ValueError("Number of labels must match number of loss files")

    cmap = plt.cm.get_cmap("viridis")
    colors = cmap(np.linspace(0, 1, len(loss_files)))

    plt.figure(figsize=(8, 5))
    for idx, (path, label) in enumerate(zip(loss_files, labels)):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Loss file '{path}' not found")

        data = np.load(path)
        if "train_loss" not in data or "val_loss" not in data:
            raise KeyError(f"Loss file '{path}' is missing 'train_loss' or 'val_loss'")

        plt.plot(data["train_loss"], label=f"Train ({label})", alpha=1.0, color=colors[idx],)
        plt.plot(data["val_loss"], label=f"Val ({label})",  alpha=0.5, color=colors[idx],)

    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.grid(True, linestyle=":", linewidth=0.5)
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()


def plot_case_predictions(plot_predictions: Dict[int, Dict[str, Dict[str, np.ndarray]]]):

    # The number of samples for plotting
    num_cases = len(plot_predictions)

    if num_cases == 0:
        raise ValueError("No predictions provided for plotting")
    fig, axes = plt.subplots(num_cases, 2, figsize=(12, 3.5 * num_cases), sharex=True)

    if num_cases == 1:
        axes = np.array([axes])

    # row_idx = sample index; label = iteration_method (Hybrid-AA)
    # pred = {'error': scalar, 'u_true': array, 'x_nodes': array, 'u_pred': array}
    for row_idx, (sample_idx, pred) in enumerate(plot_predictions.items()):
        for label, results in pred.items():
            # Solution Plotting
            ax_pred, ax_err = axes[row_idx]
            ax_pred.plot(results["x_nodes"], results["u_pred"], label=label + f":{results['error']:.2e}")

            # Error Plotting
            error = results["u_pred"] - results["u_true"]
            ax_err.plot(results["x_nodes"], error, label=label + f":{results['error']:.2e}")

            # finish the solution plotting
            ax_pred.plot(results["x_nodes"], results["u_true"], label="True", color="black", linewidth=2)
            ax_pred.legend()
            ax_pred.set_ylabel("u(x)")
            ax_pred.set_title(f"Prediction vs True: {sample_idx}")
            ax_pred.grid(True, linestyle=":", linewidth=0.5)

            # finish the error plotting
            ax_err.legend()
            ax_err.set_ylabel("Error")
            ax_err.set_title(f"Absolute Error: {sample_idx}")
            ax_err.grid(True, linestyle=":", linewidth=0.5)

            axes[-1, 0].set_xlabel("x")
            axes[-1, 1].set_xlabel("x")

            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.show()


__all__ = ["plot_test_samples", "plot_loss_history", "plot_case_predictions"]

