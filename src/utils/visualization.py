import os
from typing import Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch


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

    test_fig_dir = os.path.join(out_dir, "test")
    os.makedirs(test_fig_dir, exist_ok=True)

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

    test_fig_path = os.path.join(test_fig_dir, f"Fig_Test_epoch_{epoch_index}.png")
    plt.savefig(test_fig_path, dpi=150)
    plt.close(fig)


__all__ = ["plot_test_samples"]