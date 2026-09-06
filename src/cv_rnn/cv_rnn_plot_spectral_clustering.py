import numpy as np
import torch
from matplotlib.figure import Figure
import matplotlib.pyplot as plt


def plot_spectral_clustering(
    prj: np.ndarray | torch.Tensor,
    x: np.ndarray | torch.Tensor,
    phase_iter: int = 120,
) -> Figure:
    """
    Create a 3D scatter plot of spectral clustering projection colored by phase.

    Parameters
    ----------
    prj : np.ndarray | torch.Tensor
        Projection array of shape (N_fg, n_dims, n_windows) where N_fg is the
        number of foreground nodes, n_dims is typically 3 (for 3D plot), and
        n_windows is the number of sliding windows. Contains foreground rows only.
    x : np.ndarray | torch.Tensor
        Complex state array of shape (N_fg, T) containing foreground nodes only.
    phase_iter : int, optional
        MATLAB-style 1-based iteration index for phase coloring (default 120).
        This is converted to 0-based Python index internally.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure containing the 3D scatter plot.

    Raises
    ------
    ValueError
        If phase_iter is out of bounds or array dimensions are inconsistent.
    """
    # Convert torch to numpy
    if isinstance(prj, torch.Tensor):
        prj = prj.detach().cpu().numpy()
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()

    # Validate dimensions
    if prj.ndim != 3:
        raise ValueError(f"`prj` must be 3-D array (N, n_dims, n_windows), got shape {prj.shape}")
    if x.ndim != 2:
        raise ValueError(f"`x` must be 2-D array (N, T), got shape {x.shape}")
    if not np.iscomplexobj(x):
        raise ValueError("`x` must be complex-valued")

    n_fg = prj.shape[0]
    n_dims = prj.shape[1]
    n_windows = prj.shape[2]
    n_samples, n_timesteps = x.shape

    if n_fg != n_samples:
        raise ValueError(
            f"First dimension of prj ({n_fg}) does not match first dimension of x ({n_samples})"
        )

    # Convert phase_iter to 0-based index
    phase_idx = phase_iter - 1
    if phase_idx < 0 or phase_idx >= n_timesteps:
        raise ValueError(
            f"phase_iter={phase_iter} (0-based index {phase_idx}) out of bounds for x with T={n_timesteps}"
        )

    if n_dims < 3:
        raise ValueError(
            f"Need at least 3 dimensions for 3D plot, got n_dims={n_dims}"
        )

    # Extract last window projection
    last_proj = prj[:, :3, -1]  # (N_fg, 3) - take first 3 dimensions

    # Get phase colors from x at phase_iter
    phase = np.angle(x[:, phase_idx])  # (N_fg,)

    # Create figure with 3D subplot
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    # Create scatter plot
    scatter = ax.scatter(
        last_proj[:, 0],
        last_proj[:, 1],
        last_proj[:, 2],
        c=phase,
        cmap="hsv",
        s=50,
        alpha=0.8,
        edgecolors="none",
    )

    # Set phase colormap limits to [-π, π]
    scatter.set_clim([-np.pi, np.pi])

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, pad=0.1, shrink=0.8)
    cbar.set_label("phase (rad)", fontsize=10)

    # Set labels
    ax.set_xlabel("dimension 1", fontsize=10)
    ax.set_ylabel("dimension 2", fontsize=10)
    ax.set_zlabel("dimension 3", fontsize=10)
    ax.set_title("spectral clustering projection", fontsize=11)

    return fig
