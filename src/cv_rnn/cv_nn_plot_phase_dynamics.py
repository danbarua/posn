# Animation API replacement for plot_dynamics and plot_dynamics_animated.
# This module provides animate_dynamics(), the canonical animation function.
# (plot_dynamics and plot_dynamics_animated are removed as per contract.)

import numpy as np
import torch
from matplotlib.figure import Figure
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt


def animate_dynamics(
    states: np.ndarray | torch.Tensor,
    image_shape: tuple[int, int],
    layer_1_steps: int = 60,
    *,
    interval: int = 100,
) -> tuple[Figure, FuncAnimation]:
    """
    Animate complex phase dynamics with transparent NaN backgrounds.

    Parameters
    ----------
    states : np.ndarray | torch.Tensor
        Complex array of shape (N, T) where N is total pixels (H*W in F-order)
        and T is number of timesteps.
    image_shape : tuple[int, int]
        (H, W) dimensions of the image.
    layer_1_steps : int, optional
        Timestep where layer 1 ends (default 60). Color limits switch to [-π, π]
        starting at this index.
    interval : int, optional
        Milliseconds between frames (default 100).

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure containing the animation.
    anim : matplotlib.animation.FuncAnimation
        The animation object.

    Raises
    ------
    ValueError
        If states is not 2-D, image_shape does not match states shape,
        layer_1_steps is out of range, or complex input is required.
    """
    # Convert torch to numpy
    if isinstance(states, torch.Tensor):
        states = states.detach().cpu().numpy()

    if states.ndim != 2:
        raise ValueError(f"`states` must be a 2-D array (N, T), got shape {states.shape}")

    if not np.iscomplexobj(states):
        raise ValueError("`states` must be complex-valued")

    H, W = image_shape
    N, T = states.shape

    if N != H * W:
        raise ValueError(
            f"Total pixels N={N} does not match image shape H*W={H*W}"
        )

    if layer_1_steps < 0 or layer_1_steps >= T:
        raise ValueError(
            f"layer_1_steps={layer_1_steps} out of range [0, {T})"
        )

    if interval <= 0:
        raise ValueError(f"interval must be positive, got {interval}")

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axis("off")

    # Initial frame: reshape from F-order flat to (H, W)
    phase_0 = np.angle(states[:, 0])
    phase_image_0 = phase_0.reshape((H, W), order="F")

    # Create initial imshow with hsv colormap
    im = ax.imshow(
        phase_image_0,
        cmap="hsv",
        interpolation="nearest",
        vmin=-np.pi,
        vmax=np.pi,
    )

    # Set initial alpha based on NaN values
    alpha_0 = ~np.isnan(phase_image_0)
    im.set_alpha(alpha_0.astype(float))

    # Title and text for iteration counter
    title = ax.set_title("")
    layer_label = ax.text(0.02, 0.98, "", transform=ax.transAxes, 
                          verticalalignment="top", fontsize=10)

    def update(frame_idx: int):
        """Update function for FuncAnimation."""
        # Reshape from F-order flat to (H, W)
        phase = np.angle(states[:, frame_idx])
        phase_image = phase.reshape((H, W), order="F")

        # Update image data
        im.set_data(phase_image)

        # Update alpha: transparent where NaN, opaque elsewhere
        alpha = ~np.isnan(phase_image)
        im.set_alpha(alpha.astype(float))

        # Adjust color limits based on layer
        if frame_idx >= layer_1_steps:
            im.set_clim(-np.pi, np.pi)
        else:
            # Before layer 2, auto-scale to show phase variation
            valid_mask = ~np.isnan(phase_image)
            if np.any(valid_mask):
                valid_phases = phase_image[valid_mask]
                if len(valid_phases) > 0:
                    vmin, vmax = np.percentile(valid_phases, [5, 95])
                    # Ensure symmetric range for better visualization
                    vmax = max(abs(vmin), abs(vmax))
                    vmin = -vmax
                    im.set_clim(vmin, vmax)

        # Update title and layer label
        title.set_text(f"t={frame_idx + 1}")
        if frame_idx < layer_1_steps:
            layer_label.set_text(f"Layer 1, iteration {frame_idx + 1}")
        else:
            layer_label.set_text(
                f"Layer 2, iteration {frame_idx - layer_1_steps + 1}"
            )

        return [im, title, layer_label]

    # Create animation using FuncAnimation without blitting
    # (blitting fails with alpha changes)
    anim = FuncAnimation(
        fig,
        update,
        frames=range(T),
        interval=interval,
        blit=False,
        repeat=True,
    )

    return fig, anim
