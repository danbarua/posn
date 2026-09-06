"""Animate complex pixel states with MATLAB column-major image layout."""

import numpy as np
import torch
from matplotlib.figure import Figure
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt


def _finite_clim(values: np.ndarray) -> tuple[float, float]:
    """Return a non-degenerate (vmin, vmax) covering ``values``."""
    if values.size == 0:
        return -np.pi, np.pi
    vmin, vmax = float(np.min(values)), float(np.max(values))
    if vmax > vmin:
        return vmin, vmax
    pad = 1e-12 if vmin == 0.0 else abs(vmin) * 1e-12
    return vmin - pad, vmax + pad


def animate_dynamics(
    states: np.ndarray | torch.Tensor,
    image_shape: tuple[int, int],
    layer_1_steps: int = 60,
    *,
    interval: int = 100,
) -> tuple[Figure, FuncAnimation]:
    """Animate phase dynamics with paper Fig. 3 colormap conventions.

    Layer 1 uses viridis scaled to each frame's finite phase range. Layer-1
    coupling (α=0.5, σ=0.9) globally phase-locks within a few steps; the
    background/foreground split that the mask reads is a milliradian offset
    around a shared phase. HSV is circular, so that offset maps both clusters
    to red. A sequential map with per-frame scaling makes it visible, matching
    the paper's "phase (scaled)" panels.

    Layer 2 uses HSV over [-π, π]. Masked (NaN) pixels are white.

    ``states`` is complex ``(H*W, T)`` in MATLAB column-major pixel order.
    """
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

    layer1_cmap = plt.get_cmap("viridis").copy()
    layer1_cmap.set_bad("#ffffff")
    layer2_cmap = plt.get_cmap("hsv").copy()
    layer2_cmap.set_bad("#ffffff")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axis("off")

    phase_image_0 = np.ma.masked_invalid(
        np.angle(states[:, 0]).reshape((H, W), order="F")
    )
    if layer_1_steps == 0:
        cmap0, clim0 = layer2_cmap, (-np.pi, np.pi)
    else:
        cmap0, clim0 = layer1_cmap, _finite_clim(phase_image_0.compressed())

    im = ax.imshow(
        phase_image_0,
        cmap=cmap0,
        interpolation="nearest",
        vmin=clim0[0],
        vmax=clim0[1],
    )

    title = ax.set_title("")
    layer_label = ax.text(
        0.02, 0.98, "", transform=ax.transAxes, verticalalignment="top", fontsize=10
    )

    def update(frame_idx: int):
        phase_image = np.ma.masked_invalid(
            np.angle(states[:, frame_idx]).reshape((H, W), order="F")
        )
        im.set_data(phase_image)
        if frame_idx >= layer_1_steps:
            im.set_cmap(layer2_cmap)
            im.set_clim(-np.pi, np.pi)
            layer_label.set_text(f"Layer 2, iteration {frame_idx + 1}")
        else:
            im.set_cmap(layer1_cmap)
            im.set_clim(*_finite_clim(phase_image.compressed()))
            layer_label.set_text(f"Layer 1, iteration {frame_idx + 1}")
        title.set_text(f"t={frame_idx + 1}")
        return [im, title, layer_label]

    anim = FuncAnimation(
        fig,
        update,
        frames=range(T),
        interval=interval,
        blit=False,
        repeat=True,
    )

    return fig, anim
