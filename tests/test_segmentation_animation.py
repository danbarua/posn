"""Check actual GIF pixels, not private animation callbacks or nonempty figures."""

import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
import numpy as np
from PIL import Image
import torch

from src.cv_rnn import animate_dynamics


def test_gif_preserves_pixel_order_phase_and_layer_boundary(tmp_path):
    phases = np.array([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi, np.pi / 4])
    states = np.repeat(np.exp(1j * phases)[:, None], 3, axis=1)
    states[1, 1:] = np.nan  # bottom-left pixel disappears only in layer 2
    states[2, 2] = np.exp(-0.5j * np.pi)  # top-middle changes in the final frame
    fig, animation = animate_dynamics(torch.from_numpy(states), (2, 3), layer_1_steps=1)
    path = tmp_path / "dynamics.gif"
    try:
        animation.save(path, writer=PillowWriter(fps=10), dpi=40)
        with Image.open(path) as gif:
            assert gif.n_frames == 3
            width, height = gif.size
            frames = []
            for frame in range(3):
                gif.seek(frame)
                frames.append(np.asarray(gif.convert("RGB")).copy())
        canvas_width, canvas_height = fig.canvas.get_width_height()
        positions = []
        for pixel in range(6):
            # Explicit MATLAB mapping: row = index % H, column = index // H.
            x, y = fig.axes[0].transData.transform((pixel // 2, pixel % 2))
            positions.append((height - 1 - round(y * height / canvas_height), round(x * width / canvas_width)))
        for frame in range(3):
            for pixel, (row, col) in enumerate(positions):
                if pixel == 1 and frame >= 1:
                    expected = np.array([255, 255, 255])
                else:
                    phase = -np.pi / 2 if (frame, pixel) == (2, 2) else phases[pixel]
                    if frame == 0:
                        vmin, vmax = float(phases.min()), float(phases.max())
                        frac = (phase - vmin) / (vmax - vmin)
                        expected = 255 * np.asarray(plt.get_cmap("viridis")(frac)[:3])
                    else:
                        expected = 255 * np.asarray(
                            plt.get_cmap("hsv")((phase + np.pi) / (2 * np.pi))[:3]
                        )
                np.testing.assert_allclose(frames[frame][row, col], expected, atol=12)
    finally:
        plt.close(fig)


def test_layer1_milliradian_offset_is_not_hsv_red(tmp_path):
    """HSV autoscaling maps a tight [θ, θ+ε] range to both ends of the wheel (red).

    Layer 1 of the paper's α=0.5, σ=0.9 sheet does exactly this: global lock with
    a milliradian background/foreground offset. Sequential viridis must keep the
    two clusters visually distinct.
    """
    low, high = 1.0, 1.005
    phases = np.array([low, low, high, high])
    states = np.repeat(np.exp(1j * phases)[:, None], 3, axis=1)
    fig, animation = animate_dynamics(torch.from_numpy(states), (2, 2), layer_1_steps=2)
    path = tmp_path / "tight.gif"
    try:
        animation.save(path, writer=PillowWriter(fps=10), dpi=40)
        with Image.open(path) as gif:
            gif.seek(1)
            frame = np.asarray(gif.convert("RGB"))
            width, height = gif.size
        canvas_width, canvas_height = fig.canvas.get_width_height()
        samples = []
        for pixel in (0, 2):
            x, y = fig.axes[0].transData.transform((pixel // 2, pixel % 2))
            row = height - 1 - round(y * height / canvas_height)
            col = round(x * width / canvas_width)
            samples.append(frame[row, col].astype(float))
        low_color, high_color = samples
        hsv_red = np.array([255.0, 0.0, 0.0])
        viridis_low = 255 * np.asarray(plt.get_cmap("viridis")(0.0)[:3])
        viridis_high = 255 * np.asarray(plt.get_cmap("viridis")(1.0)[:3])
        np.testing.assert_allclose(low_color, viridis_low, atol=12)
        np.testing.assert_allclose(high_color, viridis_high, atol=12)
        assert np.linalg.norm(low_color - high_color) > 50
        assert np.linalg.norm(low_color - hsv_red) > 50
        assert np.linalg.norm(high_color - hsv_red) > 50
    finally:
        plt.close(fig)
