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
                    expected = 255 * np.asarray(plt.get_cmap("hsv")((phase + np.pi) / (2 * np.pi))[:3])
                np.testing.assert_allclose(frames[frame][row, col], expected, atol=12)
    finally:
        plt.close(fig)
