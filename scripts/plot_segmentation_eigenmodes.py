"""Figure-6-style diagnostics of the layer-2 recurrence matrix B, not rho.

Run from the repository root:
    uv run --locked python scripts/plot_segmentation_eigenmodes.py
    uv run --locked python scripts/plot_segmentation_eigenmodes.py --synthetic

Uses the bundled 2shapes images and production recurrence parameters. These
are comparable diagnostics, not an exact reproduction of the paper artwork:
upstream provides no Figure 6 driver.

**Bundled images give real eigenvectors, not traveling waves.** The bundled
2shapes frequencies are binary (-1 background / +1 foreground). Every
foreground pixel shares one frequency, so the foreground block of B is
exactly `(real symmetric K) + i*1*I`: an eigenvector of K shifted by a
constant imaginary offset. Eigenvectors of a real symmetric matrix are real
(up to a global phase), so `Arg[(v_i)_j]` can only land on {0, pi} — this is
forced by the bundled data, not a rendering or gauge choice. Measured on
2shapes/0: eigenvalue imaginary parts are all exactly 1.0 (the omega shift);
eigenvector imaginary parts are ~5e-9 after gauge-fixing (torch's iterative
non-Hermitian `eig`, not the ~1e-15 of a symmetric solver, but the same
conclusion). None of Fig. 6B's continuous 0->pi/2 gradients can appear from
this dataset regardless of solver or display gauge.

`--synthetic` instead builds a **synthetic, clearly-labeled** per-object
frequency gradient (not bundled data): each foreground object gets a linear
ramp along its own principal axis (from the bundled ground-truth `labels`),
giving a genuinely non-uniform, non-real-eigenvector B whose leading modes
show within-object traveling-wave phase gradients comparable to Fig. 6B.

Eigenvector display phases are anchored to each mode's largest component;
amplitudes below 1e-8 of a mode's maximum are left blank.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.cv_rnn import gaussian_sheet_torch, run_2layer_torch
from src.cv_rnn.cv_rnn_segmentation import _canonical_phase
from src.cv_rnn.segmentation_demo import run_segmentation_example


def _initial_state(numel: int, seed: int) -> torch.Tensor:
    """Match segmentation_demo.run_segmentation_example's RNG convention."""
    rng = np.random.RandomState(5489 if seed == 0 else seed)
    phases = (rng.random_sample(numel) - 0.5) * 2 * np.pi
    return torch.from_numpy(np.exp(1j * phases))


def _synthetic_gradient_image(
    image: torch.Tensor, labels: np.ndarray, magnitude: float = 0.4
) -> torch.Tensor:
    """Replace the uniform foreground frequency with a per-object row ramp.

    Synthetic, not bundled data. Each object's rows span
    ``[1 - magnitude, 1 + magnitude]``; background stays at the bundled
    value (-1). Exists only to demonstrate traveling-wave eigenvectors.
    """
    synthetic = image.clone().numpy()
    for label in np.unique(labels):
        if label == 0:
            continue
        rows, cols = np.nonzero(labels == label)
        span = rows.max() - rows.min()
        ramp = ((rows - rows.min()) / span) * 2 - 1 if span else np.zeros_like(rows)
        synthetic[rows, cols] = 1 + magnitude * ramp
    return torch.from_numpy(synthetic)

def _render_eigenmodes(
    image: torch.Tensor,
    states: torch.Tensor,
    mask: torch.Tensor,
    title: str,
) -> tuple[plt.Figure, dict]:
    foreground = ~mask
    shape = tuple(image.shape)
    # Background rows/columns of layer-2 B are zero. Diagonalize only its
    # active block; append the exact zero eigenvalues for the full spectrum.
    weights = gaussian_sheet_torch(*shape, 0.5, 0.0313)
    operator = weights[foreground][:, foreground]
    operator.diagonal().add_(1j * image.T.reshape(-1)[foreground])
    values, vectors = torch.linalg.eig(operator)  # B is not Hermitian.
    order = values.abs().argsort(descending=True)
    values, vectors = values[order], vectors[:, order]
    # Display gauge only; the coefficient solve absorbs the same rotation.
    vectors = _canonical_phase(vectors)
    initial = states[foreground, 0]
    coefficients = torch.linalg.solve(vectors, initial)
    steps = torch.arange(141)
    modes = coefficients[:, None] * values[:, None] ** steps[None, :]
    reconstructed = vectors @ modes
    expected = states[foreground, 60:]
    relative_error = (
        torch.linalg.vector_norm(reconstructed[:, 1:] - expected, dim=0)
        / torch.linalg.vector_norm(expected, dim=0)
    )
    eig_residual = (
        torch.linalg.matrix_norm(operator @ vectors - vectors * values)
        / (torch.linalg.matrix_norm(operator) * torch.linalg.matrix_norm(vectors))
    ).item()
    # This verifies the non-orthogonal coefficient solve, pixel order, and the
    # layer-2 restart against production's full raw complex trajectory.
    np.testing.assert_allclose(
        reconstructed[:, 1:].numpy(), expected.numpy(), rtol=1e-9, atol=1e-10
    )
    assert eig_residual < 1e-12, eig_residual

    figure = plt.figure(figsize=(13, 13), layout="constrained")
    grid = figure.add_gridspec(4, 6, height_ratios=(0.85, 1, 1, 1))
    figure.suptitle(
        f"{title}\n"
        r"$B=K+i\,\mathrm{diag}(\omega)$ · production parameters · not the correlation matrix",
        fontsize=15,
    )
    axis = figure.add_subplot(grid[0, :2])
    axis.imshow(image.numpy(), cmap="twilight_shifted", interpolation="nearest")
    axis.set_title("Input frequency image")
    axis.set_axis_off()
    spectrum = np.concatenate((values.numpy(), np.zeros(int(mask.sum()))))
    for column, data, ylabel in (
        (2, np.abs(spectrum), r"$|\lambda_i|$"),
        (4, np.angle(spectrum), r"$\mathrm{Arg}(\lambda_i)$ [rad]"),
    ):
        axis = figure.add_subplot(grid[0, column : column + 2])
        axis.plot(np.arange(1, len(spectrum) + 1), data, ".", ms=3, color="black")
        axis.set(xlabel="Mode rank (decreasing magnitude)", ylabel=ylabel)
        axis.set_title("Eigenvalue magnitude" if column == 2 else "Eigenvalue phase")
        axis.grid(alpha=0.2)
    # Zero eigenvalues have undefined phase, even though numpy.angle(0)=0.
    figure.axes[-1].lines[0].set_ydata(
        np.where(np.abs(spectrum) > 0, np.angle(spectrum), np.nan)
    )

    cmap = plt.get_cmap("twilight_shifted").copy()
    cmap.set_bad("white")

    def phase_image(axis, vector, subtitle, hide_negligible=False):
        phase = torch.full((image.numel(),), torch.nan, dtype=torch.float64)
        displayed = torch.angle(vector).clone()
        if hide_negligible:
            displayed[vector.abs() < 1e-8 * vector.abs().max()] = torch.nan
        phase[foreground] = displayed
        artist = axis.imshow(
            phase.reshape(shape[1], shape[0]).T.numpy(), cmap=cmap,
            vmin=-np.pi, vmax=np.pi, interpolation="nearest",
        )
        axis.set_title(subtitle, fontsize=10)
        axis.set_axis_off()
        return artist

    mode_axes = []
    for mode in range(6):
        axis = figure.add_subplot(grid[1 + mode // 3, 2 * (mode % 3) : 2 * (mode % 3) + 2])
        phase_artist = phase_image(
            axis, vectors[:, mode],
            f"Mode {mode + 1} · |λ|={abs(values[mode]):.5f}", True,
        )
        mode_axes.append(axis)
    figure.colorbar(phase_artist, ax=mode_axes, shrink=0.8, label="Eigenvector phase [rad]")

    axis = figure.add_subplot(grid[3, :2])
    magnitudes = modes.abs()
    normalized = magnitudes / magnitudes.sum(dim=0)
    for mode in range(6):
        axis.plot(steps, normalized[mode], label=str(mode + 1))
    axis.set(xlabel="Layer-2 updates k", ylabel=r"$|\mu_i(k)| / \sum_j |\mu_j(k)|$",
             title="Leading modal weights")
    axis.legend(title="Mode", ncol=3, fontsize=8)
    axis.grid(alpha=0.2)
    snapshot = 80
    full = reconstructed[:, snapshot]
    truncated = vectors[:, :6] @ modes[:6, snapshot]
    truncated_error = (
        torch.linalg.vector_norm(truncated - full) / torch.linalg.vector_norm(full)
    ).item()
    snapshot_axes = [figure.add_subplot(grid[3, 2:4]), figure.add_subplot(grid[3, 4:6])]
    phase_image(snapshot_axes[0], full, f"All modes · k={snapshot}\nMatches raw recurrence")
    artist = phase_image(
        snapshot_axes[1], truncated,
        f"First 6 modes · k={snapshot}\nComplex relative error={truncated_error:.2e}",
    )
    figure.colorbar(artist, ax=snapshot_axes, shrink=0.75, label="State phase [rad]")
    figure.supxlabel(
        "Display gauge: largest mode component positive real; amplitudes < 10⁻⁸ of mode maximum hidden.\n"
        "Background modes have λ=0 (phase undefined). Modal weights use all foreground modes in the denominator.",
        fontsize=9,
    )
    return figure, {
        "eigenpair_relative_residual": eig_residual,
        "max_trajectory_relative_error": relative_error.max().item(),
        "six_mode_relative_error_at_k80": truncated_error,
    }


def plot_eigenmodes(image_index: int, seed: int) -> tuple[plt.Figure, dict]:
    example = run_segmentation_example("2shapes", image_index=image_index, seed=seed)
    figure, diagnostics = _render_eigenmodes(
        example.image, example.states, example.mask,
        f"Layer-2 recurrence eigenmodes · 2shapes/{image_index} · seed {seed}",
    )
    diagnostics["image_index"] = image_index
    return figure, diagnostics


def plot_synthetic_eigenmodes(
    image_index: int, seed: int, magnitude: float = 0.4
) -> tuple[plt.Figure, dict]:
    from scipy.io import loadmat

    from defns import DATA_DIR

    data = loadmat(Path(DATA_DIR) / "2shapes.mat")
    bundled_image = torch.from_numpy(
        np.asarray(data["images"][:, :, image_index], dtype=np.float64)
    )
    labels = data["labels"][:, :, image_index]
    synthetic_image = _synthetic_gradient_image(bundled_image, labels, magnitude)
    initial = _initial_state(synthetic_image.numel(), seed)
    states, mask = run_2layer_torch(synthetic_image, initial_state=initial)
    bundled = run_segmentation_example("2shapes", image_index=image_index, seed=seed)
    figure, diagnostics = _render_eigenmodes(
        synthetic_image, states, mask,
        f"Layer-2 recurrence eigenmodes · SYNTHETIC per-object gradient · "
        f"2shapes/{image_index} · seed {seed}",
    )
    diagnostics["image_index"] = image_index
    diagnostics["synthetic_gradient_magnitude"] = magnitude
    diagnostics["synthetic_background_pixels"] = int(mask.sum())
    diagnostics["bundled_background_pixels"] = int(bundled.mask.sum())
    return figure, diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-indices", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=Path("plots"))
    parser.add_argument(
        "--synthetic", action="store_true",
        help="use a synthetic per-object frequency gradient instead of the bundled binary image",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for image_index in args.image_indices:
        if args.synthetic:
            figure, diagnostics = plot_synthetic_eigenmodes(image_index, args.seed)
            name = f"segmentation-eigenmodes-2shapes-{image_index}-synthetic.png"
        else:
            figure, diagnostics = plot_eigenmodes(image_index, args.seed)
            name = f"segmentation-eigenmodes-2shapes-{image_index}.png"
        destination = args.output_dir / name
        figure.savefig(destination, dpi=140, bbox_inches="tight")
        plt.close(figure)
        print(destination, diagnostics)


if __name__ == "__main__":
    main()
