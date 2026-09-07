"""Reproducible counterparts to paper Figures 3 and 4A/C using bundled inputs.

Reference examples use unit amplitudes and MT19937 phases, as in the MATLAB
release, rather than Torch's different random stream. No labels enter inference.
The paper's coins/MNIST inputs and its 1,000-image benchmarks are not bundled.
"""

from dataclasses import dataclass
from pathlib import Path
import json

import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.figure import Figure
import numpy as np
from scipy.io import loadmat, savemat
from sklearn.metrics import adjusted_rand_score
import torch

from defns import DATA_DIR
from .cv_nn_plot_phase_dynamics import animate_dynamics
from .cv_rnn_segmentation import run_2layer_torch, spatiotemporal_segmentation_torch


@dataclass(frozen=True)
class SegmentationExample:
    dataset: str
    image_index: int
    seed: int
    n_clusters: int
    image: torch.Tensor
    states: torch.Tensor
    mask: torch.Tensor
    labels: torch.Tensor
    projection: torch.Tensor
    ground_truth: np.ndarray | None

    def scores(self) -> dict[str, float]:
        """Score shape labels without letting the large background hide errors.

        Natural-image ``lb`` contains multiple scene/edge IDs without a binary
        background convention; do not misreport it as a two-object annotation.
        """
        if self.ground_truth is None:
            return {}
        predicted = self.labels.numpy()
        truth = self.ground_truth
        foreground = truth != 0
        predicted_foreground = predicted != -1
        return {
            "foreground_ari": float(adjusted_rand_score(truth[foreground], predicted[foreground])),
            "foreground_mask_iou": float(
                np.count_nonzero(foreground & predicted_foreground)
                / np.count_nonzero(foreground | predicted_foreground)
            ),
            "background_pixel_accuracy": float(np.mean((predicted == -1)[~foreground])),
        }


def run_segmentation_example(
    dataset: str = "2shapes",
    image_index: int = 0,
    *,
    seed: int | None = None,
    n_clusters: int | None = None,
    dataset_dir: str | Path = DATA_DIR,
) -> SegmentationExample:
    """Run one reference example, keeping initialization independent of Torch.

    Defaults follow ``cvrnn_image_segmentation.m``: seeds 1/9/1 and cluster counts
    2/3/2 for 2shapes/3shapes/natural. RandomState deliberately uses legacy MT19937
    doubles, not NumPy's newer default_rng or Torch's generator. Seed zero uses
    the original MT19937 default state (5489), MATLAB's factory twister state.
    Explicit exported x0 arrays are still required to establish runtime parity.
    """
    settings = {
        "2shapes": ("2shapes.mat", 1, 2),
        "3shapes": ("3shapes.mat", 9, 3),
        "natural": ("natural_image.mat", 1, 2),
    }
    if dataset not in settings:
        raise ValueError("dataset must be 2shapes, 3shapes or natural")
    filename, default_seed, default_clusters = settings[dataset]
    seed = default_seed if seed is None else seed
    n_clusters = default_clusters if n_clusters is None else n_clusters
    data = loadmat(Path(dataset_dir) / filename)
    if dataset == "natural":
        if image_index != 0:
            raise ValueError("the natural dataset contains only image index zero")
        array = data["im"]
    else:
        if not 0 <= image_index < data["images"].shape[2]:
            raise ValueError("image_index is outside the bundled dataset")
        array = data["images"][:, :, image_index]
    image = torch.from_numpy(np.asarray(array, dtype=np.float64))
    rng = np.random.RandomState(5489 if seed == 0 else seed)
    phases = (rng.random_sample(image.numel()) - 0.5) * 2 * np.pi
    initial = torch.from_numpy(np.exp(1j * phases))
    states, mask = run_2layer_torch(image, initial_state=initial)
    labels, _, _, _, projection = spatiotemporal_segmentation_torch(
        states, image, mask, nt_mask=60, n_clusters=n_clusters
    )
    # Ground truth is read only AFTER inference, solely for evaluation/display.
    truth = None if dataset == "natural" else data["labels"][:, :, image_index]
    return SegmentationExample(
        dataset, image_index, seed, n_clusters, image, states, mask, labels, projection, truth
    )


def plot_segmentation_comparison(example: SegmentationExample) -> Figure:
    """Plot the quantities in Fig. 3 / Fig. 4A,C, not a pixel-identical facsimile.

    All times are one-based stored iterations. Layer-1 colors are independently
    scaled; layer-2 colors use fixed phase limits. Predictions retain their raw
    KMeans labels: ground truth is never used to recolor or repair predictions.
    """
    figure = plt.figure(figsize=(15, 9), layout="constrained")
    grid = figure.add_gridspec(3, 5)
    shape = tuple(example.image.shape)
    phases = np.angle(example.states.numpy()).reshape((*shape, -1), order="F")
    phase_cmap = plt.get_cmap("hsv").copy()
    phase_cmap.set_bad("#eeeeee")
    for row, iterations in enumerate(((1, 2, 5, 20, 60), (61, 90, 120, 150, 180))):
        for column, iteration in enumerate(iterations):
            axis = figure.add_subplot(grid[row, column])
            phase = np.ma.masked_invalid(phases[:, :, iteration - 1])
            limits = (float(phase.min()), float(phase.max())) if row == 0 else (-np.pi, np.pi)
            axis.imshow(phase, cmap="viridis" if row == 0 else phase_cmap,
                        vmin=limits[0], vmax=limits[1], interpolation="nearest")
            axis.set_title(f"Layer {row + 1} · iteration {iteration}", fontsize=10)
            axis.set_axis_off()
    axis = figure.add_subplot(grid[2, 0])
    axis.imshow(example.image.numpy(), cmap="gray", interpolation="nearest")
    axis.set_title("Frequency input")
    axis.set_axis_off()
    axis = figure.add_subplot(grid[2, 1])
    axis.imshow(example.mask.numpy().reshape(shape, order="F"), cmap="gray", vmin=0, vmax=1)
    axis.set_title("Background mask (white)")
    axis.set_axis_off()
    axis = figure.add_subplot(grid[2, 2], projection="3d")
    projected = example.projection[:, :, -1].numpy()
    colors = np.angle(example.states[~example.mask, 119].numpy())
    scatter = axis.scatter(*projected.T, c=colors, cmap="hsv", vmin=-np.pi, vmax=np.pi, s=8)
    axis.set(xlabel="Dimension 1", ylabel="Dimension 2", zlabel="Dimension 3")
    axis.set_title("Projection · iterations 140–180")
    figure.colorbar(scatter, ax=axis, shrink=0.55, label="Phase at iteration 120 (rad)")
    palette = ["#eeeeee"] + [plt.get_cmap("tab10")(i % 10) for i in range(example.n_clusters)]
    cmap = ListedColormap(palette)
    norm = BoundaryNorm(np.arange(-1.5, example.n_clusters + 0.5), cmap.N)
    axis = figure.add_subplot(grid[2, 3])
    axis.imshow(example.labels.numpy(), cmap=cmap, norm=norm, interpolation="nearest")
    axis.set_title("Predicted object labels")
    axis.set_axis_off()
    axis = figure.add_subplot(grid[2, 4])
    if example.ground_truth is not None:
        axis.imshow(example.ground_truth - 1.0, cmap=cmap, norm=norm, interpolation="nearest")
        axis.set_title("Ground truth · IDs arbitrary")
    else:
        axis.text(0.5, 0.5, "Natural image: qualitative comparison\nNo binary object ground truth assumed",
                  ha="center", va="center", wrap=True, transform=axis.transAxes, fontsize=9)
    axis.set_axis_off()
    scores = example.scores()
    score_text = " · ".join(f"{name}={value:.3f}" for name, value in scores.items())
    figure.suptitle(
        f"{example.dataset} · image {example.image_index} · MT19937 seed {example.seed}\n"
        f"Layer 1: independently scaled phase (viridis); layer 2: −π to π (HSV)\n{score_text}",
        fontsize=12,
    )
    return figure


def save_segmentation_comparison(
    example: SegmentationExample, directory: str | Path
) -> dict[str, Path]:
    """Save comparison PNG, full dynamics GIF, shared inputs and numeric metadata."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"{example.dataset}-{example.image_index}"
    paths = {kind: directory / f"{stem}{suffix}" for kind, suffix in {
        "figure": ".png", "animation": ".gif", "inputs": "-inputs.mat", "metadata": ".json"
    }.items()}
    figure = plot_segmentation_comparison(example)
    try:
        figure.savefig(paths["figure"], dpi=140, bbox_inches="tight")
    finally:
        plt.close(figure)
    figure, animation = animate_dynamics(example.states, tuple(example.image.shape))
    try:
        animation.save(paths["animation"], writer=PillowWriter(fps=10), dpi=75)
    finally:
        plt.close(figure)
    savemat(paths["inputs"], {
        "im": example.image.numpy(), "x0": example.states[:, 0].numpy()[:, None],
        "alpha": [0.5, 0.5], "sigma": [0.9, 0.0313], "nt": [60, 200], "seed": example.seed,
    })
    metadata = {
        "dataset": example.dataset, "image_index": example.image_index,
        "seed": example.seed, "initialization": "MT19937 doubles, unit amplitudes",
        "n_clusters": example.n_clusters, "alpha": [0.5, 0.5], "sigma": [0.9, 0.0313],
        "nt": [60, 200], "pixel_order": "MATLAB column-major",
        "analysis_windows_one_based_inclusive": [[60, 100], [100, 140], [140, 180]],
        "supplement_reported_window": [141, 181],
        "scores": example.scores(),
        "scope": "MATLAB demo equations and supplied inputs; not cross-runtime MATLAB parity or the paper's full benchmark",
    }
    paths["metadata"].write_text(json.dumps(metadata, indent=2) + "\n")
    return paths
