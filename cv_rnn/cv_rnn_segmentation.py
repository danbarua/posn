"""
cv_rnn_segmentation.py

A faithful, GPU-ready Python re-implementation of the cv-RNN image–
segmentation pipeline from

    “Image segmentation with traveling waves in an exactly solvable
     recurrent neural network”, Liboni et al., PNAS 122 (1): e2321319121 (2025)

Differences from the original MATLAB release were closed as follows
-------------------------------------------------------------------
✓ full spatio-temporal clustering (`rho`, `eig`, sliding window)
✓ sparse-friendly Gaussian sheet kernel (no huge N×N diagonals)
✓ optional complex‐phase modulation (`phi`)
✓ hyper-parameters exposed with paper defaults
✓ deterministic seeding via an optional torch.Generator argument
✓ background pixels labelled **-1** (clusters start at 0)
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, Literal, Tuple, Union

import numpy as np
import torch
from sklearn.cluster import KMeans

__all__ = [
    "gaussian_sheet_torch",
    "run_2layer_torch",
    "spatiotemporal_segmentation_torch",
]


# --------------------------------------------------------------------------- #
# 1.  Gaussian connectivity sheet                                             #
# --------------------------------------------------------------------------- #
def gaussian_sheet_torch(
    nrow: int,
    ncol: int,
    amp: float,
    sigma: float,
    *,
    device: Union[str, torch.device] = "cpu",
    dtype: torch.dtype = torch.float32,
    phi: float | None = None,
    phi_threshold: float = 0.04,
) -> torch.Tensor:
    """
    Distance–dependent weight matrix  **W ∈ ℂ^{N×N}**  with optional complex
    phase modulation.

    Parameters
    ----------
    nrow, ncol : int
        Image grid dimensions.
    amp, sigma : float
        Amplitude α and spatial spread σ of the Gaussian.
        σ is expressed *relative to the unit square* (same convention
        as the MATLAB version).
    device, dtype : placement arguments
    phi : float, optional
        If given, connections with weight < `phi_threshold` receive a
        distance-dependent phase   *exp(1j·ϕ)*  (see Liboni et al.,
        Methods).
    phi_threshold : float
        Magnitude below which the phase term is applied.

    Notes
    -----
    •  Uses  **torch.cdist** – fine for ≤64×64, but you can switch to a
       separable convolution kernel or sparse CSR outside this helper.
    """
    # Grid centres in (0,1] × (0,1]  as in the MATLAB code
    rows = torch.linspace(1 / nrow, 1.0, nrow, device=device, dtype=dtype)
    cols = torch.linspace(1 / ncol, 1.0, ncol, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(rows, cols, indexing="ij")
    pos = torch.stack((yy.flatten(), xx.flatten()), dim=-1)  # (N,2)

    d_sq = torch.cdist(pos, pos).pow_(2)
    w = amp * torch.exp(-d_sq / (2.0 * sigma**2))

    if phi is not None:
        mask = w < phi_threshold
        # Rescale distance linearly to [ϕ_min,ϕ_max]   (same heuristic)
        phi_min, phi_range = 0.0, phi
        d_norm = (d_sq.sqrt() - d_sq.min()) / (d_sq.max() - d_sq.min())
        phase = (d_norm * phi_range + phi_min)[mask]
        w[mask] = w[mask] * torch.exp(1j * phase)

    return w.to(torch.complex64)


# --------------------------------------------------------------------------- #
# 2.  Two-layer cv-RNN dynamics                                               #
# --------------------------------------------------------------------------- #
def _step(matrix: torch.Tensor, omega: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    One Euler step of     x ← (K + i·diag(ω))·x
    implemented without forming the dense diagonal matrix.
    """
    return matrix @ x + 1j * omega.unsqueeze(1) * x


def run_2layer_torch(
    im: torch.Tensor,
    alpha: Tuple[float, float] = (0.5, 0.5),
    sigma: Tuple[float, float] = (0.9, 0.0313),
    nt: Tuple[int, int] = (60, 200),
    *,
    generator: torch.Generator | None = None,
    device: Union[str, torch.device] = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Parameters
    ----------
    im : (H,W) tensor
        Input image, **values in [-π, π] rad/s** in the paper’s examples.
    alpha, sigma : tuple(float,float)
        Gaussian amplitude & spread for layer 1 and 2.
    nt : (t_mask, t_end)
        Number of iterations for layer 1, total number of iterations.
    generator : torch.Generator, optional
        Pass your own generator for reproducible initial conditions.
    device, dtype : placement arguments

    Returns
    -------
    save_x : (N, nt[1]) complex
        Dynamics of all pixels across time.
    mask   : (N,) bool
        Background mask applied at t = nt[0] − 1.
    """
    im = im.to(device, dtype)
    nrow, ncol = im.shape
    n = nrow * ncol
    t1, t_end = nt

    # ------------------------------------------------------------------ #
    # initial condition                                                  #
    # ------------------------------------------------------------------ #
    rand = torch.rand((n, 1), device=device, generator=generator, dtype=dtype)
    x0 = torch.exp(1j * (rand * 2 * math.pi))  # phase ∈ [0,2π)
    x = x0.clone()

    omega = im.flatten()  # (N,)

    save_x = torch.empty((n, t_end), dtype=torch.complex64, device=device)
    save_x[:, 0] = x0.squeeze()

    # ------------------------------------------------------------------ #
    # layer 1                                                            #
    # ------------------------------------------------------------------ #
    k1 = gaussian_sheet_torch(
        nrow, ncol, alpha[0], sigma[0], device=device, dtype=dtype
    )

    for t in range(1, t1):
        x = _step(k1, omega, x)
        save_x[:, t] = x.squeeze()

    # mask = majority side of mean phase
    phase_t1 = torch.angle(x.squeeze())
    thr = phase_t1.mean()
    mask = (
        phase_t1 > thr
        if (phase_t1 > thr).sum() > (phase_t1 < thr).sum()
        else phase_t1 < thr
    )
    mask_idx = mask.nonzero(as_tuple=False).squeeze()

    # ------------------------------------------------------------------ #
    # layer 2 (masked)                                                   #
    # ------------------------------------------------------------------ #
    k2 = gaussian_sheet_torch(
        nrow, ncol, alpha[1], sigma[1], device=device, dtype=dtype
    ).to(torch.complex64)
    k2[mask, :] = 0
    k2[:, mask] = 0

    omega2 = omega.clone()
    omega2[mask] = 0

    # re-use x0 but zero masked nodes
    x = x0.clone()
    x[mask] = 0

    for t in range(t1, t_end):
        x = _step(k2, omega2, x)
        save_x[:, t] = x.squeeze()

    save_x[mask, t1:t_end] = torch.nan
    return save_x, mask


# --------------------------------------------------------------------------- #
# 3.  Spatio-temporal segmentation                                            #
# --------------------------------------------------------------------------- #
def _similarity_tensor(
    x: torch.Tensor,
    win_start: int,
    win_end: int,
) -> torch.Tensor:
    """
    Compute ρ = (1/Δt) ∑_{t∈window} x(t) · x(t)ᴴ
    """
    p = x[:, win_start:win_end]  # (N, Δt) complex
    #  einsum → (N,N)
    return torch.einsum("it,jt->ij", p, p.conj()) / (win_end - win_start)


def spatiotemporal_segmentation_torch(
    save_x: torch.Tensor,
    im: torch.Tensor,
    mask: torch.Tensor,
    *,
    n_clusters: int = 2,
    dim: Iterable[int] = (0, 1, 2),
    window_size: int = 40,
    window_step: int = 40,
    nt_mask: int,
    device: Union[str, torch.device] = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Full port of `spatiotemporal_segmentation.m`.

    Parameters
    ----------
    save_x : (N,T) complex
        Dynamics from `run_2layer_torch`.
    im     : (H,W)  float
        Image array (only used for shape).
    mask   : (N,)   bool
    n_clusters : int
        Desired number of output segments.
    dim : iterable[int]
        Eigenvector indices to keep for the projection (default 1:3).
    window_size, window_step : int
        Sliding-window parameters (Δt, Δt_step).
    nt_mask : int
        Time index where layer-1 ended ( == nt[0]  originally).
    device : torch.device

    Returns
    -------
    cluster_map  : (H,W) long    (-1 = masked / background)
    rho, V, D, P : see MATLAB code (all tensors on `device`)
    """
    save_x = save_x.to(device)
    n, t_end = save_x.shape
    h, w = im.shape

    # ------------------------------------------------------------------ #
    # sliding-window similarity tensor                                   #
    # ------------------------------------------------------------------ #
    win_starts = torch.arange(nt_mask, t_end - window_size + 1, window_step)
    n_wins = len(win_starts)

    rho = torch.empty((n, n, n_wins), dtype=torch.complex64, device=device)
    V = torch.empty_like(rho)
    D = torch.empty_like(rho)
    prj = torch.empty((n, len(tuple(dim)), n_wins), dtype=torch.float32, device=device)

    for k, ws in enumerate(win_starts):
        we = ws + window_size
        rho_k = _similarity_tensor(save_x, ws, we)
        rho[:, :, k] = rho_k

        # eig(ρ) — real symmetric in practice after taking Re, but follow MATLAB
        vals, vecs = torch.linalg.eig(rho_k)
        order = vals.abs().argsort(descending=True)
        vals, vecs = vals[order], vecs[:, order]
        D[:, :, k] = torch.diag_embed(vals)
        V[:, :, k] = vecs
        prj[:, :, k] = (rho_k.real @ vecs.real[:, dim]).float()

    # ------------------------------------------------------------------ #
    # final clustering on the *last* window’s projection                 #
    # ------------------------------------------------------------------ #
    last_proj = prj[:, :, -1]  # (N, len(dim))
    fg_mask = ~mask
    data = last_proj[fg_mask].cpu().numpy()

    kmeans = KMeans(
        n_clusters=min(n_clusters, data.shape[0]),
        random_state=0,
        n_init="auto",
    ).fit(data)
    labels = kmeans.labels_.astype(np.int64)

    cluster_idx = torch.full((n,), -1, dtype=torch.long, device=device)
    cluster_idx[fg_mask] = torch.from_numpy(labels).to(device)

    return (
        cluster_idx.view(h, w),
        rho,
        V,
        D,
        prj,
    )


# --------------------------------------------------------------------------- #
# 4.  Minimal example (2-Shapes dataset from the paper)                       #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import argparse
    import matplotlib.pyplot as plt
    from torchvision.datasets.utils import download_url

    parser = argparse.ArgumentParser(description="cv-RNN demo on 2-Shapes")
    parser.add_argument("--gpu", action="store_true", help="run on CUDA if available")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("../datasets"),
        help="where to download 2shapes.mat",
    )
    args = parser.parse_args()

    dev = torch.device("cuda:0" if args.gpu and torch.cuda.is_available() else "cpu")
    g = torch.Generator(device=dev).manual_seed(1)

    # ------------------------------------------------------------------ #
    # load dataset                                                       #
    # ------------------------------------------------------------------ #
    from scipy.io import loadmat

    url = "https://raw.githubusercontent.com/mullerlab/liboniEA2025image/main/dataset/2shapes.mat"
    mat_path = args.dataset_dir / "2shapes.mat"
    mat_path.parent.mkdir(parents=True, exist_ok=True)
    if not mat_path.exists():
        download_url(url, str(mat_path.parent))
    data = loadmat(mat_path)
    image = torch.from_numpy(data["images"][:, :, 0]).float()  # pick first example

    # ------------------------------------------------------------------ #
    # run model                                                          #
    # ------------------------------------------------------------------ #
    save_x, mask = run_2layer_torch(
        image,
        generator=g,
        device=dev,
    )
    cluster_map, rho, V, D, prj = spatiotemporal_segmentation_torch(
        save_x,
        image,
        mask,
        n_clusters=2,
        dim=(0, 1, 2),
        window_size=40,
        window_step=40,
        nt_mask=60,
        device=dev,
    )

    # ------------------------------------------------------------------ #
    # quick visualisation                                                #
    # ------------------------------------------------------------------ #
    fig, ax = plt.subplots(1, 3, figsize=(9, 3))
    ax[0].imshow(image.cpu(), cmap="gray")
    ax[0].set_title("input")
    ax[1].imshow(mask.view(*image.shape).cpu(), cmap="gray")
    ax[1].set_title("mask")
    im_ = ax[2].imshow(cluster_map.cpu(), cmap="tab10", vmin=-1, vmax=1)
    ax[2].set_title("segments")
    for a in ax:
        a.axis("off")
    fig.colorbar(im_, ax=ax.ravel().tolist(), shrink=0.6)
    plt.tight_layout()
    plt.show()
