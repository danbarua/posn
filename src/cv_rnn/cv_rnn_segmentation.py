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
from typing import Iterable, Tuple, Union

import numpy as np
import torch
from sklearn.cluster import KMeans  # type: ignore

from defns import DATA_DIR
from src.cv_rnn.cv_nn_plot_phase_dynamics import plot_dynamics


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
def _step(
    matrix: torch.Tensor, omega: torch.Tensor, x: torch.Tensor, eps: float = 1e-12
) -> torch.Tensor:
    """
    One Euler step      x ← (K + i·diag(ω)) · x,
    implemented without forming the dense diagonal matrix, **and**
    projected back onto the unit circle so that |x| ≈ 1.

    Parameters
    ----------
    matrix : (..., N, N)      coupling matrix  K
    omega  : (..., N)         natural frequencies ω
    x      : (..., N, 1)      current complex state
    eps    : float            small value to avoid division by zero

    Returns
    -------
    torch.Tensor              updated, re-normalised complex state
    """
    # Euler RHS
    x_new = matrix @ x + 1j * omega.unsqueeze(-1) * x

    # Project back to |x| = 1
    magn = x_new.abs().clamp(min=eps)  # real-valued
    x_new = x_new / magn

    return x_new


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
    # x0 = torch.exp(1j * (rand * 2 * math.pi))  # phase ∈ [0,2π)
    x0 = torch.exp(1j * (rand - 0.5) * 2 * math.pi)  # phase ∈ [-π, π)
    x = x0.clone()

    print_complex_tensor_stats("x0", x0)

    omega = im.flatten()  # (N,)

    save_x = torch.empty((n, t_end), dtype=torch.complex64, device=device)
    save_x[:, 0] = x0.squeeze()

    # ------------------------------------------------------------------ #
    # layer 1                                                            #
    # ------------------------------------------------------------------ #
    k1 = gaussian_sheet_torch(
        nrow, ncol, alpha[0], sigma[0], device=device, dtype=dtype
    )

    print_complex_tensor_stats("k1", k1)

    for t in range(1, t1):
        x = _step(k1, omega, x)
        print_complex_tensor_stats(f"x @ t={t}", x)
        save_x[:, t] = x.squeeze()

    # mask = majority side of mean phase
    phase_t1 = torch.angle(x.squeeze())
    print(f"phase_t1: {phase_t1.shape}, {phase_t1.min()}, {phase_t1.max()}")

    thr = phase_t1.mean()
    print(f"Mean phase: {thr}")
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

    print_complex_tensor_stats("k2", k2)
    omega2 = omega.clone()
    omega2[mask] = 0

    # re-use x0 but zero masked nodes
    x = x0.clone()
    x[mask] = 0

    print_complex_tensor_stats("x @ t1", x)

    for t in range(t1, t_end):
        x = _step(k2, omega2, x)
        save_x[:, t] = x.squeeze()

    print(f"Any Nans in save_x (before masking): {torch.isnan(save_x).any().item()}")
    save_x[mask, t1:t_end] = torch.nan  # Note: Explicit NaN setting
    return save_x, mask


def print_complex_tensor_stats(name: str, input: torch.Tensor):
    print(
        f"{name} real: [{input.real.min()}, {input.real.max()}] imag: [{input.imag.min()}, {input.imag.max()}] mean: {input.mean()}, std: {input.std()}"
    )


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

    rho = torch.empty(
        (n, n, n_wins), dtype=torch.complex64, device=device
    )  # rho: similarity matrix (nodes x nodes x time window)
    V = torch.empty_like(rho)  #  V: eigenvectors of rho
    D = torch.empty_like(rho)  #  D: eigenvalues of rho
    prj = torch.empty(
        (n, len(tuple(dim)), n_wins), dtype=torch.float32, device=device
    )  # prj: rho*V

    for k, ws in enumerate(win_starts):
        we = ws + window_size
        rho_k = _similarity_tensor(save_x, ws, we)
        print("Any NaNs in rho_k:", torch.isnan(rho_k).any().item())
        print("Any infs in rho_k:", torch.isinf(rho_k).any().item())
        rho[:, :, k] = rho_k

        # eig(ρ) — real symmetric in practice after taking Re, but follow MATLAB
        # rho_k is (N, N) complex and may contain NaNs coming from the mask
        rho_k = torch.nan_to_num(rho_k, nan=0.0)
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
    from torchvision.datasets.utils import download_url  # type:ignore

    parser = argparse.ArgumentParser(description="cv-RNN demo on 2-Shapes")
    parser.add_argument("--gpu", action="store_true", help="run on CUDA if available")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path(DATA_DIR),
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

    print("Any NaNs in image:", torch.isnan(image).any().item())
    print("Any infs in image:", torch.isinf(image).any().item())

    # normalize image from [-1, 1] to [-pi, pi]
    image = image * math.pi
    print(f"Image shape: {image.shape}, Total: {image.numel()}")
    print(f"Image min: {image.min()}, max: {image.max()}")

    # ------------------------------------------------------------------ #
    # run model                                                          #
    # ------------------------------------------------------------------ #
    save_x, mask = run_2layer_torch(
        image,
        generator=g,
        device=dev,
    )

    print(f"Mask shape: {mask.shape}, Mask sum: {mask.sum()}, Total: {mask.numel()}")
    print("Any masked?", (mask.sum() > 0))
    if mask.sum() == mask.numel() or mask.sum() == 0:
        print(
            "Warning: All or zero nodes masked! Check phase distribution or threshold logic."
        )
    print(torch.isnan(save_x).sum())  # See when NaNs appear
    print("Any NaNs in save_x:", torch.isnan(save_x).any().item())
    print("Any infs in save_x:", torch.isinf(save_x).any().item())
    print("Any NaNs in mask:", torch.isnan(mask).any().item())
    print("Any infs in mask:", torch.isinf(mask).any().item())

    cluster_map, rho, V, D, prj = spatiotemporal_segmentation_torch(
        save_x,
        image,
        mask,
        n_clusters=2,
        dim=(0, 1, 2),
        window_size=40,
        window_step=40,
        nt_mask=61,
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

    # MATPLOTLIB ported visualisation
    plot_dynamics(save_x.detach().cpu().numpy(), image, 60)
