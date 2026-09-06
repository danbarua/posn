"""Two-layer image segmentation following ``matlab/liboniEA2025image``.

Dynamics are raw discrete matrix iterates, not normalized Euler steps. Pixel
vectors use MATLAB column-major order. Clustering alone replaces amplitudes by
unit phases and excludes the background. Double precision is needed for the
large amplitudes reached by the supplied examples.
"""

from collections.abc import Iterable
import math
from operator import index

import torch
from sklearn.cluster import KMeans


def _complex_dtype(dtype: torch.dtype) -> torch.dtype:
    if dtype == torch.float64:
        return torch.complex128
    if dtype == torch.float32:
        return torch.complex64
    raise ValueError("dtype must be torch.float32 or torch.float64")


def gaussian_sheet_torch(
    nrow: int,
    ncol: int,
    amp: float,
    sigma: float,
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float64,
    phi: float | None = None,
    phi_threshold: float = 0.04,
) -> torch.Tensor:
    """Return the dense Gaussian weights from MATLAB ``gaussian_sheet.m``.

    Positions match ``meshgrid(row, col)`` followed by MATLAB's ``(:)``. This
    preserves the upstream rectangular-grid convention as well as its square
    examples; it is not a change to MATLAB's connectivity geometry.

    Optional ``phi`` applies a constant phase to weights below ``phi_threshold``.
    This is the defined operation in the upstream optional branch, which then
    references undefined ``phi_range``/``phi_min``. No distance-dependent phase
    law is inferred from that unfinished branch; the published demos omit phi.
    """
    complex_dtype = _complex_dtype(dtype)
    if nrow < 1 or ncol < 1:
        raise ValueError("image dimensions must be positive")
    if not math.isfinite(amp) or amp < 0:
        raise ValueError("amp must be finite and nonnegative")
    if not math.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be finite and positive")
    if phi is not None and not math.isfinite(phi):
        raise ValueError("phi must be finite")
    if not math.isfinite(phi_threshold) or phi_threshold < 0:
        raise ValueError("phi_threshold must be finite and nonnegative")

    rows = torch.arange(1, nrow + 1, device=device, dtype=dtype) / nrow
    cols = torch.arange(1, ncol + 1, device=device, dtype=dtype) / ncol
    yy, xx = torch.meshgrid(rows, cols, indexing="ij")
    positions = torch.stack((yy.flatten(), xx.flatten()), dim=-1)
    weights = torch.cdist(
        positions, positions, compute_mode="donot_use_mm_for_euclid_dist"
    ).square_()
    weights.div_(-2.0 * sigma**2).exp_().mul_(amp)
    weights = weights.to(complex_dtype)
    if phi is not None:
        weak = weights.real < phi_threshold
        weights[weak] *= complex(math.cos(phi), math.sin(phi))
    return weights


def run_2layer_torch(
    im: torch.Tensor,
    alpha: tuple[float, float] = (0.5, 0.5),
    sigma: tuple[float, float] = (0.9, 0.0313),
    nt: tuple[int, int] = (60, 200),
    *,
    generator: torch.Generator | None = None,
    initial_state: torch.Tensor | None = None,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run ``x_next = (K + i*diag(im(:))) @ x`` in two layers.

    ``im`` is the real frequency image used directly by MATLAB; do not rescale
    the bundled images by pi. ``nt`` gives the layer-1 sample count and total
    sample count. The initial state occupies index zero. Layer 2 restarts from
    the same initial state, masks its background, and stores its first update
    at index ``nt[0]`` (MATLAB iteration ``nt[0] + 1``).

    Returns raw complex states ``(H*W, nt[1])`` and a Boolean background vector
    ``(H*W,)``, both in MATLAB column-major pixel order. Background samples are
    NaN only in layer 2. Magnitudes are deliberately not normalized.

    A supplied ``initial_state`` must be a finite complex vector ``(H*W,)``;
    it takes precedence over the generator. Use this for cross-language
    comparisons: equal MATLAB and Torch seeds do not imply equal random states.
    """
    complex_dtype = _complex_dtype(dtype)
    if im.ndim != 2 or im.numel() == 0 or im.is_complex():
        raise ValueError("im must be a nonempty real two-dimensional image")
    if len(alpha) != 2 or len(sigma) != 2 or len(nt) != 2:
        raise ValueError("alpha, sigma and nt must each contain two entries")
    t1, t_end = (index(value) for value in nt)
    if not 1 <= t1 < t_end:
        raise ValueError("nt must satisfy 1 <= layer_1_samples < total_samples")
    im = im.to(device=device, dtype=dtype)
    if not torch.isfinite(im).all():
        raise ValueError("im must contain finite frequencies")
    nrow, ncol = im.shape
    n = im.numel()
    omega = im.T.reshape(-1)

    if initial_state is None:
        phases = torch.rand(n, device=device, dtype=dtype, generator=generator)
        x0 = torch.exp(1j * (phases - 0.5) * (2 * math.pi))
    else:
        if initial_state.shape != (n,) or not initial_state.is_complex():
            raise ValueError("initial_state must be a complex vector with H*W entries")
        x0 = initial_state.to(device=device, dtype=complex_dtype)
        if not torch.isfinite(x0).all():
            raise ValueError("initial_state must contain finite values")

    save_x = torch.empty((n, t_end), device=device, dtype=complex_dtype)
    save_x[:, 0] = x0
    weights = gaussian_sheet_torch(
        nrow, ncol, alpha[0], sigma[0], device=device, dtype=dtype
    )
    x = x0
    for time in range(1, t1):
        x = weights @ x + 1j * omega * x
        save_x[:, time] = x
    del weights
    if not torch.isfinite(x).all():
        raise FloatingPointError(
            "Layer-1 raw dynamics overflowed; use float64 or reduce iterations/coupling"
        )

    phases = torch.angle(x)
    threshold = phases.mean()
    above, below = phases > threshold, phases < threshold
    mask = above if above.sum() > below.sum() else below

    weights = gaussian_sheet_torch(
        nrow, ncol, alpha[1], sigma[1], device=device, dtype=dtype
    )
    weights[mask, :] = 0
    weights[:, mask] = 0
    omega2 = omega.masked_fill(mask, 0)
    x = x0.masked_fill(mask, 0)
    for time in range(t1, t_end):
        x = weights @ x + 1j * omega2 * x
        save_x[:, time] = x
    if not torch.isfinite(x).all():
        raise FloatingPointError(
            "Layer-2 raw dynamics overflowed; use float64 or reduce iterations/coupling"
        )
    save_x[mask, t1:] = torch.nan
    return save_x, mask


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
    device: str | torch.device = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Cluster foreground phase dynamics using MATLAB's inclusive windows.

    ``nt_mask`` is the layer-1 sample count, not its zero-based final index.
    Thus defaults analyze Python slices ``59:100``, ``99:140``, ``139:180``:
    MATLAB's window size 40 is an index span containing 41 samples, including
    the last layer-1 sample in the first window.

    Returns ``(cluster_map, rho, V, D, projection)``. The image-shaped label map
    uses -1 for background and zero-based foreground labels. Other tensors
    contain ONLY foreground nodes in original column-major order: rho/V/D
    have shape ``(F,F,windows)`` and projection ``(F,len(dim),windows)``.

    Eigenpairs are sorted by decreasing eigenvalue magnitude; D is the sorted
    diagonal matrix, correcting upstream ``D(:,ind)=d`` bookkeeping. Projection
    follows ``real(rho) @ real(V[:,dim])``. Eigenvector complex phases and KMeans
    initialization are solver-specific, so raw projections are not portable
    bitwise parity targets.
    """
    if save_x.ndim != 2 or not save_x.is_complex():
        raise ValueError("save_x must be a complex (pixels, samples) tensor")
    n, t_end = save_x.shape
    if im.ndim != 2 or im.numel() != n:
        raise ValueError("im shape must match the number of pixels in save_x")
    if mask.shape != (n,) or mask.dtype != torch.bool:
        raise ValueError("mask must be a Boolean vector with one entry per pixel")
    nt_mask, window_size, window_step = map(index, (nt_mask, window_size, window_step))
    if not 1 <= nt_mask <= t_end or window_size < 0 or window_step < 1:
        raise ValueError("invalid layer boundary or window parameters")
    windows = range(nt_mask - 1, t_end - window_size, window_step)
    if not windows:
        raise ValueError("no complete inclusive analysis window fits the trajectory")

    save_x = save_x.to(device)
    mask = mask.to(device)
    foreground = save_x[~mask]
    n_fg = foreground.shape[0]
    dimensions = tuple(index(value) for value in dim)
    n_clusters = index(n_clusters)
    if not dimensions or any(value < 0 or value >= n_fg for value in dimensions):
        raise ValueError("projection dimensions must index existing foreground eigenvectors")
    if not 1 <= n_clusters <= n_fg:
        raise ValueError("n_clusters must be between one and the foreground pixel count")
    if not torch.isfinite(foreground).all():
        raise ValueError("unmasked foreground dynamics contain nonfinite values")
    phases = torch.exp(1j * torch.angle(foreground))
    n_windows = len(windows)
    rho = torch.empty((n_fg, n_fg, n_windows), dtype=save_x.dtype, device=device)
    eigenvectors = torch.empty_like(rho)
    eigenvalues = torch.zeros_like(rho)
    projection = torch.empty(
        (n_fg, len(dimensions), n_windows), dtype=save_x.real.dtype, device=device
    )
    for window, start in enumerate(windows):
        samples = phases[:, start : start + window_size + 1]
        similarity = samples @ samples.mH / samples.shape[1]
        values, vectors = torch.linalg.eigh(similarity)
        order = values.abs().argsort(descending=True)
        values, vectors = values[order], vectors[:, order]
        rho[:, :, window] = similarity
        eigenvectors[:, :, window] = vectors
        eigenvalues[:, :, window].diagonal().copy_(values)
        projection[:, :, window] = similarity.real @ vectors.real[:, dimensions]

    data = projection[:, :, -1].detach().cpu().numpy()
    labels = KMeans(n_clusters=n_clusters, random_state=0, n_init=1).fit_predict(data)
    cluster_idx = torch.full((n,), -1, dtype=torch.long, device=device)
    cluster_idx[~mask] = torch.as_tensor(labels, dtype=torch.long, device=device)
    return (
        cluster_idx.reshape(im.shape[1], im.shape[0]).T,
        rho,
        eigenvectors,
        eigenvalues,
        projection,
    )
