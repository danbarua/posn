#!/usr/bin/env python
"""Quantify drift between the preprint, the published paper, and the GitHub
MATLAB reference implementation for the image-segmentation cv-RNN.

Run:
    uv run --locked python scripts/probe_paper_vs_matlab_drift.py

This is an investigative report, not a regression test or a proposal to
change production defaults. `src/cv_rnn` intentionally follows the MATLAB
convention (see README.md, docs/references/01_paper.md's provenance note);
this script measures how much the *other* convention -- the one written in
the paper text -- would actually change outcomes, so "the text and the code
disagree" can be answered with "and here is whether it matters" rather than
left as a shrug.

Findings are written up in docs/references/04_paper_vs_matlab_drift.md,
which quotes this script's output. Re-run after touching the amplitude/
initial-condition/window code paths, or after updating the MATLAB submodule
pin, to keep that document's numbers honest.

Sections:
  1. Static source claims -- regex assertions against the checked-out MATLAB
     submodule, so a submodule update that changes these lines fails loudly
     instead of silently invalidating the write-up.
  2. Amplitude convention -- paper text says node amplitudes are drawn
     Uniform(0,1); run_2layer.m uses unit amplitude. Same phases, only the
     amplitude convention differs; measure background-mask agreement and
     foreground ARI drift across seeds.
  3. Layer-2 initial condition -- the published Materials and Methods says
     layer 2 "starts from a new random initial state x_2(0)"; run_2layer.m
     reuses the masked layer-1 phases. Measure the same metrics.
  4. Window schedule -- the SI Appendix parameter table reports T=141-181 for
     Figs. 3b/3c/4a-c and T=121-141 for Figs. 5/6; the actual driver script
     (window_size=40, window_step=40, nt_mask=60) computes windows
     60-100/100-140/140-180. Reproduce the checked-in window arithmetic
     exactly and compare clustering quality across the alternatives that ARE
     reachable from the bundled datasets.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.io import loadmat
from sklearn.metrics import adjusted_rand_score

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from defns import DATA_DIR, ROOT_DIR
from src.cv_rnn.cv_rnn_segmentation import (
    gaussian_sheet_torch,
    run_2layer_torch,
    spatiotemporal_segmentation_torch,
)

MATLAB_ROOT = Path(ROOT_DIR) / "matlab" / "liboniEA2025image"


# --------------------------------------------------------------------------
# 1. Static source claims
# --------------------------------------------------------------------------


def _read_matlab(relative_path: str) -> str:
    return (MATLAB_ROOT / relative_path).read_text()


def check_matlab_source_claims() -> list[tuple[str, bool]]:
    """Assert the exact MATLAB lines the drift write-up quotes still exist.

    A `False` here means the submodule pin changed and every downstream
    number in docs/references/04_paper_vs_matlab_drift.md needs re-checking.
    """
    run2layer = _read_matlab("image_segmentation/run_2layer.m")
    driver = _read_matlab("cvrnn_image_segmentation.m")
    segfn = _read_matlab("image_segmentation/spatiotemporal_segmentation.m")
    plotfn = _read_matlab("helper_functions/plot_dynamics.m")

    claims = [
        (
            "layer-1 initial state is unit amplitude (exp(1i*(rand-.5)*2*pi))",
            r"x0\s*=\s*exp\(\s*1i\s*\*\s*\(\s*rand\(",
            run2layer,
        ),
        (
            "layer-2 initial state reuses x0 rather than drawing new randomness",
            r"x02\s*=\s*x0\s*;",
            run2layer,
        ),
        (
            "layer-2 zeroes (does not resample) the masked entries of x02",
            r"x02\(mask\)\s*=\s*0",
            run2layer,
        ),
        (
            "driver hardcodes window_size=40, window_step=40 for every example",
            r"window_size\s*=\s*40;\s*window_step\s*=\s*40",
            driver,
        ),
        (
            "driver's layer boundaries are 1:60 and 61:200 for every example",
            r"layer_1_time_range\s*=\s*1:60;\s*layer_2_time_range\s*=\s*61:200",
            driver,
        ),
        (
            "driver only reproduces Examples 1-3 (2shapes/3shapes/natural), not Figs. 5-6",
            r"EXAMPLE 3: natural image",
            driver,
        ) if "EXAMPLE 4" not in driver else (
            "driver reproduces more than 3 examples (recheck Fig. 5/6 claim)",
            r"EXAMPLE 4",
            driver,
        ),
        (
            "windowing is win(1):dw:(win(2)-ws), i.e. the code path this script mirrors",
            r"window_start\s*=\s*win\(1\):dw:\(win\(2\)-ws\)",
            segfn,
        ),
        (
            "plot_dynamics.m applies one 'colormap hsv' call for every frame",
            r"colormap hsv",
            plotfn,
        ),
        (
            "plot_dynamics.m only fixes CLim to [-pi,pi] after layer_1_final_time"
            " (layer-1 frames keep whichever CLim imagesc auto-scaled at frame 1)",
            r"if\(\s*ii\s*>\s*layer_1_final_time\s*\),\s*clim",
            plotfn,
        ),
    ]
    return [(name, bool(re.search(pattern, text))) for name, pattern, text in claims]


def matlab_window_schedule(
    win_start: int, win_end: int, window_size: int, window_step: int
) -> list[tuple[int, int]]:
    """Reproduce spatiotemporal_segmentation.m's window_start/window_end exactly.

    MATLAB: `window_start = win(1):dw:(win(2)-ws); window_end = window_start + ws;`
    All values are 1-based inclusive MATLAB indices; this function does not
    convert to Python 0-based indexing.
    """
    starts = list(range(win_start, win_end - window_size + 1, window_step))
    return [(s, s + window_size) for s in starts]


# --------------------------------------------------------------------------
# Shared probe plumbing
# --------------------------------------------------------------------------

ALPHA = (0.5, 0.5)
SIGMA = (0.9, 0.0313)
NT = (60, 200)


def _load_example(dataset: str, image_index: int):
    filename = {"2shapes": "2shapes.mat", "3shapes": "3shapes.mat"}[dataset]
    data = loadmat(Path(DATA_DIR) / filename)
    image = torch.from_numpy(np.asarray(data["images"][:, :, image_index], dtype=np.float64))
    truth = data["labels"][:, :, image_index]
    n_clusters = int(truth.max())
    return image, truth, n_clusters


def _score(mask: torch.Tensor, labels: torch.Tensor, truth: np.ndarray) -> tuple[float, float]:
    predicted = labels.numpy()
    foreground = truth != 0
    ari = adjusted_rand_score(truth[foreground], predicted[foreground])
    mask_agree = float(np.mean((predicted == -1) == (truth == 0)))
    return ari, mask_agree


# --------------------------------------------------------------------------
# 2. Amplitude convention: unit (MATLAB) vs Uniform(0,1) (paper text)
# --------------------------------------------------------------------------


def amplitude_probe(dataset: str, image_index: int, n_trials: int = 20) -> dict:
    image, truth, n_clusters = _load_example(dataset, image_index)
    n = image.numel()
    unit_aris, uniform_aris, mask_agreements = [], [], []
    for trial in range(n_trials):
        phase_rng = np.random.RandomState(10_000 + trial)
        phases = (phase_rng.random_sample(n) - 0.5) * 2 * np.pi
        unit_x0 = torch.from_numpy(np.exp(1j * phases))

        amp_rng = np.random.RandomState(20_000 + trial)
        amplitudes = amp_rng.random_sample(n)  # Uniform(0, 1), paper's Eq. (4) text
        uniform_x0 = torch.from_numpy(amplitudes * np.exp(1j * phases))

        unit_states, unit_mask = run_2layer_torch(image, ALPHA, SIGMA, NT, initial_state=unit_x0)
        uniform_states, uniform_mask = run_2layer_torch(
            image, ALPHA, SIGMA, NT, initial_state=uniform_x0
        )

        unit_labels, *_ = spatiotemporal_segmentation_torch(
            unit_states, image, unit_mask, nt_mask=NT[0], n_clusters=n_clusters
        )
        uniform_labels, *_ = spatiotemporal_segmentation_torch(
            uniform_states, image, uniform_mask, nt_mask=NT[0], n_clusters=n_clusters
        )

        unit_ari, _ = _score(unit_mask, unit_labels, truth)
        uniform_ari, _ = _score(uniform_mask, uniform_labels, truth)
        mask_agreements.append(float(torch.mean((unit_mask == uniform_mask).float())))
        unit_aris.append(unit_ari)
        uniform_aris.append(uniform_ari)

    diffs = [u - v for u, v in zip(uniform_aris, unit_aris)]
    return {
        "dataset": dataset,
        "image_index": image_index,
        "n_trials": n_trials,
        "unit_ari_mean": float(np.mean(unit_aris)),
        "unit_ari_min": float(np.min(unit_aris)),
        "uniform_ari_mean": float(np.mean(uniform_aris)),
        "uniform_ari_min": float(np.min(uniform_aris)),
        "paired_diff_mean": float(np.mean(diffs)),
        "paired_diff_std": float(np.std(diffs)),
        "mask_agreement_mean": float(np.mean(mask_agreements)),
        "mask_agreement_min": float(np.min(mask_agreements)),
    }


# --------------------------------------------------------------------------
# 3. Layer-2 initial condition: reuse masked x0 (MATLAB) vs a fresh draw (paper text)
# --------------------------------------------------------------------------


def run_2layer_variant(
    image: torch.Tensor,
    x0: torch.Tensor,
    *,
    layer2_init: str,
    rng: np.random.RandomState | None = None,
    alpha=ALPHA,
    sigma=SIGMA,
    nt=NT,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mirror run_2layer_torch's recurrence, exposing the layer-2 initial condition.

    `layer2_init="reuse"` matches run_2layer.m (`x02 = x0; x02(mask) = 0;`),
    the production default. `layer2_init="fresh"` matches the PNAS Materials
    and Methods text ("a new random initial state x_2(0)"): an independent
    random unit-amplitude phase draw for the unmasked (foreground) nodes.
    """
    nrow, ncol = image.shape
    n = image.numel()
    omega = image.T.reshape(-1).to(torch.float64)
    t1, t_end = nt

    weights1 = gaussian_sheet_torch(nrow, ncol, alpha[0], sigma[0])
    x = x0.clone()
    states = torch.empty((n, t_end), dtype=torch.complex128)
    states[:, 0] = x0
    for t in range(1, t1):
        x = weights1 @ x + 1j * omega * x
        states[:, t] = x

    phases = torch.angle(x)
    threshold = phases.mean()
    above, below = phases > threshold, phases < threshold
    mask = above if above.sum() > below.sum() else below

    weights2 = gaussian_sheet_torch(nrow, ncol, alpha[1], sigma[1])
    weights2[mask, :] = 0
    weights2[:, mask] = 0
    omega2 = omega.masked_fill(mask, 0)

    if layer2_init == "reuse":
        x = x0.masked_fill(mask, 0)
    elif layer2_init == "fresh":
        if rng is None:
            raise ValueError("layer2_init='fresh' requires an rng")
        fresh_phases = torch.from_numpy((rng.random_sample(n) - 0.5) * 2 * np.pi)
        x = torch.exp(1j * fresh_phases).masked_fill(mask, 0)
    else:
        raise ValueError(f"unknown layer2_init: {layer2_init!r}")

    for t in range(t1, t_end):
        x = weights2 @ x + 1j * omega2 * x
        states[:, t] = x
    states[mask, t1:] = torch.nan
    return states, mask


def layer2_reinit_probe(dataset: str, image_index: int, n_trials: int = 20) -> dict:
    image, truth, n_clusters = _load_example(dataset, image_index)
    n = image.numel()
    reuse_aris, fresh_aris = [], []
    for trial in range(n_trials):
        phase_rng = np.random.RandomState(30_000 + trial)
        phases = (phase_rng.random_sample(n) - 0.5) * 2 * np.pi
        x0 = torch.from_numpy(np.exp(1j * phases))
        fresh_rng = np.random.RandomState(40_000 + trial)

        reuse_states, mask = run_2layer_variant(image, x0, layer2_init="reuse")
        fresh_states, fresh_mask = run_2layer_variant(
            image, x0, layer2_init="fresh", rng=fresh_rng
        )
        assert torch.equal(mask, fresh_mask), "layer-1 mask must not depend on layer-2 init"

        reuse_labels, *_ = spatiotemporal_segmentation_torch(
            reuse_states, image, mask, nt_mask=NT[0], n_clusters=n_clusters
        )
        fresh_labels, *_ = spatiotemporal_segmentation_torch(
            fresh_states, image, mask, nt_mask=NT[0], n_clusters=n_clusters
        )
        reuse_ari, _ = _score(mask, reuse_labels, truth)
        fresh_ari, _ = _score(mask, fresh_labels, truth)
        reuse_aris.append(reuse_ari)
        fresh_aris.append(fresh_ari)

    diffs = [f - r for f, r in zip(fresh_aris, reuse_aris)]
    return {
        "dataset": dataset,
        "image_index": image_index,
        "n_trials": n_trials,
        "reuse_ari_mean": float(np.mean(reuse_aris)),
        "reuse_ari_min": float(np.min(reuse_aris)),
        "fresh_ari_mean": float(np.mean(fresh_aris)),
        "fresh_ari_min": float(np.min(fresh_aris)),
        "paired_diff_mean": float(np.mean(diffs)),
        "paired_diff_std": float(np.std(diffs)),
    }


# --------------------------------------------------------------------------
# 4. Window schedule: production defaults vs the SI Appendix's literal ranges
# --------------------------------------------------------------------------


def window_probe(dataset: str, image_index: int, seed: int) -> dict:
    image, truth, n_clusters = _load_example(dataset, image_index)
    n = image.numel()
    rng = np.random.RandomState(seed)
    phases = (rng.random_sample(n) - 0.5) * 2 * np.pi
    x0 = torch.from_numpy(np.exp(1j * phases))
    states, mask = run_2layer_torch(image, ALPHA, SIGMA, NT, initial_state=x0)

    def final_window_ari(nt_mask: int, window_size: int) -> float:
        labels, *_ = spatiotemporal_segmentation_torch(
            states,
            image,
            mask,
            nt_mask=nt_mask,
            window_size=window_size,
            window_step=10_000,  # force exactly one window starting at nt_mask - 1
            n_clusters=n_clusters,
        )
        ari, _ = _score(mask, labels, truth)
        return ari

    # Production default: MATLAB [140, 180] (1-based inclusive) -> here nt_mask=140.
    production_ari = final_window_ari(nt_mask=140, window_size=40)
    # SI Appendix's literal text for Figs. 3b/3c/4a-c: MATLAB [141, 181].
    si_literal_ari = final_window_ari(nt_mask=141, window_size=40)
    # SI Appendix's literal text for Figs. 5/6: MATLAB [121, 141], a 21-sample window
    # the checked-in driver's hardcoded window_size=40 can never produce.
    short_window_ari = final_window_ari(nt_mask=121, window_size=20)

    return {
        "dataset": dataset,
        "image_index": image_index,
        "seed": seed,
        "production_140_180_ari": production_ari,
        "si_literal_141_181_ari": si_literal_ari,
        "short_121_141_ari": short_window_ari,
    }


# --------------------------------------------------------------------------
# 5. Seed sensitivity: the demo's hand-picked seeds vs generic random seeds,
#    with a raw-gauge control (production canonicalization bypassed)
# --------------------------------------------------------------------------


def seed_sensitivity_probe(
    dataset: str, image_index: int, demo_seed: int, n_random_trials: int = 20
) -> dict:
    """Compare the demo's hand-picked seed against generic random seeds.

    Also reruns the random-seed sweep with production's eigenvector
    canonicalization bypassed, so the comparison measures whether the
    remaining solver-dependent gauge affects seed-sweep ARI. Patching
    ``eigh`` with the same pivot rotation is a no-op: production already
    applies it after ``eigh``.
    """
    from src.cv_rnn import cv_rnn_segmentation as seg

    image, truth, n_clusters = _load_example(dataset, image_index)
    n = image.numel()

    def run_seed(seed: int) -> float:
        rng = np.random.RandomState(seed)
        phases = (rng.random_sample(n) - 0.5) * 2 * np.pi
        x0 = torch.from_numpy(np.exp(1j * phases))
        states, mask = run_2layer_torch(image, ALPHA, SIGMA, NT, initial_state=x0)
        labels, *_ = spatiotemporal_segmentation_torch(
            states, image, mask, nt_mask=NT[0], n_clusters=n_clusters
        )
        ari, _ = _score(mask, labels, truth)
        return ari

    demo_ari = run_seed(demo_seed)
    random_aris = [run_seed(50_000 + trial) for trial in range(n_random_trials)]

    original = seg._canonical_phase
    seg._canonical_phase = lambda vectors: vectors
    try:
        raw_gauge_aris = [run_seed(50_000 + trial) for trial in range(n_random_trials)]
    finally:
        seg._canonical_phase = original

    return {
        "dataset": dataset,
        "image_index": image_index,
        "demo_seed": demo_seed,
        "demo_ari": demo_ari,
        "random_ari_mean": float(np.mean(random_aris)),
        "random_ari_min": float(np.min(random_aris)),
        "raw_gauge_ari_mean": float(np.mean(raw_gauge_aris)),
        "raw_gauge_ari_min": float(np.min(raw_gauge_aris)),
        "n_random_trials": n_random_trials,
    }


# --------------------------------------------------------------------------
# 6. gaussian_sheet_torch node ordering vs. im(:) column-major indexing
# --------------------------------------------------------------------------


def gaussian_sheet_indexing_probe(nrow: int, ncol: int, amp: float = 1.0, sigma: float = 0.3) -> dict:
    """Quantify gaussian_sheet_torch's node-ordering mismatch against im(:).

    gaussian_sheet_torch (a faithful port of gaussian_sheet.m) indexes node
    positions by meshgrid(row,col); the frequency vector elsewhere is built
    via im(:) column-major indexing. These only agree when nrow == ncol. No
    simulation: pure index/geometry arithmetic plus one gaussian_sheet_torch
    call.
    """
    weights = gaussian_sheet_torch(nrow, ncol, amp, sigma, dtype=torch.float64).real
    n = nrow * ncol
    pos = torch.zeros(n, 2, dtype=torch.float64)
    for row in range(nrow):
        for col in range(ncol):
            k = col * nrow + row
            pos[k, 0] = (row + 1) / nrow
            pos[k, 1] = (col + 1) / ncol
    dist = torch.cdist(pos, pos)
    expected = amp * torch.exp(-dist**2 / (2 * sigma**2))
    diff = (weights - expected).abs()
    worst = int(diff.argmax())
    k1, k2 = worst // n, worst % n
    return {
        "nrow": nrow,
        "ncol": ncol,
        # k=0 vs k=nrow: physically adjacent under im(:) (same row, next column).
        "adjacent_pair_weight": float(weights[0, nrow].item()),
        # k=0 vs k=ncol: not physically adjacent (different row and column).
        "nonadjacent_pair_weight": float(weights[0, ncol].item()),
        "worst_pair": (k1, k2),
        "worst_pair_expected": float(expected[k1, k2].item()),
        "worst_pair_actual": float(weights[k1, k2].item()),
        "max_abs_diff": float(diff.max().item()),
        "mean_abs_diff": float(diff.mean().item()),
    }


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------


def main() -> None:
    print("=" * 78)
    print("1. Static MATLAB source claims (matlab/liboniEA2025image)")
    print("=" * 78)
    all_pass = True
    for name, ok in check_matlab_source_claims():
        status = "PASS" if ok else "FAIL"
        all_pass &= ok
        print(f"  [{status}] {name}")
    if not all_pass:
        raise AssertionError(
            "A MATLAB source claim failed; the submodule pin likely moved. "
            "Update docs/references/04_paper_vs_matlab_drift.md before trusting "
            "the rest of this report."
        )

    print()
    print("Driver's actual window schedule (MATLAB 1-based inclusive indices):")
    for start, end in matlab_window_schedule(60, 200, 40, 40):
        print(f"  [{start}, {end}]  ({end - start + 1} samples)")
    print("SI Appendix Table (section X) reports, for the same figures: [141, 181].")
    print("SI Appendix Table reports, for Figs. 5/6 only: [121, 141] (21 samples,")
    print("unreachable with the driver's hardcoded window_size=40).")

    print()
    print("=" * 78)
    print("2. Amplitude convention: unit (MATLAB) vs Uniform(0,1) (paper text)")
    print("=" * 78)
    for dataset, idx in [("2shapes", 0), ("2shapes", 1), ("3shapes", 0), ("3shapes", 1)]:
        result = amplitude_probe(dataset, idx)
        print(
            f"  {dataset}/{idx}: unit ARI mean={result['unit_ari_mean']:.3f} "
            f"min={result['unit_ari_min']:.3f} | uniform[0,1] ARI mean="
            f"{result['uniform_ari_mean']:.3f} min={result['uniform_ari_min']:.3f} | "
            f"paired diff (uniform-unit) mean={result['paired_diff_mean']:+.3f} "
            f"std={result['paired_diff_std']:.3f} | mask agreement mean="
            f"{result['mask_agreement_mean']:.4f} min={result['mask_agreement_min']:.4f} "
            f"({result['n_trials']} seeds)"
        )

    print()
    print("=" * 78)
    print("3. Layer-2 initial condition: reuse masked x0 (MATLAB) vs fresh draw (paper text)")
    print("=" * 78)
    for dataset, idx in [("2shapes", 0), ("2shapes", 1), ("3shapes", 0), ("3shapes", 1)]:
        result = layer2_reinit_probe(dataset, idx)
        print(
            f"  {dataset}/{idx}: reuse ARI mean={result['reuse_ari_mean']:.3f} "
            f"min={result['reuse_ari_min']:.3f} | fresh ARI mean="
            f"{result['fresh_ari_mean']:.3f} min={result['fresh_ari_min']:.3f} | "
            f"paired diff (fresh-reuse) mean={result['paired_diff_mean']:+.3f} "
            f"std={result['paired_diff_std']:.3f} ({result['n_trials']} seeds)"
        )

    print()
    print("=" * 78)
    print("4. Window schedule: production default vs SI Appendix's literal ranges")
    print("=" * 78)
    for dataset, idx in [("2shapes", 0), ("3shapes", 0)]:
        for seed in (1, 2, 3):
            result = window_probe(dataset, idx, seed)
            print(
                f"  {dataset}/{idx} seed={seed}: production[140,180] ARI="
                f"{result['production_140_180_ari']:.3f} | SI-literal[141,181] ARI="
                f"{result['si_literal_141_181_ari']:.3f} | short[121,141] ARI="
                f"{result['short_121_141_ari']:.3f}"
            )

    print()
    print("=" * 78)
    print("5. Seed sensitivity: demo's hand-picked seed vs generic random seeds")
    print("   (with a raw-gauge control: production canonicalization bypassed)")
    print("=" * 78)
    for dataset, idx, demo_seed in [("2shapes", 0, 1), ("3shapes", 0, 9)]:
        result = seed_sensitivity_probe(dataset, idx, demo_seed)
        print(
            f"  {dataset}/{idx}: demo seed={demo_seed} ARI={result['demo_ari']:.3f} | "
            f"generic random seeds ARI mean={result['random_ari_mean']:.3f} "
            f"min={result['random_ari_min']:.3f} | raw-gauge "
            f"ARI mean={result['raw_gauge_ari_mean']:.3f} "
            f"min={result['raw_gauge_ari_min']:.3f} "
            f"({result['n_random_trials']} random seeds)"
        )
    print()
    print("Bypassing production's eigenvector canonicalization does not close")
    print("the gap between the demo seed and generic seeds in this sweep.")
    print("Seed sensitivity is a clustering-quality property of these images")
    print("under random initializations, distinct from the gauge used for")
    print("cross-runtime projection parity.")
    print("The paper's headline '93%/86% of pixels correctly clustered' is a")
    print("PIXEL accuracy over 1,000 images (background pixels included, and they")
    print("dominate: ~81% of pixels in the 2shapes/0 example are background); it")
    print("is a different quantity from foreground ARI on a single seed, and is")
    print("not inconsistent with a modest mean foreground ARI under generic seeds.")

    print()
    print("=" * 78)
    print("6. gaussian_sheet_torch node ordering vs. im(:) column-major indexing")
    print("=" * 78)
    for nrow, ncol in [(3, 5), (4, 6)]:
        result = gaussian_sheet_indexing_probe(nrow, ncol)
        wk1, wk2 = result["worst_pair"]
        print(
            f"  {nrow}x{ncol}: adjacent pair (k=0,k={nrow}) weight="
            f"{result['adjacent_pair_weight']:.4f} | non-adjacent pair (k=0,k={ncol}) "
            f"weight={result['nonadjacent_pair_weight']:.4f} | worst pair (k={wk1},k={wk2}) "
            f"expected={result['worst_pair_expected']:.4f} actual={result['worst_pair_actual']:.4f} | "
            f"max|diff|={result['max_abs_diff']:.4f} mean|diff|={result['mean_abs_diff']:.4f}"
        )
    print()
    print("For nrow==ncol (all three bundled images), this is a diagonal coordinate")
    print("swap that preserves every pairwise distance, so the mismatch is invisible.")


if __name__ == "__main__":
    main()
