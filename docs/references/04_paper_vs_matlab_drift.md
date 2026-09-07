# Preprint vs. published paper vs. GitHub MATLAB: drift and its measured effect

This documents concrete places where the text of
[the preprint](00_preprint.md) (arXiv:2311.16943v1), [the published paper](01_paper.md)
/ [supplement](02_supplementary.md) (PNAS 2025), and the checked-in
[`matlab/liboniEA2025image`](../../matlab/liboniEA2025image) submodule
disagree, plus how much each disagreement actually changes segmentation
outcomes. All numbers below come from
[`scripts/probe_paper_vs_matlab_drift.py`](../../scripts/probe_paper_vs_matlab_drift.py);
re-run it after changing numerical conventions or the MATLAB submodule pin.
The tables below were regenerated with uv and upper-triangle eigensolves.
The separate cross-runtime result in section 4 comes from the Octave fixtures
consumed by `tests/test_image_segmentation.py`.

`src/cv_rnn` follows the upstream recurrence and real-projection formula.
Matching that formula alone does not guarantee matching eigensolver phases.

## 1. Amplitude convention: Uniform(0,1) (paper) vs. unit (MATLAB code)

Both the [preprint](00_preprint.md#network-architecture) and the
[published paper](01_paper.md) state, identically: *"node amplitudes
$|x_i(0)|$ distributed uniformly in the interval $[0,1]$."* `run_2layer.m`
instead draws `x0 = exp(1i*(rand(Nr*Nc,1)-0.5)*2*pi)` — unit amplitude, no
separate amplitude draw at all. This is not a preprint-vs-published
difference; both paper versions agree with each other and disagree with the
code.

**Measured effect** (same phase draw, only the amplitude convention differs;
20 seeds per image, paired difference = uniform-amplitude ARI − unit-amplitude
ARI):

| Image | unit-amplitude ARI (mean, min) | Uniform(0,1) ARI (mean, min) | paired diff (mean ± std) |
| --- | --- | --- | --- |
| 2shapes/0 | 0.589, −0.005 | 0.621, −0.009 | +0.031 ± 0.516 |
| 2shapes/1 | 0.622, −0.022 | 0.714, −0.029 | +0.092 ± 0.366 |
| 3shapes/0 | 0.507, 0.014 | 0.522, 0.144 | +0.016 ± 0.360 |
| 3shapes/1 | 0.636, 0.237 | 0.596, 0.055 | −0.040 ± 0.169 |

The paired mean differences are small relative to their standard deviations
and not consistently signed. This sweep does not establish statistical
significance. Background masks agree in all 80 paired trials; that is an
observed result, not a general claim that initial amplitudes cannot affect
the phase dynamics.

## 2. Layer-2 initial condition: fresh draw (published paper) vs. reuse (MATLAB code)

The **published** paper's Materials and Methods states: *"The dynamics in the
second layer start from a new random initial state $\mathbf{x}_2(0)$."*
`run_2layer.m` instead does `x02 = x0; x02(mask) = 0;` — it reuses the
layer-1 phases, restricted to the foreground mask; it never draws new
randomness for layer 2.

This sentence has **no preprint counterpart to compare against**: the v1
LaTeX source (`main.tex`) contains no Methods section in any form (see
[`00_preprint.md`](00_preprint.md)'s provenance note, verified against the
arXiv e-print tarball, not just a lossy PDF conversion). Whether this text was
written for the PNAS submission or existed earlier cannot be determined from
what's public.

**Measured effect** (paired difference = fresh-draw ARI − reuse ARI; masks
are identical between the two by construction, since the mask is computed
before either layer-2 initial condition is applied):

| Image | reuse ARI (mean, min) | fresh-draw ARI (mean, min) | paired diff (mean ± std) |
| --- | --- | --- | --- |
| 2shapes/0 | 0.606, 0.044 | 0.750, 0.033 | +0.143 ± 0.634 |
| 2shapes/1 | 0.622, −0.029 | 0.674, −0.019 | +0.052 ± 0.667 |
| 3shapes/0 | 0.646, 0.066 | 0.618, 0.038 | −0.028 ± 0.388 |
| 3shapes/1 | 0.495, −0.035 | 0.621, 0.243 | +0.126 ± 0.355 |

The paired differences are not consistently signed and their standard
deviations exceed their means. A larger sweep would be needed to establish
a systematic effect of reusing versus redrawing the initial state.

## 3. Time windows: SI Appendix table vs. the actual driver script

[`02_supplementary.md`](02_supplementary.md) section X reports, for
Figs. 3b/3c/4a-c: $T=$ 141–181 (MATLAB 1-based, inclusive). The actual driver,
`cvrnn_image_segmentation.m` (`window_size=40; window_step=40;`,
`layer_time_points=[60,200]`), feeding `spatiotemporal_segmentation.m`'s
`window_start = win(1):dw:(win(2)-ws); window_end = window_start + ws;`,
computes exactly:

```
[60, 100], [100, 140], [140, 180]   (41 samples each, 1-based inclusive)
```

The SI's reported **141–181 is a one-sample shift** from the driver's actual
third window, **140–180**, not the same range. For Figs. 5/6 the SI table
instead reports $T=$ 121–141 — a **21-sample** window the driver's hardcoded
`window_size=40` can never produce at all. Figures 5 (overlapping objects) and
6 (eigenvector reconstruction) have **no driver code in this repository**;
`cvrnn_image_segmentation.m` only reproduces the three examples corresponding
to Figs. 3–4 (confirmed: `check_matlab_source_claims()` asserts the driver's
last section is "EXAMPLE 3: natural image", nothing further).

**Measured effect** (final-window ARI only; window fixed to a single range
via `window_step` large enough to force exactly one):

| Image, seed | production [140,180] | SI-literal [141,181] | short [121,141] |
| --- | --- | --- | --- |
| 2shapes/0, seed 1 | 1.000 | 1.000 | 1.000 |
| 2shapes/0, seed 2 | 1.000 | 1.000 | 1.000 |
| 2shapes/0, seed 3 | 1.000 | 1.000 | 0.212 |
| 3shapes/0, seed 1 | 0.490 | 0.490 | 0.338 |
| 3shapes/0, seed 2 | 1.000 | 1.000 | 0.295 |
| 3shapes/0, seed 3 | 0.487 | 0.507 | 0.449 |

The one-sample shift produces identical or nearly identical ARI here. The
short window degrades four of six rows and leaves two unchanged; it is not
interchangeable with the longer window. These are bundled images, not a
reconstruction of the papers' Fig. 5/6 experiments.


## 4. Seed sensitivity (the actual headline finding)

The specific seeds hardcoded in `cvrnn_image_segmentation.m` — 1 for 2shapes,
9 for 3shapes — give foreground ARI = 1.000 (confirmed independently by
`tests/test_segmentation_objects.py` for every bundled image, using these
exact demo defaults). **Generic random seeds do not reproduce this.** Across
20 random seeds per image:

| Image | demo seed → ARI | generic random seeds ARI (mean, min) | raw-gauge mean, min |
| --- | --- | --- | --- |
| 2shapes/0 | seed 1 → 1.000 | 0.606, −0.038 | 0.598, −0.038 |
| 3shapes/0 | seed 9 → 1.000 | 0.639, −0.001 | 0.616, −0.001 |

The seed-sensitivity control bypasses production canonicalization (a
raw-gauge sweep). Patching ``eigh`` with the same pivot rotation is a no-op
because production already applies it. Bypassing the gauge does not close
the demo-vs-generic gap.

**Direct Octave verification found a phase-convention mismatch, then a
backend-independent projection fix.** With shared images and initial states,
the default lower-triangle Torch `eigh` matches all raw trajectories, masks,
and final correlations, but `3shapes` foreground partition ARI against
Octave's `kmeans` labels was 0.9721244344 before canonicalization. Leading
eigenvalues are well-separated (λ = [263.32, 0.674, 0.00523, …]). Both
sides now canonicalize each eigenvector (largest-magnitude entry rotated to
real-positive) before `real(rho) @ real(V)`. Projections then agree to
~4e-12. sklearn `KMeans` on both projections yields ARI 1.0. Octave's own
`kmeans` labels remain a non-target: on canonicalized `3shapes` they
disagree with sklearn on the same array (ARI ~0.40). Under Octave's
exported `x0`, neither clustering hits ground truth (`3shapes.mat`
`labels[:,:,0]`): Python/sklearn foreground ARI 0.496, Octave `kmeans`
0.077. The demo seed 9 still scores ARI 1.0 — that is a different initial
state, not this fixture. Canonicalization does not resolve genuinely
degenerate eigenspaces; the bundled/reference cases have none.

This does not contradict the paper's headline claim ("93%/86% of pixels
correctly clustered" over 1,000 images, `01_paper.md`): that is **pixel**
accuracy, and background pixels dominate the count (background is ~81% of
pixels in the 2shapes/0 example) — a classifier that gets the mask right and
the foreground clustering wrong still scores well by that metric. It is a
different quantity from foreground ARI on one image/seed, and a modest mean
foreground ARI under generic seeds is not inconsistent with it. But it does
mean: **treat Figs. 3/4's specific example images as illustrations of what
the architecture can do under a favorable seed, not evidence that any random
seed reliably segments any image this cleanly.** `tests/test_segmentation_objects.py`
and this README's Image Segmentation section are worded to make this
explicit rather than imply "ARI=1 always."

## 5. Plotting: `plot_dynamics.m` does not reproduce Fig. 3A's rescaled colormap

`plot_dynamics.m` issues one `colormap hsv` call for every frame and only
fixes `CLim` to `[-pi,pi]` *after* `layer_1_final_time`
(`if(ii > layer_1_final_time), clim([-pi,pi]); end`). For layer-1 frames,
`CLim` is left at whatever `imagesc` auto-scaled at frame 1 (a fully random
initial condition spanning the whole circle). Layer 1 with the paper's own
parameters ($\alpha=0.5$, $\sigma=0.9$) phase-locks within a couple of steps
to a background/foreground split of a few milliradians (see the CV-RNN
`animate_dynamics` fix and its regression test,
`test_layer1_milliradian_offset_is_not_hsv_red`) — running `plot_dynamics.m`
literally as checked in would show a solid, uniform color for the converged
layer-1 frames, not the "phase (scaled)" gradient in the published Fig. 3A.
**The code that produced that specific rescaled-colormap panel is not in this
repository.** `src/cv_rnn/cv_nn_plot_phase_dynamics.py` fills this gap with
per-frame-scaled viridis for layer 1 and HSV `[-π,π]` for layer 2.

## 6. An unreferenced figure in the preprint's own source bundle

The v1 arXiv LaTeX tarball (`https://arxiv.org/e-print/2311.16943v1`)
contains `figures/FM1.png`, which is **not referenced by any
`\includegraphics` in `main.tex`** — every other figure file (F1–F6, FS1,
FS2) is used exactly once. `FM1.png`'s content (three panels: log-scale
connectivity-strength matrices for local, broad, and masked connectivity)
matches the published paper's **Figure 7** almost exactly. The authors had
this figure prepared before the v1 submission but did not include it in the
v1 preprint text; it surfaced only in the published version. Kept locally as
[`00_figures/arxiv.2311.16943v1.unreferenced-fig7-draft.png`](00_figures/arxiv.2311.16943v1.unreferenced-fig7-draft.png)
for reference.

## 7. `gaussian_sheet.m`'s node ordering silently assumes a square image

`gaussian_sheet_torch` (`src/cv_rnn/cv_rnn_segmentation.py`) is a faithful,
node-for-node port of `matlab/liboniEA2025image/graphs/gaussian_sheet.m`: its
`meshgrid(rows, cols, indexing="ij")` + row-major flatten reproduces MATLAB's
`[ROW,COL] = meshgrid(row,col); pos = [ROW(:) COL(:)]` exactly, including
MATLAB's `meshgrid(x,y)` axis-order swap (output size `(length(y),
length(x))`, not `(length(x), length(y))`). This is confirmed by
`tests/test_segmentation_math.py::test_rectangular_gaussian_matches_matlab_positions`.

Separately, `run_2layer_torch` builds the frequency vector as
`omega = im.T.reshape(-1)` — MATLAB `im(:)` column-major indexing (pixel
`(row, col)` at index `k = col*nrow + row`).

**These two node-index conventions only coincide when `nrow == ncol`.**
`gaussian_sheet.m`'s `meshgrid(row,col)` axis swap means node `k`'s
*position* is drawn from `(row[k // ncol], col[k % ncol])` — indexed by
`ncol`, not `nrow`. For a square image this is invisible: coordinate-swapping
every node's `(x, y)` position uniformly preserves all pairwise Euclidean
distances (a diagonal reflection), so the resulting weight matrix is
identical regardless. For a **non-square** image it is not a reflection —
`ncol != nrow` means indexing by the wrong dimension picks genuinely
different points, and the effect inverts adjacency rather than just
attenuating it. `scripts/probe_paper_vs_matlab_drift.py`'s
`gaussian_sheet_indexing_probe`, section 6:

```
3x5: adjacent pair (k=0,k=3) weight=0.1353 | non-adjacent pair (k=0,k=5) weight=0.5394 | worst pair (k=5,k=6) expected=0.0678 actual=0.8007 | max|diff|=0.7329 mean|diff|=0.2065
4x6: adjacent pair (k=0,k=4) weight=0.0847 | non-adjacent pair (k=0,k=6) weight=0.7066 | worst pair (k=7,k=8) expected=0.0377 actual=0.8570 | max|diff|=0.8193 mean|diff|=0.2207
```

For a 4-row × 6-col grid, `k=0` and `k=4` are physically **adjacent** under
`im(:)` (same row, next column) yet `gaussian_sheet_torch` assigns them
coupling **0.08** — as if far apart. `k=0` and `k=6` are **not** adjacent
(two rows down, one column over) yet get coupling **0.71** — as if close.
The full weight matrix vs. the physically-correct distance-based matrix
disagrees by up to 0.82 (mean 0.22) out of an amplitude-1 kernel.

**This is an upstream MATLAB quirk** (`gaussian_sheet.m`'s own `pos` array
has the same property relative to `im(:)`), not something introduced by the
Python port — `cv_rnn_segmentation.py`'s docstring already notes the port
"preserves the upstream rectangular-grid convention as well as its square
examples." **It does not affect this repository today**: all three bundled
images are square (`2shapes`/`3shapes`: 32×32, `natural`: 64×64). It would
matter if a non-square image were ever added — the connectivity would not
represent genuine spatial locality, corrupting the Gaussian sheet's
"nearby pixels are strongly coupled" property. Do **not** silently fix the
`gaussian_sheet_torch`/`omega` indexing mismatch to be square-agnostic; that
would diverge from MATLAB parity for a code path nothing currently exercises.
If a non-square dataset is ever bundled, this needs a real decision (match
MATLAB's quirk exactly, or diverge and document it), not a silent choice.
