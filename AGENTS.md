# Repository Guidelines

## Project Overview

POSN implements fixed-weight complex-valued oscillator networks in PyTorch:
image segmentation (Liboni et al., PNAS 2025), XOR, memory, and experimental
message transmission (Budzinski et al., Communications Physics 2024).
There is no training loop. MATLAB submodules are the numerical references;
`docs/references/` records the papers and known paper/code discrepancies.

## Architecture & Data Flow

Two numerical families live in `src/cv_rnn/`; do not mix their conventions:

- `cv_rnn_segmentation.py`: dense Gaussian coupling, raw discrete recurrence
  `x_next = (K + i*diag(image(:))) @ x`, strict-majority phase background
  mask, then restart from the original initial state with background disabled.
  No Euler timestep or magnitude normalization. Windowed phase correlations,
  eigenvectors, real projection, and CPU KMeans produce object labels.
- Segmentation uses MATLAB column-major pixel order, float64/complex128,
  `-1` background labels, and NaNs for masked layer-2 history. The Gaussian
  geometry deliberately preserves upstream's rectangular-grid convention;
  see the drift document before changing it.
- `segmentation_demo.py` bundles numerical results in `SegmentationExample`;
  loading labels/scoring/plotting happen after inference. Shapes have an
  explicit zero-valued background label; natural-image `lb` is a region map
  scored with whole-image ARI, not foreground/background metrics.
- `cv_nn.py` defines `RingCVNN`, a plain Python class with cached FFT
  eigen-rates and exact Fourier propagation. `XorCVNN`/`MemoryCVNN` subclass
  it; `MessageDemo` composes it. `design_input` propagates backwards to
  construct inputs reaching a specified target at a specified time.
- `src/cv_rnn/__main__.py` dispatches `xor`, `memory`, `message`, and
  `segmentation`. Plot functions return figures; callers choose display.

## Key Directories

- `src/cv_rnn/`: simulation, evaluation, plotting, and CLI.
- `tests/`: pytest numerical/behavior checks; `conftest.py` selects Agg before
  pyplot imports. Ring-network references use independent SciPy `expm`.
- `datasets/`: bundled images plus exported `*_ref.mat` Octave fixtures.
- `matlab/liboniEA2025image/`, `matlab/budzinskiEAexact/`: upstream submodules;
  do not edit them to make Python comparisons pass.
- `matlab/export_segmentation_references.m`: reference exporter calling
  upstream functions with shared images/initial states and recorded parameters.
- `docs/references/04_paper_vs_matlab_drift.md` and
  `scripts/probe_paper_vs_matlab_drift.py`: paired investigation and report.
- `plots/`: committed PNG/GIF examples embedded by README; unrelated scratch
  output is not a deliverable to overwrite.
- `src/utils.py`: legacy standalone `simple_nanvar`, unused by `src/cv_rnn`.

## Development Commands

Run from the repository root; Python imports depend on it:

```bash
uv sync --locked
uv run --locked python -m pytest -q
uv run --locked python -m src.cv_rnn xor
uv run --locked python -m src.cv_rnn memory --plot
uv run --locked python -m src.cv_rnn message --text "HELLO WORLD"
uv run --locked python -m src.cv_rnn segmentation --dataset natural --plot
uv run --locked python scripts/probe_paper_vs_matlab_drift.py
```

Generate Octave reference fixtures through the container's `$SCRIPT` contract:

```bash
docker build -t octave:optimized .
docker run --rm -v "$PWD:/work" \
  -e SCRIPT=/work/matlab/export_segmentation_references.m octave:optimized
uv run --locked python -m pytest tests/test_image_segmentation.py -q
```

The Dockerfile installs Octave and `octave-statistics` (`pdist2`, `kmeans`).
Default `SCRIPT` is `/matlab/hello_world.m`. Mount the repository read/write
for export so generated files persist under host `datasets/`. The exporter uses
explicit MATLAB binary `-v7` format; `matlab/octave_compat/rng.m` supplies only
the upstream `rng(seed)` call on Octave versions lacking it.

## Code Conventions & Common Patterns

- Both numerical families default to float64/complex128. Segmentation raw
  amplitudes can overflow float32; do not downcast parity fixtures.
- Explicit device and RNG inputs; equal seeds across Torch/MATLAB/Octave do
  not imply equal draws. Cross-runtime tests share the exported initial array.
- Plain descriptive `ValueError` for invalid inputs; numerical overflow is
  reported rather than normalized away. Preserve NaN/background semantics.
- No DI framework, structured logging, or async model code.
- Label IDs are arbitrary: compare partitions with ARI. Eigenvector global
  phase is canonicalized (largest-magnitude entry rotated to real-positive)
  on both Python and Octave sides before taking `real(V)`: this is
  backend-independent, not tied to one LAPACK's triangle-convention. The
  three reference cases enforce trajectory/mask/rho/projection parity and
  sklearn-on-both-projections agreement. Octave's own `kmeans` labels are
  not a parity target (3shapes ARI ~0.40 vs sklearn on the same array).
  Canonicalization does not resolve genuinely degenerate eigenspaces.

## Important Files

- `pyproject.toml` + committed `uv.lock`: dependency definition and resolution.
- `.python-version`: Python 3.12 selection.
- `src/cv_rnn/__init__.py`: public exports; `__main__.py`: CLI flags/dispatch.
- `defns.py`: root/data paths (`DATA_DIR` is a string; wrap with `Path`).
- `Dockerfile`, `scripts/octave_runner.sh`: Octave runtime and script dispatch.
- `main.py`: greeting template, not the application entry point.
- `README.md`: usage and scientific caveats; message transmission is not
  secure encryption, and segmentation quality is seed-sensitive.

## Runtime/Tooling Preferences

Use **uv**. `uv.lock` is the committed, locked dependency authority;
`uv sync --locked` installs it and `uv run --locked` executes commands in
it. Python 3.12 is pinned; project metadata requires >=3.12. No installed
`posn` console command, build backend, or project lint/format/CI gate is
configured.

## Testing & QA

Use focused tests for changed behavior, then `uv run --locked python -m pytest`.
Keep independent numerical references independent of production helpers.
`test_image_segmentation.py` consumes Octave-generated references, sharing the
actual image and complex initial state; it must not derive image frequencies
from initial phases or assume cross-runtime RNG parity. Regenerate fixtures
when upstream inputs, numerical parameters, or the submodule revision change.
Octave execution of MATLAB source is not verification on proprietary MATLAB.
Ring-network tests remain SciPy-reference checks, not cross-runtime verification.
Shape-demo ARI=1 applies to the provided examples and selected seeds, not all
random initializations. No coverage threshold is configured.

Under uv, `threadpoolctl` emits a RuntimeWarning about Intel OpenMP
(`libiomp`, bundled with torch) and LLVM OpenMP (`libomp`, used by sklearn)
being loaded together. Observed as a warning on this macOS machine; it is a
documented deadlock source on Linux. Do not set `KMP_DUPLICATE_LIB_OK`.
See the threadpoolctl multiple-OpenMP guidance if a hang appears.
