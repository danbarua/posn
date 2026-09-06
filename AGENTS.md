# Repository Guidelines

## Project Overview

POSN is a Python/PyTorch research implementation of complex-valued oscillator networks for image segmentation and computational tasks, based on Liboni and Budzinski papers. It uses phase dynamics and synchrony, not a conventional neural-network training loop. MATLAB submodules provide scientific reference implementations.

## Architecture & Data Flow

- **Segmentation — `src/cv_rnn/cv_rnn_segmentation.py`:** an `(H, W)` image scaled to radians enters `run_2layer_torch`. `gaussian_sheet_torch` builds dense Gaussian coupling; normalized Euler updates evolve complex states. Layer one identifies background; layer two disables its coupling and restarts from the initial phases. Outputs are complex history `(N, T)`, where `N = H * W`, and a boolean background mask; masked history contains NaNs.
- **Clustering:** `spatiotemporal_segmentation_torch` computes windowed conjugate correlations, eigendecomposes them, projects into real features, then runs scikit-learn KMeans on CPU. It returns an `(H, W)` label map plus intermediates; background is `-1`, foreground clusters start at `0`.
- **XOR — `src/cv_rnn/cv_nn_xor.py`:** `XorCVNN` owns cached ring-coupling/system matrices (`_K`, `_M`). Phase-pattern inputs evolve through eigendecomposition; regional synchrony thresholds produce truth-table readouts. This class is not a `torch.nn.Module` and has no optimizer.
- Plotting lives in `cv_nn_plot_phase_dynamics.py` and `cv_rnn_plot_spectral_clustering.py`. Numerical code is synchronous; device and randomness dependencies are explicit parameters, not framework-managed services.

## Key Directories

- `src/cv_rnn/`: numerical models and visualization helpers.
- `tests/`: CPU synchrony checks and intended MATLAB-parity tests.
- `datasets/`: bundled `2shapes.mat`, `3shapes.mat`, and `natural_image.mat`; these are inputs, not reference outputs.
- `matlab/liboniEA2025image/`, `matlab/budzinskiEAexact/`: upstream reference submodules; initialize with `git submodule update --init --recursive` when needed.
- `docs/references/`: extracted papers and supplementary material for mathematical context. `docs/_archive/` contains historical proposals, not current architecture or a task list.

## Development Commands

Run from the repository root so `src` and `defns` imports resolve:

```sh
uv sync --locked
uv run --locked python -m pytest tests/
uv run --locked python -m pytest tests/test_xor_cv_nn.py::test_convergence_to_perfect_synchrony
uv run --locked python -m src.cv_rnn.cv_rnn_segmentation --dataset-dir datasets
uv run --locked python -m src.cv_rnn.cv_nn_xor
```

The segmentation demo accepts `--gpu` (CUDA when available, otherwise CPU). Demos are interactive and computationally expensive: segmentation uses dense matrices/eigendecompositions and downloads `2shapes.mat` if absent; XOR's `truth_table` unconditionally plots. Do not use demos as cheap unattended smoke checks.

No build/release command, installed CLI, or lint/format task is configured. Ruff and mypy are listed only in the Conda environment; if installed, `ruff check src tests` and `mypy src` are optional diagnostics, not established project gates.

## Code Conventions & Common Patterns

- Follow existing four-space indentation, `snake_case` functions, `PascalCase` classes, and `_private` helpers. Mathematical identifiers such as `K`, `M`, `rho`, `V`, and `D` are conventional here; document tensor shapes and units.
- Preserve complex arithmetic: complex64 state/coupling, phase angles in radians, conjugation in correlations, and float32 projections. Background NaNs carry meaning; do not indiscriminately replace them throughout the pipeline.
- Use explicit `device` and `torch.Generator` parameters where supported; seed numerical checks. Keep simulation state local except for `XorCVNN`'s cached matrices. KMeans and plotting cross into CPU/NumPy; do not assume end-to-end GPU support.
- Public numerical functions generally have type hints and scientific docstrings; match the surrounding style rather than imposing a new convention. Errors mostly propagate from numerical libraries, with some `ValueError`s and printed diagnostics. No async, dependency-injection framework, structured logging policy, or configured formatter is present.

## Important Files

- `src/cv_rnn/__init__.py`: public exports (`gaussian_sheet_torch`, `run_2layer_torch`, `spatiotemporal_segmentation_torch`, `XorCVNN`); eagerly imports model and plotting dependencies.
- `defns.py`: repository-relative data locations. `DATA_DIR` is a string; use `Path(DATA_DIR)` for `/` joins. `DATASET_2SHAPES` omits the `.mat` suffix.
- `src/utils.py`: `simple_nanvar`; inspect its low-count and complex-number behavior before reuse.
- `pyproject.toml`, `uv.lock`, `.python-version`: dependency metadata, locked resolution, and interpreter selection.
- `main.py` is a greeting template, not the model runner. `README.md` is useful for scientific context, but its `cvrnn`/`computational_cvrnn` imports and directory tree do not describe current code.

## Runtime/Tooling Preferences

Use **Python 3.12** (`.python-version`), despite metadata allowing `>=3.12`. Prefer `uv` for reproducibility with the committed lockfile; no explicit maintainer mandate requires it. Key constraints include PyTorch `~=2.2.2`, torchvision `~=0.17.2`, and NumPy `~=1.26.4`.

`requirements.txt` mirrors direct dependency constraints and supports a pip/venv alternative. `environment.yml` is a broader, divergent Conda environment, not an equivalent locked setup. `scripts/setup_conda.sh` only checks for Conda; it does not create or activate an environment. There is no configured packaging build backend.

## Testing & QA

Pytest 7.4.4 is locked. Tests use plain `test_*` functions, assertions, and parametrization; all select CPU. No CI workflow, coverage plugin/threshold, or shared pytest configuration is present.

Static inspection identifies existing caveats; these are not test-run results:

- `tests/test_image_segmentation.py` expects `datasets/{2shapes,3shapes,natural}_ref.mat` containing `mask`, `save_x`, and `cluster_map`. These fixtures are absent. Its string `DATA_DIR / filename` expression raises before the intended missing-fixture skip. Intended assertions compare wrapped phase MSE, identical masks, and label-permutation-invariant adjusted Rand score.
- `tests/test_xor_cv_nn.py` checks oscillator synchrony, not the public XOR truth table. Several synchrony cases use unseeded randomness.
- `tests/test_local_synchrony.py` includes a width-100 case leaving one outside oscillator; its synchrony is necessarily one, contradicting the expected lower outside synchrony.

For numerical changes, run focused CPU checks first, use seeded inputs, and distinguish known baseline defects or missing references from regressions. Do not claim MATLAB equivalence from bundled input data alone.
