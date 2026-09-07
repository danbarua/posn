# Repository Guidelines

## Project Overview

POSN is a PyTorch research implementation of complex-valued recurrent neural
networks (cv-RNNs): fixed-weight, linear-dynamics oscillator networks that
perform image segmentation (Liboni et al., PNAS 2025) and computation — XOR,
short-term memory, message transmission (Budzinski et al., *Communications
Physics* 2024) — with no training loop. `matlab/liboniEA2025image/` and
`matlab/budzinskiEAexact/` are the upstream MATLAB reference submodules each
Python module ports from; `docs/references/` holds the papers, preprint, and
a dedicated write-up quantifying where the paper text and the MATLAB code
disagree (see Key Directories).

## Architecture & Data Flow

Two independent simulation families live under `src/cv_rnn/`; **do not mix
their conventions**:

- **Segmentation** (`cv_rnn_segmentation.py`): explicit-Euler recurrence with
  unit-circle renormalization, complex64/float32. `gaussian_sheet_torch`
  builds dense Gaussian coupling → `run_2layer_torch` evolves two layers
  (layer 1 finds the background via strict-majority mean phase; layer 2
  disables background coupling and restarts from the original phases,
  masked entries become NaN) → `spatiotemporal_segmentation_torch` computes
  windowed conjugate correlations, eigendecomposes, projects to real
  features, and runs scikit-learn KMeans on CPU (background label is `-1`).
  `segmentation_demo.py` wraps this into a `SegmentationExample` dataclass
  (`run_segmentation_example`, `plot_segmentation_comparison`,
  `save_segmentation_comparison`) used by the CLI, README, and tests. State
  vectors use **MATLAB column-major pixel order** throughout — flattening in
  the wrong order silently transposes images.
- **Ring-network demos** (`cv_nn.py`'s `RingCVNN` base, plain class, *not*
  `torch.nn.Module`): exact analytical Fourier propagator via cached FFT
  eigen-rates — no Euler step, no renormalization, complex128/float64.
  `XorCVNN` and `MemoryCVNN` subclass it with fixed MATLAB-matching
  parameters baked into `__init__` defaults (`XorCVNN`: N=201, ε=50,
  φ=1.56; `MemoryCVNN`: N=321, ε=45, φ=1.55); `MessageDemo` *composes* a
  `RingCVNN` internally (`self._dynamics`) rather than subclassing it.
  `design_input(target, target_time)` runs evolution backwards in time to
  inverse-design an initial state that reaches `target` at `target_time`.

Entry point: `src/cv_rnn/__main__.py`'s `main()` dispatches
`python -m src.cv_rnn {xor,memory,message,segmentation}` — read its
docstring for the exact per-subcommand flag set before adding a new one.
Public API surface is consolidated in `src/cv_rnn/__init__.py`; plotting
(`cv_nn_plot_phase_dynamics.animate_dynamics`,
`cv_rnn_plot_spectral_clustering.plot_spectral_clustering`, per-class
`.plot()` methods) always returns a `Figure` and never calls `plt.show()` —
callers/CLI decide whether to display it.

## Key Directories

- `src/cv_rnn/`: both model families above, plus plotting.
- `src/utils.py`: `simple_nanvar` — NaN-aware variance shared by the
  segmentation path; `n=0` → NaN, `n=1` → `0.0` regardless of `unbiased`.
- `tests/`: 9 files. Ring-network demos check against a hand-written
  `scipy.linalg.expm` reference at `rtol/atol≈2e-11`; segmentation math is
  checked against independent NumPy/SciPy re-derivations, not production
  code. `tests/conftest.py` forces the `Agg` matplotlib backend at
  collection time (must run before any test module imports `pyplot`).
- `datasets/`: bundled `2shapes.mat`, `3shapes.mat`, `natural_image.mat` —
  these are **inputs**, not reference outputs.
- `matlab/liboniEA2025image/`, `matlab/budzinskiEAexact/`: upstream
  reference submodules; `git submodule update --init --recursive` if empty.
- `docs/references/`: `00_preprint.md` (arXiv v1 excerpts, verified against
  the actual LaTeX source, not a lossy PDF conversion), `01_paper.md` /
  `02_supplementary.md` (published PNAS 2025 + SI), `03_paper.md` (Budzinski
  *Communications Physics* 2024), `04_paper_vs_matlab_drift.md` (measured
  effect of every paper-text-vs-MATLAB-code discrepancy found so far — read
  this before trusting "the paper says X" in isolation). `docs/_archive/` is
  untracked historical proposals, not current architecture.
- `scripts/probe_paper_vs_matlab_drift.py`: the investigative script behind
  `04_paper_vs_matlab_drift.md`. Re-run and update the doc together after
  touching amplitude/layer-2-init/window code or the MATLAB submodule pin.
  This is **not** a regression test (~4 min runtime, prints a report).
- `plots/`: committed demo output (PNG/GIF) referenced by `README.md`.

## Development Commands

Run from the repository root (`src`/`defns` imports depend on it):

```bash
conda activate posn
python -m pytest -q                                          # 37 passed, 3 skipped
python -m src.cv_rnn xor [--seed N] [--plot]
python -m src.cv_rnn memory [--seed N] [--plot]
python -m src.cv_rnn message [--text "HELLO"] [--frequency-hz 10] [--plot]
python -m src.cv_rnn segmentation --dataset {2shapes,3shapes,natural} [--image-index N] [--n-clusters N] [--plot]
python scripts/probe_paper_vs_matlab_drift.py                 # investigative, not a test
```

`environment.yml` (Conda env `posn`) is what this repo is actually developed
and verified against. `uv.lock` exists on disk but is **untracked in git**
(`git status` shows `?? uv.lock`) — `uv sync --locked` only works if that
file happens to already be present locally; it is not reproducible from a
fresh clone. Treat `pyproject.toml` + `requirements.txt` as the source of
truth for direct dependencies, not `uv.lock`. `requirements.txt` mirrors
`pyproject.toml`'s constraints exactly, for a pip/venv alternative.

No build system, no installed console-script entry point (always
`python -m src.cv_rnn`, never a bare `posn` command), no lint/format/CI
config (`ruff`/`mypy` appear only as IDE inspection profiles under `.idea/`
and Conda dev-tool entries in `environment.yml`, not as committed config or
a project gate).

## Code Conventions & Common Patterns

- Explicit `device`/`torch.Generator(device=...).manual_seed(seed)`
  parameters everywhere; no class manages global RNG state, and no state
  survives past one `encode`/`make_key`/`run` call.
- Complex dtype tracks the family: segmentation is complex64 (from float32
  images); every ring-network demo is complex128 (float64). Don't downcast
  across the boundary.
- Errors are plain `ValueError` with a descriptive message (unsupported
  alphabet characters, out-of-bounds target delays, missing cue-boundary
  samples), not custom exception types.
- Background/masked sentinel values are load-bearing: `-1` for background
  cluster labels, NaN for masked-out history entries — don't blanket-replace
  them without checking what reads them downstream.
- No dependency-injection framework, no structured logging, no async.

## Important Files

- `src/cv_rnn/__init__.py`: the public API (`gaussian_sheet_torch`,
  `run_2layer_torch`, `spatiotemporal_segmentation_torch`,
  `animate_dynamics`, `plot_spectral_clustering`, `XorCVNN`, `MemoryCVNN`,
  `MemoryRun`, `MessageDemo`, `MessageKey`, `Ciphertext`, `InputEvent`,
  `MessageTrace`, `DecodedSymbol`, `ChimeraAlphabet`).
- `src/cv_rnn/__main__.py`: CLI entry point; `main()`'s docstring is the
  authoritative flag reference.
- `defns.py`: `ROOT_DIR`/`DATA_DIR`/`DATASET_2SHAPES` path constants.
  `DATA_DIR` is a plain `str` (use `Path(DATA_DIR)` for `/` joins);
  `DATASET_2SHAPES` omits the `.mat` suffix.
- `docs/references/04_paper_vs_matlab_drift.md` +
  `scripts/probe_paper_vs_matlab_drift.py`: keep these two in sync.
- `main.py`: still a `print_hi` greeting template, not a real entry point.
- `README.md`: the primary usage doc, per feature, including known caveats
  (seed-sensitive segmentation ARI, message demo is explicitly not secure
  encryption, ring-network MATLAB parity is unverified — see Testing & QA).

## Runtime/Tooling Preferences

Python **3.12** pinned (`.python-version`); `pyproject.toml` only requires
`>=3.12`. Core pinned deps: `numpy~=1.26.4`, `scipy~=1.15.2`,
`scikit-learn~=1.6.1`, `matplotlib~=3.10.1`, `torch~=2.2.2`,
`torchvision~=0.17.2`, `pytest~=7.4.4`. `scripts/setup_conda.sh` only checks
whether Conda is present; it does not create or activate an environment.

## Testing & QA

Plain `test_*` functions and assertions, heavy `@pytest.mark.parametrize`,
no custom fixtures beyond `tmp_path`. Last verified: `python -m pytest -q`
→ 37 passed, 3 skipped, ~13s on CPU.

- **Independent-reference pattern**: ring-network tests (`test_xor_cv_nn.py`,
  `test_memory_cv_nn.py`, `test_message_cv_nn.py`, `test_local_synchrony.py`)
  validate against a from-scratch `scipy.linalg.expm` matrix-exponential
  reference at `rtol/atol≈2e-11` — this checks internal consistency, **not**
  MATLAB output. `test_segmentation_math.py` similarly hand-codes
  MATLAB-equivalent NumPy/SciPy formulas independent of production code.
- **The one skip**: `test_image_segmentation.py` needs
  `datasets/{2shapes,3shapes,natural}_ref.mat` (MATLAB-produced) that aren't
  bundled — true MATLAB/Octave cross-runtime parity is not yet established
  for any demo.
- `test_segmentation_objects.py` documents that ARI=1.0 on every bundled
  image is specific to the demo's hardcoded seeds (1 for 2shapes, 9 for
  3shapes); see `docs/references/04_paper_vs_matlab_drift.md` section 4 for
  the measured seed sensitivity (generic seeds: mean ARI≈0.6).
- No CI workflow, no coverage config or threshold.
