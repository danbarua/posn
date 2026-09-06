"""
test_cv_rnn.py

Intended as a CI-server smoke test to verify the outputs of the Python port
against the MATLAB reference implementation.

TODO: Run the MATLAB reference implementation and save outputs in ./references folder

"""

import math
from pathlib import Path

import numpy as np
import pytest
import scipy.io as sio
import torch
from sklearn.metrics import adjusted_rand_score

from defns import DATA_DIR

# ---- module under test -------------------------------------------------
from src.cv_rnn import (
    run_2layer_torch,
    spatiotemporal_segmentation_torch,
)

DEVICE = torch.device("cpu")  # keep GPU out of the loop for CI
DTYPE = torch.float32
SEED = 1  # must match MATLAB ‘seed’ arg


# helpers ----------------------------------------------------------------
def _load_reference(mat_path):
    mat = sio.loadmat(mat_path)
    ref = {
        k: torch.from_numpy(v.squeeze()).to(
            torch.complex64 if np.iscomplexobj(v) else torch.float32
        )
        for k, v in mat.items()
        if not k.startswith("__")
    }
    return ref


def _phase_mse(a, b):
    """mean-squared error of wrapped phases in [-π, π)."""
    diff = torch.angle(torch.exp(1j * (torch.angle(a) - torch.angle(b))))
    return torch.mean(diff.abs() ** 2).item()


# ----------------------------------------------------------------------- #
#                               parametrised                              #
# ----------------------------------------------------------------------- #
@pytest.mark.parametrize("demo_name", ["2shapes", "3shapes", "natural"])
def test_cv_rnn_against_matlab(demo_name):
    ref_file = DATA_DIR / f"{demo_name}_ref.mat"
    if not ref_file.exists():
        pytest.skip(f"reference file {ref_file} missing")

    ref = _load_reference(ref_file)
    H = int(math.sqrt(ref["mask"].numel()))
    image = torch.angle(ref["save_x"][:, 0]).view(H, H).to(DTYPE)

    # ------ run Python pipeline ----------------------------------------
    g = torch.Generator(device=DEVICE).manual_seed(SEED)
    save_x_py, mask_py = run_2layer_torch(
        image,
        generator=g,
        device=DEVICE,
        dtype=DTYPE,
    )
    (cluster_py, _, _, _, _) = spatiotemporal_segmentation_torch(
        save_x_py,
        image,
        mask_py,
        nt_mask=60,  # paper default
        n_clusters=int(ref["cluster_map"].max().item() + 1),
        device=DEVICE,
    )

    # ------------------------------------------------------------------ #
    # 1.  dynamics: compare final two frames (last of each layer)        #
    # ------------------------------------------------------------------ #
    t1 = 59  # 0-based index
    t_end = save_x_py.shape[1] - 1
    mse_l1 = _phase_mse(save_x_py[:, t1], ref["save_x"][:, t1])
    mse_l2 = _phase_mse(save_x_py[:, t_end], ref["save_x"][:, t_end])
    assert mse_l1 < 1e-6
    assert mse_l2 < 1e-6

    # ------------------------------------------------------------------ #
    # 2.  mask identical                                                 #
    # ------------------------------------------------------------------ #
    assert torch.equal(mask_py, ref["mask"])

    # ------------------------------------------------------------------ #
    # 3.  segmentation: label-permutation-invariant equality             #
    # ------------------------------------------------------------------ #
    ari = adjusted_rand_score(
        ref["cluster_map"].flatten().int().cpu(),
        cluster_py.flatten().int().cpu(),
    )
    assert ari == 1.0
