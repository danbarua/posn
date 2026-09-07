"""Cross-runtime checks against upstream MATLAB source executed by GNU Octave.

Regenerate datasets/*_ref.mat with matlab/export_segmentation_references.m.
Shared images/initial states avoid cross-runtime RNG assumptions. Keep raw
complex128 amplitudes: the supplied trajectories overflow single precision.
Eigenvector phase is canonicalized identically on both sides (see
spatiotemporal_segmentation_torch and the exporter), so the real projection
is a legitimate cross-runtime target independent of LAPACK backend.

The sklearn-on-both-projections ARI==1.0 assert is same-algorithm consistency
on arrays that already agree to ~4e-12, not clustering-parity with Octave's
kmeans. The 3shapes fixture under Octave's exported x0 is poorly separable
(Python vs ground truth ARI 0.496); do not read it as a quality result.
"""

from pathlib import Path

import numpy as np
import pytest
from scipy.io import loadmat
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import torch

from defns import DATA_DIR
from src.cv_rnn import run_2layer_torch, spatiotemporal_segmentation_torch


@pytest.mark.parametrize("demo_name", ["2shapes", "3shapes", "natural"])
def test_cv_rnn_against_matlab(demo_name):
    ref_file = Path(DATA_DIR) / f"{demo_name}_ref.mat"
    assert ref_file.exists(), (
        f"Missing {ref_file}; run matlab/export_segmentation_references.m in Octave"
    )
    ref = loadmat(ref_file)
    schema = int(np.asarray(ref["schema_version"]).reshape(-1)[0])
    assert schema == 2, (
        f"{ref_file} has schema_version={schema}; regenerate with "
        "matlab/export_segmentation_references.m"
    )
    image = torch.from_numpy(ref["im"])
    initial_state = torch.from_numpy(ref["initial_state"].reshape(-1))
    nt = tuple(int(value) for value in ref["nt"].ravel())
    states, mask = run_2layer_torch(
        image,
        initial_state=initial_state,
        alpha=tuple(ref["alpha"].ravel()),
        sigma=tuple(ref["sigma"].ravel()),
        nt=nt,
    )
    reference_mask = ref["mask"].ravel().astype(bool)
    np.testing.assert_array_equal(mask.numpy(), reference_mask)
    np.testing.assert_allclose(
        states.numpy(), ref["save_x"], rtol=1e-10, atol=1e-12, equal_nan=True
    )
    n_clusters = int(ref["n_clusters"].item())
    labels, rho, _, _, projection = spatiotemporal_segmentation_torch(
        states,
        image,
        mask,
        nt_mask=nt[0],
        n_clusters=n_clusters,
        window_size=int(ref["window_size"].item()),
        window_step=int(ref["window_step"].item()),
    )
    np.testing.assert_allclose(
        rho[:, :, -1].numpy(), ref["rho_final"], rtol=1e-10, atol=1e-12
    )
    np.testing.assert_allclose(
        projection[:, :, -1].numpy(), ref["projection_final"], rtol=1e-9, atol=1e-11
    )
    predicted = labels.numpy()
    truth = ref["cluster_map"]
    np.testing.assert_array_equal(predicted == -1, truth == -1)
    # projection_final is foreground-only in MATLAB column-major order.
    # Extract Python's labels in the same order before comparing partitions.
    sklearn_on_octave = KMeans(
        n_clusters=n_clusters, random_state=0, n_init=1
    ).fit_predict(ref["projection_final"])
    python_foreground = predicted.T.reshape(-1)[reference_mask == False]
    assert adjusted_rand_score(python_foreground, sklearn_on_octave) == 1.0
