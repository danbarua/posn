"""All bundled shape images must separate actual objects, not just background.

This exercises the reference demo's own hardcoded seeds (1 for 2shapes, 9 for
3shapes) and their perfect ARI is specific to those seeds, not a general
property of generic random initial conditions: see
docs/references/04_paper_vs_matlab_drift.md section 4 and
scripts/probe_paper_vs_matlab_drift.py for the measured seed sensitivity.
"""

from pathlib import Path

import numpy as np
import pytest
from scipy.io import loadmat
from sklearn.metrics import adjusted_rand_score

from defns import DATA_DIR
from src.cv_rnn.segmentation_demo import run_segmentation_example


@pytest.mark.parametrize("dataset", ["2shapes", "3shapes"])
@pytest.mark.parametrize("image_index", [0, 1, 2])
def test_reference_examples_segment_every_object(dataset, image_index):
    # Exercise exactly the reference-demo workflow, including its fixed RNG,
    # default object count, background removal, dynamics and final clustering.
    # Every supplied geometry is covered; no seed searches or accepted subsets.
    result = run_segmentation_example(dataset, image_index)
    truth = loadmat(Path(DATA_DIR) / f"{dataset}.mat")["labels"][:, :, image_index]
    predicted = result.labels.numpy()
    np.testing.assert_array_equal(predicted == -1, truth == 0)
    foreground = truth != 0
    assert adjusted_rand_score(truth[foreground], predicted[foreground]) == 1.0
    # ARI on foreground alone cannot guarantee labels stayed out of background.
    assert np.all(predicted[foreground] >= 0)
