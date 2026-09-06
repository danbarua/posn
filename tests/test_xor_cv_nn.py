"""Numerical and behavioral checks against the MATLAB XOR construction."""

import numpy as np
import pytest
import torch
from scipy.linalg import expm

from src.cv_rnn import XorCVNN


def _matlab_operator(n):
    """Independent translation of distance_dependent_graph and K in MATLAB."""
    idx = np.arange(n)
    distance = np.abs(idx[:, None] - idx[None, :]).astype(float)
    distance = np.minimum(distance, n - distance)
    np.fill_diagonal(distance, np.inf)
    adjacency = 1 / distance
    adjacency /= adjacency[0].sum()
    return 50 * np.exp(-1.56j) * adjacency + 1j * 2 * np.pi * 10 * np.eye(n)


@pytest.mark.parametrize("n", [16, 201])  # even ring and MATLAB's odd-sized ring
def test_evolution_matches_matlab_matrix_exponential(n):
    network = XorCVNN(N=n, device="cpu")
    rng = np.random.default_rng(2)
    x0 = (1.5 + 2 * rng.random(n)) * np.exp(2j * np.pi * rng.random(n))
    times = [0.0, 0.001, 0.12, 3.0]
    operator = _matlab_operator(n)
    expected = np.column_stack([expm(operator * t) @ x0 for t in times])

    actual = network.evolve(torch.from_numpy(x0), times)

    # Compare amplitudes as well as phases. This exposes a non-unitary
    # eigenvector inverse, missing natural frequency, or state normalization.
    np.testing.assert_allclose(actual.numpy(), expected, rtol=2e-11, atol=2e-11)


def test_xor_inputs_recover_shared_targets_and_interfere():
    network = XorCVNN(device="cpu")
    inputs = network.xor_inputs(target_time=3.0, seed=1)
    forward = expm(3.0 * _matlab_operator(201))
    states = {bits: forward @ x0.numpy() for bits, x0 in inputs.items()}
    cluster = slice(50, 150)  # MATLAB's one-based nodes 51:150

    for bits, phase in [((1, 0), -1.5), ((0, 1), 1.5)]:
        target = states[bits]
        np.testing.assert_allclose(np.angle(target[cluster]), phase, atol=2e-11)
        assert np.all(np.abs(target) >= 1.5)
        assert np.all(np.abs(target) < 3.5)

    # With both inputs present, linear superposition must produce the sum of
    # the complex targets, not their product or two separate coherent patches.
    np.testing.assert_allclose(
        states[(1, 1)], states[(1, 0)] + states[(0, 1)], rtol=2e-11, atol=2e-11
    )
    combined_synchrony = np.abs(np.mean(np.exp(1j * np.angle(states[(1, 1)][cluster]))))
    assert combined_synchrony < 0.8


def test_truth_table_decodes_xor_at_the_target_time():
    network = XorCVNN(device="cpu")
    assert network.truth_table(seed=1) == {
        (0, 0): 0,
        (1, 0): 1,
        (0, 1): 1,
        (1, 1): 0,
    }


def test_readout_threshold_is_applied_to_one_cluster():
    # At zero threshold all four seeded states trigger the shared decoder.
    # XORing two thresholded region bits would incorrectly return four zeros.
    network = XorCVNN(device="cpu")
    assert network.truth_table(seed=1, threshold=0.0) == {
        (0, 0): 1,
        (1, 0): 1,
        (0, 1): 1,
        (1, 1): 1,
    }
