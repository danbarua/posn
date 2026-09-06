"""A coherent patch is designed at a specified time, not induced by phase masking."""

import math

import numpy as np
import pytest
import torch
from scipy.linalg import expm

from src.cv_rnn import XorCVNN


def test_inverse_designed_input_recovers_local_phase_and_amplitude():
    network = XorCVNN(device="cpu")
    target_time = 1.25  # exercise a designed time other than the demo's 3 seconds
    mask = torch.zeros(network.N, dtype=torch.bool)
    mask[50:150] = True
    phases = torch.full((network.N,), -1.5, dtype=torch.float64)
    outside_count = int((~mask).sum())
    # Equally spaced outside phases have zero order parameter; the central
    # patch has a single phase, with nonuniform amplitudes everywhere.
    phases[~mask] = (
        2 * math.pi * torch.arange(outside_count, dtype=torch.float64) / outside_count
    )
    amplitudes = torch.linspace(1.5, 3.4, network.N, dtype=torch.float64)
    target = amplitudes * torch.exp(1j * phases)

    initial = network.design_input(target, target_time)
    # An independent matrix exponential prevents matching errors in forward
    # and backward FFT implementations from cancelling inside this test.
    recovered = expm(network.M.numpy() * target_time) @ initial.numpy()

    np.testing.assert_allclose(recovered, target.numpy(), rtol=2e-11, atol=2e-11)
    state = torch.from_numpy(recovered)
    assert network.synchrony(state) == pytest.approx(1.0, abs=1e-12)
    outside_synchrony = torch.exp(1j * torch.angle(state[~mask])).mean().abs().item()
    assert outside_synchrony == pytest.approx(0.0, abs=1e-12)
