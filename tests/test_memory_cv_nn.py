"""Reference equations, boundary transitions, and local decoding for memory."""

import numpy as np
import pytest
import torch
from scipy.linalg import expm

from src.cv_rnn import MemoryCVNN


def _matlab_operator(n):
    idx = np.arange(n)
    distance = np.abs(idx[:, None] - idx[None, :]).astype(float)
    distance = np.minimum(distance, n - distance)
    np.fill_diagonal(distance, np.inf)
    adjacency = 1 / distance
    adjacency /= adjacency[0].sum()
    return 45 * np.exp(-1.55j) * adjacency + 1j * 2 * np.pi * 10 * np.eye(n)


def test_memory_inputs_recover_matlab_targets_under_independent_dynamics():
    memory = MemoryCVNN(device="cpu")
    _, first, second, _ = memory.memory_inputs(seed=1)
    forward = expm(3.0 * _matlab_operator(321))
    targets = forward @ np.column_stack((first.numpy(), second.numpy()))

    # MATLAB item 2 is nodes 41:80; item 6 is nodes 201:240, one-based.
    np.testing.assert_allclose(np.angle(targets[40:80, 0]), 0.0, atol=2e-11)
    np.testing.assert_allclose(np.angle(targets[200:240, 1]), 0.0, atol=2e-11)
    assert np.all(np.abs(targets) >= 2.0)
    assert np.all(np.abs(targets) < 2.5)
    expected = torch.zeros((8, 2), dtype=torch.bool)
    expected[1, 0] = True
    expected[5, 1] = True
    torch.testing.assert_close(memory.decode(torch.from_numpy(targets)), expected)


def test_timeline_matches_matlab_inclusive_boundary_overwrites():
    # A small ring makes an independent step-by-step matrix-exponential
    # reference cheap while retaining the full 8001-sample MATLAB time grid.
    memory = MemoryCVNN(N=17, device="cpu")
    initials = memory.memory_inputs(seed=3)
    result = memory.run(seed=3)
    dt = 0.001
    expected = np.empty((17, 8001), dtype=np.complex128)
    recalls = []
    step = expm(dt * _matlab_operator(17))
    offset = 0
    for segment, (duration, initial) in enumerate(zip((1, 3, 3, 1), initials)):
        state = initial.numpy().copy()
        steps = round(duration / dt)
        # Each inclusive segment overwrites the previous endpoint. Retain
        # copies of the pre-update states so later writes cannot change them.
        for index in range(steps + 1):
            expected[:, offset + index] = state
            if index != steps:
                state = step @ state
        if segment in (1, 2):
            recalls.append(state.copy())
        offset += steps

    np.testing.assert_array_equal(result.times.numpy(), np.arange(8001) * dt)
    np.testing.assert_allclose(
        result.trajectory.numpy(), expected, rtol=2e-10, atol=2e-10
    )
    np.testing.assert_array_equal(result.recall_times.numpy(), [4.0, 7.0])
    np.testing.assert_allclose(
        result.recall_states.numpy(), np.column_stack(recalls), rtol=2e-10, atol=2e-10
    )


def test_default_sequence_recalls_two_items_then_clears():
    memory = MemoryCVNN(device="cpu")
    result = memory.run(seed=1, dt=0.01)
    expected = torch.zeros((8, 2), dtype=torch.bool)
    expected[1, 0] = True
    expected[5, 1] = True
    torch.testing.assert_close(memory.decode(result.recall_states), expected)
    # At exactly 7 s the second target has already been replaced by the
    # independent asynchronous clearing state, not appended to or retained.
    torch.testing.assert_close(
        memory.decode(result.trajectory[:, 700]), torch.zeros(8, dtype=torch.bool)
    )


def test_decoder_ignores_remainder_node_without_forcing_one_item():
    memory = MemoryCVNN(device="cpu")
    phases = 2 * np.pi * torch.arange(40, dtype=torch.float64) / 40
    state = torch.cat((torch.exp(1j * phases).repeat(8), torch.ones(1)))
    state[40:80] = 1.0
    state[200:240] = np.exp(1.2j)
    state *= torch.linspace(0.5, 3.5, 321, dtype=torch.float64)
    state[-1] = 1e12 * (1 + 1j)  # participates in dynamics, but no decoder owns it
    expected = torch.zeros(8, dtype=torch.float64)
    expected[[1, 5]] = 1.0

    torch.testing.assert_close(memory.synchrony(state), expected, rtol=0, atol=1e-12)
    torch.testing.assert_close(memory.decode(state), expected.bool())


def test_sampling_grid_must_include_each_cue_boundary():
    memory = MemoryCVNN(N=17, device="cpu")
    with pytest.raises(ValueError):
        memory.run(dt=0.3)
