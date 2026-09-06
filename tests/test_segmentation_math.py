"""Independent equations, precision and boundaries for the actual segmentation code."""

import numpy as np
import pytest
import torch

from src.cv_rnn import gaussian_sheet_torch, run_2layer_torch, spatiotemporal_segmentation_torch


def _gaussian(shape, alpha, sigma):
    # Literal MATLAB meshgrid(row,col), then column-major (:), not a call to
    # the production connectivity helper. Retain its rectangular convention.
    rows, cols = np.meshgrid(
        np.arange(1, shape[0] + 1) / shape[0],
        np.arange(1, shape[1] + 1) / shape[1],
    )
    points = np.column_stack((rows.ravel(order="F"), cols.ravel(order="F")))
    squared = ((points[:, None] - points[None, :]) ** 2).sum(axis=2)
    return alpha * np.exp(-squared / (2 * sigma**2))


def test_rectangular_gaussian_matches_matlab_positions():
    actual = gaussian_sheet_torch(2, 3, 0.5, 0.3).numpy()
    np.testing.assert_allclose(actual, _gaussian((2, 3), 0.5, 0.3), rtol=1e-13, atol=1e-14)


def test_weak_connections_receive_constant_complex_phase():
    expected = _gaussian((2, 3), 0.5, 0.3).astype(complex)
    expected[expected.real < 0.2] *= np.exp(0.9j)
    actual = gaussian_sheet_torch(2, 3, 0.5, 0.3, phi=0.9, phi_threshold=0.2)
    np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-13, atol=1e-14)


def test_full_raw_recurrence_restart_and_nan_boundary():
    image = np.array([[0.2, -0.4, 0.7], [1.1, -0.1, 0.3]])
    initial = np.array([0.5, 1.2, 0.8, 2.0, 0.4, 1.7]) * np.exp(1j * np.array([-1, -2, 2, 0.2, 0.7, 1]))
    alpha, sigma, nt = (0.5, 0.3), (0.9, 0.3), (4, 8)
    omega = image.ravel(order="F")
    operator = _gaussian(image.shape, alpha[0], sigma[0]) + 1j * np.diag(omega)
    expected = np.empty((image.size, nt[1]), dtype=complex)
    expected[:, 0] = initial
    for time in range(1, nt[0]):
        expected[:, time] = operator @ expected[:, time - 1]
    phase = np.angle(expected[:, nt[0] - 1])
    above, below = phase > phase.mean(), phase < phase.mean()
    background = above if above.sum() > below.sum() else below
    operator = _gaussian(image.shape, alpha[1], sigma[1]) + 1j * np.diag(omega)
    operator[background, :] = 0
    operator[:, background] = 0
    state = initial.copy()
    state[background] = 0
    for time in range(nt[0], nt[1]):
        state = operator @ state
        expected[:, time] = state
    expected[background, nt[0]:] = np.nan
    actual, mask = run_2layer_torch(
        torch.from_numpy(image), alpha, sigma, nt, initial_state=torch.from_numpy(initial)
    )
    np.testing.assert_array_equal(mask.numpy(), background)
    np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-12, atol=1e-12, equal_nan=True)


def test_mask_tie_selects_strictly_below_mean_not_equal():
    initial = torch.exp(1j * torch.tensor([-1, -0.5, 0, 0.5, 1], dtype=torch.float64))
    states, mask = run_2layer_torch(torch.zeros(1, 5), nt=(1, 3), initial_state=initial)
    torch.testing.assert_close(mask, torch.tensor([True, True, False, False, False]))
    np.testing.assert_array_equal(np.isnan(states.numpy()), [[False, True, True]] * 2 + [[False] * 3] * 3)


def test_large_gain_retains_raw_amplitudes_beyond_float32_range():
    states, mask = run_2layer_torch(
        torch.zeros(2, 2), alpha=(50, 0.5), nt=(25, 30),
        initial_state=torch.ones(4, dtype=torch.complex128),
    )
    gain = _gaussian((2, 2), 50, 0.9)[0].sum()
    expected = np.r_[gain ** np.arange(25), 0.5 ** np.arange(1, 6)]
    np.testing.assert_allclose(states.numpy(), np.tile(expected, (4, 1)), rtol=1e-12, atol=1e-14)
    np.testing.assert_array_equal(mask.numpy(), np.zeros(4, dtype=bool))


def test_every_inclusive_window_uses_foreground_phase_and_sorted_eigenpairs():
    node, time = np.arange(6)[:, None], np.arange(8)[None, :]
    raw = (0.3 + node / 2 + 0.1 * time**2) * np.exp(1j * (0.23 * node * time + 0.07 * time**2))
    mask = np.array([False, True, False, False, True, False])
    raw[mask, 2:] = np.nan
    image = torch.zeros(2, 3)
    result, rho, vectors, diagonal, projection = spatiotemporal_segmentation_torch(
        torch.from_numpy(raw), image, torch.from_numpy(mask), nt_mask=2,
        window_size=2, window_step=2, dim=(value for value in (0, 2)),
    )
    expected_phase = np.exp(1j * np.angle(raw[~mask]))
    for window, start in enumerate((1, 3, 5)):
        samples = expected_phase[:, start:start + 3]
        expected = samples @ samples.conj().T / 3
        s, v, d = rho[:, :, window].numpy(), vectors[:, :, window].numpy(), diagonal[:, :, window].numpy()
        np.testing.assert_allclose(s, expected, atol=1e-13)
        np.testing.assert_allclose(s @ v, v @ d, atol=1e-12)
        np.testing.assert_allclose(v.conj().T @ v, np.eye(4), atol=1e-12)
        eigenvalues = np.linalg.eigvalsh(expected)
        expected_diagonal = eigenvalues[np.argsort(np.abs(eigenvalues))[::-1]]
        np.testing.assert_allclose(d, np.diag(expected_diagonal), atol=1e-12)
        np.testing.assert_allclose(projection[:, :, window].numpy(), expected.real @ v[:, (0, 2)].real, atol=1e-12)
    np.testing.assert_array_equal(result.numpy() == -1, mask.reshape((2, 3), order="F"))
    _, scaled_rho, _, _, _ = spatiotemporal_segmentation_torch(
        torch.from_numpy(raw * (1 + time) * (1 + node)), image, torch.from_numpy(mask),
        nt_mask=2, window_size=2, window_step=2,
    )
    torch.testing.assert_close(rho, scaled_rho, rtol=1e-12, atol=1e-12)


def test_nonfinite_foreground_is_not_silently_zero_filled():
    states = torch.ones((4, 8), dtype=torch.complex128)
    states[0, 3] = torch.nan
    with pytest.raises(ValueError):
        spatiotemporal_segmentation_torch(
            states, torch.zeros(2, 2), torch.zeros(4, dtype=torch.bool),
            nt_mask=2, window_size=2,
        )


def test_requested_clusters_are_not_silently_reduced():
    with pytest.raises(ValueError):
        spatiotemporal_segmentation_torch(
            torch.ones((4, 8), dtype=torch.complex128), torch.zeros(2, 2),
            torch.tensor([False, True, True, True]), nt_mask=2, window_size=2,
            dim=(0,), n_clusters=2,
        )
