import math
import torch
import pytest
from src.cv_rnn import XorCVNN


# Parameterize over several candidate target widths, desired local synchrony threshold,
# and minimal required difference between the synchronous and asynchronous regions.
@pytest.mark.parametrize(
    "width, desired_sync, min_sync_diff",
    [
        # (10, 0.70, 0.15),
        # (20, 0.75, 0.15),
        # (30, 0.80, 0.15),
        # (40, 0.80, 0.10),
        (60, 0.80, 0.10),
        (80, 0.80, 0.10),
        (100, 0.80, 0.15),
    ],
)
def test_chimera_state_local_synchrony(width, desired_sync, min_sync_diff):
    """
    Induce a chimera state by applying the network's chimera input to a given center.
    The test is parameterized by the target width and the desired minimum synchrony level in the target region.
    It then verifies that at a chosen readout time the targeted (chimera) region reaches at least the given synchrony,
    and that the asynchronous region remains sufficiently less synchronous.
    """
    N = 200
    network = XorCVNN(N=N, device="cpu")

    # Center of the chimera target region: use the middle of the ring.
    center = N // 2

    # Create a random initial phase vector
    x0 = torch.exp(1j * 2 * math.pi * torch.rand(N, device=network.device))

    # Create the chimera input for the chosen center region with the given width.
    chimera_input = network._chimera_input(center, width)

    # Combine to produce a chimera-like initial state.
    x_init = x0 * chimera_input

    # Run the closed-form simulation: run long enough for the state to settle.
    nt = 800
    dt = 1e-3
    traj = network._run_exactly(x_init, nt=nt, dt=dt)

    # Choose a readout time (near the end of simulation)
    readout_time = -1  # nt - 1
    final_state = traj[:, readout_time]

    # Create an index vector for nodes.
    idx = torch.arange(N, device=network.device)

    # Use the same criterion as _chimera_input:
    # The target (synchronous) region is defined where the circular distance from center < width.
    dist = torch.minimum((idx - center).abs(), (N - (idx - center).abs()))
    mask_sync = dist < width
    mask_async = ~mask_sync

    # Compute local Kuramoto synchrony in each region.
    phases_sync = torch.angle(final_state[mask_sync])
    phases_async = torch.angle(final_state[mask_async])
    sync_level_sync = network._sync_level(phases_sync)
    sync_level_async = network._sync_level(phases_async)

    # Assert that the synchronous region reaches the desired level.
    assert sync_level_sync > desired_sync, (
        f"With width={width}, expected sync in target region > {desired_sync:.2f}, got {sync_level_sync:.3f}"
    )

    # Assert that the asynchronous region is less synchronous by at least min_sync_diff.
    assert sync_level_async < sync_level_sync - min_sync_diff, (
        f"With width={width}, expected asynchronous region to be at least {min_sync_diff:.2f} "
        f"less synchronous than target: sync_target={sync_level_sync:.3f}, sync_async={sync_level_async:.3f}"
    )
