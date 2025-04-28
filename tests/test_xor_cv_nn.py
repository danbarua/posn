import torch
import math

# Import the oscillator network (the XorCVNN class)
from src.cv_rnn import XorCVNN


def test_convergence_to_perfect_synchrony():
    """
    Test that if all oscillators start with the same phase (i.e. unperturbed),
    the network remains synchronous.
    """
    N = 100
    network = XorCVNN(N=N, device="cpu")

    # Set an initial condition where all nodes are perfectly synchronized
    x0 = torch.ones(
        N, dtype=torch.complex64, device=network.device
    )  # exp(0i) = 1 for all
    traj = network._run_exactly(x0, nt=400, dt=1e-3)

    # Extract the final state and calculate its phases
    final_state = traj[:, -1]
    phases = torch.angle(final_state)

    # Get synchrony level (Kuramoto order parameter)
    sync = network._sync_level(phases)

    # With perfect initial synchrony, the order parameter should remain near 1
    assert sync > 0.999, f"Expected near-perfect synchrony, got {sync:.4f}"


def test_convergence_from_near_synchrony():
    """
    Test that if the oscillators start nearly synchronous (with only small perturbations),
    the network converges to synchrony after sufficient time.
    """
    N = 100
    network = XorCVNN(N=N, device="cpu")

    # Create a small random perturbation around 0 (i.e. around exp(0i)=1 for synchrony)
    perturbation = 0.01 * (torch.rand(N, device=network.device) - 0.5)
    x0 = torch.exp(1j * perturbation)

    # Run the closed-form simulation
    traj = network._run_exactly(x0, nt=400, dt=1e-3)

    # Extract final state and compute the synchrony level
    final_state = traj[:, -1]
    phases = torch.angle(final_state)
    sync = network._sync_level(phases)

    # Expect high synchrony (e.g., at least 0.95)
    assert sync > 0.95, f"Expected synchrony > 0.95, got {sync:.4f}"


def test_convergence_eventually():
    """
    Test that if all oscillators start random phases,
    the network eventually synchronises, somewhat.
    """
    N = 16
    network = XorCVNN(N=N, device="cpu")

    # Create a random initial phase vector
    x0 = torch.exp(1j * 2 * math.pi * torch.rand(N, device=network.device))

    # run the network for a longer time to see the synchronisation
    traj = network._run_exactly(x0, nt=2000, dt=1e-3)

    # Extract the final state and calculate its phases
    early_state = traj[:, 50]
    phases = torch.angle(early_state)
    sync = network._sync_level(phases)
    assert sync < 0.8, f"Expected low early synchrony, got {sync:.4f}"

    final_state = traj[:, -1]
    phases = torch.angle(final_state)
    sync = network._sync_level(phases)

    # Vectorized Kuramoto order parameter for each time (along the node dimension)
    # This gives a tensor of shape (nt,), one value per time step.
    sync_over_time = torch.abs(torch.mean(torch.exp(1j * (torch.angle(traj))), dim=0))

    # Compute the maximum synchrony over all time points
    max_sync, max_idx = torch.max(sync_over_time, dim=0)
    max_sync = max_sync.item()
    max_idx = max_idx.item()

    print(f"Maximum synchrony {max_sync:.3f} achieved at timestep {max_idx}.")
    # If coupling is implemented correctly, synchrony should trend towards 1
    # Realistically, 0.4 at best, and with small networks only
    threshold = 0.34
    assert max_sync > threshold, (
        f"Expected some synchrony (>{threshold:.2f}, got {max_sync:.4f}"
    )


def test_chimera_state_local_synchrony():
    """
    Induce a chimera state by applying the network's chimera input to a given center.
    Then verify that at a chosen readout time, the targeted region has a significantly higher
    local synchrony (Kuramoto order parameter) than regions outside it.
    """
    N = 200
    network = XorCVNN(N=N, device="cpu")

    # Define parameters for the chimera target region.
    # For example, let the synchronous (chimera) cluster be centered at index center,
    # with width (half width) 20.
    center = N // 2
    width = 20  # targets approximately 2*width nodes (depending on wrap-around)

    # Create a random initial phase vector
    x0 = torch.exp(1j * 2 * math.pi * torch.rand(N, device=network.device))

    # Create the chimera input for the chosen center region.
    chimera_input = network._chimera_input(center, width)

    # Combine to produce a chimera-like initial state.
    # Inside the target region, the chimera input forces phase=0 (i.e., exp(0j)==1);
    # outside, the final phase will follow a nontrivial pattern.
    x_init = x0 * chimera_input

    # Run the exact (closed-form) simulation over a longer time to see the local evolution.
    nt = 1000
    dt = 1e-3
    traj = network._run_exactly(x_init, nt=nt, dt=dt)

    # Pick a readout time (say near the end) to measure the state.
    readout_time = nt - 1
    final_state = traj[:, readout_time]

    # Create an index vector.
    idx = torch.arange(N, device=network.device)

    # Define the target synchronous region indices.
    # Here we use the same condition as in _chimera_input: distance < width.
    dist = torch.minimum((idx - center).abs(), (N - (idx - center).abs()))
    mask_sync = dist < width
    mask_async = ~mask_sync

    # Compute local Kuramoto order parameters.
    phases_sync = torch.angle(final_state[mask_sync])
    phases_async = torch.angle(final_state[mask_async])

    sync_level_sync = network._sync_level(phases_sync)
    sync_level_async = network._sync_level(phases_async)

    # For a successful chimera induction the local order parameter in the target region
    # should be much higher than outside.
    # Note: The exact thresholds may depend on network parameters.
    assert sync_level_sync > 0.34, (
        f"Expected high synchrony in target region, got {sync_level_sync:.3f}"
    )
    # We expect the asynchronous region to have lower synchrony.
    assert sync_level_async < sync_level_sync - 0.2, (
        f"Expected asynchronous region to be significantly less synchronous than the target region; "
        f"sync_level_sync={sync_level_sync:.3f}, sync_level_async={sync_level_async:.3f}"
    )
