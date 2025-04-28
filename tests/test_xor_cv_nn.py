import torch

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
