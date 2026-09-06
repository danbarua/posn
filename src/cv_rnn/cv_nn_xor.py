"""XOR by transient phase synchrony, ported from cvnn_xor_gate.m.

Reference: matlab/budzinskiEAexact/cvnn_xor_gate.m and its circulant eigensystem.
The dynamics are linear: x(t) = exp((i*omega*I + K)*t) x(0). Inputs are designed
backwards from complex target states, then added when both inputs are active.
Only the shared-cluster synchrony readout is nonlinear; there is no Boolean XOR.
"""

import cmath
import math

import matplotlib.pyplot as plt
import torch
from matplotlib.figure import Figure


class XorCVNN:
    """Distance-coupled ring with the MATLAB XOR parameters.

    Defaults reproduce N=201, epsilon=50, phi=1.56, and f=10 Hz. The shared
    cluster is Python slice 50:150 (MATLAB nodes 51:150); other sizes use the
    central half of the ring. Computation uses float64/complex128, like MATLAB.
    Amplitudes are part of the computation and must not be normalized away.
    """

    def __init__(self, N: int = 201, device: str | torch.device = "cpu") -> None:
        if N < 4:
            raise ValueError("N must be at least 4 for a central synchronized cluster")
        self.N = N
        self.device = torch.device(device)
        self.cluster = slice(N // 4, 3 * N // 4)

        idx = torch.arange(N, dtype=torch.float64, device=self.device)
        distance = (idx[:, None] - idx[None, :]).abs()
        distance = torch.minimum(distance, N - distance)
        distance.fill_diagonal_(math.inf)
        adjacency = distance.reciprocal()  # power-law exponent alpha=1
        adjacency /= adjacency[0].sum()

        self._K = 50.0 * cmath.exp(-1.56j) * adjacency
        self._M = self._K.clone()
        self._M.diagonal().add_(1j * 2 * math.pi * 10.0)
        # MATLAB's negative-sign Fourier basis diagonalizes this circulant K.
        self._rates = torch.fft.fft(self._K[0]) + 1j * 2 * math.pi * 10.0

    @property
    def K(self) -> torch.Tensor:
        """Complex coupling K = epsilon * exp(-i*phi) * adjacency."""
        return self._K

    @property
    def M(self) -> torch.Tensor:
        """Full linear operator M = i*omega*I + K."""
        return self._M

    def evolve(
        self, x0: torch.Tensor, times: torch.Tensor | list[float]
    ) -> torch.Tensor:
        """Return complex states (N, len(times)) at times in seconds.

        Negative times implement the inverse dynamics. FFTs apply the same
        orthonormal Fourier basis as MATLAB's circulant_eigensystem, without
        constructing eigenvectors or assuming a generic eigensolver is unitary.
        """
        x0 = torch.as_tensor(x0, dtype=torch.complex128, device=self.device)
        times = torch.as_tensor(times, dtype=torch.float64, device=self.device)
        if x0.shape != (self.N,) or times.ndim != 1:
            raise ValueError("Expected x0 with shape (N,) and one-dimensional times")
        coefficients = torch.fft.ifft(x0, norm="ortho")
        modes = coefficients[:, None] * torch.exp(self._rates[:, None] * times)
        return torch.fft.fft(modes, dim=0, norm="ortho")

    def design_input(
        self, target: torch.Tensor, target_time: float = 3.0
    ) -> torch.Tensor:
        """Calculate x(0) = exp(-M*target_time) target, preserving amplitudes."""
        return self.evolve(target, [-target_time])[:, 0]

    def xor_inputs(
        self, target_time: float = 3.0, seed: int = 1
    ) -> dict[tuple[int, int], torch.Tensor]:
        """Construct the four initial states in the MATLAB example.

        X and Y target the same cluster at phases -1.5 and +1.5. Each target
        has independent amplitudes in [1.5, 3.5) and random outside phases.
        The zero-input case is a separate random unit-amplitude state, not an
        extra baseline added to the active inputs. PyTorch and MATLAB seeds
        do not produce identical random samples; compare with shared targets
        and initial states when checking cross-runtime numerical parity.
        """
        generator = torch.Generator(device=self.device).manual_seed(seed)
        inputs = []
        for phase in (-1.5, 1.5):
            theta = (
                2
                * math.pi
                * (
                    torch.rand(
                        self.N,
                        generator=generator,
                        dtype=torch.float64,
                        device=self.device,
                    )
                    - 0.5
                )
            )
            theta[self.cluster] = phase
            amplitudes = 1.5 + 2 * torch.rand(
                self.N, generator=generator, dtype=torch.float64, device=self.device
            )
            target = amplitudes * torch.exp(1j * theta)
            inputs.append(self.design_input(target, target_time))

        theta0 = (
            2
            * math.pi
            * (
                torch.rand(
                    self.N, generator=generator, dtype=torch.float64, device=self.device
                )
                - 0.5
            )
        )
        x, y = inputs
        return {
            (0, 0): torch.exp(1j * theta0),
            (1, 0): x,
            (0, 1): y,
            (1, 1): x + y,
        }

    def synchrony(self, state: torch.Tensor) -> float:
        """Kuramoto order parameter of the shared central cluster."""
        phases = torch.angle(state[self.cluster])
        return torch.exp(1j * phases).mean().abs().item()

    def truth_table(
        self,
        target_time: float = 3.0,
        threshold: float = 0.8,
        seed: int = 1,
        *,
        plot: bool = False,
        dt: float = 1e-3,
    ) -> dict[tuple[int, int], int]:
        """Decode each input from one cluster at the designed target time.

        The MATLAB script supplies trajectories; threshold=0.8 implements
        the paper's synchrony decoder, not a threshold specified by the script.
        Random realizations and non-reference sizes need not all separate at
        this threshold. No seeds are retried and no outputs are forced.

        With plot=True, dt sets the plot sampling interval in seconds; the
        exact target time is always included. Otherwise only that time is
        evaluated, avoiding allocation of four full trajectories.
        """
        if not math.isfinite(target_time) or target_time <= 0:
            raise ValueError("target_time must be positive and finite")
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be positive and finite")
        times = torch.tensor([target_time], dtype=torch.float64, device=self.device)
        if plot:
            times = torch.cat(
                (
                    torch.arange(
                        0, target_time, dt, dtype=torch.float64, device=self.device
                    ),
                    times,
                )
            )

        result = {}
        for bits, x0 in self.xor_inputs(target_time, seed).items():
            trajectory = self.evolve(x0, times)
            result[bits] = int(self.synchrony(trajectory[:, -1]) > threshold)
            if plot:
                self.plot_trajectory(str(bits), trajectory, times)
        if plot:
            plt.show()
        return result

    def plot_trajectory(
        self, title: str, trajectory: torch.Tensor, times: torch.Tensor
    ) -> tuple[Figure, Figure]:
        """Plot nodes versus time in seconds and the final phase snapshot.

        Returns figures without showing them; the caller controls display.
        """
        phases = torch.angle(trajectory).detach().cpu().numpy()
        seconds = times.detach().cpu().numpy()
        phase_figure, phase_axes = plt.subplots(figsize=(8, 3), layout="constrained")
        image = phase_axes.imshow(
            phases,
            aspect="auto",
            origin="upper",
            extent=(seconds[0], seconds[-1], self.N + 0.5, 0.5),
            interpolation="nearest",
            cmap="bone",
            vmin=-math.pi,
            vmax=math.pi,
        )
        phase_axes.set(
            title=f"Phase dynamics {title}", xlabel="time (s)", ylabel="nodes"
        )
        phase_figure.colorbar(image, ax=phase_axes, label="phase (rad)")

        state_figure, state_axes = plt.subplots(figsize=(6, 3), layout="constrained")
        state_axes.scatter(range(1, self.N + 1), phases[:, -1], c="black", s=10)
        state_axes.set(
            title=f"State {title} at {seconds[-1]:g} s",
            xlabel="nodes",
            ylabel="phase (rad)",
            xlim=(0, self.N + 1),
            ylim=(-math.pi, math.pi),
        )
        return phase_figure, state_figure


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MATLAB-reference cv-NN XOR demo")
    parser.add_argument(
        "--plot", action="store_true", help="show all four trajectories"
    )
    args = parser.parse_args()
    table = XorCVNN().truth_table(plot=args.plot)
    print("XOR truth table (shared-cluster synchrony at 3 s)\n X  Y | f(X,Y)")
    for (x, y), output in sorted(table.items()):
        print(f" {x}  {y} |   {output}")
