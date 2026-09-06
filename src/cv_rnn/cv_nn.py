"""Shared linear ring dynamics for the MATLAB computational cv-NN examples."""

import cmath
import math

import torch


class RingCVNN:
    """Double-precision power-law ring with an analytical Fourier propagator."""

    def __init__(
        self,
        N: int,
        epsilon: float,
        phi: float,
        device: str | torch.device = "cpu",
        *,
        frequency_hz: float = 10.0,
    ) -> None:
        if N < 2:
            raise ValueError("N must be at least 2 for a coupled ring")
        if not math.isfinite(frequency_hz):
            raise ValueError("frequency_hz must be finite")
        self.N = N
        self.device = torch.device(device)
        self.frequency_hz = frequency_hz

        idx = torch.arange(N, dtype=torch.float64, device=self.device)
        distance = (idx[:, None] - idx[None, :]).abs()
        distance = torch.minimum(distance, N - distance)
        distance.fill_diagonal_(math.inf)
        adjacency = distance.reciprocal()  # power-law exponent alpha=1
        adjacency /= adjacency[0].sum()

        self._K = epsilon * cmath.exp(-1j * phi) * adjacency
        self._M = self._K.clone()
        omega = 2 * math.pi * frequency_hz
        self._M.diagonal().add_(1j * omega)
        # MATLAB's negative-sign Fourier basis diagonalizes this circulant K.
        self._rates = torch.fft.fft(self._K[0]) + 1j * omega

    @property
    def K(self) -> torch.Tensor:
        """Complex coupling K = epsilon * exp(-i*phi) * adjacency."""
        return self._K

    @property
    def M(self) -> torch.Tensor:
        """Full linear operator M = i*omega*I + K; omega = 2*pi*frequency_hz."""
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
