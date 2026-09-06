"""Port of matlab/budzinskiEAexact/cvnn_memory_task.m.

Four independently initialized segments implement background, recall of item 2,
recall of item 6, and clearing. Cue boundaries overwrite the preceding segment's
last sample, exactly as in MATLAB; pre-update recall states are retained separately.
"""

import math
from dataclasses import dataclass

import matplotlib.pyplot as plt
import torch
from matplotlib.figure import Figure

from .cv_nn import RingCVNN


@dataclass(frozen=True)
class MemoryRun:
    """Sampled MATLAB timeline and the two states immediately before updates.

    times: (T,) seconds; trajectory: (N, T) complex states, post-cue at boundaries.
    recall_times: (2,) seconds; recall_states: (N, 2) complex left-limit states.
    """

    times: torch.Tensor
    trajectory: torch.Tensor
    recall_times: torch.Tensor
    recall_states: torch.Tensor


class MemoryCVNN(RingCVNN):
    """Eight-item transient memory with the MATLAB network and timing defaults.

    N=321, epsilon=45, phi=1.55, f=10 Hz. Item numbers are one-based. Each
    decoder covers floor(N/n_items) nodes; remainder nodes participate in the
    dynamics but not the readout (node 321 in the reference configuration).
    """

    def __init__(
        self,
        N: int = 321,
        device: str | torch.device = "cpu",
        *,
        n_items: int = 8,
    ) -> None:
        if not 1 <= n_items <= N:
            raise ValueError("n_items must be between 1 and N")
        super().__init__(N, epsilon=45.0, phi=1.55, device=device)
        self.n_items = n_items
        self.nodes_per_item = N // n_items

    def memory_inputs(
        self, items: tuple[int, int] = (2, 6), seed: int = 1
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return initial states for background, first item, second item, clearing.

        Targets have phase zero in the selected block, random phases outside,
        and independent amplitudes in [2, 2.5). Both are designed three seconds
        ahead. Random draws follow the MATLAB script's order, but matching seed
        numbers across PyTorch and MATLAB do not produce matching samples.
        """
        if len(items) != 2 or any(not 1 <= item <= self.n_items for item in items):
            raise ValueError(
                "Expected two one-based item numbers between 1 and n_items"
            )
        generator = torch.Generator(device=self.device).manual_seed(seed)
        designed = []
        for item in items:
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
            start = (item - 1) * self.nodes_per_item
            theta[start : start + self.nodes_per_item] = 0.0
            amplitudes = 2.0 + 0.5 * torch.rand(
                self.N, generator=generator, dtype=torch.float64, device=self.device
            )
            target = amplitudes * torch.exp(1j * theta)
            designed.append(self.design_input(target, 3.0))

        background = []
        for _ in range(2):
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
            background.append(torch.exp(1j * theta))
        return background[0], designed[0], designed[1], background[1]

    def run(
        self, items: tuple[int, int] = (2, 6), seed: int = 1, dt: float = 1e-3
    ) -> MemoryRun:
        """Simulate the reference's 1 + 3 + 3 + 1 second sequence.

        Each segment restarts its local clock and state. At 1, 4, and 7 s the
        newer segment replaces the shared boundary sample, not the entire past
        trajectory. Recall at 4- and 7- is kept separately from those post-cue
        samples. The final 8 s sample is included: default shape (321, 8001).

        dt must divide one second so cue boundaries lie on the uniform grid.
        """
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be positive and finite")
        steps_per_second = round(1.0 / dt)
        if steps_per_second < 1 or not math.isclose(
            steps_per_second * dt, 1.0, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError("dt must divide one second exactly")

        initial_states = self.memory_inputs(items, seed)
        times = (
            torch.arange(
                8 * steps_per_second + 1, dtype=torch.float64, device=self.device
            )
            * dt
        )
        trajectory = torch.empty(
            (self.N, times.numel()), dtype=torch.complex128, device=self.device
        )
        recalls = torch.empty((self.N, 2), dtype=torch.complex128, device=self.device)
        offset = 0
        for segment, (duration, initial) in enumerate(
            zip((1, 3, 3, 1), initial_states)
        ):
            steps = duration * steps_per_second
            local_times = (
                torch.arange(steps + 1, dtype=torch.float64, device=self.device) * dt
            )
            states = self.evolve(initial, local_times)
            trajectory[:, offset : offset + steps + 1] = states
            if segment in (1, 2):
                recalls[:, segment - 1] = states[:, -1]
            offset += steps

        return MemoryRun(
            times=times,
            trajectory=trajectory,
            recall_times=times[[4 * steps_per_second, 7 * steps_per_second]],
            recall_states=recalls,
        )

    def synchrony(self, state: torch.Tensor) -> torch.Tensor:
        """Return each item's phase order parameter, shape (n_items, ...).

        Accepts a state (N,) or trajectory (N, T). Amplitude does not weight the
        readout; unassigned remainder nodes are excluded, as in the paper's
        equal-sized local decoder groups.
        """
        if state.ndim not in (1, 2) or state.shape[0] != self.N:
            raise ValueError("Expected a state (N,) or trajectory (N, T)")
        assigned = state[: self.n_items * self.nodes_per_item]
        phases = torch.angle(assigned).reshape(
            self.n_items, self.nodes_per_item, *state.shape[1:]
        )
        return torch.exp(1j * phases).mean(dim=1).abs()

    def decode(self, state: torch.Tensor, threshold: float = 0.8) -> torch.Tensor:
        """Threshold each local decoder independently; do not force a winner.

        This readout follows the paper; the MATLAB memory script only plots
        phases and does not specify a threshold. Returns boolean activation(s).
        """
        return self.synchrony(state) > threshold

    def plot(self, result: MemoryRun) -> Figure:
        """Plot the reference phase timeline and local synchrony without showing."""
        seconds = result.times.detach().cpu().numpy()
        phases = torch.angle(result.trajectory).detach().cpu().numpy()
        levels = self.synchrony(result.trajectory).detach().cpu().numpy()
        figure, (phase_axes, readout_axes) = plt.subplots(
            2, 1, figsize=(11, 7), sharex=True, layout="constrained"
        )
        image = phase_axes.imshow(
            phases,
            origin="upper",
            aspect="auto",
            extent=(seconds[0], seconds[-1], self.N + 0.5, 0.5),
            interpolation="nearest",
            cmap="bone",
            vmin=-math.pi,
            vmax=math.pi,
        )
        phase_axes.set(title="Memory: phase dynamics", ylabel="nodes")
        figure.colorbar(image, ax=phase_axes, label="phase (rad)")
        recall_levels = self.synchrony(result.recall_states).detach().cpu().numpy()
        recall_times = result.recall_times.detach().cpu().numpy()
        for item, level in enumerate(levels):
            (line,) = readout_axes.plot(seconds, level, label=f"item {item + 1}")
            readout_axes.scatter(
                recall_times, recall_levels[item], color=line.get_color(), s=20
            )
        for cue in (1, 4, 7):
            phase_axes.axvline(cue, color="tab:red", linestyle="--", linewidth=0.8)
            readout_axes.axvline(cue, color="gray", linestyle="--", linewidth=0.8)
        readout_axes.set(
            title="Local synchrony (dots: immediately before updates)",
            xlabel="time (s)",
            ylabel="order parameter",
            ylim=(0, 1.05),
        )
        readout_axes.legend(ncol=4, loc="upper left")
        return figure
