"""Additive, framed dynamics-based message transmission: NOT secure encryption.

Inspired by the computational cv-NN paper, not an exact Figure 4 reproduction.
The public alphabet and one-symbol-per-input-frame convention are design choices.
Ciphertext contains pulse vectors/times and an observation endpoint, never targets
or their private delays. Frequency aliases and floating-point drift remain real
limitations; there are no confidentiality or authentication guarantees.
"""

import math
from dataclasses import dataclass
from typing import Sequence

import matplotlib.pyplot as plt
import torch
from matplotlib.figure import Figure

from .cv_nn import RingCVNN


@dataclass(frozen=True)
class ChimeraAlphabet:
    """Public contiguous node blocks; a space is a symbol, not an erasure."""

    symbols: str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ "
    nodes_per_symbol: int = 16

    def __post_init__(self) -> None:
        if len(self.symbols) < 2 or len(set(self.symbols)) != len(self.symbols):
            raise ValueError("Alphabet must contain at least two distinct symbols")
        if self.nodes_per_symbol < 2:
            raise ValueError("Each symbol needs at least two nodes")

    @property
    def N(self) -> int:
        return len(self.symbols) * self.nodes_per_symbol

    def target(
        self,
        symbol: str,
        generator: torch.Generator,
        device: str | torch.device = "cpu",
    ) -> torch.Tensor:
        """Complex target: coherent selected block, random outside, amplitudes [2,2.5)."""
        if len(symbol) != 1 or symbol not in self.symbols:
            raise ValueError(f"Unsupported symbol: {symbol!r}")
        phases = (
            2
            * math.pi
            * (
                torch.rand(
                    self.N, generator=generator, dtype=torch.float64, device=device
                )
                - 0.5
            )
        )
        start = self.symbols.index(symbol) * self.nodes_per_symbol
        phases[start : start + self.nodes_per_symbol] = 0.0
        amplitudes = 2.0 + 0.5 * torch.rand(
            self.N, generator=generator, dtype=torch.float64, device=device
        )
        return amplitudes * torch.exp(1j * phases)

    def synchrony(self, state: torch.Tensor) -> torch.Tensor:
        """Phase-only local order parameters, shape (symbols,) or (symbols, times)."""
        if state.ndim not in (1, 2) or state.shape[0] != self.N:
            raise ValueError("Expected a state (N,) or trajectory (N, T)")
        phases = torch.angle(state).reshape(
            len(self.symbols), self.nodes_per_symbol, *state.shape[1:]
        )
        return torch.exp(1j * phases).mean(dim=1).abs()


@dataclass(frozen=True)
class MessageKey:
    """Shared frequency in Hz and complex initial state; not a secure cryptographic key."""

    frequency_hz: float
    x0: torch.Tensor


@dataclass(frozen=True)
class InputEvent:
    """Complex state correction to add at a public time in seconds."""

    time: float
    impulse: torch.Tensor


@dataclass(frozen=True)
class Ciphertext:
    """Public transmission only: ordered additive pulses and the observation endpoint."""

    events: tuple[InputEvent, ...]
    end_time: float


@dataclass(frozen=True)
class MessageTrace:
    """Receiver samples: global times (T,), complex states (N,T), half-open frame slices."""

    times: torch.Tensor
    trajectory: torch.Tensor
    frames: tuple[slice, ...]


@dataclass(frozen=True)
class DecodedSymbol:
    """Observed peak and simultaneous competitor; symbol=None means an erasure."""

    symbol: str | None
    time: float
    score: float
    runner_up: float


class MessageDemo:
    """Public network used by independent sender and receiver instances.

    Defaults borrow the memory task's coupling parameters, not an unavailable
    reference encryption implementation. No sender state is retained by encode.
    """

    def __init__(
        self,
        alphabet: ChimeraAlphabet | None = None,
        device: str | torch.device = "cpu",
        *,
        epsilon: float = 45.0,
        phi: float = 1.55,
    ) -> None:
        self.alphabet = alphabet if alphabet is not None else ChimeraAlphabet()
        self.device = torch.device(device)
        # All keys share K. Their scalar frequency commutes with K, so reuse one
        # zero-frequency Fourier propagator rather than rebuilding dense matrices.
        self._dynamics = RingCVNN(
            self.alphabet.N, epsilon, phi, self.device, frequency_hz=0.0
        )

    def _key_state(self, key: MessageKey) -> torch.Tensor:
        if not math.isfinite(key.frequency_hz):
            raise ValueError("Key frequency must be finite")
        state = torch.as_tensor(key.x0, dtype=torch.complex128, device=self.device)
        if state.shape != (self.alphabet.N,) or not torch.isfinite(state).all():
            raise ValueError("Key initial state must be finite with shape (N,)")
        return state

    def _flow(
        self,
        state: torch.Tensor,
        times: torch.Tensor | list[float],
        frequency_hz: float,
    ) -> torch.Tensor:
        times = torch.as_tensor(times, dtype=torch.float64, device=self.device)
        rotation = torch.exp(1j * 2 * math.pi * frequency_hz * times)
        return self._dynamics.evolve(state, times).mul_(rotation)

    def make_key(self, seed: int = 7, frequency_hz: float = 10.0) -> MessageKey:
        """Seeded unit phases for reproducible experiments, not cryptographic randomness."""
        if not math.isfinite(frequency_hz):
            raise ValueError("Key frequency must be finite")
        generator = torch.Generator(device=self.device).manual_seed(seed)
        phases = (
            2
            * math.pi
            * torch.rand(
                self.alphabet.N,
                generator=generator,
                dtype=torch.float64,
                device=self.device,
            )
        )
        return MessageKey(frequency_hz, torch.exp(1j * phases))

    def encode(
        self,
        message: str,
        key: MessageKey,
        *,
        seed: int = 1,
        frame_duration: float = 3.0,
        target_delays: Sequence[float] | None = None,
    ) -> Ciphertext:
        """Design additive pulses I_j = D(-delay_j) target_j - x(t_j^-).

        Events occur at j*frame_duration. Private delays are uniform in
        [0.35,0.50)*frame_duration unless supplied explicitly. Target generation
        uses a local RNG; neither delays nor its seed are transmitted. Text must
        be nonempty and supported exactly: no uppercasing or whitespace stripping.
        """
        state = self._key_state(key)
        if not message or any(
            symbol not in self.alphabet.symbols for symbol in message
        ):
            raise ValueError(
                f"Message must be nonempty and use only {self.alphabet.symbols!r}"
            )
        if not math.isfinite(frame_duration) or frame_duration <= 0:
            raise ValueError("frame_duration must be positive and finite")
        if target_delays is not None:
            if len(target_delays) != len(message) or any(
                not math.isfinite(delay) or not 0 < delay < frame_duration
                for delay in target_delays
            ):
                raise ValueError("Provide one finite delay strictly inside each frame")

        generator = torch.Generator(device=self.device).manual_seed(seed)
        events = []
        previous_time = 0.0
        for index, symbol in enumerate(message):
            at = index * frame_duration
            if at > previous_time:
                state = self._flow(state, [at - previous_time], key.frequency_hz)[:, 0]
            delay = (
                target_delays[index]
                if target_delays is not None
                else frame_duration
                * (
                    0.35
                    + 0.15
                    * torch.rand(
                        (), generator=generator, dtype=torch.float64, device=self.device
                    ).item()
                )
            )
            target = self.alphabet.target(symbol, generator, self.device)
            required = self._flow(target, [-delay], key.frequency_hz)[:, 0]
            impulse = required - state
            events.append(InputEvent(at, impulse))
            state = state + impulse
            previous_time = at
        return Ciphertext(tuple(events), len(message) * frame_duration)

    def receive(
        self, ciphertext: Ciphertext, key: MessageKey, *, dt: float = 0.01
    ) -> MessageTrace:
        """Apply received corrections to the running state; never replace it.

        Samples are dt-spaced within each half-open frame. Propagation to the
        exact frame end carries state into the next input, even for non-dividing
        dt. A delayed first pulse evolves x0 from time zero before applying it.
        No plaintext, private target times, or sender object are consulted.
        """
        state = self._key_state(key)
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be positive and finite")
        if not ciphertext.events or not math.isfinite(ciphertext.end_time):
            raise ValueError(
                "Ciphertext needs events and a finite observation endpoint"
            )
        previous_time = -math.inf
        impulses = []
        for event in ciphertext.events:
            if (
                not math.isfinite(event.time)
                or event.time < 0
                or event.time <= previous_time
            ):
                raise ValueError(
                    "Pulse times must be finite, nonnegative and strictly increasing"
                )
            impulse = torch.as_tensor(
                event.impulse, dtype=torch.complex128, device=self.device
            )
            if impulse.shape != (self.alphabet.N,) or not torch.isfinite(impulse).all():
                raise ValueError("Each impulse must be finite with shape (N,)")
            impulses.append(impulse)
            previous_time = event.time
        if ciphertext.end_time <= previous_time:
            raise ValueError("Observation endpoint must follow the final pulse")

        boundaries = [event.time for event in ciphertext.events] + [ciphertext.end_time]
        local_times = [
            torch.arange(0, stop - start, dt, dtype=torch.float64, device=self.device)
            for start, stop in zip(boundaries[:-1], boundaries[1:])
        ]
        times = torch.cat(
            [start + local for start, local in zip(boundaries, local_times)]
        )
        trajectory = torch.empty(
            (self.alphabet.N, times.numel()), dtype=torch.complex128, device=self.device
        )
        if boundaries[0] > 0:
            state = self._flow(state, [boundaries[0]], key.frequency_hz)[:, 0]
        frames = []
        offset = 0
        for index, (local, impulse) in enumerate(zip(local_times, impulses)):
            duration = boundaries[index + 1] - boundaries[index]
            sample_times = torch.cat((local, local.new_tensor([duration])))
            states = self._flow(state + impulse, sample_times, key.frequency_hz)
            if not torch.isfinite(states).all():
                raise ValueError(
                    "Non-finite dynamics; reduce transmission duration or coupling"
                )
            stop = offset + local.numel()
            trajectory[:, offset:stop] = states[:, :-1]
            frames.append(slice(offset, stop))
            state = states[:, -1]
            offset = stop
        return MessageTrace(times, trajectory, tuple(frames))

    def decode_frames(
        self, trace: MessageTrace, *, threshold: float = 0.9, margin: float = 0.15
    ) -> tuple[DecodedSymbol, ...]:
        """One observed peak per public frame, with no deduplication across frames.

        The runner-up is measured at the winning peak's time, not at another
        symbol's unrelated maximum. Insufficient or ambiguous coherence yields
        None, never a guessed letter or a space.
        """
        if not 0 <= threshold <= 1 or not 0 <= margin <= 1:
            raise ValueError("threshold and margin must lie in [0,1]")
        if trace.times.ndim != 1 or trace.trajectory.shape != (
            self.alphabet.N,
            trace.times.numel(),
        ):
            raise ValueError("Trace times and trajectory dimensions must agree")
        levels = self.alphabet.synchrony(trace.trajectory)
        if not torch.isfinite(levels).all() or not torch.isfinite(trace.times).all():
            raise ValueError("Cannot decode non-finite receiver samples")
        results = []
        for frame in trace.frames:
            if (
                frame.step not in (None, 1)
                or frame.start is None
                or frame.stop is None
                or not (0 <= frame.start < frame.stop <= trace.times.numel())
            ):
                raise ValueError(
                    "Each frame must be a nonempty slice of receiver samples"
                )
            frame_levels = levels[:, frame]
            winner, sample = divmod(frame_levels.argmax().item(), frame_levels.shape[1])
            score = frame_levels[winner, sample].item()
            runner_up = frame_levels[:, sample].topk(2).values[1].item()
            symbol = (
                self.alphabet.symbols[winner]
                if score >= threshold and score - runner_up >= margin
                else None
            )
            results.append(
                DecodedSymbol(
                    symbol, trace.times[frame.start + sample].item(), score, runner_up
                )
            )
        return tuple(results)

    def plot(self, trace: MessageTrace, decoded: tuple[DecodedSymbol, ...]) -> Figure:
        """Receiver-only phase/coherence heatmaps, public frames, and observed peaks."""
        seconds = trace.times.detach().cpu().numpy()
        phases = torch.angle(trace.trajectory).detach().cpu().numpy()
        levels = self.alphabet.synchrony(trace.trajectory).detach().cpu().numpy()
        figure, (phase_axes, symbol_axes) = plt.subplots(
            2, 1, figsize=(12, 9), sharex=True, layout="constrained"
        )
        image = phase_axes.pcolormesh(
            seconds,
            range(1, self.alphabet.N + 1),
            phases,
            shading="nearest",
            cmap="bone",
            vmin=-math.pi,
            vmax=math.pi,
            rasterized=True,
        )
        phase_axes.invert_yaxis()
        phase_axes.set(title="Receiver phase dynamics", ylabel="nodes")
        figure.colorbar(image, ax=phase_axes, label="phase (rad)")
        image = symbol_axes.pcolormesh(
            seconds,
            range(len(self.alphabet.symbols)),
            levels,
            shading="nearest",
            cmap="magma",
            vmin=0,
            vmax=1,
            rasterized=True,
        )
        symbol_axes.invert_yaxis()
        symbol_axes.set(
            title="Local synchrony and observed peaks",
            xlabel="time (s)",
            ylabel="symbol",
        )
        symbol_axes.set_yticks(range(len(self.alphabet.symbols)))
        symbol_axes.set_yticklabels(
            ["SPACE" if symbol == " " else symbol for symbol in self.alphabet.symbols],
            fontsize=8,
        )
        figure.colorbar(image, ax=symbol_axes, label="order parameter")
        for frame in trace.frames:
            for axes in (phase_axes, symbol_axes):
                axes.axvline(
                    seconds[frame.start], color="gray", linestyle="--", linewidth=0.7
                )
        for item in decoded:
            if item.symbol is None:
                symbol_axes.annotate(
                    "ERASURE",
                    (item.time, 0.98),
                    xycoords=("data", "axes fraction"),
                    xytext=(0, 0),
                    textcoords="offset points",
                    ha="center",
                    va="top",
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85},
                    fontsize=8,
                )
            else:
                index = self.alphabet.symbols.index(item.symbol)
                symbol_axes.scatter(
                    item.time, index, s=55, facecolors="none", edgecolors="cyan"
                )
        text = "".join("?" if item.symbol is None else item.symbol for item in decoded)
        figure.suptitle(
            f"Decoded: {text!r} — research message demo, NOT secure encryption"
        )
        return figure
