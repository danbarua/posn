"""Numerical pulse propagation and time-blind message decoding contracts."""

import math

import numpy as np
import pytest
import torch
from scipy.linalg import expm

from src.cv_rnn import (
    ChimeraAlphabet,
    Ciphertext,
    InputEvent,
    MessageDemo,
    MessageKey,
    MessageTrace,
)


def _operator(n, frequency_hz):
    """Independent complex operator; no production matrices or Fourier helpers."""
    index = np.arange(n)
    distance = np.abs(index[:, None] - index[None, :]).astype(float)
    distance = np.minimum(distance, n - distance)
    np.fill_diagonal(distance, np.inf)
    adjacency = 1 / distance
    adjacency /= adjacency[0].sum()
    return 45 * np.exp(-1.55j) * adjacency + 2j * np.pi * frequency_hz * np.eye(n)


def _reference_state(operator, key, ciphertext, time):
    state = expm(operator * time) @ key.x0.numpy()
    for event in ciphertext.events:
        if event.time <= time:
            state += expm(operator * (time - event.time)) @ event.impulse.numpy()
    return state


def _incoherent_states(alphabet, samples):
    phases = (
        2
        * math.pi
        * torch.arange(alphabet.nodes_per_symbol, dtype=torch.float64)
        / alphabet.nodes_per_symbol
    )
    return (
        torch.exp(1j * phases).repeat(len(alphabet.symbols))[:, None].repeat(1, samples)
    )


def test_receiver_matches_independent_impulse_dynamics_and_frame_boundaries():
    alphabet = ChimeraAlphabet(symbols="AB ", nodes_per_symbol=4)
    receiver = MessageDemo(alphabet)
    rng = np.random.default_rng(2)
    initial = (1.5 + rng.random(alphabet.N)) * np.exp(
        2j * np.pi * rng.random(alphabet.N)
    )
    key = MessageKey(7.3, torch.from_numpy(initial))
    events = tuple(
        InputEvent(
            time,
            torch.from_numpy(
                rng.normal(size=alphabet.N) + 1j * rng.normal(size=alphabet.N)
            ),
        )
        for time in (0.25, 1.3)
    )
    ciphertext = Ciphertext(events, end_time=2.45)
    trace = receiver.receive(ciphertext, key, dt=0.3)
    # Delayed first input, exact post-input samples, and dt-spaced half-open
    # frames: linspace stretching or retaining the pre-input boundary is wrong.
    expected_frames = ([0.25, 0.55, 0.85, 1.15], [1.3, 1.6, 1.9, 2.2])
    assert len(trace.frames) == len(expected_frames)
    for frame, expected in zip(trace.frames, expected_frames):
        np.testing.assert_allclose(
            trace.times[frame].numpy(), expected, rtol=0, atol=1e-15
        )
    operator = _operator(alphabet.N, key.frequency_hz)
    expected = np.column_stack(
        [
            _reference_state(operator, key, ciphertext, time)
            for time in trace.times.tolist()
        ]
    )
    np.testing.assert_allclose(
        trace.trajectory.numpy(), expected, rtol=2e-11, atol=2e-11
    )


def test_encoded_pulses_recover_targets_at_private_delays_under_independent_dynamics():
    alphabet = ChimeraAlphabet(symbols="AB ", nodes_per_symbol=4)
    sender = MessageDemo(alphabet)
    key = sender.make_key(seed=4, frequency_hz=7.3)
    delays = (0.47, 1.13)
    ciphertext = sender.encode(
        "BA", key, seed=8, frame_duration=1.5, target_delays=delays
    )
    operator = _operator(alphabet.N, key.frequency_hz)
    # Pulse times are public frame starts, not the private delays. Test their
    # resulting physical states rather than inspecting metadata field names.
    for index, symbol in enumerate("BA"):
        target = _reference_state(
            operator, key, ciphertext, index * 1.5 + delays[index]
        )
        start = alphabet.symbols.index(symbol) * alphabet.nodes_per_symbol
        np.testing.assert_allclose(
            np.angle(target[start : start + alphabet.nodes_per_symbol]), 0, atol=2e-11
        )
        assert np.all(np.abs(target) >= 2.0)
        assert np.all(np.abs(target) < 2.5)


def test_separate_receiver_recovers_repeated_letters_and_surrounding_spaces():
    sender = MessageDemo()
    key = sender.make_key(seed=7)
    text = " HELLO "
    encoded = sender.encode(text, key, seed=1)
    transported = Ciphertext(
        tuple(
            InputEvent(event.time, torch.from_numpy(event.impulse.numpy().copy()))
            for event in encoded.events
        ),
        encoded.end_time,
    )
    receiver = MessageDemo()
    decoded = receiver.decode_frames(receiver.receive(transported, key))
    assert [item.symbol for item in decoded] == list(text)


def test_multiple_peaks_in_one_frame_do_not_erase_repeated_letters_in_later_frames():
    alphabet = ChimeraAlphabet(symbols="HL ", nodes_per_symbol=4)
    receiver = MessageDemo(alphabet)
    states = _incoherent_states(alphabet, 9)
    states[0:4, 1] = 1.0
    states[0:4, 3] = 1.0  # a second H peak, still the same frame
    states[4:8, 5] = 1.0
    states[4:8, 8] = 1.0  # an actual repeated L in a separate frame
    trace = MessageTrace(
        torch.arange(9, dtype=torch.float64),
        states,
        (slice(0, 5), slice(5, 7), slice(7, 9)),
    )
    assert [item.symbol for item in receiver.decode_frames(trace)] == list("HLL")


def test_space_is_distinct_from_an_ambiguous_global_synchrony_erasure():
    alphabet = ChimeraAlphabet(symbols="AB ", nodes_per_symbol=4)
    receiver = MessageDemo(alphabet)
    states = _incoherent_states(alphabet, 4)
    states[8:12, 1] = 1.0
    states[:, 2:] = 1.0  # every block coherent: no unique symbol
    trace = MessageTrace(
        torch.arange(4, dtype=torch.float64), states, (slice(0, 2), slice(2, 4))
    )
    assert [item.symbol for item in receiver.decode_frames(trace)] == [" ", None]


def test_competing_peak_at_another_time_does_not_reject_an_unambiguous_symbol():
    alphabet = ChimeraAlphabet(symbols="AB ", nodes_per_symbol=4)
    receiver = MessageDemo(alphabet)
    states = _incoherent_states(alphabet, 2)
    states[:4, 0] = 1.0
    angle = math.acos(0.95)
    states[4:8, 1] = torch.exp(
        1j * torch.tensor([angle, -angle, angle, -angle], dtype=torch.float64)
    )
    trace = MessageTrace(
        torch.tensor([0.0, 1.0], dtype=torch.float64), states, (slice(0, 2),)
    )
    (decoded,) = receiver.decode_frames(trace)
    assert decoded.symbol == "A"
    assert decoded.time == 0.0
    assert decoded.runner_up == pytest.approx(0.0, abs=1e-12)


def test_different_frequency_key_can_have_identical_synchrony_on_regular_frames():
    # Explicit security counterexample, not an assertion that wrong keys fail.
    demo = MessageDemo()
    key = demo.make_key(seed=7, frequency_hz=10.0)
    ciphertext = demo.encode("HI", key, seed=1, frame_duration=3.0)
    correct = demo.receive(ciphertext, key)
    alias = demo.receive(ciphertext, MessageKey(9.0, key.x0))
    torch.testing.assert_close(
        demo.alphabet.synchrony(alias.trajectory),
        demo.alphabet.synchrony(correct.trajectory),
        rtol=1e-8,
        atol=1e-8,
    )
    assert [item.symbol for item in demo.decode_frames(correct)] == list("HI")
    assert [item.symbol for item in demo.decode_frames(alias)] == list("HI")


def test_unsupported_text_is_not_silently_normalized():
    demo = MessageDemo(ChimeraAlphabet(symbols="AB ", nodes_per_symbol=4))
    with pytest.raises(ValueError):
        demo.encode("ab", demo.make_key())


def test_target_at_the_next_input_boundary_is_rejected():
    demo = MessageDemo(ChimeraAlphabet(symbols="AB ", nodes_per_symbol=4))
    with pytest.raises(ValueError):
        demo.encode("AB", demo.make_key(), target_delays=[1.0, 3.0])


def test_receiver_rejects_unordered_inputs_instead_of_reordering_them():
    demo = MessageDemo(ChimeraAlphabet(symbols="AB ", nodes_per_symbol=4))
    impulse = torch.ones(demo.alphabet.N, dtype=torch.complex128)
    ciphertext = Ciphertext((InputEvent(2.0, impulse), InputEvent(1.0, impulse)), 3.0)
    with pytest.raises(ValueError):
        demo.receive(ciphertext, demo.make_key())
