"""Run the computational cv-NN examples from the repository root."""

import argparse

import matplotlib.pyplot as plt

from .cv_nn_memory import MemoryCVNN
from .cv_nn_message import MessageDemo, MessageKey
from .cv_nn_xor import XorCVNN


def main() -> None:
    parser = argparse.ArgumentParser(description="Computational cv-NN research demos")
    tasks = parser.add_subparsers(dest="task", required=True)
    task_parsers = {
        "xor": tasks.add_parser("xor", help="MATLAB-reference XOR gate"),
        "memory": tasks.add_parser("memory", help="MATLAB-reference memory task"),
        "message": tasks.add_parser(
            "message", help="framed transmission, not secure encryption"
        ),
    }
    for task_parser in task_parsers.values():
        task_parser.add_argument("--seed", type=int, default=1)
        task_parser.add_argument(
            "--plot", action="store_true", help="show phase dynamics"
        )
    message_parser = task_parsers["message"]
    message_parser.add_argument(
        "--text", default="HELLO", help="nonempty uppercase A-Z and spaces"
    )
    message_parser.add_argument(
        "--frequency-hz", type=float, default=10.0, help="sender key frequency"
    )
    message_parser.add_argument(
        "--receiver-frequency-hz",
        type=float,
        help="override receiver frequency for experiments",
    )
    message_parser.add_argument(
        "--receiver-seed", type=int, help="use a different receiver initial state"
    )
    message_parser.add_argument(
        "--dt", type=float, default=0.01, help="receiver sampling interval in seconds"
    )
    args = parser.parse_args()

    if args.task == "xor":
        table = XorCVNN().truth_table(seed=args.seed, plot=args.plot)
        print("XOR truth table (shared-cluster synchrony at 3 s)\n X  Y | f(X,Y)")
        for (x, y), output in sorted(table.items()):
            print(f" {x}  {y} |   {output}")
    elif args.task == "memory":
        memory = MemoryCVNN()
        result = memory.run(seed=args.seed)
        levels = memory.synchrony(result.recall_states)
        decoded = memory.decode(result.recall_states)
        print("Memory recall immediately before updates (threshold 0.8)")
        for column, time in enumerate(result.recall_times.tolist()):
            active = (decoded[:, column].nonzero(as_tuple=True)[0] + 1).tolist()
            scores = ", ".join(f"{value:.6f}" for value in levels[:, column].tolist())
            print(f"t={time:g} s, before update: items {active}; synchrony [{scores}]")
        if args.plot:
            memory.plot(result)
            plt.show()
    else:
        try:
            sender = MessageDemo()
            key = sender.make_key(seed=args.seed, frequency_hz=args.frequency_hz)
            ciphertext = sender.encode(args.text, key, seed=args.seed)
            receiver = MessageDemo()
            receiver_state = (
                key.x0
                if args.receiver_seed is None
                else receiver.make_key(seed=args.receiver_seed).x0
            )
            receiver_key = MessageKey(
                frequency_hz=(
                    key.frequency_hz
                    if args.receiver_frequency_hz is None
                    else args.receiver_frequency_hz
                ),
                x0=receiver_state,
            )
            trace = receiver.receive(ciphertext, receiver_key, dt=args.dt)
            symbols = receiver.decode_frames(trace)
        except ValueError as error:
            message_parser.error(str(error))

        received = "".join(
            "?" if item.symbol is None else item.symbol for item in symbols
        )
        pulse_bytes = sum(
            event.impulse.numel() * event.impulse.element_size()
            for event in ciphertext.events
        )
        print("Research message transmission demo — NOT secure encryption")
        print(f"Sent:     {args.text!r}")
        print(f"Received: {received!r}")
        print(
            f"Pulse data: {len(ciphertext.events)} vectors, {pulse_bytes} bytes (excluding timing/metadata)"
        )
        print(
            f"Key frequencies: sender {key.frequency_hz:g} Hz, receiver {receiver_key.frequency_hz:g} Hz"
        )
        for frame, item in enumerate(symbols, start=1):
            label = "<erasure>" if item.symbol is None else repr(item.symbol)
            print(
                f"Frame {frame}: {label} at {item.time:.3f} s; R={item.score:.6f}, runner-up={item.runner_up:.6f}"
            )
        if args.plot:
            receiver.plot(trace, symbols)
            plt.show()


if __name__ == "__main__":
    main()
