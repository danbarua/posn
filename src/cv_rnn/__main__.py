"""Run the computational cv-NN examples from the repository root."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from .cv_nn_memory import MemoryCVNN
from .cv_nn_message import MessageDemo, MessageKey
from .cv_nn_plot_phase_dynamics import animate_dynamics
from .cv_nn_xor import XorCVNN
from .segmentation_demo import (
    plot_segmentation_comparison,
    run_segmentation_example,
    save_segmentation_comparison,
)


def main() -> None:
    """Entry point for ``python -m src.cv_rnn <task> [flags]``.

    Four mutually exclusive subcommands (``args.task``):

    - ``xor``: ``[--seed INT=1] [--plot]``
    - ``memory``: ``[--seed INT=1] [--plot]``
    - ``message``: ``[--seed INT=1] [--plot] [--text STR="HELLO"]
      [--frequency-hz FLOAT=10.0] [--receiver-frequency-hz FLOAT]
      [--receiver-seed INT] [--dt FLOAT=0.01]``
    - ``segmentation``: ``[--seed INT] [--plot]
      [--dataset {2shapes,3shapes,natural}=2shapes] [--image-index INT=0]
      [--n-clusters INT] [--output-dir PATH]`` — ``--seed`` defaults to
      ``None`` here (not ``1``), meaning "use the demo's own per-dataset
      seed" rather than a shared default across datasets.

    ``--seed`` and ``--plot`` are added to every subparser in the loop
    above and so are common to all four; the rest are subcommand-specific.
    Dispatch below is ``if/elif`` on ``args.task`` for the first three,
    falling through to an unconditional ``else`` for ``message`` — relies
    on argparse's ``choices`` already having rejected anything else.
    """
    parser = argparse.ArgumentParser(description="Computational cv-NN research demos")
    tasks = parser.add_subparsers(dest="task", required=True)
    task_parsers = {
        "xor": tasks.add_parser("xor", help="MATLAB-reference XOR gate"),
        "memory": tasks.add_parser("memory", help="MATLAB-reference memory task"),
        "message": tasks.add_parser(
            "message", help="framed transmission, not secure encryption"
        ),
        "segmentation": tasks.add_parser(
            "segmentation", help="paper 2-layer image segmentation demo"
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
    segmentation_parser = task_parsers["segmentation"]
    segmentation_parser.set_defaults(seed=None)
    segmentation_parser.add_argument(
        "--dataset",
        choices=("2shapes", "3shapes", "natural"),
        default="2shapes",
        help="bundled dataset in ./datasets",
    )
    segmentation_parser.add_argument(
        "--image-index",
        type=int,
        default=0,
        help="image 0, 1, or 2 within the 2shapes/3shapes dataset",
    )
    segmentation_parser.add_argument(
        "--n-clusters", type=int, help="override dataset-specific foreground segment count"
    )
    segmentation_parser.add_argument(
        "--output-dir", type=Path, help="save comparison PNG, animated GIF, inputs and metrics"
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
    elif args.task == "segmentation":
        try:
            example = run_segmentation_example(
                args.dataset, args.image_index, seed=args.seed, n_clusters=args.n_clusters
            )
        except (ValueError, FileNotFoundError) as error:
            segmentation_parser.error(str(error))
        print(
            f"{example.dataset} image {example.image_index}: MT19937 seed {example.seed}, "
            f"{int(example.mask.sum())}/{example.mask.numel()} background pixels, "
            f"{example.n_clusters} clusters"
        )
        for metric, score in example.scores().items():
            print(f"{metric}: {score:.6f}")
        if args.output_dir is not None:
            for kind, path in save_segmentation_comparison(example, args.output_dir).items():
                print(f"{kind}: {path}")
        if args.plot:
            plot_segmentation_comparison(example)
            _, animation = animate_dynamics(example.states, tuple(example.image.shape))
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
