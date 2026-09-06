"""Run the computational cv-NN examples from the repository root."""

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import scipy.io as sio
import torch

from defns import DATA_DIR
from .cv_nn_memory import MemoryCVNN
from .cv_nn_message import MessageDemo, MessageKey
from .cv_nn_plot_phase_dynamics import animate_dynamics
from .cv_nn_xor import XorCVNN
from .cv_rnn_plot_spectral_clustering import plot_spectral_clustering
from .cv_rnn_segmentation import run_2layer_torch, spatiotemporal_segmentation_torch


def main() -> None:
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
        "--n-clusters", type=int, default=2, help="requested foreground segment count"
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
        dataset_files = {
            "2shapes": "2shapes.mat",
            "3shapes": "3shapes.mat",
            "natural": "natural_image.mat",
        }
        mat = sio.loadmat(Path(DATA_DIR) / dataset_files[args.dataset])
        if args.dataset == "natural":
            image = torch.from_numpy(mat["im"]).to(torch.float64)
        else:
            image = torch.from_numpy(
                mat["images"][:, :, args.image_index].astype("float64")
            )
        generator = torch.Generator().manual_seed(args.seed)
        states, mask = run_2layer_torch(image, generator=generator, dtype=torch.float64)
        cluster_map, _, _, _, projection = spatiotemporal_segmentation_torch(
            states, image, mask, n_clusters=args.n_clusters, nt_mask=60
        )
        background = int(mask.sum())
        print(
            f"Segmented {args.dataset} image {args.image_index}: "
            f"{background} background px of {mask.numel()}, "
            f"{args.n_clusters} requested clusters"
        )
        if args.plot:
            mask_image = mask.cpu().numpy().reshape(image.shape, order="F")
            fig, axes = plt.subplots(1, 3, figsize=(9, 3))
            axes[0].imshow(image.cpu().numpy(), cmap="gray")
            axes[0].set_title("input")
            axes[1].imshow(mask_image, cmap="gray")
            axes[1].set_title("mask")
            segments = axes[2].imshow(
                cluster_map.cpu().numpy(), cmap="tab10", vmin=-1, vmax=9
            )
            axes[2].set_title("segments")
            for axis in axes:
                axis.axis("off")
            fig.colorbar(segments, ax=axes.ravel().tolist(), shrink=0.6)
            _, animation = animate_dynamics(states, tuple(image.shape))
            plot_spectral_clustering(projection, states[~mask], phase_iter=60)
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
