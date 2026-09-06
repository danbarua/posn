"""Run the computational cv-NN examples from the repository root."""

import argparse

import matplotlib.pyplot as plt

from .cv_nn_memory import MemoryCVNN
from .cv_nn_xor import XorCVNN


def main() -> None:
    parser = argparse.ArgumentParser(description="MATLAB-reference cv-NN demos")
    parser.add_argument("task", choices=("xor", "memory"))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--plot", action="store_true", help="show phase dynamics")
    args = parser.parse_args()

    if args.task == "xor":
        table = XorCVNN().truth_table(seed=args.seed, plot=args.plot)
        print("XOR truth table (shared-cluster synchrony at 3 s)\n X  Y | f(X,Y)")
        for (x, y), output in sorted(table.items()):
            print(f" {x}  {y} |   {output}")
    else:
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


if __name__ == "__main__":
    main()
