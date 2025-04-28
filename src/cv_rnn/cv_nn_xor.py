import math
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

"""
xor_cv_nn.py
-------------------------------------------
A *functional* XOR gate realised with a
complex-valued neural network
-------------------------------------------
The approaches are conceptually similar but implemented differently.
The MATLAB version calculates inputs that will evolve into specific target patterns,
while the Python version directly constructs chimera patterns.

The Python implementation is a faithful adaptation of the core concepts from the original MATLAB code, 
but with several improvements:

- More compact implementation: Encapsulates the entire XOR gate in a clean class structure
- Explicit readout mechanism: Clearly defines how synchronization maps to logical outputs
- Hardware acceleration: Leverages PyTorch for potential GPU acceleration
- Parameterization: Makes it easier to adjust hyperparameters

Both implementations demonstrate the fundamental concept from the paper: that complex-valued linear dynamics with
phase delays can generate sophisticated spatiotemporal patterns capable of performing logical operations like XOR.

The Python version is more practical for integration with modern ML frameworks while preserving the mathematical
principles of the original implementation.
"""


class XorCVNN:
    """
    Simple 1-D ring CV-NN that implements XOR by
    measuring phase-synchrony in two independent regions.
    """

    def __init__(self, N: int = 200, device: str | torch.device = "cpu") -> None:
        """
        Initialise the Complex-Valued-Neural-Net with a ring topology and a fixed number of nodes.
        Args:
            N (int): number of nodes in the ring. (default: 200).
            device (str|torch.device): device on which to run the simulation. (default: "cpu").
        """
        self.N = N
        self.device = torch.device(device)

        # --- build once, reuse everywhere ---------------------------------

        self._K = self._powerlaw_connectivity(alpha=1.0, exponent=1.0)
        self._M = self._system_matrix(self.K)  # (N,N) complex64

    @property
    def K(self) -> torch.Tensor:
        """
        The matrix `K` contains information about the connectivity
        pattern, the coupling strength, and the phase-delay in the interaction term.

        Specifically, `K = ϵe −iϕ A`, where `ϵ` is the coupling strength and `ϕ` is a phase-delay.
        """
        return self._K

    @property
    def M(self) -> torch.Tensor:
        """
        The matrix `M` is the linear operator that describes the dynamics of the CV-NN.
        """
        return self._M

    # ------------------------------------------------------------------ #
    # 1.  Connectivity & system matrix                                   #
    # ------------------------------------------------------------------ #
    def _ring_distance(self) -> torch.Tensor:
        """

        Returns:
            tensor: (N,N) float, with self-loops. 0 for diagonal.
        """
        idx = torch.arange(self.N, device=self.device)
        d = torch.minimum(
            (idx.unsqueeze(0) - idx.unsqueeze(1)).abs(),
            self.N - (idx.unsqueeze(0) - idx.unsqueeze(1)).abs(),
        ).to(torch.float32)
        d.fill_diagonal_(float("inf"))
        return d  # (N,N) float

    def _powerlaw_connectivity(self, alpha: float, exponent: float) -> torch.Tensor:
        """
        Row-normalised power-law kernel on a ring.

        We consider the nodes in the cv-NN to be coupled in a one-dimensional ring with periodic boundary conditions
        where the connection weight decays as a power-law with distance between the two nodes.

        Args:
            alpha (float):
            exponent (float):

        Returns:
            tensor: (N,N) float32, row-stochastic, with self-loops. 0 for diagonal.
        """
        d = self._ring_distance()
        W = alpha / d.pow(exponent)  # 1 / d^γ
        W[torch.isinf(W)] = 0.0  # remove self-loops
        W = W / W.sum(dim=1, keepdim=True)  # stochastic rows
        return W.to(torch.float32)  # (N,N) float32

    def _system_matrix(
        self,
        K: torch.Tensor,
        omega_hz: float = 10.0,
        epsilon: float = 50.0,
        phi: float = math.pi / 2,
    ) -> torch.Tensor:
        """
        Linear operator `M = iω + εe^{-iφ}K

        Args:
            K (tensor): (N,N) float32,
              The matrix `K` contains information about the connectivity pattern,
              the coupling strength, and the phase-delay in the interaction term.
            omega_hz (float, optional): (default: 10.0 Hz)):
            epsilon (float, optional): Coupling Strength (default: 50.0):
            phi (float): The phase delay parameter (default: 1/2 pi).

        Returns:

        """
        ω = torch.full((self.N,), 2 * math.pi * omega_hz, device=self.device)
        diag = 1j * ω
        phase_shift = torch.tensor(-1j * phi, dtype=torch.complex64, device=self.device)
        return torch.diag(diag).to(torch.complex64) + epsilon * torch.exp(
            phase_shift
        ) * K.to(torch.complex64)

    # ------------------------------------------------------------------ #
    # 2.  Helpers                                                        #
    # ------------------------------------------------------------------ #
    def _chimera_input(self, center: int, width: int) -> torch.Tensor:
        """
        Creates chimera inputs directly through spatial phase patterns.

        Region *center ± width* is phase-zero; elsewhere phase varies linearly.
        Returns a vector of complex phases (|x| = 1).

        Original MATLAB:

        Args:
            center (int): center of the region.
            width (int): width of the region.
        Returns:
            tensor (N,): (complex) phases of the input vector.
        """
        idx = torch.arange(self.N, device=self.device)
        dist = torch.minimum((idx - center).abs(), self.N - (idx - center).abs())
        same = dist < width
        phases = torch.zeros(self.N, device=self.device)
        phases[~same] = 2 * math.pi * idx[~same] / self.N
        return torch.exp(1j * phases).to(torch.complex64)

    @staticmethod
    def _sync_level(phases: torch.Tensor) -> float:
        """
        Kuramoto order parameter |⟨e^{iθ}⟩|.

        `phases` is a real-valued 1-D tensor (radians).

        Args:
            phases (tensor): (N,) phases of the input vector.
        Returns:
            float: Kuramoto order parameter.
        """
        return torch.abs(torch.mean(torch.exp(1j * phases))).item()

    # ------------------------------------------------------------------ #
    # 3.  Core simulation                                                #
    # ------------------------------------------------------------------ #
    def _run_iteratively(
        self, x0: torch.Tensor, nt: int = 200, dt: float = 1e-3
    ) -> torch.Tensor:
        """
        Linear recurrence  x[t+1] = M x[t].

        Returns (N,nt) complex.

        Original MATLAB: Uses eigendecomposition for exact simulation with closed-form solution.
        Python: Uses iterative matrix multiplication.

        The MATLAB implementation leverages the exact solution more explicitly, while the Python implementation uses an
        iterative approach that's equivalent but more straightforward to implement.

        Args:
            x0 (tensor): (N,) initial state.
            nt (int): number of timesteps. (default: 200)
            dt (float): timestep. (default: 1e-3).

        Returns:
            tensor: (N,nt) complex, trajectory.
        """
        traj = torch.empty((self.N, nt), dtype=torch.complex64, device=self.device)
        traj[:, 0] = x0
        x = x0
        for t in range(1, nt):
            if torch.isnan(x).any():
                print(f"NaN detected at timestep {t}")
                break
            x = torch.exp(self.M * dt) @ x
            traj[:, t] = x
            if torch.isinf(x).any():
                print(f"Inf detected at timestep {t}")
                break
        return traj

    def _run_exactly(
        self, x0: torch.Tensor, nt: int = 200, dt: float = 1e-3
    ) -> torch.Tensor:
        """
        Exact simulation of the CV-NN using the closed-form solution.

        This function computes the trajectory x(t) = exp(M*t) x0 exactly.
        The eigen decomposition of self.M is computed:

            self.M = v diag(λ) v⁻¹,

        and for each time t we evaluate:

            x(t) = v diag(exp(λ * t)) v⁻¹ x0.

        Since our connectivity matrix is well-behaved, and the eigenvectors are unitary,
        we can use v.conj().T as the inverse.

        Args:
            x0 (torch.Tensor): (N,) initial state, complex64.
            nt (int): number of timesteps.
            dt (float): time step.

        Returns:
            torch.Tensor: (N, nt) trajectory of the system.
        """
        traj = torch.empty((self.N, nt), dtype=torch.complex64, device=self.device)
        traj[:, 0] = x0

        # Compute eigen decomposition of self.M:
        eigenvalues, eigenvectors = torch.linalg.eig(self.M)
        # In our case the eigenvectors are unitary (or nearly so) so that the
        # inverse can be obtained by the conjugate transpose.
        v_inv = eigenvectors.conj().T

        # Compute the state at each time using the closed-form solution:
        for t in range(1, nt):
            time = t * dt
            # Diagonal: exp(λ * time)
            exp_diag = torch.exp(eigenvalues * time)
            # Compute exp(M*time)*x0 = v @ diag(exp_diag) @ v_inv @ x0
            traj[:, t] = eigenvectors @ (exp_diag * (v_inv @ x0))
        return traj

    # ------------------------------------------------------------------ #
    # 4.  XOR gate                                                       #
    # ------------------------------------------------------------------ #
    def truth_table(
        self,
        nt: int = 200,
        dt: float = 1e-3,
        readout_time: int = 150,
        width: int | None = None,
        threshold: float = 0.8,
        seed: int = 1,
    ) -> Dict[Tuple[int, int], int]:
        """
        Simulate all four input combinations and return a dictionary
        `{(X,Y): XOR(X,Y)}` where the output is determined *solely* by
        synchrony in the two stimulus regions.

        Args:
            nt (int): number of timesteps. (default: 200).
            dt (float): timestep. (default: 1e-3).
            readout_time (int): time at which to measure synchrony. (default: 150).
            width (int): width of the two stimulus regions. (default: N/10).
            threshold (float): synchrony threshold. (default: 0.8).
            seed (int): random seed. (default: 1).
        """
        if width is None:
            width = self.N // 10

        center1 = self.N // 3
        center2 = 2 * self.N // 3

        # --- deterministic initial phases ------------------------------
        g = torch.Generator(device=self.device).manual_seed(seed)
        x0 = torch.exp(
            1j * 2 * math.pi * torch.rand(self.N, generator=g, device=self.device)
        )

        inp1 = self._chimera_input(center1, width)
        inp2 = self._chimera_input(center2, width)

        # enumerate input combinations
        combos = {
            (0, 0): x0,
            (1, 0): x0 * inp1,
            (0, 1): x0 * inp2,
            (1, 1): x0 * inp1 * inp2,
        }

        out: Dict[Tuple[int, int], int] = {}
        for key, state in combos.items():
            traj = self._run_exactly(state, nt, dt)
            x_t = traj[:, readout_time]

            # synchrony in each stimulus region
            idx = torch.arange(self.N, device=self.device)
            dist1 = torch.minimum((idx - center1).abs(), self.N - (idx - center1).abs())
            dist2 = torch.minimum((idx - center2).abs(), self.N - (idx - center2).abs())
            mask1 = dist1 < width
            mask2 = dist2 < width

            s1 = self._sync_level(torch.angle(x_t[mask1]))
            s2 = self._sync_level(torch.angle(x_t[mask2]))

            bit1 = int(s1 > threshold)
            bit2 = int(s2 > threshold)
            print(f"XOR({key}): {bit1} XOR {bit2}")
            out[key] = bit1 ^ bit2  # XOR in Python

            self.plot_thickens(str(key), traj, readout_time)
        return out

    def plot_thickens(
        self, title: str, x: torch.Tensor, readout_time: int = -1
    ) -> None:
        # Determine the number of timesteps from the tensor shape
        nt = x.shape[1]
        # Create a time vector from 0 to nt steps
        t = np.linspace(0, nt)

        # ---- Figure 1: Spatiotemporal phase cv-NN ----
        plt.figure(figsize=(7.83, 1.92))
        plt.title(f"Spatiotemporal phase ({title})", fontsize=16, fontname="Arial")
        # Plot phase dynamics: note np.angle(x).T gives a (nt x N) array for imshow
        img = plt.imshow(
            np.angle(x).T,
            aspect="auto",
            origin="lower",
            extent=[t[0], t[-1], 1, self.N],
            cmap="bone",
        )
        plt.xlabel("time (steps)", fontsize=16, fontname="Arial")
        plt.ylabel("nodes", fontsize=16, fontname="Arial")
        plt.colorbar(img, label="phase (rad)")
        plt.tick_params(labelsize=14, width=2)
        plt.tight_layout()

        # ---- Figure 2: Final state cv-NN ----
        plt.figure(figsize=(4.05, 1.84))
        plt.title(f"State {title} at ({readout_time})", fontsize=16, fontname="Arial")
        # Extract final state across all nodes (column axis)
        final_phase = np.angle(x[:, readout_time])
        nodes = np.arange(1, self.N + 1)  # nodes from 1 to N
        plt.scatter(
            nodes, final_phase, c="black", marker="o", s=40
        )  # 'filled' circle marker
        plt.xlabel("nodes", fontsize=16, fontname="Arial")
        plt.ylabel("phase (rad)", fontsize=16, fontname="Arial")
        plt.xlim(0, self.N + 1)
        plt.ylim(-4, 4)
        plt.tick_params(labelsize=14, width=2)
        plt.tight_layout()

        plt.show()


# --------------------------------------------------------------------------- #
# Quick demo                                                                  #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    gate = XorCVNN(N=201, device="cpu")
    table = gate.truth_table(nt=400, dt=1e-3, readout_time=120, threshold=0.2)
    print("-------------")
    print("XOR truth table (synchrony-based read-out)\n X  Y |  f(X,Y)")
    for (x, y), z in sorted(table.items()):
        print(f" {x}  {y} |   {z}")
