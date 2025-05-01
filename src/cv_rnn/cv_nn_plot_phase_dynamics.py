import torch
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation


def plot_dynamics(
    x: np.ndarray, im: np.ndarray | torch.Tensor, layer_1_final_time: int
):
    """
    Replicates the MATLAB `plot_dynamics` routine in Python/Matplotlib.

    Parameters
    ----------
    x : np.ndarray
        A 2-D array of (H * W, T) complex values.
    im : np.ndarray | torch.Tensor
        (H,W) for shape only
    layer_1_final_time : int
        Time index at which the colour limits switch to [-π, π].

    Returns
    -------
    matplotlib.axes.Axes
        The Axes instance that contains the image.
    """

    H, W = im.shape
    # x is (H * W, T)
    # we need (H, W, T)
    x = x.reshape((H, W, -1))

    if x.ndim != 3:
        raise ValueError("`x` must be a 3-D array (H, W, T).")

    plot_dynamics_animated(x, layer_1_final_time)
    return

    # --- initial plot --------------------------------------------------------
    fig, ax = plt.subplots()
    im = ax.imshow(np.angle(x[:, :, 0]), cmap="hsv", interpolation="nearest")
    ax.axis("off")
    ax.set_title("")  # Keep title area free
    ax.set_xlabel("")  # Remove any axes labels
    ax.set_ylabel("")

    # Use a consistent font (like the original 'arial')
    plt.rcParams.update({"font.size": 15, "font.family": "Arial"})

    # Timesteps text in the same spot used in the MATLAB code
    t1 = ax.text(13, -1, "1", fontsize=15, fontname="Arial")

    # -------------------------------------------------------------------------
    times = range(1, min(180, x.shape[2]))  # MATLAB 2:180  -> Python 1 .. 179
    for ii in times:
        tmp = x[:, :, ii]
        tmp = np.angle(tmp)
        # Update image data
        im.set_data(tmp)

        # Handle transparency (AlphaData in MATLAB)
        alpha = ~np.isnan(tmp)  # True where data is NOT NaN
        im.set_alpha(alpha.astype(float))

        # Update colour limits after `layer_1_final_time`
        if ii > layer_1_final_time:
            im.set_clim(-np.pi, np.pi)

        # Update the text string
        t1.set_text(f"{ii + 1} timesteps")  # MATLAB indices start at 1

        # Let Matplotlib redraw
        plt.pause(0.1)

    return ax


def plot_dynamics_animated(x: np.ndarray, layer_1_final_time: int):
    """
    Animate the phase dynamics of the network using matplotlib's FuncAnimation.

    Parameters:
    -----------
    x : np.ndarray
        A 3D array (H, W, T) containing the phase dynamics.
    layer_1_final_time : int
        The timestep after which color limits are constrained to [-π, π].

    Returns:
    --------
    anim : FuncAnimation
        The matplotlib animation object.
    """
    if x.ndim != 3:
        raise ValueError("`x` must be a 3-D array (H, W, T).")

    fig, ax = plt.subplots()
    plt.rcParams.update({"font.size": 15, "font.family": "Arial"})
    ax.axis("off")

    # Initial image plot with the first timestep
    im = ax.imshow(np.angle(x[:, :, 0]), cmap="hsv", interpolation="nearest", alpha=1.0)

    # Set initial color limits if needed
    # Here we initially do not limit, as in MATLAB code
    # Color limits will be set in update function conditionally

    # Add text annotation for timestep display at (13, -1)
    t1 = ax.text(13, -1, "1", fontsize=15, fontname="Arial")

    # Define update function for FuncAnimation
    def update(ii):
        phase_data = np.angle(x[:, :, ii])

        im.set_data(phase_data)
        # Set transparency: opaque where not NaN, transparent where NaN
        alpha = ~np.isnan(phase_data)
        im.set_alpha(alpha.astype(float))

        # Update color limits after layer_1_final_time
        if ii > layer_1_final_time:
            im.set_clim(-np.pi, np.pi)
        else:
            # If before or equal to layer_1_final_time, autoscale color limits
            im.autoscale()

        # Update text with timestep (1-based index)
        t1.set_text(f"{ii + 1} timesteps")

        return im, t1

    # Create animation: interval in ms to match ~0.1s pause in MATLAB code
    frames = min(180, x.shape[2])
    anim = FuncAnimation(fig, update, frames=frames, interval=200, blit=True)

    ax.set_xlim(0, x.shape[1])
    ax.set_ylim(x.shape[0], 0)
    ax.set_aspect("equal")

    return anim


def anim_test():
    x = np.random.rand(10, 10, 50) * 2 * np.pi - np.pi  # (H, W, T)
    fig, ax = plt.subplots()
    im = ax.imshow(np.angle(x[:, :, 0]), cmap="hsv")

    def update(i):
        data = np.angle(x[:, :, i])
        im.set_data(data)
        return (im,)

    ani = FuncAnimation(fig, update, frames=50, interval=100, blit=True)
    plt.show()


if __name__ == "__main__":
    anim_test()
