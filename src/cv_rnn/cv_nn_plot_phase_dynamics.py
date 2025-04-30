# Python
import torch
import torch
import numpy as np
import numpy as np
import matplotlib.pyplot as plt


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
