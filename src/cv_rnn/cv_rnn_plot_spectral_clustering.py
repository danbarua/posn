import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # needed for 3D plotting
from sklearn.cluster import KMeans

def plot_spectral_clustering(prj: np.ndarray, x: np.ndarray, phase_iter: int = 120) -> np.ndarray:
    """
    Visualize the result of spectral clustering in 3D and perform KMeans clustering.

    Parameters:
        prj        : np.ndarray of shape (N, 3, W)
                     The projection matrix over sliding windows.
        x          : np.ndarray of shape (N, T) (complex valued)
                     The complex state from which the phase is computed.
        phase_iter : int, optional (default: 120)
                     The iteration index for phase visualization (MATLAB index 120 converts to Python index 119).

    Returns:
        predict    : np.ndarray of shape (N,)
                     Cluster labels (0 or 1) for each data point based on the last window’s projection.
    """
    # Select the last window index.
    # In MATLAB: w = size(prj, 3); then prj(:, :, w)
    last_window = prj[:, :, -1]

    # Get colours from the phase of x at iteration 'phase_iter'.
    # MATLAB uses x(:,120) where Python indices are 0 based.
    # Make sure that phase_iter does not exceed x.shape[1].
    phase_index = phase_iter - 1
    if phase_index >= x.shape[1]:
        raise ValueError(f"phase_iter ({phase_iter}) is out of bounds for x with shape {x.shape}")

    colors = np.angle(x[:, phase_index])

    # Create figure and 3D axis.
    fig = plt.figure(figsize=(5.5, 3.3))  # approximate conversion from MATLAB's position
    ax = fig.add_subplot(111, projection='3d')

    # Create scatter plot for the projection (dimensions 1, 2 and 3).
    sc = ax.scatter(last_window[:, 0], last_window[:, 1], last_window[:, 2],
                    s=50, c=colors, cmap='hsv', marker='o', depthshade=True)
    sc.set_clim([-np.pi, np.pi])

    # Set labels and title.
    ax.set_xlabel('dimension 1', fontname='Arial', fontsize=15)
    ax.set_ylabel('dimension 2', fontname='Arial', fontsize=15)
    ax.set_zlabel('dimension 3', fontname='Arial', fontsize=15)
    ax.set_title('similarity projection', fontsize=15)

    # Add a colorbar with label.
    cbar = plt.colorbar(sc, ax=ax, pad=0.1)
    cbar.set_label('phase (rad)', fontsize=15)

    plt.show()

    # Run KMeans on the last window's projection data.
    # MATLAB clusters prj(:,1:3,end), which in Python is last_window (shape (N,3))
    kmeans = KMeans(n_clusters=2, random_state=0)
    # Fit and predict the cluster labels.
    predict = kmeans.fit_predict(last_window)

    return predict
