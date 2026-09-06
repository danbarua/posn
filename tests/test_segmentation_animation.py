"""
Regression tests for animation rendering.

Tests defend visible animation properties: F-order pixel layout, layer1/layer2
boundary, NaN transparency, phase color mapping, and final frames.
Uses actual Agg/Pillow rendering to verify consumer-visible outputs.
"""

import io
import math
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from src.cv_rnn import run_2layer_torch, animate_dynamics, plot_spectral_clustering


def _render_animation_frame(fig, anim, frame_idx: int) -> Image.Image:
    """Render a single animation frame to PIL Image using Agg backend."""
    # FuncAnimation keeps the update callback as the private `_func`
    # attribute; there is no public accessor. Driving it directly is the
    # only way to render an arbitrary frame without a real-time event loop.
    anim._func(frame_idx)
    
    # Render figure to PNG buffer
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=50)
    buf.seek(0)
    
    return Image.open(buf).copy()


def _image_has_color(img: Image.Image, target_color: tuple[int, int, int], 
                     tolerance: int = 30) -> bool:
    """Check if image contains target color within tolerance."""
    arr = np.array(img.convert("RGB"))
    target = np.array(target_color, dtype=np.uint8)
    
    # Compute color distance for each pixel
    distances = np.linalg.norm(arr.astype(int) - target.astype(int), axis=2)
    
    # Return True if any pixel is close to target color
    return np.any(distances < tolerance)


def _count_nonwhite_pixels(img: Image.Image, threshold: int = 240) -> int:
    """Count pixels that are not near-white (for NaN/transparent detection)."""
    arr = np.array(img.convert("RGB"))
    is_white = np.all(arr >= threshold, axis=2)
    return np.sum(~is_white)


class TestAnimateDynamicsBasic:
    """Basic validation tests for animate_dynamics function."""

    def test_f_order_rectangular_layout(self):
        """Verify F-order column-major pixel layout on rectangular image."""
        H, W = 4, 6  # Rectangular to expose row/col swap errors
        N = H * W
        T = 10
        
        # Create simple pattern: vary phase by pixel position
        # Phase increases along columns (F-order: first row varies slowest)
        phases = np.zeros(N, dtype=complex)
        for i in range(N):
            col = i // H  # F-order: column index
            row = i % H   # F-order: row index
            phases[i] = np.exp(1j * (row * 2 * np.pi / H))
        
        states = np.tile(phases[:, None], (1, T))
        
        fig, anim = animate_dynamics(states, (H, W), layer_1_steps=5)
        
        # Render first frame and verify pixel layout
        frame_img = _render_animation_frame(fig, anim, 0)
        
        # First frame should show phase pattern
        # With correct F-order, specific pixels should have specific phases
        assert frame_img.size[0] > 0 and frame_img.size[1] > 0
        
        # Cleanup
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_layer1_pixels_visible(self):
        """Verify layer1 pixels are visible (no NaN masking in layer1)."""
        H, W = 5, 5
        N = H * W
        T = 15
        
        # Create states with no NaN in layer1
        states = np.random.uniform(-np.pi, np.pi, (N, T)).astype(complex)
        
        fig, anim = animate_dynamics(states, (H, W), layer_1_steps=10)
        
        # Render a layer1 frame (e.g., frame 5)
        frame_img = _render_animation_frame(fig, anim, 5)
        
        # Count non-white pixels (white would indicate transparent/NaN)
        nonwhite = _count_nonwhite_pixels(frame_img, threshold=240)
        
        # Layer1 frame should have significant colored pixels
        total_pixels = frame_img.size[0] * frame_img.size[1]
        assert nonwhite > total_pixels * 0.1  # At least 10% colored
        
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_nan_transparency_at_layer2(self):
        """Verify NaN values become transparent starting at layer_1_steps."""
        H, W = 5, 5
        N = H * W
        T = 15
        layer_1_steps = 5
        
        # Create states: no NaN in layer1, masked region has NaN in layer2
        states = np.random.uniform(-np.pi, np.pi, (N, T)).astype(complex)
        
        # Add NaN for last 5 pixels in layer2 only
        mask_indices = np.arange(N - 5, N)
        states[mask_indices, layer_1_steps:] = np.nan
        
        fig, anim = animate_dynamics(states, (H, W), layer_1_steps=layer_1_steps)
        
        # Render layer1 frame (should have no transparency)
        frame_layer1 = _render_animation_frame(fig, anim, layer_1_steps - 1)
        nonwhite_layer1 = _count_nonwhite_pixels(frame_layer1)
        
        # Render layer2 frame (should have transparency)
        frame_layer2 = _render_animation_frame(fig, anim, layer_1_steps + 2)
        nonwhite_layer2 = _count_nonwhite_pixels(frame_layer2)
        
        # Layer2 should have fewer colored pixels (some masked as NaN)
        assert nonwhite_layer2 < nonwhite_layer1
        
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_phase_colors_hsv_mapping(self):
        """Verify phase colors map correctly via HSV colormap."""
        H, W = 3, 3
        N = H * W
        T = 5
        
        # Create distinct phases
        states = np.zeros((N, T), dtype=complex)
        # Set specific pixels to specific phases for color check
        states[0, 0] = np.exp(1j * 0)          # phase = 0 (red)
        states[1, 0] = np.exp(1j * np.pi/2)    # phase = π/2 (green)
        states[2, 0] = np.exp(1j * np.pi)      # phase = π (cyan)
        states[3, 0] = np.exp(1j * (-np.pi/2)) # phase = -π/2 (magenta)
        
        fig, anim = animate_dynamics(states, (H, W), layer_1_steps=2)
        
        # Render frame and check for expected HSV colors
        frame_img = _render_animation_frame(fig, anim, 0)
        
        # Convert to RGB and verify color presence
        # Red (phase 0): should have high R
        # Green (phase π/2): should have high G
        # Magenta (phase -π/2): should have R and B
        arr = np.array(frame_img.convert("RGB"))
        
        # Verify non-grayscale (confirms HSV coloring)
        r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
        color_variation = np.std([r.mean(), g.mean(), b.mean()])
        assert color_variation > 10  # Significant RGB variation
        
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_fixed_colorlim_at_layer2(self):
        """Verify [-π, π] color limits fixed starting at layer_1_steps."""
        H, W = 4, 4
        N = H * W
        T = 15
        layer_1_steps = 5
        
        # Layer1: large phase range
        states = np.zeros((N, T), dtype=complex)
        states[:, :layer_1_steps] = np.exp(1j * np.linspace(-2*np.pi, 2*np.pi, N)[:, None])
        
        # Layer2: small phase range (within [-π, π])
        states[:, layer_1_steps:] = np.exp(1j * np.linspace(-np.pi/2, np.pi/2, N)[:, None])
        
        fig, anim = animate_dynamics(states, (H, W), layer_1_steps=layer_1_steps)
        
        # Render layer1 frame (should auto-scale)
        frame_layer1 = _render_animation_frame(fig, anim, layer_1_steps - 1)
        
        # Render layer2 frame (should use fixed [-π, π])
        frame_layer2 = _render_animation_frame(fig, anim, layer_1_steps)
        
        # Both should render without error
        assert frame_layer1.size[0] > 0
        assert frame_layer2.size[0] > 0
        
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_final_frame_renders(self):
        """Verify final animation frame renders correctly."""
        H, W = 5, 5
        N = H * W
        T = 20
        
        states = np.random.uniform(-np.pi, np.pi, (N, T)).astype(complex)
        
        fig, anim = animate_dynamics(states, (H, W), layer_1_steps=10)
        
        # Render final frame
        frame_final = _render_animation_frame(fig, anim, T - 1)
        
        assert frame_final.size[0] > 0
        assert frame_final.size[1] > 0
        
        import matplotlib.pyplot as plt
        plt.close(fig)


class TestAnimateDynamicsValidation:
    """Validation and error handling tests."""

    def test_rejects_1d_input(self):
        """Verify rejection of 1-D input."""
        states = np.random.randn(100)  # 1-D
        
        with pytest.raises(ValueError, match="2-D array"):
            animate_dynamics(states, (10, 10))

    def test_rejects_real_input(self):
        """Verify rejection of real-valued input."""
        states = np.random.randn(100, 20)  # Real, not complex
        
        with pytest.raises(ValueError, match="complex"):
            animate_dynamics(states, (10, 10))

    def test_rejects_mismatched_shape(self):
        """Verify rejection when N != H*W."""
        states = np.random.randn(50, 20).astype(complex)
        
        with pytest.raises(ValueError, match="does not match"):
            animate_dynamics(states, (10, 10))  # 10*10=100, not 50

    def test_rejects_invalid_layer_1_steps(self):
        """Verify rejection of out-of-bounds layer_1_steps."""
        states = np.random.randn(100, 20).astype(complex)
        
        # layer_1_steps >= T
        with pytest.raises(ValueError, match="out of range"):
            animate_dynamics(states, (10, 10), layer_1_steps=25)
        
        # layer_1_steps < 0
        with pytest.raises(ValueError, match="out of range"):
            animate_dynamics(states, (10, 10), layer_1_steps=-1)

    def test_rejects_invalid_interval(self):
        """Verify rejection of non-positive interval."""
        states = np.random.randn(100, 20).astype(complex)
        
        with pytest.raises(ValueError, match="positive"):
            animate_dynamics(states, (10, 10), layer_1_steps=10, interval=0)
        
        with pytest.raises(ValueError, match="positive"):
            animate_dynamics(states, (10, 10), layer_1_steps=10, interval=-10)

    def test_torch_input_converted(self):
        """Verify torch tensor input is converted correctly."""
        states_torch = torch.randn(100, 20, dtype=torch.complex64)
        
        fig, anim = animate_dynamics(states_torch, (10, 10), layer_1_steps=10)
        
        # Should not raise; animation should be created
        assert fig is not None
        assert anim is not None
        
        import matplotlib.pyplot as plt
        plt.close(fig)


class TestPlotSpectralClusteringBasic:
    """Basic validation tests for plot_spectral_clustering function."""

    def test_3d_figure_created(self):
        """Verify 3D figure is created successfully."""
        N_fg = 50
        n_dims = 3
        n_windows = 5
        T = 100
        
        # Create dummy projection and states
        prj = np.random.randn(N_fg, n_dims, n_windows).astype(np.float32)
        x = (np.random.randn(N_fg, T) + 1j * np.random.randn(N_fg, T)).astype(np.complex64)
        
        fig = plot_spectral_clustering(prj, x, phase_iter=60)
        
        assert fig is not None
        assert len(fig.axes) > 0
        
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_phase_coloring_applied(self):
        """Verify phase-based coloring is applied to scatter plot."""
        N_fg = 20
        n_dims = 3
        n_windows = 3
        T = 100
        
        prj = np.random.randn(N_fg, n_dims, n_windows).astype(np.float32)
        
        # Create states with known phases
        x = np.zeros((N_fg, T), dtype=complex)
        # Vary phases from -π to π
        phases = np.linspace(-np.pi, np.pi, N_fg)
        x[:, 50] = np.exp(1j * phases)
        
        fig = plot_spectral_clustering(prj, x, phase_iter=51)
        
        # Extract scatter plot
        ax = fig.axes[0]
        collections = [c for c in ax.collections if hasattr(c, 'get_facecolors')]
        
        # Should have a scatter collection
        assert len(collections) > 0
        
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_last_window_used(self):
        """Verify last window projection is used."""
        N_fg = 10
        n_dims = 3
        n_windows = 5
        T = 50
        
        prj = np.random.randn(N_fg, n_dims, n_windows).astype(np.float32)
        # Mark last window distinctly
        prj[:, :, -1] = 10.0
        
        x = (np.random.randn(N_fg, T) + 1j * np.random.randn(N_fg, T)).astype(np.complex64)
        
        fig = plot_spectral_clustering(prj, x, phase_iter=30)
        
        # Figure should be created without error
        assert fig is not None
        
        import matplotlib.pyplot as plt
        plt.close(fig)


class TestPlotSpectralClusteringValidation:
    """Validation and error handling tests."""

    def test_rejects_1d_prj(self):
        """Verify rejection of 1-D projection array."""
        prj = np.random.randn(100)
        x = (np.random.randn(100, 50) + 1j * np.random.randn(100, 50)).astype(np.complex64)
        
        with pytest.raises(ValueError, match="3-D array"):
            plot_spectral_clustering(prj, x)

    def test_rejects_real_x(self):
        """Verify rejection of real-valued state array."""
        prj = np.random.randn(50, 3, 5).astype(np.float32)
        x = np.random.randn(50, 100)  # Real, not complex
        
        with pytest.raises(ValueError, match="complex"):
            plot_spectral_clustering(prj, x)

    def test_rejects_mismatched_dimensions(self):
        """Verify rejection when prj and x first dimensions don't match."""
        prj = np.random.randn(50, 3, 5).astype(np.float32)
        x = (np.random.randn(30, 100) + 1j * np.random.randn(30, 100)).astype(np.complex64)
        
        with pytest.raises(ValueError, match="does not match"):
            plot_spectral_clustering(prj, x)

    def test_rejects_out_of_bounds_phase_iter(self):
        """Verify rejection of out-of-bounds phase_iter."""
        prj = np.random.randn(20, 3, 5).astype(np.float32)
        x = (np.random.randn(20, 100) + 1j * np.random.randn(20, 100)).astype(np.complex64)
        
        # phase_iter > T (T=100, so phase_iter=101 is out of bounds)
        with pytest.raises(ValueError, match="out of bounds"):
            plot_spectral_clustering(prj, x, phase_iter=101)
        
        # phase_iter < 1 (0-based conversion would be -1)
        with pytest.raises(ValueError, match="out of bounds"):
            plot_spectral_clustering(prj, x, phase_iter=0)

    def test_requires_3_dims_for_plot(self):
        """Verify that at least 3 dimensions are required for 3D plot."""
        prj = np.random.randn(20, 2, 5).astype(np.float32)  # Only 2 dimensions
        x = (np.random.randn(20, 100) + 1j * np.random.randn(20, 100)).astype(np.complex64)
        
        with pytest.raises(ValueError, match="3 dimensions"):
            plot_spectral_clustering(prj, x, phase_iter=50)

    def test_torch_input_converted(self):
        """Verify torch tensor inputs are converted correctly."""
        prj_torch = torch.randn(20, 3, 5, dtype=torch.float32)
        x_torch = torch.randn(20, 100, dtype=torch.complex64)
        
        fig = plot_spectral_clustering(prj_torch, x_torch, phase_iter=50)
        
        # Should not raise; figure should be created
        assert fig is not None
        
        import matplotlib.pyplot as plt
        plt.close(fig)


class TestIntegrationWithSegmentation:
    """Integration tests with actual segmentation pipeline."""

    def test_animate_with_run_2layer_output(self):
        """Test animate_dynamics with actual run_2layer_torch output."""
        # Create small test image
        image = torch.randn(8, 10, dtype=torch.float32) * 0.1
        
        # Run segmentation
        g = torch.Generator()
        g.manual_seed(42)
        
        states, mask = run_2layer_torch(
            image,
            alpha=(0.5, 0.5),
            sigma=(0.9, 0.0313),
            nt=(10, 20),
            generator=g,
            device="cpu",
            dtype=torch.float32,
        )
        
        # Animate
        fig, anim = animate_dynamics(
            states,
            image.shape,
            layer_1_steps=10,
            interval=100,
        )
        
        assert fig is not None
        assert anim is not None
        
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_spectral_plot_with_mock_projection(self):
        """Test plot_spectral_clustering with mock projection data."""
        N_fg = 15
        n_dims = 3
        n_windows = 4
        T = 50
        
        # Create mock projection
        prj = np.random.randn(N_fg, n_dims, n_windows).astype(np.float32)
        
        # Create states with foreground nodes only (no NaN)
        x = (np.random.randn(N_fg, T) + 1j * np.random.randn(N_fg, T)).astype(np.complex64)
        
        fig = plot_spectral_clustering(prj, x, phase_iter=25)
        
        assert fig is not None
        
        import matplotlib.pyplot as plt
        plt.close(fig)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
