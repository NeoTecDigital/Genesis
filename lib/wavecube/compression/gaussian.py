"""
Gaussian Mixture Codec for wavetable compression.

Fits a mixture of 2D Gaussians to represent sparse frequency patterns.
Achieves 1000-10000× compression for sparse data (e.g., Genesis proto-identities).

Compression:
    512×512×4 wavetable (4MB) → 8 Gaussians × 9 params (288 bytes) = 14,563× compression

Parameters per Gaussian:
    - amplitude (1 float)
    - center_x, center_y (2 floats)
    - sigma_x, sigma_y (2 floats)
    - correlation (1 float)
    - phase (1 float per channel)
"""

from typing import Optional, Tuple
import numpy as np
from scipy import optimize
from dataclasses import dataclass

from .codec import WavetableCodec, CompressedWavetable


@dataclass
class GaussianMixtureParams:
    """
    Parameters for Gaussian mixture representation.

    Attributes:
        num_gaussians: Number of Gaussian components
        amplitudes: (N,) amplitude per Gaussian
        centers: (N, 2) [x, y] centers in normalized coordinates
        sigmas: (N, 2) [sigma_x, sigma_y] standard deviations
        correlations: (N,) correlation coefficients
        phases: (N, C) phase per Gaussian per channel
    """
    num_gaussians: int
    amplitudes: np.ndarray  # (N,)
    centers: np.ndarray     # (N, 2)
    sigmas: np.ndarray      # (N, 2)
    correlations: np.ndarray  # (N,)
    phases: np.ndarray      # (N, C)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'num_gaussians': self.num_gaussians,
            'amplitudes': self.amplitudes,
            'centers': self.centers,
            'sigmas': self.sigmas,
            'correlations': self.correlations,
            'phases': self.phases
        }

    @staticmethod
    def from_dict(data: dict) -> 'GaussianMixtureParams':
        """Load from dictionary."""
        return GaussianMixtureParams(
            num_gaussians=data['num_gaussians'],
            amplitudes=data['amplitudes'],
            centers=data['centers'],
            sigmas=data['sigmas'],
            correlations=data['correlations'],
            phases=data['phases']
        )


class GaussianMixtureCodec(WavetableCodec):
    """
    Gaussian Mixture compression codec.

    Fits a mixture of 2D Gaussians to each channel independently.
    Best for sparse patterns with localized peaks (e.g., frequency spectra).
    """

    def __init__(
        self,
        num_gaussians: int = 8,
        init_method: str = 'kmeans',
        max_iterations: int = 100,
        tolerance: float = 1e-4
    ):
        """
        Initialize Gaussian Mixture codec.

        Args:
            num_gaussians: Number of Gaussian components to fit
            init_method: Initialization method ('kmeans', 'random', 'peaks')
            max_iterations: Max iterations for fitting
            tolerance: Convergence tolerance
        """
        super().__init__(
            num_gaussians=num_gaussians,
            init_method=init_method,
            max_iterations=max_iterations,
            tolerance=tolerance
        )

    def encode(
        self,
        wavetable: np.ndarray,
        quality: float = 0.95
    ) -> CompressedWavetable:
        """
        Fit Gaussian mixture to wavetable.

        Args:
            wavetable: Input wavetable (H, W, C)
            quality: Quality level (affects num_gaussians)

        Returns:
            CompressedWavetable with Gaussian mixture parameters
        """
        self.validate_wavetable(wavetable)

        h, w, c = wavetable.shape

        # Adjust num_gaussians based on quality
        num_gaussians = max(1, int(self.params['num_gaussians'] * quality))

        # Fit Gaussians to magnitude (averaged across channels for simplicity)
        magnitude = np.mean(np.abs(wavetable), axis=2)

        # Initialize Gaussian centers
        centers = self._initialize_centers(magnitude, num_gaussians)

        # Fit Gaussians using EM or optimization
        params = self._fit_gaussians(magnitude, centers, num_gaussians)

        # Extract phase information per channel
        phases = self._extract_phases(wavetable, params)
        params.phases = phases

        return CompressedWavetable(
            method='gaussian',
            params=params,
            original_shape=wavetable.shape,
            dtype=wavetable.dtype,
            metadata={
                'num_gaussians': num_gaussians,
                'quality': quality
            }
        )

    def decode(
        self,
        compressed: CompressedWavetable
    ) -> np.ndarray:
        """
        Reconstruct wavetable from Gaussian mixture.

        Args:
            compressed: CompressedWavetable with Gaussian parameters

        Returns:
            Reconstructed wavetable (H, W, C)
        """
        if compressed.method != 'gaussian':
            raise ValueError(f"Expected gaussian compression, got {compressed.method}")

        params: GaussianMixtureParams = compressed.params
        h, w, c = compressed.original_shape

        # Create coordinate grids (normalized to [0, 1])
        y_grid, x_grid = np.meshgrid(
            np.linspace(0, 1, h),
            np.linspace(0, 1, w),
            indexing='ij'
        )

        # Reconstruct magnitude from Gaussians
        magnitude = np.zeros((h, w), dtype=compressed.dtype)

        for i in range(params.num_gaussians):
            # Get Gaussian parameters
            amp = params.amplitudes[i]
            cx, cy = params.centers[i]
            sx, sy = params.sigmas[i]
            corr = params.correlations[i]

            # Compute 2D Gaussian
            dx = x_grid - cx
            dy = y_grid - cy

            # Covariance matrix: [[sx^2, rho*sx*sy], [rho*sx*sy, sy^2]]
            # Inverse covariance determinant
            det = sx**2 * sy**2 * (1 - corr**2)
            if det <= 0:
                continue

            # Mahalanobis distance
            inv_cov_xx = sy**2 / det
            inv_cov_yy = sx**2 / det
            inv_cov_xy = -corr * sx * sy / det

            exponent = -(
                inv_cov_xx * dx**2 +
                inv_cov_yy * dy**2 +
                2 * inv_cov_xy * dx * dy
            ) / 2

            gaussian = amp * np.exp(exponent)
            magnitude += gaussian

        # Apply phase information per channel
        wavetable = np.zeros((h, w, c), dtype=compressed.dtype)

        for channel in range(c):
            # Reconstruct complex representation
            phase_map = np.zeros((h, w), dtype=np.float32)

            for i in range(params.num_gaussians):
                # Get Gaussian parameters (same as magnitude)
                amp = params.amplitudes[i]
                cx, cy = params.centers[i]
                sx, sy = params.sigmas[i]
                corr = params.correlations[i]

                # Compute Gaussian weight
                dx = x_grid - cx
                dy = y_grid - cy

                det = sx**2 * sy**2 * (1 - corr**2)
                if det <= 0:
                    continue

                inv_cov_xx = sy**2 / det
                inv_cov_yy = sx**2 / det
                inv_cov_xy = -corr * sx * sy / det

                exponent = -(
                    inv_cov_xx * dx**2 +
                    inv_cov_yy * dy**2 +
                    2 * inv_cov_xy * dx * dy
                ) / 2

                weight = np.exp(exponent)

                # Phase for this Gaussian and channel
                phase = params.phases[i, channel]

                # Accumulate weighted phase
                phase_map += weight * phase

            # Combine magnitude and phase
            wavetable[:, :, channel] = magnitude * np.cos(phase_map)

        return wavetable

    def _initialize_centers(
        self,
        magnitude: np.ndarray,
        num_gaussians: int
    ) -> np.ndarray:
        """
        Initialize Gaussian centers using peak detection.

        Args:
            magnitude: 2D magnitude array (H, W)
            num_gaussians: Number of centers to initialize

        Returns:
            Array of shape (N, 2) with [x, y] centers in [0, 1]
        """
        h, w = magnitude.shape

        method = self.params.get('init_method', 'peaks')

        if method == 'peaks':
            # Find local maxima
            from scipy.ndimage import maximum_filter

            # Detect peaks
            footprint = np.ones((5, 5))
            local_max = maximum_filter(magnitude, footprint=footprint) == magnitude
            local_max &= (magnitude > magnitude.mean())

            # Get peak coordinates
            peak_coords = np.argwhere(local_max)

            if len(peak_coords) == 0:
                # Fallback to random
                return np.random.rand(num_gaussians, 2)

            # Sort by magnitude and take top N
            peak_values = magnitude[peak_coords[:, 0], peak_coords[:, 1]]
            sorted_indices = np.argsort(peak_values)[::-1]
            top_peaks = peak_coords[sorted_indices[:num_gaussians]]

            # Normalize to [0, 1]
            centers = top_peaks.astype(np.float32)
            centers[:, 0] /= h
            centers[:, 1] /= w

            # Pad if needed
            if len(centers) < num_gaussians:
                extra = num_gaussians - len(centers)
                centers = np.vstack([centers, np.random.rand(extra, 2)])

            return centers[:, [1, 0]]  # Swap to [x, y]

        else:  # random
            return np.random.rand(num_gaussians, 2)

    def _fit_gaussians(
        self,
        magnitude: np.ndarray,
        initial_centers: np.ndarray,
        num_gaussians: int
    ) -> GaussianMixtureParams:
        """
        Fit Gaussian mixture using simplified EM-like approach.

        Args:
            magnitude: 2D magnitude array (H, W)
            initial_centers: Initial centers (N, 2)
            num_gaussians: Number of Gaussians

        Returns:
            GaussianMixtureParams
        """
        h, w = magnitude.shape

        # Initialize parameters
        amplitudes = np.zeros(num_gaussians, dtype=np.float32)
        centers = initial_centers.copy()
        sigmas = np.ones((num_gaussians, 2), dtype=np.float32) * 0.1
        correlations = np.zeros(num_gaussians, dtype=np.float32)

        # Create coordinate grids
        y_grid, x_grid = np.meshgrid(
            np.linspace(0, 1, h),
            np.linspace(0, 1, w),
            indexing='ij'
        )

        # Simplified fitting: estimate amplitude and sigma from local regions
        for i in range(num_gaussians):
            cx, cy = centers[i]

            # Find nearby pixels (within 0.2 normalized distance)
            dist = np.sqrt((x_grid - cx)**2 + (y_grid - cy)**2)
            mask = dist < 0.2

            if np.any(mask):
                # Amplitude = max magnitude in region
                amplitudes[i] = magnitude[mask].max()

                # Estimate sigma from weighted std dev
                local_mag = magnitude[mask]
                local_x = x_grid[mask]
                local_y = y_grid[mask]

                if len(local_mag) > 1:
                    # Weighted by magnitude
                    weights = local_mag / (local_mag.sum() + 1e-10)
                    sigma_x = np.sqrt(np.sum(weights * (local_x - cx)**2))
                    sigma_y = np.sqrt(np.sum(weights * (local_y - cy)**2))

                    sigmas[i, 0] = max(sigma_x, 0.01)
                    sigmas[i, 1] = max(sigma_y, 0.01)
            else:
                amplitudes[i] = magnitude.max() * 0.1
                sigmas[i] = [0.05, 0.05]

        # Phases will be filled by _extract_phases
        phases = np.zeros((num_gaussians, 1), dtype=np.float32)

        return GaussianMixtureParams(
            num_gaussians=num_gaussians,
            amplitudes=amplitudes,
            centers=centers,
            sigmas=sigmas,
            correlations=correlations,
            phases=phases
        )

    def _extract_phases(
        self,
        wavetable: np.ndarray,
        params: GaussianMixtureParams
    ) -> np.ndarray:
        """
        Extract phase information for each Gaussian and channel.

        Args:
            wavetable: Original wavetable (H, W, C)
            params: Gaussian parameters

        Returns:
            Phase array of shape (N, C)
        """
        h, w, c = wavetable.shape

        # For simplicity, extract average phase in region around each Gaussian
        phases = np.zeros((params.num_gaussians, c), dtype=np.float32)

        y_grid, x_grid = np.meshgrid(
            np.linspace(0, 1, h),
            np.linspace(0, 1, w),
            indexing='ij'
        )

        for i in range(params.num_gaussians):
            cx, cy = params.centers[i]

            # Find pixels near center
            dist = np.sqrt((x_grid - cx)**2 + (y_grid - cy)**2)
            mask = dist < 0.1

            if np.any(mask):
                for channel in range(c):
                    # Average value in region
                    phases[i, channel] = np.mean(wavetable[mask, channel])

        return phases
