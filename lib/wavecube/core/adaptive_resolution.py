"""
Adaptive resolution system for WaveCube.

Dynamically adjusts wavetable resolution based on proto-identity density.
Provides upsampling/downsampling with quality preservation and smooth
transitions between resolution levels.
"""

from typing import Tuple, Optional, Dict, Any
import numpy as np

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    from scipy.ndimage import zoom
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def upsample_wavetable(
    wavetable: np.ndarray,
    target_shape: Tuple[int, int, int],
    method: str = 'cubic'
) -> np.ndarray:
    """
    Upsample wavetable to higher resolution.

    Uses cubic interpolation to preserve frequency information during
    upsampling. Supports both OpenCV (faster) and scipy fallback.

    Args:
        wavetable: Input wavetable (H, W, C)
        target_shape: Target shape (H, W, C)
        method: Interpolation method ('cubic', 'linear', 'nearest')

    Returns:
        Upsampled wavetable with target shape

    Raises:
        ValueError: If target shape is smaller than current shape
    """
    if wavetable.ndim != 3:
        raise ValueError(f"Wavetable must be 3D (H, W, C), got {wavetable.ndim}D")

    h, w, c = wavetable.shape
    target_h, target_w, target_c = target_shape

    if target_h < h or target_w < w:
        raise ValueError(
            f"Target shape {target_shape} smaller than current {wavetable.shape}"
        )

    if target_c != c:
        raise ValueError(
            f"Channel mismatch: current={c}, target={target_c}"
        )

    # Use OpenCV if available (faster)
    if HAS_CV2 and method in ('cubic', 'linear', 'nearest'):
        interp_map = {
            'cubic': cv2.INTER_CUBIC,
            'linear': cv2.INTER_LINEAR,
            'nearest': cv2.INTER_NEAREST
        }
        interpolation = interp_map[method]

        # OpenCV resize expects (width, height)
        result = cv2.resize(
            wavetable,
            (target_w, target_h),
            interpolation=interpolation
        )

        # Ensure shape is correct (H, W, C)
        if result.ndim == 2:
            result = result[:, :, np.newaxis]

        return result.astype(wavetable.dtype)

    # Fallback to scipy
    elif HAS_SCIPY:
        order_map = {
            'cubic': 3,
            'linear': 1,
            'nearest': 0
        }
        order = order_map.get(method, 3)

        zoom_factors = (
            target_h / h,
            target_w / w,
            1.0  # Don't interpolate across channels
        )

        result = zoom(wavetable, zoom_factors, order=order)
        return result.astype(wavetable.dtype)

    else:
        raise RuntimeError(
            "No interpolation library available. "
            "Install opencv-python or scipy."
        )


def downsample_wavetable(
    wavetable: np.ndarray,
    target_shape: Tuple[int, int, int],
    method: str = 'area'
) -> np.ndarray:
    """
    Downsample wavetable to lower resolution.

    Uses area averaging for downsampling to reduce aliasing.

    Args:
        wavetable: Input wavetable (H, W, C)
        target_shape: Target shape (H, W, C)
        method: Downsampling method ('area', 'cubic', 'linear')

    Returns:
        Downsampled wavetable with target shape

    Raises:
        ValueError: If target shape is larger than current shape
    """
    if wavetable.ndim != 3:
        raise ValueError(f"Wavetable must be 3D (H, W, C), got {wavetable.ndim}D")

    h, w, c = wavetable.shape
    target_h, target_w, target_c = target_shape

    if target_h > h or target_w > w:
        raise ValueError(
            f"Target shape {target_shape} larger than current {wavetable.shape}"
        )

    if target_c != c:
        raise ValueError(
            f"Channel mismatch: current={c}, target={target_c}"
        )

    # Use OpenCV for area averaging (best for downsampling)
    if HAS_CV2:
        interp_map = {
            'area': cv2.INTER_AREA,
            'cubic': cv2.INTER_CUBIC,
            'linear': cv2.INTER_LINEAR
        }
        interpolation = interp_map.get(method, cv2.INTER_AREA)

        # OpenCV resize expects (width, height)
        result = cv2.resize(
            wavetable,
            (target_w, target_h),
            interpolation=interpolation
        )

        # Ensure shape is correct (H, W, C)
        if result.ndim == 2:
            result = result[:, :, np.newaxis]

        return result.astype(wavetable.dtype)

    # Fallback to scipy
    elif HAS_SCIPY:
        # Use cubic for downsampling
        zoom_factors = (
            target_h / h,
            target_w / w,
            1.0
        )

        result = zoom(wavetable, zoom_factors, order=3)
        return result.astype(wavetable.dtype)

    else:
        raise RuntimeError(
            "No interpolation library available. "
            "Install opencv-python or scipy."
        )


def resize_wavetable(
    wavetable: np.ndarray,
    target_shape: Tuple[int, int, int],
    method: Optional[str] = None
) -> np.ndarray:
    """
    Resize wavetable to target shape (up or down).

    Automatically chooses appropriate method based on direction.

    Args:
        wavetable: Input wavetable (H, W, C)
        target_shape: Target shape (H, W, C)
        method: Interpolation method (auto-selected if None)

    Returns:
        Resized wavetable
    """
    h, w, c = wavetable.shape
    target_h, target_w, target_c = target_shape

    if (h, w, c) == target_shape:
        return wavetable

    # Determine direction
    is_upsampling = target_h >= h and target_w >= w

    # Auto-select method if not specified
    if method is None:
        method = 'cubic' if is_upsampling else 'area'

    if is_upsampling:
        return upsample_wavetable(wavetable, target_shape, method)
    else:
        return downsample_wavetable(wavetable, target_shape, method)


def blend_edge_transitions(
    low_res: np.ndarray,
    high_res: np.ndarray,
    blend_width: int = 8
) -> np.ndarray:
    """
    Blend edges between different resolution wavetables.

    Creates smooth transitions to avoid discontinuities at chunk boundaries.

    Args:
        low_res: Lower resolution wavetable
        high_res: Higher resolution wavetable (will be downsampled)
        blend_width: Width of blend region in pixels

    Returns:
        Blended wavetable at low_res resolution
    """
    if low_res.shape != high_res.shape:
        # Downsample high_res to match low_res
        high_res = downsample_wavetable(high_res, low_res.shape)

    h, w, c = low_res.shape

    # Create blend mask (alpha channel)
    mask = np.ones((h, w, 1), dtype=np.float32)

    # Apply Gaussian blur at edges
    for i in range(blend_width):
        alpha = i / blend_width
        # Top edge
        if i < h:
            mask[i, :, :] = alpha
        # Bottom edge
        if h - i - 1 >= 0:
            mask[h - i - 1, :, :] = alpha
        # Left edge
        if i < w:
            mask[:, i, :] = np.minimum(mask[:, i, :], alpha)
        # Right edge
        if w - i - 1 >= 0:
            mask[:, w - i - 1, :] = np.minimum(mask[:, w - i - 1, :], alpha)

    # Blend
    result = low_res * mask + high_res * (1 - mask)
    return result.astype(low_res.dtype)


class AdaptiveResolutionManager:
    """
    Manages adaptive resolution for WaveCube chunks.

    Tracks resolution changes, handles upsampling/downsampling,
    and maintains quality metrics.
    """

    def __init__(
        self,
        default_resolution: Tuple[int, int, int] = (512, 512, 4),
        min_resolution: Tuple[int, int, int] = (16, 16, 4),
        max_resolution: Tuple[int, int, int] = (3840, 3840, 48),
        interpolation_method: str = 'cubic'
    ):
        """
        Initialize adaptive resolution manager.

        Args:
            default_resolution: Default wavetable resolution
            min_resolution: Minimum allowed resolution
            max_resolution: Maximum allowed resolution
            interpolation_method: Default interpolation method
        """
        self.default_resolution = default_resolution
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.interpolation_method = interpolation_method

        # Track adaptations
        self.stats = {
            'total_adaptations': 0,
            'upsamples': 0,
            'downsamples': 0,
            'avg_mse': 0.0,
            'total_mse': 0.0
        }

    def adapt_wavetable(
        self,
        wavetable: np.ndarray,
        target_resolution: Tuple[int, int, int],
        track_error: bool = True
    ) -> Dict[str, Any]:
        """
        Adapt wavetable to target resolution.

        Args:
            wavetable: Input wavetable
            target_resolution: Target resolution
            track_error: Track reconstruction error

        Returns:
            Dict with resized wavetable and metadata
        """
        original_shape = wavetable.shape

        # Validate target resolution
        target_resolution = self._validate_resolution(target_resolution)

        # Resize
        resized = resize_wavetable(
            wavetable,
            target_resolution,
            self.interpolation_method
        )

        # Calculate error if tracking
        mse = 0.0
        if track_error:
            # Resize back to original for comparison
            reconstructed = resize_wavetable(
                resized,
                original_shape,
                self.interpolation_method
            )
            mse = np.mean((wavetable - reconstructed) ** 2)

            # Update statistics
            self._update_stats(original_shape, target_resolution, mse)

        return {
            'wavetable': resized,
            'original_shape': original_shape,
            'target_shape': target_resolution,
            'mse': mse,
            'method': self.interpolation_method
        }

    def _validate_resolution(
        self,
        resolution: Tuple[int, int, int]
    ) -> Tuple[int, int, int]:
        """
        Validate and clamp resolution to allowed range.

        Args:
            resolution: Requested resolution

        Returns:
            Validated resolution
        """
        h, w, c = resolution
        min_h, min_w, min_c = self.min_resolution
        max_h, max_w, max_c = self.max_resolution

        # Clamp to range
        h = max(min_h, min(max_h, h))
        w = max(min_w, min(max_w, w))
        c = max(min_c, min(max_c, c))

        return (h, w, c)

    def _update_stats(
        self,
        original_shape: Tuple[int, int, int],
        target_shape: Tuple[int, int, int],
        mse: float
    ) -> None:
        """
        Update adaptation statistics.

        Args:
            original_shape: Original wavetable shape
            target_shape: Target wavetable shape
            mse: Mean squared error
        """
        self.stats['total_adaptations'] += 1

        # Track direction
        if target_shape[0] > original_shape[0]:
            self.stats['upsamples'] += 1
        else:
            self.stats['downsamples'] += 1

        # Update MSE
        total = self.stats['total_adaptations']
        self.stats['total_mse'] += mse
        self.stats['avg_mse'] = self.stats['total_mse'] / total

    def get_statistics(self) -> Dict[str, Any]:
        """Get adaptation statistics."""
        return self.stats.copy()

    def reset_statistics(self) -> None:
        """Reset statistics to initial values."""
        self.stats = {
            'total_adaptations': 0,
            'upsamples': 0,
            'downsamples': 0,
            'avg_mse': 0.0,
            'total_mse': 0.0
        }

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"AdaptiveResolutionManager("
            f"default={self.default_resolution[0]}×{self.default_resolution[1]}×{self.default_resolution[2]}, "
            f"adaptations={self.stats['total_adaptations']}, "
            f"avg_mse={self.stats['avg_mse']:.6f})"
        )
