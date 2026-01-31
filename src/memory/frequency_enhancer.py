"""
Frequency Enhancement for Multi-Band Abstraction Shifting.

Shifts frequency spectrums between abstraction levels:
- LOW (<10Hz): Concept level
- MID (10-100Hz): Word/context level
- HIGH (>100Hz): Phoneme/character level

Enables iterative cluster→enhance→cluster pipeline for
multi-scale pattern emergence.
"""

import numpy as np
from typing import Tuple, List

from src.memory.frequency_bands import FrequencyBand


class FrequencyEnhancer:
    """
    Shift frequency spectrums between abstraction levels.

    Preserves harmonic relationships via proportional scaling:
    - All frequencies multiplied by same factor
    - Phase relationships maintained
    - Amplitude ratios preserved

    Example:
        enhancer = FrequencyEnhancer()

        # Shift concept to word level
        mid_spectrum = enhancer.shift_band(
            low_spectrum,
            FrequencyBand.LOW,
            FrequencyBand.MID
        )

        # Validate roundtrip preservation
        path = [FrequencyBand.LOW, FrequencyBand.MID,
                FrequencyBand.HIGH, FrequencyBand.MID,
                FrequencyBand.LOW]
        error, similarity = enhancer.validate_roundtrip(
            low_spectrum,
            path
        )
    """

    def __init__(self):
        """Initialize frequency enhancer with shift factors."""
        # Band transition matrix
        self.shift_factors = {
            # Forward shifts (increase frequency)
            (FrequencyBand.LOW, FrequencyBand.MID): 10.0,
            (FrequencyBand.MID, FrequencyBand.HIGH): 4.0,
            (FrequencyBand.LOW, FrequencyBand.HIGH): 40.0,

            # Reverse shifts (decrease frequency)
            (FrequencyBand.MID, FrequencyBand.LOW): 0.1,
            (FrequencyBand.HIGH, FrequencyBand.MID): 0.25,
            (FrequencyBand.HIGH, FrequencyBand.LOW): 0.025,
        }

    def shift_band(self,
                   spectrum: np.ndarray,
                   from_band: FrequencyBand,
                   to_band: FrequencyBand) -> np.ndarray:
        """
        Shift frequency spectrum between bands.

        Args:
            spectrum: Frequency spectrum (H, W, 4) quaternionic
            from_band: Source frequency band
            to_band: Target frequency band

        Returns:
            Shifted spectrum with preserved harmonics

        Raises:
            ValueError: If spectrum shape invalid or bands equal
        """
        # Validate input
        if spectrum.ndim != 3 or spectrum.shape[2] != 4:
            raise ValueError(
                f"Expected (H, W, 4) spectrum, got {spectrum.shape}"
            )

        # No-op if same band
        if from_band == to_band:
            return spectrum.copy()

        # Get shift factor
        factor = self._compute_shift_factor(from_band, to_band)

        # Apply shift to each quaternion component
        shifted = self._shift_spectrum(spectrum, factor)

        return shifted

    def _compute_shift_factor(self,
                              from_band: FrequencyBand,
                              to_band: FrequencyBand) -> float:
        """
        Get shift factor for band transition.

        Args:
            from_band: Source band
            to_band: Target band

        Returns:
            Multiplicative shift factor

        Raises:
            KeyError: If transition not defined
        """
        key = (from_band, to_band)
        if key not in self.shift_factors:
            raise KeyError(
                f"No shift factor defined for {from_band.name} → "
                f"{to_band.name}"
            )

        return self.shift_factors[key]

    def _compute_radial_coordinates(self,
                                   h: int,
                                   w: int,
                                   factor: float) -> Tuple[np.ndarray,
                                                          np.ndarray]:
        """
        Compute source coordinates for radial frequency shift.

        Args:
            h: Height of spectrum
            w: Width of spectrum
            factor: Frequency multiplication factor

        Returns:
            (source_y, source_x): Source coordinate grids for interpolation
        """
        center_h, center_w = h // 2, w // 2

        # Create coordinate grids for OUTPUT positions
        y_coords = np.arange(h).reshape(-1, 1) - center_h
        x_coords = np.arange(w).reshape(1, -1) - center_w

        # Compute radial distance and angle for each output pixel
        distances = np.sqrt(y_coords**2 + x_coords**2)
        angles = np.arctan2(y_coords, x_coords)

        # Inverse mapping: output pixel at distance D
        # should get value from input pixel at distance D/factor
        source_distances = distances / factor

        # Convert back to cartesian coordinates (source positions)
        source_y = source_distances * np.sin(angles) + center_h
        source_x = source_distances * np.cos(angles) + center_w

        return source_y, source_x

    def _interpolate_channel(self,
                           channel: np.ndarray,
                           source_y: np.ndarray,
                           source_x: np.ndarray) -> np.ndarray:
        """
        Interpolate single channel at source coordinates.

        Args:
            channel: 2D channel data (H, W)
            source_y: Y coordinates to sample from
            source_x: X coordinates to sample from

        Returns:
            Interpolated channel, same shape as input
        """
        from scipy.ndimage import map_coordinates

        h, w = channel.shape

        # Create coordinate array for map_coordinates
        coords = np.array([source_y.flatten(), source_x.flatten()])

        # Interpolate
        interpolated = map_coordinates(
            channel,
            coords,
            order=1,
            mode='constant',
            cval=0.0
        )

        return interpolated.reshape(h, w)

    def _shift_spectrum(self,
                       spectrum: np.ndarray,
                       factor: float) -> np.ndarray:
        """
        Apply frequency shift via radial remapping.

        In the spectrum representation, distance from center = frequency.
        To multiply frequency by N, we sample from distance/N in original.

        Uses inverse mapping: for each output pixel, find corresponding
        input pixel at scaled radial distance.

        Args:
            spectrum: Input spectrum (H, W, 4)
            factor: Frequency multiplication factor

        Returns:
            Shifted spectrum, same shape as input
        """
        h, w, c = spectrum.shape

        # Handle edge case: zero spectrum
        if np.allclose(spectrum, 0):
            return spectrum.copy()

        # Compute source coordinates for radial shift
        source_y, source_x = self._compute_radial_coordinates(h, w, factor)

        # Interpolate each quaternion component
        shifted = np.zeros_like(spectrum)
        for i in range(c):
            shifted[:, :, i] = self._interpolate_channel(
                spectrum[:, :, i],
                source_y,
                source_x
            )

        return shifted

    def _compute_frequency_error(self,
                                original: np.ndarray,
                                current: np.ndarray) -> float:
        """
        Compute mean absolute percentage error between spectrums.

        Args:
            original: Original spectrum
            current: Current spectrum after transformations

        Returns:
            MAPE error (0.0 if original is near-zero)
        """
        orig_mean = np.mean(np.abs(original))
        if orig_mean > 1e-10:
            return np.mean(np.abs(current - original)) / orig_mean
        return 0.0

    def _compute_structure_similarity(self,
                                     original: np.ndarray,
                                     current: np.ndarray) -> float:
        """
        Compute cosine similarity between spectrums.

        Args:
            original: Original spectrum
            current: Current spectrum after transformations

        Returns:
            Cosine similarity (1.0 if both near-zero)
        """
        flat_orig = original.flatten()
        flat_curr = current.flatten()

        norm_orig = np.linalg.norm(flat_orig)
        norm_curr = np.linalg.norm(flat_curr)

        if norm_orig > 1e-10 and norm_curr > 1e-10:
            return np.dot(flat_orig, flat_curr) / (norm_orig * norm_curr)
        return 1.0  # Both zero = perfect similarity

    def validate_roundtrip(self,
                          original: np.ndarray,
                          path: List[FrequencyBand]) -> Tuple[float, float]:
        """
        Validate roundtrip preservation.

        Applies sequence of band shifts and measures preservation:
        - Frequency error: Mean absolute percentage error
        - Structure similarity: Cosine similarity

        Args:
            original: Original spectrum (H, W, 4)
            path: Sequence of bands to traverse
                  (e.g., [LOW, MID, HIGH, MID, LOW])

        Returns:
            (frequency_error, structure_similarity)
            - frequency_error: MAPE, should be < 0.01 (1%)
            - structure_similarity: Cosine sim, should be > 0.95

        Raises:
            ValueError: If path has < 2 bands
        """
        if len(path) < 2:
            raise ValueError("Path must have at least 2 bands")

        # Apply all shifts
        current = original.copy()
        for i in range(len(path) - 1):
            current = self.shift_band(current, path[i], path[i+1])

        # Compute metrics
        freq_error = self._compute_frequency_error(original, current)
        structure_sim = self._compute_structure_similarity(original, current)

        return freq_error, structure_sim
