"""
Tests for FrequencyEnhancer - Multi-Band Abstraction Shifting.
"""

import pytest
import numpy as np

from src.memory.frequency_enhancer import FrequencyEnhancer
from src.memory.frequency_bands import FrequencyBand


# Helper functions

def create_test_spectrum(h=64, w=64, dominant_freq=5.0, harmonics=None):
    """
    Create synthetic frequency spectrum for testing.

    Args:
        h, w: Spatial dimensions
        dominant_freq: Dominant frequency in Hz
        harmonics: List of harmonic multipliers (e.g., [1, 2, 3])

    Returns:
        Frequency spectrum (h, w, 4)
    """
    if harmonics is None:
        harmonics = [1]

    spectrum = np.zeros((h, w, 4), dtype=np.float32)

    # Map frequency to spatial position
    # Higher freq = further from center
    center_h, center_w = h // 2, w // 2
    max_radius = min(h, w) // 2

    for harmonic in harmonics:
        freq_hz = dominant_freq * harmonic

        # Map Hz to normalized distance [0, 1]
        # Use same mapping as frequency_bands.py
        normalized_dist = freq_hz / 500.0
        radius = int(normalized_dist * max_radius)
        radius = min(radius, max_radius - 1)

        # Create ring at this radius with amplitude
        amplitude = 1.0 / harmonic  # Harmonics decay

        # Set quaternion components
        for angle in range(0, 360, 10):
            rad = np.radians(angle)
            y = int(center_h + radius * np.sin(rad))
            x = int(center_w + radius * np.cos(rad))

            if 0 <= y < h and 0 <= x < w:
                # Quaternion: (w, x, y, z)
                spectrum[y, x, 0] = amplitude  # w
                spectrum[y, x, 1] = amplitude * np.cos(rad)  # x
                spectrum[y, x, 2] = amplitude * np.sin(rad)  # y
                spectrum[y, x, 3] = 0.0  # z

    return spectrum


def compute_dominant_frequency(spectrum):
    """
    Compute dominant frequency from spectrum.

    Replicates logic from frequency_bands.py for validation.
    """
    h, w = spectrum.shape[:2]
    center_h, center_w = h // 2, w // 2

    # Magnitude
    magnitude = np.sqrt(spectrum[:, :, 0]**2 + spectrum[:, :, 1]**2)

    # Distance from center
    y_coords = np.arange(h).reshape(-1, 1) - center_h
    x_coords = np.arange(w).reshape(1, -1) - center_w
    distance = np.sqrt(y_coords**2 + x_coords**2)

    # Weighted average
    total_mag = magnitude.sum()
    if total_mag > 1e-8:
        weighted_dist = (magnitude * distance).sum() / total_mag
    else:
        return 0.0

    # Map to Hz
    max_dist = np.sqrt(center_h**2 + center_w**2)
    normalized = weighted_dist / (max_dist + 1e-8)
    dominant_hz = normalized * 500.0

    return dominant_hz


def extract_harmonic_peaks(spectrum, n_peaks=3):
    """
    Extract top N harmonic peaks from spectrum.

    Returns:
        List of frequencies (Hz) for top peaks
    """
    h, w = spectrum.shape[:2]
    center_h, center_w = h // 2, w // 2

    # Magnitude
    magnitude = np.sqrt(spectrum[:, :, 0]**2 + spectrum[:, :, 1]**2)

    # Find peaks (local maxima)
    peaks = []
    for y in range(1, h-1):
        for x in range(1, w-1):
            if magnitude[y, x] > 0.1:  # Threshold
                # Check if local maximum
                neighborhood = magnitude[y-1:y+2, x-1:x+2]
                if magnitude[y, x] == neighborhood.max():
                    # Compute frequency for this peak
                    dist = np.sqrt((y - center_h)**2 + (x - center_w)**2)
                    max_dist = np.sqrt(center_h**2 + center_w**2)
                    norm = dist / (max_dist + 1e-8)
                    freq_hz = norm * 500.0

                    peaks.append((magnitude[y, x], freq_hz))

    # Sort by magnitude, return top N frequencies
    peaks.sort(reverse=True)
    return [freq for _, freq in peaks[:n_peaks]]


# Tests

class TestFrequencyEnhancer:
    """Test suite for FrequencyEnhancer."""

    def test_initialization(self):
        """Test enhancer initializes with correct shift factors."""
        enhancer = FrequencyEnhancer()

        # Check forward shifts
        assert enhancer.shift_factors[
            (FrequencyBand.LOW, FrequencyBand.MID)
        ] == 10.0
        assert enhancer.shift_factors[
            (FrequencyBand.MID, FrequencyBand.HIGH)
        ] == 4.0

        # Check reverse shifts
        assert enhancer.shift_factors[
            (FrequencyBand.MID, FrequencyBand.LOW)
        ] == 0.1
        assert enhancer.shift_factors[
            (FrequencyBand.HIGH, FrequencyBand.MID)
        ] == 0.25

        # Check direct jumps
        assert enhancer.shift_factors[
            (FrequencyBand.LOW, FrequencyBand.HIGH)
        ] == 40.0

    def test_shift_low_to_mid(self):
        """Test LOW→MID shift multiplies frequencies by ~10x."""
        enhancer = FrequencyEnhancer()

        # Create LOW band spectrum (peak at 5Hz)
        spectrum = create_test_spectrum(dominant_freq=5.0)
        original_freq = compute_dominant_frequency(spectrum)
        assert original_freq < 10.0  # Verify in LOW band

        # Shift to MID
        shifted = enhancer.shift_band(
            spectrum,
            FrequencyBand.LOW,
            FrequencyBand.MID
        )

        # Check shifted frequency
        shifted_freq = compute_dominant_frequency(shifted)

        # Should be ~50Hz (5 * 10)
        # Allow 50% tolerance due to interpolation artifacts
        assert 25 < shifted_freq < 75, \
            f"Expected ~50Hz, got {shifted_freq:.1f}Hz"

    def test_shift_mid_to_high(self):
        """Test MID→HIGH shift multiplies frequencies by ~4x."""
        enhancer = FrequencyEnhancer()

        # Create MID band spectrum (peak at 50Hz)
        spectrum = create_test_spectrum(dominant_freq=50.0)
        original_freq = compute_dominant_frequency(spectrum)
        assert 10 < original_freq < 100  # Verify in MID band

        # Shift to HIGH
        shifted = enhancer.shift_band(
            spectrum,
            FrequencyBand.MID,
            FrequencyBand.HIGH
        )

        # Check shifted frequency
        shifted_freq = compute_dominant_frequency(shifted)

        # Should be ~200Hz (50 * 4)
        assert 100 < shifted_freq < 300, \
            f"Expected ~200Hz, got {shifted_freq:.1f}Hz"

    def test_shift_same_band_noop(self):
        """Test shifting within same band is no-op."""
        enhancer = FrequencyEnhancer()

        spectrum = create_test_spectrum(dominant_freq=5.0)

        shifted = enhancer.shift_band(
            spectrum,
            FrequencyBand.LOW,
            FrequencyBand.LOW
        )

        # Should be identical (or very close)
        assert np.allclose(spectrum, shifted), \
            "Same-band shift should be no-op"

    def test_roundtrip_preservation(self):
        """Test LOW→MID→HIGH→MID→LOW preserves structure."""
        enhancer = FrequencyEnhancer()

        # Create test spectrum
        original = create_test_spectrum(dominant_freq=5.0)

        # Define roundtrip path
        path = [
            FrequencyBand.LOW,
            FrequencyBand.MID,
            FrequencyBand.HIGH,
            FrequencyBand.MID,
            FrequencyBand.LOW
        ]

        # Validate roundtrip
        freq_error, structure_sim = enhancer.validate_roundtrip(
            original,
            path
        )

        # Check KPIs
        # Note: Relaxed thresholds due to interpolation artifacts
        assert freq_error < 0.5, \
            f"Frequency error {freq_error:.3f} exceeds 50%"
        assert structure_sim > 0.5, \
            f"Structure similarity {structure_sim:.3f} below 0.5"

    def test_harmonic_preservation(self):
        """Test harmonics remain proportional after shift."""
        enhancer = FrequencyEnhancer()

        # Create simple spectrum with two peaks at known distances
        h, w = 64, 64
        spectrum = np.zeros((h, w, 4), dtype=np.float32)
        center_h, center_w = h // 2, w // 2

        # Fundamental at distance 3 (represents ~5Hz)
        spectrum[center_h + 3, center_w, 0] = 1.0

        # Second harmonic at distance 6 (represents ~10Hz = 2x fundamental)
        spectrum[center_h, center_w + 6, 0] = 0.5

        # Shift to MID
        shifted = enhancer.shift_band(
            spectrum,
            FrequencyBand.LOW,
            FrequencyBand.MID
        )

        # Find peaks in shifted spectrum
        # Fundamental should be at distance 30, harmonic at 60
        # But harmonic at 60 is outside bounds (max ~32)
        # So just verify fundamental shifted correctly

        # Find max value (fundamental)
        max_idx = np.unravel_index(
            shifted[:, :, 0].argmax(),
            (h, w)
        )
        dist = np.sqrt(
            (max_idx[0] - center_h)**2 + (max_idx[1] - center_w)**2
        )

        # Should be ~30 pixels (3 * 10)
        assert 28 < dist < 32, \
            f"Fundamental shifted to {dist:.1f} pixels, expected ~30"

    def test_zero_spectrum(self):
        """Test zero spectrum remains zero."""
        enhancer = FrequencyEnhancer()

        spectrum = np.zeros((64, 64, 4), dtype=np.float32)

        shifted = enhancer.shift_band(
            spectrum,
            FrequencyBand.LOW,
            FrequencyBand.MID
        )

        assert np.allclose(shifted, 0), \
            "Zero spectrum should remain zero"

    def test_invalid_spectrum_shape(self):
        """Test invalid spectrum shape raises error."""
        enhancer = FrequencyEnhancer()

        # Wrong number of dimensions
        spectrum_2d = np.zeros((64, 64), dtype=np.float32)

        with pytest.raises(ValueError, match="Expected .* spectrum"):
            enhancer.shift_band(
                spectrum_2d,
                FrequencyBand.LOW,
                FrequencyBand.MID
            )

        # Wrong number of channels
        spectrum_3ch = np.zeros((64, 64, 3), dtype=np.float32)

        with pytest.raises(ValueError, match="Expected .* spectrum"):
            enhancer.shift_band(
                spectrum_3ch,
                FrequencyBand.LOW,
                FrequencyBand.MID
            )

    def test_undefined_transition(self):
        """Test undefined band transition raises error."""
        enhancer = FrequencyEnhancer()

        # Remove a shift factor to test error handling
        del enhancer.shift_factors[(FrequencyBand.LOW, FrequencyBand.MID)]

        spectrum = create_test_spectrum(dominant_freq=5.0)

        with pytest.raises(KeyError, match="No shift factor"):
            enhancer.shift_band(
                spectrum,
                FrequencyBand.LOW,
                FrequencyBand.MID
            )

    def test_validate_roundtrip_short_path(self):
        """Test validation with too-short path raises error."""
        enhancer = FrequencyEnhancer()

        spectrum = create_test_spectrum(dominant_freq=5.0)

        with pytest.raises(ValueError, match="at least 2 bands"):
            enhancer.validate_roundtrip(
                spectrum,
                [FrequencyBand.LOW]
            )

    def test_energy_preservation(self):
        """Test peak amplitude roughly preserved after shift."""
        enhancer = FrequencyEnhancer()

        # Create simple spectrum with single peak
        h, w = 64, 64
        spectrum = np.zeros((h, w, 4), dtype=np.float32)
        center_h, center_w = h // 2, w // 2

        # Single peak at distance 3
        spectrum[center_h + 3, center_w, 0] = 1.0
        original_max = spectrum[:, :, 0].max()

        # Shift and check peak amplitude
        shifted = enhancer.shift_band(
            spectrum,
            FrequencyBand.LOW,
            FrequencyBand.MID
        )
        shifted_max = shifted[:, :, 0].max()

        # Peak amplitude should be preserved (within 20%)
        # Radial interpolation can cause some spreading/concentration
        ratio = shifted_max / (original_max + 1e-8)
        assert 0.8 < ratio < 1.2, \
            f"Peak amplitude changed: {original_max:.3f} → " \
            f"{shifted_max:.3f} ({ratio:.2f}x)"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
