"""
Octave-based frequency extraction for memory system.

Extracts fundamental frequency and harmonics from text frequency spectra
and maps them to Gen/Res path parameters.
"""

import numpy as np
from typing import Dict


def extract_fundamental(freq_spectrum: np.ndarray) -> float:
    """
    Extract dominant frequency f₀ from frequency spectrum.

    Args:
        freq_spectrum: Frequency spectrum (height, width, channels)

    Returns:
        Fundamental frequency as float
    """
    # Handle both 2-channel (from text_to_frequency) and 4-channel inputs
    if freq_spectrum.ndim == 3:
        if freq_spectrum.shape[-1] == 2:
            # Complex representation (real, imag)
            magnitude = np.sqrt(freq_spectrum[..., 0]**2 + freq_spectrum[..., 1]**2)
        elif freq_spectrum.shape[-1] == 4:
            # RGBA or multi-channel, use first 2 as complex
            magnitude = np.sqrt(freq_spectrum[..., 0]**2 + freq_spectrum[..., 1]**2)
        else:
            # Single channel or other format
            magnitude = np.abs(freq_spectrum[..., 0])
    else:
        magnitude = np.abs(freq_spectrum)

    # Get dimensions
    height, width = magnitude.shape[:2]
    center_y, center_x = height // 2, width // 2

    # Find peak in magnitude spectrum
    peak_idx = np.unravel_index(np.argmax(magnitude), magnitude.shape)
    peak_y, peak_x = peak_idx[:2]

    # Convert to frequency (distance from center in normalized units)
    freq_y = (peak_y - center_y) / height
    freq_x = (peak_x - center_x) / width

    # Compute radial frequency
    fundamental = np.sqrt(freq_y**2 + freq_x**2)

    # Scale to reasonable range [0.5, 10.0] Hz equivalent
    fundamental = fundamental * 20.0 + 0.5

    return float(fundamental)


def _compute_magnitude(freq_spectrum: np.ndarray) -> np.ndarray:
    """Helper to compute magnitude spectrum from frequency data."""
    if freq_spectrum.ndim == 3:
        if freq_spectrum.shape[-1] >= 2:
            magnitude = np.sqrt(freq_spectrum[..., 0]**2 + freq_spectrum[..., 1]**2)
        else:
            magnitude = np.abs(freq_spectrum[..., 0])
    else:
        magnitude = np.abs(freq_spectrum)
    return magnitude


def _sample_harmonic_ring(magnitude: np.ndarray, center_y: int, center_x: int,
                         target_radius: float, height: int, width: int) -> float:
    """Sample magnitude at a circular ring for harmonic extraction."""
    if target_radius >= min(height, width) / 2:
        return 0.0

    angle = np.linspace(0, 2*np.pi, 32, endpoint=False)
    sample_y = center_y + target_radius * np.sin(angle)
    sample_x = center_x + target_radius * np.cos(angle)

    # Clip to valid indices
    sample_y = np.clip(sample_y.astype(int), 0, height-1)
    sample_x = np.clip(sample_x.astype(int), 0, width-1)

    return magnitude[sample_y, sample_x].mean()


def extract_harmonics(freq_spectrum: np.ndarray, fundamental: float) -> np.ndarray:
    """
    Extract harmonic coefficients at integer multiples of f₀.

    Args:
        freq_spectrum: Frequency spectrum (height, width, channels)
        fundamental: Fundamental frequency f₀

    Returns:
        Array of 10 harmonic amplitudes normalized to sum=1.0
    """
    magnitude = _compute_magnitude(freq_spectrum)
    height, width = magnitude.shape[:2]
    center_y, center_x = height // 2, width // 2

    # Create radial frequency grid
    y_coords, x_coords = np.ogrid[:height, :width]
    freq_y = (y_coords - center_y) / height
    freq_x = (x_coords - center_x) / width
    radial_freq = np.sqrt(freq_y**2 + freq_x**2)

    # Normalize fundamental to grid scale
    f0_normalized = (fundamental - 0.5) / 20.0

    # Extract energy at harmonic frequencies
    harmonics = np.zeros(10)
    bandwidth = 0.01

    for i in range(10):
        harmonic_freq = f0_normalized * (i + 1)
        mask = np.abs(radial_freq - harmonic_freq) < bandwidth

        if mask.any():
            harmonics[i] = magnitude[mask].mean()
        else:
            # Fallback: sample at ring
            target_radius = harmonic_freq * min(height, width) / 2
            harmonics[i] = _sample_harmonic_ring(magnitude, center_y, center_x,
                                                target_radius, height, width)

    # Normalize to sum = 1.0
    if harmonics.sum() > 0:
        harmonics /= harmonics.sum()
    else:
        harmonics = np.array([0.6] + [0.1]*3 + [0.01]*6)
        harmonics /= harmonics.sum()

    return harmonics.astype(np.float32)


def frequency_to_gen_params(octave_freq: float, harmonics: np.ndarray) -> Dict:
    """
    Generate gamma_params and iota_params for Gen path.

    Args:
        octave_freq: Base octave frequency
        harmonics: Array of harmonic coefficients

    Returns:
        Dict with 'gamma_params' and 'iota_params'
    """
    # Gamma parameters for carrier generation
    gamma_params = {
        'amplitude': float(1.0 + harmonics[0] * 0.5),  # Boost by fundamental
        'base_frequency': float(octave_freq),
        'envelope_sigma': 0.45,
        'num_harmonics': 12,
        'harmonic_decay': float(0.5 + harmonics[1] * 0.4),  # Second harmonic affects decay
        'initial_phase': 0.0
    }

    # Iota parameters for instantiation
    # Ensure exactly 10 harmonic coefficients
    harmonic_list = harmonics[:10].tolist() if len(harmonics) >= 10 else harmonics.tolist() + [0.0] * (10 - len(harmonics))

    iota_params = {
        'harmonic_coeffs': harmonic_list[:10],
        'global_amplitude': float(0.5 + np.sqrt(np.sum(harmonics**2))),  # RMS amplitude
        'frequency_range': float(octave_freq * 2.0)  # Octave range
    }

    return {
        'gamma_params': gamma_params,
        'iota_params': iota_params
    }


def frequency_to_res_params(octave_freq: float, harmonics: np.ndarray) -> Dict:
    """
    Generate epsilon_params and tau_params for Res path.

    Args:
        octave_freq: Base octave frequency
        harmonics: Array of harmonic coefficients

    Returns:
        Dict with 'epsilon_params' and 'tau_params'
    """
    # Compute spectral characteristics
    spectral_centroid = np.sum(harmonics * np.arange(1, len(harmonics) + 1)) / (harmonics.sum() + 1e-8)
    spectral_spread = np.sqrt(np.sum(harmonics * (np.arange(1, len(harmonics) + 1) - spectral_centroid)**2) / (harmonics.sum() + 1e-8))

    # Epsilon parameters for extraction/focus
    epsilon_params = {
        'extraction_rate': float(np.clip(octave_freq / 10.0, 0.1, 0.9)),
        'focus_sigma': float(0.2 + spectral_spread * 0.1),
        'threshold': float(0.1 * (1.0 - harmonics[0])),  # Lower threshold for distributed energy
        'preserve_peaks': True
    }

    # Tau parameters for projection/assertion
    tau_params = {
        'projection_strength': float(1.0 + spectral_centroid * 0.1),
        'eigen_components': int(min(8, max(3, int(spectral_spread * 4)))),
        'regularization': float(0.01 * (1.0 + harmonics[-1])),  # More regularization for high harmonics
        'kernel_size': 5
    }

    return {
        'epsilon_params': epsilon_params,
        'tau_params': tau_params
    }


def extract_fundamental_from_image(image: np.ndarray) -> float:
    """
    Extract fundamental frequency from image via 2D FFT.

    Args:
        image: RGB/grayscale image (H, W, C) or (H, W)

    Returns:
        Fundamental frequency f₀
    """
    # Convert to grayscale if RGB
    if image.ndim == 3:
        gray = np.mean(image, axis=2)
    else:
        gray = image

    # 2D FFT
    fft = np.fft.fft2(gray)
    fft_shifted = np.fft.fftshift(fft)
    magnitude = np.abs(fft_shifted)

    # Get dimensions
    h, w = magnitude.shape
    center_y, center_x = h // 2, w // 2

    # Mask out center (DC component)
    mask = np.ones_like(magnitude, dtype=bool)
    mask[center_y-5:center_y+5, center_x-5:center_x+5] = False
    magnitude_masked = magnitude * mask

    # Find peak
    peak_idx = np.unravel_index(np.argmax(magnitude_masked), magnitude.shape)
    peak_y, peak_x = peak_idx

    # Convert to frequency
    freq_y = (peak_y - center_y) / h
    freq_x = (peak_x - center_x) / w
    fundamental = np.sqrt(freq_y**2 + freq_x**2)

    # Scale to [0.5, 10.0] Hz range (matching text)
    fundamental = fundamental * 20.0 + 0.5

    return float(fundamental)


def extract_harmonics_from_image(image: np.ndarray, fundamental: float) -> np.ndarray:
    """
    Extract harmonic coefficients from image frequency spectrum.

    Args:
        image: RGB/grayscale image (H, W, C) or (H, W)
        fundamental: Fundamental frequency f₀

    Returns:
        Array of 10 harmonic amplitudes normalized to sum=1.0
    """
    # Convert to grayscale if RGB
    if image.ndim == 3:
        gray = np.mean(image, axis=2)
    else:
        gray = image

    # 2D FFT
    fft = np.fft.fft2(gray)
    fft_shifted = np.fft.fftshift(fft)
    magnitude = np.abs(fft_shifted)

    h, w = magnitude.shape
    center_y, center_x = h // 2, w // 2

    # Create radial frequency grid
    y_coords, x_coords = np.ogrid[:h, :w]
    freq_y = (y_coords - center_y) / h
    freq_x = (x_coords - center_x) / w
    radial_freq = np.sqrt(freq_y**2 + freq_x**2)

    # Normalize fundamental to grid scale
    f0_normalized = (fundamental - 0.5) / 20.0

    # Extract energy at harmonic frequencies
    harmonics = np.zeros(10)
    bandwidth = 0.01

    for i in range(10):
        harmonic_freq = f0_normalized * (i + 1)
        mask = np.abs(radial_freq - harmonic_freq) < bandwidth

        if mask.any():
            harmonics[i] = magnitude[mask].mean()

    # Normalize to sum = 1.0
    if harmonics.sum() > 0:
        harmonics /= harmonics.sum()
    else:
        harmonics = np.array([0.6] + [0.1]*3 + [0.01]*6)
        harmonics /= harmonics.sum()

    return harmonics.astype(np.float32)


def extract_fundamental_from_audio(audio: np.ndarray, sample_rate: int = 16000) -> float:
    """
    Extract fundamental frequency from audio via 1D FFT.

    Args:
        audio: Audio samples (mono)
        sample_rate: Sample rate in Hz

    Returns:
        Fundamental frequency f₀
    """
    # 1D FFT
    fft = np.fft.fft(audio)
    magnitude = np.abs(fft[:len(fft)//2])  # Positive frequencies only
    freqs = np.fft.fftfreq(len(audio), 1/sample_rate)[:len(fft)//2]

    # Find peak (exclude DC component)
    magnitude[0] = 0
    peak_idx = np.argmax(magnitude)
    fundamental = freqs[peak_idx] if peak_idx > 0 else 50.0

    # Normalize to [0.5, 10.0] Hz range
    fundamental = np.clip(fundamental / 100.0, 0.5, 10.0)

    return float(fundamental)


def extract_harmonics_from_audio(audio: np.ndarray, fundamental: float,
                                sample_rate: int = 16000) -> np.ndarray:
    """
    Extract harmonic coefficients from audio spectrum.

    Args:
        audio: Audio samples (mono)
        fundamental: Fundamental frequency f₀
        sample_rate: Sample rate in Hz

    Returns:
        Array of 10 harmonic amplitudes normalized to sum=1.0
    """
    # 1D FFT
    fft = np.fft.fft(audio)
    magnitude = np.abs(fft[:len(fft)//2])
    freqs = np.fft.fftfreq(len(audio), 1/sample_rate)[:len(fft)//2]

    # Denormalize fundamental to actual frequency
    f0_actual = fundamental * 100.0  # Hz

    # Extract harmonics
    harmonics = np.zeros(10)

    for i in range(10):
        target_freq = f0_actual * (i + 1)
        # Find closest frequency bin
        closest_idx = np.argmin(np.abs(freqs - target_freq))

        if closest_idx < len(magnitude):
            # Average around the target
            start = max(0, closest_idx - 2)
            end = min(len(magnitude), closest_idx + 3)
            harmonics[i] = magnitude[start:end].mean()

    # Normalize to sum = 1.0
    if harmonics.sum() > 0:
        harmonics /= harmonics.sum()
    else:
        harmonics = np.array([0.6] + [0.1]*3 + [0.01]*6)
        harmonics /= harmonics.sum()

    return harmonics.astype(np.float32)


# Test functions
if __name__ == "__main__":
    import sys
    sys.path.append('/home/persist/alembic/genesis')

    from src.memory.frequency_field import TextFrequencyAnalyzer

    # Create analyzer and get frequency spectrum
    analyzer = TextFrequencyAnalyzer(512, 512)
    freq_spectrum, _ = analyzer.analyze("test text for octave frequency extraction")

    # Test extract_fundamental
    f0 = extract_fundamental(freq_spectrum)
    print(f"✓ f0: {f0:.3f}")

    # Test extract_harmonics
    harmonics = extract_harmonics(freq_spectrum, f0)
    print(f"✓ harmonics shape: {harmonics.shape}")
    print(f"✓ harmonics sum: {harmonics.sum():.3f}")
    print(f"✓ harmonics: {harmonics[:5].round(3)}")  # Show first 5

    # Test frequency_to_gen_params
    gen_params = frequency_to_gen_params(f0, harmonics)
    print(f"✓ gen_params keys: {list(gen_params.keys())}")
    print(f"✓ gamma_params keys: {list(gen_params['gamma_params'].keys())}")
    print(f"✓ iota harmonic_coeffs length: {len(gen_params['iota_params']['harmonic_coeffs'])}")

    # Test frequency_to_res_params
    res_params = frequency_to_res_params(f0, harmonics)
    print(f"✓ res_params keys: {list(res_params.keys())}")
    print(f"✓ epsilon_params keys: {list(res_params['epsilon_params'].keys())}")
    print(f"✓ tau_params keys: {list(res_params['tau_params'].keys())}")