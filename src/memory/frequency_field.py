"""
Multi-Modal Frequency Field Mapping - Convert text/audio/image to frequency spectra.

Key Principle: Frequency → Field Convergence
- Phonetic "ah" → formant frequencies (F1=730Hz, F2=1090Hz)
- Graphemic "A" → semantic Unicode group → frequency bins
- Audio waveform → FFT → fundamental + harmonics
- Image pixels → 2D FFT → dominant frequency

Multi-modal convergence: "A" (text) + "ah" (audio) + visual "A" → SAME f₀
"""

import numpy as np
from typing import Dict, Tuple, Optional
import os
from pathlib import Path
import cv2


class FrequencyFieldMapper:
    """Base class for frequency field mapping."""

    def __init__(self, width: int = 512, height: int = 512):
        self.width = width
        self.height = height

    def to_frequency_spectrum(self, data) -> np.ndarray:
        """Convert data to frequency spectrum (H, W, 2)."""
        raise NotImplementedError


class AudioFrequencyMapper(FrequencyFieldMapper):
    """Map audio waveform to frequency spectrum via FFT."""

    def to_frequency_spectrum(self, audio_path: str, sr: int = 22050) -> np.ndarray:
        """Load audio → FFT → frequency spectrum."""
        try:
            import librosa
        except ImportError:
            # If librosa not available, return zero spectrum
            print("Warning: librosa not installed. Audio processing unavailable.")
            return np.zeros((self.height, self.width, 2), dtype=np.float32)

        if not os.path.exists(audio_path):
            print(f"Warning: Audio file not found: {audio_path}")
            return np.zeros((self.height, self.width, 2), dtype=np.float32)

        # Load audio
        try:
            y, sr = librosa.load(audio_path, sr=sr)
        except Exception as e:
            print(f"Warning: Failed to load audio: {e}")
            return np.zeros((self.height, self.width, 2), dtype=np.float32)

        # Compute spectrogram using STFT for better 2D representation
        D = librosa.stft(y, n_fft=2048, hop_length=512)
        magnitude = np.abs(D)
        phase = np.angle(D)

        # Resize to target dimensions
        from scipy.ndimage import zoom

        # Calculate zoom factors
        zoom_y = self.height / magnitude.shape[0]
        zoom_x = self.width / magnitude.shape[1]

        # Resize magnitude and phase
        magnitude_resized = zoom(magnitude, (zoom_y, zoom_x), order=1)
        phase_resized = zoom(phase, (zoom_y, zoom_x), order=1)

        # Stack as complex spectrum
        spectrum = np.stack([magnitude_resized, phase_resized], axis=-1)

        # Normalize
        max_val = np.abs(spectrum).max()
        if max_val > 0:
            spectrum /= max_val

        return spectrum.astype(np.float32)


class ImageFrequencyMapper(FrequencyFieldMapper):
    """Map image to frequency spectrum via 2D FFT."""

    def to_frequency_spectrum(self, image_path: str) -> np.ndarray:
        """Load image → 2D FFT → frequency spectrum."""
        try:
            from PIL import Image
        except ImportError:
            print("Warning: PIL not installed. Image processing unavailable.")
            return np.zeros((self.height, self.width, 2), dtype=np.float32)

        if not os.path.exists(image_path):
            print(f"Warning: Image file not found: {image_path}")
            return np.zeros((self.height, self.width, 2), dtype=np.float32)

        try:
            # Load image as grayscale
            img = Image.open(image_path).convert('L')
            img = img.resize((self.width, self.height))
            img_array = np.array(img, dtype=np.float32) / 255.0
        except Exception as e:
            print(f"Warning: Failed to load image: {e}")
            return np.zeros((self.height, self.width, 2), dtype=np.float32)

        # 2D FFT
        fft = np.fft.fft2(img_array)
        fft_shifted = np.fft.fftshift(fft)

        magnitude = np.abs(fft_shifted)
        phase = np.angle(fft_shifted)

        spectrum = np.stack([magnitude, phase], axis=-1)

        # Normalize
        max_val = np.abs(spectrum).max()
        if max_val > 0:
            spectrum /= max_val

        return spectrum.astype(np.float32)


class TextFrequencyAnalyzer(FrequencyFieldMapper):
    """STFT-based text → frequency spectrum mapping.

    Uses the SAME mathematical foundation (FFT) as audio/image:
    - Text is a 1D sequence (like audio)
    - Apply sliding window FFT (STFT) on character codes
    - Produces diverse, deterministic, reversible spectrums
    """

    def __init__(self, width: int = 512, height: int = 512,
                 window_size: int = 128, hop_length: int = 32):
        super().__init__(width, height)
        self.window_size = window_size
        self.hop_length = hop_length
        self.TAU = 2.0 * np.pi

    def text_to_frequency(self, text: str) -> Tuple[np.ndarray, np.ndarray]:
        """Convert text to frequency spectrum using STFT (dual-path).

        Process:
        1. Text → Character codes (Unicode)
        2. Sliding window FFT → STFT matrix
        3. Extract magnitude and phase
        4. STORE native STFT (for lossless reconstruction)
        5. Resize to target shape (for proto-identity learning)

        Returns:
            Tuple of (resized_spectrum, native_stft):
            - resized_spectrum: (H, W, 2) for proto-identity [magnitude, phase]
            - native_stft: (H', W', 2) native dims for lossless ISTFT
        """
        if not text or len(text) == 0:
            zero_spectrum = np.zeros((self.height, self.width, 2), dtype=np.float32)
            return zero_spectrum, zero_spectrum

        # Step 1: Text → Character codes (Unicode)
        char_codes = np.array([ord(c) for c in text], dtype=np.float32)

        # Step 2: Handle short texts (pad if needed)
        if len(char_codes) < self.window_size:
            char_codes = np.pad(char_codes,
                               (0, self.window_size - len(char_codes)))

        # Step 3: Sliding window STFT
        num_windows = (len(char_codes) - self.window_size) // self.hop_length + 1
        stft_matrix = []

        for i in range(num_windows):
            start = i * self.hop_length
            end = start + self.window_size
            window = char_codes[start:end]

            # Apply Hamming window for smoother edges
            hamming = np.hamming(self.window_size)
            windowed = window * hamming

            # Apply FFT to window
            fft_result = np.fft.fft(windowed)
            stft_matrix.append(fft_result)

        # Shape: (num_windows, fft_bins) → transpose to (fft_bins, num_windows)
        stft_matrix = np.array(stft_matrix).T

        # Step 4: Extract magnitude and phase
        magnitude = np.abs(stft_matrix)
        phase = np.angle(stft_matrix)

        # Step 4.5: STORE NATIVE STFT (for lossless reconstruction)
        native_stft = np.stack([magnitude, phase], axis=-1).astype(np.float32)

        # Step 5: Resize to target shape (match audio/image pipeline)
        # Ensure we have valid dimensions
        if magnitude.size == 0:
            zero_spectrum = np.zeros((self.height, self.width, 2), dtype=np.float32)
            return zero_spectrum, zero_spectrum

        magnitude_resized = cv2.resize(magnitude, (self.width, self.height))
        phase_resized = cv2.resize(phase, (self.width, self.height))

        # Step 6: Normalize magnitude while preserving phase
        max_mag = magnitude_resized.max()
        if max_mag > 0:
            magnitude_resized /= max_mag

        # Step 7: Stack as (H, W, 2)
        resized_spectrum = np.stack([magnitude_resized, phase_resized], axis=-1)

        return resized_spectrum.astype(np.float32), native_stft

    def from_frequency_spectrum(self, spectrum: np.ndarray,
                               native_stft: Optional[np.ndarray] = None,
                               original_length: Optional[int] = None) -> str:
        """Reconstruct text from frequency spectrum using inverse STFT.

        Args:
            spectrum: (H, W, 2) array where [:,:,0] = magnitude, [:,:,1] = phase
            native_stft: Optional (H', W', 2) native STFT for lossless reconstruction
            original_length: Original text length (for trimming)

        Returns:
            Reconstructed text (lossless if native_stft provided, lossy otherwise)
        """
        # Step 1: Extract magnitude and phase (PREFER NATIVE STFT)
        if native_stft is not None:
            magnitude = native_stft[:, :, 0]
            phase = native_stft[:, :, 1]
        else:
            # Fallback to resized spectrum (lossy path)
            magnitude = spectrum[:, :, 0]
            phase = spectrum[:, :, 1]

        # Step 2: Reconstruct complex STFT matrix
        stft_matrix = magnitude * np.exp(1j * phase)

        # Step 3: Handle native vs resized STFT paths
        if native_stft is not None:
            # LOSSLESS PATH: Use native STFT dimensions directly
            # Native STFT is (fft_bins, num_windows, 2)
            fft_bins, num_windows = stft_matrix.shape[:2]
            # Transpose to (num_windows, fft_bins)
            stft_transposed = stft_matrix.T
        else:
            # LOSSY PATH: Resize from 512x512 to estimated STFT dimensions
            # Step 4: Determine reconstruction dimensions
            # Estimate the number of windows based on expected text length
            if original_length:
                estimated_windows = (original_length - self.window_size) // self.hop_length + 1
                num_windows = min(estimated_windows, self.width)
            else:
                num_windows = 32  # Default to reasonable number

            fft_bins = self.window_size

            # Denormalize magnitude (for lossy path)
            magnitude_denorm = magnitude * 1000.0
            stft_matrix_denorm = magnitude_denorm * np.exp(1j * phase)

            # Downsample from full spectrum to expected STFT size
            # cv2.resize expects (width, height) which maps to (num_windows, fft_bins)
            if num_windows > 0 and fft_bins > 0:
                stft_real = cv2.resize(
                    stft_matrix_denorm.real,
                    (num_windows, fft_bins),  # cv2.resize: (width, height)
                    interpolation=cv2.INTER_LINEAR
                )
                stft_imag = cv2.resize(
                    stft_matrix_denorm.imag,
                    (num_windows, fft_bins),  # cv2.resize: (width, height)
                    interpolation=cv2.INTER_LINEAR
                )
                stft_resized = stft_real + 1j * stft_imag
            else:
                # Fallback: use a portion of the spectrum directly
                stft_resized = stft_matrix_denorm[:min(fft_bins, 512), :min(num_windows, 512)]

            # Result is already (fft_bins, num_windows) from cv2.resize
            # Transpose to (num_windows, fft_bins) for ISTFT
            stft_transposed = stft_resized.T

        # Step 5: Inverse STFT (overlap-add method)
        signal_length = (num_windows - 1) * self.hop_length + self.window_size
        reconstructed = np.zeros(signal_length, dtype=np.complex128)
        window_count = np.zeros(signal_length)

        hamming = np.hamming(self.window_size)

        for i in range(num_windows):
            fft_window = stft_transposed[i, :]

            # Inverse FFT
            time_window = np.fft.ifft(fft_window, n=self.window_size)

            # Undo Hamming window (divide by it to reverse the forward multiplication)
            # Add small epsilon to avoid division by zero
            time_window_unwindowed = time_window.real / (hamming + 1e-8)

            # Overlap-add
            start = i * self.hop_length
            end = start + self.window_size
            if end <= signal_length:
                reconstructed[start:end] += time_window_unwindowed
                window_count[start:end] += 1  # Simple counting for averaging

        # Average overlapping regions
        with np.errstate(divide='ignore', invalid='ignore'):
            reconstructed = np.where(
                window_count > 0,
                reconstructed / window_count,
                reconstructed
            ).real

        # Step 6: Convert to character codes
        char_codes = np.round(np.real(reconstructed)).astype(int)

        # Step 7: Ensure valid character range
        # Map to printable ASCII (32-126) with some tolerance
        char_codes = np.where(char_codes < 32, 32, char_codes)  # Replace control chars with space
        char_codes = np.where(char_codes > 126, 63, char_codes)  # Replace high chars with '?'

        # Step 8: Convert to text
        text_chars = []
        for i, code in enumerate(char_codes):
            if original_length and i >= original_length:
                break
            try:
                char = chr(int(code))
                text_chars.append(char)
            except (ValueError, OverflowError):
                text_chars.append('?')

        text = ''.join(text_chars)

        # Trim to original length if needed
        if original_length and len(text) > original_length:
            text = text[:original_length]

        return text

    def frequency_to_params(self, freq_spectrum: np.ndarray) -> Dict:
        """Map frequency spectrum to morphism parameters.

        Maintains compatibility with existing code.
        """
        # Compute magnitude spectrum
        magnitude = np.sqrt(freq_spectrum[..., 0]**2 + freq_spectrum[..., 1]**2)

        # Find dominant frequency (peak in magnitude)
        center_y, center_x = self.height // 2, self.width // 2
        y_coords, x_coords = np.ogrid[:self.height, :self.width]
        freq_y = (y_coords - center_y) / self.height
        freq_x = (x_coords - center_x) / self.width

        # Weight by magnitude to find center of mass
        total_mag = magnitude.sum()
        if total_mag > 0:
            weighted_y = (magnitude * freq_y).sum() / total_mag
            weighted_x = (magnitude * freq_x).sum() / total_mag
            dominant_freq = np.sqrt(weighted_y**2 + weighted_x**2) * 10.0 + 1.0
        else:
            dominant_freq = 2.0

        # Compute harmonic structure (radial profile)
        radial_dist = np.sqrt(freq_y**2 + freq_x**2)
        num_bins = 16
        harmonic_coeffs = []

        for i in range(num_bins):
            bin_min = i / num_bins * 0.5
            bin_max = (i + 1) / num_bins * 0.5
            mask = (radial_dist >= bin_min) & (radial_dist < bin_max)
            bin_energy = magnitude[mask].mean() if mask.any() else 0.0
            harmonic_coeffs.append(float(bin_energy))

        # Normalize harmonics
        max_harmonic = max(harmonic_coeffs) if harmonic_coeffs else 1.0
        if max_harmonic > 0:
            harmonic_coeffs = [h / max_harmonic for h in harmonic_coeffs]

        # Compute phase distribution for modulation
        phase = np.arctan2(freq_spectrum[..., 1], freq_spectrum[..., 0])
        phase_variance = phase.var()

        # Gamma parameters (carrier generation)
        gamma_params = {
            'base_frequency': float(dominant_freq),
            'initial_phase': float(phase.mean()),
            'amplitude': float(magnitude.mean() * 2.0 + 0.5),
            'envelope_sigma': 0.45,
            'num_harmonics': 12,
            'harmonic_decay': 0.75
        }

        # Iota parameters (instantiation)
        # Ensure we have exactly 10 coefficients for iota
        harmonic_coeffs_10 = (harmonic_coeffs[:10] if len(harmonic_coeffs) >= 10
                              else harmonic_coeffs + [0.0] * (10 - len(harmonic_coeffs)))
        iota_params = {
            'harmonic_coeffs': harmonic_coeffs_10[:10],  # Exactly 10 coefficients
            'global_amplitude': float(magnitude.mean() + 0.5),
            'frequency_range': float(dominant_freq)
        }

        # Tau parameters (projection/assertion)
        energy_concentration = (magnitude**2).sum() / (magnitude.size + 1e-8)
        tau_params = {
            'projection_strength': float(np.clip(energy_concentration * 10, 0.5, 2.0)),
            'eigen_components': 8,
            'regularization': 0.01,
            'kernel_size': 5
        }

        # Epsilon parameters (extraction/focus)
        spectral_entropy = -np.sum(magnitude * np.log(magnitude + 1e-8))
        epsilon_params = {
            'extraction_rate': float(np.clip(spectral_entropy / 100, 0.1, 0.9)),
            'focus_sigma': 0.3,
            'threshold': 0.1,
            'preserve_peaks': True
        }

        return {
            'gamma_params': gamma_params,
            'iota_params': iota_params,
            'tau_params': tau_params,
            'epsilon_params': epsilon_params
        }

    def analyze(self, text: str) -> Tuple[np.ndarray, Dict]:
        """Full analysis: text → frequency spectrum → morphism parameters."""
        freq_spectrum, _ = self.text_to_frequency(text)  # Use resized for params
        params = self.frequency_to_params(freq_spectrum)
        return freq_spectrum, params