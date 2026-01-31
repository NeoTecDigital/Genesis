"""
Proto-identity generator from synchronized Kuramoto phases.

Converts synchronized phase oscillators into spatial interference fields
that serve as proto-identities for the Oracle memory system.
"""

import numpy as np
import hashlib
from typing import Tuple, Optional, Union


class ProtoIdentityGenerator:
    """
    Converts synchronized phases to spatial interference field Φ(x,y).

    Each oscillator creates a wave: A_j · e^{i(k_j·r + θ_j)}
    Superposition creates the proto-identity field.
    """

    def __init__(
        self,
        resolution: Tuple[int, int] = (512, 512),
        seed: int = 42,
        wavelength_range: Tuple[float, float] = (10.0, 100.0)
    ):
        """
        Initialize proto-identity generator.

        Args:
            resolution: Spatial grid size (width, height)
            seed: Random seed for deterministic wave vectors
            wavelength_range: Range of wavelengths for wave vectors
        """
        if resolution[0] <= 0 or resolution[1] <= 0:
            raise ValueError("Resolution must be positive")
        if wavelength_range[0] <= 0 or wavelength_range[1] <= wavelength_range[0]:
            raise ValueError("Invalid wavelength range")

        self.resolution = resolution
        self.seed = seed
        self.wavelength_range = wavelength_range

        # Pre-compute spatial grid
        self._setup_spatial_grid()

    def _setup_spatial_grid(self):
        """Pre-compute spatial coordinate grid."""
        width, height = self.resolution

        # Create normalized coordinates in [-1, 1]
        x = np.linspace(-1, 1, width)
        y = np.linspace(-1, 1, height)

        # Create meshgrid
        self.X, self.Y = np.meshgrid(x, y)

        # Combine into position vectors
        self.positions = np.stack([self.X, self.Y], axis=-1)

    def generate(
        self,
        thetas: np.ndarray,
        text: Optional[str] = None,
        return_components: bool = False
    ) -> Union[np.ndarray, dict]:
        """
        Create spatial interference field from synchronized phases.

        Args:
            thetas: Synchronized phases [θ_1, ..., θ_N]
            text: Original text (for deterministic wave vector generation)
            return_components: If True, return dict with field components

        Returns:
            np.ndarray: Complex-valued field Φ(x,y) of shape resolution
            or dict if return_components=True: {
                'field': complex field,
                'magnitude': |Φ|,
                'phase': arg(Φ),
                'wave_vectors': k_j vectors used,
                'amplitudes': A_j values used
            }
        """
        thetas = np.asarray(thetas)
        N = len(thetas)

        if N == 0:
            raise ValueError("Must provide at least one phase")

        # Generate deterministic wave vectors from text
        wave_vectors = self._generate_wave_vectors(N, text)

        # Uniform amplitudes (could be made frequency-dependent)
        amplitudes = np.ones(N) / np.sqrt(N)  # Normalized

        # Initialize complex field
        field = np.zeros(self.resolution, dtype=np.complex128)

        # Compute interference pattern
        for j in range(N):
            # Wave vector k_j
            kx, ky = wave_vectors[j]

            # Phase factor: k_j · r + θ_j
            phase = kx * self.X + ky * self.Y + thetas[j]

            # Add wave contribution: A_j · e^{i·phase}
            field += amplitudes[j] * np.exp(1j * phase)

        if return_components:
            return {
                'field': field,
                'magnitude': np.abs(field),
                'phase': np.angle(field),
                'wave_vectors': wave_vectors,
                'amplitudes': amplitudes
            }

        return field

    def _generate_wave_vectors(
        self,
        N: int,
        text: Optional[str] = None
    ) -> np.ndarray:
        """
        Generate deterministic wave vectors for oscillators.

        Args:
            N: Number of oscillators
            text: Optional text for deterministic generation

        Returns:
            Array of wave vectors k_j = (kx_j, ky_j)
        """
        # Create deterministic seed
        if text is not None:
            text_hash = hashlib.sha256(text.encode()).digest()
            seed = int.from_bytes(text_hash[:4], byteorder='big')
        else:
            seed = self.seed

        rng = np.random.RandomState(seed)

        # Generate wave vectors
        wave_vectors = np.zeros((N, 2))

        for j in range(N):
            # Random wavelength in specified range
            wavelength = rng.uniform(*self.wavelength_range)
            k_magnitude = 2 * np.pi / wavelength

            # Random direction
            angle = rng.uniform(0, 2 * np.pi)

            wave_vectors[j] = [
                k_magnitude * np.cos(angle),
                k_magnitude * np.sin(angle)
            ]

        return wave_vectors

    def to_quaternion(self, field: np.ndarray) -> np.ndarray:
        """
        Convert complex field to quaternion representation.

        Maps complex field Φ = a + bi to quaternion q = (w, x, y, z).
        We store the real and imaginary parts in w and x components,
        and derive y, z from magnitude and phase for additional features.

        Args:
            field: Complex-valued field

        Returns:
            Quaternion field of shape (*field.shape, 4)
        """
        # Store the max for later reconstruction
        self._field_max = np.abs(field).max() if np.abs(field).max() > 0 else 1.0

        # Normalize field
        normalized_field = field / self._field_max

        magnitude = np.abs(normalized_field)
        phase = np.angle(normalized_field)

        # Map to quaternion components
        quaternion = np.zeros((*field.shape, 4))

        # Store real and imaginary directly (lossless within normalization)
        quaternion[..., 0] = np.real(normalized_field)
        quaternion[..., 1] = np.imag(normalized_field)

        # Additional components for redundancy/features
        quaternion[..., 2] = magnitude
        quaternion[..., 3] = phase / (2 * np.pi)  # Normalize phase to [-0.5, 0.5]

        return quaternion

    def from_quaternion(self, quaternion: np.ndarray) -> np.ndarray:
        """
        Reconstruct complex field from quaternion representation.

        Args:
            quaternion: Quaternion field of shape (*shape, 4)

        Returns:
            Complex-valued field
        """
        # Direct reconstruction from first two components
        real_part = quaternion[..., 0]
        imag_part = quaternion[..., 1]

        # Reconstruct normalized field
        normalized_field = real_part + 1j * imag_part

        # Scale back to original magnitude
        if hasattr(self, '_field_max'):
            field = normalized_field * self._field_max
        else:
            # Fallback if _field_max not set
            field = normalized_field

        return field

    def coherence_measure(self, field: np.ndarray) -> float:
        """
        Calculate spatial coherence of the interference field.

        Higher coherence indicates better synchronization.

        Args:
            field: Complex-valued field

        Returns:
            Coherence measure in [0, 1]
        """
        # Normalize field
        normalized = field / (np.abs(field).max() + 1e-10)

        # Calculate spatial autocorrelation
        fft_field = np.fft.fft2(normalized)
        power_spectrum = np.abs(fft_field) ** 2

        # Coherence from concentration of power
        total_power = np.sum(power_spectrum)
        if total_power > 0:
            # Fraction of power in low frequencies (coherent part)
            low_freq_mask = (
                (np.abs(np.fft.fftfreq(self.resolution[0])) < 0.1)[:, np.newaxis] &
                (np.abs(np.fft.fftfreq(self.resolution[1])) < 0.1)[np.newaxis, :]
            )
            coherent_power = np.sum(power_spectrum[low_freq_mask])
            coherence = coherent_power / total_power
        else:
            coherence = 0.0

        return float(coherence)


if __name__ == "__main__":
    # Quick validation
    print("Testing proto-identity generator...")

    generator = ProtoIdentityGenerator(resolution=(128, 128))

    # Test with synchronized phases
    N = 10
    thetas = np.linspace(0, np.pi, N)  # Partially synchronized

    # Generate proto-identity
    components = generator.generate(thetas, text="Hello world", return_components=True)
    field = components['field']

    print(f"Field shape: {field.shape}")
    print(f"Field dtype: {field.dtype}")
    print(f"Magnitude range: [{components['magnitude'].min():.3f}, {components['magnitude'].max():.3f}]")
    print(f"Phase range: [{components['phase'].min():.3f}, {components['phase'].max():.3f}]")

    # Test quaternion conversion
    quaternion = generator.to_quaternion(field)
    print(f"Quaternion shape: {quaternion.shape}")

    # Test reconstruction
    reconstructed = generator.from_quaternion(quaternion)
    reconstruction_error = np.mean(np.abs(field - reconstructed))
    print(f"Reconstruction error: {reconstruction_error:.6f}")

    # Test coherence
    coherence = generator.coherence_measure(field)
    print(f"Field coherence: {coherence:.3f}")