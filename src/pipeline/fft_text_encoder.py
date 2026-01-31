"""
FFT-based text encoding without metadata storage.

This module implements pure FFT-based text encoding that stores all information
in the frequency domain proto-identity itself, eliminating metadata violations.

Phase B: Supports interference-based proto-identity derivation via standing waves.
"""

import numpy as np
from typing import Tuple, Optional


class FFTTextEncoder:
    """Pure FFT-based text encoding without metadata storage."""

    def __init__(self, width: int = 512, height: int = 512):
        """
        Initialize FFT encoder with grid dimensions.

        Args:
            width: Grid width (default 512)
            height: Grid height (default 512)
        """
        self.width = width
        self.height = height

    def encode_text(self, text: str) -> np.ndarray:
        """
        Encode text to proto-identity via FFT.

        Args:
            text: Input text string

        Returns:
            proto_identity: (512, 512, 4) XYZW quaternion array
        """
        # Convert text to 2D spatial grid
        grid = self._text_to_grid(text)

        # Transform to frequency domain
        spectrum = self._grid_to_frequency(grid)

        # Convert to proto-identity quaternion
        proto = self._frequency_to_proto(spectrum)

        return proto

    def encode(
        self,
        text: str,
        proto_unity_carrier: Optional[np.ndarray] = None,
        use_interference: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Encode text to frequency spectrum and optionally derive proto-identity.

        Phase B Integration: Supports interference-based proto-identity derivation
        via StandingWaveInterference when carrier is provided.

        Args:
            text: Text to encode
            proto_unity_carrier: Proto-unity carrier (γ ∪ ε) for interference
            use_interference: If True and carrier provided, derive proto-identity via interference

        Returns:
            Tuple of (frequency_spectrum, proto_identity)
            - If use_interference=False or carrier=None: (frequency, frequency) for backward compat
            - If use_interference=True and carrier provided: (frequency, derived_proto_identity)
        """
        # Convert text to 2D spatial grid
        grid = self._text_to_grid(text)

        # Transform to frequency domain
        frequency_spectrum = self._grid_to_frequency(grid)

        # Convert frequency to proto-identity (baseline)
        proto_from_fft = self._frequency_to_proto(frequency_spectrum)

        # If interference requested and carrier provided, derive via standing waves
        if use_interference and proto_unity_carrier is not None:
            # Import interference system directly (avoid wavecube __init__.py)
            import sys
            from pathlib import Path
            wavecube_path = Path(__file__).parent.parent.parent / "lib" / "wavecube"
            if str(wavecube_path) not in sys.path:
                sys.path.insert(0, str(wavecube_path))

            from spatial.interference import (
                StandingWaveInterference,
                InterferenceMode
            )

            # Create interference system
            interference = StandingWaveInterference(
                carrier_weight=1.0,
                modulation_weight=0.5,
                phase_coherence=0.9
            )

            # Derive proto-identity via MODULATION mode (ι ∪ τ application)
            # Carrier = proto-unity carrier (γ ∪ ε)
            # Modulation = frequency spectrum (text's signature)
            # Result = proto-identity inheriting structure from both
            proto_identity = interference.interfere(
                carrier=proto_unity_carrier,
                modulation=proto_from_fft,
                mode=InterferenceMode.MODULATION
            )

            return frequency_spectrum, proto_identity

        # Backward compatibility: return frequency as both spectrum and proto
        return frequency_spectrum, proto_from_fft

    def _text_to_grid(self, text: str) -> np.ndarray:
        """
        Convert text to 2D spatial grid with spiral embedding.

        Args:
            text: Input text

        Returns:
            grid: (512, 512) complex array with text encoded spatially
        """
        # Convert text to UTF-8 bytes
        text_bytes = text.encode('utf-8')

        # Create 2D grid
        grid = np.zeros((self.height, self.width), dtype=np.complex128)

        # Early return for empty text
        if len(text_bytes) == 0:
            return grid

        # Embed bytes in spiral pattern (center-out for locality)
        cx, cy = self.width // 2, self.height // 2

        # Generate spiral positions
        positions = self._generate_spiral_positions(cx, cy, len(text_bytes))

        # Embed bytes at spiral positions
        for byte_val, (x, y) in zip(text_bytes, positions):
            if 0 <= x < self.width and 0 <= y < self.height:
                # Normalize byte value to [0, 1] and store as real part
                grid[y, x] = complex(byte_val / 255.0, 0)

        return grid

    def _generate_spiral_positions(self, cx: int, cy: int,
                                   count: int) -> list[Tuple[int, int]]:
        """
        Generate spiral positions from center outward.

        Args:
            cx: Center x coordinate
            cy: Center y coordinate
            count: Number of positions needed

        Returns:
            List of (x, y) positions in spiral order
        """
        positions = []
        x, y = cx, cy
        dx, dy = 0, -1

        for _ in range(count):
            positions.append((x, y))

            # Spiral logic: turn right when hitting boundary
            if x == cx + max(abs(x - cx), abs(y - cy)) and y == cy - max(abs(x - cx), abs(y - cy)):
                dx, dy = dy, -dx
            elif abs(x - cx) == abs(y - cy) and (dx, dy) != (1, 0):
                dx, dy = dy, -dx

            x, y = x + dx, y + dy

            # Boundary check
            if len(positions) >= count or x < 0 or x >= self.width or y < 0 or y >= self.height:
                break

        return positions

    def _grid_to_frequency(self, grid: np.ndarray) -> np.ndarray:
        """
        Apply 2D FFT to spatial grid.

        Args:
            grid: (512, 512) spatial representation

        Returns:
            spectrum: (512, 512, 2) [magnitude, phase] array
        """
        # Apply 2D FFT
        freq_complex = np.fft.fft2(grid)

        # Shift zero frequency to center for better organization
        freq_complex = np.fft.fftshift(freq_complex)

        # Extract magnitude and phase
        magnitude = np.abs(freq_complex)
        phase = np.angle(freq_complex)

        # Stack as [magnitude, phase]
        spectrum = np.stack([magnitude, phase], axis=-1)

        return spectrum.astype(np.float32)

    def _frequency_to_proto(self, spectrum: np.ndarray) -> np.ndarray:
        """
        Convert frequency spectrum to proto-identity quaternion.

        Args:
            spectrum: (512, 512, 2) [magnitude, phase]

        Returns:
            proto: (512, 512, 4) XYZW quaternion
        """
        # Initialize proto-identity
        proto = np.zeros((self.height, self.width, 4), dtype=np.float32)

        # Extract magnitude and phase
        mag = spectrum[:, :, 0]
        phase = spectrum[:, :, 1]

        # XYZW quaternion encoding
        # X: Real component (magnitude * cos(phase))
        proto[:, :, 0] = mag * np.cos(phase)

        # Y: Imaginary component (magnitude * sin(phase))
        proto[:, :, 1] = mag * np.sin(phase)

        # Z: Magnitude (for sparse indexing and retrieval)
        proto[:, :, 2] = mag

        # W: Normalized phase [0, 1]
        # Handle phase wrapping properly
        proto[:, :, 3] = (phase + np.pi) / (2 * np.pi)

        return proto

    def compress_spectrum(self, spectrum: np.ndarray,
                         keep_ratio: float = 0.01) -> Tuple[np.ndarray, float]:
        """
        Compress spectrum by keeping only significant frequencies.

        Args:
            spectrum: Input frequency spectrum
            keep_ratio: Minimum magnitude ratio to keep (default 1%)

        Returns:
            Tuple of (compressed spectrum, compression ratio achieved)
        """
        # Get magnitude channel
        magnitude = spectrum[:, :, 0] if len(spectrum.shape) == 3 else np.abs(spectrum)

        # Calculate threshold - handle edge case where max is 0
        max_mag = np.max(magnitude)
        if max_mag == 0:
            return spectrum, 0.0

        threshold = max_mag * keep_ratio

        # Create mask for significant frequencies
        mask = magnitude > threshold

        # Apply mask
        if len(spectrum.shape) == 3:
            compressed = spectrum.copy()
            compressed[~mask] = 0
        else:
            compressed = spectrum * mask

        # Calculate compression ratio (zeroed elements / total elements)
        zeroed = np.sum(~mask)
        total = mask.size
        compression_ratio = zeroed / total if total > 0 else 0.0

        return compressed, compression_ratio