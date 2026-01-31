"""
FFT-based text decoding without metadata reads.

This module implements pure IFFT-based text decoding that reconstructs text
from the frequency domain proto-identity, eliminating metadata dependencies.
"""

import numpy as np
from typing import Tuple


class FFTTextDecoder:
    """Pure IFFT-based text decoding without metadata reads."""

    def __init__(self, width: int = 512, height: int = 512):
        """
        Initialize FFT decoder with grid dimensions.

        Args:
            width: Grid width (default 512)
            height: Grid height (default 512)
        """
        self.width = width
        self.height = height

    def decode_text(self, proto_identity: np.ndarray) -> str:
        """
        Decode proto-identity to text via IFFT.

        Args:
            proto_identity: (512, 512, 4) XYZW quaternion

        Returns:
            text: Reconstructed text string
        """
        # Convert proto-identity back to frequency spectrum
        spectrum = self._proto_to_frequency(proto_identity)

        # Transform back to spatial domain
        grid = self._frequency_to_grid(spectrum)

        # Extract text from grid
        text = self._grid_to_text(grid)

        return text

    def _proto_to_frequency(self, proto: np.ndarray) -> np.ndarray:
        """
        Convert proto-identity quaternion to frequency spectrum.

        Args:
            proto: (512, 512, 4) XYZW quaternion

        Returns:
            spectrum: (512, 512) complex frequency array
        """
        # Extract XYZW components
        x = proto[:, :, 0]  # Real component
        y = proto[:, :, 1]  # Imaginary component
        z = proto[:, :, 2]  # Magnitude
        w = proto[:, :, 3]  # Normalized phase

        # Reconstruct phase from normalized value
        phase = w * 2 * np.pi - np.pi

        # Reconstruct complex frequency from magnitude and phase
        # Use magnitude from Z channel for better precision
        freq_complex = z * np.exp(1j * phase)

        return freq_complex

    def _frequency_to_grid(self, spectrum: np.ndarray) -> np.ndarray:
        """
        Apply 2D IFFT to frequency spectrum.

        Args:
            spectrum: (512, 512) complex frequency

        Returns:
            grid: (512, 512) spatial representation
        """
        # Unshift to move zero frequency back to corner
        spectrum_unshifted = np.fft.ifftshift(spectrum)

        # Apply inverse 2D FFT
        grid = np.fft.ifft2(spectrum_unshifted)

        return grid

    def _grid_to_text(self, grid: np.ndarray) -> str:
        """
        Extract text from spatial grid using reverse spiral extraction.

        Args:
            grid: (512, 512) spatial array

        Returns:
            text: Decoded text string
        """
        # Get center position
        cx, cy = self.width // 2, self.height // 2

        # Extract values in spiral order
        bytes_list = []
        positions = self._generate_spiral_positions(cx, cy,
                                                    self.width * self.height)

        for x, y in positions:
            if 0 <= x < self.width and 0 <= y < self.height:
                # Get real part and denormalize
                val = np.real(grid[y, x])
                byte_val = int(np.round(val * 255))

                # Stop at first zero byte (end of text)
                if byte_val == 0:
                    break

                # Clamp to valid byte range
                byte_val = max(0, min(255, byte_val))
                bytes_list.append(byte_val)

        # Convert bytes to string
        if not bytes_list:
            return ""

        try:
            # Decode UTF-8 bytes
            text = bytes(bytes_list).decode('utf-8', errors='ignore')
            # Remove any trailing null characters
            text = text.rstrip('\x00')
            return text
        except Exception:
            # Fallback to ASCII if UTF-8 fails
            return ''.join(chr(b) if 32 <= b < 127 else '?' for b in bytes_list)

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

    def validate_proto_identity(self, proto: np.ndarray) -> bool:
        """
        Validate proto-identity dimensions and values.

        Args:
            proto: Input array to validate

        Returns:
            True if valid proto-identity, False otherwise
        """
        # Check dimensions
        if proto.shape != (self.height, self.width, 4):
            return False

        # Check for NaN or Inf values
        if np.any(np.isnan(proto)) or np.any(np.isinf(proto)):
            return False

        # Check phase normalization (W channel should be in [0, 1])
        w_channel = proto[:, :, 3]
        if np.any(w_channel < -0.1) or np.any(w_channel > 1.1):
            # Allow small tolerance for numerical errors
            return False

        return True