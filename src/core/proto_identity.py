"""ProtoIdentity data class for unified field representation."""

from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple
import numpy as np


@dataclass
class ProtoIdentity:
    """
    Proto-identity field representation.

    Attributes:
        field: Complex-valued spatial field (512×512 or 512×512×4 for spinor)
        metadata: Dictionary containing:
            - 'order_parameter': Tuple[float, float] - (r, Ψ) from Kuramoto
            - 'mass_gap': float - Δ from UFT evolution
            - 'chiral_phase': float - δ from text analysis
            - 'coupling': float - K coupling strength
            - 'resonance': float - R resonance factor
            - 'encoder_type': str - 'kuramoto' | 'uft' | 'fft'
            - 'evolved': bool - Whether UFT evolution has been applied
            - 'text': str - Original text (optional)
            - 'steps': int - Evolution/synchronization steps
            - 'converged': bool - Whether dynamics converged
    """
    field: np.ndarray
    metadata: Dict[str, Any]

    def __post_init__(self):
        """Validate field dimensions and metadata."""
        # Validate field shape
        shape = self.field.shape
        valid_shapes = [
            (512, 512),        # 2D complex field
            (512, 512, 2),     # 2D with real/imag channels
            (512, 512, 4),     # Quaternion/spinor field
        ]

        if len(shape) not in [2, 3] or (len(shape) == 3 and shape not in valid_shapes):
            raise ValueError(
                f"Invalid field shape {shape}. Expected 512×512, 512×512×2, or 512×512×4"
            )

        # Ensure field is complex for 2D case
        if len(shape) == 2 and not np.iscomplexobj(self.field):
            self.field = self.field.astype(np.complex128)

    def to_quaternion(self) -> np.ndarray:
        """
        Convert to WaveCube quaternion format (512×512×4).

        Returns:
            np.ndarray: XYZW quaternion field of shape (512, 512, 4)
        """
        shape = self.field.shape

        # Already in quaternion format
        if shape == (512, 512, 4):
            return self.field.copy()

        # Convert from complex field
        if len(shape) == 2 or shape == (512, 512, 2):
            # Ensure complex
            if len(shape) == 2:
                field_complex = self.field
            else:
                field_complex = self.field[..., 0] + 1j * self.field[..., 1]

            # Map to quaternion
            # Real part → X component
            # Imaginary part → Y component
            # Zero → Z, W components (can be enhanced with metadata)
            quaternion = np.zeros((512, 512, 4), dtype=np.float64)
            quaternion[..., 0] = np.real(field_complex)  # X
            quaternion[..., 1] = np.imag(field_complex)  # Y

            # Optional: Use metadata to inform Z, W components
            if 'mass_gap' in self.metadata:
                # Mass gap modulates W component
                quaternion[..., 3] = self.metadata['mass_gap'] * np.abs(field_complex)

            if 'chiral_phase' in self.metadata:
                # Chiral phase modulates Z component
                delta = self.metadata['chiral_phase']
                quaternion[..., 2] = np.cos(delta) * np.abs(field_complex)

            return quaternion

        raise ValueError(f"Cannot convert field shape {shape} to quaternion")

    def get_order_parameter(self) -> Optional[Tuple[float, float]]:
        """
        Get Kuramoto order parameter (r, Ψ) if available.

        Returns:
            Optional[Tuple[float, float]]: (r, Ψ) or None
        """
        return self.metadata.get('order_parameter')

    def get_mass_gap(self) -> Optional[float]:
        """
        Get UFT mass gap Δ if available.

        Returns:
            Optional[float]: Mass gap or None
        """
        return self.metadata.get('mass_gap')

    def get_chiral_phase(self) -> Optional[float]:
        """
        Get chiral phase δ if available.

        Returns:
            Optional[float]: Chiral phase or None
        """
        return self.metadata.get('chiral_phase')

    def is_evolved(self) -> bool:
        """Check if UFT evolution has been applied."""
        return self.metadata.get('evolved', False)

    def is_converged(self) -> bool:
        """Check if dynamics converged."""
        return self.metadata.get('converged', False)