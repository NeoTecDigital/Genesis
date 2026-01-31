"""
State Classifier - Classify signal state based on temporal pattern and coherence.

Three signal states:
- PARADOX: Conflicting, low coherence (∂proto/∂t is large, coherence low)
- EVOLUTION: Changing, derivatives non-zero (∂proto/∂t significant, coherence moderate)
- IDENTITY: Stable, converged (∂proto/∂t ≈ 0, coherence high)
"""

from enum import Enum
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .temporal_buffer import TemporalBuffer


class SignalState(Enum):
    """Three signal states for proto-identity classification."""
    PARADOX = 0    # Conflicting, low coherence
    EVOLUTION = 1  # Changing, derivatives non-zero
    IDENTITY = 2   # Stable, converged


class StateClassifier:
    """Classify signal state based on temporal pattern and coherence.

    Uses temporal derivatives and coherence to determine if a proto-identity is:
    - PARADOX: High rate of change, low coherence (conflicting signals)
    - EVOLUTION: Moderate rate of change, moderate coherence (learning/adapting)
    - IDENTITY: Low rate of change, high coherence (stable convergence)
    """

    def __init__(self,
                 evolution_threshold: float = 0.1,
                 identity_coherence: float = 0.85,
                 paradox_coherence: float = 0.3):
        """Initialize state classifier.

        Args:
            evolution_threshold: Threshold for derivative magnitude to be considered "changing"
            identity_coherence: Minimum coherence for IDENTITY state
            paradox_coherence: Maximum coherence for PARADOX state
        """
        self.evolution_threshold = evolution_threshold
        self.identity_coherence = identity_coherence
        self.paradox_coherence = paradox_coherence

    def classify(self, temporal_buffer: 'TemporalBuffer', coherence: float) -> SignalState:
        """Classify current state based on temporal buffer and coherence.

        Classification logic:
        1. Compute derivative magnitude
        2. High derivative + low coherence → PARADOX (conflicting)
        3. Low derivative + high coherence → IDENTITY (stable)
        4. Otherwise → EVOLUTION (changing)

        Args:
            temporal_buffer: Temporal buffer with history
            coherence: Current coherence value [0, 1]

        Returns:
            Classified signal state
        """
        # Get first derivative (velocity)
        deriv_1 = temporal_buffer.get_derivatives(order=1)

        if deriv_1 is None:
            # Insufficient data - default to EVOLUTION
            return SignalState.EVOLUTION

        # Compute derivative magnitude (normalized)
        deriv_magnitude = self._compute_derivative_magnitude(deriv_1)

        # Classify based on derivative magnitude and coherence
        return self._classify_from_metrics(deriv_magnitude, coherence)

    def _compute_derivative_magnitude(self, derivative: np.ndarray) -> float:
        """Compute normalized magnitude of derivative.

        Uses RMS (root mean square) of derivative values.

        Args:
            derivative: Derivative array (H, W, 4)

        Returns:
            Normalized magnitude [0, inf) - typically [0, 1] for well-behaved signals
        """
        # RMS of all derivative values
        rms = np.sqrt(np.mean(derivative ** 2))
        return rms

    def _classify_from_metrics(self, deriv_magnitude: float, coherence: float) -> SignalState:
        """Classify state from derivative magnitude and coherence.

        Decision tree:
        - deriv_magnitude > threshold AND coherence < paradox_threshold → PARADOX
        - deriv_magnitude < threshold AND coherence > identity_threshold → IDENTITY
        - Otherwise → EVOLUTION

        Args:
            deriv_magnitude: Magnitude of temporal derivative
            coherence: Coherence value [0, 1]

        Returns:
            Classified signal state
        """
        is_changing = deriv_magnitude > self.evolution_threshold
        is_stable = deriv_magnitude < (self.evolution_threshold * 0.5)  # Tighter threshold for stability

        # IDENTITY: stable and high coherence
        if is_stable and coherence >= self.identity_coherence:
            return SignalState.IDENTITY

        # PARADOX: changing and low coherence (conflicting signals)
        if is_changing and coherence <= self.paradox_coherence:
            return SignalState.PARADOX

        # EVOLUTION: everything else (learning/adapting)
        return SignalState.EVOLUTION

    def __repr__(self) -> str:
        """String representation."""
        return (f"StateClassifier(evolution_threshold={self.evolution_threshold}, "
                f"identity_coherence={self.identity_coherence}, "
                f"paradox_coherence={self.paradox_coherence})")
