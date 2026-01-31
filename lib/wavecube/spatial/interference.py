"""
Standing wave interference for proto-identity derivation.

This module implements the core mechanism for deriving proto-identities
via interference between the proto-unity carrier and frequency patterns.

Categorical Theory:
- Proto-unity carrier = γ ∪ ε (stable baseline)
- Frequency spectrum = n's signature (from FFT)
- Proto-identity = Interference(carrier, frequency)
- Implements ι ∪ τ morphism application

Physical Intuition:
- Carrier: stable waveform (proto-unity baseline)
- Modulation: frequency pattern that shapes the carrier
- Result: proto-identity inheriting structure from both
"""

from enum import Enum
from typing import Optional
import numpy as np


class InterferenceMode(Enum):
    """
    Interference modes for combining waveforms.

    Modes:
        CONSTRUCTIVE: Aligned phases - patterns reinforce (carrier + modulation)
        DESTRUCTIVE: Opposed phases - patterns cancel where different (carrier - modulation)
        MODULATION: Frequency modulation - modulation shapes carrier (carrier × (1 + mod))
    """
    CONSTRUCTIVE = "constructive"
    DESTRUCTIVE = "destructive"
    MODULATION = "modulation"


class StandingWaveInterference:
    """
    Standing wave interference system for proto-identity derivation.

    Derives proto-identities by combining the proto-unity carrier with
    frequency patterns via controlled interference. This implements the
    ι ∪ τ morphism in the categorical architecture.

    Attributes:
        carrier_weight: Weight for carrier component (default: 1.0)
        modulation_weight: Weight for modulation component (default: 0.5)
        io_weight: Weight for input/output patterns (default: 0.3)
        phase_coherence: Phase coherence blending factor (default: 0.9)
    """

    def __init__(
        self,
        carrier_weight: float = 1.0,
        modulation_weight: float = 0.5,
        io_weight: float = 0.3,
        phase_coherence: float = 0.9
    ):
        """
        Initialize interference system with configurable weights.

        Args:
            carrier_weight: Weight for proto-unity carrier (γ ∪ ε)
            modulation_weight: Weight for frequency modulation
            io_weight: Weight for input/output patterns in multi-layer
            phase_coherence: Phase coherence factor [0-1], higher = more coherent

        Raises:
            ValueError: If weights are negative or phase_coherence out of range
        """
        if carrier_weight < 0 or modulation_weight < 0 or io_weight < 0:
            raise ValueError("Weights must be non-negative")

        if not (0.0 <= phase_coherence <= 1.0):
            raise ValueError("phase_coherence must be in [0, 1]")

        self.carrier_weight = carrier_weight
        self.modulation_weight = modulation_weight
        self.io_weight = io_weight
        self.phase_coherence = phase_coherence

    def interfere(
        self,
        carrier: np.ndarray,
        modulation: np.ndarray,
        mode: InterferenceMode = InterferenceMode.MODULATION
    ) -> np.ndarray:
        """
        Derive proto-identity via interference between carrier and modulation.

        This implements the core ι ∪ τ morphism: combining the proto-unity
        carrier (γ ∪ ε) with a frequency pattern to derive a proto-identity.

        Physical Interpretation:
        - CONSTRUCTIVE: Patterns reinforce where aligned (additive)
        - DESTRUCTIVE: Patterns cancel where opposed (subtractive)
        - MODULATION: Frequency modulation - modulation shapes carrier (FM)

        Args:
            carrier: Proto-unity carrier (H, W, 4) - stable baseline
            modulation: Frequency pattern (H, W, 4) - from FFT or similar
            mode: Interference mode (default: MODULATION for proto-identity)

        Returns:
            Proto-identity (H, W, 4) - derived via interference

        Raises:
            ValueError: If carrier and modulation shapes don't match
        """
        if carrier.shape != modulation.shape:
            raise ValueError(
                f"Shape mismatch: carrier {carrier.shape} != modulation {modulation.shape}"
            )

        if carrier.ndim != 3 or carrier.shape[2] != 4:
            raise ValueError(
                f"Expected (H, W, 4) quaternion arrays, got {carrier.shape}"
            )

        if mode == InterferenceMode.CONSTRUCTIVE:
            return self._constructive_interference(carrier, modulation)
        elif mode == InterferenceMode.DESTRUCTIVE:
            return self._destructive_interference(carrier, modulation)
        elif mode == InterferenceMode.MODULATION:
            return self._modulation_interference(carrier, modulation)
        else:
            raise ValueError(f"Unknown interference mode: {mode}")

    def _constructive_interference(
        self,
        carrier: np.ndarray,
        modulation: np.ndarray
    ) -> np.ndarray:
        """
        Constructive interference - aligned phases reinforce.

        Pattern: result = carrier_weight × carrier + modulation_weight × modulation

        Physical meaning: Waveforms with aligned phases add together,
        creating reinforcement where patterns align.

        Args:
            carrier: Proto-unity carrier (H, W, 4)
            modulation: Frequency pattern (H, W, 4)

        Returns:
            Reinforced proto-identity (H, W, 4)
        """
        result = (
            self.carrier_weight * carrier +
            self.modulation_weight * modulation
        )

        # Normalize to preserve magnitude scale
        return self._normalize_quaternion(result)

    def _destructive_interference(
        self,
        carrier: np.ndarray,
        modulation: np.ndarray
    ) -> np.ndarray:
        """
        Destructive interference - opposed phases cancel.

        Pattern: result = carrier_weight × carrier - modulation_weight × modulation

        Physical meaning: Waveforms with opposed phases subtract,
        canceling where patterns differ.

        Args:
            carrier: Proto-unity carrier (H, W, 4)
            modulation: Frequency pattern (H, W, 4)

        Returns:
            Canceled proto-identity (H, W, 4)
        """
        result = (
            self.carrier_weight * carrier -
            self.modulation_weight * modulation
        )

        # Normalize to preserve magnitude scale
        return self._normalize_quaternion(result)

    def _modulation_interference(
        self,
        carrier: np.ndarray,
        modulation: np.ndarray
    ) -> np.ndarray:
        """
        Frequency modulation - modulation shapes carrier.

        Pattern: result = carrier × (1 + modulation_weight × modulation)

        Physical meaning: This is FM synthesis - the modulation pattern
        shapes the carrier waveform, creating a new proto-identity that
        inherits structure from both carrier and modulation.

        This is the primary mode for proto-identity derivation (ι ∪ τ).

        Args:
            carrier: Proto-unity carrier (H, W, 4)
            modulation: Frequency pattern (H, W, 4)

        Returns:
            Modulated proto-identity (H, W, 4)
        """
        # FM modulation: carrier shaped by modulation
        result = carrier * (1.0 + self.modulation_weight * modulation)

        # Apply phase coherence blending
        result = self._apply_phase_coherence(result, carrier, modulation)

        return result

    def _apply_phase_coherence(
        self,
        result: np.ndarray,
        carrier: np.ndarray,
        modulation: np.ndarray
    ) -> np.ndarray:
        """
        Apply phase coherence blending to preserve phase relationships.

        Blends result with phase-adjusted version to maintain coherence
        between carrier and modulation phases.

        Args:
            result: Raw interference result (H, W, 4)
            carrier: Original carrier (H, W, 4)
            modulation: Original modulation (H, W, 4)

        Returns:
            Phase-coherent result (H, W, 4)
        """
        # Extract phase information from carrier and modulation
        # Use W channel as phase reference (quaternion W = real part)
        carrier_phase = np.angle(carrier[..., 3])
        modulation_phase = np.angle(modulation[..., 3])

        # Compute phase difference
        phase_diff = np.abs(carrier_phase - modulation_phase)

        # Create coherence mask (higher where phases are aligned)
        coherence_mask = np.cos(phase_diff)  # [-1, 1]
        coherence_mask = (coherence_mask + 1.0) / 2.0  # [0, 1]

        # Blend based on phase coherence parameter
        coherent_blend = (
            self.phase_coherence * coherence_mask +
            (1.0 - self.phase_coherence)
        )

        # Apply coherence weighting per-pixel
        for channel in range(4):
            result[..., channel] *= coherent_blend

        return result

    def interfere_multi_layer(
        self,
        proto_unity: np.ndarray,
        experiential: Optional[np.ndarray] = None,
        io: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Combine proto-identities from multiple memory layers with interference.

        Used for query synthesis - combines patterns from core memory
        (proto-unity), experiential memory, and I/O patterns into a
        unified proto-identity for querying.

        Combination strategy:
        1. Start with proto_unity as base carrier
        2. Interfere with experiential (if provided)
        3. Interfere with I/O (if provided)
        4. Each layer modulates the previous result

        Args:
            proto_unity: Proto-unity base pattern (H, W, 4)
            experiential: Experiential memory pattern (H, W, 4), optional
            io: Input/output pattern (H, W, 4), optional

        Returns:
            Combined proto-identity (H, W, 4)

        Raises:
            ValueError: If shapes don't match
        """
        result = proto_unity.copy()

        # Apply experiential layer if provided
        if experiential is not None:
            result = self._apply_layer(result, experiential, weight=1.0)

        # Apply I/O layer if provided (with reduced weight)
        if io is not None:
            result = self._apply_layer(result, io, weight=self.io_weight)

        return result

    def _apply_layer(
        self,
        carrier: np.ndarray,
        modulation: np.ndarray,
        weight: float
    ) -> np.ndarray:
        """
        Apply a memory layer via interference.

        Args:
            carrier: Current result (H, W, 4)
            modulation: Layer to apply (H, W, 4)
            weight: Weight for modulation

        Returns:
            Modulated result (H, W, 4)

        Raises:
            ValueError: If shapes don't match
        """
        if modulation.shape != carrier.shape:
            raise ValueError(
                f"Layer shape {modulation.shape} != carrier {carrier.shape}"
            )

        # Scale modulation by weight if needed
        weighted_mod = modulation * weight if weight != 1.0 else modulation

        # Apply interference
        return self.interfere(
            carrier=carrier,
            modulation=weighted_mod,
            mode=InterferenceMode.MODULATION
        )

    def _normalize_quaternion(self, quat: np.ndarray) -> np.ndarray:
        """
        Normalize quaternion array to unit magnitude.

        Preserves direction but normalizes magnitude to prevent
        accumulation of numerical errors.

        Args:
            quat: Quaternion array (H, W, 4)

        Returns:
            Normalized quaternion (H, W, 4)
        """
        # Compute magnitude: sqrt(x² + y² + z² + w²)
        magnitude = np.sqrt(np.sum(quat ** 2, axis=-1, keepdims=True))

        # Avoid division by zero
        magnitude = np.maximum(magnitude, 1e-8)

        # Normalize
        return quat / magnitude

    def compute_interference_strength(
        self,
        carrier: np.ndarray,
        modulation: np.ndarray
    ) -> float:
        """
        Compute interference strength between carrier and modulation.

        Returns a scalar [0, 1] indicating how strongly the patterns
        interfere. Higher values mean stronger interference.

        Useful for adaptive weighting or debugging.

        Args:
            carrier: Proto-unity carrier (H, W, 4)
            modulation: Frequency pattern (H, W, 4)

        Returns:
            Interference strength [0, 1]
        """
        if carrier.shape != modulation.shape:
            raise ValueError("Shape mismatch")

        # Compute dot product (measures alignment)
        dot_product = np.sum(carrier * modulation)

        # Normalize by magnitudes
        carrier_norm = np.sqrt(np.sum(carrier ** 2))
        modulation_norm = np.sqrt(np.sum(modulation ** 2))

        # Cosine similarity
        similarity = dot_product / (carrier_norm * modulation_norm + 1e-8)

        # Map to [0, 1] range
        strength = (similarity + 1.0) / 2.0

        return float(strength)
