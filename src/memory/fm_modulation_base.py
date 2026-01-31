"""
FM Modulation Base - Shared Core Functionality

Provides common FM modulation operations shared between:
- FMModulationMemory (basic FM with clustering)
- StratifiedFMModulation (FM with stratified memory pool)

Core operations:
- XYZW channel extraction and semantics
- Phase velocity computation
- FM modulation (carrier * input)
- Delta computation (deviation from carrier)
- Resonance detection (constructive interference)
"""

import numpy as np
from typing import Dict


class FMModulationBase:
    """
    Base class providing core FM modulation operations.

    Channel semantics (XYZW):
    - X: Real(Concept) - real part of complex oscillation
    - Y: Imaginary(Concept) - imaginary part of complex oscillation
    - Z: Weight/Amplitude - "truth" that increases with resonance
    - W: Phase Velocity - time derivative for prediction
    """

    def extract_xyzw_channels(self, state: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract XYZW channels from state.

        Args:
            state: (H, W, 4) RGBA32F array

        Returns:
            Dict with X, Y, Z, W channels
        """
        return {
            'X': state[:, :, 0],  # Real(Concept)
            'Y': state[:, :, 1],  # Imaginary(Concept)
            'Z': state[:, :, 2],  # Weight/Truth
            'W': state[:, :, 3],  # Phase Velocity
        }

    def compute_phase_velocity(self, xy_channels: np.ndarray) -> np.ndarray:
        """
        Compute phase velocity (W channel) from XY oscillation.

        W = dφ/dt where φ = arctan2(Y, X)

        For discrete approximation, we compute spatial gradient as proxy:
        W ≈ ∇φ (gradient magnitude)

        Args:
            xy_channels: (H, W, 2) array with X, Y channels

        Returns:
            (H, W) phase velocity array
        """
        X = xy_channels[:, :, 0]
        Y = xy_channels[:, :, 1]

        # Compute phase
        phase = np.arctan2(Y, X)

        # Compute gradient (toroidal wrapping)
        phase_dx = np.roll(phase, -1, axis=1) - np.roll(phase, 1, axis=1)
        phase_dy = np.roll(phase, -1, axis=0) - np.roll(phase, 1, axis=0)

        # Unwrap phase jumps (handle 2π discontinuities)
        phase_dx = np.where(phase_dx > np.pi, phase_dx - 2*np.pi, phase_dx)
        phase_dx = np.where(phase_dx < -np.pi, phase_dx + 2*np.pi, phase_dx)
        phase_dy = np.where(phase_dy > np.pi, phase_dy - 2*np.pi, phase_dy)
        phase_dy = np.where(phase_dy < -np.pi, phase_dy + 2*np.pi, phase_dy)

        # Gradient magnitude
        velocity = np.sqrt(phase_dx**2 + phase_dy**2)

        return velocity

    def modulate(self, carrier: np.ndarray, input_data: np.ndarray,
                 modulation_depth: float = 0.5) -> np.ndarray:
        """
        FM modulation: carrier modulated by input with controllable depth.

        Modulation semantics:
        - XY: Complex multiplication with depth control
        - Z: Input amplitude (initial weight/truth)
        - W: Recompute phase velocity after modulation

        Args:
            carrier: (H, W, 4) carrier signal
            input_data: (H, W, 4) input signal
            modulation_depth: Strength of modulation (0.0 to 2.0, default 0.5)

        Returns:
            (H, W, 4) modulated signal
        """
        # Extract channels and convert to complex for proper FM modulation
        carrier_complex = carrier[:, :, 0] + 1j * carrier[:, :, 1]
        input_complex = input_data[:, :, 0] + 1j * input_data[:, :, 1]

        # Extract phase and magnitude from input
        input_phase = np.angle(input_complex)
        input_magnitude = np.abs(input_complex)

        # FM modulation: modulate carrier phase by input phase
        # and scale amplitude by input magnitude
        phase_modulation = input_phase * modulation_depth
        magnitude_modulation = 1 + (input_magnitude * modulation_depth)

        # Apply FM modulation: carrier * magnitude * exp(i * phase)
        # This creates interference patterns unique to each input
        modulated_complex = carrier_complex * magnitude_modulation * np.exp(1j * phase_modulation)

        # Convert back to real components
        modulated_x = np.real(modulated_complex)
        modulated_y = np.imag(modulated_complex)
        modulated_xy = np.stack([modulated_x, modulated_y], axis=-1)

        # Z channel: Use input amplitude (weight/truth)
        modulated_z = input_data[:, :, 2]

        # W channel: Recompute phase velocity
        modulated_w = self.compute_phase_velocity(modulated_xy)

        # Combine channels
        modulated = np.concatenate([
            modulated_xy,
            modulated_z[..., None],
            modulated_w[..., None]
        ], axis=-1).astype(np.float32)

        return modulated

    def demodulate(self, modulated: np.ndarray, carrier: np.ndarray,
                   modulation_depth: float = 0.5) -> np.ndarray:
        """
        Demodulate proto-identity to extract original signal.

        Inverse of modulate():
        modulated = carrier * (1 + input_mag * depth) * exp(i * input_phase * depth)

        To recover input:
        modulated / carrier = (1 + input_mag * depth) * exp(i * input_phase * depth)

        Args:
            modulated: (H, W, 4) modulated signal
            carrier: (H, W, 4) carrier signal
            modulation_depth: Strength of modulation used during modulation

        Returns:
            (H, W, 4) demodulated signal
        """
        # Extract XY channels (complex representation)
        modulated_x, modulated_y = modulated[:, :, 0], modulated[:, :, 1]
        carrier_x, carrier_y = carrier[:, :, 0], carrier[:, :, 1]

        # Convert to complex
        modulated_complex = modulated_x + 1j * modulated_y
        carrier_complex = carrier_x + 1j * carrier_y

        # Divide modulated by carrier to extract modulation
        # Handle division by zero
        carrier_mag = np.abs(carrier_complex)
        demod_complex = np.divide(
            modulated_complex,
            carrier_complex,
            out=np.zeros_like(modulated_complex),
            where=carrier_mag > 1e-8
        )

        # Extract modulation parameters
        demod_magnitude = np.abs(demod_complex)
        demod_phase = np.angle(demod_complex)

        # Recover input magnitude and phase
        # From: magnitude_modulation = 1 + (input_magnitude * modulation_depth)
        # Thus: input_magnitude = (demod_magnitude - 1) / modulation_depth
        input_magnitude = np.divide(
            demod_magnitude - 1,
            modulation_depth,
            out=np.zeros_like(demod_magnitude),
            where=np.abs(modulation_depth) > 1e-8
        )

        # From: phase_modulation = input_phase * modulation_depth
        # Thus: input_phase = demod_phase / modulation_depth
        input_phase = np.divide(
            demod_phase,
            modulation_depth,
            out=np.zeros_like(demod_phase),
            where=np.abs(modulation_depth) > 1e-8
        )

        # Reconstruct input complex signal
        input_complex = input_magnitude * np.exp(1j * input_phase)

        # Convert back to real components
        input_x = np.real(input_complex)
        input_y = np.imag(input_complex)

        # Z channel: Use the correctly recovered magnitude
        # Since modulation passes through Z directly, and Z contains the original magnitude,
        # we should use the demodulated magnitude which represents the signal amplitude
        input_z = np.sqrt(input_x**2 + input_y**2)

        # W channel: Recompute phase velocity from demodulated XY
        demod_xy = np.stack([input_x, input_y], axis=-1)
        input_w = self.compute_phase_velocity(demod_xy)

        # Combine channels
        signal = np.stack([input_x, input_y, input_z, input_w], axis=-1).astype(np.float32)

        return signal

    def compute_residual(self, modulated: np.ndarray, carrier: np.ndarray) -> np.ndarray:
        """
        Compute residual (deviation from carrier).

        Memory storage = residual from carrier, not absolute state.
        This is used for delta encoding in memory storage.

        Args:
            modulated: (H, W, 4) modulated signal
            carrier: (H, W, 4) carrier signal

        Returns:
            (H, W, 4) residual (deviation)
        """
        return modulated - carrier

    def compute_delta(self, modulated: np.ndarray, carrier: np.ndarray) -> np.ndarray:
        """
        DEPRECATED: Use compute_residual() for delta encoding or demodulate() for signal recovery.

        Kept for backward compatibility.
        """
        return self.compute_residual(modulated, carrier)

    def detect_resonance(self, state: np.ndarray, carrier: np.ndarray) -> float:
        """
        Detect constructive interference (resonance) between state and carrier.

        High resonance = state aligns with carrier harmonics.

        Args:
            state: (H, W, 4) current state
            carrier: (H, W, 4) carrier signal

        Returns:
            Resonance strength (0-1)
        """
        # Compute dot product in XY space (concept alignment)
        state_xy = state[:, :, :2]
        carrier_xy = carrier[:, :, :2]

        dot_product = np.sum(state_xy * carrier_xy, axis=-1)

        # Normalize by magnitudes
        state_mag = np.linalg.norm(state_xy, axis=-1) + 1e-6
        carrier_mag = np.linalg.norm(carrier_xy, axis=-1) + 1e-6

        # Cosine similarity
        alignment = dot_product / (state_mag * carrier_mag)

        # Resonance = mean alignment across space
        resonance = float(np.mean(np.abs(alignment)))

        return resonance
