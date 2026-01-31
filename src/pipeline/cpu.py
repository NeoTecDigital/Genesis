"""
CPU implementation of Genesis morphisms (γ, ι, τ, ε)

Implements the same operations as GPU shaders but in NumPy.
Used as fallback when GPU is unavailable.
"""

import numpy as np
from typing import Dict

TAU = 2.0 * np.pi


class CPUPipeline:
    """CPU-based implementation of Genesis categorical morphisms."""

    def __init__(self, width: int = 512, height: int = 512):
        self.width = width
        self.height = height

        # Precompute frequency coordinates (optimization)
        y, x = np.meshgrid(
            np.arange(height) - height / 2,
            np.arange(width) - width / 2,
            indexing='ij'
        )
        self.uv = np.stack([x / width, y / height], axis=-1)  # (H, W, 2) normalized coords

    def apply_gamma(self, empty: np.ndarray, params: Dict) -> np.ndarray:
        """
        Gamma Genesis: γ : ∅ → 𝟙

        Creates proto-unity from empty state using multi-scale harmonics.

        Args:
            empty: (H, W, 4) empty state
            params: {
                'base_frequency': float,
                'initial_phase': float,
                'amplitude': float,
                'envelope_sigma': float,
                'num_harmonics': int,
                'harmonic_decay': float
            }

        Returns:
            (H, W, 4) proto-unity state
        """
        # Extract parameters
        base_freq = params.get('base_frequency', 2.0)
        initial_phase = params.get('initial_phase', 0.0)
        amplitude = params.get('amplitude', 1.0)
        envelope_sigma = params.get('envelope_sigma', 0.45)
        num_harmonics = params.get('num_harmonics', 12)
        harmonic_decay = params.get('harmonic_decay', 0.75)

        # Frequency domain coordinates
        freq = self.uv * base_freq  # (H, W, 2)
        freq_mag = np.linalg.norm(freq, axis=-1)  # (H, W)

        # Multi-scale Gaussian envelopes
        envelope = np.zeros_like(freq_mag)
        for scale in range(4):
            sigma = envelope_sigma * (1.0 + scale * 0.3)
            scale_weight = 1.0 / (1.0 + scale)
            envelope += np.exp(-freq_mag**2 / (2.0 * sigma**2)) * scale_weight
        envelope /= 4.0

        # Harmonic series
        complex_sum = np.zeros((self.height, self.width, 2), dtype=np.float32)

        for h in range(1, num_harmonics + 1):
            harmonic_freq = float(h)
            harmonic_amp = amplitude * (harmonic_decay ** (h - 1))

            # Phase varies with harmonic and position
            phase = (initial_phase
                    + TAU * harmonic_freq * (freq[..., 0] + freq[..., 1])
                    + TAU * h * 0.1)

            # Add harmonic (real, imaginary)
            complex_sum[..., 0] += harmonic_amp * np.cos(phase)
            complex_sum[..., 1] += harmonic_amp * np.sin(phase)

        # Apply envelope
        complex_sum *= envelope[..., None]

        # Encode as RGBA (2 complex channels)
        output = np.zeros((self.height, self.width, 4), dtype=np.float32)
        output[..., :2] = complex_sum  # Channel 1
        output[..., 2:] = complex_sum * 0.7  # Channel 2 (scaled)

        # Blend with empty state
        proto = output + empty * 0.1

        return proto

    def apply_iota(self, proto: np.ndarray, params: Dict) -> np.ndarray:
        """
        Iota Instantiation: ι : 𝟙 → n

        Modulates proto-unity with instance-specific harmonics.

        Args:
            proto: (H, W, 4) proto-unity state
            params: {
                'harmonic_coeffs': List[float],  # 10 coefficients
                'global_amplitude': float,
                'frequency_range': float
            }

        Returns:
            (H, W, 4) instance state
        """
        # Extract parameters
        harmonic_coeffs = np.array(params.get('harmonic_coeffs', [1.0] * 10), dtype=np.float32)
        global_amplitude = params.get('global_amplitude', 1.0)
        frequency_range = params.get('frequency_range', 2.0)

        # Frequency coordinates
        freq = self.uv * frequency_range
        freq_mag = np.linalg.norm(freq, axis=-1)

        # Map frequency to harmonic bins [0, 9]
        bin_indices = np.clip(freq_mag * 10.0, 0, 9).astype(np.int32)

        # Get amplitude modulation for each pixel
        amp_mod = harmonic_coeffs[bin_indices]  # (H, W)

        # Apply modulation to both channels
        instance = np.zeros_like(proto)
        instance[..., :2] = proto[..., :2] * amp_mod[..., None] * global_amplitude
        instance[..., 2:] = proto[..., 2:] * amp_mod[..., None] * global_amplitude

        return instance

    def apply_epsilon_reverse(self, infinity: np.ndarray, params: Dict) -> np.ndarray:
        """
        Epsilon Preservation: ε_res : ∞ → 𝟙

        Focus/condensation from infinite to proto-unity.
        Creates wave pattern from infinity (dual to gamma from empty).
        """
        # Extract parameters (dual to gamma)
        focus_sigma = params.get('focus_sigma', 2.222)
        extraction_rate = params.get('extraction_rate', 0.0)
        preserve_peaks = params.get('preserve_peaks', True)

        # Frequency domain coordinates
        freq = self.uv * 2.0  # Base frequency
        freq_mag = np.linalg.norm(freq, axis=-1)

        # Focused Gaussian envelope (opposite of gamma's spread)
        envelope = np.exp(-freq_mag**2 / (2.0 * focus_sigma**2))

        # Create wave pattern from infinity (dual to gamma)
        # For standing wave: waves travel in OPPOSITE directions with SAME frequency
        # Gamma: sin(kx - ωt), Epsilon: sin(kx + ωt) → Standing wave: 2sin(kx)cos(ωt)
        num_harmonics = 12
        harmonic_decay = 0.75

        complex_sum = np.zeros((self.height, self.width, 2), dtype=np.float32)

        for h in range(1, num_harmonics + 1):
            harmonic_freq = float(h)
            harmonic_amp = (1.0 - extraction_rate) * (harmonic_decay ** (h - 1))

            # Phase (opposite DIRECTION from gamma, same initial phase for constructive interference)
            # Gamma uses: +TAU * harmonic_freq * (freq[..., 0] + freq[..., 1])
            # Epsilon uses: -TAU * harmonic_freq * (freq[..., 0] + freq[..., 1]) (opposite direction)
            phase = (-TAU * harmonic_freq * (freq[..., 0] + freq[..., 1])
                    - TAU * h * 0.1)  # Match gamma's phase offset but opposite sign

            # Add harmonic
            complex_sum[..., 0] += harmonic_amp * np.cos(phase)
            complex_sum[..., 1] += harmonic_amp * np.sin(phase)

        # Apply focused envelope
        complex_sum *= envelope[..., None]

        # Encode as RGBA
        output = np.zeros((self.height, self.width, 4), dtype=np.float32)
        output[..., :2] = complex_sum
        output[..., 2:] = complex_sum * 0.7

        # Blend with infinity
        proto = output + infinity * extraction_rate

        return proto.astype(np.float32)

    def apply_tau_reverse(self, proto: np.ndarray, params: Dict) -> np.ndarray:
        """
        Tau Expansion: τ_res : 𝟙 → n

        Reconstruction from proto-unity to instance.
        """
        # Simplified: apply projection strength
        projection_strength = params.get('projection_strength', 1.0)
        return proto * projection_strength

    def apply_iota_reverse(self, n: np.ndarray, params: Dict) -> np.ndarray:
        """
        Iota Abstraction: ι_res : n → 𝟙

        Abstract instance back to proto-unity.
        """
        # Reverse: remove instance-specific modulation
        harmonic_coeffs = np.array(params.get('harmonic_coeffs', [1.0] * 10), dtype=np.float32)
        frequency_range = params.get('frequency_range', 2.0)

        freq = self.uv * frequency_range
        freq_mag = np.linalg.norm(freq, axis=-1)
        bin_indices = np.clip(freq_mag * 10.0, 0, 9).astype(np.int32)

        # Inverse amplitude modulation
        amp_mod = harmonic_coeffs[bin_indices]
        amp_mod = np.where(amp_mod > 1e-6, 1.0 / amp_mod, 1.0)

        proto = np.zeros_like(n)
        proto[..., :2] = n[..., :2] * amp_mod[..., None]
        proto[..., 2:] = n[..., 2:] * amp_mod[..., None]

        return proto

    def apply_gamma_reverse(self, proto: np.ndarray, params: Dict) -> np.ndarray:
        """
        Gamma Revelation: γ_res : 𝟙 → ∅

        Grounding proto-unity back to empty.
        """
        # Remove the proto structure, leaving minimal trace
        return proto * 0.1  # Keep 10% as trace

    def apply_tau(self, n: np.ndarray, params: Dict) -> np.ndarray:
        """
        Tau Reduction: τ_gen : n → 𝟙

        Assert instance structure into proto-unity.
        """
        # Compress instance to proto-unity via normalization
        projection_strength = params.get('projection_strength', 1.0)
        normalization_epsilon = params.get('normalization_epsilon', 1e-6)

        # Normalize magnitude
        magnitude = np.linalg.norm(n, axis=-1, keepdims=True) + normalization_epsilon
        proto = n / magnitude * projection_strength

        return proto

    def apply_epsilon(self, proto: np.ndarray, params: Dict) -> np.ndarray:
        """
        Epsilon Erasure: ε_gen : 𝟙 → ∞

        Expose proto-unity to infinity (projection to universal space).
        """
        # Amplify and expand proto to infinity
        energy_weight = params.get('energy_weight', 1.0)
        infinity = proto * energy_weight + 0.5  # Shift toward white
        return np.clip(infinity, 0, 1)

    def compute_cohesion_state(
        self,
        instance: np.ndarray,
        tau_params: Dict,
        epsilon_params: Dict
    ) -> Dict:
        """
        Actualization test: Does instance n cohere with totality?

        Process:
        1. Assertion: n → τ_gen → 𝟙_gen → ε_gen → ∞
        2. Verdict: ∞ → ε_res → 𝟙_res
        3. Cohesion: distance(𝟙_gen, 𝟙_res)

        Returns:
            {
                'assertion': 𝟙_gen,
                'verdict': 𝟙_res,
                'infinity': ∞,
                'delta': float,
                'cohesion': float (0-1),
                'state': 'paradox' | 'evolution' | 'truth'
            }
        """
        # Assertion path
        proto_gen = self.apply_tau(instance, tau_params)
        infinity = self.apply_epsilon(proto_gen, epsilon_params)

        # Verdict path
        proto_res = self.apply_epsilon_reverse(infinity, epsilon_params)

        # Measure cohesion
        delta = float(np.linalg.norm(proto_gen - proto_res))
        cohesion = float(np.exp(-delta / 10.0))  # Map to [0, 1]

        # Classify state
        if cohesion > 0.9:
            state = 'truth'
        elif cohesion > 0.5:
            state = 'evolution'
        else:
            state = 'paradox'

        return {
            'assertion': proto_gen,
            'verdict': proto_res,
            'infinity': infinity,
            'delta': delta,
            'cohesion': cohesion,
            'state': state
        }
