"""
Origin Object V3 - Correct Ouroboros Standing Wave Model

The Origin (○) contains both {} (empty) and ∞ (infinity).
Identity (n) is a STANDING WAVE where Gen and Res paths CONVERGE.

CONVERGENCE (○ → n): Two paths meet at n
├─ Gen path: ∅ → γ_gen → 1 → ι_gen → n (from Empty)
└─ Res path: ∞ → ε_res → 1 → τ_res → n (from Infinite)

DIVERGENCE (n → ○): Act splits n into dual fuel
├─ To Empty:    n → ι_res → 1 → γ_res → ∅ (recycling matter)
└─ To Infinite: n → τ_gen → 1 → ε_gen → ∞ (recycling law)

Each morphism has TWO FACES:
- γ: γ_gen (∅→1) actualization    | γ_res (1→∅) grounding
- ι: ι_gen (1→n) instantiation    | ι_res (n→1) abstraction
- τ: τ_gen (n→1) assertion        | τ_res (1→n) reconstruction
- ε: ε_gen (1→∞) exposure         | ε_res (∞→1) focus

Shader Mapping:
- gamma_genesis.comp      = γ_gen
- gamma_revelation.comp   = γ_res
- iota_instantiation.comp = ι_gen
- iota_abstraction.comp   = ι_res
- tau_reduction.comp      = τ_gen (assertion/compression)
- tau_expansion.comp      = τ_res (reconstruction)
- epsilon_erasure.comp    = ε_gen (exposure/projection)
- epsilon_preservation.comp = ε_res (focus/condensation)
"""

import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
from src.proto_identity import ProtoIdentityManager

@dataclass
class ConvergenceResult:
    """Result from convergence (○ → n)"""
    n_gen: np.ndarray       # n from Gen path (∅ → γ_gen → 1 → ι_gen → n)
    n_res: np.ndarray       # n from Res path (∞ → ε_res → 1 → τ_res → n)
    proto_gen: np.ndarray   # 1 from Gen path
    proto_res: np.ndarray   # 1 from Res path (verdict)
    standing_wave_coherence: float  # How well Gen and Res paths meet at n
    cohesion_gen: Dict      # Actualization test of n_gen (assertion/verdict)
    cohesion_res: Dict      # Actualization test of n_res (assertion/verdict)

@dataclass
class DivergenceResult:
    """Result from divergence (n → ○) with quaternionic vector extraction."""
    empty_output: np.ndarray      # ∅ from matter recycling path
    infinity_output: np.ndarray   # ∞ from law recycling path
    proto_matter: np.ndarray      # 1 from matter path (ι_res)
    proto_law: np.ndarray         # 1 from law path (τ_gen)
    quaternionic_vector: np.ndarray = None  # NEW: (4,) unit quaternion
    multi_octave_quaternions: Dict[int, np.ndarray] = None  # NEW: Octave → quaternion
    proto_identity: np.ndarray = None  # NEW: Reconstructed proto-identity


class Origin:
    """
    Singular origin object ○ containing both {} and ∞

    Implements Standing Wave model:
    - Convergence: Two paths (Gen, Res) meet at n
    - Divergence: Act splits n back into (∅, ∞)
    """

    def __init__(self, width: int = 512, height: int = 512, use_gpu: bool = True):
        self.width = width
        self.height = height
        self.use_gpu = use_gpu

        # Base values
        self.empty = np.zeros((height, width, 4), dtype=np.float32)      # ∅ = #00000000
        self.infinity = np.ones((height, width, 4), dtype=np.float32)    # ∞ = #FFFFFFFF

        # Proto-unity carrier (Phase 1)
        self.proto_unity_carrier: Optional[np.ndarray] = None

        # Initialize Vulkan pipeline
        if use_gpu:
            try:
                from src.gpu.genesis_gpu import GenesisBatchPipeline
                self.pipeline = GenesisBatchPipeline(batch_size=1, width=width, height=height)
                self.has_gpu = True
            except ImportError:
                self.pipeline = None
                self.has_gpu = False
        else:
            self.pipeline = None
            self.has_gpu = False

        # Initialize proto-identity manager
        self.proto_manager = ProtoIdentityManager(width, height, self.pipeline)

    def initialize_carrier(self) -> np.ndarray:
        """
        Initialize stable proto-unity carrier (γ ∪ ε convergence).

        This creates the reference frame for all FM modulation.
        Called once per session to establish the proto-unity carrier.

        Returns:
            carrier: (H, W, 4) proto-unity carrier
        """
        # Default stable parameters for carrier
        gamma_params = {
            'amplitude': 1.0,
            'base_frequency': 2.0,
            'envelope_sigma': 0.45,
            'num_harmonics': 12,
            'harmonic_decay': 0.75,
            'initial_phase': 0.0
        }

        epsilon_params = {
            'extraction_rate': 0.0,
            'focus_sigma': 2.222,
            'base_frequency': 2.0,
            'threshold': 0.1,
            'preserve_peaks': True
        }

        # Create proto-identity via convergence
        carrier = self.create_proto_identity(gamma_params, epsilon_params)
        self.proto_unity_carrier = carrier

        return carrier

    def modulate_carrier(self, input_signal: np.ndarray,
                        iota_params: Dict, tau_params: Dict) -> np.ndarray:
        """
        Modulate proto-unity carrier with input using ι/τ (FM modulation).

        This implements the encoding path:
        carrier + input → modulated proto-identity

        Args:
            input_signal: (H, W, 4) input frequency representation
            iota_params: Parameters for iota modulation
            tau_params: Parameters for tau modulation

        Returns:
            modulated: (H, W, 4) modulated proto-identity
        """
        if self.proto_unity_carrier is None:
            raise ValueError("Carrier not initialized. Call initialize_carrier() first.")

        from src.memory.fm_modulation_base import FMModulationBase
        fm = FMModulationBase()

        # FM modulation: carrier * input
        modulated = fm.modulate(self.proto_unity_carrier, input_signal)

        return modulated

    def demodulate_carrier(self, proto_identity: np.ndarray) -> np.ndarray:
        """
        Reverse: Extract signal from proto-identity using carrier.

        This implements the decoding path:
        modulated proto-identity → signal

        Args:
            proto_identity: (H, W, 4) modulated proto-identity

        Returns:
            signal: (H, W, 4) extracted signal
        """
        if self.proto_unity_carrier is None:
            raise ValueError("Carrier not initialized. Call initialize_carrier() first.")

        from src.memory.fm_modulation_base import FMModulationBase
        fm = FMModulationBase()

        # Demodulate proto-identity to extract original signal
        # Using default modulation_depth of 0.5 (same as in modulate())
        signal = fm.demodulate(proto_identity, self.proto_unity_carrier, modulation_depth=0.5)

        return signal

    def create_proto_identity(self, gamma_params: Dict, epsilon_params: Dict) -> np.ndarray:
        """
        Delegate to ProtoIdentityManager to create proto-identity.

        Args:
            gamma_params: Parameters for gamma morphism
            epsilon_params: Parameters for epsilon morphism

        Returns:
            proto_identity: Standing wave from Gen ∪ Res convergence
        """
        return self.proto_manager.create_proto_identity(
            gamma_params, epsilon_params, self.empty, self.infinity, self.has_gpu
        )

    def project_proto_identity(self, proto_identity: np.ndarray, n: np.ndarray) -> np.ndarray:
        """
        Delegate to ProtoIdentityManager to project proto against input.

        Args:
            proto_identity: Converged standing wave
            n: Input frequency representation

        Returns:
            standing_wave: Interference pattern
        """
        return self.proto_manager.project_proto_identity(proto_identity, n)

    def Gen(self, gamma_params: Dict, iota_params: Dict,
            input_n: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Gen path: ∅ → γ_gen → 1 → ι_gen → n (Generation)

        MODIFIED: Support query mode (input_n=None) and encode mode (input_n provided).

        Query mode: Return proto-identity only
        Encode mode: Project proto against input_n, then apply iota

        Args:
            gamma_params: Parameters for gamma morphism
            iota_params: Parameters for iota morphism
            input_n: Optional input for projection (encode mode)

        Returns:
            proto_identity (query mode) or n (encode mode)
        """
        # Derive epsilon for convergence
        epsilon_params = self.proto_manager.derive_epsilon_from_gamma(gamma_params)

        # Create proto-identity via Gen ∪ Res convergence
        proto_identity = self.create_proto_identity(gamma_params, epsilon_params)

        if input_n is None:
            # Query mode: Return proto-identity
            return proto_identity

        # Encode mode: Project and apply iota
        standing_wave = self.project_proto_identity(proto_identity, input_n)

        if self.has_gpu:
            # GPU path
            n = self.pipeline.execute_iota(iota_params, standing_wave)
        else:
            # CPU fallback
            from src.pipeline.cpu import CPUPipeline
            cpu_pipeline = CPUPipeline(self.width, self.height)
            n = cpu_pipeline.apply_iota(standing_wave, iota_params)

        return n

    def Res(self, epsilon_params: Dict, tau_params: Dict,
            input_n: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Res path: ∞ → ε_res → 1 → τ_res → n (Resolution)

        MODIFIED: Support query mode (input_n=None) and encode mode (input_n provided).

        Query mode: Return proto-identity only
        Encode mode: Project proto against input_n, then apply tau

        Args:
            epsilon_params: Parameters for epsilon morphism
            tau_params: Parameters for tau morphism
            input_n: Optional input for projection (encode mode)

        Returns:
            proto_identity (query mode) or n (encode mode)
        """
        # Derive gamma for convergence
        gamma_params = self.proto_manager.derive_gamma_from_epsilon(epsilon_params)

        # Create proto-identity via Gen ∪ Res convergence
        proto_identity = self.create_proto_identity(gamma_params, epsilon_params)

        if input_n is None:
            # Query mode: Return proto-identity
            return proto_identity

        # Encode mode: Project and apply tau
        standing_wave = self.project_proto_identity(proto_identity, input_n)

        if self.has_gpu:
            # GPU path
            n = self.pipeline.execute_tau_reverse(tau_params, standing_wave)
        else:
            # CPU fallback
            from src.pipeline.cpu import CPUPipeline
            cpu_pipeline = CPUPipeline(self.width, self.height)
            n = cpu_pipeline.apply_tau_reverse(standing_wave, tau_params)

        return n

    def _act_dual_gpu(self, instance: np.ndarray, iota_params: Dict,
                      gamma_params: Dict, tau_params: Dict, epsilon_params: Dict) -> DivergenceResult:
        """GPU path for Act_dual divergence."""
        # Gen reversal: n → ι_rev → 𝟙_gen
        identity_gen = self.pipeline.execute_iota_reverse(iota_params, instance)

        # Res reversal: n → τ_rev → 𝟙_res
        identity_res = self.pipeline.execute_tau_reverse_divergence(tau_params, instance)

        # Combine identities (standing wave superposition)
        identity_combined = (identity_gen + identity_res) / 2.0

        # Diverge to empty: 𝟙 → γ_rev → ∅
        empty_output = self.pipeline.execute_gamma_reverse(gamma_params, identity_combined)

        # Diverge to infinity: 𝟙 → ε_rev → ∞
        infinity_output = self.pipeline.execute_epsilon_reverse_divergence(epsilon_params, identity_combined)

        return DivergenceResult(
            empty_output=empty_output,
            infinity_output=infinity_output,
            proto_matter=identity_combined,
            proto_law=identity_combined
        )

    def _act_dual_cpu(self, instance: np.ndarray, iota_params: Dict,
                      gamma_params: Dict, tau_params: Dict, epsilon_params: Dict) -> DivergenceResult:
        """CPU path for Act_dual divergence."""
        from src.pipeline.cpu import CPUPipeline
        cpu_pipeline = CPUPipeline(self.width, self.height)

        # Gen reversal: n → ι_rev → 𝟙_gen
        identity_gen = cpu_pipeline.apply_iota_reverse(instance, iota_params)

        # Res reversal: n → τ_rev → 𝟙_res
        identity_res = cpu_pipeline.apply_tau_reverse(instance, tau_params)

        # Combine identities
        identity_combined = (identity_gen + identity_res) / 2.0

        # Diverge to empty: 𝟙 → γ_rev → ∅
        empty_output = cpu_pipeline.apply_gamma_reverse(identity_combined, gamma_params)

        # Diverge to infinity: 𝟙 → ε_rev → ∞
        infinity_output = cpu_pipeline.apply_epsilon_reverse(identity_combined, epsilon_params)

        return DivergenceResult(
            empty_output=empty_output,
            infinity_output=infinity_output,
            proto_matter=identity_combined,
            proto_law=identity_combined
        )

    def Act_dual(self, instance: np.ndarray, iota_params: Dict,
                 gamma_params: Dict, tau_params: Dict, epsilon_params: Dict) -> DivergenceResult:
        """Act_dual: n → (ι ∪ τ) → 𝟙 → (γ ∪ ε) → (∅ ∪ ∞) → ○"""
        if self.has_gpu:
            return self._act_dual_gpu(instance, iota_params, gamma_params, tau_params, epsilon_params)
        else:
            return self._act_dual_cpu(instance, iota_params, gamma_params, tau_params, epsilon_params)

    def compute_cohesion_state(
        self,
        instance: np.ndarray,
        tau_params: Dict,
        epsilon_params: Dict
    ) -> Dict:
        """
        Actualization: Test if instance n is coherent (real)

        n as lens - assertion/verdict dialogue:
        1. Assertion: n → τ_gen → 𝟙_gen → ε_gen → ∞ (n's claim)
        2. Verdict: ∞ → ε_res → 𝟙_res (what emerges from totality)
        3. Cohesion: distance(𝟙_gen, 𝟙_res) (reality test)

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
        if self.has_gpu:
            # Use GPU implementation
            return self.pipeline.compute_cohesion_state(instance, tau_params, epsilon_params)
        else:
            # Use CPU implementation
            from src.pipeline.cpu import CPUPipeline
            cpu_pipeline = CPUPipeline(self.width, self.height)
            return cpu_pipeline.compute_cohesion_state(instance, tau_params, epsilon_params)

    def _convergence_gpu(self, gamma_params: Dict, iota_params: Dict,
                        epsilon_params: Dict, tau_params: Dict) -> Tuple:
        """GPU path for convergence computation."""
        # --- Gen Path: ∅ → γ_gen → 1 → ι_gen → n_gen ---
        self.pipeline.execute_gamma_once(gamma_params)
        proto_gen = self.pipeline.download_working_buffer()
        n_gen = self.pipeline.execute_iota_once(iota_params)

        # --- Res Path: ∞ → ε_res → 1 → τ_res → n_res ---
        # The verdict (𝟙_res) comes directly from ε_res
        proto_res = self.pipeline.execute_epsilon_reverse(epsilon_params, self.infinity)
        n_res = self.pipeline.execute_tau_reverse(tau_params, proto_res)

        return n_gen, n_res, proto_gen, proto_res

    def _convergence_cpu(self, gamma_params: Dict, iota_params: Dict,
                        epsilon_params: Dict, tau_params: Dict) -> Tuple:
        """CPU path for convergence computation."""
        from src.pipeline.cpu import CPUPipeline
        cpu_pipeline = CPUPipeline(self.width, self.height)

        # --- Gen Path: ∅ → γ_gen → 1 → ι_gen → n_gen ---
        proto_gen = cpu_pipeline.apply_gamma(self.empty, gamma_params)
        n_gen = cpu_pipeline.apply_iota(proto_gen, iota_params)

        # --- Res Path: ∞ → ε_res → 1 → τ_res → n_res ---
        proto_res = cpu_pipeline.apply_epsilon_reverse(self.infinity, epsilon_params)
        n_res = cpu_pipeline.apply_tau_reverse(proto_res, tau_params)

        return n_gen, n_res, proto_gen, proto_res

    def Convergence(
        self,
        gamma_params: Dict,
        iota_params: Dict,
        epsilon_params: Dict,
        tau_params: Dict
    ) -> ConvergenceResult:
        """
        Convergence: ○ → n (Standing Wave Formation)

        Two paths meet at n:
        - Gen path: ∅ → γ_gen → 1 → ι_gen → n_gen
        - Res path: ∞ → ε_res → 1 → τ_res → n_res

        Then test BOTH n_gen and n_res for actualization (assertion/verdict).

        Returns:
            ConvergenceResult with:
            - standing_wave_coherence: How well Gen and Res meet
            - cohesion_gen: Actualization test of n_gen
            - cohesion_res: Actualization test of n_res
        """
        if self.has_gpu:
            n_gen, n_res, proto_gen, proto_res = self._convergence_gpu(
                gamma_params, iota_params, epsilon_params, tau_params)
        else:
            n_gen, n_res, proto_gen, proto_res = self._convergence_cpu(
                gamma_params, iota_params, epsilon_params, tau_params)

        # Measure Standing Wave: How well Gen and Res paths meet at n
        n_distance = float(np.linalg.norm(n_gen - n_res))
        standing_wave_coherence = float(np.exp(-n_distance / 100.0))

        # Actualization Test: Test BOTH n_gen and n_res for coherence
        cohesion_gen = self.compute_cohesion_state(n_gen, tau_params, epsilon_params)
        cohesion_res = self.compute_cohesion_state(n_res, tau_params, epsilon_params)

        return ConvergenceResult(
            n_gen=n_gen,
            n_res=n_res,
            proto_gen=proto_gen,
            proto_res=proto_res,
            standing_wave_coherence=standing_wave_coherence,
            cohesion_gen=cohesion_gen,
            cohesion_res=cohesion_res
        )

    def _act_gpu(self, n: np.ndarray, iota_params: Dict, gamma_params: Dict,
                 tau_params: Dict, epsilon_params: Dict) -> DivergenceResult:
        """GPU path for Act divergence."""
        # Upload n for processing
        self.pipeline.upload_instance(0, n)

        # --- To Empty (Matter Recycling): n → ι_res → 1 → γ_res → ∅ ---
        # ι_res: n → 1 (iota_abstraction.comp)
        proto_matter = self.pipeline.execute_iota_reverse(iota_params, n)

        # γ_res: 1 → ∅ (gamma_revelation.comp)
        empty_output = self.pipeline.execute_gamma_reverse(gamma_params, proto_matter)

        # --- To Infinite (Law Recycling): n → τ_gen → 1 → ε_gen → ∞ ---
        # τ_gen: n → 1 (tau_reduction.comp)
        self.pipeline.execute_tau_once(tau_params, 0)
        proto_law = self.pipeline.download_working_buffer()

        # ε_gen: 1 → ∞ (epsilon_erasure.comp)
        infinity_output = self.pipeline.execute_epsilon_once(epsilon_params)

        return DivergenceResult(
            empty_output=empty_output,
            infinity_output=infinity_output,
            proto_matter=proto_matter,
            proto_law=proto_law
        )

    def _act_cpu(self, n: np.ndarray, iota_params: Dict, gamma_params: Dict,
                 tau_params: Dict, epsilon_params: Dict) -> DivergenceResult:
        """CPU path for Act divergence."""
        from src.pipeline.cpu import CPUPipeline
        cpu_pipeline = CPUPipeline(self.width, self.height)

        # --- To Empty (Matter Recycling): n → ι_res → 1 → γ_res → ∅ ---
        # ι_res: n → 1 (iota_abstraction.comp)
        proto_matter = cpu_pipeline.apply_iota_reverse(n, iota_params)

        # γ_res: 1 → ∅ (gamma_revelation.comp)
        empty_output = cpu_pipeline.apply_gamma_reverse(proto_matter, gamma_params)

        # --- To Infinite (Law Recycling): n → τ_gen → 1 → ε_gen → ∞ ---
        # τ_gen: n → 1 (tau_reduction.comp)
        proto_law = cpu_pipeline.apply_tau(n, tau_params)

        # ε_gen: 1 → ∞ (epsilon_erasure.comp)
        infinity_output = cpu_pipeline.apply_epsilon(proto_law, epsilon_params)

        return DivergenceResult(
            empty_output=empty_output,
            infinity_output=infinity_output,
            proto_matter=proto_matter,
            proto_law=proto_law
        )

    def Act(self, standing_wave: np.ndarray) -> DivergenceResult:
        """
        Act: Extract quaternionic vector from standing wave.

        MODIFIED for Week 2: Simplified to extract quaternionic vector.
        Full divergence (n → ○) available via Act_full().

        Args:
            standing_wave: (H, W, 4) proto-identity interference pattern

        Returns:
            DivergenceResult with quaternionic_vector and multi_octave_quaternions
        """
        # Extract quaternionic vector (NEW - Week 2)
        quaternionic_vector = self.proto_manager.extract_quaternion(standing_wave)

        # Extract multi-octave quaternions (optional, for storage)
        multi_octave_quaternions = self.proto_manager.extract_multi_octave_quaternions(standing_wave)

        # Existing reverse projection to empty/infinity (simplified)
        energy_to_empty = float(np.linalg.norm(standing_wave - self.empty))
        energy_to_infinity = float(np.linalg.norm(standing_wave - self.infinity))

        return DivergenceResult(
            empty_output=self.empty,  # For compatibility
            infinity_output=self.infinity,  # For compatibility
            proto_matter=standing_wave,  # For compatibility
            proto_law=standing_wave,  # For compatibility
            quaternionic_vector=quaternionic_vector,  # NEW
            multi_octave_quaternions=multi_octave_quaternions,  # NEW
            proto_identity=standing_wave  # NEW
        )

    def Act_full(
        self,
        n: np.ndarray,
        iota_params: Dict,
        gamma_params: Dict,
        tau_params: Dict,
        epsilon_params: Dict
    ) -> DivergenceResult:
        """
        Act_full: n → ○ (Full Divergence - Return to Origin)

        Splits n into dual fuel:
        - To Empty (matter):    n → ι_res → 1 → γ_res → ∅
        - To Infinite (law):    n → τ_gen → 1 → ε_gen → ∞

        Shaders:
        - ι_res: iota_abstraction.comp (n → 1)
        - γ_res: gamma_revelation.comp (1 → ∅)
        - τ_gen: tau_reduction.comp (n → 1)
        - ε_gen: epsilon_erasure.comp (1 → ∞)
        """
        if self.has_gpu:
            return self._act_gpu(n, iota_params, gamma_params, tau_params, epsilon_params)
        else:
            return self._act_cpu(n, iota_params, gamma_params, tau_params, epsilon_params)

    def _extract_metrics(self, conv_result: ConvergenceResult) -> Tuple[float, float]:
        """Extract coherence metrics from convergence result."""
        sw_coherence = conv_result.standing_wave_coherence
        gen_cohesion = conv_result.cohesion_gen['cohesion']
        res_cohesion = conv_result.cohesion_res['cohesion']
        avg_cohesion = (gen_cohesion + res_cohesion) / 2.0
        return sw_coherence, avg_cohesion

    def _print_iteration(self, iteration: int, max_iterations: int, conv_result: ConvergenceResult,
                        sw_coherence: float, gamma_params: Dict) -> None:
        """Print iteration progress."""
        gen_cohesion = conv_result.cohesion_gen['cohesion']
        res_cohesion = conv_result.cohesion_res['cohesion']
        print(f"  Iter {iteration+1:3d}/{max_iterations} | "
              f"SW: {sw_coherence:.4f} | "
              f"Gen: {gen_cohesion:.4f} ({conv_result.cohesion_gen['state']}) | "
              f"Res: {res_cohesion:.4f} ({conv_result.cohesion_res['state']}) | "
              f"γ_amp: {gamma_params.get('amplitude', 0):.2f}")

    def _check_convergence(self, sw_coherence: float, avg_cohesion: float,
                          threshold: float, conv_result: ConvergenceResult) -> bool:
        """Check if Ouroboros cycle has converged."""
        if sw_coherence >= threshold and avg_cohesion >= threshold:
            print(f"\n  ✓ Ouroboros converged!")
            print(f"    Standing Wave: {sw_coherence:.6f}")
            print(f"    Gen Cohesion: {conv_result.cohesion_gen['cohesion']:.6f} ({conv_result.cohesion_gen['state']})")
            print(f"    Res Cohesion: {conv_result.cohesion_res['cohesion']:.6f} ({conv_result.cohesion_res['state']})")
            return True
        return False

    def _update_params(self, gamma_params: Dict, tau_params: Dict,
                      sw_coherence: float, avg_cohesion: float,
                      gamma_lr: float, tau_lr: float) -> None:
        """Update parameters based on coherence metrics."""
        sw_error = 1.0 - sw_coherence
        cohesion_error = 1.0 - avg_cohesion
        total_error = (sw_error + cohesion_error) / 2.0

        # Adjust γ (Initial Condition)
        if 'amplitude' in gamma_params:
            gamma_params['amplitude'] *= (1.0 - gamma_lr * total_error)
        if 'base_frequency' in gamma_params:
            gamma_params['base_frequency'] += (np.random.rand() - 0.5) * 0.1 * total_error

        # Adjust τ (Assertion strength)
        if 'projection_strength' in tau_params:
            tau_params['projection_strength'] *= (1.0 + tau_lr * cohesion_error * 0.5)
            tau_params['projection_strength'] = np.clip(tau_params['projection_strength'], 0.1, 2.0)

    def ouroboros_cycle(
        self,
        gamma_params: Dict,
        iota_params: Dict,
        tau_params: Dict,
        epsilon_params: Dict,
        max_iterations: int = 100,
        gamma_lr: float = 0.01,
        tau_lr: float = 0.005,
        convergence_threshold: float = 0.95
    ) -> Tuple[Dict, Dict, Dict, Dict, ConvergenceResult]:
        """
        Complete Ouroboros cycle: Convergence (○ → n) + Divergence (n → ○).

        Autopoietic loop where the system maintains itself through continuous
        convergence and divergence cycles.
        """
        print("\n╔═══════════════════════════════════════════════════════════╗")
        print("║          OUROBOROS: Standing Wave Autopoiesis            ║")
        print("║  Convergence: ○ → n | Divergence: n → ○                 ║")
        print("╚═══════════════════════════════════════════════════════════╝\n")

        for iteration in range(max_iterations):
            # Phase 1: Convergence (○ → n)
            conv_result = self.Convergence(gamma_params, iota_params, epsilon_params, tau_params)
            sw_coherence, avg_cohesion = self._extract_metrics(conv_result)
            self._print_iteration(iteration, max_iterations, conv_result, sw_coherence, gamma_params)

            # Check convergence
            if self._check_convergence(sw_coherence, avg_cohesion, convergence_threshold, conv_result):
                break

            # Phase 2: Divergence (n → ○)
            div_result = self.Act_full(conv_result.n_gen, iota_params, gamma_params, tau_params, epsilon_params)

            # Phase 3: Update parameters
            self._update_params(gamma_params, tau_params, sw_coherence, avg_cohesion, gamma_lr, tau_lr)

        print(f"\n╚═══════════════════════════════════════════════════════════╝")
        print(f"Ouroboros cycle completed.\n")

        return gamma_params, iota_params, tau_params, epsilon_params, conv_result

    def cleanup(self):
        if self.has_gpu and self.pipeline is not None:
            self.pipeline.free()

    def __del__(self):
        try:
            self.cleanup()
        except:
            pass

    def __repr__(self) -> str:
        return f"Origin(○) [Standing Wave Model, width={self.width}, height={self.height}, gpu={self.has_gpu}]"
