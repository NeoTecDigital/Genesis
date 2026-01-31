"""TaylorSynthesizer - Iterative refinement with natural convergence.

The core synthesis engine that iteratively refines proto-identities through
Taylor series expansion via Origin morphisms (ι ∪ τ).

Key principles:
1. Identity ≠ must match core - new discoveries are valid identities
2. Paradox NOT rejected - split into P/!P branches, both valid
3. No max_iterations as normal limit - natural convergence via delta < epsilon only
4. Evolution includes cycling AND chaos - both are valid outputs
5. All states produce output - no rejection
6. Unstable states quantified - entropy metrics + resistance map + explanation
"""

import numpy as np
from typing import List, Optional, Dict, Tuple
from src.memory.synthesis_types import (
    SynthesisResult,
    IdentityBranch,
    UnstableSystemStub
)
from src.memory.experiential_reflector import ExperientialReflector
from src.memory.identity_branch_manager import IdentityBranchManager
from src.memory.memory_hierarchy import MemoryHierarchy
from src.origin import Origin
import uuid


class TaylorSynthesizer:
    """Main iterative 'thinking' loop with natural convergence detection.

    Refines proto-identities through Taylor series expansion, detecting:
    - Identity: Natural convergence (delta < epsilon)
    - Paradox: Multiple stable attractors (split into P/!P branches)
    - Evolution-cycling: Periodic patterns
    - Evolution-chaotic: High entropy, no pattern
    """

    def __init__(
        self,
        epsilon: float = 1e-4,
        safety_max_iterations: int = 10000,
        clustering_check_interval: int = 10
    ):
        """Initialize Taylor synthesizer.

        Args:
            epsilon: Natural convergence threshold (delta < epsilon)
            safety_max_iterations: Bug detection only (not normal termination)
            clustering_check_interval: How often to check for state changes
        """
        self.epsilon = epsilon
        self.safety_max_iterations = safety_max_iterations
        self.clustering_check_interval = clustering_check_interval
        self.reflector = None  # Initialized in synthesize()
        self.branch_manager = IdentityBranchManager()
        self.origin = Origin(width=512, height=512, use_gpu=False)

    def synthesize(
        self,
        query: str,
        memory_hierarchy: MemoryHierarchy,
        proto_unity_carrier: np.ndarray
    ) -> SynthesisResult:
        """Main synthesis entry point.

        Args:
            query: Query text to synthesize
            memory_hierarchy: Memory system with core/experiential layers
            proto_unity_carrier: Stable carrier for FM modulation

        Returns:
            SynthesisResult with state and proto-identities
        """
        # Set carrier on Origin for FM modulation
        self.origin.proto_unity_carrier = proto_unity_carrier

        # Initialize reflector
        from src.memory.feedback_loop import FeedbackLoop
        feedback_loop = FeedbackLoop(
            memory_hierarchy.core_memory,
            memory_hierarchy.experiential_memory
        )
        self.reflector = ExperientialReflector(feedback_loop)

        # Initial encoding - create proto from query
        initial_proto = self._encode_query(query, proto_unity_carrier)

        # Create initial branch
        branch = IdentityBranch(
            branch_id=f"branch_{uuid.uuid4().hex[:8]}",
            proto_identity=initial_proto,
            trajectory=[initial_proto.copy()],
            coherence_history=[],
            state='active'
        )

        # Run Taylor loop
        refined_branch = self._taylor_loop(branch, memory_hierarchy)

        # Detect final state and handle
        state = self._detect_state(refined_branch)

        if state == 'identity':
            return self._handle_identity(refined_branch, memory_hierarchy.core_memory)
        elif state == 'paradox':
            return self._handle_paradox(refined_branch, memory_hierarchy)
        elif state == 'evolution_cycling':
            return self._handle_evolution_cycling(refined_branch)
        else:  # evolution_chaotic
            return self._handle_evolution_chaotic(refined_branch, query)

    def _taylor_loop(
        self,
        branch: IdentityBranch,
        memory_hierarchy: MemoryHierarchy
    ) -> IdentityBranch:
        """Iterative refinement loop with natural convergence.

        Refines until delta < epsilon (natural convergence).
        safety_max_iterations is for bug detection only.

        Args:
            branch: Branch to refine
            memory_hierarchy: Memory system

        Returns:
            Refined branch with updated state
        """
        iteration = 0
        delta = float('inf')
        prev_proto = branch.proto_identity.copy()

        while delta >= self.epsilon and iteration < self.safety_max_iterations:
            # Apply Taylor step (Origin morphisms)
            refined_proto = self._apply_taylor_step(
                branch.proto_identity,
                memory_hierarchy.proto_unity_carrier
            )

            # Measure dual coherence
            core_coherence, internal_coherence = self.reflector.measure_dual_coherence(
                refined_proto,
                memory_hierarchy.core_memory,
                branch.trajectory
            )

            # Update branch
            branch.proto_identity = refined_proto
            branch.trajectory.append(refined_proto.copy())
            branch.coherence_history.append((core_coherence + internal_coherence) / 2.0)

            # Calculate delta
            delta = float(np.linalg.norm(refined_proto - prev_proto))
            prev_proto = refined_proto.copy()

            # Check state periodically
            if iteration % self.clustering_check_interval == 0 and iteration > 0:
                potential_state = self._detect_state(branch)
                if potential_state in ['paradox', 'evolution_cycling', 'evolution_chaotic']:
                    branch.state = potential_state
                    break

            iteration += 1

        # Set final state if converged
        if delta < self.epsilon:
            branch.state = 'converged'

        return branch

    def _apply_taylor_step(
        self,
        proto: np.ndarray,
        carrier: np.ndarray
    ) -> np.ndarray:
        """Apply Taylor series step via Origin morphisms.

        Uses ι ∪ τ application for refinement.

        Args:
            proto: Current proto-identity (H, W, 4)
            carrier: Proto-unity carrier

        Returns:
            Refined proto-identity
        """
        # Parameters for ι/τ morphisms
        iota_params = {
            'frequency_shift': 0.1,
            'phase_modulation': 0.05
        }
        tau_params = {
            'projection_strength': 0.8,
            'compression_rate': 0.9
        }

        # Apply ι (instantiation) and τ (reconstruction) for refinement
        # This creates standing wave interference
        iota_refined = self.origin.modulate_carrier(proto, iota_params, tau_params)

        return iota_refined

    def _detect_state(self, branch: IdentityBranch) -> str:
        """Detect synthesis state.

        Args:
            branch: Branch to analyze

        Returns:
            State: 'identity', 'paradox', 'evolution_cycling', 'evolution_chaotic'
        """
        # Need sufficient history
        if len(branch.trajectory) < 5:
            return 'active'

        # Check for natural convergence
        if len(branch.trajectory) >= 2:
            recent_delta = float(np.linalg.norm(
                branch.trajectory[-1] - branch.trajectory[-2]
            ))
            if recent_delta < self.epsilon:
                return 'identity'

        # Check for paradox (multiple attractors)
        if self.branch_manager.detect_paradox(
            branch.trajectory,
            branch.coherence_history
        ):
            return 'paradox'

        # Check for cycling
        periodicity = self._detect_periodicity(branch.trajectory)
        if periodicity is not None:
            return 'evolution_cycling'

        # Check for chaos (high entropy)
        if len(branch.trajectory) >= 10:
            entropy_metrics = self._calculate_entropy(branch.trajectory[-10:])
            if entropy_metrics['sample_entropy'] > 0.5:
                return 'evolution_chaotic'

        return 'active'

    def _handle_identity(self, branch: IdentityBranch, core_memory) -> SynthesisResult:
        """Handle converged identity state. New discoveries are valid identities."""
        core_coherence = self.reflector.measure_core_coherence(
            branch.proto_identity, core_memory
        )

        # Determine identity type
        if core_coherence > 0.9:
            explanation = "Identity: Aligned with core knowledge"
        elif core_coherence > 0.5:
            explanation = "Identity: Refined existing knowledge"
        else:
            explanation = "Identity: Novel discovery (valid new knowledge)"

        return SynthesisResult(
            proto_identities=[branch.proto_identity],
            state='identity',
            coherence_scores=[core_coherence],
            explanation=explanation,
            branches=[branch]
        )

    def _handle_paradox(
        self, branch: IdentityBranch, memory_hierarchy: MemoryHierarchy
    ) -> SynthesisResult:
        """Handle paradox state - split into P/!P branches. Both are valid."""
        attractors = self.branch_manager.detect_attractors(branch.trajectory)
        split_branches = self.branch_manager.split_paradox(
            branch.proto_identity, attractors
        )

        # Refine each branch independently
        refined_branches = [
            self._taylor_loop(sb, memory_hierarchy) for sb in split_branches
        ]

        protos = [b.proto_identity for b in refined_branches]
        coherences = [
            self.reflector.measure_core_coherence(p, memory_hierarchy.core_memory)
            for p in protos
        ]
        resistance_map = self._measure_resistance(protos)

        return SynthesisResult(
            proto_identities=protos,
            state='paradox',
            coherence_scores=coherences,
            resistance_map=resistance_map,
            explanation=f"Paradox: {len(protos)} attractors. P and !P both valid.",
            branches=refined_branches
        )

    def _handle_evolution_cycling(self, branch: IdentityBranch) -> SynthesisResult:
        """Handle evolution-cycling state."""
        periodicity = self._detect_periodicity(branch.trajectory)
        period = periodicity['period']
        cycle_protos = branch.trajectory[-period:]

        return SynthesisResult(
            proto_identities=cycle_protos,
            state='evolution_cycling',
            coherence_scores=[0.0] * len(cycle_protos),
            entropy_metrics=periodicity,
            explanation=f"Evolution (cycling): Period {period}, "
                       f"Confidence {periodicity['confidence']:.2f}",
            branches=[branch]
        )

    def _handle_evolution_chaotic(
        self, branch: IdentityBranch, query: str
    ) -> SynthesisResult:
        """Handle evolution-chaotic state.

        Note: query parameter retained for signature compatibility but NOT stored in stub.
        Identification via proto_identities (frequency-based), maintaining zero-text-storage.
        """
        entropy_metrics = self._calculate_entropy(branch.trajectory)
        resistance_map = self._measure_resistance(branch.trajectory[-5:])
        explanation = (
            f"Evolution (chaotic): Entropy {entropy_metrics['sample_entropy']:.3f}, "
            f"Variance {entropy_metrics['coherence_variance']:.3f}"
        )

        stub = UnstableSystemStub(
            proto_identities=branch.trajectory[-5:],
            entropy_metrics=entropy_metrics,
            resistance_map=resistance_map,
            explanation=explanation,
            iterations_attempted=len(branch.trajectory)
        )

        return SynthesisResult(
            proto_identities=[branch.proto_identity],
            state='evolution_chaotic',
            coherence_scores=[0.0],
            entropy_metrics=entropy_metrics,
            resistance_map=resistance_map,
            explanation=explanation,
            branches=[branch],
            unstable_stub=stub
        )

    def _calculate_entropy(self, trajectory: List[np.ndarray]) -> Dict[str, float]:
        """Calculate entropy metrics for trajectory."""
        if len(trajectory) < 2:
            return {'sample_entropy': 0.0, 'coherence_variance': 0.0}

        # Sample entropy (complexity via pairwise distances)
        flattened = np.array([p.flatten() for p in trajectory])
        dists = [np.linalg.norm(flattened[i] - flattened[i + 1])
                 for i in range(len(flattened) - 1)]
        sample_entropy = float(np.std(dists) / (np.mean(dists) + 1e-8))

        # Coherence variance (stability)
        coherence_variance = 0.0
        if hasattr(self, 'reflector') and self.reflector is not None:
            coherences = [
                self.reflector.measure_internal_coherence(trajectory[i], trajectory[:i])
                for i in range(1, len(trajectory))
            ]
            coherence_variance = float(np.var(coherences)) if coherences else 0.0

        return {'sample_entropy': sample_entropy, 'coherence_variance': coherence_variance}

    def _measure_resistance(self, protos: List[np.ndarray]) -> Dict[Tuple[int, int], float]:
        """Measure pairwise semantic distances (resistance)."""
        return {
            (i, j): float(np.linalg.norm(protos[i].flatten() - protos[j].flatten()))
            for i in range(len(protos))
            for j in range(i + 1, len(protos))
        }

    def _detect_periodicity(self, trajectory: List[np.ndarray]) -> Optional[Dict]:
        """Detect periodic pattern via autocorrelation."""
        if len(trajectory) < 10:
            return None

        # Compute pairwise distances along trajectory
        flattened = np.array([p.flatten() for p in trajectory])
        distances = np.array([
            np.linalg.norm(flattened[i] - flattened[i + 1])
            for i in range(len(flattened) - 1)
        ])
        mean_dist = distances.mean()

        # Look for repeating patterns
        for period in range(2, min(20, len(distances) // 2)):
            pattern_match = sum(
                1 for i in range(len(distances) - period)
                if abs(distances[i] - distances[i + period]) < mean_dist * 0.3
            )
            confidence = pattern_match / (len(distances) - period)
            if confidence > 0.6:
                return {'period': period, 'confidence': confidence}

        return None

    def _encode_query(self, query: str, carrier: np.ndarray) -> np.ndarray:
        """Encode query text into initial proto-identity."""
        import hashlib
        query_hash = hashlib.sha256(query.encode()).digest()
        H, W = carrier.shape[:2]
        proto = np.zeros((H, W, 4), dtype=np.float32)

        # Create Gaussian peaks from hash
        for i in range(0, len(query_hash), 4):
            x = int.from_bytes(query_hash[i:i+2], 'big') % W
            y = int.from_bytes(query_hash[i+2:i+4], 'big') % H
            y_grid, x_grid = np.ogrid[:H, :W]
            gaussian = np.exp(-((x_grid - x)**2 + (y_grid - y)**2) / 200.0)
            for c in range(4):
                proto[:, :, c] += gaussian * 0.1

        return np.clip(proto, 0, 1)

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"TaylorSynthesizer("
            f"epsilon={self.epsilon}, "
            f"safety_max={self.safety_max_iterations})"
        )
