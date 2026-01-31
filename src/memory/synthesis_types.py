"""Synthesis Result Data Structures for Taylor Series Refinement.

This module defines data structures for Genesis's Phase C iterative synthesis
system, where the experiential layer refines proto-identities through Taylor
series expansion to achieve coherence-driven "thinking" capability.

Key Concepts:
- UnstableSystemStub: Captures failed synthesis attempts with diagnostic data
- IdentityBranch: Tracks individual refinement trajectories during synthesis
- SynthesisResult: Comprehensive result of synthesis operation (identity/paradox/evolution)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import time


@dataclass
class UnstableSystemStub:
    """Records failed synthesis attempts with diagnostic information.

    Captures the state of a synthesis that failed to converge, providing
    insight into why the system couldn't reach coherence. Essential for
    analyzing paradoxes, cycling, and chaotic behavior.

    The stub identifies failed synthesis via proto_identities (frequency-based),
    NOT raw text. This maintains Genesis's zero-text-storage principle.

    Attributes:
        proto_identities: List of proto-identity candidates (H×W×4)
        entropy_metrics: Sample entropy and coherence variance measurements
        resistance_map: Pairwise conflicts {(i,j): resistance_value}
        oscillation_pattern: Cyclic behavior pattern if detected
        explanation: Human-readable behavioral description (e.g., "Ideas A and B resist synthesis")
        timestamp: Unix timestamp of synthesis attempt
        iterations_attempted: Number of refinement iterations before failure
    """
    proto_identities: List[np.ndarray]
    entropy_metrics: Dict[str, float]  # sample_entropy, coherence_variance
    resistance_map: Dict[Tuple[int, int], float]  # pairwise conflicts
    oscillation_pattern: Optional[Dict] = None
    explanation: str = ""
    timestamp: float = field(default_factory=time.time)
    iterations_attempted: int = 0


@dataclass
class IdentityBranch:
    """Tracks a single refinement trajectory during synthesis.

    Each branch represents one possible path through the synthesis space,
    maintaining history of refinement steps and coherence evolution.

    Attributes:
        branch_id: Unique identifier for this trajectory
        proto_identity: Current proto-identity state (H×W×4)
        trajectory: History of proto-identity states during refinement
        coherence_history: Coherence scores at each refinement step
        state: Branch status ('active', 'converged', 'cycling', 'chaotic')
    """
    branch_id: str
    proto_identity: np.ndarray  # Current state (H×W×4)
    trajectory: List[np.ndarray] = field(default_factory=list)  # Refinement history
    coherence_history: List[float] = field(default_factory=list)
    state: str = 'active'  # 'active', 'converged', 'cycling', 'chaotic'


@dataclass
class SynthesisResult:
    """Comprehensive result of synthesis operation.

    Encapsulates all possible outcomes of iterative refinement:
    - Identity: Single converged proto-identity (coherent answer)
    - Paradox: Multiple stable proto-identities (legitimate ambiguity)
    - Evolution: Cycling or chaotic behavior (unstable system)

    Attributes:
        proto_identities: List of resulting proto-identities (H×W×4)
        state: Synthesis outcome ('identity', 'paradox', 'evolution_cycling', 'evolution_chaotic')
        coherence_scores: Coherence values for each proto-identity
        entropy_metrics: Optional entropy measurements for unstable systems
        resistance_map: Optional pairwise conflicts for paradox/evolution
        explanation: Human-readable description of synthesis outcome
        branches: All refinement branches explored during synthesis
        unstable_stub: Diagnostic data if synthesis failed to converge
    """
    proto_identities: List[np.ndarray]  # Can be multiple for paradox
    state: str  # 'identity', 'paradox', 'evolution_cycling', 'evolution_chaotic'
    coherence_scores: List[float]
    entropy_metrics: Optional[Dict[str, float]] = None
    resistance_map: Optional[Dict[Tuple[int, int], float]] = None
    explanation: str = ""
    branches: List[IdentityBranch] = field(default_factory=list)
    unstable_stub: Optional[UnstableSystemStub] = None
