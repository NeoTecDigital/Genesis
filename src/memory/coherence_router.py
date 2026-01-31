"""
Coherence-based routing for memory storage decisions.

Routes proto-identities to appropriate memory layers based on
coherence measurements from Origin.compute_cohesion_state().
"""

import numpy as np
from typing import Dict
from dataclasses import dataclass
from src.memory.state_classifier import SignalState


@dataclass
class RoutingDecision:
    """Result of coherence-based routing decision."""
    destination: str  # 'core' | 'experiential' | 'rejected'
    coherence: float
    state: SignalState
    reason: str


class CoherenceRouter:
    """
    Routes proto-identities based on coherence measurements.

    Uses Origin.compute_cohesion_state() to measure coherence and
    determines appropriate memory layer:
    - IDENTITY (coherence ≥ 0.85) → core memory
    - EVOLUTION (0.3 < coherence < 0.85) → experiential memory
    - PARADOX (coherence ≤ 0.3) → rejected
    """

    def __init__(self,
                 width: int,
                 height: int,
                 identity_threshold: float = 0.85,
                 paradox_threshold: float = 0.3,
                 use_gpu: bool = False):
        """
        Initialize coherence router.

        Args:
            width: Proto-identity width
            height: Proto-identity height
            identity_threshold: Minimum coherence for IDENTITY state (default: 0.85)
            paradox_threshold: Maximum coherence for PARADOX state (default: 0.3)
            use_gpu: Enable GPU acceleration for Origin (default: False)
        """
        self.width = width
        self.height = height
        self.identity_threshold = identity_threshold
        self.paradox_threshold = paradox_threshold
        self.use_gpu = use_gpu

        # Lazy-load Origin (expensive import)
        self._origin = None

    def _get_origin(self):
        """Lazy-load Origin instance."""
        if self._origin is None:
            from src.origin import Origin
            self._origin = Origin(
                width=self.width,
                height=self.height,
                use_gpu=self.use_gpu
            )
        return self._origin

    def route_by_coherence(
        self,
        proto_identity: np.ndarray,
        frequency: np.ndarray,
        metadata: Dict,
        tau_params: Dict = None,
        epsilon_params: Dict = None
    ) -> RoutingDecision:
        """Measure coherence and determine routing destination."""
        tau_params = tau_params or {'tau_center': 0.5, 'tau_bandwidth': 0.3}
        epsilon_params = epsilon_params or {'epsilon_center': 0.5, 'epsilon_bandwidth': 0.3}

        coherence, state = self._measure_coherence(
            proto_identity, tau_params, epsilon_params
        )

        return self._create_decision(coherence, state)

    def _measure_coherence(
        self,
        proto_identity: np.ndarray,
        tau_params: Dict,
        epsilon_params: Dict
    ) -> tuple:
        """Measure coherence via Origin and map to state."""
        origin = self._get_origin()
        cohesion = origin.compute_cohesion_state(
            instance=proto_identity,
            tau_params=tau_params,
            epsilon_params=epsilon_params
        )
        coherence = cohesion['cohesion']
        state = self._map_state(cohesion['state'])
        return coherence, state

    def _create_decision(self, coherence: float, state: SignalState) -> RoutingDecision:
        """Create routing decision based on state."""
        destinations = {
            SignalState.IDENTITY: ('core', 'High coherence - stable pattern'),
            SignalState.EVOLUTION: ('experiential', 'Moderate coherence - learning pattern'),
            SignalState.PARADOX: ('rejected', 'Low coherence - conflicting pattern')
        }
        dest, reason = destinations.get(state, ('experiential', 'Unknown state'))
        return RoutingDecision(
            destination=dest,
            coherence=coherence,
            state=state,
            reason=reason
        )

    def _map_state(self, origin_state: str) -> SignalState:
        """
        Map Origin state string to SignalState enum.

        Args:
            origin_state: 'paradox' | 'evolution' | 'truth'

        Returns:
            SignalState enum value
        """
        state_map = {
            'paradox': SignalState.PARADOX,
            'evolution': SignalState.EVOLUTION,
            'truth': SignalState.IDENTITY
        }
        return state_map.get(origin_state, SignalState.EVOLUTION)

    def batch_route(
        self,
        proto_identities: list,
        frequencies: list,
        metadatas: list,
        tau_params: Dict = None,
        epsilon_params: Dict = None
    ) -> list:
        """
        Route multiple proto-identities efficiently.

        Args:
            proto_identities: List of proto-identities
            frequencies: List of frequency spectrums
            metadatas: List of metadata dicts
            tau_params: Tau parameters for coherence
            epsilon_params: Epsilon parameters

        Returns:
            List of RoutingDecision objects
        """
        decisions = []
        for proto, freq, meta in zip(proto_identities, frequencies, metadatas):
            decision = self.route_by_coherence(
                proto, freq, meta, tau_params, epsilon_params
            )
            decisions.append(decision)
        return decisions
