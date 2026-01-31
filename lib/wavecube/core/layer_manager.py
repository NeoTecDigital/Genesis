"""
Automatic memory layer management for Genesis's three-layer architecture.

Manages promotion, demotion, and eviction of proto-identities across:
- proto_unity: Stable consolidated knowledge (high resonance)
- experiential: Working memory (moderate resonance)
- io: Transient buffer (low commitment)
"""

from typing import Dict, Tuple, Optional, Any, TYPE_CHECKING
import logging

from .node_lifecycle import NodeLifecycle
from .layer_config import (
    DEFAULT_PROMOTION_CONFIG,
    DEFAULT_DEMOTION_CONFIG,
    DEFAULT_EVICTION_CONFIG
)

if TYPE_CHECKING:
    from .layered_matrix import LayeredWaveCube

logger = logging.getLogger(__name__)


class LayerManager:
    """
    Automatic layer management for three-tier memory architecture.

    Manages lifecycle of proto-identities across:
    - proto_unity: Stable consolidated knowledge (high resonance)
    - experiential: Working memory (moderate resonance)
    - io: Transient buffer (low commitment)

    Lifecycle flow:
        io → experiential → proto_unity (promotion via high resonance)
        proto_unity → experiential → evicted (demotion via low access)
    """

    def __init__(
        self,
        layered_wavecube: 'LayeredWaveCube',
        promotion_config: Optional[Dict[str, Any]] = None,
        demotion_config: Optional[Dict[str, Any]] = None,
        eviction_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize layer manager with configurable policies.

        Args:
            layered_wavecube: The layered memory structure to manage
            promotion_config: Promotion thresholds and settings
            demotion_config: Demotion thresholds and settings
            eviction_config: Eviction thresholds and settings
        """
        self.wavecube = layered_wavecube
        self.promotion_config = promotion_config or DEFAULT_PROMOTION_CONFIG.copy()
        self.demotion_config = demotion_config or DEFAULT_DEMOTION_CONFIG.copy()
        self.eviction_config = eviction_config or DEFAULT_EVICTION_CONFIG.copy()

        # Metadata tracking
        self._metadata: Dict[Tuple[int, int, int], NodeLifecycle] = {}
        self._operation_count = 0

        # Statistics
        self._stats = {
            'promoted': 0,
            'demoted': 0,
            'evicted': 0,
            'total_optimizations': 0
        }

    def on_store(
        self,
        x: int,
        y: int,
        z: int,
        layer: str,
        resonance: float
    ) -> None:
        """
        Called when a node is stored. Track metadata.

        Args:
            x: X coordinate
            y: Y coordinate
            z: Z coordinate
            layer: Target layer ('io', 'experiential', 'proto_unity')
            resonance: Initial resonance value
        """
        coords = (x, y, z)

        if coords not in self._metadata:
            self._metadata[coords] = NodeLifecycle(
                x=x, y=y, z=z,
                current_layer=layer,
                resonance=resonance,
                creation_time=self._operation_count
            )
        else:
            metadata = self._metadata[coords]
            metadata.resonance = resonance
            metadata.current_layer = layer

        self._operation_count += 1

    def on_access(self, x: int, y: int, z: int, layer: str) -> None:
        """
        Called when a node is accessed. Update access count.

        Args:
            x: X coordinate
            y: Y coordinate
            z: Z coordinate
            layer: Layer being accessed
        """
        coords = (x, y, z)

        if coords in self._metadata:
            self._metadata[coords].record_access(self._operation_count)
        else:
            self._metadata[coords] = NodeLifecycle(
                x=x, y=y, z=z,
                current_layer=layer,
                access_count=1,
                last_access_time=self._operation_count,
                creation_time=self._operation_count
            )

        self._operation_count += 1
        self._maybe_optimize()

    def optimize_layers(self) -> Dict[str, int]:
        """
        Run full optimization cycle: promotion, demotion, eviction.

        Returns:
            Statistics: {'promoted': count, 'demoted': count, 'evicted': count}
        """
        results = {'promoted': 0, 'demoted': 0, 'evicted': 0}
        coords_to_process = list(self._metadata.keys())

        for coords in coords_to_process:
            x, y, z = coords
            metadata = self._metadata[coords]

            # Check promotion
            if self.check_promotion(x, y, z, metadata.current_layer):
                target = self._get_promotion_target(metadata.current_layer)
                if self.promote_node(x, y, z, metadata.current_layer, target):
                    results['promoted'] += 1
                    continue

            # Check demotion
            if self.check_demotion(x, y, z, metadata.current_layer):
                target = self._get_demotion_target(metadata.current_layer)
                if self.demote_node(x, y, z, metadata.current_layer, target):
                    results['demoted'] += 1
                    continue

            # Check eviction
            if self.check_eviction(x, y, z):
                if self.evict_node(x, y, z, metadata.current_layer):
                    results['evicted'] += 1

        # Update stats
        self._stats['promoted'] += results['promoted']
        self._stats['demoted'] += results['demoted']
        self._stats['evicted'] += results['evicted']
        self._stats['total_optimizations'] += 1

        return results

    def check_promotion(
        self,
        x: int,
        y: int,
        z: int,
        from_layer: str
    ) -> bool:
        """Check if node should be promoted to higher layer."""
        if from_layer == 'proto_unity':
            return False

        coords = (x, y, z)
        if coords not in self._metadata:
            return False

        metadata = self._metadata[coords]
        return (
            metadata.resonance >= self.promotion_config['resonance_threshold'] and
            metadata.access_count >= self.promotion_config['access_threshold']
        )

    def check_demotion(
        self,
        x: int,
        y: int,
        z: int,
        from_layer: str
    ) -> bool:
        """Check if node should be demoted to lower layer."""
        if from_layer == 'io':
            return False

        coords = (x, y, z)
        if coords not in self._metadata:
            return False

        metadata = self._metadata[coords]
        age = self._operation_count - metadata.creation_time

        return (
            metadata.access_count < self.demotion_config['access_threshold'] and
            age > self.demotion_config['time_threshold']
        )

    def check_eviction(self, x: int, y: int, z: int) -> bool:
        """Check if node should be evicted from memory."""
        coords = (x, y, z)
        if coords not in self._metadata:
            return False

        metadata = self._metadata[coords]
        if metadata.current_layer != 'io':
            return False

        total_memory_mb = self._get_total_memory_mb()
        return (
            total_memory_mb > self.eviction_config['memory_threshold_mb'] and
            metadata.resonance < self.eviction_config['resonance_threshold']
        )

    def promote_node(
        self,
        x: int,
        y: int,
        z: int,
        from_layer: str,
        to_layer: str
    ) -> bool:
        """Promote node from one layer to another."""
        if not self._validate_promotion(from_layer, to_layer):
            return False

        node = self.wavecube.get_node(x, y, z, layer=from_layer)
        if node is None:
            return False

        self.wavecube.set_node(x, y, z, node, layer=to_layer)
        self.wavecube.remove_node(x, y, z, layer=from_layer)

        coords = (x, y, z)
        if coords in self._metadata:
            self._metadata[coords].record_transition(
                self._operation_count,
                from_layer,
                to_layer
            )

        return True

    def demote_node(
        self,
        x: int,
        y: int,
        z: int,
        from_layer: str,
        to_layer: str
    ) -> bool:
        """Demote node from one layer to another."""
        if not self._validate_demotion(from_layer, to_layer):
            return False

        node = self.wavecube.get_node(x, y, z, layer=from_layer)
        if node is None:
            return False

        self.wavecube.set_node(x, y, z, node, layer=to_layer)
        self.wavecube.remove_node(x, y, z, layer=from_layer)

        coords = (x, y, z)
        if coords in self._metadata:
            self._metadata[coords].record_transition(
                self._operation_count,
                from_layer,
                to_layer
            )

        return True

    def evict_node(self, x: int, y: int, z: int, layer: str) -> bool:
        """Remove node from memory entirely."""
        coords = (x, y, z)
        success = self.wavecube.remove_node(x, y, z, layer=layer)

        if success and coords in self._metadata:
            del self._metadata[coords]

        return success

    def get_statistics(self) -> Dict[str, Any]:
        """Get layer management statistics."""
        return {
            'operation_count': self._operation_count,
            'tracked_nodes': len(self._metadata),
            'total_promoted': self._stats['promoted'],
            'total_demoted': self._stats['demoted'],
            'total_evicted': self._stats['evicted'],
            'optimization_cycles': self._stats['total_optimizations'],
            'memory_mb': self._get_total_memory_mb()
        }

    def _maybe_optimize(self) -> None:
        """Check if optimization should run based on intervals."""
        promo_interval = self.promotion_config['check_interval']
        demo_interval = self.demotion_config['check_interval']

        if (self._operation_count % promo_interval == 0 or
            self._operation_count % demo_interval == 0):
            self.optimize_layers()

    def _get_promotion_target(self, from_layer: str) -> str:
        """Get target layer for promotion."""
        if from_layer == 'io':
            return 'experiential'
        elif from_layer == 'experiential':
            return 'proto_unity'
        else:
            raise ValueError(f"Cannot promote from {from_layer}")

    def _get_demotion_target(self, from_layer: str) -> str:
        """Get target layer for demotion."""
        if from_layer == 'proto_unity':
            return 'experiential'
        elif from_layer == 'experiential':
            return 'io'
        else:
            raise ValueError(f"Cannot demote from {from_layer}")

    def _validate_promotion(self, from_layer: str, to_layer: str) -> bool:
        """Validate promotion path."""
        valid_paths = {
            ('io', 'experiential'),
            ('experiential', 'proto_unity')
        }
        return (from_layer, to_layer) in valid_paths

    def _validate_demotion(self, from_layer: str, to_layer: str) -> bool:
        """Validate demotion path."""
        valid_paths = {
            ('proto_unity', 'experiential'),
            ('experiential', 'io')
        }
        return (from_layer, to_layer) in valid_paths

    def _get_total_memory_mb(self) -> float:
        """Get total memory usage across all layers."""
        memory_usage = self.wavecube.get_memory_usage()
        return memory_usage['total']['total_mb']
