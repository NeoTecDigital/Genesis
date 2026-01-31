"""
Experiential Memory Manager - Manages experiential memory lifecycle.

Handles:
- Reset to baseline (proto-unity carrier)
- Consolidation to core (Identity-state patterns)
- Memory lifecycle management
"""

import numpy as np
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .voxel_cloud import VoxelCloud

from .state_classifier import SignalState


class ExperientialMemoryManager:
    """Manages experiential memory lifecycle (reset, consolidation).

    Experiential memory is short-term working memory that can be:
    1. Reset to baseline (proto-unity carrier) - clears all working thoughts
    2. Consolidated to core - moves stable patterns to long-term memory
    """

    def __init__(self, proto_unity_carrier: np.ndarray,
                 experiential_memory: 'VoxelCloud'):
        """Initialize experiential memory manager.

        Args:
            proto_unity_carrier: Baseline proto-unity from Origin (H, W, 4)
            experiential_memory: Experiential VoxelCloud to manage
        """
        self.baseline_carrier = proto_unity_carrier.copy()
        self.experiential = experiential_memory
        self.current_carrier = proto_unity_carrier.copy()

    def reset_to_baseline(self) -> None:
        """Reset experiential memory to proto-unity baseline.

        This clears all experiential entries and resets temporal buffer,
        returning the system to the stable proto-unity state from Origin.

        Use cases:
        - PARADOX state persists (conflicting thoughts)
        - Memory overflow
        - Session restart
        """
        # Clear all experiential entries
        self.experiential.entries.clear()

        # Clear spatial index
        self.experiential.spatial_index.clear()

        # Reset temporal buffer
        self.experiential.temporal_buffer.clear()

        # Reset carrier to baseline proto-unity
        self.current_carrier = self.baseline_carrier.copy()

    def consolidate_to_core(self, core_memory: 'VoxelCloud',
                           threshold: float = 0.8) -> int:
        """Move stable Identity-state patterns from experiential to core.

        Only consolidates entries that:
        1. Are in IDENTITY state (stable, converged)
        2. Have resonance_strength >= threshold (well-established patterns)

        Args:
            core_memory: Core VoxelCloud to consolidate into
            threshold: Minimum resonance strength for consolidation (default: 0.8)

        Returns:
            Number of patterns consolidated
        """
        consolidated = 0

        # Find Identity-state entries with high resonance
        for entry in self.experiential.entries:
            # Check state is IDENTITY (stable, converged)
            if entry.current_state != SignalState.IDENTITY:
                continue

            # Check resonance strength meets threshold
            if entry.resonance_strength < threshold:
                continue

            # Add to core memory
            core_memory.add(
                entry.proto_identity,
                entry.frequency,
                entry.metadata
            )
            consolidated += 1

        return consolidated

    def selective_reset(self, keep_states: Optional[set] = None) -> int:
        """Reset experiential memory but keep entries in specified states.

        Args:
            keep_states: Set of SignalState to preserve (default: {IDENTITY})

        Returns:
            Number of entries removed
        """
        if keep_states is None:
            keep_states = {SignalState.IDENTITY}

        # Count entries before
        before = len(self.experiential.entries)

        # Filter entries - keep only those in keep_states
        self.experiential.entries = [
            entry for entry in self.experiential.entries
            if entry.current_state in keep_states
        ]

        # Rebuild spatial index for remaining entries
        self.experiential.spatial_index.clear()
        for idx, entry in enumerate(self.experiential.entries):
            position = entry.position
            grid_pos = tuple((position / [
                self.experiential.width,
                self.experiential.height,
                self.experiential.depth
            ] * 10).astype(int))

            if grid_pos not in self.experiential.spatial_index:
                self.experiential.spatial_index[grid_pos] = []
            self.experiential.spatial_index[grid_pos].append(idx)

        # Count entries removed
        removed = before - len(self.experiential.entries)
        return removed

    def get_baseline_carrier(self) -> np.ndarray:
        """Get baseline proto-unity carrier.

        Returns:
            Copy of baseline carrier (H, W, 4)
        """
        return self.baseline_carrier.copy()

    def get_current_carrier(self) -> np.ndarray:
        """Get current carrier state.

        Returns:
            Copy of current carrier (H, W, 4)
        """
        return self.current_carrier.copy()

    def __repr__(self) -> str:
        """String representation."""
        return (f"ExperientialMemoryManager("
                f"experiential_entries={len(self.experiential)}, "
                f"carrier_shape={self.baseline_carrier.shape})")
