"""
Node lifecycle tracking for layer management.

Tracks metadata for nodes across their lifetime in the three-layer architecture.
"""

from typing import Tuple
from dataclasses import dataclass, field


@dataclass
class NodeLifecycle:
    """Metadata tracking for a single node across its lifecycle."""

    # Position
    x: int
    y: int
    z: int

    # Current state
    current_layer: str
    resonance: float = 0.0

    # Access tracking
    access_count: int = 0
    last_access_time: int = 0  # Operation count
    creation_time: int = 0

    # Transition history
    transitions: list[Tuple[int, str, str]] = field(default_factory=list)

    @property
    def age(self) -> int:
        """Age in operations since creation."""
        return 0  # Will be computed relative to current operation count

    def record_access(self, operation_count: int) -> None:
        """Update access metadata."""
        self.access_count += 1
        self.last_access_time = operation_count

    def record_transition(
        self,
        operation_count: int,
        from_layer: str,
        to_layer: str
    ) -> None:
        """Record layer transition."""
        self.transitions.append((operation_count, from_layer, to_layer))
        self.current_layer = to_layer
