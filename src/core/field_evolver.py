"""Abstract base class for field evolution."""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
from .proto_identity import ProtoIdentity


class FieldEvolver(ABC):
    """
    Abstract interface for proto-identity field evolution.

    Implementations evolve an initial proto-identity field to a
    stable state according to specific dynamics.

    Concrete implementations:
    - UFTEvolver: Mass gap + chiral dynamics
    - DiracEvolver: Full 4-spinor quantum evolution (future)
    - AdaptiveEvolver: Self-tuning evolution (future)
    """

    @abstractmethod
    def evolve(self, proto: ProtoIdentity) -> ProtoIdentity:
        """
        Evolve proto-identity field to stable state.

        Args:
            proto: Initial proto-identity from encoder

        Returns:
            ProtoIdentity: Evolved proto-identity with updated field and metadata

        Raises:
            ValueError: If proto is invalid or evolution fails
        """
        pass

    def evolve_batch(self, protos: List[ProtoIdentity]) -> List[ProtoIdentity]:
        """
        Evolve multiple proto-identities.

        Default implementation evolves sequentially.
        Subclasses can override for parallel processing.

        Args:
            protos: List of proto-identities to evolve

        Returns:
            List[ProtoIdentity]: Evolved proto-identities
        """
        return [self.evolve(proto) for proto in protos]

    @property
    @abstractmethod
    def evolver_type(self) -> str:
        """Return evolver type identifier."""
        pass

    @property
    def config(self) -> Dict[str, Any]:
        """
        Return evolver configuration.

        Subclasses should override to provide specific config.

        Returns:
            Dict[str, Any]: Configuration parameters
        """
        return {
            'evolver_type': self.evolver_type,
        }

    def validate_proto(self, proto: ProtoIdentity) -> bool:
        """
        Validate that proto-identity is suitable for evolution.

        Args:
            proto: Proto-identity to validate

        Returns:
            bool: True if valid, False otherwise
        """
        # Check field dimensions
        if proto.field.shape[:2] != (512, 512):
            return False

        # Check for required metadata
        if 'order_parameter' not in proto.metadata:
            return False

        return True