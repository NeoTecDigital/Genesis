"""Abstract base class for field encoders."""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from .proto_identity import ProtoIdentity


class FieldEncoder(ABC):
    """
    Abstract interface for text → proto-identity encoding.

    Implementations should encode text into a complex-valued field
    that represents the proto-identity of the input.

    Concrete implementations:
    - KuramotoEncoder: Phase oscillator synchronization
    - FFTEncoder: Frequency-based encoding (legacy)
    - UFTEncoder: Direct UFT field encoding (future)
    """

    @abstractmethod
    def encode(self, text: str) -> ProtoIdentity:
        """
        Encode text to proto-identity field.

        Args:
            text: Input text string to encode

        Returns:
            ProtoIdentity: Contains complex field and metadata

        Raises:
            ValueError: If text is empty or invalid
        """
        pass

    def encode_batch(self, texts: list[str]) -> list[ProtoIdentity]:
        """
        Encode multiple texts to proto-identities.

        Default implementation encodes sequentially.
        Subclasses can override for parallel processing.

        Args:
            texts: List of input text strings

        Returns:
            List[ProtoIdentity]: Encoded proto-identities

        Raises:
            ValueError: If any text is invalid
        """
        return [self.encode(text) for text in texts]

    @property
    @abstractmethod
    def encoder_type(self) -> str:
        """Return encoder type identifier."""
        pass

    @property
    def config(self) -> Dict[str, Any]:
        """
        Return encoder configuration.

        Subclasses should override to provide specific config.

        Returns:
            Dict[str, Any]: Configuration parameters
        """
        return {
            'encoder_type': self.encoder_type,
        }