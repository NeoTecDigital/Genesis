"""
Core abstractions for Oracle field-theoretic memory system.

Separates interface from implementation per modular design principles.
"""

from .field_encoder import FieldEncoder
from .field_evolver import FieldEvolver
from .proto_identity import ProtoIdentity

__all__ = ['FieldEncoder', 'FieldEvolver', 'ProtoIdentity']