"""
Memory pool for storing proto-identities and cluster centers.
"""

import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class MemoryEntry:
    """Single memory entry."""
    proto_identity: np.ndarray  # Feature vector or proto-unity state
    coherence_score: float  # Quality/coherence metric
    metadata: Dict[str, Any]  # Associated metadata
    timestamp: float = 0.0  # When stored


class MemoryPool:
    """
    Storage pool for proto-identities and learned cluster centers.

    Used for associative recall and semantic memory.
    """

    def __init__(self, max_size: Optional[int] = None):
        """
        Initialize memory pool.

        Args:
            max_size: Maximum number of memories to store (None = unlimited)
        """
        self.memories: List[MemoryEntry] = []
        self.max_size = max_size

    def add(
        self,
        proto_identity: np.ndarray,
        coherence_score: float,
        metadata: Dict[str, Any],
        timestamp: Optional[float] = None
    ) -> int:
        """
        Add a new memory to the pool.

        Args:
            proto_identity: Feature vector or proto state
            coherence_score: Quality metric (0-1)
            metadata: Associated metadata
            timestamp: Optional timestamp

        Returns:
            Index of added memory
        """
        import time

        if timestamp is None:
            timestamp = time.time()

        entry = MemoryEntry(
            proto_identity=proto_identity.copy(),
            coherence_score=coherence_score,
            metadata=metadata.copy(),
            timestamp=timestamp
        )

        self.memories.append(entry)

        # Enforce max size if set
        if self.max_size is not None and len(self.memories) > self.max_size:
            # Remove lowest coherence entry
            min_idx = min(range(len(self.memories)), key=lambda i: self.memories[i].coherence_score)
            self.memories.pop(min_idx)

        return len(self.memories) - 1

    def query(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        coherence_threshold: float = 0.0
    ) -> List[tuple]:
        """
        Query memory pool for similar entries.

        Args:
            query_vector: Query feature vector
            top_k: Number of results to return
            coherence_threshold: Minimum coherence to consider

        Returns:
            List of (index, similarity, entry) tuples
        """
        if len(self.memories) == 0:
            return []

        # Compute similarities (cosine similarity)
        similarities = []
        query_norm = query_vector / (np.linalg.norm(query_vector) + 1e-8)

        for idx, entry in enumerate(self.memories):
            if entry.coherence_score < coherence_threshold:
                continue

            stored_norm = entry.proto_identity / (np.linalg.norm(entry.proto_identity) + 1e-8)
            similarity = np.dot(query_norm.flatten(), stored_norm.flatten())

            # Map to [0, 1]
            similarity = (similarity + 1) / 2

            similarities.append((idx, float(similarity), entry))

        # Sort by similarity and return top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    def clear(self):
        """Clear all memories."""
        self.memories.clear()

    def __len__(self):
        return len(self.memories)

    def __getitem__(self, idx: int) -> MemoryEntry:
        return self.memories[idx]
