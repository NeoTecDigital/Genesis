"""
Stratified memory pool with provenance tracking.

Extension of basic MemoryPool with stratification by source/type.
"""

from enum import Enum
from typing import Dict, List, Optional
from dataclasses import dataclass

from .pool import MemoryPool, MemoryEntry


class MemoryProvenance(Enum):
    """Memory source/provenance tracking."""
    TRAINING = 'training'
    INTERACTION = 'interaction'
    SYSTEM = 'system'
    IMPORTED = 'imported'


class StratifiedMemoryPool:
    """
    Memory pool with stratification by provenance.

    Maintains separate pools for different memory sources while
    providing unified query interface.
    """

    def __init__(self, max_size_per_stratum: Optional[int] = None):
        """
        Initialize stratified pool.

        Args:
            max_size_per_stratum: Max memories per provenance type
        """
        self.strata: Dict[MemoryProvenance, MemoryPool] = {
            prov: MemoryPool(max_size=max_size_per_stratum)
            for prov in MemoryProvenance
        }

    def add(
        self,
        proto_identity,
        coherence_score: float,
        metadata: Dict,
        provenance: MemoryProvenance = MemoryProvenance.TRAINING
    ) -> int:
        """Add memory to appropriate stratum."""
        # Ensure provenance is in metadata
        metadata['provenance'] = provenance.value

        return self.strata[provenance].add(
            proto_identity=proto_identity,
            coherence_score=coherence_score,
            metadata=metadata
        )

    def query(
        self,
        query_vector,
        top_k: int = 5,
        coherence_threshold: float = 0.0,
        provenance_filter: Optional[List[MemoryProvenance]] = None
    ) -> List[tuple]:
        """
        Query across strata.

        Args:
            query_vector: Query vector
            top_k: Results to return
            coherence_threshold: Minimum coherence
            provenance_filter: Only query specific strata (None = all)

        Returns:
            List of (index, similarity, entry, provenance) tuples
        """
        results = []

        strata_to_query = provenance_filter if provenance_filter else list(MemoryProvenance)

        for prov in strata_to_query:
            stratum_results = self.strata[prov].query(
                query_vector,
                top_k=top_k * 2,  # Get more from each stratum
                coherence_threshold=coherence_threshold
            )

            # Add provenance to results
            for idx, sim, entry in stratum_results:
                results.append((idx, sim, entry, prov))

        # Sort all results and return top-k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def clear(self, provenance: Optional[MemoryProvenance] = None):
        """Clear memories from specific stratum or all."""
        if provenance is None:
            for pool in self.strata.values():
                pool.clear()
        else:
            self.strata[provenance].clear()

    def __len__(self):
        return sum(len(pool) for pool in self.strata.values())
