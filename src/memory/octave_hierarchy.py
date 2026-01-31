"""
Octave-Based Multi-Scale Memory Hierarchy.

12 octave levels (spatial pyramid spanning letters to documents):
- Octave 0: Full resolution (letter/phoneme-level) - 512×512
- Octave 1: 2x downsampled (syllable-level) - 256×256
- Octave 2: 4x downsampled (morpheme-level) - 128×128
- Octave 3: 8x downsampled (word-level) - 64×64
- Octave 4: 16x downsampled (phrase-level) - 32×32
- Octave 5: 32x downsampled (clause-level) - 16×16
- Octave 6: 64x downsampled (sentence-level) - 8×8
- Octave 7: 128x downsampled (multi-sentence) - 4×4
- Octave 8: 256x downsampled (paragraph-level) - 2×2
- Octave 9: 512x downsampled (section-level) - 1×1
- Octave 10: 1024x downsampled (chapter-level) - 0.5×0.5
- Octave 11: 2048x downsampled (document-level) - 0.25×0.25

Enables multi-resolution semantic matching from letters to full documents.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class OctaveProtoIdentity:
    """Proto-identity with multi-octave quaternionic vectors."""
    proto_identity: np.ndarray  # (H, W, 4) full resolution
    quaternions: Dict[int, np.ndarray]  # octave → (4,) quaternion
    frequency: float
    modality: str
    octave_level: int = 0  # Default query level


class OctaveHierarchy:
    """Manage multi-octave proto-identity storage and retrieval."""

    def __init__(self, num_octaves: int = 12):
        self.num_octaves = num_octaves
        self.octave_storage: Dict[int, List[OctaveProtoIdentity]] = {
            i: [] for i in range(num_octaves)
        }

    def add_proto_identity(self, octave_proto: OctaveProtoIdentity):
        """Add proto-identity to all relevant octave levels."""
        # Store in all octaves
        for octave in range(self.num_octaves):
            self.octave_storage[octave].append(octave_proto)

    def query_at_octave(self, query_quaternion: np.ndarray, octave: int,
                        top_k: int = 10) -> List[Tuple[OctaveProtoIdentity, float]]:
        """Query at specific octave level using quaternionic distance."""
        results = []

        for proto in self.octave_storage[octave]:
            # Get quaternion at this octave level
            proto_quat = proto.quaternions.get(octave)
            if proto_quat is None:
                continue

            # Quaternionic distance (geodesic on unit sphere)
            dot_product = np.dot(query_quaternion, proto_quat)
            # Clamp to [-1, 1] to avoid numerical errors
            dot_product = np.clip(dot_product, -1.0, 1.0)
            distance = np.arccos(np.abs(dot_product))  # Absolute for quaternion double-cover

            results.append((proto, distance))

        # Sort by distance and return top_k
        results.sort(key=lambda x: x[1])
        return results[:top_k]

    def multi_octave_query(self, query_quaternions: Dict[int, np.ndarray],
                          top_k: int = 10,
                          octave_weights: Optional[Dict[int, float]] = None) -> List[Tuple[OctaveProtoIdentity, float]]:
        """Query across multiple octaves with weighted fusion.

        Args:
            query_quaternions: octave → query quaternion
            top_k: Number of results to return
            octave_weights: octave → weight (default: equal weights)

        Returns:
            List of (proto, fused_distance) sorted by distance
        """
        if octave_weights is None:
            # Equal weights by default
            octave_weights = {i: 1.0 / len(query_quaternions)
                            for i in query_quaternions}

        # Normalize weights
        total_weight = sum(octave_weights.values())
        octave_weights = {k: v / total_weight
                         for k, v in octave_weights.items()}

        # Collect all unique protos (using list to avoid hashable issues)
        seen_texts = set()
        all_protos = []
        for octave in query_quaternions.keys():
            octave_results = self.query_at_octave(
                query_quaternions[octave], octave, top_k=top_k * 2
            )
            for proto, _ in octave_results:
                # Use proto_identity hash as unique identifier (no text stored)
                proto_id = id(proto.proto_identity)
                if proto_id not in seen_texts:
                    seen_texts.add(proto_id)
                    all_protos.append(proto)

        # Compute weighted fusion distance for each proto
        results = []
        for proto in all_protos:
            fused_distance = 0.0

            for octave, query_quat in query_quaternions.items():
                proto_quat = proto.quaternions.get(octave)
                if proto_quat is None:
                    continue

                dot_product = np.dot(query_quat, proto_quat)
                dot_product = np.clip(dot_product, -1.0, 1.0)
                distance = np.arccos(np.abs(dot_product))

                weight = octave_weights.get(octave, 0.0)
                fused_distance += weight * distance

            results.append((proto, fused_distance))

        # Sort and return top_k
        results.sort(key=lambda x: x[1])
        return results[:top_k]

    def adaptive_octave_selection(self, query_text: str) -> int:
        """Select appropriate octave level based on query characteristics.

        Adapts octave selection to 12-level hierarchy:
        - Single letter → octave 0 (letter-level)
        - 1-5 words → octave 3 (word-level)
        - 6-15 words → octave 6 (sentence-level)
        - 16-50 words → octave 8 (paragraph-level)
        - 50+ words → octave 11 (document-level)
        """
        words = query_text.split()
        num_words = len(words)

        if num_words == 1 and len(query_text.strip()) == 1:
            return 0  # Single letter
        elif num_words <= 5:
            return 3  # Word-level
        elif num_words <= 15:
            return 6  # Sentence-level
        elif num_words <= 50:
            return 8  # Paragraph-level
        else:
            return 11  # Document-level

    def get_octave_statistics(self) -> Dict:
        """Get statistics about stored proto-identities per octave."""
        stats = {}
        for octave in range(self.num_octaves):
            stats[octave] = {
                'count': len(self.octave_storage[octave]),
                'has_quaternions': sum(
                    1 for p in self.octave_storage[octave]
                    if octave in p.quaternions
                )
            }
        return stats