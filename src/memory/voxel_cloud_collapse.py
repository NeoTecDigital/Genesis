"""Gravitational collapse operations for VoxelCloud.

This module contains the frequency-based similarity matching and merging logic
that implements gravitational collapse via constructive interference.
"""

import numpy as np
from typing import List, Dict

from .voxel_helpers import compute_cosine_similarity, check_frequency_match


def find_similar_by_frequency(voxel_cloud, fundamental: float,
                              harmonics: np.ndarray) -> List:
    """Find existing proto-identities with similar frequency signatures.

    Args:
        voxel_cloud: VoxelCloud instance
        fundamental: Fundamental frequency
        harmonics: Harmonic coefficients array

    Returns:
        List of matching ProtoIdentityEntry instances
    """
    if not voxel_cloud.collapse_config['enable']:
        return []  # Skip collapse if disabled

    return find_by_frequency_internal(
        voxel_cloud,
        fundamental,
        harmonics,
        octave_tolerance=voxel_cloud.collapse_config['octave_tolerance'],
        harmonic_tolerance=voxel_cloud.collapse_config['harmonic_tolerance'],
        max_results=3  # Get top 3 candidates
    )


def find_by_frequency_internal(voxel_cloud, query_freq: float,
                               query_harmonics: np.ndarray,
                               octave_tolerance: int = 1,
                               harmonic_tolerance: float = 0.3,
                               max_results: int = 10) -> List:
    """
    Find proto-identities by frequency signature matching using frequency_index.
    O(k * octaves) where k = entries per frequency bin, not O(n).

    Args:
        voxel_cloud: VoxelCloud instance
        query_freq: Query fundamental frequency
        query_harmonics: Query harmonic coefficients (length 10)
        octave_tolerance: How many octaves to search (±1 = search 0.5x to 2x)
        harmonic_tolerance: Max harmonic difference threshold
        max_results: Maximum results to return

    Returns:
        List of matching entries, sorted by similarity
    """
    matches = []

    # Search across octaves using frequency_index
    for octave_shift in range(-octave_tolerance, octave_tolerance + 1):
        target_freq = query_freq * (2 ** octave_shift)
        freq_bin = voxel_cloud._get_frequency_bin(target_freq)

        # Only check entries at THIS frequency bin (not spatial voxel!)
        if freq_bin in voxel_cloud.frequency_index:
            for idx in voxel_cloud.frequency_index[freq_bin]:
                entry = voxel_cloud.entries[idx]
                diff = check_frequency_match(entry, target_freq,
                                            query_harmonics, harmonic_tolerance)
                if diff is not None:
                    matches.append((diff, entry))

    # Sort by similarity and return top matches
    matches.sort(key=lambda x: x[0])
    return [entry for _, entry in matches[:max_results]]


def merge_proto_identity(existing, new_proto: np.ndarray,
                        new_freq: np.ndarray, new_metadata: Dict,
                        voxel_cloud) -> None:
    """
    Merge new proto-identity into existing one using weighted average.
    This implements gravitational collapse via constructive interference.

    Args:
        existing: Existing ProtoIdentityEntry
        new_proto: New proto-identity to merge
        new_freq: New frequency spectrum
        new_metadata: Metadata for new proto
        voxel_cloud: VoxelCloud instance for context
    """
    from src.memory.octave_frequency import extract_fundamental, extract_harmonics

    k = existing.resonance_strength

    # Store old frequency bin for index update
    old_freq_bin = voxel_cloud._get_frequency_bin(existing.fundamental_freq)

    # Weighted average of proto-identities
    existing.proto_identity = (k * existing.proto_identity + new_proto) / (k + 1)

    # Weighted average of frequencies
    existing.frequency = (k * existing.frequency + new_freq) / (k + 1)

    # Update position based on new frequency
    existing.position = voxel_cloud._frequency_to_position(existing.frequency)

    # Regenerate MIP levels for merged proto
    existing.mip_levels = voxel_cloud._generate_mip_levels(existing.proto_identity)

    # Update frequency signature
    existing.fundamental_freq = extract_fundamental(existing.frequency)
    existing.harmonic_signature = extract_harmonics(existing.frequency,
                                                    existing.fundamental_freq)

    # Update frequency_index if the frequency bin changed
    new_freq_bin = voxel_cloud._get_frequency_bin(existing.fundamental_freq)
    if new_freq_bin != old_freq_bin:
        # Find the entry index
        entry_idx = voxel_cloud.entries.index(existing)

        # Remove from old bin
        if old_freq_bin in voxel_cloud.frequency_index:
            if entry_idx in voxel_cloud.frequency_index[old_freq_bin]:
                voxel_cloud.frequency_index[old_freq_bin].remove(entry_idx)
            if not voxel_cloud.frequency_index[old_freq_bin]:
                del voxel_cloud.frequency_index[old_freq_bin]

        # Add to new bin
        if new_freq_bin not in voxel_cloud.frequency_index:
            voxel_cloud.frequency_index[new_freq_bin] = []
        if entry_idx not in voxel_cloud.frequency_index[new_freq_bin]:
            voxel_cloud.frequency_index[new_freq_bin].append(entry_idx)

    # Increment resonance strength
    existing.resonance_strength = k + 1

    # Update metadata with resonance strength and track merged texts
    existing.metadata['resonance_strength'] = existing.resonance_strength
    existing.metadata['merge_count'] = existing.metadata.get('merge_count', 1) + 1

    # NOTE: Text tracking removed - proto-identities contain encoded text via FFT
    # No need to track 'merged_texts' as text is encoded in the proto itself


def check_similarity_for_collapse(proto_identity: np.ndarray,
                                  candidate,
                                  cosine_threshold: float) -> bool:
    """Check if proto-identity is similar enough to merge via collapse.

    Args:
        proto_identity: New proto-identity to check
        candidate: Existing ProtoIdentityEntry candidate
        cosine_threshold: Minimum cosine similarity for merging

    Returns:
        True if should merge, False otherwise
    """
    similarity = compute_cosine_similarity(proto_identity, candidate.proto_identity)
    return similarity > cosine_threshold
