"""
Multi-Band Frequency Clustering for VoxelCloud.

Groups proto-identities into 3 frequency bands:
- LOW: Concept-level abstractions (slow oscillations)
- MID: Word-level patterns (medium frequency)
- HIGH: Character-level details (fast oscillations)
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from enum import IntEnum
from collections import defaultdict


class FrequencyBand(IntEnum):
    """Frequency band classifications."""
    LOW = 0   # Concept-level (< 10 Hz)
    MID = 1   # Word-level (10-100 Hz)
    HIGH = 2  # Character-level (> 100 Hz)


class FrequencyBandClustering:
    """
    Cluster proto-identities by frequency spectrum into 3 bands.

    Uses spectral magnitude distribution to classify proto-identities
    into LOW (concept), MID (word), or HIGH (character) frequency bands.
    """

    def __init__(self, num_bands: int = 3):
        """
        Initialize frequency band clustering.

        Args:
            num_bands: Number of bands (default: 3)
        """
        if num_bands != 3:
            raise ValueError("Only 3 bands supported (LOW, MID, HIGH)")

        self.num_bands = num_bands

        # Frequency ranges (in Hz) for each band
        self.band_ranges = {
            FrequencyBand.LOW: (0.0, 10.0),
            FrequencyBand.MID: (10.0, 100.0),
            FrequencyBand.HIGH: (100.0, float('inf'))
        }

        # Band statistics
        self.band_stats: Dict[FrequencyBand, Dict] = {
            FrequencyBand.LOW: {'count': 0, 'mean_freq': 0.0},
            FrequencyBand.MID: {'count': 0, 'mean_freq': 0.0},
            FrequencyBand.HIGH: {'count': 0, 'mean_freq': 0.0}
        }

    def assign_band(self, frequency_spectrum: np.ndarray) -> FrequencyBand:
        """
        Classify proto-identity into frequency band.

        Args:
            frequency_spectrum: Frequency spectrum (H, W, 4)

        Returns:
            FrequencyBand enum (LOW, MID, or HIGH)
        """
        # Compute dominant frequency from spectrum
        dominant_freq = self._compute_dominant_frequency(frequency_spectrum)

        # Classify into band based on dominant frequency
        if dominant_freq < self.band_ranges[FrequencyBand.LOW][1]:
            return FrequencyBand.LOW
        elif dominant_freq < self.band_ranges[FrequencyBand.MID][1]:
            return FrequencyBand.MID
        else:
            return FrequencyBand.HIGH

    def _compute_dominant_frequency(self, freq_spectrum: np.ndarray) -> float:
        """
        Compute dominant frequency from spectrum.

        Args:
            freq_spectrum: Frequency spectrum (H, W, 4)

        Returns:
            Dominant frequency in Hz
        """
        # Compute magnitude
        magnitude = np.sqrt(freq_spectrum[..., 0]**2 + freq_spectrum[..., 1]**2)

        # Compute distance from center for each pixel
        h, w = freq_spectrum.shape[:2]
        center_y, center_x = h // 2, w // 2

        # Create distance map from center (DC component)
        y_coords = np.arange(h).reshape(-1, 1) - center_y
        x_coords = np.arange(w).reshape(1, -1) - center_x
        distance_from_center = np.sqrt(y_coords**2 + x_coords**2)

        # Weighted average distance (weighted by magnitude)
        total_mag = magnitude.sum()
        if total_mag > 1e-8:
            weighted_dist = (magnitude * distance_from_center).sum() / total_mag
        else:
            weighted_dist = 0.0

        # Map distance to Hz
        # Near center (small distance) = LOW frequency
        # Far from center (large distance) = HIGH frequency
        max_dist = np.sqrt(center_y**2 + center_x**2)

        # Normalize to [0, 1] then scale to Hz
        # Low: 0-10 Hz, Mid: 10-100 Hz, High: 100+ Hz
        normalized_dist = weighted_dist / (max_dist + 1e-8)
        dominant_hz = normalized_dist * 500.0

        return dominant_hz

    def cluster_by_band(self, voxel_cloud, band: FrequencyBand) -> List:
        """
        Group voxels by frequency band.

        Args:
            voxel_cloud: VoxelCloud instance
            band: FrequencyBand to filter by

        Returns:
            List of ProtoIdentityEntry in specified band
        """
        clustered = []

        for entry in voxel_cloud.entries:
            # Compute band for this entry
            entry_band = self.assign_band(entry.frequency)

            if entry_band == band:
                clustered.append(entry)

        # Update stats
        self.band_stats[band]['count'] = len(clustered)
        if clustered:
            freqs = [self._compute_dominant_frequency(e.frequency)
                    for e in clustered]
            self.band_stats[band]['mean_freq'] = np.mean(freqs)

        return clustered

    def compute_band_coherence(self, protos: List, band: FrequencyBand) -> float:
        """
        Measure within-band coherence.

        Coherence measures how similar proto-identities within a band are.
        High coherence = band contains related patterns.

        Args:
            protos: List of ProtoIdentityEntry
            band: FrequencyBand to measure coherence for

        Returns:
            Coherence score [0, 1]
        """
        if len(protos) < 2:
            return 1.0  # Perfect coherence if only one proto

        # Filter to band
        band_protos = [p for p in protos
                      if self.assign_band(p.frequency) == band]

        if len(band_protos) < 2:
            return 1.0

        # Compute pairwise frequency similarities
        similarities = []
        for i, proto_i in enumerate(band_protos):
            freq_i = self._compute_dominant_frequency(proto_i.frequency)

            for proto_j in band_protos[i+1:]:
                freq_j = self._compute_dominant_frequency(proto_j.frequency)

                # Frequency similarity (normalized)
                freq_diff = abs(freq_i - freq_j)
                band_range = (self.band_ranges[band][1] -
                             self.band_ranges[band][0])

                if band_range > 0:
                    similarity = 1.0 - min(freq_diff / band_range, 1.0)
                else:
                    similarity = 1.0 if freq_diff < 1.0 else 0.0

                similarities.append(similarity)

        if similarities:
            coherence = np.mean(similarities)
        else:
            coherence = 1.0

        return coherence

    def get_band_representatives(self, voxel_cloud,
                                 band: FrequencyBand, k: int = 5) -> List:
        """
        Get top-k representative patterns per band.

        Representatives are selected based on:
        1. Resonance strength (how often pattern appeared)
        2. Centrality within band (frequency close to band mean)

        Args:
            voxel_cloud: VoxelCloud instance
            band: FrequencyBand to get representatives from
            k: Number of representatives to return

        Returns:
            List of top-k ProtoIdentityEntry for band
        """
        # Get all protos in band
        band_protos = self.cluster_by_band(voxel_cloud, band)

        if not band_protos:
            return []

        # Compute scores for each proto
        scores = []
        band_mean_freq = self.band_stats[band]['mean_freq']

        for proto in band_protos:
            # Resonance score (normalized)
            resonance = proto.resonance_strength
            max_resonance = max(p.resonance_strength for p in band_protos)
            res_score = resonance / (max_resonance + 1e-8)

            # Centrality score (distance from band mean frequency)
            proto_freq = self._compute_dominant_frequency(proto.frequency)
            freq_diff = abs(proto_freq - band_mean_freq)
            band_range = (self.band_ranges[band][1] -
                         self.band_ranges[band][0])

            if band_range > 0:
                centrality = 1.0 - min(freq_diff / band_range, 1.0)
            else:
                centrality = 1.0

            # Combined score
            score = 0.6 * res_score + 0.4 * centrality
            scores.append((score, proto))

        # Sort by score and return top-k
        scores.sort(key=lambda x: x[0], reverse=True)
        representatives = [proto for _, proto in scores[:k]]

        return representatives

    def analyze_band_distribution(self, voxel_cloud) -> Dict[str, any]:
        """
        Analyze distribution of protos across bands.

        Args:
            voxel_cloud: VoxelCloud instance

        Returns:
            Dictionary with band statistics
        """
        band_counts = {
            'LOW': 0,
            'MID': 0,
            'HIGH': 0
        }

        band_resonances = defaultdict(list)
        band_frequencies = defaultdict(list)

        for entry in voxel_cloud.entries:
            band = self.assign_band(entry.frequency)
            band_name = band.name

            band_counts[band_name] += 1
            band_resonances[band_name].append(entry.resonance_strength)
            band_frequencies[band_name].append(
                self._compute_dominant_frequency(entry.frequency)
            )

        # Compute statistics
        stats = {}
        for band_name in ['LOW', 'MID', 'HIGH']:
            stats[band_name] = {
                'count': band_counts[band_name],
                'avg_resonance': (np.mean(band_resonances[band_name])
                                 if band_resonances[band_name] else 0.0),
                'avg_frequency': (np.mean(band_frequencies[band_name])
                                 if band_frequencies[band_name] else 0.0),
                'frequency_std': (np.std(band_frequencies[band_name])
                                 if band_frequencies[band_name] else 0.0)
            }

        return stats

    def __repr__(self) -> str:
        low = self.band_stats[FrequencyBand.LOW]
        mid = self.band_stats[FrequencyBand.MID]
        high = self.band_stats[FrequencyBand.HIGH]

        return (f"FrequencyBandClustering("
                f"LOW={low['count']} ({low['mean_freq']:.1f}Hz), "
                f"MID={mid['count']} ({mid['mean_freq']:.1f}Hz), "
                f"HIGH={high['count']} ({high['mean_freq']:.1f}Hz))")
