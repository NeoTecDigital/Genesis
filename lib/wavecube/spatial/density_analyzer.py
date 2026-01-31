"""
Density analysis for adaptive resolution in WaveCube.

Analyzes proto-identity density in chunks to determine optimal wavetable
resolution. Supports density-based classification (low/medium/high) for
dynamic resolution adaptation.
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np


class DensityLevel:
    """Density classification levels."""
    LOW = 'low'
    MEDIUM = 'medium'
    HIGH = 'high'


# Density thresholds (protos per chunk volume)
DENSITY_THRESHOLDS = {
    DensityLevel.LOW: (0, 10),      # < 10 protos/chunk
    DensityLevel.MEDIUM: (10, 100),  # 10-100 protos/chunk
    DensityLevel.HIGH: (100, float('inf'))  # > 100 protos/chunk
}


# Resolution mapping for each density level
RESOLUTION_MAP = {
    DensityLevel.LOW: (16, 16, 4),      # Minimal resolution
    DensityLevel.MEDIUM: (512, 512, 4),  # Standard resolution
    DensityLevel.HIGH: (1024, 1024, 4)   # High resolution
}


# Alternative high-resolution for extreme density
ULTRA_HIGH_RESOLUTION = (3840, 3840, 48)


def compute_chunk_density(num_nodes: int, chunk_volume: int) -> float:
    """
    Compute density metric for a chunk.

    Args:
        num_nodes: Number of proto-identities in chunk
        chunk_volume: Volume of chunk (width × height × depth)

    Returns:
        Density value (protos per volume unit)
    """
    if chunk_volume <= 0:
        raise ValueError(f"Chunk volume must be positive: {chunk_volume}")

    return num_nodes / chunk_volume


def classify_density_level(density: float) -> str:
    """
    Classify density into low/medium/high categories.

    Args:
        density: Density value (protos per volume)

    Returns:
        Density level string ('low', 'medium', or 'high')
    """
    if density < DENSITY_THRESHOLDS[DensityLevel.LOW][1]:
        return DensityLevel.LOW
    elif density < DENSITY_THRESHOLDS[DensityLevel.MEDIUM][1]:
        return DensityLevel.MEDIUM
    else:
        return DensityLevel.HIGH


def get_target_resolution(density_level: str) -> Tuple[int, int, int]:
    """
    Get target wavetable resolution for density level.

    Args:
        density_level: Density classification ('low', 'medium', 'high')

    Returns:
        Target resolution tuple (height, width, channels)
    """
    if density_level not in RESOLUTION_MAP:
        raise ValueError(f"Unknown density level: {density_level}")

    return RESOLUTION_MAP[density_level]


def should_use_ultra_high(density: float, threshold: float = 500) -> bool:
    """
    Determine if ultra-high resolution should be used.

    Args:
        density: Density value (protos per volume)
        threshold: Threshold for ultra-high resolution

    Returns:
        True if ultra-high resolution recommended
    """
    return density >= threshold


class DensityAnalyzer:
    """
    Analyzes chunk density and recommends resolution adaptations.

    Tracks density statistics, identifies patterns, and provides
    recommendations for optimal resolution levels.
    """

    def __init__(
        self,
        low_threshold: float = 10,
        medium_threshold: float = 100,
        ultra_high_threshold: float = 500
    ):
        """
        Initialize density analyzer.

        Args:
            low_threshold: Threshold for low→medium transition
            medium_threshold: Threshold for medium→high transition
            ultra_high_threshold: Threshold for ultra-high resolution
        """
        self.low_threshold = low_threshold
        self.medium_threshold = medium_threshold
        self.ultra_high_threshold = ultra_high_threshold

        # Statistics
        self.stats = {
            'total_chunks_analyzed': 0,
            'low_density_chunks': 0,
            'medium_density_chunks': 0,
            'high_density_chunks': 0,
            'ultra_high_density_chunks': 0,
            'avg_density': 0.0,
            'max_density': 0.0,
            'min_density': float('inf')
        }

        # Track density history for adaptive thresholds
        self.density_history = []

    def analyze_chunk(
        self,
        num_nodes: int,
        chunk_volume: int
    ) -> Dict[str, Any]:
        """
        Analyze chunk density and recommend resolution.

        Args:
            num_nodes: Number of nodes in chunk
            chunk_volume: Chunk volume (width × height × depth)

        Returns:
            Analysis dict with density, level, and resolution
        """
        # Compute density
        density = compute_chunk_density(num_nodes, chunk_volume)

        # Classify
        if density < self.low_threshold:
            level = DensityLevel.LOW
        elif density < self.medium_threshold:
            level = DensityLevel.MEDIUM
        else:
            level = DensityLevel.HIGH

        # Check for ultra-high
        use_ultra = density >= self.ultra_high_threshold

        # Get target resolution
        if use_ultra:
            resolution = ULTRA_HIGH_RESOLUTION
        else:
            resolution = get_target_resolution(level)

        # Update statistics
        self._update_stats(density, level, use_ultra)

        return {
            'density': density,
            'level': level,
            'resolution': resolution,
            'ultra_high': use_ultra,
            'num_nodes': num_nodes,
            'chunk_volume': chunk_volume
        }

    def _update_stats(
        self,
        density: float,
        level: str,
        ultra_high: bool
    ) -> None:
        """
        Update internal statistics.

        Args:
            density: Density value
            level: Density level
            ultra_high: Whether ultra-high resolution was recommended
        """
        self.stats['total_chunks_analyzed'] += 1

        # Update level counts
        if level == DensityLevel.LOW:
            self.stats['low_density_chunks'] += 1
        elif level == DensityLevel.MEDIUM:
            self.stats['medium_density_chunks'] += 1
        elif level == DensityLevel.HIGH:
            self.stats['high_density_chunks'] += 1

        if ultra_high:
            self.stats['ultra_high_density_chunks'] += 1

        # Update density statistics
        self.density_history.append(density)
        self.stats['max_density'] = max(self.stats['max_density'], density)
        self.stats['min_density'] = min(self.stats['min_density'], density)

        # Update average
        total = self.stats['total_chunks_analyzed']
        self.stats['avg_density'] = (
            (self.stats['avg_density'] * (total - 1) + density) / total
        )

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get density statistics.

        Returns:
            Statistics dictionary
        """
        stats = self.stats.copy()

        # Add percentages
        total = stats['total_chunks_analyzed']
        if total > 0:
            stats['low_density_pct'] = (
                stats['low_density_chunks'] / total * 100
            )
            stats['medium_density_pct'] = (
                stats['medium_density_chunks'] / total * 100
            )
            stats['high_density_pct'] = (
                stats['high_density_chunks'] / total * 100
            )
            stats['ultra_high_density_pct'] = (
                stats['ultra_high_density_chunks'] / total * 100
            )

        return stats

    def reset_statistics(self) -> None:
        """Reset all statistics to initial values."""
        self.stats = {
            'total_chunks_analyzed': 0,
            'low_density_chunks': 0,
            'medium_density_chunks': 0,
            'high_density_chunks': 0,
            'ultra_high_density_chunks': 0,
            'avg_density': 0.0,
            'max_density': 0.0,
            'min_density': float('inf')
        }
        self.density_history.clear()

    def recommend_threshold_adjustments(self) -> Optional[Dict[str, float]]:
        """
        Recommend threshold adjustments based on density history.

        Returns:
            Recommended thresholds or None if insufficient data
        """
        if len(self.density_history) < 10:
            return None

        # Calculate percentiles
        densities = np.array(self.density_history)
        p25 = np.percentile(densities, 25)
        p75 = np.percentile(densities, 75)
        p95 = np.percentile(densities, 95)

        return {
            'low_threshold': p25,
            'medium_threshold': p75,
            'ultra_high_threshold': p95
        }

    def __repr__(self) -> str:
        """String representation."""
        stats = self.get_statistics()
        return (
            f"DensityAnalyzer("
            f"chunks={stats['total_chunks_analyzed']}, "
            f"avg_density={stats['avg_density']:.2f}, "
            f"low={stats['low_density_chunks']}, "
            f"medium={stats['medium_density_chunks']}, "
            f"high={stats['high_density_chunks']})"
        )
