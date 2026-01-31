"""
Quaternionic coordinate system for multi-modal phase-locked encoding.

Extends 3D spatial coordinates with a W dimension representing modality phase offset.
"""

from typing import Tuple, Optional, List
import numpy as np
from dataclasses import dataclass
from enum import IntEnum


class Modality(IntEnum):
    """Modality types with their phase offsets."""
    TEXT = 0      # 0° phase offset
    AUDIO = 90    # 90° phase offset
    IMAGE = 180   # 180° phase offset
    VIDEO = 270   # 270° phase offset


@dataclass
class QuaternionicCoord:
    """
    Quaternionic coordinate in 4D space.

    The W dimension represents modality phase offset, enabling
    cross-modal relationships through phase-locking.

    Attributes:
        x: Spatial X coordinate
        y: Spatial Y coordinate
        z: Spatial Z coordinate
        w: Phase offset (0-360 degrees)
    """
    x: int
    y: int
    z: int
    w: float  # Phase in degrees (0-360)

    def __post_init__(self):
        """Normalize phase to 0-360 range."""
        self.w = self.w % 360.0

    @classmethod
    def from_modality(
        cls,
        x: int, y: int, z: int,
        modality: Modality
    ) -> 'QuaternionicCoord':
        """
        Create coordinate from modality enum.

        Args:
            x, y, z: Spatial coordinates
            modality: Modality enum value

        Returns:
            QuaternionicCoord with appropriate phase
        """
        return cls(x, y, z, float(modality.value))

    @classmethod
    def from_tuple(cls, coords: Tuple[int, int, int, float]) -> 'QuaternionicCoord':
        """Create from tuple (x, y, z, w)."""
        return cls(*coords)

    def to_tuple(self) -> Tuple[int, int, int, float]:
        """Convert to tuple (x, y, z, w)."""
        return (self.x, self.y, self.z, self.w)

    def to_array(self) -> np.ndarray:
        """Convert to numpy array [x, y, z, w]."""
        return np.array([self.x, self.y, self.z, self.w], dtype=np.float32)

    def spatial_distance(self, other: 'QuaternionicCoord') -> float:
        """
        Compute Euclidean distance in spatial dimensions only.

        Args:
            other: Another quaternionic coordinate

        Returns:
            Euclidean distance in XYZ space
        """
        dx = self.x - other.x
        dy = self.y - other.y
        dz = self.z - other.z
        return np.sqrt(dx**2 + dy**2 + dz**2)

    def phase_distance(self, other: 'QuaternionicCoord') -> float:
        """
        Compute phase distance (shortest angular distance).

        Args:
            other: Another quaternionic coordinate

        Returns:
            Phase distance in degrees (0-180)
        """
        diff = abs(self.w - other.w)
        # Take shortest path around circle
        if diff > 180:
            diff = 360 - diff
        return diff

    def total_distance(
        self,
        other: 'QuaternionicCoord',
        phase_weight: float = 1.0 / 360.0
    ) -> float:
        """
        Compute combined spatial and phase distance.

        Args:
            other: Another quaternionic coordinate
            phase_weight: Weight for phase component (default 1/360)

        Returns:
            Combined distance metric
        """
        spatial = self.spatial_distance(other)
        phase = self.phase_distance(other)
        return spatial + phase * phase_weight

    def shift_phase(self, degrees: float) -> 'QuaternionicCoord':
        """
        Create new coordinate with shifted phase.

        Args:
            degrees: Phase shift in degrees

        Returns:
            New coordinate with shifted phase
        """
        new_phase = (self.w + degrees) % 360.0
        return QuaternionicCoord(self.x, self.y, self.z, new_phase)

    def interpolate(
        self,
        other: 'QuaternionicCoord',
        t: float
    ) -> 'QuaternionicCoord':
        """
        Linearly interpolate to another coordinate.

        Args:
            other: Target coordinate
            t: Interpolation factor (0=self, 1=other)

        Returns:
            Interpolated coordinate
        """
        # Spatial interpolation
        x = int(self.x + (other.x - self.x) * t)
        y = int(self.y + (other.y - self.y) * t)
        z = int(self.z + (other.z - self.z) * t)

        # Phase interpolation (handle wraparound)
        phase_diff = other.w - self.w
        if phase_diff > 180:
            phase_diff -= 360
        elif phase_diff < -180:
            phase_diff += 360

        w = (self.w + phase_diff * t) % 360.0

        return QuaternionicCoord(x, y, z, w)

    def get_modality(self, tolerance: float = 45.0) -> Optional[Modality]:
        """
        Determine modality from phase offset.

        Args:
            tolerance: Phase tolerance in degrees

        Returns:
            Closest modality or None if not within tolerance
        """
        for modality in Modality:
            phase_diff = abs(self.w - modality.value)
            if phase_diff > 180:
                phase_diff = 360 - phase_diff

            if phase_diff <= tolerance:
                return modality

        return None

    def __eq__(self, other: object) -> bool:
        """Equality comparison."""
        if not isinstance(other, QuaternionicCoord):
            return False
        return (
            self.x == other.x and
            self.y == other.y and
            self.z == other.z and
            abs(self.w - other.w) < 1e-6
        )

    def __hash__(self) -> int:
        """Hash for use in sets/dicts."""
        # Round phase to nearest degree for hashing
        return hash((self.x, self.y, self.z, int(self.w)))

    def __repr__(self) -> str:
        """String representation."""
        modality = self.get_modality()
        mod_str = f" ({modality.name})" if modality else ""
        return f"QuaternionicCoord(x={self.x}, y={self.y}, z={self.z}, w={self.w:.1f}°{mod_str})"


def create_phase_locked_set(
    base_coord: QuaternionicCoord,
    modalities: List[Modality]
) -> List[QuaternionicCoord]:
    """
    Create a set of phase-locked coordinates for different modalities.

    Args:
        base_coord: Base coordinate
        modalities: List of target modalities

    Returns:
        List of phase-locked coordinates
    """
    coords = []
    for modality in modalities:
        new_coord = QuaternionicCoord(
            base_coord.x,
            base_coord.y,
            base_coord.z,
            float(modality.value)
        )
        coords.append(new_coord)
    return coords


def find_nearest_phase_locked(
    coord: QuaternionicCoord,
    target_modality: Modality,
    search_radius: int = 5
) -> Optional[QuaternionicCoord]:
    """
    Find nearest phase-locked position for target modality.

    Searches in a cube around the given coordinate for an optimal
    position that maintains phase-locking with the target modality.

    Args:
        coord: Starting coordinate
        target_modality: Target modality to lock to
        search_radius: Search radius in spatial dimensions

    Returns:
        Optimal phase-locked coordinate or None
    """
    target_phase = float(target_modality.value)

    # Check if current position works
    if abs(coord.phase_distance(QuaternionicCoord(0, 0, 0, target_phase))) < 1.0:
        return QuaternionicCoord(coord.x, coord.y, coord.z, target_phase)

    # Search nearby positions
    best_coord = None
    best_score = float('inf')

    for dx in range(-search_radius, search_radius + 1):
        for dy in range(-search_radius, search_radius + 1):
            for dz in range(-search_radius, search_radius + 1):
                new_x = coord.x + dx
                new_y = coord.y + dy
                new_z = coord.z + dz

                # Skip negative coordinates
                if new_x < 0 or new_y < 0 or new_z < 0:
                    continue

                # Create candidate coordinate
                candidate = QuaternionicCoord(new_x, new_y, new_z, target_phase)

                # Score based on distance from original
                score = abs(dx) + abs(dy) + abs(dz)

                if score < best_score:
                    best_score = score
                    best_coord = candidate

    return best_coord


def compute_phase_matrix(
    coords: List[QuaternionicCoord]
) -> np.ndarray:
    """
    Compute phase relationship matrix for a set of coordinates.

    Args:
        coords: List of quaternionic coordinates

    Returns:
        NxN matrix of phase distances
    """
    n = len(coords)
    matrix = np.zeros((n, n), dtype=np.float32)

    for i in range(n):
        for j in range(n):
            if i != j:
                matrix[i, j] = coords[i].phase_distance(coords[j])

    return matrix