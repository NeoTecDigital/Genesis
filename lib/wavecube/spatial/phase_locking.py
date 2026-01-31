"""
Phase-locking utilities for cross-modal binding in quaternionic space.

Provides functions for phase shifting, finding phase-locked positions,
and binding multiple modalities together.
"""

from typing import List, Tuple, Optional, Dict, Any
import numpy as np

from .coordinates import QuaternionicCoord, Modality


def phase_shift(coord: QuaternionicCoord, degrees: float) -> QuaternionicCoord:
    """
    Shift the phase of a quaternionic coordinate.

    Args:
        coord: Input coordinate
        degrees: Phase shift in degrees

    Returns:
        New coordinate with shifted phase
    """
    return coord.shift_phase(degrees)


def find_phase_locked(
    base_coord: QuaternionicCoord,
    target_modality: Modality,
    search_radius: int = 5
) -> QuaternionicCoord:
    """
    Find optimal phase-locked position for target modality.

    Args:
        base_coord: Starting coordinate
        target_modality: Target modality to lock to
        search_radius: Search radius in spatial dimensions

    Returns:
        Phase-locked coordinate for target modality
    """
    target_phase = float(target_modality.value)

    # First try direct phase substitution
    direct = QuaternionicCoord(
        base_coord.x,
        base_coord.y,
        base_coord.z,
        target_phase
    )

    # Check if we need to search for better position
    phase_diff = base_coord.phase_distance(direct)

    # If phase difference is small, use direct mapping
    if phase_diff < 45.0:
        return direct

    # Otherwise search for optimal nearby position
    best_coord = direct
    best_score = float('inf')

    for dx in range(-search_radius, search_radius + 1):
        for dy in range(-search_radius, search_radius + 1):
            for dz in range(-search_radius, search_radius + 1):
                # Skip center
                if dx == 0 and dy == 0 and dz == 0:
                    continue

                new_x = base_coord.x + dx
                new_y = base_coord.y + dy
                new_z = base_coord.z + dz

                # Skip invalid coordinates
                if new_x < 0 or new_y < 0 or new_z < 0:
                    continue

                candidate = QuaternionicCoord(new_x, new_y, new_z, target_phase)

                # Score based on spatial distance
                score = abs(dx) + abs(dy) + abs(dz)

                if score < best_score:
                    best_score = score
                    best_coord = candidate

    return best_coord


def cross_modal_bind(
    protos_list: List[np.ndarray],
    modalities_list: List[Modality],
    base_position: Optional[Tuple[int, int, int]] = None
) -> List[QuaternionicCoord]:
    """
    Bind multiple proto-identities across modalities.

    Creates phase-locked coordinates for storing related proto-identities
    from different modalities in a coherent spatial arrangement.

    Args:
        protos_list: List of proto-identity arrays
        modalities_list: List of corresponding modalities
        base_position: Optional base spatial position (default: origin)

    Returns:
        List of bound quaternionic coordinates
    """
    if len(protos_list) != len(modalities_list):
        raise ValueError(
            f"Mismatch: {len(protos_list)} protos vs {len(modalities_list)} modalities"
        )

    if base_position is None:
        base_position = (0, 0, 0)

    bound_coords = []

    # Create base coordinate for first modality
    base_coord = QuaternionicCoord(
        base_position[0],
        base_position[1],
        base_position[2],
        float(modalities_list[0].value)
    )
    bound_coords.append(base_coord)

    # Create phase-locked coordinates for other modalities
    for i in range(1, len(modalities_list)):
        modality = modalities_list[i]

        # Find phase-locked position
        locked_coord = find_phase_locked(base_coord, modality, search_radius=3)
        bound_coords.append(locked_coord)

    return bound_coords


def create_phase_ring(
    center: QuaternionicCoord,
    radius: int,
    modalities: List[Modality]
) -> List[QuaternionicCoord]:
    """
    Create a ring of phase-locked coordinates around a center.

    Arranges modalities in a circular pattern in XY plane.

    Args:
        center: Center coordinate
        radius: Spatial radius
        modalities: List of modalities to arrange

    Returns:
        List of coordinates arranged in a ring
    """
    coords = []
    n = len(modalities)

    for i, modality in enumerate(modalities):
        # Compute angle for this modality
        angle = 2 * np.pi * i / n

        # Compute position on ring
        x = center.x + int(radius * np.cos(angle))
        y = center.y + int(radius * np.sin(angle))
        z = center.z

        # Create phase-locked coordinate
        coord = QuaternionicCoord(x, y, z, float(modality.value))
        coords.append(coord)

    return coords


def compute_phase_coherence(coords: List[QuaternionicCoord]) -> float:
    """
    Compute phase coherence metric for a set of coordinates.

    Measures how well-aligned the phase relationships are.

    Args:
        coords: List of quaternionic coordinates

    Returns:
        Coherence score (0=random, 1=perfectly locked)
    """
    if len(coords) < 2:
        return 1.0

    # Compute all pairwise phase distances
    phase_diffs = []

    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            phase_diff = coords[i].phase_distance(coords[j])
            phase_diffs.append(phase_diff)

    # Check if phase differences follow expected pattern
    # For locked modalities, we expect 90° separations
    expected_diffs = [90.0, 180.0, 270.0]

    coherence_scores = []
    for diff in phase_diffs:
        # Find closest expected difference
        min_deviation = min(
            abs(diff - expected) for expected in expected_diffs + [0.0]
        )
        # Normalize to 0-1 (0° deviation = 1.0, 90° deviation = 0.0)
        score = max(0.0, 1.0 - min_deviation / 90.0)
        coherence_scores.append(score)

    return np.mean(coherence_scores) if coherence_scores else 1.0


def find_phase_clusters(
    coords: List[QuaternionicCoord],
    phase_tolerance: float = 15.0
) -> Dict[Modality, List[QuaternionicCoord]]:
    """
    Group coordinates by their phase (modality).

    Args:
        coords: List of quaternionic coordinates
        phase_tolerance: Tolerance for phase matching in degrees

    Returns:
        Dict mapping modalities to lists of coordinates
    """
    clusters: Dict[Modality, List[QuaternionicCoord]] = {
        modality: [] for modality in Modality
    }

    for coord in coords:
        modality = coord.get_modality(tolerance=phase_tolerance)
        if modality is not None:
            clusters[modality].append(coord)

    # Remove empty clusters
    return {k: v for k, v in clusters.items() if v}


def create_phase_gradient(
    start: QuaternionicCoord,
    end: QuaternionicCoord,
    steps: int
) -> List[QuaternionicCoord]:
    """
    Create a gradient of coordinates with smoothly varying phase.

    Args:
        start: Starting coordinate
        end: Ending coordinate
        steps: Number of steps in gradient

    Returns:
        List of interpolated coordinates
    """
    if steps <= 1:
        return [start]

    coords = []
    for i in range(steps):
        t = i / (steps - 1)
        coord = start.interpolate(end, t)
        coords.append(coord)

    return coords


def optimize_phase_arrangement(
    coords: List[QuaternionicCoord],
    iterations: int = 10
) -> List[QuaternionicCoord]:
    """
    Optimize spatial arrangement to minimize phase conflicts.

    Uses iterative relaxation to find optimal positions.

    Args:
        coords: Initial coordinates
        iterations: Number of optimization iterations

    Returns:
        Optimized coordinates
    """
    optimized = [
        QuaternionicCoord(c.x, c.y, c.z, c.w) for c in coords
    ]

    for _ in range(iterations):
        # Compute forces between coordinates
        forces = [(0.0, 0.0, 0.0) for _ in optimized]

        for i in range(len(optimized)):
            for j in range(i + 1, len(optimized)):
                # Compute repulsion based on phase similarity
                phase_diff = optimized[i].phase_distance(optimized[j])

                if phase_diff < 45.0:  # Too similar, push apart
                    spatial_dist = optimized[i].spatial_distance(optimized[j])

                    if spatial_dist > 0.1:
                        # Compute normalized direction
                        dx = optimized[j].x - optimized[i].x
                        dy = optimized[j].y - optimized[i].y
                        dz = optimized[j].z - optimized[i].z

                        norm = np.sqrt(dx**2 + dy**2 + dz**2)
                        dx /= norm
                        dy /= norm
                        dz /= norm

                        # Apply force
                        force = (45.0 - phase_diff) / 45.0
                        forces[i] = (
                            forces[i][0] - dx * force,
                            forces[i][1] - dy * force,
                            forces[i][2] - dz * force
                        )
                        forces[j] = (
                            forces[j][0] + dx * force,
                            forces[j][1] + dy * force,
                            forces[j][2] + dz * force
                        )

        # Apply forces with damping
        damping = 0.5
        for i, (fx, fy, fz) in enumerate(forces):
            new_x = max(0, int(optimized[i].x + fx * damping))
            new_y = max(0, int(optimized[i].y + fy * damping))
            new_z = max(0, int(optimized[i].z + fz * damping))

            optimized[i] = QuaternionicCoord(
                new_x, new_y, new_z, optimized[i].w
            )

    return optimized