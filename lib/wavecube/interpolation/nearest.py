"""
Nearest neighbor interpolation for fast wavetable sampling.

Provides fast lookup without interpolation overhead - useful for
real-time applications or when smooth interpolation isn't needed.
"""

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.matrix import WavetableMatrix


def nearest_neighbor(
    matrix: 'WavetableMatrix',
    x: float,
    y: float,
    z: float
) -> np.ndarray:
    """
    Sample wavetable using nearest neighbor (no interpolation).

    Simply rounds coordinates to nearest integer and returns that node.
    Much faster than interpolation but produces discontinuous results.

    Args:
        matrix: WavetableMatrix to sample from
        x, y, z: Coordinates (will be rounded to nearest integer)

    Returns:
        Wavetable at nearest grid point

    Raises:
        ValueError: If rounded coordinates are out of bounds
        RuntimeError: If node doesn't exist at rounded coordinates
    """
    # Round to nearest integer
    xi = int(round(x))
    yi = int(round(y))
    zi = int(round(z))

    # Clamp to bounds
    xi = max(0, min(xi, matrix.width - 1))
    yi = max(0, min(yi, matrix.height - 1))
    zi = max(0, min(zi, matrix.depth - 1))

    # Get node
    wavetable = matrix.get_node(xi, yi, zi)

    if wavetable is None:
        raise RuntimeError(
            f"No node at nearest position ({xi}, {yi}, {zi}). "
            "Matrix is sparse and node doesn't exist."
        )

    return wavetable


def nearest_neighbor_batch(
    matrix: 'WavetableMatrix',
    coords: np.ndarray
) -> np.ndarray:
    """
    Batch nearest neighbor sampling.

    Args:
        matrix: WavetableMatrix to sample from
        coords: (N, 3) array of [x, y, z] coordinates

    Returns:
        (N, H, W, C) array of nearest wavetables

    Raises:
        ValueError: If coords shape is invalid
    """
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"coords must be (N, 3), got {coords.shape}")

    n = len(coords)
    h, w = matrix.resolution
    c = matrix.channels

    results = np.zeros((n, h, w, c), dtype=matrix.dtype)

    for i in range(n):
        x, y, z = coords[i]
        results[i] = nearest_neighbor(matrix, x, y, z)

    return results


def nearest_neighbor_fill(
    matrix: 'WavetableMatrix',
    x: float,
    y: float,
    z: float,
    fill_value: float = 0.0
) -> np.ndarray:
    """
    Nearest neighbor with fill value for missing nodes.

    Returns fill value instead of raising error when node doesn't exist.
    Useful for sparse matrices with many empty regions.

    Args:
        matrix: WavetableMatrix to sample from
        x, y, z: Coordinates
        fill_value: Value to return if node doesn't exist

    Returns:
        Wavetable at nearest position or filled array
    """
    # Round to nearest integer
    xi = int(round(x))
    yi = int(round(y))
    zi = int(round(z))

    # Clamp to bounds
    xi = max(0, min(xi, matrix.width - 1))
    yi = max(0, min(yi, matrix.height - 1))
    zi = max(0, min(zi, matrix.depth - 1))

    # Get node
    wavetable = matrix.get_node(xi, yi, zi)

    if wavetable is None:
        # Return filled array
        h, w = matrix.resolution
        c = matrix.channels
        return np.full((h, w, c), fill_value, dtype=matrix.dtype)

    return wavetable


def round_coordinates(x: float, y: float, z: float) -> tuple[int, int, int]:
    """
    Round coordinates to nearest integers.

    Utility function for nearest neighbor lookup.

    Args:
        x, y, z: Fractional coordinates

    Returns:
        Tuple of rounded (x, y, z) integers
    """
    return (int(round(x)), int(round(y)), int(round(z)))


def floor_coordinates(x: float, y: float, z: float) -> tuple[int, int, int]:
    """
    Floor coordinates to integers.

    Alternative to rounding - always picks lower corner.

    Args:
        x, y, z: Fractional coordinates

    Returns:
        Tuple of floored (x, y, z) integers
    """
    return (int(np.floor(x)), int(np.floor(y)), int(np.floor(z)))


def ceil_coordinates(x: float, y: float, z: float) -> tuple[int, int, int]:
    """
    Ceil coordinates to integers.

    Alternative to rounding - always picks upper corner.

    Args:
        x, y, z: Fractional coordinates

    Returns:
        Tuple of ceiled (x, y, z) integers
    """
    return (int(np.ceil(x)), int(np.ceil(y)), int(np.ceil(z)))
