"""
Trilinear interpolation for 3D wavetable matrices.

Implements trilinear interpolation to sample wavetables at fractional
coordinates by blending the 8 surrounding grid nodes.
"""

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.matrix import WavetableMatrix


def trilinear_interpolate(
    matrix: 'WavetableMatrix',
    x: float,
    y: float,
    z: float
) -> np.ndarray:
    """
    Sample wavetable at fractional coordinates using trilinear interpolation.

    Trilinear interpolation blends the wavetables at the 8 surrounding
    integer grid points based on the fractional parts of the coordinates.

    Algorithm:
        1. Find 8 surrounding integer grid points
        2. Extract wavetables at those points
        3. Interpolate along X axis (4 results)
        4. Interpolate along Y axis (2 results)
        5. Interpolate along Z axis (1 final result)

    Args:
        matrix: WavetableMatrix to sample from
        x: X coordinate (float in [0, matrix.width])
        y: Y coordinate (float in [0, matrix.height])
        z: Z coordinate (float in [0, matrix.depth])

    Returns:
        Interpolated wavetable with shape matching matrix resolution

    Raises:
        ValueError: If coordinates are out of bounds
        RuntimeError: If no nodes exist in the neighborhood
    """
    # Validate bounds
    if not (0 <= x <= matrix.width):
        raise ValueError(f"X coordinate {x} out of bounds [0, {matrix.width}]")
    if not (0 <= y <= matrix.height):
        raise ValueError(f"Y coordinate {y} out of bounds [0, {matrix.height}]")
    if not (0 <= z <= matrix.depth):
        raise ValueError(f"Z coordinate {z} out of bounds [0, {matrix.depth}]")

    # Find surrounding integer grid points
    x0 = int(np.floor(x))
    x1 = min(x0 + 1, matrix.width - 1)
    y0 = int(np.floor(y))
    y1 = min(y0 + 1, matrix.height - 1)
    z0 = int(np.floor(z))
    z1 = min(z0 + 1, matrix.depth - 1)

    # Compute fractional parts
    xd = x - x0
    yd = y - y0
    zd = z - z0

    # Get wavetables at 8 corners
    # Handle sparse matrices - return zeros if node doesn't exist
    def get_wavetable(xi: int, yi: int, zi: int) -> np.ndarray:
        """Get wavetable or return zeros if doesn't exist."""
        wt = matrix.get_node(xi, yi, zi)
        if wt is None:
            # Return zeros with default resolution
            return np.zeros((*matrix.resolution, matrix.channels), dtype=matrix.dtype)
        return wt

    c000 = get_wavetable(x0, y0, z0)
    c001 = get_wavetable(x0, y0, z1)
    c010 = get_wavetable(x0, y1, z0)
    c011 = get_wavetable(x0, y1, z1)
    c100 = get_wavetable(x1, y0, z0)
    c101 = get_wavetable(x1, y0, z1)
    c110 = get_wavetable(x1, y1, z0)
    c111 = get_wavetable(x1, y1, z1)

    # Check if all corners are empty (sparse matrix with no data in region)
    if not any(matrix.has_node(xi, yi, zi)
               for xi in [x0, x1]
               for yi in [y0, y1]
               for zi in [z0, z1]):
        raise RuntimeError(
            f"No nodes found in neighborhood of ({x}, {y}, {z}). "
            "Cannot interpolate in empty region."
        )

    # Interpolate along X axis (4 interpolations)
    c00 = c000 * (1 - xd) + c100 * xd
    c01 = c001 * (1 - xd) + c101 * xd
    c10 = c010 * (1 - xd) + c110 * xd
    c11 = c011 * (1 - xd) + c111 * xd

    # Interpolate along Y axis (2 interpolations)
    c0 = c00 * (1 - yd) + c10 * yd
    c1 = c01 * (1 - yd) + c11 * yd

    # Interpolate along Z axis (1 final interpolation)
    c = c0 * (1 - zd) + c1 * zd

    return c.astype(matrix.dtype)


def trilinear_interpolate_batch(
    matrix: 'WavetableMatrix',
    coords: np.ndarray
) -> np.ndarray:
    """
    Batch trilinear interpolation for multiple coordinates.

    Args:
        matrix: WavetableMatrix to sample from
        coords: (N, 3) array of [x, y, z] coordinates

    Returns:
        (N, H, W, C) array of interpolated wavetables

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
        results[i] = trilinear_interpolate(matrix, x, y, z)

    return results


def trilinear_weights(x: float, y: float, z: float) -> tuple[
    tuple[int, int, int, int, int, int],  # Indices (x0, x1, y0, y1, z0, z1)
    np.ndarray  # Weights (8,) for 8 corners
]:
    """
    Compute trilinear interpolation indices and weights.

    Useful for understanding interpolation or custom implementations.

    Args:
        x, y, z: Fractional coordinates

    Returns:
        Tuple of (indices, weights):
            - indices: (x0, x1, y0, y1, z0, z1)
            - weights: (8,) array of weights for corners in order:
                [c000, c001, c010, c011, c100, c101, c110, c111]
    """
    # Integer parts
    x0 = int(np.floor(x))
    x1 = x0 + 1
    y0 = int(np.floor(y))
    y1 = y0 + 1
    z0 = int(np.floor(z))
    z1 = z0 + 1

    # Fractional parts
    xd = x - x0
    yd = y - y0
    zd = z - z0

    # Compute weights for 8 corners
    weights = np.array([
        (1 - xd) * (1 - yd) * (1 - zd),  # c000
        (1 - xd) * (1 - yd) * zd,        # c001
        (1 - xd) * yd * (1 - zd),        # c010
        (1 - xd) * yd * zd,              # c011
        xd * (1 - yd) * (1 - zd),        # c100
        xd * (1 - yd) * zd,              # c101
        xd * yd * (1 - zd),              # c110
        xd * yd * zd,                    # c111
    ])

    indices = (x0, x1, y0, y1, z0, z1)

    return indices, weights


# Validation function for testing
def validate_trilinear(
    c000: np.ndarray, c001: np.ndarray, c010: np.ndarray, c011: np.ndarray,
    c100: np.ndarray, c101: np.ndarray, c110: np.ndarray, c111: np.ndarray,
    x: float, y: float, z: float
) -> np.ndarray:
    """
    Reference implementation for testing trilinear interpolation.

    Direct implementation without matrix wrapper for validation.

    Args:
        c000-c111: 8 corner wavetables
        x, y, z: Fractional coordinates (fractional parts only, [0, 1])

    Returns:
        Interpolated wavetable
    """
    # Interpolate along X
    c00 = c000 * (1 - x) + c100 * x
    c01 = c001 * (1 - x) + c101 * x
    c10 = c010 * (1 - x) + c110 * x
    c11 = c011 * (1 - x) + c111 * x

    # Interpolate along Y
    c0 = c00 * (1 - y) + c10 * y
    c1 = c01 * (1 - y) + c11 * y

    # Interpolate along Z
    c = c0 * (1 - z) + c1 * z

    return c
