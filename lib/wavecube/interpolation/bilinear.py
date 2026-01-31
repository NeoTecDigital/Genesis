"""
Bilinear interpolation for 2D wavetable slices.

Useful for extracting 2D slices through the matrix and interpolating
within those slices.
"""

import numpy as np
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from ..core.matrix import WavetableMatrix


def bilinear_interpolate(
    wavetable: np.ndarray,
    u: float,
    v: float
) -> np.ndarray:
    """
    Bilinear interpolation within a 2D wavetable.

    Interpolates at fractional coordinates (u, v) within a single wavetable
    by blending the 4 surrounding pixels.

    Args:
        wavetable: 2D wavetable array (H, W, C)
        u: Row coordinate (float in [0, H])
        v: Column coordinate (float in [0, W])

    Returns:
        Interpolated values (C,) at position (u, v)

    Raises:
        ValueError: If coordinates are out of bounds
    """
    if wavetable.ndim != 3:
        raise ValueError(f"wavetable must be 3D (H, W, C), got {wavetable.ndim}D")

    h, w, c = wavetable.shape

    # Validate bounds
    if not (0 <= u <= h - 1):
        raise ValueError(f"u coordinate {u} out of bounds [0, {h-1}]")
    if not (0 <= v <= w - 1):
        raise ValueError(f"v coordinate {v} out of bounds [0, {w-1}]")

    # Find surrounding integer points
    u0 = int(np.floor(u))
    u1 = min(u0 + 1, h - 1)
    v0 = int(np.floor(v))
    v1 = min(v0 + 1, w - 1)

    # Fractional parts
    ud = u - u0
    vd = v - v0

    # Get 4 corners
    c00 = wavetable[u0, v0]
    c01 = wavetable[u0, v1]
    c10 = wavetable[u1, v0]
    c11 = wavetable[u1, v1]

    # Interpolate along v (columns)
    c0 = c00 * (1 - vd) + c01 * vd
    c1 = c10 * (1 - vd) + c11 * vd

    # Interpolate along u (rows)
    c = c0 * (1 - ud) + c1 * ud

    return c


def extract_slice(
    matrix: 'WavetableMatrix',
    axis: Literal['x', 'y', 'z'],
    index: float,
    interpolate: bool = True
) -> np.ndarray:
    """
    Extract a 2D slice through the matrix along specified axis.

    Args:
        matrix: WavetableMatrix to slice
        axis: Axis to slice along ('x', 'y', or 'z')
        index: Position along axis (integer or fractional if interpolate=True)
        interpolate: If True, interpolate between nodes; if False, use nearest

    Returns:
        2D slice (depends on axis):
            - axis='x': (height, depth, H, W, C)
            - axis='y': (width, depth, H, W, C)
            - axis='z': (width, height, H, W, C)

    Raises:
        ValueError: If axis is invalid or index out of bounds
    """
    if axis not in ['x', 'y', 'z']:
        raise ValueError(f"axis must be 'x', 'y', or 'z', got {axis}")

    # Get grid dimensions
    if axis == 'x':
        if not (0 <= index < matrix.width):
            raise ValueError(f"x index {index} out of bounds [0, {matrix.width})")
        dim1, dim2 = matrix.height, matrix.depth
    elif axis == 'y':
        if not (0 <= index < matrix.height):
            raise ValueError(f"y index {index} out of bounds [0, {matrix.height})")
        dim1, dim2 = matrix.width, matrix.depth
    else:  # z
        if not (0 <= index < matrix.depth):
            raise ValueError(f"z index {index} out of bounds [0, {matrix.depth})")
        dim1, dim2 = matrix.width, matrix.height

    h, w = matrix.resolution
    c = matrix.channels

    # Create output slice
    slice_data = np.zeros((dim1, dim2, h, w, c), dtype=matrix.dtype)

    # Fill slice by sampling matrix
    for i in range(dim1):
        for j in range(dim2):
            # Determine 3D coordinates based on axis
            if axis == 'x':
                x, y, z = index, float(i), float(j)
            elif axis == 'y':
                x, y, z = float(i), index, float(j)
            else:  # z
                x, y, z = float(i), float(j), index

            # Sample from matrix (uses interpolation if fractional)
            if interpolate or (index != int(index)):
                try:
                    slice_data[i, j] = matrix.sample(x, y, z)
                except RuntimeError:
                    # Empty region - leave as zeros
                    pass
            else:
                # Nearest neighbor
                xi, yi, zi = int(round(x)), int(round(y)), int(round(z))
                wt = matrix.get_node(xi, yi, zi)
                if wt is not None:
                    slice_data[i, j] = wt

    return slice_data


def sample_slice_2d(
    slice_data: np.ndarray,
    u: float,
    v: float
) -> np.ndarray:
    """
    Sample from a 2D slice at fractional coordinates using bilinear interpolation.

    Args:
        slice_data: 2D slice array (dim1, dim2, H, W, C)
        u: First dimension coordinate (float in [0, dim1])
        v: Second dimension coordinate (float in [0, dim2])

    Returns:
        Interpolated wavetable (H, W, C)
    """
    if slice_data.ndim != 5:
        raise ValueError(f"slice_data must be 5D (dim1, dim2, H, W, C), got {slice_data.ndim}D")

    dim1, dim2, h, w, c = slice_data.shape

    # Validate bounds
    if not (0 <= u <= dim1 - 1):
        raise ValueError(f"u coordinate {u} out of bounds [0, {dim1-1}]")
    if not (0 <= v <= dim2 - 1):
        raise ValueError(f"v coordinate {v} out of bounds [0, {dim2-1}]")

    # Find surrounding integer points
    u0 = int(np.floor(u))
    u1 = min(u0 + 1, dim1 - 1)
    v0 = int(np.floor(v))
    v1 = min(v0 + 1, dim2 - 1)

    # Fractional parts
    ud = u - u0
    vd = v - v0

    # Get 4 corner wavetables
    w00 = slice_data[u0, v0]
    w01 = slice_data[u0, v1]
    w10 = slice_data[u1, v0]
    w11 = slice_data[u1, v1]

    # Interpolate along v
    w0 = w00 * (1 - vd) + w01 * vd
    w1 = w10 * (1 - vd) + w11 * vd

    # Interpolate along u
    w = w0 * (1 - ud) + w1 * ud

    return w


def bilinear_weights(u: float, v: float) -> tuple[
    tuple[int, int, int, int],  # Indices (u0, u1, v0, v1)
    np.ndarray  # Weights (4,) for 4 corners
]:
    """
    Compute bilinear interpolation indices and weights.

    Args:
        u, v: Fractional coordinates

    Returns:
        Tuple of (indices, weights):
            - indices: (u0, u1, v0, v1)
            - weights: (4,) array of weights for corners in order:
                [w00, w01, w10, w11]
    """
    # Integer parts
    u0 = int(np.floor(u))
    u1 = u0 + 1
    v0 = int(np.floor(v))
    v1 = v0 + 1

    # Fractional parts
    ud = u - u0
    vd = v - v0

    # Compute weights for 4 corners
    weights = np.array([
        (1 - ud) * (1 - vd),  # w00
        (1 - ud) * vd,        # w01
        ud * (1 - vd),        # w10
        ud * vd,              # w11
    ])

    indices = (u0, u1, v0, v1)

    return indices, weights
