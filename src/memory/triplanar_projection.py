"""Triplanar projection from frequency spectrum to 3D WaveCube coordinates.

This module implements the core transformation that maps frequency spectra
to 3D volumetric positions (x,y,z,w) in the WaveCube space.

Architecture:
    - Input: Frequency spectrum (512×512×2) [magnitude, phase] from FFT/STFT
    - Triplanar sampling: Extract XYZ from three orthogonal plane projections
    - Modality encoding: W dimension encodes input modality via phase
    - Output: (x, y, z, w) coordinates in 128×128×128 WaveCube grid

Triplanar Mapping:
    - XY plane projection → X coordinate (horizontal frequency distribution)
    - XZ plane projection → Y coordinate (vertical frequency distribution)
    - YZ plane projection → Z coordinate (depth frequency distribution)
    - Modality phase → W coordinate (text=0°, audio=90°, image=180°, video=270°)

Spatial Tolerance:
    - Exact matching: spatial_tolerance=1.0 means A=A=A and A≠B
    - Same input → same (x,y,z,w) coordinates
    - Different input → different coordinates

Multi-Octave Support:
    - Character level: octave +4 (finest granularity)
    - Word level: octave 0
    - Short phrase: octave -2
    - Long phrase: octave -4
    - Octave affects frequency band selection for projection

NO hashing, NO neural networks - pure frequency analysis.
"""

import numpy as np
from typing import Tuple, Literal
from dataclasses import dataclass

# Modality phase encoding (quaternionic W dimension)
MODALITY_PHASES = {
    'text': 0.0,      # 0°
    'audio': 90.0,    # 90°
    'image': 180.0,   # 180°
    'video': 270.0    # 270°
}

ModalityType = Literal['text', 'audio', 'image', 'video']


@dataclass
class WaveCubeCoordinates:
    """3D WaveCube coordinates with modality encoding.

    Attributes:
        x: X coordinate [0, 127] in 128×128×128 grid
        y: Y coordinate [0, 127]
        z: Z coordinate [0, 127]
        w: W phase [0°, 360°] encoding modality
        modality: Original modality type
        octave: Octave level used for extraction
    """
    x: int
    y: int
    z: int
    w: float
    modality: ModalityType
    octave: int

    def as_tuple(self) -> Tuple[int, int, int, float]:
        """Return as (x, y, z, w) tuple."""
        return (self.x, self.y, self.z, self.w)

    def as_spatial(self) -> Tuple[int, int, int]:
        """Return spatial coordinates only (x, y, z)."""
        return (self.x, self.y, self.z)


def extract_triplanar_coordinates(
    freq_spectrum: np.ndarray,
    modality: ModalityType = 'text',
    octave: int = 0,
    grid_size: int = 128
) -> WaveCubeCoordinates:
    """Extract 3D WaveCube coordinates from frequency spectrum via triplanar projection.

    This is the core transformation that maps frequency analysis to spatial positions.

    Algorithm:
        1. Extract magnitude from frequency spectrum
        2. Find dominant frequency peaks in three orthogonal planes:
           - XY plane (horizontal × vertical) → X coordinate
           - XZ plane (horizontal × depth) → Y coordinate
           - YZ plane (vertical × depth) → Z coordinate
        3. Normalize coordinates to WaveCube grid [0, grid_size)
        4. Add modality phase to W dimension

    Args:
        freq_spectrum: Frequency spectrum (H, W, 2) [magnitude, phase]
                      or (H, W, 4) [x, y, z, w] from prior processing
        modality: Input modality ('text', 'audio', 'image', 'video')
        octave: Octave level for frequency band selection
        grid_size: WaveCube grid dimension (default: 128)

    Returns:
        WaveCubeCoordinates with (x, y, z, w) position

    Example:
        >>> freq = extract_frequency_spectrum(text_input)
        >>> coords = extract_triplanar_coordinates(freq, modality='text', octave=4)
        >>> print(f"Character position: ({coords.x}, {coords.y}, {coords.z})")
    """
    # Validate input
    if freq_spectrum.ndim != 3:
        raise ValueError(f"Frequency spectrum must be 3D (H, W, C), got {freq_spectrum.ndim}D")

    h, w, c = freq_spectrum.shape

    # Extract magnitude channel
    if c == 2:
        # Standard frequency spectrum [magnitude, phase]
        magnitude = freq_spectrum[:, :, 0]
    elif c >= 4:
        # XYZW quaternion format - extract Z channel for magnitude
        magnitude = freq_spectrum[:, :, 2]
    else:
        raise ValueError(f"Frequency spectrum must have 2 or 4 channels, got {c}")

    # Apply octave-specific frequency band selection
    # Higher octaves (characters) focus on high frequencies
    # Lower octaves (phrases) focus on low frequencies
    magnitude = _apply_octave_band(magnitude, octave)

    # === TRIPLANAR PROJECTION ===

    # XY Plane Projection → X coordinate
    # Sum along depth (collapse Z) to get XY plane view
    xy_plane = np.sum(magnitude, axis=0) if magnitude.ndim > 2 else magnitude
    x_coord = _extract_dominant_position(xy_plane, axis=1, grid_size=grid_size)

    # XZ Plane Projection → Y coordinate
    # Sum along vertical (collapse Y) to get XZ plane view
    xz_plane = np.sum(magnitude, axis=1) if magnitude.ndim > 2 else magnitude.T
    y_coord = _extract_dominant_position(xz_plane, axis=1, grid_size=grid_size)

    # YZ Plane Projection → Z coordinate
    # Sum along horizontal (collapse X) to get YZ plane view
    yz_plane = np.sum(magnitude, axis=2) if magnitude.ndim > 2 else magnitude
    z_coord = _extract_dominant_position(yz_plane, axis=0, grid_size=grid_size)

    # W Dimension: Modality phase encoding
    w_phase = MODALITY_PHASES[modality]

    return WaveCubeCoordinates(
        x=x_coord,
        y=y_coord,
        z=z_coord,
        w=w_phase,
        modality=modality,
        octave=octave
    )


def _apply_octave_band(magnitude: np.ndarray, octave: int) -> np.ndarray:
    """Apply octave-specific frequency band filtering.

    Higher octaves (characters) emphasize high frequencies.
    Lower octaves (phrases) emphasize low frequencies.

    Args:
        magnitude: Frequency magnitude (H, W)
        octave: Octave level (+4 to -4)

    Returns:
        Band-filtered magnitude
    """
    h, w = magnitude.shape
    center_y, center_x = h // 2, w // 2

    # Create radial frequency grid
    y_coords = np.arange(h).reshape(-1, 1) - center_y
    x_coords = np.arange(w).reshape(1, -1) - center_x
    radial_freq = np.sqrt(y_coords**2 + x_coords**2)

    # Octave-dependent frequency band
    # octave +4 (char): high frequencies (0.3-1.0 normalized)
    # octave 0 (word): mid frequencies (0.1-0.5 normalized)
    # octave -4 (phrase): low frequencies (0.0-0.2 normalized)

    max_radius = np.sqrt(center_y**2 + center_x**2)
    normalized_freq = radial_freq / max_radius

    if octave >= 2:  # Character level
        band_min, band_max = 0.3, 1.0
    elif octave >= -1:  # Word level
        band_min, band_max = 0.1, 0.5
    else:  # Phrase level
        band_min, band_max = 0.0, 0.2

    # Apply band-pass filter
    band_mask = (normalized_freq >= band_min) & (normalized_freq <= band_max)
    filtered = magnitude * band_mask

    return filtered


def _extract_dominant_position(
    plane: np.ndarray,
    axis: int,
    grid_size: int
) -> int:
    """Extract dominant position from projected plane.

    Finds the center of mass of the frequency distribution along specified axis.

    Args:
        plane: 2D projected plane
        axis: Axis to extract position along (0 or 1)
        grid_size: Target grid size for normalization

    Returns:
        Integer coordinate in [0, grid_size)
    """
    if plane.ndim == 1:
        # 1D projection
        distribution = plane
    else:
        # 2D projection - sum along orthogonal axis
        distribution = np.sum(plane, axis=1-axis)

    # Find center of mass
    positions = np.arange(len(distribution))
    total_mass = distribution.sum()

    if total_mass > 1e-8:
        center_of_mass = (distribution * positions).sum() / total_mass
    else:
        # No signal - default to center
        center_of_mass = len(distribution) / 2.0

    # Normalize to grid size
    normalized = center_of_mass / len(distribution) * grid_size
    coord = int(np.clip(normalized, 0, grid_size - 1))

    return coord


def extract_multi_octave_coordinates(
    freq_spectrum: np.ndarray,
    modality: ModalityType = 'text',
    octaves: list[int] = [4, 0, -2]
) -> list[WaveCubeCoordinates]:
    """Extract coordinates at multiple octave levels.

    Enables multi-scale spatial clustering (character → word → phrase).

    Args:
        freq_spectrum: Frequency spectrum (H, W, 2 or 4)
        modality: Input modality
        octaves: List of octave levels to extract

    Returns:
        List of WaveCubeCoordinates, one per octave
    """
    coordinates = []

    for octave in octaves:
        coords = extract_triplanar_coordinates(
            freq_spectrum,
            modality=modality,
            octave=octave
        )
        coordinates.append(coords)

    return coordinates


def compute_spatial_distance(
    coords1: WaveCubeCoordinates,
    coords2: WaveCubeCoordinates
) -> float:
    """Compute Euclidean distance between two WaveCube coordinates.

    Used for 3D spatial clustering with tolerance=1.0 (exact matching).

    Args:
        coords1: First coordinates
        coords2: Second coordinates

    Returns:
        Euclidean distance in 3D space (ignoring W dimension)
    """
    dx = coords1.x - coords2.x
    dy = coords1.y - coords2.y
    dz = coords1.z - coords2.z

    distance = np.sqrt(dx**2 + dy**2 + dz**2)
    return float(distance)


def are_coordinates_equal(
    coords1: WaveCubeCoordinates,
    coords2: WaveCubeCoordinates,
    spatial_tolerance: float = 1.0
) -> bool:
    """Check if two coordinates are equal within spatial tolerance.

    Implements A=A=A and A≠B exact matching principle.

    Args:
        coords1: First coordinates
        coords2: Second coordinates
        spatial_tolerance: Distance threshold (default: 1.0 for exact matching)

    Returns:
        True if coordinates are within tolerance
    """
    distance = compute_spatial_distance(coords1, coords2)
    return distance < spatial_tolerance
