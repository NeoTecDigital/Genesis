"""
Spatial coordinate systems and phase-locking utilities for WaveCube.

This module provides:
- Quaternionic coordinates (X, Y, Z, W) for multi-modal encoding
- Phase-locking utilities for cross-modal binding
- Spatial indexing and distance metrics
- Standing wave interference for proto-identity derivation
"""

from .coordinates import (
    QuaternionicCoord,
    Modality,
    create_phase_locked_set,
    find_nearest_phase_locked,
    compute_phase_matrix
)

from .phase_locking import (
    phase_shift,
    find_phase_locked,
    cross_modal_bind,
    create_phase_ring,
    compute_phase_coherence,
    find_phase_clusters,
    create_phase_gradient,
    optimize_phase_arrangement
)

from .interference import (
    InterferenceMode,
    StandingWaveInterference
)

__all__ = [
    # Coordinates
    'QuaternionicCoord',
    'Modality',
    'create_phase_locked_set',
    'find_nearest_phase_locked',
    'compute_phase_matrix',
    # Phase locking
    'phase_shift',
    'find_phase_locked',
    'cross_modal_bind',
    'create_phase_ring',
    'compute_phase_coherence',
    'find_phase_clusters',
    'create_phase_gradient',
    'optimize_phase_arrangement',
    # Interference
    'InterferenceMode',
    'StandingWaveInterference'
]