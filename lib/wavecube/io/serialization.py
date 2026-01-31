"""
Serialization and file I/O for wavetable matrices.

Supports multiple formats:
- NPZ: NumPy compressed archive (simple, portable)
- HDF5: Hierarchical format for large matrices (Phase 2)
- Wavecube Binary: Custom optimized format (Phase 3)
"""

import numpy as np
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Any
import json

if TYPE_CHECKING:
    from ..core.matrix import WavetableMatrix


def save_matrix(matrix: 'WavetableMatrix', path: str, format: str = 'npz') -> None:
    """
    Save wavetable matrix to disk.

    Args:
        matrix: WavetableMatrix to save
        path: File path
        format: Format ('npz', 'hdf5', 'wavecube')

    Raises:
        ValueError: If format is not supported
    """
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)

    if format == 'npz':
        _save_npz(matrix, path_obj)
    elif format == 'hdf5':
        raise NotImplementedError("HDF5 format will be implemented in Phase 2")
    elif format == 'wavecube':
        raise NotImplementedError("Wavecube binary format will be implemented in Phase 3")
    else:
        raise ValueError(f"Unknown format: {format}. Supported: npz, hdf5, wavecube")


def load_matrix(path: str) -> 'WavetableMatrix':
    """
    Load wavetable matrix from disk.

    Automatically detects format from file extension.

    Args:
        path: File path

    Returns:
        Loaded WavetableMatrix

    Raises:
        ValueError: If format cannot be determined
        FileNotFoundError: If file doesn't exist
    """
    path_obj = Path(path)

    if not path_obj.exists():
        raise FileNotFoundError(f"File not found: {path}")

    # Detect format from extension
    suffix = path_obj.suffix.lower()

    if suffix == '.npz':
        return _load_npz(path_obj)
    elif suffix == '.h5' or suffix == '.hdf5':
        raise NotImplementedError("HDF5 format will be implemented in Phase 2")
    elif suffix == '.wvcb' or suffix == '.wavecube':
        raise NotImplementedError("Wavecube binary format will be implemented in Phase 3")
    else:
        raise ValueError(
            f"Cannot determine format from extension: {suffix}. "
            "Supported: .npz, .h5/.hdf5 (Phase 2), .wvcb/.wavecube (Phase 3)"
        )


def _save_npz(matrix: 'WavetableMatrix', path: Path) -> None:
    """
    Save matrix to NPZ format.

    NPZ format stores:
    - metadata.json: Matrix configuration and statistics
    - node_{x}_{y}_{z}.npy: Individual wavetable arrays
    - node_{x}_{y}_{z}_meta.json: Node metadata (if any)

    Args:
        matrix: WavetableMatrix to save
        path: Path object
    """
    # Ensure .npz extension
    if path.suffix != '.npz':
        path = path.with_suffix('.npz')

    # Prepare data dict for np.savez_compressed
    data = {}

    # Save metadata
    metadata = {
        'width': matrix.width,
        'height': matrix.height,
        'depth': matrix.depth,
        'resolution': list(matrix.resolution),
        'channels': matrix.channels,
        'dtype': np.dtype(matrix.dtype).name,  # Use dtype.name for proper serialization
        'sparse': matrix.sparse,
        'compression': matrix.compression,
        'stats': matrix._stats,
        'num_nodes': len(matrix._nodes),
    }
    data['_metadata'] = np.array([json.dumps(metadata)])

    # Save each node
    for coords, node in matrix._nodes.items():
        x, y, z = coords
        key = f"node_{x}_{y}_{z}"

        if node.wavetable is not None:
            data[key] = node.wavetable

        # Save node metadata if present
        if node.metadata:
            meta_key = f"{key}_meta"
            data[meta_key] = np.array([json.dumps(node.metadata)])

        # Save compression info if compressed
        if node.compressed:
            comp_key = f"{key}_compressed"
            comp_data = {
                'method': node.compression_method,
                'resolution': list(node.resolution),
                'channels': node.channels,
            }
            # TODO: Add compressed_params serialization in Phase 2
            data[comp_key] = np.array([json.dumps(comp_data)])

    # Save to NPZ
    np.savez_compressed(str(path), **data)


def _load_npz(path: Path) -> 'WavetableMatrix':
    """
    Load matrix from NPZ format.

    Args:
        path: Path object

    Returns:
        Loaded WavetableMatrix
    """
    # Import here to avoid circular dependency
    from ..core.matrix import WavetableMatrix

    # Load NPZ file
    data = np.load(str(path), allow_pickle=True)

    # Load metadata
    if '_metadata' not in data:
        raise ValueError("Invalid NPZ file: missing _metadata")

    metadata = json.loads(str(data['_metadata'][0]))

    # Create matrix
    matrix = WavetableMatrix(
        width=metadata['width'],
        height=metadata['height'],
        depth=metadata['depth'],
        resolution=tuple(metadata['resolution']),
        channels=metadata['channels'],
        dtype=np.dtype(metadata['dtype']),
        sparse=metadata['sparse'],
        compression=metadata.get('compression'),
    )

    # Load nodes
    node_keys = [k for k in data.keys() if k.startswith('node_') and not k.endswith('_meta') and not k.endswith('_compressed')]

    for key in node_keys:
        # Parse coordinates from key "node_x_y_z"
        parts = key.split('_')
        if len(parts) != 4:
            continue

        try:
            x = int(parts[1])
            y = int(parts[2])
            z = int(parts[3])
        except ValueError:
            continue

        # Get wavetable
        wavetable = data[key]

        # Get metadata if present
        meta_key = f"{key}_meta"
        node_metadata = {}
        if meta_key in data:
            node_metadata = json.loads(str(data[meta_key][0]))

        # Check if compressed
        comp_key = f"{key}_compressed"
        compressed = comp_key in data

        if compressed:
            # TODO: Load compressed params in Phase 2
            # For now, just store the wavetable
            pass

        # Set node
        matrix.set_node(x, y, z, wavetable, metadata=node_metadata)

    # Restore statistics (they get recalculated during set_node, but we can verify)
    # matrix._stats = metadata['stats']

    return matrix


def export_matrix_info(matrix: 'WavetableMatrix') -> Dict[str, Any]:
    """
    Export matrix information as dictionary (for debugging/inspection).

    Args:
        matrix: WavetableMatrix

    Returns:
        Dict with matrix information
    """
    info = {
        'grid_dimensions': {
            'width': matrix.width,
            'height': matrix.height,
            'depth': matrix.depth,
        },
        'resolution': {
            'height': matrix.resolution[0],
            'width': matrix.resolution[1],
        },
        'channels': matrix.channels,
        'dtype': str(matrix.dtype),
        'storage': {
            'sparse': matrix.sparse,
            'compression': matrix.compression,
        },
        'statistics': matrix.get_memory_usage(),
        'populated_nodes': len(matrix._nodes),
        'node_coordinates': matrix.get_populated_nodes(),
    }

    return info
