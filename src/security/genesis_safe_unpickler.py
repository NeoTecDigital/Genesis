"""
Genesis-specific safe unpickling configuration.

This module extends the base safe unpickler with Genesis-specific
classes that are allowed to be unpickled.
"""

from typing import Optional, Any, Union
from pathlib import Path
import logging
import warnings

from .safe_unpickler import (
    SafeLoadConfig,
    RestrictedUnpickler,
    safe_load as _base_safe_load,
    safe_dump as _base_safe_dump,
    add_torch_support
)

# Setup logging
logger = logging.getLogger(__name__)


def get_genesis_safe_config(
    verify_signature: bool = True,
    backward_compatible: bool = True
) -> SafeLoadConfig:
    """
    Get Genesis-specific safe loading configuration.

    Args:
        verify_signature: Whether to verify HMAC signatures
        backward_compatible: Whether to allow loading without signatures (with warning)

    Returns:
        SafeLoadConfig with Genesis classes whitelisted
    """
    config = SafeLoadConfig(verify_signature=verify_signature)

    # Add Genesis-specific modules
    config.allowed_modules.update({
        'memory.voxel_cloud',
        'memory.voxel_helpers',
        'memory.octave_hierarchy',
        'memory.temporal_buffer',
        'memory.state_classifier',
        'memory.frequency_bands',
        'memory.voxel_cloud_collapse',
        'memory.voxel_cloud_query',
        'src.memory.voxel_cloud',
        'src.memory.voxel_helpers',
        'src.memory.octave_hierarchy',
        'src.memory.temporal_buffer',
        'src.memory.state_classifier',
        'src.memory.frequency_bands',
        'src.memory.voxel_cloud_collapse',
        'src.memory.voxel_cloud_query',
        'dataclasses',
        'enum'
    })

    # Add Genesis-specific classes
    config.allowed_classes.update({
        # Core data structures
        'ProtoIdentityEntry',
        'VoxelCloud',
        'OctaveHierarchy',
        'OctaveProtoIdentity',
        'TemporalBuffer',
        'StateClassifier',
        'SignalState',
        'FrequencyBandClustering',
        'FrequencyBand',

        # Memory hierarchy structures
        'MemoryHierarchy',
        'CoreMemory',
        'ExperientialMemory',

        # Encoder/decoder classes
        'MultiOctaveEncoder',
        'MultiOctaveDecoder',
        'VoxelCloudClustering',

        # Dataclasses and enums
        'dataclasses._HAS_DEFAULT_FACTORY_CLASS',
        'collections.defaultdict',
        'enum.EnumMeta',

        # Python built-ins for Genesis structures
        'range',
        'slice',
        'complex',
        'bytearray',

        # Prefix-based matching for Genesis classes
        'src.memory.frequency_bands.FrequencyBand',
        'src.memory.frequency_bands.FrequencyBandClustering',
        'src.memory.voxel_cloud.ProtoIdentityEntry',
        'src.memory.voxel_cloud.VoxelCloud',
        'src.memory.octave_hierarchy.OctaveHierarchy',
        'src.memory.octave_hierarchy.OctaveProtoIdentity',
        'src.memory.temporal_buffer.TemporalBuffer',
        'src.memory.state_classifier.StateClassifier',
        'src.memory.state_classifier.SignalState'
    })

    # Add PyTorch support if needed
    try:
        import torch
        add_torch_support(config)
    except ImportError:
        pass

    return config


def safe_load_pickle(
    filepath: Union[str, Path],
    verify_signature: bool = True,
    backward_compatible: bool = True,
    expected_signature: Optional[str] = None
) -> Any:
    """
    Safely load a Genesis pickle file with security checks.

    This function provides backward compatibility for existing pickle files
    while encouraging migration to signed files.

    Args:
        filepath: Path to pickle file
        verify_signature: Whether to verify HMAC signatures
        backward_compatible: Whether to allow loading without signatures (with warning)
        expected_signature: Expected HMAC signature (if verification enabled)

    Returns:
        Unpickled object

    Raises:
        ValueError: If verification fails or file is unsafe
        FileNotFoundError: If file doesn't exist
    """
    filepath = Path(filepath)

    # Get Genesis-specific config
    config = get_genesis_safe_config(
        verify_signature=verify_signature,
        backward_compatible=backward_compatible
    )

    # Check if signature file exists
    sig_file = filepath.with_suffix(filepath.suffix + '.sig')
    has_signature = sig_file.exists() or expected_signature is not None

    # Handle backward compatibility
    if verify_signature and not has_signature and backward_compatible:
        warnings.warn(
            f"Loading unsigned pickle file: {filepath}\n"
            "This file should be re-saved with HMAC signature for security.\n"
            "Use 'genesis migrate-pickles' to update all pickle files.",
            UserWarning
        )
        logger.warning(f"Loading unsigned pickle file: {filepath}")

        # Load without signature verification for backward compatibility
        config.verify_signature = False

    try:
        # Use base safe_load with Genesis config
        return _base_safe_load(filepath, config, expected_signature)

    except Exception as e:
        logger.error(f"Failed to load pickle file {filepath}: {e}")

        # Provide helpful error message
        if "not in whitelist" in str(e):
            logger.error(
                "The pickle file contains classes not in the security whitelist.\n"
                "This may be a malicious file or an outdated Genesis file.\n"
                "If this is a trusted Genesis file, please report this issue."
            )
        raise


def safe_save_pickle(
    obj: Any,
    filepath: Union[str, Path],
    create_signature: bool = True
) -> Optional[str]:
    """
    Safely save an object to pickle with optional HMAC signature.

    Args:
        obj: Object to pickle
        filepath: Path to save to
        create_signature: Whether to create signature file

    Returns:
        HMAC signature if created, None otherwise
    """
    filepath = Path(filepath)

    # Get Genesis config for HMAC key
    config = get_genesis_safe_config(verify_signature=create_signature)

    # Use base safe_dump
    signature = _base_safe_dump(obj, filepath, config, save_signature=create_signature)

    if signature:
        logger.info(f"Saved pickle with HMAC signature: {filepath}")
    else:
        logger.info(f"Saved pickle without signature: {filepath}")

    return signature


def migrate_pickle_file(
    filepath: Union[str, Path],
    force: bool = False
) -> bool:
    """
    Migrate an existing pickle file to use HMAC signatures.

    Args:
        filepath: Path to pickle file
        force: Force re-signing even if signature exists

    Returns:
        True if migration successful
    """
    filepath = Path(filepath)
    sig_file = filepath.with_suffix(filepath.suffix + '.sig')

    # Check if already has signature
    if sig_file.exists() and not force:
        logger.info(f"File already has signature: {filepath}")
        return True

    try:
        # Load the pickle file
        logger.info(f"Migrating pickle file: {filepath}")
        obj = safe_load_pickle(filepath, verify_signature=False)

        # Re-save with signature
        safe_save_pickle(obj, filepath, create_signature=True)

        logger.info(f"Successfully migrated: {filepath}")
        return True

    except Exception as e:
        logger.error(f"Failed to migrate {filepath}: {e}")
        return False