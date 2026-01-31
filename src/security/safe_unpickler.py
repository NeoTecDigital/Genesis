"""
Safe unpickling module with restricted class loading and HMAC verification.

This module provides a secure alternative to pickle.load() that prevents
arbitrary code execution by whitelisting allowed classes and verifying
file integrity through HMAC signatures.
"""

import pickle
import hashlib
import hmac
import os
from pathlib import Path
from typing import Any, Set, Optional, Union, Type
from dataclasses import dataclass
import numpy as np

# Constants
DEFAULT_MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB
HMAC_KEY_ENV_VAR = "GENESIS_HMAC_KEY"


@dataclass
class SafeLoadConfig:
    """Configuration for safe unpickling operations."""

    max_file_size: int = DEFAULT_MAX_FILE_SIZE
    verify_signature: bool = True
    hmac_key: Optional[bytes] = None
    allowed_modules: Optional[Set[str]] = None
    allowed_classes: Optional[Set[str]] = None

    def __post_init__(self):
        """Initialize default allowed modules and classes if not provided."""
        if self.allowed_modules is None:
            self.allowed_modules = {
                'numpy',
                'numpy.core.multiarray',
                'numpy.core.numeric',
                'numpy._core.multiarray',  # Newer numpy versions
                'numpy._core.numeric',
                '__builtin__',
                'builtins',
                'collections',
                'collections.abc',
            }

        if self.allowed_classes is None:
            self.allowed_classes = {
                'numpy.ndarray',
                'numpy.dtype',
                'numpy.float32',
                'numpy.float64',
                'numpy.int32',
                'numpy.int64',
                'numpy.uint8',
                'numpy.core.multiarray.scalar',  # Old numpy versions
                'numpy._core.multiarray.scalar',  # New numpy versions
                '_reconstruct',  # numpy internal
                '_frombuffer',  # numpy internal
                'dict',
                'list',
                'tuple',
                'set',
                'frozenset',
                'OrderedDict',
            }

        # Get HMAC key from environment if not provided
        if self.verify_signature and self.hmac_key is None:
            env_key = os.environ.get(HMAC_KEY_ENV_VAR)
            if env_key:
                self.hmac_key = env_key.encode('utf-8')


class RestrictedUnpickler(pickle.Unpickler):
    """
    Restricted unpickler that only allows whitelisted classes.

    This unpickler prevents arbitrary code execution by restricting
    which classes can be instantiated during unpickling.
    """

    def __init__(self, file, config: Optional[SafeLoadConfig] = None):
        """
        Initialize the restricted unpickler.

        Args:
            file: File object to read from
            config: Configuration for safe loading
        """
        super().__init__(file)
        self.config = config or SafeLoadConfig()

    def find_class(self, module: str, name: str) -> Type:
        """
        Override find_class to restrict which classes can be loaded.

        Args:
            module: Module name
            name: Class name

        Returns:
            The requested class if allowed

        Raises:
            pickle.UnpicklingError: If class is not whitelisted
        """
        # Check if module is allowed
        if module not in self.config.allowed_modules:
            raise pickle.UnpicklingError(
                f"Module '{module}' is not in whitelist"
            )

        # Check if class is allowed
        full_name = f"{module}.{name}"
        if full_name not in self.config.allowed_classes and \
           name not in self.config.allowed_classes:
            raise pickle.UnpicklingError(
                f"Class '{full_name}' is not in whitelist"
            )

        # Use parent's find_class if checks pass
        return super().find_class(module, name)


def compute_file_hmac(filepath: Path, key: bytes) -> str:
    """
    Compute HMAC-SHA256 for a file.

    Args:
        filepath: Path to file
        key: HMAC key

    Returns:
        Hex string of HMAC digest
    """
    h = hmac.new(key, digestmod=hashlib.sha256)

    with open(filepath, 'rb') as f:
        # Read in chunks for memory efficiency
        while chunk := f.read(8192):
            h.update(chunk)

    return h.hexdigest()


def verify_file_signature(
    filepath: Path,
    expected_signature: str,
    key: bytes
) -> bool:
    """
    Verify file HMAC signature.

    Args:
        filepath: Path to file
        expected_signature: Expected HMAC hex string
        key: HMAC key

    Returns:
        True if signature matches
    """
    actual_signature = compute_file_hmac(filepath, key)
    return hmac.compare_digest(actual_signature, expected_signature)


def safe_load(
    filepath: Union[str, Path],
    config: Optional[SafeLoadConfig] = None,
    expected_signature: Optional[str] = None
) -> Any:
    """
    Safely load a pickled object with security checks.

    Args:
        filepath: Path to pickle file
        config: Configuration for safe loading
        expected_signature: Expected HMAC signature (if verification enabled)

    Returns:
        Unpickled object

    Raises:
        ValueError: If file size exceeds limit or signature verification fails
        pickle.UnpicklingError: If unpickling encounters disallowed classes
    """
    filepath = Path(filepath)
    config = config or SafeLoadConfig()

    # Check file exists
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    # Check file size
    file_size = filepath.stat().st_size
    if file_size > config.max_file_size:
        raise ValueError(
            f"File size {file_size} exceeds limit {config.max_file_size}"
        )

    # Verify signature if enabled
    if config.verify_signature:
        if not config.hmac_key:
            raise ValueError(
                "HMAC verification enabled but no key provided. "
                f"Set {HMAC_KEY_ENV_VAR} environment variable or provide key."
            )

        if not expected_signature:
            # Try to load signature from companion file
            sig_file = filepath.with_suffix(filepath.suffix + '.sig')
            if sig_file.exists():
                expected_signature = sig_file.read_text().strip()
            else:
                raise ValueError(
                    "HMAC verification enabled but no signature provided"
                )

        if not verify_file_signature(filepath, expected_signature, config.hmac_key):
            raise ValueError("HMAC signature verification failed")

    # Load with restricted unpickler
    with open(filepath, 'rb') as f:
        unpickler = RestrictedUnpickler(f, config)
        return unpickler.load()


def safe_dump(
    obj: Any,
    filepath: Union[str, Path],
    config: Optional[SafeLoadConfig] = None,
    save_signature: bool = True
) -> Optional[str]:
    """
    Safely dump an object with optional HMAC signature.

    Args:
        obj: Object to pickle
        filepath: Path to save to
        config: Configuration (used for HMAC key)
        save_signature: Whether to save signature file

    Returns:
        HMAC signature if verification is enabled, None otherwise
    """
    filepath = Path(filepath)
    config = config or SafeLoadConfig()

    # Save the object
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Generate and save signature if enabled
    if config.verify_signature and config.hmac_key:
        signature = compute_file_hmac(filepath, config.hmac_key)

        if save_signature:
            sig_file = filepath.with_suffix(filepath.suffix + '.sig')
            sig_file.write_text(signature)

        return signature

    return None


# Convenience function for torch tensor support (when available)
def add_torch_support(config: SafeLoadConfig) -> None:
    """
    Add PyTorch tensor classes to the whitelist.

    Args:
        config: Configuration to update
    """
    config.allowed_modules.update({
        'torch',
        'torch._utils',
        'torch.nn',
        'torch.storage',
    })

    config.allowed_classes.update({
        'torch.Tensor',
        'torch.FloatTensor',
        'torch.DoubleTensor',
        'torch.IntTensor',
        'torch.LongTensor',
        'torch.ByteTensor',
        'torch.nn.parameter.Parameter',
        'torch.storage._TypedStorage',
    })