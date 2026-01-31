"""
Security module for Genesis project.

Provides secure alternatives for potentially dangerous operations:
- Safe pickle loading/saving with class whitelisting
- Input validation and sanitization
- Path traversal prevention
"""

from .safe_unpickler import (
    RestrictedUnpickler,
    SafeLoadConfig,
    safe_load,
    safe_dump,
    compute_file_hmac,
    verify_file_signature,
    add_torch_support,
)

from .genesis_safe_unpickler import (
    safe_load_pickle,
    safe_save_pickle,
    migrate_pickle_file,
    get_genesis_safe_config,
)

from .validators import (
    # Core validation functions
    sanitize_file_path,
    validate_text_input,
    validate_file_extension,
    create_validator_chain,
    # Types and enums
    SecurityLevel,
    PathValidationResult,
    NumericRange,
    # Pydantic models
    EncodingArgs,
    DecodingArgs,
    TrainingArgs,
    # Constants
    MAX_TEXT_LENGTH,
    MAX_PATH_LENGTH,
)

__all__ = [
    # Safe unpickler
    "RestrictedUnpickler",
    "SafeLoadConfig",
    "safe_load",
    "safe_dump",
    "compute_file_hmac",
    "verify_file_signature",
    "add_torch_support",
    # Genesis-specific safe unpickling
    "safe_load_pickle",
    "safe_save_pickle",
    "migrate_pickle_file",
    "get_genesis_safe_config",
    # Validators
    "sanitize_file_path",
    "validate_text_input",
    "validate_file_extension",
    "create_validator_chain",
    "SecurityLevel",
    "PathValidationResult",
    "NumericRange",
    "EncodingArgs",
    "DecodingArgs",
    "TrainingArgs",
    "MAX_TEXT_LENGTH",
    "MAX_PATH_LENGTH",
]

# Version info
__version__ = "1.0.0"
__author__ = "Genesis Security Team"