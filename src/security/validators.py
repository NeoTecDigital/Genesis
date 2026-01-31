"""
Input validation module for security hardening.

This module provides validators for various input types including
file paths, text input, numeric ranges, and CLI arguments using
Pydantic models for type safety and validation.
"""

import os
import re
from pathlib import Path
from typing import Optional, Any, Union, List, Tuple
from enum import Enum

from pydantic import BaseModel, Field, field_validator, ConfigDict


# Constants
MAX_TEXT_LENGTH = 1_000_000  # 1MB of text
MAX_PATH_LENGTH = 4096  # Linux PATH_MAX
FORBIDDEN_PATH_CHARS = set('<>|"\0')
FORBIDDEN_PATH_SEQUENCES = ['../', '/../', '\\..', '..\\']

# Sensitive system paths that should trigger warnings
SENSITIVE_PATHS = {
    '/etc/passwd', '/etc/shadow', '/etc/group', '/etc/sudoers',
    '/root/.ssh', '/var/log/auth.log', '/var/log/secure',
    '/boot', '/sys', '/proc'
}

# Sensitive path patterns (substring matches)
SENSITIVE_PATH_PATTERNS = [
    '/.ssh/', '/.ssh/id_rsa', '/etc/', '/root/', '/var/log/'
]


class SecurityLevel(str, Enum):
    """Security validation strictness levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    PARANOID = "paranoid"


class PathValidationResult(BaseModel):
    """Result of path validation."""
    is_valid: bool
    sanitized_path: Optional[Path] = None
    error_message: Optional[str] = None
    warnings: List[str] = Field(default_factory=list)


def sanitize_file_path(
    path: Union[str, Path],
    base_dir: Optional[Path] = None,
    security_level: SecurityLevel = SecurityLevel.MEDIUM
) -> PathValidationResult:
    """
    Sanitize and validate a file path.

    Args:
        path: Path to sanitize
        base_dir: Optional base directory to restrict paths within
        security_level: How strict to be with validation

    Returns:
        PathValidationResult with sanitized path or error
    """
    result = PathValidationResult(is_valid=False)

    try:
        # Convert to string for processing
        path_str = str(path)

        # Expand user home directory (~) before validation
        path_obj = Path(path_str).expanduser()
        path_str = str(path_obj)

        # Check length
        if len(path_str) > MAX_PATH_LENGTH:
            result.error_message = f"Path exceeds maximum length {MAX_PATH_LENGTH}"
            return result

        # Check for forbidden characters
        if any(char in path_str for char in FORBIDDEN_PATH_CHARS):
            result.error_message = f"Path contains forbidden characters"
            return result

        # Check for path traversal attempts
        for seq in FORBIDDEN_PATH_SEQUENCES:
            if seq in path_str:
                result.error_message = f"Path traversal attempt detected"
                return result

        # Convert to Path object and resolve
        clean_path = Path(path_str).resolve()

        # Check for sensitive system paths
        clean_path_str = str(clean_path)
        for sensitive_path in SENSITIVE_PATHS:
            if clean_path_str.startswith(sensitive_path) or clean_path_str == sensitive_path:
                result.warnings.append(f"Path accesses sensitive system location: {sensitive_path}")

        # Check for sensitive path patterns
        for pattern in SENSITIVE_PATH_PATTERNS:
            if pattern in clean_path_str:
                result.warnings.append(f"Path contains sensitive pattern: {pattern}")

        # If base_dir specified, ensure path is within it
        if base_dir:
            base_dir = Path(base_dir).resolve()
            try:
                clean_path.relative_to(base_dir)
            except ValueError:
                result.error_message = f"Path is outside allowed base directory"
                return result

        # Additional checks based on security level
        if security_level in [SecurityLevel.HIGH, SecurityLevel.PARANOID]:
            # Check for symbolic links
            if clean_path.is_symlink():
                if security_level == SecurityLevel.PARANOID:
                    result.error_message = "Symbolic links not allowed"
                    return result
                else:
                    result.warnings.append("Path is a symbolic link")

            # Check for hidden files
            if any(part.startswith('.') for part in clean_path.parts):
                if security_level == SecurityLevel.PARANOID:
                    result.error_message = "Hidden files/directories not allowed"
                    return result
                else:
                    result.warnings.append("Path contains hidden components")

        result.is_valid = True
        result.sanitized_path = clean_path
        return result

    except Exception as e:
        result.error_message = f"Path validation error: {str(e)}"
        return result


def validate_text_input(
    text: str,
    max_length: int = MAX_TEXT_LENGTH,
    allow_control_chars: bool = False,
    allow_unicode: bool = True
) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Validate text input for safety.

    Args:
        text: Text to validate
        max_length: Maximum allowed length
        allow_control_chars: Whether to allow control characters
        allow_unicode: Whether to allow unicode characters

    Returns:
        Tuple of (is_valid, sanitized_text, error_message)
    """
    if not text:
        return True, text, None

    # Check length
    if len(text) > max_length:
        return False, None, f"Text exceeds maximum length {max_length}"

    # Check for null bytes
    if '\0' in text:
        return False, None, "Text contains null bytes"

    sanitized = text

    # Remove control characters if not allowed
    if not allow_control_chars:
        # Keep newlines and tabs, remove other control chars
        control_char_pattern = r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]'
        sanitized = re.sub(control_char_pattern, '', sanitized)
        if sanitized != text:
            # Text was modified, log warning
            pass

    # Check for Unicode if not allowed
    if not allow_unicode:
        try:
            sanitized.encode('ascii')
        except UnicodeEncodeError:
            return False, None, "Text contains non-ASCII characters"

    return True, sanitized, None


class NumericRange(BaseModel):
    """Validator for numeric ranges."""

    model_config = ConfigDict(frozen=True)

    min_value: Optional[float] = None
    max_value: Optional[float] = None
    allow_negative: bool = True
    allow_zero: bool = True
    allow_infinity: bool = False

    def validate(self, value: Union[int, float]) -> Tuple[bool, Optional[str]]:
        """
        Validate a numeric value against the range.

        Args:
            value: Value to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check for NaN (always invalid)
        import math
        if isinstance(value, float) and math.isnan(value):
            return False, "NaN values not allowed"

        # Check for infinity
        if not self.allow_infinity and (value == float('inf') or value == float('-inf')):
            return False, "Infinite values not allowed"

        # Check for zero
        if not self.allow_zero and value == 0:
            return False, "Zero not allowed"

        # Check for negative
        if not self.allow_negative and value < 0:
            return False, "Negative values not allowed"

        # Check min bound
        if self.min_value is not None and value < self.min_value:
            return False, f"Value {value} below minimum {self.min_value}"

        # Check max bound
        if self.max_value is not None and value > self.max_value:
            return False, f"Value {value} above maximum {self.max_value}"

        return True, None


# Pydantic models for CLI argument validation

class EncodingArgs(BaseModel):
    """Validated arguments for encoding operations."""

    model_config = ConfigDict(str_strip_whitespace=True)

    input_text: str = Field(..., min_length=1, max_length=MAX_TEXT_LENGTH)
    output_path: Optional[Path] = None
    octave_levels: List[int] = Field(default=[-4, -2, 0, 4])
    clustering_threshold: float = Field(default=0.9, ge=0.0, le=1.0)

    @field_validator('input_text')
    @classmethod
    def validate_input_text(cls, v: str) -> str:
        """Validate and sanitize input text."""
        is_valid, sanitized, error = validate_text_input(v)
        if not is_valid:
            raise ValueError(error)
        return sanitized

    @field_validator('output_path')
    @classmethod
    def validate_output_path(cls, v: Optional[Path]) -> Optional[Path]:
        """Validate output path if provided."""
        if v is None:
            return None

        result = sanitize_file_path(v)
        if not result.is_valid:
            raise ValueError(result.error_message)

        return result.sanitized_path

    @field_validator('octave_levels')
    @classmethod
    def validate_octave_levels(cls, v: List[int]) -> List[int]:
        """Validate octave levels are within reasonable range."""
        for level in v:
            if abs(level) > 10:
                raise ValueError(f"Octave level {level} out of range [-10, 10]")
        return v


class DecodingArgs(BaseModel):
    """Validated arguments for decoding operations."""

    model_config = ConfigDict(str_strip_whitespace=True)

    input_path: Path = Field(...)
    output_path: Optional[Path] = None
    reconstruction_mode: str = Field(default="hierarchical", pattern="^(hierarchical|direct)$")
    batch_size: int = Field(default=32, ge=1, le=1024)

    @field_validator('input_path')
    @classmethod
    def validate_input_path(cls, v: Path) -> Path:
        """Validate input path exists and is safe."""
        result = sanitize_file_path(v)
        if not result.is_valid:
            raise ValueError(result.error_message)

        if not result.sanitized_path.exists():
            raise ValueError(f"Input file does not exist: {v}")

        if not result.sanitized_path.is_file():
            raise ValueError(f"Input path is not a file: {v}")

        return result.sanitized_path

    @field_validator('output_path')
    @classmethod
    def validate_output_path(cls, v: Optional[Path]) -> Optional[Path]:
        """Validate output path if provided."""
        if v is None:
            return None

        result = sanitize_file_path(v)
        if not result.is_valid:
            raise ValueError(result.error_message)

        return result.sanitized_path


class TrainingArgs(BaseModel):
    """Validated arguments for training operations."""

    model_config = ConfigDict(str_strip_whitespace=True)

    data_dir: Path = Field(...)
    checkpoint_dir: Path = Field(default=Path("./checkpoints"))
    batch_size: int = Field(default=32, ge=1, le=512)
    learning_rate: float = Field(default=1e-4, gt=0, lt=1.0)
    epochs: int = Field(default=10, ge=1, le=1000)
    validation_split: float = Field(default=0.2, ge=0.0, le=0.5)
    seed: Optional[int] = Field(default=None, ge=0, le=2**32-1)

    @field_validator('data_dir')
    @classmethod
    def validate_data_dir(cls, v: Path) -> Path:
        """Validate data directory exists and is safe."""
        result = sanitize_file_path(v)
        if not result.is_valid:
            raise ValueError(result.error_message)

        if not result.sanitized_path.exists():
            raise ValueError(f"Data directory does not exist: {v}")

        if not result.sanitized_path.is_dir():
            raise ValueError(f"Data path is not a directory: {v}")

        return result.sanitized_path

    @field_validator('checkpoint_dir')
    @classmethod
    def validate_checkpoint_dir(cls, v: Path) -> Path:
        """Validate checkpoint directory."""
        result = sanitize_file_path(v)
        if not result.is_valid:
            raise ValueError(result.error_message)

        # Create directory if it doesn't exist
        result.sanitized_path.mkdir(parents=True, exist_ok=True)

        return result.sanitized_path


def validate_file_extension(
    filepath: Path,
    allowed_extensions: List[str]
) -> Tuple[bool, Optional[str]]:
    """
    Validate file has an allowed extension.

    Args:
        filepath: Path to check
        allowed_extensions: List of allowed extensions (with dots)

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not allowed_extensions:
        return True, None

    suffix = filepath.suffix.lower()
    if suffix not in [ext.lower() for ext in allowed_extensions]:
        return False, f"File extension {suffix} not in allowed list {allowed_extensions}"

    return True, None


def create_validator_chain(*validators) -> Any:
    """
    Create a chain of validators that all must pass.

    Args:
        *validators: Validator functions

    Returns:
        Composite validator function
    """
    def chained_validator(value):
        for validator in validators:
            result = validator(value)
            if isinstance(result, tuple) and not result[0]:
                return result
        return True, None

    return chained_validator