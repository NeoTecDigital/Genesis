"""CLI input validation for Genesis commands."""

import argparse
import logging
from pathlib import Path
from typing import Any, Optional

from src.security import (
    sanitize_file_path,
    SecurityLevel,
    validate_file_extension
)

logger = logging.getLogger(__name__)


class ValidatedPath(argparse.Action):
    """
    Custom argparse action that validates file paths.

    Prevents path traversal attacks and ensures paths are safe.
    """

    def __init__(
        self,
        option_strings,
        dest,
        required=False,
        must_exist=False,
        must_be_file=False,
        must_be_dir=False,
        allowed_extensions=None,
        security_level=SecurityLevel.MEDIUM,
        **kwargs
    ):
        """
        Initialize validated path action.

        Args:
            must_exist: Path must exist
            must_be_file: Path must be a file
            must_be_dir: Path must be a directory
            allowed_extensions: List of allowed extensions (e.g., ['.txt', '.pkl'])
            security_level: Security validation strictness
        """
        self.must_exist = must_exist
        self.must_be_file = must_be_file
        self.must_be_dir = must_be_dir
        self.allowed_extensions = allowed_extensions
        self.security_level = security_level
        super().__init__(option_strings, dest, required=required, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        """Validate and sanitize the path argument."""
        # Sanitize the path
        result = sanitize_file_path(values, security_level=self.security_level)

        if not result.is_valid:
            parser.error(f"Invalid path for {option_string}: {result.error_message}")

        path = result.sanitized_path

        # Warn about any security concerns
        for warning in result.warnings:
            logger.warning(f"Path validation warning for {option_string}: {warning}")

        # Check existence requirements
        if self.must_exist and not path.exists():
            parser.error(f"Path does not exist: {path}")

        # Check file/directory requirements
        if self.must_be_file and path.exists() and not path.is_file():
            parser.error(f"Path is not a file: {path}")

        if self.must_be_dir and path.exists() and not path.is_dir():
            parser.error(f"Path is not a directory: {path}")

        # Check extension requirements
        if self.allowed_extensions:
            is_valid, error = validate_file_extension(path, self.allowed_extensions)
            if not is_valid:
                parser.error(f"Invalid file extension for {option_string}: {error}")

        # Set the validated path
        setattr(namespace, self.dest, str(path))


class BoundedFloat(argparse.Action):
    """Custom argparse action for bounded float values."""

    def __init__(
        self,
        option_strings,
        dest,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        allow_negative: bool = True,
        allow_zero: bool = True,
        **kwargs
    ):
        """Initialize bounded float action."""
        self.min_value = min_value
        self.max_value = max_value
        self.allow_negative = allow_negative
        self.allow_zero = allow_zero
        super().__init__(option_strings, dest, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        """Validate float is within bounds."""
        try:
            value = float(values)
        except ValueError:
            parser.error(f"Invalid float value for {option_string}: {values}")

        if not self.allow_zero and value == 0:
            parser.error(f"Zero not allowed for {option_string}")

        if not self.allow_negative and value < 0:
            parser.error(f"Negative value not allowed for {option_string}")

        if self.min_value is not None and value < self.min_value:
            parser.error(f"Value for {option_string} must be >= {self.min_value}")

        if self.max_value is not None and value > self.max_value:
            parser.error(f"Value for {option_string} must be <= {self.max_value}")

        setattr(namespace, self.dest, value)


class BoundedInt(argparse.Action):
    """Custom argparse action for bounded integer values."""

    def __init__(
        self,
        option_strings,
        dest,
        min_value: Optional[int] = None,
        max_value: Optional[int] = None,
        allow_negative: bool = True,
        allow_zero: bool = True,
        **kwargs
    ):
        """Initialize bounded int action."""
        self.min_value = min_value
        self.max_value = max_value
        self.allow_negative = allow_negative
        self.allow_zero = allow_zero
        super().__init__(option_strings, dest, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        """Validate int is within bounds."""
        try:
            value = int(values)
        except ValueError:
            parser.error(f"Invalid integer value for {option_string}: {values}")

        if not self.allow_zero and value == 0:
            parser.error(f"Zero not allowed for {option_string}")

        if not self.allow_negative and value < 0:
            parser.error(f"Negative value not allowed for {option_string}")

        if self.min_value is not None and value < self.min_value:
            parser.error(f"Value for {option_string} must be >= {self.min_value}")

        if self.max_value is not None and value > self.max_value:
            parser.error(f"Value for {option_string} must be <= {self.max_value}")

        setattr(namespace, self.dest, value)


def add_secure_path_argument(
    parser: argparse.ArgumentParser,
    *args,
    must_exist: bool = False,
    must_be_file: bool = False,
    must_be_dir: bool = False,
    allowed_extensions: Optional[list] = None,
    security_level: SecurityLevel = SecurityLevel.MEDIUM,
    **kwargs
) -> None:
    """
    Add a secure path argument to a parser.

    This is a convenience function that adds ValidatedPath action.

    Args:
        parser: ArgumentParser to add to
        *args: Positional arguments for add_argument
        must_exist: Path must exist
        must_be_file: Path must be a file
        must_be_dir: Path must be a directory
        allowed_extensions: List of allowed extensions
        security_level: Security validation level
        **kwargs: Additional arguments for add_argument
    """
    parser.add_argument(
        *args,
        action=ValidatedPath,
        must_exist=must_exist,
        must_be_file=must_be_file,
        must_be_dir=must_be_dir,
        allowed_extensions=allowed_extensions,
        security_level=security_level,
        **kwargs
    )


def add_bounded_float_argument(
    parser: argparse.ArgumentParser,
    *args,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    allow_negative: bool = True,
    allow_zero: bool = True,
    **kwargs
) -> None:
    """Add a bounded float argument to a parser."""
    parser.add_argument(
        *args,
        action=BoundedFloat,
        min_value=min_value,
        max_value=max_value,
        allow_negative=allow_negative,
        allow_zero=allow_zero,
        **kwargs
    )


def add_bounded_int_argument(
    parser: argparse.ArgumentParser,
    *args,
    min_value: Optional[int] = None,
    max_value: Optional[int] = None,
    allow_negative: bool = True,
    allow_zero: bool = True,
    **kwargs
) -> None:
    """Add a bounded int argument to a parser."""
    parser.add_argument(
        *args,
        action=BoundedInt,
        min_value=min_value,
        max_value=max_value,
        allow_negative=allow_negative,
        allow_zero=allow_zero,
        **kwargs
    )