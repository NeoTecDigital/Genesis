"""
Tests for the input validators security module prototype.
"""

from pathlib import Path
import tempfile
import pytest
from pydantic import ValidationError

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.security.validators import (
    sanitize_file_path,
    validate_text_input,
    validate_file_extension,
    create_validator_chain,
    NumericRange,
    EncodingArgs,
    DecodingArgs,
    TrainingArgs,
    SecurityLevel,
    PathValidationResult,
    MAX_TEXT_LENGTH,
    MAX_PATH_LENGTH
)


class TestPathSanitization:
    """Test suite for path sanitization."""

    def test_valid_path(self):
        """Test that valid paths pass validation."""
        result = sanitize_file_path("/home/user/file.txt")
        assert result.is_valid
        assert result.sanitized_path is not None
        assert result.error_message is None

    def test_path_traversal_detection(self):
        """Test that path traversal attempts are detected."""
        dangerous_paths = [
            "../etc/passwd",
            "../../sensitive.txt",
            "valid/../../../etc/passwd",
            "some/path/../../../root",
            "..\\windows\\system32"
        ]

        for path in dangerous_paths:
            result = sanitize_file_path(path)
            assert not result.is_valid
            assert "traversal" in result.error_message.lower()

    def test_forbidden_characters(self):
        """Test that forbidden characters are rejected."""
        bad_paths = [
            "file<name>.txt",
            "file>name.txt",
            'file"name".txt',
            "file|name.txt",
            "file\0name.txt"
        ]

        for path in bad_paths:
            result = sanitize_file_path(path)
            assert not result.is_valid
            assert "forbidden" in result.error_message.lower()

    def test_path_length_limit(self):
        """Test that overly long paths are rejected."""
        long_path = "a" * (MAX_PATH_LENGTH + 1)
        result = sanitize_file_path(long_path)
        assert not result.is_valid
        assert "length" in result.error_message.lower()

    def test_base_directory_restriction(self, tmp_path):
        """Test that paths can be restricted to base directory."""
        base_dir = tmp_path
        allowed_file = base_dir / "allowed.txt"
        allowed_file.touch()

        # Path within base_dir should be allowed
        result = sanitize_file_path(allowed_file, base_dir=base_dir)
        assert result.is_valid

        # Path outside base_dir should be rejected
        outside_path = tmp_path.parent / "outside.txt"
        result = sanitize_file_path(outside_path, base_dir=base_dir)
        assert not result.is_valid
        assert "outside" in result.error_message.lower()

    def test_security_levels(self, tmp_path):
        """Test different security levels."""
        # Create a hidden file
        hidden_file = tmp_path / ".hidden"
        hidden_file.touch()

        # Low security - should allow hidden files
        result = sanitize_file_path(hidden_file, security_level=SecurityLevel.LOW)
        assert result.is_valid

        # High security - should warn about hidden files
        result = sanitize_file_path(hidden_file, security_level=SecurityLevel.HIGH)
        assert result.is_valid
        assert len(result.warnings) > 0
        assert "hidden" in result.warnings[0].lower()

        # Paranoid security - should reject hidden files
        result = sanitize_file_path(hidden_file, security_level=SecurityLevel.PARANOID)
        assert not result.is_valid
        assert "hidden" in result.error_message.lower()


class TestTextValidation:
    """Test suite for text validation."""

    def test_valid_text(self):
        """Test that valid text passes validation."""
        valid, sanitized, error = validate_text_input("Hello, World!")
        assert valid
        assert sanitized == "Hello, World!"
        assert error is None

    def test_empty_text(self):
        """Test that empty text is allowed."""
        valid, sanitized, error = validate_text_input("")
        assert valid
        assert sanitized == ""
        assert error is None

    def test_text_length_limit(self):
        """Test that text length is enforced."""
        long_text = "a" * (MAX_TEXT_LENGTH + 1)
        valid, sanitized, error = validate_text_input(long_text)
        assert not valid
        assert sanitized is None
        assert "length" in error.lower()

    def test_null_byte_rejection(self):
        """Test that null bytes are rejected."""
        text_with_null = "Hello\0World"
        valid, sanitized, error = validate_text_input(text_with_null)
        assert not valid
        assert "null" in error.lower()

    def test_control_character_removal(self):
        """Test that control characters are removed when not allowed."""
        text_with_control = "Hello\x01\x02World\x7F"
        valid, sanitized, error = validate_text_input(
            text_with_control,
            allow_control_chars=False
        )
        assert valid
        assert sanitized == "HelloWorld"
        assert error is None

    def test_control_characters_allowed(self):
        """Test that control characters pass when allowed."""
        text_with_control = "Hello\x01World"
        valid, sanitized, error = validate_text_input(
            text_with_control,
            allow_control_chars=True
        )
        assert valid
        assert sanitized == text_with_control
        assert error is None

    def test_unicode_rejection(self):
        """Test that unicode is rejected when not allowed."""
        unicode_text = "Hello 世界"
        valid, sanitized, error = validate_text_input(
            unicode_text,
            allow_unicode=False
        )
        assert not valid
        assert "ASCII" in error

    def test_unicode_allowed(self):
        """Test that unicode passes when allowed."""
        unicode_text = "Hello 世界"
        valid, sanitized, error = validate_text_input(
            unicode_text,
            allow_unicode=True
        )
        assert valid
        assert sanitized == unicode_text


class TestNumericRange:
    """Test suite for numeric range validation."""

    def test_within_range(self):
        """Test values within range pass."""
        range_validator = NumericRange(min_value=0, max_value=100)
        valid, error = range_validator.validate(50)
        assert valid
        assert error is None

    def test_below_minimum(self):
        """Test values below minimum fail."""
        range_validator = NumericRange(min_value=10)
        valid, error = range_validator.validate(5)
        assert not valid
        assert "below minimum" in error

    def test_above_maximum(self):
        """Test values above maximum fail."""
        range_validator = NumericRange(max_value=100)
        valid, error = range_validator.validate(150)
        assert not valid
        assert "above maximum" in error

    def test_negative_rejection(self):
        """Test negative values rejected when not allowed."""
        range_validator = NumericRange(allow_negative=False)
        valid, error = range_validator.validate(-5)
        assert not valid
        assert "Negative" in error

    def test_zero_rejection(self):
        """Test zero rejected when not allowed."""
        range_validator = NumericRange(allow_zero=False)
        valid, error = range_validator.validate(0)
        assert not valid
        assert "Zero" in error

    def test_infinity_rejection(self):
        """Test infinity rejected when not allowed."""
        range_validator = NumericRange(allow_infinity=False)
        valid, error = range_validator.validate(float('inf'))
        assert not valid
        assert "Infinite" in error


class TestPydanticModels:
    """Test suite for Pydantic validation models."""

    def test_encoding_args_valid(self):
        """Test valid encoding arguments."""
        args = EncodingArgs(
            input_text="Test text to encode",
            octave_levels=[0, 2, 4],
            clustering_threshold=0.85
        )
        assert args.input_text == "Test text to encode"
        assert args.octave_levels == [0, 2, 4]
        assert args.clustering_threshold == 0.85

    def test_encoding_args_text_validation(self):
        """Test text validation in encoding args."""
        # Empty text should fail
        with pytest.raises(ValidationError, match="at least 1 character"):
            EncodingArgs(input_text="")

        # Overly long text should fail
        long_text = "x" * (MAX_TEXT_LENGTH + 1)
        with pytest.raises(ValidationError):
            EncodingArgs(input_text=long_text)

    def test_encoding_args_octave_validation(self):
        """Test octave level validation."""
        # Out of range octave should fail
        with pytest.raises(ValidationError, match="out of range"):
            EncodingArgs(
                input_text="test",
                octave_levels=[15]  # Too high
            )

    def test_encoding_args_threshold_validation(self):
        """Test clustering threshold validation."""
        # Out of range threshold should fail
        with pytest.raises(ValidationError):
            EncodingArgs(
                input_text="test",
                clustering_threshold=1.5  # Above 1.0
            )

    def test_decoding_args_valid(self, tmp_path):
        """Test valid decoding arguments."""
        test_file = tmp_path / "test.pkl"
        test_file.touch()

        args = DecodingArgs(
            input_path=test_file,
            reconstruction_mode="hierarchical",
            batch_size=64
        )
        assert args.input_path == test_file
        assert args.reconstruction_mode == "hierarchical"
        assert args.batch_size == 64

    def test_decoding_args_file_validation(self):
        """Test file path validation in decoding args."""
        # Non-existent file should fail
        with pytest.raises(ValidationError, match="does not exist"):
            DecodingArgs(input_path=Path("/nonexistent/file.pkl"))

    def test_decoding_args_mode_validation(self):
        """Test reconstruction mode validation."""
        test_file = Path(__file__)  # Use this file as valid input

        # Invalid mode should fail
        with pytest.raises(ValidationError):
            DecodingArgs(
                input_path=test_file,
                reconstruction_mode="invalid_mode"
            )

    def test_training_args_valid(self, tmp_path):
        """Test valid training arguments."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        args = TrainingArgs(
            data_dir=data_dir,
            batch_size=16,
            learning_rate=0.001,
            epochs=5
        )
        assert args.data_dir == data_dir
        assert args.batch_size == 16
        assert args.learning_rate == 0.001
        assert args.epochs == 5

    def test_training_args_directory_validation(self):
        """Test directory validation in training args."""
        # Non-existent directory should fail
        with pytest.raises(ValidationError, match="does not exist"):
            TrainingArgs(data_dir=Path("/nonexistent/dir"))

    def test_training_args_range_validation(self):
        """Test numeric range validation in training args."""
        tmp_dir = Path(__file__).parent  # Valid directory

        # Invalid batch size
        with pytest.raises(ValidationError):
            TrainingArgs(data_dir=tmp_dir, batch_size=0)

        # Invalid learning rate
        with pytest.raises(ValidationError):
            TrainingArgs(data_dir=tmp_dir, learning_rate=1.5)

        # Invalid epochs
        with pytest.raises(ValidationError):
            TrainingArgs(data_dir=tmp_dir, epochs=0)


class TestFileExtensionValidation:
    """Test suite for file extension validation."""

    def test_allowed_extension(self):
        """Test that allowed extensions pass."""
        valid, error = validate_file_extension(
            Path("file.txt"),
            [".txt", ".md"]
        )
        assert valid
        assert error is None

    def test_disallowed_extension(self):
        """Test that disallowed extensions fail."""
        valid, error = validate_file_extension(
            Path("file.exe"),
            [".txt", ".md"]
        )
        assert not valid
        assert "not in allowed list" in error

    def test_case_insensitive(self):
        """Test that extension checking is case-insensitive."""
        valid, error = validate_file_extension(
            Path("file.TXT"),
            [".txt"]
        )
        assert valid

    def test_empty_allowed_list(self):
        """Test that empty allowed list allows all."""
        valid, error = validate_file_extension(
            Path("file.anything"),
            []
        )
        assert valid


class TestValidatorChain:
    """Test suite for validator chaining."""

    def test_all_pass(self):
        """Test chain when all validators pass."""
        def always_pass(value):
            return True, None

        chain = create_validator_chain(always_pass, always_pass)
        result = chain("test")
        assert result == (True, None)

    def test_first_fails(self):
        """Test chain stops at first failure."""
        def always_fail(value):
            return False, "First failed"

        def always_pass(value):
            return True, None

        chain = create_validator_chain(always_fail, always_pass)
        result = chain("test")
        assert result == (False, "First failed")

    def test_mixed_validators(self):
        """Test chain with mixed validators."""
        def check_length(value):
            if len(value) < 5:
                return False, "Too short"
            return True, None

        def check_content(value):
            if "bad" in value:
                return False, "Contains bad word"
            return True, None

        chain = create_validator_chain(check_length, check_content)

        # Should pass both
        assert chain("good text") == (True, None)

        # Should fail length check
        assert chain("hi")[0] is False

        # Should fail content check
        assert chain("bad text")[0] is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])