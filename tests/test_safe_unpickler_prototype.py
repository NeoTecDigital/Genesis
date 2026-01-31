"""
Tests for the RestrictedUnpickler security module prototype.
"""

import os
import pickle
import tempfile
from pathlib import Path
import numpy as np
import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.security.safe_unpickler import (
    RestrictedUnpickler,
    SafeLoadConfig,
    safe_load,
    safe_dump,
    compute_file_hmac,
    verify_file_signature,
    add_torch_support
)


class TestRestrictedUnpickler:
    """Test suite for RestrictedUnpickler."""

    def test_allows_numpy_arrays(self, tmp_path):
        """Test that numpy arrays can be safely loaded."""
        # Create test data
        test_array = np.array([1, 2, 3, 4, 5], dtype=np.float32)
        test_file = tmp_path / "test_array.pkl"

        # Save with regular pickle
        with open(test_file, 'wb') as f:
            pickle.dump(test_array, f)

        # Load with restricted unpickler
        config = SafeLoadConfig(verify_signature=False)
        loaded = safe_load(test_file, config)

        assert np.array_equal(loaded, test_array)

    def test_allows_basic_types(self, tmp_path):
        """Test that basic Python types can be loaded."""
        test_data = {
            'list': [1, 2, 3],
            'tuple': (4, 5, 6),
            'dict': {'a': 1, 'b': 2},
            'set': {7, 8, 9},
        }
        test_file = tmp_path / "test_basic.pkl"

        with open(test_file, 'wb') as f:
            pickle.dump(test_data, f)

        config = SafeLoadConfig(verify_signature=False)
        loaded = safe_load(test_file, config)

        assert loaded['list'] == test_data['list']
        assert loaded['tuple'] == test_data['tuple']
        assert loaded['dict'] == test_data['dict']
        assert loaded['set'] == test_data['set']

    def test_blocks_dangerous_classes(self, tmp_path):
        """Test that dangerous classes are blocked."""
        # Try to pickle an os module function (should be blocked on load)
        test_file = tmp_path / "test_dangerous.pkl"

        import os
        # Pickle os.system (dangerous!)
        with open(test_file, 'wb') as f:
            pickle.dump(os.system, f)

        config = SafeLoadConfig(verify_signature=False)

        with pytest.raises(pickle.UnpicklingError, match="not in whitelist"):
            safe_load(test_file, config)

    def test_file_size_limit(self, tmp_path):
        """Test that file size limits are enforced."""
        test_file = tmp_path / "test_large.pkl"

        # Create a file larger than limit
        large_data = np.zeros(1000000, dtype=np.uint8)
        with open(test_file, 'wb') as f:
            pickle.dump(large_data, f)

        # Set small limit
        config = SafeLoadConfig(
            max_file_size=100,  # 100 bytes
            verify_signature=False
        )

        with pytest.raises(ValueError, match="exceeds limit"):
            safe_load(test_file, config)

    def test_hmac_signature_verification(self, tmp_path):
        """Test HMAC signature generation and verification."""
        test_data = {'test': 'data'}
        test_file = tmp_path / "test_signed.pkl"
        key = b"test_secret_key_12345"

        # Save with signature
        config = SafeLoadConfig(
            verify_signature=True,
            hmac_key=key
        )
        signature = safe_dump(test_data, test_file, config, save_signature=True)

        assert signature is not None

        # Verify signature file was created
        sig_file = test_file.with_suffix('.pkl.sig')
        assert sig_file.exists()
        assert sig_file.read_text().strip() == signature

        # Load with signature verification
        loaded = safe_load(test_file, config)
        assert loaded == test_data

        # Modify file and verify it fails
        with open(test_file, 'ab') as f:
            f.write(b'corrupted')

        with pytest.raises(ValueError, match="verification failed"):
            safe_load(test_file, config)

    def test_missing_signature_file(self, tmp_path):
        """Test behavior when signature file is missing."""
        test_file = tmp_path / "test_unsigned.pkl"

        with open(test_file, 'wb') as f:
            pickle.dump({'test': 'data'}, f)

        config = SafeLoadConfig(
            verify_signature=True,
            hmac_key=b"test_key"
        )

        with pytest.raises(ValueError, match="no signature provided"):
            safe_load(test_file, config)

    def test_environment_key_loading(self, tmp_path):
        """Test loading HMAC key from environment."""
        test_file = tmp_path / "test_env_key.pkl"
        test_data = {'env': 'test'}

        # Set environment variable
        os.environ['GENESIS_HMAC_KEY'] = 'env_secret_key'

        try:
            # Create config without explicit key
            config = SafeLoadConfig(verify_signature=True)
            assert config.hmac_key == b'env_secret_key'

            # Save and load with env key
            safe_dump(test_data, test_file, config)
            loaded = safe_load(test_file, config)
            assert loaded == test_data

        finally:
            # Clean up environment
            del os.environ['GENESIS_HMAC_KEY']

    def test_compute_file_hmac(self, tmp_path):
        """Test HMAC computation for files."""
        test_file = tmp_path / "test_hmac.txt"
        test_file.write_bytes(b"test content for hmac")

        key = b"secret_key"
        hmac1 = compute_file_hmac(test_file, key)
        hmac2 = compute_file_hmac(test_file, key)

        # Same file and key should produce same HMAC
        assert hmac1 == hmac2

        # Different key should produce different HMAC
        hmac3 = compute_file_hmac(test_file, b"different_key")
        assert hmac1 != hmac3

    def test_verify_file_signature(self, tmp_path):
        """Test file signature verification."""
        test_file = tmp_path / "test_verify.txt"
        test_file.write_bytes(b"content to verify")

        key = b"verification_key"
        signature = compute_file_hmac(test_file, key)

        # Correct signature should verify
        assert verify_file_signature(test_file, signature, key)

        # Wrong signature should not verify
        assert not verify_file_signature(test_file, "wrong_signature", key)

        # Wrong key should not verify
        assert not verify_file_signature(test_file, signature, b"wrong_key")

    def test_torch_support(self):
        """Test adding PyTorch support to config."""
        config = SafeLoadConfig()

        # Initially torch not in whitelist
        assert 'torch' not in config.allowed_modules
        assert 'torch.Tensor' not in config.allowed_classes

        # Add torch support
        add_torch_support(config)

        # Now torch should be allowed
        assert 'torch' in config.allowed_modules
        assert 'torch.Tensor' in config.allowed_classes
        assert 'torch.nn' in config.allowed_modules

    def test_custom_whitelist(self, tmp_path):
        """Test using custom module/class whitelist."""
        # Very restrictive config - no numpy allowed
        config = SafeLoadConfig(
            allowed_modules={'builtins', '__builtin__'},
            allowed_classes={'dict', 'list', 'str', 'int'},
            verify_signature=False
        )

        # Only dict and list should be allowed
        test_file = tmp_path / "test_custom.pkl"

        # Dict should work
        with open(test_file, 'wb') as f:
            pickle.dump({'allowed': 'dict'}, f)
        loaded = safe_load(test_file, config)
        assert loaded == {'allowed': 'dict'}

        # Numpy array should fail (not in whitelist)
        import numpy as np
        with open(test_file, 'wb') as f:
            pickle.dump(np.array([1, 2, 3]), f)

        with pytest.raises(pickle.UnpicklingError, match="not in whitelist"):
            safe_load(test_file, config)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])