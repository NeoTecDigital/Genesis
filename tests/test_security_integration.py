#!/usr/bin/env python3
"""Integration tests for security hardening."""

import os
import sys
import tempfile
import pickle
import pytest
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.security import (
    safe_load_pickle,
    safe_save_pickle,
    migrate_pickle_file
)
from src.memory.voxel_cloud import VoxelCloud, ProtoIdentityEntry
import numpy as np


class TestSecurePickling:
    """Test secure pickling functionality."""

    def test_save_and_load_voxel_cloud(self, tmp_path):
        """Test saving and loading VoxelCloud with security."""
        # Create a VoxelCloud with some data
        cloud = VoxelCloud(width=128, height=128, depth=32)

        # Add a test entry
        entry = ProtoIdentityEntry(
            proto_identity=np.random.randn(128, 128, 4).astype(np.float32),
            mip_levels=[],
            frequency=np.random.randn(128, 128, 2).astype(np.float32),
            metadata={'text': 'test'},
            position=np.array([0.5, 0.5, 0.5]),
            fundamental_freq=440.0,
            octave=0
        )
        cloud.entries.append(entry)

        # Save with security
        test_file = tmp_path / "test_cloud.pkl"
        cloud.save(str(test_file))

        # Verify signature file was created
        sig_file = test_file.with_suffix('.pkl.sig')
        assert sig_file.exists(), "Signature file should be created"

        # Load with security
        cloud2 = VoxelCloud(width=128, height=128, depth=32)
        cloud2.load(str(test_file))

        # Verify data integrity
        assert len(cloud2.entries) == 1
        assert cloud2.entries[0].fundamental_freq == 440.0
        assert cloud2.entries[0].metadata['text'] == 'test'

    def test_backward_compatibility(self, tmp_path):
        """Test loading old pickle files without signatures."""
        # Create test data
        test_data = {
            'width': 128,
            'height': 128,
            'depth': 32,
            'entries': [],
            'spatial_index': {},
            'frequency_index': {}
        }

        # Save using old method (no signature)
        test_file = tmp_path / "old_format.pkl"
        with open(test_file, 'wb') as f:
            pickle.dump(test_data, f)

        # Should load with warning but succeed
        loaded = safe_load_pickle(test_file, backward_compatible=True)
        assert loaded['width'] == 128

    def test_migration(self, tmp_path):
        """Test migrating old pickle to new format."""
        # Create old format file
        test_data = {'test': 'data', 'value': 42}
        test_file = tmp_path / "migrate_me.pkl"

        with open(test_file, 'wb') as f:
            pickle.dump(test_data, f)

        # Migrate
        assert migrate_pickle_file(test_file)

        # Verify signature exists
        sig_file = test_file.with_suffix('.pkl.sig')
        assert sig_file.exists()

        # Load and verify
        loaded = safe_load_pickle(test_file)
        assert loaded['test'] == 'data'
        assert loaded['value'] == 42

    def test_tampering_detection(self, tmp_path):
        """Test that tampering is detected."""
        # Save data with signature
        test_data = {'secure': 'data'}
        test_file = tmp_path / "secure.pkl"
        safe_save_pickle(test_data, test_file)

        # Tamper with the file
        with open(test_file, 'rb') as f:
            content = f.read()

        # Modify content
        tampered = content[:-10] + b'TAMPERED!!'
        with open(test_file, 'wb') as f:
            f.write(tampered)

        # Should fail to load due to signature mismatch
        with pytest.raises(ValueError, match="signature verification failed"):
            safe_load_pickle(test_file, verify_signature=True)

    def test_malicious_class_rejection(self, tmp_path):
        """Test that dangerous classes are rejected."""
        # Try to pickle an os.system call (should be rejected on load)
        import subprocess

        class MaliciousClass:
            def __reduce__(self):
                return (subprocess.call, (['echo', 'pwned'],))

        test_file = tmp_path / "malicious.pkl"

        # Save using regular pickle (simulating attacker)
        with open(test_file, 'wb') as f:
            pickle.dump(MaliciousClass(), f)

        # Should fail to load due to class restrictions
        with pytest.raises(pickle.UnpicklingError, match="not in whitelist"):
            safe_load_pickle(test_file, verify_signature=False)


class TestInputValidation:
    """Test input validation for CLI arguments."""

    def test_path_traversal_prevention(self):
        """Test that path traversal attempts are blocked."""
        from src.security import sanitize_file_path

        # Test various traversal attempts
        dangerous_paths = [
            "../../../etc/passwd",
            "..\\..\\windows\\system32\\config\\sam",
            "/tmp/../../../etc/shadow",
            "data/../../../../../../etc/hosts"
        ]

        for path in dangerous_paths:
            result = sanitize_file_path(path)
            assert not result.is_valid or "traversal" in str(result.error_message).lower()

    def test_safe_paths(self):
        """Test that legitimate paths work."""
        from src.security import sanitize_file_path

        safe_paths = [
            "models/genesis.pkl",
            "./data/test.txt",
            "/tmp/output.pkl"
        ]

        for path in safe_paths:
            result = sanitize_file_path(path)
            assert result.is_valid, f"Safe path {path} should be valid"

    def test_numeric_bounds(self):
        """Test numeric range validation."""
        from src.security import NumericRange

        # Test float range
        float_range = NumericRange(min_value=0.0, max_value=1.0)
        assert float_range.validate(0.5)[0]
        assert not float_range.validate(1.5)[0]
        assert not float_range.validate(-0.1)[0]

        # Test integer range
        int_range = NumericRange(
            min_value=1,
            max_value=100,
            allow_zero=False,
            allow_negative=False
        )
        assert int_range.validate(50)[0]
        assert not int_range.validate(0)[0]
        assert not int_range.validate(-10)[0]
        assert not int_range.validate(101)[0]


def run_tests():
    """Run all security integration tests."""
    # Set up test environment
    os.environ['GENESIS_HMAC_KEY'] = 'test-key-for-testing'

    # Run pytest
    pytest.main([__file__, '-v'])


if __name__ == '__main__':
    run_tests()