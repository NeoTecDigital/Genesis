"""
Test to verify that text is not stored in metadata (Phase 7 compliance).

Text storage in metadata violates the core principle that voxel space
should ONLY store quaternionic parameters. All semantic content must
be derived from frequency signals, not stored explicitly.
"""

import os
import sys
import pytest
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline.encoding import EncodingPipeline
from src.memory.memory_hierarchy import MemoryHierarchy
from src.memory.voxel_cloud import VoxelCloud
from src.memory.octave_hierarchy import OctaveProtoIdentity
from src.origin import Origin


def test_encoding_no_text_in_metadata():
    """Verify encoding doesn't add text to metadata."""
    # Create Origin and carrier
    origin = Origin(512, 512, use_gpu=False)
    carrier = origin.initialize_carrier()

    # Create encoder with carrier
    encoder = EncodingPipeline(carrier=carrier, width=512, height=512)

    # Encode text
    proto, metadata = encoder.encode_text("The Tao that can be spoken")

    # Verify no text in metadata
    assert 'text' not in metadata, "Text should not be stored in metadata"

    # Check only deterministic fields are present
    assert 'modality' in metadata
    assert 'source' in metadata
    assert 'timestamp' in metadata
    assert 'fundamental_freq' in metadata


def test_memory_hierarchy_no_text_storage():
    """Verify MemoryHierarchy doesn't store text."""
    hierarchy = MemoryHierarchy(width=64, height=64, depth=10)

    # Create Origin and carrier
    origin = Origin(64, 64, use_gpu=False)
    carrier = origin.initialize_carrier()

    # Create encoder with carrier
    encoder = EncodingPipeline(carrier=carrier, width=64, height=64)

    # Create and store a proto
    proto, metadata = encoder.encode_text("Essential oneness")
    freq_spectrum = np.random.rand(64, 64, 2)  # Proper frequency spectrum shape

    # Store in hierarchy
    hierarchy.store_core(proto, freq_spectrum, metadata)

    # Retrieve and verify no text
    entries = hierarchy.core_memory.entries
    assert len(entries) > 0

    for entry in entries:
        assert 'text' not in entry.metadata, "Text found in stored metadata"


def test_octave_proto_identity_no_text_field():
    """Verify OctaveProtoIdentity dataclass has no text field."""
    # Create an octave proto identity
    proto = OctaveProtoIdentity(
        proto_identity=np.random.rand(32, 32, 4),
        quaternions={0: np.array([1, 0, 0, 0])},
        frequency=100.0,
        modality='text',
        octave_level=0
    )

    # Verify no text attribute
    assert not hasattr(proto, 'text'), "OctaveProtoIdentity should not have text field"

    # Check only expected fields
    assert hasattr(proto, 'proto_identity')
    assert hasattr(proto, 'quaternions')
    assert hasattr(proto, 'frequency')
    assert hasattr(proto, 'modality')
    assert hasattr(proto, 'octave_level')


def test_voxel_cloud_add_with_octaves_no_text():
    """Verify add_with_octaves doesn't accept text parameter."""
    cloud = VoxelCloud(width=10, height=10, depth=10)
    proto = np.random.rand(32, 32, 4)

    # This should work without text parameter
    cloud.add_with_octaves(
        proto_identity=proto,
        frequency=100.0,
        modality='text',
        quaternions={0: np.array([1, 0, 0, 0])},
        resonance_strength=1.0
    )

    # Verify stored entry has no text
    assert len(cloud.octave_hierarchy.octave_storage[0]) > 0
    stored_proto = cloud.octave_hierarchy.octave_storage[0][0]
    assert not hasattr(stored_proto, 'text')


def test_grep_verification_no_text_storage():
    """Verify no text storage patterns in codebase."""
    import subprocess

    # Check for metadata['text'] = assignments (not just 'text' in general)
    result = subprocess.run(
        ['grep', '-rE', r"metadata\['text'\]\s*=", 'src/', '--include=*.py'],
        capture_output=True,
        text=True
    )

    assert result.returncode != 0 or not result.stdout, \
        f"Found text storage violations:\n{result.stdout}"

    # Check for text: str field in dataclasses
    # This looks for class-level field definitions, not function parameters
    result = subprocess.run(
        ['grep', '-rE', r'@dataclass', 'src/', '--include=*.py', '-A', '20'],
        capture_output=True,
        text=True
    )

    # Check if any dataclass has a 'text:' field
    violations = []
    lines = result.stdout.strip().split('\n')
    in_dataclass = False
    for i, line in enumerate(lines):
        if '@dataclass' in line:
            in_dataclass = True
        elif in_dataclass and line.strip().startswith('class '):
            # Check next 20 lines for text: field
            for j in range(i+1, min(i+20, len(lines))):
                if lines[j].strip().startswith('text:'):
                    violations.append(f"{line.split(':')[0]}:{lines[j]}")
            in_dataclass = False

    assert not violations, f"Found text field in dataclasses:\n{chr(10).join(violations)}"