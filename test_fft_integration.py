#!/usr/bin/env python3
"""Test FFT integration with MultiOctave pipeline."""

import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline.multi_octave_encoder import MultiOctaveEncoder
from src.pipeline.multi_octave_decoder import MultiOctaveDecoder
from src.memory.voxel_cloud import VoxelCloud


def test_basic_integration():
    """Test basic FFT integration with multi-octave encoding/decoding."""

    print("Testing FFT Integration with MultiOctave Pipeline")
    print("=" * 50)

    # Initialize components
    carrier = np.zeros((512, 512, 4), dtype=np.float32)
    encoder = MultiOctaveEncoder(carrier, width=512, height=512)
    decoder = MultiOctaveDecoder(carrier)
    voxel_cloud = VoxelCloud(width=512, height=512, depth=512)

    # Test text
    test_text = "Hello FFT!"
    print(f"\nOriginal text: '{test_text}'")

    # Encode at character level
    octave_units = encoder.encode_text_hierarchical(test_text, octaves=[4])
    print(f"Encoded {len(octave_units)} units at octave 4 (character level)")

    # Verify no text in metadata
    for unit in octave_units:
        assert not hasattr(unit, 'text'), "OctaveUnit should not have 'text' attribute"
        print(f"  Unit at octave {unit.octave}: proto shape {unit.proto_identity.shape}")

    # Store in voxel cloud
    for unit in octave_units:
        metadata = {
            'octave': unit.octave,
            'modality': 'text',
            'encoder': 'fft'
        }
        voxel_cloud.add(unit.proto_identity, unit.frequency, metadata)

    print(f"\nStored {len(octave_units)} units in VoxelCloud")

    # Query and decode
    query_proto = octave_units[0].proto_identity  # Use first character as query
    results = voxel_cloud.query_by_proto_similarity(query_proto, max_results=10)

    print(f"\nQuery returned {len(results)} results")

    # Try to decode using FFT decoder directly
    if results:
        entry = results[0]
        decoded_text = decoder.fft_decoder.decode_text(entry.proto_identity)
        print(f"Decoded first result: '{decoded_text}'")

    # Test decoding from memory
    reconstructed = decoder.decode_from_memory(query_proto, voxel_cloud)
    print(f"Reconstructed from memory: '{reconstructed}'")

    print("\n✅ FFT Integration Test Passed!")


if __name__ == "__main__":
    test_basic_integration()