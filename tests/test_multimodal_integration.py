#!/usr/bin/env python3
"""
Test multimodal encoding and dataset loading integration.

Tests:
1. Image encoding with real dataset images
2. Audio encoding with real dataset audio
3. Arrow dataset loading
4. Multimodal dataset batch processing
"""

import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from src.data.dataset_loader import UniversalDatasetLoader
from src.pipeline.encoding import EncodingPipeline

print("=" * 80)
print("MULTIMODAL INTEGRATION TEST")
print("=" * 80)
print()

# Initialize components
loader = UniversalDatasetLoader()
carrier = np.random.randn(512, 512, 4).astype(np.float32)  # Temporary carrier
pipeline = EncodingPipeline(carrier)

tests_passed = 0
tests_failed = 0

# Test 1: Dataset Loading
print("1. DATASET LOADING")
print("-" * 60)

# Test text file
text_path = Path("/usr/lib/alembic/data/datasets/text/curated/foundation/tractatus_logico-wittgenstein.txt")
if text_path.exists():
    try:
        sample = loader.load(text_path)
        print(f"  ✓ Loaded text: {len(sample.data)} chars")
        tests_passed += 1
    except Exception as e:
        print(f"  ✗ Text loading failed: {e}")
        tests_failed += 1
else:
    print(f"  ⚠  Text file not found: {text_path}")
    tests_failed += 1

# Test image file
image_path = Path("/usr/lib/alembic/data/datasets/images/be_here_now/page-006.jpg")
if image_path.exists():
    try:
        sample = loader.load(image_path)
        print(f"  ✓ Loaded image: shape {sample.data.shape}")
        tests_passed += 1
    except Exception as e:
        print(f"  ✗ Image loading failed: {e}")
        tests_failed += 1
else:
    print(f"  ⚠  Image file not found: {image_path}")
    tests_failed += 1

# Test audio file
audio_path = Path("/usr/lib/alembic/data/datasets/multimodal/PianoVAM/PianoVAM_v1.0/Audio/2024-02-15_21-40-43.wav")
if audio_path.exists():
    try:
        sample = loader.load(audio_path)
        print(f"  ✓ Loaded audio: {sample.metadata['duration_sec']:.1f}s at {sample.metadata['sample_rate']}Hz")
        tests_passed += 1
    except Exception as e:
        print(f"  ✗ Audio loading failed: {e}")
        tests_failed += 1
else:
    print(f"  ⚠  Audio file not found: {audio_path}")
    tests_failed += 1

print()

# Test 2: Arrow Dataset Loading
print("2. ARROW DATASET LOADING")
print("-" * 60)

arrow_dir = Path("/usr/lib/alembic/data/datasets/text/medical/reasonmed")
if arrow_dir.exists():
    try:
        # Try loading as directory
        samples = list(loader.load_arrow_dataset(arrow_dir, limit=5))
        if samples:
            print(f"  ✓ Loaded Arrow dataset: {len(samples)} samples")
            print(f"  Sample schema: {samples[0].metadata.get('schema', 'N/A')}")
            tests_passed += 1
        else:
            print(f"  ✗ Arrow dataset returned no samples")
            tests_failed += 1
    except Exception as e:
        print(f"  ✗ Arrow loading failed: {e}")
        tests_failed += 1
else:
    print(f"  ⚠  Arrow directory not found: {arrow_dir}")
    tests_failed += 1

print()

# Test 3: Image Encoding
print("3. IMAGE ENCODING")
print("-" * 60)

if image_path.exists():
    try:
        proto, metadata = pipeline.encode_image(str(image_path))
        print(f"  ✓ Encoded image → proto shape: {proto.shape}")
        print(f"  Modality: {metadata['modality']}")
        print(f"  Carrier filtered: {metadata.get('carrier_filtered', False)}")
        if proto.shape == (512, 512, 4):
            tests_passed += 1
        else:
            print(f"  ✗ Proto shape incorrect: {proto.shape} != (512, 512, 4)")
            tests_failed += 1
    except Exception as e:
        print(f"  ✗ Image encoding failed: {e}")
        tests_failed += 1
else:
    print(f"  ⚠  Skipping - image file not found")
    tests_failed += 1

print()

# Test 4: Audio Encoding
print("4. AUDIO ENCODING")
print("-" * 60)

if audio_path.exists():
    try:
        proto, metadata = pipeline.encode_audio(str(audio_path))
        print(f"  ✓ Encoded audio → proto shape: {proto.shape}")
        print(f"  Modality: {metadata['modality']}")
        print(f"  Carrier filtered: {metadata.get('carrier_filtered', False)}")
        if proto.shape == (512, 512, 4):
            tests_passed += 1
        else:
            print(f"  ✗ Proto shape incorrect: {proto.shape} != (512, 512, 4)")
            tests_failed += 1
    except Exception as e:
        print(f"  ✗ Audio encoding failed: {e}")
        tests_failed += 1
else:
    print(f"  ⚠  Skipping - audio file not found")
    tests_failed += 1

print()

# Summary
print("=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"Tests Passed: {tests_passed}")
print(f"Tests Failed: {tests_failed}")
print(f"Pass Rate: {tests_passed / (tests_passed + tests_failed) * 100:.1f}%")
print()

if tests_failed == 0:
    print("✅ ALL MULTIMODAL TESTS PASSED")
    print("   Dataset loading works for text, images, audio")
    print("   Image and audio encoding functional")
    print("   Arrow dataset loading working")
elif tests_passed >= 6:
    print("⚠️  PARTIAL SUCCESS")
    print("   Core functionality working")
    print("   Some tests failed - check logs above")
else:
    print("❌ TESTS FAILED")
    print("   Multiple components need fixing")

print()
print("=" * 80)
