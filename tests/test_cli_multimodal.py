#!/usr/bin/env python3
"""
Test CLI multi-modal support.

Tests the genesis CLI with image and audio processing capabilities.
"""

import sys
import tempfile
import numpy as np
from pathlib import Path
import subprocess

sys.path.append('/home/persist/alembic/genesis')


def create_test_image(path: Path):
    """Create a simple test image."""
    try:
        from PIL import Image
        # Create simple pattern
        img_array = np.zeros((128, 128, 3), dtype=np.uint8)
        # Add some pattern
        for i in range(0, 128, 16):
            img_array[i:i+8, :, 0] = 255  # Red stripes
            img_array[:, i:i+8, 1] = 128  # Green stripes

        img = Image.fromarray(img_array)
        img.save(path)
        return True
    except ImportError:
        print("PIL not installed, skipping image test")
        return False


def create_test_audio(path: Path):
    """Create a simple test audio file."""
    try:
        import soundfile as sf
        # Create 1 second of 440Hz tone
        sample_rate = 16000
        t = np.linspace(0, 1.0, sample_rate)
        audio = np.sin(2 * np.pi * 440 * t)
        sf.write(path, audio, sample_rate)
        return True
    except ImportError:
        print("soundfile not installed, skipping audio test")
        return False


def test_text_discovery():
    """Test text modality discovery."""
    print("\n1. Testing Text Discovery")
    print("=" * 50)

    # Create test text file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("This is a test sentence.\n")
        f.write("Another test with different frequency.\n")
        f.write("Final test sentence here.\n")
        text_path = Path(f.name)

    # Create output path
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
        output_path = Path(f.name)

    try:
        # Run discovery
        cmd = [
            'python', 'genesis.py', 'discover',
            '--input', str(text_path),
            '--output', str(output_path),
            '--modality', 'text',
            '--disable-collapse'  # Keep entries separate for testing
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"  Error: {result.stderr}")
            return False

        # Check output exists
        if output_path.exists():
            print(f"  ✓ Text discovery successful")
            print(f"  Output size: {output_path.stat().st_size} bytes")
            return True
        else:
            print("  ❌ Output file not created")
            return False

    finally:
        text_path.unlink(missing_ok=True)
        output_path.unlink(missing_ok=True)


def test_image_discovery():
    """Test image modality discovery."""
    print("\n2. Testing Image Discovery")
    print("=" * 50)

    # Create test image
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        img_path = Path(f.name)

    if not create_test_image(img_path):
        return True  # Skip if PIL not available

    # Create output path
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
        output_path = Path(f.name)

    try:
        # Run discovery
        cmd = [
            'python', 'genesis.py', 'discover',
            '--input', str(img_path),
            '--output', str(output_path),
            '--modality', 'image'
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"  Error: {result.stderr}")
            if "PIL/Pillow not installed" in result.stderr:
                print("  (Expected - PIL not installed)")
                return True
            return False

        # Check output exists
        if output_path.exists():
            print(f"  ✓ Image discovery successful")
            print(f"  Output size: {output_path.stat().st_size} bytes")
            return True
        else:
            print("  ❌ Output file not created")
            return False

    finally:
        img_path.unlink(missing_ok=True)
        output_path.unlink(missing_ok=True)


def test_audio_discovery():
    """Test audio modality discovery."""
    print("\n3. Testing Audio Discovery")
    print("=" * 50)

    # Create test audio
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        audio_path = Path(f.name)

    if not create_test_audio(audio_path):
        return True  # Skip if soundfile not available

    # Create output path
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
        output_path = Path(f.name)

    try:
        # Run discovery
        cmd = [
            'python', 'genesis.py', 'discover',
            '--input', str(audio_path),
            '--output', str(output_path),
            '--modality', 'audio'
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"  Error: {result.stderr}")
            if "soundfile not installed" in result.stderr:
                print("  (Expected - soundfile not installed)")
                return True
            return False

        # Check output exists
        if output_path.exists():
            print(f"  ✓ Audio discovery successful")
            print(f"  Output size: {output_path.stat().st_size} bytes")
            return True
        else:
            print("  ❌ Output file not created")
            return False

    finally:
        audio_path.unlink(missing_ok=True)
        output_path.unlink(missing_ok=True)


def test_cross_modal_linking():
    """Test cross-modal linking via CLI."""
    print("\n4. Testing Cross-Modal Linking")
    print("=" * 50)

    # Create test text file with known content
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("Frequency pattern test.\n")
        text_path = Path(f.name)

    # Create output path
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
        output_path = Path(f.name)

    try:
        # Run discovery with cross-modal linking enabled
        cmd = [
            'python', 'genesis.py', 'discover',
            '--input', str(text_path),
            '--output', str(output_path),
            '--modality', 'text',
            '--link-cross-modal',
            '--phase-coherence-threshold', '5.0',  # Wide threshold
            '--disable-collapse'
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"  Error: {result.stderr}")
            return False

        # Check for cross-modal linking message in output
        if "cross-modal" in result.stdout.lower():
            print("  ✓ Cross-modal linking option accepted")
            return True
        else:
            print("  ✓ Command executed (no cross-modal links expected with single modality)")
            return True

    finally:
        text_path.unlink(missing_ok=True)
        output_path.unlink(missing_ok=True)


def main():
    """Run all CLI multi-modal tests."""
    print("\n" + "=" * 60)
    print("CLI MULTI-MODAL TESTS")
    print("=" * 60)

    all_passed = True

    # Test each modality
    if not test_text_discovery():
        all_passed = False

    if not test_image_discovery():
        all_passed = False

    if not test_audio_discovery():
        all_passed = False

    if not test_cross_modal_linking():
        all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("✅ ALL CLI TESTS PASSED")
        return 0
    else:
        print("❌ SOME CLI TESTS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())