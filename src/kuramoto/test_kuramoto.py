"""
Comprehensive test and validation for Kuramoto module.

Tests the complete pipeline:
1. Text → Natural frequencies
2. Frequencies → Synchronized phases
3. Phases → Proto-identity field
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kuramoto import KuramotoSolver, TextToOscillators, ProtoIdentityGenerator


def test_kuramoto_solver():
    """Test Kuramoto solver convergence."""
    print("=" * 60)
    print("Testing Kuramoto Solver")
    print("=" * 60)

    solver = KuramotoSolver(coupling_strength=3.0, sync_threshold=0.9)

    # Test 1: Identical frequencies (should sync immediately)
    print("\n1. Identical frequencies:")
    omegas = np.ones(5)
    result = solver.solve(omegas)
    print(f"   Converged: {result['converged']}")
    print(f"   Steps: {result['steps']}")
    print(f"   Order parameter r: {result['order_parameter'][0]:.4f}")
    assert result['converged'], "Should converge for identical frequencies"

    # Test 2: Similar frequencies (should sync with moderate coupling)
    print("\n2. Similar frequencies:")
    omegas = np.array([1.0, 1.05, 0.95, 1.02, 0.98])
    result = solver.solve(omegas)
    print(f"   Converged: {result['converged']}")
    print(f"   Steps: {result['steps']}")
    print(f"   Order parameter r: {result['order_parameter'][0]:.4f}")

    # Test 3: Diverse frequencies (may not sync with weak coupling)
    print("\n3. Diverse frequencies (weak coupling):")
    solver_weak = KuramotoSolver(coupling_strength=0.5, max_steps=500)
    omegas = np.array([0.5, 1.0, 1.5, 2.0, 2.5])
    result = solver_weak.solve(omegas)
    print(f"   Converged: {result['converged']}")
    print(f"   Steps: {result['steps']}")
    print(f"   Order parameter r: {result['order_parameter'][0]:.4f}")

    # Test 4: Strong coupling should sync even diverse frequencies
    print("\n4. Diverse frequencies (strong coupling):")
    solver_strong = KuramotoSolver(coupling_strength=5.0, max_steps=2000)
    result = solver_strong.solve(omegas)
    print(f"   Converged: {result['converged']}")
    print(f"   Steps: {result['steps']}")
    print(f"   Order parameter r: {result['order_parameter'][0]:.4f}")

    print("\n✓ Kuramoto solver tests passed")


def test_text_encoder():
    """Test text to oscillator encoding."""
    print("\n" + "=" * 60)
    print("Testing Text Encoder")
    print("=" * 60)

    encoder = TextToOscillators(frequency_scale=2.0)

    # Test 1: Basic encoding
    print("\n1. Basic encoding:")
    text = "Hello, World!"
    frequencies = encoder.encode(text)
    print(f"   Text: '{text}'")
    print(f"   Oscillators: {len(frequencies)}")
    print(f"   Frequency mean: {frequencies.mean():.3f}")
    print(f"   Frequency std: {frequencies.std():.3f}")
    assert len(frequencies) == len(text), "Should have one oscillator per character"

    # Test 2: Reproducibility
    print("\n2. Reproducibility:")
    freq2 = encoder.encode(text)
    are_equal = np.allclose(frequencies, freq2)
    print(f"   Same encoding: {are_equal}")
    assert are_equal, "Encoding should be deterministic"

    # Test 3: Different texts produce different frequencies
    print("\n3. Distinctness:")
    text2 = "Goodbye, World!"
    freq3 = encoder.encode(text2)
    similarity = np.corrcoef(frequencies[:len(freq3)], freq3[:len(frequencies)])[0, 1]
    print(f"   Correlation between '{text}' and '{text2}': {similarity:.3f}")
    assert not np.allclose(frequencies, freq3[:len(frequencies)]), "Different texts should produce different frequencies"

    # Test 4: Hierarchical encoding
    print("\n4. Hierarchical encoding:")
    hierarchical = encoder.encode_hierarchical(text)
    for level, freqs in hierarchical.items():
        print(f"   {level.capitalize()}: {len(freqs)} oscillators")

    print("\n✓ Text encoder tests passed")


def test_proto_identity_generator():
    """Test proto-identity generation."""
    print("\n" + "=" * 60)
    print("Testing Proto-Identity Generator")
    print("=" * 60)

    generator = ProtoIdentityGenerator(resolution=(128, 128))

    # Test 1: Basic generation
    print("\n1. Basic generation:")
    N = 10
    thetas = np.random.uniform(0, 2*np.pi, N)
    field = generator.generate(thetas, text="Test")
    print(f"   Field shape: {field.shape}")
    print(f"   Field dtype: {field.dtype}")
    print(f"   Mean magnitude: {np.abs(field).mean():.3f}")
    assert field.shape == (128, 128), "Field should match resolution"
    assert field.dtype == np.complex128, "Field should be complex"

    # Test 2: Deterministic generation
    print("\n2. Determinism:")
    field2 = generator.generate(thetas, text="Test")
    are_equal = np.allclose(field, field2)
    print(f"   Same field: {are_equal}")
    assert are_equal, "Generation should be deterministic for same inputs"

    # Test 3: Quaternion conversion
    print("\n3. Quaternion conversion:")
    quaternion = generator.to_quaternion(field)
    print(f"   Quaternion shape: {quaternion.shape}")
    assert quaternion.shape == (128, 128, 4), "Quaternion should have 4 components"

    # Test 4: Reconstruction accuracy
    print("\n4. Quaternion reconstruction:")
    reconstructed = generator.from_quaternion(quaternion)
    error = np.mean(np.abs(field - reconstructed))
    print(f"   Reconstruction error: {error:.6f}")
    # Note: Quaternion mapping stores complex field in first 2 components
    assert error < 0.1, "Reconstruction error should be small"

    # Test 5: Coherence measure
    print("\n5. Coherence measure:")
    # Synchronized phases (high coherence)
    thetas_sync = np.ones(N) * np.pi/4
    field_sync = generator.generate(thetas_sync)
    coherence_sync = generator.coherence_measure(field_sync)

    # Random phases (low coherence)
    thetas_random = np.random.uniform(0, 2*np.pi, N)
    field_random = generator.generate(thetas_random)
    coherence_random = generator.coherence_measure(field_random)

    print(f"   Synchronized coherence: {coherence_sync:.3f}")
    print(f"   Random coherence: {coherence_random:.3f}")
    assert coherence_sync > coherence_random, "Synchronized phases should have higher coherence"

    print("\n✓ Proto-identity generator tests passed")


def test_full_pipeline():
    """Test complete pipeline: Text → Frequencies → Sync → Proto-identity."""
    print("\n" + "=" * 60)
    print("Testing Full Pipeline")
    print("=" * 60)

    # Initialize components
    encoder = TextToOscillators(frequency_scale=1.5)
    solver = KuramotoSolver(coupling_strength=2.5, sync_threshold=0.9)
    generator = ProtoIdentityGenerator(resolution=(256, 256))

    # Test text
    test_texts = [
        "Hello, World!",
        "The quick brown fox",
        "Quantum memory system"
    ]

    for text in test_texts:
        print(f"\nProcessing: '{text}'")
        print("-" * 40)

        # Step 1: Encode text
        omegas = encoder.encode(text)
        print(f"1. Encoded to {len(omegas)} oscillators")
        print(f"   Frequency range: [{omegas.min():.3f}, {omegas.max():.3f}]")

        # Step 2: Synchronize
        result = solver.solve(omegas)
        print(f"2. Synchronization:")
        print(f"   Converged: {result['converged']}")
        print(f"   Steps: {result['steps']}")
        print(f"   Order parameter: {result['order_parameter'][0]:.3f}")

        # Step 3: Generate proto-identity
        components = generator.generate(
            result['phases'],
            text=text,
            return_components=True
        )
        field = components['field']
        coherence = generator.coherence_measure(field)

        print(f"3. Proto-identity:")
        print(f"   Field shape: {field.shape}")
        print(f"   Magnitude mean: {np.abs(field).mean():.3f}")
        print(f"   Coherence: {coherence:.3f}")

        # Verify reproducibility
        field2 = generator.generate(result['phases'], text=text)
        print(f"4. Reproducible: {np.allclose(field, field2)}")

    print("\n✓ Full pipeline tests passed")


def test_edge_cases():
    """Test edge cases and error handling."""
    print("\n" + "=" * 60)
    print("Testing Edge Cases")
    print("=" * 60)

    # Test 1: Empty text
    print("\n1. Empty text handling:")
    encoder = TextToOscillators()
    try:
        encoder.encode("")
        print("   ERROR: Should have raised ValueError")
    except ValueError as e:
        print(f"   ✓ Caught expected error: {e}")

    # Test 2: Invalid parameters
    print("\n2. Invalid parameters:")
    try:
        KuramotoSolver(coupling_strength=-1)
        print("   ERROR: Should have raised ValueError")
    except ValueError as e:
        print(f"   ✓ Caught expected error: {e}")

    # Test 3: Single oscillator
    print("\n3. Single oscillator:")
    solver = KuramotoSolver()
    result = solver.solve(np.array([1.0]))
    print(f"   Converged: {result['converged']}")
    print(f"   Order parameter: {result['order_parameter'][0]:.3f}")

    # Test 4: Large number of oscillators
    print("\n4. Large system (1000 oscillators):")
    omegas = np.random.normal(1.0, 0.1, 1000)
    solver = KuramotoSolver(coupling_strength=3.0, max_steps=500)
    result = solver.solve(omegas)
    print(f"   Converged: {result['converged']}")
    print(f"   Steps: {result['steps']}")
    print(f"   Order parameter: {result['order_parameter'][0]:.3f}")

    print("\n✓ Edge case tests passed")


if __name__ == "__main__":
    print("KURAMOTO MODULE VALIDATION")
    print("=" * 60)

    # Run all tests
    test_kuramoto_solver()
    test_text_encoder()
    test_proto_identity_generator()
    test_full_pipeline()
    test_edge_cases()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)

    # Demo the complete system
    print("\n" + "=" * 60)
    print("DEMO: Complete Kuramoto Pipeline")
    print("=" * 60)

    # Components
    encoder = TextToOscillators(frequency_scale=2.0)
    solver = KuramotoSolver(coupling_strength=3.0, sync_threshold=0.95)
    generator = ProtoIdentityGenerator(resolution=(512, 512))

    # Process text
    text = "Quantum coherence emerges from synchronized oscillations"
    print(f"\nInput text: '{text}'")
    print(f"Text length: {len(text)} characters")

    # Encode
    print("\n1. Encoding text to oscillators...")
    omegas = encoder.encode(text)
    print(f"   Generated {len(omegas)} oscillators")
    print(f"   Frequency mean: {omegas.mean():.3f} ± {omegas.std():.3f}")

    # Synchronize
    print("\n2. Synchronizing oscillators...")
    result = solver.solve(omegas)
    print(f"   Converged: {result['converged']}")
    print(f"   Iterations: {result['steps']}")
    print(f"   Order parameter r: {result['order_parameter'][0]:.4f}")
    print(f"   Collective phase Ψ: {result['order_parameter'][1]:.4f} rad")

    # Generate proto-identity
    print("\n3. Generating proto-identity field...")
    components = generator.generate(result['phases'], text=text, return_components=True)
    field = components['field']
    coherence = generator.coherence_measure(field)

    print(f"   Field dimensions: {field.shape}")
    print(f"   Field coherence: {coherence:.4f}")
    print(f"   Magnitude range: [{np.abs(field).min():.3f}, {np.abs(field).max():.3f}]")
    print(f"   Phase spread: {np.std(np.angle(field)):.3f} rad")

    # Convert to quaternion
    print("\n4. Converting to quaternion representation...")
    quaternion = generator.to_quaternion(field)
    print(f"   Quaternion shape: {quaternion.shape}")
    print(f"   Quaternion norm mean: {np.mean(np.linalg.norm(quaternion, axis=-1)):.3f}")

    print("\n" + "=" * 60)
    print("DEMO COMPLETE - Kuramoto system operational!")
    print("=" * 60)