"""Example: Text → Kuramoto → Spinor UFT → Analysis."""

import sys
import numpy as np
sys.path.append('/home/persist/alembic/genesis')
from src.kuramoto.kuramoto_encoder import KuramotoEncoder
from src.uft.evolver_factory import create_evolver
from src.uft.mass_gap import MassGapCalculator
from src.uft.chirality import ChiralityAnalyzer


def main():
    """Demonstrate complete pipeline with spinor evolver."""
    print("Spinor UFT Integration Example")
    print("=" * 60)

    # Initialize components
    print("\n1. Initializing components...")
    encoder = KuramotoEncoder(coupling_strength=3.0)
    evolver_scalar = create_evolver(mode="scalar", evolution_time=1.0)
    evolver_spinor = create_evolver(mode="spinor", evolution_time=1.0, chiral_damping=0.5)

    # Test texts with different characteristics
    texts = [
        "Because the sun rises in the east, morning light appears first on eastern horizons",
        "The quantum field fluctuates continuously in spacetime",
        "מְלָכִים וּנְבִיאִים",  # Hebrew (right-to-left)
        "古池や蛙飛び込む水の音",  # Japanese haiku
    ]

    for i, text in enumerate(texts, 1):
        print(f"\n{i}. Processing text {i}:")
        print(f"   Text: {text[:50]}..." if len(text) > 50 else f"   Text: {text}")

        # Kuramoto encoding
        proto_kuramoto = encoder.encode(text)
        r, phase = proto_kuramoto.metadata['order_parameter']
        print(f"\n   Kuramoto encoding:")
        print(f"   - Synchronization: r={r:.3f}")
        print(f"   - Collective phase: φ={phase:.3f}")

        # Scalar UFT evolution
        proto_scalar = evolver_scalar.evolve(proto_kuramoto)
        print(f"\n   Scalar UFT evolution:")
        print(f"   - Mass gap: Δ={proto_scalar.metadata['mass_gap']:.3f}")
        print(f"   - Chiral phase: δ={proto_scalar.metadata['chiral_phase']:.3f}")
        print(f"   - Converged: {proto_scalar.metadata.get('converged', False)}")
        print(f"   - Field shape: {proto_scalar.field.shape}")

        # Spinor UFT evolution
        proto_spinor = evolver_spinor.evolve(proto_kuramoto)
        print(f"\n   Spinor UFT evolution:")
        print(f"   - Mass gap: Δ={proto_spinor.metadata['mass_gap']:.3f}")
        print(f"   - Chiral phase: δ={proto_spinor.metadata['chiral_phase']:.3f}")
        print(f"   - Converged: {proto_spinor.metadata['converged']}")
        print(f"   - Steps taken: {proto_spinor.metadata['steps_taken']}")
        print(f"   - Field shape: {proto_spinor.field.shape}")
        print(f"   - Component norms: {[f'{n:.2f}' for n in proto_spinor.metadata['component_norms']]}")
        print(f"   - Norm growth: {proto_spinor.metadata['norm_growth']:.2f}x")

        # Compare scalar vs spinor
        print(f"\n   Comparison:")
        # Extract dominant spinor component for comparison
        spinor_dominant = proto_spinor.field[:, :, 0]
        scalar_field = proto_scalar.field

        # Compute correlation
        correlation = np.abs(np.vdot(
            scalar_field.flatten() / np.linalg.norm(scalar_field.flatten()),
            spinor_dominant.flatten() / np.linalg.norm(spinor_dominant.flatten())
        ))
        print(f"   - Scalar-Spinor correlation: {correlation:.3f}")

        # Analyze chiral asymmetry in spinor
        left_norm = np.linalg.norm(proto_spinor.field[:, :, :2])  # Upper components
        right_norm = np.linalg.norm(proto_spinor.field[:, :, 2:])  # Lower components
        chiral_asymmetry = (left_norm - right_norm) / (left_norm + right_norm)
        print(f"   - Chiral asymmetry: {chiral_asymmetry:.3f}")

        print("-" * 60)

    # Demonstrate adaptive evolution
    print("\n" + "=" * 60)
    print("Adaptive Evolution Demonstration")
    print("=" * 60)

    # Create test cases with different synchronization levels
    test_cases = [
        ("Highly synchronized", 0.9),
        ("Moderately synchronized", 0.5),
        ("Weakly synchronized", 0.2)
    ]

    for case_name, r_value in test_cases:
        print(f"\n{case_name} (r={r_value}):")

        # Create proto with specific synchronization
        from src.core.proto_identity import ProtoIdentity
        test_field = np.random.randn(512, 512).astype(np.complex128) * 0.1
        test_proto = ProtoIdentity(
            field=test_field,
            metadata={
                'order_parameter': (r_value, 0.0),
                'coupling': 3.0,
                'text': f'test case {case_name}'
            }
        )

        # Evolve with spinor
        spinor_result = evolver_spinor.evolve(test_proto)

        # Report results
        print(f"  - Mass gap: Δ={spinor_result.metadata['mass_gap']:.3f}")
        print(f"  - Converged: {spinor_result.metadata['converged']}")
        print(f"  - Steps: {spinor_result.metadata['steps_taken']}")
        print(f"  - Norm growth: {spinor_result.metadata['norm_growth']:.2f}x")

    # Performance comparison
    print("\n" + "=" * 60)
    print("Performance Comparison")
    print("=" * 60)

    import time

    test_text = "Performance test text for timing measurements"
    proto_test = encoder.encode(test_text)

    # Time scalar evolution
    start = time.time()
    _ = evolver_scalar.evolve(proto_test)
    scalar_time = time.time() - start

    # Time spinor evolution
    start = time.time()
    _ = evolver_spinor.evolve(proto_test)
    spinor_time = time.time() - start

    print(f"\nEvolution times (100 steps):")
    print(f"  Scalar UFT: {scalar_time:.3f} seconds")
    print(f"  Spinor UFT: {spinor_time:.3f} seconds")
    print(f"  Slowdown factor: {spinor_time/scalar_time:.2f}x")

    # Memory usage estimate
    scalar_memory = 512 * 512 * 16  # complex128
    spinor_memory = 512 * 512 * 4 * 16  # 4 components
    print(f"\nMemory usage:")
    print(f"  Scalar field: {scalar_memory / 1e6:.1f} MB")
    print(f"  Spinor field: {spinor_memory / 1e6:.1f} MB")
    print(f"  Memory factor: {spinor_memory/scalar_memory:.1f}x")

    print("\n" + "=" * 60)
    print("✅ Integration example completed successfully!")


if __name__ == "__main__":
    main()