"""
Integration example: Using Kuramoto dynamics with Oracle memory system.

This demonstrates how the Kuramoto module can be integrated with the
existing VoxelCloud memory architecture.
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kuramoto import KuramotoSolver, TextToOscillators, ProtoIdentityGenerator


class KuramotoMemoryEncoder:
    """
    Wrapper class to integrate Kuramoto dynamics with Oracle memory.

    This replaces the FFT-based encoding with phase synchronization.
    """

    def __init__(
        self,
        coupling_strength: float = 3.0,
        frequency_scale: float = 2.0,
        resolution: tuple = (512, 512),
        sync_threshold: float = 0.95
    ):
        """Initialize Kuramoto memory encoder."""
        self.text_encoder = TextToOscillators(frequency_scale=frequency_scale)
        self.solver = KuramotoSolver(
            coupling_strength=coupling_strength,
            sync_threshold=sync_threshold,
            max_steps=1000
        )
        self.proto_generator = ProtoIdentityGenerator(resolution=resolution)

    def encode_text(self, text: str) -> dict:
        """
        Encode text into proto-identity using Kuramoto dynamics.

        Args:
            text: Input text to encode

        Returns:
            dict: {
                'proto_identity': Quaternion field (H, W, 4),
                'field': Complex interference field,
                'phases': Synchronized phases,
                'frequencies': Natural frequencies,
                'order_parameter': Synchronization measure,
                'metadata': Additional information
            }
        """
        # Step 1: Convert text to natural frequencies
        frequencies = self.text_encoder.encode(text)

        # Step 2: Synchronize oscillators
        sync_result = self.solver.solve(frequencies)

        if not sync_result['converged']:
            print(f"Warning: Synchronization did not converge for '{text[:20]}...'")

        # Step 3: Generate proto-identity field
        field = self.proto_generator.generate(
            sync_result['phases'],
            text=text
        )

        # Step 4: Convert to quaternion for VoxelCloud compatibility
        proto_identity = self.proto_generator.to_quaternion(field)

        # Calculate coherence
        coherence = self.proto_generator.coherence_measure(field)

        return {
            'proto_identity': proto_identity,
            'field': field,
            'phases': sync_result['phases'],
            'frequencies': frequencies,
            'order_parameter': sync_result['order_parameter'],
            'metadata': {
                'text_length': len(text),
                'num_oscillators': len(frequencies),
                'sync_steps': sync_result['steps'],
                'converged': sync_result['converged'],
                'coherence': coherence,
                'resolution': proto_identity.shape[:2]
            }
        }

    def decode_estimate(self, proto_identity: np.ndarray) -> dict:
        """
        Attempt to reconstruct information from proto-identity.

        Note: This is lossy - original text cannot be perfectly recovered.

        Args:
            proto_identity: Quaternion field (H, W, 4)

        Returns:
            dict: Reconstruction information
        """
        # Reconstruct complex field
        field = self.proto_generator.from_quaternion(proto_identity)

        # Extract phase information (simplified - real system would be more complex)
        # This is just a demonstration of the concept
        coherence = self.proto_generator.coherence_measure(field)

        # Get field statistics
        magnitude_mean = np.abs(field).mean()
        phase_std = np.std(np.angle(field))

        return {
            'field': field,
            'coherence': coherence,
            'magnitude_mean': magnitude_mean,
            'phase_std': phase_std,
            'field_energy': np.sum(np.abs(field)**2)
        }

    def compute_similarity(self, proto1: np.ndarray, proto2: np.ndarray) -> float:
        """
        Compute similarity between two proto-identities.

        Args:
            proto1: First quaternion field
            proto2: Second quaternion field

        Returns:
            Similarity score in [0, 1]
        """
        # Reconstruct fields
        field1 = self.proto_generator.from_quaternion(proto1)
        field2 = self.proto_generator.from_quaternion(proto2)

        # Normalize fields
        field1_norm = field1 / (np.abs(field1).max() + 1e-10)
        field2_norm = field2 / (np.abs(field2).max() + 1e-10)

        # Compute correlation
        correlation = np.abs(np.vdot(field1_norm.flatten(), field2_norm.flatten()))
        correlation /= (np.linalg.norm(field1_norm) * np.linalg.norm(field2_norm) + 1e-10)

        return float(correlation)


def demonstrate_kuramoto_memory():
    """Demonstrate Kuramoto memory encoding."""
    print("=" * 60)
    print("KURAMOTO MEMORY ENCODING DEMONSTRATION")
    print("=" * 60)

    # Initialize encoder
    encoder = KuramotoMemoryEncoder(
        coupling_strength=3.0,
        frequency_scale=2.0,
        resolution=(256, 256)
    )

    # Test texts
    test_texts = [
        "Hello world",
        "Hello world!",  # Similar but different
        "Goodbye world",
        "Quantum memory system",
        "The quick brown fox jumps over the lazy dog"
    ]

    encoded_texts = []

    # Encode all texts
    print("\n1. ENCODING PHASE")
    print("-" * 40)
    for i, text in enumerate(test_texts):
        print(f"\nEncoding text {i+1}: '{text}'")
        result = encoder.encode_text(text)
        encoded_texts.append(result)

        meta = result['metadata']
        print(f"  - Oscillators: {meta['num_oscillators']}")
        print(f"  - Sync steps: {meta['sync_steps']}")
        print(f"  - Converged: {meta['converged']}")
        print(f"  - Coherence: {meta['coherence']:.4f}")
        print(f"  - Order parameter: {result['order_parameter'][0]:.4f}")

    # Test similarity
    print("\n2. SIMILARITY ANALYSIS")
    print("-" * 40)
    print("\nSimilarity matrix:")
    print("     ", end="")
    for i in range(len(test_texts)):
        print(f" T{i+1:2}", end="  ")
    print()

    for i in range(len(test_texts)):
        print(f"T{i+1:2}: ", end="")
        for j in range(len(test_texts)):
            sim = encoder.compute_similarity(
                encoded_texts[i]['proto_identity'],
                encoded_texts[j]['proto_identity']
            )
            print(f"{sim:5.3f}", end=" ")
        print()

    print("\nText legend:")
    for i, text in enumerate(test_texts):
        print(f"T{i+1}: '{text[:30]}{'...' if len(text) > 30 else ''}'")

    # Test reconstruction
    print("\n3. RECONSTRUCTION TEST")
    print("-" * 40)
    for i in range(min(3, len(test_texts))):
        print(f"\nReconstructing text {i+1}: '{test_texts[i]}'")
        recon = encoder.decode_estimate(encoded_texts[i]['proto_identity'])
        print(f"  - Coherence: {recon['coherence']:.4f}")
        print(f"  - Magnitude mean: {recon['magnitude_mean']:.4f}")
        print(f"  - Phase std: {recon['phase_std']:.4f}")
        print(f"  - Field energy: {recon['field_energy']:.2f}")

    # Memory footprint comparison
    print("\n4. MEMORY EFFICIENCY")
    print("-" * 40)
    total_text_bytes = sum(len(text.encode('utf-8')) for text in test_texts)
    total_proto_bytes = sum(
        result['proto_identity'].nbytes for result in encoded_texts
    )

    print(f"Total text size: {total_text_bytes} bytes")
    print(f"Total proto-identity size: {total_proto_bytes:,} bytes")
    print(f"Compression ratio: {total_proto_bytes/total_text_bytes:.1f}x")
    print("\nNote: Proto-identities are fixed-size regardless of text length,")
    print("enabling O(1) lookup and comparison operations.")

    print("\n" + "=" * 60)
    print("DEMONSTRATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    demonstrate_kuramoto_memory()