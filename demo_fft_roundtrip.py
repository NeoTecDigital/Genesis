#!/usr/bin/env python3
"""
FFT Roundtrip Demonstration

This script demonstrates the FFT-based text encoding/decoding pipeline
with various text samples and saves results for verification.
"""

import numpy as np
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline.fft_text_encoder import FFTTextEncoder
from src.pipeline.fft_text_decoder import FFTTextDecoder


def test_roundtrip(text: str, label: str) -> dict:
    """Test encoding/decoding roundtrip and collect metrics."""
    encoder = FFTTextEncoder()
    decoder = FFTTextDecoder()

    print(f"\n{'='*60}")
    print(f"Testing: {label}")
    print(f"{'='*60}")
    print(f"Original text length: {len(text)} bytes")

    # Encode
    start = time.perf_counter()
    proto = encoder.encode_text(text)
    encode_time = (time.perf_counter() - start) * 1000

    print(f"Proto-identity shape: {proto.shape}")
    print(f"Proto-identity dtype: {proto.dtype}")
    print(f"Encoding time: {encode_time:.2f}ms")

    # Analyze frequency spectrum
    grid = encoder._text_to_grid(text)
    spectrum = encoder._grid_to_frequency(grid)
    magnitude = spectrum[:, :, 0]

    # Calculate sparsity
    max_mag = np.max(magnitude)
    if max_mag > 0:
        sparse_mask = magnitude < (max_mag * 0.01)
        sparsity = np.sum(sparse_mask) / sparse_mask.size
        print(f"Frequency sparsity (< 1% max): {sparsity:.2%}")

    # Decode
    start = time.perf_counter()
    decoded = decoder.decode_text(proto)
    decode_time = (time.perf_counter() - start) * 1000

    print(f"Decoded text length: {len(decoded)} bytes")
    print(f"Decoding time: {decode_time:.2f}ms")

    # Check accuracy
    min_len = min(len(text), len(decoded))
    if min_len > 0:
        matching = sum(1 for a, b in zip(text[:min_len], decoded[:min_len]) if a == b)
        accuracy = matching / min_len
    else:
        accuracy = 0.0

    print(f"Accuracy: {accuracy:.2%}")

    # Show sample
    if len(decoded) > 100:
        print(f"\nFirst 100 chars of decoded text:")
        print(f"  {decoded[:100]}")
    else:
        print(f"\nDecoded text:")
        print(f"  {decoded}")

    return {
        'label': label,
        'original_len': len(text),
        'decoded_len': len(decoded),
        'encode_time': encode_time,
        'decode_time': decode_time,
        'accuracy': accuracy,
        'sparsity': sparsity if max_mag > 0 else 0.0,
        'success': accuracy >= 0.95
    }


def main():
    """Run FFT roundtrip demonstrations."""
    print("FFT-Based Text Encoding/Decoding Demonstration")
    print("=" * 60)

    results = []

    # Test 1: Simple ASCII
    text1 = "Hello, World! This is a test of the FFT-based encoding system."
    results.append(test_roundtrip(text1, "Simple ASCII"))

    # Test 2: UTF-8 Multilingual
    text2 = "Hello 世界! Здравствуй мир! مرحبا بالعالم! 🌍🌎🌏"
    results.append(test_roundtrip(text2, "UTF-8 Multilingual"))

    # Test 3: Code snippet
    text3 = """
def fibonacci(n):
    '''Calculate nth Fibonacci number'''
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

# Test the function
for i in range(10):
    print(f"F({i}) = {fibonacci(i)}")
"""
    results.append(test_roundtrip(text3, "Python Code"))

    # Test 4: Structured data
    text4 = """
{
    "project": "Genesis",
    "architecture": "FFT-based encoding",
    "features": [
        "Perfect reversibility",
        "Frequency domain representation",
        "No metadata storage",
        "UTF-8 support"
    ],
    "version": "1.0.0"
}
"""
    results.append(test_roundtrip(text4, "JSON Data"))

    # Test 5: Long philosophical text
    text5 = """
Consciousness emerges from the intricate dance of billions of neurons,
each firing in patterns that somehow give rise to the subjective experience
of being. The hard problem of consciousness - explaining how and why we have
qualitative, subjective experiences - remains one of the most profound mysteries
in science and philosophy.

Consider the redness of red, the painfulness of pain, or the taste of chocolate.
These qualia, as philosophers call them, seem to resist reduction to mere
physical processes. Yet everything we know about the brain suggests it operates
according to the same physical laws that govern the rest of the universe.

How then does the gray matter in our skulls produce the rich inner life we
each experience? Some propose that consciousness is fundamental to the universe,
others that it emerges from complex information processing, and still others
that it's an illusion created by the brain's predictive models.

The implications extend far beyond philosophy. As we create increasingly
sophisticated artificial systems, we must grapple with questions of machine
consciousness. Can a sufficiently complex artificial neural network experience
qualia? Would we even know if it did?

Perhaps consciousness, like life itself, exists on a spectrum rather than
as a binary property. From the simplest organisms responding to stimuli,
to the rich inner worlds of humans contemplating their own existence,
consciousness may be a matter of degree rather than kind.
"""
    results.append(test_roundtrip(text5, "Philosophical Text"))

    # Test 6: Mathematical expressions
    text6 = """
∫₀^∞ e^(-x²) dx = √π/2
∇²φ = ρ/ε₀
E = mc²
∑ᵢ₌₁ⁿ i = n(n+1)/2
lim(x→∞) (1 + 1/x)ˣ = e
"""
    results.append(test_roundtrip(text6, "Mathematical Notation"))

    # Summary
    print("\n" + "="*60)
    print("SUMMARY RESULTS")
    print("="*60)

    # Create output directory if it doesn't exist
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    # Save results to file
    output_file = output_dir / "fft_roundtrip_demo.txt"

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("FFT Roundtrip Demonstration Results\n")
        f.write("="*60 + "\n\n")

        for result in results:
            status = "✓ PASS" if result['success'] else "✗ FAIL"

            line = (f"{status} {result['label']:20s} | "
                   f"Acc: {result['accuracy']:6.2%} | "
                   f"Encode: {result['encode_time']:6.2f}ms | "
                   f"Decode: {result['decode_time']:6.2f}ms | "
                   f"Sparse: {result['sparsity']:6.2%}")

            print(line)
            f.write(line + "\n")

        # Overall statistics
        successful = sum(1 for r in results if r['success'])
        total = len(results)
        success_rate = successful / total if total > 0 else 0

        avg_encode = np.mean([r['encode_time'] for r in results])
        avg_decode = np.mean([r['decode_time'] for r in results])
        avg_accuracy = np.mean([r['accuracy'] for r in results])

        stats = f"\nOverall Success Rate: {successful}/{total} ({success_rate:.1%})"
        stats += f"\nAverage Accuracy: {avg_accuracy:.2%}"
        stats += f"\nAverage Encode Time: {avg_encode:.2f}ms"
        stats += f"\nAverage Decode Time: {avg_decode:.2f}ms"

        print(stats)
        f.write(stats + "\n")

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()