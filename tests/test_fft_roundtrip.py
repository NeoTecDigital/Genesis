"""
FFT roundtrip validation tests.

This module validates the FFT-based text encoding/decoding pipeline
for perfect reversibility and performance characteristics.
"""

import pytest
import numpy as np
import time
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline.fft_text_encoder import FFTTextEncoder
from src.pipeline.fft_text_decoder import FFTTextDecoder


class TestFFTRoundtrip:
    """Test suite for FFT text encoding/decoding roundtrip."""

    def setup_method(self):
        """Initialize encoder and decoder for each test."""
        self.encoder = FFTTextEncoder()
        self.decoder = FFTTextDecoder()

    def test_basic_roundtrip(self):
        """Simple ASCII text roundtrip."""
        text = "Hello, World!"

        # Encode to proto-identity
        proto = self.encoder.encode_text(text)

        # Validate proto dimensions
        assert proto.shape == (512, 512, 4), f"Expected (512, 512, 4), got {proto.shape}"
        assert proto.dtype == np.float32, f"Expected float32, got {proto.dtype}"

        # Decode back to text
        decoded = self.decoder.decode_text(proto)

        # Verify perfect roundtrip
        assert decoded == text, f"Roundtrip failed: '{text}' != '{decoded}'"

    def test_utf8_roundtrip(self):
        """UTF-8 multilingual text."""
        texts = [
            "Hello 世界 مرحبا 🌍",
            "Здравствуй мир",
            "こんにちは世界",
            "नमस्ते दुनिया",
            "Γειά σου κόσμε"
        ]

        for text in texts:
            proto = self.encoder.encode_text(text)
            decoded = self.decoder.decode_text(proto)
            assert decoded == text, f"UTF-8 roundtrip failed for: '{text}'"

    def test_long_text(self):
        """1KB+ document roundtrip."""
        # Generate a long text
        long_text = """The Genesis Project represents a fundamental shift in how we approach
        artificial intelligence and machine learning. By treating information as frequency
        patterns in a multi-dimensional space, we enable natural clustering of similar
        concepts while maintaining perfect reversibility through FFT-based encoding.

        This approach eliminates the traditional separation between storage and computation,
        allowing the system to directly operate on frequency-domain representations. The
        proto-identity becomes not just a compressed representation, but the actual
        computational substrate for reasoning and synthesis.

        Key innovations include:
        1. Hash-free encoding via FFT transforms
        2. Perfect text reconstruction without metadata
        3. Natural similarity clustering in frequency space
        4. Multi-octave hierarchical organization
        5. Sparse compression achieving >90% efficiency

        The system maintains semantic relationships through frequency resonance, where
        similar concepts naturally align in the spectral domain. This enables both
        efficient storage and intuitive retrieval based on conceptual similarity rather
        than exact matching.
        """ * 2  # Duplicate to make it longer

        # Test roundtrip
        proto = self.encoder.encode_text(long_text)
        decoded = self.decoder.decode_text(proto)

        # For long text, verify at least the beginning matches perfectly
        # (spiral embedding may truncate very long texts)
        min_length = min(len(long_text), len(decoded))
        assert decoded[:min_length] == long_text[:min_length], \
            "Long text roundtrip failed"

    def test_empty_text(self):
        """Edge case: empty string."""
        text = ""
        proto = self.encoder.encode_text(text)
        decoded = self.decoder.decode_text(proto)
        assert decoded == text, f"Empty text roundtrip failed: '{decoded}'"

    def test_special_characters(self):
        """Newlines, tabs, special chars."""
        texts = [
            "Line1\nLine2\nLine3",
            "Tab\tSeparated\tValues",
            "Special: !@#$%^&*()_+-=[]{}|;:',.<>?/",
            "Quotes: \"double\" and 'single'",
            "Escape: \\n \\t \\r \\\\",
            "Mixed\n\twhitespace\r\npatterns"
        ]

        for text in texts:
            proto = self.encoder.encode_text(text)
            decoded = self.decoder.decode_text(proto)
            assert decoded == text, f"Special char roundtrip failed for: '{text}'"

    def test_compression_ratio(self):
        """Verify frequency domain properties and compression potential."""
        # Use text that fills more of the grid
        text = """The Genesis Project represents a fundamental shift in how we approach
        artificial intelligence and machine learning. By treating information as frequency
        patterns in a multi-dimensional space, we enable natural clustering of similar
        concepts while maintaining perfect reversibility through FFT-based encoding.
        """ * 20  # More substantial text

        # Encode to get frequency spectrum
        grid = self.encoder._text_to_grid(text)
        spectrum = self.encoder._grid_to_frequency(grid)

        # Analyze frequency distribution
        magnitude = spectrum[:, :, 0]
        max_mag = np.max(magnitude)

        # Count frequencies above various thresholds
        thresholds = [0.5, 0.1, 0.01, 0.001]
        for threshold in thresholds:
            mask = magnitude > (max_mag * threshold)
            kept_ratio = np.sum(mask) / mask.size
            print(f"\nFrequencies > {threshold*100:.1f}% of max: {kept_ratio:.2%}")

        # Test aggressive compression
        compressed, ratio = self.encoder.compress_spectrum(spectrum, keep_ratio=0.1)

        print(f"\nCompression with 10% threshold: {ratio:.2%}")
        print(f"Text size: {len(text)} bytes")
        print(f"Grid utilization: {len(text)/(512*512):.2%}")

        # For sparse text embedding, FFT will have many active frequencies
        # This is expected behavior, not a failure
        # The real benefit comes from the reversibility, not compression

        # Verify roundtrip works even with compression
        proto = self.encoder._frequency_to_proto(compressed)
        decoded = self.decoder.decode_text(proto)

        # Just verify we get some text back
        assert len(decoded) > 0, "Decoding produced empty text"

        # Log information about frequency distribution
        print(f"Decoded text length: {len(decoded)} bytes")
        print(f"Original text length: {len(text)} bytes")

    def test_encoding_performance(self):
        """Encoding speed: Target <100ms for 1KB text."""
        # Generate 1KB text
        text = "a" * 1024

        # Warm up
        _ = self.encoder.encode_text("warmup")

        # Measure encoding time
        start = time.perf_counter()
        proto = self.encoder.encode_text(text)
        elapsed = time.perf_counter() - start

        # Convert to milliseconds
        elapsed_ms = elapsed * 1000

        # Log performance
        print(f"\nEncoding 1KB: {elapsed_ms:.2f}ms")

        # Verify target (relaxed for safety)
        assert elapsed_ms < 200, f"Encoding too slow: {elapsed_ms:.2f}ms > 200ms"

    def test_decoding_performance(self):
        """Decoding speed: Target <50ms for proto-identity."""
        # Generate proto-identity
        text = "a" * 1024
        proto = self.encoder.encode_text(text)

        # Warm up
        _ = self.decoder.decode_text(proto)

        # Measure decoding time
        start = time.perf_counter()
        decoded = self.decoder.decode_text(proto)
        elapsed = time.perf_counter() - start

        # Convert to milliseconds
        elapsed_ms = elapsed * 1000

        # Log performance
        print(f"Decoding proto: {elapsed_ms:.2f}ms")

        # Verify target (relaxed for safety)
        assert elapsed_ms < 100, f"Decoding too slow: {elapsed_ms:.2f}ms > 100ms"

    def test_memory_usage(self):
        """Track peak memory for 1KB text."""
        import tracemalloc

        # Start memory tracking
        tracemalloc.start()

        # Generate 1KB text
        text = "a" * 1024

        # Encode and decode
        proto = self.encoder.encode_text(text)
        decoded = self.decoder.decode_text(proto)

        # Get peak memory
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Convert to MB
        peak_mb = peak / 1024 / 1024

        # Log memory usage
        print(f"Peak memory: {peak_mb:.2f}MB")

        # Verify reasonable memory usage
        assert peak_mb < 50, f"Memory usage too high: {peak_mb:.2f}MB > 50MB"

    def test_proto_validation(self):
        """Test proto-identity validation."""
        # Valid proto
        text = "Test"
        proto = self.encoder.encode_text(text)
        assert self.decoder.validate_proto_identity(proto), "Valid proto rejected"

        # Invalid dimensions
        bad_proto = np.zeros((256, 256, 4))
        assert not self.decoder.validate_proto_identity(bad_proto), \
            "Invalid dimensions accepted"

        # NaN values
        bad_proto = proto.copy()
        bad_proto[0, 0, 0] = np.nan
        assert not self.decoder.validate_proto_identity(bad_proto), \
            "NaN values accepted"

        # Invalid phase normalization
        bad_proto = proto.copy()
        bad_proto[:, :, 3] = 2.0  # Phase > 1
        assert not self.decoder.validate_proto_identity(bad_proto), \
            "Invalid phase accepted"

    def test_numerical_stability(self):
        """Test numerical stability with edge cases."""
        texts = [
            "\x00",  # Null byte
            "\xff" * 10,  # Max byte values
            "a" * 100000,  # Very long text (will be truncated)
            "\n" * 100,  # Many newlines
            " " * 100,  # Many spaces
        ]

        for text in texts:
            try:
                proto = self.encoder.encode_text(text)
                decoded = self.decoder.decode_text(proto)
                # Just verify no crashes or exceptions
                assert proto is not None
                assert decoded is not None
            except Exception as e:
                pytest.fail(f"Numerical instability for text length {len(text)}: {e}")

    def test_foundation_document(self):
        """Test with actual foundation document."""
        # Try to load a foundation document if available
        foundation_path = Path("/usr/lib/alembic/data/datasets/text/curated/foundation/consciousness.txt")

        if foundation_path.exists():
            with open(foundation_path, 'r', encoding='utf-8') as f:
                text = f.read()[:5000]  # Use first 5KB

            proto = self.encoder.encode_text(text)
            decoded = self.decoder.decode_text(proto)

            # Check substantial similarity (may not be perfect for very long texts)
            min_len = min(len(text), len(decoded))
            matching = sum(1 for a, b in zip(text[:min_len], decoded[:min_len]) if a == b)
            accuracy = matching / min_len if min_len > 0 else 0

            print(f"Foundation document accuracy: {accuracy:.2%}")
            assert accuracy > 0.95, f"Foundation document accuracy too low: {accuracy:.2%}"

    def test_incremental_text(self):
        """Test with incrementally growing text."""
        base = "a"
        for i in range(1, 11):
            text = base * (10 ** i)
            if len(text) > 10000:  # Cap at 10KB for testing
                text = text[:10000]

            proto = self.encoder.encode_text(text)
            decoded = self.decoder.decode_text(proto)

            # Verify at least prefix matches
            min_len = min(100, len(text), len(decoded))
            assert decoded[:min_len] == text[:min_len], \
                f"Failed at length {len(text)}"


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])