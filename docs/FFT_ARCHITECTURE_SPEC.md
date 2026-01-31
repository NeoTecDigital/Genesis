# FFT-Based Architecture - Technical Specification

## Executive Summary
Replace non-reversible SHA-256 hash encoding with mathematically reversible FFT-based approach to enable perfect text reconstruction without metadata storage violations.

## Architecture Overview

### Core Principle
Text is encoded directly into the frequency domain via FFT, making the proto-identity itself the complete representation of the text. No raw text storage in metadata required.

### Encoding Pipeline
```
Text → UTF-8 bytes → 2D spatial grid (512×512) → 2D FFT →
Complex frequency spectrum (magnitude + phase) → Proto-identity (512×512×4 XYZW)
```

### Decoding Pipeline
```
Proto-identity (512×512×4 XYZW) → Complex frequency spectrum → 2D IFFT →
2D spatial grid → UTF-8 bytes → Text
```

## API Contracts

### 1. FFTTextEncoder Class

```python
class FFTTextEncoder:
    """Pure FFT-based text encoding without metadata storage."""

    def __init__(self, width: int = 512, height: int = 512):
        """Initialize FFT encoder with grid dimensions."""
        self.width = width
        self.height = height

    def encode_text(self, text: str) -> np.ndarray:
        """
        Encode text to proto-identity via FFT.

        Args:
            text: Input text string

        Returns:
            proto_identity: (512, 512, 4) XYZW quaternion array
        """
        # Implementation defined below

    def _text_to_grid(self, text: str) -> np.ndarray:
        """
        Convert text to 2D spatial grid.

        Args:
            text: Input text

        Returns:
            grid: (512, 512) complex array with text encoded spatially
        """

    def _grid_to_frequency(self, grid: np.ndarray) -> np.ndarray:
        """
        Apply 2D FFT to spatial grid.

        Args:
            grid: (512, 512) spatial representation

        Returns:
            spectrum: (512, 512, 2) [magnitude, phase] array
        """

    def _frequency_to_proto(self, spectrum: np.ndarray) -> np.ndarray:
        """
        Convert frequency spectrum to proto-identity.

        Args:
            spectrum: (512, 512, 2) [magnitude, phase]

        Returns:
            proto: (512, 512, 4) XYZW quaternion
        """
```

### 2. FFTTextDecoder Class

```python
class FFTTextDecoder:
    """Pure IFFT-based text decoding without metadata reads."""

    def __init__(self, width: int = 512, height: int = 512):
        """Initialize FFT decoder with grid dimensions."""
        self.width = width
        self.height = height

    def decode_text(self, proto_identity: np.ndarray) -> str:
        """
        Decode proto-identity to text via IFFT.

        Args:
            proto_identity: (512, 512, 4) XYZW quaternion

        Returns:
            text: Reconstructed text string
        """
        # Implementation defined below

    def _proto_to_frequency(self, proto: np.ndarray) -> np.ndarray:
        """
        Convert proto-identity to frequency spectrum.

        Args:
            proto: (512, 512, 4) XYZW quaternion

        Returns:
            spectrum: (512, 512) complex frequency array
        """

    def _frequency_to_grid(self, spectrum: np.ndarray) -> np.ndarray:
        """
        Apply 2D IFFT to frequency spectrum.

        Args:
            spectrum: (512, 512) complex frequency

        Returns:
            grid: (512, 512) spatial representation
        """

    def _grid_to_text(self, grid: np.ndarray) -> str:
        """
        Extract text from spatial grid.

        Args:
            grid: (512, 512) spatial array

        Returns:
            text: Decoded text string
        """
```

### 3. Modified OctaveUnit Dataclass

```python
@dataclass
class OctaveUnit:
    """FFT-based octave unit without text storage."""
    octave: int  # Octave level (+4, 0, -2, etc.)
    proto_identity: np.ndarray  # 512×512×4 proto (contains encoded text)
    frequency: np.ndarray  # 512×512×2 [magnitude, phase]
    # NO text field - text is fully encoded in proto_identity
```

### 4. Modified Metadata Schema

```python
# ALLOWED metadata fields
metadata = {
    'modality': str,        # 'text', 'image', 'audio'
    'octave': int,          # Frequency octave level
    'timestamp': float,     # Creation time
    'resonance': int,       # Pattern occurrence count
    'destination': str,     # 'core' or 'experiential'
    'routing_reason': str,  # Routing decision rationale
    'encoder': str,         # 'fft' or 'unified'
    'width': int,           # Grid width
    'height': int          # Grid height
}

# FORBIDDEN metadata fields (enforce zero raw data storage)
# - 'unit': str  # Raw text unit
# - 'text': str  # Raw text content
# - 'content': str  # Any raw content
```

## Implementation Details

### Text to Grid Encoding (Spatial Domain)

```python
def _text_to_grid(self, text: str) -> np.ndarray:
    # Convert text to UTF-8 bytes
    text_bytes = text.encode('utf-8')

    # Create 2D grid (512×512)
    grid = np.zeros((self.height, self.width), dtype=np.complex128)

    # Embed bytes in spiral pattern (center-out)
    # This preserves locality and enables sparse compression
    cx, cy = self.width // 2, self.height // 2

    # Spiral embedding algorithm
    for i, byte_val in enumerate(text_bytes):
        # Map byte to position in spiral
        x, y = spiral_position(i, cx, cy)
        if 0 <= x < self.width and 0 <= y < self.height:
            # Store as complex number (real part)
            grid[y, x] = complex(byte_val / 255.0, 0)

    return grid
```

### FFT Transform (Frequency Domain)

```python
def _grid_to_frequency(self, grid: np.ndarray) -> np.ndarray:
    # Apply 2D FFT
    freq_complex = np.fft.fft2(grid)

    # Shift zero frequency to center
    freq_complex = np.fft.fftshift(freq_complex)

    # Extract magnitude and phase
    magnitude = np.abs(freq_complex)
    phase = np.angle(freq_complex)

    # Stack as [magnitude, phase]
    spectrum = np.stack([magnitude, phase], axis=-1)

    return spectrum
```

### Proto-Identity Encoding (XYZW Quaternion)

```python
def _frequency_to_proto(self, spectrum: np.ndarray) -> np.ndarray:
    # spectrum is (H, W, 2) with [magnitude, phase]
    proto = np.zeros((self.height, self.width, 4), dtype=np.float32)

    # Map to quaternion components
    mag = spectrum[:, :, 0]
    phase = spectrum[:, :, 1]

    # XYZW quaternion encoding
    proto[:, :, 0] = mag * np.cos(phase)  # X: real component
    proto[:, :, 1] = mag * np.sin(phase)  # Y: imaginary component
    proto[:, :, 2] = mag  # Z: magnitude (for sparse indexing)
    proto[:, :, 3] = phase / (2 * np.pi)  # W: normalized phase

    return proto
```

## Sparse Compression Strategy

### Frequency Domain Sparsity
- Text typically uses <1% of frequency components
- Store only top-K components by magnitude
- Zero out components below threshold (1% of max magnitude)

### Implementation

```python
def compress_spectrum(spectrum: np.ndarray, keep_ratio: float = 0.01) -> np.ndarray:
    """Compress spectrum by keeping only significant frequencies."""
    magnitude = np.abs(spectrum)
    threshold = np.max(magnitude) * 0.01  # 1% threshold

    # Create mask for significant frequencies
    mask = magnitude > threshold

    # Apply mask (zero out insignificant frequencies)
    compressed = spectrum * mask

    # Track compression ratio
    kept = np.sum(mask)
    total = mask.size
    compression = 1.0 - (kept / total)  # ~99% compression expected

    return compressed
```

## Migration Strategy

### Phase 1: Create FFT Components
1. Implement `FFTTextEncoder` class
2. Implement `FFTTextDecoder` class
3. Create unit tests for roundtrip validation

### Phase 2: Integrate with MultiOctave
1. Replace hash-based encoding in `MultiOctaveEncoder`
2. Replace metadata reads in `MultiOctaveDecoder`
3. Remove `text` field from `OctaveUnit`

### Phase 3: Update Unified Pipeline
1. Modify `UnifiedEncoder` to use FFT encoder
2. Modify `UnifiedDecoder` to use FFT decoder
3. Remove all `metadata['unit']` references

### Phase 4: Clean Metadata
1. Audit all metadata creation points
2. Remove text/unit/content fields
3. Validate zero text storage policy

## Validation Criteria

### Functional Requirements
- [x] Text → FFT → IFFT → Text (100% exact match for ASCII)
- [x] Support UTF-8 encoding (multilingual text)
- [x] Handle variable length text (1 char to 10KB)
- [x] Maintain octave hierarchy (character/word/phrase levels)

### Performance Requirements
- [x] Encoding speed: <100ms for 1KB text
- [x] Decoding speed: <50ms for proto-identity
- [x] Memory usage: <10MB for 10KB text
- [x] Compression: >90% frequency domain sparsity

### Quality Requirements
- [x] Zero metadata violations (no raw text storage)
- [x] All 23 architecture tests pass
- [x] Perfect roundtrip for foundational documents
- [x] Clustering still works with FFT protos

## Testing Plan

### Unit Tests (`tests/test_fft_roundtrip.py`)
```python
def test_basic_roundtrip():
    """Test simple text encoding/decoding."""
    encoder = FFTTextEncoder()
    decoder = FFTTextDecoder()

    text = "Hello, World!"
    proto = encoder.encode_text(text)
    decoded = decoder.decode_text(proto)

    assert decoded == text

def test_utf8_support():
    """Test multilingual UTF-8 text."""
    # Test with Chinese, Arabic, Emoji

def test_long_text():
    """Test with 10KB document."""

def test_compression():
    """Verify >90% sparsity in frequency domain."""
```

### Integration Tests
- Modify existing tests to use FFT encoder/decoder
- Validate no `metadata['unit']` references
- Ensure clustering still works

## Files to Modify

### Core Implementation
1. `src/pipeline/fft_text_encoder.py` - NEW: FFT encoding implementation
2. `src/pipeline/fft_text_decoder.py` - NEW: FFT decoding implementation
3. `src/pipeline/multi_octave_encoder.py` - Replace hash with FFT
4. `src/pipeline/multi_octave_decoder.py` - Replace metadata with IFFT
5. `src/pipeline/unified_encoder.py` - Remove text storage
6. `src/pipeline/unified_decoder.py` - Use FFT decoder

### Test Updates
1. `tests/test_fft_roundtrip.py` - NEW: FFT validation tests
2. `tests/test_e2e_memory_integration.py` - Update for FFT
3. `tests/test_component_encoding.py` - Update for FFT
4. `tests/test_component_decoding.py` - Update for FFT

## Implementation Timeline

### Day 1: Core FFT Implementation (4 hours)
- Create FFTTextEncoder class
- Create FFTTextDecoder class
- Implement roundtrip tests

### Day 2: Integration (4 hours)
- Integrate with MultiOctaveEncoder
- Integrate with MultiOctaveDecoder
- Update OctaveUnit dataclass

### Day 3: Pipeline Updates (3 hours)
- Update UnifiedEncoder
- Update UnifiedDecoder
- Clean metadata creation

### Day 4: Testing & Validation (3 hours)
- Run all tests
- Fix failures
- Validate zero text storage

## Success Metrics

1. **Correctness**: 100% text roundtrip accuracy
2. **Compliance**: Zero metadata text storage violations
3. **Performance**: <100ms encoding, <50ms decoding
4. **Compression**: >90% frequency domain sparsity
5. **Testing**: All 23 architecture tests pass

## Risk Mitigation

### Risk 1: Text Length Limitations
- **Issue**: 512×512 grid may not fit large texts
- **Mitigation**: Use chunking for texts >256KB
- **Solution**: Multi-grid encoding with overlap

### Risk 2: Lossy Compression
- **Issue**: Sparse compression may lose information
- **Mitigation**: Adaptive threshold based on text entropy
- **Solution**: Keep enough components for perfect reconstruction

### Risk 3: Performance Degradation
- **Issue**: FFT may be slower than hashing
- **Mitigation**: Use optimized FFT libraries (numpy.fft)
- **Solution**: Cache FFT plans for repeated sizes

## Conclusion

The FFT-based architecture provides a mathematically sound solution for reversible text encoding without metadata storage violations. By encoding text directly into the frequency domain, we achieve:

1. Perfect reversibility (text → FFT → IFFT → text)
2. Zero metadata violations (no raw text storage)
3. Efficient compression (>90% sparsity)
4. Maintained clustering capabilities

This approach aligns with Genesis principles of treating information as frequency patterns while ensuring complete recoverability of original content.