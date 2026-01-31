# Genesis: Multi-Octave Hierarchical Memory System

A frequency-based memory system that encodes text at multiple octave levels using FFT-based proto-identities with dynamic clustering, achieving O(vocabulary_size) storage efficiency.

**Status**: FFT encoding implementation active
**Encoding**: Reversible FFT-based text encoding (no metadata storage)
**Last Updated**: 2026-01-30

---

## Quick Start

### Installation

```bash
# Install dependencies
pip install numpy

# Verify installation
python test_multi_octave_clustering.py
```

### Basic Usage

```python
from src.pipeline.multi_octave_encoder import MultiOctaveEncoder
from src.pipeline.multi_octave_decoder import MultiOctaveDecoder
from src.memory.voxel_cloud import VoxelCloud
from src.memory.voxel_cloud_clustering import add_or_strengthen_proto

# Create voxel cloud memory
voxel_cloud = VoxelCloud()

# Encode text at character and word levels
carrier = np.zeros((512, 512, 4))  # Unused, kept for API compatibility
encoder = MultiOctaveEncoder(carrier)

text = "Hello world"
units = encoder.encode_text_hierarchical(text, octaves=[4, 0])

# Add to memory with dynamic clustering
for unit in units:
    entry, is_new = add_or_strengthen_proto(
        voxel_cloud,
        unit.proto_identity,
        unit.frequency,
        unit.octave
    )

# Query and decode
decoder = MultiOctaveDecoder(carrier)
query_proto = units[0].proto_identity
response = decoder.decode_from_memory(query_proto, voxel_cloud)
print(response)
```

---

## Architecture Overview

### Multi-Octave Hierarchy

Text is decomposed at multiple frequency scales:

```
Octave +4: Characters     ('a', 'b', 'c', ...)
Octave  0: Words          ('hello', 'world', ...)
Octave -2: Short phrases  ('hello world', ...)
Octave -4: Long phrases   ('the way that can', ...)
```

### Proto-Identity Representation

- **Format**: 512×512×4 XYZW quaternion field
- **Generation**: Text → UTF-8 bytes → 2D FFT → Complex spectrum → Proto-identity
- **Reversibility**: IFFT decodes proto-identity back to original text

### Dynamic Clustering

**Similarity Wells**: Proto-identities cluster when similarity ≥ 0.90

```
Input: 154 character occurrences
Output: 27 unique protos (82.5% compression)

Example: Character 'e' appears 10 times
Result: Single proto-identity with resonance_strength = 10
```

**Benefits**:
- O(vocabulary_size) storage instead of O(corpus_size)
- Automatic frequency-based retrieval via resonance
- No explicit vocabulary management required

### FFT-Based Encoding

**Critical**: Uses 2D FFT for reversible text encoding. Proto-identities contain the complete text representation in frequency domain.

**Why**: Mathematically reversible via IFFT - no metadata storage required for text reconstruction.

---

## Project Structure

```
genesis/
├── src/
│   ├── pipeline/
│   │   ├── multi_octave_encoder.py    # Hash-based encoding
│   │   └── multi_octave_decoder.py    # Hierarchical decoding
│   ├── memory/
│   │   ├── voxel_cloud.py             # 3D spatial memory
│   │   └── voxel_cloud_clustering.py  # Dynamic clustering
│   └── ...
├── test_multi_octave_clustering.py    # Primary validation test
├── ARCHITECTURE.md                     # Detailed architecture docs
├── CLAUDE.md                           # Project standards
└── README.md                           # This file
```

---

## Testing

### Run Primary Test

```bash
python test_multi_octave_clustering.py
```

**Expected Output**:
```
✅ Character convergence: 1.000 (perfect)
✅ Character protos: 27/154 (82.5% compression)
✅ Word protos: 26/31 (16.1% compression)
✅ Common characters: 8/8 pass
✅ OVERALL: TEST PASSED
```

### Validation Criteria

1. **Character Convergence**: Average similarity ≥ 0.85 for same character
2. **Character Protos**: < 100 unique protos
3. **Word Protos**: > 0 unique protos
4. **Resonance**: Common characters have high resonance (count = occurrences)

---

## Key Features

### FFT Text Encoding

```python
def encode_text(text: str) -> np.ndarray:
    """Encode text to proto-identity via 2D FFT."""
    # 1. Convert text to UTF-8 bytes
    text_bytes = text.encode('utf-8')

    # 2. Arrange bytes in 2D spatial grid (512×512)
    grid = arrange_bytes_to_grid(text_bytes)

    # 3. Apply 2D FFT
    freq_spectrum = np.fft.fft2(grid)

    # 4. Convert to proto-identity (XYZW quaternion)
    return spectrum_to_proto(freq_spectrum)
```

### Weighted Averaging

When proto-identities cluster, new occurrences blend via constructive interference:

```python
weight_new = 1.0 / resonance_strength
proto_identity = (1 - weight_new) * existing + weight_new * new
```

### Hierarchical Decoding

```python
def decode_from_memory(query_proto, voxel_cloud):
    """Decode by querying multiple octaves."""
    # Query each octave
    char_results = query_by_octave(voxel_cloud, query_proto, octave=4)
    word_results = query_by_octave(voxel_cloud, query_proto, octave=0)

    # Reconstruct hierarchically (characters → words → phrases)
    return hierarchical_reconstruction(char_results, word_results)
```

---

## Performance

### Storage Efficiency

| Octave | Input | Stored | Compression |
|--------|-------|--------|-------------|
| +4 (char) | 154 occurrences | 27 protos | 82.5% |
| 0 (word) | 31 occurrences | 26 protos | 16.1% |

### Complexity

- **Encoding**: O(n) where n = number of units
- **Clustering**: O(m) where m = protos at octave (similarity computation)
- **Retrieval**: O(m) linear scan (spatial indexing available but not used)

---

## Design Rationale

### Why Multi-Octave?

Single-scale encoding loses hierarchical structure. Multi-octave enables:
- Character-level precision for spelling
- Word-level semantics for meaning
- Phrase-level context for understanding

### Why FFT-Based?

FFT encoding provides mathematical reversibility via IFFT:
- No metadata storage required (text encoded in proto-identity)
- Perfect reconstruction via inverse transform
- Frequency domain representation enables similarity clustering
- Lossless encoding/decoding cycle

---

## Documentation

- **[ARCHITECTURE.md](ARCHITECTURE.md)** - Detailed architecture and implementation
- **[CLAUDE.md](CLAUDE.md)** - Project standards and guidelines
- **[API.md](API.md)** - API reference (if available)

Legacy documentation (historical reference only):
- **ARCHITECTURE_LEGACY.md** - Old carrier-based architecture
- **README_LEGACY.md** - Old project documentation

---

## Contributing

See **[CONTRIBUTING.md](CONTRIBUTING.md)** for development guidelines.

**Code Quality Standards**:
- Files: Maximum 500 lines
- Functions: Maximum 50 lines
- Nesting: Maximum 3 indentation levels
- Always update existing code rather than creating duplicates

---

## License

[Add license information]

---

## Status

**Current Phase**: Core architecture validated
**Next Steps**:
- Extended octave levels (sentence, paragraph, document)
- Multi-modal proto-identities (text + image + audio)
- GPU acceleration for similarity computations

---

## Citation

If you use Genesis in your research, please cite:

```bibtex
@software{genesis2026,
  title = {Genesis: Multi-Octave Hierarchical Memory System},
  author = {Christopher, Richard I},
  year = {2026},
  url = {https://github.com/NeoTecDigital/Genesis},
  license = {MIT}
}
```

See `CITATION.cff` for structured citation metadata.

---

## License

MIT License - see `LICENSE` file for details.

Copyright (c) 2025 Richard I Christopher
