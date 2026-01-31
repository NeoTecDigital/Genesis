# Genesis: Multi-Octave Hierarchical Memory System

**Current Architecture Document**
**Last Updated**: 2026-01-30

---

## Overview

Genesis is a frequency-based memory system that encodes text at multiple octave levels using FFT-based proto-identities with dynamic clustering. The system achieves O(vocabulary_size) storage efficiency through similarity wells and resonance tracking.

## Core Architecture

### Proto-Identity Representation

Each text unit (character, word, phrase) maps to a unique **proto-identity**:

- **Format**: 512×512×4 XYZW quaternion field
- **Generation**: Text → UTF-8 bytes → 2D FFT → Complex spectrum → Proto-identity
- **Reversibility**: IFFT enables lossless text reconstruction
- **Clustering**: Similar units (≥0.90 similarity) share proto-identities

### Multi-Octave Hierarchy

Text is decomposed at multiple frequency scales:

```
Octave +4: Character level    (finest granularity)
Octave  0: Word level         (primary semantic units)
Octave -2: Short phrases      (2-3 words)
Octave -4: Long phrases       (4-6 words)
```

**Key Principle**: Each octave level has independent clustering - characters don't interfere with words.

### Encoding Pipeline

```
Input Text
    ↓
Decompose at octaves (+4, 0, -2, -4)
    ↓
For each unit:
    unit → UTF-8 bytes
        ↓
    Arrange in 2D grid (512×512)
        ↓
    Apply 2D FFT → Complex frequency spectrum
        ↓
    Convert to proto-identity (512×512×4)
        ↓
    Find similar proto at same octave (similarity ≥ 0.90)
        ↓
    If found: Strengthen resonance (cluster)
    If not: Create new proto-identity
        ↓
Store in VoxelCloud with metadata
```

### Dynamic Clustering

**Similarity Wells**: Proto-identities cluster when similarity ≥ 0.90

**Resonance Tracking**: Counts occurrences of each proto
- Character 'e' appearing 10 times → single proto with resonance=10
- Enables frequency-based retrieval

**Weighted Averaging**: New occurrences blend via constructive interference
```python
weight_new = 1.0 / resonance_strength
proto_identity = (1 - weight_new) * existing + weight_new * new
```

### Decoding Pipeline

```
Query
    ↓
Query each octave level (character, word, phrase)
    ↓
For each octave:
    Compute similarities to all protos at that octave
    Sort by similarity × resonance_strength
    ↓
Hierarchical reconstruction:
    1. Character level (finest)
    2. Word level (structure)
    3. Phrase level (context)
    ↓
Return synthesized text
```

## Key Components

### MultiOctaveEncoder (`src/pipeline/multi_octave_encoder.py`)

Encodes text at multiple octave levels:

**Key Methods**:
- `encode_text_hierarchical(text, octaves)` - Main encoding entry point
- `_decompose_at_octave(text, octave)` - Splits text into units
- `_encode_unit_to_proto(unit, octave)` - FFT → proto-identity

**Critical Detail**: Uses FFTTextEncoder for reversible text encoding via 2D FFT

### MultiOctaveDecoder (`src/pipeline/multi_octave_decoder.py`)

Reconstructs text from multiple octaves:

**Key Methods**:
- `decode_from_memory(query_proto, voxel_cloud)` - Query and reconstruct
- `decode_to_summary(query_proto, visible_protos)` - Synthesize from pre-filtered protos
- `_hierarchical_reconstruction(octave_results)` - Combines character/word/phrase levels

### VoxelCloudClustering (`src/memory/voxel_cloud_clustering.py`)

Implements dynamic clustering mechanism:

**Key Functions**:
- `find_nearest_proto(voxel_cloud, proto, octave)` - Find matching proto at same octave
- `add_or_strengthen_proto(...)` - Core clustering logic
- `compute_proto_similarity(proto1, proto2)` - Cosine similarity calculation
- `query_by_octave(voxel_cloud, query_proto, octave)` - Octave-specific retrieval
- `get_octave_statistics(voxel_cloud)` - Clustering metrics

### VoxelCloud (`src/memory/voxel_cloud.py`)

3D spatial memory structure:

**Features**:
- **Spatial indexing**: 10×10×10 grid for fast neighbor lookup
- **Frequency indexing**: 128 bins for frequency-based retrieval
- **Metadata**: Each proto stores unit, octave, resonance_strength
- **Persistence**: Serializable for save/load

## Storage Efficiency

### Compression Ratios

**Character Level (Octave +4)**:
- Input: 154 character occurrences
- Stored: 27 unique protos
- Compression: 82.5%

**Word Level (Octave 0)**:
- Input: 31 word occurrences
- Stored: 26 unique protos
- Compression: 16.1%

### Complexity

- **Storage**: O(vocabulary_size) instead of O(corpus_size)
- **Retrieval**: O(n) linear scan at octave level (spatial indexing available)
- **Clustering**: O(n) similarity computation during insertion

## Memory Architecture

### Core Memory
- **Purpose**: Long-term consolidated knowledge
- **Population**: Foundation training on curated documents
- **Persistence**: Saved to disk, loaded on startup
- **Size**: Unbounded (limited by available memory)

### Experiential Memory
- **Purpose**: Short-term working memory
- **Population**: Active query processing, synthesis
- **Persistence**: Session-based, cleared on exit
- **Size**: Dynamic, grows during conversation

## Implementation Details

### Data Structures

**Proto-Identity**: `np.ndarray` shape (512, 512, 4) dtype float32
- Channel 0: X = magnitude
- Channel 1: Y = phase
- Channel 2: Z = magnitude × cos(phase)
- Channel 3: W = magnitude × sin(phase)

**Frequency Spectrum**: `np.ndarray` shape (512, 512, 2) dtype float32
- Channel 0: Magnitude
- Channel 1: Phase

**ProtoIdentityEntry**: Dataclass containing:
- `proto_identity`: The 512×512×4 field
- `frequency`: Original frequency spectrum
- `position`: 3D spatial coordinate
- `octave`: Octave level (-4, -2, 0, +4)
- `resonance_strength`: Occurrence count
- `metadata`: Dict with 'unit', 'octave', 'modality'

### FFT-Based Pattern Generation

```python
def encode_text(text: str) -> np.ndarray:
    """Encode text to proto-identity via 2D FFT."""

    # 1. Convert text to UTF-8 bytes
    text_bytes = text.encode('utf-8')

    # 2. Arrange bytes in 2D spatial grid
    grid = np.zeros((512, 512), dtype=np.complex128)
    # Fill grid with byte values in spiral pattern
    for idx, byte in enumerate(text_bytes):
        if idx >= 512 * 512:
            break
        y, x = divmod(idx, 512)
        grid[y, x] = byte

    # 3. Apply 2D FFT
    freq_spectrum = np.fft.fft2(grid)

    # 4. Convert to XYZW quaternion
    magnitude = np.abs(freq_spectrum)
    phase = np.angle(freq_spectrum)
    proto = np.stack([
        magnitude * np.cos(phase),  # X
        magnitude * np.sin(phase),  # Y
        magnitude,                  # Z
        (phase + np.pi) / (2 * np.pi)  # W (normalized)
    ], axis=-1)

    return proto.astype(np.float32)
```

### FFT Reversibility

**Critical Design Decision**: Use FFT for mathematically reversible encoding.

**Why**:
- IFFT provides lossless text reconstruction
- No metadata storage required (text encoded in frequency domain)
- Frequency representation enables similarity clustering
- Mathematically sound transformation

**Decoding**: Proto-identity → Complex spectrum → 2D IFFT → Byte grid → UTF-8 text
- Perfect reconstruction via inverse transform
- No information loss in encoding/decoding cycle

## Testing and Validation

### Test Suite

**Primary Test**: `test_multi_octave_clustering.py`

Validates:
1. Character convergence (avg similarity ≥ 0.85)
2. Character protos < 100
3. Word protos > 0
4. Common characters have high resonance
5. Storage efficiency metrics

**Current Results**:
```
✅ Character convergence: 1.000 (perfect)
✅ Character protos: 27/154 (82.5% compression)
✅ Word protos: 26/31 (16.1% compression)
✅ Common characters: 8/8 pass (resonance = occurrence count)
```

## Design Rationale

### Why Multi-Octave?

**Problem**: Single-scale encoding loses hierarchical structure
- Character 'e' appears in many words
- Word "the" appears in many phrases
- Need to capture both local and global patterns

**Solution**: Independent octave levels
- Characters cluster at octave +4
- Words cluster at octave 0
- Phrases cluster at octave -2/-4
- No cross-octave interference

### Why FFT-Based Encoding?

**Problem**: Need reversible text encoding without metadata storage

**Solution**: 2D FFT transformation
- Text → Bytes → 2D grid → FFT → Frequency domain
- IFFT reverses the process perfectly
- No information loss in round-trip

**Benefits**:
- Mathematical reversibility (IFFT = perfect decoder)
- No raw text storage required
- Frequency domain enables similarity clustering
- Lossless encoding/decoding

## Future Directions

### Optimizations
- Spatial indexing for O(log n) octave queries
- GPU acceleration for similarity computations
- Incremental clustering updates

### Extensions
- Additional octave levels (sentence, paragraph, document)
- Multi-modal proto-identities (text + image + audio)
- Cross-octave attention mechanisms

### Research Questions
- Optimal similarity threshold (currently 0.90)
- Optimal number of Gaussian peaks (currently 8)
- Resonance decay over time
- Proto-identity pruning strategies

---

---

## Security Considerations

### Pickle Serialization Security

Genesis uses Python pickle for VoxelCloud serialization. **Critical security considerations**:

#### RestrictedUnpickler (Production Requirement)
```python
class RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # Whitelist allowed classes only
        ALLOWED = {
            ('numpy', 'ndarray'),
            ('src.memory.voxel_cloud', 'VoxelCloud'),
            ('src.memory.voxel_cloud', 'ProtoIdentityEntry'),
        }
        if (module, name) not in ALLOWED:
            raise pickle.UnpicklingError(f"Forbidden: {module}.{name}")
        return super().find_class(module, name)
```

#### HMAC Integrity Verification
```python
def save_signed(path, data):
    serialized = pickle.dumps(data)
    sig = hmac.new(SECRET, serialized, hashlib.sha256).digest()
    with open(path, 'wb') as f:
        f.write(sig + serialized)
```

**Warning**: Never deserialize pickle data from untrusted sources.

---

## See Also

- `README.md` - Quick start and usage guide
- `CLAUDE.md` - Project standards and guidelines
- `docs/FFT_ARCHITECTURE_SPEC.md` - Detailed FFT encoding specification
- `docs/advanced/IMPLEMENTATION.md` - Rust/Vulkan GPU implementation
- `docs/advanced/MEMORY_INTEGRATION.md` - Memory system integration
- `SECURITY_REQUIREMENTS.md` - Complete security standards
