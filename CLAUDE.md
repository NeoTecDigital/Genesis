# Genesis Project Standards

## Code Quality Standards

### Professional Excellence
- Maintain utmost professional standards in all code and documentation
- No duplicate implementations or bloated code
- No conflated intentions - clear separation of concerns
- Always update and maintain existing code/documentation rather than creating duplicates

### Code Structure Rules
- **Files**: Maximum 500 lines
- **Functions**: Maximum 50 lines
- **Nesting**: Maximum 3 indentation levels
- Anything exceeding these limits must be abstracted to maintain modular compositional paradigm

### Model Management
- Training models: Store in `./tmp/` during development
- Production models: Move to `./models/` with descriptive names
- Delete unsuccessful models to avoid storage bloat
- Maintain consistent naming, training, and logging practices

## Genesis Architecture (Current Implementation)

### Multi-Octave Hierarchical Encoding

The current architecture uses **FFT-based frequency encoding** with **dynamic clustering** at multiple octave levels, achieving O(vocabulary_size) storage efficiency.

#### Core Principles

1. **FFT-Based Encoding**: Proto-identities encode text via 2D FFT transformation
   - Text → UTF-8 bytes → 2D spatial grid → 2D FFT → Proto-identity
   - Reversible via IFFT for lossless text reconstruction
   - No metadata storage required (text encoded in frequency domain)

2. **Octave Hierarchy**:
   - Octave +4: Character level (finest granularity)
   - Octave 0: Word level
   - Octave -2: Short phrases (2-3 words)
   - Octave -4: Long phrases (4-6 words)

3. **Dynamic Clustering**: Similarity wells enable shared proto-identities
   - Threshold: 0.90 similarity → cluster together
   - Resonance tracking: Counts how many times pattern appears
   - Weighted averaging: New occurrences blend via constructive interference

#### Encoding Flow

```
Text → Decompose at octave levels →
  For each unit (char/word/phrase):
    unit → UTF-8 bytes → 2D grid → 2D FFT → frequency spectrum →
    convert to proto-identity (512×512×4) →
    cluster with similar protos (similarity ≥ 0.90) →
    store in VoxelCloud with resonance tracking
```

#### Storage Efficiency

- **Character level**: ~82.5% compression (e.g., 27 protos for 154 occurrences)
- **Word level**: ~16.1% compression (e.g., 26 protos for 31 occurrences)
- **Total**: O(vocabulary_size) instead of O(corpus_size)

#### Key Components

- **MultiOctaveEncoder**: Hash-based proto-identity generation at multiple octaves
- **MultiOctaveDecoder**: Hierarchical reconstruction from character → word → phrase levels
- **VoxelCloudClustering**: Dynamic similarity-based clustering with resonance tracking
- **VoxelCloud**: 3D spatial memory structure with frequency and spatial indexing

#### Memory Architecture

- **Core Memory**: Long-term consolidated knowledge (foundation training)
- **Experiential Memory**: Short-term working memory (active queries, synthesis)

### Implementation Notes

- Proto-identities are 512×512×4 XYZW quaternions
- Frequency spectrums are 512×512×2 [magnitude, phase]
- No GPU acceleration currently (pure NumPy implementation)
- Testing validates perfect character convergence (similarity = 1.000)
- we need to ensure tests outputs go into ./outputs/ for verification purposes, and that we use the complete foundational documents in /usr/lib/alembic/data/datasets/text/curated/foundation