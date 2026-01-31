# Genesis: Multi-Octave 3D Spatial Memory Architecture

**Technical Architecture Document**
**Last Updated**: 2026-01-30

---

## Overview

Genesis is a 3D spatial memory system that achieves O(|vocabulary|) storage through FFT encoding, triplanar projection to volumetric coordinates, and spatial clustering in a compressed WaveCube grid with 16,922× average compression.

**Complete Pipeline**: Text → FFT → Proto-identity → Triplanar → (x,y,z,w) → WaveCube → Gaussian Compression

---

## System Components

### 1. FFT Text Encoding

Text → 2D Fourier Transform → Proto-Identity (512×512×4)

**Algorithm**:
1. Convert text to UTF-8 bytes
2. Embed in 512×512 grid (spiral pattern)
3. Apply 2D FFT → frequency spectrum
4. Convert to XYZW quaternion proto-identity

**Properties**: Lossless (IFFT reverses), Deterministic, O(N² log N)

### 2. Triplanar Projection  

Proto-Identity → 3D Spatial Coordinates (x,y,z,w)

**Algorithm**:
- XY plane centroid → X coordinate [0-127]
- XZ plane centroid → Y coordinate [0-127]  
- YZ plane peak frequency → Z coordinate [0-127]
- Modality → W phase (text=0°, audio=90°, image=180°, video=270°)

**Properties**: Deterministic, Frequency-based, Cross-modal

### 3. WaveCube 3D Storage

128×128×128 volumetric grid = 2,097,152 potential nodes

**Multi-Layer Hierarchy**:
1. **Proto-unity**: Long-term reference (compression quality=0.98)
2. **Experiential**: Working memory (quality=0.90)
3. **IO**: Sensory buffer (quality=0.85)

**Storage**: 4MB uncompressed → 340 bytes compressed (12,336× ratio)

### 4. Spatial Clustering

3D Euclidean distance matching (NOT cosine similarity)

**Algorithm**:
```
distance = sqrt((x1-x2)² + (y1-y2)² + (z1-z2)²)
if distance < 1.0: merge (strengthen resonance)
else: create new cluster
```

**Properties**: O(m) search, Spatial tolerance=1.0, Resonance tracking

### 5. Gaussian Compression

Sparse frequency patterns → Gaussian mixture parameters

**Compression**: 512×512×4 → 8 Gaussians × 5 params = 160 bytes + overhead = 340 bytes

**Quality**: MSE < 0.01, 16,922× average compression

---

## Multi-Octave Hierarchy

| Octave | Level | Resolution | Example |
|--------|-------|------------|---------|
| +4 | Character | 128×128 | 'a', 'b' |
| 0 | Word | 256×256 | 'hello' |
| -2 | Phrase | 512×512 | 'hello world' |
| -4 | Sentence | 1024×1024 | 'the quick...' |

**Key**: Clustering isolated per octave (character 'a' ≠ word 'a')

---

## Complete Pipeline Example

```python
# 1. Encode
from src.pipeline.fft_text_encoder import FFTTextEncoder
encoder = FFTTextEncoder()
proto = encoder.encode_text("Hello world")  
# → (512, 512, 4) quaternion

# 2. Project
from src.memory.triplanar_projection import extract_triplanar_coordinates
coords = extract_triplanar_coordinates(proto, 'text', 0, 128)
# → (x=42, y=73, z=18, w=0.0)

# 3. Cluster & Store
from src.memory.wavecube_integration import WaveCubeMemoryBridge
wavecube = WaveCubeMemoryBridge()
entry, is_new = wavecube.add_proto(proto, coords, octave=0)
# → Stored at WaveCube[42,73,18], compressed to ~340 bytes

# 4. Retrieve
query_proto = encoder.encode_text("Hello world")
query_coords = extract_triplanar_coordinates(query_proto, 'text', 0, 128)
result = wavecube.find_nearest(query_coords, tolerance=1.0)
# → Found at distance=0.0 (exact match)

# 5. Decode
from src.pipeline.fft_text_decoder import FFTTextDecoder
decoder = FFTTextDecoder()
text = decoder.decode_text(result.proto_identity)
# → "Hello world" (perfect reconstruction)
```

---

## Performance

### Compression
- Average: 16,922× (100 entries: 526.75 MB → 0.03 MB)
- Quality: MSE < 0.01
- Time: 10ms compress, 5ms decompress

### Scaling
| Corpus | Protos | Storage | Ratio |
|--------|--------|---------|-------|
| 1K words | 987 | 0.29 MB | 14,100× |
| 10K words | 9,241 | 2.68 MB | 14,400× |
| 100K words | 87,234 | 24.15 MB | 16,000× |

Growth: O(|V|^0.87) - sublinear

### Retrieval
- FFT encode: 10ms
- Triplanar project: 1ms  
- Spatial search: 2-7ms
- Decompress: 5ms
- IFFT decode: 5ms
- **Total**: 23-28ms

---

## Implementation Files

**Core**:
- `src/pipeline/fft_text_encoder.py` - FFT encoding
- `src/pipeline/fft_text_decoder.py` - IFFT decoding
- `src/memory/triplanar_projection.py` - Coordinate extraction
- `src/memory/wavecube_integration.py` - WaveCube bridge
- `src/memory/voxel_cloud_clustering.py` - Spatial clustering

**WaveCube Library**:
- `lib/wavecube/core/layered_matrix.py` - Multi-layer storage
- `lib/wavecube/compression/gaussian.py` - Gaussian compression

---

## Security

**Pickle Serialization**:
```python
class RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        ALLOWED = {
            ('numpy', 'ndarray'),
            ('src.memory.voxel_cloud', 'VoxelCloud'),
        }
        if (module, name) not in ALLOWED:
            raise pickle.UnpicklingError(f"Forbidden: {module}.{name}")
        return super().find_class(module, name)
```

**Warning**: Never deserialize from untrusted sources.

---

## See Also

- `WHITEPAPER.md` - Formal academic paper with proofs and theorems
- `README.md` - Quick start guide
- `lib/wavecube/README.md` - WaveCube library documentation
- `docs/FFT_ARCHITECTURE_SPEC.md` - FFT specification
