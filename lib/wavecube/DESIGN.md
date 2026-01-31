# Wavecube: Multi-Resolution Wavetable Matrix Library

## Executive Summary

Wavecube is a high-performance library for storing, manipulating, and interpolating multi-dimensional data in a 3D grid structure. It provides variable-resolution wavetable storage with advanced compression, GPU-accelerated interpolation, and multi-modal data support.

**Design Goals:**
- Replace fixed 512×512×4 architecture with flexible variable-resolution system
- Achieve 100-10000× compression through parametric representations
- Enable real-time trilinear interpolation for smooth data morphing
- Support multi-octave hierarchies for multi-scale processing
- Provide modality-agnostic foundation for text, audio, image, and video processing

---

## Core Architecture

### 1. Wavetable Matrix Structure

A **WavetableMatrix** is a 3D grid where each node at coordinates (x, y, z) stores a wavetable (2D array of shape [H, W, C]).

```
Grid Dimensions: (width, height, depth)
Node Storage: Each node → wavetable [resolution_h, resolution_w, channels]
Interpolation: Trilinear interpolation between 8 surrounding nodes
```

#### Example Use Cases

**Audio Synthesis:**
- X-axis: Timbre (sine → saw → square → noise)
- Y-axis: Brightness (dark → bright)
- Z-axis: Harmonicity (harmonic → inharmonic)

**Text Embeddings:**
- X-axis: Formality (casual → formal)
- Y-axis: Sentiment (negative → positive)
- Z-axis: Domain (technical → creative)

**Image/Video:**
- X-axis: Time of day (dawn → day → dusk → night)
- Y-axis: Color temperature (warm → cool)
- Z-axis: Motion intensity (static → dynamic)

### 2. Multi-Resolution Support

Unlike fixed-resolution systems (e.g., always 512×512), Wavecube supports:

1. **Uniform Resolution**: All nodes use same resolution (e.g., 256×256)
2. **Variable Resolution**: Each node can have different resolution based on content complexity
3. **Adaptive Resolution**: Automatically adjust resolution based on frequency content

```python
# Uniform resolution (simple)
matrix = WavetableMatrix(width=10, height=10, depth=10, resolution=256)

# Variable resolution (advanced)
matrix = MultiResolutionWavetableMatrix(width=10, height=10, depth=10)
matrix.set_node(0, 0, 0, wavetable_64x64)   # Low-frequency content
matrix.set_node(5, 5, 5, wavetable_1024x1024)  # High-frequency content
```

### 3. Octave Hierarchy

Multi-scale representation with different octave levels:

```
Octave +4: Character-level (finest granularity)
Octave  0: Word-level (medium granularity)
Octave -2: Phrase-level (coarse granularity)
Octave -4: Paragraph-level (coarsest granularity)
```

Each octave has its own WavetableMatrix with appropriate resolution:

```python
hierarchy = OctaveHierarchy(octave_levels=[-4, -2, 0, +4])
hierarchy.matrices[+4].resolution = 1024  # Fine details
hierarchy.matrices[-4].resolution = 64    # Coarse structure
```

---

## Compression & Parametric Representations

### Compression Methods

#### 1. Gaussian Mixture Compression (Highest Compression)

**Use Case:** Sparse frequency patterns (e.g., Genesis proto-identities)

**Storage:**
```python
@dataclass
class GaussianMixtureParams:
    num_gaussians: int                    # Typically 4-16
    amplitudes: np.ndarray                # (N,) amplitude per Gaussian
    centers: np.ndarray                   # (N, 2) [x, y] centers
    covariances: np.ndarray               # (N, 2, 2) covariance matrices
    phases: np.ndarray                    # (N,) phase per Gaussian
    resolution: tuple[int, int]           # Original resolution
    channels: int                         # Number of channels
```

**Compression Ratio:** 1000-10000× for sparse patterns

**Example:**
- Dense: 512×512×4×4 bytes = 4,194,304 bytes (4MB)
- Compressed: 8 Gaussians × 9 params × 4 bytes = 288 bytes
- Ratio: **14,563×**

#### 2. DCT Compression (JPEG-like)

**Use Case:** General-purpose image/wavetable compression

**Storage:** Top-K DCT coefficients + quantization

**Compression Ratio:** 10-100× depending on quality

#### 3. FFT Compression (Frequency Domain)

**Use Case:** Frequency-domain data (audio spectrograms, frequency fields)

**Storage:** Magnitude + phase with threshold-based sparsification

**Compression Ratio:** 5-50× with aggressive thresholding

#### 4. Wavelet Compression

**Use Case:** Multi-scale image data, smooth gradients

**Storage:** Wavelet coefficients at multiple scales

**Compression Ratio:** 20-200× depending on content

#### 5. Quantization

**Use Case:** Reduce precision without full compression

**Storage:** Reduce float32 → int16 or int8

**Compression Ratio:** 2-4×

### Codec API

```python
class WavetableCodec:
    """Base class for wavetable compression."""

    def encode(self, wavetable: np.ndarray, quality: float = 0.95) -> CompressedWavetable:
        """Compress wavetable to parametric representation."""
        raise NotImplementedError

    def decode(self, compressed: CompressedWavetable) -> np.ndarray:
        """Reconstruct wavetable from parameters."""
        raise NotImplementedError

# Concrete implementations
class GaussianMixtureCodec(WavetableCodec): ...
class DCTCodec(WavetableCodec): ...
class FFTCodec(WavetableCodec): ...
class WaveletCodec(WavetableCodec): ...
```

---

## Core API Reference

### WavetableMatrix

```python
class WavetableMatrix:
    """3D grid of wavetables with trilinear interpolation."""

    def __init__(
        self,
        width: int,
        height: int,
        depth: int,
        resolution: tuple[int, int] | int = 512,
        channels: int = 4,
        dtype: np.dtype = np.float32,
        sparse: bool = True,
        compression: str | None = None
    ):
        """
        Initialize wavetable matrix.

        Args:
            width, height, depth: Grid dimensions
            resolution: Wavetable resolution (H, W) or single int
            channels: Number of channels (4 for XYZW quaternions)
            dtype: Data type (float32, float16)
            sparse: Use sparse storage (only allocate populated nodes)
            compression: Compression method ('gaussian', 'dct', 'fft', None)
        """

    # Node Operations
    def set_node(self, x: int, y: int, z: int, wavetable: np.ndarray) -> None:
        """Set wavetable at grid position (x, y, z)."""

    def get_node(self, x: int, y: int, z: int) -> np.ndarray:
        """Get wavetable at grid position (x, y, z)."""

    def has_node(self, x: int, y: int, z: int) -> bool:
        """Check if node exists at position."""

    def delete_node(self, x: int, y: int, z: int) -> None:
        """Delete node at position (for sparse matrices)."""

    # Interpolation
    def sample(self, x: float, y: float, z: float) -> np.ndarray:
        """
        Sample wavetable at fractional coordinates using trilinear interpolation.

        Args:
            x, y, z: Fractional coordinates in [0, width/height/depth]

        Returns:
            Interpolated wavetable [resolution_h, resolution_w, channels]
        """

    def sample_batch(self, coords: np.ndarray) -> np.ndarray:
        """
        Batch sampling for multiple coordinates.

        Args:
            coords: (N, 3) array of [x, y, z] coordinates

        Returns:
            (N, resolution_h, resolution_w, channels) interpolated wavetables
        """

    # Compression
    def compress_node(self, x: int, y: int, z: int,
                     method: str = 'gaussian', quality: float = 0.95) -> None:
        """Compress node in-place."""

    def compress_all(self, method: str = 'gaussian', quality: float = 0.95) -> None:
        """Compress all nodes."""

    def decompress_node(self, x: int, y: int, z: int) -> np.ndarray:
        """Decompress and return wavetable."""

    def decompress_all(self) -> None:
        """Decompress all nodes in-place."""

    # Resolution Management
    def get_resolution(self, x: int, y: int, z: int) -> tuple[int, int]:
        """Get resolution of node at (x, y, z)."""

    def set_global_resolution(self, resolution: tuple[int, int] | int) -> None:
        """Change resolution for all future nodes."""

    def upscale_node(self, x: int, y: int, z: int, factor: int) -> None:
        """Upscale node resolution by factor."""

    def downscale_node(self, x: int, y: int, z: int, factor: int) -> None:
        """Downscale node resolution by factor."""

    # I/O
    def save(self, path: str, format: str = 'npz') -> None:
        """Save matrix to disk (supports 'npz', 'hdf5', 'wavecube')."""

    @staticmethod
    def load(path: str) -> 'WavetableMatrix':
        """Load matrix from disk."""

    # Utilities
    def get_memory_usage(self) -> dict[str, int]:
        """Get memory usage statistics."""

    def get_compression_ratio(self) -> float:
        """Get overall compression ratio."""

    def visualize_slice(self, axis: str = 'z', index: int = 0) -> np.ndarray:
        """Visualize 2D slice through matrix."""
```

### MultiResolutionWavetableMatrix

```python
class MultiResolutionWavetableMatrix(WavetableMatrix):
    """Wavetable matrix with per-node variable resolution."""

    def set_node(
        self,
        x: int, y: int, z: int,
        wavetable: np.ndarray,
        auto_compress: bool = True
    ) -> None:
        """
        Set node with automatic resolution detection.

        Args:
            x, y, z: Grid coordinates
            wavetable: Input wavetable (any resolution)
            auto_compress: Automatically compress based on content
        """

    def get_resolution_map(self) -> dict[tuple[int, int, int], tuple[int, int]]:
        """Get map of all node resolutions."""
```

### OctaveHierarchy

```python
class OctaveHierarchy:
    """Multi-octave wavetable matrices."""

    def __init__(
        self,
        width: int,
        height: int,
        depth: int,
        octave_levels: list[int] = [-4, -2, 0, +4],
        base_resolution: int = 512
    ):
        """
        Initialize octave hierarchy.

        Args:
            width, height, depth: Grid dimensions (shared across octaves)
            octave_levels: Octave levels to support
            base_resolution: Base resolution (octave 0)
        """

    def get_matrix(self, octave: int) -> WavetableMatrix:
        """Get wavetable matrix for specific octave."""

    def sample_at_octave(
        self,
        octave: int,
        x: float, y: float, z: float
    ) -> np.ndarray:
        """Sample from specific octave level."""

    def sample_multi_octave(
        self,
        x: float, y: float, z: float,
        octave_range: tuple[int, int] | None = None
    ) -> dict[int, np.ndarray]:
        """Sample from multiple octave levels simultaneously."""

    def save(self, path: str) -> None:
        """Save entire hierarchy."""

    @staticmethod
    def load(path: str) -> 'OctaveHierarchy':
        """Load hierarchy from disk."""
```

---

## Interpolation

### Trilinear Interpolation

Given fractional coordinates (x, y, z), interpolate between 8 surrounding nodes:

```
     (x0,y0,z1)────────(x1,y0,z1)
         /│                /│
        / │               / │
   (x0,y1,z1)────────(x1,y1,z1)
       │  │              │  │
       │ (x0,y0,z0)──────│─(x1,y0,z0)
       │ /               │ /
       │/                │/
   (x0,y1,z0)────────(x1,y1,z0)
```

**Algorithm:**
1. Find 8 surrounding integer grid points
2. Compute interpolation weights based on fractional parts
3. Linearly interpolate along X axis (4 interpolations)
4. Linearly interpolate along Y axis (2 interpolations)
5. Linearly interpolate along Z axis (1 interpolation)

**Complexity:** O(1) for single sample, O(N) for N samples

```python
def trilinear_interpolate(
    matrix: WavetableMatrix,
    x: float, y: float, z: float
) -> np.ndarray:
    """CPU implementation of trilinear interpolation."""

def trilinear_interpolate_gpu(
    matrix: WavetableMatrix,
    coords: np.ndarray  # (N, 3)
) -> np.ndarray:
    """GPU-accelerated batch trilinear interpolation."""
```

---

## File Format Specification

### NPZ Format (Simple)

Standard NumPy compressed archive:

```python
{
    'metadata': {
        'width': int,
        'height': int,
        'depth': int,
        'resolution': tuple[int, int],
        'channels': int,
        'compression': str | None
    },
    'nodes': {
        '0_0_0': np.ndarray,  # Wavetable at (0,0,0)
        '1_2_3': np.ndarray,  # Wavetable at (1,2,3)
        ...
    }
}
```

### HDF5 Format (Advanced)

Hierarchical structure for large matrices:

```
/
├── metadata (attributes)
├── grid/
│   ├── 0_0_0/
│   │   ├── wavetable (dataset)
│   │   └── compressed_params (dataset, if compressed)
│   ├── 1_2_3/
│   ...
└── octaves/  (if OctaveHierarchy)
    ├── -4/
    ├── -2/
    ├── 0/
    └── +4/
```

### Wavecube Format (Custom Binary)

Optimized binary format for fast loading:

```
Header (256 bytes):
  - Magic number: "WVCB" (4 bytes)
  - Version: uint32
  - Width, height, depth: uint32 × 3
  - Resolution: uint32 × 2
  - Channels: uint32
  - Compression method: uint8
  - Sparse: bool
  - Reserved: 228 bytes

Node Index (variable):
  - Num nodes: uint64
  - For each node:
    - Coordinates: uint32 × 3
    - Offset: uint64
    - Size: uint64
    - Compression params offset: uint64 (if compressed)

Node Data (variable):
  - Raw wavetable data or compressed parameters
```

---

## Performance Considerations

### Memory Usage

**Dense Storage (No Compression):**
- Single node: `resolution_h × resolution_w × channels × sizeof(dtype)`
- Full matrix: `width × height × depth × node_size`
- Example: 10×10×10 grid, 512×512×4, float32 = **4GB**

**Compressed Storage (Gaussian Mixture):**
- Single node: `num_gaussians × 9 params × sizeof(float32)` ≈ 288 bytes
- Full matrix: `width × height × depth × 288 bytes`
- Example: 10×10×10 grid = **288KB** (14,000× reduction)

**Sparse Storage:**
- Only allocated nodes consume memory
- Overhead: ~48 bytes per node (indexing)

### Computation Performance

**CPU Interpolation:**
- Single sample: ~0.1ms (trilinear + decompression)
- Batch (1000 samples): ~50ms
- Bottleneck: Decompression if using Gaussian mixture

**GPU Interpolation:**
- Single sample: ~0.01ms
- Batch (1000 samples): ~5ms (20× speedup)
- Batch (100,000 samples): ~100ms (50× speedup)

**Recommendations:**
- Use compression for storage, decompress on-demand
- Cache frequently accessed nodes
- Use GPU for batch operations (>100 samples)
- Pre-decompress "hot" nodes for real-time access

---

## Integration with Genesis

### Replacing 512×512×4 Architecture

**Current Genesis:**
```python
proto_identity: np.ndarray  # (512, 512, 4) - FIXED
frequency: np.ndarray       # (512, 512, 2)
```

**With Wavecube:**
```python
# Store compressed parameters instead of full array
wavetable_index: tuple[int, int, int]  # Position in grid
compressed_params: GaussianMixtureParams  # 288 bytes
octave: int
resolution: tuple[int, int]  # Variable resolution

# Reconstruct on demand
proto_identity = octave_hierarchy.sample_at_octave(
    octave, x, y, z
)  # Any resolution!
```

### Migration Path

**Phase 1:** Parallel Storage
- Keep existing 512×512×4 storage
- Add Wavecube storage alongside
- Validate equivalence

**Phase 2:** Hybrid Usage
- Store new entries in Wavecube
- Lazy-migrate old entries on access
- Benchmark performance

**Phase 3:** Full Migration
- Convert all entries to Wavecube
- Remove 512×512×4 storage
- Optimize for new architecture

---

## Quality Gates

### Unit Tests
- ✅ All core operations have unit tests (coverage >95%)
- ✅ Interpolation accuracy within 1e-6 of reference
- ✅ Compression/decompression lossless or within quality threshold
- ✅ I/O round-trip preserves data

### Integration Tests
- ✅ Genesis integration: Can replace proto-identity storage
- ✅ Performance: <10ms for batch interpolation (1000 samples, GPU)
- ✅ Memory: Compression achieves >100× for sparse patterns
- ✅ Multi-octave: Seamless sampling across octave levels

### Performance Benchmarks
- ✅ Memory usage: Measure dense vs compressed vs sparse
- ✅ Interpolation speed: CPU vs GPU, single vs batch
- ✅ Compression ratio: Measure per codec per data type
- ✅ Load time: <1s for 10×10×10 matrix from disk

### Documentation
- ✅ All public APIs documented with docstrings
- ✅ Type hints for all functions
- ✅ README with quick start examples
- ✅ Tutorial notebooks for common use cases

---

## Future Extensions

### Advanced Interpolation
- Cubic interpolation (higher quality, slower)
- Catmull-Rom splines
- Hermite interpolation with gradients

### Advanced Compression
- Learned compression (VAE/GAN latent space)
- Adaptive quantization
- Temporal compression for video

### GPU Acceleration
- CUDA kernels for interpolation
- Vulkan compute shaders
- Metal (for Apple Silicon)

### Distributed Storage
- Shard large matrices across multiple files
- Lazy loading of nodes
- Network-based access (gRPC, HTTP)

### Machine Learning Integration
- PyTorch tensor backend
- Differentiable interpolation
- Gradient-based optimization of node positions
