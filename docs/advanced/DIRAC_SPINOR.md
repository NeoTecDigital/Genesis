# Dirac Spinor Implementation Design for Oracle UFT System

## Executive Summary

This document specifies the complete architecture for upgrading Oracle's UFT field evolution from scalar approximation to full 4-component Dirac spinor formalism. The design maintains backward compatibility while introducing quantum relativistic dynamics through gamma matrices and the chiral γ⁵ operator.

**Current**: Scalar field equation `i∂_tΨ + i∇Ψ = ΔRe^{iδ}Ψ`
**Target**: Dirac spinor equation `(iγ^μ∂_μ)Ψ = ΔRe^{iδγ⁵}Ψ`

## 1. Dirac Matrices Specification

### 1.1 Representation Choice

**Selected**: **Dirac-Pauli Representation** (Standard)

**Rationale**:
- Natural separation of positive/negative energy states
- Chiral operator γ⁵ has simple block structure
- Non-relativistic limit is transparent
- Most common in literature (easier validation)

### 1.2 Matrix Definitions

```python
# Pauli matrices (2×2)
σ₁ = [[0, 1],    σ₂ = [[0, -i],   σ₃ = [[1,  0],
      [1, 0]]          [i,  0]]         [0, -1]]

# Identity (2×2)
I₂ = [[1, 0],
      [0, 1]]

# Gamma matrices (4×4) in Dirac-Pauli basis
γ⁰ = [[I₂,  0 ],    # Timelike
      [0,  -I₂]]

γ¹ = [[0,   σ₁],    # Spatial x
      [-σ₁, 0 ]]

γ² = [[0,   σ₂],    # Spatial y
      [-σ₂, 0 ]]

γ³ = [[0,   σ₃],    # Spatial z (set ∂_z = 0 for 2D)
      [-σ₃, 0 ]]

# Chiral operator
γ⁵ = iγ⁰γ¹γ²γ³ = [[0,  I₂],
                   [I₂, 0 ]]
```

### 1.3 Storage Strategy

```python
class GammaMatrices:
    """Efficient storage and caching of gamma matrices."""

    def __init__(self, representation: str = "dirac"):
        self._cache = {}
        self._representation = representation

        # Pre-compute and store as complex128 for numerical stability
        self.gamma = {
            0: self._build_gamma0(),  # γ⁰
            1: self._build_gamma1(),  # γ¹
            2: self._build_gamma2(),  # γ²
            3: self._build_gamma3(),  # γ³
            5: self._build_gamma5(),  # γ⁵
        }

        # Pre-compute products for efficiency
        self.gamma_products = {
            (0, 1): self.gamma[0] @ self.gamma[1],
            (0, 2): self.gamma[0] @ self.gamma[2],
            # ... other useful products
        }
```

**Numerical Precision**: Use `np.complex128` (double precision) for all gamma matrices to minimize accumulation errors during evolution.

## 2. Spinor Field Structure

### 2.1 Field Representation

```python
# Current scalar field
Ψ_scalar: np.ndarray[512, 512, dtype=complex128]

# Target spinor field
Ψ_spinor: np.ndarray[512, 512, 4, dtype=complex128]
```

### 2.2 Memory Layout

**Selected**: **Component-last layout** `(512, 512, 4)`

**Rationale**:
- Natural for applying pointwise matrix operations
- Efficient for spatial derivatives (contiguous in x, y)
- Direct indexing: `Ψ[x, y]` gives 4-spinor at point (x, y)
- Cache-friendly for gamma matrix multiplication

### 2.3 Scalar-to-Spinor Initialization

```python
def scalar_to_spinor(scalar_field: np.ndarray,
                     metadata: Dict) -> np.ndarray:
    """
    Initialize 4-component spinor from scalar field.

    Strategy: Embed scalar as upper two components (positive energy)
    with phase information distributed across spinor components.
    """
    Ψ = np.zeros((512, 512, 4), dtype=np.complex128)

    # Normalize scalar field
    ψ_norm = scalar_field / (np.abs(scalar_field).max() + 1e-10)

    # Extract phase and magnitude
    magnitude = np.abs(ψ_norm)
    phase = np.angle(ψ_norm)

    # Upper components (positive energy states)
    Ψ[..., 0] = magnitude * np.exp(1j * phase)        # Spin up
    Ψ[..., 1] = magnitude * np.exp(1j * (phase + π/4)) # Spin down (phase shifted)

    # Lower components (negative energy states, initially suppressed)
    suppression = 0.1  # Small initial amplitude
    Ψ[..., 2] = suppression * magnitude * np.exp(1j * (phase + π/2))
    Ψ[..., 3] = suppression * magnitude * np.exp(1j * (phase + 3*π/4))

    # Optional: Use metadata to inform initialization
    if 'chiral_phase' in metadata:
        δ = metadata['chiral_phase']
        # Apply initial chiral rotation
        Ψ = apply_chiral_rotation(Ψ, δ * 0.1)  # Small initial chirality

    return Ψ
```

## 3. Dirac Operator Implementation

### 3.1 Operator Structure

The Dirac operator in 2+1D (spatial 2D + time):
```
D = iγ⁰∂_t + iγ¹∂_x + iγ²∂_y
```

Note: We drop γ³∂_z term for 2D spatial grid.

### 3.2 Spatial Discretization

**Selected**: **4th-order centered differences** with periodic BC

```python
def compute_spatial_derivative(field: np.ndarray,
                              axis: int,
                              dx: float = 1.0) -> np.ndarray:
    """
    4th-order accurate spatial derivative with periodic BC.

    Stencil: f'(x) ≈ (-f(x+2h) + 8f(x+h) - 8f(x-h) + f(x-2h)) / (12h)
    """
    # Pad for periodic BC
    pad_width = [(0, 0)] * field.ndim
    pad_width[axis] = (2, 2)
    padded = np.pad(field, pad_width, mode='wrap')

    # 4th-order stencil coefficients
    c = np.array([-1, 8, 0, -8, 1]) / (12 * dx)

    # Apply stencil
    derivative = np.zeros_like(field)
    for i, coeff in enumerate(c):
        if coeff != 0:
            shift = i - 2
            derivative += coeff * np.roll(padded, shift, axis=axis)[
                (slice(2, -2) if j == axis else slice(None))
                for j in range(field.ndim)
            ]

    return derivative
```

### 3.3 Boundary Conditions

**Selected**: **Periodic BC** (toroidal topology)

**Rationale**:
- Conserves probability/norm
- No artificial reflections
- Natural for Fourier-based methods
- Matches existing scalar implementation

## 4. Chiral Phase Implementation

### 4.1 Matrix Exponential

The chiral rotation operator: `e^{iδγ⁵}`

```python
def chiral_rotation_matrix(delta: float) -> np.ndarray:
    """
    Compute exact matrix exponential e^{iδγ⁵}.

    Using: γ⁵ has eigenvalues ±1, so
    e^{iδγ⁵} = cos(δ)I + i·sin(δ)γ⁵
    """
    I4 = np.eye(4, dtype=np.complex128)
    gamma5 = build_gamma5()

    return np.cos(delta) * I4 + 1j * np.sin(delta) * gamma5
```

### 4.2 Chiral Parameter Scaling

```python
def compute_chiral_phase(text: str, metadata: Dict) -> float:
    """
    Map text directionality to chiral phase δ ∈ [-π, π].

    Heuristics:
    - Left-to-right (English): δ → 0
    - Right-to-left (Arabic): δ → ±π
    - Bidirectional: δ based on dominant direction
    - Vertical (Chinese traditional): δ → ±π/2
    """
    # Extract directionality features
    ltr_score = compute_ltr_score(text)  # [0, 1]

    # Map to chiral phase
    # Center at 0 for pure LTR, rotate for RTL
    delta = π * (1 - 2 * ltr_score)

    # Apply damping for stability
    damping = metadata.get('chiral_damping', 0.5)
    return delta * damping
```

### 4.3 Validation

```python
def validate_chiral_operator(delta: float, tol: float = 1e-10):
    """Verify unitarity and properties of chiral rotation."""

    R = chiral_rotation_matrix(delta)
    gamma5 = build_gamma5()

    # Check unitarity: R†R = I
    assert np.allclose(R.conj().T @ R, np.eye(4), atol=tol)

    # Check commutation: [R, γ⁵] = 0
    commutator = R @ gamma5 - gamma5 @ R
    assert np.allclose(commutator, 0, atol=tol)

    # Check limit: δ→0 gives R→I
    R0 = chiral_rotation_matrix(0)
    assert np.allclose(R0, np.eye(4), atol=tol)
```

## 5. Evolution Algorithm

### 5.1 Time Integration Scheme

**Selected**: **Implicit-Explicit (IMEX) Runge-Kutta**

Splits equation into stiff (mass gap) and non-stiff (derivatives) parts:
- Implicit: Mass gap term `ΔRe^{iδγ⁵}Ψ` (stiff, requires stability)
- Explicit: Spatial derivatives `γⁱ∂_iΨ` (smooth, efficient)

```python
class IMEXEvolver:
    """
    2nd-order IMEX-RK scheme for Dirac equation.

    Split: ∂_tΨ = F_explicit(Ψ) + F_implicit(Ψ)
    where:
        F_explicit = -iγ⁰⁻¹(γ¹∂_x + γ²∂_y)Ψ
        F_implicit = -iγ⁰⁻¹ΔRe^{iδγ⁵}Ψ
    """

    def step(self, Ψ: np.ndarray, dt: float,
             Delta: float, delta: float) -> np.ndarray:

        # Stage 1: Explicit part
        F_exp = self.compute_explicit_term(Ψ)
        Ψ_stage = Ψ + dt * F_exp

        # Stage 2: Implicit solve
        # Solve: (I + dt·A)Ψ_new = Ψ_stage
        # where A = -iγ⁰⁻¹ΔRe^{iδγ⁵}
        A = self.build_implicit_operator(Delta, delta)
        Ψ_new = self.implicit_solve(Ψ_stage, A, dt)

        return Ψ_new
```

### 5.2 Stability Analysis

**CFL Condition** (explicit part):
```
dt ≤ C_cfl * dx² / (||γ¹|| + ||γ²||) ≈ 0.5 * dx²
```

For 512×512 grid with dx=1: `dt_max ≈ 0.5`

**Implicit Stability**: Unconditionally stable for mass gap term.

### 5.3 Complete Evolution Loop

```python
def evolve_dirac(
    Ψ_init: np.ndarray,
    Delta: float,
    R: float,
    delta: float,
    T: float = 1.0,
    dt: float = 0.01,
    tolerance: float = 1e-6
) -> Tuple[np.ndarray, Dict]:
    """
    Evolve spinor field via Dirac equation.

    Equation: (iγ^μ∂_μ)Ψ = ΔRe^{iδγ⁵}Ψ
    """

    # Initialize
    Ψ = Ψ_init.copy()
    evolver = IMEXEvolver()

    # Pre-compute operators
    gamma = GammaMatrices()
    chiral_op = chiral_rotation_matrix(delta)

    # Evolution metrics
    steps = int(T / dt)
    converged = False
    norm_history = []

    for step in range(steps):
        Ψ_old = Ψ.copy()

        # Time step
        Ψ = evolver.step(Ψ, dt, Delta * R, delta)

        # Monitor norm (should be conserved or slowly growing)
        norm = np.linalg.norm(Ψ.reshape(-1))
        norm_history.append(norm)

        # Renormalize if needed (prevent explosion)
        if norm > 10.0:
            Ψ = Ψ * (10.0 / norm)

        # Check convergence
        if step > 10 and step % 10 == 0:
            change = np.linalg.norm((Ψ - Ψ_old).reshape(-1))
            if change < tolerance:
                converged = True
                break

    return Ψ, {
        'steps_taken': step + 1,
        'converged': converged,
        'final_norm': float(norm_history[-1]),
        'norm_history': norm_history
    }
```

## 6. WaveCube Integration

### 6.1 Storage Strategy

**Selected**: **Option A - Separate Real/Imaginary Storage**

```python
def spinor_to_wavecube(Ψ_spinor: np.ndarray) -> np.ndarray:
    """
    Map 4-component complex spinor to 8-channel real storage.

    Layout:
    - Channels 0-3: Real parts of ψ₁, ψ₂, ψ₃, ψ₄
    - Channels 4-7: Imaginary parts of ψ₁, ψ₂, ψ₃, ψ₄

    This preserves full information for exact reconstruction.
    """
    wavecube = np.zeros((512, 512, 8), dtype=np.float64)

    # Real parts
    wavecube[..., 0:4] = np.real(Ψ_spinor)

    # Imaginary parts
    wavecube[..., 4:8] = np.imag(Ψ_spinor)

    return wavecube

def wavecube_to_spinor(wavecube: np.ndarray) -> np.ndarray:
    """Reconstruct complex spinor from 8-channel storage."""

    Ψ_spinor = np.zeros((512, 512, 4), dtype=np.complex128)

    # Combine real and imaginary parts
    Ψ_spinor = wavecube[..., 0:4] + 1j * wavecube[..., 4:8]

    return Ψ_spinor
```

**Compression Strategy** (future optimization):
```python
def spinor_to_quaternion_compressed(Ψ_spinor: np.ndarray) -> np.ndarray:
    """
    Compress spinor to quaternion via Cayley-Klein parameters.
    Maps SU(2) spinor → quaternion H.

    Warning: This is lossy for full Dirac spinor!
    Only use for visualization or when full fidelity not required.
    """
    # Extract upper 2 components (positive energy)
    ψ_upper = Ψ_spinor[..., :2]

    # Map to quaternion via stereographic projection
    # ... (implementation details)

    return quaternion_field
```

## 7. API Design

### 7.1 Core Interface

```python
class DiracSpinorEvolver(FieldEvolver):
    """
    Full Dirac spinor field evolver for UFT dynamics.

    Implements: (iγ^μ∂_μ)Ψ = ΔRe^{iδγ⁵}Ψ
    """

    def __init__(
        self,
        representation: str = "dirac",
        evolution_time: float = 1.0,
        dt: float = 0.01,
        integration_method: str = "imex",  # "imex" | "rk4" | "cn"
        boundary_conditions: str = "periodic",
        max_norm: float = 10.0,
        convergence_tol: float = 1e-6,
        chiral_damping: float = 0.5,
    ):
        self.representation = representation
        self.gamma = GammaMatrices(representation)
        self.evolution_time = evolution_time
        self.dt = dt
        self.integration_method = integration_method
        self.bc = boundary_conditions
        self.max_norm = max_norm
        self.convergence_tol = convergence_tol
        self.chiral_damping = chiral_damping

        # Initialize integrator
        self._init_integrator()

    def evolve(self, proto: ProtoIdentity) -> ProtoIdentity:
        """
        Evolve proto-identity via Dirac equation.

        Args:
            proto: Initial proto-identity (scalar or spinor)

        Returns:
            ProtoIdentity with evolved 4-spinor field
        """
        # Validate input
        if not self.validate_proto(proto):
            raise ValueError("Invalid proto-identity")

        # Extract/compute UFT parameters
        Delta, delta, R = self._extract_parameters(proto)

        # Convert to spinor if needed
        if proto.field.shape[-1] != 4:
            spinor = self._scalar_to_spinor(proto.field, proto.metadata)
        else:
            spinor = proto.field

        # Evolve via Dirac equation
        spinor_evolved, info = self._evolve_dirac(
            spinor, Delta, R, delta
        )

        # Package as ProtoIdentity
        metadata = proto.metadata.copy()
        metadata.update({
            'evolver_type': 'dirac_spinor',
            'representation': self.representation,
            'evolved': True,
            'spinor_components': 4,
            **info
        })

        return ProtoIdentity(field=spinor_evolved, metadata=metadata)

    def _scalar_to_spinor(
        self,
        scalar: np.ndarray,
        metadata: Dict
    ) -> np.ndarray:
        """Initialize spinor from scalar field."""
        # Implementation as specified in Section 2.3
        ...

    def _evolve_dirac(
        self,
        spinor: np.ndarray,
        Delta: float,
        R: float,
        delta: float
    ) -> Tuple[np.ndarray, Dict]:
        """Main Dirac evolution loop."""
        # Implementation as specified in Section 5.3
        ...

    def _extract_parameters(
        self,
        proto: ProtoIdentity
    ) -> Tuple[float, float, float]:
        """Extract UFT parameters from proto metadata."""

        # Mass gap from order parameter
        order_param = proto.metadata['order_parameter']
        Delta = self.mass_gap_calc.calculate(order_param)

        # Chiral phase from text
        text = proto.metadata.get('text', '')
        delta = self.chirality.analyze(text) * self.chiral_damping

        # Resonance from coupling
        R = proto.metadata.get('coupling', 3.0)

        return Delta, delta, R

    @property
    def evolver_type(self) -> str:
        return 'dirac_spinor'

    @property
    def config(self) -> Dict[str, Any]:
        return {
            'evolver_type': self.evolver_type,
            'representation': self.representation,
            'evolution_time': self.evolution_time,
            'dt': self.dt,
            'integration_method': self.integration_method,
            'boundary_conditions': self.bc,
            'max_norm': self.max_norm,
            'convergence_tol': self.convergence_tol,
            'chiral_damping': self.chiral_damping,
        }
```

### 7.2 Factory Pattern

```python
def create_evolver(mode: str = "scalar", **kwargs) -> FieldEvolver:
    """
    Factory for creating field evolvers.

    Args:
        mode: "scalar" | "spinor" | "adaptive"
        **kwargs: Mode-specific configuration

    Returns:
        FieldEvolver instance
    """

    if mode == "scalar":
        from src.uft.evolver import UFTEvolver
        return UFTEvolver(**kwargs)

    elif mode == "spinor":
        from src.uft.dirac_evolver import DiracSpinorEvolver
        return DiracSpinorEvolver(**kwargs)

    elif mode == "adaptive":
        from src.uft.adaptive_evolver import AdaptiveEvolver
        return AdaptiveEvolver(**kwargs)

    else:
        raise ValueError(f"Unknown evolver mode: {mode}")
```

## 8. Testing Strategy

### 8.1 Unit Tests

```python
# test_gamma_matrices.py
class TestGammaMatrices:
    """Test gamma matrix properties."""

    def test_anticommutation(self):
        """Verify {γ^μ, γ^ν} = 2g^{μν}"""
        gamma = GammaMatrices()

        # Minkowski metric (-, +, +, +)
        metric = np.diag([-1, 1, 1, 1])

        for mu in range(4):
            for nu in range(4):
                anticomm = (
                    gamma.gamma[mu] @ gamma.gamma[nu] +
                    gamma.gamma[nu] @ gamma.gamma[mu]
                )
                expected = 2 * metric[mu, nu] * np.eye(4)
                assert np.allclose(anticomm, expected)

    def test_gamma5_properties(self):
        """Test γ⁵ = iγ⁰γ¹γ²γ³ and (γ⁵)² = I"""
        gamma = GammaMatrices()

        # Check definition
        g5_computed = (1j * gamma.gamma[0] @ gamma.gamma[1] @
                      gamma.gamma[2] @ gamma.gamma[3])
        assert np.allclose(g5_computed, gamma.gamma[5])

        # Check square
        g5_squared = gamma.gamma[5] @ gamma.gamma[5]
        assert np.allclose(g5_squared, np.eye(4))

    def test_chiral_anticommutation(self):
        """Verify {γ⁵, γ^μ} = 0"""
        gamma = GammaMatrices()

        for mu in range(4):
            anticomm = (
                gamma.gamma[5] @ gamma.gamma[mu] +
                gamma.gamma[mu] @ gamma.gamma[5]
            )
            assert np.allclose(anticomm, np.zeros((4, 4)))
```

### 8.2 Integration Tests

```python
# test_dirac_evolution.py
class TestDiracEvolution:
    """Test full Dirac spinor evolution."""

    def test_norm_conservation(self):
        """Verify norm conservation (or controlled growth)."""
        evolver = DiracSpinorEvolver()

        # Create test spinor
        spinor = create_test_spinor()
        initial_norm = np.linalg.norm(spinor.reshape(-1))

        # Evolve
        proto = ProtoIdentity(field=spinor, metadata={...})
        evolved = evolver.evolve(proto)
        final_norm = np.linalg.norm(evolved.field.reshape(-1))

        # Check norm growth is controlled
        assert final_norm / initial_norm < 2.0

    def test_convergence(self):
        """Test convergence to steady state."""
        evolver = DiracSpinorEvolver(evolution_time=10.0)

        # High mass gap should converge quickly
        metadata = {
            'order_parameter': (0.9, 0.0),  # High synchronization
            'coupling': 5.0,
            'text': 'test'
        }

        spinor = create_test_spinor()
        proto = ProtoIdentity(field=spinor, metadata=metadata)
        evolved = evolver.evolve(proto)

        assert evolved.metadata['converged'] == True
        assert evolved.metadata['steps_taken'] < 1000

    def test_scalar_compatibility(self):
        """Verify spinor reduces to scalar for small δ."""

        # Create scalar evolver result
        scalar_evolver = UFTEvolver()
        scalar_field = create_test_field()
        scalar_proto = ProtoIdentity(field=scalar_field, metadata={...})
        scalar_evolved = scalar_evolver.evolve(scalar_proto)

        # Create spinor evolver with small chirality
        spinor_evolver = DiracSpinorEvolver(chiral_damping=0.01)
        spinor_evolved = spinor_evolver.evolve(scalar_proto)

        # Extract dominant component
        dominant = spinor_evolved.field[..., 0]

        # Should be similar to scalar result
        correlation = np.corrcoef(
            scalar_evolved.field.flatten(),
            dominant.flatten()
        )[0, 1]

        assert correlation > 0.9
```

### 8.3 Performance Tests

```python
# test_performance.py
class TestPerformance:
    """Benchmark Dirac evolution performance."""

    def test_evolution_speed(self):
        """Verify performance targets."""
        evolver = DiracSpinorEvolver()
        spinor = create_test_spinor()

        start = time.time()
        proto = ProtoIdentity(field=spinor, metadata={...})
        evolved = evolver.evolve(proto)
        elapsed = time.time() - start

        # Should complete in reasonable time
        # Target: <5 seconds for 100 steps
        assert elapsed < 5.0

        # Compare with scalar
        scalar_evolver = UFTEvolver()
        scalar_field = spinor[..., 0]  # Take one component

        start = time.time()
        scalar_proto = ProtoIdentity(field=scalar_field, metadata={...})
        scalar_evolved = scalar_evolver.evolve(scalar_proto)
        scalar_elapsed = time.time() - start

        # Spinor should be <5x slower than scalar
        slowdown = elapsed / scalar_elapsed
        assert slowdown < 5.0
```

## 9. Performance Optimization

### 9.1 Computational Analysis

**Operation Counts per Time Step**:
- Spatial derivatives: `2 × 512² × 4 × 5` = 10.5M ops (4th-order stencil)
- Gamma matrix multiplications: `4 × 512² × 4 × 16` = 67M ops
- Chiral rotation: `512² × 4 × 16` = 16.8M ops
- **Total**: ~95M ops/step

For 100 steps: ~9.5B operations

### 9.2 Optimization Strategies

**Immediate** (NumPy-based):
1. **Vectorization**: Use `np.einsum` for gamma operations
2. **Memory reuse**: Pre-allocate arrays, avoid copies
3. **Caching**: Store gamma products, chiral matrices
4. **Parallelization**: Use `numexpr` for element-wise operations

```python
def optimized_gamma_multiply(gamma: np.ndarray,
                            spinor: np.ndarray) -> np.ndarray:
    """
    Optimized gamma matrix multiplication.
    Uses einsum for efficient contraction.
    """
    # gamma: (4, 4)
    # spinor: (512, 512, 4)
    # result: (512, 512, 4)

    return np.einsum('ab,xyb->xya', gamma, spinor,
                     optimize='optimal')
```

**Future** (GPU acceleration):
```python
class CuPyDiracEvolver(DiracSpinorEvolver):
    """GPU-accelerated Dirac evolver using CuPy."""

    def _evolve_dirac(self, spinor, Delta, R, delta):
        import cupy as cp

        # Transfer to GPU
        spinor_gpu = cp.asarray(spinor)
        gamma_gpu = {k: cp.asarray(v) for k, v in self.gamma.gamma.items()}

        # Evolution on GPU
        # ... (implementation using CuPy operations)

        # Transfer back
        return spinor_gpu.get()
```

### 9.3 Adaptive Strategies

```python
class AdaptiveTimestepping:
    """Adaptive dt based on field dynamics."""

    def compute_timestep(self, spinor: np.ndarray,
                         prev_dt: float,
                         error: float) -> float:
        """
        Adjust timestep based on truncation error.

        Uses Richardson extrapolation to estimate error.
        """
        # Target relative error
        target_error = 1e-4

        # Safety factor
        safety = 0.9

        # Compute new timestep
        if error > 0:
            dt_new = prev_dt * safety * (target_error / error) ** 0.5
        else:
            dt_new = prev_dt * 1.5  # Increase if error tiny

        # Clamp to reasonable range
        dt_min, dt_max = 0.001, 0.1
        return np.clip(dt_new, dt_min, dt_max)
```

## 10. Migration Strategy

### 10.1 Backward Compatibility

```python
# src/uft/__init__.py
from .evolver import UFTEvolver  # Keep scalar evolver
from .dirac_evolver import DiracSpinorEvolver  # New spinor evolver
from .factory import create_evolver  # Factory pattern

__all__ = ['UFTEvolver', 'DiracSpinorEvolver', 'create_evolver']
```

### 10.2 CLI Integration

```python
# src/cli/commands_core.py
@click.option('--evolver-mode',
              type=click.Choice(['scalar', 'spinor', 'auto']),
              default='scalar',
              help='Field evolution mode')
def evolve(text: str, evolver_mode: str, **kwargs):
    """Evolve text through UFT dynamics."""

    # Create appropriate evolver
    evolver = create_evolver(mode=evolver_mode, **kwargs)

    # Process text
    # ... rest of implementation
```

### 10.3 Configuration Management

```yaml
# config/uft_config.yaml
evolver:
  default_mode: scalar  # Change to spinor when ready

  scalar:
    evolution_time: 1.0
    dt: 0.01
    stability_threshold: 0.001

  spinor:
    representation: dirac
    evolution_time: 1.0
    dt: 0.01
    integration_method: imex
    chiral_damping: 0.5

  auto_switch_threshold: 0.7  # Use spinor if order_param > threshold
```

## 11. Risk Assessment

### 11.1 Numerical Stability Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Norm explosion | Medium | High | Adaptive normalization, implicit methods |
| Accumulation errors | Low | Medium | Double precision, error monitoring |
| CFL violation | Low | High | Conservative dt, adaptive stepping |
| Matrix conditioning | Low | Medium | Regularization, condition monitoring |

### 11.2 Performance Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| 4-5x slowdown | High | Medium | GPU acceleration, optimization |
| Memory usage (4x) | High | Low | Streaming processing, chunking |
| Cache misses | Medium | Medium | Optimize memory layout |

### 11.3 Accuracy Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Scalar compatibility | Low | High | Extensive testing, gradual rollout |
| Chiral phase tuning | Medium | Medium | Adaptive damping, validation |
| Boundary artifacts | Low | Low | Higher-order BC, padding |

## 12. Implementation Priorities

### Phase 1: Core Infrastructure (Week 1)
1. **CRITICAL**: Gamma matrix implementation with validation
2. **CRITICAL**: Scalar-to-spinor conversion
3. **CRITICAL**: Basic Dirac operator (spatial derivatives)
4. **HIGH**: Unit tests for gamma matrices

### Phase 2: Evolution Engine (Week 2)
1. **CRITICAL**: IMEX time integration
2. **CRITICAL**: Chiral rotation operator
3. **HIGH**: Convergence monitoring
4. **HIGH**: Integration tests

### Phase 3: Integration (Week 3)
1. **HIGH**: ProtoIdentity spinor support
2. **HIGH**: WaveCube storage mapping
3. **MEDIUM**: Factory pattern implementation
4. **MEDIUM**: CLI integration

### Phase 4: Optimization (Week 4)
1. **MEDIUM**: NumPy optimization (einsum, vectorization)
2. **LOW**: GPU exploration (CuPy)
3. **LOW**: Adaptive timestepping
4. **LOW**: Performance benchmarks

### Phase 5: Validation & Rollout (Week 5)
1. **HIGH**: Comparison with scalar evolver
2. **HIGH**: End-to-end testing
3. **MEDIUM**: Documentation
4. **MEDIUM**: Gradual rollout strategy

## 13. Success Criteria

### Functional Requirements
- ✅ Correct implementation of Dirac equation
- ✅ Gamma matrices satisfy all algebraic properties
- ✅ Chiral operator is unitary
- ✅ Backward compatible with scalar evolver
- ✅ Convergence to steady states

### Performance Requirements
- ✅ <5x slowdown vs scalar
- ✅ <4x memory usage increase
- ✅ Completes 100 steps in <5 seconds (512×512)
- ✅ No memory leaks or accumulation

### Quality Requirements
- ✅ >90% correlation with scalar for δ→0
- ✅ Norm conservation or controlled growth
- ✅ Numerical stability (no NaN/Inf)
- ✅ All tests pass
- ✅ Clean API with documentation

## 14. Architecture Diagrams

### Data Flow
```
Text Input
    ↓
Kuramoto Encoder (oscillators)
    ↓
ProtoIdentity (scalar field)
    ↓
Scalar→Spinor Conversion
    ↓
┌─────────────────────────┐
│   Dirac Evolution Loop  │
│  ┌────────────────────┐ │
│  │ Spatial Derivatives│ │
│  └──────────┬─────────┘ │
│             ↓           │
│  ┌────────────────────┐ │
│  │  Gamma Operations  │ │
│  └──────────┬─────────┘ │
│             ↓           │
│  ┌────────────────────┐ │
│  │  Chiral Rotation   │ │
│  └──────────┬─────────┘ │
│             ↓           │
│  ┌────────────────────┐ │
│  │  Time Integration  │ │
│  └──────────┬─────────┘ │
│             ↓           │
│  ┌────────────────────┐ │
│  │ Convergence Check  │ │
│  └────────┬─┴─────────┘ │
│           ↓ No           │
│         Loop ←───────────┘
│           ↓ Yes
└───────────┼─────────────
            ↓
    Evolved Spinor Field
            ↓
    WaveCube Storage
            ↓
    Memory Integration
```

### Component Architecture
```
src/uft/
├── __init__.py           # Exports
├── evolver.py            # Scalar UFTEvolver (existing)
├── dirac_evolver.py      # DiracSpinorEvolver (new)
├── gamma_matrices.py     # Gamma matrix definitions
├── chiral_operators.py   # Chiral rotation implementations
├── integrators/          # Time integration schemes
│   ├── __init__.py
│   ├── imex.py          # IMEX-RK methods
│   ├── rk4.py           # Classic RK4
│   └── cn.py            # Crank-Nicolson
├── validators.py         # Property validation
├── factory.py           # Evolver factory
└── tests/
    ├── test_gamma.py
    ├── test_chiral.py
    ├── test_evolution.py
    └── test_integration.py
```

## 15. Mathematical Validation

### Dirac Equation Derivation

Starting from Lagrangian density:
```
ℒ = Ψ̄(iγ^μ∂_μ - m)Ψ
```

Euler-Lagrange gives:
```
(iγ^μ∂_μ - m)Ψ = 0
```

With our UFT modification:
```
m → ΔRe^{iδγ⁵}
```

This preserves gauge invariance under U(1) × chiral transformations.

### Non-Relativistic Limit

For v << c, upper components dominate:
```
Ψ ≈ [ψ_upper, ε·ψ_lower]ᵀ where ε ~ v/c
```

The equation reduces to:
```
i∂_tψ_upper ≈ (-∇²/2m + V)ψ_upper
```

recovering Schrödinger equation with V = ΔRcos(δ).

### Chiral Symmetry

The mass term ΔRe^{iδγ⁵} breaks chiral symmetry explicitly, generating:
- Mass gap Δ (energy scale)
- Chiral condensate ⟨Ψ̄γ⁵Ψ⟩ ≠ 0
- Topological phase δ

This matches QCD phenomenology where chiral symmetry breaking generates hadron masses.

## Conclusion

This design provides a complete blueprint for implementing Dirac spinor evolution in the Oracle UFT system. The architecture maintains backward compatibility while introducing sophisticated quantum dynamics through gamma matrices and chiral operators. The phased implementation plan ensures systematic development with continuous validation.

Key advantages:
- **Physical**: Proper relativistic quantum mechanics
- **Numerical**: Stable IMEX integration
- **Practical**: Backward compatible, testable
- **Scalable**: GPU-ready architecture

The design balances theoretical rigor with practical implementation concerns, providing a clear path from the current scalar approximation to full spinor dynamics.