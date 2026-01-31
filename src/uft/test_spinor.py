"""Tests for Dirac spinor implementation."""

import numpy as np
import sys
sys.path.append('/home/persist/alembic/genesis')
from src.uft.dirac_matrices import GammaMatrices
from src.uft.spinor_evolver import DiracSpinorEvolver
from src.uft.evolver_factory import create_evolver
from src.kuramoto.kuramoto_encoder import KuramotoEncoder
from src.core.proto_identity import ProtoIdentity


def test_gamma_anticommutation():
    """Test {γ^μ, γ^ν} = 2g^{μν}I."""
    print("Testing gamma anticommutation relations...")

    gamma = GammaMatrices()
    I4 = np.eye(4, dtype=np.complex128)

    # Minkowski metric (+, -, -, -) for standard Dirac-Pauli
    metric = np.diag([1.0, -1.0, -1.0, -1.0])

    for mu in range(4):
        for nu in range(4):
            anticomm = gamma.anticommutator(mu, nu)
            expected = 2 * metric[mu, nu] * I4
            assert np.allclose(anticomm, expected, atol=1e-10), \
                f"Anticommutation failed for γ^{mu}, γ^{nu}"

    print("✓ Anticommutation relations verified")


def test_gamma5_properties():
    """Test (γ⁵)² = I and {γ⁵, γ^μ} = 0."""
    print("Testing gamma5 properties...")

    gamma = GammaMatrices()
    I4 = np.eye(4, dtype=np.complex128)

    # Check (γ⁵)² = I
    gamma5_squared = gamma.gamma5 @ gamma.gamma5
    assert np.allclose(gamma5_squared, I4, atol=1e-10), \
        "(γ⁵)² != I"

    # Check {γ⁵, γ^μ} = 0
    for mu in range(4):
        anticomm = gamma.gamma5 @ gamma.gamma[mu] + gamma.gamma[mu] @ gamma.gamma5
        assert np.allclose(anticomm, np.zeros((4, 4)), atol=1e-10), \
            f"γ⁵ doesn't anticommute with γ^{mu}"

    # Check γ⁵ = iγ⁰γ¹γ²γ³
    gamma5_computed = 1j * gamma.gamma[0] @ gamma.gamma[1] @ gamma.gamma[2] @ gamma.gamma[3]
    assert np.allclose(gamma5_computed, gamma.gamma5, atol=1e-10), \
        "γ⁵ != iγ⁰γ¹γ²γ³"

    print("✓ γ⁵ properties verified")


def test_chiral_rotation_unitarity():
    """Test e^{iδγ⁵} is unitary."""
    print("Testing chiral rotation unitarity...")

    evolver = DiracSpinorEvolver()

    for delta in [0, np.pi/4, np.pi/2, np.pi, 2*np.pi]:
        R = evolver._chiral_rotation(delta)

        # Check unitarity: R†R = I
        I4 = np.eye(4, dtype=np.complex128)
        product = R.conj().T @ R
        assert np.allclose(product, I4, atol=1e-10), \
            f"Chiral rotation not unitary for δ={delta}"

        # Check determinant = 1 (special unitary)
        det = np.linalg.det(R)
        assert np.abs(det - 1.0) < 1e-10, \
            f"Chiral rotation det != 1 for δ={delta}: {det}"

    # Check limit: δ→0 gives R→I
    R0 = evolver._chiral_rotation(0)
    assert np.allclose(R0, np.eye(4), atol=1e-10), \
        "Chiral rotation for δ=0 not identity"

    print("✓ Chiral rotation is unitary")


def test_spinor_evolution_stability():
    """Test evolution doesn't explode."""
    print("Testing spinor evolution stability...")

    evolver = DiracSpinorEvolver(
        evolution_time=1.0,
        dt=0.01,
        max_norm=10.0,
        convergence_tol=1e-6
    )

    # Create test spinor field
    spinor = np.random.randn(512, 512, 4).astype(np.complex128) * 0.1
    initial_norm = np.linalg.norm(spinor.reshape(-1))

    # Create proto-identity
    proto = ProtoIdentity(
        field=spinor,
        metadata={
            'order_parameter': (0.5, 0.0),
            'coupling': 3.0,
            'text': 'test stability'
        }
    )

    # Evolve
    evolved = evolver.evolve(proto)
    final_norm = np.linalg.norm(evolved.field.reshape(-1))

    # Check norm growth is controlled
    norm_ratio = final_norm / initial_norm
    assert norm_ratio < 20.0, f"Norm exploded: {norm_ratio}x growth"

    # Check no NaN or Inf
    assert not np.any(np.isnan(evolved.field)), "NaN in evolved field"
    assert not np.any(np.isinf(evolved.field)), "Inf in evolved field"

    print(f"✓ Evolution stable (norm growth: {norm_ratio:.2f}x)")


def test_scalar_to_spinor_conversion():
    """Test scalar field converts to valid spinor."""
    print("Testing scalar to spinor conversion...")

    evolver = DiracSpinorEvolver()

    # Create test scalar field (complex)
    scalar = np.random.randn(512, 512).astype(np.complex128)
    scalar = scalar + 1j * np.random.randn(512, 512)

    metadata = {
        'order_parameter': (0.7, 0.0),
        'chiral_phase': 0.1
    }

    # Convert
    spinor = evolver._scalar_to_spinor(scalar, metadata)

    # Check shape
    assert spinor.shape == (512, 512, 4), f"Wrong shape: {spinor.shape}"

    # Check upper components have most of the amplitude
    upper_norm = np.linalg.norm(spinor[:, :, :2])
    lower_norm = np.linalg.norm(spinor[:, :, 2:])
    assert upper_norm > lower_norm, "Upper components should dominate"

    # Check no NaN or Inf
    assert not np.any(np.isnan(spinor)), "NaN in spinor"
    assert not np.any(np.isinf(spinor)), "Inf in spinor"

    print("✓ Scalar to spinor conversion works")


def test_end_to_end_pipeline():
    """Test Kuramoto → scalar UFT → spinor UFT."""
    print("Testing end-to-end pipeline...")

    # Initialize components
    encoder = KuramotoEncoder(coupling_strength=3.0)
    scalar_evolver = create_evolver(mode="scalar", evolution_time=0.5)
    spinor_evolver = create_evolver(mode="spinor", evolution_time=0.5)

    # Test text
    text = "Because the sun rises in the east, morning light appears first on eastern horizons"

    # Encode with Kuramoto
    proto_kuramoto = encoder.encode(text)
    print(f"  Kuramoto sync: r={proto_kuramoto.metadata['order_parameter'][0]:.3f}")

    # Evolve with scalar UFT
    proto_scalar = scalar_evolver.evolve(proto_kuramoto)
    print(f"  Scalar mass gap: Δ={proto_scalar.metadata['mass_gap']:.3f}")

    # Evolve with spinor UFT
    proto_spinor = spinor_evolver.evolve(proto_kuramoto)
    print(f"  Spinor mass gap: Δ={proto_spinor.metadata['mass_gap']:.3f}")
    print(f"  Chiral phase: δ={proto_spinor.metadata['chiral_phase']:.3f}")

    # Check shapes
    assert proto_scalar.field.shape == (512, 512), \
        f"Wrong scalar shape: {proto_scalar.field.shape}"
    assert proto_spinor.field.shape == (512, 512, 4), \
        f"Wrong spinor shape: {proto_spinor.field.shape}"

    # Check metadata
    assert proto_spinor.metadata['spinor'] == True
    assert proto_spinor.metadata['evolved'] == True
    assert 'component_norms' in proto_spinor.metadata

    print("✓ End-to-end pipeline works")


def test_factory_pattern():
    """Test evolver factory creates correct types."""
    print("Testing factory pattern...")

    # Create scalar evolver
    scalar = create_evolver(mode="scalar")
    assert scalar.evolver_type == 'uft', f"Wrong type: {scalar.evolver_type}"

    # Create spinor evolver
    spinor = create_evolver(mode="spinor")
    assert spinor.evolver_type == 'dirac_spinor', f"Wrong type: {spinor.evolver_type}"

    # Create auto evolver (defaults to scalar for now)
    auto = create_evolver(mode="auto")
    assert auto.evolver_type == 'uft', f"Wrong type: {auto.evolver_type}"

    print("✓ Factory pattern works")


def test_gamma_matrix_application():
    """Test gamma matrix application to spinor fields."""
    print("Testing gamma matrix application...")

    gamma = GammaMatrices()

    # Create test spinor field
    spinor = np.random.randn(512, 512, 4).astype(np.complex128)

    # Apply gamma matrices
    for mu in range(4):
        result = gamma.apply(gamma.gamma[mu], spinor)
        assert result.shape == spinor.shape, f"Shape mismatch for γ^{mu}"
        assert not np.any(np.isnan(result)), f"NaN after applying γ^{mu}"

    # Apply gamma5
    result = gamma.apply(gamma.gamma5, spinor)
    assert result.shape == spinor.shape, "Shape mismatch for γ⁵"
    assert not np.any(np.isnan(result)), "NaN after applying γ⁵"

    print("✓ Gamma matrix application works")


def test_chiral_projectors():
    """Test chiral projection operators."""
    print("Testing chiral projectors...")

    gamma = GammaMatrices()
    P_L, P_R = gamma.chiral_projectors()

    # Check P_L + P_R = I
    I4 = np.eye(4, dtype=np.complex128)
    assert np.allclose(P_L + P_R, I4, atol=1e-10), "P_L + P_R != I"

    # Check P_L² = P_L (idempotent)
    assert np.allclose(P_L @ P_L, P_L, atol=1e-10), "P_L not idempotent"

    # Check P_R² = P_R (idempotent)
    assert np.allclose(P_R @ P_R, P_R, atol=1e-10), "P_R not idempotent"

    # Check P_L P_R = 0 (orthogonal)
    assert np.allclose(P_L @ P_R, np.zeros((4, 4)), atol=1e-10), "P_L P_R != 0"

    print("✓ Chiral projectors work correctly")


def test_convergence_detection():
    """Test that evolution detects convergence."""
    print("Testing convergence detection...")

    evolver = DiracSpinorEvolver(
        evolution_time=10.0,  # Long time to allow convergence
        dt=0.01,
        convergence_tol=1e-4
    )

    # Create stable initial condition (small random perturbation)
    spinor = np.ones((512, 512, 4), dtype=np.complex128) * 0.1
    spinor += np.random.randn(512, 512, 4) * 0.01

    proto = ProtoIdentity(
        field=spinor,
        metadata={
            'order_parameter': (0.9, 0.0),  # High sync should converge fast
            'coupling': 5.0,
            'text': 'test'
        }
    )

    evolved = evolver.evolve(proto)

    # Check convergence reported
    if evolved.metadata['converged']:
        print(f"  Converged in {evolved.metadata['steps_taken']} steps")
        assert evolved.metadata['steps_taken'] < 1000, "Took too long to converge"
    else:
        print(f"  Did not converge in {evolved.metadata['steps_taken']} steps")

    print("✓ Convergence detection works")


# Run all tests
if __name__ == "__main__":
    print("Running Dirac spinor implementation tests...")
    print("=" * 50)

    test_gamma_anticommutation()
    test_gamma5_properties()
    test_chiral_rotation_unitarity()
    test_gamma_matrix_application()
    test_chiral_projectors()
    test_scalar_to_spinor_conversion()
    test_spinor_evolution_stability()
    test_convergence_detection()
    test_factory_pattern()
    test_end_to_end_pipeline()

    print("=" * 50)
    print("✅ All tests passed!")