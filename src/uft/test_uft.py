"""Unit tests for UFT implementation."""

import sys
sys.path.append('/home/persist/alembic/oracle')

import numpy as np
import pytest
from typing import Tuple

from src.core.proto_identity import ProtoIdentity
from src.uft.mass_gap import MassGapCalculator
from src.uft.chirality import ChiralityAnalyzer
from src.uft.evolver import UFTEvolver
from src.kuramoto.kuramoto_encoder import KuramotoEncoder


class TestMassGapCalculator:
    """Test mass gap calculations."""

    def test_initialization(self):
        """Test calculator initialization."""
        calc = MassGapCalculator(base_mass_gap=2.0, power=3.0, min_mass_gap=0.1)
        assert calc.Delta_0 == 2.0
        assert calc.power == 3.0
        assert calc.min_gap == 0.1

    def test_invalid_params(self):
        """Test invalid parameter handling."""
        with pytest.raises(ValueError):
            MassGapCalculator(base_mass_gap=-1.0)  # Negative base

        with pytest.raises(ValueError):
            MassGapCalculator(power=-2.0)  # Negative power

        with pytest.raises(ValueError):
            MassGapCalculator(min_mass_gap=2.0, base_mass_gap=1.0)  # Min > base

    def test_calculate_basic(self):
        """Test basic mass gap calculation."""
        calc = MassGapCalculator(base_mass_gap=1.0, power=2.0, min_mass_gap=0.01)

        # Fully synchronized
        delta = calc.calculate((1.0, 0.0))
        assert np.isclose(delta, 1.0)  # Δ = 1.0 * 1.0^2 = 1.0

        # Half synchronized
        delta = calc.calculate((0.5, 0.0))
        assert np.isclose(delta, 0.25)  # Δ = 1.0 * 0.5^2 = 0.25

        # Unsynchronized
        delta = calc.calculate((0.0, 0.0))
        assert np.isclose(delta, 0.01)  # Min gap

    def test_calculate_with_coupling(self):
        """Test mass gap with coupling modulation."""
        calc = MassGapCalculator()

        # Standard coupling
        delta1 = calc.calculate_from_coupling((0.8, 0.0), coupling=1.0)

        # Strong coupling
        delta2 = calc.calculate_from_coupling((0.8, 0.0), coupling=5.0)

        # Stronger coupling should give larger mass gap
        assert delta2 > delta1

    def test_stability_factor(self):
        """Test stability factor calculation."""
        calc = MassGapCalculator(base_mass_gap=1.0, min_mass_gap=0.1)

        # Maximum stability
        assert calc.stability_factor(1.0) == 1.0

        # Minimum stability
        assert calc.stability_factor(0.1) == 0.0

        # Mid-range
        factor = calc.stability_factor(0.55)
        assert 0 < factor < 1

    def test_critical_synchronization(self):
        """Test critical synchronization calculation."""
        calc = MassGapCalculator(power=2.0)
        r_crit = calc.critical_synchronization

        # At critical r, mass gap should be half maximum
        delta = calc.calculate((r_crit, 0.0))
        assert np.isclose(delta, calc.Delta_0 / 2, rtol=0.01)


class TestChiralityAnalyzer:
    """Test chirality analysis."""

    def test_initialization(self):
        """Test analyzer initialization."""
        analyzer = ChiralityAnalyzer(max_chiral_phase=np.pi / 6)
        assert analyzer.delta_max == np.pi / 6
        assert len(analyzer.forward_markers) > 0
        assert len(analyzer.backward_markers) > 0

    def test_invalid_params(self):
        """Test invalid parameter handling."""
        with pytest.raises(ValueError):
            ChiralityAnalyzer(max_chiral_phase=-0.1)  # Negative

        with pytest.raises(ValueError):
            ChiralityAnalyzer(max_chiral_phase=np.pi)  # Too large

    def test_empty_text(self):
        """Test empty text handling."""
        analyzer = ChiralityAnalyzer()
        assert analyzer.analyze("") == 0.0
        assert analyzer.analyze("   ") == 0.0

    def test_forward_text(self):
        """Test forward-oriented text."""
        analyzer = ChiralityAnalyzer()

        # Strong forward causality
        text = "The experiment was successful, therefore we can conclude that the hypothesis is correct."
        delta = analyzer.analyze(text)
        assert delta > 0  # Forward-oriented

        # Multiple forward markers
        text = "First we observe, then we measure, consequently we understand, thus we predict."
        delta = analyzer.analyze(text)
        assert delta > 0

    def test_backward_text(self):
        """Test backward-oriented text."""
        analyzer = ChiralityAnalyzer()

        # Strong backward causality
        text = "The result was unexpected because the initial conditions were not properly set."
        delta = analyzer.analyze(text)
        assert delta < 0  # Backward-oriented

        # Multiple backward markers
        text = "The failure occurred due to errors, since the system was misconfigured, because of poor planning."
        delta = analyzer.analyze(text)
        assert delta < 0

    def test_neutral_text(self):
        """Test neutral/descriptive text."""
        analyzer = ChiralityAnalyzer()

        # Descriptive, no clear direction
        text = "The sky is blue. The grass is green. Water flows downhill."
        delta = analyzer.analyze(text)
        assert abs(delta) < 0.1  # Near neutral

    def test_mixed_text(self):
        """Test text with mixed directionality."""
        analyzer = ChiralityAnalyzer()

        # Balanced forward and backward
        text = "We succeeded because we planned well, therefore we can continue."
        delta = analyzer.analyze(text)
        # Should be small due to cancellation
        assert abs(delta) < analyzer.delta_max / 2

    def test_marker_counting(self):
        """Test marker counting functionality."""
        analyzer = ChiralityAnalyzer()

        text = "Therefore, we conclude that the result follows from the premise."
        markers = analyzer.get_markers_found(text)

        assert markers['forward'] >= 2  # 'therefore', 'follows'
        assert markers['backward'] == 0
        assert markers['net'] == markers['forward']

    def test_detailed_analysis(self):
        """Test detailed analysis output."""
        analyzer = ChiralityAnalyzer()

        text = "The reaction produced heat, thus the temperature increased."
        analysis = analyzer.get_detailed_analysis(text)

        assert 'chiral_phase' in analysis
        assert 'forward_markers' in analysis
        assert 'backward_markers' in analysis
        assert 'interpretation' in analysis
        assert analysis['forward_markers'] > 0


class TestUFTEvolver:
    """Test UFT field evolution."""

    def test_initialization(self):
        """Test evolver initialization."""
        evolver = UFTEvolver(
            evolution_time=0.5,
            dt=0.05,
            stability_threshold=0.01
        )
        assert evolver.T == 0.5
        assert evolver.dt == 0.05
        assert evolver.steps == 10

    def test_invalid_params(self):
        """Test invalid parameter handling."""
        with pytest.raises(ValueError):
            UFTEvolver(evolution_time=-1.0)  # Negative time

        with pytest.raises(ValueError):
            UFTEvolver(dt=2.0, evolution_time=1.0)  # dt > T

        with pytest.raises(ValueError):
            UFTEvolver(max_norm=-1.0)  # Negative norm

    def test_evolve_simple(self):
        """Test basic field evolution."""
        # Create simple proto-identity
        field = np.random.randn(512, 512) + 1j * np.random.randn(512, 512)
        field = field * 0.1  # Small amplitude

        proto = ProtoIdentity(
            field=field,
            metadata={
                'order_parameter': (0.8, 0.0),
                'coupling': 3.0,
                'text': 'Test text',
                'encoder_type': 'test'
            }
        )

        evolver = UFTEvolver(evolution_time=0.1, dt=0.01)
        evolved = evolver.evolve(proto)

        # Check metadata was updated
        assert evolved.metadata['evolved'] == True
        assert 'mass_gap' in evolved.metadata
        assert 'chiral_phase' in evolved.metadata
        assert evolved.metadata['encoder_type'] == 'uft'

        # Check field was modified
        assert not np.allclose(evolved.field, proto.field)

    def test_evolve_with_chirality(self):
        """Test evolution with different chiralities."""
        evolver = UFTEvolver(evolution_time=0.2, dt=0.02)

        # Forward-oriented text
        field1 = np.ones((512, 512), dtype=complex) * 0.1
        proto1 = ProtoIdentity(
            field=field1,
            metadata={
                'order_parameter': (0.7, 0.0),
                'text': 'Therefore, we conclude that the result follows.',
                'encoder_type': 'test'
            }
        )
        evolved1 = evolver.evolve(proto1)

        # Backward-oriented text
        field2 = np.ones((512, 512), dtype=complex) * 0.1
        proto2 = ProtoIdentity(
            field=field2,
            metadata={
                'order_parameter': (0.7, 0.0),
                'text': 'The failure occurred because of poor planning.',
                'encoder_type': 'test'
            }
        )
        evolved2 = evolver.evolve(proto2)

        # Different chiralities should produce different evolution
        assert evolved1.metadata['chiral_phase'] > 0  # Forward
        assert evolved2.metadata['chiral_phase'] < 0  # Backward
        assert not np.allclose(evolved1.field, evolved2.field)

    def test_evolve_mass_gap_effect(self):
        """Test mass gap effect on evolution."""
        evolver = UFTEvolver(evolution_time=0.2, dt=0.02)

        # High synchronization → high mass gap
        field1 = np.ones((512, 512), dtype=complex) * 0.1
        proto1 = ProtoIdentity(
            field=field1,
            metadata={
                'order_parameter': (0.95, 0.0),  # High sync
                'text': 'Test',
                'encoder_type': 'test'
            }
        )
        evolved1 = evolver.evolve(proto1)

        # Low synchronization → low mass gap
        field2 = np.ones((512, 512), dtype=complex) * 0.1
        proto2 = ProtoIdentity(
            field=field2,
            metadata={
                'order_parameter': (0.2, 0.0),  # Low sync
                'text': 'Test',
                'encoder_type': 'test'
            }
        )
        evolved2 = evolver.evolve(proto2)

        # Check mass gaps
        assert evolved1.metadata['mass_gap'] > evolved2.metadata['mass_gap']

    def test_convergence(self):
        """Test evolution convergence."""
        evolver = UFTEvolver(
            evolution_time=1.0,
            dt=0.01,
            stability_threshold=0.001
        )

        # Create stable field (high sync)
        field = np.ones((512, 512), dtype=complex) * 0.1
        proto = ProtoIdentity(
            field=field,
            metadata={
                'order_parameter': (0.9, 0.0),
                'text': 'Stable pattern',
                'encoder_type': 'test'
            }
        )

        evolved = evolver.evolve(proto)

        # High sync should converge
        if evolved.metadata['converged']:
            assert evolved.metadata['evolution_steps'] < evolver.steps

    def test_config_property(self):
        """Test configuration property."""
        evolver = UFTEvolver(evolution_time=0.5, dt=0.05)
        config = evolver.config

        assert config['evolver_type'] == 'uft'
        assert config['evolution_time'] == 0.5
        assert config['time_step'] == 0.05
        assert 'mass_gap_base' in config
        assert 'max_chiral_phase' in config


class TestIntegration:
    """Test full pipeline integration."""

    def test_kuramoto_to_uft_pipeline(self):
        """Test complete Kuramoto → UFT pipeline."""
        # Initialize components
        encoder = KuramotoEncoder(coupling_strength=3.0)
        evolver = UFTEvolver(evolution_time=0.2, dt=0.02)

        # Test text
        text = "The quick brown fox jumps over the lazy dog"

        # Encode via Kuramoto
        proto_initial = encoder.encode(text)

        # Check Kuramoto output
        assert proto_initial.metadata['encoder_type'] == 'kuramoto'
        assert 'order_parameter' in proto_initial.metadata
        r, psi = proto_initial.metadata['order_parameter']
        assert 0 <= r <= 1

        # Evolve via UFT
        proto_evolved = evolver.evolve(proto_initial)

        # Check UFT output
        assert proto_evolved.metadata['encoder_type'] == 'uft'
        assert proto_evolved.metadata['evolved'] == True
        assert 'mass_gap' in proto_evolved.metadata
        assert 'chiral_phase' in proto_evolved.metadata

        # Verify field was modified
        assert not np.allclose(proto_evolved.field, proto_initial.field)

        # Print summary
        print(f"\nPipeline Test Summary:")
        print(f"Text: '{text[:50]}...'")
        print(f"Kuramoto sync: r={r:.3f}")
        print(f"Mass gap: Δ={proto_evolved.metadata['mass_gap']:.3f}")
        print(f"Chiral phase: δ={proto_evolved.metadata['chiral_phase']:.3f} rad")
        print(f"Evolution converged: {proto_evolved.metadata.get('converged', False)}")

    def test_different_texts(self):
        """Test pipeline with different text characteristics."""
        encoder = KuramotoEncoder()
        evolver = UFTEvolver(evolution_time=0.1, dt=0.01)

        texts = [
            "A simple statement.",  # Neutral
            "Therefore, we conclude that the hypothesis is proven.",  # Forward
            "The error occurred because the system failed.",  # Backward
        ]

        results = []
        for text in texts:
            proto = encoder.encode(text)
            evolved = evolver.evolve(proto)
            results.append({
                'text': text[:30],
                'r': proto.metadata['order_parameter'][0],
                'mass_gap': evolved.metadata['mass_gap'],
                'chiral_phase': evolved.metadata['chiral_phase']
            })

        # Different texts should produce different results
        mass_gaps = [r['mass_gap'] for r in results]
        chiral_phases = [r['chiral_phase'] for r in results]

        assert len(set(np.round(mass_gaps, 3))) > 1  # Different mass gaps
        assert max(chiral_phases) > 0  # Some forward
        assert min(chiral_phases) < 0  # Some backward


def run_tests():
    """Run all tests with pytest."""
    import subprocess
    result = subprocess.run(
        ['python', '-m', 'pytest', __file__, '-v', '--tb=short'],
        capture_output=True,
        text=True
    )
    print(result.stdout)
    if result.stderr:
        print(result.stderr)
    return result.returncode == 0


if __name__ == '__main__':
    # Run integration demo
    print("=== UFT Integration Demo ===\n")

    encoder = KuramotoEncoder(coupling_strength=3.0)
    evolver = UFTEvolver(evolution_time=0.5, dt=0.01)

    test_texts = [
        "The experiment succeeded, therefore we can proceed with production.",
        "The system failed because the configuration was incorrect.",
        "Quantum fields exhibit both particle and wave properties.",
    ]

    for text in test_texts:
        print(f"\nText: '{text[:60]}...'")

        # Encode
        proto_initial = encoder.encode(text)
        r, psi = proto_initial.metadata['order_parameter']
        print(f"  Kuramoto: r={r:.3f}, Ψ={psi:.3f}")

        # Evolve
        proto_evolved = evolver.evolve(proto_initial)
        print(f"  Mass gap: Δ={proto_evolved.metadata['mass_gap']:.3f}")
        print(f"  Chirality: δ={proto_evolved.metadata['chiral_phase']:.3f} rad")
        print(f"  Converged: {proto_evolved.metadata.get('converged', False)}")

    print("\n" + "="*50)
    print("Running unit tests...\n")
    run_tests()