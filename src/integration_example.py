"""Integration example demonstrating the modular abstraction layer."""

import sys
sys.path.append('/home/persist/alembic/oracle')

import numpy as np
from typing import Dict, Any, List

# Core abstractions
from src.core.proto_identity import ProtoIdentity
from src.core.field_encoder import FieldEncoder
from src.core.field_evolver import FieldEvolver

# Concrete implementations
from src.kuramoto.kuramoto_encoder import KuramotoEncoder
from src.uft import UFTEvolver, MassGapCalculator, ChiralityAnalyzer


def demonstrate_abstraction_layer():
    """
    Demonstrate the clean separation between abstraction and implementation.

    The modular architecture allows:
    1. Swappable encoders (Kuramoto, FFT, future implementations)
    2. Swappable evolvers (UFT, Dirac, adaptive)
    3. Clear interfaces via abstract base classes
    """
    print("="*60)
    print("ORACLE MODULAR ABSTRACTION LAYER DEMONSTRATION")
    print("="*60)

    # Initialize concrete implementations
    encoder: FieldEncoder = KuramotoEncoder(coupling_strength=3.0)
    evolver: FieldEvolver = UFTEvolver(evolution_time=1.0, dt=0.01)

    print("\n1. ABSTRACTION LAYER")
    print("-" * 40)
    print(f"Encoder Type: {encoder.encoder_type}")
    print(f"Encoder Config: {encoder.config}")
    print(f"\nEvolver Type: {evolver.evolver_type}")
    print(f"Evolver Config: {evolver.config}")

    # Test texts with different characteristics
    test_cases = [
        {
            'text': "The discovery led to a breakthrough, therefore revolutionizing the field.",
            'expected': 'forward-oriented'
        },
        {
            'text': "The project failed because the requirements were unclear.",
            'expected': 'backward-oriented'
        },
        {
            'text': "Water molecules consist of hydrogen and oxygen atoms.",
            'expected': 'neutral/descriptive'
        }
    ]

    print("\n2. ENCODING PIPELINE (Text → Proto-Identity)")
    print("-" * 40)

    protos = []
    for i, case in enumerate(test_cases, 1):
        print(f"\nCase {i}: {case['expected'].upper()}")
        print(f"Text: '{case['text'][:50]}...'")

        # Encode via Kuramoto
        proto = encoder.encode(case['text'])
        protos.append(proto)

        # Extract synchronization info
        r, psi = proto.metadata['order_parameter']
        print(f"  Kuramoto Sync: r={r:.3f}, Ψ={psi:.3f} rad")
        print(f"  Converged: {proto.metadata['converged']}")
        print(f"  Steps: {proto.metadata['steps']}")

    print("\n3. UFT EVOLUTION PIPELINE")
    print("-" * 40)

    evolved_protos = []
    for i, proto in enumerate(protos, 1):
        print(f"\nCase {i}: Evolving proto-identity...")

        # Evolve via UFT
        evolved = evolver.evolve(proto)
        evolved_protos.append(evolved)

        # Extract UFT parameters
        Delta = evolved.metadata['mass_gap']
        delta = evolved.metadata['chiral_phase']
        R = evolved.metadata['resonance']

        print(f"  Mass Gap: Δ={Delta:.3f}")
        print(f"  Chiral Phase: δ={delta:.3f} rad ({np.degrees(delta):.1f}°)")
        print(f"  Resonance: R={R:.3f}")
        print(f"  Evolution Converged: {evolved.metadata['converged']}")
        print(f"  Evolution Steps: {evolved.metadata['evolution_steps']}")

    print("\n4. PROTO-IDENTITY ANALYSIS")
    print("-" * 40)

    for i, (original, evolved) in enumerate(zip(protos, evolved_protos), 1):
        print(f"\nCase {i}: Field Properties")

        # Compare fields
        orig_field = original.field
        evol_field = evolved.field

        # Field statistics
        orig_norm = np.abs(orig_field).mean()
        evol_norm = np.abs(evol_field).mean()

        print(f"  Original Field Norm: {orig_norm:.6f}")
        print(f"  Evolved Field Norm: {evol_norm:.6f}")
        print(f"  Norm Change: {(evol_norm - orig_norm)/orig_norm * 100:.2f}%")

        # Convert to quaternion
        quat = evolved.to_quaternion()
        print(f"  Quaternion Shape: {quat.shape}")
        print(f"  Quaternion Components (mean): X={quat[...,0].mean():.3f}, "
              f"Y={quat[...,1].mean():.3f}, Z={quat[...,2].mean():.3f}, "
              f"W={quat[...,3].mean():.3f}")

    print("\n5. PARAMETER RELATIONSHIPS")
    print("-" * 40)

    # Analyze relationships between parameters
    for i, evolved in enumerate(evolved_protos, 1):
        r, _ = evolved.metadata['order_parameter']
        Delta = evolved.metadata['mass_gap']
        delta = evolved.metadata['chiral_phase']

        # Stability classification
        if Delta > 0.5:
            stability = "HIGHLY STABLE"
        elif Delta > 0.2:
            stability = "MODERATELY STABLE"
        else:
            stability = "WEAKLY STABLE"

        # Chirality classification
        if abs(delta) < 0.1:
            chirality = "NEUTRAL"
        elif delta > 0:
            chirality = "FORWARD"
        else:
            chirality = "BACKWARD"

        print(f"\nCase {i}: {stability} | {chirality}-ORIENTED")
        print(f"  r={r:.3f} → Δ={Delta:.3f} (quadratic coupling)")
        print(f"  δ={delta:.3f} rad (semantic directionality)")


def demonstrate_mass_gap_dynamics():
    """Demonstrate adaptive mass gap dynamics."""
    print("\n" + "="*60)
    print("MASS GAP DYNAMICS (Law of Selection)")
    print("="*60)

    calc = MassGapCalculator(base_mass_gap=1.0, power=2.0)

    # Test different synchronization levels
    r_values = np.linspace(0, 1, 11)

    print("\nSynchronization → Mass Gap Mapping:")
    print("-" * 40)
    print("r\tΔ\tStability")
    print("-" * 40)

    for r in r_values:
        Delta = calc.calculate((r, 0))
        stability = calc.stability_factor(Delta)

        # Visual bar
        bar_length = int(stability * 20)
        bar = "█" * bar_length + "░" * (20 - bar_length)

        print(f"{r:.1f}\t{Delta:.3f}\t{bar} {stability:.2f}")

    print(f"\nCritical synchronization (Δ = Δ₀/2): r = {calc.critical_synchronization:.3f}")


def demonstrate_chirality_analysis():
    """Demonstrate chirality analysis."""
    print("\n" + "="*60)
    print("CHIRALITY ANALYSIS (Semantic Directionality)")
    print("="*60)

    analyzer = ChiralityAnalyzer()

    # Test texts with varying directionality
    test_texts = [
        ("Strong Forward", "First we observe, then analyze, consequently understand, therefore predict."),
        ("Moderate Forward", "The experiment succeeded, thus we can proceed."),
        ("Neutral", "The sky is blue. Water is transparent. Grass grows."),
        ("Moderate Backward", "The result was unexpected because of the initial conditions."),
        ("Strong Backward", "Failure occurred due to errors, since planning was poor, because of negligence."),
        ("Mixed", "We succeeded because we planned, therefore we celebrate, since we worked hard.")
    ]

    print("\nText Directionality Analysis:")
    print("-" * 60)

    for label, text in test_texts:
        analysis = analyzer.get_detailed_analysis(text)
        delta = analysis['chiral_phase']

        # Visual indicator
        if abs(delta) < 0.05:
            arrow = "↔"
        elif delta > 0:
            arrow = "→" * min(3, int(abs(delta) * 10))
        else:
            arrow = "←" * min(3, int(abs(delta) * 10))

        print(f"\n{label}:")
        print(f"  Text: '{text[:60]}...'")
        print(f"  Chirality: δ={delta:.3f} rad ({arrow})")
        print(f"  Markers: {analysis['forward_markers']} forward, {analysis['backward_markers']} backward")
        print(f"  Interpretation: {analysis['interpretation']}")


def main():
    """Run all demonstrations."""
    # Core abstraction demo
    demonstrate_abstraction_layer()

    # Component demos
    demonstrate_mass_gap_dynamics()
    demonstrate_chirality_analysis()

    print("\n" + "="*60)
    print("DEMONSTRATION COMPLETE")
    print("="*60)
    print("\nKey Design Principles:")
    print("1. SEPARATION: Abstraction (interfaces) vs Implementation (concrete)")
    print("2. MODULARITY: Swappable encoders and evolvers")
    print("3. COMPOSABILITY: Pipeline components work together")
    print("4. PHYSICS-INSPIRED: Mass gap for stability, chirality for directionality")
    print("5. EXTENSIBILITY: Easy to add new encoders/evolvers via base classes")


if __name__ == "__main__":
    main()