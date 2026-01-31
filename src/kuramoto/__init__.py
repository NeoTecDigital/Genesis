"""
Kuramoto phase oscillator dynamics for Oracle memory system.

Based on theoretical framework from reference images:
- Core fundamental: (iγ^μ∂_μ)Ψ = [ℏω/c²·R(t)]Ψ
- Universal formula: Heisenberg + Dirac + Kuramoto + Einstein
- Path conditions: Queries must go through proto-identity Φ

This module implements:
1. KuramotoSolver: Numerical integration of coupled phase oscillators
2. TextToOscillators: Text encoding into natural frequencies
3. ProtoIdentityGenerator: Synchronized phases → spatial interference fields

The Kuramoto model provides a natural way to encode text into coherent
proto-identities through phase synchronization. The coupling strength K
controls how strongly oscillators influence each other, leading to
emergent collective behavior that captures the essence of the input text.

Mathematical Foundation:
- Kuramoto equation: dθ_j/dt = ω_j + (K/N) ∑_k sin(θ_k - θ_j)
- Order parameter: r·e^{iΨ} = (1/N) ∑_j e^{iθ_j}
- Interference field: Φ(r) = ∑_j A_j·e^{i(k_j·r + θ_j)}

Usage:
    from kuramoto import KuramotoSolver, TextToOscillators, ProtoIdentityGenerator

    # Encode text
    encoder = TextToOscillators()
    omegas = encoder.encode("Hello world")

    # Synchronize oscillators
    solver = KuramotoSolver(coupling_strength=2.0)
    result = solver.solve(omegas)

    # Generate proto-identity
    generator = ProtoIdentityGenerator()
    proto_id = generator.generate(result['phases'], text="Hello world")
"""

from .dynamics import KuramotoSolver
from .text_encoder import TextToOscillators
from .proto_identity import ProtoIdentityGenerator

__version__ = "0.1.0"
__all__ = ['KuramotoSolver', 'TextToOscillators', 'ProtoIdentityGenerator']

# Default configurations
DEFAULT_COUPLING_STRENGTH = 2.0
DEFAULT_SYNC_THRESHOLD = 0.95
DEFAULT_RESOLUTION = (512, 512)