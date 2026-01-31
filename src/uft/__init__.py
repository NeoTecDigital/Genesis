"""
Unified Field Theory implementation for Oracle.

Supports both scalar and full Dirac spinor evolution:

1. Scalar equation (simplified):
    i∂_tΨ + i∇Ψ = ΔRe^{iδ}Ψ

2. Dirac spinor equation (full):
    (iγ^μ∂_μ)Ψ = ΔRe^{iδγ^5}Ψ

Where:
    Ψ: Proto-identity field (scalar or 4-spinor)
    Δ: Mass gap (adaptive, from order parameter)
    R: Resonance (from Kuramoto coupling)
    δ: Chiral phase (from text directionality)
    γ^μ: Dirac gamma matrices (spinor mode only)
    γ^5: Chiral operator (spinor mode only)

Components:
- MassGapCalculator: Δ = Δ₀·r² (adaptive stability)
- ChiralityAnalyzer: δ from text semantic directionality
- UFTEvolver: Simplified scalar field evolution
- DiracSpinorEvolver: Full 4-component spinor evolution
- GammaMatrices: Dirac gamma matrices in Pauli representation
- create_evolver: Factory for creating evolvers
"""

from .mass_gap import MassGapCalculator
from .chirality import ChiralityAnalyzer
from .evolver import UFTEvolver
from .spinor_evolver import DiracSpinorEvolver
from .dirac_matrices import GammaMatrices
from .evolver_factory import create_evolver

__all__ = [
    'MassGapCalculator',
    'ChiralityAnalyzer',
    'UFTEvolver',
    'DiracSpinorEvolver',
    'GammaMatrices',
    'create_evolver'
]