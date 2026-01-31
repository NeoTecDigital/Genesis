"""UFT field evolver - simplified scalar approximation."""

import numpy as np
from typing import Tuple, Optional, Dict, Any
import sys
sys.path.append('/home/persist/alembic/oracle')

from src.core.field_evolver import FieldEvolver
from src.core.proto_identity import ProtoIdentity
from .mass_gap import MassGapCalculator
from .chirality import ChiralityAnalyzer


class UFTEvolver(FieldEvolver):
    """
    Evolves proto-identity fields via simplified UFT dynamics.

    Implements scalar field approximation of the full spinor equation:
        Scalar: i∂_tΨ + i∇Ψ = ΔRe^{iδ}Ψ
        Full (future): (iγ^μ∂_μ)Ψ = ΔRe^{iδγ^5}Ψ

    Parameters:
        Δ: Mass gap (from Kuramoto order parameter)
        R: Resonance (from Kuramoto coupling K)
        δ: Chiral phase (from text semantic analysis)

    The evolution stabilizes well-synchronized patterns (high r → high Δ)
    while allowing poorly synchronized patterns to decay (low r → low Δ).
    """

    def __init__(
        self,
        mass_gap_calc: Optional[MassGapCalculator] = None,
        chirality_analyzer: Optional[ChiralityAnalyzer] = None,
        evolution_time: float = 1.0,
        dt: float = 0.01,
        stability_threshold: float = 0.001,
        max_norm: float = 10.0
    ):
        """
        Initialize UFT evolver.

        Args:
            mass_gap_calc: Calculator for mass gap Δ
            chirality_analyzer: Analyzer for chiral phase δ
            evolution_time: Total evolution time T
            dt: Time step for integration
            stability_threshold: Threshold for early stopping
            max_norm: Maximum field norm to prevent explosion

        Raises:
            ValueError: If parameters are invalid
        """
        if evolution_time <= 0:
            raise ValueError("Evolution time must be positive")
        if dt <= 0 or dt > evolution_time:
            raise ValueError("Time step must be positive and less than evolution time")
        if stability_threshold < 0:
            raise ValueError("Stability threshold must be non-negative")
        if max_norm <= 0:
            raise ValueError("Max norm must be positive")

        self.mass_gap_calc = mass_gap_calc or MassGapCalculator()
        self.chirality = chirality_analyzer or ChiralityAnalyzer()
        self.T = evolution_time
        self.dt = dt
        self.steps = int(self.T / self.dt)
        self.stability_threshold = stability_threshold
        self.max_norm = max_norm

    def evolve(self, proto: ProtoIdentity) -> ProtoIdentity:
        """
        Evolve proto-identity field to stable state.

        Args:
            proto: Initial proto-identity from Kuramoto encoder

        Returns:
            ProtoIdentity: Evolved proto-identity with UFT dynamics applied

        Raises:
            ValueError: If proto is invalid or missing required metadata
        """
        # Validate proto
        if not self.validate_proto(proto):
            raise ValueError("Invalid proto-identity for UFT evolution")

        # Extract parameters from metadata
        order_param = proto.metadata['order_parameter']
        r, collective_phase = order_param
        K = proto.metadata.get('coupling', 3.0)
        text = proto.metadata.get('text', '')

        # Calculate UFT parameters
        Delta = self.mass_gap_calc.calculate(order_param)
        delta = self.chirality.analyze(text)
        R = K  # Resonance coupled to Kuramoto coupling

        # Get initial field as complex
        if len(proto.field.shape) == 2:
            psi = proto.field.astype(np.complex128)
        else:
            # Convert from real/imag or quaternion
            if proto.field.shape[-1] >= 2:
                psi = proto.field[..., 0] + 1j * proto.field[..., 1]
            else:
                psi = proto.field[..., 0].astype(np.complex128)

        # Evolve via scalar field equation
        psi_evolved, evolution_info = self._evolve_scalar(psi, Delta, R, delta)

        # Create new ProtoIdentity with evolved field
        metadata = proto.metadata.copy()
        metadata.update({
            'mass_gap': Delta,
            'chiral_phase': delta,
            'resonance': R,
            'encoder_type': 'uft',
            'evolved': True,
            'evolution_steps': evolution_info['steps_taken'],
            'converged': evolution_info['converged'],
            'final_change': evolution_info['final_change']
        })

        return ProtoIdentity(field=psi_evolved, metadata=metadata)

    def _evolve_scalar(
        self,
        psi: np.ndarray,
        Delta: float,
        R: float,
        delta: float
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Evolve scalar field via: i∂_tΨ = (ΔRe^{iδ} - ∇²)Ψ

        Uses 4th-order Runge-Kutta for stability.
        Includes periodic boundary conditions and norm regularization.

        Args:
            psi: Initial complex field (512×512)
            Delta: Mass gap
            R: Resonance factor
            delta: Chiral phase

        Returns:
            Tuple of (evolved_field, evolution_info)
        """
        # Effective coupling with chiral rotation
        coupling = Delta * R * np.exp(1j * delta)

        # Initialize
        psi_current = psi.copy()
        converged = False
        steps_taken = 0

        # Pre-compute Laplacian operator (for efficiency)
        def compute_laplacian(field):
            """Compute Laplacian with periodic BC."""
            # Use numpy's gradient twice for smoother results
            grad_x = np.gradient(field, axis=1, edge_order=2)
            grad_y = np.gradient(field, axis=0, edge_order=2)
            laplacian = np.gradient(grad_x, axis=1, edge_order=2) + \
                       np.gradient(grad_y, axis=0, edge_order=2)
            return laplacian

        # RK4 derivative function
        def field_derivative(psi_state):
            """Compute dΨ/dt = -i(ΔRe^{iδ} - ∇²)Ψ"""
            laplacian = compute_laplacian(psi_state)
            return -1j * (coupling * psi_state - laplacian)

        # Evolution loop with RK4
        prev_psi = psi_current.copy()

        for step in range(self.steps):
            # RK4 integration
            k1 = field_derivative(psi_current)
            k2 = field_derivative(psi_current + 0.5 * self.dt * k1)
            k3 = field_derivative(psi_current + 0.5 * self.dt * k2)
            k4 = field_derivative(psi_current + self.dt * k3)

            psi_current = psi_current + (self.dt / 6) * (k1 + 2*k2 + 2*k3 + k4)

            # Regularization: prevent norm explosion
            current_norm = np.abs(psi_current).max()
            if current_norm > self.max_norm:
                psi_current = psi_current * (self.max_norm / current_norm)

            # Check for convergence every 10 steps
            if step % 10 == 0 and step > 0:
                change = np.abs(psi_current - prev_psi).mean()
                if change < self.stability_threshold:
                    converged = True
                    steps_taken = step + 1
                    break
                prev_psi = psi_current.copy()

        if not converged:
            steps_taken = self.steps

        # Final change for diagnostics
        final_change = np.abs(psi_current - psi).mean()

        evolution_info = {
            'steps_taken': steps_taken,
            'converged': converged,
            'final_change': float(final_change),
            'final_norm': float(np.abs(psi_current).max())
        }

        return psi_current, evolution_info

    @property
    def evolver_type(self) -> str:
        """Return evolver type identifier."""
        return 'uft'

    @property
    def config(self) -> Dict[str, Any]:
        """Return evolver configuration."""
        return {
            'evolver_type': self.evolver_type,
            'evolution_time': self.T,
            'time_step': self.dt,
            'total_steps': self.steps,
            'stability_threshold': self.stability_threshold,
            'max_norm': self.max_norm,
            'mass_gap_base': self.mass_gap_calc.Delta_0,
            'max_chiral_phase': self.chirality.delta_max
        }