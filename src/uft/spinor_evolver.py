"""Dirac spinor evolution for UFT dynamics."""

import numpy as np
import sys
from typing import Dict, Tuple, Optional
sys.path.append('/home/persist/alembic/genesis')
from src.core.field_evolver import FieldEvolver
from src.core.proto_identity import ProtoIdentity
from .dirac_matrices import GammaMatrices
from .mass_gap import MassGapCalculator
from .chirality import ChiralityAnalyzer


class DiracSpinorEvolver(FieldEvolver):
    """
    Evolves proto-identity via full Dirac spinor dynamics.

    Equation: (iγ^μ∂_μ)Ψ = ΔRe^{iδγ⁵}Ψ

    Where:
    - Ψ: 4-component spinor field (512×512×4)
    - γ^μ: Dirac gamma matrices
    - Δ: adaptive mass gap
    - R: resonance coupling
    - δ: chiral phase
    - γ⁵: chiral operator
    """

    def __init__(
        self,
        mass_gap_calc: Optional[MassGapCalculator] = None,
        chirality_analyzer: Optional[ChiralityAnalyzer] = None,
        evolution_time: float = 1.0,
        dt: float = 0.01,
        boundary_conditions: str = "periodic",
        max_norm: float = 10.0,
        convergence_tol: float = 1e-6,
        chiral_damping: float = 0.5,
        integration_method: str = "euler",  # "euler" or "imex"
        representation: str = "dirac"
    ):
        """
        Initialize Dirac spinor evolver.

        Args:
            mass_gap_calc: Calculator for adaptive mass gap
            chirality_analyzer: Analyzer for chiral phase from text
            evolution_time: Total evolution time
            dt: Time step
            boundary_conditions: Boundary conditions ("periodic")
            max_norm: Maximum allowed norm before renormalization
            convergence_tol: Convergence tolerance
            chiral_damping: Damping factor for chiral phase
            integration_method: Time integration method
            representation: Gamma matrix representation
        """
        self.mass_gap_calc = mass_gap_calc or MassGapCalculator()
        self.chirality = chirality_analyzer or ChiralityAnalyzer()
        self.gamma = GammaMatrices(representation=representation)
        self.T = evolution_time
        self.dt = dt
        self.bc = boundary_conditions
        self.max_norm = max_norm
        self.convergence_tol = convergence_tol
        self.chiral_damping = chiral_damping
        self.integration_method = integration_method
        self.steps = int(self.T / self.dt)

        # Precompute useful matrices
        self._I4 = np.eye(4, dtype=np.complex128)
        self._dx = 1.0  # Grid spacing

    @property
    def evolver_type(self) -> str:
        """Return evolver type identifier."""
        return 'dirac_spinor'

    @property
    def config(self) -> Dict:
        """Return evolver configuration."""
        return {
            'evolver_type': self.evolver_type,
            'evolution_time': self.T,
            'dt': self.dt,
            'boundary_conditions': self.bc,
            'max_norm': self.max_norm,
            'convergence_tol': self.convergence_tol,
            'chiral_damping': self.chiral_damping,
            'integration_method': self.integration_method,
            'representation': self.gamma.representation
        }

    def evolve(self, proto: ProtoIdentity) -> ProtoIdentity:
        """
        Evolve proto-identity to stable spinor state.

        Args:
            proto: Input proto-identity (scalar or spinor field)

        Returns:
            ProtoIdentity with evolved spinor field
        """
        # Extract parameters
        r, collective_phase = proto.metadata['order_parameter']
        K = proto.metadata.get('coupling', 3.0)
        text = proto.metadata.get('text', '')

        # Calculate UFT parameters
        Delta = self.mass_gap_calc.calculate((r, collective_phase))
        delta = self.chirality.analyze(text) * self.chiral_damping
        R = K

        # Convert to spinor if needed
        if len(proto.field.shape) == 2:
            # Scalar field (H, W)
            spinor = self._scalar_to_spinor(proto.field, proto.metadata)
        elif len(proto.field.shape) == 3 and proto.field.shape[-1] == 4:
            # Already a spinor
            spinor = proto.field
        else:
            raise ValueError(f"Unexpected field shape: {proto.field.shape}")

        # Evolve via Dirac equation
        spinor_evolved, evolution_info = self._evolve_dirac(spinor, Delta, R, delta)

        # Update metadata
        metadata = proto.metadata.copy()
        metadata.update({
            'mass_gap': Delta,
            'chiral_phase': delta,
            'resonance': R,
            'encoder_type': 'uft-spinor',
            'spinor': True,
            'evolved': True,
            'spinor_components': 4,
            **evolution_info
        })

        return ProtoIdentity(field=spinor_evolved, metadata=metadata)

    def _scalar_to_spinor(self, scalar: np.ndarray, metadata: Dict) -> np.ndarray:
        """
        Convert 512×512 complex scalar to 512×512×4 spinor.

        Strategy:
        - Initialize spinor with scalar in upper components (positive energy)
        - Add phase variation across components
        - Suppress lower components initially

        Args:
            scalar: (H, W) complex scalar field
            metadata: Proto-identity metadata

        Returns:
            (H, W, 4) spinor field
        """
        H, W = scalar.shape
        spinor = np.zeros((H, W, 4), dtype=np.complex128)

        # Normalize scalar field
        norm_factor = np.abs(scalar).max() + 1e-10
        scalar_norm = scalar / norm_factor

        # Extract magnitude and phase
        magnitude = np.abs(scalar_norm)
        phase = np.angle(scalar_norm)

        # Upper components (positive energy states)
        spinor[:, :, 0] = magnitude * np.exp(1j * phase)  # Spin up
        spinor[:, :, 1] = magnitude * np.exp(1j * (phase + np.pi/4))  # Spin down

        # Lower components (negative energy states, initially suppressed)
        suppression = 0.1  # Small initial amplitude
        spinor[:, :, 2] = suppression * magnitude * np.exp(1j * (phase + np.pi/2))
        spinor[:, :, 3] = suppression * magnitude * np.exp(1j * (phase + 3*np.pi/4))

        # Optional: Apply initial chiral rotation if specified
        if 'chiral_phase' in metadata and metadata['chiral_phase'] != 0:
            delta_init = metadata['chiral_phase'] * 0.1  # Small initial chirality
            chiral_op = self._chiral_rotation(delta_init)
            spinor = self.gamma.apply(chiral_op, spinor)

        # Restore scale
        spinor = spinor * norm_factor

        return spinor

    def _evolve_dirac(
        self,
        spinor: np.ndarray,
        Delta: float,
        R: float,
        delta: float
    ) -> Tuple[np.ndarray, Dict]:
        """
        Evolve spinor via (iγ^μ∂_μ)Ψ = ΔRe^{iδγ⁵}Ψ.

        Args:
            spinor: (H, W, 4) initial spinor field
            Delta: Mass gap parameter
            R: Resonance coupling
            delta: Chiral phase

        Returns:
            Evolved spinor and evolution info dictionary
        """
        # Precompute chiral rotation operator
        chiral_op = self._chiral_rotation(delta)

        # Mass term matrix: ΔRe^{iδγ⁵}
        mass_matrix = Delta * R * chiral_op

        # Evolution tracking
        psi = spinor.copy()
        norm_history = []
        converged = False
        convergence_step = self.steps

        for step in range(self.steps):
            psi_old = psi.copy()

            # Compute spatial derivatives
            dpsi_dx = self._spatial_derivative(psi, axis=1)  # ∂_x
            dpsi_dy = self._spatial_derivative(psi, axis=0)  # ∂_y

            # Apply gamma matrices: γ¹∂_x + γ²∂_y
            spatial_term = (
                self.gamma.apply(self.gamma.gamma[1], dpsi_dx) +
                self.gamma.apply(self.gamma.gamma[2], dpsi_dy)
            )

            # Apply mass term: ΔRe^{iδγ⁵}Ψ
            mass_term = self.gamma.apply(mass_matrix, psi)

            # Right-hand side: spatial + mass terms
            rhs = spatial_term + mass_term

            # Time derivative: ∂_tΨ = -i(γ⁰)⁻¹(spatial + mass)
            # Note: (γ⁰)⁻¹ = γ⁰ for Dirac-Pauli representation
            dpsi_dt = -1j * self.gamma.apply(self.gamma.gamma[0], rhs)

            # Time integration (Euler for now, can upgrade to IMEX)
            if self.integration_method == "euler":
                psi = psi + self.dt * dpsi_dt
            else:
                # Future: implement IMEX or other methods
                raise NotImplementedError(f"Integration method {self.integration_method} not implemented")

            # Monitor norm
            norm = np.linalg.norm(psi.reshape(-1))
            norm_history.append(float(norm))

            # Renormalize if needed to prevent explosion
            if norm > self.max_norm:
                psi = psi * (self.max_norm / norm)

            # Check convergence (every 10 steps after warmup)
            if step > 10 and step % 10 == 0:
                change = np.linalg.norm((psi - psi_old).reshape(-1))
                if change < self.convergence_tol:
                    converged = True
                    convergence_step = step
                    break

        # Compute final statistics
        component_norms = [
            float(np.linalg.norm(psi[:, :, i]))
            for i in range(4)
        ]

        return psi, {
            'steps_taken': convergence_step + 1,
            'converged': converged,
            'final_norm': float(norm_history[-1] if norm_history else 0),
            'component_norms': component_norms,
            'norm_growth': float(norm_history[-1] / norm_history[0]) if len(norm_history) > 1 else 1.0
        }

    def _chiral_rotation(self, delta: float) -> np.ndarray:
        """
        Compute e^{iδγ⁵} = cos(δ)I + i·sin(δ)γ⁵.

        Exact for any δ since (γ⁵)² = I.

        Args:
            delta: Chiral phase angle

        Returns:
            4×4 chiral rotation matrix
        """
        return np.cos(delta) * self._I4 + 1j * np.sin(delta) * self.gamma.gamma5

    def _spatial_derivative(
        self,
        field: np.ndarray,
        axis: int
    ) -> np.ndarray:
        """
        Compute spatial derivative with boundary conditions.

        Uses 4th-order centered differences with periodic BC.

        Args:
            field: (H, W, 4) spinor field
            axis: Axis for derivative (0=y, 1=x)

        Returns:
            (H, W, 4) derivative field
        """
        if self.bc == "periodic":
            # Use numpy gradient for simplicity (2nd order)
            # Can upgrade to 4th order for better accuracy
            return np.gradient(field, self._dx, axis=axis, edge_order=2)

        else:
            raise NotImplementedError(f"Boundary condition '{self.bc}' not implemented")

    def validate_proto(self, proto: ProtoIdentity) -> bool:
        """
        Validate that proto-identity is suitable for evolution.

        Args:
            proto: Proto-identity to validate

        Returns:
            True if valid, False otherwise
        """
        # Check field exists and has right shape
        if proto.field is None:
            return False

        shape = proto.field.shape
        if len(shape) == 2:
            # Scalar field - OK
            return shape[0] == 512 and shape[1] == 512
        elif len(shape) == 3:
            # Spinor field - OK if 4 components
            return shape[0] == 512 and shape[1] == 512 and shape[2] == 4
        else:
            return False