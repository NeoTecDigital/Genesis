"""Kuramoto encoder implementing FieldEncoder interface."""

import sys
sys.path.append('/home/persist/alembic/oracle')

import numpy as np
from typing import Dict, Any, Optional

from src.core.field_encoder import FieldEncoder
from src.core.proto_identity import ProtoIdentity
from .dynamics import KuramotoSolver
from .text_encoder import TextToOscillators
from .proto_identity import ProtoIdentityGenerator


class KuramotoEncoder(FieldEncoder):
    """
    Kuramoto phase oscillator encoder.

    Encodes text via coupled phase oscillator synchronization.

    Pipeline:
    1. Text → natural frequencies ω_j (via hash-based assignment)
    2. Kuramoto synchronization → synchronized phases θ_j
    3. Wave interference → proto-identity field Ψ(x,y)

    The resulting proto-identity captures:
    - Collective synchronization (order parameter r)
    - Phase coherence patterns
    - Spatial interference structure
    """

    def __init__(
        self,
        coupling_strength: float = 3.0,
        resolution: tuple = (512, 512),
        dt: float = 0.01,
        max_steps: int = 1000,
        sync_threshold: float = 0.95
    ):
        """
        Initialize Kuramoto encoder.

        Args:
            coupling_strength (K): Oscillator coupling strength
            resolution: Proto-identity grid size (height, width)
            dt: Time step for Kuramoto integration
            max_steps: Maximum synchronization steps
            sync_threshold: Order parameter threshold for convergence

        Raises:
            ValueError: If parameters are invalid
        """
        if coupling_strength < 0:
            raise ValueError("Coupling strength must be non-negative")
        if len(resolution) != 2 or resolution[0] <= 0 or resolution[1] <= 0:
            raise ValueError("Resolution must be positive (height, width) tuple")

        self.K = coupling_strength
        self.resolution = resolution

        # Initialize components
        self.text_encoder = TextToOscillators()
        self.solver = KuramotoSolver(
            coupling_strength=coupling_strength,
            dt=dt,
            max_steps=max_steps,
            sync_threshold=sync_threshold
        )
        self.generator = ProtoIdentityGenerator(resolution=resolution)

    def encode(self, text: str) -> ProtoIdentity:
        """
        Encode text via Kuramoto synchronization.

        Args:
            text: Input text to encode

        Returns:
            ProtoIdentity: Contains complex field and synchronization metadata

        Raises:
            ValueError: If text is empty
        """
        if not text or len(text.strip()) == 0:
            raise ValueError("Cannot encode empty text")

        # Step 1: Text → natural frequencies
        omegas = self.text_encoder.encode(text)

        # Step 2: Synchronize oscillators
        result = self.solver.solve(omegas)

        # Step 3: Generate proto-identity field from synchronized phases
        field = self.generator.generate(result['phases'], text=text)

        # Package as ProtoIdentity with metadata
        metadata = {
            'order_parameter': result['order_parameter'],  # (r, Ψ)
            'coupling': self.K,
            'steps': result['steps'],
            'converged': result['converged'],
            'encoder_type': 'kuramoto',
            'text': text,
            'num_oscillators': len(omegas),
            'mean_frequency': float(np.mean(omegas)),
            'frequency_variance': float(np.var(omegas))
        }

        return ProtoIdentity(field=field, metadata=metadata)

    @property
    def encoder_type(self) -> str:
        """Return encoder type identifier."""
        return 'kuramoto'

    @property
    def config(self) -> Dict[str, Any]:
        """Return encoder configuration."""
        return {
            'encoder_type': self.encoder_type,
            'coupling_strength': self.K,
            'resolution': self.resolution,
            'solver_config': {
                'dt': self.solver.dt,
                'max_steps': self.solver.max_steps,
                'sync_threshold': self.solver.sync_threshold
            }
        }

    def get_synchronization_info(self, proto: ProtoIdentity) -> Dict[str, Any]:
        """
        Extract synchronization information from proto-identity.

        Args:
            proto: Proto-identity to analyze

        Returns:
            Dict with synchronization metrics
        """
        if proto.metadata.get('encoder_type') != 'kuramoto':
            raise ValueError("Proto-identity not from Kuramoto encoder")

        r, psi = proto.metadata['order_parameter']

        return {
            'order_parameter_magnitude': r,
            'collective_phase': psi,
            'synchronization_quality': 'high' if r > 0.9 else 'medium' if r > 0.5 else 'low',
            'converged': proto.metadata.get('converged', False),
            'steps_to_converge': proto.metadata.get('steps', 0),
            'coupling_strength': proto.metadata.get('coupling', 0)
        }