"""
Kuramoto phase oscillator dynamics solver.

Implements the Kuramoto model for coupled phase oscillators:
dθ_j/dt = ω_j + (K/N) ∑_k sin(θ_k - θ_j)

Uses 4th-order Runge-Kutta integration for numerical stability.
"""

import numpy as np
from typing import Dict, Tuple, Optional, Any


class KuramotoSolver:
    """
    Solves Kuramoto model: dθ_j/dt = ω_j + (K/N) ∑_k sin(θ_k - θ_j)

    Uses RK4 integration for numerical stability.
    """

    def __init__(
        self,
        coupling_strength: float = 1.0,
        dt: float = 0.01,
        max_steps: int = 1000,
        sync_threshold: float = 0.95
    ):
        """
        Initialize Kuramoto solver.

        Args:
            coupling_strength (K): Oscillator coupling strength
            dt: Time step for integration
            max_steps: Max iterations before timeout
            sync_threshold: Order parameter threshold for convergence (r > threshold)
        """
        if coupling_strength < 0:
            raise ValueError("Coupling strength must be non-negative")
        if dt <= 0:
            raise ValueError("Time step must be positive")
        if not 0 < sync_threshold <= 1:
            raise ValueError("Sync threshold must be in (0, 1]")

        self.K = coupling_strength
        self.dt = dt
        self.max_steps = max_steps
        self.sync_threshold = sync_threshold

    def solve(
        self,
        omegas: np.ndarray,
        initial_phases: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Evolve oscillators until synchronized.

        Args:
            omegas: Array of natural frequencies [ω_1, ..., ω_N]
            initial_phases: Optional initial phases (random if None)

        Returns:
            dict: {
                'phases': final synchronized phases [θ_1, ..., θ_N],
                'order_parameter': (r, Ψ) - synchronization measure,
                'steps': number of iterations taken,
                'converged': bool
            }
        """
        omegas = np.asarray(omegas)
        N = len(omegas)

        if N == 0:
            raise ValueError("Must provide at least one oscillator")

        # Initialize phases
        if initial_phases is None:
            # Random initial phases in [0, 2π)
            rng = np.random.RandomState(42)  # Reproducible randomness
            thetas = rng.uniform(0, 2 * np.pi, N)
        else:
            thetas = np.asarray(initial_phases)
            if len(thetas) != N:
                raise ValueError(f"Initial phases length {len(thetas)} != omegas length {N}")

        # Evolution loop
        for step in range(self.max_steps):
            # Calculate order parameter
            r, psi = self.order_parameter(thetas)

            # Check convergence
            if r > self.sync_threshold:
                return {
                    'phases': thetas % (2 * np.pi),  # Wrap to [0, 2π)
                    'order_parameter': (r, psi),
                    'steps': step,
                    'converged': True
                }

            # RK4 integration step
            thetas = self._rk4_step(thetas, omegas, self.dt)

        # Max steps reached without convergence
        r, psi = self.order_parameter(thetas)
        return {
            'phases': thetas % (2 * np.pi),
            'order_parameter': (r, psi),
            'steps': self.max_steps,
            'converged': False
        }

    def order_parameter(self, thetas: np.ndarray) -> Tuple[float, float]:
        """
        Calculate Kuramoto order parameter: r·e^{iΨ} = (1/N) ∑_j e^{iθ_j}

        Args:
            thetas: Array of phases

        Returns:
            tuple: (r, Ψ) where:
                r ∈ [0,1] - synchronization strength (1 = perfect sync)
                Ψ - collective phase angle
        """
        z = np.mean(np.exp(1j * thetas))
        return np.abs(z), np.angle(z)

    def _rk4_step(self, thetas: np.ndarray, omegas: np.ndarray, dt: float) -> np.ndarray:
        """
        Perform single RK4 integration step.

        Args:
            thetas: Current phases
            omegas: Natural frequencies
            dt: Time step

        Returns:
            Updated phases after dt
        """
        # 4th-order Runge-Kutta
        k1 = self._kuramoto_derivative(thetas, omegas)
        k2 = self._kuramoto_derivative(thetas + 0.5 * dt * k1, omegas)
        k3 = self._kuramoto_derivative(thetas + 0.5 * dt * k2, omegas)
        k4 = self._kuramoto_derivative(thetas + dt * k3, omegas)

        return thetas + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    def _kuramoto_derivative(self, thetas: np.ndarray, omegas: np.ndarray) -> np.ndarray:
        """
        Calculate dθ/dt for Kuramoto model.

        Args:
            thetas: Current phases
            omegas: Natural frequencies

        Returns:
            Phase derivatives dθ/dt
        """
        N = len(thetas)

        # Calculate coupling term: (K/N) ∑_k sin(θ_k - θ_j)
        # Vectorized for efficiency
        # theta_diff[j, k] = θ_k - θ_j
        theta_diff = thetas[np.newaxis, :] - thetas[:, np.newaxis]
        coupling = (self.K / N) * np.sum(np.sin(theta_diff), axis=1)

        # dθ_j/dt = ω_j + coupling_j
        return omegas + coupling


if __name__ == "__main__":
    # Quick validation
    print("Testing Kuramoto solver...")

    # Create solver
    solver = KuramotoSolver(coupling_strength=2.0)

    # Test with simple frequencies
    omegas = np.array([1.0, 1.1, 0.9, 1.05, 0.95])
    result = solver.solve(omegas)

    print(f"Converged: {result['converged']}")
    print(f"Steps taken: {result['steps']}")
    print(f"Order parameter r: {result['order_parameter'][0]:.3f}")
    print(f"Collective phase Ψ: {result['order_parameter'][1]:.3f}")
    print(f"Final phases: {result['phases']}")