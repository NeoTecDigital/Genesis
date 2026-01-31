"""Mass gap calculator for UFT dynamics."""

import numpy as np
from typing import Tuple, Optional


class MassGapCalculator:
    """
    Calculates adaptive mass gap: Δ = Δ₀·r²

    The mass gap provides natural stability selection:
    - High synchronization (r → 1) → large Δ → stable proto-identity
    - Low synchronization (r → 0) → small Δ → unstable, decays

    This implements the "Law of Selection" naturally through dynamics,
    where well-synchronized patterns persist while poorly synchronized
    patterns decay away.

    Physical interpretation:
    - Δ represents the "binding energy" of the proto-identity
    - Higher Δ means stronger binding, more stable pattern
    - Lower Δ means weaker binding, pattern disperses
    """

    def __init__(
        self,
        base_mass_gap: float = 1.0,
        power: float = 2.0,
        min_mass_gap: float = 0.01
    ):
        """
        Initialize mass gap calculator.

        Args:
            base_mass_gap (Δ₀): Base scale for mass gap
            power: Exponent for r dependence (default 2 for quadratic)
            min_mass_gap: Minimum mass gap to prevent complete decay

        Raises:
            ValueError: If parameters are invalid
        """
        if base_mass_gap <= 0:
            raise ValueError("Base mass gap must be positive")
        if power <= 0:
            raise ValueError("Power must be positive")
        if min_mass_gap < 0:
            raise ValueError("Minimum mass gap must be non-negative")
        if min_mass_gap >= base_mass_gap:
            raise ValueError("Minimum mass gap must be less than base mass gap")

        self.Delta_0 = base_mass_gap
        self.power = power
        self.min_gap = min_mass_gap

    def calculate(self, order_parameter: Tuple[float, float]) -> float:
        """
        Calculate adaptive mass gap from Kuramoto order parameter.

        Args:
            order_parameter: (r, Ψ) from Kuramoto synchronization
                r ∈ [0,1]: synchronization strength (0=incoherent, 1=fully synchronized)
                Ψ: collective phase (not used in mass gap calculation)

        Returns:
            float: Mass gap value Δ ∈ [min_gap, Δ₀]

        Formula:
            Δ = max(Δ₀·r^power, min_gap)
        """
        r, _ = order_parameter

        if not 0 <= r <= 1:
            raise ValueError(f"Order parameter r must be in [0,1], got {r}")

        Delta = self.Delta_0 * (r ** self.power)
        Delta = max(Delta, self.min_gap)  # Ensure minimum gap

        return Delta

    def calculate_from_coupling(
        self,
        order_parameter: Tuple[float, float],
        coupling: float
    ) -> float:
        """
        Calculate mass gap with coupling modulation.

        For stronger coupling, we expect stronger mass gap formation.

        Args:
            order_parameter: (r, Ψ) from Kuramoto
            coupling: Kuramoto coupling strength K

        Returns:
            float: Modulated mass gap
        """
        base_gap = self.calculate(order_parameter)

        # Modulate by coupling strength (normalized around K=1)
        # Stronger coupling → larger mass gap
        coupling_factor = 1 + 0.1 * np.log1p(coupling)
        modulated_gap = base_gap * coupling_factor

        return modulated_gap

    def stability_factor(self, mass_gap: float) -> float:
        """
        Calculate stability factor from mass gap.

        Args:
            mass_gap: Current mass gap value

        Returns:
            float: Stability factor ∈ [0, 1]
        """
        if mass_gap <= self.min_gap:
            return 0.0
        elif mass_gap >= self.Delta_0:
            return 1.0
        else:
            # Linear interpolation
            return (mass_gap - self.min_gap) / (self.Delta_0 - self.min_gap)

    @property
    def critical_synchronization(self) -> float:
        """
        Calculate critical r value for half-maximum mass gap.

        Returns:
            float: r_critical where Δ(r_critical) = Δ₀/2
        """
        return (0.5) ** (1 / self.power)