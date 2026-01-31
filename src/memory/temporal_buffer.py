"""
Temporal Buffer - Circular buffer for temporal proto-identity sequences.

Tracks proto-identity evolution over time and enables temporal derivative computation
and Taylor series prediction.
"""

import numpy as np
from typing import List, Tuple, Optional


class TemporalBuffer:
    """Circular buffer for temporal proto-identity sequences.

    Stores (timestamp, proto_identity) pairs and provides methods for:
    - Temporal derivative computation (∂proto/∂t, ∂²proto/∂t²)
    - Taylor series prediction for future states
    """

    def __init__(self, max_length: int = 100):
        """Initialize temporal buffer.

        Args:
            max_length: Maximum number of temporal entries to store
        """
        self.buffer: List[Tuple[float, np.ndarray]] = []
        self.max_length = max_length

    def add(self, proto: np.ndarray, timestamp: float) -> None:
        """Add proto-identity to buffer with timestamp.

        Implements circular buffer logic - oldest entries are removed
        when max_length is exceeded.

        Args:
            proto: Proto-identity array (H, W, 4)
            timestamp: Timestamp in seconds (unix time or relative)
        """
        self.buffer.append((timestamp, proto.copy()))

        # Circular buffer: remove oldest if exceeded
        if len(self.buffer) > self.max_length:
            self.buffer.pop(0)

    def get_derivatives(self, order: int = 1) -> Optional[np.ndarray]:
        """Compute temporal derivatives using finite differences.

        - order=1: ∂proto/∂t (first derivative - velocity)
        - order=2: ∂²proto/∂t² (second derivative - acceleration)

        Uses central differences for interior points, forward/backward
        differences for boundaries.

        Args:
            order: Derivative order (1 or 2)

        Returns:
            Derivative array with same shape as proto-identity, or None if insufficient data
        """
        if len(self.buffer) < 2:
            return None

        if order == 1:
            return self._compute_first_derivative()
        elif order == 2:
            return self._compute_second_derivative()
        else:
            raise ValueError(f"Unsupported derivative order: {order}. Use 1 or 2.")

    def _compute_first_derivative(self) -> np.ndarray:
        """Compute first derivative (∂proto/∂t) using finite differences.

        Uses backward difference at the most recent point:
        ∂proto/∂t ≈ (proto[n] - proto[n-1]) / Δt
        """
        # Get last two entries
        t_prev, proto_prev = self.buffer[-2]
        t_curr, proto_curr = self.buffer[-1]

        dt = t_curr - t_prev
        if dt <= 0:
            # Timestamps not monotonic or zero delta - return zeros
            return np.zeros_like(proto_curr)

        # Backward difference
        derivative = (proto_curr - proto_prev) / dt
        return derivative

    def _compute_second_derivative(self) -> Optional[np.ndarray]:
        """Compute second derivative (∂²proto/∂t²) using finite differences.

        Uses central difference formula:
        ∂²proto/∂t² ≈ (proto[n+1] - 2*proto[n] + proto[n-1]) / Δt²

        For most recent point, uses backward differences:
        ∂²proto/∂t² ≈ (∂proto/∂t[n] - ∂proto/∂t[n-1]) / Δt
        """
        if len(self.buffer) < 3:
            return None

        # Get last three entries
        t_0, proto_0 = self.buffer[-3]
        t_1, proto_1 = self.buffer[-2]
        t_2, proto_2 = self.buffer[-1]

        # Compute time deltas
        dt_01 = t_1 - t_0
        dt_12 = t_2 - t_1

        if dt_01 <= 0 or dt_12 <= 0:
            return np.zeros_like(proto_2)

        # Compute first derivatives at two points
        deriv_1 = (proto_1 - proto_0) / dt_01
        deriv_2 = (proto_2 - proto_1) / dt_12

        # Second derivative from difference of first derivatives
        dt_avg = (dt_01 + dt_12) / 2
        second_deriv = (deriv_2 - deriv_1) / dt_avg

        return second_deriv

    def predict_next(self, delta_t: float, order: int = 2) -> Optional[np.ndarray]:
        """Taylor series prediction: proto(t+Δt).

        - order=1: proto(t+Δt) ≈ proto(t) + ∂proto/∂t · Δt
        - order=2: proto(t+Δt) ≈ proto(t) + ∂proto/∂t · Δt + 0.5 · ∂²proto/∂t² · Δt²

        Args:
            delta_t: Time step into the future (seconds)
            order: Taylor series order (1 or 2)

        Returns:
            Predicted proto-identity, or None if insufficient data
        """
        if len(self.buffer) < 2:
            return None

        # Current state
        _, proto_current = self.buffer[-1]

        # First derivative
        deriv_1 = self.get_derivatives(order=1)
        if deriv_1 is None:
            return None

        # First-order Taylor series
        predicted = proto_current + deriv_1 * delta_t

        # Second-order Taylor series
        if order >= 2:
            deriv_2 = self.get_derivatives(order=2)
            if deriv_2 is not None:
                predicted += 0.5 * deriv_2 * (delta_t ** 2)

        return predicted

    def clear(self) -> None:
        """Clear all entries from buffer."""
        self.buffer.clear()

    def __len__(self) -> int:
        """Number of entries in buffer."""
        return len(self.buffer)

    def __repr__(self) -> str:
        """String representation."""
        if len(self.buffer) == 0:
            return f"TemporalBuffer(empty, max_length={self.max_length})"

        t_start, _ = self.buffer[0]
        t_end, _ = self.buffer[-1]
        duration = t_end - t_start

        return f"TemporalBuffer({len(self.buffer)} entries, {duration:.2f}s span, max_length={self.max_length})"
