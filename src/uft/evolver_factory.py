"""Factory for creating UFT evolvers."""

from typing import Literal, Any, Dict
from src.core.field_evolver import FieldEvolver


def create_evolver(
    mode: Literal["scalar", "spinor", "auto"] = "scalar",
    **kwargs
) -> FieldEvolver:
    """
    Create UFT evolver based on mode.

    Args:
        mode: Evolution mode:
            - "scalar": Simplified scalar UFT (existing)
            - "spinor": Full Dirac spinor UFT (new)
            - "auto": Choose based on order parameter (future)
        **kwargs: Pass to evolver constructor

    Returns:
        FieldEvolver instance

    Raises:
        ValueError: If mode is unknown
    """
    if mode == "scalar":
        from .evolver import UFTEvolver
        return UFTEvolver(**kwargs)

    elif mode == "spinor":
        from .spinor_evolver import DiracSpinorEvolver
        return DiracSpinorEvolver(**kwargs)

    elif mode == "auto":
        # Future: implement automatic mode selection based on order parameter
        # For now, default to scalar
        from .evolver import UFTEvolver
        return UFTEvolver(**kwargs)

    else:
        raise ValueError(f"Unknown evolver mode: {mode}")


def create_adaptive_evolver(
    scalar_threshold: float = 0.7,
    **kwargs
) -> FieldEvolver:
    """
    Create adaptive evolver that switches between scalar and spinor.

    Args:
        scalar_threshold: Use spinor if order_parameter[0] > threshold
        **kwargs: Pass to evolver constructors

    Returns:
        AdaptiveEvolver instance
    """
    from .adaptive_evolver import AdaptiveEvolver
    return AdaptiveEvolver(scalar_threshold=scalar_threshold, **kwargs)