"""
Configuration dataclasses and defaults for layer management policies.
"""

from typing import Dict, Any
from dataclasses import dataclass


@dataclass
class PromotionConfig:
    """Configuration for node promotion policies."""
    resonance_threshold: float = 0.8
    access_threshold: int = 10
    check_interval: int = 50  # Check every N operations


@dataclass
class DemotionConfig:
    """Configuration for node demotion policies."""
    access_threshold: int = 2
    time_threshold: int = 1000  # Age in operations
    check_interval: int = 100


@dataclass
class EvictionConfig:
    """Configuration for node eviction policies."""
    memory_threshold_mb: float = 900.0
    resonance_threshold: float = 0.3


# Default configurations as module constants
DEFAULT_PROMOTION_CONFIG = {
    'resonance_threshold': 0.8,
    'access_threshold': 10,
    'check_interval': 50
}

DEFAULT_DEMOTION_CONFIG = {
    'access_threshold': 2,
    'time_threshold': 1000,
    'check_interval': 100
}

DEFAULT_EVICTION_CONFIG = {
    'memory_threshold_mb': 900.0,
    'resonance_threshold': 0.3
}


def get_default_configs() -> Dict[str, Dict[str, Any]]:
    """Get all default configurations."""
    return {
        'promotion': DEFAULT_PROMOTION_CONFIG.copy(),
        'demotion': DEFAULT_DEMOTION_CONFIG.copy(),
        'eviction': DEFAULT_EVICTION_CONFIG.copy()
    }
