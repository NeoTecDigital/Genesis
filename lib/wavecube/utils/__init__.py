"""Utility functions for wavecube."""

from .benchmarks import (
    benchmark_interpolation,
    benchmark_batch_interpolation,
    benchmark_memory_usage,
    benchmark_save_load,
    run_full_benchmark_suite
)

__all__ = [
    'benchmark_interpolation',
    'benchmark_batch_interpolation',
    'benchmark_memory_usage',
    'benchmark_save_load',
    'run_full_benchmark_suite',
]
