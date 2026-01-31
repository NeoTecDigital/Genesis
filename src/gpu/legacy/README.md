# Legacy GPU Code - Archived 2025-11-26

## Why Archived

This C++ batch pipeline implementation was never integrated with the active system and had fundamental architectural issues.

**Issues**:
1. **Not in use**: Python code couldn't load it (FFI symbol mismatch)
2. **Wrong architecture**: Batched γ parameters (violates uniqueness constraint)
3. **Build broken**: Rust lib doesn't re-export C symbols
4. **Superseded**: CPUPipeline is the working implementation

## What to use instead

- **CPU**: `src/pipeline/cpu.py` (active, architecturally correct)
- **GPU future**: Will be implemented in Rust or Python+ROCm when needed

## Files archived
- batch_pipeline.cpp (119 KB - 2,834 lines)
- batch_pipeline.h (17 KB)
- batch_pipeline.o (object file)
- test_batch_pipeline (test binary)

## Archival context

This decision removes dead code that was:
- Developed pre-Origin refactor
- Never productionized (Python-to-C++ FFI never materialized)
- Architecturally incompatible with the cycle structure

The Origin architecture shift (commit d86f9ee) identified that batched processing violates the uniqueness constraint on γ parameters, making this implementation fundamentally unsound.

**Next steps for GPU acceleration**:
1. If GPU support needed, implement in Rust (leverage ROCm bindings)
2. Keep Python as primary layer for now (meets performance requirements)
3. Profile before optimizing (premature optimization risk)
