# Contributing to Genesis

Thank you for your interest in contributing to Genesis. This guide outlines our development standards and contribution workflow.

## Table of Contents

- [Code Standards](#code-standards)
- [Development Environment](#development-environment)
- [Development Workflow](#development-workflow)
- [Testing](#testing)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)
- [Architecture Principles](#architecture-principles)

---

## Code Standards

### Structural Limits (MANDATORY)

All code must adhere to these hard limits:

- **Files**: Maximum 500 lines
  - Split into modules if exceeded
  - Use clear module boundaries

- **Functions**: Maximum 50 lines
  - Extract subfunctions if exceeded
  - Single responsibility principle

- **Nesting**: Maximum 3 levels of indentation
  - Abstract into functions/modules if exceeded
  - Use early returns to reduce nesting

**Example of proper abstraction:**

```python
# BAD - exceeds nesting limit
def process_data(data):
    if data:
        for item in data:
            if item.valid:
                if item.score > 0.5:
                    return item.value
    return None

# GOOD - reduced nesting via early returns and extraction
def process_data(data):
    if not data:
        return None

    for item in data:
        value = process_item(item)
        if value is not None:
            return value

    return None

def process_item(item):
    if not item.valid:
        return None
    if item.score <= 0.5:
        return None
    return item.value
```

### Code Quality Requirements

- **Clean, readable, self-documenting code**
  - Descriptive naming (no cryptic abbreviations like `tmp_x`, `proc_buf`)
  - Clear function/variable names that express intent

- **Comprehensive error handling**
  - No silent failures
  - Meaningful error messages
  - Proper exception types

- **Proper logging**
  - Appropriate log levels (DEBUG, INFO, WARNING, ERROR)
  - Contextual information in log messages

- **Zero compiler/linter warnings tolerated**
  - Run linters before committing
  - Fix all warnings, don't suppress

- **No TODO comments in production code**
  - Either implement or create an issue
  - TODOs allowed only in WIP branches

### Naming Conventions

```python
# Classes: PascalCase
class VoxelCloud:
    pass

# Functions/methods: snake_case
def compute_coherence_state(instance):
    pass

# Constants: UPPER_SNAKE_CASE
MAX_ITERATIONS = 100
CONVERGENCE_THRESHOLD = 0.95

# Private members: leading underscore
def _internal_helper(data):
    pass

# Descriptive names over short names
# BAD
def proc_freq(f, h):
    pass

# GOOD
def process_frequency_spectrum(frequency, harmonics):
    pass
```

---

## Development Environment

### Package Management

**ALWAYS use uv/uvx - NEVER use venv, pip, or pyenv**

We're on Arch Linux and use uv for all Python package management.

```bash
# Install dependencies
uv pip install <package>

# Run scripts
uvx <command>
uv run <script>
```

### Hardware Context

- **Platform**: Arch Linux with Sway/Wayland
- **GPU**: ROCm (AMD GPU)
- **Sleep Management**: systemd-inhibit + swayidle (auto-handled by train.py)

### Resource Limits

**CRITICAL: Check Before GPU Use**

NEVER run a model or use the GPU without FIRST verifying it's not already in use:

```bash
# Check GPU usage
rocm-smi

# Check for training processes
ps aux | grep -i "python.*train"

# Check training logs for active sessions
tail -f /tmp/*_training.log
```

**Resource Caps (DO NOT EXCEED):**
- **CPU**: ≤60% utilization
- **GPU**: ≤60% utilization (check with `rocm-smi`)
- **RAM**: ≤50% of total system RAM
- **Swap**: ≤10% of total swap (ideally 0%)

**Monitoring during testing:**
```bash
# Real-time monitoring
watch -n 1 'rocm-smi && free -h && mpstat 1 1'

# Individual commands
rocm-smi              # GPU usage
free -h               # RAM/swap
mpstat 1 1            # CPU usage
```

**If resources exceed limits:**
1. STOP IMMEDIATELY - Kill the process
2. Reduce batch size
3. Reduce model size
4. Enable gradient accumulation
5. Use MFN optimizer state offloading (crucible)
6. Check for memory leaks

---

## Development Workflow

### 1. Anti-Duplication Protocol (MANDATORY)

**SEARCH BEFORE CREATE. UPDATE BEFORE DUPLICATE.**

Before creating ANY new file or component:

1. Search codebase for existing implementations:
   ```bash
   # Search for similar functionality
   grep -r "class VoxelCloud" src/
   find src/ -name "*voxel*"
   ```

2. If found: **UPDATE existing file**
   - NEVER create alternatives (`_simple.*`, `_fixed.*`, `_new.*`, `_v2.*`)
   - Fix and maintain the original

3. If not found: Create new + document decision

**Absolute Zero Tolerance:**
- Duplicate files/components/functions/APIs
- Copy-paste code without refactoring
- Variant implementations

### 2. Branch Naming

```bash
# Features
git checkout -b feat/multi-octave-query

# Fixes
git checkout -b fix/frequency-extraction-bug

# Refactoring
git checkout -b refactor/consolidate-fm-modulation

# Documentation
git checkout -b docs/api-reference
```

### 3. Commit Messages

Follow conventional commits:

```bash
# Format
<type>(<scope>): <subject>

# Examples
feat(voxel): Add multi-octave quaternion support
fix(origin): Correct standing wave convergence logic
refactor(clustering): Consolidate duplicate k-means code
docs(api): Add FrequencyField API reference
test(foundation): Add Q&A test suite
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `refactor`: Code refactoring (no functional change)
- `docs`: Documentation only
- `test`: Adding/updating tests
- `perf`: Performance improvement
- `chore`: Maintenance tasks

---

## State Management Guidelines

### Signal States

The Genesis architecture recognizes three distinct signal states:

- **Paradox**: VALID state for conflicting information
  - Do NOT treat as error
  - Store both perspectives separately
  - Mark with paradox relationship
  - Expected during complex reasoning or conflicting inputs

- **Evolution**: Temporary transitional state
  - Expected during learning/processing
  - Track derivatives carefully
  - Monitor convergence metrics
  - Transition to Identity or Paradox based on coherence

- **Identity**: Stable converged state
  - Consolidation candidate
  - Low temporal drift (<0.1 per timestep)
  - High coherence (>0.8)
  - Ready for Core memory integration

### State Transitions

Proper state management requires:

- Use hysteresis to prevent oscillation (min 3 timesteps in state)
- Log state changes for debugging with timestamps
- Test all three states in unit tests
- Handle transitions gracefully without data loss
- Validate state consistency across layers

**Example state handling:**

```python
def update_signal_state(self, signal, coherence, drift):
    """Update signal state with hysteresis."""
    if coherence > 0.8 and drift < 0.1:
        if self.state_counter.get(signal.id, 0) >= 3:
            signal.state = SignalState.IDENTITY
    elif coherence < 0.3:
        if self.state_counter.get(signal.id, 0) >= 3:
            signal.state = SignalState.PARADOX
    else:
        signal.state = SignalState.EVOLUTION
```

---

## Temporal Processing

### Buffer Management

Temporal buffers track signal evolution over time:

- Fixed circular buffer (no dynamic resizing)
- Always store with timestamp (monotonic clock)
- Compute derivatives using finite differences
- Handle edge cases (buffer not full yet)
- Maintain buffer integrity during resets

**Buffer implementation requirements:**

```python
class TemporalBuffer:
    def __init__(self, size=100):
        self.buffer = np.zeros((size, signal_dim))
        self.timestamps = np.zeros(size)
        self.head = 0
        self.count = 0  # Track filled slots

    def add(self, signal, timestamp):
        self.buffer[self.head] = signal
        self.timestamps[self.head] = timestamp
        self.head = (self.head + 1) % len(self.buffer)
        self.count = min(self.count + 1, len(self.buffer))
```

### Taylor Series Extrapolation

For predictive synthesis:

- Support orders 1-3 (velocity, acceleration, jerk)
- Validate prediction horizon (<10 timesteps)
- Test extrapolation accuracy (MSE < 0.1)
- Handle discontinuities gracefully (fallback to order 1)
- Log prediction errors for tuning

**Validation requirements:**

```python
def validate_taylor_prediction(predicted, actual):
    """Validate Taylor series prediction accuracy."""
    mse = np.mean((predicted - actual)**2)
    if mse > 0.1:
        logger.warning(f"High prediction error: {mse:.3f}")
        return False
    return True
```

---

## Memory Hierarchy

### When to Use Core Memory

Core memory stores validated, stable knowledge:

- Stable patterns (Identity state for >10 timesteps)
- High resonance strength (>5 occurrences)
- Validated knowledge (coherence >0.8)
- Low contradiction rate (<0.1)
- Significant importance (relevance >0.7)

### When to Use Experiential Memory

Experiential memory handles active processing:

- Active conversation context
- Temporary working thoughts
- Evolution/Paradox states
- Recent interactions (<1000 timesteps)
- Unvalidated patterns

### Consolidation Rules

Memory consolidation follows strict criteria:

- Only consolidate Identity-state patterns
- Require minimum coherence (>0.8)
- Check for duplicates before adding to Core
- Log consolidation events with metrics
- Maintain consolidation history for rollback

**Consolidation example:**

```python
def consolidate_to_core(self, pattern):
    """Consolidate validated pattern to Core memory."""
    if pattern.state != SignalState.IDENTITY:
        return False

    if pattern.coherence < 0.8:
        return False

    if self.core_memory.exists(pattern.signature):
        logger.info(f"Pattern already in Core: {pattern.id}")
        return False

    self.core_memory.add(pattern)
    logger.info(f"Consolidated pattern {pattern.id} to Core")
    return True
```

---

## Self-Reflection Testing

### Test Cases Required

Comprehensive feedback loop testing:

- **High coherence** (>0.9): Aligned thinking, stable patterns
- **Low coherence** (<0.3): Conflict detected, paradox handling
- **Gradual drift**: Auto-reset trigger at threshold
- **Consolidation**: After Identity state stabilization
- **Edge cases**: Buffer overflow, rapid state changes

### Metrics to Track

Monitor these metrics during testing:

- Coherence vs Core alignment over time
- Reset frequency (target: <1 per 1000 timesteps)
- False positive rate (incorrect consolidation <0.01)
- False negative rate (missed consolidation <0.05)
- State transition stability (oscillation rate <0.1)

**Test example:**

```python
def test_coherence_drift_reset(feedback_loop):
    """Test gradual coherence drift triggers reset."""
    for i in range(100):
        feedback_loop.update(coherence=0.9 - i*0.01)

    assert feedback_loop.reset_count == 1
    assert feedback_loop.coherence > 0.8  # Post-reset
```

---

## Streaming Synthesis

### Chunk Size Selection

Optimal chunk sizes by modality:

- **Text**: 1 word per chunk (natural boundary)
- **Audio**: 10-20ms frames (phoneme-level)
- **Video**: 1 frame per chunk (16-33ms at 30-60fps)
- **Hybrid**: Align to slowest modality

### Prediction Guidelines

Streaming with prediction requires:

- Only predict when in Evolution state
- Validate predictions against actual (MSE threshold)
- Log prediction accuracy for tuning
- Fallback if prediction fails (>3 consecutive misses)
- Maintain prediction confidence scores

**Streaming implementation:**

```python
class StreamingSynthesizer:
    def __init__(self, chunk_size, prediction_order=2):
        self.chunk_size = chunk_size
        self.prediction_order = prediction_order
        self.prediction_confidence = 1.0

    def process_chunk(self, chunk):
        if self.state == SignalState.EVOLUTION:
            predicted = self.predict_next(chunk)
            if self.validate_prediction(predicted, chunk):
                self.prediction_confidence *= 1.1
            else:
                self.prediction_confidence *= 0.9
                if self.prediction_confidence < 0.5:
                    self.disable_prediction()
```

---

## Testing

### Test Organization

Tests are organized in `/tests/` hierarchy:

```
tests/
├── test_origin.py              # Origin morphisms
├── test_voxel_cloud.py         # VoxelCloud memory
├── test_frequency_field.py     # Frequency mapping
├── test_clustering.py          # Clustering algorithms
├── test_foundation_qa.py       # Foundation model Q&A
└── integration/
    ├── test_discovery.py       # Discovery pipeline
    └── test_synthesis.py       # Synthesis pipeline
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_voxel_cloud.py -v

# Run specific test class
pytest tests/test_foundation_qa.py::TestPhilosophicalQuestions -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

### Writing Tests

**Requirements:**
- Unit tests for all public functions
- Integration tests for pipelines
- 100% coverage for critical paths (morphisms, clustering, synthesis)
- Tests must be deterministic (use fixed random seeds)
- Test all three signal states (Identity, Evolution, Paradox)
- Test temporal buffer edge cases (empty, full, wraparound)
- Test feedback loop scenarios (high/low coherence, drift reset)
- Test streaming with various chunk sizes
- Test consolidation and reset mechanisms

**Example test cases:**

```python
import pytest
import numpy as np
from src.origin import Origin
from src.temporal import TemporalBuffer, SignalState
from src.feedback import FeedbackLoop

class TestOriginMorphisms:
    @pytest.fixture
    def origin(self):
        return Origin(512, 512, use_gpu=False)

    def test_gen_creates_proto_identity(self, origin):
        gamma_params = {
            'base_frequency': 2.5,
            'amplitude': 1.0,
            'envelope_sigma': 0.45,
            'num_harmonics': 12,
            'harmonic_decay': 0.75,
            'initial_phase': 0.0
        }
        iota_params = {
            'harmonic_coeffs': [1.0] * 10,
            'global_amplitude': 1.0,
            'frequency_range': 2.5
        }

        proto = origin.Gen(gamma_params, iota_params)

        assert proto.shape == (512, 512, 4)
        assert proto.dtype == np.float32
        assert -1.0 <= proto.min() <= proto.max() <= 1.0

class TestSignalStates:
    """Test all three signal states and transitions."""

    def test_identity_state_stability(self):
        signal = Signal(coherence=0.9, drift=0.05)
        for _ in range(10):
            signal.update()
        assert signal.state == SignalState.IDENTITY

    def test_paradox_state_handling(self):
        signal = Signal(coherence=0.2)
        signal.add_conflicting_data(data1, data2)
        assert signal.state == SignalState.PARADOX
        assert len(signal.perspectives) == 2

    def test_evolution_state_transition(self):
        signal = Signal(coherence=0.5)
        assert signal.state == SignalState.EVOLUTION
        # Should transition based on coherence changes

class TestTemporalBuffer:
    """Test temporal buffer edge cases."""

    def test_empty_buffer_handling(self):
        buffer = TemporalBuffer(size=10)
        assert buffer.count == 0
        derivatives = buffer.compute_derivatives()
        assert derivatives is None  # Not enough data

    def test_buffer_wraparound(self):
        buffer = TemporalBuffer(size=5)
        for i in range(10):
            buffer.add(signal=np.array([i]), timestamp=i)
        assert buffer.count == 5  # Max capacity
        assert buffer.head == 0  # Wrapped around

    def test_buffer_finite_differences(self):
        buffer = TemporalBuffer(size=10)
        for i in range(5):
            buffer.add(signal=np.array([i**2]), timestamp=i)
        derivatives = buffer.compute_derivatives()
        # First derivative should approximate 2*x
        assert np.allclose(derivatives[0], [2, 4, 6], atol=1.0)

class TestFeedbackLoop:
    """Test feedback loop scenarios."""

    def test_high_coherence_no_reset(self):
        loop = FeedbackLoop(reset_threshold=0.3)
        for _ in range(100):
            loop.update(coherence=0.95)
        assert loop.reset_count == 0

    def test_low_coherence_triggers_reset(self):
        loop = FeedbackLoop(reset_threshold=0.3)
        for _ in range(10):
            loop.update(coherence=0.2)
        assert loop.reset_count == 1
        assert loop.coherence > 0.8  # Reset to baseline

    def test_gradual_drift_detection(self):
        loop = FeedbackLoop(drift_threshold=0.5)
        initial = loop.coherence
        for i in range(50):
            loop.update(coherence=initial - i*0.01)
        assert loop.drift_detected == True

class TestStreamingSynthesis:
    """Test streaming with various chunk sizes."""

    def test_text_chunk_processing(self):
        synth = StreamingSynthesizer(chunk_size='word')
        text = "The quick brown fox"
        chunks = text.split()
        for chunk in chunks:
            output = synth.process_chunk(chunk)
            assert output is not None

    def test_audio_chunk_alignment(self):
        synth = StreamingSynthesizer(chunk_size='20ms')
        audio = np.random.randn(16000)  # 1 second at 16kHz
        chunk_samples = 320  # 20ms at 16kHz
        for i in range(0, len(audio), chunk_samples):
            chunk = audio[i:i+chunk_samples]
            output = synth.process_chunk(chunk)
            assert len(output) == chunk_samples

    def test_prediction_fallback(self):
        synth = StreamingSynthesizer(prediction_enabled=True)
        # Force prediction failures
        for _ in range(5):
            synth.process_chunk(np.random.randn(100))
        assert synth.prediction_enabled == False  # Should fallback
```

### Test Before Commit

**ALWAYS run tests before committing:**

```bash
# Run relevant test suite
pytest tests/test_<module>.py -v

# Run full suite before PR
pytest tests/ -v
```

---

## Documentation

### Types of Documentation

| Type | Purpose | Location | Action |
|------|---------|----------|--------|
| **API Reference** | Public API documentation | `API.md` | Update when API changes |
| **Architecture** | System design and structure | `ARCHITECTURE.md` | Update for design changes |
| **User Guide** | End-user documentation | `README.md` | Update for user-facing changes |
| **Contributing** | Developer guide | `CONTRIBUTING.md` | Update for workflow changes |
| **Training** | Model training guides | `FOUNDATION_TRAINING.md` | Update for training changes |

### Documentation Standards

- **Update documentation to match code**
  - API changes require API.md updates
  - Architecture changes require ARCHITECTURE.md updates

- **No progress reports or status files**
  - Don't create PROGRESS.md, STATUS.md, NOTES.md
  - Use git commits and PR descriptions for progress tracking

- **Code comments for complex logic only**
  - Self-documenting code is preferred
  - Comments explain "why", not "what"

- **Docstrings for all public functions**
  - Use NumPy/Google style docstrings
  - Include parameters, returns, examples

**Example docstring:**

```python
def synthesize(self, visible_protos: List[ProtoIdentityEntry],
               query_freq: np.ndarray) -> np.ndarray:
    """
    Synthesize new proto-identity from visible elements.

    This creates a NEW proto-identity by blending visible elements
    based on their relevance to the query.

    Args:
        visible_protos: List of visible proto-identities
        query_freq: Query frequency spectrum (H, W, 2)

    Returns:
        Synthesized proto-identity (H, W, 4)

    Example:
        >>> visible = voxel_cloud.query_viewport(query_freq, radius=100.0)
        >>> synthesized = voxel_cloud.synthesize(visible, query_freq)
    """
```

---

## Pull Request Process

### Before Creating a PR

1. **Run all tests**
   ```bash
   pytest tests/ -v
   ```

2. **Check for duplicates**
   ```bash
   # Search for similar implementations
   grep -r "class MyFeature" src/
   ```

3. **Verify resource compliance**
   - Files <500 lines
   - Functions <50 lines
   - Nesting <3 levels

4. **Update documentation**
   - API.md for API changes
   - README.md for user-facing changes
   - ARCHITECTURE.md for design changes

### PR Template

```markdown
## Summary
Brief description of changes (1-2 sentences)

## Changes
- Added: New features
- Changed: Modified functionality
- Fixed: Bug fixes
- Removed: Deprecated code

## Testing
- [ ] All tests pass
- [ ] Added tests for new functionality
- [ ] Updated documentation

## Checklist
- [ ] No duplicate implementations
- [ ] Files <500 lines
- [ ] Functions <50 lines
- [ ] Nesting <3 levels
- [ ] Zero linter warnings
- [ ] Documentation updated
```

### Review Criteria

PRs will be reviewed for:

1. **Standards compliance**
   - 500/50/3 limits enforced
   - No duplicates
   - Proper naming conventions

2. **Code quality**
   - Clean, readable code
   - Comprehensive error handling
   - Proper logging

3. **Testing**
   - Tests pass
   - Adequate coverage
   - Deterministic tests

4. **Documentation**
   - API docs updated
   - Code comments where needed
   - Examples provided

---

## Architecture Principles

### Core Principle

**Know it early → lock it down. Know it late → keep it flexible.**

### Application

1. **State Management**
   - Compile-time knowable → `immutable`, `const`
   - Runtime-contingent → computed dynamically

2. **Type Safety**
   - Static knowledge → strict typing, no implicit conversion
   - Dynamic knowledge → flexible typing with runtime validation

3. **Separation of Concerns**
   - **Compute vs Runtime**: Separate calculation logic from execution context
   - **Dynamic vs Static**: Compile-time knowable → static, runtime-contingent → dynamic
   - **Server vs Client**: Clear boundaries, no logic leakage

### Data Structures & Algorithms

- **Time complexity**: Prefer O(log n) or O(n) over O(n²)
- **Space complexity**: Avoid unnecessary allocations
- **Use appropriate data structures**:
  - Arrays for fixed-size, ordered data
  - Maps/dicts for key-value lookups
  - Sets for uniqueness constraints
  - Trees for hierarchical data
  - Graphs for relational data

### Design Patterns

**Use established patterns:**
- Factory: Object creation
- Strategy: Interchangeable algorithms
- Observer: Event-driven updates
- Adapter: Interface compatibility

**Avoid anti-patterns:**
- God objects
- Circular dependencies
- Magic numbers
- Global state

**Composition over inheritance:**
```python
# BAD - deep inheritance
class A:
    pass

class B(A):
    pass

class C(B):
    pass

# GOOD - composition
class DataProcessor:
    def __init__(self, validator, transformer, serializer):
        self.validator = validator
        self.transformer = transformer
        self.serializer = serializer
```

---

## Training Protocol (MANDATORY)

### Universal Training Script

**ALWAYS use `train.py` for ALL training tasks**
- NEVER run individual training scripts directly
- NEVER create new training wrapper scripts

`train.py` provides:
- Universal architecture/size/type selection
- Automatic sleep prevention (systemd-inhibit + swayidle kill)
- MFN service management
- Standardized checkpoint paths
- Resource monitoring

### Usage

```bash
# Interactive menu (recommended)
python train.py

# CLI arguments
python train.py --arch {alchemy|crucible|apex} \
                --size {nano|tiny|small|medium|large|xl|giant|frontier} \
                --type {curriculum|aggregate} \
                --steps 10000
```

### Training Requirements

ALL models MUST have:
1. **Safety**: Resource limits enforced (≤60% CPU/GPU, ≤50% RAM)
2. **Logging**: Real-time JSONL metrics via MetricsLogger
3. **Data Loading**: Universal data loading (lib/dataloader/unified_dataset.py)
4. **Checkpointing**: Standardized paths (`/usr/lib/alembic/checkpoints/{arch}_{size}_{type}/`)
5. **Sleep Prevention**: Auto-enabled via systemd-inhibit

### Training Logs

Format: `/tmp/{model}_training.log`

Examples: `/tmp/crucible_training.log`, `/tmp/alchemy_training.log`

---

## MFN Services

**MFN WILL NOT BE RUNNING IF YOU DON'T START IT**

```bash
# Check MFN status
ls -la /tmp/mfn_layer*.sock

# Start MFN (auto-started by train.py)
# Check train.py for MFN startup configuration
```

---

## Questions?

- Check [README.md](README.md) for project overview
- Check [ARCHITECTURE.md](ARCHITECTURE.md) for system design
- Check [API.md](API.md) for API reference
- Open an issue for questions or clarifications

---

**Remember: Maintain the utmost professional standards. We exhibit the leading and bleeding edge of technology and innovation.**
