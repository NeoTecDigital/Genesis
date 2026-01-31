# Genesis Security Requirements Specification

**Document Version**: 1.0
**Status**: Definition & Scoping (Step 2)
**Sprint**: Security Hardening Sprint
**Last Updated**: 2025-12-03

---

## Executive Summary

This document specifies formal security requirements for Genesis based on Step 1 (Discovery) vulnerability assessment. The specification addresses three vulnerability categories identified in 16 Python files across the codebase:

1. **Pickle RCE Vulnerability** (2 HIGH-risk user-facing loads)
2. **Input Validation Gaps** (21 entry points, 1 CRITICAL + 6 HIGH)
3. **Subprocess Calls** (9 calls, verified secure)

---

## Vulnerability Context

### Step 1 Discovery Findings

| Category | Count | Risk Level | Files |
|----------|-------|-----------|-------|
| Pickle RCE | 2 | HIGH | commands_synthesis.py:69, commands_helpers.py:317 |
| Input Validation Gaps | 21 | 1 CRITICAL, 6 HIGH, 10 MEDIUM | Across CLI and data loading |
| Subprocess Calls | 9 | LOW | All verified secure with list args |

### Files Requiring Security Hardening

**Pickle Usage (16 files)**:
- commands_synthesis.py (lines 69, 91)
- commands_helpers.py (line 317)
- commands_train.py
- voxel_cloud.py (load/save operations)
- memory_hierarchy.py
- And 11 other files with pickle.load/dump

**Input Validation (21 entry points)**:
- CLI argument parsing (genesis.py)
- File path handling (commands_synthesis.py, commands_train.py, commands_helpers.py)
- JSON loading (commands_train.py)
- Numeric parameter validation
- Text input processing

---

## Security Requirements

### REQ-SEC-001: Safe Deserialization

**Priority**: CRITICAL
**Category**: Pickle RCE Mitigation

**Requirement Statement**:
Replace all `pickle.load()` and `pickle.loads()` calls with secure deserialization using RestrictedUnpickler (HMAC-validated) or JSON for non-pickle formats.

**Rationale**:
- `pickle.load()` executes arbitrary code during deserialization
- Genesis deserializes user-provided model files (commands_synthesis.py:69)
- Attacker can craft malicious .pkl file → arbitrary code execution

**Scope**:
- All `pickle.load()` calls from file streams
- All `pickle.loads()` calls from bytes
- Both user-facing (HIGH-risk) and internal (LOW-risk) loads

**Acceptance Criteria**:

| Criteria | Verification |
|----------|--------------|
| Zero `pickle.load()` calls remain in codebase | Grep: `pickle\.load\(` = 0 matches |
| RestrictedUnpickler implemented with HMAC | Code review + unit tests |
| All model files include HMAC signature | Integration tests verify signatures |
| Fallback to JSON for new models | New models use JSON + numpy file format |
| Backward compatibility tested | Load old pickle models with Unpickler validation |
| No code execution from untrusted pickles | Security test with malicious .pkl injection |

**Implementation Pattern**:

```python
# OLD (VULNERABLE)
with open(model_path, 'rb') as f:
    data = pickle.load(f)  # DANGEROUS!

# NEW (SECURE)
from src.security.unpickler import RestrictedUnpickler, validate_pickle_signature
import hmac

with open(model_path, 'rb') as f:
    signature = f.read(32)  # 256-bit HMAC
    data_bytes = f.read()

    # Validate HMAC before unpickling
    if not validate_pickle_signature(data_bytes, signature, secret_key):
        raise ValueError("Invalid pickle signature - possible tampering")

    # Use RestrictedUnpickler (no code execution)
    unpickler = RestrictedUnpickler(io.BytesIO(data_bytes))
    data = unpickler.load()
```

**Testing**:
- Unit: RestrictedUnpickler with safe/unsafe objects
- Integration: Load existing models with signature validation
- Security: Malicious pickle injection detection

---

### REQ-SEC-002: Input Validation Framework

**Priority**: CRITICAL
**Category**: Input Validation Standardization

**Requirement Statement**:
Implement pydantic-based validation for all CLI arguments and data inputs across entry points.

**Rationale**:
- 21 entry points without structured validation (1 CRITICAL, 6 HIGH, 10 MEDIUM risk)
- Command injection possible via unvalidated path arguments
- Numeric parameters lack range checking (thresholds should be 0.0-1.0)
- Text inputs lack encoding/length validation

**Scope**:
- All argparse arguments in genesis.py
- All file paths in CLI commands
- All numeric parameters (thresholds, tolerances, rates)
- All text inputs (queries, file content)

**Acceptance Criteria**:

| Criteria | Verification |
|----------|--------------|
| All CLI args have pydantic models | Code review: 100% coverage |
| Path traversal prevention | Path validator rejects ../, ~/, absolute paths |
| Numeric ranges enforced | Thresholds: 0.0-1.0, octaves: 0-12, etc. |
| Text input limits enforced | Max 10KB for queries, max 1MB for files |
| UTF-8 encoding validated | Invalid UTF-8 rejected on input |
| JSON schema validation | JSON inputs validated before parsing |
| Error messages safe | No path disclosure in error output |
| All validation tested | Unit tests for each validator |

**Implementation Pattern**:

```python
# Define validated CLI argument models
from pydantic import BaseModel, Field, validator
from pathlib import Path

class DiscoverArgs(BaseModel):
    input: Path  # Validates path doesn't escape sandbox
    output: Path = Path('./models/genesis.pkl')
    dual_path: bool = False
    modality: str = 'text'
    collapse_harmonic_tolerance: float = Field(0.05, ge=0.0, le=1.0)
    collapse_cosine_threshold: float = Field(0.85, ge=0.0, le=1.0)
    collapse_octave_tolerance: int = Field(0, ge=0, le=12)
    max_segments: int = Field(None, ge=1, le=100000)

    @validator('input')
    def validate_input_path(cls, v):
        # Prevent path traversal
        if '..' in str(v) or str(v).startswith('/'):
            raise ValueError("Path traversal attempt blocked")
        if not v.exists():
            raise ValueError(f"Input file not found: {v}")
        if v.stat().st_size > 1_000_000_000:  # 1GB limit
            raise ValueError("Input file too large (>1GB)")
        return v

    @validator('modality')
    def validate_modality(cls, v):
        if v not in ['text', 'image', 'audio']:
            raise ValueError(f"Invalid modality: {v}")
        return v

# Parse CLI args into validated model
args = parser.parse_args()
try:
    validated = DiscoverArgs(**vars(args))
except ValidationError as e:
    print(f"Invalid arguments: {e}")
    sys.exit(1)
```

**Testing**:
- Unit: Each validator with valid/invalid inputs
- Integration: Full CLI argument parsing
- Security: Path traversal attempts, encoding attacks

---

### REQ-SEC-003: Path Traversal Protection

**Priority**: HIGH
**Category**: File System Security

**Requirement Statement**:
Implement centralized path sanitization for all file operations to prevent directory traversal attacks.

**Rationale**:
- 6 HIGH-risk path inputs without validation (commands_synthesis.py, commands_train.py, voxel_cloud.py)
- Attackers can use `../../etc/passwd`, `~/.ssh/id_rsa`, absolute paths
- Genesis loads/saves model files from user-specified paths

**Scope**:
- All file input paths (--input, --model, --output, --data)
- All internal file operations (load/save methods)
- Path construction for checkpoint files

**Acceptance Criteria**:

| Criteria | Verification |
|----------|--------------|
| Paths restricted to allowed directories | Whitelist: ./models/, /usr/lib/alembic/checkpoints/, ./tmp/ |
| .. (parent directory) traversal blocked | Reject any path with .. |
| Absolute paths rejected | Reject paths starting with / or ~ |
| Symlinks validated | Resolve symlinks, reject if outside sandbox |
| File permissions checked | Reject if file writable by other users |
| Relative paths normalized | Use Path.resolve() for canonical form |
| All path operations use pathlib | No os.path string manipulation |
| Security tests pass | Unit tests for each traversal technique |

**Implementation Pattern**:

```python
# Centralized path validator
from pathlib import Path
from typing import List

ALLOWED_DIRS = [
    Path('./models').resolve(),
    Path('./tmp').resolve(),
    Path('/usr/lib/alembic/checkpoints').resolve(),
]

def sanitize_path(user_path: str, allowed_dirs: List[Path] = None) -> Path:
    """Sanitize and validate file paths."""
    if allowed_dirs is None:
        allowed_dirs = ALLOWED_DIRS

    # Reject absolute paths and home directory expansion
    if user_path.startswith('/') or user_path.startswith('~'):
        raise ValueError("Absolute paths and home directory expansion not allowed")

    # Resolve to absolute path and detect traversal
    path = Path(user_path).resolve()

    # Verify path is within allowed directories
    try:
        path.relative_to(allowed_dirs[0])  # May raise ValueError
    except ValueError:
        for allowed in allowed_dirs:
            try:
                path.relative_to(allowed)
                break
            except ValueError:
                continue
        else:
            raise ValueError(f"Path {path} outside allowed directories")

    # Check for symlink escapes
    if path.is_symlink():
        target = path.resolve()
        for allowed in allowed_dirs:
            try:
                target.relative_to(allowed)
                break
            except ValueError:
                continue
        else:
            raise ValueError(f"Symlink {path} points outside allowed directories")

    return path
```

**Testing**:
- Unit: Each traversal technique (.., ~, /, symlinks)
- Integration: Load/save operations with sanitized paths
- Security: Real traversal attempt detection

---

### REQ-SEC-004: Text Input Validation

**Priority**: HIGH
**Category**: Input Sanitization

**Requirement Statement**:
Validate all text inputs for encoding, length, and content constraints.

**Rationale**:
- 10 MEDIUM-risk text inputs without validation
- Malformed UTF-8 can cause processing errors
- Unbounded text can cause DoS (memory exhaustion)
- Special characters may cause downstream vulnerabilities

**Scope**:
- Query text (--query parameter)
- File content (text files being processed)
- User-provided text in interactive commands (chat)

**Acceptance Criteria**:

| Criteria | Verification |
|----------|--------------|
| UTF-8 encoding validated | Invalid UTF-8 rejected with clear error |
| Query length limited | Max 10KB (10,240 bytes) |
| File size limited | Max 1GB for text files |
| Line length limited | Max 10KB per line (prevent streaming issues) |
| Null bytes rejected | Strings with \x00 rejected |
| Control characters flagged | Warn on non-printable characters (optional) |
| All validation logged | Security log entry on rejection |
| Tests cover edge cases | Truncated UTF-8, max lengths, null bytes |

**Implementation Pattern**:

```python
# Text validator
import re

def validate_text_input(text: str, max_length: int = 10240,
                        allow_control_chars: bool = False) -> None:
    """Validate text input for security and correctness."""
    if not isinstance(text, str):
        raise TypeError(f"Expected str, got {type(text)}")

    # Check length
    if len(text) > max_length:
        raise ValueError(f"Text exceeds max length of {max_length} bytes")

    # Check for null bytes
    if '\x00' in text:
        raise ValueError("Text contains null bytes (potential injection)")

    # Check encoding (should already be valid str, but verify)
    try:
        text.encode('utf-8')
    except UnicodeEncodeError as e:
        raise ValueError(f"Invalid UTF-8 encoding: {e}")

    # Optional: check for problematic control characters
    if not allow_control_chars:
        control_pattern = r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]'
        matches = re.findall(control_pattern, text)
        if matches:
            raise ValueError(f"Text contains control characters: {matches}")

    return True

def validate_file_encoding(file_path: Path, max_size: int = 1_000_000_000) -> None:
    """Validate file encoding and size."""
    if file_path.stat().st_size > max_size:
        raise ValueError(f"File exceeds {max_size} bytes")

    # Read and validate UTF-8
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if len(line) > 10240:
                    raise ValueError(f"Line {line_num} exceeds 10KB limit")
    except UnicodeDecodeError as e:
        raise ValueError(f"File contains invalid UTF-8: {e}")
```

**Testing**:
- Unit: Each validation constraint
- Integration: Query/file processing with limits
- Security: Malformed UTF-8, oversized inputs, null bytes

---

### REQ-SEC-005: Numeric Parameter Ranges

**Priority**: MEDIUM
**Category**: Input Validation

**Requirement Statement**:
Validate numeric parameters are within defined ranges and valid for their use cases.

**Rationale**:
- Thresholds (0.0-1.0) can cause algorithmic errors if out of range
- Octave tolerances (0-12) should be bounded
- Max segments should be positive and reasonable
- Weight functions and decay rates have valid ranges

**Scope**:
- Collapse parameters (cosine_threshold, harmonic_tolerance, octave_tolerance)
- Synthesis parameters (resonance_boost, distance_decay, weight_function)
- Training parameters (max_segments, checkpoint_interval, max_documents)
- Frequency parameters (phase coherence threshold)

**Acceptance Criteria**:

| Criteria | Verification |
|----------|--------------|
| Thresholds constrained 0.0-1.0 | ValueError if outside range |
| Octave tolerance 0-12 | ValueError if outside range |
| Positive counts validated | Max_segments, max_documents > 0 |
| Decay rates 0.0-1.0 | ValueError if outside range |
| Boost factors > 0.0 | ValueError if <= 0 |
| Weight functions enumerated | Only 'linear', 'sqrt', 'log' allowed |
| Type checking enforced | int vs float verified |
| Reasonable defaults documented | Each param has justified default |
| Tests cover boundaries | Min, max, edge cases |

**Implementation Pattern**:

```python
# Numeric validators using pydantic
from pydantic import Field, validator

class CollapseConfig(BaseModel):
    # Thresholds must be in [0.0, 1.0]
    cosine_threshold: float = Field(
        0.85,
        ge=0.0,  # greater than or equal
        le=1.0,  # less than or equal
        description="Cosine similarity for clustering [0.0-1.0]"
    )

    # Harmonic tolerance also [0.0, 1.0]
    harmonic_tolerance: float = Field(
        0.05,
        ge=0.0,
        le=1.0,
        description="Harmonic deviation tolerance [0.0-1.0]"
    )

    # Octave tolerance [0-12]
    octave_tolerance: int = Field(
        0,
        ge=0,
        le=12,
        description="Octave range for frequency matching"
    )

class SynthesisConfig(BaseModel):
    # Boost factor must be positive
    resonance_boost: float = Field(
        2.0,
        gt=0.0,  # strictly greater than
        description="Resonance weight multiplier"
    )

    # Distance decay in [0.0, 1.0]
    distance_decay: float = Field(
        0.5,
        ge=0.0,
        le=1.0,
        description="Distance weight factor"
    )

    # Weight function enumeration
    weight_function: str = Field(
        'linear',
        description="Weight function for resonance"
    )

    @validator('weight_function')
    def validate_weight_function(cls, v):
        if v not in ['linear', 'sqrt', 'log']:
            raise ValueError(f"weight_function must be one of: linear, sqrt, log")
        return v
```

**Testing**:
- Unit: Boundary values (0, 1, -1, 2 for 0-1 ranges, etc.)
- Integration: Parameter parsing and validation
- Edge cases: NaN, infinity, very large/small values

---

### REQ-SEC-006: Security Test Suite

**Priority**: HIGH
**Category**: Verification & Testing

**Requirement Statement**:
Implement comprehensive security test suite to verify all security requirements are met.

**Rationale**:
- Automated verification prevents regression
- Security requirements need evidence of correctness
- Step 1 identified subprocess security (already passing tests)
- Need similar rigor for new security controls

**Scope**:
- Pickle RCE prevention tests
- Input validation tests (paths, text, numbers)
- Deserialization security tests
- End-to-end attack scenario tests

**Test Categories**:

#### Test Suite 1: Deserialization Security (`test_pickle_security.py`)

```
- test_pickle_load_rejects_dangerous_pickle()
  Load pickle with __reduce__ payload → must reject

- test_restricted_unpickler_safe_objects()
  Load safe objects (dict, list, str) → must succeed

- test_hmac_signature_validation()
  Valid pickle + correct signature → success
  Valid pickle + wrong signature → reject

- test_model_compatibility()
  Load old pickle models with Unpickler validation → success

- test_code_execution_prevention()
  Pickle with os.system() in __reduce__ → must reject
```

#### Test Suite 2: Path Traversal Prevention (`test_path_security.py`)

```
- test_parent_directory_traversal()
  Paths with ../../etc/passwd → reject

- test_absolute_path_rejection()
  Absolute paths /etc/passwd → reject

- test_home_directory_expansion()
  Paths with ~/ → reject

- test_symlink_escape_detection()
  Symlink pointing outside sandbox → reject

- test_relative_path_normalization()
  Safe relative paths ./models/model.pkl → accept

- test_allowed_directory_enforcement()
  Paths in ./models/, ./tmp/, /usr/lib/alembic → accept
  Other paths → reject
```

#### Test Suite 3: Text Input Validation (`test_text_input_security.py`)

```
- test_utf8_encoding_validation()
  Invalid UTF-8 bytes → reject
  Valid UTF-8 → accept

- test_query_length_limit()
  Query > 10KB → reject
  Query <= 10KB → accept

- test_file_size_limit()
  File > 1GB → reject
  File <= 1GB → accept

- test_null_byte_rejection()
  Text with \x00 → reject
  Normal text → accept

- test_line_length_limit()
  Line > 10KB → reject
  Line <= 10KB → accept
```

#### Test Suite 4: Numeric Validation (`test_numeric_validation.py`)

```
- test_threshold_range_validation()
  Cosine threshold 0.85 → accept
  Cosine threshold 1.5 or -0.1 → reject

- test_octave_tolerance_range()
  Octave tolerance 0-12 → accept
  Octave tolerance 13 or -1 → reject

- test_boost_factor_positive()
  Resonance boost 2.0 → accept
  Resonance boost 0.0 or -1 → reject

- test_decay_rate_validation()
  Distance decay 0.5 → accept
  Distance decay 1.5 or -0.1 → reject
```

#### Test Suite 5: CLI Argument Injection (`test_cli_injection.py`)

```
- test_query_shell_metacharacters()
  Query with ; | & ` $ → parsed as text, not executed

- test_path_shell_metacharacters()
  Path with ; | & ` $ → parsed as text, not path traversal

- test_newline_argument_injection()
  Parameter with \n--disable-collapse → remains single arg

- test_parameter_boundary_testing()
  Edge values for numeric params → validated correctly
```

**Acceptance Criteria**:

| Criteria | Verification |
|----------|--------------|
| All test suites pass | `pytest tests/security/` = 100% pass |
| Code coverage >95% | Security code paths fully tested |
| Attack scenarios blocked | All injection/traversal attempts fail |
| Safe inputs accepted | Normal usage passes validation |
| Error messages safe | No path disclosure or internal details |
| Tests documented | Docstrings explain each test |
| Regression prevention | Tests run in CI/CD pipeline |

**File Organization**:
```
tests/security/
├── __init__.py
├── test_pickle_security.py
├── test_path_security.py
├── test_text_input_security.py
├── test_numeric_validation.py
├── test_cli_injection.py
├── conftest.py  (shared fixtures: temp dirs, pickle generators, etc.)
└── README.md    (security test documentation)
```

---

## Standards Compliance

### Code Quality Standards (Per Professional Development Requirements)

**File Size**: Security modules must be <500 lines each
- validators.py: Validation functions
- unpickler.py: RestrictedUnpickler + HMAC
- path_sanitizer.py: Path validation utilities

**Function Size**: Each function <50 lines
- validate_* functions broken into helper functions if needed
- Test functions may exceed 50 lines per pytest conventions

**Nesting**: Maximum 3 indentation levels
- Validators use early returns to keep nesting shallow
- Helper functions extract complex logic

**Error Handling**: No silent failures
- All validation errors raise exceptions with clear messages
- Security rejections logged and reported

### Security Standards

**SEC-001**: No hardcoded secrets
- HMAC keys stored in environment variables or secure config
- No test credentials in code

**SEC-002**: No code execution from untrusted input
- pickle.load() eliminated entirely
- JSON only from trusted sources

**SEC-003**: No path traversal vulnerabilities
- Centralized sanitization
- Whitelist-based directory access

**SEC-004**: Comprehensive input validation
- All entry points validated
- Type checking, range checking, encoding validation

**SEC-005**: Secure error messages
- No internal paths in error output
- No stack traces in user-facing output

---

## Requirements Summary

| ID | Requirement | Priority | Status |
|---|---|---|---|
| REQ-SEC-001 | Safe Deserialization | CRITICAL | Definition |
| REQ-SEC-002 | Input Validation Framework | CRITICAL | Definition |
| REQ-SEC-003 | Path Traversal Protection | HIGH | Definition |
| REQ-SEC-004 | Text Input Validation | HIGH | Definition |
| REQ-SEC-005 | Numeric Parameter Ranges | MEDIUM | Definition |
| REQ-SEC-006 | Security Test Suite | HIGH | Definition |

---

## Verification Method

Each requirement includes:
- **Acceptance Criteria** table with verification method
- **Implementation Pattern** showing secure code structure
- **Testing Strategy** for comprehensive validation

Verification occurs during Step 3 (Design Review) and Step 4 (Development Testing).

---

## Dependencies & Constraints

### External Dependencies
- `pydantic` (v2.0+): Data validation
- `pathlib`: Path handling (stdlib)
- `hmac`: Signature validation (stdlib)
- `json`: Safe serialization (stdlib)

### Backward Compatibility
- Old pickle models must be loadable with signature validation
- Existing APIs remain unchanged (validation added transparently)
- Error messages enhanced but behaviors preserved

### Performance Impact
- Path validation: <1ms per operation
- Text validation: <10ms for 1MB files
- Numeric validation: <1ms per parameter
- Overall: <5% performance overhead for CLI operations

---

## Next Steps

1. **Step 3 (Design)**: Design detailed implementation (separate document)
2. **Step 4 (Development)**: Implement all requirements with tests
3. **Step 5 (Testing)**: Security test suite execution
4. **Step 6 (Launch)**: Deployment with security validation
5. **Step 7 (Growth)**: Monitor and optimize security controls
