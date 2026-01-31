# Memory Integration Architecture Specification

**Sprint**: Memory Integration
**Step**: 2 (Definition & Scoping)
**Status**: COMPLETE
**Date**: 2025-12-03
**Version**: 1.0

---

## Executive Summary

This specification defines the architecture to integrate the multi-octave encoding system with the memory hierarchy, bridging the gap between hash-based proto-identity generation and the core/experiential memory layers.

---

## Current State Analysis

### System A: Multi-Octave Encoding (NEW)
- **Location**: `src/pipeline/multi_octave_encoder.py`, `src/pipeline/multi_octave_decoder.py`
- **Architecture**: Hash-based frequency patterns with dynamic clustering
- **Storage**: `VoxelCloudClustering` with O(vocabulary_size) efficiency
- **Octave Levels**: +4 (character), 0 (word), -2 (short phrase), -4 (long phrase)
- **Output**: List of `OctaveUnit` objects with proto_identity and frequency

### System B: Memory Hierarchy (EXISTING)
- **Location**: `src/memory/memory_hierarchy.py`
- **Architecture**: Three-layer system (carrier, core, experiential)
- **Storage**: Two `VoxelCloud` instances (core + experiential)
- **Features**: Feedback loops, consolidation, temporal tracking
- **Input**: Individual proto_identity + frequency + metadata via `store_core()` or `store_experiential()`

### The Gap
- No routing logic to direct multi-octave units to appropriate memory layer
- No unified API for encoding/decoding across both systems
- No octave-aware querying across memory layers
- No integration of feedback loops with multi-octave clustering
- No preservation of octave hierarchy during consolidation

---

## Requirements Specification

### REQ-INT-001: MemoryRouter Routes Proto-Identities to Correct Layer
**Priority**: CRITICAL
**Description**: MemoryRouter must analyze context and route proto-identities to core or experiential memory.

**Acceptance Criteria**:
- Foundation texts (training data) → core memory
- Query/inference inputs → experiential memory
- User-provided context flag overrides default routing
- Octave-level routing: character/word → both layers, phrases → experiential only

---

### REQ-INT-002: Unified Encoder API
**Priority**: CRITICAL
**Description**: Single `encode()` function that works with both systems.

**Acceptance Criteria**:
- `encode(text, destination='auto', octaves=[4,0])` returns stored entries
- Auto-routing based on context (foundation vs query)
- Backward compatible with existing `MultiOctaveEncoder` API
- Metadata includes: octave, unit, modality, timestamp, destination_layer

---

### REQ-INT-003: Unified Decoder API
**Priority**: CRITICAL
**Description**: Single `decode()` function for retrieval across layers.

**Acceptance Criteria**:
- `decode(query_proto, layers='both', octaves=[4,0])` queries specified layers
- Blends results from core + experiential with configurable weights
- Preserves octave hierarchy in reconstruction
- Returns text + metadata (source layers, octaves used, confidence)

---

### REQ-INT-004: Octave-Aware Cross-Layer Queries
**Priority**: HIGH
**Description**: Query system must understand octave relationships across layers.

**Acceptance Criteria**:
- Query at octave N also checks octaves N±1 in both layers
- Character-level queries can upscale to word-level results
- Phrase-level queries can downscale to word/character components
- Resonance weighting factors in octave proximity

---

### REQ-INT-005: Feedback Loop Integration with Multi-Octave Clustering
**Priority**: MEDIUM
**Description**: Feedback loops must account for octave-specific coherence.

**Acceptance Criteria**:
- Coherence computed per-octave (character coherence, word coherence, etc.)
- Overall coherence = weighted average across octaves
- PARADOX state triggers octave-specific analysis (which octave conflicts?)
- Feedback recommendations include octave-level guidance

---

### REQ-INT-006: Consolidation Preserves Octave Hierarchy
**Priority**: MEDIUM
**Description**: Moving entries from experiential → core maintains octave structure.

**Acceptance Criteria**:
- Octave metadata preserved during consolidation
- Clustering continues to work post-consolidation
- Resonance strength carries over from experiential to core
- Units set preserved (no data loss)

---

## Architecture Design

### Component 1: MemoryRouter

**Purpose**: Intelligent routing of proto-identities to appropriate memory layer.

**Location**: `src/memory/memory_router.py`

**Class Structure**:
```python
class MemoryRouter:
    """Routes proto-identities to core or experiential memory based on context."""

    def __init__(self, memory_hierarchy: MemoryHierarchy):
        self.hierarchy = memory_hierarchy
        self.routing_policy = RoutingPolicy()

    def route(self, octave_units: List[OctaveUnit],
              context: str = 'auto') -> Dict[str, List[OctaveUnit]]:
        """Route units to appropriate layers.

        Returns:
            {"core": [...], "experiential": [...]}
        """
        pass

    def route_single(self, unit: OctaveUnit,
                     context: str = 'auto') -> str:
        """Route single unit.

        Returns:
            'core' or 'experiential'
        """
        pass
```

**Routing Logic**:

1. **Context-based routing**:
   - `context='foundation'` → all units to core
   - `context='query'` → all units to experiential
   - `context='auto'` → infer from metadata

2. **Octave-based routing** (when context='auto'):
   - Octave +4 (character): both layers (for vocabulary building)
   - Octave 0 (word): both layers (for semantic grounding)
   - Octave -2/-4 (phrases): experiential only (query-specific)

3. **Metadata inspection**:
   - Check for `is_training`, `is_foundation` flags
   - Timestamp analysis (old = core, recent = experiential)
   - Resonance strength (high = core candidate)

**Integration Points**:
- **Input**: `List[OctaveUnit]` from `MultiOctaveEncoder.encode_text_hierarchical()`
- **Output**: Routed units stored in appropriate `VoxelCloud` via `MemoryHierarchy.store_core()` or `.store_experiential()`

---

### Component 2: UnifiedEncoder

**Purpose**: Single API for encoding text with automatic routing.

**Location**: `src/pipeline/unified_encoder.py`

**Class Structure**:
```python
class UnifiedEncoder:
    """Unified encoding API with automatic memory routing."""

    def __init__(self, memory_hierarchy: MemoryHierarchy,
                 carrier: np.ndarray, width: int = 512, height: int = 512):
        self.multi_octave_encoder = MultiOctaveEncoder(carrier, width, height)
        self.router = MemoryRouter(memory_hierarchy)
        self.hierarchy = memory_hierarchy

    def encode(self, text: str,
               destination: str = 'auto',
               octaves: List[int] = [4, 0],
               metadata: Optional[Dict] = None) -> Dict[str, int]:
        """Encode text and route to appropriate memory layers.

        Args:
            text: Input text
            destination: 'foundation', 'query', or 'auto'
            octaves: Octave levels to encode at
            metadata: Optional metadata

        Returns:
            {"core_added": N, "experiential_added": M}
        """
        pass
```

**Workflow**:
1. Call `multi_octave_encoder.encode_text_hierarchical(text, octaves)`
2. Route units via `router.route(units, context=destination)`
3. Store routed units in appropriate layers using clustering
4. Return statistics (how many added to each layer)

**Metadata Schema** (unified across both systems):
```python
{
    "unit": str,           # The text unit (char/word/phrase)
    "octave": int,         # Octave level
    "modality": str,       # "text", "image", "audio"
    "timestamp": float,    # Unix timestamp
    "destination_layer": str,  # "core" or "experiential"
    "context": str,        # "foundation", "query", "auto"
    "is_training": bool,   # Training data flag
    "units": Set[str]      # All units that cluster here
}
```

---

### Component 3: UnifiedDecoder

**Purpose**: Single API for decoding with cross-layer queries.

**Location**: `src/pipeline/unified_decoder.py`

**Class Structure**:
```python
class UnifiedDecoder:
    """Unified decoding API with cross-layer querying."""

    def __init__(self, memory_hierarchy: MemoryHierarchy,
                 carrier: np.ndarray):
        self.multi_octave_decoder = MultiOctaveDecoder(carrier)
        self.hierarchy = memory_hierarchy

    def decode(self, query_proto: np.ndarray,
               layers: str = 'both',  # 'core', 'experiential', 'both'
               octaves: List[int] = [4, 0],
               max_results_per_octave: int = 10) -> DecodingResult:
        """Decode from memory with cross-layer queries.

        Returns:
            DecodingResult with text + metadata
        """
        pass

    def decode_text(self, query_text: str, **kwargs) -> str:
        """Convenience: encode query_text → query_proto → decode."""
        pass
```

**Cross-Layer Query Strategy**:
1. Query core memory at specified octaves
2. Query experiential memory at specified octaves
3. Blend results:
   - Default weights: core=0.7, experiential=0.3
   - If core empty: experiential=1.0
   - If experiential empty: core=1.0
4. Apply octave-aware resonance weighting
5. Reconstruct text hierarchically

**Result Structure**:
```python
@dataclass
class DecodingResult:
    text: str                          # Reconstructed text
    source_layers: Dict[str, int]      # {"core": N, "experiential": M}
    octaves_used: List[int]            # Which octaves contributed
    confidence: float                  # Overall confidence [0,1]
    coherence: Optional[float]         # Coherence (if feedback enabled)
    state: Optional[SignalState]       # IDENTITY/EVOLUTION/PARADOX
```

---

### Component 4: OctaveAwareConsolidation

**Purpose**: Preserve octave hierarchy during consolidation.

**Location**: `src/memory/octave_consolidation.py`

**Key Functions**:
```python
def consolidate_with_octaves(experiential_memory: VoxelCloud,
                              core_memory: VoxelCloud,
                              threshold: float = 0.8) -> int:
    """Consolidate experiential → core with octave preservation.

    Process:
        1. Group experiential entries by octave
        2. For each octave:
           - Find high-resonance entries (>= threshold)
           - Check for existing core matches at same octave
           - If match: strengthen core entry + merge units
           - If no match: create new core entry
        3. Preserve: octave, units set, resonance_strength

    Returns:
        Number of entries consolidated
    """
    pass
```

---

### Component 5: OctaveAwareFeedback

**Purpose**: Compute coherence per-octave for fine-grained feedback.

**Location**: `src/memory/octave_feedback.py`

**Enhanced Feedback**:
```python
class OctaveAwareFeedback:
    """Per-octave coherence computation for fine-grained feedback."""

    def __init__(self, feedback_loop: FeedbackLoop):
        self.feedback = feedback_loop

    def self_reflect_octaves(self,
                             experiential_protos: Dict[int, np.ndarray],
                             query_quaternions: Dict[int, np.ndarray]) -> OctaveFeedbackResult:
        """Self-reflection with per-octave coherence.

        Returns:
            OctaveFeedbackResult with detailed octave-level feedback
        """
        pass
```

**Feedback Structure**:
```python
@dataclass
class OctaveFeedbackResult:
    overall_coherence: float               # Weighted average
    octave_coherences: Dict[int, float]    # Per-octave coherence
    overall_state: SignalState             # IDENTITY/EVOLUTION/PARADOX
    octave_states: Dict[int, SignalState]  # Per-octave states
    recommendations: List[str]             # Actionable guidance
```

---

## Integration Flow

### Encoding Flow (Foundation Training)
```
Text → UnifiedEncoder.encode(text, destination='foundation') →
    MultiOctaveEncoder.encode_text_hierarchical() →
    MemoryRouter.route(units, context='foundation') →
    MemoryHierarchy.store_core() (via VoxelCloudClustering) →
    Core Memory Updated
```

### Encoding Flow (Query/Inference)
```
Query → UnifiedEncoder.encode(query, destination='query') →
    MultiOctaveEncoder.encode_text_hierarchical() →
    MemoryRouter.route(units, context='query') →
    MemoryHierarchy.store_experiential() →
    Experiential Memory Updated
```

### Decoding Flow (Cross-Layer Query)
```
Query Proto → UnifiedDecoder.decode(query_proto, layers='both') →
    Query Core at octaves [4, 0] →
    Query Experiential at octaves [4, 0] →
    Blend results (core=0.7, experiential=0.3) →
    MultiOctaveDecoder.decode_from_memory() →
    Hierarchical reconstruction →
    Text Output
```

### Feedback Flow (Self-Reflection)
```
Experiential Proto → OctaveAwareFeedback.self_reflect_octaves() →
    Query Core at each octave →
    Compute per-octave coherence →
    Aggregate to overall coherence →
    Classify state (IDENTITY/EVOLUTION/PARADOX) →
    Generate octave-specific recommendations
```

### Consolidation Flow
```
Trigger Consolidation → OctaveAwareConsolidation.consolidate_with_octaves() →
    Group experiential entries by octave →
    For each octave:
        Find high-resonance entries (>= 0.8) →
        Check core for existing match (same octave, similarity > 0.9) →
        If match: strengthen core + merge units set →
        If no match: create new core entry →
    Return count consolidated
```

---

## Implementation Plan

### Phase 1: Core Integration Components (Days 1-3)
**Effort**: 16 hours

**Tasks**:
1. **MemoryRouter** (4 hours)
   - File: `src/memory/memory_router.py`
   - Implement routing logic (context-based, octave-based, metadata-based)
   - Unit tests: `tests/test_memory_router.py`

2. **UnifiedEncoder** (4 hours)
   - File: `src/pipeline/unified_encoder.py`
   - Integrate MultiOctaveEncoder + MemoryRouter
   - Implement unified metadata schema
   - Unit tests: `tests/test_unified_encoder.py`

3. **UnifiedDecoder** (4 hours)
   - File: `src/pipeline/unified_decoder.py`
   - Implement cross-layer querying
   - Implement result blending
   - Unit tests: `tests/test_unified_decoder.py`

4. **Integration Testing** (4 hours)
   - File: `tests/test_memory_integration.py`
   - End-to-end test: encode foundation → query → decode
   - Verify octave preservation
   - Verify routing correctness

### Phase 2: Feedback & Consolidation (Days 4-5)
**Effort**: 12 hours

**Tasks**:
1. **OctaveAwareFeedback** (4 hours)
   - File: `src/memory/octave_feedback.py`
   - Extend FeedbackLoop with per-octave coherence
   - Implement octave-specific recommendations
   - Unit tests: `tests/test_octave_feedback.py`

2. **OctaveAwareConsolidation** (4 hours)
   - File: `src/memory/octave_consolidation.py`
   - Implement octave-preserving consolidation
   - Integrate with MemoryHierarchy.consolidate()
   - Unit tests: `tests/test_octave_consolidation.py`

3. **Integration & Documentation** (4 hours)
   - Update `MemoryHierarchy` to use new consolidation
   - Update `FeedbackLoop` to use octave-aware feedback
   - Integration tests
   - Update README with unified API examples

### Total Effort: 28 hours (3.5 days)

---

## File Modification List

### New Files to Create
1. `src/memory/memory_router.py` - MemoryRouter class
2. `src/pipeline/unified_encoder.py` - UnifiedEncoder class
3. `src/pipeline/unified_decoder.py` - UnifiedDecoder class
4. `src/memory/octave_feedback.py` - OctaveAwareFeedback class
5. `src/memory/octave_consolidation.py` - Consolidation functions
6. `tests/test_memory_router.py` - Router tests
7. `tests/test_unified_encoder.py` - Encoder tests
8. `tests/test_unified_decoder.py` - Decoder tests
9. `tests/test_octave_feedback.py` - Feedback tests
10. `tests/test_octave_consolidation.py` - Consolidation tests
11. `tests/test_memory_integration.py` - End-to-end tests

### Files to Modify
1. `src/memory/memory_hierarchy.py` - Add router integration, update consolidate()
2. `src/memory/feedback_loop.py` - Add octave-aware methods
3. `src/memory/voxel_cloud_clustering.py` - Ensure octave preservation in clustering

### Files to Preserve (No Changes)
1. `src/pipeline/multi_octave_encoder.py` - API stable
2. `src/pipeline/multi_octave_decoder.py` - API stable
3. `src/memory/voxel_cloud.py` - Core data structure stable
4. `src/origin.py` - Carrier generation unchanged
5. `src/memory/octave_hierarchy.py` - Already handles octave storage

---

## Backward Compatibility

**Guaranteed**:
- Existing `MultiOctaveEncoder` and `MultiOctaveDecoder` APIs unchanged
- Existing `MemoryHierarchy` methods (`store_core()`, `store_experiential()`) work as before
- Existing `VoxelCloud` clustering continues to function
- New unified API is additive (no breaking changes)

**Migration Path**:
```python
# Old way (still works)
encoder = MultiOctaveEncoder(carrier)
units = encoder.encode_text_hierarchical(text)
for unit in units:
    hierarchy.store_core(unit.proto_identity, unit.frequency, metadata)

# New way (recommended)
unified = UnifiedEncoder(hierarchy, carrier)
stats = unified.encode(text, destination='foundation')
```

---

## Success Criteria

### Functional
- ✓ Foundation text encodes to core memory
- ✓ Query text encodes to experiential memory
- ✓ Cross-layer queries return blended results
- ✓ Octave hierarchy preserved throughout pipeline
- ✓ Consolidation moves entries without data loss

### Performance
- ✓ Routing adds <1ms latency per unit
- ✓ Cross-layer queries complete in <100ms (10k entries)
- ✓ Consolidation handles 1000+ entries in <1s

### Quality
- ✓ Zero code duplication
- ✓ All functions <50 lines
- ✓ All files <500 lines
- ✓ 100% test coverage for new components
- ✓ Documentation includes examples

---

## Next Steps (Step 3: Design/Prototyping)

1. Create `MemoryRouter` class with routing logic
2. Implement `UnifiedEncoder` with metadata schema
3. Implement `UnifiedDecoder` with cross-layer blending
4. Write comprehensive integration tests
5. Validate end-to-end flow (foundation → query → decode)

---

**Document Status**: COMPLETE - Ready for Step 3
**Last Updated**: 2025-12-03
**Version**: 1.0
