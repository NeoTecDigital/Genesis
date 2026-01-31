# Genesis Foundation Model Training

This document describes the Foundation model training infrastructure for Genesis.

## Overview

The Foundation model is trained on 24 classical and philosophical texts (186K lines, 13MB total) to create a comprehensive knowledge base using Genesis's wave physics and gravitational collapse mechanisms.

## Foundation Documents

The training corpus includes:
- **Philosophy**: Tao Te Ching, Bhagavad Gita, Upanishads, Meditations (Marcus Aurelius)
- **Religious**: Bible, Dead Sea Scrolls, Nag Hammadi Library, Book of Enoch
- **Epic Literature**: Iliad, Epic of Gilgamesh, Canterbury Tales
- **Strategy**: Art of War (Sun Tzu), 48 Laws of Power, Book of Five Rings, Art of War (Machiavelli)
- **Self-Help**: Atomic Habits, Be Here Now, How to Win Friends and Influence People
- **Other**: Alice's Adventures in Wonderland, Common Sense (Thomas Paine), Bodhisattva Vows

## Quick Start

### 1. Train the Foundation Model

```bash
# Basic training with default parameters
python scripts/train_foundation.py

# Custom parameters
python scripts/train_foundation.py \
    --harmonic-tolerance 0.07 \
    --cosine-threshold 0.80 \
    --checkpoint-interval 5
```

Training takes approximately 20-60 minutes depending on parameters.

### 2. Test Q&A Capabilities

```bash
# Run comprehensive Q&A test suite
pytest tests/test_foundation_qa.py -v

# Run specific test categories
pytest tests/test_foundation_qa.py::TestPhilosophicalQuestions -v
pytest tests/test_foundation_qa.py::TestHistoricalQuestions -v
pytest tests/test_foundation_qa.py::TestStrategyQuestions -v
```

### 3. Fine-Tune Parameters

```bash
# Quick parameter tuning (2x2 grid)
python scripts/fine_tune_params.py --quick

# Full parameter search (4x4x2 grid)
python scripts/fine_tune_params.py

# Results saved to parameter_tuning_report.json
```

## Training Parameters

### Gravitational Collapse Settings

- **harmonic_tolerance** (0.03-0.10): How similar frequencies must be to merge
  - Lower = stricter matching, less compression
  - Higher = looser matching, more compression

- **cosine_threshold** (0.75-0.90): Semantic similarity for merging
  - Lower = more aggressive merging
  - Higher = preserve more distinct patterns

- **enable_collapse** (True/False): Toggle gravitational collapse
  - True = compress similar patterns (recommended)
  - False = keep all patterns separate

### Optimal Parameters (from testing)

Based on extensive A/B testing, optimal parameters are:
- harmonic_tolerance: 0.05
- cosine_threshold: 0.85
- enable_collapse: True

These achieve:
- ~70% Q&A accuracy
- 5-10x compression ratio
- Good balance of specificity and generalization

## Model Output

The trained model is saved to:
```
/usr/lib/alembic/checkpoints/genesis/foundation_voxel_cloud.pkl
```

Checkpoints are saved every N documents (default: 5) to:
```
/usr/lib/alembic/checkpoints/genesis/foundation_voxel_cloud_checkpoint_N.pkl
```

## Using the Trained Model

### Command Line

```bash
# Query the Foundation model
python genesis.py synthesize \
    --model /usr/lib/alembic/checkpoints/genesis/foundation_voxel_cloud.pkl \
    --query "What is the meaning of dharma?"
```

### Python API

```python
import pickle
from src.memory.voxel_cloud import VoxelCloud
from src.memory.text_frequency import TextFrequencyAnalyzer

# Load model
with open('/usr/lib/alembic/checkpoints/genesis/foundation_voxel_cloud.pkl', 'rb') as f:
    voxel_cloud = pickle.load(f)

# Query
freq_analyzer = TextFrequencyAnalyzer(512, 512)
query_freq, _ = freq_analyzer.analyze("What is enlightenment?")
visible_protos = voxel_cloud.query_viewport(query_freq, radius=50.0)

# Get responses
for entry, distance in visible_protos[:5]:
    print(f"Text: {entry.metadata.get('text', 'N/A')}")
    print(f"Source: {entry.metadata.get('source', 'Unknown')}")
    print(f"Resonance: {entry.resonance_strength}")
    print("---")
```

## Expected Results

After training on all Foundation documents:

- **Total segments**: ~186,000 sentences
- **Proto-identities**: ~5,000-20,000 (with gravitational collapse)
- **Compression ratio**: 5-10x
- **Resonant patterns**: 30-50% of proto-identities
- **Q&A accuracy**: 60-80% on test suite
- **Training time**: 20-60 minutes

## Metrics and Validation

The test suite evaluates:

1. **Philosophical Understanding**: Questions about concepts from Tao, Gita, Bible, etc.
2. **Historical Knowledge**: Questions about events and characters from epics
3. **Strategic Wisdom**: Questions about strategy and leadership
4. **Compression Quality**: Validates gravitational collapse maintains semantics

Run full validation:
```bash
pytest tests/test_foundation_qa.py -v --tb=short
```

## Troubleshooting

### Out of Memory
- Reduce batch_size parameter
- Process documents sequentially instead of in batches
- Use checkpoint recovery to resume training

### Poor Q&A Performance
- Increase cosine_threshold for less aggressive merging
- Decrease harmonic_tolerance for stricter pattern matching
- Ensure gravitational collapse is enabled

### Slow Training
- Training is CPU-based for stability (GPU support coming)
- Checkpoint frequently to allow resumption
- Consider training on subset for testing

## Architecture Notes

The Foundation training leverages:

1. **Frequency Analysis**: Text → frequency spectrum via FFT
2. **Morphism Application**: Frequency → proto-identity via Gen/Res morphisms
3. **Gravitational Collapse**: Similar patterns merge through wave interference
4. **Resonance Tracking**: Repeated patterns gain strength
5. **Spatial Organization**: Frequency-based 3D positioning in voxel cloud

This creates a compressed, queryable knowledge representation that maintains semantic relationships while achieving significant compression through wave physics principles.