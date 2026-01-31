# Genesis Scripts

All Genesis operations are now unified through the `genesis.py` CLI. Legacy scripts have been archived as they are incompatible with the new MemoryHierarchy architecture.

## Using genesis.py

The main CLI provides all functionality:

```bash
# Show help
genesis.py --help

# Show specific command help
genesis.py train --help
```

## Available Commands

### Training
Train Genesis on foundation data to build core memory:

```bash
# Train with foundation dataset
genesis.py train \
  --data /usr/lib/alembic/data/datasets/curated/foundation/ \
  --output ./models/genesis_foundation.pkl \
  --max-documents 100 \
  --checkpoint-interval 10

# Resume from checkpoint
genesis.py train \
  --data /usr/lib/alembic/data/datasets/curated/foundation/ \
  --output ./models/genesis_foundation.pkl \
  --resume ./checkpoints/genesis_checkpoint_50.pkl
```

### Interactive Chat
Have conversations with Genesis:

```bash
# Chat with trained model
genesis.py chat --model ./models/genesis_foundation.pkl

# Enable response streaming
genesis.py chat --model ./models/genesis_foundation.pkl --stream

# Adjust temperature
genesis.py chat --model ./models/genesis_foundation.pkl --temperature 0.7
```

### Evaluation
Run A/B testing and benchmarks:

```bash
# Evaluate model performance
genesis.py eval \
  --model ./models/genesis_foundation.pkl \
  --test-cases tests/test_cases.json \
  --metrics coherence,diversity,stability

# Compare two models
genesis.py eval \
  --model-a ./models/genesis_v1.pkl \
  --model-b ./models/genesis_v2.pkl \
  --test-cases tests/test_cases.json
```

### Testing
Run the comprehensive test suite:

```bash
# Run all tests
genesis.py test

# Run specific phase tests
genesis.py test --phase 1

# Run with GPU enabled
genesis.py test --gpu

# Run benchmarks
genesis.py test --benchmark
```

### Discovery & Synthesis (Legacy Commands)
These commands are being updated to use the new architecture:

```bash
# Discover patterns (currently being updated)
genesis.py discover --input "explore consciousness"

# Synthesize responses (currently being updated)
genesis.py synthesize --prompt "tell me about wisdom"
```

## Directory Structure

```
scripts/
├── README.md              # This file
├── archive/               # Legacy scripts (deprecated)
│   ├── README.md         # Explanation of archived scripts
│   ├── train_foundation.py
│   ├── demo_foundation.py
│   └── test_foundation.py
└── fine_tune_params.py    # Parameter tuning utilities (still valid)
```

## Legacy Scripts

The following scripts have been archived in `scripts/archive/` as they use the pre-Phase 1-6 architecture:

- **train_foundation.py**: Used Gen() directly without MemoryHierarchy
- **demo_foundation.py**: Old demonstration patterns
- **test_foundation.py**: Pre-integration testing approach

These scripts are incompatible with the current architecture which includes:
- MemoryHierarchy (3-layer: Core ↔ Experiential ↔ External)
- TemporalBuffer with state classification
- ConversationPipeline with real-time handling
- StreamingSynthesizer with encoding/decoding pipelines

## Migration Guide

If you were using the old scripts, here's how to migrate:

| Old Command | New Command |
|------------|-------------|
| `python scripts/train_foundation.py` | `genesis.py train --data /path/to/data --output model.pkl` |
| `python scripts/demo_foundation.py` | `genesis.py chat --model model.pkl` |
| `python scripts/test_foundation.py` | `genesis.py test` |

## Parameter Tuning

The `fine_tune_params.py` script remains valid for parameter optimization:

```bash
# Run parameter tuning
python scripts/fine_tune_params.py \
  --param-grid configs/param_grid.yaml \
  --output results/tuning_results.json
```

## Notes

- All commands now properly use the Phases 1-6 architecture
- GPU support is available but disabled by default (use `--gpu` flag)
- Models are saved in pickle format with full state preservation
- Checkpointing is automatic during training

For more details, see:
- `docs/INTEGRATION_PLAN.md` - Full integration roadmap
- `docs/ARCHITECTURE.md` - System architecture details
- `examples/conversation_demo.py` - Example usage