# Genesis: Production Deployment Guide

Comprehensive guide for deploying Genesis in production environments with monitoring, scaling, and troubleshooting.

**Last Updated**: 2025-11-26

---

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [Running Training](#running-training)
5. [Monitoring Metrics](#monitoring-metrics)
6. [Troubleshooting](#troubleshooting)
7. [Performance Tuning](#performance-tuning)
8. [Scaling](#scaling)
9. [Disaster Recovery](#disaster-recovery)

---

## System Requirements

### Minimum Specifications

| Component | Requirement | Notes |
|-----------|-------------|-------|
| **OS** | Linux 5.10+ (kernel) | Arch Linux, Ubuntu 20.04+, RHEL 8+ |
| **Architecture** | x86-64 with AVX2 | Intel Skylake+ or AMD EPYC+ |
| **CPU Cores** | 8+ | 16+ for streaming synthesis |
| **RAM** | 16 GB | 32 GB+ for dual memory hierarchy |
| **GPU** | AMD ROCm 5.0+ | NVIDIA CUDA 11.8+ possible |
| **GPU Memory** | 6 GB VRAM | 8 GB+ for temporal buffers |
| **Storage** | 100 GB SSD | Core + Experiential memory |
| **Network** | Gigabit+ | For distributed training |

### Recommended Specifications (Production)

```
- Dual-socket CPU: 32+ cores, 3.0+ GHz base clock
- RAM: 64 GB DDR4+ with ECC
- GPU: Dual AMD MI250X or NVIDIA H100
- Storage: NVMe RAID 1 (500 GB+) for checkpoints
- Network: 10 Gbps interconnect for multi-GPU
- Backup: Dedicated NAS or S3 for model snapshots
```

### Networking

- **Inbound**: Port 5000 (optional, for training API)
- **Outbound**: Internet access for package updates
- **NFS** (optional): For shared model storage across nodes

---

## Installation

### 1. System Preparation

```bash
# Update system packages
sudo pacman -Syu

# Install dependencies
sudo pacman -S --needed \
    base-devel \
    rustup \
    python \
    python-pip \
    cmake \
    vulkan-tools \
    rocm-core \
    rocm-opencl-runtime \
    git

# Activate Rust toolchain
rustup default stable
rustup update

# Verify versions
python --version    # 3.10+
cargo --version     # 1.70+
rocm-smi           # Check GPU detection
```

### 2. Clone Repository

```bash
# Clone with submodules
git clone --recursive https://github.com/your-org/genesis.git
cd genesis

# Verify submodules
git submodule status
```

### 3. Build from Source

```bash
# Build release binary
./build.sh

# Verify build
ls -lh target/release/libgenesis.so
file target/release/libgenesis.so
```

### 4. Install Python Dependencies

```bash
# Using uv (preferred)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install package dependencies
uv pip install \
    numpy \
    scipy \
    scikit-learn \
    tqdm \
    pyyaml \
    pytest

# Verify installation
python -c "import sys; print(f'Python {sys.version}')"
uv pip list | grep -E 'numpy|scipy|scikit'
```

### 5. Verify GPU Access

```bash
# Check GPU detection
rocm-smi

# Expected output:
# ROCm Agent (0): GPU Device (0x{id})
#   Name: {GPU Model}
#   Uuid: {UUID}
#   VRAM: {Size} MB

# Test GPU with sample code
python -c "
import subprocess
result = subprocess.run(['rocm-smi'], capture_output=True, text=True)
if 'GPU' in result.stdout:
    print('GPU detected successfully')
else:
    print('ERROR: GPU not detected')
"
```

---

## Configuration

### Environment Variables

```bash
# Set environment variables in ~/.bashrc or systemd service

# GPU configuration
export HIP_DEVICE_ORDER=PCI                    # Use PCI ordering for GPU IDs
export HIP_VISIBLE_DEVICES=0                   # Use GPU 0 (or 0,1 for dual)
export ROC_ENABLE_PRE_VEGA=0                   # Disable pre-VEGA (if not needed)

# Resource limits
export KERAS_BACKEND=jax                       # Use JAX backend (optional)

# Logging
export GENESIS_LOG_LEVEL=INFO                  # INFO, DEBUG, WARNING, ERROR
export GENESIS_LOG_FILE=/var/log/genesis/training.log

# Model checkpoints
export GENESIS_CHECKPOINT_DIR=/var/lib/genesis/checkpoints
export GENESIS_MEMORY_DIR=/var/lib/genesis/memory
```

### Application Configuration

**File**: `/home/persist/alembic/genesis/config.yaml`

```yaml
# Training configuration
training:
  max_epochs: 200
  batch_size: 32
  learning_rate: 0.01
  momentum: 0.9
  convergence_threshold: 0.001

# Memory hierarchy
memory_hierarchy:
  core_memory_size: 10000        # Max proto-identities in long-term
  experiential_memory_size: 1000  # Max in short-term
  temporal_buffer_size: 100       # Temporal history length
  auto_reset_threshold: 0.2       # Coherence for auto-reset
  consolidation_threshold: 0.8    # Min coherence to consolidate

# State classifier
state_classifier:
  evolution_threshold: 0.1      # Min derivative for Evolution
  identity_coherence: 0.85      # Min coherence for Identity
  paradox_coherence: 0.3        # Max coherence for Paradox

# Streaming configuration
streaming:
  enable_prediction: true
  taylor_order: 2              # Taylor series expansion order
  prediction_horizon: 0.5      # Seconds ahead to predict
  chunk_size: 1                # Words/frames per chunk

# GPU configuration
gpu:
  device_id: 0
  enable_mixed_precision: false
  max_memory_percent: 60

# Logging
logging:
  level: INFO
  file: /var/log/genesis/training.log
  max_size_mb: 500
  backup_count: 5
```

### Systemd Service (Production)

**File**: `/etc/systemd/system/genesis-training.service`

```ini
[Unit]
Description=Genesis Categorical Training Service
After=network.target

[Service]
Type=simple
User=genesis
WorkingDirectory=/home/genesis/genesis
ExecStart=/usr/bin/python /home/genesis/genesis/genesis.py train --data /usr/lib/alembic/data/datasets/curated/foundation/

# Resource limits
MemoryLimit=32G
CPUQuota=75%
TasksMax=100

# Environment
Environment="PATH=/usr/local/bin:/usr/bin"
Environment="HIP_VISIBLE_DEVICES=0"
Environment="GENESIS_LOG_FILE=/var/log/genesis/training.log"

# Restart policy
Restart=on-failure
RestartSec=30s
StartLimitInterval=600
StartLimitBurst=10

# Logging
StandardOutput=journal
StandardError=journal
SyslogIdentifier=genesis

[Install]
WantedBy=multi-user.target
```

Enable and start service:

```bash
sudo systemctl daemon-reload
sudo systemctl enable genesis-training.service
sudo systemctl start genesis-training.service

# Monitor
sudo systemctl status genesis-training.service
sudo journalctl -u genesis-training.service -f
```

---

## Running Training

### Deployment Modes

**Learning Mode**: Build Core Memory from training data
```bash
genesis.py train --data /usr/lib/alembic/data/datasets/curated/foundation/ --mode learning
# Builds long-term Core Memory from dataset
```

**Interactive Mode**: Use Experiential Memory for conversation
```bash
python scripts/run_interactive.py --mode experiential
# Uses short-term Experiential Memory for dialogue
```

**Streaming Mode**: Real-time audio/video processing
```bash
python scripts/stream_processor.py --mode text --predict
# Processes streaming input with temporal prediction
```

### Operational Commands

```bash
# Start with carrier initialization
genesis serve --init-carrier

# Reset experiential memory
genesis reset --experiential

# Consolidate experiential to core
genesis consolidate --threshold 0.8

# Monitor state distribution
genesis status --states

# Enable streaming mode
genesis stream --mode text --predict
```

### Production Deployment

```bash
# 1. Verify configuration
cat /home/persist/alembic/genesis/config.yaml

# 2. Start service
sudo systemctl start genesis-training.service

# 3. Monitor
sudo journalctl -u genesis-training.service -f

# 4. Verify checkpoint saving
ls -lh /var/lib/genesis/checkpoints/
```

---

## Monitoring Metrics

### Key Performance Indicators

#### Training Metrics

```bash
# Real-time monitoring from logs
tail -f /var/log/genesis/training.log | grep "Epoch"

# Expected output format:
# Epoch 001/100 | Loss: 0.8234 | Cohesion: 0.72 | Clusters: 4 | Memory: 234 items
# Epoch 002/100 | Loss: 0.7891 | Cohesion: 0.74 | Clusters: 4 | Memory: 456 items
# ...
# Epoch 100/100 | Loss: 0.0234 | Cohesion: 0.96 | Clusters: 8 | Memory: 3567 items
```

#### System Resources

```bash
# GPU Memory
rocm-smi --json | jq '.[] | {gpu_id: .product_name, vram_used: .mem_info.vram_used}'

# CPU/Memory
free -h
ps aux | grep genesis

# Disk I/O
iostat -x 1 5

# Network (if distributed)
nethogs -c
```

### Metrics to Track

| Metric | Target | Warning | Critical |
|--------|--------|---------|----------|
| **Factorization Loss** | < 0.001 | > 0.01 | > 0.05 |
| **Cohesion Score** | > 0.95 | < 0.85 | < 0.70 |
| **State Distribution** | Balanced | Paradox > 50% | Paradox > 70% |
| **Exp/Core Coherence** | > 0.6 | < 0.4 | < 0.2 |
| **Temporal Derivative** | < 1.0 | > 2.0 | > 5.0 |
| **Reset Frequency** | < 10/hr | > 20/hr | > 50/hr |
| **Consolidation Rate** | > 5/hr | < 2/hr | < 1/hr |
| **GPU Memory Usage** | < 60% | > 75% | > 90% |

### Prometheus Metrics (Optional)

For integration with monitoring systems, export metrics:

```python
# Example: Export metrics to Prometheus
from prometheus_client import start_http_server, Gauge, Counter
import time

# Define metrics
loss_gauge = Gauge('genesis_loss', 'Factorization loss')
cohesion_gauge = Gauge('genesis_cohesion', 'Coherence score')
epoch_counter = Counter('genesis_epochs', 'Total epochs completed')

# Start HTTP server
start_http_server(8000)

# In training loop:
loss_gauge.set(current_loss)
cohesion_gauge.set(current_cohesion)
epoch_counter.inc()
```

Access metrics at `http://localhost:8000/metrics`

---

## Troubleshooting

### Common Issues

**GPU Not Detected**: Run `rocm-smi`, check `lsmod | grep amdgpu`, verify render group membership

**Out of Memory**: Clear cache with `rocm-smi --resetpfm`, reduce batch size or buffer size

**Training Not Converging**: Check state distribution, adjust thresholds in config

**High GPU Temperature**: Monitor with `rocm-smi --showtemp`, reduce utilization

**Checkpoint Failures**: Verify permissions and disk space in `/var/lib/genesis/`

**Data Loading Errors**: Validate tensor shapes match expected dimensions

---

## Performance Tuning

### Temporal-Specific Optimizations

#### 1. Temporal Buffer Sizing
```yaml
# Optimize based on use case
temporal_buffer_size: 100   # Text: 100 tokens
temporal_buffer_size: 500   # Audio: 500ms chunks
temporal_buffer_size: 30    # Video: 30 frames (1s @ 30fps)
```

#### 2. Prediction Horizon Tuning
```yaml
# Balance accuracy vs computation
prediction_horizon: 0.1    # Fast response, lower accuracy
prediction_horizon: 0.5    # Balanced (recommended)
prediction_horizon: 1.0    # High accuracy, more compute
```

#### 3. State Transition Hysteresis
```python
# Prevent rapid state oscillations
hysteresis_window = 5  # Require 5 consistent classifications
min_state_duration = 0.5  # Minimum seconds in state
```

#### 4. Feedback Loop Frequency
```yaml
# Adjust based on modality
text_feedback_hz: 10      # 10 Hz for text
audio_feedback_hz: 100    # 100 Hz for audio
video_feedback_hz: 30     # 30 Hz for video
```

### Memory Hierarchy Tuning

```yaml
# Core Memory (slow update, stable)
core_update_rate: 0.01    # Low learning rate
core_momentum: 0.99       # High momentum

# Experiential Memory (fast update, adaptive)
exp_update_rate: 0.1      # High learning rate
exp_momentum: 0.9         # Lower momentum
```

### Streaming Performance

```bash
# Enable prediction for smoother output
genesis stream --predict --taylor-order 2

# Disable for lower latency
genesis stream --no-predict

# Adjust chunk size for throughput vs latency
genesis stream --chunk-size 1   # Low latency
genesis stream --chunk-size 10  # Higher throughput
```

---

## Scaling

### Vertical Scaling (More GPU Compute)

Current architecture supports single-GPU only.

**Upgrade path**:
1. CPU: Increase cores for batch preprocessing
2. GPU: Use higher-end GPU (MI250X) for more VRAM
3. RAM: Increase for larger memory pools

### Horizontal Scaling (Multiple Nodes)

**Planned for v0.2.0**:
- Ray distributed training
- Model parameter server
- Collective communication (NCCL)

**Interim workaround**: Train multiple independent models on separate nodes, merge learned memories.

### Memory Pool Scaling

```bash
# Increase max proto-identities
genesis.py train --data /usr/lib/alembic/data/datasets/curated/foundation/ --max-segments 1000

# Monitor memory usage
watch -n 1 'ps aux | grep genesis | awk "{print \$6, \$7, \$8, \$9}"'
```

---

## Disaster Recovery

### Checkpoint Recovery

```bash
# List available checkpoints
ls -lh /var/lib/genesis/checkpoints/

# Resume from specific checkpoint
# Resume from checkpoint
genesis.py train --resume ./checkpoints/genesis_checkpoint.pkl

# Recovery expected: Ouroboros cycle resumes from saved state
```

### Data Backup Strategy

```bash
# Daily backup of checkpoints and memory
0 2 * * * rsync -av /var/lib/genesis/checkpoints/ /backup/genesis/checkpoints/
0 2 * * * rsync -av /var/lib/genesis/memory/ /backup/genesis/memory/

# Verify backups
ls -lh /backup/genesis/checkpoints/ | tail -5
```

### Log Rotation

```bash
# /etc/logrotate.d/genesis
/var/log/genesis/training.log {
    daily
    rotate 30
    compress
    delaycompress
    notifempty
    create 0644 genesis genesis
    sharedscripts
}

# Test rotation
sudo logrotate -f /etc/logrotate.d/genesis
```

### Emergency Shutdown

```bash
# Graceful shutdown (saves checkpoint)
sudo systemctl stop genesis-training.service
# Waits for current epoch to complete, then saves state

# Force shutdown (loses current epoch)
sudo systemctl kill -9 genesis-training.service
```

---

## Performance Baseline

### Reference Hardware

**Test Configuration**:
- CPU: AMD EPYC 7003 (32 cores)
- GPU: AMD MI250X (128 GB VRAM)
- RAM: 256 GB DDR4
- Storage: NVMe RAID 1

**Baseline Metrics**:
- Epoch time: 3-5 seconds
- Batch size: 256
- GPU utilization: 85-95%
- Convergence time: 50-100 epochs
- Final loss: 0.0001-0.0005
- Final cohesion: 0.96-0.99

### Expected Variance

Different hardware will show:
- Small batch size (16): 2x slower epoch time, 20% lower GPU utilization
- High learning rate (0.1): 30% faster initial convergence, higher oscillation
- Large cluster count (32): 40% longer memory update time, finer learned structure

---

## Security Considerations

### File Permissions

```bash
# Restrict access to model files
chmod 750 /var/lib/genesis/
chmod 640 /var/lib/genesis/checkpoints/*
chown genesis:genesis /var/lib/genesis/

# Log file access
chmod 640 /var/log/genesis/*
chown genesis:genesis /var/log/genesis/
```

### Network Security

```bash
# If using training API (future feature)
# Bind only to localhost
GENESIS_API_HOST=127.0.0.1:5000

# Use firewall to restrict access
sudo ufw allow 22/tcp   # SSH only
sudo ufw allow from 192.168.1.0/24 to any port 5000
```

### Model Validation

```bash
# Before deployment, verify model integrity
sha256sum /var/lib/genesis/checkpoints/latest.ckpt
# Compare against known good hash

# Run validation suite
pytest tests/ -v --tb=short
```

---

## Further Reading

- **ARCHITECTURE.md**: Detailed system design
- **README.md**: Quick start and overview
- **tests/**: Working examples and integration tests

For additional support or issues, consult the test suite examples or open an issue on GitHub.
