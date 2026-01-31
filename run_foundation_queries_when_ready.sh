#!/bin/bash
# Wait for foundation training to complete, then run queries

MODEL="/usr/lib/alembic/checkpoints/genesis/foundation_voxel_cloud.pkl"
OUT_DIR="output/foundation_queries"
mkdir -p "$OUT_DIR"

# Wait for training to complete
echo "Waiting for foundation training to complete..."
while ps aux | grep -q "[t]rain_foundation.py"; do
    progress=$(tail -1 /tmp/foundation_training.log)
    echo "$(date '+%H:%M:%S') - Still training: $progress"
    sleep 30
done

echo "✓ Training complete! Running queries..."

QUERIES=(
    "What is the Tao?"
    "What is wisdom?"
    "How should one live?"
    "What is virtue?"
    "What is true power?"
    "How do I find inner peace?"
    "What is the nature of reality?"
    "What is compassion?"
)

for i in "${!QUERIES[@]}"; do
    num=$(printf "%02d" $((i+1)))
    query="${QUERIES[$i]}"
    out="$OUT_DIR/query_${num}.txt"
    
    echo ""
    echo "[${num}/${#QUERIES[@]}] Query: $query"
    timeout 120 python genesis.py synthesize --model "$MODEL" --query "$query" --output "$out" 2>&1 | tail -30
    
    if [ -f "$out" ]; then
        echo "  ✓ Saved: $out"
        head -20 "$out"
    else
        echo "  ✗ Failed to save output"
    fi
done

echo ""
echo "========================================="
echo "COMPLETE: ${#QUERIES[@]} queries saved"
echo "========================================="
ls -lh "$OUT_DIR"
