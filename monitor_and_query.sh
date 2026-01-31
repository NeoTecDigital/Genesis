#!/bin/bash
# Proper monitor: Wait for training completion, then query

LOG="/tmp/foundation_training.log"
MODEL="/usr/lib/alembic/checkpoints/genesis/foundation_voxel_cloud.pkl"
OUT="output/foundation_queries"
mkdir -p "$OUT"

echo "Monitoring foundation training..."
echo "Progress: $(grep -c "Processing:" $LOG)/24 datasets"

# Wait for training to actually complete
while true; do
    # Check if final model exists AND training process is done
    if [ -f "$MODEL" ] && ! ps aux | grep -q "[t]rain_foundation.py"; then
        echo "✓ Training complete! Model found at $MODEL"
        break
    fi
    
    # Show progress
    datasets_done=$(grep -c "Processing:" $LOG)
    last_line=$(tail -1 $LOG)
    echo "[$(date '+%H:%M:%S')] Progress: $datasets_done/24 - $last_line"
    sleep 60
done

echo ""
echo "Running queries on foundation knowledge..."
echo "Model: $MODEL ($(ls -lh $MODEL | awk '{print $5}'))"

QUERIES=(
    "What is the Tao?"
    "What is wisdom?"
    "How should one live?"
    "What is virtue?"
    "What is compassion?"
)

for i in "${!QUERIES[@]}"; do
    num=$(printf "%02d" $((i+1)))
    query="${QUERIES[$i]}"
    out="$OUT/query_${num}.txt"
    
    echo ""
    echo "=== Query $num/${#QUERIES[@]}: $query ==="
    
    timeout 180 python genesis.py synthesize \
        --model "$MODEL" \
        --query "$query" \
        --output "$out" 2>&1 | tail -40
    
    if [ -f "$out" ]; then
        echo "✓ Saved: $out"
        echo "Preview:"
        head -15 "$out"
    else
        echo "✗ Failed"
    fi
done

echo ""
echo "========================================"
echo "COMPLETE: ${#QUERIES[@]} foundation queries"
echo "========================================"
ls -lh "$OUT"
