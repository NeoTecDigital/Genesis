#!/bin/bash
# Process ALL 24 foundation datasets using the working genesis.py discover command

OUT_DIR="/usr/lib/alembic/checkpoints/genesis/foundation"
mkdir -p "$OUT_DIR"

FILES=(/usr/lib/alembic/data/datasets/curated/foundation/*.txt)
TOTAL=${#FILES[@]}

echo "Processing $TOTAL foundation datasets..."
echo "Output: $OUT_DIR"
echo ""

for i in "${!FILES[@]}"; do
    file="${FILES[$i]}"
    name=$(basename "$file" .txt)
    out="$OUT_DIR/${name}.pkl"
    
    num=$((i+1))
    echo "[$num/$TOTAL] Processing: $name"
    
    timeout 600 python genesis.py discover \
        --input "$file" \
        --output "$out" \
        --dual-path 2>&1 | tail -5
    
    if [ -f "$out" ]; then
        size=$(du -h "$out" | cut -f1)
        echo "  ✓ Saved: $out ($size)"
    else
        echo "  ✗ Failed"
    fi
    echo ""
done

echo "Complete: $TOTAL datasets processed"
ls -lh "$OUT_DIR"/*.pkl
