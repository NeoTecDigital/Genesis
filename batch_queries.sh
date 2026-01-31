#!/bin/bash
# Batch query script - run multiple queries and save outputs

MODEL="output/tao_foundation.pkl"
QUERIES=(
    "What is the Tao?"
    "What is wisdom?"
    "How should I live?"
    "What is virtue?"
    "What is true power?"
)

for i in "${!QUERIES[@]}"; do
    num=$(printf "%02d" $((i+1)))
    query="${QUERIES[$i]}"
    out="output/foundation_queries/query_${num}.txt"
    
    echo "[$((i+1))/${#QUERIES[@]}] Running: $query"
    python genesis.py synthesize --model "$MODEL" --query "$query" --output "$out" 2>&1 | tail -20
    echo ""
done

echo "Complete: ${#QUERIES[@]} queries → output/foundation_queries/"
ls -lh output/foundation_queries/
