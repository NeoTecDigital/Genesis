#!/usr/bin/env python
"""End-to-end test: Query Tao knowledge and save outputs."""
import sys
import pickle
import numpy as np
from pathlib import Path
from src.origin import Origin
from src.memory.frequency_field import TextFrequencyAnalyzer
from src.pipeline.fft_text_decoder import FFTTextDecoder

MODEL = "/usr/lib/alembic/checkpoints/genesis/tao_voxel_cloud.pkl"
OUT_DIR = Path("output/foundation_queries")
OUT_DIR.mkdir(parents=True, exist_ok=True)

queries = [
    "What is the Tao?",
    "What is wisdom?",
    "How should I live?",
    "What is virtue?",
    "What is true power?",
    "How do I find peace?",
    "What is the nature of reality?",
    "What is the way of the sage?"
]

print(f"Loading: {MODEL}")
with open(MODEL, 'rb') as f:
    cloud = pickle.load(f)
print(f"Loaded: {cloud}\n")

origin = Origin(width=512, height=512, use_gpu=False)
analyzer = TextFrequencyAnalyzer(width=512, height=512)
decoder = FFTTextDecoder()

for i, q in enumerate(queries, 1):
    print(f"[{i}/{len(queries)}] {q}")
    out = OUT_DIR / f"query_{i:02d}.txt"
    
    # Convert query
    _, params = analyzer.analyze(q)
    proto = origin.Gen(params['gamma_params'], params['iota_params'])
    result = origin.Act(proto)
    
    # Query cloud
    matches = cloud.query_multi_octave(result.multi_octave_quaternions, top_k=5)
    
    # Weighted synthesis
    dists = np.array([d for _, d in matches])
    if dists.max() > 0:
        dists = dists / dists.max()
    weights = np.exp(-dists * 5.0)
    weights /= weights.sum()
    
    # Save
    with open(out, 'w') as f:
        f.write(f"QUERY: {q}\n{'='*60}\n\n")
        f.write("SYNTHESIZED RESPONSE:\n" + "-"*60 + "\n")
        for (p, _), w in zip(matches, weights):
            # Decode text from proto_identity using FFT decoder
            decoded_text = decoder.decode_text(p.proto_identity) if hasattr(p, 'proto_identity') else "N/A"
            f.write(f"[{w*100:.1f}%] {decoded_text}\n")
        f.write("\n" + "-"*60 + "\nTOP MATCHES:\n" + "-"*60 + "\n")
        for j, (p, d) in enumerate(matches, 1):
            decoded_text = decoder.decode_text(p.proto_identity) if hasattr(p, 'proto_identity') else "N/A"
            f.write(f"{j}. Distance: {d:.4f} | Resonance: {p.resonance_strength:.1f}\n")
            f.write(f"   {decoded_text}\n\n")
    
    print(f"  → Saved: {out}")
    if matches:
        decoded_text = decoder.decode_text(matches[0][0].proto_identity) if hasattr(matches[0][0], 'proto_identity') else "N/A"
        print(f"  → Top: [{weights[0]*100:.1f}%] {decoded_text[:50]}...\n")

print(f"\n✓ Complete: {len(queries)} queries → {OUT_DIR}")
