#!/usr/bin/env python
"""
End-to-end test: Load foundation model and run queries, saving all outputs.
"""
import sys
import pickle
from pathlib import Path

# Use the latest checkpoint (15 datasets processed)
MODEL_PATH = "/usr/lib/alembic/checkpoints/genesis/foundation_voxel_cloud_checkpoint_15.pkl"
OUTPUT_DIR = Path("output/foundation_queries")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

queries = [
    "What is the Tao?",
    "What is wisdom?",
    "How should I live?",
    "What is virtue?",
    "What is power?",
    "What is compassion?",
    "What is the nature of reality?",
    "How do I find inner peace?"
]

print(f"Loading foundation model from: {MODEL_PATH}")
print(f"Model size: {Path(MODEL_PATH).stat().st_size / (1024**3):.1f} GB")

try:
    with open(MODEL_PATH, 'rb') as f:
        voxel_cloud = pickle.load(f)
    print(f"✓ Loaded: {voxel_cloud}")
    print(f"  Proto-identities: {len(voxel_cloud.entries)}")
    
    # Now import genesis to run synthesis
    from src.origin import Origin
    from src.memory.frequency_field import TextFrequencyAnalyzer
    from src.pipeline.fft_text_decoder import FFTTextDecoder

    origin = Origin(width=512, height=512, use_gpu=False)
    analyzer = TextFrequencyAnalyzer(width=512, height=512)
    decoder = FFTTextDecoder()
    
    print(f"\nRunning {len(queries)} queries on foundation knowledge...\n")
    
    for i, query in enumerate(queries, 1):
        print(f"[{i}/{len(queries)}] Query: {query}")
        output_file = OUTPUT_DIR / f"query_{i:02d}.txt"
        
        try:
            # Convert query to proto-identity
            freq_spectrum, params = analyzer.analyze(query)
            query_proto = origin.Gen(params['gamma_params'], params['iota_params'], input_n=None)
            query_result = origin.Act(query_proto)
            
            # Multi-octave query
            results = voxel_cloud.query_multi_octave(query_result.multi_octave_quaternions, top_k=5)
            
            if results:
                # Synthesis with exponential decay
                import numpy as np
                distances = np.array([d for _, d in results])
                if distances.max() > 0:
                    distances = distances / distances.max()
                decay_factor = 5.0
                weights = np.exp(-distances * decay_factor)
                weights /= weights.sum()
                
                # Save output
                with open(output_file, 'w') as f:
                    f.write(f"Query: {query}\n")
                    f.write(f"=" * 60 + "\n\n")
                    f.write("SYNTHESIZED RESPONSE:\n")
                    f.write("-" * 60 + "\n")
                    for (proto, distance), weight in zip(results, weights):
                        decoded_text = decoder.decode_text(proto.proto_identity) if hasattr(proto, 'proto_identity') else "N/A"
                        f.write(f"[{weight*100:.1f}%] {decoded_text}\n")
                    f.write("-" * 60 + "\n\n")
                    f.write("TOP MATCHING PROTO-IDENTITIES:\n")
                    f.write("-" * 60 + "\n")
                    for j, (proto, distance) in enumerate(results, 1):
                        decoded_text = decoder.decode_text(proto.proto_identity) if hasattr(proto, 'proto_identity') else "N/A"
                        f.write(f"{j}. Distance: {distance:.4f}\n")
                        f.write(f"   Text: {decoded_text}\n")
                        f.write(f"   Frequency: {proto.frequency:.2f} Hz\n")
                        f.write(f"   Resonance: {proto.resonance_strength:.1f}\n\n")
                
                print(f"  ✓ Saved to: {output_file}")
                decoded_text = decoder.decode_text(results[0][0].proto_identity) if hasattr(results[0][0], 'proto_identity') else "N/A"
                print(f"  Top match: {decoded_text[:60]}...")
            else:
                print(f"  ✗ No results found")
                
        except Exception as e:
            print(f"  ✗ Error: {e}")
            with open(output_file, 'w') as f:
                f.write(f"Query: {query}\n")
                f.write(f"ERROR: {e}\n")
        
        print()
    
    print(f"\n{'='*60}")
    print(f"COMPLETE: All query outputs saved to {OUTPUT_DIR}")
    print(f"{'='*60}")
    
except Exception as e:
    print(f"✗ Failed to load model: {e}")
    sys.exit(1)
