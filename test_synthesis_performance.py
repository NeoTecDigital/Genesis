#!/usr/bin/env python3
"""Performance validation for hierarchical synthesis."""

import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline.unified_encoder import UnifiedEncoder
from src.pipeline.unified_decoder import UnifiedDecoder
from src.memory.memory_hierarchy import MemoryHierarchy

print("=" * 80)
print("HIERARCHICAL SYNTHESIS - PERFORMANCE VALIDATION")
print("=" * 80)

# Initialize
print("\n[1/3] Initializing components...")
memory = MemoryHierarchy()
encoder = UnifiedEncoder(memory)
decoder = UnifiedDecoder(memory)
print("   ✓ Components ready")

# Encode foundation data
print("\n[2/3] Encoding foundation knowledge...")
foundation_texts = [
    "The supreme art of war is to subdue the enemy without fighting",
    "Strategy without tactics is the slowest route to victory",
    "All warfare is based on deception",
    "Victory comes from finding opportunities in problems",
    "Know the enemy and know yourself"
]

for i, text in enumerate(foundation_texts, 1):
    encoder.encode(text, destination='core')
    print(f"   {i}. '{text[:50]}...'")

# Test performance across query sizes
print("\n[3/3] Testing synthesis performance...")
queries = [
    "war",
    "war strategy",
    "war strategy tactics",
    "war strategy tactics enemy victory"
]

results = []
for query in queries:
    # Encode query
    q = encoder.encode(query, destination='experiential')

    # Time synthesis
    start = time.time()
    result = decoder.hierarchical_synthesis(q.octave_units[0].proto_identity)
    duration = time.time() - start

    results.append({
        'query': query,
        'query_chars': len(query),
        'query_words': len(query.split()),
        'synthesis': result.text,
        'synth_chars': len(result.text),
        'synth_words': len(result.text.split()),
        'confidence': result.confidence,
        'time_ms': duration * 1000
    })

# Display results
print("\n" + "=" * 80)
print("PERFORMANCE RESULTS")
print("=" * 80)

for r in results:
    print(f"\nQuery: '{r['query']}' ({r['query_chars']} chars, {r['query_words']} words)")
    print(f"  Synthesis: '{r['synthesis']}' ({r['synth_chars']} chars, {r['synth_words']} words)")
    print(f"  Confidence: {r['confidence']:.3f}")
    print(f"  Time: {r['time_ms']:.2f}ms {'✓' if r['time_ms'] < 100 else '⚠'}")

# Summary statistics
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

avg_time = sum(r['time_ms'] for r in results) / len(results)
max_time = max(r['time_ms'] for r in results)
min_time = min(r['time_ms'] for r in results)

print(f"\nTiming Statistics:")
print(f"  Average: {avg_time:.2f}ms")
print(f"  Min: {min_time:.2f}ms")
print(f"  Max: {max_time:.2f}ms")

print(f"\nSynthesis Quality:")
print(f"  Avg words synthesized: {sum(r['synth_words'] for r in results) / len(results):.1f}")
print(f"  Avg confidence: {sum(r['confidence'] for r in results) / len(results):.3f}")

# Quality gates
print("\n" + "=" * 80)
print("QUALITY GATES")
print("=" * 80)

gates_passed = []
gates_failed = []

if max_time < 100:
    gates_passed.append("✓ All queries < 100ms")
else:
    gates_failed.append(f"✗ Max time {max_time:.2f}ms exceeds 100ms threshold")

if all(r['synth_words'] > 0 for r in results):
    gates_passed.append("✓ All queries produced text")
else:
    gates_failed.append("✗ Some queries produced no text")

if all(r['confidence'] > 0 for r in results):
    gates_passed.append("✓ All queries have confidence > 0")
else:
    gates_failed.append("✗ Some queries have zero confidence")

for gate in gates_passed:
    print(f"\n{gate}")

for gate in gates_failed:
    print(f"\n{gate}")

print("\n" + "=" * 80)
if gates_failed:
    print("⚠ PERFORMANCE VALIDATION: PARTIAL PASS")
    print("Some quality gates failed - review recommended")
else:
    print("✅ PERFORMANCE VALIDATION: FULL PASS")
    print("All quality gates passed - ready for production")
print("=" * 80)
