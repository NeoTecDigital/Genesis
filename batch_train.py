#!/usr/bin/env python3
"""Comprehensive parameter grid sweep across cosine, harmonic, and octave ranges."""

import subprocess
import json
import itertools
from pathlib import Path
from datetime import datetime

# Parameter grids - Optimized gradient ranges (12×12×6 = 864 configs)
COSINE_RANGE = [0.0, 0.0275, 0.05, 0.125, 0.25, 0.5, 0.75, 0.875, 0.95, 0.9725, 1.0]
HARMONIC_RANGE = [0.0, 0.0275, 0.05, 0.125, 0.25, 0.5, 0.75, 0.875, 0.95, 0.9725, 1.0]
OCTAVE_RANGE = [0, 1, 3, 5, 7, 12]

def generate_configs():
    """Generate all parameter combinations with meaningful names."""
    configs = {}

    for cosine, harmonic, octave in itertools.product(COSINE_RANGE, HARMONIC_RANGE, OCTAVE_RANGE):
        # Generate meaningful name
        cosine_label = f"c{int(cosine*100):03d}"  # c000, c030, c100, etc
        harmonic_label = f"h{int(harmonic*100):03d}"  # h000, h001, h050, etc
        octave_label = f"o{octave:02d}"  # o00, o01, o12, etc

        config_name = f"{cosine_label}_{harmonic_label}_{octave_label}"

        configs[config_name] = {
            "cosine": cosine,
            "harmonic": harmonic,
            "octave": octave,
            "desc": f"cosine={cosine:.2f}, harmonic={harmonic:.2f}, octave=±{octave}"
        }

    return configs

def train_with_config(config_name, config, data_path, output_dir):
    """Train model with specific configuration."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"{config_name}.pkl"  # No timestamp in name for clarity
    output_path = output_dir / model_name

    cmd = [
        "python", "genesis.py", "train",
        "--data", str(data_path),
        "--output", str(output_path),
        "--collapse-cosine-threshold", str(config["cosine"]),
        "--collapse-harmonic-tolerance", str(config["harmonic"]),
        "--collapse-octave-tolerance", str(config["octave"]),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    # Parse output for metrics
    output_lines = result.stdout.split('\n')
    metrics = {
        "config_name": config_name,
        "cosine": config["cosine"],
        "harmonic": config["harmonic"],
        "octave": config["octave"],
        "model_path": str(output_path),
        "timestamp": timestamp,
        "proto_count": None,
        "document_count": None,
        "segments": None,
        "success": result.returncode == 0,
    }

    for line in output_lines:
        if "Proto-identities in core:" in line:
            metrics["proto_count"] = int(line.split(':')[1].strip())
        elif "Documents processed:" in line:
            metrics["document_count"] = int(line.split(':')[1].strip())
        elif "Total segments:" in line:
            metrics["segments"] = int(line.split(':')[1].strip())

    if metrics['success'] and metrics['proto_count']:
        metrics['ratio'] = metrics['proto_count'] / metrics['document_count'] if metrics['document_count'] else 0

    return metrics

def main():
    """Run comprehensive parameter grid sweep."""
    # Use small test documents for fast parameter exploration
    data_path = Path("./tmp/sweep_test_docs")
    output_dir = Path("./tmp/models/grid_sweep")
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path.mkdir(parents=True, exist_ok=True)

    # Create small test documents (100 lines each)
    import shutil
    foundation_path = Path("/usr/lib/alembic/data/datasets/text/curated/foundation")
    foundation_files = sorted(foundation_path.glob("*.txt"))[:3]

    for i, f in enumerate(foundation_files):
        # Only use first 100 lines of each doc
        with open(f, 'r') as src:
            lines = src.readlines()[:100]
        with open(data_path / f"doc{i+1}.txt", 'w') as dst:
            dst.writelines(lines)

    print(f"\n{'='*90}")
    print("GENESIS COMPREHENSIVE PARAMETER GRID SWEEP")
    print(f"{'='*90}")
    print(f"Cosine range: {COSINE_RANGE}")
    print(f"Harmonic range: {HARMONIC_RANGE}")
    print(f"Octave range: {OCTAVE_RANGE}")

    configs = generate_configs()
    total_configs = len(configs)

    # Count created files
    created_files = list(data_path.glob("*.txt"))
    print(f"\nTotal configurations: {total_configs}")
    print(f"Data: {data_path} ({len(created_files)} documents, ~100 lines each)")
    print(f"Output: {output_dir}")
    print(f"{'='*90}\n")

    results = []
    start_time = datetime.now()

    for i, (config_name, config) in enumerate(configs.items(), 1):
        elapsed = (datetime.now() - start_time).total_seconds()
        avg_time = elapsed / i if i > 1 else 0
        eta = avg_time * (total_configs - i)

        print(f"[{i}/{total_configs}] {config_name} (ETA: {eta/60:.1f}min)... ", end='', flush=True)

        try:
            metrics = train_with_config(config_name, config, data_path, output_dir)
            results.append(metrics)

            if metrics['success'] and metrics.get('proto_count'):
                print(f"✓ {metrics['proto_count']} protos (ratio={metrics['ratio']:.2f})")
            else:
                print(f"✗ FAILED")

        except Exception as e:
            print(f"✗ ERROR: {str(e)[:50]}")
            results.append({
                "config_name": config_name,
                "cosine": config["cosine"],
                "harmonic": config["harmonic"],
                "octave": config["octave"],
                "success": False,
                "error": str(e)
            })

    # Save results
    results_path = output_dir / f"grid_sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Generate CSV for plotting
    csv_path = output_dir / "sweep_results.csv"
    with open(csv_path, 'w') as f:
        f.write("config_name,cosine,harmonic,octave,proto_count,document_count,ratio,success\n")
        for r in results:
            if r['success']:
                f.write(f"{r['config_name']},{r['cosine']},{r['harmonic']},{r['octave']},{r.get('proto_count', 0)},{r.get('document_count', 0)},{r.get('ratio', 0):.4f},True\n")
            else:
                f.write(f"{r['config_name']},{r['cosine']},{r['harmonic']},{r['octave']},0,0,0,False\n")

    # Print summary
    print(f"\n{'='*90}")
    print("GRID SWEEP COMPLETE")
    print(f"{'='*90}\n")

    successful = [r for r in results if r['success'] and r.get('proto_count')]
    if successful:
        print(f"Successful configs: {len(successful)}/{total_configs}")
        print(f"\nTop 10 by diversity (ratio closest to 1.0):")
        print(f"{'Config':<30} {'Cosine':<8} {'Harm':<8} {'Oct':<5} {'Protos':<8} {'Ratio':<8}")
        print("-" * 90)

        sorted_by_ratio = sorted(successful, key=lambda x: abs(x.get('ratio', 0) - 1.0))[:10]
        for r in sorted_by_ratio:
            print(f"{r['config_name']:<30} {r['cosine']:<8.2f} {r['harmonic']:<8.3f} {r['octave']:<5} {r['proto_count']:<8} {r['ratio']:<8.2f}")

    print(f"\n{'='*90}")
    print(f"Results saved to:")
    print(f"  JSON: {results_path}")
    print(f"  CSV:  {csv_path}")
    print(f"\nPlot results with:")
    print(f"  import pandas as pd")
    print(f"  import matplotlib.pyplot as plt")
    print(f"  df = pd.read_csv('{csv_path}')")
    print(f"  # 3D scatter: cosine vs harmonic vs ratio, colored by octave")
    print(f"{'='*90}\n")

if __name__ == "__main__":
    main()
