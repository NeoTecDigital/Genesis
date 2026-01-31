"""Synthesis command implementation for Genesis CLI."""
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict

from src.origin import Origin
from src.memory.frequency_field import TextFrequencyAnalyzer
from src.memory.voxel_cloud import VoxelCloud


def cmd_synthesize(args):
    """Synthesize new proto-identity from voxel cloud."""
    print("=" * 60)
    print("Genesis Pattern Synthesis (Viewport → Synthesis)")
    print("=" * 60)

    # Load voxel cloud
    if not Path(args.model).exists():
        print(f"❌ Model not found: {args.model}")
        return 1

    print(f"\n📂 Loading voxel cloud: {args.model}")
    voxel_cloud, meta, use_dual = _load_voxel_cloud_and_metadata(args.model)

    # Override synthesis config with CLI args if provided
    _update_synthesis_config(voxel_cloud, args)

    _print_metadata(meta, use_dual)
    print(f"  Voxel cloud: {voxel_cloud}")
    print(f"  Synthesis config: {voxel_cloud.synthesis_config}")

    # Convert query to frequency
    print(f"\n🔍 Query: {args.query}")
    freq_analyzer = TextFrequencyAnalyzer(512, 512)
    query_freq, query_params = freq_analyzer.analyze(args.query)

    if args.debug:
        _debug_frequency_analysis(query_freq, query_params)

    # Query and synthesize
    result = _query_and_synthesize(voxel_cloud, query_freq, args)
    if result is None:
        return 0
    synthesized, visible_protos, query_pos = result

    # Apply reverse morphisms to get semantic meaning
    origin = Origin(512, 512, use_gpu=False)
    divergence_result = _apply_divergence(origin, synthesized, query_params, use_dual)

    _print_synthesis_interpretation(divergence_result, visible_protos, synthesized, args.debug)
    _print_weighted_texts(visible_protos, query_pos)

    # Save output if requested
    if args.output:
        np.save(args.output, synthesized)
        print(f"\n💾 Saved synthesized proto-identity to: {args.output}")

    return 0


def _load_voxel_cloud_and_metadata(model_path: str):
    """Load voxel cloud and associated metadata securely."""
    import logging
    from src.security import safe_load_pickle

    logger = logging.getLogger(__name__)

    # Try to load as MemoryHierarchy first (new format) with security checks
    try:
        logger.info(f"Loading model from {model_path} with security checks")
        data = safe_load_pickle(model_path, backward_compatible=True)

        # Check if it's a MemoryHierarchy format
        if 'core_memory' in data:
            # Extract VoxelCloud from MemoryHierarchy
            voxel_cloud = data['core_memory']
            use_dual = False
            meta = None
            logger.info("Successfully loaded MemoryHierarchy format model")
            return voxel_cloud, meta, use_dual
    except FileNotFoundError:
        raise
    except Exception as e:
        logger.debug(f"Not MemoryHierarchy format or load failed: {e}")
        pass

    # Fallback to direct VoxelCloud loading (old format)
    voxel_cloud = VoxelCloud()
    voxel_cloud.load(model_path)

    # Load metadata
    metadata_path = model_path.replace('.pkl', '_meta.pkl')
    use_dual = False
    meta = None
    if Path(metadata_path).exists():
        with open(metadata_path, 'rb') as f:
            meta = pickle.load(f)
            use_dual = meta.get('dual_path', False)

    return voxel_cloud, meta, use_dual


def _update_synthesis_config(voxel_cloud, args):
    """Update synthesis config from CLI args."""
    if hasattr(args, 'resonance_weighting'):
        voxel_cloud.synthesis_config['use_resonance_weighting'] = args.resonance_weighting
    if hasattr(args, 'weight_function'):
        voxel_cloud.synthesis_config['weight_function'] = args.weight_function
    if hasattr(args, 'resonance_boost'):
        voxel_cloud.synthesis_config['resonance_boost'] = args.resonance_boost
    if hasattr(args, 'distance_decay'):
        voxel_cloud.synthesis_config['distance_decay'] = args.distance_decay


def _debug_frequency_analysis(query_freq, query_params):
    """Print debug information for frequency analysis."""
    print("\n" + "="*60)
    print("FREQUENCY ANALYSIS:")
    print("="*60)
    # Get dominant frequency
    fft_magnitude = np.abs(np.fft.fft2(query_freq[:,:,0]))
    freqs = np.fft.fftfreq(fft_magnitude.shape[0])
    dominant_idx = np.unravel_index(np.argmax(fft_magnitude), fft_magnitude.shape)
    dominant_freq = freqs[dominant_idx[0]]

    print(f"  Dominant frequency: {dominant_freq:.3f}")
    print(f"  Harmonic structure: {query_params['iota_params']['harmonic_coeffs'][:8]}")
    print(f"  Global amplitude: {query_params['iota_params']['global_amplitude']:.3f}")

    print("\nMORPHISM PARAMETERS:")
    print(f"  gamma_params: {query_params['gamma_params']}")
    print(f"  iota_params: {query_params['iota_params']}")


def _debug_viewport_query(query_pos, visible_protos):
    """Print debug information for viewport query."""
    print("\n" + "="*60)
    print("VIEWPORT QUERY:")
    print("="*60)
    print(f"  Query position: ({query_pos[0]:.1f}, {query_pos[1]:.1f}, {query_pos[2]:.1f})")
    print(f"  Radius: 100.0")
    print(f"  Found {len(visible_protos)} visible proto-identities")


def _print_visible_protos_debug(visible_protos, query_pos):
    """Print detailed debug information about visible proto-identities."""
    print("\n" + "="*60)
    print("VISIBLE PROTO-IDENTITIES:")
    print("="*60)

    distances = []
    for entry in visible_protos:
        dist = np.linalg.norm(entry.position - query_pos)
        distances.append(dist)

    weights = 1.0 / (np.array(distances) + 1e-6)
    weights = weights / weights.sum()

    for i, (entry, weight) in enumerate(zip(visible_protos, weights)):
        modality = entry.metadata.get('modality', 'unknown')
        index = entry.metadata.get('index', '?')
        dist = np.linalg.norm(entry.position - query_pos)

        proto_mean = entry.proto_identity.mean()
        proto_std = entry.proto_identity.std()
        energy_empty = np.abs(entry.proto_identity[:,:,0]).mean()

        print(f"\n  {i+1}. [Distance: {dist:.1f}] {modality}#{index}")
        print(f"     Proto: mean={proto_mean:.3f}, std={proto_std:.3f}")
        print(f"     Energy_to_empty={energy_empty:.2f}, Weight: {weight:.2f}")


def _print_visible_protos_normal(visible_protos, query_pos):
    """Print normal information about visible proto-identities."""
    print(f"\n📝 Nearby proto-identities:\n")
    for i, entry in enumerate(visible_protos[:5]):
        modality = entry.metadata.get('modality', 'unknown')
        index = entry.metadata.get('index', '?')
        f0 = entry.fundamental_freq
        print(f"{i+1}. {modality}#{index} (f0={f0:.2f}Hz)")
        dist = np.linalg.norm(entry.position - query_pos)
        print(f"   Distance: {dist:.2f}")
        print()


def _debug_synthesis(visible_protos, query_pos, synthesized):
    """Print debug information about synthesis."""
    print("\n" + "="*60)
    print("SYNTHESIS:")
    print("="*60)

    distances = []
    for entry in visible_protos:
        dist = np.linalg.norm(entry.position - query_pos)
        distances.append(dist)

    weights = 1.0 / (np.array(distances) + 1e-6)
    weights = weights / weights.sum()

    max_dist = max(distances)
    mip_levels = [min(int(d / max_dist * 3), 2) for d in distances]

    print(f"  Weighted blend of {len(visible_protos)} proto-identities")
    print(f"  MIP levels used: {mip_levels}")
    print(f"  Noise variance: {0.01:.2f}")

    print("\nSYNTHESIZED PROTO-IDENTITY:")
    print(f"  Shape: {synthesized.shape}")
    print(f"  Mean: {synthesized.mean():.3f}")
    print(f"  Std: {synthesized.std():.3f}")


def _print_synthesis_interpretation(divergence_result, visible_protos, synthesized, debug=False):
    """Print interpretation of synthesis results."""
    if debug:
        print(f"  Energy to empty: {divergence_result.empty_output.mean():.3f}")
        print(f"  Energy to infinity: {divergence_result.infinity_output.mean():.3f}")

        min_diff = float('inf')
        for entry in visible_protos:
            diff = np.linalg.norm(synthesized - entry.proto_identity)
            min_diff = min(min_diff, diff)

        print(f"\nINTERPRETATION:")
        if divergence_result.empty_output.mean() > divergence_result.infinity_output.mean():
            print("  Leans toward: material/concrete aspects")
        else:
            print("  Leans toward: abstract/law-like aspects")
        print(f"  Distance from nearest stored proto: {min_diff:.1f}")
        print(f"  Unique: {'YES' if min_diff > 1.0 else 'NO'}")
    else:
        print(f"\n✨ Synthesis complete:")
        print(f"  Proto-identity shape: {synthesized.shape}")
        print(f"  Energy to empty: {divergence_result.empty_output.mean():.4f}")
        print(f"  Energy to infinity: {divergence_result.infinity_output.mean():.4f}")

        print(f"\n💭 Interpretation:")
        if divergence_result.empty_output.mean() > divergence_result.infinity_output.mean():
            print("  Synthesis leans toward material/concrete aspects")
        else:
            print("  Synthesis leans toward abstract/law-like aspects")


def _apply_divergence(origin, synthesized, query_params, use_dual):
    """Apply divergence morphisms to extract meaning."""
    if use_dual:
        # Act_dual: n → (ι ∪ τ) → 𝟙 → (γ ∪ ε) → (∅ ∪ ∞) → ○
        return origin.Act_dual(
            synthesized,
            query_params['iota_params'],
            query_params['gamma_params'],
            query_params['tau_params'],
            query_params['epsilon_params']
        )
    else:
        # Act: n → ○ (divergence to extract meaning)
        return origin.Act(synthesized)


def _print_weighted_texts(visible_protos, query_pos):
    """Print weighted text output from visible proto-identities."""
    if not visible_protos:
        return

    distances = []
    for entry in visible_protos:
        dist = np.linalg.norm(entry.position - query_pos)
        distances.append(dist)

    print(f"\n💬 SYNTHESIZED RESPONSE:")
    print("-" * 60)
    # Add exponential decay weighting with decay factor
    # Scale distances to reasonable range first (distances are often very small)
    distances_array = np.array(distances)
    if distances_array.max() > 0:
        distances_array = distances_array / distances_array.max()  # Normalize to [0, 1]
    decay_factor = 5.0
    weights_normalized = np.exp(-distances_array * decay_factor)

    # Optional: minimum weight threshold
    min_weight = 0.01
    weights_normalized = np.maximum(weights_normalized, min_weight)

    # Normalize
    weights_normalized /= weights_normalized.sum()

    # Show top 3 weighted proto-identities
    top_indices = np.argsort(weights_normalized)[::-1][:3]
    for idx in top_indices:
        entry = visible_protos[idx]
        weight = weights_normalized[idx]
        modality = entry.metadata.get('modality', 'unknown')
        index = entry.metadata.get('index', '?')
        f0 = entry.fundamental_freq
        print(f"[{weight:.2%}] {modality}#{index} (f0={f0:.2f}Hz)")
    print("-" * 60)


def _print_metadata(meta, use_dual):
    """Print loaded metadata information."""
    if meta:
        print(f"  Source: {meta['source_file']}")
        print(f"  Proto-identities: {meta['num_proto_identities']}")
        if use_dual:
            print(f"  Mode: dual-path (Gen + Res)")


def _query_and_synthesize(voxel_cloud, query_freq, args):
    """Query viewport and synthesize proto-identity."""
    # Generate query proto-identity
    query_proto, query_quaternions = _generate_query_proto(query_freq)

    print("\n🎯 Querying voxel cloud with multi-octave matching...")
    print(f"   Query proto-identity shape: {query_proto.shape}")

    # Query voxel cloud for visible proto-identities
    visible_protos = _query_voxel_cloud(
        voxel_cloud, query_freq, query_proto, query_quaternions, args
    )

    query_pos = voxel_cloud._frequency_to_position(query_freq)

    if args.debug:
        _debug_viewport_query(query_pos, visible_protos)

    print(f"  Visible proto-identities: {len(visible_protos)}")

    if not visible_protos:
        print("  No proto-identities found in viewport")
        return None

    # Show nearby proto-identities
    if args.debug:
        _print_visible_protos_debug(visible_protos, query_pos)
    else:
        _print_visible_protos_normal(visible_protos, query_pos)

    # Synthesize new proto-identity
    return _synthesize_proto(voxel_cloud, visible_protos, query_freq, query_pos, args)


def _generate_query_proto(query_freq):
    """Generate query proto-identity from frequency."""
    from src.origin import Origin
    from src.memory.octave_frequency import (
        extract_fundamental, extract_harmonics,
        frequency_to_gen_params
    )

    origin = Origin(512, 512, use_gpu=False)
    f0 = extract_fundamental(query_freq)
    harmonics = extract_harmonics(query_freq, f0)
    gen_params = frequency_to_gen_params(f0, harmonics)

    # Create query proto-identity using Gen
    query_standing_wave = origin.Gen(gen_params['gamma_params'], gen_params['iota_params'])

    # Extract multi-octave quaternions from query
    query_result = origin.Act(query_standing_wave)
    return query_result.proto_identity, query_result.multi_octave_quaternions


def _query_voxel_cloud(voxel_cloud, query_freq, query_proto, query_quaternions, args):
    """Query voxel cloud using octave or traditional methods."""
    # Check if voxel cloud has octave support
    if hasattr(voxel_cloud, 'octave_hierarchy') and query_quaternions:
        visible_protos = _query_with_octaves(
            voxel_cloud, query_quaternions, args.query
        )
        if visible_protos:
            return visible_protos

    # Fall back to traditional query
    return voxel_cloud.query_viewport(
        query_freq, radius=100.0, max_results=10,
        use_frequency_matching=args.use_frequency,
        query_proto=query_proto
    )


def _query_with_octaves(voxel_cloud, query_quaternions, query_text):
    """Query using multi-octave quaternions."""
    # Adaptive octave selection based on query length
    primary_octave = voxel_cloud.octave_hierarchy.adaptive_octave_selection(query_text)
    print(f"   Primary octave level: {primary_octave} (0=phoneme, 4=sentence)")

    # Multi-octave query with weighted fusion
    octave_weights = {primary_octave: 0.5}  # 50% weight on primary
    # Distribute remaining weight to adjacent octaves
    for i in range(max(0, primary_octave - 1), min(5, primary_octave + 2)):
        if i != primary_octave:
            octave_weights[i] = 0.25

    # Query using multi-octave quaternions
    octave_results = voxel_cloud.query_multi_octave(
        query_quaternions, top_k=10, octave_weights=octave_weights
    )

    # Convert octave results to proto entries for compatibility
    visible_protos = []
    for octave_proto, dist in octave_results:
        # Find corresponding entry in voxel cloud by proto_identity match
        for entry in voxel_cloud.entries:
            # Match by proto_identity similarity (no text stored)
            if np.allclose(entry.proto_identity, octave_proto.proto_identity, rtol=1e-5):
                visible_protos.append(entry)
                break

    return visible_protos


def _synthesize_proto(voxel_cloud, visible_protos, query_freq, query_pos, args):
    """Synthesize new proto-identity from visible elements."""
    print("\n🔮 Synthesizing new proto-identity from visible elements...")
    synthesized = voxel_cloud.synthesize(visible_protos, query_freq)

    if args.debug:
        _debug_synthesis(visible_protos, query_pos, synthesized)
    return synthesized, visible_protos, query_pos
