"""Core command implementations for Genesis CLI (test, discover)."""
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict

from src.origin import Origin
from src.memory.frequency_field import TextFrequencyAnalyzer
from src.memory.voxel_cloud import VoxelCloud
from src.memory.octave_frequency import extract_fundamental, extract_harmonics
from src.memory.memory_hierarchy import MemoryHierarchy
from src.pipeline.unified_encoder import UnifiedEncoder
from src.pipeline.unified_decoder import UnifiedDecoder
from src.cli.commands_helpers import extract_sentences
from src.cli.commands_modality import (
    process_text_input, process_image_input, process_audio_input
)
# Import synthesize command
from src.cli.commands_synthesis import cmd_synthesize


def _build_test_configs():
    """Build collapse and synthesis configs for testing."""
    collapse_config = {
        'harmonic_tolerance': 0.05,
        'cosine_threshold': 0.85,
        'octave_tolerance': 0,
        'enable': True
    }
    synthesis_config = {
        'use_resonance_weighting': True,
        'weight_function': 'linear',
        'resonance_boost': 2.0,
        'distance_decay': 0.5
    }
    return collapse_config, synthesis_config


def cmd_test(args):
    """Test Genesis frequency-based morphism system."""
    print("=" * 60)
    print("Genesis Frequency-Morphism Test")
    print("=" * 60)

    # Test frequency analyzer
    print("\n1. Testing TextFrequencyAnalyzer:")
    analyzer = TextFrequencyAnalyzer(512, 512)
    test_text = "The Tao that can be spoken is not the eternal Tao"
    freq_spectrum, params = analyzer.analyze(test_text)
    print(f"  ✓ Frequency spectrum shape: {freq_spectrum.shape}")
    print(f"  ✓ Dominant frequency: {params['gamma_params']['base_frequency']:.3f}")
    print(f"  ✓ Amplitude: {params['gamma_params']['amplitude']:.3f}")

    # Test Origin with morphisms
    print("\n2. Testing Origin morphisms:")
    origin = Origin(512, 512, use_gpu=False)  # Use CPU for test
    proto_identity = origin.Gen(params['gamma_params'], params['iota_params'])
    print(f"  ✓ Proto-identity shape: {proto_identity.shape}")
    print(f"  ✓ Proto-identity range: [{proto_identity.min():.3f}, {proto_identity.max():.3f}]")

    # Test voxel cloud with configurable parameters
    print("\n3. Testing VoxelCloud with configs:")
    collapse_config, synthesis_config = _build_test_configs()
    voxel_cloud = VoxelCloud(512, 512, 128, collapse_config, synthesis_config)
    import time
    voxel_cloud.add(proto_identity, freq_spectrum, {
        'modality': 'text',
        'source': 'cmd_test',
        'timestamp': int(time.time()),
        'params': params
    })
    print(f"  ✓ Added to voxel cloud")
    print(f"  ✓ Collapse config: {collapse_config}")
    print(f"  ✓ Synthesis config: {synthesis_config}")

    visible = voxel_cloud.query_viewport(freq_spectrum, radius=50.0)
    print(f"  ✓ Visible proto-identities: {len(visible)}")
    synthesized = voxel_cloud.synthesize(visible, freq_spectrum)
    print(f"  ✓ Synthesized shape: {synthesized.shape}")

    print("\n✅ All tests passed")
    return 0


def _create_voxel_cloud_for_discovery(args):
    """Create VoxelCloud with configurations from arguments."""
    collapse_config = {
        'harmonic_tolerance': args.collapse_harmonic_tolerance,
        'cosine_threshold': args.collapse_cosine_threshold,
        'octave_tolerance': args.collapse_octave_tolerance,
        'enable': args.enable_collapse
    }
    synthesis_config = {
        'use_resonance_weighting': True,
        'weight_function': 'linear',
        'resonance_boost': 2.0,
        'distance_decay': 0.5
    }
    print(f"  Collapse config: {collapse_config}")
    return VoxelCloud(512, 512, 128, collapse_config, synthesis_config)


def cmd_discover(args):
    """Discover patterns in text/image/audio via frequency-based morphisms."""
    print("=" * 60)
    print(f"Genesis Pattern Discovery ({args.modality.upper()} → Frequency → Morphisms → Voxel Cloud)")
    print("=" * 60)

    # Check for unified mode flag (default to legacy for backward compatibility)
    use_unified = getattr(args, 'unified', False)

    # Load input based on modality
    if not Path(args.input).exists():
        print(f"❌ File not found: {args.input}")
        return 1

    print(f"\n📖 Loading {args.modality}: {args.input}")

    # Initialize components
    print("\n🔧 Initializing components...")
    origin = Origin(512, 512, use_gpu=False)

    if use_unified:
        # New unified approach with memory hierarchy
        return _cmd_discover_unified(args, origin)
    else:
        # Legacy approach for backward compatibility
        voxel_cloud = _create_voxel_cloud_for_discovery(args)

        # Process based on modality
        if args.modality == 'text':
            freq_analyzer = TextFrequencyAnalyzer(512, 512)
            result = process_text_input(args.input, freq_analyzer, origin, voxel_cloud, args)
        elif args.modality == 'image':
            result = process_image_input(args.input, origin, voxel_cloud, args)
        elif args.modality == 'audio':
            result = process_audio_input(args.input, origin, voxel_cloud, args)
        else:
            print(f"❌ Unknown modality: {args.modality}")
            return 1

        if result != 0:
            return result

        # Cross-modal linking
        if args.link_cross_modal:
            print(f"\n🔗 Performing cross-modal phase-locking...")
            links = voxel_cloud.link_cross_modal_protos(args.phase_coherence_threshold)
            print(f"  Created {links} cross-modal links")

        print(f"\n📊 Discovery Results:")
        print(f"  {voxel_cloud}")

        print(f"\n💾 Saving voxel cloud to: {args.output}")
        voxel_cloud.save(args.output)
        _save_discovery_metadata(args.output, None, args.input, voxel_cloud, args.dual_path)

        print("\n✅ Discovery complete - Voxel cloud populated with proto-identities")
        return 0


def _cmd_discover_unified(args, origin):
    """Unified discover command using UnifiedEncoder and MemoryHierarchy."""
    # Initialize unified components
    collapse_config = {
        'harmonic_tolerance': args.collapse_harmonic_tolerance,
        'cosine_threshold': args.collapse_cosine_threshold,
        'octave_tolerance': args.collapse_octave_tolerance,
        'enable': args.enable_collapse
    }

    # Create memory hierarchy with routing enabled
    memory_hierarchy = MemoryHierarchy(
        width=512, height=512, depth=128,
        collapse_config=collapse_config,
        use_routing=True
    )

    # Initialize carrier
    memory_hierarchy.create_carrier(origin)

    # Create unified encoder
    encoder = UnifiedEncoder(
        memory_hierarchy=memory_hierarchy,
        carrier=memory_hierarchy.proto_unity_carrier
    )

    # Read input text
    with open(args.input, 'r', encoding='utf-8') as f:
        text = f.read()

    # Determine context type
    context_type = getattr(args, 'context_type', 'foundation')  # Default to foundation for discover

    print(f"\n🔄 Processing with UnifiedEncoder (context: {context_type})...")

    # Encode text with unified encoder
    # Determine destination based on context_type
    if context_type == 'foundation':
        destination = 'core'
    elif context_type == 'query':
        destination = 'experiential'
    else:
        destination = 'auto'

    result = encoder.encode(
        text=text,
        destination=destination,
        octaves=[4, 0, -2],  # Characters, words, short phrases
        metadata={'source': args.input, 'context': context_type}
    )

    # Display results
    print(f"\n📊 Unified Discovery Results:")
    print(f"  Total units encoded: {len(result.octave_units)}")
    print(f"  Core memory: {result.core_added} entries")
    print(f"  Experiential memory: {result.experiential_added} entries")

    # Show octave distribution
    octave_counts = {}
    for unit in result.octave_units:
        octave_counts[unit.octave] = octave_counts.get(unit.octave, 0) + 1
    print(f"  Octave distribution: {octave_counts}")

    # Save memory state
    print(f"\n💾 Saving memory state to: {args.output}")
    _save_unified_memory(args.output, memory_hierarchy, result)

    print("\n✅ Unified discovery complete - Memory hierarchy populated")
    return 0


def _save_unified_memory(output_path: str, memory_hierarchy, encode_result):
    """Save unified memory state and metadata."""
    import pickle

    # Save main memory state
    with open(output_path, 'wb') as f:
        pickle.dump({
            'core_memory': memory_hierarchy.core_memory,
            'experiential_memory': memory_hierarchy.experiential_memory,
            'carrier': memory_hierarchy.proto_unity_carrier,
            'encode_result': encode_result
        }, f)

    # Save metadata
    metadata_path = output_path.replace('.pkl', '_meta.pkl')

    # Compute octave distribution
    octave_counts = {}
    for unit in encode_result.octave_units:
        octave_counts[unit.octave] = octave_counts.get(unit.octave, 0) + 1

    with open(metadata_path, 'wb') as f:
        pickle.dump({
            'stats': {
                'total_units': len(encode_result.octave_units),
                'core_added': encode_result.core_added,
                'experiential_added': encode_result.experiential_added
            },
            'octave_distribution': octave_counts,
            'routing_history': memory_hierarchy.memory_router.routing_history
                             if memory_hierarchy.memory_router else []
        }, f)


def _save_discovery_metadata(output_path: str, sentences: list, input_path: str,
                            voxel_cloud, dual_path: bool):
    """Save discovery metadata."""
    metadata_path = output_path.replace('.pkl', '_meta.pkl')
    with open(metadata_path, 'wb') as f:
        pickle.dump({
            'sentences': sentences,
            'source_file': str(input_path),
            'num_proto_identities': len(voxel_cloud),
            'dual_path': dual_path
        }, f)
