"""Training command implementations for Genesis CLI (train, chat, eval)."""
import pickle
import re
import json
from pathlib import Path
from typing import List

from src.memory.voxel_cloud import VoxelCloud
from src.cli.commands_helpers import extract_sentences


def cmd_train(args):
    """Train Genesis on foundation data using MemoryHierarchy architecture."""
    print("=" * 60)
    print("Genesis Foundation Training (Phases 1-6 Architecture)")
    print("=" * 60)

    # Build collapse config from args
    collapse_config = {
        'enable': args.enable_collapse,
        'cosine_threshold': args.collapse_cosine_threshold,
        'harmonic_tolerance': args.collapse_harmonic_tolerance,
        'octave_tolerance': args.collapse_octave_tolerance,
    }

    print(f"\n🔧 Collapse Configuration:")
    print(f"  Enable: {collapse_config['enable']}")
    print(f"  Cosine threshold: {collapse_config['cosine_threshold']}")
    print(f"  Harmonic tolerance: {collapse_config['harmonic_tolerance']}")
    print(f"  Octave tolerance: {collapse_config['octave_tolerance']}")

    # Initialize components and load documents
    hierarchy, encoding, origin = _init_training_components(args.use_gpu, collapse_config)
    text_files = _load_foundation_documents(args.data, args.max_documents)

    if text_files is None:
        return 1  # Error loading documents

    # Process all documents
    total_protos = _process_training_documents(
        text_files, encoding, hierarchy, args.checkpoint_interval, args.output
    )

    # Save and print statistics
    _save_training_results(hierarchy, args.output, len(text_files), total_protos)
    return 0


def _init_training_components(use_gpu, collapse_config):
    """Initialize MemoryHierarchy, Origin, and EncodingPipeline."""
    from src.memory.memory_hierarchy import MemoryHierarchy
    from src.origin import Origin
    from src.pipeline.encoding import EncodingPipeline

    print("\n🔧 Initializing components...")
    hierarchy = MemoryHierarchy(width=512, height=512, depth=128, collapse_config=collapse_config)
    origin = Origin(512, 512, use_gpu=use_gpu)

    print("🌟 Creating proto-unity carrier...")
    carrier = hierarchy.create_carrier(origin)
    print(f"  ✓ Carrier shape: {carrier.shape}")

    encoding = EncodingPipeline(hierarchy.proto_unity_carrier, width=512, height=512)
    return hierarchy, encoding, origin


def _load_foundation_documents(data_path, max_documents):
    """Load foundation text files."""
    data_dir = Path(data_path)
    if not data_dir.exists():
        print(f"❌ Data path not found: {data_path}")
        return None

    text_files = sorted(data_dir.glob("*.txt"))
    if max_documents:
        text_files = text_files[:max_documents]

    print(f"\n📚 Found {len(text_files)} documents to process")
    return text_files


def _process_training_documents(text_files, encoding, hierarchy,
                                checkpoint_interval, output):
    """Process documents using signal-derived architecture.

    Each document encoded once as unified signal. Frequency spectrum
    naturally contains all hierarchical levels. VoxelCloud's
    frequency_to_position() maps spectrum to voxel space across octaves.
    """
    stats = {'processed': 0, 'protos_created': 0, 'collapsed': 0}

    for idx, doc_path in enumerate(text_files, 1):
        print(f"\n[{idx}/{len(text_files)}] Processing: {doc_path.name}")

        with open(doc_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        # Detect modality and encode
        if doc_path.suffix == '.txt':
            proto, metadata = encoding.encode_text(content)
        elif doc_path.suffix in ['.png', '.jpg', '.jpeg']:
            proto, metadata = encoding.encode_image(doc_path)
        elif doc_path.suffix in ['.wav', '.mp3']:
            proto, metadata = encoding.encode_audio(doc_path)
        elif doc_path.suffix == '.mp4':
            video_protos = encoding.encode_video(doc_path)
            for proto, metadata in video_protos:
                _store_proto(proto, metadata, hierarchy, stats, doc_path)
            stats['processed'] += 1
            continue
        elif doc_path.suffix == '.pdf':
            pdf_protos = encoding.encode_pdf(doc_path)
            for proto, metadata in pdf_protos:
                _store_proto(proto, metadata, hierarchy, stats, doc_path)
            stats['processed'] += 1
            continue
        else:
            print(f"  ⚠️  Unsupported file type: {doc_path.suffix}")
            continue

        # Store proto (VoxelCloud handles octave distribution)
        _store_proto(proto, metadata, hierarchy, stats, doc_path)

        stats['processed'] += 1

        # Checkpoint if requested
        if checkpoint_interval and idx % checkpoint_interval == 0:
            checkpoint_path = f"{output}.checkpoint_{idx}"
            print(f"  💾 Saving checkpoint: {checkpoint_path}")
            _save_hierarchy(hierarchy, checkpoint_path)

    return stats['protos_created']


def _store_proto(proto, metadata, hierarchy, stats, doc_path):
    """Store proto with deterministic metadata."""
    import time

    # Extract frequency spectrum (VoxelCloud needs this)
    spectrum = proto[:, :, :2]  # First 2 channels: magnitude, phase

    # Deterministic metadata only
    clean_metadata = {
        'source_file': doc_path.name,
        'modality': metadata.get('modality', 'text'),
        'timestamp': time.time(),
    }

    # Store in core memory
    # VoxelCloud.frequency_to_position() maps spectrum to voxel space
    # Gravitational collapse handles deduplication at each octave
    hierarchy.store_core(proto, spectrum, clean_metadata)

    stats['protos_created'] += 1
    print(f"  ✓ Proto created (total: {stats['protos_created']})")


def _save_training_results(hierarchy, output_path, num_docs, total_protos):
    """Save model and print statistics."""
    print(f"\n💾 Saving trained model to: {output_path}")
    _save_hierarchy(hierarchy, output_path)

    print(f"\n📊 Training Complete:")
    print(f"  Documents processed: {num_docs}")
    print(f"  Proto-identities in core: {len(hierarchy.core_memory.entries)}")
    print(f"  Total segments: {total_protos}")
    print(f"  Model saved: {output_path}")


def cmd_chat(args):
    """Interactive conversation with trained Genesis model."""
    print("=" * 60)
    print("Genesis Interactive Chat")
    print("=" * 60)

    # Initialize chat session
    hierarchy, handler = _init_chat_session(args.model)
    if hierarchy is None:
        return 1  # Failed to load

    print("\n✅ Ready! Commands: /stats, /consolidate, /reset, /save, /quit")
    print("=" * 60)

    # Run interactive loop
    _run_chat_loop(args, hierarchy, handler)
    return 0


def _init_chat_session(model_path):
    """Load model and initialize chat components."""
    from src.origin import Origin
    from src.pipeline.conversation import ConversationPipeline
    from src.pipeline.realtime_io import RealTimeHandler

    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        print(f"\n💡 Train a model first:")
        print(f"   genesis.py train --data /usr/lib/alembic/data/datasets/curated/foundation/")
        return None, None

    print(f"\n📖 Loading model: {model_path}")
    hierarchy = _load_hierarchy(model_path)
    print(f"  ✓ Core memory: {len(hierarchy.core_memory.entries)} proto-identities")

    print("\n🔧 Initializing conversation pipeline...")
    origin = Origin(512, 512, use_gpu=False)

    # Recreate carrier if not present
    if hierarchy.proto_unity_carrier is None:
        print("  Creating new proto-unity carrier...")
        hierarchy.create_carrier(origin)

    pipeline = ConversationPipeline(hierarchy, origin)
    handler = RealTimeHandler(pipeline)
    handler.start_session()

    return hierarchy, handler


def _run_chat_loop(args, hierarchy, handler):
    """Main interactive chat loop."""
    while True:
        try:
            user_input = input("\nYou: ").strip()
            if not user_input:
                continue

            if user_input.startswith("/"):
                if _handle_chat_command(user_input, args, hierarchy, handler):
                    break  # /quit command
            else:
                _process_chat_input(user_input, args, handler)

        except KeyboardInterrupt:
            print("\n\n👋 Interrupted. Use /quit to save and exit.")
        except Exception as e:
            print(f"\n❌ Error: {e}")


def _handle_chat_command(command, args, hierarchy, handler):
    """Handle chat commands. Returns True if should exit."""
    if command == "/quit":
        print("\n👋 Saving session and exiting...")
        if args.save_on_exit:
            handler.end_session(consolidate=True)
            _save_hierarchy(hierarchy, args.model)
        return True  # Signal to exit

    elif command == "/stats":
        stats = handler.get_session_state()
        print(f"\n📊 Session Statistics:")
        print(f"  Coherence: {stats.get('coherence', 0):.3f}")
        print(f"  State: {stats.get('state', 'N/A')}")
        print(f"  Experiential memories: {stats.get('experiential_count', 0)}")

    elif command == "/consolidate":
        print("\n💾 Consolidating experiential → core memory...")
        consolidated = hierarchy.consolidate()
        print(f"  ✓ Consolidated {consolidated} patterns")

    elif command == "/reset":
        print("\n🔄 Resetting experiential memory...")
        hierarchy.reset_experiential()
        print("  ✓ Reset complete")

    elif command == "/save":
        print(f"\n💾 Saving model...")
        _save_hierarchy(hierarchy, args.model)
        print(f"  ✓ Saved to {args.model}")

    else:
        print(f"Unknown command: {command}")

    return False  # Don't exit


def _process_chat_input(user_input, args, handler):
    """Process user input and display response."""
    response_data = handler.handle_input(user_input)

    if args.stream:
        print("\nGenesis: ", end="", flush=True)
        for chunk in handler.stream_response(response_data):
            print(chunk, end="", flush=True)
        print()  # Newline after streaming
    else:
        print(f"\nGenesis: {response_data['response']}")

    if args.show_stats:
        print(f"  [Coherence: {response_data['coherence']:.3f}, State: {response_data['state'].name}]")


def cmd_eval(args):
    """Run A/B evaluation tests on trained model."""
    print("=" * 60)
    print("Genesis A/B Evaluation")
    print("=" * 60)

    # Initialize evaluation components
    pipeline, test_cases = _init_evaluation(args.model, args.test_cases)
    if pipeline is None:
        return 1  # Failed to initialize

    # Run test cases
    results, passed, failed = _run_test_cases(test_cases, pipeline)

    # Save results
    _save_evaluation_results(results, passed, failed, args.output, len(test_cases))

    return 0 if failed == 0 else 1


def _init_evaluation(model_path, test_cases_path):
    """Load model and test cases."""
    # Load test cases first
    test_cases = _load_test_cases(test_cases_path)
    if test_cases is None:
        return None, None

    # Load model and create pipeline
    pipeline = _create_eval_pipeline(model_path, test_cases)
    if pipeline is None:
        return None, None

    return pipeline, test_cases


def _load_test_cases(test_cases_path):
    """Load test cases from JSON file."""
    if not Path(test_cases_path).exists():
        print(f"❌ Test cases not found: {test_cases_path}")
        print(f"\n💡 Create test_cases.json with format:")
        print("""
{
  "test_cases": [
    {
      "id": "001",
      "input": "What is the Tao?",
      "expected_concepts": ["unity", "way"],
      "min_coherence": 0.7,
      "label": "works"
    }
  ]
}
        """)
        return None

    with open(test_cases_path) as f:
        data = json.load(f)
        return data['test_cases']


def _create_eval_pipeline(model_path, test_cases):
    """Create evaluation pipeline from model."""
    from src.origin import Origin
    from src.pipeline.conversation import ConversationPipeline

    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        return None

    print(f"\n📖 Loading model: {model_path}")
    hierarchy = _load_hierarchy(model_path)

    print(f"📝 Running {len(test_cases)} test cases...")

    origin = Origin(512, 512, use_gpu=False)

    # Recreate carrier if needed
    if hierarchy.proto_unity_carrier is None:
        hierarchy.create_carrier(origin)

    pipeline = ConversationPipeline(hierarchy, origin)
    pipeline.initialize_session()

    return pipeline


def _run_test_cases(test_cases, pipeline):
    """Execute all test cases and collect results."""
    results = []
    passed = 0
    failed = 0

    for case in test_cases:
        print(f"\n[{case['id']}] Testing: {case['input'][:50]}...")

        response = pipeline.process_input(case['input'], input_type='text')

        # Evaluate test case
        test_passed, reasons = _evaluate_test_case(case, response)

        if test_passed:
            print(f"  ✅ PASS")
            passed += 1
        else:
            print(f"  ❌ FAIL: {', '.join(reasons)}")
            failed += 1

        results.append({
            'test_id': case['id'],
            'input': case['input'],
            'response': response['response'],
            'coherence': float(response['coherence']),
            'state': response['state'].name,
            'passed': test_passed,
            'reasons': reasons if not test_passed else [],
            'label': case.get('label', 'unknown')
        })

    return results, passed, failed


def _evaluate_test_case(case, response):
    """Evaluate a single test case."""
    test_passed = True
    reasons = []

    if 'min_coherence' in case:
        if response['coherence'] < case['min_coherence']:
            test_passed = False
            reasons.append(f"Low coherence: {response['coherence']:.3f} < {case['min_coherence']}")

    if 'expected_state' in case:
        if response['state'].name != case['expected_state']:
            test_passed = False
            reasons.append(f"Wrong state: {response['state'].name} != {case['expected_state']}")

    return test_passed, reasons


def _save_evaluation_results(results, passed, failed, output_path, total_cases):
    """Save results JSON and print summary."""
    # Create output directory if needed
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save results
    with open(output_path, 'w') as f:
        json.dump({
            'results': results,
            'summary': {'passed': passed, 'failed': failed, 'total': total_cases}
        }, f, indent=2)

    print(f"\n📊 Evaluation Complete:")
    print(f"  Passed: {passed}/{total_cases}")
    print(f"  Failed: {failed}/{total_cases}")
    print(f"  Results saved: {output_path}")


def _save_hierarchy(hierarchy, path: str):
    """Save MemoryHierarchy to disk."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump({
            'width': hierarchy.width,
            'height': hierarchy.height,
            'depth': hierarchy.depth,
            'proto_unity_carrier': hierarchy.proto_unity_carrier,
            'core_memory': hierarchy.core_memory,
            'experiential_memory': hierarchy.experiential_memory
        }, f)


def _load_hierarchy(path: str):
    """Load MemoryHierarchy from disk."""
    from src.memory.memory_hierarchy import MemoryHierarchy

    with open(path, 'rb') as f:
        data = pickle.load(f)

    # Handle both direct hierarchy format and wrapped format
    if 'hierarchy' in data and isinstance(data['hierarchy'], MemoryHierarchy):
        # Format with wrapped hierarchy object
        return data['hierarchy']
    else:
        # Format with individual fields
        hierarchy = MemoryHierarchy(data['width'], data['height'], data['depth'])
        hierarchy.proto_unity_carrier = data.get('proto_unity_carrier')
        hierarchy.core_memory = data['core_memory']
        hierarchy.experiential_memory = data.get('experiential_memory',
                                                VoxelCloud(512, 512, 128))
        return hierarchy


