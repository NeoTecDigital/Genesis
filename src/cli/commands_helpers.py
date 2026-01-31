"""Helper functions for Genesis CLI commands (modality processing, display, etc)."""
import numpy as np
import pickle
from pathlib import Path
from typing import List, Dict, Optional
from scipy import signal

from src.origin import Origin
from src.memory.octave_frequency import (
    frequency_to_gen_params, frequency_to_res_params,
    extract_fundamental, extract_harmonics,
    extract_fundamental_from_image, extract_harmonics_from_image,
    extract_fundamental_from_audio, extract_harmonics_from_audio
)


# Modality-specific processing functions
def process_text_input(input_path: str, freq_analyzer, origin, voxel_cloud, args) -> int:
    """Process text input file."""
    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()

    sentences = extract_sentences(content)
    print(f"  Found {len(sentences)} text segments")

    if len(sentences) == 0:
        print("❌ No valid sentences found")
        return 1

    # Apply max_segments limit if specified (for testing/QA)
    if hasattr(args, 'max_segments') and args.max_segments is not None:
        if len(sentences) > args.max_segments:
            print(f"  ⚠️  Limiting to first {args.max_segments} segments (--max-segments)")
            sentences = sentences[:args.max_segments]

    print(f"\n🔄 Converting text → frequency → morphisms → voxel cloud...")
    for i, sentence in enumerate(sentences):
        process_single_text_segment(i, sentence, freq_analyzer, origin, voxel_cloud, args)
        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{len(sentences)} segments")

    return 0


def process_single_text_segment(i: int, sentence: str, freq_analyzer, origin, voxel_cloud, args):
    """Process a single text segment into proto-identity and store in voxel cloud."""
    freq_spectrum, params_dict = freq_analyzer.analyze(sentence)

    # Generate proto-identity and quaternions
    proto_identity, quaternions = generate_text_proto_identity(
        origin, params_dict, args.dual_path
    )

    # Store in voxel cloud
    store_text_in_voxel_cloud(
        voxel_cloud, proto_identity, freq_spectrum, quaternions,
        sentence, i, params_dict, args.dual_path
    )


def generate_text_proto_identity(origin, params_dict, dual_path):
    """Generate proto-identity from text parameters."""
    if dual_path:
        # Dual path: Gen ∪ Res convergence
        n_gen = origin.Gen(params_dict['gamma_params'], params_dict['iota_params'])
        # Create simpler epsilon/tau params for Res path
        epsilon_simple = {
            'extraction_rate': 0.3,
            'focus_sigma': 2.222,
            'base_frequency': params_dict['gamma_params']['base_frequency']
        }
        tau_simple = {'decay_rate': 0.9}
        n_res = origin.Res(epsilon_simple, tau_simple)
        standing_wave = n_gen + n_res  # Superposition (not averaging!)

        # Extract multi-octave quaternions via Act
        result = origin.Act(standing_wave)
        return result.proto_identity, result.multi_octave_quaternions
    else:
        proto_identity = origin.Gen(
            params_dict['gamma_params'],
            params_dict['iota_params']
        )
        # For single path, still extract quaternions
        result = origin.Act(proto_identity)
        return result.proto_identity, result.multi_octave_quaternions


def store_text_in_voxel_cloud(voxel_cloud, proto_identity, freq_spectrum,
                               quaternions, sentence, index, params_dict, dual_path):
    """Store processed text in voxel cloud with octave and legacy support."""
    # Store in voxel cloud with octave support
    f0 = extract_fundamental(freq_spectrum)
    voxel_cloud.add_with_octaves(
        proto_identity=proto_identity,
        frequency=f0,
        modality='text',
        quaternions=quaternions,
        resonance_strength=1.0
    )

    # Also store with traditional metadata for backward compatibility
    metadata = {
        'index': index,
        'modality': 'text',
        'gamma_params': params_dict['gamma_params'],
        'iota_params': params_dict['iota_params'],
        'tau_params': params_dict['tau_params'],
        'epsilon_params': params_dict['epsilon_params'],
        'dual_path': dual_path,
        'has_octaves': True
    }
    voxel_cloud.add(proto_identity, freq_spectrum, metadata)


def process_image_input(input_path: str, origin, voxel_cloud, args) -> int:
    """Process image input file."""
    # Load and validate image
    img_array = load_image_file(input_path)
    if img_array is None:
        return 1

    print(f"  Image shape: {img_array.shape}")

    # Extract frequency information
    f0 = extract_fundamental_from_image(img_array)
    harmonics = extract_harmonics_from_image(img_array, f0)

    print(f"  Fundamental frequency: {f0:.3f} Hz")
    print(f"  Harmonics: {harmonics[:5].round(3)}")

    # Process image into proto-identity and store
    process_and_store_image(
        img_array, f0, harmonics, origin, voxel_cloud,
        input_path, args.dual_path
    )

    print("  Added image proto-identity to voxel cloud")
    return 0


def load_image_file(input_path: str):
    """Load image file and convert to numpy array."""
    try:
        from PIL import Image
        img = Image.open(input_path)
        return np.array(img)
    except ImportError:
        print("❌ PIL/Pillow not installed. Install with: pip install pillow")
        return None
    except Exception as e:
        print(f"❌ Error loading image: {e}")
        return None


def process_and_store_image(img_array, f0, harmonics, origin, voxel_cloud,
                             input_path, dual_path):
    """Process image array into proto-identity and store in voxel cloud."""
    # Generate parameters from frequency
    gen_params = frequency_to_gen_params(f0, harmonics)
    res_params = frequency_to_res_params(f0, harmonics)

    # Create proto-identity
    if dual_path:
        n_gen = origin.Gen(gen_params['gamma_params'], gen_params['iota_params'])
        n_res = origin.Res(res_params['epsilon_params'], res_params['tau_params'])
        proto_identity = (n_gen + n_res) / 2.0
    else:
        proto_identity = origin.Gen(gen_params['gamma_params'], gen_params['iota_params'])

    # Create frequency spectrum placeholder
    freq_spectrum = np.zeros((512, 512, 4), dtype=np.float32)
    freq_spectrum[:img_array.shape[0], :img_array.shape[1], :min(4, img_array.shape[2] if img_array.ndim > 2 else 1)] = \
        img_array[:512, :512] / 255.0 if img_array.ndim > 2 else img_array[:512, :512, np.newaxis] / 255.0

    # Store in voxel cloud
    metadata = {
        'filename': Path(input_path).name,
        'modality': 'image',
        'shape': img_array.shape,
        'fundamental_freq': f0,
        'gamma_params': gen_params['gamma_params'],
        'iota_params': gen_params['iota_params'],
        'tau_params': res_params['tau_params'],
        'epsilon_params': res_params['epsilon_params'],
        'dual_path': dual_path
    }
    voxel_cloud.add(proto_identity, freq_spectrum, metadata)


def process_audio_input(input_path: str, origin, voxel_cloud, args) -> int:
    """Process audio input file."""
    # Load and preprocess audio
    audio_data = load_audio_file(input_path)
    if audio_data is None:
        return 1

    audio, sample_rate = audio_data
    print(f"  Audio shape: {audio.shape}, Sample rate: {sample_rate} Hz")

    # Extract frequency information
    f0 = extract_fundamental_from_audio(audio, sample_rate)
    harmonics = extract_harmonics_from_audio(audio, f0, sample_rate)

    print(f"  Fundamental frequency: {f0:.3f} Hz")
    print(f"  Harmonics: {harmonics[:5].round(3)}")

    # Process audio into proto-identity and store
    process_and_store_audio(
        audio, sample_rate, f0, harmonics, origin,
        voxel_cloud, input_path, args.dual_path
    )

    print("  Added audio proto-identity to voxel cloud")
    return 0


def load_audio_file(input_path: str):
    """Load audio file and convert to mono if needed."""
    try:
        import soundfile as sf
        audio, sample_rate = sf.read(input_path)

        # Convert to mono if stereo
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        return (audio, sample_rate)
    except ImportError:
        print("❌ soundfile not installed. Install with: pip install soundfile")
        return None
    except Exception as e:
        print(f"❌ Error loading audio: {e}")
        return None


def process_and_store_audio(audio, sample_rate, f0, harmonics, origin,
                             voxel_cloud, input_path, dual_path):
    """Process audio into proto-identity and store in voxel cloud."""
    # Generate parameters from frequency
    gen_params = frequency_to_gen_params(f0, harmonics)
    res_params = frequency_to_res_params(f0, harmonics)

    # Create proto-identity
    if dual_path:
        n_gen = origin.Gen(gen_params['gamma_params'], gen_params['iota_params'])
        n_res = origin.Res(res_params['epsilon_params'], res_params['tau_params'])
        proto_identity = (n_gen + n_res) / 2.0
    else:
        proto_identity = origin.Gen(gen_params['gamma_params'], gen_params['iota_params'])

    # Create frequency spectrum from audio spectrogram
    _, _, Sxx = signal.spectrogram(audio, sample_rate, nperseg=512, noverlap=256)
    freq_spectrum = np.zeros((512, 512, 4), dtype=np.float32)
    freq_spectrum[:min(512, Sxx.shape[0]), :min(512, Sxx.shape[1]), 0] = \
        Sxx[:512, :512] / (Sxx.max() + 1e-8)

    # Store in voxel cloud
    metadata = {
        'filename': Path(input_path).name,
        'modality': 'audio',
        'duration': len(audio) / sample_rate,
        'sample_rate': sample_rate,
        'fundamental_freq': f0,
        'gamma_params': gen_params['gamma_params'],
        'iota_params': gen_params['iota_params'],
        'tau_params': res_params['tau_params'],
        'epsilon_params': res_params['epsilon_params'],
        'dual_path': dual_path
    }
    voxel_cloud.add(proto_identity, freq_spectrum, metadata)


# Utility functions
def extract_sentences(content: str) -> List[str]:
    """Extract meaningful sentences from text content."""
    sentences = []
    for line in content.split('\n'):
        line = line.strip()
        if not line:
            continue
        # Split on periods but keep meaningful chunks
        parts = line.split('.')
        for part in parts:
            part = part.strip()
            if len(part) > 10:  # Skip very short fragments
                sentences.append(part)
    return sentences


def save_discovery_metadata(output_path: str, sentences: list, input_path: str,
                            voxel_cloud, dual_path: bool):
    """Save discovery metadata securely."""
    from src.security import safe_save_pickle

    metadata_path = output_path.replace('.pkl', '_meta.pkl')
    safe_save_pickle({
        'sentences': sentences,
        'source_file': str(input_path),
        'num_proto_identities': len(voxel_cloud),
        'dual_path': dual_path
    }, metadata_path)


def load_voxel_cloud_and_metadata(model_path: str):
    """Load voxel cloud and associated metadata securely."""
    from src.memory.voxel_cloud import VoxelCloud
    from src.security import safe_load_pickle
    import logging

    logger = logging.getLogger(__name__)

    voxel_cloud = VoxelCloud()
    voxel_cloud.load(model_path)

    # Load metadata with security checks
    metadata_path = model_path.replace('.pkl', '_meta.pkl')
    use_dual = False
    meta = None
    if Path(metadata_path).exists():
        try:
            logger.info(f"Loading metadata from {metadata_path} with security checks")
            meta = safe_load_pickle(metadata_path, backward_compatible=True)
            use_dual = meta.get('dual_path', False)
            logger.info("Successfully loaded metadata")
        except Exception as e:
            logger.warning(f"Failed to load metadata: {e}")
            # Continue without metadata if loading fails

    return voxel_cloud, meta, use_dual


def update_synthesis_config(voxel_cloud, args):
    """Update synthesis config from CLI args."""
    if hasattr(args, 'resonance_weighting'):
        voxel_cloud.synthesis_config['use_resonance_weighting'] = args.resonance_weighting
    if hasattr(args, 'weight_function'):
        voxel_cloud.synthesis_config['weight_function'] = args.weight_function
    if hasattr(args, 'resonance_boost'):
        voxel_cloud.synthesis_config['resonance_boost'] = args.resonance_boost
    if hasattr(args, 'distance_decay'):
        voxel_cloud.synthesis_config['distance_decay'] = args.distance_decay
