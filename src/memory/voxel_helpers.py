"""Helper functions for VoxelCloud operations."""
import numpy as np
from typing import Dict, List, Optional


def compute_frequency_position(freq_spectrum: np.ndarray, width: int, height: int, depth: int) -> np.ndarray:
    """Map frequency spectrum to 3D position in voxel cloud."""
    # Compute spectral characteristics
    magnitude = np.sqrt(freq_spectrum[..., 0]**2 + freq_spectrum[..., 1]**2)

    # X: Dominant frequency
    center_y, center_x = freq_spectrum.shape[0] // 2, freq_spectrum.shape[1] // 2
    y_coords, x_coords = np.ogrid[:freq_spectrum.shape[0], :freq_spectrum.shape[1]]
    freq_y = (y_coords - center_y) / freq_spectrum.shape[0]
    freq_x = (x_coords - center_x) / freq_spectrum.shape[1]

    total_mag = magnitude.sum()
    if total_mag > 0:
        weighted_y = (magnitude * freq_y).sum() / total_mag
        weighted_x = (magnitude * freq_x).sum() / total_mag
        dominant_freq = np.sqrt(weighted_y**2 + weighted_x**2)
    else:
        dominant_freq = 0.5

    # Y: Phase variance
    phase = np.arctan2(freq_spectrum[..., 1], freq_spectrum[..., 0])
    phase_var = phase.var()

    # Z: Energy concentration
    energy = (magnitude**2).sum() / (magnitude.size + 1e-8)

    # Map to voxel cloud coordinates
    x = dominant_freq * width
    y = np.clip(phase_var / np.pi, 0, 1) * height
    z = np.clip(energy * 10, 0, 1) * depth

    return np.array([x, y, z], dtype=np.float32)


def box_filter_downsample(img: np.ndarray) -> Optional[np.ndarray]:
    """Apply 2x2 box filter downsampling."""
    h, w = img.shape[0] // 2, img.shape[1] // 2
    if h < 4 or w < 4:
        return None

    # Box filter downsample
    downsampled = np.zeros((h, w, 4), dtype=np.float32)
    for i in range(2):
        for j in range(2):
            downsampled += img[i::2, j::2, :]
    downsampled /= 4.0
    return downsampled


def generate_mip_levels(proto_identity: np.ndarray, levels: int = 5) -> List[np.ndarray]:
    """Generate MIP pyramid for multi-scale representation."""
    mips = [proto_identity]
    current = proto_identity

    for level in range(1, levels):
        downsampled = box_filter_downsample(current)
        if downsampled is None:
            break
        mips.append(downsampled)
        current = downsampled

    return mips


def compute_cosine_similarity(proto1: np.ndarray, proto2: np.ndarray) -> float:
    """Compute cosine similarity between two proto-identities."""
    proto1_flat = proto1.flatten()
    proto2_flat = proto2.flatten()

    dot_product = np.dot(proto1_flat, proto2_flat)
    norm1 = np.linalg.norm(proto1_flat)
    norm2 = np.linalg.norm(proto2_flat)

    if norm1 > 0 and norm2 > 0:
        return dot_product / (norm1 * norm2)
    # If either is zero, check if both are zero (identical)
    return 1.0 if norm1 == 0 and norm2 == 0 else 0.0


def resize_proto(proto: np.ndarray, target_height: int, target_width: int) -> np.ndarray:
    """Resize proto-identity to target dimensions using bilinear sampling."""
    if proto.shape[:2] == (target_height, target_width):
        return proto

    h_ratio = target_height / proto.shape[0]
    w_ratio = target_width / proto.shape[1]
    resized = np.zeros((target_height, target_width, 4), dtype=np.float32)

    for i in range(target_height):
        for j in range(target_width):
            src_i = min(i / h_ratio, proto.shape[0] - 1)
            src_j = min(j / w_ratio, proto.shape[1] - 1)
            resized[i, j] = proto[int(src_i), int(src_j)]
    return resized


def _apply_weight_function(resonances_norm: np.ndarray, function_name: str) -> np.ndarray:
    """Apply specified weight function to normalized resonances."""
    if function_name == 'sqrt':
        return np.sqrt(resonances_norm)
    elif function_name == 'log':
        return np.log1p(resonances_norm * 10) / np.log(11)
    else:  # 'linear'
        return resonances_norm


def compute_resonance_weights(visible_protos: List, query_pos: np.ndarray,
                             config: Dict) -> np.ndarray:
    """
    Compute weights based on resonance_strength and distance.
    Uses configuration to determine how to weight resonance vs distance.
    """
    # Extract resonance strengths from metadata
    resonances = np.array([
        p.metadata.get('resonance_strength', p.resonance_strength)
        for p in visible_protos
    ], dtype=np.float32)

    # Normalize resonances to [0, 1]
    if resonances.max() > resonances.min():
        resonances_norm = (resonances - resonances.min()) / (resonances.max() - resonances.min())
    else:
        resonances_norm = np.ones_like(resonances, dtype=np.float32)

    # Calculate distances
    distances = np.array([
        np.linalg.norm(p.position - query_pos) for p in visible_protos
    ])

    # Distance-based weights (inverse distance)
    dist_weights = 1.0 / (distances + 1e-6)
    dist_weights /= dist_weights.sum()

    # Apply weight function to resonance scores
    res_weights = _apply_weight_function(resonances_norm, config['weight_function'])

    # Normalize resonance weights
    if res_weights.sum() > 0:
        res_weights /= res_weights.sum()
    else:
        res_weights = np.ones_like(res_weights) / len(res_weights)

    # Combine weights based on config
    boost = config['resonance_boost']
    decay = config['distance_decay']

    if boost + decay > 0:
        combined = (boost * res_weights + decay * dist_weights) / (boost + decay)
    else:
        combined = np.ones(len(visible_protos)) / len(visible_protos)

    combined /= (combined.sum() + 1e-8)
    return combined


def compute_distance_weights(visible_protos: List, query_freq: np.ndarray,
                            query_pos: np.ndarray) -> np.ndarray:
    """Original distance and frequency correlation weighting."""
    weights = []
    for entry in visible_protos:
        # Distance weight
        dist = np.linalg.norm(entry.position - query_pos)
        dist_weight = np.exp(-dist / 20.0)

        # Frequency correlation weight
        freq_corr = np.corrcoef(
            query_freq.flatten(),
            entry.frequency.flatten()
        )[0, 1]
        freq_weight = max(0, freq_corr)

        # Combined weight
        weight = dist_weight * (0.5 + 0.5 * freq_weight)
        weights.append(weight)

    weights = np.array(weights)
    return weights / (weights.sum() + 1e-8)


def check_frequency_match(entry, target_freq: float, query_harmonics: np.ndarray,
                         harmonic_tolerance: float) -> Optional[float]:
    """Check if an entry matches frequency criteria."""
    # Check if fundamental frequency matches (within 10%)
    freq_ratio = entry.fundamental_freq / (target_freq + 1e-8)
    if not (0.9 < freq_ratio < 1.1):
        return None

    # Check harmonic signature similarity
    if entry.harmonic_signature is None:
        return None

    harmonic_diff = np.linalg.norm(entry.harmonic_signature - query_harmonics)
    if harmonic_diff >= harmonic_tolerance:
        return None

    return harmonic_diff