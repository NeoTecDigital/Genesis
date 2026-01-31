"""
Encoding Pipeline - Encode multimodal input to proto-identities.

Converts text, images, and audio to proto-identities via:
1. Input → Frequency spectrum
2. Frequency → Signal
3. Signal → Modulate carrier
4. Output proto-identity
"""

import numpy as np
import time
from typing import Tuple, Dict, Optional, List
from pathlib import Path

from src.memory.frequency_field import (
    TextFrequencyAnalyzer,
    ImageFrequencyMapper,
    AudioFrequencyMapper
)
from src.memory.fm_modulation_base import FMModulationBase


class EncodingPipeline:
    """Encode multimodal input to proto-identities."""

    def __init__(self, carrier: np.ndarray, width: int = 512, height: int = 512,
                 modulation_depth: float = 0.5):
        """Initialize encoding pipeline.

        Args:
            carrier: Proto-unity carrier (H, W, 4)
            width: Proto-identity width
            height: Proto-identity height
            modulation_depth: FM modulation depth (0.0 to 2.0)
        """
        self.carrier = carrier
        self.width = width
        self.height = height
        self.modulation_depth = modulation_depth

        # Initialize frequency mappers
        self.text_analyzer = TextFrequencyAnalyzer(width, height)
        self.image_mapper = ImageFrequencyMapper(width, height)
        self.audio_mapper = AudioFrequencyMapper(width, height)

        # Initialize modulator
        self.modulator = FMModulationBase()

    def encode_text(
        self,
        text: str,
        modality: str = 'text'
    ) -> Tuple[np.ndarray, Dict]:
        """Text → Proto-Identity via carrier filter architecture.

        The carrier acts as a FIXED filter that creates interference patterns
        when applied to input signals via FM modulation.

        Flow:
        1. Text → Frequency spectrum
        2. Frequency → Signal (n)
        3. Apply carrier filter via FM modulation
        4. Output proto-identity (interference pattern)

        Args:
            text: Input text
            modality: Modality label for metadata

        Returns:
            Tuple of (proto_identity, metadata)
        """
        # 1. Text → Frequency spectrum (dual-path: resized + native)
        freq_spectrum, native_stft = self.text_analyzer.text_to_frequency(text)

        # 2. Frequency → Signal (this is 'n' - the input)
        signal_n = self._frequency_to_signal(freq_spectrum)

        # 2.5. Add phase diversity based on content hash
        # This prevents identical texts from producing identical protos
        text_hash = hash(text) % 1000 / 1000.0 * 2 * np.pi  # Map to [0, 2π]
        carrier_shifted = self.carrier.copy()
        phase_shift = np.exp(1j * text_hash)
        carrier_shifted[:, :, 0] = np.real(
            (carrier_shifted[:, :, 0] + 1j * carrier_shifted[:, :, 1]) * phase_shift
        )
        carrier_shifted[:, :, 1] = np.imag(
            (carrier_shifted[:, :, 0] + 1j * carrier_shifted[:, :, 1]) * phase_shift
        )

        # 3. Apply carrier filter via FM modulation
        # Carrier acts as FIXED filter creating interference patterns
        proto_identity = self.modulator.modulate(carrier_shifted, signal_n,
                                                 modulation_depth=self.modulation_depth)

        # 4. Build metadata (deterministic only - no semantic content)
        metadata = {
            'modality': modality,
            'source': 'text_input',
            'timestamp': int(time.time()),
            'fundamental_freq': self._compute_fundamental_freq(freq_spectrum),
            'carrier_filtered': True,  # Flag for carrier filter architecture
            'original_length': len(text),
            # Store native STFT for ISTFT reconstruction
            # NOTE: This makes the system RETRIEVAL-based, not purely generative
            # The 512×512 proto stores patterns, but reconstruction needs native STFT
            'native_stft': native_stft.tobytes(),
            'native_stft_shape': native_stft.shape
        }

        return proto_identity, metadata

    def encode_image(self, image_path: str) -> Tuple[np.ndarray, Dict]:
        """Image → Signal → Modulate → Proto.

        Args:
            image_path: Path to image file

        Returns:
            Tuple of (proto_identity, metadata)
        """
        # 1. Image → Frequency spectrum
        freq_spectrum = self.image_mapper.to_frequency_spectrum(image_path)

        # 2. Frequency → Signal
        signal = self._frequency_to_signal(freq_spectrum)

        # 3. Modulate carrier
        proto_identity = self.modulator.modulate(self.carrier, signal,
                                                 modulation_depth=self.modulation_depth)

        # 4. Build metadata
        metadata = {
            'modality': 'image',
            'source': str(image_path),
            'fundamental_freq': self._compute_fundamental_freq(freq_spectrum),
            'image_path': str(image_path),
            'carrier_filtered': True  # Flag for carrier filter architecture
        }

        return proto_identity, metadata

    def encode_audio(self, audio_path: str) -> Tuple[np.ndarray, Dict]:
        """Audio → Signal → Modulate → Proto.

        Args:
            audio_path: Path to audio file

        Returns:
            Tuple of (proto_identity, metadata)
        """
        # 1. Audio → Frequency spectrum
        freq_spectrum = self.audio_mapper.to_frequency_spectrum(audio_path)

        # 2. Frequency → Signal
        signal = self._frequency_to_signal(freq_spectrum)

        # 3. Modulate carrier
        proto_identity = self.modulator.modulate(self.carrier, signal,
                                                 modulation_depth=self.modulation_depth)

        # 4. Build metadata
        metadata = {
            'modality': 'audio',
            'source': str(audio_path),
            'fundamental_freq': self._compute_fundamental_freq(freq_spectrum),
            'audio_path': str(audio_path),
            'carrier_filtered': True  # Flag for carrier filter architecture
        }

        return proto_identity, metadata

    def encode_video(self, video_path: str,
                    frame_sample_rate: int = 30) -> List[Tuple[np.ndarray, Dict]]:
        """Video → Frames → Signal → Modulate → Proto sequence.

        Args:
            video_path: Path to video file
            frame_sample_rate: Sample every Nth frame

        Returns:
            List of (proto_identity, metadata) tuples
        """
        try:
            import cv2
        except ImportError:
            raise ImportError("OpenCV (cv2) required for video encoding. Install with: pip install opencv-python")

        cap = cv2.VideoCapture(str(video_path))
        frames = []
        frame_idx = 0
        fps = cap.get(cv2.CAP_PROP_FPS)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_sample_rate == 0:
                # Convert frame to frequency spectrum via image mapper
                freq_spectrum = self.image_mapper.to_frequency_spectrum_from_array(frame)
                signal = self._frequency_to_signal(freq_spectrum)
                proto = self.modulator.modulate(self.carrier, signal,
                                               modulation_depth=self.modulation_depth)

                metadata = {
                    'modality': 'video',
                    'source': str(video_path),
                    'frame_index': frame_idx,
                    'timestamp': frame_idx / fps if fps > 0 else 0,
                    'fundamental_freq': self._compute_fundamental_freq(freq_spectrum),
                    'carrier_filtered': True  # Flag for carrier filter architecture
                }
                frames.append((proto, metadata))

            frame_idx += 1

        cap.release()
        return frames

    def encode_pdf(self, pdf_path: str) -> List[Tuple[np.ndarray, Dict]]:
        """PDF → Pages → Text+Images → Proto sequence.

        Args:
            pdf_path: Path to PDF file

        Returns:
            List of (proto_identity, metadata) tuples per page
        """
        try:
            import fitz  # PyMuPDF
        except ImportError:
            raise ImportError("PyMuPDF required for PDF encoding. Install with: pip install PyMuPDF")

        doc = fitz.open(str(pdf_path))
        pages = []

        for page_num in range(len(doc)):
            page = doc[page_num]

            # Extract text from page
            text = page.get_text()

            # Encode text as primary proto
            if text.strip():
                proto, _ = self.encode_text(text)
            else:
                # Empty page - create zero proto
                proto = np.zeros((self.height, self.width, 4), dtype=np.float32)

            # TODO: Extract and blend images from PDF page
            # For now, just use text encoding

            metadata = {
                'modality': 'pdf',
                'source': str(pdf_path),
                'page_number': page_num,
                'has_text': bool(text.strip())
            }
            pages.append((proto, metadata))

        doc.close()
        return pages

    def encode_batch_with_clustering(
        self,
        texts: List[str],
        cluster_before_storage: bool = True
    ) -> Dict:
        """Encode batch of texts with optional clustering.

        This method provides integration points for future clustering
        implementations as defined in docs/genesis_interface_definition.md.

        Args:
            texts: List of text strings to encode
            cluster_before_storage: If True, prepare for clustering
                                  (actual clustering left for Step 7)

        Returns:
            Dictionary with:
                - 'proto_identities': List of proto-identity arrays
                - 'metadata': List of metadata dicts
                - 'quaternions': List of multi-octave quaternion dicts
                - 'cluster_ready': Boolean flag for clustering readiness
        """
        results = {
            'proto_identities': [],
            'metadata': [],
            'quaternions': [],
            'cluster_ready': cluster_before_storage
        }

        for text in texts:
            # Encode text to proto-identity
            proto_identity, meta = self.encode_text(text)
            results['proto_identities'].append(proto_identity)
            results['metadata'].append(meta)

            # Extract multi-octave quaternions if clustering enabled
            if cluster_before_storage:
                quaternions = self.extract_multi_octave_quaternions(proto_identity)
                results['quaternions'].append(quaternions)
            else:
                results['quaternions'].append(None)

        # TODO (Step 7): Add actual clustering logic here
        # For now, just mark as ready for clustering
        if cluster_before_storage:
            # Placeholder for clustering integration
            # See docs/genesis_interface_definition.md for full spec
            pass

        return results

    def extract_multi_octave_quaternions(
        self,
        proto_identity: np.ndarray,
        num_octaves: int = 12
    ) -> Dict[int, np.ndarray]:
        """Extract unit quaternions at multiple octave levels.

        Implements spatial pyramid with average pooling to extract
        quaternions at different resolutions, corresponding to different
        semantic levels (see docs/genesis_interface_definition.md).

        Args:
            proto_identity: (H, W, 4) proto-identity array
            num_octaves: Number of octave levels to extract (default 12)

        Returns:
            Dictionary mapping octave level to unit quaternion (w, x, y, z)
        """
        quaternions = {}

        for octave in range(num_octaves):
            if octave == 0:
                # Full resolution
                downsampled = proto_identity
            else:
                # Downsample by factor of 2^octave
                factor = 2 ** octave
                target_h = max(1, self.height // factor)
                target_w = max(1, self.width // factor)

                # Average pooling for downsampling
                downsampled = self._average_pool(proto_identity, target_h, target_w)

            # Extract quaternion from downsampled level
            quaternion = self._extract_quaternion(downsampled)
            quaternions[octave] = quaternion

            # Stop if we've reached 1x1
            if downsampled.shape[0] == 1 and downsampled.shape[1] == 1:
                break

        return quaternions

    def _average_pool(self, data: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
        """Apply average pooling for downsampling.

        Args:
            data: (H, W, C) array to downsample
            target_h: Target height
            target_w: Target width

        Returns:
            (target_h, target_w, C) downsampled array
        """
        h, w, c = data.shape

        # Calculate pool sizes
        pool_h = h // target_h
        pool_w = w // target_w

        # Truncate to exact multiple
        pooled = data[:target_h*pool_h, :target_w*pool_w]

        # Reshape for pooling
        pooled = pooled.reshape(target_h, pool_h, target_w, pool_w, c)

        # Average across pool dimensions
        pooled = pooled.mean(axis=(1, 3))

        return pooled

    def _extract_quaternion(self, data: np.ndarray) -> np.ndarray:
        """Extract energy-weighted quaternion from proto-identity data.

        Args:
            data: (H, W, 4) proto-identity array [X, Y, Z, W]

        Returns:
            Unit quaternion (w, x, y, z)
        """
        # Weight by Z channel (amplitude/truth)
        weights = data[:, :, 2:3]  # Keep dims for broadcasting

        # Energy-weighted average across spatial dimensions
        weighted_data = data * weights
        quaternion = np.mean(weighted_data, axis=(0, 1))

        # Normalize to unit quaternion
        norm = np.linalg.norm(quaternion) + 1e-8
        quaternion /= norm

        # Reorder to standard quaternion format
        # From [X, Y, Z, W] to [W, X, Y, Z]
        quaternion = np.array([
            quaternion[3],  # W (phase velocity) -> w
            quaternion[0],  # X (real) -> x
            quaternion[1],  # Y (imaginary) -> y
            quaternion[2],  # Z (weight) -> z
        ])

        return quaternion

    def cluster_proto_identities_by_octave(
        self,
        proto_identities: List[np.ndarray],
        octave_level: int = 3,  # Word level by default
        n_clusters: Optional[int] = None  # Auto-detect if None
    ) -> Dict:
        """
        Cluster proto-identities at specific octave level.

        For document "The Tao of virtue":
        - Octave 0 (letter): Cluster T, h, e, space, T, a, o...
        - Octave 3 (word): Cluster "The", "Tao", "of", "virtue"
        - Octave 6 (sentence): Cluster sentence-level patterns
        - Octave 11 (document): Cluster document-level patterns

        Args:
            proto_identities: List of proto-identity arrays (H, W, 4)
            octave_level: Which octave to cluster at (0-11)
            n_clusters: Number of clusters (auto if None)

        Returns:
            Dict with cluster_labels, centroids, quaternions, octave_level
        """
        from src.clustering import ProtoUnityClusterer

        # Extract quaternions at specified octave level
        quaternions = []
        for proto in proto_identities:
            octave_quats = self.extract_multi_octave_quaternions(proto)
            if octave_level in octave_quats:
                quaternions.append(octave_quats[octave_level])
            else:
                # Use highest available octave
                max_octave = max(octave_quats.keys())
                quaternions.append(octave_quats[max_octave])

        quaternions = np.array(quaternions)

        # Auto-determine cluster count if not specified
        if n_clusters is None:
            # Use sqrt(n) heuristic, bounded between 2 and 10
            n_samples = len(quaternions)
            n_clusters = min(max(2, int(np.sqrt(n_samples))), 10)

        # Cluster quaternions
        clusterer = ProtoUnityClusterer(n_clusters=n_clusters)
        result = clusterer.fit(quaternions)

        return {
            'cluster_labels': result.cluster_labels,
            'centroids': result.cluster_centers,
            'quaternions': quaternions,
            'octave_level': octave_level,
            'n_clusters': result.n_clusters,
            'cluster_sizes': result.cluster_sizes,
            'inertia': result.inertia
        }

    def _frequency_to_signal(self, freq_spectrum: np.ndarray) -> np.ndarray:
        """Convert frequency spectrum to signal (XYZW format).

        Args:
            freq_spectrum: (H, W, 2) frequency spectrum [magnitude, phase]

        Returns:
            (H, W, 4) signal [X, Y, Z, W]
        """
        # Extract magnitude and phase
        if freq_spectrum.shape[-1] == 2:
            magnitude = freq_spectrum[:, :, 0]
            phase = freq_spectrum[:, :, 1]
        else:
            # If single channel, use as magnitude with zero phase
            magnitude = freq_spectrum.squeeze()
            phase = np.zeros_like(magnitude)

        # Convert to complex representation (XYZW)
        X = magnitude * np.cos(phase)  # Real part
        Y = magnitude * np.sin(phase)  # Imaginary part
        Z = magnitude  # Weight/amplitude
        W = self._compute_phase_velocity(phase)  # Phase velocity

        return np.stack([X, Y, Z, W], axis=-1).astype(np.float32)

    def _compute_phase_velocity(self, phase: np.ndarray) -> np.ndarray:
        """Compute phase velocity from phase field.

        Args:
            phase: (H, W) phase values

        Returns:
            (H, W) phase velocity
        """
        # Compute gradient magnitude
        grad_y, grad_x = np.gradient(phase)
        velocity = np.sqrt(grad_x**2 + grad_y**2)
        return velocity

    def _compute_fundamental_freq(self, freq_spectrum: np.ndarray) -> float:
        """Compute fundamental frequency from spectrum.

        Args:
            freq_spectrum: (H, W, 2) frequency spectrum

        Returns:
            Fundamental frequency (Hz)
        """
        # Extract magnitude
        if freq_spectrum.shape[-1] == 2:
            magnitude = freq_spectrum[:, :, 0]
        else:
            magnitude = freq_spectrum.squeeze()

        # Find peak in magnitude spectrum
        H, W = magnitude.shape
        center_y, center_x = H // 2, W // 2

        # Create frequency coordinates
        y_coords = np.arange(H) - center_y
        x_coords = np.arange(W) - center_x

        # Find weighted center of mass
        total_mag = magnitude.sum()
        if total_mag > 0:
            weighted_y = (magnitude * y_coords[:, None]).sum() / total_mag
            weighted_x = (magnitude * x_coords[None, :]).sum() / total_mag

            # Convert to frequency (assuming 0-8000 Hz range)
            freq_y = abs(weighted_y) / H * 8000.0
            freq_x = abs(weighted_x) / W * 8000.0
            fundamental = np.sqrt(freq_y**2 + freq_x**2)
        else:
            fundamental = 0.0

        return float(fundamental)
