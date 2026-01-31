"""Decoding Pipeline - Proto-Identity → Multi-Modal Output.

Universal frequency-based synthesis - no static semantic tables.
All modalities decode from frequency via inverse FFT.
"""

import numpy as np
from typing import List, Optional, Union
from dataclasses import dataclass
from scipy.ndimage import label
from src.memory.fm_modulation_base import FMModulationBase


@dataclass
class DecodingConfig:
    """Configuration for decoding pipeline."""
    similarity_threshold: float = 0.3
    resonance_weight: float = 0.5
    synthesis_temperature: float = 1.0
    energy_threshold: float = 0.2
    modulation_depth: float = 0.5  # Must match encoding modulation_depth


class DecodingPipeline:
    """Universal frequency → modality decoder.

    Core principle: All modalities decode from frequency patterns.
    - Text: proto → frequency → inverse FFT → signal → text
    - Image: proto → frequency → inverse FFT → pixels
    - Audio: proto → frequency → inverse FFT → waveform
    """

    def __init__(self, carrier: np.ndarray, config: Optional[DecodingConfig] = None):
        """Initialize decoding pipeline.

        Args:
            carrier: Proto-unity carrier (H, W, 4)
            config: Optional decoding configuration
        """
        self.carrier = carrier
        self.config = config if config is not None else DecodingConfig()
        self.demodulator = FMModulationBase()

    def decode_to_text(self, proto: np.ndarray, metadata: dict = None) -> str:
        """Reconstruct text using native STFT from metadata.

        ARCHITECTURE NOTE:
        - The 512×512 proto stores interference patterns for similarity matching
        - But text reconstruction requires the original STFT time-frequency structure
        - System is RETRIEVAL-based with proto-identity for clustering/matching

        Args:
            proto: Proto-identity (H, W, 4) - used for similarity, not reconstruction
            metadata: Metadata with native_stft for ISTFT reconstruction

        Returns:
            Text reconstructed from native STFT
        """
        from src.memory.frequency_field import TextFrequencyAnalyzer
        text_analyzer = TextFrequencyAnalyzer(width=self.carrier.shape[1], height=self.carrier.shape[0])

        # Extract native STFT from metadata for lossless reconstruction
        if metadata and 'native_stft' in metadata:
            native_stft_bytes = metadata['native_stft']
            native_stft_shape = metadata['native_stft_shape']
            original_length = metadata.get('original_length', None)

            # Deserialize native STFT
            native_stft = np.frombuffer(native_stft_bytes, dtype=np.float32)
            native_stft = native_stft.reshape(native_stft_shape)

            # ISTFT → text (lossless reconstruction)
            text = text_analyzer.from_frequency_spectrum(
                None,  # Not used when native_stft provided
                native_stft=native_stft,
                original_length=original_length
            )
        else:
            # No native STFT - cannot reconstruct
            text = "[no reconstruction data]"

        return text

    def decode_to_image(self, proto: np.ndarray) -> np.ndarray:
        """Proto → image via inverse frequency transform.

        Args:
            proto: Proto-identity (H, W, 4)

        Returns:
            RGB image (H, W, 3)
        """
        # Extract frequency spectrum
        freq_spectrum = self._proto_to_frequency(proto)

        # Inverse FFT to spatial domain
        signal = self._inverse_fft_to_signal(freq_spectrum)

        # Normalize to image range [0, 255]
        signal_min = signal.min()
        signal_max = signal.max()
        if signal_max > signal_min:
            normalized = (signal - signal_min) / (signal_max - signal_min)
        else:
            normalized = np.zeros_like(signal)

        # Convert to RGB
        image = (normalized * 255).astype(np.uint8)
        image = np.stack([image, image, image], axis=-1)

        return image

    def decode_to_audio(
        self,
        proto: np.ndarray,
        sample_rate: int = 44100,
        duration: float = 1.0
    ) -> np.ndarray:
        """Proto → audio via inverse frequency transform.

        Args:
            proto: Proto-identity (H, W, 4)
            sample_rate: Audio sample rate in Hz
            duration: Duration in seconds

        Returns:
            Audio waveform (samples,)
        """
        # Extract frequency spectrum
        freq_spectrum = self._proto_to_frequency(proto)

        # Generate temporal signal
        signal = self._inverse_fft_to_signal(freq_spectrum)

        # Reshape to 1D and interpolate to target length
        signal_1d = signal.flatten()
        target_length = int(sample_rate * duration)

        # Simple linear interpolation
        x_old = np.linspace(0, 1, len(signal_1d))
        x_new = np.linspace(0, 1, target_length)
        waveform = np.interp(x_new, x_old, signal_1d)

        # Normalize to audio range [-1, 1]
        if waveform.max() > waveform.min():
            waveform = 2 * (waveform - waveform.min()) / (waveform.max() - waveform.min()) - 1

        return waveform.astype(np.float32)

    def decode_to_video(
        self,
        protos: List[np.ndarray],
        output_path: str,
        fps: int = 30
    ) -> None:
        """Proto sequence → video.

        Args:
            protos: List of proto-identities (frames)
            output_path: Path to save video
            fps: Frames per second
        """
        try:
            import cv2
        except ImportError:
            raise ImportError("OpenCV required for video generation. Install with: pip install opencv-python")

        # Decode first frame to get dimensions
        first_frame = self.decode_to_image(protos[0])
        height, width = first_frame.shape[:2]

        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        # Decode and write each frame
        for proto in protos:
            frame = self.decode_to_image(proto)
            out.write(frame)

        out.release()

    def decode_to_pdf(self, protos: List[np.ndarray],
                     output_path: str,
                     page_size: tuple = (612, 792)) -> None:
        """Proto sequence → PDF.

        Args:
            protos: List of proto-identities (pages)
            output_path: Path to save PDF
            page_size: PDF page size in points (width, height)
        """
        try:
            import fitz  # PyMuPDF
        except ImportError:
            raise ImportError("PyMuPDF required for PDF generation. Install with: pip install PyMuPDF")

        doc = fitz.open()  # Create new PDF

        for proto in protos:
            # Create new page
            page = doc.new_page(width=page_size[0], height=page_size[1])

            # Decode proto to text
            text = self.decode_to_text(proto)

            # Insert text on page
            page.insert_text((72, 72), text, fontsize=12)

        doc.save(str(output_path))
        doc.close()

    def decode_to_summary(
        self,
        query_proto: np.ndarray,
        visible_protos: List
    ) -> str:
        """Generate response via weighted synthesis from proto-identities.

        GENERATIVE approach - synthesizes new proto-identity from context:
        1. Gather proto-identities from context
        2. Weight by resonance_strength * similarity
        3. Synthesize weighted proto-identity (quaternionic blend)
        4. Decode via: proto → demodulate → INVERSE FFT → text

        NO text storage or retrieval - pure generation from proto patterns.

        Args:
            query_proto: Query proto-identity (H, W, 4)
            visible_protos: List of ProtoIdentityEntry objects

        Returns:
            Generated text response
        """
        if not visible_protos:
            return "[no context]"

        # 1. Gather proto-identities and compute weights
        weighted_protos = []
        weights = []

        for entry in visible_protos:
            # Compute similarity
            similarity = self._compute_similarity(query_proto, entry.proto_identity)

            # Weight by resonance (how often seen) AND similarity (relevance)
            weight = similarity * entry.resonance_strength

            weighted_protos.append(entry.proto_identity)
            weights.append(weight)

        # 2. Synthesize new proto-identity from weighted context
        # This creates CONSTRUCTIVE INTERFERENCE in quaternionic space
        synthesized_proto = self._weighted_synthesis(weighted_protos, weights)

        # 3. Decode via TRUE GENERATIVE path: proto → demodulate → signal → text
        # NO metadata, NO native_stft - pure pattern extraction
        freq_spectrum = self._proto_to_frequency(synthesized_proto)
        signal = self._inverse_fft_to_signal(freq_spectrum)
        text = self._signal_to_text(signal)

        return text

    def _weighted_synthesis(
        self,
        protos: List[np.ndarray],
        weights: List[float]
    ) -> np.ndarray:
        """Synthesize new proto-identity from weighted context.

        Constructive interference in quaternionic space:
        - Combines proto-identities weighted by relevance
        - Similar to gravitational collapse but for response generation

        Args:
            protos: List of proto-identities (H, W, 4)
            weights: List of weights (resonance * similarity)

        Returns:
            Synthesized proto-identity (H, W, 4)
        """
        # Normalize weights
        weights = np.array(weights)
        if weights.sum() > 0:
            weights = weights / weights.sum()
        else:
            weights = np.ones(len(weights)) / len(weights)

        # Apply temperature scaling
        if self.config.synthesis_temperature != 1.0:
            weights = np.power(weights, 1.0 / self.config.synthesis_temperature)
            weights = weights / weights.sum()

        # Weighted combination in quaternionic space
        synthesized = np.zeros_like(protos[0], dtype=np.float32)
        for proto, weight in zip(protos, weights):
            synthesized += weight * proto

        return synthesized

    def _proto_to_frequency(self, proto: np.ndarray) -> np.ndarray:
        """Extract frequency spectrum from proto-identity via demodulation.

        CRITICAL: Demodulates proto with carrier to extract original signal.
        This is the INVERSE of encoding's FM modulation step.

        Encoding: signal → modulate(carrier, signal) → proto
        Decoding: proto → demodulate(proto, carrier) → signal

        Args:
            proto: Proto-identity (H, W, 4)

        Returns:
            Frequency spectrum (H, W, 2) [magnitude, phase]
        """
        # DEMODULATE proto-identity with carrier to extract original signal
        signal = self.demodulator.demodulate(
            proto,
            self.carrier,
            modulation_depth=self.config.modulation_depth
        )

        # Extract XY channels (complex representation of frequency)
        X = signal[:, :, 0]
        Y = signal[:, :, 1]

        # Convert to magnitude and phase
        magnitude = np.sqrt(X**2 + Y**2)
        phase = np.arctan2(Y, X)

        return np.stack([magnitude, phase], axis=-1).astype(np.float32)

    def _inverse_fft_to_signal(self, freq_spectrum: np.ndarray) -> np.ndarray:
        """INVERSE FFT: frequency domain → spatial/temporal domain.

        Args:
            freq_spectrum: (H, W, 2) [magnitude, phase]

        Returns:
            Spatial signal (H, W)
        """
        magnitude = freq_spectrum[:, :, 0]
        phase = freq_spectrum[:, :, 1]

        # Reconstruct complex signal
        complex_signal = magnitude * np.exp(1j * phase)

        # INVERSE FFT
        spatial_signal = np.fft.ifft2(complex_signal).real

        return spatial_signal

    def _signal_to_text(self, signal: np.ndarray) -> str:
        """Extract text from spatial signal energy patterns.

        Universal approach - works for any language/script:
        1. Find energy peaks in signal
        2. Extract local patterns (character-like regions)
        3. Map to nearest printable characters by energy signature

        NO language-specific phoneme tables!

        Args:
            signal: Spatial signal (H, W)

        Returns:
            Text approximation
        """
        # Find local energy maxima
        threshold = signal.max() * self.config.energy_threshold
        peaks = signal > threshold

        # Extract character-like patterns
        labeled, num_features = label(peaks)

        # Limit to reasonable number of characters
        max_chars = min(num_features, 100)

        # For each region, extract representative character
        chars = []
        for i in range(1, max_chars + 1):
            region = (labeled == i)
            region_signal = signal[region]

            if len(region_signal) > 0:
                # Map energy pattern to character
                char = self._energy_to_char(region_signal)
                chars.append(char)

        return ''.join(chars) if chars else "[silence]"

    def _energy_to_char(self, energy: np.ndarray) -> str:
        """Map energy pattern to character.

        Universal mapping based on signal properties, not phonetics.
        Different energy patterns → different characters.

        Args:
            energy: Signal energy in region

        Returns:
            Character approximation
        """
        # Compute signal signature
        mean_energy = energy.mean()
        std_energy = energy.std()
        max_energy = energy.max() + 1e-8

        # Map to printable ASCII based on energy characteristics
        # Higher energy → later alphabet
        # Higher variance → consonants vs vowels

        char_index = int((mean_energy / max_energy) * 26)
        char_index = np.clip(char_index, 0, 25)

        if std_energy > mean_energy * 0.5:
            # High variance → consonants
            consonants = 'bcdfghjklmnpqrstvwxyz'
            char = consonants[char_index % len(consonants)]
        else:
            # Low variance → vowels
            vowels = 'aeiou'
            char = vowels[char_index % len(vowels)]

        return char

    def _compute_similarity(
        self,
        proto1: np.ndarray,
        proto2: np.ndarray
    ) -> float:
        """Compute similarity between proto-identities.

        Args:
            proto1: First proto-identity (H, W, 4)
            proto2: Second proto-identity (H, W, 4)

        Returns:
            Similarity score [0, 1]
        """
        # Flatten and normalize
        v1 = proto1.flatten()
        v2 = proto2.flatten()

        # Compute cosine similarity
        dot_product = np.dot(v1, v2)
        norm1 = np.linalg.norm(v1) + 1e-8
        norm2 = np.linalg.norm(v2) + 1e-8

        similarity = dot_product / (norm1 * norm2)

        # Map to [0, 1]
        return (similarity + 1.0) / 2.0