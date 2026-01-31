#!/usr/bin/env python3
"""
Universal Dataset Loader for Genesis Multimodal Encoding.

Supports:
- Apache Arrow datasets (HuggingFace format)
- Plain text files (.txt)
- JSON datasets
- Images (.jpg, .png)
- Audio (.wav, .mp3)
- Video (.mp4, .avi)
"""

from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional, Iterator, Any
from dataclasses import dataclass
import json

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
from PIL import Image
import librosa
import soundfile as sf
import cv2


@dataclass
class DatasetSample:
    """Unified dataset sample container."""
    data: Union[str, np.ndarray, Dict[str, Any]]
    modality: str  # 'text', 'image', 'audio', 'video'
    metadata: Dict[str, Any]
    source_path: Optional[str] = None


class UniversalDatasetLoader:
    """
    Load datasets from multiple formats and modalities.

    Supported formats:
    - Arrow: HuggingFace datasets (instruction/input/output, etc.)
    - Text: Plain .txt files
    - JSON: {title, content} or list of objects
    - Images: JPG, PNG (RGB arrays)
    - Audio: WAV, MP3 (audio arrays + sample rate)
    - Video: MP4, AVI (frame arrays + audio)
    """

    def __init__(self):
        self.supported_extensions = {
            'text': ['.txt', '.md'],
            'arrow': ['.arrow'],
            'json': ['.json'],
            'image': ['.jpg', '.jpeg', '.png', '.bmp'],
            'audio': ['.wav', '.mp3', '.flac', '.ogg'],
            'video': ['.mp4', '.avi', '.mkv', '.mov']
        }

    def detect_modality(self, path: Union[str, Path]) -> str:
        """Auto-detect modality from file extension."""
        path = Path(path)
        ext = path.suffix.lower()

        for modality, extensions in self.supported_extensions.items():
            if ext in extensions:
                return modality

        raise ValueError(f"Unsupported file extension: {ext}")

    # ========================================================================
    # ARROW DATASET LOADING (HuggingFace format)
    # ========================================================================

    def load_arrow_dataset(
        self,
        dataset_path: Union[str, Path],
        columns: Optional[List[str]] = None,
        limit: Optional[int] = None
    ) -> Iterator[DatasetSample]:
        """
        Load Apache Arrow dataset.

        Common schemas:
        - {instruction, input, output}
        - {question, answer}
        - {text, label}
        """
        dataset_path = Path(dataset_path)

        if dataset_path.is_file():
            arrow_ds = ds.dataset(dataset_path, format='arrow')
        else:
            arrow_ds = ds.dataset(dataset_path, format='arrow')

        table = arrow_ds.to_table(columns=columns)

        for i, batch in enumerate(table.to_batches()):
            if limit and i >= limit:
                break

            batch_dict = batch.to_pydict()
            num_rows = len(batch_dict[list(batch_dict.keys())[0]])

            for row_idx in range(num_rows):
                row = {k: v[row_idx] for k, v in batch_dict.items()}

                yield DatasetSample(
                    data=row,
                    modality='text',
                    metadata={
                        'format': 'arrow',
                        'schema': list(row.keys()),
                        'source_file': str(dataset_path)
                    },
                    source_path=str(dataset_path)
                )

    # ========================================================================
    # TEXT LOADING
    # ========================================================================

    def load_text_file(self, file_path: Union[str, Path]) -> DatasetSample:
        """Load plain text file."""
        file_path = Path(file_path)

        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        return DatasetSample(
            data=content,
            modality='text',
            metadata={
                'format': 'text',
                'filename': file_path.name,
                'size_bytes': len(content.encode('utf-8'))
            },
            source_path=str(file_path)
        )

    # ========================================================================
    # JSON LOADING
    # ========================================================================

    def load_json_dataset(
        self,
        file_path: Union[str, Path]
    ) -> Iterator[DatasetSample]:
        """Load JSON dataset (single object or array of objects)."""
        file_path = Path(file_path)

        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if isinstance(data, dict):
            yield DatasetSample(
                data=data,
                modality='text',
                metadata={'format': 'json', 'type': 'object'},
                source_path=str(file_path)
            )
        elif isinstance(data, list):
            for item in data:
                yield DatasetSample(
                    data=item,
                    modality='text',
                    metadata={'format': 'json', 'type': 'array_item'},
                    source_path=str(file_path)
                )
        else:
            raise ValueError(f"Unsupported JSON structure: {type(data)}")

    # ========================================================================
    # IMAGE LOADING
    # ========================================================================

    def load_image(
        self,
        file_path: Union[str, Path],
        target_size: Optional[Tuple[int, int]] = None
    ) -> DatasetSample:
        """
        Load image file.

        Returns RGB array (H, W, 3).
        """
        file_path = Path(file_path)

        image = Image.open(file_path).convert('RGB')

        if target_size:
            image = image.resize(target_size, Image.Resampling.LANCZOS)

        image_array = np.array(image)  # (H, W, 3)

        return DatasetSample(
            data=image_array,
            modality='image',
            metadata={
                'format': 'image',
                'shape': image_array.shape,
                'dtype': str(image_array.dtype),
                'filename': file_path.name
            },
            source_path=str(file_path)
        )

    # ========================================================================
    # AUDIO LOADING
    # ========================================================================

    def load_audio(
        self,
        file_path: Union[str, Path],
        target_sr: int = 22050,
        mono: bool = True
    ) -> DatasetSample:
        """
        Load audio file.

        Returns audio array + sample rate in metadata.
        """
        file_path = Path(file_path)

        audio, sr = librosa.load(file_path, sr=target_sr, mono=mono)

        return DatasetSample(
            data=audio,  # (num_samples,) if mono, (num_samples, channels) if stereo
            modality='audio',
            metadata={
                'format': 'audio',
                'sample_rate': sr,
                'duration_sec': len(audio) / sr,
                'shape': audio.shape,
                'dtype': str(audio.dtype),
                'filename': file_path.name
            },
            source_path=str(file_path)
        )

    # ========================================================================
    # VIDEO LOADING
    # ========================================================================

    def load_video(
        self,
        file_path: Union[str, Path],
        max_frames: Optional[int] = None
    ) -> DatasetSample:
        """
        Load video file.

        Returns dict with:
        - frames: (num_frames, H, W, 3) array
        - audio: audio array (if present)
        - fps: frames per second
        - sample_rate: audio sample rate
        """
        file_path = Path(file_path)

        cap = cv2.VideoCapture(str(file_path))
        fps = cap.get(cv2.CAP_PROP_FPS)

        frames = []
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
            frame_count += 1

            if max_frames and frame_count >= max_frames:
                break

        cap.release()

        frames_array = np.array(frames) if frames else np.array([])

        video_data = {
            'frames': frames_array,
            'fps': fps,
            'num_frames': len(frames)
        }

        return DatasetSample(
            data=video_data,
            modality='video',
            metadata={
                'format': 'video',
                'fps': fps,
                'num_frames': len(frames),
                'frame_shape': frames_array[0].shape if len(frames) > 0 else None,
                'filename': file_path.name
            },
            source_path=str(file_path)
        )

    # ========================================================================
    # UNIFIED LOAD METHOD
    # ========================================================================

    def load(
        self,
        path: Union[str, Path],
        modality: Optional[str] = None,
        **kwargs
    ) -> Union[DatasetSample, Iterator[DatasetSample]]:
        """
        Universal load method - auto-detects format and modality.

        Args:
            path: Path to file/directory
            modality: Optional explicit modality (auto-detected if None)
            **kwargs: Format-specific arguments

        Returns:
            Single DatasetSample or iterator of samples
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Path not found: {path}")

        if modality is None:
            modality = self.detect_modality(path)

        if modality == 'arrow':
            return self.load_arrow_dataset(path, **kwargs)
        elif modality == 'text':
            return self.load_text_file(path)
        elif modality == 'json':
            return self.load_json_dataset(path)
        elif modality == 'image':
            return self.load_image(path, **kwargs)
        elif modality == 'audio':
            return self.load_audio(path, **kwargs)
        elif modality == 'video':
            return self.load_video(path, **kwargs)
        else:
            raise ValueError(f"Unknown modality: {modality}")

    # ========================================================================
    # BATCH LOADING
    # ========================================================================

    def load_directory(
        self,
        directory: Union[str, Path],
        pattern: str = "*",
        modality: Optional[str] = None,
        limit: Optional[int] = None
    ) -> Iterator[DatasetSample]:
        """Load all files matching pattern from directory."""
        directory = Path(directory)

        files = sorted(directory.glob(pattern))

        for i, file_path in enumerate(files):
            if limit and i >= limit:
                break

            try:
                result = self.load(file_path, modality=modality)

                if hasattr(result, '__iter__') and not isinstance(result, (str, bytes, np.ndarray)):
                    yield from result
                else:
                    yield result
            except Exception as e:
                print(f"Warning: Failed to load {file_path}: {e}")
                continue
