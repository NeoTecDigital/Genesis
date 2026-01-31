"""
Text to Kuramoto oscillator encoder.

Maps text characters to natural frequencies for phase oscillators.
"""

import numpy as np
import hashlib
from typing import Optional, Union


class TextToOscillators:
    """
    Maps text to natural frequencies for Kuramoto oscillators.

    Each character in the text becomes an oscillator with a deterministic
    natural frequency derived from its character value and position.
    """

    def __init__(self, frequency_scale: float = 1.0, position_weight: float = 0.1):
        """
        Initialize text encoder.

        Args:
            frequency_scale: Scaling factor for frequencies
            position_weight: Weight for position-dependent frequency modulation
        """
        if frequency_scale <= 0:
            raise ValueError("Frequency scale must be positive")
        if position_weight < 0:
            raise ValueError("Position weight must be non-negative")

        self.frequency_scale = frequency_scale
        self.position_weight = position_weight

    def encode(self, text: str) -> np.ndarray:
        """
        Convert text to natural frequencies.

        Args:
            text: Input string

        Returns:
            np.ndarray: Natural frequencies [ω_1, ..., ω_N]

        Implementation:
            - Base frequency from character value normalized to [0, 1]
            - Position modulation adds context-dependent variation
            - Deterministic hash ensures reproducibility
        """
        if not text:
            raise ValueError("Cannot encode empty text")

        N = len(text)
        frequencies = np.zeros(N)

        # Generate deterministic seed from text for reproducibility
        text_hash = hashlib.sha256(text.encode()).digest()
        seed = int.from_bytes(text_hash[:4], byteorder='big')
        rng = np.random.RandomState(seed)

        for i, char in enumerate(text):
            # Base frequency from character value
            char_value = ord(char)
            base_freq = (char_value / 255.0) * self.frequency_scale

            # Position-dependent modulation
            position_mod = self.position_weight * np.sin(2 * np.pi * i / N)

            # Add small deterministic noise for diversity
            noise = rng.normal(0, 0.01 * self.frequency_scale)

            frequencies[i] = base_freq + position_mod + noise

        return frequencies

    def encode_hierarchical(self, text: str, levels: int = 3) -> dict:
        """
        Encode text at multiple hierarchical levels.

        Args:
            text: Input string
            levels: Number of hierarchical levels

        Returns:
            dict: Frequencies at each level {
                'character': character-level frequencies,
                'word': word-level frequencies,
                'phrase': phrase-level frequencies (if applicable)
            }
        """
        result = {}

        # Character level
        result['character'] = self.encode(text)

        # Word level
        words = text.split()
        if words:
            word_freqs = []
            for word in words:
                # Average frequency of characters in word
                char_freqs = self.encode(word)
                word_freqs.append(np.mean(char_freqs))
            result['word'] = np.array(word_freqs)

        # Phrase level (sliding window)
        if levels >= 3 and len(words) > 2:
            phrase_size = 3
            phrase_freqs = []
            for i in range(len(words) - phrase_size + 1):
                phrase = ' '.join(words[i:i+phrase_size])
                char_freqs = self.encode(phrase)
                phrase_freqs.append(np.mean(char_freqs))
            result['phrase'] = np.array(phrase_freqs)

        return result

    def decode_estimate(self, frequencies: np.ndarray) -> str:
        """
        Attempt to estimate original text from frequencies (lossy).

        This is a best-effort reconstruction for debugging purposes.
        The original text cannot be perfectly recovered from frequencies alone.

        Args:
            frequencies: Natural frequencies

        Returns:
            Estimated text string
        """
        chars = []
        for freq in frequencies:
            # Reverse the encoding (approximate)
            char_value = int((freq / self.frequency_scale) * 255)
            char_value = np.clip(char_value, 32, 126)  # Printable ASCII range
            chars.append(chr(char_value))

        return ''.join(chars)


if __name__ == "__main__":
    # Quick validation
    print("Testing text encoder...")

    encoder = TextToOscillators(frequency_scale=2.0)

    # Test basic encoding
    text = "Hello world"
    frequencies = encoder.encode(text)

    print(f"Text: '{text}'")
    print(f"Number of oscillators: {len(frequencies)}")
    print(f"Frequency range: [{frequencies.min():.3f}, {frequencies.max():.3f}]")
    print(f"Mean frequency: {frequencies.mean():.3f}")

    # Test reproducibility
    freq2 = encoder.encode(text)
    print(f"Reproducible: {np.allclose(frequencies, freq2)}")

    # Test hierarchical encoding
    hierarchical = encoder.encode_hierarchical(text)
    for level, freqs in hierarchical.items():
        print(f"{level.capitalize()} level: {len(freqs)} oscillators")

    # Test approximate decoding
    estimated = encoder.decode_estimate(frequencies)
    print(f"Estimated text: '{estimated}'")