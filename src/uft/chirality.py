"""Chirality analyzer for semantic directionality."""

import numpy as np
import re
from typing import Dict, List, Optional, Tuple


class ChiralityAnalyzer:
    """
    Analyzes text for semantic directionality → chiral phase δ.

    Chirality captures the "handedness" or directional bias in text:
    - Temporal flow (past → future vs future → past)
    - Causal structure (cause → effect vs effect → cause)
    - Grammatical direction (subject → object vs object → subject)
    - Logical flow (premise → conclusion vs conclusion → premise)

    The chiral phase δ affects how the proto-identity evolves:
    - δ > 0: Forward-oriented (progressive, causal)
    - δ < 0: Backward-oriented (retrospective, consequential)
    - δ ≈ 0: Symmetric/neutral (balanced, descriptive)
    """

    def __init__(
        self,
        max_chiral_phase: float = np.pi / 4,
        smoothing_factor: float = 3.0
    ):
        """
        Initialize chirality analyzer.

        Args:
            max_chiral_phase (δ_max): Maximum chiral phase magnitude
            smoothing_factor: Normalization factor for marker counts

        Raises:
            ValueError: If parameters are invalid
        """
        if max_chiral_phase <= 0 or max_chiral_phase > np.pi / 2:
            raise ValueError("Max chiral phase must be in (0, π/2]")
        if smoothing_factor <= 0:
            raise ValueError("Smoothing factor must be positive")

        self.delta_max = max_chiral_phase
        self.smoothing = smoothing_factor

        # Directionality markers (expanded set)
        self.forward_markers = {
            # Causal
            'therefore', 'thus', 'consequently', 'hence', 'so',
            'accordingly', 'as a result', 'it follows',
            # Temporal
            'then', 'next', 'after', 'afterwards', 'following',
            'subsequently', 'later', 'eventually', 'finally',
            # Productive
            'leads to', 'causes', 'results in', 'produces',
            'generates', 'creates', 'yields', 'brings about',
            # Logical
            'implies', 'entails', 'necessitates', 'proves'
        }

        self.backward_markers = {
            # Causal
            'because', 'since', 'due to', 'owing to', 'as a result of',
            'thanks to', 'on account of', 'for',
            # Temporal
            'before', 'previously', 'earlier', 'formerly',
            'originally', 'initially', 'at first',
            # Consequential
            'caused by', 'resulted from', 'stemmed from', 'arose from',
            'originated from', 'derived from', 'came from',
            # Logical
            'given that', 'assuming', 'provided that', 'if'
        }

        # Weight different marker types
        self.marker_weights = {
            'causal': 1.0,
            'temporal': 0.8,
            'logical': 0.9
        }

    def analyze(self, text: str) -> float:
        """
        Analyze text directionality → chiral phase δ.

        Args:
            text: Input text to analyze

        Returns:
            float: Chiral phase δ ∈ [-δ_max, +δ_max]

        Algorithm:
            1. Count directional markers with weights
            2. Analyze sentence structure
            3. Combine into net directionality
            4. Map to chiral phase via tanh
        """
        if not text or len(text.strip()) == 0:
            return 0.0

        text_lower = text.lower()

        # Count directional markers
        forward_score = self._count_markers(text_lower, self.forward_markers)
        backward_score = self._count_markers(text_lower, self.backward_markers)

        # Analyze sentence structure
        structure_bias = self._analyze_structure(text)

        # Net directionality
        net_direction = forward_score - backward_score + 0.5 * structure_bias

        # Normalize with tanh for smooth mapping to [-1, 1]
        normalized = np.tanh(net_direction / self.smoothing)

        # Scale to chiral phase range
        delta = normalized * self.delta_max

        return delta

    def _count_markers(self, text: str, markers: set) -> float:
        """
        Count occurrence of directional markers.

        Args:
            text: Lowercase text to search
            markers: Set of marker words/phrases

        Returns:
            float: Weighted marker count
        """
        count = 0.0

        for marker in markers:
            # Use word boundaries for accurate matching
            pattern = r'\b' + re.escape(marker) + r'\b'
            matches = re.findall(pattern, text)
            count += len(matches)

        return count

    def _analyze_structure(self, text: str) -> float:
        """
        Analyze sentence structure for directionality.

        Args:
            text: Original text

        Returns:
            float: Structure bias (-1 to +1)
        """
        sentences = re.split(r'[.!?]+', text)
        if len(sentences) <= 1:
            return 0.0

        # Check if questions come before statements (backward)
        # or statements come before questions (forward)
        question_positions = []
        statement_positions = []

        for i, sent in enumerate(sentences):
            sent = sent.strip()
            if not sent:
                continue

            if sent.endswith('?') or '?' in sent:
                question_positions.append(i)
            else:
                statement_positions.append(i)

        if question_positions and statement_positions:
            avg_question_pos = np.mean(question_positions)
            avg_statement_pos = np.mean(statement_positions)

            # Questions before statements suggests inquiry (backward)
            # Statements before questions suggests assertion (forward)
            if avg_statement_pos < avg_question_pos:
                return 0.3  # Forward bias
            else:
                return -0.3  # Backward bias

        return 0.0

    def get_markers_found(self, text: str) -> Dict[str, int]:
        """
        Return counts of markers found (for debugging/analysis).

        Args:
            text: Input text

        Returns:
            Dict with 'forward', 'backward' counts
        """
        text_lower = text.lower()

        forward_count = sum(
            len(re.findall(r'\b' + re.escape(m) + r'\b', text_lower))
            for m in self.forward_markers
        )

        backward_count = sum(
            len(re.findall(r'\b' + re.escape(m) + r'\b', text_lower))
            for m in self.backward_markers
        )

        return {
            'forward': forward_count,
            'backward': backward_count,
            'net': forward_count - backward_count
        }

    def get_detailed_analysis(self, text: str) -> Dict[str, any]:
        """
        Perform detailed chirality analysis.

        Args:
            text: Input text

        Returns:
            Dict with detailed analysis results
        """
        markers = self.get_markers_found(text)
        structure_bias = self._analyze_structure(text)
        delta = self.analyze(text)

        return {
            'chiral_phase': delta,
            'chiral_phase_degrees': np.degrees(delta),
            'forward_markers': markers['forward'],
            'backward_markers': markers['backward'],
            'net_markers': markers['net'],
            'structure_bias': structure_bias,
            'interpretation': self._interpret_chirality(delta)
        }

    def _interpret_chirality(self, delta: float) -> str:
        """
        Provide human-readable interpretation of chiral phase.

        Args:
            delta: Chiral phase value

        Returns:
            str: Interpretation
        """
        abs_delta = abs(delta)
        ratio = abs_delta / self.delta_max

        if ratio < 0.1:
            direction = "neutral/balanced"
        elif ratio < 0.3:
            direction = "slightly " + ("forward" if delta > 0 else "backward")
        elif ratio < 0.6:
            direction = "moderately " + ("forward" if delta > 0 else "backward")
        else:
            direction = "strongly " + ("forward" if delta > 0 else "backward")

        return f"{direction}-oriented (δ={delta:.3f} rad)"