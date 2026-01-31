"""
Q&A test suite for Foundation model validation.

Tests Genesis ability to answer questions about Foundation texts.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import pickle
import numpy as np
from pathlib import Path
from typing import List, Optional

from src.memory.voxel_cloud import VoxelCloud
from src.memory.frequency_field import TextFrequencyAnalyzer


# Load voxel cloud once for all tests
@pytest.fixture(scope='module')
def foundation_model():
    """Load the trained Foundation model."""
    model_path = Path('/usr/lib/alembic/checkpoints/genesis/foundation_voxel_cloud.pkl')
    if not model_path.exists():
        pytest.skip("Foundation model not trained yet. Run: python scripts/train_foundation.py")

    with open(model_path, 'rb') as f:
        return pickle.load(f)


@pytest.fixture(scope='module')
def freq_analyzer():
    """Create frequency analyzer for queries."""
    return TextFrequencyAnalyzer(512, 512)


def synthesize(voxel_cloud: VoxelCloud, query: str, freq_analyzer: TextFrequencyAnalyzer) -> str:
    """Helper to synthesize response from voxel cloud."""
    # Convert query to frequency
    query_freq, _ = freq_analyzer.analyze(query)

    # Query viewport for relevant proto-identities
    visible_protos = voxel_cloud.query_viewport(query_freq, radius=50.0)

    if not visible_protos:
        return "No relevant patterns found"

    # TODO Phase 11: Derive text from frequency spectrum
    # Text is no longer stored in metadata - will be derived from signal
    # For testing purposes, return placeholder

    return "Text derivation from frequency spectrum pending (Phase 11)"


class TestPhilosophicalQuestions:
    """Test philosophical understanding from Tao, Gita, Upanishads, etc."""

    def test_tao_emptiness(self, foundation_model, freq_analyzer):
        """Can it explain emptiness from Tao Te Ching?"""
        result = synthesize(foundation_model, "What is emptiness in Taoism?", freq_analyzer)
        assert len(result) > 10  # Non-trivial response
        # Check for relevant concepts
        keywords = ['empty', 'void', 'nothing', 'tao', 'way']
        assert any(word in result.lower() for word in keywords)

    def test_dharma_concept(self, foundation_model, freq_analyzer):
        """Can it explain dharma from Bhagavad Gita?"""
        result = synthesize(foundation_model, "What is dharma?", freq_analyzer)
        assert len(result) > 10
        keywords = ['duty', 'righteousness', 'path', 'law', 'order']
        assert any(word in result.lower() for word in keywords)

    def test_karma_understanding(self, foundation_model, freq_analyzer):
        """Can it explain karma?"""
        result = synthesize(foundation_model, "What is karma?", freq_analyzer)
        assert len(result) > 10
        keywords = ['action', 'consequence', 'deed', 'cause', 'effect']
        assert any(word in result.lower() for word in keywords)

    def test_enlightenment(self, foundation_model, freq_analyzer):
        """Can it describe enlightenment?"""
        result = synthesize(foundation_model, "What is enlightenment?", freq_analyzer)
        assert len(result) > 10
        keywords = ['wisdom', 'knowledge', 'truth', 'liberation', 'awakening']
        assert any(word in result.lower() for word in keywords)

    def test_suffering_concept(self, foundation_model, freq_analyzer):
        """Can it explain suffering from various traditions?"""
        result = synthesize(foundation_model, "What causes suffering?", freq_analyzer)
        assert len(result) > 10
        keywords = ['desire', 'attachment', 'pain', 'loss', 'ignorance']
        assert any(word in result.lower() for word in keywords)

    def test_virtue_ethics(self, foundation_model, freq_analyzer):
        """Can it discuss virtue from Meditations?"""
        result = synthesize(foundation_model, "What is virtue according to Marcus Aurelius?", freq_analyzer)
        assert len(result) > 10
        keywords = ['virtue', 'wisdom', 'justice', 'courage', 'temperance']
        assert any(word in result.lower() for word in keywords)

    def test_biblical_love(self, foundation_model, freq_analyzer):
        """Can it explain love from Bible?"""
        result = synthesize(foundation_model, "What does the Bible say about love?", freq_analyzer)
        assert len(result) > 10
        keywords = ['love', 'compassion', 'kindness', 'patient', 'god']
        assert any(word in result.lower() for word in keywords)

    def test_upanishad_self(self, foundation_model, freq_analyzer):
        """Can it explain the Self from Upanishads?"""
        result = synthesize(foundation_model, "What is Atman?", freq_analyzer)
        assert len(result) > 10
        keywords = ['self', 'soul', 'consciousness', 'eternal', 'brahman']
        assert any(word in result.lower() for word in keywords)

    def test_stoic_control(self, foundation_model, freq_analyzer):
        """Can it explain Stoic control?"""
        result = synthesize(foundation_model, "What can we control according to Stoicism?", freq_analyzer)
        assert len(result) > 10
        keywords = ['control', 'choice', 'judgment', 'opinion', 'will']
        assert any(word in result.lower() for word in keywords)

    def test_mystical_knowledge(self, foundation_model, freq_analyzer):
        """Can it discuss mystical knowledge from Nag Hammadi?"""
        result = synthesize(foundation_model, "What is gnosis?", freq_analyzer)
        assert len(result) > 10
        keywords = ['knowledge', 'wisdom', 'divine', 'spiritual', 'hidden']
        assert any(word in result.lower() for word in keywords)


class TestHistoricalQuestions:
    """Test historical understanding from Iliad, Gilgamesh, Canterbury Tales, etc."""

    def test_achilles_character(self, foundation_model, freq_analyzer):
        """Can it describe Achilles from Iliad?"""
        result = synthesize(foundation_model, "Who was Achilles?", freq_analyzer)
        assert len(result) > 10
        keywords = ['warrior', 'hero', 'greek', 'troy', 'heel']
        assert any(word in result.lower() for word in keywords)

    def test_gilgamesh_quest(self, foundation_model, freq_analyzer):
        """Can it explain Gilgamesh's quest?"""
        result = synthesize(foundation_model, "What did Gilgamesh seek?", freq_analyzer)
        assert len(result) > 10
        keywords = ['immortality', 'eternal', 'life', 'death', 'friend']
        assert any(word in result.lower() for word in keywords)

    def test_trojan_war(self, foundation_model, freq_analyzer):
        """Can it describe the Trojan War?"""
        result = synthesize(foundation_model, "What was the Trojan War?", freq_analyzer)
        assert len(result) > 10
        keywords = ['troy', 'greek', 'war', 'helen', 'horse']
        assert any(word in result.lower() for word in keywords)

    def test_canterbury_pilgrims(self, foundation_model, freq_analyzer):
        """Can it describe Canterbury Tales?"""
        result = synthesize(foundation_model, "What are the Canterbury Tales about?", freq_analyzer)
        assert len(result) > 10
        keywords = ['pilgrim', 'tale', 'story', 'canterbury', 'journey']
        assert any(word in result.lower() for word in keywords)

    def test_dead_sea_scrolls(self, foundation_model, freq_analyzer):
        """Can it discuss Dead Sea Scrolls?"""
        result = synthesize(foundation_model, "What are the Dead Sea Scrolls?", freq_analyzer)
        assert len(result) > 10
        keywords = ['scroll', 'dead', 'sea', 'qumran', 'ancient']
        assert any(word in result.lower() for word in keywords)

    def test_alice_wonderland(self, foundation_model, freq_analyzer):
        """Can it describe Alice in Wonderland?"""
        result = synthesize(foundation_model, "Who is Alice in Wonderland?", freq_analyzer)
        assert len(result) > 10
        keywords = ['alice', 'wonderland', 'rabbit', 'queen', 'mad']
        assert any(word in result.lower() for word in keywords)

    def test_book_of_enoch(self, foundation_model, freq_analyzer):
        """Can it discuss Book of Enoch?"""
        result = synthesize(foundation_model, "What is the Book of Enoch about?", freq_analyzer)
        assert len(result) > 10
        keywords = ['enoch', 'angel', 'heaven', 'watcher', 'vision']
        assert any(word in result.lower() for word in keywords)

    def test_common_sense_paine(self, foundation_model, freq_analyzer):
        """Can it discuss Common Sense by Thomas Paine?"""
        result = synthesize(foundation_model, "What did Thomas Paine argue in Common Sense?", freq_analyzer)
        assert len(result) > 10
        keywords = ['independence', 'america', 'britain', 'freedom', 'government']
        assert any(word in result.lower() for word in keywords)

    def test_biblical_creation(self, foundation_model, freq_analyzer):
        """Can it describe Biblical creation?"""
        result = synthesize(foundation_model, "How does the Bible describe creation?", freq_analyzer)
        assert len(result) > 10
        keywords = ['god', 'create', 'heaven', 'earth', 'light']
        assert any(word in result.lower() for word in keywords)

    def test_epic_heroes(self, foundation_model, freq_analyzer):
        """Can it compare epic heroes?"""
        result = synthesize(foundation_model, "What makes an epic hero?", freq_analyzer)
        assert len(result) > 10
        keywords = ['hero', 'courage', 'strength', 'quest', 'journey']
        assert any(word in result.lower() for word in keywords)


class TestStrategyQuestions:
    """Test strategic understanding from Art of War, 48 Laws, Book of Five Rings, etc."""

    def test_sun_tzu_deception(self, foundation_model, freq_analyzer):
        """Can it recall Sun Tzu on deception?"""
        result = synthesize(foundation_model, "What did Sun Tzu say about deception?", freq_analyzer)
        assert len(result) > 10
        keywords = ['war', 'deception', 'enemy', 'strategy', 'art']
        assert any(word in result.lower() for word in keywords)

    def test_48_laws_power(self, foundation_model, freq_analyzer):
        """Can it discuss laws of power?"""
        result = synthesize(foundation_model, "What are the laws of power?", freq_analyzer)
        assert len(result) > 10
        keywords = ['power', 'law', 'enemy', 'friend', 'master']
        assert any(word in result.lower() for word in keywords)

    def test_five_rings_musashi(self, foundation_model, freq_analyzer):
        """Can it discuss Book of Five Rings?"""
        result = synthesize(foundation_model, "What are the five rings according to Musashi?", freq_analyzer)
        assert len(result) > 10
        keywords = ['ring', 'water', 'fire', 'earth', 'void', 'sword']
        assert any(word in result.lower() for word in keywords)

    def test_machiavelli_prince(self, foundation_model, freq_analyzer):
        """Can it discuss Machiavelli's advice?"""
        result = synthesize(foundation_model, "What did Machiavelli say about ruling?", freq_analyzer)
        assert len(result) > 10
        keywords = ['prince', 'fear', 'love', 'power', 'rule']
        assert any(word in result.lower() for word in keywords)

    def test_winning_friends(self, foundation_model, freq_analyzer):
        """Can it discuss How to Win Friends?"""
        result = synthesize(foundation_model, "How do you win friends and influence people?", freq_analyzer)
        assert len(result) > 10
        keywords = ['friend', 'influence', 'people', 'interest', 'smile']
        assert any(word in result.lower() for word in keywords)

    def test_atomic_habits(self, foundation_model, freq_analyzer):
        """Can it discuss habit formation?"""
        result = synthesize(foundation_model, "How do you build good habits?", freq_analyzer)
        assert len(result) > 10
        keywords = ['habit', 'atomic', 'small', 'compound', 'system']
        assert any(word in result.lower() for word in keywords)

    def test_be_here_now(self, foundation_model, freq_analyzer):
        """Can it discuss presence from Be Here Now?"""
        result = synthesize(foundation_model, "What does it mean to be here now?", freq_analyzer)
        assert len(result) > 10
        keywords = ['present', 'now', 'moment', 'awareness', 'consciousness']
        assert any(word in result.lower() for word in keywords)

    def test_bodhisattva_vows(self, foundation_model, freq_analyzer):
        """Can it discuss Bodhisattva vows?"""
        result = synthesize(foundation_model, "What are the Bodhisattva vows?", freq_analyzer)
        assert len(result) > 10
        keywords = ['vow', 'beings', 'save', 'enlightenment', 'compassion']
        assert any(word in result.lower() for word in keywords)

    def test_strategy_principles(self, foundation_model, freq_analyzer):
        """Can it discuss general strategy?"""
        result = synthesize(foundation_model, "What makes a good strategy?", freq_analyzer)
        assert len(result) > 10
        keywords = ['strategy', 'plan', 'goal', 'advantage', 'position']
        assert any(word in result.lower() for word in keywords)

    def test_leadership_wisdom(self, foundation_model, freq_analyzer):
        """Can it discuss leadership?"""
        result = synthesize(foundation_model, "What makes a good leader?", freq_analyzer)
        assert len(result) > 10
        keywords = ['leader', 'wisdom', 'courage', 'vision', 'people']
        assert any(word in result.lower() for word in keywords)


class TestCompressionQuality:
    """Test that gravitational collapse maintains semantic integrity."""

    def test_high_resonance_patterns(self, foundation_model):
        """High-resonance protos should be common concepts."""
        # Find top 10 highest resonance protos
        sorted_protos = sorted(
            foundation_model.entries,
            key=lambda p: p.resonance_strength,
            reverse=True
        )

        top_10 = sorted_protos[:10]

        # All should have resonance > 1 (appeared multiple times)
        for proto in top_10:
            assert proto.resonance_strength > 1, \
                f"Top proto should have resonance > 1, got {proto.resonance_strength}"

    def test_compression_ratio(self, foundation_model):
        """Compression should be reasonable (gravitational collapse working)."""
        num_protos = len(foundation_model.entries)

        # Estimate original segments from metadata
        doc_sources = set()
        total_segments_estimate = 0

        for entry in foundation_model.entries:
            if 'source' in entry.metadata:
                doc_sources.add(entry.metadata['source'])
            total_segments_estimate += entry.resonance_strength

        # Should have protos from multiple documents
        assert len(doc_sources) >= 10, \
            f"Should have protos from many docs, got {len(doc_sources)}"

        # Compression ratio should be reasonable
        compression = total_segments_estimate / num_protos
        assert 1.1 < compression < 50.0, \
            f"Compression ratio {compression:.2f}x out of expected range"

    def test_modality_consistency(self, foundation_model):
        """All Foundation entries should be text modality."""
        for entry in foundation_model.entries:
            assert entry.modality == 'text', \
                f"Foundation should only have text, got {entry.modality}"

    def test_metadata_completeness(self, foundation_model):
        """All entries should have required metadata."""
        for i, entry in enumerate(foundation_model.entries[:100]):  # Check first 100
            # Text no longer stored in metadata (removed in Phase 7)
            # Check only deterministic metadata
            assert 'source' in entry.metadata, f"Entry {i} missing source metadata"

    def test_frequency_distribution(self, foundation_model):
        """Proto-identities should be well-distributed in frequency space."""
        positions = [entry.position for entry in foundation_model.entries]
        positions = np.array(positions)

        # Check distribution across each dimension
        for dim in range(3):
            dim_values = positions[:, dim]
            # Should use reasonable range of the dimension
            dim_range = np.max(dim_values) - np.min(dim_values)
            dim_size = [foundation_model.width, foundation_model.height, foundation_model.depth][dim]
            assert dim_range > dim_size * 0.3, \
                f"Dimension {dim} not well distributed: range {dim_range} < {dim_size * 0.3}"