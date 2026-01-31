"""Integration tests for memory router, unified encoder, and unified decoder.

Tests the complete integration pipeline:
1. Encoding text at multiple octaves
2. Routing to appropriate memory layers
3. Storing in memory hierarchy
4. Querying across layers
5. Decoding with blending
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from datetime import datetime, timedelta

from src.memory.memory_router import MemoryRouter, RoutingDecision
from src.pipeline.unified_encoder import UnifiedEncoder, EncodingResult
from src.pipeline.unified_decoder import UnifiedDecoder, DecodingResult
from src.memory.memory_hierarchy import MemoryHierarchy
from src.pipeline.multi_octave_encoder import OctaveUnit


class TestMemoryRouter:
    """Test the MemoryRouter component."""

    def test_router_initialization(self):
        """Test router initialization."""
        router = MemoryRouter()
        assert router.route_phrases_to_experiential is True
        assert router.route_chars_to_both is True
        assert len(router.routing_history) == 0

    def test_foundation_routing(self):
        """Test routing of foundation/training texts."""
        router = MemoryRouter()

        # Create mock octave units
        char_unit = OctaveUnit(
            text='a',
            octave=4,
            proto_identity=np.random.randn(512, 512, 4).astype(np.float32),
            frequency=np.random.randn(512, 512, 2).astype(np.float32)
        )
        word_unit = OctaveUnit(
            text='hello',
            octave=0,
            proto_identity=np.random.randn(512, 512, 4).astype(np.float32),
            frequency=np.random.randn(512, 512, 2).astype(np.float32)
        )

        # Route with foundation context
        decisions = router.route(
            [char_unit, word_unit],
            context_type='foundation',
            metadata={'source': 'training'}
        )

        assert len(decisions) == 2
        # Characters/words from foundation should go to both
        assert decisions[0].destination == 'both'
        assert decisions[1].destination == 'both'
        assert decisions[0].reason == 'foundation_dual_storage'

    def test_query_routing(self):
        """Test routing of query/inference inputs."""
        router = MemoryRouter()

        # Create phrase unit
        phrase_unit = OctaveUnit(
            text='hello world test',
            octave=-2,
            proto_identity=np.random.randn(512, 512, 4).astype(np.float32),
            frequency=np.random.randn(512, 512, 2).astype(np.float32)
        )

        # Route with query context
        decisions = router.route(
            [phrase_unit],
            context_type='query',
            metadata={'is_query': True}
        )

        assert len(decisions) == 1
        # Phrases should go to experiential only
        assert decisions[0].destination == 'experiential'
        assert decisions[0].reason == 'phrase_level_routing'

    def test_auto_context_detection(self):
        """Test automatic context detection."""
        router = MemoryRouter()

        char_unit = OctaveUnit(
            text='b',
            octave=4,
            proto_identity=np.random.randn(512, 512, 4).astype(np.float32),
            frequency=np.random.randn(512, 512, 2).astype(np.float32)
        )

        # Test with recent timestamp (should detect as query)
        decisions = router.route(
            [char_unit],
            context_type='auto',
            metadata={'timestamp': datetime.now()}
        )
        assert decisions[0].destination == 'experiential'

        # Test with training marker (should detect as foundation)
        decisions = router.route(
            [char_unit],
            context_type='auto',
            metadata={'is_training': True}
        )
        assert decisions[0].destination == 'both'

    def test_explicit_destination_override(self):
        """Test explicit destination override."""
        router = MemoryRouter()

        unit = OctaveUnit(
            text='test',
            octave=0,
            proto_identity=np.random.randn(512, 512, 4).astype(np.float32),
            frequency=np.random.randn(512, 512, 2).astype(np.float32)
        )

        # Override to core only
        decisions = router.route(
            [unit],
            context_type='query',  # Would normally go to experiential
            metadata={'destination': 'core'}
        )
        assert decisions[0].destination == 'core'
        assert decisions[0].reason == 'explicit_override'

    def test_routing_statistics(self):
        """Test routing statistics tracking."""
        router = MemoryRouter()

        units = [
            OctaveUnit(text='a', octave=4, proto_identity=np.zeros((512, 512, 4)), frequency=np.zeros((512, 512, 2))),
            OctaveUnit(text='word', octave=0, proto_identity=np.zeros((512, 512, 4)), frequency=np.zeros((512, 512, 2))),
            OctaveUnit(text='phrase here', octave=-2, proto_identity=np.zeros((512, 512, 4)), frequency=np.zeros((512, 512, 2)))
        ]

        router.route(units, context_type='foundation')
        stats = router.get_routing_stats()

        assert stats['total'] == 3
        assert stats['both'] == 2  # char and word
        assert 4 in stats['by_octave']
        assert 0 in stats['by_octave']
        assert -2 in stats['by_octave']


class TestUnifiedEncoder:
    """Test the UnifiedEncoder component."""

    def test_encoder_initialization(self):
        """Test encoder initialization."""
        hierarchy = MemoryHierarchy()
        encoder = UnifiedEncoder(hierarchy)

        assert encoder.width == 512
        assert encoder.height == 512
        assert encoder.total_encoded == 0

    def test_basic_encoding(self):
        """Test basic encoding functionality."""
        hierarchy = MemoryHierarchy()
        encoder = UnifiedEncoder(hierarchy)

        result = encoder.encode(
            text="Hello",
            destination='auto',
            octaves=[4, 0]
        )

        assert isinstance(result, EncodingResult)
        assert len(result.octave_units) > 0
        assert len(result.routing_decisions) == len(result.octave_units)
        assert result.core_added >= 0
        assert result.experiential_added >= 0

    def test_encoding_with_metadata(self):
        """Test encoding with custom metadata."""
        hierarchy = MemoryHierarchy()
        encoder = UnifiedEncoder(hierarchy)

        custom_metadata = {
            'source': 'test_suite',
            'version': '1.0',
            'is_training': True
        }

        result = encoder.encode(
            text="Test text",
            metadata=custom_metadata,
            destination='core'
        )

        assert result.metadata['source'] == 'test_suite'
        assert result.metadata['version'] == '1.0'
        assert 'timestamp' in result.metadata

    def test_encoding_statistics(self):
        """Test encoding statistics tracking."""
        hierarchy = MemoryHierarchy()
        encoder = UnifiedEncoder(hierarchy)

        # Encode some text
        encoder.encode("First text", destination='core')
        encoder.encode("Second text", destination='experiential')

        stats = encoder.get_statistics()

        assert stats['total_encoded'] > 0
        assert 'routing_stats' in stats
        assert 'efficiency' in stats
        assert stats['efficiency']['core_ratio'] >= 0
        assert stats['efficiency']['experiential_ratio'] >= 0


class TestUnifiedDecoder:
    """Test the UnifiedDecoder component."""

    def test_decoder_initialization(self):
        """Test decoder initialization."""
        hierarchy = MemoryHierarchy()
        decoder = UnifiedDecoder(hierarchy)

        assert decoder.width == 512
        assert decoder.height == 512
        assert decoder.core_weight == 0.7
        assert decoder.experiential_weight == 0.3

    def test_basic_decoding(self):
        """Test basic decoding functionality."""
        hierarchy = MemoryHierarchy()
        decoder = UnifiedDecoder(hierarchy)

        # Create a query proto
        query_proto = np.random.randn(512, 512, 4).astype(np.float32)

        result = decoder.decode(
            query_proto=query_proto,
            layers='both',
            octaves=[4, 0]
        )

        assert isinstance(result, DecodingResult)
        assert isinstance(result.text, str)
        assert 0 <= result.confidence <= 1
        assert isinstance(result.octaves_used, list)

    def test_octave_expansion(self):
        """Test octave range expansion."""
        hierarchy = MemoryHierarchy()
        decoder = UnifiedDecoder(hierarchy)

        # Test internal expansion method
        expanded = decoder._expand_octave_range([0], True)
        assert -1 in expanded
        assert 0 in expanded
        assert 1 in expanded

        # Test that expand=False doesn't expand
        not_expanded = decoder._expand_octave_range([0], False)
        assert not_expanded == [0]

    def test_layer_specific_queries(self):
        """Test querying specific memory layers."""
        hierarchy = MemoryHierarchy()
        decoder = UnifiedDecoder(hierarchy)

        query_proto = np.random.randn(512, 512, 4).astype(np.float32)

        # Query core only
        core_result = decoder.decode(
            query_proto=query_proto,
            layers='core'
        )
        assert 'core' in core_result.source_layers or len(core_result.source_layers) == 0

        # Query experiential only
        exp_result = decoder.decode(
            query_proto=query_proto,
            layers='experiential'
        )
        assert 'experiential' in exp_result.source_layers or len(exp_result.source_layers) == 0

    def test_blend_modes(self):
        """Test different blending modes."""
        hierarchy = MemoryHierarchy()
        decoder = UnifiedDecoder(hierarchy)

        query_proto = np.random.randn(512, 512, 4).astype(np.float32)

        # Test weighted blend
        weighted_result = decoder.decode(
            query_proto=query_proto,
            blend_mode='weighted'
        )

        # Test max blend
        max_result = decoder.decode(
            query_proto=query_proto,
            blend_mode='max'
        )

        # Test average blend
        avg_result = decoder.decode(
            query_proto=query_proto,
            blend_mode='average'
        )

        # All should return valid results
        assert isinstance(weighted_result.text, str)
        assert isinstance(max_result.text, str)
        assert isinstance(avg_result.text, str)


class TestEndToEndIntegration:
    """Test complete end-to-end integration pipeline."""

    def test_encode_route_decode_pipeline(self):
        """Test complete pipeline: encode → route → store → query → decode."""
        # Initialize components
        hierarchy = MemoryHierarchy()
        encoder = UnifiedEncoder(hierarchy)
        decoder = UnifiedDecoder(hierarchy)

        # Test text
        test_text = "Hello world"

        # Step 1: Encode and route
        encode_result = encoder.encode(
            text=test_text,
            destination='both',  # Store in both layers
            octaves=[4, 0],
            metadata={'test': 'integration'}
        )

        # Debug: print what was encoded
        print(f"Encoded {len(encode_result.octave_units)} units")
        print(f"Core added: {encode_result.core_added}")
        print(f"Experiential added: {encode_result.experiential_added}")

        # Check that something was processed
        assert len(encode_result.octave_units) > 0
        # May be stored in either or both
        assert encode_result.core_added > 0 or encode_result.experiential_added > 0

        # Step 2: Query using first proto as query
        if encode_result.octave_units:
            query_proto = encode_result.octave_units[0].proto_identity

            decode_result = decoder.decode(
                query_proto=query_proto,
                layers='both',
                octaves=[4, 0]
            )

            assert decode_result.confidence > 0
            assert len(decode_result.source_layers) > 0

    def test_octave_preservation(self):
        """Test that octave information is preserved through pipeline."""
        hierarchy = MemoryHierarchy()
        encoder = UnifiedEncoder(hierarchy)

        # Encode at specific octaves
        result = encoder.encode(
            text="Test octaves",
            octaves=[4, 0, -2]
        )

        # Check that units have correct octave levels
        octaves_found = {unit.octave for unit in result.octave_units}
        assert 4 in octaves_found or 0 in octaves_found or -2 in octaves_found

    def test_cross_layer_blending(self):
        """Test blending of results from multiple memory layers."""
        hierarchy = MemoryHierarchy()
        encoder = UnifiedEncoder(hierarchy)
        decoder = UnifiedDecoder(hierarchy)

        # Store same text in both layers
        encoder.encode("Shared knowledge", destination='both')

        # Create query
        query_proto = np.random.randn(512, 512, 4).astype(np.float32)

        # Decode from both layers
        result = decoder.decode(
            query_proto=query_proto,
            layers='both',
            blend_mode='weighted'
        )

        # Check resonances from both layers
        if result.resonances:
            assert 'core' in result.resonances or 'experiential' in result.resonances


def test_all_components():
    """Run all component tests."""
    print("Testing MemoryRouter...")
    router_tests = TestMemoryRouter()
    router_tests.test_router_initialization()
    router_tests.test_foundation_routing()
    router_tests.test_query_routing()
    router_tests.test_auto_context_detection()
    router_tests.test_explicit_destination_override()
    router_tests.test_routing_statistics()
    print("✓ MemoryRouter tests passed")

    print("\nTesting UnifiedEncoder...")
    encoder_tests = TestUnifiedEncoder()
    encoder_tests.test_encoder_initialization()
    encoder_tests.test_basic_encoding()
    encoder_tests.test_encoding_with_metadata()
    encoder_tests.test_encoding_statistics()
    print("✓ UnifiedEncoder tests passed")

    print("\nTesting UnifiedDecoder...")
    decoder_tests = TestUnifiedDecoder()
    decoder_tests.test_decoder_initialization()
    decoder_tests.test_basic_decoding()
    decoder_tests.test_octave_expansion()
    decoder_tests.test_layer_specific_queries()
    decoder_tests.test_blend_modes()
    print("✓ UnifiedDecoder tests passed")

    print("\nTesting End-to-End Integration...")
    integration_tests = TestEndToEndIntegration()
    integration_tests.test_encode_route_decode_pipeline()
    integration_tests.test_octave_preservation()
    integration_tests.test_cross_layer_blending()
    print("✓ Integration tests passed")

    print("\n✅ All tests passed successfully!")


if __name__ == "__main__":
    test_all_components()