#!/usr/bin/env python3
"""
Genesis - Text Learning via Frequency-based Morphism Application.

The morphisms ARE the frequency filters. Text is converted to frequency,
which determines morphism parameters, which project into voxel cloud space.

Usage:
    genesis.py test
    genesis.py discover --input TEXT_FILE [--output MODEL.pkl]
    genesis.py synthesize --model MODEL.pkl --query "text to find"
"""

import sys
import argparse
import logging
from src.cli import (
    cmd_test, cmd_discover, cmd_synthesize,
    cmd_train, cmd_chat, cmd_eval
)
from src.cli.validators import (
    add_secure_path_argument,
    add_bounded_float_argument,
    add_bounded_int_argument
)
from src.security import SecurityLevel

# Configure logging
logging.basicConfig(
    level=logging.WARNING,
    format='%(levelname)s: %(message)s'
)


def _setup_discover_parser(subparsers):
    """Set up discover command parser with input validation."""
    parser = subparsers.add_parser('discover', help='Discover patterns in text/image/audio')

    # Use secure path validation for input/output files
    add_secure_path_argument(
        parser, '--input',
        required=True,
        must_exist=True,
        must_be_file=True,
        help='Input file (text/image/audio)'
    )
    add_secure_path_argument(
        parser, '--output',
        default='/usr/lib/alembic/checkpoints/genesis/text_memory.pkl',
        allowed_extensions=['.pkl'],
        help='Output model path'
    )
    parser.add_argument('--dual-path', action='store_true',
                       help='Use dual-path convergence (Gen + Res)')

    # Modality arguments
    parser.add_argument('--modality', choices=['text', 'image', 'audio'], default='text',
                       help='Input modality type (default: text)')
    parser.add_argument('--phase-coherence-threshold', type=float, default=0.1,
                       help='Phase coherence threshold for cross-modal linking (default: 0.1)')
    parser.add_argument('--link-cross-modal', action='store_true',
                       help='Enable cross-modal phase-locking')

    # Collapse configuration arguments with validation
    add_bounded_float_argument(
        parser, '--collapse-harmonic-tolerance',
        default=0.05, min_value=0.0, max_value=1.0,
        help='Harmonic tolerance for gravitational collapse (default: 0.05)'
    )
    add_bounded_float_argument(
        parser, '--collapse-cosine-threshold',
        default=0.85, min_value=0.0, max_value=1.0,
        help='Cosine similarity threshold for merging (default: 0.85)'
    )
    add_bounded_int_argument(
        parser, '--collapse-octave-tolerance',
        default=0, min_value=0, max_value=5,
        help='Octave tolerance for frequency matching (default: 0)'
    )
    parser.add_argument('--enable-collapse', action='store_true', default=True,
                       help='Enable gravitational collapse (default: True)')
    parser.add_argument('--disable-collapse', dest='enable_collapse', action='store_false',
                       help='Disable gravitational collapse')

    # Performance/testing arguments
    parser.add_argument('--max-segments', type=int, default=None,
                       help='Maximum number of segments to process (for testing/QA)')
    return parser


def _setup_synthesize_parser(subparsers):
    """Set up synthesize command parser with input validation."""
    parser = subparsers.add_parser('synthesize', help='Synthesize from patterns')

    # Use secure path validation for model and output files
    add_secure_path_argument(
        parser, '--model',
        required=True,
        must_exist=True,
        must_be_file=True,
        allowed_extensions=['.pkl'],
        help='Model path'
    )
    parser.add_argument('--query', type=str, required=True, help='Query text')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    add_secure_path_argument(
        parser, '--output',
        allowed_extensions=['.pkl', '.npy'],
        help='Save synthesized proto to file'
    )
    parser.add_argument('--use-frequency', action='store_true',
                       help='Use frequency-based matching instead of spatial')

    # Synthesis configuration arguments
    parser.add_argument('--resonance-weighting', action='store_true', default=True,
                       help='Use resonance-weighted synthesis (default: True)')
    parser.add_argument('--no-resonance-weighting', dest='resonance_weighting',
                       action='store_false', help='Disable resonance weighting')
    parser.add_argument('--weight-function', choices=['linear', 'sqrt', 'log'],
                       default='linear', help='Weight function for resonance (default: linear)')
    parser.add_argument('--resonance-boost', type=float, default=2.0,
                       help='Multiplier for resonance importance (default: 2.0)')
    parser.add_argument('--distance-decay', type=float, default=0.5,
                       help='Distance weight factor, 0=ignore distance, 1=equal weight (default: 0.5)')
    return parser


def _setup_train_parser(subparsers):
    """Set up train command parser with input validation."""
    parser = subparsers.add_parser('train', help='Train on foundation data')

    # Use secure path validation
    add_secure_path_argument(
        parser, '--data',
        default='/usr/lib/alembic/data/datasets/curated/foundation/',
        must_exist=True,
        must_be_dir=True,
        help='Foundation data directory'
    )
    add_secure_path_argument(
        parser, '--output',
        default='./models/genesis_foundation.pkl',
        allowed_extensions=['.pkl'],
        help='Output model path'
    )
    parser.add_argument('--use-gpu', action='store_true',
                       help='Enable GPU acceleration')
    parser.add_argument('--checkpoint-interval', type=int, default=None,
                       help='Save checkpoint every N documents')
    parser.add_argument('--max-documents', type=int, default=None,
                       help='Limit documents (for testing)')

    # Collapse configuration
    parser.add_argument('--collapse-cosine-threshold', type=float, default=0.85,
                       help='Cosine similarity threshold for merging (default: 0.85)')
    parser.add_argument('--collapse-harmonic-tolerance', type=float, default=0.05,
                       help='Harmonic tolerance for gravitational collapse (default: 0.05)')
    parser.add_argument('--collapse-octave-tolerance', type=int, default=1,
                       help='Octave tolerance for frequency matching (default: 1)')
    parser.add_argument('--enable-collapse', action='store_true', default=True,
                       help='Enable gravitational collapse (default: True)')
    parser.add_argument('--disable-collapse', dest='enable_collapse', action='store_false',
                       help='Disable gravitational collapse')
    return parser


def _setup_chat_parser(subparsers):
    """Set up chat command parser with input validation."""
    parser = subparsers.add_parser('chat', help='Interactive conversation')

    # Use secure path validation
    add_secure_path_argument(
        parser, '--model',
        default='./models/genesis_foundation.pkl',
        must_exist=True,
        must_be_file=True,
        allowed_extensions=['.pkl'],
        help='Trained model path'
    )
    parser.add_argument('--stream', action='store_true',
                       help='Stream responses')
    parser.add_argument('--show-stats', action='store_true',
                       help='Show coherence/state after each response')
    parser.add_argument('--save-on-exit', action='store_true', default=True,
                       help='Save session on exit')
    return parser


def _setup_eval_parser(subparsers):
    """Set up eval command parser with input validation."""
    parser = subparsers.add_parser('eval', help='Run A/B evaluation tests')

    # Use secure path validation
    add_secure_path_argument(
        parser, '--model',
        default='./models/genesis_foundation.pkl',
        must_exist=True,
        must_be_file=True,
        allowed_extensions=['.pkl'],
        help='Trained model path'
    )
    add_secure_path_argument(
        parser, '--test-cases',
        default='./tests/test_cases.json',
        must_exist=True,
        must_be_file=True,
        allowed_extensions=['.json'],
        help='Test cases JSON file'
    )
    add_secure_path_argument(
        parser, '--output',
        default='./results/eval_results.json',
        allowed_extensions=['.json'],
        help='Results output path'
    )
    return parser


def main():
    """Main entry point for Genesis CLI."""
    parser = argparse.ArgumentParser(description="Genesis - Text Learning System")
    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Core Commands
    _setup_train_parser(subparsers)
    _setup_chat_parser(subparsers)
    _setup_eval_parser(subparsers)

    # Legacy Commands
    subparsers.add_parser('test', help='Test system components')
    _setup_discover_parser(subparsers)
    _setup_synthesize_parser(subparsers)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Core commands
    if args.command == 'train':
        return cmd_train(args)
    elif args.command == 'chat':
        return cmd_chat(args)
    elif args.command == 'eval':
        return cmd_eval(args)
    # Legacy commands
    elif args.command == 'test':
        return cmd_test(args)
    elif args.command == 'discover':
        return cmd_discover(args)
    elif args.command == 'synthesize':
        return cmd_synthesize(args)


if __name__ == "__main__":
    sys.exit(main())