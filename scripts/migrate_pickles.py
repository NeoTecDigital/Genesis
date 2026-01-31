#!/usr/bin/env python3
"""
Migration script to add HMAC signatures to existing pickle files.

Usage:
    python scripts/migrate_pickles.py [directory_or_file]

This script will:
1. Find all .pkl files in the specified directory (or process single file)
2. Load each file using backward-compatible mode
3. Re-save with HMAC signature
4. Create .sig companion files
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import List, Tuple

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.security import migrate_pickle_file

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def find_pickle_files(path: Path) -> List[Path]:
    """Find all pickle files in a directory or return single file."""
    if path.is_file():
        if path.suffix == '.pkl':
            return [path]
        else:
            logger.warning(f"Not a pickle file: {path}")
            return []

    if path.is_dir():
        return list(path.rglob('*.pkl'))

    logger.error(f"Path does not exist: {path}")
    return []


def migrate_pickles(
    path: Path,
    force: bool = False,
    dry_run: bool = False
) -> Tuple[int, int]:
    """
    Migrate pickle files to include HMAC signatures.

    Args:
        path: Directory or file path
        force: Force re-signing even if signature exists
        dry_run: Show what would be done without doing it

    Returns:
        Tuple of (success_count, failure_count)
    """
    pickle_files = find_pickle_files(path)

    if not pickle_files:
        logger.warning("No pickle files found")
        return 0, 0

    logger.info(f"Found {len(pickle_files)} pickle files to process")

    success_count = 0
    failure_count = 0

    for pkl_file in pickle_files:
        sig_file = pkl_file.with_suffix(pkl_file.suffix + '.sig')

        # Check if already has signature
        if sig_file.exists() and not force:
            logger.info(f"Skipping (has signature): {pkl_file}")
            success_count += 1
            continue

        if dry_run:
            action = "Would migrate" if not sig_file.exists() else "Would re-sign"
            logger.info(f"{action}: {pkl_file}")
            continue

        # Perform migration
        logger.info(f"Migrating: {pkl_file}")

        try:
            if migrate_pickle_file(pkl_file, force=force):
                logger.info(f"Successfully migrated: {pkl_file}")
                success_count += 1
            else:
                logger.error(f"Failed to migrate: {pkl_file}")
                failure_count += 1
        except Exception as e:
            logger.error(f"Error migrating {pkl_file}: {e}")
            failure_count += 1

    return success_count, failure_count


def main():
    """Main entry point for migration script."""
    parser = argparse.ArgumentParser(
        description="Migrate existing pickle files to use HMAC signatures"
    )
    parser.add_argument(
        'path',
        type=Path,
        nargs='?',
        default='.',
        help='Directory or file to process (default: current directory)'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force re-signing even if signature exists'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without doing it'
    )
    parser.add_argument(
        '--set-key',
        type=str,
        help='Set GENESIS_HMAC_KEY environment variable'
    )

    args = parser.parse_args()

    # Set HMAC key if provided
    if args.set_key:
        os.environ['GENESIS_HMAC_KEY'] = args.set_key
        logger.info("HMAC key set for this session")

    # Check if HMAC key is available
    if 'GENESIS_HMAC_KEY' not in os.environ:
        logger.warning(
            "GENESIS_HMAC_KEY environment variable not set.\n"
            "Generating a random key for this session.\n"
            "To use a persistent key, set: export GENESIS_HMAC_KEY='your-secret-key'"
        )
        import secrets
        os.environ['GENESIS_HMAC_KEY'] = secrets.token_hex(32)

    # Perform migration
    success, failure = migrate_pickles(
        args.path,
        force=args.force,
        dry_run=args.dry_run
    )

    # Report results
    total = success + failure
    if total > 0:
        logger.info(f"\nMigration complete: {success}/{total} successful")
        if failure > 0:
            logger.error(f"{failure} files failed to migrate")
            return 1
    else:
        logger.info("No files were migrated")

    return 0


if __name__ == '__main__':
    sys.exit(main())