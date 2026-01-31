"""
Migration utility for converting existing proto-identities to chunked storage.

Provides utilities to:
- Migrate from VoxelCloud to ChunkedWaveCube
- Convert legacy proto-identities
- Validate migration integrity
- Rollback if needed
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import time
from pathlib import Path
from typing import Dict, Any, Optional
import pickle
import json

from src.memory.voxel_cloud import VoxelCloud
from src.pipeline.encoder_with_chunking import ChunkedMultiOctaveEncoder
from lib.wavecube.core.chunked_matrix import ChunkedWaveCube
from lib.wavecube.spatial.coordinates import QuaternionicCoord, Modality


class MigrationManager:
    """Manages migration from VoxelCloud to ChunkedWaveCube."""

    def __init__(
        self,
        backup_dir: Optional[Path] = None,
        cache_dir: Optional[Path] = None
    ):
        """
        Initialize migration manager.

        Args:
            backup_dir: Directory for backups
            cache_dir: Directory for chunked storage
        """
        if backup_dir is None:
            backup_dir = Path.home() / '.cache' / 'genesis' / 'migration_backup'
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(parents=True, exist_ok=True)

        if cache_dir is None:
            cache_dir = Path.home() / '.cache' / 'genesis' / 'chunks'
        self.cache_dir = cache_dir

        self.migration_log = []

    def backup_voxel_cloud(self, voxel_cloud: VoxelCloud) -> Path:
        """
        Create backup of VoxelCloud before migration.

        Args:
            voxel_cloud: VoxelCloud to backup

        Returns:
            Path to backup file
        """
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        backup_file = self.backup_dir / f'voxel_backup_{timestamp}.pkl'

        print(f"Creating backup at {backup_file}...")

        # Serialize VoxelCloud
        backup_data = {
            'entries': [],
            'metadata': {
                'width': voxel_cloud.width,
                'height': voxel_cloud.height,
                'depth': voxel_cloud.depth,
                'timestamp': timestamp,
                'num_entries': len(voxel_cloud.entries)
            }
        }

        for entry in voxel_cloud.entries:
            entry_data = {
                'proto_identity': entry.proto_identity,
                'frequency': entry.frequency,
                'metadata': entry.metadata,
                'position': entry.position,
                'wavecube_coords': entry.wavecube_coords
            }
            backup_data['entries'].append(entry_data)

        # Save backup
        with open(backup_file, 'wb') as f:
            pickle.dump(backup_data, f)

        print(f"Backup complete: {len(voxel_cloud.entries)} entries saved")
        self.migration_log.append({
            'action': 'backup',
            'timestamp': timestamp,
            'file': str(backup_file),
            'entries': len(voxel_cloud.entries)
        })

        return backup_file

    def migrate_voxel_to_chunks(
        self,
        voxel_cloud: VoxelCloud,
        create_backup: bool = True,
        batch_size: int = 100,
        compression: str = 'gaussian'
    ) -> ChunkedWaveCube:
        """
        Migrate VoxelCloud to ChunkedWaveCube.

        Args:
            voxel_cloud: Source VoxelCloud
            create_backup: Whether to create backup first
            batch_size: Number of entries per batch
            compression: Compression method

        Returns:
            New ChunkedWaveCube with migrated data
        """
        print("=" * 70)
        print("MIGRATION: VoxelCloud → ChunkedWaveCube")
        print("=" * 70)

        # Create backup if requested
        if create_backup:
            backup_file = self.backup_voxel_cloud(voxel_cloud)
            print(f"Backup created: {backup_file}")

        # Create ChunkedWaveCube
        cube = ChunkedWaveCube(
            chunk_size=(16, 16, 16),
            resolution=512,
            channels=4,
            cache_radius=2,
            compression=compression
        )

        # Create encoder for migration
        dummy_carrier = np.zeros((512, 512, 2), dtype=np.float32)
        encoder = ChunkedMultiOctaveEncoder(
            carrier=dummy_carrier,
            width=512,
            height=512,
            chunk_size=(16, 16, 16),
            cache_dir=self.cache_dir,
            enable_persistence=True
        )

        # Migration statistics
        stats = {
            'total_entries': len(voxel_cloud.entries),
            'migrated': 0,
            'failed': 0,
            'start_time': time.time(),
            'memory_before': 0,
            'memory_after': 0
        }

        # Get initial memory usage
        if hasattr(voxel_cloud, 'get_memory_usage'):
            stats['memory_before'] = voxel_cloud.get_memory_usage()['total_mb']

        print(f"\nMigrating {stats['total_entries']} entries...")
        print(f"Initial memory: {stats['memory_before']:.2f} MB")

        # Migrate in batches
        for batch_start in range(0, len(voxel_cloud.entries), batch_size):
            batch_end = min(batch_start + batch_size, len(voxel_cloud.entries))
            batch_entries = voxel_cloud.entries[batch_start:batch_end]

            for i, entry in enumerate(batch_entries):
                global_idx = batch_start + i

                try:
                    # Determine position based on octave
                    octave = entry.metadata.get('octave', 0)
                    position = self._calculate_position(global_idx, octave)

                    # Get modality
                    modality_str = entry.metadata.get('modality', 'TEXT')
                    if modality_str in Modality.__members__:
                        modality = Modality[modality_str]
                    else:
                        modality = Modality.TEXT

                    # Create quaternionic coordinate
                    coord = QuaternionicCoord.from_modality(
                        position[0], position[1], position[2], modality
                    )

                    # Store in cube
                    cube.set_node(
                        coord.x, coord.y, coord.z,
                        entry.proto_identity,
                        metadata=entry.metadata
                    )

                    # Update VoxelCloud with reference
                    voxel_cloud.set_wavecube_reference(global_idx, coord.to_tuple())

                    stats['migrated'] += 1

                except Exception as e:
                    print(f"  Failed to migrate entry {global_idx}: {e}")
                    stats['failed'] += 1

            # Progress update
            progress = (batch_end / stats['total_entries']) * 100
            print(f"  Progress: {batch_end}/{stats['total_entries']} "
                  f"({progress:.1f}%), "
                  f"Chunks: {cube.stats['chunks_total']}, "
                  f"Compressed: {cube.stats['chunks_compressed']}")

            # Offload inactive chunks periodically
            if batch_end % (batch_size * 5) == 0:
                cube.offload_all_inactive()

        # Final statistics
        stats['end_time'] = time.time()
        stats['duration'] = stats['end_time'] - stats['start_time']
        stats['memory_after'] = cube.get_memory_usage()['total_mb']

        print("\n" + "=" * 70)
        print("MIGRATION COMPLETE")
        print("=" * 70)
        print(f"Migrated: {stats['migrated']}/{stats['total_entries']} entries")
        print(f"Failed: {stats['failed']} entries")
        print(f"Duration: {stats['duration']:.2f} seconds")
        print(f"Memory reduction: {stats['memory_before']:.2f} MB → "
              f"{stats['memory_after']:.2f} MB "
              f"({stats['memory_before']/max(stats['memory_after'], 0.001):.1f}x compression)")
        print(f"Chunks created: {cube.stats['chunks_total']}")
        print(f"Chunks compressed: {cube.stats['chunks_compressed']}")

        # Log migration
        self.migration_log.append({
            'action': 'migrate',
            'timestamp': time.strftime('%Y%m%d_%H%M%S'),
            'stats': stats
        })

        # Save migration log
        self._save_migration_log()

        # Shutdown encoder to persist chunks
        encoder.shutdown()

        return cube

    def validate_migration(
        self,
        voxel_cloud: VoxelCloud,
        cube: ChunkedWaveCube,
        sample_size: int = 100
    ) -> Dict[str, Any]:
        """
        Validate migration integrity.

        Args:
            voxel_cloud: Original VoxelCloud
            cube: Migrated ChunkedWaveCube
            sample_size: Number of entries to sample

        Returns:
            Validation results
        """
        print("\n" + "=" * 70)
        print("VALIDATION")
        print("=" * 70)

        results = {
            'total_entries': len(voxel_cloud.entries),
            'samples_tested': 0,
            'matches': 0,
            'mismatches': 0,
            'missing': 0,
            'avg_similarity': 0.0
        }

        # Sample random entries
        indices = np.random.choice(
            len(voxel_cloud.entries),
            min(sample_size, len(voxel_cloud.entries)),
            replace=False
        )

        similarities = []

        for idx in indices:
            entry = voxel_cloud.entries[idx]
            reference = voxel_cloud.get_wavecube_reference(idx)

            if reference is None:
                results['missing'] += 1
                continue

            # Retrieve from cube
            x, y, z, w = reference
            retrieved = cube.get_node(int(x), int(y), int(z))

            if retrieved is None:
                results['missing'] += 1
                continue

            # Calculate similarity
            similarity = self._calculate_similarity(
                entry.proto_identity,
                retrieved
            )

            similarities.append(similarity)
            results['samples_tested'] += 1

            if similarity > 0.95:
                results['matches'] += 1
            else:
                results['mismatches'] += 1

            if results['samples_tested'] <= 5:
                print(f"  Sample {idx}: similarity = {similarity:.4f}")

        if similarities:
            results['avg_similarity'] = np.mean(similarities)
            results['min_similarity'] = np.min(similarities)
            results['max_similarity'] = np.max(similarities)

        print(f"\nValidation Results:")
        print(f"  Samples tested: {results['samples_tested']}")
        print(f"  Matches (>0.95): {results['matches']}")
        print(f"  Mismatches: {results['mismatches']}")
        print(f"  Missing: {results['missing']}")
        print(f"  Average similarity: {results['avg_similarity']:.4f}")

        if results['avg_similarity'] > 0.95:
            print("✓ Validation PASSED")
        else:
            print("⚠ Validation concerns - similarity below threshold")

        return results

    def rollback_migration(self, backup_file: Path) -> VoxelCloud:
        """
        Rollback migration from backup.

        Args:
            backup_file: Path to backup file

        Returns:
            Restored VoxelCloud
        """
        print(f"Rolling back from {backup_file}...")

        with open(backup_file, 'rb') as f:
            backup_data = pickle.load(f)

        # Create new VoxelCloud
        metadata = backup_data['metadata']
        voxel = VoxelCloud(
            width=metadata['width'],
            height=metadata['height'],
            depth=metadata['depth']
        )

        # Restore entries
        for entry_data in backup_data['entries']:
            voxel.add(
                entry_data['proto_identity'],
                entry_data['frequency'],
                entry_data['metadata']
            )

            # Restore wavecube reference if exists
            if entry_data.get('wavecube_coords'):
                idx = len(voxel.entries) - 1
                voxel.set_wavecube_reference(idx, entry_data['wavecube_coords'])

        print(f"Rollback complete: {len(voxel.entries)} entries restored")

        self.migration_log.append({
            'action': 'rollback',
            'timestamp': time.strftime('%Y%m%d_%H%M%S'),
            'backup_file': str(backup_file),
            'entries_restored': len(voxel.entries)
        })

        return voxel

    def _calculate_position(self, index: int, octave: int) -> Tuple[int, int, int]:
        """Calculate spatial position for entry."""
        # Distribute in grid pattern
        grid_size = 20  # 20x20 grid per octave level

        x = (index % grid_size) * 8
        y = ((index // grid_size) % grid_size) * 8
        z = (octave + 4) * 10  # Separate octaves by Z level

        return (x, y, z)

    def _calculate_similarity(self, proto1: np.ndarray, proto2: np.ndarray) -> float:
        """Calculate similarity between protos."""
        p1_flat = proto1.flatten()
        p2_flat = proto2.flatten()

        # Handle shape mismatch
        if p1_flat.shape != p2_flat.shape:
            min_len = min(len(p1_flat), len(p2_flat))
            p1_flat = p1_flat[:min_len]
            p2_flat = p2_flat[:min_len]

        p1_norm = p1_flat / (np.linalg.norm(p1_flat) + 1e-8)
        p2_norm = p2_flat / (np.linalg.norm(p2_flat) + 1e-8)

        return float(np.dot(p1_norm, p2_norm))

    def _save_migration_log(self):
        """Save migration log to file."""
        log_file = self.backup_dir / 'migration_log.json'

        with open(log_file, 'w') as f:
            json.dump(self.migration_log, f, indent=2)


def main():
    """Run migration utility."""
    print("Genesis Migration Utility")
    print("=" * 70)

    # Create test VoxelCloud with sample data
    print("\nCreating test VoxelCloud...")
    voxel = VoxelCloud(width=512, height=512, depth=128)

    # Add sample entries
    num_entries = 100
    for i in range(num_entries):
        proto = np.random.randn(512, 512, 4).astype(np.float32) * 0.1
        freq = np.random.randn(512, 512, 2).astype(np.float32) * 0.1

        metadata = {
            'octave': np.random.choice([4, 0, -2]),
            'modality': np.random.choice(['TEXT', 'AUDIO', 'IMAGE']),
            'index': i
        }

        voxel.add(proto, freq, metadata)

    print(f"Created VoxelCloud with {len(voxel.entries)} entries")
    print(f"Memory usage: {voxel.get_memory_usage()['total_mb']:.2f} MB")

    # Run migration
    manager = MigrationManager()

    # Migrate to chunks
    cube = manager.migrate_voxel_to_chunks(
        voxel,
        create_backup=True,
        batch_size=20,
        compression='gaussian'
    )

    # Validate migration
    validation = manager.validate_migration(voxel, cube, sample_size=20)

    # Show final statistics
    print("\n" + "=" * 70)
    print("MIGRATION SUMMARY")
    print("=" * 70)
    print(f"Original memory: {voxel.get_memory_usage()['total_mb']:.2f} MB")
    print(f"Chunked memory: {cube.get_memory_usage()['total_mb']:.2f} MB")
    print(f"Compression ratio: {voxel.get_memory_usage()['total_mb'] / max(cube.get_memory_usage()['total_mb'], 0.001):.1f}x")
    print(f"Validation similarity: {validation['avg_similarity']:.4f}")

    if validation['avg_similarity'] > 0.95:
        print("\n✓ Migration successful!")
    else:
        print("\n⚠ Migration completed with warnings")

    return 0


if __name__ == "__main__":
    exit(main())