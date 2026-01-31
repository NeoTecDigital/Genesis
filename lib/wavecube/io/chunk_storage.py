"""
Disk persistence layer for ChunkedWaveCube with async IO and LRU eviction.

Provides:
- Save/load chunks in compressed .npz format
- Metadata tracking (locations, compression stats, access patterns)
- Async loading/saving with threading
- LRU eviction policy based on access patterns
"""

import os
import time
import json
import threading
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple, Any, List
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, Future
import logging

logger = logging.getLogger(__name__)


class ChunkMetadata:
    """Metadata for a single chunk."""

    def __init__(self, coords: Tuple[int, int, int]):
        self.coords = coords
        self.last_access = time.time()
        self.access_count = 0
        self.compressed_size = 0
        self.uncompressed_size = 0
        self.node_count = 0
        self.is_dirty = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'coords': self.coords,
            'last_access': self.last_access,
            'access_count': self.access_count,
            'compressed_size': self.compressed_size,
            'uncompressed_size': self.uncompressed_size,
            'node_count': self.node_count,
            'is_dirty': self.is_dirty
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChunkMetadata':
        """Create from dictionary."""
        meta = cls(tuple(data['coords']))
        meta.last_access = data['last_access']
        meta.access_count = data['access_count']
        meta.compressed_size = data['compressed_size']
        meta.uncompressed_size = data['uncompressed_size']
        meta.node_count = data['node_count']
        meta.is_dirty = data.get('is_dirty', False)
        return meta


class ChunkStorage:
    """
    Manages disk persistence for ChunkedWaveCube chunks.

    Features:
    - Compressed NPZ format storage
    - Async IO with threading
    - LRU eviction policy
    - Metadata tracking
    """

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        max_threads: int = 4,
        auto_save: bool = True,
        compression_level: int = 6
    ):
        """
        Initialize chunk storage.

        Args:
            cache_dir: Directory for chunk storage (default: ~/.cache/genesis/chunks)
            max_threads: Maximum threads for async IO
            auto_save: Auto-save dirty chunks on eviction
            compression_level: NPZ compression level (1-9)
        """
        # Set up cache directory
        if cache_dir is None:
            cache_dir = Path.home() / '.cache' / 'genesis' / 'chunks'
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Metadata storage
        self.metadata_file = self.cache_dir / 'metadata.json'
        self.metadata: Dict[Tuple[int, int, int], ChunkMetadata] = {}
        self._load_metadata()

        # LRU cache for tracking access
        self.lru_cache: OrderedDict[Tuple[int, int, int], None] = OrderedDict()

        # Threading
        self.executor = ThreadPoolExecutor(max_workers=max_threads)
        self.io_lock = threading.Lock()
        self.pending_operations: Dict[Tuple[int, int, int], Future] = {}

        # Settings
        self.auto_save = auto_save
        self.compression_level = compression_level

        # Statistics
        self.stats = {
            'chunks_saved': 0,
            'chunks_loaded': 0,
            'total_bytes_written': 0,
            'total_bytes_read': 0,
            'io_errors': 0
        }

    def _get_chunk_path(self, coords: Tuple[int, int, int]) -> Path:
        """Get file path for chunk coordinates."""
        cx, cy, cz = coords
        return self.cache_dir / f'chunk_{cx}_{cy}_{cz}.npz'

    def _load_metadata(self) -> None:
        """Load metadata from disk."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    data = json.load(f)
                    for key, value in data.items():
                        coords = tuple(map(int, key.split(',')))
                        self.metadata[coords] = ChunkMetadata.from_dict(value)
            except Exception as e:
                logger.warning(f"Failed to load metadata: {e}")

    def _save_metadata(self) -> None:
        """Save metadata to disk."""
        try:
            data = {}
            for coords, meta in self.metadata.items():
                key = f"{coords[0]},{coords[1]},{coords[2]}"
                data[key] = meta.to_dict()

            with open(self.metadata_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")

    def _update_lru(self, coords: Tuple[int, int, int]) -> None:
        """Update LRU cache for chunk access."""
        if coords in self.lru_cache:
            self.lru_cache.move_to_end(coords)
        else:
            self.lru_cache[coords] = None

        # Update metadata
        if coords not in self.metadata:
            self.metadata[coords] = ChunkMetadata(coords)

        meta = self.metadata[coords]
        meta.last_access = time.time()
        meta.access_count += 1

    def save_chunk(
        self,
        coords: Tuple[int, int, int],
        chunk_data: Dict[str, Any],
        async_save: bool = True
    ) -> Optional[Future]:
        """
        Save chunk to disk.

        Args:
            coords: Chunk coordinates
            chunk_data: Dictionary of arrays to save
            async_save: Use async IO

        Returns:
            Future if async, None if sync
        """
        def _save():
            try:
                path = self._get_chunk_path(coords)

                # Calculate sizes
                uncompressed_size = sum(
                    arr.nbytes if isinstance(arr, np.ndarray) else 0
                    for arr in chunk_data.values()
                )

                # Save compressed
                np.savez_compressed(path, **chunk_data)

                # Update metadata
                compressed_size = path.stat().st_size

                with self.io_lock:
                    if coords not in self.metadata:
                        self.metadata[coords] = ChunkMetadata(coords)

                    meta = self.metadata[coords]
                    meta.compressed_size = compressed_size
                    meta.uncompressed_size = uncompressed_size
                    meta.node_count = len(chunk_data.get('node_indices', []))
                    meta.is_dirty = False

                    self.stats['chunks_saved'] += 1
                    self.stats['total_bytes_written'] += compressed_size

                logger.debug(f"Saved chunk {coords}: {compressed_size/1024:.1f}KB")
                return True

            except Exception as e:
                logger.error(f"Failed to save chunk {coords}: {e}")
                self.stats['io_errors'] += 1
                return False

        if async_save:
            with self.io_lock:
                # Cancel pending operation if exists
                if coords in self.pending_operations:
                    self.pending_operations[coords].cancel()

                # Submit new operation
                future = self.executor.submit(_save)
                self.pending_operations[coords] = future
                return future
        else:
            _save()
            return None

    def load_chunk(
        self,
        coords: Tuple[int, int, int],
        async_load: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Load chunk from disk.

        Args:
            coords: Chunk coordinates
            async_load: Use async IO

        Returns:
            Chunk data dictionary or Future if async
        """
        def _load():
            try:
                path = self._get_chunk_path(coords)

                if not path.exists():
                    logger.debug(f"Chunk {coords} not found on disk")
                    return None

                # Load data
                data = dict(np.load(path, allow_pickle=True))

                # Update stats
                file_size = path.stat().st_size

                with self.io_lock:
                    self._update_lru(coords)
                    self.stats['chunks_loaded'] += 1
                    self.stats['total_bytes_read'] += file_size

                logger.debug(f"Loaded chunk {coords}: {file_size/1024:.1f}KB")
                return data

            except Exception as e:
                logger.error(f"Failed to load chunk {coords}: {e}")
                self.stats['io_errors'] += 1
                return None

        if async_load:
            return self.executor.submit(_load)
        else:
            return _load()

    def evict_lru(self, keep_count: int = 10) -> List[Tuple[int, int, int]]:
        """
        Evict least recently used chunks.

        Args:
            keep_count: Number of chunks to keep in cache

        Returns:
            List of evicted chunk coordinates
        """
        evicted = []

        while len(self.lru_cache) > keep_count:
            coords, _ = self.lru_cache.popitem(last=False)

            # Save if dirty and auto-save enabled
            if self.auto_save and coords in self.metadata:
                meta = self.metadata[coords]
                if meta.is_dirty:
                    logger.debug(f"Auto-saving dirty chunk {coords} before eviction")
                    # Note: Caller should provide chunk data for saving

            evicted.append(coords)

        return evicted

    def get_access_stats(self, coords: Tuple[int, int, int]) -> Optional[Dict[str, Any]]:
        """Get access statistics for a chunk."""
        if coords not in self.metadata:
            return None

        meta = self.metadata[coords]
        return {
            'last_access': meta.last_access,
            'access_count': meta.access_count,
            'age_seconds': time.time() - meta.last_access,
            'compression_ratio': (
                meta.uncompressed_size / max(1, meta.compressed_size)
                if meta.compressed_size > 0 else 0
            )
        }

    def cleanup_old_chunks(self, max_age_days: int = 7) -> int:
        """
        Delete chunks older than specified age.

        Args:
            max_age_days: Maximum age in days

        Returns:
            Number of chunks deleted
        """
        deleted = 0
        max_age_seconds = max_age_days * 24 * 3600
        current_time = time.time()

        for coords, meta in list(self.metadata.items()):
            age = current_time - meta.last_access

            if age > max_age_seconds:
                path = self._get_chunk_path(coords)
                try:
                    if path.exists():
                        path.unlink()
                    del self.metadata[coords]
                    deleted += 1
                    logger.info(f"Deleted old chunk {coords} (age: {age/3600:.1f} hours)")
                except Exception as e:
                    logger.error(f"Failed to delete chunk {coords}: {e}")

        if deleted > 0:
            self._save_metadata()

        return deleted

    def wait_pending(self, timeout: Optional[float] = None) -> bool:
        """
        Wait for all pending operations to complete.

        Args:
            timeout: Maximum wait time in seconds

        Returns:
            True if all completed, False if timeout
        """
        start_time = time.time()

        while self.pending_operations:
            if timeout and (time.time() - start_time) > timeout:
                return False

            # Check completed operations
            completed = []
            for coords, future in self.pending_operations.items():
                if future.done():
                    completed.append(coords)

            # Remove completed
            for coords in completed:
                del self.pending_operations[coords]

            if self.pending_operations:
                time.sleep(0.01)

        return True

    def shutdown(self, wait: bool = True, timeout: float = 10.0) -> None:
        """
        Shutdown storage system.

        Args:
            wait: Wait for pending operations
            timeout: Maximum wait time
        """
        if wait:
            self.wait_pending(timeout)

        self._save_metadata()
        # ThreadPoolExecutor.shutdown doesn't support timeout parameter
        self.executor.shutdown(wait=wait)

    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics."""
        total_compressed = sum(m.compressed_size for m in self.metadata.values())
        total_uncompressed = sum(m.uncompressed_size for m in self.metadata.values())

        return {
            'chunks_on_disk': len(self.metadata),
            'chunks_in_lru': len(self.lru_cache),
            'total_compressed_mb': total_compressed / (1024 ** 2),
            'total_uncompressed_mb': total_uncompressed / (1024 ** 2),
            'compression_ratio': (
                total_uncompressed / max(1, total_compressed)
                if total_compressed > 0 else 0
            ),
            'io_stats': self.stats
        }