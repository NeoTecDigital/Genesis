"""
Spatial indexing for O(k) queries in ChunkedWaveCube.

Provides:
- Active chunk caching strategy
- Spatial hash grid for O(1) chunk lookups
- K-nearest neighbor queries with chunk-based partitioning
- Distance-based chunk prioritization
- Query result caching with invalidation
"""

import heapq
import numpy as np
from typing import Dict, List, Tuple, Set, Optional, Any, Callable
from dataclasses import dataclass, field
from collections import defaultdict
import time


@dataclass
class QueryResult:
    """Result from a spatial query."""
    position: Tuple[float, float, float]
    data: Any
    distance: float
    metadata: Optional[Dict[str, Any]] = None

    def __lt__(self, other):
        """Less than comparison for heap operations."""
        return self.distance < other.distance


@dataclass
class ChunkInfo:
    """Information about a chunk for spatial queries."""
    coords: Tuple[int, int, int]
    bounds_min: Tuple[float, float, float]
    bounds_max: Tuple[float, float, float]
    node_count: int
    is_loaded: bool
    last_query_time: float = 0.0


class SpatialHashGrid:
    """
    Hash grid for O(1) chunk lookups by spatial region.

    Maps spatial regions to chunk coordinates for fast lookups.
    """

    def __init__(self, grid_resolution: int = 64):
        """
        Initialize spatial hash grid.

        Args:
            grid_resolution: Resolution of hash grid
        """
        self.grid_resolution = grid_resolution
        self.grid: Dict[Tuple[int, int, int], Set[Tuple[int, int, int]]] = defaultdict(set)
        self.chunk_info: Dict[Tuple[int, int, int], ChunkInfo] = {}

    def _hash_position(self, x: float, y: float, z: float) -> Tuple[int, int, int]:
        """Hash a position to grid cell."""
        gx = int(x / self.grid_resolution)
        gy = int(y / self.grid_resolution)
        gz = int(z / self.grid_resolution)
        return (gx, gy, gz)

    def add_chunk(
        self,
        chunk_coords: Tuple[int, int, int],
        chunk_size: Tuple[int, int, int],
        node_count: int = 0,
        is_loaded: bool = False
    ) -> None:
        """
        Add chunk to spatial index.

        Args:
            chunk_coords: Chunk coordinates
            chunk_size: Size of chunk
            node_count: Number of nodes in chunk
            is_loaded: Whether chunk is currently loaded
        """
        cx, cy, cz = chunk_coords
        sx, sy, sz = chunk_size

        # Calculate bounds
        bounds_min = (cx * sx, cy * sy, cz * sz)
        bounds_max = ((cx + 1) * sx, (cy + 1) * sy, (cz + 1) * sz)

        # Store chunk info
        self.chunk_info[chunk_coords] = ChunkInfo(
            coords=chunk_coords,
            bounds_min=bounds_min,
            bounds_max=bounds_max,
            node_count=node_count,
            is_loaded=is_loaded
        )

        # Add to grid cells
        min_gx = int(bounds_min[0] / self.grid_resolution)
        max_gx = int(bounds_max[0] / self.grid_resolution)
        min_gy = int(bounds_min[1] / self.grid_resolution)
        max_gy = int(bounds_max[1] / self.grid_resolution)
        min_gz = int(bounds_min[2] / self.grid_resolution)
        max_gz = int(bounds_max[2] / self.grid_resolution)

        for gx in range(min_gx, max_gx + 1):
            for gy in range(min_gy, max_gy + 1):
                for gz in range(min_gz, max_gz + 1):
                    self.grid[(gx, gy, gz)].add(chunk_coords)

    def remove_chunk(self, chunk_coords: Tuple[int, int, int]) -> None:
        """Remove chunk from spatial index."""
        if chunk_coords not in self.chunk_info:
            return

        # Remove from all grid cells
        for chunks in self.grid.values():
            chunks.discard(chunk_coords)

        # Remove chunk info
        del self.chunk_info[chunk_coords]

    def find_chunks_near(
        self,
        position: Tuple[float, float, float],
        radius: float
    ) -> List[Tuple[int, int, int]]:
        """
        Find chunks within radius of position.

        Args:
            position: Query position
            radius: Search radius

        Returns:
            List of chunk coordinates
        """
        x, y, z = position
        result = set()

        # Calculate grid cells to check
        min_gx = int((x - radius) / self.grid_resolution)
        max_gx = int((x + radius) / self.grid_resolution)
        min_gy = int((y - radius) / self.grid_resolution)
        max_gy = int((y + radius) / self.grid_resolution)
        min_gz = int((z - radius) / self.grid_resolution)
        max_gz = int((z + radius) / self.grid_resolution)

        # Check each grid cell
        for gx in range(min_gx, max_gx + 1):
            for gy in range(min_gy, max_gy + 1):
                for gz in range(min_gz, max_gz + 1):
                    if (gx, gy, gz) in self.grid:
                        result.update(self.grid[(gx, gy, gz)])

        # Filter by actual distance
        filtered = []
        for chunk_coords in result:
            info = self.chunk_info[chunk_coords]

            # Check if chunk overlaps with sphere
            if self._chunk_overlaps_sphere(info, position, radius):
                filtered.append(chunk_coords)

        return filtered

    def _chunk_overlaps_sphere(
        self,
        chunk: ChunkInfo,
        center: Tuple[float, float, float],
        radius: float
    ) -> bool:
        """Check if chunk overlaps with sphere."""
        # Find closest point on chunk to sphere center
        cx = max(chunk.bounds_min[0], min(center[0], chunk.bounds_max[0]))
        cy = max(chunk.bounds_min[1], min(center[1], chunk.bounds_max[1]))
        cz = max(chunk.bounds_min[2], min(center[2], chunk.bounds_max[2]))

        # Check distance
        dx = cx - center[0]
        dy = cy - center[1]
        dz = cz - center[2]
        dist_sq = dx * dx + dy * dy + dz * dz

        return dist_sq <= radius * radius

    def get_chunk_at(self, x: float, y: float, z: float) -> Optional[Tuple[int, int, int]]:
        """Get chunk containing position."""
        for chunk_coords, info in self.chunk_info.items():
            if (info.bounds_min[0] <= x < info.bounds_max[0] and
                info.bounds_min[1] <= y < info.bounds_max[1] and
                info.bounds_min[2] <= z < info.bounds_max[2]):
                return chunk_coords
        return None


class SpatialIndex:
    """
    Main spatial index for ChunkedWaveCube with O(k) query support.
    """

    def __init__(
        self,
        chunk_size: Tuple[int, int, int] = (16, 16, 16),
        cache_radius: int = 2,
        query_cache_size: int = 100,
        query_cache_ttl: float = 10.0
    ):
        """
        Initialize spatial index.

        Args:
            chunk_size: Size of chunks
            cache_radius: Radius of chunks to keep loaded
            query_cache_size: Maximum cached query results
            query_cache_ttl: Time-to-live for cached queries (seconds)
        """
        self.chunk_size = chunk_size
        self.cache_radius = cache_radius

        # Spatial hash grid
        self.hash_grid = SpatialHashGrid(grid_resolution=max(chunk_size))

        # Active chunks
        self.active_chunks: Set[Tuple[int, int, int]] = set()
        self.active_center: Optional[Tuple[int, int, int]] = None

        # Query cache
        self.query_cache: Dict[str, Tuple[List[QueryResult], float]] = {}
        self.query_cache_size = query_cache_size
        self.query_cache_ttl = query_cache_ttl

        # Statistics
        self.stats = {
            'queries_total': 0,
            'cache_hits': 0,
            'chunks_examined': 0,
            'nodes_examined': 0
        }

    def add_chunk(
        self,
        chunk_coords: Tuple[int, int, int],
        node_count: int = 0,
        is_loaded: bool = False
    ) -> None:
        """Add chunk to index."""
        self.hash_grid.add_chunk(
            chunk_coords,
            self.chunk_size,
            node_count,
            is_loaded
        )

        if is_loaded:
            self.active_chunks.add(chunk_coords)

    def remove_chunk(self, chunk_coords: Tuple[int, int, int]) -> None:
        """Remove chunk from index."""
        self.hash_grid.remove_chunk(chunk_coords)
        self.active_chunks.discard(chunk_coords)
        self.invalidate_cache()  # Invalidate queries

    def set_active_center(
        self,
        position: Tuple[float, float, float]
    ) -> Set[Tuple[int, int, int]]:
        """
        Set active center and return chunks to load.

        Args:
            position: New active position

        Returns:
            Set of chunk coordinates to load
        """
        # Convert to chunk coordinates
        cx = int(position[0] / self.chunk_size[0])
        cy = int(position[1] / self.chunk_size[1])
        cz = int(position[2] / self.chunk_size[2])
        chunk_coords = (cx, cy, cz)

        if self.active_center == chunk_coords:
            return set()

        self.active_center = chunk_coords

        # Determine chunks to keep loaded
        chunks_to_load = set()
        for dx in range(-self.cache_radius, self.cache_radius + 1):
            for dy in range(-self.cache_radius, self.cache_radius + 1):
                for dz in range(-self.cache_radius, self.cache_radius + 1):
                    neighbor = (cx + dx, cy + dy, cz + dz)

                    # Only include if chunk exists
                    if neighbor in self.hash_grid.chunk_info:
                        chunks_to_load.add(neighbor)

        return chunks_to_load

    def knn_query(
        self,
        position: Tuple[float, float, float],
        k: int,
        max_distance: Optional[float] = None,
        filter_fn: Optional[Callable[[QueryResult], bool]] = None,
        node_getter: Optional[Callable] = None
    ) -> List[QueryResult]:
        """
        K-nearest neighbor query with O(k) performance.

        Args:
            position: Query position
            k: Number of neighbors
            max_distance: Maximum search distance
            filter_fn: Optional filter function
            node_getter: Function to get node data from chunk

        Returns:
            List of k nearest results
        """
        self.stats['queries_total'] += 1

        # Check cache
        cache_key = f"{position}_{k}_{max_distance}"
        if cache_key in self.query_cache:
            cached_results, cache_time = self.query_cache[cache_key]
            if time.time() - cache_time < self.query_cache_ttl:
                self.stats['cache_hits'] += 1
                return cached_results

        # Priority queue for results (min heap by distance)
        results = []
        examined_positions = set()

        # Start with small radius and expand
        initial_radius = max(self.chunk_size) * 2
        search_radius = initial_radius
        max_radius = max_distance if max_distance else float('inf')

        while len(results) < k and search_radius < max_radius:
            # Find chunks in radius
            chunk_coords_list = self.hash_grid.find_chunks_near(position, search_radius)

            # Sort chunks by distance to center
            chunks_with_dist = []
            for chunk_coords in chunk_coords_list:
                info = self.hash_grid.chunk_info[chunk_coords]

                # Calculate distance to chunk center
                chunk_center = (
                    (info.bounds_min[0] + info.bounds_max[0]) / 2,
                    (info.bounds_min[1] + info.bounds_max[1]) / 2,
                    (info.bounds_min[2] + info.bounds_max[2]) / 2
                )

                dist = self._euclidean_distance(position, chunk_center)
                chunks_with_dist.append((dist, chunk_coords))

            chunks_with_dist.sort()
            self.stats['chunks_examined'] += len(chunks_with_dist)

            # Examine chunks in order of distance
            for _, chunk_coords in chunks_with_dist:
                info = self.hash_grid.chunk_info[chunk_coords]

                # Priority: loaded chunks first
                if not info.is_loaded and self.active_chunks:
                    continue

                # Get nodes from chunk (using provided getter)
                if node_getter:
                    nodes = node_getter(chunk_coords)
                    if nodes:
                        for node_pos, node_data in nodes:
                            if node_pos not in examined_positions:
                                examined_positions.add(node_pos)
                                self.stats['nodes_examined'] += 1

                                dist = self._euclidean_distance(position, node_pos)

                                if max_distance and dist > max_distance:
                                    continue

                                result = QueryResult(
                                    position=node_pos,
                                    data=node_data,
                                    distance=dist
                                )

                                if filter_fn and not filter_fn(result):
                                    continue

                                heapq.heappush(results, (dist, result))

            # Expand search radius
            search_radius *= 1.5

            # Early exit if we have enough results
            if len(results) >= k * 2:
                break

        # Get top k results
        top_results = []
        while results and len(top_results) < k:
            _, result = heapq.heappop(results)
            top_results.append(result)

        # Cache results
        self._cache_query(cache_key, top_results)

        return top_results

    def radius_query(
        self,
        position: Tuple[float, float, float],
        radius: float,
        filter_fn: Optional[Callable[[QueryResult], bool]] = None,
        node_getter: Optional[Callable] = None
    ) -> List[QueryResult]:
        """
        Find all nodes within radius.

        Args:
            position: Query position
            radius: Search radius
            filter_fn: Optional filter function
            node_getter: Function to get node data

        Returns:
            List of results within radius
        """
        # Find relevant chunks
        chunk_coords_list = self.hash_grid.find_chunks_near(position, radius)

        results = []
        for chunk_coords in chunk_coords_list:
            info = self.hash_grid.chunk_info[chunk_coords]

            if node_getter:
                nodes = node_getter(chunk_coords)
                if nodes:
                    for node_pos, node_data in nodes:
                        dist = self._euclidean_distance(position, node_pos)

                        if dist <= radius:
                            result = QueryResult(
                                position=node_pos,
                                data=node_data,
                                distance=dist
                            )

                            if not filter_fn or filter_fn(result):
                                results.append(result)

        # Sort by distance
        results.sort(key=lambda r: r.distance)
        return results

    def _euclidean_distance(
        self,
        p1: Tuple[float, float, float],
        p2: Tuple[float, float, float]
    ) -> float:
        """Calculate Euclidean distance."""
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        dz = p1[2] - p2[2]
        return np.sqrt(dx * dx + dy * dy + dz * dz)

    def _cache_query(
        self,
        key: str,
        results: List[QueryResult]
    ) -> None:
        """Cache query results."""
        self.query_cache[key] = (results, time.time())

        # Evict old entries if cache is full
        if len(self.query_cache) > self.query_cache_size:
            # Remove oldest entry
            oldest_key = min(
                self.query_cache.keys(),
                key=lambda k: self.query_cache[k][1]
            )
            del self.query_cache[oldest_key]

    def invalidate_cache(self, position: Optional[Tuple[float, float, float]] = None) -> None:
        """
        Invalidate cached queries.

        Args:
            position: If provided, only invalidate queries near position
        """
        if position is None:
            self.query_cache.clear()
        else:
            # Remove queries near position
            to_remove = []
            for key in self.query_cache:
                # Parse position from key (simple approach)
                if str(position) in key:
                    to_remove.append(key)

            for key in to_remove:
                del self.query_cache[key]

    def get_stats(self) -> Dict[str, Any]:
        """Get query statistics."""
        cache_ratio = (
            self.stats['cache_hits'] / max(1, self.stats['queries_total'])
            if self.stats['queries_total'] > 0 else 0
        )

        return {
            'total_chunks': len(self.hash_grid.chunk_info),
            'active_chunks': len(self.active_chunks),
            'queries_total': self.stats['queries_total'],
            'cache_hits': self.stats['cache_hits'],
            'cache_hit_ratio': cache_ratio,
            'chunks_examined_avg': (
                self.stats['chunks_examined'] / max(1, self.stats['queries_total'])
                if self.stats['queries_total'] > 0 else 0
            ),
            'nodes_examined_avg': (
                self.stats['nodes_examined'] / max(1, self.stats['queries_total'])
                if self.stats['queries_total'] > 0 else 0
            ),
            'cache_size': len(self.query_cache)
        }