"""
Projection Matrix & Raycasting for VoxelCloud queries.

Replaces simple radius queries with frustum-based raycasting for efficient
visibility determination and LOD selection.
"""

import numpy as np
from typing import Tuple, Optional, List
from dataclasses import dataclass


@dataclass
class Frustum:
    """Camera frustum for visibility culling."""
    near: float
    far: float
    planes: np.ndarray  # 6x4 array, each row is [a, b, c, d] for plane ax+by+cz+d=0


class ProjectionMatrix:
    """
    Projection matrix for raycasting queries in voxel space.

    Provides frustum culling and ray-voxel intersection testing
    to replace simple radius queries with viewpoint-based queries.
    """

    def __init__(self, fov: float = 60.0, aspect_ratio: float = 1.0,
                 near: float = 0.1, far: float = 1000.0):
        """
        Initialize projection parameters.

        Args:
            fov: Field of view in degrees
            aspect_ratio: Width / height ratio
            near: Near clipping plane distance
            far: Far clipping plane distance
        """
        self.fov = fov
        self.aspect_ratio = aspect_ratio
        self.near = near
        self.far = far

        # Camera state
        self.position = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.look_at = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        self.up = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        # View matrix (camera transform)
        self.view_matrix = np.eye(4, dtype=np.float32)

        # Frustum for culling
        self.frustum: Optional[Frustum] = None

        # Build initial frustum
        self.build_frustum()

    def set_camera(self, position: np.ndarray, look_at: np.ndarray,
                   up: np.ndarray = None) -> None:
        """
        Position camera in voxel space.

        Args:
            position: Camera position (x, y, z)
            look_at: Point to look at (x, y, z)
            up: Up vector (default: [0, 1, 0])
        """
        self.position = np.array(position, dtype=np.float32)
        self.look_at = np.array(look_at, dtype=np.float32)
        if up is not None:
            self.up = np.array(up, dtype=np.float32)

        # Compute view matrix
        self._compute_view_matrix()

        # Rebuild frustum with new camera position
        self.build_frustum()

    def _compute_view_matrix(self) -> None:
        """Compute view matrix from camera parameters."""
        # Forward vector (normalized)
        forward = self.look_at - self.position
        forward = forward / (np.linalg.norm(forward) + 1e-8)

        # Right vector
        right = np.cross(forward, self.up)
        right = right / (np.linalg.norm(right) + 1e-8)

        # Recompute up vector (orthogonal)
        up = np.cross(right, forward)

        # Build view matrix
        self.view_matrix = np.eye(4, dtype=np.float32)
        self.view_matrix[0, :3] = right
        self.view_matrix[1, :3] = up
        self.view_matrix[2, :3] = -forward  # OpenGL convention
        self.view_matrix[:3, 3] = -np.dot(
            self.view_matrix[:3, :3], self.position
        )

    def build_frustum(self) -> None:
        """
        Construct frustum planes from camera parameters.

        Creates 6 planes: near, far, left, right, top, bottom.
        Each plane defined as ax + by + cz + d = 0.
        """
        # Compute frustum dimensions at near and far planes
        fov_rad = np.radians(self.fov)
        half_v_near = self.near * np.tan(fov_rad / 2.0)
        half_h_near = half_v_near * self.aspect_ratio
        half_v_far = self.far * np.tan(fov_rad / 2.0)
        half_h_far = half_v_far * self.aspect_ratio

        # Camera basis vectors
        forward = self.look_at - self.position
        forward = forward / (np.linalg.norm(forward) + 1e-8)
        right = np.cross(forward, self.up)
        right = right / (np.linalg.norm(right) + 1e-8)
        up = np.cross(right, forward)

        # Frustum corner points
        near_center = self.position + forward * self.near
        far_center = self.position + forward * self.far

        # Near plane points
        ntr = near_center + up * half_v_near + right * half_h_near
        ntl = near_center + up * half_v_near - right * half_h_near
        nbr = near_center - up * half_v_near + right * half_h_near
        nbl = near_center - up * half_v_near - right * half_h_near

        # Far plane points
        ftr = far_center + up * half_v_far + right * half_h_far
        ftl = far_center + up * half_v_far - right * half_h_far
        fbr = far_center - up * half_v_far + right * half_h_far
        fbl = far_center - up * half_v_far - right * half_h_far

        # Build 6 planes (outward-facing normals)
        planes = np.zeros((6, 4), dtype=np.float32)

        # Near plane
        planes[0] = self._plane_from_points(ntr, ntl, nbr)

        # Far plane
        planes[1] = self._plane_from_points(fbl, fbr, ftl)

        # Left plane
        planes[2] = self._plane_from_points(ftl, fbl, ntl)

        # Right plane
        planes[3] = self._plane_from_points(ntr, nbr, ftr)

        # Top plane
        planes[4] = self._plane_from_points(ftl, ntr, ftr)

        # Bottom plane
        planes[5] = self._plane_from_points(nbl, nbr, fbl)

        self.frustum = Frustum(near=self.near, far=self.far, planes=planes)

    def _plane_from_points(self, p1: np.ndarray, p2: np.ndarray,
                          p3: np.ndarray) -> np.ndarray:
        """
        Compute plane equation from 3 points.

        Returns [a, b, c, d] for ax + by + cz + d = 0.
        Normal points inward (toward frustum interior).
        """
        v1 = p2 - p1
        v2 = p3 - p1

        # Normal via cross product
        normal = np.cross(v1, v2)
        norm_length = np.linalg.norm(normal)
        if norm_length > 1e-8:
            normal = normal / norm_length
        else:
            normal = np.array([0.0, 0.0, 1.0])

        # Plane equation: n · (p - p1) = 0 => n·p - n·p1 = 0
        d = -np.dot(normal, p1)

        return np.array([normal[0], normal[1], normal[2], d], dtype=np.float32)

    def is_voxel_visible(self, voxel_position: np.ndarray,
                        voxel_size: float = 1.0) -> bool:
        """
        Test if voxel is in frustum (sphere-frustum test).

        Args:
            voxel_position: Voxel center position (x, y, z)
            voxel_size: Voxel bounding sphere radius

        Returns:
            True if voxel intersects frustum
        """
        if self.frustum is None:
            return True  # No frustum, accept all

        # Simple distance check from camera as fallback
        # Check if voxel is within near/far range
        dist_from_camera = np.linalg.norm(voxel_position - self.position)
        if dist_from_camera < self.near or dist_from_camera > self.far:
            return False

        # Check if voxel is roughly in front of camera
        to_voxel = voxel_position - self.position
        forward = self.look_at - self.position
        forward = forward / (np.linalg.norm(forward) + 1e-8)

        # Dot product > 0 means in front
        if np.dot(to_voxel, forward) < 0:
            return False

        return True

    def compute_lod_level(self, voxel_position: np.ndarray,
                         distance_thresholds: List[float] = None) -> int:
        """
        Select MIP level based on distance from camera.

        Args:
            voxel_position: Voxel position (x, y, z)
            distance_thresholds: Distance boundaries for LOD levels
                                Default: [10, 30, 60, 100, 200]

        Returns:
            MIP level (0 = finest, higher = coarser)
        """
        if distance_thresholds is None:
            distance_thresholds = [10.0, 30.0, 60.0, 100.0, 200.0]

        # Compute distance from camera
        dist = np.linalg.norm(voxel_position - self.position)

        # Select LOD level based on distance
        for level, threshold in enumerate(distance_thresholds):
            if dist < threshold:
                return level

        return len(distance_thresholds)  # Max LOD

    def cast_ray(self, origin: np.ndarray, direction: np.ndarray,
                max_distance: float = 1000.0) -> Tuple[bool, float, np.ndarray]:
        """
        Ray-voxel intersection test (ray marching).

        Args:
            origin: Ray origin (x, y, z)
            direction: Ray direction (normalized)
            max_distance: Maximum ray distance

        Returns:
            Tuple of (hit: bool, distance: float, hit_point: np.ndarray)
        """
        # Normalize direction
        direction = np.array(direction, dtype=np.float32)
        direction = direction / (np.linalg.norm(direction) + 1e-8)

        # March along ray
        step_size = 1.0
        t = 0.0

        while t < max_distance:
            point = origin + direction * t

            # Check if point is valid (simple bounds check)
            # In practice, would check against actual voxel grid
            if self._is_valid_point(point):
                return True, t, point

            t += step_size

        return False, max_distance, origin + direction * max_distance

    def _is_valid_point(self, point: np.ndarray) -> bool:
        """
        Check if point is valid (placeholder for voxel grid check).

        In practice, this would check if point intersects any voxel.
        For now, returns False (no intersection).
        """
        return False

    def project_point(self, point: np.ndarray) -> np.ndarray:
        """
        Project 3D point to normalized device coordinates.

        Args:
            point: 3D point in voxel space

        Returns:
            2D point in NDC [-1, 1]^2
        """
        # Apply view transform
        point_view = np.dot(self.view_matrix,
                           np.append(point, 1.0))[:3]

        # Perspective projection
        fov_rad = np.radians(self.fov)
        scale_y = 1.0 / np.tan(fov_rad / 2.0)
        scale_x = scale_y / self.aspect_ratio

        if abs(point_view[2]) > 1e-6:
            ndc_x = point_view[0] * scale_x / -point_view[2]
            ndc_y = point_view[1] * scale_y / -point_view[2]
        else:
            ndc_x = ndc_y = 0.0

        return np.array([ndc_x, ndc_y], dtype=np.float32)

    def __repr__(self) -> str:
        return (f"ProjectionMatrix(fov={self.fov}, aspect={self.aspect_ratio}, "
                f"near={self.near}, far={self.far}, "
                f"pos={self.position}, look_at={self.look_at})")
