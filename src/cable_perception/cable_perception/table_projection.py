"""Monocular reconstruction of cable markers lying on a known table plane.

Pure geometry, no ROS. A single eye-to-hand camera cannot triangulate a point,
but the cable rests on a plane whose pose is known from TF, so each pixel maps
to exactly one 3-D point through a ray/plane intersection.
"""

import numpy as np


def quaternion_to_matrix(x, y, z, w):
    """Rotation matrix of a unit quaternion (no external dependency)."""
    n = np.sqrt(x * x + y * y + z * z + w * w)
    if n == 0.0:
        raise ValueError("zero-norm quaternion")
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def pixel_to_ray(u, v, fx, fy, cx, cy):
    """Unit direction of the optical ray through pixel (u, v), camera frame."""
    if fx == 0.0 or fy == 0.0:
        raise ValueError("focal lengths must be non-zero")
    d = np.array([(u - cx) / fx, (v - cy) / fy, 1.0], dtype=float)
    return d / np.linalg.norm(d)


def ray_plane_intersection(origin, direction, plane_point, plane_normal,
                           min_range_m=1e-3):
    """Intersect a ray with a plane; both expressed in the same frame.

    Returns the 3-D point, or ``None`` when the ray is parallel to the plane or
    the intersection falls behind the camera.
    """
    origin = np.asarray(origin, dtype=float)
    direction = np.asarray(direction, dtype=float)
    plane_point = np.asarray(plane_point, dtype=float)
    normal = np.asarray(plane_normal, dtype=float)

    norm = np.linalg.norm(normal)
    if norm == 0.0:
        raise ValueError("plane_normal must be non-zero")
    normal = normal / norm

    denom = float(normal @ direction)
    if abs(denom) < 1e-9:
        return None  # ray parallel to the table

    t = float(normal @ (plane_point - origin)) / denom
    if t < min_range_m:
        return None  # plane is behind the camera
    return origin + t * direction


def project_pixels_to_plane(pixels, intrinsics, cam_to_plane_rotation,
                            cam_position, plane_point, plane_normal):
    """Reconstruct several pixels onto the table plane.

    Parameters
    ----------
    pixels
        (M, 2) array of (u, v) pixel coordinates.
    intrinsics
        ``(fx, fy, cx, cy)``.
    cam_to_plane_rotation
        3x3 rotation taking camera-frame vectors into the plane's frame.
    cam_position
        Camera origin expressed in the plane's frame.
    plane_point, plane_normal
        Any point on the table and its normal, in the plane's frame.

    Returns a list of 3-D points (``None`` where the ray misses the plane).
    """
    pixels = np.asarray(pixels, dtype=float)
    if pixels.ndim != 2 or pixels.shape[1] != 2:
        raise ValueError(f"expected an (M, 2) array, got {pixels.shape}")

    rotation = np.asarray(cam_to_plane_rotation, dtype=float)
    if rotation.shape != (3, 3):
        raise ValueError("cam_to_plane_rotation must be 3x3")

    fx, fy, cx, cy = intrinsics
    points = []
    for u, v in pixels:
        ray_cam = pixel_to_ray(u, v, fx, fy, cx, cy)
        ray_world = rotation @ ray_cam
        points.append(ray_plane_intersection(
            cam_position, ray_world, plane_point, plane_normal))
    return points
