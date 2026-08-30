"""Unit tests for the monocular table-plane reconstruction."""

import numpy as np
import pytest

from cable_perception.table_projection import (
    pixel_to_ray,
    project_pixels_to_plane,
    ray_plane_intersection,
)

INTRINSICS = (500.0, 500.0, 320.0, 240.0)


def test_principal_point_maps_to_the_optical_axis():
    assert pixel_to_ray(320.0, 240.0, *INTRINSICS) == pytest.approx([0.0, 0.0, 1.0])


def test_rays_are_unit_length():
    ray = pixel_to_ray(100.0, 50.0, *INTRINSICS)

    assert np.linalg.norm(ray) == pytest.approx(1.0)


def test_pixel_offset_tilts_the_ray_in_the_expected_direction():
    ray = pixel_to_ray(420.0, 240.0, *INTRINSICS)

    assert ray[0] > 0.0
    assert ray[1] == pytest.approx(0.0)


def test_zero_focal_length_is_rejected():
    with pytest.raises(ValueError):
        pixel_to_ray(0.0, 0.0, 0.0, 500.0, 320.0, 240.0)


def test_ray_hits_the_plane_straight_below_the_camera():
    point = ray_plane_intersection(
        origin=[0.0, 0.0, 2.0], direction=[0.0, 0.0, -1.0],
        plane_point=[0.0, 0.0, 0.0], plane_normal=[0.0, 0.0, 1.0])

    assert point == pytest.approx([0.0, 0.0, 0.0])


def test_slanted_ray_hits_the_expected_point():
    point = ray_plane_intersection(
        origin=[0.0, 0.0, 1.0], direction=[1.0, 0.0, -1.0],
        plane_point=[0.0, 0.0, 0.0], plane_normal=[0.0, 0.0, 1.0])

    assert point == pytest.approx([1.0, 0.0, 0.0])


def test_ray_parallel_to_the_plane_never_intersects():
    assert ray_plane_intersection(
        origin=[0.0, 0.0, 1.0], direction=[1.0, 0.0, 0.0],
        plane_point=[0.0, 0.0, 0.0], plane_normal=[0.0, 0.0, 1.0]) is None


def test_plane_behind_the_camera_is_rejected():
    assert ray_plane_intersection(
        origin=[0.0, 0.0, 2.0], direction=[0.0, 0.0, 1.0],
        plane_point=[0.0, 0.0, 0.0], plane_normal=[0.0, 0.0, 1.0]) is None


def test_degenerate_plane_normal_is_rejected():
    with pytest.raises(ValueError):
        ray_plane_intersection([0.0, 0.0, 1.0], [0.0, 0.0, -1.0],
                               [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])


def test_round_trip_a_known_table_point_through_the_camera():
    """Project a 3-D table point to a pixel, then reconstruct it back."""
    # Camera 1.5 m above the table looking straight down; its optical z axis
    # points along -z of the table frame, and its x axis is aligned.
    rotation = np.array([[1.0, 0.0, 0.0],
                         [0.0, -1.0, 0.0],
                         [0.0, 0.0, -1.0]])
    origin = np.array([0.0, 0.0, 1.5])
    fx, fy, cx, cy = INTRINSICS

    truth = np.array([0.12, -0.07, 0.0])
    in_camera = rotation.T @ (truth - origin)
    pixel = [fx * in_camera[0] / in_camera[2] + cx,
             fy * in_camera[1] / in_camera[2] + cy]

    (point,) = project_pixels_to_plane(
        [pixel], INTRINSICS, rotation, origin,
        plane_point=[0.0, 0.0, 0.0], plane_normal=[0.0, 0.0, 1.0])

    assert point == pytest.approx(truth, abs=1e-9)


def test_projection_reports_misses_as_none():
    rotation = np.eye(3)  # camera looks along +z, away from the table below it

    (point,) = project_pixels_to_plane(
        [[320.0, 240.0]], INTRINSICS, rotation, [0.0, 0.0, 1.5],
        plane_point=[0.0, 0.0, 0.0], plane_normal=[0.0, 0.0, 1.0])

    assert point is None


def test_wrong_shaped_pixel_array_is_rejected():
    with pytest.raises(ValueError):
        project_pixels_to_plane(
            [[1.0, 2.0, 3.0]], INTRINSICS, np.eye(3), [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0], [0.0, 0.0, 1.0])
