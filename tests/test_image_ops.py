import math

import numpy as np
import cv2

from angle_cal.image_ops import (
    acute_angle_difference,
    adjust_image_bgr,
    angle_to_axis,
    bgr_to_rgb8_for_display,
    intersection,
    line_angle_degrees,
    measure_cliff_curvature,
    read_image,
    rotate_image_and_points,
    snap_line_to_gradient_curve,
    snap_polyline_to_gradient,
    snap_line_to_gradient,
)


def test_angle_helpers_use_undirected_lines():
    assert math.isclose(line_angle_degrees((0, 0), (10, 0)), 0)
    assert math.isclose(line_angle_degrees((0, 0), (0, 10)), 90)
    assert math.isclose(acute_angle_difference(5, 175), 10)
    assert math.isclose(angle_to_axis((0, 0), (10, 10), "horizontal"), 45)
    assert math.isclose(angle_to_axis((0, 0), (10, 10), "vertical"), 45)


def test_intersection_on_segments():
    pt = intersection(((0, 0), (10, 10)), ((0, 10), (10, 0)))
    assert pt is not None
    assert math.isclose(pt[0], 5)
    assert math.isclose(pt[1], 5)
    assert intersection(((0, 0), (1, 0)), ((2, 1), (2, -1))) is None


def test_snap_line_to_synthetic_vertical_edge():
    image = np.zeros((120, 160), dtype=np.float32)
    image[:, 82:] = 255
    result = snap_line_to_gradient(image, (70, 20), (70, 100), search_radius_px=30)
    assert result is not None
    snapped_x = (result.start[0] + result.end[0]) / 2
    assert 80 <= snapped_x <= 83


def test_adjust_image_bgr_applies_brightness_contrast_and_sharpness():
    image = np.full((12, 12, 3), 100, dtype=np.uint8)
    image[:, 6:] = 150

    bright = adjust_image_bgr(image, brightness=35, contrast_percent=100, sharpness_percent=0)
    contrasted = adjust_image_bgr(image, brightness=0, contrast_percent=160, sharpness_percent=0)
    sharpened = adjust_image_bgr(image, brightness=0, contrast_percent=100, sharpness_percent=120)

    assert bright.dtype == np.uint8
    assert float(bright.mean()) > float(image.mean())
    assert float(contrasted[:, 6:].mean() - contrasted[:, :6].mean()) > 50.0
    assert sharpened.shape == image.shape


def test_asymmetric_search_range_limits_normal_sides():
    image = np.zeros((120, 160), dtype=np.float32)
    image[:, 82:] = 255

    too_short_right = snap_line_to_gradient(
        image,
        (70, 20),
        (70, 100),
        search_radius_px=30,
        search_radius_left_px=30,
        search_radius_right_px=5,
    )
    enough_right = snap_line_to_gradient(
        image,
        (70, 20),
        (70, 100),
        search_radius_px=30,
        search_radius_left_px=5,
        search_radius_right_px=20,
    )

    assert too_short_right is not None
    assert enough_right is not None
    assert ((too_short_right.start[0] + too_short_right.end[0]) / 2) < 78
    assert 80 <= ((enough_right.start[0] + enough_right.end[0]) / 2) <= 83


def test_snap_line_to_gradient_curve_follows_curved_edge():
    image = np.zeros((140, 180), dtype=np.float32)
    for y in range(image.shape[0]):
        boundary_x = int(round(82 + 10 * math.sin((y - 20) / 18)))
        image[y, boundary_x:] = 255

    result = snap_line_to_gradient_curve(image, (70, 15), (70, 125), search_radius_px=30, segment_size_px=5)

    assert result is not None
    xs = np.array([point[0] for point in result.points])
    assert len(result.points) > 10
    assert xs.std() > 3
    assert 72 <= xs.mean() <= 90


def test_snap_polyline_to_gradient_uses_drawn_connected_segments():
    image = np.zeros((140, 180), dtype=np.float32)
    for y in range(image.shape[0]):
        boundary_x = int(round(62 + 0.34 * y))
        image[y, boundary_x:] = 255

    result = snap_polyline_to_gradient(
        image,
        [(50, 15), (70, 70), (95, 125)],
        search_radius_px=25,
        segment_size_px=7,
    )

    assert result is not None
    xs = np.array([point[0] for point in result.points])
    ys = np.array([point[1] for point in result.points])
    assert len(result.points) > 10
    assert ys.max() - ys.min() > 90
    assert xs[-1] - xs[0] > 25


def test_measure_cliff_curvature_fits_rounded_corner_radius():
    image = np.zeros((130, 130), dtype=np.uint8)
    center = (58.0, 66.0)
    radius = 18.0
    arc_points = [
        (
            center[0] + radius * math.cos(math.radians(angle)),
            center[1] + radius * math.sin(math.radians(angle)),
        )
        for angle in np.linspace(-90.0, 0.0, 30)
    ]
    boundary = [(0.0, center[1] - radius), (center[0], center[1] - radius), *arc_points, (center[0] + radius, 129.0), (0.0, 129.0)]
    polygon = np.array(boundary, dtype=np.int32)
    cv2.fillPoly(image, [polygon], 255)
    image = cv2.GaussianBlur(image, (3, 3), 0)

    result = measure_cliff_curvature(image)

    assert result is not None
    assert 13.0 <= result.radius_px <= 24.0
    assert math.hypot(result.center[0] - center[0], result.center[1] - center[1]) < 8.0
    assert result.quality > 0.7


def test_rotate_points_can_align_downward_line_to_horizontal():
    image = np.zeros((80, 100), dtype=np.uint8)
    start = (20.0, 40.0)
    end = (80.0, 50.0)
    angle = line_angle_degrees(start, end)
    _, points = rotate_image_and_points(image, [start, end], angle)
    rotated_angle = line_angle_degrees(points[0], points[1])
    assert rotated_angle < 0.5 or rotated_angle > 179.5


def test_read_image_supports_unicode_png_path(tmp_path):
    path = tmp_path / "한글 SEM 이미지.png"
    image = np.zeros((12, 16), dtype=np.uint16)
    image[:, 8:] = 4095
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    encoded.tofile(path)

    loaded = read_image(path)
    assert loaded is not None
    assert loaded.shape == (12, 16, 3)
    assert loaded.dtype == np.uint16

    display = bgr_to_rgb8_for_display(loaded)
    assert display.shape == (12, 16, 3)
    assert display.dtype == np.uint8
    assert display[:, :4].max() == 0
    assert display[:, 8:].max() == 255
