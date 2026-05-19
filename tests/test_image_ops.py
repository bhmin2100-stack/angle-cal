import math

import numpy as np

from angle_cal.image_ops import (
    acute_angle_difference,
    angle_to_axis,
    intersection,
    line_angle_degrees,
    rotate_image_and_points,
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


def test_rotate_points_can_align_downward_line_to_horizontal():
    image = np.zeros((80, 100), dtype=np.uint8)
    start = (20.0, 40.0)
    end = (80.0, 50.0)
    angle = line_angle_degrees(start, end)
    _, points = rotate_image_and_points(image, [start, end], angle)
    rotated_angle = line_angle_degrees(points[0], points[1])
    assert rotated_angle < 0.5 or rotated_angle > 179.5
