from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import cv2
import numpy as np

Point = Tuple[float, float]
Line = Tuple[Point, Point]


@dataclass(frozen=True)
class SnapResult:
    start: Point
    end: Point
    offset_px: float
    gradient: float
    offsets: np.ndarray
    intensity_profile: np.ndarray
    gradient_profile: np.ndarray


def line_length(start: Point, end: Point) -> float:
    return float(math.hypot(end[0] - start[0], end[1] - start[1]))


def line_angle_degrees(start: Point, end: Point) -> float:
    """Return the image-coordinate angle in [0, 180)."""
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    if dx == 0 and dy == 0:
        return 0.0
    return math.degrees(math.atan2(dy, dx)) % 180.0


def acute_angle_difference(angle_a: float, angle_b: float) -> float:
    """Smallest unsigned difference between two undirected line angles."""
    return abs(((angle_a - angle_b + 90.0) % 180.0) - 90.0)


def angle_to_axis(start: Point, end: Point, axis: str) -> float:
    target = 0.0 if axis == "horizontal" else 90.0
    return acute_angle_difference(line_angle_degrees(start, end), target)


def normal_for_line(start: Point, end: Point) -> Point:
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length = math.hypot(dx, dy)
    if length == 0:
        return (0.0, 0.0)
    return (-dy / length, dx / length)


def to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image.astype(np.float32)
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)


def ensure_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    if image.ndim == 3 and image.shape[2] == 3:
        return image
    raise ValueError(f"Unsupported image shape: {image.shape}")


def read_image(path: str | Path) -> Optional[np.ndarray]:
    """Read images through imdecode so Windows Unicode paths work."""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    image = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if image is None:
        return None
    return ensure_bgr(image)


def bgr_to_rgb8_for_display(image: np.ndarray) -> np.ndarray:
    bgr = ensure_bgr(image)
    if bgr.dtype == np.uint8:
        display = bgr
    else:
        values = bgr.astype(np.float32)
        min_value = float(np.nanmin(values))
        max_value = float(np.nanmax(values))
        if max_value <= min_value:
            display = np.zeros(bgr.shape, dtype=np.uint8)
        else:
            display = np.clip((values - min_value) * (255.0 / (max_value - min_value)), 0, 255).astype(np.uint8)
    return cv2.cvtColor(display, cv2.COLOR_BGR2RGB)


def _bilinear_sample(gray: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    h, w = gray.shape[:2]
    x0 = np.floor(xs).astype(np.int32)
    y0 = np.floor(ys).astype(np.int32)
    x1 = x0 + 1
    y1 = y0 + 1
    valid = (x0 >= 0) & (y0 >= 0) & (x1 < w) & (y1 < h)
    values = np.full(xs.shape, np.nan, dtype=np.float32)
    if not np.any(valid):
        return values

    xv = xs[valid]
    yv = ys[valid]
    x0v = x0[valid]
    y0v = y0[valid]
    x1v = x1[valid]
    y1v = y1[valid]
    wx = xv - x0v
    wy = yv - y0v

    top = gray[y0v, x0v] * (1.0 - wx) + gray[y0v, x1v] * wx
    bottom = gray[y1v, x0v] * (1.0 - wx) + gray[y1v, x1v] * wx
    values[valid] = top * (1.0 - wy) + bottom * wy
    return values


def snap_line_to_gradient(
    gray: np.ndarray,
    start: Point,
    end: Point,
    search_radius_px: int = 30,
    samples_along_line: int = 160,
) -> Optional[SnapResult]:
    """Move a line along its normal to the strongest brightness change.

    The method samples average brightness on parallel offsets around the user's
    drawn line. The chosen offset is where the absolute derivative of that
    profile is largest.
    """
    if search_radius_px <= 0 or samples_along_line < 2:
        return None

    length = line_length(start, end)
    if length < 2:
        return None

    nx, ny = normal_for_line(start, end)
    if nx == 0 and ny == 0:
        return None

    offsets = np.arange(-search_radius_px, search_radius_px + 1, dtype=np.float32)
    t = np.linspace(0.0, 1.0, samples_along_line, dtype=np.float32)
    base_x = start[0] + (end[0] - start[0]) * t
    base_y = start[1] + (end[1] - start[1]) * t

    profile = np.empty(offsets.shape, dtype=np.float32)
    for i, offset in enumerate(offsets):
        xs = base_x + nx * offset
        ys = base_y + ny * offset
        samples = _bilinear_sample(gray, xs, ys)
        valid_count = np.count_nonzero(~np.isnan(samples))
        if valid_count < max(2, samples_along_line // 4):
            profile[i] = np.nan
        else:
            profile[i] = float(np.nanmean(samples))

    finite = np.isfinite(profile)
    if np.count_nonzero(finite) < 5:
        return None

    filled = profile.copy()
    if not np.all(finite):
        valid_x = offsets[finite]
        valid_y = profile[finite]
        filled = np.interp(offsets, valid_x, valid_y).astype(np.float32)

    if filled.size >= 5:
        kernel = np.array([1, 2, 3, 2, 1], dtype=np.float32)
        kernel /= kernel.sum()
        padded = np.pad(filled, (2, 2), mode="edge")
        filled = np.convolve(padded, kernel, mode="valid").astype(np.float32)

    gradient = np.gradient(filled)
    best_index = int(np.nanargmax(np.abs(gradient)))
    best_offset = float(offsets[best_index])
    best_gradient = float(gradient[best_index])
    shifted_start = (start[0] + nx * best_offset, start[1] + ny * best_offset)
    shifted_end = (end[0] + nx * best_offset, end[1] + ny * best_offset)
    return SnapResult(
        start=shifted_start,
        end=shifted_end,
        offset_px=best_offset,
        gradient=best_gradient,
        offsets=offsets,
        intensity_profile=filled,
        gradient_profile=gradient,
    )


def rotate_image_and_points(
    image: np.ndarray,
    points: Sequence[Point],
    angle_degrees: float,
) -> Tuple[np.ndarray, list[Point]]:
    h, w = image.shape[:2]
    center = (w / 2.0, h / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle_degrees, 1.0)
    cos = abs(matrix[0, 0])
    sin = abs(matrix[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    matrix[0, 2] += (new_w / 2.0) - center[0]
    matrix[1, 2] += (new_h / 2.0) - center[1]

    rotated = cv2.warpAffine(
        image,
        matrix,
        (new_w, new_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    transformed: list[Point] = []
    for x, y in points:
        xp = matrix[0, 0] * x + matrix[0, 1] * y + matrix[0, 2]
        yp = matrix[1, 0] * x + matrix[1, 1] * y + matrix[1, 2]
        transformed.append((float(xp), float(yp)))
    return rotated, transformed


def intersection(line_a: Line, line_b: Line, as_segments: bool = True) -> Optional[Point]:
    (x1, y1), (x2, y2) = line_a
    (x3, y3), (x4, y4) = line_b
    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(den) < 1e-9:
        return None

    t_num = (x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)
    u_num = (x1 - x3) * (y1 - y2) - (y1 - y3) * (x1 - x2)
    t = t_num / den
    u = u_num / den
    if as_segments and not (-1e-6 <= t <= 1.0 + 1e-6 and -1e-6 <= u <= 1.0 + 1e-6):
        return None
    return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))


def all_points_from_lines(lines: Iterable[Line]) -> list[Point]:
    points: list[Point] = []
    for start, end in lines:
        points.extend([start, end])
    return points
