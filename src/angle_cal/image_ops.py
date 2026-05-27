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


@dataclass(frozen=True)
class SnapCurveResult:
    points: list[Point]
    offsets: np.ndarray
    gradient_strength: np.ndarray

    @property
    def start(self) -> Point:
        return self.points[0]

    @property
    def end(self) -> Point:
        return self.points[-1]


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


def snap_line_to_gradient_curve(
    gray: np.ndarray,
    start: Point,
    end: Point,
    search_radius_px: int = 30,
    sensitivity: int = 65,
) -> Optional[SnapCurveResult]:
    return snap_polyline_to_gradient(gray, [start, end], search_radius_px, sensitivity)


def _resample_polyline(points: Sequence[Point], step_px: float) -> list[Point]:
    cleaned: list[Point] = []
    for point in points:
        if not cleaned or line_length(cleaned[-1], point) > 1e-6:
            cleaned.append((float(point[0]), float(point[1])))
    if len(cleaned) < 2:
        return cleaned

    sampled: list[Point] = [cleaned[0]]
    carry = 0.0
    for start, end in zip(cleaned, cleaned[1:]):
        segment_length = line_length(start, end)
        if segment_length <= 0:
            continue
        distance = step_px - carry if carry > 0 else step_px
        while distance < segment_length:
            fraction = distance / segment_length
            sampled.append(
                (
                    start[0] + (end[0] - start[0]) * fraction,
                    start[1] + (end[1] - start[1]) * fraction,
                )
            )
            distance += step_px
        carry = max(0.0, segment_length - (distance - step_px))
    if line_length(sampled[-1], cleaned[-1]) > 1e-6:
        sampled.append(cleaned[-1])
    return sampled


def snap_polyline_to_gradient(
    gray: np.ndarray,
    points: Sequence[Point],
    search_radius_px: int = 30,
    sensitivity: int = 65,
) -> Optional[SnapCurveResult]:
    """Trace a connected polyline boundary near user-drawn connected segments.

    Sensitivity controls both point density and smoothing. Higher values follow
    local brightness changes more closely; lower values smooth the resulting
    boundary.
    """
    if search_radius_px <= 0:
        return None
    if len(points) < 2:
        return None

    sensitivity = int(np.clip(sensitivity, 1, 100))

    step_px = 18.0 - sensitivity * 0.14
    step_px = float(np.clip(step_px, 3.0, 18.0))
    sampled_points = _resample_polyline(points, step_px)
    if len(sampled_points) < 2:
        return None
    if len(sampled_points) > 240:
        indexes = np.linspace(0, len(sampled_points) - 1, 240).round().astype(int)
        sampled_points = [sampled_points[int(index)] for index in indexes]
    point_count = len(sampled_points)
    offsets = np.arange(-search_radius_px, search_radius_px + 1, dtype=np.float32)
    local_half_width = float(np.clip(step_px * 0.55, 1.5, 8.0))
    local_tangent_offsets = np.linspace(-local_half_width, local_half_width, 5, dtype=np.float32)

    best_offsets = np.full(point_count, np.nan, dtype=np.float32)
    strengths = np.full(point_count, np.nan, dtype=np.float32)
    tangents: list[Point] = []
    normals: list[Point] = []
    for idx, point in enumerate(sampled_points):
        prev_point = sampled_points[max(0, idx - 1)]
        next_point = sampled_points[min(point_count - 1, idx + 1)]
        tangent_length = line_length(prev_point, next_point)
        if tangent_length <= 0:
            tangents.append((0.0, 0.0))
            normals.append((0.0, 0.0))
            continue
        tx = (next_point[0] - prev_point[0]) / tangent_length
        ty = (next_point[1] - prev_point[1]) / tangent_length
        tangents.append((tx, ty))
        normals.append((-ty, tx))

    for idx, point in enumerate(sampled_points):
        tx, ty = tangents[idx]
        nx, ny = normals[idx]
        if nx == 0 and ny == 0:
            continue
        bx, by = point
        profile = np.empty(offsets.shape, dtype=np.float32)
        for offset_idx, offset in enumerate(offsets):
            xs = bx + tx * local_tangent_offsets + nx * offset
            ys = by + ty * local_tangent_offsets + ny * offset
            samples = _bilinear_sample(gray, xs, ys)
            valid_count = np.count_nonzero(~np.isnan(samples))
            profile[offset_idx] = float(np.nanmean(samples)) if valid_count >= 2 else np.nan

        finite = np.isfinite(profile)
        if np.count_nonzero(finite) < 5:
            continue
        filled = profile.copy()
        if not np.all(finite):
            filled = np.interp(offsets, offsets[finite], profile[finite]).astype(np.float32)
        if filled.size >= 5:
            kernel = np.array([1, 2, 3, 2, 1], dtype=np.float32)
            kernel /= kernel.sum()
            filled = np.convolve(np.pad(filled, (2, 2), mode="edge"), kernel, mode="valid").astype(np.float32)
        gradient = np.gradient(filled)
        best_idx = int(np.nanargmax(np.abs(gradient)))
        best_offsets[idx] = offsets[best_idx]
        strengths[idx] = abs(float(gradient[best_idx]))

    finite_offsets = np.isfinite(best_offsets)
    if np.count_nonzero(finite_offsets) < 3:
        return None
    if not np.all(finite_offsets):
        valid_x = np.flatnonzero(finite_offsets)
        best_offsets = np.interp(np.arange(point_count), valid_x, best_offsets[finite_offsets]).astype(np.float32)
        strengths = np.interp(np.arange(point_count), valid_x, strengths[finite_offsets]).astype(np.float32)

    smoothing_window = int(round((101 - sensitivity) / 14.0)) * 2 + 1
    smoothing_window = int(np.clip(smoothing_window, 1, 15))
    if smoothing_window > 1:
        kernel = np.ones(smoothing_window, dtype=np.float32) / smoothing_window
        pad = smoothing_window // 2
        best_offsets = np.convolve(np.pad(best_offsets, (pad, pad), mode="edge"), kernel, mode="valid").astype(np.float32)

    points = []
    for (bx, by), (nx, ny), offset in zip(sampled_points, normals, best_offsets):
        points.append((float(bx + nx * offset), float(by + ny * offset)))

    return SnapCurveResult(points=points, offsets=best_offsets, gradient_strength=strengths)


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
