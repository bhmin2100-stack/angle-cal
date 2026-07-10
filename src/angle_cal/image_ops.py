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


@dataclass(frozen=True)
class SegmentProfileResult:
    offsets: np.ndarray
    distances: np.ndarray
    sample_grid: np.ndarray
    sample_counts: np.ndarray
    intensity_profile: np.ndarray
    gradient_profile: np.ndarray
    best_offset_px: float
    best_gradient: float


@dataclass(frozen=True)
class CurvatureResult:
    center: Point
    radius_px: float
    apex: Point
    edge_points: list[Point]
    fit_points: list[Point]
    quality: float


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


def _scalar_intensity(image: np.ndarray) -> np.ndarray:
    return to_gray(image)


def _normal_intensity_change(intensity: np.ndarray, nx: float, ny: float) -> np.ndarray:
    gy, gx = np.gradient(intensity.astype(np.float32))
    directional_change = gx * float(nx) + gy * float(ny)
    return np.abs(directional_change).astype(np.float32)


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


def adjust_image_bgr(
    image: np.ndarray,
    brightness: int = 0,
    contrast_percent: int = 100,
    sharpness_percent: int = 0,
) -> np.ndarray:
    bgr = ensure_bgr(image).astype(np.float32)
    contrast = max(0.0, float(contrast_percent)) / 100.0
    adjusted = (bgr - 127.5) * contrast + 127.5 + float(brightness)
    adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)
    if sharpness_percent > 0:
        amount = min(5.0, float(sharpness_percent) / 100.0)
        blurred = cv2.GaussianBlur(adjusted, (0, 0), 1.0)
        adjusted = cv2.addWeighted(adjusted, 1.0 + amount, blurred, -amount, 0)
    return adjusted


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
    search_radius_left_px: Optional[int] = None,
    search_radius_right_px: Optional[int] = None,
) -> Optional[SnapResult]:
    """Move a line along its normal to the strongest brightness change.

    The method samples average brightness on parallel offsets around the user's
    drawn line. The chosen offset is where the absolute derivative of that
    profile is largest.
    """
    left_radius_px, right_radius_px = _resolve_search_radii(
        search_radius_px,
        search_radius_left_px,
        search_radius_right_px,
    )
    if (left_radius_px <= 0 and right_radius_px <= 0) or samples_along_line < 2:
        return None

    length = line_length(start, end)
    if length < 2:
        return None

    nx, ny = normal_for_line(start, end)
    if nx == 0 and ny == 0:
        return None

    offsets = _search_offsets(left_radius_px, right_radius_px)
    if offsets.size < 3:
        return None
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
    segment_size_px: float = 9.0,
    search_radius_left_px: Optional[int] = None,
    search_radius_right_px: Optional[int] = None,
) -> Optional[SnapCurveResult]:
    return snap_polyline_to_gradient(
        gray,
        [start, end],
        search_radius_px,
        segment_size_px,
        search_radius_left_px,
        search_radius_right_px,
    )


def segment_brightness_profile(
    image: np.ndarray,
    start: Point,
    end: Point,
    search_radius_px: int = 30,
    search_radius_left_px: Optional[int] = None,
    search_radius_right_px: Optional[int] = None,
    boundary_mode: str = "max_gradient",
) -> Optional[SegmentProfileResult]:
    left_radius_px, right_radius_px = _resolve_search_radii(
        search_radius_px,
        search_radius_left_px,
        search_radius_right_px,
    )
    if left_radius_px <= 0 and right_radius_px <= 0:
        return None

    length = line_length(start, end)
    if length < 2:
        return None

    dx = end[0] - start[0]
    dy = end[1] - start[1]
    tx = dx / length
    ty = dy / length
    nx, ny = normal_for_line(start, end)
    if nx == 0 and ny == 0:
        return None

    offsets = _search_offsets(left_radius_px, right_radius_px)
    if offsets.size < 3:
        return None

    sample_count = max(2, int(math.ceil(length)) + 1)
    distances = np.linspace(0.0, float(length), sample_count, dtype=np.float32)
    min_valid_samples = max(2, int(distances.size // 4))

    patch = _segment_pixel_patch(
        image,
        start,
        end,
        offsets,
        distances,
        left_radius_px,
        right_radius_px,
    )
    if patch is None:
        return None
    sample_grid, counts, gradient_grid = patch

    filled = _row_mean_profile(sample_grid, counts, offsets, min_valid_samples)
    if filled is None:
        return None
    gradient = _area_gradient_profile(gradient_grid, counts, offsets, min_valid_samples)
    if gradient is None:
        gradient = np.abs(np.gradient(filled)).astype(np.float32)
    best_index = _boundary_profile_index(offsets, filled, gradient, boundary_mode)
    return SegmentProfileResult(
        offsets=offsets,
        distances=distances,
        sample_grid=sample_grid,
        sample_counts=counts,
        intensity_profile=filled,
        gradient_profile=gradient,
        best_offset_px=float(offsets[best_index]),
        best_gradient=float(gradient[best_index]),
    )


def measure_cliff_curvature(image: np.ndarray) -> Optional[CurvatureResult]:
    """Estimate the local radius of curvature for a cliff-like rounded edge.

    The input is a user-selected ROI. The method extracts high-gradient edge
    contours, finds the strongest local turning point, and fits a circle to the
    neighboring contour points around that apex. Returned coordinates are
    relative to the ROI origin.
    """
    gray = to_gray(image)
    if gray.size < 64:
        return None
    height, width = gray.shape[:2]
    if height < 8 or width < 8:
        return None

    gray8 = _normalize_gray_uint8(gray)
    if gray8 is None:
        return None
    blurred = cv2.GaussianBlur(gray8, (5, 5), 0)
    edges = _auto_canny(blurred)
    if int(np.count_nonzero(edges)) < 12:
        edges = _gradient_edge_mask(blurred)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    best: Optional[tuple[float, CurvatureResult]] = None
    for contour in contours:
        points = contour.reshape(-1, 2).astype(np.float32)
        if len(points) < 16:
            continue
        if cv2.arcLength(contour, False) < 12.0:
            continue
        candidate = _curvature_from_contour(points, width, height)
        if candidate is None:
            continue
        score = candidate.quality * math.sqrt(len(candidate.fit_points))
        if best is None or score > best[0]:
            best = (score, candidate)
    return best[1] if best is not None else None


def _normalize_gray_uint8(gray: np.ndarray) -> Optional[np.ndarray]:
    values = gray.astype(np.float32)
    finite = np.isfinite(values)
    if not np.any(finite):
        return None
    min_value = float(np.nanmin(values[finite]))
    max_value = float(np.nanmax(values[finite]))
    if max_value - min_value < 1e-6:
        return None
    scaled = (values - min_value) * (255.0 / (max_value - min_value))
    return np.clip(scaled, 0, 255).astype(np.uint8)


def _auto_canny(gray8: np.ndarray) -> np.ndarray:
    median = float(np.median(gray8))
    lower = int(max(5, 0.66 * median))
    upper = int(min(255, max(lower + 10, 1.33 * median)))
    edges = cv2.Canny(gray8, lower, upper, L2gradient=True)
    kernel = np.ones((2, 2), dtype=np.uint8)
    return cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)


def _gradient_edge_mask(gray8: np.ndarray) -> np.ndarray:
    values = gray8.astype(np.float32)
    gy, gx = np.gradient(values)
    magnitude = np.hypot(gx, gy)
    finite = np.isfinite(magnitude)
    if not np.any(finite):
        return np.zeros(gray8.shape, dtype=np.uint8)
    threshold = float(np.percentile(magnitude[finite], 92.0))
    mask = (magnitude >= threshold).astype(np.uint8) * 255
    kernel = np.ones((2, 2), dtype=np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def _curvature_from_contour(points: np.ndarray, width: int, height: int) -> Optional[CurvatureResult]:
    points = _smooth_contour_points(points)
    count = int(points.shape[0])
    if count < 16:
        return None
    closed = bool(np.linalg.norm(points[0] - points[-1]) <= 2.0)
    step = max(3, min(14, count // 12))
    scores: list[tuple[float, int]] = []
    start_index = 0 if closed else step
    end_index = count if closed else count - step
    for index in range(start_index, end_index):
        p0 = points[(index - step) % count]
        p1 = points[index]
        p2 = points[(index + step) % count]
        a = p0 - p1
        b = p2 - p1
        len_a = float(np.linalg.norm(a))
        len_b = float(np.linalg.norm(b))
        if len_a < 1e-6 or len_b < 1e-6:
            continue
        cosine = float(np.clip(np.dot(a, b) / (len_a * len_b), -1.0, 1.0))
        turn = abs(math.pi - math.acos(cosine))
        scores.append((turn, index))
    if not scores:
        return None

    max_radius = max(float(width), float(height)) * 2.0
    neighborhood = max(6, min(24, step + 2))
    margin = max(4.0, min(float(width), float(height)) * 0.05)
    for _, index in sorted(scores, reverse=True)[:80]:
        apex_x = float(points[index, 0])
        apex_y = float(points[index, 1])
        if apex_x <= margin or apex_y <= margin or apex_x >= width - 1 - margin or apex_y >= height - 1 - margin:
            continue
        fit_points = _contour_window(points, index, neighborhood, closed)
        if fit_points.shape[0] < 8:
            continue
        fit = _fit_circle(fit_points)
        if fit is None:
            continue
        center_x, center_y, radius, residual = fit
        if not (2.0 <= radius <= max_radius):
            continue
        if residual > 0.35:
            continue
        center = (float(center_x), float(center_y))
        apex = (apex_x, apex_y)
        sampled_edge = _sample_points(points, 220)
        sampled_fit = _sample_points(fit_points, 80)
        quality = float(max(0.0, min(1.0, 1.0 - residual)))
        return CurvatureResult(
            center=center,
            radius_px=float(radius),
            apex=apex,
            edge_points=sampled_edge,
            fit_points=sampled_fit,
            quality=quality,
        )
    return None


def _smooth_contour_points(points: np.ndarray) -> np.ndarray:
    if points.shape[0] < 7:
        return points.astype(np.float32)
    kernel = np.array([1, 2, 3, 2, 1], dtype=np.float32)
    kernel /= kernel.sum()
    pad = len(kernel) // 2
    x = np.convolve(np.pad(points[:, 0], (pad, pad), mode="edge"), kernel, mode="valid")
    y = np.convolve(np.pad(points[:, 1], (pad, pad), mode="edge"), kernel, mode="valid")
    return np.column_stack([x, y]).astype(np.float32)


def _contour_window(points: np.ndarray, index: int, radius: int, closed: bool) -> np.ndarray:
    count = int(points.shape[0])
    if closed:
        indexes = [(index + offset) % count for offset in range(-radius, radius + 1)]
        return points[indexes]
    left = max(0, index - radius)
    right = min(count, index + radius + 1)
    return points[left:right]


def _fit_circle(points: np.ndarray) -> Optional[tuple[float, float, float, float]]:
    if points.shape[0] < 3:
        return None
    x = points[:, 0].astype(np.float64)
    y = points[:, 1].astype(np.float64)
    matrix = np.column_stack([2.0 * x, 2.0 * y, np.ones_like(x)])
    rhs = x * x + y * y
    try:
        center_x, center_y, c = np.linalg.lstsq(matrix, rhs, rcond=None)[0]
    except np.linalg.LinAlgError:
        return None
    radius_sq = float(center_x * center_x + center_y * center_y + c)
    if radius_sq <= 0:
        return None
    radius = math.sqrt(radius_sq)
    distances = np.hypot(x - center_x, y - center_y)
    residual = float(np.sqrt(np.mean((distances - radius) ** 2)) / max(radius, 1e-6))
    return float(center_x), float(center_y), float(radius), residual


def _sample_points(points: np.ndarray, limit: int) -> list[Point]:
    if points.shape[0] <= limit:
        return [(float(x), float(y)) for x, y in points]
    indexes = np.linspace(0, points.shape[0] - 1, limit).round().astype(int)
    return [(float(points[index, 0]), float(points[index, 1])) for index in indexes]


def _boundary_profile_index(
    offsets: np.ndarray,
    intensity_profile: np.ndarray,
    gradient_profile: np.ndarray,
    boundary_mode: str,
) -> int:
    mode = str(boundary_mode or "max_gradient")
    if mode == "brightest":
        return _nanarg_profile(intensity_profile, maximize=True)
    if mode == "darkest":
        return _nanarg_profile(intensity_profile, maximize=False)
    if mode == "left_gradient":
        return _nanarg_profile(gradient_profile, maximize=True, mask=offsets < 0)
    if mode == "right_gradient":
        return _nanarg_profile(gradient_profile, maximize=True, mask=offsets > 0)
    return _nanarg_profile(gradient_profile, maximize=True)


def _nanarg_profile(values: np.ndarray, maximize: bool, mask: Optional[np.ndarray] = None) -> int:
    valid = np.isfinite(values)
    if mask is not None:
        valid &= mask
    if not np.any(valid):
        valid = np.isfinite(values)
    if not np.any(valid):
        return 0
    indexes = np.flatnonzero(valid)
    subset = values[indexes]
    selected = int(np.nanargmax(subset) if maximize else np.nanargmin(subset))
    return int(indexes[selected])


def _segment_pixel_patch(
    image: np.ndarray,
    start: Point,
    end: Point,
    offsets: np.ndarray,
    distances: np.ndarray,
    left_radius_px: int,
    right_radius_px: int,
) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    intensity = _scalar_intensity(image)
    h, w = intensity.shape[:2]
    length = line_length(start, end)
    if length < 2:
        return None

    tx = (end[0] - start[0]) / length
    ty = (end[1] - start[1]) / length
    nx, ny = normal_for_line(start, end)
    if nx == 0 and ny == 0:
        return None

    corners = np.array(
        [
            (start[0] + nx * left_radius_px, start[1] + ny * left_radius_px),
            (end[0] + nx * left_radius_px, end[1] + ny * left_radius_px),
            (end[0] - nx * right_radius_px, end[1] - ny * right_radius_px),
            (start[0] - nx * right_radius_px, start[1] - ny * right_radius_px),
        ],
        dtype=np.float32,
    )
    min_x = max(0, int(math.floor(float(np.min(corners[:, 0])))) - 1)
    max_x = min(w - 1, int(math.ceil(float(np.max(corners[:, 0])))) + 1)
    min_y = max(0, int(math.floor(float(np.min(corners[:, 1])))) - 1)
    max_y = min(h - 1, int(math.ceil(float(np.max(corners[:, 1])))) + 1)
    if min_x > max_x or min_y > max_y:
        return None

    xs = np.arange(min_x, max_x + 1, dtype=np.float32)
    ys = np.arange(min_y, max_y + 1, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xs, ys)
    rel_x = grid_x - float(start[0])
    rel_y = grid_y - float(start[1])
    along = rel_x * tx + rel_y * ty
    signed_offset = rel_x * nx + rel_y * ny
    mask = (
        (along >= 0.0)
        & (along <= float(length))
        & (signed_offset >= -float(right_radius_px))
        & (signed_offset <= float(left_radius_px))
    )
    if not np.any(mask):
        return None

    offset_indexes = np.rint(signed_offset[mask]).astype(np.int32) - int(offsets[0])
    distance_indexes = np.rint(along[mask]).astype(np.int32)
    valid = (
        (offset_indexes >= 0)
        & (offset_indexes < offsets.size)
        & (distance_indexes >= 0)
        & (distance_indexes < distances.size)
    )
    if not np.any(valid):
        return None

    offset_indexes = offset_indexes[valid]
    distance_indexes = distance_indexes[valid]
    pixel_x = grid_x[mask].astype(np.int32)[valid]
    pixel_y = grid_y[mask].astype(np.int32)[valid]

    sample_sums = np.zeros((offsets.size, distances.size), dtype=np.float32)
    counts = np.zeros((offsets.size, distances.size), dtype=np.float32)
    np.add.at(sample_sums, (offset_indexes, distance_indexes), intensity[pixel_y, pixel_x])
    np.add.at(counts, (offset_indexes, distance_indexes), 1.0)
    sample_grid = np.divide(sample_sums, counts, out=np.full_like(sample_sums, np.nan), where=counts > 0)

    intensity_change = _normal_intensity_change(intensity, nx, ny)
    gradient_sums = np.zeros((offsets.size, distances.size), dtype=np.float32)
    np.add.at(gradient_sums, (offset_indexes, distance_indexes), intensity_change[pixel_y, pixel_x])
    gradient_grid = np.divide(gradient_sums, counts, out=np.full_like(gradient_sums, np.nan), where=counts > 0)
    return sample_grid, counts, gradient_grid


def _row_mean_profile(
    sample_grid: np.ndarray,
    counts: np.ndarray,
    offsets: np.ndarray,
    min_valid_samples: int,
) -> Optional[np.ndarray]:
    row_counts = counts.sum(axis=1)
    row_sums = np.nansum(sample_grid * counts, axis=1)
    profile = np.divide(row_sums, row_counts, out=np.full(offsets.shape, np.nan, dtype=np.float32), where=row_counts >= min_valid_samples)
    finite = np.isfinite(profile)
    if np.count_nonzero(finite) < 5:
        return None
    if not np.all(finite):
        profile = np.interp(offsets, offsets[finite], profile[finite]).astype(np.float32)
    if profile.size >= 5:
        kernel = np.array([1, 2, 3, 2, 1], dtype=np.float32)
        kernel /= kernel.sum()
        profile = np.convolve(np.pad(profile, (2, 2), mode="edge"), kernel, mode="valid").astype(np.float32)
    return profile


def _area_gradient_profile(
    gradient_grid: np.ndarray,
    counts: np.ndarray,
    offsets: np.ndarray,
    min_valid_samples: int,
) -> Optional[np.ndarray]:
    if gradient_grid.ndim != 2 or gradient_grid.shape[0] < 3:
        return None

    row_counts = counts.sum(axis=1)
    row_sums = np.nansum(gradient_grid * counts, axis=1)
    profile = np.divide(row_sums, row_counts, out=np.full(offsets.shape, np.nan, dtype=np.float32), where=row_counts >= min_valid_samples)
    finite_profile = np.isfinite(profile)
    if np.count_nonzero(finite_profile) < 5:
        return None
    if not np.all(finite_profile):
        profile = np.interp(offsets, offsets[finite_profile], profile[finite_profile]).astype(np.float32)
    if profile.size >= 5:
        kernel = np.array([1, 2, 3, 2, 1], dtype=np.float32)
        kernel /= kernel.sum()
        profile = np.convolve(np.pad(profile, (2, 2), mode="edge"), kernel, mode="valid").astype(np.float32)
    return profile


def _resolve_search_radii(
    search_radius_px: int,
    search_radius_left_px: Optional[int],
    search_radius_right_px: Optional[int],
) -> tuple[int, int]:
    symmetric_radius = max(0, int(search_radius_px))
    left_radius = symmetric_radius if search_radius_left_px is None else max(0, int(search_radius_left_px))
    right_radius = symmetric_radius if search_radius_right_px is None else max(0, int(search_radius_right_px))
    return left_radius, right_radius


def _search_offsets(left_radius_px: int, right_radius_px: int) -> np.ndarray:
    return np.arange(-right_radius_px, left_radius_px + 1, dtype=np.float32)


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
    segment_size_px: float = 9.0,
    search_radius_left_px: Optional[int] = None,
    search_radius_right_px: Optional[int] = None,
    boundary_mode: str = "max_gradient",
) -> Optional[SnapCurveResult]:
    """Trace a connected polyline boundary near user-drawn connected segments.

    Segment size is the target distance in pixels between recognition points.
    Smaller values create denser segments and follow local changes more closely.
    """
    left_radius_px, right_radius_px = _resolve_search_radii(
        search_radius_px,
        search_radius_left_px,
        search_radius_right_px,
    )
    if left_radius_px <= 0 and right_radius_px <= 0:
        return None
    if len(points) < 2:
        return None

    step_px = float(np.clip(segment_size_px, 2.0, 80.0))
    sampled_points = _resample_polyline(points, step_px)
    if len(sampled_points) < 2:
        return None
    if len(sampled_points) > 240:
        indexes = np.linspace(0, len(sampled_points) - 1, 240).round().astype(int)
        sampled_points = [sampled_points[int(index)] for index in indexes]
    point_count = len(sampled_points)
    offsets = _search_offsets(left_radius_px, right_radius_px)
    if offsets.size < 3:
        return None

    segment_count = point_count - 1
    segment_offsets = np.full(segment_count, np.nan, dtype=np.float32)
    segment_strengths = np.full(segment_count, np.nan, dtype=np.float32)
    segment_normals: list[Point] = []
    for idx, (start, end) in enumerate(zip(sampled_points, sampled_points[1:])):
        nx, ny = normal_for_line(start, end)
        segment_normals.append((nx, ny))
        if nx == 0 and ny == 0:
            continue
        profile = segment_brightness_profile(
            gray,
            start,
            end,
            search_radius_px,
            left_radius_px,
            right_radius_px,
            boundary_mode,
        )
        if profile is None:
            continue
        segment_offsets[idx] = profile.best_offset_px
        segment_strengths[idx] = profile.best_gradient

    finite_offsets = np.isfinite(segment_offsets)
    if np.count_nonzero(finite_offsets) < 1:
        return None
    if not np.all(finite_offsets):
        valid_x = np.flatnonzero(finite_offsets)
        segment_offsets = np.interp(np.arange(segment_count), valid_x, segment_offsets[finite_offsets]).astype(np.float32)
        segment_strengths = np.interp(np.arange(segment_count), valid_x, segment_strengths[finite_offsets]).astype(np.float32)

    smoothing_window = int(round(step_px / 4.0)) * 2 + 1
    smoothing_window = int(np.clip(smoothing_window, 1, 15))
    if smoothing_window > 1 and segment_offsets.size > 1:
        kernel = np.ones(smoothing_window, dtype=np.float32) / smoothing_window
        pad = smoothing_window // 2
        segment_offsets = np.convolve(np.pad(segment_offsets, (pad, pad), mode="edge"), kernel, mode="valid").astype(np.float32)

    shifted_segments: list[Line] = []
    for (start, end), (nx, ny), offset in zip(zip(sampled_points, sampled_points[1:]), segment_normals, segment_offsets):
        shifted_segments.append(
            (
                (float(start[0] + nx * offset), float(start[1] + ny * offset)),
                (float(end[0] + nx * offset), float(end[1] + ny * offset)),
            )
        )

    points = [shifted_segments[0][0]]
    for idx in range(1, point_count - 1):
        prev_end = shifted_segments[idx - 1][1]
        next_start = shifted_segments[idx][0]
        points.append(((prev_end[0] + next_start[0]) / 2.0, (prev_end[1] + next_start[1]) / 2.0))
    points.append(shifted_segments[-1][1])

    vertex_offsets = np.empty(point_count, dtype=np.float32)
    vertex_strengths = np.empty(point_count, dtype=np.float32)
    vertex_offsets[0] = segment_offsets[0]
    vertex_strengths[0] = segment_strengths[0]
    vertex_offsets[-1] = segment_offsets[-1]
    vertex_strengths[-1] = segment_strengths[-1]
    for idx in range(1, point_count - 1):
        vertex_offsets[idx] = (segment_offsets[idx - 1] + segment_offsets[idx]) / 2.0
        vertex_strengths[idx] = (segment_strengths[idx - 1] + segment_strengths[idx]) / 2.0

    return SnapCurveResult(points=points, offsets=vertex_offsets, gradient_strength=vertex_strengths)


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
