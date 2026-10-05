"""Entrance-anchored representative radii, in original image coordinates.

Trace the material-to-void transition along rays spanning the top and wall.
This avoids selecting a high-curvature texture contour inside the material.
The residual is a fit diagnostic, not metrological uncertainty.
"""
from __future__ import annotations

import math
import cv2
import numpy as np

from .image_ops import to_gray


def fit_circle(points: np.ndarray):
    """Deterministic Huber refinement of geometric radial distances."""
    if len(points) < 8 or not np.isfinite(points).all():
        return None
    origin = points.mean(axis=0)
    p = points - origin
    matrix = np.column_stack((2 * p, np.ones(len(p))))
    solution, _, rank, _ = np.linalg.lstsq(matrix, np.sum(p * p, axis=1), rcond=None)
    if rank < 3:
        return None
    center = solution[:2]
    radius = math.sqrt(max(0, solution[2] + center @ center))
    for _ in range(25):
        delta = center - p
        distance = np.linalg.norm(delta, axis=1)
        if radius <= 0 or np.any(distance < 1e-9):
            return None
        error = distance - radius
        scale = max(.15, 1.4826 * np.median(np.abs(error - np.median(error))))
        weights = np.minimum(1, 1.5 * scale / np.maximum(np.abs(error), 1e-9))
        jac = np.column_stack((delta / distance[:, None], -np.ones(len(p))))
        step = np.linalg.lstsq(jac * np.sqrt(weights[:, None]), -error * np.sqrt(weights), rcond=None)[0]
        center += step[:2]
        radius += step[2]
        if np.linalg.norm(step) < 1e-7:
            break
    if radius <= 0 or not np.isfinite(radius) or radius > 1e7:
        return None
    error = np.linalg.norm(p - center, axis=1) - radius
    return center + origin, float(radius), float(np.sqrt(np.mean(error ** 2)))


def _resample(points):
    lengths = np.r_[0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    keep = np.r_[True, np.diff(lengths) > 1e-5]
    positions = np.arange(0, lengths[-1], 1.0)
    return np.column_stack([np.interp(positions, lengths[keep], points[keep, i]) for i in (0, 1)])


def _monotone(values):
    # Pool adjacent violations: the representative shoulder turns from the
    # surface into the wall once, while texture creates oscillating tangents.
    blocks = []
    for value in values:
        blocks.append([float(value), 1])
        while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0]:
            b, a = blocks.pop(), blocks.pop()
            count = a[1] + b[1]
            blocks.append([(a[0] * a[1] + b[0] * b[1]) / count, count])
    return np.concatenate([np.full(count, value) for value, count in blocks])


def _trace(gray, wall_x, top, reach):
    """Canonical orientation: material left/below, void right/above."""
    origin = np.array([max(4, wall_x - .8 * reach), min(gray.shape[0] - 5, top + 1.05 * reach)])
    angles = np.linspace(-math.pi / 2, 0, max(180, int(reach * 3)))
    directions = np.column_stack((np.cos(angles), np.sin(angles)))
    top_dist = np.where(directions[:, 1] < -.01, (top - origin[1]) / np.minimum(directions[:, 1], -.01), np.inf)
    wall_dist = np.where(directions[:, 0] > .01, (wall_x - origin[0]) / np.maximum(directions[:, 0], .01), np.inf)
    predicted = np.minimum(top_dist, wall_dist)
    offsets = np.arange(-.65 * reach, .25 * reach + .25, .25)
    distances = predicted[:, None] + offsets
    coords = origin + directions[:, None, :] * distances[:, :, None]
    intensities = cv2.remap(gray, coords[:, :, 0].astype(np.float32), coords[:, :, 1].astype(np.float32),
                            cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    strength = -np.gradient(intensities, .25, axis=1)
    inside = ((coords[:, :, 0] > 1) & (coords[:, :, 0] < gray.shape[1] - 2) &
              (coords[:, :, 1] > 1) & (coords[:, :, 1] < gray.shape[0] - 2) & (distances > 2))
    strength[~inside] = 0
    peaks = (strength >= np.roll(strength, 1, axis=1)) & (strength >= np.roll(strength, -1, axis=1))
    peaks[:, [0, -1]] = False
    scores = np.where(peaks, strength, 0)
    # Several nearby edge candidates, with continuity enforced across rays.
    indexes = np.argsort(scores, axis=1)[:, -8:]
    values = np.take_along_axis(scores, indexes, axis=1)
    points = np.take_along_axis(coords, indexes[:, :, None], axis=1)
    cost = -values / np.maximum(values.max(axis=1, keepdims=True), 1)
    cost += .12 * np.abs(np.take_along_axis(distances, indexes, axis=1) - predicted[:, None]) / reach
    cost[values < 3] += 100
    back = np.zeros(indexes.shape, dtype=int)
    accumulated = cost[0].copy()
    for i in range(1, len(indexes)):
        jumps = np.linalg.norm(points[i, :, None, :] - points[i - 1, None, :, :], axis=2)
        transition = accumulated[None, :] + .2 * jumps ** 2
        back[i] = np.argmin(transition, axis=1)
        accumulated = cost[i] + transition[np.arange(8), back[i]]
    chosen = [int(np.argmin(accumulated))]
    for i in range(len(indexes) - 1, 0, -1):
        chosen.append(int(back[i, chosen[-1]]))
    chosen = np.asarray(chosen[::-1])
    if np.mean(values[np.arange(len(values)), chosen] >= 3) < .95:
        return None
    trace = points[np.arange(len(points)), chosen]
    if np.max(np.linalg.norm(np.diff(trace, axis=0), axis=1)) > max(5, reach * .08):
        return None
    # Estimate image resolution from the straight surface/wall supports. A
    # blurred sharp corner can otherwise supply many apparently circular
    # points. The derivative FWHM of a Gaussian edge corresponds to a 10--90%
    # edge spread of 1.088 * FWHM; this is an observed blur diagnostic, not an
    # estimate of the microscope's physical resolution.
    spreads = []
    support_count = max(5, len(angles) // 10)
    for i in list(range(support_count)) + list(range(len(angles) - support_count, len(angles))):
        k = indexes[i, chosen[i]]
        profile = strength[i]
        half = profile[k] * .5
        if half < 1.5:
            continue
        left = right = int(k)
        while left > 0 and profile[left] >= half:
            left -= 1
        while right < len(profile) - 1 and profile[right] >= half:
            right += 1
        if left == 0 or right == len(profile) - 1:
            continue
        left_cross = left + (half - profile[left]) / max(1e-9, profile[left + 1] - profile[left])
        right_cross = right - (half - profile[right]) / max(1e-9, profile[right - 1] - profile[right])
        normal_projection = abs(directions[i, 1 if i < support_count else 0])
        spreads.append(float((right_cross - left_cross) * .25 * 1.088 * normal_projection))
    edge_spread = float(np.median(spreads)) if len(spreads) >= 6 else None
    return _resample(trace), edge_spread


def measure_entrance_corner(image, side, entrance_y, wall_y, wall_x, window=24,
                            roi=None, span=None, bright=False):
    """Return a diagnostic record even when a radius cannot be determined.

    span is an optional pair of fractions along the detected rounded transition,
    stored independently for each side. roi is in full-image pixel coordinates.
    """
    reach = max(32, int(window) * 3)
    manual = roi is not None or span is not None
    anchor_y = min(float(wall_y[-1]), entrance_y + reach * 1.5)
    anchor_x = float(np.interp(anchor_y, wall_y, wall_x))
    if roi is None:
        x0, x1 = max(0, int(anchor_x - reach * 1.8)), min(image.shape[1], int(anchor_x + reach * .7))
        if side == "right":
            x0, x1 = max(0, int(anchor_x - reach * .7)), min(image.shape[1], int(anchor_x + reach * 1.8))
        y0, y1 = max(0, int(entrance_y - reach * .5)), min(image.shape[0], int(entrance_y + reach * 1.9))
        roi = (x0, y0, x1 - x0, y1 - y0)
    else:
        reach = max(12, min(reach, int(roi[2] * .65), int(roi[3] * .65)))
        anchor_y = min(float(wall_y[-1]), roi[1] + roi[3] * .8)
        anchor_x = float(np.interp(anchor_y, wall_y, wall_x))
    roi = tuple(int(v) for v in roi)
    x0, y0, width, height = roi
    result = dict(side=side, roi=list(roi), selection=list(span or (0, 1)),
                  source="manual" if manual else "auto", status="unconfirmed",
                  reason="입구의 윗면과 측벽이 함께 보이도록 검색 영역을 조절하세요.",
                  radius_px=None, center=None, rms_px=None, arc_angle_deg=None,
                  stability_fraction=None, edge_spread_px=None,
                  fit_points=[], trace_points=[], arc_points=[])
    if x0 < 0 or y0 < 0 or width < 12 or height < 16 or x0 + width > image.shape[1] or y0 + height > image.shape[0]:
        result["reason"] = "검색 영역이 이미지 범위를 벗어나거나 너무 작습니다."
        return result
    gray = to_gray(image[y0:y0 + height, x0:x0 + width]).astype(np.float32)
    if not np.isfinite(gray).all():
        return result
    low, high = np.percentile(gray, (1, 99))
    if high - low < 1e-6:
        return result
    gray = np.clip((gray - low) * (255 / (high - low)), 0, 255)
    if bright:
        gray = 255 - gray
    gray = cv2.GaussianBlur(gray, (5, 5), .8)
    local_wall = anchor_x - x0
    if side == "right":
        gray = np.ascontiguousarray(gray[:, ::-1])
        local_wall = width - 1 - local_wall
    if span is not None and (len(span) != 2 or not np.isfinite(span).all() or not 0 <= span[0] < span[1] <= 1):
        result["reason"] = "원호 시작·끝 범위가 올바르지 않습니다."
        return result
    traced = _trace(gray, local_wall, entrance_y - y0, reach)
    if traced is None:
        return result
    trace, edge_spread = traced
    result["edge_spread_px"] = edge_spread
    if len(trace) < 15:
        return result
    smoothed = np.column_stack([cv2.GaussianBlur(trace[:, i:i + 1], (1, 7), 1.1).ravel() for i in (0, 1)])
    tangent = np.gradient(smoothed, axis=0)
    turns = _monotone(np.clip(np.degrees(np.arctan2(tangent[:, 1], tangent[:, 0])), -30, 120))
    # Both straight supports must be present; a layer or a cropped arc cannot
    # become a valid entrance just because a circle happens to fit locally.
    if abs(float(np.median(turns[2:8]))) > 20 or abs(float(np.median(turns[-8:-2])) - 90) > 20:
        return result
    a_candidates = np.flatnonzero(turns > 8)
    b_candidates = np.flatnonzero(turns < 82)
    if not len(a_candidates) or not len(b_candidates):
        return result
    a, b = max(0, int(a_candidates[0]) - 2), min(len(trace) - 1, int(b_candidates[-1]) + 2)
    # Require the support near the specified entrance, not a lower material layer.
    if abs(np.median(trace[:max(5, a), 1]) - (entrance_y - y0)) > max(6, reach * .2):
        return result
    rounded = trace[a:b + 1].copy()
    if side == "right":
        rounded[:, 0] = width - 1 - rounded[:, 0]
    rounded += (x0, y0)
    result["trace_points"] = rounded.tolist()
    result["transition_point_count"] = int(b_candidates[-1] - a_candidates[0] + 1)
    start, end = span or (0, 1)
    points = rounded[int(round(start * max(0, len(rounded) - 1))):int(round(end * max(0, len(rounded) - 1))) + 1]
    result["fit_points"] = points.tolist()
    if result["transition_point_count"] < 8:
        result["reason"] = ("경계 흐림 폭에 비해 둥근 전이 윤곽이 부족합니다. 실제 곡률과 흐림을 구분할 수 없습니다."
                            if edge_spread is not None and edge_spread>5 else
                            "둥근 전이 윤곽이 8개 픽셀 미만입니다. 해상도가 부족하거나 날카로운 모서리입니다.")
        return result
    fit = fit_circle(points)
    if fit is None:
        result["reason"] = "원호 윤곽점이 8개 미만이거나 원을 결정할 수 없습니다."
        return result
    center, radius, rms = fit
    angles = np.unwrap(np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0]))
    angle_span = abs(float(np.degrees(angles[-1] - angles[0])))
    trim = max(1, int(round((len(points) - 1) * .1)))
    variations = [fit_circle(p) for p in (points[trim:], points[:-trim])]
    stability = max((abs(f[1] - radius) / radius if f else 1) for f in variations)
    result.update(rms_px=rms, arc_angle_deg=angle_span, stability_fraction=stability)
    if angle_span < 45:
        result["reason"] = "원호 각도가 45° 미만입니다. 선택 구간을 넓혀주세요."
    elif rms > max(1, radius * .05):
        result["reason"] = "윤곽과 원호의 맞춤 오차가 기준을 초과합니다."
    elif stability > .2:
        result["reason"] = "구간을 10% 조정하면 R이 20% 넘게 달라집니다."
    elif radius < 3:
        result["reason"] = "모서리의 둥근 구간이 해상도에 비해 너무 작습니다."
    elif edge_spread is not None and radius < 2 * edge_spread:
        result["reason"] = "R이 경계 흐림 폭의 2배 미만입니다. 실제 곡률과 흐림에 의한 둥글어짐을 구분할 수 없습니다."
    else:
        theta = np.linspace(angles[0], angles[-1], 80)
        arc = center + radius * np.column_stack((np.cos(theta), np.sin(theta)))
        result.update(status="valid", reason="유효 · 입구 전이 구간의 대표 원 맞춤", radius_px=radius,
                      center=center.tolist(), arc_points=arc.tolist())
    return result
