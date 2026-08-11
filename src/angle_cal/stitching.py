from __future__ import annotations

from dataclasses import dataclass, replace
import json
import math
from pathlib import Path
from typing import Callable, Iterable

import cv2
import numpy as np


class StitchingError(RuntimeError):
    pass


class StitchingCancelled(StitchingError):
    pass


class StitchingNeedsManual(StitchingError):
    def __init__(self, message, suggested_pair=None):
        super().__init__(message)
        self.suggested_pair = suggested_pair


@dataclass(frozen=True)
class StitchOptions:
    overlay_crop_fraction: float = 0.0
    min_matches: int = 8
    ratio_test: float = 0.75
    ransac_threshold: float = 3.0
    min_inlier_ratio: float = 0.22
    min_feature_coverage: float = 0.012
    min_pixel_correlation: float = 0.20
    board_tolerance_fraction: float = 0.28


@dataclass(frozen=True)
class StitchLayoutHint:
    """Approximate source-image placement supplied by the merge board.

    ``source_to_board`` maps original source pixels to board scene coordinates.
    It is a hint, not an output transform; automatic registration must still
    pass feature and pixel-domain validation.
    """

    path: str
    source_to_board: np.ndarray
    match_rect: tuple[float, float, float, float] | None = None


@dataclass
class StitchPlacement:
    path: str
    transform: np.ndarray
    mode: str
    inlier_count: int = 0
    reprojection_error: float = 0.0


@dataclass
class StitchResult:
    image: np.ndarray
    valid_mask: np.ndarray
    placements: list[StitchPlacement]
    output_size: tuple[int, int]


@dataclass
class _PairAlignment:
    source: int
    target: int
    matrix: np.ndarray
    inlier_count: int
    reprojection_error: float
    confidence: float
    mode: str


def read_raw_image(path):
    data = np.fromfile(str(path), np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise StitchingError(f"이미지를 읽을 수 없습니다: {path}")
    return image


def _gray8(image):
    gray = cv2.cvtColor(image[..., :3], cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    if gray.dtype == np.uint8:
        return gray
    lo, hi = np.percentile(gray, (1, 99))
    if hi <= lo:
        return np.zeros(gray.shape, np.uint8)
    return np.clip((gray.astype(np.float32) - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)


def detect_bottom_overlay_fraction(images):
    if not images:
        return 0.0

    # SEM/TEM footers often contain text and a scale bar, so they are not
    # necessarily low-variance.  Their stronger signature is that the same
    # bottom rows repeat across otherwise different captures.
    if len(images) >= 2:
        previews = []
        for image in images:
            gray = cv2.resize(_gray8(image), (320, 320), interpolation=cv2.INTER_AREA).astype(np.float32)
            low, high = np.percentile(gray, (2, 98))
            if high > low:
                gray = np.clip((gray - low) * 255.0 / (high - low), 0, 255)
            previews.append(gray)
        stack = np.stack(previews)
        center = np.median(stack, axis=0)
        row_disagreement = np.median(np.abs(stack - center), axis=(0, 2))
        row_disagreement = np.convolve(row_disagreement, np.ones(7) / 7.0, mode="same")
        window = 20
        best: tuple[float, int] | None = None
        for y in range(int(320 * 0.55), int(320 * 0.94)):
            above = float(np.median(row_disagreement[max(0, y - window) : y]))
            below = float(np.median(row_disagreement[y : min(320, y + max(window, (320 - y) // 2))]))
            drop = above - below
            if above >= 3.0 and below <= above * 0.68 and drop >= 2.0:
                score = drop / max(above, 1e-6)
                if best is None or score > best[0]:
                    best = (score, y)
        if best is not None:
            fraction = (320 - best[1]) / 320.0
            if 0.025 <= fraction <= 0.45:
                return float(fraction)

    # Fallback for a single image or a footer whose text differs per capture.
    found = []
    for image in images:
        gray = _gray8(image)
        height = gray.shape[0]
        row_std = gray.std(1)
        for y in range(height - 2, int(height * 0.55), -1):
            reference = row_std[max(0, y - height // 15) : y].mean()
            if row_std[y:].mean() < max(4.0, reference * 0.4):
                found.append((height - y) / height)
                break
    return float(np.median(found)) if found else 0.0


def alignment_from_manual_points(source, target):
    if len(source) != len(target) or len(source) < 4:
        raise StitchingError("수동 기준점은 양쪽에 같은 개수로 4개 이상 필요합니다.")
    matrix, _ = cv2.findHomography(np.float32(source), np.float32(target), 0)
    if matrix is None:
        raise StitchingError("기준점 변환을 계산하지 못했습니다.")
    return matrix


def _ratio_matches(descriptors_a, descriptors_b, ratio: float) -> dict[int, cv2.DMatch]:
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    accepted: dict[int, cv2.DMatch] = {}
    for pair in matcher.knnMatch(descriptors_a, descriptors_b, k=2):
        if len(pair) == 2 and pair[0].distance < ratio * pair[1].distance:
            accepted[pair[0].queryIdx] = pair[0]
    return accepted


def _mutual_matches(descriptors_a, descriptors_b, ratio: float) -> list[cv2.DMatch]:
    forward = _ratio_matches(descriptors_a, descriptors_b, ratio)
    backward = _ratio_matches(descriptors_b, descriptors_a, ratio)
    return [
        match
        for match in forward.values()
        if (reverse := backward.get(match.trainIdx)) is not None and reverse.trainIdx == match.queryIdx
    ]


def _hint_map(layout_hints: Iterable[StitchLayoutHint] | None) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for hint in layout_hints or []:
        matrix = np.asarray(hint.source_to_board, dtype=np.float64)
        if matrix.shape == (3, 3) and np.isfinite(matrix).all() and abs(np.linalg.det(matrix)) > 1e-12:
            result[str(Path(hint.path).resolve()).casefold()] = matrix
    return result


def _match_region_map(
    layout_hints: Iterable[StitchLayoutHint] | None,
) -> dict[str, tuple[float, float, float, float]]:
    result: dict[str, tuple[float, float, float, float]] = {}
    for hint in layout_hints or []:
        if hint.match_rect is None or len(hint.match_rect) != 4:
            continue
        x, y, width, height = (float(value) for value in hint.match_rect)
        if not np.isfinite([x, y, width, height]).all():
            continue
        x = min(1.0, max(0.0, x))
        y = min(1.0, max(0.0, y))
        width = min(1.0 - x, max(0.01, width))
        height = min(1.0 - y, max(0.01, height))
        result[str(Path(hint.path).resolve()).casefold()] = (x, y, width, height)
    return result


def _match_mask(
    image: np.ndarray,
    region: tuple[float, float, float, float] | None,
    bottom_crop_fraction: float,
) -> np.ndarray:
    height, width = image.shape[:2]
    mask = np.zeros((height, width), np.uint8)
    if region is None:
        left, top, right, bottom = 0, 0, width, height
    else:
        x, y, region_width, region_height = region
        left = int(round(x * width))
        top = int(round(y * height))
        right = int(round((x + region_width) * width))
        bottom = int(round((y + region_height) * height))
    bottom = min(bottom, max(1, int(round(height * (1.0 - bottom_crop_fraction)))))
    left, right = max(0, left), min(width, right)
    top, bottom = max(0, top), min(height, bottom)
    if right > left and bottom > top:
        mask[top:bottom, left:right] = 255
    return mask


def _prior_for_pair(path_a: str, path_b: str, hints: dict[str, np.ndarray]) -> np.ndarray | None:
    board_a = hints.get(str(Path(path_a).resolve()).casefold())
    board_b = hints.get(str(Path(path_b).resolve()).casefold())
    if board_a is None or board_b is None:
        return None
    return np.linalg.inv(board_b) @ board_a


def _board_pair_is_nearby(
    image_a: np.ndarray,
    image_b: np.ndarray,
    board_a: np.ndarray,
    board_b: np.ndarray,
) -> bool:
    def board_bounds(image, matrix):
        height, width = image.shape[:2]
        corners = np.float32([[[0, 0], [width, 0], [width, height], [0, height]]])
        points = cv2.perspectiveTransform(corners, matrix)[0]
        return points.min(0), points.max(0)

    low_a, high_a = board_bounds(image_a, board_a)
    low_b, high_b = board_bounds(image_b, board_b)
    gap = np.maximum(0.0, np.maximum(low_a - high_b, low_b - high_a))
    size_a = high_a - low_a
    size_b = high_b - low_b
    tolerance = 0.35 * max(1.0, float(min(np.max(size_a), np.max(size_b))))
    return float(np.linalg.norm(gap)) <= tolerance


def _apply_prior_gate(
    source: np.ndarray,
    target: np.ndarray,
    prior: np.ndarray | None,
    target_shape: tuple[int, int],
    options: StitchOptions,
) -> tuple[np.ndarray, np.ndarray]:
    if prior is None or len(source) == 0:
        return source, target
    predicted = cv2.perspectiveTransform(source[:, None, :], prior)[:, 0]
    errors = np.linalg.norm(predicted - target, axis=1)
    diagonal = math.hypot(target_shape[1], target_shape[0])
    tolerance = max(32.0, options.board_tolerance_fraction * diagonal)
    keep = errors <= tolerance
    return source[keep], target[keep]


def _reprojection(matrix: np.ndarray, source: np.ndarray, target: np.ndarray, mask: np.ndarray):
    keep = np.asarray(mask).ravel().astype(bool)
    if not np.any(keep):
        return np.empty((0,), np.float32), float("inf")
    projected = cv2.perspectiveTransform(source[keep, None, :], matrix)[:, 0]
    errors = np.linalg.norm(projected - target[keep], axis=1)
    return errors, float(np.mean(errors))


def _coverage(points: np.ndarray, shape: tuple[int, int]) -> float:
    if len(points) < 3:
        return 0.0
    hull = cv2.convexHull(np.float32(points))
    return float(cv2.contourArea(hull)) / max(1.0, float(shape[0] * shape[1]))


def _geometry_quality(matrix: np.ndarray, source_shape: tuple[int, int], target_shape: tuple[int, int]):
    source_height, source_width = source_shape
    target_height, target_width = target_shape
    corners = np.float32([[[0, 0], [source_width, 0], [source_width, source_height], [0, source_height]]])
    transformed = cv2.perspectiveTransform(corners, matrix)[0]
    if not np.isfinite(transformed).all() or not cv2.isContourConvex(transformed.astype(np.float32)):
        return None
    transformed_area = abs(float(cv2.contourArea(transformed.astype(np.float32))))
    source_area = float(source_width * source_height)
    area_ratio = transformed_area / max(1.0, source_area)
    if not 0.45 <= area_ratio <= 2.2:
        return None
    target_corners = np.float32([[0, 0], [target_width, 0], [target_width, target_height], [0, target_height]])
    overlap_area, _ = cv2.intersectConvexConvex(transformed.astype(np.float32), target_corners)
    overlap_fraction = float(overlap_area) / max(1.0, min(transformed_area, float(target_width * target_height)))
    if overlap_fraction < 0.025:
        return None
    perspective = max(abs(matrix[2, 0]) * source_width, abs(matrix[2, 1]) * source_height)
    if perspective > 0.20:
        return None
    return overlap_fraction, transformed


def _prior_distance(
    matrix: np.ndarray,
    prior: np.ndarray | None,
    source_shape: tuple[int, int],
    target_shape: tuple[int, int],
) -> float:
    if prior is None:
        return 0.0
    source_height, source_width = source_shape
    points = np.float32(
        [[[0, 0], [source_width, 0], [source_width, source_height], [0, source_height], [source_width / 2, source_height / 2]]]
    )
    actual = cv2.perspectiveTransform(points, matrix)[0]
    expected = cv2.perspectiveTransform(points, prior)[0]
    diagonal = max(1.0, math.hypot(target_shape[1], target_shape[0]))
    return float(np.median(np.linalg.norm(actual - expected, axis=1))) / diagonal


def _normalized_correlation(first: np.ndarray, second: np.ndarray) -> float:
    first = first.astype(np.float32)
    second = second.astype(np.float32)
    first -= first.mean()
    second -= second.mean()
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    return -1.0 if denominator <= 1e-6 else float(np.dot(first, second) / denominator)


def _pixel_correlation(
    image_a: np.ndarray,
    image_b: np.ndarray,
    matrix: np.ndarray,
    mask_a: np.ndarray,
    mask_b: np.ndarray,
) -> float:
    gray_a = _gray8(image_a)
    gray_b = _gray8(image_b)
    warped = cv2.warpPerspective(gray_a, matrix, (gray_b.shape[1], gray_b.shape[0]), flags=cv2.INTER_LINEAR)
    valid = cv2.warpPerspective(mask_a, matrix, (gray_b.shape[1], gray_b.shape[0]), flags=cv2.INTER_NEAREST) > 0
    valid &= mask_b > 0
    valid = cv2.erode(valid.astype(np.uint8), np.ones((5, 5), np.uint8), iterations=1).astype(bool)
    if int(valid.sum()) < 256:
        return -1.0
    ys, xs = np.where(valid)
    step = max(1, int(math.sqrt(len(xs) / 250_000)))
    ys, xs = ys[::step], xs[::step]
    intensity = _normalized_correlation(warped[ys, xs], gray_b[ys, xs])
    gradient_a = cv2.Laplacian(warped, cv2.CV_32F, ksize=3)
    gradient_b = cv2.Laplacian(gray_b, cv2.CV_32F, ksize=3)
    gradient = _normalized_correlation(gradient_a[ys, xs], gradient_b[ys, xs])
    return max(intensity, gradient)


def _candidate_matrices(source: np.ndarray, target: np.ndarray, options: StitchOptions):
    candidates: list[tuple[str, np.ndarray, np.ndarray]] = []

    delta = np.median(target - source, axis=0)
    translation_errors = np.linalg.norm((source + delta) - target, axis=1)
    translation_mask = (translation_errors <= options.ransac_threshold).astype(np.uint8)
    if int(translation_mask.sum()) >= options.min_matches:
        translation = np.array([[1.0, 0.0, delta[0]], [0.0, 1.0, delta[1]], [0.0, 0.0, 1.0]])
        candidates.append(("translation", translation, translation_mask))

    affine, affine_mask = cv2.estimateAffinePartial2D(
        source,
        target,
        method=cv2.RANSAC,
        ransacReprojThreshold=options.ransac_threshold,
        maxIters=3000,
        confidence=0.995,
        refineIters=15,
    )
    if affine is not None and affine_mask is not None:
        affine_h = np.vstack((affine, [0.0, 0.0, 1.0]))
        candidates.append(("affine", affine_h, affine_mask))

    if len(source) >= max(12, options.min_matches):
        homography, homography_mask = cv2.findHomography(
            source,
            target,
            cv2.RANSAC,
            options.ransac_threshold,
            maxIters=3500,
            confidence=0.995,
        )
        if homography is not None and homography_mask is not None:
            candidates.append(("perspective", homography, homography_mask))
    return candidates


def _phase_translation(
    index_a: int,
    index_b: int,
    images: list[np.ndarray],
    match_masks: list[np.ndarray],
    options: StitchOptions,
    prior: np.ndarray | None,
) -> _PairAlignment | None:
    image_a = _gray8(images[index_a])
    image_b = _gray8(images[index_b])

    if prior is None:
        if image_a.shape != image_b.shape:
            return None
        base_x = base_y = 0
    else:
        normalized = prior / prior[2, 2]
        if not np.allclose(normalized[:2, :2], np.eye(2), atol=0.08) or not np.allclose(
            normalized[2, :2], 0.0, atol=1e-5
        ):
            return None
        base_x, base_y = np.round(normalized[:2, 2]).astype(int)

    base_matrix = np.array(
        [[1.0, 0.0, base_x], [0.0, 1.0, base_y], [0.0, 0.0, 1.0]], dtype=np.float64
    )
    warped_mask_a = cv2.warpPerspective(
        match_masks[index_a], base_matrix, (image_b.shape[1], image_b.shape[0]), flags=cv2.INTER_NEAREST
    )
    common = (warped_mask_a > 0) & (match_masks[index_b] > 0)
    points = cv2.findNonZero(common.astype(np.uint8))
    if points is None:
        return None
    left, top, width, height = cv2.boundingRect(points)
    right, bottom = left + width, top + height
    if width < 48 or height < 48:
        return None
    crop_a = image_a[top - base_y : bottom - base_y, left - base_x : right - base_x]
    crop_b = image_b[top:bottom, left:right]

    if min(crop_a.shape) < 48 or crop_a.shape != crop_b.shape:
        return None
    first = crop_a.astype(np.float32)
    second = crop_b.astype(np.float32)
    window = cv2.createHanningWindow((first.shape[1], first.shape[0]), cv2.CV_32F)
    (residual_x, residual_y), response = cv2.phaseCorrelate(first, second, window)
    if not np.isfinite([residual_x, residual_y, response]).all() or response < 0.08:
        return None
    matrix = np.array(
        [[1.0, 0.0, base_x + residual_x], [0.0, 1.0, base_y + residual_y], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    if _geometry_quality(matrix, images[index_a].shape[:2], images[index_b].shape[:2]) is None:
        return None
    prior_distance = _prior_distance(matrix, prior, images[index_a].shape[:2], images[index_b].shape[:2])
    if prior is not None and prior_distance > options.board_tolerance_fraction:
        return None
    pixel_score = _pixel_correlation(
        images[index_a], images[index_b], matrix, match_masks[index_a], match_masks[index_b]
    )
    if pixel_score < options.min_pixel_correlation:
        return None

    rounded = matrix.copy()
    rounded[:2, 2] = np.round(rounded[:2, 2])
    rounded_score = _pixel_correlation(
        images[index_a], images[index_b], rounded, match_masks[index_a], match_masks[index_b]
    )
    if rounded_score >= max(options.min_pixel_correlation, pixel_score - 0.015):
        matrix = rounded
        mode = "exact translation"
        pixel_score = rounded_score
    else:
        mode = "Lanczos phase translation"
    confidence = 0.28 + 0.32 * min(1.0, float(response)) + 0.38 * max(0.0, pixel_score) - 0.35 * prior_distance
    return _PairAlignment(index_a, index_b, matrix, 0, 0.0, confidence, mode)


def _align_pair(
    index_a: int,
    index_b: int,
    images: list[np.ndarray],
    match_masks: list[np.ndarray],
    features,
    options: StitchOptions,
    prior: np.ndarray | None,
) -> _PairAlignment | None:
    keypoints_a, descriptors_a = features[index_a]
    keypoints_b, descriptors_b = features[index_b]
    best = _phase_translation(index_a, index_b, images, match_masks, options, prior)
    if descriptors_a is None or descriptors_b is None:
        return best
    matches = _mutual_matches(descriptors_a, descriptors_b, options.ratio_test)
    if len(matches) < options.min_matches:
        return best
    source = np.float32([keypoints_a[match.queryIdx].pt for match in matches])
    target = np.float32([keypoints_b[match.trainIdx].pt for match in matches])
    source, target = _apply_prior_gate(source, target, prior, images[index_b].shape[:2], options)
    if len(source) < options.min_matches:
        return best

    for kind, matrix, mask in _candidate_matrices(source, target, options):
        inliers = np.asarray(mask).ravel().astype(bool)
        count = int(inliers.sum())
        ratio = count / max(1, len(source))
        if count < options.min_matches or ratio < options.min_inlier_ratio:
            continue
        errors, mean_error = _reprojection(matrix, source, target, inliers)
        if not len(errors) or mean_error > options.ransac_threshold * 0.85:
            continue
        coverage = min(
            _coverage(source[inliers], images[index_a].shape[:2]),
            _coverage(target[inliers], images[index_b].shape[:2]),
        )
        if coverage < options.min_feature_coverage:
            continue
        geometry = _geometry_quality(matrix, images[index_a].shape[:2], images[index_b].shape[:2])
        if geometry is None:
            continue
        prior_distance = _prior_distance(matrix, prior, images[index_a].shape[:2], images[index_b].shape[:2])
        if prior is not None and prior_distance > options.board_tolerance_fraction:
            continue
        pixel_score = _pixel_correlation(
            images[index_a], images[index_b], matrix, match_masks[index_a], match_masks[index_b]
        )
        if pixel_score < options.min_pixel_correlation:
            continue

        complexity_penalty = {"translation": 0.0, "affine": 0.035, "perspective": 0.09}[kind]
        confidence = (
            0.42 * min(1.0, count / 40.0)
            + 0.18 * min(1.0, ratio)
            + 0.18 * min(1.0, coverage * 4.0)
            + 0.30 * max(0.0, pixel_score)
            - 0.35 * prior_distance
            - 0.06 * min(1.0, mean_error / max(options.ransac_threshold, 1e-6))
            - complexity_penalty
        )

        mode = "Lanczos affine" if kind == "affine" else "Lanczos perspective"
        if kind == "translation":
            rounded = np.round(matrix[:2, 2])
            rounded_errors = np.linalg.norm((source[inliers] + rounded) - target[inliers], axis=1)
            if float(np.mean(rounded_errors)) <= 0.35:
                matrix = np.array([[1.0, 0.0, rounded[0]], [0.0, 1.0, rounded[1]], [0.0, 0.0, 1.0]])
                mean_error = float(np.mean(rounded_errors))
                mode = "exact translation"
            else:
                mode = "Lanczos translation"

        candidate = _PairAlignment(index_a, index_b, matrix, count, mean_error, confidence, mode)
        if best is None or candidate.confidence > best.confidence:
            best = candidate
    return best


def stitch_paths(
    paths,
    options=StitchOptions(),
    *,
    manual_links=None,
    layout_hints: Iterable[StitchLayoutHint] | None = None,
    progress: Callable | None = None,
    cancelled: Callable[[], bool] | None = None,
):
    if not 2 <= len(paths) <= 20:
        raise StitchingError("이미지는 2~20장을 선택할 수 있습니다.")
    layout_hints = list(layout_hints or [])
    images = [read_raw_image(path) for path in paths]
    formats = {(image.dtype.str, image.ndim, image.shape[2] if image.ndim == 3 else 1) for image in images}
    if len(formats) != 1:
        raise StitchingError("픽셀 보존을 위해 비트 깊이와 채널 수가 같아야 합니다.")

    hints = _hint_map(layout_hints)
    regions = _match_region_map(layout_hints)
    auto_detected_crop = False
    if options.overlay_crop_fraction <= 0.0:
        detected_crop = detect_bottom_overlay_fraction(images)
        if detected_crop >= 0.025:
            options = replace(options, overlay_crop_fraction=min(0.45, detected_crop))
            auto_detected_crop = True
            if progress:
                progress("preprocess", 0, 1, f"하단 장비 정보 영역 {options.overlay_crop_fraction:.0%} 제외")

    match_masks = []
    for path, image in zip(paths, images):
        region = regions.get(str(Path(path).resolve()).casefold())
        bottom_crop = 0.0 if auto_detected_crop and region is not None else options.overlay_crop_fraction
        match_masks.append(_match_mask(image, region, bottom_crop))

    sift = cv2.SIFT_create(nfeatures=6000, contrastThreshold=0.025)
    features = []
    for index, image in enumerate(images):
        if cancelled and cancelled():
            raise StitchingCancelled()
        gray = _gray8(image)
        features.append(sift.detectAndCompute(gray, match_masks[index]))
        if progress:
            progress("features", index, len(images), f"특징점 추출 {index + 1}/{len(images)}")

    manual_links = manual_links or {}
    edges: list[_PairAlignment] = []
    pair_total = len(images) * (len(images) - 1) // 2
    pair_index = 0
    for index_a in range(len(images)):
        for index_b in range(index_a + 1, len(images)):
            if cancelled and cancelled():
                raise StitchingCancelled()
            pair_index += 1
            if (index_a, index_b) in manual_links:
                matrix = alignment_from_manual_points(*manual_links[(index_a, index_b)])
                edges.append(_PairAlignment(index_a, index_b, matrix, 0, 0.0, 2.0, "manual Lanczos"))
                continue
            board_a = hints.get(str(Path(paths[index_a]).resolve()).casefold())
            board_b = hints.get(str(Path(paths[index_b]).resolve()).casefold())
            if board_a is not None and board_b is not None and not _board_pair_is_nearby(images[index_a], images[index_b], board_a, board_b):
                continue
            prior = _prior_for_pair(paths[index_a], paths[index_b], hints)
            aligned = _align_pair(index_a, index_b, images, match_masks, features, options, prior)
            if aligned is not None:
                edges.append(aligned)
            if progress:
                progress("matching", pair_index - 1, pair_total, f"겹침 검증 {pair_index}/{pair_total}")

    transforms = {0: np.eye(3)}
    metadata = {0: (0, 0.0, "anchor")}
    while len(transforms) < len(images):
        candidates = []
        for edge in edges:
            if edge.source in transforms and edge.target not in transforms:
                candidates.append(
                    (edge.confidence, edge.target, transforms[edge.source] @ np.linalg.inv(edge.matrix), edge)
                )
            elif edge.target in transforms and edge.source not in transforms:
                candidates.append((edge.confidence, edge.source, transforms[edge.target] @ edge.matrix, edge))
        if not candidates:
            missing = next(index for index in range(len(images)) if index not in transforms)
            raise StitchingNeedsManual(
                "자동 정렬을 신뢰할 수 없어 결과 생성을 중단했습니다. "
                "보드에서 실제 겹침 위치에 더 가깝게 배치하거나, 무늬가 뚜렷한 겹침 영역을 사용해 주세요.",
                (0, missing),
            )
        _, node, matrix, edge = max(candidates, key=lambda item: item[0])
        transforms[node] = matrix
        metadata[node] = (edge.inlier_count, edge.reprojection_error, edge.mode)

    corners = []
    for index, image in enumerate(images):
        valid_points = cv2.findNonZero(match_masks[index])
        if valid_points is None:
            raise StitchingError(f"정합에 사용할 영역이 비어 있습니다: {paths[index]}")
        left, top, width, height = cv2.boundingRect(valid_points)
        source_corners = np.float32(
            [[[left, top], [left + width, top], [left + width, top + height], [left, top + height]]]
        )
        corners.append(cv2.perspectiveTransform(source_corners, transforms[index])[0])
    all_corners = np.concatenate(corners)
    low = np.floor(all_corners.min(0)).astype(int)
    high = np.ceil(all_corners.max(0)).astype(int)
    width, height = (high - low).tolist()
    if width <= 0 or height <= 0 or width * height > 1_200_000_000:
        raise StitchingError("결과 캔버스 크기가 유효하지 않거나 너무 큽니다.")

    shift = np.array([[1.0, 0.0, -low[0]], [0.0, 1.0, -low[1]], [0.0, 0.0, 1.0]])
    output = np.zeros((height, width) + images[0].shape[2:], images[0].dtype)
    valid = np.zeros((height, width), np.uint8)
    placements = []
    for index, (path, image) in enumerate(zip(paths, images)):
        matrix = shift @ transforms[index]
        exact = (
            np.allclose(matrix[:2, :2], np.eye(2))
            and np.allclose(matrix[2], [0, 0, 1])
            and np.allclose(matrix[:2, 2], np.round(matrix[:2, 2]))
        )
        if exact:
            offset_x, offset_y = np.round(matrix[:2, 2]).astype(int)
            valid_points = cv2.findNonZero(match_masks[index])
            left, top, region_width, region_height = cv2.boundingRect(valid_points)
            x, y = offset_x + left, offset_y + top
            warped = np.zeros_like(output)
            mask = np.zeros_like(valid)
            source_region = image[top : top + region_height, left : left + region_width]
            source_mask = match_masks[index][top : top + region_height, left : left + region_width]
            warped[y : y + region_height, x : x + region_width] = source_region
            mask[y : y + region_height, x : x + region_width] = source_mask
        else:
            warped = cv2.warpPerspective(image, matrix, (width, height), flags=cv2.INTER_LANCZOS4)
            mask = cv2.warpPerspective(
                match_masks[index], matrix, (width, height), flags=cv2.INTER_NEAREST
            )
        take = (mask > 0) & (valid == 0)
        output[take] = warped[take]
        valid[mask > 0] = 255
        count, error, mode = metadata[index]
        placements.append(StitchPlacement(path, matrix, mode, count, error))
        if progress:
            progress("compose", index, len(images), f"원본 픽셀 배치 {index + 1}/{len(images)}")
    return StitchResult(output, valid, placements, (width, height))


def save_stitch_result(path, result):
    output = Path(path)
    output = output if output.suffix.lower() in (".tif", ".tiff", ".png") else output.with_suffix(".tif")
    image = result.image
    alpha = result.valid_mask.astype(image.dtype) * (np.iinfo(image.dtype).max // 255)
    if image.ndim == 2:
        encoded = np.dstack((image, image, image, alpha))
    elif image.shape[2] == 3:
        encoded = np.dstack((image, alpha))
    else:
        encoded = image.copy()
        encoded[..., 3] = alpha
    ok, data = cv2.imencode(output.suffix, encoded)
    if not ok:
        raise StitchingError("결과를 저장하지 못했습니다.")
    data.tofile(str(output))
    mask = output.with_name(output.stem + ".mask.png")
    cv2.imencode(".png", result.valid_mask)[1].tofile(str(mask))
    report = output.with_suffix(".stitch.json")
    report.write_text(
        json.dumps(
            {
                "version": 1,
                "scale_status": "recalibration_required",
                "output_size": result.output_size,
                "sources": [
                    {
                        "path": placement.path,
                        "mode": placement.mode,
                        "inliers": placement.inlier_count,
                        "error": placement.reprojection_error,
                        "transform": placement.transform.tolist(),
                    }
                    for placement in result.placements
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return output, mask, report
