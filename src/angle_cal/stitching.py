from __future__ import annotations

from dataclasses import dataclass, replace, field
import json
import math
from pathlib import Path
from typing import Callable, Iterable

import cv2
import numpy as np

from .band_registration import translation_candidates, full_overlap_score, overlap_slices


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
    preserve_scale_bar: bool = True
    normalize_tone: bool = True
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
    tone_gain: float = 1.0
    tone_offset: float = 0.0
    sharpening: float = 0.0


@dataclass
class StitchResult:
    image: np.ndarray
    valid_mask: np.ndarray
    placements: list[StitchPlacement]
    output_size: tuple[int, int]
    confidence: float = 0.0
    scale_bar_source: str | None = None
    notes: list[str] = field(default_factory=list)
    saved_path: str | None = None


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


def _scale_bar_footer_start(image) -> int | None:
    """Locate a dark SEM footer containing a long bright ruler-like line."""
    gray = _gray8(image)
    height, width = gray.shape[:2]
    top = int(height * 0.55)
    roi = gray[top:]
    candidates: list[tuple[int, int]] = []
    kernel = np.ones((1, max(20, width // 16)), np.uint8)
    for threshold in (140, 180, 220):
        bright = np.uint8(roi >= threshold) * 255
        horizontal = cv2.morphologyEx(bright, cv2.MORPH_OPEN, kernel)
        row_coverage = np.count_nonzero(horizontal, axis=1)
        for y in np.where(row_coverage >= max(24, int(width * 0.08)))[0]:
            global_y = top + int(y)
            if global_y >= int(height * 0.62):
                candidates.append((int(row_coverage[y]), global_y))
    row_mean = gray.astype(np.float32).mean(axis=1)
    candidates = [
        (candidate_width, y)
        for candidate_width, y in candidates
        if y + 4 < height and float(np.median(row_mean[y + 4 :])) < 95.0
    ]
    if not candidates:
        return None
    longest = max(width for width, _ in candidates)
    bar_y = min(y for width, y in candidates if width == longest)
    if longest >= int(width * 0.75):
        return bar_y
    lower = int(height * 0.55)
    if bar_y > lower + 2:
        drops = row_mean[lower:bar_y] - row_mean[lower + 1 : bar_y + 1]
        edge = int(np.argmax(drops)) + lower + 1
        if float(drops[edge - lower - 1]) >= 8.0:
            return min(edge, bar_y)
    return max(lower, bar_y - max(12, height // 12))


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
    scale_footer_starts = [start for image in images if (start := _scale_bar_footer_start(image)) is not None]
    if scale_footer_starts:
        fractions = [
            (image.shape[0] - start) / image.shape[0]
            for image, start in zip(images, [
                _scale_bar_footer_start(image) for image in images
            ])
            if start is not None
        ]
        if fractions:
            return float(np.median(fractions))

    found = []
    for image in images:
        gray = _gray8(image)
        height = gray.shape[0]
        row_std = gray.std(1)
        for y in range(height - 2, int(height * 0.55), -1):
            reference = row_std[max(0, y - height // 15) : y].mean()
            fraction = (height - y) / height
            if reference >= 8.0 and 0.025 <= fraction <= 0.45 and row_std[y:].mean() < max(4.0, reference * 0.4):
                found.append(fraction)
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
    area_scale = abs(float(np.linalg.det(matrix[:2, :2])))
    minimum = .22 * min(np.count_nonzero(mask_a) * area_scale, np.count_nonzero(mask_b))
    if int(valid.sum()) < max(256, minimum):
        return -1.0
    return full_overlap_score(warped, gray_b, valid)


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


def _phase_translation(index_a, index_b, images, match_masks, options, prior):
    gray_a, gray_b = _gray8(images[index_a]), _gray8(images[index_b])
    # Coarse global search followed by full-overlap brightness/edge scoring.
    # Board placement and diagnostic band graphs never influence the result.
    peaks = translation_candidates(gray_a, gray_b, match_masks[index_a], match_masks[index_b])
    maximum_overlap = max(1, min(int(np.count_nonzero(match_masks[index_a])),
                                 int(np.count_nonzero(match_masks[index_b]))))
    ranked = []
    for score, dx, dy in peaks:
        if score < .70:
            continue
        sa, sb = overlap_slices(gray_a.shape, gray_b.shape, dx, dy)
        overlap_pixels = int(np.count_nonzero((match_masks[index_a][sa] > 0)
                                              & (match_masks[index_b][sb] > 0)))
        if overlap_pixels < max(64, maximum_overlap * .22):
            continue
        ranked.append((score, overlap_pixels, dx, dy))
    if not ranked:
        return None
    # Prefer more supporting pixels for exact score ties, but never resolve a
    # truly periodic ambiguity just by choosing the largest overlap.
    ranked.sort(reverse=True)
    score, overlap_pixels, dx, dy = ranked[0]
    if len(ranked) > 1 and score - ranked[1][0] < .006:
        raise StitchingNeedsManual(
            "겹친 영역 전체를 비교해도 여러 위치가 비슷하게 일치합니다. "
            "서로 다른 특징이 포함되도록 정합 영역을 지정하세요."
        )
    if overlap_pixels / maximum_overlap > .96 and score > .995:
        raise StitchingNeedsManual(
            "두 이미지가 거의 완전히 같아 확장 방향을 결정할 수 없습니다. 서로 다른 영역이 포함된 이미지를 사용하세요."
        )
    matrix = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy], [0.0, 0.0, 1.0]])
    return _PairAlignment(index_a, index_b, matrix, 0, 0.0, min(.99, score), "exact translation")


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
    if best is not None:
        return best
    if descriptors_a is None or descriptors_b is None:
        return best
    matches = _mutual_matches(descriptors_a, descriptors_b, options.ratio_test)
    if len(matches) < options.min_matches:
        return best
    source = np.float32([keypoints_a[match.queryIdx].pt for match in matches])
    target = np.float32([keypoints_b[match.trainIdx].pt for match in matches])
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
        pixel_score = _pixel_correlation(
            images[index_a], images[index_b], matrix, match_masks[index_a], match_masks[index_b]
        )
        if pixel_score < max(.55, options.min_pixel_correlation):
            continue

        complexity_penalty = {"translation": 0.0, "affine": 0.035, "perspective": 0.09}[kind]
        confidence = pixel_score - complexity_penalty

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
    if not manual_links:
        paths = sorted(paths, key=lambda path: str(Path(path).resolve()).casefold())
    layout_hints = list(layout_hints or [])
    images = [read_raw_image(path) for path in paths]
    formats = {(image.dtype.str, image.ndim, image.shape[2] if image.ndim == 3 else 1) for image in images}
    if len(formats) != 1:
        raise StitchingError("픽셀 보존을 위해 비트 깊이와 채널 수가 같아야 합니다.")

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
        match_masks.append(_match_mask(image, region, options.overlay_crop_fraction))

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
    ambiguous_pairs = {}
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
            # Board coordinates intentionally do not influence registration.
            # They are a UI arrangement only; image-derived full-overlap evidence
            # must produce the same transform after arbitrary board movement.
            try:
                aligned = _align_pair(index_a, index_b, images, match_masks, features, options, None)
            except StitchingNeedsManual as exc:
                ambiguous_pairs[(index_a, index_b)] = str(exc)
                aligned = None
            if aligned is not None:
                edges.append(aligned)
            if progress:
                progress("matching", pair_index - 1, pair_total, f"겹침 검증 {pair_index}/{pair_total}")

    transforms = {0: np.eye(3)}
    metadata = {0: (0, 0.0, "anchor")}
    used_confidences = []
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
            for (a, b), reason in ambiguous_pairs.items():
                if (a in transforms) != (b in transforms):
                    raise StitchingNeedsManual(reason, (a, b))
            raise StitchingNeedsManual(
                "자동 정렬을 신뢰할 수 없어 결과 생성을 중단했습니다. "
                "두 이미지에 공통으로 보이는 특징이 충분히 포함되도록 정합 영역을 지정해 주세요.",
                (0, missing),
            )
        _, node, matrix, edge = max(candidates, key=lambda item: item[0])
        used_confidences.append(min(1.0, max(0.0, edge.confidence)))
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
        gain, offset, sharpening = 1.0, 0.0, 0.0
        common = (mask > 0) & (valid > 0)
        if options.normalize_tone and common.sum() >= 256:
            warped, gain, offset, sharpening = _match_tone(warped, output, common)
        take = (mask > 0) & (valid == 0)
        output[take] = warped[take]
        valid[mask > 0] = 255
        count, error, mode = metadata[index]
        placements.append(StitchPlacement(path, matrix, mode, count, error, gain, offset, sharpening))
        if progress:
            progress("compose", index, len(images), f"원본 픽셀 배치 {index + 1}/{len(images)}")
    result = StitchResult(output, valid, placements, (width, height), min(used_confidences, default=0.0))
    if options.preserve_scale_bar and options.overlay_crop_fraction > 0:
        _preserve_footer(result, images, match_masks, options.overlay_crop_fraction)
    return result


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
                "version": 2,
                "confidence": result.confidence,
                "confidence_kind": "heuristic registration quality, not probability",
                "registration_metric": "full-overlap normalized brightness and 2-D edges; bands are display-only",
                "image_count": len(result.placements),
                "scale_bar_source": result.scale_bar_source,
                "notes": result.notes,
                "scale_status": "recalibration_required",
                "output_size": result.output_size,
                "sources": [
                    {
                        "path": placement.path,
                        "mode": placement.mode,
                        "inliers": placement.inlier_count,
                        "error": placement.reprojection_error,
                        "transform": placement.transform.tolist(),
                        "tone_gain": placement.tone_gain,
                        "tone_offset": placement.tone_offset,
                        "sharpening": placement.sharpening,
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


def _match_tone(image, reference, common):
    """Conservative overlap-derived exposure and sharpness correction."""
    source = image.astype(np.float32)
    target = reference.astype(np.float32)
    a, b = source[common], target[common]
    if np.mean(np.abs(a-b)) < .5:
        return image, 1.0, 0.0, 0.0
    # Exclude alpha from the fit.
    if a.ndim == 2:
        a, b = a[:, :3].mean(1), b[:, :3].mean(1)
    va = float(np.var(a))
    if va < 1e-6:
        return image, 1.0, 0.0, 0.0
    gain = float(np.clip(np.mean((a-a.mean())*(b-b.mean()))/va, .5, 2.0))
    maximum = float(np.iinfo(image.dtype).max)
    offset = float(np.clip(b.mean()-gain*a.mean(), -.20*maximum, .20*maximum))
    corrected = source*gain+offset
    sharp = 0.0
    blur = cv2.GaussianBlur(corrected, (0, 0), 1.0)
    detail = corrected-blur
    baseline = float(np.mean((corrected[common]-target[common])**2))
    for amount in (.15, .3):
        candidate = corrected+amount*detail
        error = float(np.mean((candidate[common]-target[common])**2))
        if error < baseline*.98:
            sharp, baseline = amount, error
    corrected += sharp*detail
    corrected = np.clip(np.rint(corrected), 0, maximum).astype(image.dtype)
    if image.ndim == 3 and image.shape[2] == 4:
        corrected[..., 3] = image[..., 3]
    return corrected, gain, offset, sharp


def _preserve_footer(result, images, masks, crop_fraction: float = 0.0):
    # Keep one original footer at native pixel size when the stitched result has
    # not materially changed source scale. This preserves one useful ruler while
    # preventing repeated SEM footers from driving registration.
    def scale_is_preserved(placement):
        linear = placement.transform[:2, :2]
        scale_x = float(np.linalg.norm(linear[:, 0]))
        scale_y = float(np.linalg.norm(linear[:, 1]))
        return (
            np.allclose(placement.transform[2], [0, 0, 1], atol=2e-4)
            and abs(scale_x - 1.0) <= 0.015
            and abs(scale_y - 1.0) <= 0.015
        )

    if not all(scale_is_preserved(placement) for placement in result.placements):
        result.notes.append("변형/배율 차이로 스케일바 자동 보존 생략; 재보정 필요")
        return
    for image, placement in zip(images, result.placements):
        start = _scale_bar_footer_start(image)
        if start is None:
            continue
        if crop_fraction > 0:
            start = min(start, max(0, int(round(image.shape[0] * (1.0 - crop_fraction)))))
        footer = image[start:]
        if footer.shape[0] < 8 or footer.shape[0] > image.shape[0]*.45:
            continue
        gray = _gray8(footer)
        # Require a long, bright, horizontal ruler-like component.
        binary = np.uint8(gray > max(150, np.percentile(gray, 80)))*255
        lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((1,max(20,image.shape[1]//12)), np.uint8))
        if not np.any(lines):
            continue
        old_h, old_w = result.image.shape[:2]
        width = max(old_w, image.shape[1])
        canvas = np.zeros((old_h+footer.shape[0], width)+image.shape[2:], image.dtype)
        valid = np.zeros(canvas.shape[:2], np.uint8)
        canvas[:old_h,:old_w] = result.image
        valid[:old_h,:old_w] = result.valid_mask
        canvas[old_h:,:image.shape[1]] = footer
        valid[old_h:,:image.shape[1]] = 255
        result.image, result.valid_mask = canvas, valid
        result.output_size = (width, canvas.shape[0])
        result.scale_bar_source = placement.path
        result.notes.append("하단에 원본 스케일바 후보 띠 1개를 원래 픽셀 크기로 보존; 실제 단위 확인 및 재보정 필요")
        return


def save_stitch_result_auto(result):
    """Reserve a unique filename alongside the canonical source, without overwrite."""
    source = Path(result.placements[0].path)
    confidence = int(round(float(np.clip(result.confidence, 0.0, 1.0)) * 100))
    stem = f"{source.stem[:96]}_combined_align_{confidence:03d}pct_{len(result.placements)}imgs"
    index = 0
    while True:
        path = source.with_name(stem + (f"_{index}" if index else "") + ".tif")
        try:
            with path.open("xb"):
                pass
            break
        except FileExistsError:
            index += 1
    try:
        output, _, _ = save_stitch_result(path, result)
    except Exception:
        # Remove only our reserved output and sidecars, never an existing image.
        for own in (path, path.with_name(path.stem+".mask.png"), path.with_suffix(".stitch.json")):
            own.unlink(missing_ok=True)
        raise
    result.saved_path = str(output.resolve())
    return output
