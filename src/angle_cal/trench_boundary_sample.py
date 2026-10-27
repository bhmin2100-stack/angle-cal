"""Image-selected boundary landmarks, transported by normalized profile matching.

The marker is a position in a measured profile, not an absolute gray threshold.
Positive gain/offset changes and reversed cross-section directions are handled;
unmatched sections remain missing observations for the normal join policy.
"""
from __future__ import annotations

import cv2
import numpy as np

from .image_ops import to_gray

MAX_SAMPLE_LENGTH = 160.
MIN_MATCH = .80


def validate_boundary_sample(value, image_shape=None):
    if not isinstance(value, dict) or value.get('version') != 1:
        raise ValueError('인식지점 샘플 형식이 올바르지 않습니다. 다시 지정하세요.')
    try:
        points = np.asarray([value['start'], value['end']], float)
        distances = np.asarray(value['distances_px'], float)
        values = np.asarray(value['intensities'], float)
        anchor = float(value['anchor_px'])
        half = float(value['half_width_px'])
        shape = np.asarray(value['image_shape'], float)
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError('인식지점 샘플 값이 올바르지 않습니다.') from exc
    if (points.shape != (2, 2) or distances.ndim != 1 or values.shape != distances.shape
            or not 9 <= len(values) <= 162 or shape.shape != (2,)
            or not all(np.isfinite(a).all() for a in (points, distances, values, shape, [anchor, half]))):
        raise ValueError('인식지점 샘플 좌표와 밝기값을 확인하세요.')
    length = float(np.linalg.norm(points[1] - points[0]))
    if (not 8 <= length <= MAX_SAMPLE_LENGTH or not 0 <= half <= 20
            or abs(distances[0]) > 1e-6 or abs(distances[-1] - length) > 1e-4
            or np.any(np.diff(distances) <= 0) or np.max(np.diff(distances)) > 1.01
            or not 0 <= anchor <= length or np.ptp(values) < 1e-6
            or np.any(shape < 2) or np.any(shape != np.floor(shape))
            or np.any(points < 0) or np.any(points >= shape[::-1])):
        raise ValueError('경계를 가로질러 8~160 px 길이의 샘플을 지정하세요.')
    if image_shape is not None and list(map(int, image_shape[:2])) != shape.astype(int).tolist():
        raise ValueError('샘플을 지정한 이미지 크기가 다릅니다. 인식지점을 다시 지정하세요.')
    return dict(version=1, start=points[0].tolist(), end=points[1].tolist(),
                half_width_px=half, distances_px=distances.tolist(), intensities=values.tolist(),
                anchor_px=anchor, image_shape=shape.astype(int).tolist(),
                method='normalized_profile_landmark', minimum_similarity=MIN_MATCH)


def sample_point(sample):
    a, b = np.asarray(sample['start']), np.asarray(sample['end'])
    return a + (b-a) * (sample['anchor_px'] / sample['distances_px'][-1])


def validate_boundary_samples(value, image_shape=None):
    """Optional per-wall overrides; v1 common samples stay independent."""
    if value is None:return {}
    if not isinstance(value, dict) or set(value)-{'left', 'right'}:
        raise ValueError('좌우 인식지점 샘플 형식이 올바르지 않습니다.')
    result = {}
    for side in ('left', 'right'):
        if value.get(side) is None:continue
        try:result[side] = validate_boundary_sample(value[side], image_shape)
        except ValueError as exc:
            raise ValueError(f"{'왼쪽' if side=='left' else '오른쪽'} 측벽: {exc}") from exc
    return result


def wall_sample_scopes(centers, normals, guide, guides=(), samples=None):
    """Assign wall branches without splitting the connected contour tracker.

    The two branches of a U are separated at its floor in path order, not at
    an image X midpoint. Thus asymmetric bowing and reverse drawing are safe.
    Horizontal Field/floor sections retain the common/default image criterion.
    """
    roles = np.full(len(centers), 'surface', dtype=object)
    vertical = abs(normals[:, 0]) >= .7
    high = float(centers[:, 1].max()); depth = float(np.ptp(centers[:, 1]))
    if (depth >= 16 and max(centers[0, 1], centers[-1, 1]) < high-max(12., depth*.3)
            and abs(centers[-1, 0]-centers[0, 0]) >= 8):
        bottom = np.flatnonzero(centers[:, 1] >= high-1.)
        split = int(np.median(bottom))
        first, second = ('left', 'right') if centers[0, 0] < centers[-1, 0] else ('right', 'left')
        roles[vertical & (np.arange(len(centers)) <= split)] = first
        roles[vertical & (np.arange(len(centers)) > split)] = second
        return roles
    # Legacy separate wall guides: use the cavity's row intersections, ignoring
    # old manual side labels. No image pixels are searched outside the guide.
    from .trench_guides import sweep_samples
    rows = {}
    for other in guides:
        p = sweep_samples(other)
        for a, b in zip(p[:-1], p[1:]):
            if abs(b[1]-a[1]) < .1:continue
            lo, hi = int(np.ceil(min(a[1], b[1]))), int(np.floor(max(a[1], b[1])))
            for y in range(lo, hi+1):
                x = float(a[0]+(y-a[1])*(b[0]-a[0])/(b[1]-a[1]))
                row = rows.setdefault(y, [x, x]); row[0] = min(row[0], x); row[1] = max(row[1], x)
    for i in np.flatnonzero(vertical):
        bounds = rows.get(round(centers[i, 1]))
        if bounds and bounds[1]-bounds[0] > 8:
            roles[i] = 'left' if centers[i, 0] < sum(bounds)/2 else 'right'
    # A standalone wall may only have one nearby sample. Use proximity to that
    # guide (not the whole image midpoint); ambiguous assignment stays default.
    if not np.any(roles != 'surface'):
        near = []
        for side, sample in (samples or {}).items():
            distance = float(np.min(np.linalg.norm(centers-sample_point(sample), axis=1)))
            if distance <= guide['half_width_px']+2:near.append((distance, side))
        near.sort()
        if near and (len(near)==1 or near[1][0]-near[0][0] > 4):roles[vertical] = near[0][1]
    return roles


def sample_polygon(sample):
    a, b = np.asarray(sample['start']), np.asarray(sample['end'])
    d = b-a
    n = np.array([-d[1], d[0]]) / np.linalg.norm(d) * sample['half_width_px']
    return np.array([a-n, b-n, b+n, a+n])


def make_boundary_sample(image, start, end, half_width=3., anchor=None):
    a, b = np.asarray(start, float), np.asarray(end, float)
    if a.shape != (2,) or b.shape != (2,) or not np.isfinite([a, b]).all():
        raise ValueError('샘플 선의 위치를 확인하세요.')
    length = float(np.linalg.norm(b-a))
    if not 8 <= length <= MAX_SAMPLE_LENGTH:
        raise ValueError('확대 후 경계를 가로질러 8~160 px 길이로 드래그하세요.')
    if not np.isfinite(half_width) or not 0 <= half_width <= 20:
        raise ValueError('샘플 띠의 반폭은 0~20 px여야 합니다.')
    gray = to_gray(image).astype(np.float32)
    normal = np.array([-(b-a)[1], (b-a)[0]]) / length
    distances = np.linspace(0, length, int(np.ceil(length))+1)
    across = np.linspace(-half_width, half_width, max(1, int(np.ceil(2*half_width))+1))
    xy = a + distances[:, None, None] / length * (b-a) + across[None, :, None] * normal
    h, w = gray.shape
    if (np.any(xy < 0) or np.any(xy[:, :, 0] > w-1) or np.any(xy[:, :, 1] > h-1)):
        raise ValueError('샘플 띠 전체를 이미지 안에 두거나 띠 폭을 줄이세요.')
    # Same small image/normal blur as the tracer; retain original 8/16-bit units.
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    levels = cv2.remap(gray, xy[:, :, 0].astype(np.float32), xy[:, :, 1].astype(np.float32),
                       cv2.INTER_LINEAR).mean(axis=1)
    levels = cv2.GaussianBlur(levels[None, :], (0, 0), .8, borderType=cv2.BORDER_REPLICATE)[0]
    if not np.isfinite(levels).all() or np.ptp(levels) < max(1e-6, float(np.ptp(gray))*.01):
        raise ValueError('샘플 안의 밝기 변화가 부족합니다. 경계 양쪽을 포함하세요.')
    if anchor is None:
        anchor = float(distances[np.argmax(np.abs(np.gradient(levels)))])
    return validate_boundary_sample(dict(version=1, start=a.tolist(), end=b.tolist(),
        half_width_px=float(half_width), distances_px=distances.tolist(), intensities=levels.tolist(),
        anchor_px=float(np.clip(anchor, 0, length)), image_shape=[h, w]))


def prepare_templates(sample, inverted=False):
    """Uniform original-pixel spacing, including the independently chosen marker."""
    sample = validate_boundary_sample(sample)
    distances = np.asarray(sample['distances_px'])
    # Exactly one pixel spacing makes the template compatible with trace offsets.
    grid = np.arange(int(np.floor(distances[-1]))+1, dtype=float)
    values = np.interp(grid, distances, sample['intensities']).astype(np.float32)
    if inverted:
        values = -values
    anchor = sample['anchor_px']
    return ((values, anchor), (values[::-1].copy(), grid[-1]-anchor))


def profile_candidates(profile, offsets, valid, radius, templates):
    """Return supported marker positions, always within the guide corridor.

    Similarity is a shape score, never a probability or dimensional accuracy.
    Both orientations are compared without allowing arbitrary contrast inversion.
    """
    profile = np.asarray(profile, np.float32)
    ranked = []
    for values, anchor in templates:
        n = len(values)
        if n > len(profile):
            continue
        scores = cv2.matchTemplate(profile[None, :], values[None, :], cv2.TM_CCOEFF_NORMED)[0]
        prefix = np.r_[0., np.cumsum(profile, dtype=float)]
        squares = np.r_[0., np.cumsum(profile.astype(float)**2)]
        variance = (squares[n:]-squares[:-n])/n - ((prefix[n:]-prefix[:-n])/n)**2
        invalid = np.r_[0, np.cumsum(~valid)]
        positions = offsets[:len(scores)] + anchor
        supported = ((invalid[n:] == invalid[:-n]) & (variance >= 4)
                     & (np.abs(positions) <= radius) & (scores >= MIN_MATCH))
        # Find peaks before applying the corridor mask: clipping a competing
        # outside maximum must not fabricate an edge at the search-band limit.
        peaks = (scores >= np.r_[-np.inf, scores[:-1]]) & (scores >= np.r_[scores[1:], -np.inf])
        for k in np.flatnonzero(supported & peaks):
            shift = 0.
            if 0 < k < len(scores)-1:
                a, b, c = map(float, scores[k-1:k+2])
                denom = a-2*b+c
                if denom < -1e-8:
                    shift = float(np.clip(.5*(a-c)/denom, -.5, .5))
            position = float(positions[k] + shift)
            if abs(position) <= radius:
                ranked.append((position, float(96*scores[k])))
    selected = []
    for point in sorted(ranked, key=lambda c: c[1], reverse=True):
        if all(abs(point[0]-old[0]) >= 1.5 for old in selected):
            selected.append(point)
        if len(selected) == 6:
            break
    return selected
