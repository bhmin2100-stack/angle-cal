"""Deterministic, placement-independent translation search for trench images."""
from __future__ import annotations

import cv2
import numpy as np


def correlation(a, b):
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    a = a - a.mean()
    b = b - b.mean()
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / norm) if norm > 1e-8 else -1.0


def overlap_slices(a_shape, b_shape, dx, dy):
    x0, y0 = max(0, dx), max(0, dy)
    x1, y1 = min(b_shape[1], a_shape[1] + dx), min(b_shape[0], a_shape[0] + dy)
    if x1 <= x0 or y1 <= y0:
        return None
    return (slice(y0-dy, y1-dy), slice(x0-dx, x1-dx)), (slice(y0, y1), slice(x0, x1))


def band_profiles(a, b, valid, *, with_regions=False):
    """Return profiles, optionally with the exact pixels and region used by each.

    Display strips run left-to-right along the profile axis; vertical bands are
    transposed so every strip column corresponds to the same graph sample.
    """
    rows, cols = np.where(valid)
    if len(rows) < 64:
        return []
    y0, y1, x0, x1 = rows.min(), rows.max()+1, cols.min(), cols.max()+1
    a, b, valid = a[y0:y1, x0:x1], b[y0:y1, x0:x1], valid[y0:y1, x0:x1]
    profiles = []
    for axis, name in ((0, "가로"), (1, "세로")):
        for index, indices in enumerate(np.array_split(np.arange(a.shape[axis]), 3)):
            if not len(indices):
                continue
            mask = np.take(valid, indices, axis=axis)
            count = mask.sum(axis=axis)
            keep = count >= max(2, len(indices)//2)
            if keep.sum() < 12:
                continue
            first = (np.take(a, indices, axis=axis)*mask).sum(axis=axis) / np.maximum(count, 1)
            second = (np.take(b, indices, axis=axis)*mask).sum(axis=axis) / np.maximum(count, 1)
            first, second = first[keep], second[keep]
            if min(first.std(), second.std()) < 0.35:
                continue
            score = .7*correlation(first, second) + .3*correlation(np.diff(first), np.diff(second))
            profile = (f"{name} 밴드 {index+1}", first, second, score)
            if with_regions:
                strip_a = np.take(a, indices, axis=axis)
                strip_b = np.take(b, indices, axis=axis)
                if axis == 1:
                    strip_a, strip_b, mask = strip_a.T, strip_b.T, mask.T
                positions = np.flatnonzero(keep)
                if axis == 0:
                    region = (x0 + int(positions[0]), y0 + int(indices[0]),
                              int(positions[-1] - positions[0] + 1), len(indices))
                else:
                    region = (x0 + int(indices[0]), y0 + int(positions[0]),
                              len(indices), int(positions[-1] - positions[0] + 1))
                profile += ({
                    "axis": axis, "region": region,
                    "first_strip": strip_a[:, keep].copy(),
                    "second_strip": strip_b[:, keep].copy(),
                    "strip_mask": mask[:, keep].copy(),
                },)
            profiles.append(profile)
    return sorted(profiles, key=lambda p: p[3], reverse=True)


def _ncc_surface(a, b, ma, mb):
    # Linear (not cyclic) masked NCC: every translation uses its own means.
    shape = (a.shape[0]+b.shape[0]-1, a.shape[1]+b.shape[1]-1)
    def fft(x):
        return np.fft.rfft2(x, s=shape)
    fa = [fft(x) for x in (ma, a*ma, a*a*ma)]
    fb = [fft(x[::-1, ::-1]) for x in (mb, b*mb, b*b*mb)]
    def conv(i, j):
        return np.fft.irfft2(fa[i]*fb[j], s=shape)
    n = np.maximum(conv(0, 0), 1)
    sa, sb = conv(1, 0), conv(0, 1)
    va = np.maximum(0, conv(2, 0)-sa*sa/n)
    vb = np.maximum(0, conv(0, 2)-sb*sb/n)
    denom = np.sqrt(va*vb)
    score = np.full(shape, -1.0)
    good = (n >= .22*min(ma.sum(), mb.sum())) & (denom > n*.1)
    score[good] = (conv(1, 1)[good]-sa[good]*sb[good]/n[good])/denom[good]
    return np.clip(score, -1, 1)


def image_gradients(image):
    """Signed 2-D edges, smoothed to tolerate modest acquisition noise/blur."""
    smooth = cv2.GaussianBlur(image.astype(np.float64), (0, 0), 1.0)
    return np.stack((cv2.Sobel(smooth, cv2.CV_64F, 1, 0),
                     cv2.Sobel(smooth, cv2.CV_64F, 0, 1)), axis=-1)


def full_overlap_score(a, b, valid, gradients_a=None, gradients_b=None):
    """Score every valid overlap pixel; no selected bands or sparse sampling.

    Normalized correlations remove additive exposure and positive gain changes.
    Erode the mask to keep excluded pixels out of the smoothed edge comparison.
    """
    if np.count_nonzero(valid) < 64:
        return -1.0
    brightness = correlation(a[valid], b[valid])
    edge_valid = cv2.erode(valid.astype(np.uint8), np.ones((9, 9), np.uint8),
                           borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    if np.count_nonzero(edge_valid) < 64:
        return brightness
    if gradients_a is None:
        gradients_a = image_gradients(a)
        gradients_b = image_gradients(b)
    edges_a, edges_b = gradients_a[edge_valid], gradients_b[edge_valid]
    if min(float(edges_a.std()), float(edges_b.std())) < .1:
        return brightness
    edges = correlation(edges_a, edges_b)
    return .70 * brightness + .30 * edges


def _position_score(a, b, ma, mb, dx, dy, sample_step, minimum_pixels):
    """Cheap brightness score used only to narrow an integer candidate position."""
    slices = overlap_slices(a.shape, b.shape, dx, dy)
    if slices is None:
        return -1.0
    sa, sb = slices
    first = a[sa][::sample_step, ::sample_step]
    second = b[sb][::sample_step, ::sample_step]
    valid = ((ma[sa] > 0) & (mb[sb] > 0))[::sample_step, ::sample_step]
    if np.count_nonzero(valid) < minimum_pixels:
        return -1.0
    return correlation(first[valid], second[valid])


def _refine_positions(positions, a, b, ma, mb, level_scale, radius, sample_step):
    refined = []
    minimum_pixels = max(
        64,
        .22 * min(
            np.count_nonzero(ma[::sample_step, ::sample_step]),
            np.count_nonzero(mb[::sample_step, ::sample_step]),
        ),
    )
    for px, py in positions:
        center_x, center_y = round(px * level_scale), round(py * level_scale)
        best = None
        for dy in range(center_y - radius, center_y + radius + 1):
            for dx in range(center_x - radius, center_x + radius + 1):
                score = _position_score(
                    a, b, ma, mb, dx, dy, sample_step, minimum_pixels
                )
                if best is None or score > best[0]:
                    best = (score, dx, dy)
        if best is not None and best[0] > -1:
            refined.append((round(best[1] / level_scale), round(best[2] / level_scale)))
    return list(dict.fromkeys(refined))


def translation_candidates(a, b, ma, mb):
    """Find global peaks cheaply, then verify each final position over every overlap pixel."""
    scale = min(1.0, 480/max(*a.shape, *b.shape))
    def small(x, mask=False):
        return cv2.resize(x.astype(np.float64), None, fx=scale, fy=scale,
                          interpolation=cv2.INTER_NEAREST if mask else cv2.INTER_AREA)
    aa, bb = small(a), small(b)
    surface = _ncc_surface(aa, bb, small(ma > 0, True), small(mb > 0, True))
    peaks = []
    radius = max(5, round(8*scale))
    for _ in range(32):
        y, x = np.unravel_index(np.argmax(surface), surface.shape)
        score = surface[y, x]
        if score < .45:
            break
        peaks.append((round((bb.shape[1]-1-x)/scale), round((bb.shape[0]-1-y)/scale)))
        surface[max(0,y-radius):y+radius+1, max(0,x-radius):x+radius+1] = -1
    max_dimension = max(*a.shape, *b.shape)
    positions = list(dict.fromkeys(peaks))

    # The coarse 480 px FFT can leave several native pixels of uncertainty.
    # Resolve it on a larger level while sampling pixels; this stage proposes
    # positions only and never determines the reported alignment score.
    refinement_scale = min(1.0, 1600 / max_dimension)
    resolved_scale = scale
    if refinement_scale > scale * 1.05:
        def refinement_image(x, mask=False):
            return cv2.resize(
                x.astype(np.float64), None, fx=refinement_scale, fy=refinement_scale,
                interpolation=cv2.INTER_NEAREST if mask else cv2.INTER_AREA,
            )
        level_a, level_b = refinement_image(a), refinement_image(b)
        level_ma = refinement_image(ma > 0, True)
        level_mb = refinement_image(mb > 0, True)
        level_radius = max(2, int(np.ceil(.5 * refinement_scale / scale)) + 1)
        level_step = max(2, int(np.ceil(max(level_a.shape + level_b.shape) / 600)))
        positions = _refine_positions(
            positions, level_a, level_b, level_ma, level_mb,
            refinement_scale, level_radius, level_step,
        )
        resolved_scale = refinement_scale

    # Pin each proposal to a native integer coordinate with a sparse positional
    # pass. The full-resolution score below still reads every valid pixel.
    if resolved_scale < 1.0:
        position_radius = max(1, int(np.ceil(.5 / resolved_scale)) + 1)
        position_step = max(2, int(np.ceil(max_dimension / 600)))
        positions = _refine_positions(
            positions, a, b, ma, mb, 1.0, position_radius, position_step,
        )

    refined = []
    gradients_a, gradients_b = image_gradients(a), image_gradients(b)
    minimum_pixels = max(64, .22 * min(np.count_nonzero(ma), np.count_nonzero(mb)))
    for dx, dy in positions:
        slices = overlap_slices(a.shape, b.shape, dx, dy)
        if slices is None:
            continue
        sa, sb = slices
        valid = (ma[sa] > 0) & (mb[sb] > 0)
        if valid.sum() < minimum_pixels:
            continue
        score = full_overlap_score(
            a[sa], b[sb], valid, gradients_a[sa], gradients_b[sb]
        )
        candidate = (score, dx, dy)
        if not any(abs(dx-p[1]) < 4 and abs(dy-p[2]) < 4 for p in refined):
            refined.append(candidate)
    return sorted(refined, reverse=True)
