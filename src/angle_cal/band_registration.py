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


def band_profiles(a, b, valid):
    """Return corresponding brightness profiles in six horizontal/vertical bands."""
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
            profiles.append((f"{name} 밴드 {index+1}", first, second, score))
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


def translation_candidates(a, b, ma, mb):
    """Global peaks, followed by full-resolution integer refinement; no board prior."""
    scale = min(1.0, 480/max(*a.shape, *b.shape))
    def small(x, mask=False):
        return cv2.resize(x.astype(np.float32), None, fx=scale, fy=scale,
                          interpolation=cv2.INTER_NEAREST if mask else cv2.INTER_AREA)
    aa, bb = small(a), small(b)
    surface = _ncc_surface(aa, bb, small(ma > 0, True), small(mb > 0, True))
    peaks = []
    radius = max(5, round(8*scale))
    for _ in range(5):
        y, x = np.unravel_index(np.argmax(surface), surface.shape)
        score = surface[y, x]
        if score < .45:
            break
        peaks.append((round((bb.shape[1]-1-x)/scale), round((bb.shape[0]-1-y)/scale)))
        surface[max(0,y-radius):y+radius+1, max(0,x-radius):x+radius+1] = -1
    refined = []
    reach = max(2, int(np.ceil(1/scale)))
    for px, py in peaks:
        best = None
        for dy in range(py-reach, py+reach+1):
            for dx in range(px-reach, px+reach+1):
                slices = overlap_slices(a.shape, b.shape, dx, dy)
                if slices is None:
                    continue
                sa, sb = slices
                valid = (ma[sa] > 0) & (mb[sb] > 0)
                if valid.sum() < .20*min(np.count_nonzero(ma), np.count_nonzero(mb)):
                    continue
                # Bound work on very large SEM images.
                stride = max(1, int(np.sqrt(valid.size/60000)))
                av, bv, vv = a[sa][::stride,::stride], b[sb][::stride,::stride], valid[::stride,::stride]
                score = correlation(av[vv], bv[vv])
                if best is None or score > best[0]:
                    best = (score, dx, dy)
        if best is not None and not any(abs(best[1]-p[1]) < 4 and abs(best[2]-p[2]) < 4 for p in refined):
            refined.append(best)
    return sorted(refined, reverse=True)
