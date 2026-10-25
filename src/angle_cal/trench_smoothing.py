"""Optional, bounded smoothing of measured contour segments in pixel space."""
from __future__ import annotations

import copy
import numpy as np

from .trench_guides import points_in_guides

METHODS = {'mean': '이동평균', 'gaussian': '가우시안',
           'savgol': 'Savitzky–Golay', 'robust': '강건 스플라인'}


def smoothing_parameters(strength):
    value = float(strength)
    if not np.isfinite(value) or not 0 <= value <= 100:
        raise ValueError('윤곽 보정 강도는 0~100이어야 합니다.')
    return 3 + 2 * int(round(value * .19)), 4 * value / 100


def _convolve(points, kernel):
    radius = len(kernel) // 2
    padded = np.pad(points, ((radius, radius), (0, 0)), mode='reflect')
    return np.column_stack([np.convolve(padded[:, i], kernel, 'valid') for i in range(2)])


def _pentadiagonal(diagonal, first, second, rhs):
    """LDL solve of a positive definite banded system, O(n) memory/time."""
    n = len(diagonal)
    d = diagonal.copy()
    a, b = np.zeros(n), np.zeros(n)
    y = rhs.copy()
    for i in range(n):
        if i >= 2:
            b[i] = second[i-2] / d[i-2]
        if i >= 1:
            a[i] = (first[i-1] - (b[i]*d[i-2]*a[i-1] if i >= 2 else 0)) / d[i-1]
            d[i] -= a[i]**2*d[i-1]
            y[i] -= a[i]*y[i-1]
        if i >= 2:
            d[i] -= b[i]**2*d[i-2]
            y[i] -= b[i]*y[i-2]
    x = y / d[:, None]
    for i in range(n-1, -1, -1):
        if i+1 < n:
            x[i] -= a[i+1]*x[i+1]
        if i+2 < n:
            x[i] -= b[i+2]*x[i+2]
    return x


def _robust_spline(points, window):
    # Discrete smoothing spline: weighted residual + squared second differences.
    penalty = (window / 4)**4
    weights = np.ones(len(points))
    for _ in range(6):
        d = weights.copy()
        d[:-2] += penalty; d[1:-1] += 4*penalty; d[2:] += penalty
        first = np.zeros(len(points)-1)
        first[:-1] -= 2*penalty; first[1:] -= 2*penalty
        fit = _pentadiagonal(d, first, np.full(len(points)-2, penalty), weights[:, None]*points)
        residual = np.linalg.norm(points-fit, axis=1)
        scale = max(.1, float(np.median(residual)) / .6745)
        weights = np.maximum(.02, np.minimum(1, 1.345*scale / np.maximum(residual, 1e-9)))
        weights[[0, -1]] = 1
    return fit


def smooth_segment(points, method, strength, guide, image_shape):
    if method not in METHODS:
        raise ValueError('지원하지 않는 윤곽 보정 방식입니다.')
    window, max_move = smoothing_parameters(strength)
    original = np.asarray(points, dtype=float)
    if strength == 0 or len(original) < 5:
        return original.copy()
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(original, axis=0), axis=1))]
    if arc[-1] < 4:
        return original.copy()
    # Uniform arc length treats horizontal Fields and vertical walls equally.
    distance = np.linspace(0, arc[-1], min(20001, max(5, int(np.ceil(arc[-1]))+1)))
    spacing = distance[1]-distance[0]
    p = np.column_stack([np.interp(distance, arc, original[:, i]) for i in range(2)])
    radius = min((len(p)-1)//2, max(1, int(round((window//2)/spacing))))
    window = 2*radius+1
    offsets = np.arange(-radius, radius+1, dtype=float)
    if method == 'mean':
        fit = _convolve(p, np.ones(window)/window)
    elif method == 'gaussian':
        kernel = np.exp(-.5*(offsets/max(.5, radius/2))**2)
        fit = _convolve(p, kernel/kernel.sum())
    elif method == 'savgol':
        design = np.column_stack((np.ones(window), offsets, offsets**2))
        fit = _convolve(p, np.linalg.pinv(design)[0])
    else:
        fit = _robust_spline(p, window)
    fit = np.column_stack([np.interp(arc, distance, fit[:, i]) for i in range(2)])
    # Ease out corrections at open ends and strong geometric turns. This is a
    # conservative protection, not a claim that noise and roughness are separable.
    local_window = min(7, 2*((len(p)-1)//2)+1)
    baseline = _convolve(p, np.ones(local_window)/local_window)
    span = min(4, (len(p)-1)//2)
    before = baseline - np.roll(baseline, span, axis=0)
    after = np.roll(baseline, -span, axis=0) - baseline
    cosine = np.sum(before*after, axis=1) / np.maximum(1e-9, np.linalg.norm(before, axis=1)*np.linalg.norm(after, axis=1))
    turn = np.arccos(np.clip(cosine, -1, 1))
    protection = np.interp(arc, distance, 1/(1+(turn/.35)**4))
    taper = np.minimum(1, np.minimum(arc, arc[-1]-arc)/max(2, radius*spacing))
    delta = (fit-original)*protection[:, None]*taper[:, None]
    lengths = np.linalg.norm(delta, axis=1)
    delta *= np.minimum(1, max_move/np.maximum(lengths, 1e-9))[:, None]
    # Reject a correction rather than projecting it onto the guide's boundary.
    # Check the swept displacement too, so it cannot cross a gap in the corridor.
    valid = np.ones(len(original), dtype=bool)
    for fraction in (.25, .5, .75, 1):
        candidate = original + fraction*delta
        valid &= points_in_guides(candidate, (guide,))
        valid &= ((candidate >= 0).all(axis=1) & (candidate[:, 0] < image_shape[1])
                  & (candidate[:, 1] < image_shape[0]))
    delta[~valid] = 0
    return original + delta


def smooth_traces(traces, guides, method, strength, image_shape):
    smoothing_parameters(strength)
    if method not in METHODS:
        raise ValueError('지원하지 않는 윤곽 보정 방식입니다.')
    if not strength:
        return traces
    result = copy.deepcopy(traces)
    for trace, guide in zip(result, guides):
        trace['raw_segments'] = copy.deepcopy(trace['segments'])
        trace['raw_points'] = copy.deepcopy(trace['points'])
        trace['segments'] = [smooth_segment(s, method, strength, guide, image_shape).tolist()
                             for s in trace['raw_segments']]
        trace['points'] = [p for segment in trace['segments'] for p in segment]
        shift = np.linalg.norm(np.asarray(trace['points']).reshape(-1, 2)
                               - np.asarray(trace['raw_points']).reshape(-1, 2), axis=1)
        window, limit = smoothing_parameters(strength)
        trace['smoothing'] = dict(method=method, strength=float(strength), window_px=window,
                                 displacement_limit_px=limit, max_displacement_px=float(shift.max()) if len(shift) else 0,
                                 rms_displacement_px=float(np.sqrt(np.mean(shift**2))) if len(shift) else 0,
                                 raw_metadata_scope='rim_peak_points, inward_normals, joins describe pre-smoothing observations')
    return result
