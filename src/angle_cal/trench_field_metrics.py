"""Field depth and virtual opening width from short upper-wall line fits."""
from __future__ import annotations

import numpy as np
from .trench_guides import points_in_guides


def contour_floor(traces, y, left, right):
    """Find a bottom joining both lower walls in ONE ordered trace segment.

    This uses the selected boundary, not the unrelated intensity column below
    it. A guide-filled bottom remains an estimate, never an image observation.
    """
    candidates = []
    for trace in traces:
        sources = {tuple(p): s for p, s in zip(trace.get('points', []),
                   trace.get('point_sources', ['measured'] * len(trace.get('points', []))))}
        for segment in trace.get('segments', []):
            p = np.asarray(segment, float)
            if len(p) < 5:
                continue
            k = int(np.argmax(p[:, 1])); bottom = p[k]
            if bottom[1] < y[-1]-2 or not min(left) <= bottom[0] <= max(right):
                continue
            rise = max(4., min(20., (bottom[1]-y[0])*.1))
            a = np.flatnonzero(p[:k, 1] <= bottom[1]-rise)
            b = np.flatnonzero(p[k+1:, 1] <= bottom[1]-rise)
            if not len(a) or not len(b):
                continue
            i, j = int(a[-1]), int(k+1+b[0])
            ends = p[[i, j]]
            l = np.interp(ends[:, 1], y, left); r = np.interp(ends[:, 1], y, right)
            mid = (l+r)/2
            if not ((ends[0, 0] < mid[0] and ends[1, 0] > mid[1]) or
                    (ends[1, 0] < mid[1] and ends[0, 0] > mid[0])):
                continue
            # Require both ends at their classified wall, not a small ledge.
            if np.any(np.minimum(abs(ends[:, 0]-l), abs(ends[:, 0]-r)) > 3):
                continue
            floor = p[i:j+1]
            observed = np.array([sources.get(tuple(q), 'measured') == 'measured' for q in floor])
            near_bottom = floor[:, 1] >= bottom[1]-1.5
            measured = bool(np.mean(observed) >= .5 and np.any(observed & near_bottom))
            candidates.append(dict(point=bottom.tolist(), source=sources.get(tuple(bottom), 'measured'),
                                   measured=measured, basis='connected_contour' if measured else 'interpolated_contour'))
    return max(candidates, key=lambda c: c['point'][1]) if candidates else None


def _fit_wall(y, edge):
    if len(y) < 5 or np.ptp(y) < 4 or np.any(np.diff(y) > 1.5):
        return None
    keep = np.ones(len(y), bool)
    for _ in range(5):
        m, c = np.polyfit(y[keep]-y[0], edge[keep], 1)
        residual = edge-(m*(y-y[0])+c)
        mad = 1.4826*np.median(abs(residual[keep]-np.median(residual[keep])))
        updated = abs(residual) <= max(.5, 3*mad)
        if updated.sum() < 5 or np.array_equal(keep, updated):
            break
        keep = updated
    m, c = np.polyfit(y[keep]-y[0], edge[keep], 1)
    residual = edge-(m*(y-y[0])+c)
    rms = float(np.sqrt(np.mean(residual[keep]**2)))
    return float(m), float(c-m*y[0]), rms, keep


def wall_field_intersection(field, y, edge, nm_per_px=None, traces=()):
    """Use the first locally straight wall after the entrance turn (100 A).

    Detect the end of the turn from slope agreement with the adjacent wall.
    Field length, fitted entrance radius and radius editor do not set the fit.
    """
    target = 10./nm_per_px if nm_per_px else 10.
    span = max(6., float(np.ceil(target)))
    result = dict(status='unconfirmed', reason='입구 근처의 직선 측벽 지지가 부족합니다.',
                  requested_span_angstrom=100., requested_span_px=target if nm_per_px else None,
                  resolution_limited=bool(nm_per_px and target < 6), scale_missing=not bool(nm_per_px),
                  points=[], line=None, intersection=None, rms_px=None)
    fm, fc = field['line']
    start = max(y[0], min(p[1] for p in field['points']))
    stop = min(y[-1], start+max(96., (y[-1]-start)*.45))
    observed = [p for t in traces for p, source in zip(t.get('points', []),
                t.get('point_sources', ['measured']*len(t.get('points', [])))) if source == 'measured']
    observed = np.asarray(observed, float).reshape(-1, 2)
    for k in np.flatnonzero((y >= start) & (y <= stop)):
        end = np.searchsorted(y, y[k]+span, side='right')
        yy, xx = y[k:end], edge[k:end]
        if not len(yy) or yy[-1]-yy[0] < span-.01:
            continue
        fit = _fit_wall(yy, xx)
        # A separate downstream support rejects the rounded transition even
        # when a short piece of that arc alone has a small line-fit residual.
        next_end = np.searchsorted(y, yy[-1]+max(8., min(16., span)), side='right')
        following = _fit_wall(y[end-1:next_end], edge[end-1:next_end])
        if fit is None or following is None:
            continue
        m, c, rms, keep = fit
        if (abs(m) > 1. or rms > .6 or following[2] > .8 or
                abs(np.arctan(m)-np.arctan(following[0])) > np.radians(1)):
            continue
        p = np.column_stack((xx, yy))
        if len(observed):
            nearby = observed[(observed[:, 1] >= yy[0]-2) & (observed[:, 1] <= yy[-1]+2)]
            if not len(nearby) or np.mean(np.min(np.linalg.norm(p[:, None]-nearby, axis=2), axis=1) <= 2) < .5:
                continue
        elif traces:
            continue
        denominator = 1-m*fm
        if abs(denominator) < .1:
            continue
        vx = (m*fc+c)/denominator; vy = fm*vx+fc
        if vy > yy[0]+2 or yy[0]-vy > max(96., (y[-1]-start)*.45):
            continue
        actual = float(yy[-1]-yy[0])
        result.update(status='valid', reason='입구 측벽 직선과 Field 직선의 교점',
                      points=p.tolist(), line=[m, c], intersection=[float(vx), float(vy)], rms_px=rms,
                      actual_span_px=actual, actual_span_angstrom=actual*nm_per_px*10 if nm_per_px else None,
                      support_count=int(keep.sum()), field_line=list(field['line']))
        return result
    return result


def _flat_field(item):
    if not item.get('curved'):
        return item
    points=np.asarray(item.get('points',[]),float)
    if item['side']=='right':points=points[::-1]
    # Smoothing can round a few entrance-adjacent Field points. Recover an
    # actually planar outer support instead of losing Field classification.
    n=int(len(points)*.65)
    if n<16:return None
    p=points[:n];half=n//2
    line=np.polyfit(p[:,0],p[:,1],1)
    slopes=[np.polyfit(q[:,0],q[:,1],1)[0] for q in (p[:half],p[half:])]
    if (np.sqrt(np.mean((p[:,1]-np.polyval(line,p[:,0]))**2))>.5
            or abs(np.degrees(np.arctan(slopes[0])-np.arctan(slopes[1])))>3):return None
    residual=np.abs(points[:,1]-np.polyval(line,points[:,0]))
    stop=next((i for i in range(n,len(points)-1) if np.all(residual[i:i+2]>.75)),len(points))
    p=points[:stop]
    if item['side']=='right':p=p[::-1]
    return dict(item,points=p.tolist(),line=line.tolist(),curved=False)


def measure_field_metrics(fields, traces, guides, y, left, right, closed, corners=(), *, nm_per_px=None,
                          wall_reference=None):
    result = dict(field_status='unconfirmed', tcd_status='unconfirmed', depth_status='unconfirmed',
                  reason='위쪽 평평한 Field가 포함되도록 곡선을 그려주세요.',
                  tcd_reason='양쪽 Field와 입구 근처 측벽 직선이 필요합니다.',
                  depth_reason='Field와 이어진 바닥 윤곽을 함께 확인해야 합니다.',
                  reference_y_px=None, tcd_px=None, depth_px=None,
                  anchors={}, anchor_sources={}, flat_supports={}, tcd_anchors={}, wall_fits={},
                  bottom_point=None, bottom_source=None, bottom_basis=None,
                  left_depth_px=None, right_depth_px=None,
                  automatic_fields=[f['side'] for f in fields if f.get('automatic_field') and f.get('status')=='valid'],
                  definition='horizontal_distance_between_upper_wall_100A_line_field_intersections; vertical_depth_from_mean_field_height')
    valid_fields = {}
    for item in fields:
        if item.get('source') != 'guide' or item.get('status') != 'valid':
            continue
        # A valid gently curved Field already carries a validated local line.
        # Prefer planar support, but do not hide depth just because it curves.
        item=_flat_field(item) or item
        points = np.asarray(item.get('points', []), float)
        if len(points) < 2:
            continue
        side = item['side']
        connection = item.get('connection', {})
        anchor = connection.get('field_tangent') if connection.get('status') == 'valid' else None
        source = 'field_tangent_circle'
        if anchor is None:
            # A large entrance radius may leave too little straight cliff in
            # the angle-fit span. Its independently valid circle can still
            # locate the Field tangent without requiring a wall angle.
            corner = next((c for c in corners if c.get('side')==side and c.get('status')=='valid'), None)
            if corner is not None:
                cx, cy = corner['center']; m, c = item['line']
                fx = (cx+m*(cy-c))/(1+m*m); fy=m*fx+c
                radius_error=abs(np.hypot(fx-cx,fy-cy)-corner['radius_px'])
                support_end=points[np.argmax(points[:,0]) if side=='left' else np.argmin(points[:,0])]
                if radius_error<=max(1.,corner['radius_px']*.05) and np.linalg.norm([fx-support_end[0],fy-support_end[1]])<=max(12.,corner['radius_px']*.2):
                    anchor=[fx,fy];source='field_tangent_independent_radius'
        if anchor is None or not points_in_guides([anchor], guides)[0]:
            # Field segmentation ends where the contour starts turning. Use
            # that measured support, not the wall's projected virtual vertex.
            p = points[np.argmax(points[:, 0]) if side == 'left' else np.argmin(points[:, 0])]
            anchor = [float(p[0]), float(np.polyval(item['line'], p[0]))]
            source = 'segmented_field_transition'
        if not points_in_guides([anchor], guides)[0]:
            continue
        result['anchors'][side] = list(map(float, anchor))
        result['anchor_sources'][side] = source
        result['flat_supports'][side] = points.tolist()
        valid_fields[side] = item
    anchors = result['anchors']
    if not anchors:
        return result
    ref = float(np.mean([p[1] for p in anchors.values()]))
    result.update(field_status='valid', reference_y_px=ref,
                  reason='좌우 Field 입구 높이의 평균 기준' if len(anchors)==2 else '검출된 한쪽 Field 높이 기준')
    # Keep Field anchors for the existing coordinate system. TCD has its own
    # virtual intersections so field extensions/fillets cannot enlarge it.
    wy, wl, wr, wt = wall_reference if wall_reference is not None else (y, left, right, traces)
    for side, edge in (('left', wl), ('right', wr)):
        if side not in valid_fields:
            continue
        fit = wall_field_intersection(valid_fields[side], wy, edge, nm_per_px, wt)
        result['wall_fits'][side] = fit
        if fit['status'] == 'valid':
            result['tcd_anchors'][side] = fit['intersection']
    vertices = result['tcd_anchors']
    if len(vertices) == 2 and vertices['right'][0] > vertices['left'][0]:
        result.update(tcd_status='valid', tcd_px=vertices['right'][0]-vertices['left'][0],
                      tcd_basis='upper_wall_field_intersections',
                      tcd_reason='입구 직전 약 100 Å 측벽 직선을 Field까지 연장한 두 교점의 수평 거리')
    else:
        failures = [f"{'좌측' if s=='left' else '우측'}: {f['reason']}" for s, f in result['wall_fits'].items() if f['status']!='valid']
        if failures:result['tcd_reason'] = ' / '.join(failures)
    floor = contour_floor(traces, y, left, right)
    floor_points = []
    for trace in traces if closed and floor is None else []:
        points = trace.get('points', [])
        sources = trace.get('point_sources', ['measured']*len(points))
        for p, source in zip(points, sources):
            # Exclude unrelated outer surface points; accept only the trench's
            # bottom neighborhood, backed by the separate closure test.
            if min(left) <= p[0] <= max(right) and p[1] >= y[-1]-8:
                floor_points.append((p, source))
    if floor is None and not floor_points:
        result['depth_reason']='양쪽 벽을 잇는 바닥 윤곽이 없습니다. 관찰된 깊이만 표시합니다.'
        result['observed_depth_px'] = max(0., float(y[-1]-ref))
        return result
    bottom, source = (floor['point'], floor['source']) if floor else max(floor_points, key=lambda item: item[0][1])
    if bottom[1] <= ref:
        return result
    estimated = bool(floor and not floor['measured'] and not closed)
    result.update(depth_status='estimated' if estimated else 'valid', depth_px=float(bottom[1]-ref),
                  bottom_point=list(map(float, bottom)), bottom_source=source,
                  bottom_basis=floor['basis'] if floor else 'intensity_closure',
                  depth_reason='Field부터 보간으로 연결된 최하단까지의 추정 깊이' if estimated else 'Field 기준 높이부터 연속 윤곽의 최하단까지의 수직 거리')
    for side, p in anchors.items():
        result[side+'_depth_px'] = float(bottom[1]-p[1])
    return result
