"""Automatic opening width and deepest contour depth from guided flat Fields."""
from __future__ import annotations

import numpy as np
from .trench_guides import points_in_guides


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


def measure_field_metrics(fields, traces, guides, y, left, right, closed, corners=()):
    result = dict(field_status='unconfirmed', tcd_status='unconfirmed', depth_status='unconfirmed',
                  reason='위쪽 평평한 Field가 포함되도록 곡선을 그려주세요.',
                  tcd_reason='양쪽 Field의 입구 끝점이 필요합니다.',
                  depth_reason='Field와 바닥 폐쇄를 함께 확인해야 합니다.',
                  reference_y_px=None, tcd_px=None, depth_px=None,
                  anchors={}, anchor_sources={}, flat_supports={}, bottom_point=None, bottom_source=None,
                  left_depth_px=None, right_depth_px=None,
                  automatic_fields=[f['side'] for f in fields if f.get('automatic_field') and f.get('status')=='valid'],
                  definition='horizontal_distance_between_field_transition_points; vertical_depth_from_mean_field_height')
    for item in fields:
        if item.get('source') != 'guide' or item.get('status') != 'valid':
            continue
        item=_flat_field(item)
        if item is None:continue
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
    anchors = result['anchors']
    if not anchors:
        return result
    ref = float(np.mean([p[1] for p in anchors.values()]))
    result.update(field_status='valid', reference_y_px=ref,
                  reason='좌우 Field 입구 높이의 평균 기준' if len(anchors)==2 else '검출된 한쪽 Field 높이 기준')
    if 'left' in anchors and 'right' in anchors and anchors['right'][0] > anchors['left'][0]:
        result.update(tcd_status='valid', tcd_px=anchors['right'][0]-anchors['left'][0],
                      tcd_basis='segmented_transition_estimate' if 'segmented_field_transition' in result['anchor_sources'].values() else 'fitted_field_tangencies',
                      tcd_reason='양쪽 평탄 Field가 입구로 전환되는 지점 사이의 수평 거리')
    if not closed:
        result['depth_reason']='바닥 폐쇄가 확인되지 않아 Field 기준 전체 깊이는 미확정입니다.'
        return result
    floor_points = []
    for trace in traces:
        points = trace.get('points', [])
        sources = trace.get('point_sources', ['measured']*len(points))
        for p, source in zip(points, sources):
            # Exclude unrelated outer surface points; accept only the trench's
            # bottom neighborhood, backed by the separate closure test.
            if min(left) <= p[0] <= max(right) and p[1] >= y[-1]-8:
                floor_points.append((p, source))
    if not floor_points:
        result['depth_reason']='인식 범위 안에서 바닥 윤곽을 확인하지 못했습니다.'
        return result
    bottom, source = max(floor_points, key=lambda item: item[0][1])
    if bottom[1] <= ref:
        return result
    result.update(depth_status='valid', depth_px=float(bottom[1]-ref),
                  bottom_point=list(map(float, bottom)), bottom_source=source,
                  depth_reason='Field 기준 높이부터 연속 윤곽의 최하단까지의 수직 거리')
    for side, p in anchors.items():
        result[side+'_depth_px'] = float(bottom[1]-p[1])
    return result
