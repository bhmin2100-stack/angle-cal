"""Curve structures shared with AngleCal's existing structure format."""
import numpy as np
from .trench_guides import normalize_guides


def structure_guides(payload,image_shape=None):
    guides=payload.get('wall_guides') or payload.get('trench_guides')
    if guides is None:
        guides=[]
        for record in payload.get('records',[]):
            if record.get('kind')!='edge':continue
            points=record.get('recognition_points') or record.get('points') or [record.get('start'),record.get('end')]
            if len(points)>200:points=[points[i] for i in np.linspace(0,len(points)-1,200).round().astype(int)]
            guides.append(dict(points=points,half_width_px=record.get('search_radius_px') or 12))
    if not guides:raise ValueError('구조에 곡선 또는 경계선이 없습니다.')
    return normalize_guides(guides,image_shape)
