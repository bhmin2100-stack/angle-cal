"""Measured geometry for the full Trench graph, in original pixel coordinates."""
from __future__ import annotations

import numpy as np


def shape_series(result, show_raw=False):
    """Keep ordered trace segments intact, including the Field and curved floor.

    The horizontal floor reference is only a depth marker when a measured floor
    trace is unavailable. It is dashed so it cannot look like measured geometry.
    """
    series = []
    measured_floor = False
    floor = (result.options.roi[1] + result.depth_px
             if result.depth_px is not None else None)
    for trace in result.guide_traces:
        if show_raw:
            for segment in trace.get('raw_segments', []):
                points = np.asarray(segment, dtype=float)
                if len(points) >= 2:
                    series.append((points[:, 0], points[:, 1], '#999999', True))
        for segment in trace.get('segments', []):
            points = np.asarray(segment, dtype=float)
            if len(points) < 2:
                continue
            series.append((points[:, 0], points[:, 1], '#8f45a4', False))
            if floor is not None:
                # The interior of the bottom, rather than either sidewall tip.
                width = result.right[-1] - result.left[-1]
                inside = ((points[:, 0] > result.left[-1] + width * .1)
                          & (points[:, 0] < result.right[-1] - width * .1))
                measured_floor |= bool(np.any(inside & (points[:, 1] >= result.y[-1] - 1)))
    for field in result.field_angles:
        if field.get('status') == 'valid' and len(field.get('points', [])) >= 2:
            points = np.asarray(field['points'], dtype=float)
            series.append((points[:, 0], points[:, 1], '#268b48', False))
            extension=np.asarray(field.get('extension',{}).get('points_pixel',[]),float)
            if len(extension)>=2:
                series.append((extension[:,0],extension[:,1],'#268b48',True))
    series.extend(((result.left, result.y, '#007d92', False),
                   (result.right, result.y, '#bb4b1e', False)))
    if floor is not None and not measured_floor:
        series.append((np.array([result.left[-1], result.left[-1], result.right[-1], result.right[-1]]),
                       np.array([result.y[-1], floor, floor, result.y[-1]]),
                       '#8f45a4', True))
    return series
