"""Recover Field support from upper horizontal contour ends.

Only measured support establishes height and TCD. An outward continuation is
explicitly constructed geometry, not a detected boundary or calibration.
"""
from copy import deepcopy
import math

import numpy as np

from .trench_guides import points_in_guides
from .trench_surface import _line, measure_wall_angle


def _support(path, measured, side, y, edge, middle):
    sign=1 if side=='left' else -1
    depth=max(1.,y[-1]-y[0])
    if len(path)<12 or sign*(path[0,0]-middle)>=0:
        return None
    # A floor, mid-wall ledge, or a horizontal guide disconnected from a wall
    # cannot supply an entrance Field.
    if path[0,1]>y[0]+max(16.,depth*.15):
        return None
    near_wall=(np.abs(path[:,0]-np.interp(path[:,1],y,edge))<6)
    if not np.any(near_wall & (path[:,1]>path[0,1]+12)):
        return None
    delta=path[np.minimum(np.arange(len(path))+2,len(path)-1)]-path[np.maximum(np.arange(len(path))-2,0)]
    angles=np.degrees(np.arctan2(delta[:,1],sign*delta[:,0]))
    baseline=float(np.median(angles[:4]))
    horizontal=((sign*delta[:,0]>.1)&(np.abs(angles)<=25)
                &(np.abs(angles-baseline)<=6))
    stop=next((i for i in range(1,len(path)-1) if not horizontal[i] and not horizontal[i+1]),len(path))
    support=path[:stop];sources=measured[:stop]
    if len(support)<8 or np.ptp(support[:,0])<10 or np.mean(sources)<.8:
        return None
    evidence=support[sources]
    fit=_line(evidence[:,0],evidence[:,1])
    if fit is None:
        return None
    m,c,rms,keep=fit
    if abs(m)>math.tan(math.radians(25)) or rms>1. or np.mean(keep)<.8:
        return None
    if not horizontal[:min(4,len(horizontal))].all():
        return None
    return support,m,c,rms


def _extension(path, slope, side, length, image_shape):
    """C1 continuation: endpoint direction eases into the fitted Field slope."""
    outward=-1 if side=='left' else 1
    start=path[0]
    h,w=image_shape[:2]
    limit=start[0] if side=='left' else w-1-start[0]
    length=min(length,limit)
    if length<1:
        return []
    dx=path[min(3,len(path)-1),0]-start[0]
    local=(path[min(3,len(path)-1),1]-start[1])/dx if abs(dx)>.1 else slope
    local=float(np.clip(local,-.58,.58))
    distance=np.linspace(0,length,max(2,int(math.ceil(length))+1))
    x=start[0]+outward*distance
    yy=start[1]+outward*(slope*distance+(local-slope)*distance*np.exp(-distance/5.))
    points=np.column_stack((x,yy))
    valid=(yy>=0)&(yy<=h-1)
    stop=next((i for i in range(1,len(points)) if not valid[i]),len(points))
    return points[:stop].tolist() if stop>=2 else []


def recover_fields(fields,traces,options,y,left,right,image_shape):
    """Fill only missing sides; never replace an already validated Field."""
    middle=float((left[0]+right[0])/2)
    length=float(np.clip((right[0]-left[0])*.25,24,64))
    for index,(side,edge) in enumerate((('left',left),('right',right))):
        if fields[index].get('status')=='valid':
            continue
        candidates=[]
        for trace in traces:
            offset=0
            sources=trace.get('point_sources',['measured']*sum(map(len,trace['segments'])))
            for segment in trace['segments']:
                path=np.asarray(segment,float)
                measured=np.array([v=='measured' for v in sources[offset:offset+len(path)]])
                offset+=len(path)
                if len(measured)!=len(path) or not np.isfinite(path).all():
                    continue
                for p,mask in ((path,measured),(path[::-1],measured[::-1])):
                    found=_support(p,mask,side,y,edge,middle)
                    if found is not None and points_in_guides(found[0],options.wall_guides).all():
                        candidates.append((found,p))
        # Multiple unrelated upper surfaces are ambiguous; do not choose a
        # different layer merely because it is longer or brighter.
        if len(candidates)!=1:
            continue
        (support,m,c,rms),path=candidates[0]
        item=deepcopy(fields[index])
        item.update(status='valid',source='guide',curved=False,line=[m,c],
                    points=support[np.argsort(support[:,0])].tolist(),
                    rms_px=rms,length_px=float(np.linalg.norm(np.diff(support,axis=0),axis=1).sum()),
                    tilt_deg=math.degrees(math.atan(m)),automatic_field=True,
                    reason='입구의 연속된 가로 윤곽으로 Field 자동 구성 · 바깥쪽은 접선 방향으로 연장',
                    field_model='automatic_horizontal_support',straight_line=[m,c],straight_rms_px=rms,
                    angle_basis='straight_support',tangent_points=support.tolist(),
                    extension=dict(method='tangent_continuation',measured=False,
                                   points_pixel=_extension(path,m,side,length,image_shape)))
        measure_wall_angle(item,y,edge,options.field_search_px,options.angle_span,options.roi[2])
        fields[index]=item
    return fields


def extend_export_surface(points,fields):
    """Append only anchored terminal extensions, preserving all contour points."""
    points=np.asarray(points,float)
    extensions=[]
    for item,index in ((fields[0],0),(fields[1],-1)):
        extension=item.get('extension',{})
        p=np.asarray(extension.get('points_pixel',[]),float)
        if len(p)<2:
            continue
        if not np.allclose(points[index],p[0],rtol=0,atol=1e-6):
            raise ValueError('자동 Field와 윤곽 끝점이 일치하지 않습니다. 다시 인식하세요.')
        points=np.vstack((p[:0:-1],points)) if index==0 else np.vstack((points,p[1:]))
        extensions.append(dict(side=item['side'],**extension))
    return points,extensions
