"""Field, entrance and angle from the same measured guide contour.

The drawn guide only selects a search corridor. Missing measured segments are
never joined to manufacture a Field or a radius. Coordinates remain pixels.
"""
from __future__ import annotations

import math
import cv2
import numpy as np

from .trench_guides import sweep_samples
from .trench_corner import _resample, _monotone, fit_circle
from .trench_surface import _line, measure_wall_angle


def _runs(mask):
    return np.split(np.flatnonzero(mask),np.flatnonzero(np.diff(np.flatnonzero(mask))>1)+1)


def _branch(points, side, top, anchor, reach, search):
    p=np.asarray(points,float)
    if len(p)<2:return []
    sign=1 if side=='left' else -1
    q=(p[:,0]-anchor)*sign
    near=(p[:,1]>=top-search)&(p[:,1]<=top+reach*1.6)&(q<=reach*.5)
    chunks=[]
    for indices in _runs(near):
        if len(indices)<8:continue
        part=p[indices]
        if part[0,0]*sign>part[-1,0]*sign:part=part[::-1]
        chunks.append(part)
    return chunks


def _corner(points, side, roi, span, spread, connected):
    result=dict(side=side,roi=list(roi),selection=list(span or (0,1)),source='guide',
                status='unconfirmed',reason='가이드 윤곽이 Field부터 측벽까지 연결되지 않았습니다.',
                radius_px=None,center=None,rms_px=None,arc_angle_deg=None,
                stability_fraction=None,edge_spread_px=spread,fit_points=[],trace_points=[],arc_points=[])
    if not connected or len(points)<8:return result
    points=np.asarray(points,float)
    if roi is not None:
        x,y,w,h=roi
        inside=(points[:,0]>=x)&(points[:,0]<=x+w)&(points[:,1]>=y)&(points[:,1]<=y+h)
        if not inside.all():
            result['reason']='선택 검색 영역이 가이드의 둥근 전이 구간 전체를 포함하지 않습니다.'
            return result
    result['trace_points']=points.tolist();result['transition_point_count']=len(points)
    start,end=span or (0,1)
    if not 0<=start<end<=1:
        result['reason']='원호 시작·끝 범위가 올바르지 않습니다.';return result
    points=points[round(start*(len(points)-1)):round(end*(len(points)-1))+1]
    result['fit_points']=points.tolist()
    fitted=fit_circle(points)
    if fitted is None:
        result['reason']='전이 윤곽점이 8개 미만이거나 원을 결정할 수 없습니다.';return result
    center,radius,rms=fitted
    angles=np.unwrap(np.arctan2(points[:,1]-center[1],points[:,0]-center[0]))
    angle=abs(float(np.degrees(angles[-1]-angles[0])))
    trim=max(1,round((len(points)-1)*.1))
    variants=[fit_circle(p) for p in (points[trim:],points[:-trim])]
    stability=max(abs(f[1]-radius)/radius if f else 1. for f in variants)
    result.update(rms_px=rms,arc_angle_deg=angle,stability_fraction=stability)
    if angle<45:result['reason']='원호 각도가 45° 미만입니다.'
    elif rms>max(1,radius*.05):result['reason']='윤곽과 원호의 맞춤 오차가 기준을 초과합니다.'
    elif stability>.2:result['reason']='구간을 10% 조정하면 R이 20% 넘게 달라집니다.'
    elif radius<3 or spread is not None and radius<2*spread:
        result['reason']='둥근 전이 구간이 해상도 또는 경계 흐림에 비해 작습니다.'
    else:
        theta=np.linspace(angles[0],angles[-1],80)
        result.update(status='valid',reason='가이드에서 추적한 실제 전이 윤곽의 대표 원 맞춤',
                      radius_px=radius,center=center.tolist(),
                      arc_points=(center+radius*np.column_stack((np.cos(theta),np.sin(theta)))).tolist())
    return result


def measure_guided_surfaces(guides,traces,options,y,left,right,fields,corners):
    """Replace only sides whose guide deliberately includes a Field support."""
    # Surface segmentation and angle support never depend on the R editor's
    # independent search rectangle or arc start/end fractions.
    top=options.roi[1];reach=max(96,options.field_search_px*2)
    for side_index,(side,edge) in enumerate((('left',left),('right',right))):
        sign=1 if side=='left' else -1
        anchor=float(np.interp(min(y[-1],top+reach),y,edge))
        candidates=[];intent=False;spread_values=[]
        for guide,trace in zip(guides,traces):
            samples=sweep_samples(guide)
            direction=np.gradient(samples,axis=0)
            horizontal=np.abs(direction[:,0])>1.5*np.abs(direction[:,1])
            field_zone=(sign*(samples[:,0]-anchor)<-12)&(np.abs(samples[:,1]-top)<=options.field_search_px)
            if not np.any(horizontal&field_zone):continue
            intent=True
            for segment in trace['segments']:
                for branch in _branch(segment,side,top,anchor,reach,options.field_search_px):
                    outward=sign*(branch[:,0]-anchor)<-10
                    if np.count_nonzero(outward)>=12:
                        candidates.append(branch)
            spread_values.extend(v for v in trace.get('edge_spreads_px',[]) if v is not None)
        if not intent:continue  # Wall-only legacy guides retain automatic Field.
        item=dict(side=side,status='unconfirmed',reason='Field 가이드 안에서 연속된 실제 표면 경계를 찾지 못했습니다.',
                  source='guide',points=[],line=None,rms_px=None,length_px=None,tilt_deg=None,
                  angle_status='unconfirmed',angle_reason='가이드 Field 기준이 미확정입니다.',angle_deg=None,
                  wall_points=[],wall_line=None,wall_rms_px=None,vertex=None,
                  coordinate_unit='px',field_model='measured_contour',wall_model='x=m*y+b',
                  angle_span=list(options.angle_span),search_px=options.field_search_px,
                  straight_angle_deg=None,local_angle_deg=None,angle_basis=None,
                  straight_line=None,straight_rms_px=None,tangent_points=[],entrance_trace=[])
        fields[side_index]=item
        # A generous display ROI is independent of the user's arc selection.
        default_roi=(max(0,int(anchor-reach*1.8)),max(0,int(top-reach*.5)),int(reach*3.6),int(reach*2.4))
        roi=options.corner_rois[side_index] or default_roi
        spread=float(np.median(spread_values)) if spread_values else None
        corners[side_index]=_corner([],side,roi,options.corner_spans[side_index],spread,False)
        if not candidates:continue
        # Prefer an uninterrupted branch that reaches the upper wall. Never
        # concatenate separated trace segments across missing image evidence.
        def rank(p):
            near_wall=(np.abs(p[:,0]-np.interp(p[:,1],y,edge))<5)&(p[:,1]>top+12)
            return (bool(near_wall.any()),len(p))
        branch=max(candidates,key=rank)
        canonical=np.column_stack((branch[:,0]*sign,branch[:,1]))
        trace=_resample(canonical)
        if len(trace)<20:continue
        smooth=np.column_stack([cv2.GaussianBlur(trace[:,i:i+1],(1,11),2).ravel() for i in (0,1)])
        delta=np.gradient(smooth,axis=0)
        turns=_monotone(np.clip(np.degrees(np.arctan2(delta[:,1],delta[:,0])),-40,120))
        baseline=float(np.median(turns[3:min(15,len(turns)-3)]))
        middle=np.flatnonzero(turns>baseline+(90-baseline)*.5)
        connected=bool(len(middle) and np.median(turns[-10:-2])>65)
        if connected:
            mid=int(middle[0]);curvature=cv2.GaussianBlur(np.gradient(turns)[:,None],(1,15),3).ravel()
            central=float(np.median(curvature[max(0,mid-3):mid+4]))
            threshold=max(.12,central*.3)
            a=mid;b=mid
            while a>2 and curvature[a-1]>threshold:a-=1
            while b<len(trace)-3 and curvature[b+1]>threshold:b+=1
            angular_a=np.flatnonzero(turns>baseline+8)
            angular_b=np.flatnonzero(turns<82)
            # For a planar Field, use the complete turning interval rather
            # than trimming a long arc at a small raster-induced curvature dip.
            initial=trace[:max(16,int(angular_a[0]) if len(angular_a) else 16)]
            initial_fit=_line(initial[:,0],initial[:,1])
            initial_count=len(initial)
            initial_turn_change=abs(float(np.median(turns[max(3,initial_count-8):initial_count-2]))-
                                    float(np.median(turns[3:9]))) if initial_count>=16 else 90
            if initial_fit is not None and initial_fit[2]<.5 and initial_turn_change<3:
                a=int(angular_a[0]) if len(angular_a) else a
            b=int(angular_b[-1]) if len(angular_b) else b
            a=max(0,a-2);b=min(len(trace)-1,b+2)
        else:
            a=len(trace)-1;b=a
        field=trace[:a+1].copy();field[:,0]*=sign
        if len(field)<16 or np.ptp(field[:,0])<15:continue
        # Quadratic residual validates a gently curved Field without flattening
        # its measured contour. A local derivative defines its entrance angle.
        x_origin=float(field[-1,0]);xx=field[:,0]-x_origin
        polynomial=np.polyfit(xx,field[:,1],2)
        residual=field[:,1]-np.polyval(polynomial,xx)
        rms=float(np.sqrt(np.mean(residual**2)))
        straight=_line(field[:,0],field[:,1])
        if straight is None or rms>1.5:
            item['reason']='가이드 Field의 표면 맞춤 오차가 큽니다.';continue
        sm,sc,srms,keep=straight
        tilt_change=abs(math.degrees(math.atan(np.polyval(np.polyder(polynomial),xx[0]))-
                                    math.atan(polynomial[1])))
        curved=srms>1.5 or tilt_change>3
        local=field[-min(32,len(field)):]
        local_x=local[:,0]-x_origin
        local_poly=np.polyfit(local_x,local[:,1],2)
        m=float(local_poly[1]);c=float(local_poly[2]-m*x_origin)
        if not curved:m,c=sm,sc
        if abs(m)>math.tan(math.radians(45)):
            item['reason']='Field 접선이 입구 지지 표면으로 보기에는 너무 가파릅니다.';continue
        item.update(status='valid',points=field[np.argsort(field[:,0])].tolist(),line=[m,c],
                    rms_px=rms if curved else srms,length_px=float(np.linalg.norm(np.diff(field,axis=0),axis=1).sum()),
                    tilt_deg=math.degrees(math.atan(m)),straight_line=[sm,sc],straight_rms_px=srms,
                    tangent_points=local.tolist(),curved=curved,angle_basis='local_tangent' if curved else 'straight_support',
                    reason='가이드의 실제 곡면 Field · 입구 근처 접선 기준' if curved else '가이드의 실제 Field · 직선 지지 구간 기준',
                    entrance_trace=np.column_stack((trace[:,0]*sign,trace[:,1])).tolist())
        rounded=trace[a:b+1].copy();rounded[:,0]*=sign
        corner=_corner(rounded,side,roi,options.corner_spans[side_index],spread,connected)
        corners[side_index]=corner
        angle_corner=_corner(rounded,side,default_roi,None,spread,connected) if curved else corner
        if curved and angle_corner['status']=='valid':
            # Locate the normal foot of the measured circle on the local Field
            # model. Only a short, recorded continuation beyond measured Field
            # support is permitted; the radius still uses actual arc points.
            from numpy.polynomial import Polynomial
            p=Polynomial(local_poly[::-1]);cx,cy=angle_corner['center']
            equation=Polynomial([x_origin-cx,1])+(p-cy)*p.deriv()
            feet=[float(v.real) for v in equation.roots() if abs(v.imag)<1e-8
                  and -8<=v.real<=8]
            if feet:
                foot=min(feet,key=abs);fy=float(p(foot));fx=x_origin+foot
                error=abs(math.hypot(fx-cx,fy-cy)-angle_corner['radius_px'])
                if error<=max(1,angle_corner['radius_px']*.05):
                    m=float(p.deriv()(foot));c=fy-m*fx
                    item.update(line=[m,c],tilt_deg=math.degrees(math.atan(m)),
                                tangent_anchor=[fx,fy],tangent_extrapolation_px=abs(foot))
        measure_wall_angle(item,y,edge,options.field_search_px,options.angle_span,options.roi[2])
        if item['angle_status']=='valid':
            item['local_angle_deg']=item['angle_deg']
            item['angle_reason']='가이드 Field의 입구 접선과 선택 측벽 직선 사이의 내각입니다.' if curved else '같은 가이드 윤곽의 Field 직선과 선택 측벽 직선 사이의 내각입니다.'
            if not curved:item['straight_angle_deg']=item['angle_deg']
            elif srms<=1.5:
                reference=dict(item,line=[sm,sc],angle_status='unconfirmed',angle_deg=None)
                measure_wall_angle(reference,y,edge,options.field_search_px,options.angle_span,options.roi[2])
                if reference['angle_status']=='valid':item['straight_angle_deg']=reference['angle_deg']
        if curved:
            # Keep the independently validated R, but a straight-support circle
            # is not an exact connection to a curved Field.
            item['connection_model']='local_tangent'
    return fields,corners
