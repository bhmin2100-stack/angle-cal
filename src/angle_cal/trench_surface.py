"""Measured Field support and sidewall angles in original image coordinates."""
from __future__ import annotations
import math
import cv2
import numpy as np
from .image_ops import to_gray


def connect_entrance(item, corner):
    """Fit a circle tangent to both measured supports, in native pixels."""
    connection = dict(status='unconfirmed', reason='Field 또는 cliff 직선이 미확정입니다.',
                      radius_px=None, rms_px=None, center=None, arc_points=[],
                      field_tangent=None, wall_tangent=None, stability_fraction=None)
    item['connection'] = connection
    if item.get('angle_status') != 'valid':
        return
    points = np.asarray(corner.get('fit_points', []) if corner else [], dtype=float)
    if len(points) < 8 or corner.get('transition_point_count',8)<8 or (corner.get('arc_angle_deg',90) or 0)<45:
        connection['reason'] = '둥근 입구 윤곽점이 8개 미만입니다. 입구 곡률 영역을 보정하세요.'
        return
    fm = item['line'][0]; wm = item['wall_line'][0]
    field_normal = np.array([-fm, 1.]) / math.hypot(fm, 1)
    wall_normal = np.array([-1., wm] if item['side']=='left' else [1., -wm]) / math.hypot(wm, 1)
    matrix = np.array([field_normal, wall_normal])
    if abs(np.linalg.det(matrix)) < 1e-5:
        connection['reason'] = 'Field와 cliff가 거의 평행해 접선 원호를 결정할 수 없습니다.'
        return
    direction = np.linalg.solve(matrix, np.ones(2))
    vertex = np.asarray(item['vertex'])
    maximum = max(8., float(np.ptp(points, axis=0).max()) * 4)

    def fit(p):
        def objective(r):
            residual = np.linalg.norm(p-(vertex+r*direction), axis=1)-r
            # Huber loss limits isolated contour outliers.
            absolute = np.abs(residual)
            return np.mean(np.where(absolute<=1.5, residual**2, 3*absolute-2.25))
        grid = np.linspace(.5, maximum, 200)
        index = int(np.argmin([objective(r) for r in grid]))
        lo, hi = grid[max(0,index-1)], grid[min(len(grid)-1,index+1)]
        for _ in range(40):
            a, b = lo+(hi-lo)/3, hi-(hi-lo)/3
            if objective(a)<objective(b): hi=b
            else: lo=a
        radius=(lo+hi)/2
        residual=np.linalg.norm(p-(vertex+radius*direction),axis=1)-radius
        return radius, float(np.sqrt(np.mean(residual**2)))

    radius, rms = fit(points)
    trim=max(1,int(round((len(points)-1)*.1)))
    stability=max(abs(fit(p)[0]-radius)/radius for p in (points[trim:],points[:-trim]))
    if corner.get('status')=='valid' and abs(radius-corner['radius_px'])>max(1.,corner['radius_px']*.1):
        connection['reason']='Field·cliff 기준과 독립 곡률 R이 일치하지 않습니다. 검색 영역과 직선 구간을 보정하세요.'
        return
    center=vertex+radius*direction
    ft=center-radius*field_normal; wt=center-radius*wall_normal
    a=math.atan2(*(ft-center)[::-1]); b=math.atan2(*(wt-center)[::-1])
    delta=(b-a+math.pi)%(2*math.pi)-math.pi
    connection.update(rms_px=rms,stability_fraction=stability,arc_angle_deg=abs(math.degrees(delta)),
                      fit_points=points.tolist(),model='circle_tangent_to_field_and_cliff')
    if radius<3 or (corner.get('edge_spread_px') is not None and radius<2*corner['edge_spread_px']):
        connection['reason']='입구 R이 해상도 또는 경계 흐림에 비해 작습니다.'
    elif abs(math.degrees(delta))<45:
        connection['reason']='접선 원호 각도가 45° 미만입니다.'
    elif rms>max(1.,radius*.05):
        connection['reason']='Field·cliff에 접하는 원호와 실제 윤곽의 맞춤 오차가 큽니다.'
    elif stability>.2:
        connection['reason']='윤곽 구간을 10% 조정하면 접선 R이 20% 넘게 변합니다.'
    else:
        theta=np.linspace(a,a+delta,max(80,int(radius*abs(delta)*2)))
        arc=center+radius*np.column_stack((np.cos(theta),np.sin(theta)))
        connection.update(status='valid',reason='측정 Field와 cliff에 접하는 입구 원호입니다.',
                          radius_px=radius,center=center.tolist(),field_tangent=ft.tolist(),
                          wall_tangent=wt.tolist(),arc_points=arc.tolist())


def _line(x, y):
    keep = np.ones(len(x), dtype=bool)
    for _ in range(8):
        if keep.sum() < 8:
            return None
        slope, intercept = np.polyfit(x[keep], y[keep], 1)
        residual = y - (slope*x + intercept)
        mad = 1.4826*np.median(np.abs(residual[keep]-np.median(residual[keep])))
        updated = np.abs(residual) <= max(1., 3*mad)
        if np.array_equal(updated, keep):
            break
        keep = updated
    slope, intercept = np.polyfit(x[keep], y[keep], 1)
    rms = float(np.sqrt(np.mean((y[keep]-(slope*x[keep]+intercept))**2)))
    return float(slope), float(intercept), rms, keep


def measure_field_angles(image, roi, y, left, right, search_px=32, angle_span=(40,100), bright=False):
    x0, top, width, _ = roi
    gray = to_gray(image).astype(np.float32)
    low, high = np.percentile(gray, (1,99))
    gray = np.clip((gray-low)*255/max(1e-9, high-low),0,255)
    if bright:
        gray = 255-gray
    gray = cv2.GaussianBlur(gray,(3,3),0)
    gradient = np.diff(gray, axis=0)
    results=[]
    for side, edge in (("left",left),("right",right)):
        item=dict(side=side,status="unconfirmed",reason="Field를 검출하지 못했습니다.",
                  points=[],line=None,rms_px=None,angle_status="unconfirmed",
                  length_px=None,tilt_deg=None,
                  angle_reason="Field 기준이 미확정입니다.",angle_deg=None,
                  wall_points=[],wall_line=None,wall_rms_px=None,vertex=None,
                  angle_span=list(angle_span),search_px=search_px)
        item.update(coordinate_unit="px",field_model="y=m*x+b",wall_model="x=m*y+b")
        results.append(item)
        # Keep the upper support separate from rounded transition/sidewall.
        margin=min(24,max(6,int((edge[0]-x0 if side=="left" else x0+width-edge[0])/2)))
        a,b=(x0,int(edge[0])-margin) if side=="left" else (int(edge[0])+margin,x0+width)
        a,b=max(1,a),min(gray.shape[1]-1,b)
        lo,hi=max(0,top-search_px),min(gradient.shape[0],top+search_px)
        if b-a<16 or hi-lo<8 or top<2:
            item['reason']='Field 바깥 재료 영역 또는 표면 위쪽 영상이 부족합니다.'
            continue
        xx=np.arange(a,b)
        block=gradient[lo:hi,a:b]
        distance=(np.arange(lo,hi)+.5-top)/max(6,search_px*.4)
        score=block*np.exp(-.5*distance[:,None]**2)
        index=np.argmax(score,axis=0)
        strengths=block[index,np.arange(len(xx))]
        rows=lo+index
        above=np.median(np.stack([gray[np.clip(rows-offset,0,len(gray)-1),xx] for offset in (2,3,4,5)]),axis=0)
        below=np.median(np.stack([gray[np.clip(rows+offset,0,len(gray)-1),xx] for offset in (2,3,4,5)]),axis=0)
        valid=(strengths>=8)&(index>0)&(index<hi-lo-1)&(below-above>=20)
        xx=xx[valid];index=index[valid];strengths=strengths[valid]
        if len(xx)<16:
            item['reason']='연속된 상단 재료/공간 명암 경계를 찾지 못했습니다.'
            continue
        # Parabolic peak interpolation places the edge between pixel centers.
        before=gradient[lo+index-1,xx];after=gradient[lo+index+1,xx]
        denom=before-2*strengths+after
        shift=np.divide(.5*(before-after),denom,out=np.zeros_like(denom),where=np.abs(denom)>1e-6)
        yy=lo+index+.5+np.clip(shift,-.5,.5)
        fit=_line(xx,yy)
        if fit is None:
            continue
        m,c,rms,keep=fit
        item.update(points=np.column_stack((xx[keep],yy[keep])).tolist(),line=[m,c],rms_px=rms)
        coverage=(xx[keep].max()-xx[keep].min())/max(1,b-a-1)
        if keep.sum()<16 or keep.sum()/len(xx)<.6 or coverage<.6 or rms>1.5 or abs(m)>math.tan(math.radians(30)):
            item['reason']='Field 지지 구간이 부족하거나 평탄한 표면의 직선 맞춤 오차가 큽니다.'
            continue
        item.update(status='valid',reason='실제 상단 윤곽의 직선 지지 구간으로 Field를 검출했습니다.',
                    length_px=float(np.ptp(xx[keep])*math.sqrt(1+m*m)),tilt_deg=math.degrees(math.atan(m)))
        reference=m*float(edge[0])+c
        mask=(y-reference>=angle_span[0])&(y-reference<=angle_span[1])
        wy,wx=y[mask],edge[mask]
        if len(wy)<16 or np.ptp(wy)<15:
            item['angle_reason']='각도 측벽 구간이 부족합니다. 시작/끝 깊이를 조정하세요.'
            continue
        fit=_line(wy,wx)
        if fit is None:
            continue
        wm,wc,wrms,wkeep=fit
        item.update(wall_points=np.column_stack((wx[wkeep],wy[wkeep])).tolist(),wall_line=[wm,wc],wall_rms_px=wrms)
        half=len(wy)//2
        slopes=[np.polyfit(wy[:half],wx[:half],1)[0],np.polyfit(wy[half:],wx[half:],1)[0]]
        change=abs(math.degrees(math.atan(slopes[0])-math.atan(slopes[1])))
        if wkeep.sum()<16 or wkeep.sum()/len(wy)<.6 or wrms>1.5 or change>5:
            item['angle_reason']='선택 측벽이 휘거나 구간별 기울기가 다릅니다. 직선 구간을 선택하세요.'
            continue
        if abs(1-m*wm)<1e-6:
            item['angle_reason']='Field와 측벽 직선의 교점을 정할 수 없습니다.'
            continue
        vy=(m*wc+c)/(1-m*wm);vx=wm*vy+wc
        if abs(vy-reference)>search_px or abs(vx-edge[0])>max(32,width*.3,np.ptp(wx)+16):
            item['angle_reason']='측벽 연장선과 Field의 교점이 입구에서 너무 멉니다.'
            continue
        outward=np.array([-1.,-m]) if side=='left' else np.array([1.,m])
        downward=np.array([wm,1.])
        angle=math.degrees(math.acos(np.clip(np.dot(outward,downward)/(np.linalg.norm(outward)*np.linalg.norm(downward)),-1,1)))
        item.update(angle_status='valid',angle_reason='검출 Field와 선택 측벽 직선 사이의 내각입니다.',angle_deg=angle,vertex=[vx,vy])
    return results
