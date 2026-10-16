"""Human curve guides and bounded image evidence, in original pixels."""
from __future__ import annotations

import math
import numpy as np


def normalize_guides(guides, image_shape=None):
    result = []
    for guide in guides or ():
        try:
            side = guide['side']
            points = np.asarray(guide.get('points', [guide.get('start'),guide.get('end')]), float)
            radius = float(guide['half_width_px'])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError('가이드의 좌우·끝점·인식 폭을 확인하세요.') from exc
        if (side not in ('left', 'right') or points.ndim!=2 or points.shape[1:]!=(2,) or not 2<=len(points)<=200
                or not np.isfinite(points).all() or not math.isfinite(radius) or not 1 <= radius <= 300):
            raise ValueError('가이드 좌표와 인식 폭(1~300 px)이 올바르지 않습니다.')
        if np.any(np.linalg.norm(np.diff(points,axis=0),axis=1)<.1):
            raise ValueError('같은 위치의 점을 연속해서 찍지 마세요.')
        if np.ptp(points[:,1]) < 5:
            raise ValueError('측벽 곡선은 위아래로 5 px 이상 이어지도록 점을 찍어주세요.')
        if image_shape is not None:
            h, w = image_shape[:2]
            if any(not (0 <= p[0] <= w-1 and 0 <= p[1] <= h-1) for p in points):
                raise ValueError('곡선의 점은 원본 이미지 안에 있어야 합니다.')
        result.append(dict(side=side, points=points.tolist(), half_width_px=radius))
    return tuple(result)


def curve_samples(guide):
    """Centripetal cubic Hermite interpolation through every clicked point.

    Chord-derived timing prevents very unevenly spaced clicks from producing
    the large loops of a uniform cubic. Shared derivatives make joins C1.
    """
    p = np.asarray(guide['points'],float)
    distance = np.linalg.norm(np.diff(p,axis=0),axis=1)
    duration = np.sqrt(np.maximum(distance,1e-6))
    velocity = np.diff(p,axis=0)/duration[:,None]
    tangent=np.empty_like(p)
    tangent[0],tangent[-1]=velocity[0],velocity[-1]
    for i in range(1,len(p)-1):
        tangent[i]=(velocity[i-1]*duration[i]+velocity[i]*duration[i-1])/(duration[i-1]+duration[i])
    samples=[]
    for i in range(len(p)-1):
        count=max(3,min(4000,int(math.ceil(distance[i]*1.5))))
        t=np.linspace(0,1,count,endpoint=False)[:,None]
        samples.append((2*t**3-3*t**2+1)*p[i]+(t**3-2*t**2+t)*duration[i]*tangent[i]
                       +(-2*t**3+3*t**2)*p[i+1]+(t**3-t**2)*duration[i]*tangent[i+1])
    return np.vstack([*samples,p[-1:]])


def guide_polygon(guide):
    points=curve_samples(guide)
    delta=np.gradient(points,axis=0)
    normal=np.column_stack((-delta[:,1],delta[:,0]))/np.maximum(np.linalg.norm(delta,axis=1)[:,None],1e-8)
    offset=normal*guide['half_width_px']
    # Sidewall guides end at a depth, not at an oblique half-window. Horizontal
    # end caps keep the full search width at the last clicked depth; beyond
    # that depth automatic tracking resumes without an artificial missing row.
    for i in (0,-1):
        if abs(delta[i,1])>.1*np.linalg.norm(delta[i]):
            offset[i]-=delta[i]*(offset[i,1]/delta[i,1])
    polygon=np.vstack((points+offset,(points-offset)[::-1]))
    polygon[:,1]=np.clip(polygon[:,1],points[:,1].min(),points[:,1].max())
    return polygon


def polygon_intervals(polygon,y):
    a,b=polygon,np.roll(polygon,-1,axis=0)
    # Half-open crossing avoids counting a vertex twice.
    keep=(np.minimum(a[:,1],b[:,1])<=y)&(y<np.maximum(a[:,1],b[:,1]))
    aa,bb=a[keep],b[keep]
    xs=np.sort(aa[:,0]+(bb[:,0]-aa[:,0])*(y-aa[:,1])/(bb[:,1]-aa[:,1]))
    return [(float(lo),float(hi)) for lo,hi in zip(xs[::2],xs[1::2]) if hi-lo>1e-6]


def prepare_guide_bands(guides, top, height):
    polygons=[guide_polygon(g) for g in guides]
    per_guide=[{yy:polygon_intervals(polygon,yy) for yy in range(top,top+height)} for polygon in polygons]
    bands={(yy-top,side):[b for g,rows in zip(guides,per_guide) if g['side']==side for b in rows[yy]]
           for yy in range(top,top+height) for side in ('left','right')}
    return bands,per_guide


def row_bands(guides, y, side):
    return [interval for g in guides if g['side']==side for interval in polygon_intervals(guide_polygon(g),y)]


def in_bands(x, bands):
    return any(lo-1e-7 <= x <= hi+1e-7 for lo, hi in bands)


def guided_edges(row, bands, direction, x_origin=0):
    """Return evidenced ridge/gradient candidates strictly within each band.

    Candidate pixels, interpolation and contrast support all stay in the
    search window. A guide's position is only a preference, never an edge.
    """
    found = []
    for lower, upper in bands:
        lo, hi = max(0, math.ceil(lower-x_origin)), min(len(row)-1, math.floor(upper-x_origin))
        if hi-lo < 3:
            continue
        center = (lower+upper)/2-x_origin
        scale = max(2., (upper-lower)/3)
        ridges, gradients = [], []
        for k in range(lo+1, hi):
            a, b, c = map(float, row[k-1:k+2])
            inside = row[max(lo,k-4):k] if direction==1 else row[k+1:min(hi+1,k+5)]
            outside = row[k+1:min(hi+1,k+5)] if direction==1 else row[max(lo,k-4):k]
            if not len(inside) or not len(outside):
                continue
            inner, outer = float(np.median(inside)), float(np.median(outside))
            if b >= max(a,c) and b > min(a,c) and b-inner >= 12 and b-outer >= 5:
                denom = a-2*b+c
                offset = .5*(a-c)/denom if denom < -1e-6 else 0.
                xx = k+float(np.clip(offset,-.5,.5))
                strength = b-inner
                ridges.append((xx,strength))
        difference = np.diff(row[lo:hi+1])*direction
        for j, strength in enumerate(difference):
            if strength < 8 or (j and difference[j-1]>strength) or (j+1<len(difference) and difference[j+1]>strength):
                continue
            xx = lo+j+.5
            gradients.append((xx,float(strength)))
        # A bright SEM ridge defines the boundary when it is resolved.
        for xx, strength in ridges or gradients:
            if lower <= xx+x_origin <= upper:
                score = strength*math.exp(-.5*((xx-center)/scale)**2)
                found.append((float(xx+x_origin),score))
    found.sort(key=lambda item:item[1], reverse=True)
    return [x for x, _ in found[:4]]
