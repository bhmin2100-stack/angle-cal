"""Human curve guides and bounded image evidence, in original pixels."""
from __future__ import annotations

import math
from functools import lru_cache
import cv2
import numpy as np


def normalize_guides(guides, image_shape=None):
    result = []
    for guide in guides or ():
        try:
            side = guide.get('side','contour')
            points = np.asarray(guide.get('points', [guide.get('start'),guide.get('end')]), float)
            radius = float(guide['half_width_px'])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError('가이드의 좌우·끝점·인식 폭을 확인하세요.') from exc
        if (side not in ('left', 'right','contour') or points.ndim!=2 or points.shape[1:]!=(2,) or not 2<=len(points)<=200
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
    return _curve_samples(tuple(tuple(map(float,p)) for p in guide['points']))


@lru_cache(maxsize=16)
def _curve_samples(points):
    p = np.asarray(points,float)
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
    result=np.vstack([*samples,p[-1:]])
    result.setflags(write=False)
    return result


def sweep_samples(guide):
    """Shared centerline for display and search, within .02 px of dense samples."""
    return _sweep_samples(tuple(tuple(map(float,p)) for p in guide['points']))


@lru_cache(maxsize=16)
def _sweep_samples(points):
    dense=_curve_samples(points)
    # Remove nearly collinear vertices before the stroke union. Without this,
    # thousands of tiny joins make wheel/drag previews needlessly expensive.
    # Simplify each clicked span separately so every user knot stays exact.
    result=[]
    start=0
    for a,b in zip(points,points[1:]):
        count=max(3,min(4000,int(math.ceil(np.linalg.norm(np.subtract(b,a))*1.5))))
        span=dense[start:start+count+1]
        origin=span[0]
        reduced=cv2.approxPolyDP((span-origin).astype(np.float32),.02,False).reshape(-1,2).astype(float)+origin
        reduced[0],reduced[-1]=span[0],span[-1]
        result.extend(reduced[:-1])
        start+=count
    samples=np.vstack([*result,dense[-1]])
    samples.setflags(write=False)
    return samples


@lru_cache(maxsize=16)
def _sweep_geometry(points, radius):
    """A disk swept along each sampled segment is a capsule, even at turns.

    The union of endpoint disks and segment rectangles has no inward-offset
    folds, parity holes, or depth clipping. Rendering uses the same centerline
    with a round stroke of diameter 2*radius, in original image coordinates.
    """
    samples=_sweep_samples(points)
    a,b=samples[:-1],samples[1:]
    delta=b-a
    lengths=np.linalg.norm(delta,axis=1)
    valid=lengths>1e-10
    a,b,delta,lengths=a[valid],b[valid],delta[valid],lengths[valid]
    normal=np.column_stack((-delta[:,1],delta[:,0]))*(radius/lengths)[:,None]
    quads=np.stack((a+normal,b+normal,b-normal,a-normal),axis=1)
    return samples,quads,np.roll(quads,-1,axis=1)


def _merge_intervals(intervals):
    merged=[]
    for lo,hi in sorted(intervals):
        if hi-lo<=1e-7:
            continue  # a tangent point alone cannot contain image evidence
        if merged and lo<=merged[-1][1]+1e-7:
            merged[-1]=(merged[-1][0],max(hi,merged[-1][1]))
        else:
            merged.append((float(lo),float(hi)))
    return merged


def guide_intervals(guide,y):
    """Exact row intersection of the disk sweep along the sampled curve."""
    radius=float(guide['half_width_px'])
    points=tuple(tuple(map(float,p)) for p in guide['points'])
    samples,a,b=_sweep_geometry(points,radius)
    if y<samples[:,1].min()-radius or y>samples[:,1].max()+radius:
        return []
    dy=y-samples[:,1]
    selected=np.abs(dy)<=radius
    half=np.sqrt(np.maximum(0,radius*radius-dy[selected]**2))
    xx=samples[selected,0]
    intervals=list(zip(xx-half,xx+half))
    # Each connecting rectangle is convex, so its valid edge crossings bound
    # one interval. Closed crossings also retain horizontal boundary edges.
    delta=b-a
    nonhorizontal=np.abs(delta[:,:,1])>1e-10
    keep=nonhorizontal & (np.minimum(a[:,:,1],b[:,:,1])<=y) & (y<=np.maximum(a[:,:,1],b[:,:,1]))
    ratio=np.divide(y-a[:,:,1],delta[:,:,1],out=np.zeros_like(a[:,:,1]),where=nonhorizontal)
    crossings=a[:,:,0]+ratio*delta[:,:,0]
    low=np.min(np.where(keep,crossings,np.inf),axis=1)
    high=np.max(np.where(keep,crossings,-np.inf),axis=1)
    horizontal=(~nonhorizontal)&(np.abs(y-a[:,:,1])<1e-10)
    low=np.minimum(low,np.min(np.where(horizontal,np.minimum(a[:,:,0],b[:,:,0]),np.inf),axis=1))
    high=np.maximum(high,np.max(np.where(horizontal,np.maximum(a[:,:,0],b[:,:,0]),-np.inf),axis=1))
    intervals.extend(zip(low[low<=high],high[low<=high]))
    return _merge_intervals(intervals)


def prepare_guide_bands(guides, top, height):
    per_guide=[{yy:guide_intervals(g,yy) for yy in range(top,top+height)} for g in guides]
    bands={(yy-top,side):_merge_intervals([b for g,rows in zip(guides,per_guide) if g['side']==side for b in rows[yy]])
           for yy in range(top,top+height) for side in ('left','right')}
    bands.update({(yy-top,'contour'):_merge_intervals([b for rows in per_guide for b in rows[yy]])
                  for yy in range(top,top+height)})
    return bands,per_guide


def row_bands(guides, y, side):
    return _merge_intervals([interval for g in guides if side=='contour' or g['side']==side for interval in guide_intervals(g,y)])


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
