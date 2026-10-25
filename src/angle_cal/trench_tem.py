"""Image-contrast selection and TEM gray-material interface evidence.

This measures a grayscale transition, not the physical position of an atomistic
interface. No image header, file name or sample-specific coordinates are used.
"""
import cv2
import numpy as np
from .image_ops import to_gray
from .trench_guides import sweep_samples


def void_normal_sign(centers):
    # An open U contour is ordered field -> wall -> floor -> wall -> field.
    # Reversing the entire guide reverses both the tangent and this sign.
    if np.ptp(centers[:, 0]) < 8 or np.ptp(centers[:, 1]) < 16:
        return None
    if abs(centers[-1, 0]-centers[0, 0]) < 8:
        return None
    return -1 if centers[-1, 0] > centers[0, 0] else 1


def image_contrast_mode(image, guides):
    gray = to_gray(image).astype(np.float32)
    low, high = np.percentile(gray, (1, 99))
    gray = (gray-low)*255/max(float(high-low), 1e-6)
    differences = []
    for guide in guides:
        centers = sweep_samples(guide)[::3]
        sign = void_normal_sign(centers)
        if sign is None:
            continue
        tangent = np.gradient(centers, axis=0)
        normals = np.column_stack((-tangent[:, 1], tangent[:, 0]))
        normals /= np.maximum(np.linalg.norm(normals, axis=1)[:, None], 1e-9)
        vertical = np.abs(normals[:, 0]) > .85
        mid = (centers[:, 1] > centers[:, 1].min()+40) & (centers[:, 1] < centers[:, 1].max()-40)
        centers, normals = centers[vertical & mid], normals[vertical & mid]
        if len(centers) < 12:
            continue
        levels = []
        valid = np.ones(len(centers), bool)
        for direction in (sign, -sign):
            xy = centers[:, None, :]+direction*normals[:, None, :]*np.arange(18, 29)[None, :, None]
            valid &= ((xy[:, :, 0] >= 0) & (xy[:, :, 0] < gray.shape[1]-1) &
                      (xy[:, :, 1] >= 0) & (xy[:, :, 1] < gray.shape[0]-1)).all(axis=1)
            values = cv2.remap(gray, xy[:, :, 0].astype(np.float32), xy[:, :, 1].astype(np.float32), cv2.INTER_LINEAR)
            levels.append(np.median(values, axis=1))
        differences.extend((levels[0]-levels[1])[valid].tolist())
    contrast = float(np.median(differences)) if differences else 0.
    agreement = float(np.mean(np.asarray(differences)>12)) if differences else 0.
    return dict(selected='tem' if contrast>18 and agreement>.7 else 'vsem',
                void_minus_material=contrast, bright_void_fraction=agreement,
                sampled_normals=len(differences), basis='guide_normal_contrast')


def _void_flanks(profile, offsets, valid, radius, sign):
    """Subpixel maximum slope on the void side of a dark band (inverted input).

    Only final positions in the guide radius are candidates. Exterior context
    checks side polarity, film-to-void contrast and a sufficiently flat void.
    """
    if sign is None:
        return []
    slope = -sign*np.gradient(profile)
    found = []
    for k in np.flatnonzero((np.abs(offsets)<=radius) & (slope>=3)):
        if k<16 or k+16>=len(profile) or not valid[k-16:k+17].all():
            continue
        if slope[k]<max(slope[k-1], slope[k+1]):
            continue
        inside = profile[k+sign*np.arange(5, 13)]
        band = profile[k-sign*np.arange(2, 9)]
        contrast = float(np.max(band)-np.median(inside))
        # Reject a stripe inside material or a shading ripple in empty space.
        if contrast<20 or np.ptp(inside)>max(20., contrast*.25):
            continue
        if np.median(inside)>np.percentile(profile[k-16:k+17], 20)+max(8., contrast*.12):
            continue
        a,b,c = slope[k-1:k+2]
        denom = float(a-2*b+c)
        sub = float(np.clip(.5*(a-c)/denom, -.5, .5)) if denom<-1e-6 else 0.
        offset = offsets[k]+sub
        if abs(offset)<=radius:
            found.append((float(offset), float(2*slope[k]+.3*contrast)))
    return sorted(found, key=lambda item:item[1], reverse=True)[:6]


def interface_candidates(profile, offsets, valid, radius, sign):
    """Locate the gray-material side, treating the dark band as excluded.

    The void-facing fall identifies which band belongs to the cavity. Walk
    back across that same band and choose its material-facing rise. The inner
    edge is context only: it is never exported as the measured interface.
    """
    if sign is None:
        return []
    rise = sign*np.gradient(profile)
    found=[]
    for inner,_ in _void_flanks(profile,offsets,valid,radius+24,sign):
        k=int(np.argmin(abs(offsets-inner)))
        indices=k-sign*np.arange(2,33)
        indices=indices[(indices>=12)&(indices<len(profile)-12)]
        if not len(indices):continue
        band_level=float(np.max(profile[indices]))
        void_level=float(np.median(profile[k+sign*np.arange(5,13)]))
        candidates=[]
        for j in indices:
            if abs(offsets[j])>radius or rise[j]<3 or rise[j]<max(rise[j-1],rise[j+1]):continue
            if not valid[j-12:j+13].all():continue
            outside=float(np.median(profile[j-sign*np.arange(5,12)]))
            contrast=band_level-outside
            # Film texture near its inner edge is too close to the band level;
            # the material-facing transition lies between gray and dark band.
            if contrast<15 or outside<void_level+8:continue
            fraction=(profile[j]-outside)/max(1.,contrast)
            if not .1<=fraction<=.85:continue
            a,b,c=rise[j-1:j+2];denom=float(a-2*b+c)
            sub=float(np.clip(.5*(a-c)/denom,-.5,.5)) if denom<-1e-6 else 0.
            offset=float(offsets[j]+sub)
            if abs(offset)<=radius:candidates.append((offset,float(2*rise[j]+.3*contrast)))
        if candidates:
            best=max(candidates,key=lambda item:item[1])
            if all(abs(best[0]-old[0])>1.5 for old in found):found.append(best)
    return sorted(found,key=lambda item:item[1],reverse=True)[:6]
