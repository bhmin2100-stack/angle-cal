"""Bright-crest reference and inward normal offset in original pixels."""
import numpy as np
from .trench_guides import points_in_guides


def inset_rim(chosen, centers, normals, profiles, offsets, guide, image_shape, amount):
    shifted={};peaks={};directions={}
    for i,point in sorted(chosen.items()):
        p=np.asarray(point,float)
        offset=float(np.dot(p-centers[i],normals[i]))
        # Compare beyond the bright band; darker means void after polarity
        # normalization. Guide drawing order cannot reverse this rule.
        profile=profiles[i]
        before=np.interp(offset-np.arange(24,49),offsets,profile)
        after=np.interp(offset+np.arange(24,49),offsets,profile)
        difference=float(np.median(before)-np.median(after))
        near_before=np.interp(offset-np.arange(6,17),offsets,profile)
        near_after=np.interp(offset+np.arange(6,17),offsets,profile)
        near_difference=float(np.median(near_before)-np.median(near_after))
        if abs(difference)<8 and abs(near_difference)>=8:difference=near_difference
        if amount and abs(difference)<3:continue
        sign=1 if difference>0 else -1
        neighbors=[j for j in range(max(0,i-4),i+5) if j in chosen]
        tangent=None
        if len(neighbors)>=3:
            t=np.asarray(neighbors,float);t-=t.mean()
            tangent=t@np.asarray([chosen[j] for j in neighbors])
        if tangent is None or np.linalg.norm(tangent)<1e-6:
            normal=normals[i]
        else:
            normal=np.array([-tangent[1],tangent[0]])/np.linalg.norm(tangent)
            if np.dot(normal,normals[i])<0:normal=-normal
        direction=normal*sign
        q=p+amount*direction
        h,w=image_shape[:2]
        if np.any(q<0) or q[0]>w-1 or q[1]>h-1:continue
        if not points_in_guides([q],(guide,))[0]:continue
        shifted[i]=q.tolist();peaks[i]=p.tolist();directions[i]=direction.tolist()
    return shifted,peaks,directions
