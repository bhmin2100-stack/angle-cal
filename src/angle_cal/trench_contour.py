"""Ordered, side-independent image contour tracing and cavity association.

Guides supply a search corridor, never measured coordinates. Trace evidence
along curve normals first; classify the connected cavity's walls afterwards.
"""
from __future__ import annotations

import math
import cv2
import numpy as np

from .trench_guides import sweep_samples, in_bands
from .image_ops import _boundary_profile_index


def trace_guide(image, guide, threshold, boundary_mode='inner_rim'):
    samples=sweep_samples(guide)
    arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(samples,axis=0),axis=1))]
    step=max(1.,arc[-1]/20000)
    distance=np.linspace(0,arc[-1],max(2,int(math.ceil(arc[-1]/step))+1))
    centers=np.column_stack([np.interp(distance,arc,samples[:,axis]) for axis in (0,1)])
    tangent=np.gradient(centers,axis=0)
    lengths=np.linalg.norm(tangent,axis=1)
    normals=np.column_stack((-tangent[:,1],tangent[:,0]))/np.maximum(lengths[:,None],1e-12)
    radius=float(guide['half_width_px'])
    # Context may extend outside the corridor for contrast/connectivity tests;
    # every candidate position itself must stay within the configured radius.
    offsets=np.arange(-math.ceil(radius)-24,math.ceil(radius)+25,dtype=float)
    xy=centers[:,None,:]+normals[:,None,:]*offsets[None,:,None]
    valid=(xy[:,:,0]>=0)&(xy[:,:,0]<image.shape[1]-1)&(xy[:,:,1]>=0)&(xy[:,:,1]<image.shape[0]-1)
    profiles=cv2.remap(image.astype(np.float32,copy=False),xy[:,:,0].astype(np.float32),
                       xy[:,:,1].astype(np.float32),cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
    material=image[image>threshold+20]
    material_level=float(np.percentile(material,75)) if len(material) else threshold+20
    candidates=[]
    ks=np.arange(6,len(offsets)-6)
    far=(ks>=15)&(ks+15<len(offsets))
    for i,profile in enumerate(profiles):
        found=[];ridge_indices=[]
        if lengths[i]>1e-10:
            prefix=np.r_[0.,np.cumsum(profile,dtype=float)]
            invalid=np.r_[0,np.cumsum(~valid[i])]
            before=(prefix[ks-2]-prefix[ks-6])/4
            after=(prefix[ks+7]-prefix[ks+3])/4
            contrast=np.abs(after-before)
            far_ks=ks[far]
            far_valid=invalid[far_ks+16]==invalid[far_ks-15]
            far_contrast=np.abs((prefix[far_ks-8]-prefix[far_ks-14])/6-
                                (prefix[far_ks+15]-prefix[far_ks+9])/6)
            contrast[far]=np.maximum(contrast[far],np.where(far_valid,far_contrast,0))
            supported=(invalid[ks+7]==invalid[ks-6])&(contrast>=8)
            a,b,c=profile[ks-1],profile[ks],profile[ks+1]
            prominence=b-np.maximum(before,after)
            # A broad material-layer plateau is not a resolved bright rim.
            # Require a local crest before suppressing its gradient flanks.
            peaks=supported&(b>=np.maximum(a,c))&(b-(a+c)/2>=.5)&(prominence>=5)
            ridge_indices=ks[peaks].tolist()
            for j in np.flatnonzero(peaks):
                k=ks[j];denom=float(a[j]-2*b[j]+c[j])
                sub=float(np.clip(.5*(float(a[j])-float(c[j]))/denom,-.5,.5)) if denom<-1e-6 else 0.
                offset=offsets[k]+sub
                if abs(offset)<=radius:
                    found.append((offset,float(1.25*prominence[j]+.65*contrast[j]),'ridge',k))
            gradient=np.abs(c-a)/2
            steep=supported&(np.abs(offsets[ks])<=radius)&(gradient>=8)
            steep&=(gradient>=np.abs(b-profile[ks-2])/2)&(gradient>=np.abs(profile[ks+2]-b)/2)
            for j in np.flatnonzero(steep):
                k=ks[j]
                before_g=abs(float(b[j]-profile[k-2]))/2
                after_g=abs(float(profile[k+2]-b[j]))/2
                denom=before_g-2*float(gradient[j])+after_g
                shift=float(np.clip(.5*(before_g-after_g)/denom,-.5,.5)) if denom<-1e-6 else 0.
                if abs(offsets[k]+shift)<=radius:
                    found.append((offsets[k]+shift,float(.8*gradient[j]+.65*contrast[j]),'gradient',k))
        ranked=[]
        dark=np.flatnonzero((np.convolve(profile,np.ones(3)/3,'same')<=threshold+8)&valid[i])
        barriers=ridge_indices[:]
        # A bright rim must rise above the material background as well as
        # the void. A finite bright layer plateau is otherwise mistaken for
        # a crest halfway inside the solid. Other local peaks still block
        # borrowing a cavity through an intervening real boundary.
        resolved=[k for k in ridge_indices if profile[k]>=material_level+5]
        ridge_indices=resolved
        found=[candidate for candidate in found if candidate[2]!='ridge' or candidate[3] in resolved]
        ridges=[offsets[j] for j in ridge_indices]
        for offset,strength,kind,k in sorted(found,key=lambda p:p[1],reverse=True):
            # Accept the first resolved boundary connected to nearby cavity
            # pixels. A farther/brighter material line behind another rim is
            # not the trench, even when it falls inside a wide guide corridor.
            nearest=[]
            before=dark[dark<k];after=dark[dark>k]
            if len(before):nearest.append(int(before[-1]))
            if len(after):nearest.append(int(after[0]))
            connected=any(abs(seed-k)<=24 and profile[k]-profile[seed]>=12
                          and not any(min(seed,k)<j<max(seed,k) for j in barriers)
                          for seed in nearest)
            if not connected:continue
            # A resolved SEM rim measures at its bright crest, rather than
            # either steep flank. Suppress only nearby flanks, not all other
            # gradients elsewhere in the search corridor.
            if boundary_mode=='inner_rim' and kind=='gradient' and any(abs(offset-ridge)<4 for ridge in ridges):continue
            if all(abs(offset-old[0])>=1.5 for old in ranked):ranked.append((offset,strength))
            if len(ranked)==6:break
        if ranked and boundary_mode!='inner_rim':
            positions=np.array([c[0] for c in ranked])
            levels=np.interp(positions,offsets,profile)
            slopes=np.interp(positions,offsets,np.abs(np.gradient(profile)))
            selected=_boundary_profile_index(positions,levels,slopes,boundary_mode)
            ranked=[(offset,strength+(96 if index==selected else 0)) for index,(offset,strength) in enumerate(ranked)]
        candidates.append(ranked)
    # Viterbi along the ordered guide. Keep multiple candidates until the
    # complete local path is evaluated; a strong isolated peak cannot win a
    # row independently. Missing samples remain missing, never interpolated.
    chosen={}
    blocks=[]
    supported=[i for i,c in enumerate(candidates) if c]
    for i in supported:
        if not blocks or i-blocks[-1][-1]>4:blocks.append([])
        blocks[-1].append(i)
    for indices in blocks:
        costs=[];parents=[]
        for j,i in enumerate(indices):
            current=np.array(candidates[i],float)
            unary=1-current[:,1]/max(64.,float(current[:,1].max()))+.15*(current[:,0]/radius)**2
            if j:
                previous=np.array(candidates[indices[j-1]],float)
                ds=max(1.,distance[i]-distance[indices[j-1]])
                transitions=costs[-1][:,None]+.06*((previous[:,0,None]-current[None,:,0])/ds)**2
                parent=np.argmin(transitions,axis=0)
                costs.append(unary+transitions[parent,np.arange(len(current))]);parents.append(parent)
            else:
                costs.append(unary);parents.append(None)
        selected=int(np.argmin(costs[-1]))
        for j in range(len(indices)-1,-1,-1):
            i=indices[j];offset=candidates[i][selected][0]
            chosen[i]=(centers[i]+normals[i]*offset).tolist()
            if j:selected=int(parents[j][selected])
    segments=[]
    spreads=[]
    for i,point in sorted(chosen.items()):
        profile=profiles[i];derivative=np.abs(np.gradient(profile))
        offset=float(np.dot(np.asarray(point)-centers[i],normals[i]))
        k=int(np.argmin(np.abs(offsets-offset)))
        lo,hi=max(1,k-4),min(len(profile)-1,k+5)
        k=lo+int(np.argmax(derivative[lo:hi]));half=derivative[k]/2
        a=b=k
        while a>0 and derivative[a]>=half:a-=1
        while b<len(profile)-1 and derivative[b]>=half:b+=1
        if half>1 and a>0 and b<len(profile)-1:
            ac=a+(half-derivative[a])/max(1e-9,derivative[a+1]-derivative[a])
            bc=b-(half-derivative[b])/max(1e-9,derivative[b-1]-derivative[b])
            spreads.append(float((bc-ac)*1.088))
        else:spreads.append(None)
    for i,point in sorted(chosen.items()):
        if not segments or i!=segments[-1][-1][0]+1:segments.append([])
        segments[-1].append((i,point))
    coverage=len(chosen)/len(centers)
    return dict(side='contour',requested_points=len(centers),detected_points=len(chosen),
                trace_coverage=coverage,status='valid' if coverage>=.95 else 'partial',
                reason='곡선 방향의 연결된 명암 경계' if coverage>=.95 else '명암 근거가 없는 구간은 윤곽을 만들지 않았습니다.',
                segments=[[point for _,point in segment] for segment in segments],
                points=[point for _,point in sorted(chosen.items())],edge_spreads_px=spreads)


def contour_row_targets(traces):
    rows={}
    for trace in traces:
        for segment in trace['segments']:
            # A subpixel displacement at an open trace endpoint must not
            # create a false missing integer row (e.g. y=30.005 vs row 30).
            for point in (segment[:1]+segment[-1:]):
                y=round(point[1])
                if abs(point[1]-y)<=.25:
                    values=rows.setdefault(y,[])
                    if not any(abs(point[0]-old)<.5 for old in values):values.append(float(point[0]))
            for a,b in zip(segment,segment[1:]):
                # Horizontal Field supports and floor tangents are not wall
                # targets. They remain in the ordered trace for surface fits.
                if abs(b[1]-a[1])<.25*max(1e-8,np.linalg.norm(np.subtract(b,a))):continue
                if abs(b[1]-a[1])<1e-8:continue
                for y in range(math.ceil(min(a[1],b[1])),math.floor(max(a[1],b[1]))+1):
                    xx=a[0]+(b[0]-a[0])*(y-a[1])/(b[1]-a[1])
                    values=rows.setdefault(y,[])
                    if not any(abs(xx-old)<.5 for old in values):values.append(float(xx))
    return rows


def guided_cavity_profile(seed_rows, ridge_image, guides, bands, traces, roi, rim_peak, strict=False):
    """Associate the ordered contour with one complete, connected cavity.

    Each node is an already paired dark-space interval, rather than an
    independently selected left and right edge. Guide side labels are ignored.
    """
    x,top,width,height=roi
    # Classify the outer boundary of the same 2-D cavity, including rows where
    # a bright inclusion divides its dark space into several intervals. Omit
    # ROI-border air intervals before labeling so neighboring trenches cannot
    # merge through the open space above their entrances.
    mask=np.zeros((height,width),np.uint8)
    for row,seeds in enumerate(seed_rows):
        for left,right in seeds:mask[row,math.ceil(left):math.floor(right)+1]=1
    _,labels=cv2.connectedComponents(mask,connectivity=8)
    connected_rows=[]
    for row,seeds in enumerate(seed_rows):
        regions={}
        for left,right in seeds:
            label=int(labels[row,int((left+right)/2)])
            if not label:continue
            if label not in regions:regions[label]=[left,right]
            else:
                regions[label][0]=min(regions[label][0],left)
                regions[label][1]=max(regions[label][1],right)
        connected_rows.append(list(regions.values()))
    seed_rows=connected_rows
    targets=contour_row_targets(traces)
    spans=[(float(sweep_samples(g)[:,1].min()),float(sweep_samples(g)[:,1].max())) for g in guides]
    nodes=[];active=[]
    for row_index,seeds in enumerate(seed_rows):
        yy=top+row_index
        search=bands.get((row_index,'contour'),[])
        values=targets.get(yy,[])
        if strict and not search:continue
        body=any(lo<=yy<=hi for lo,hi in spans)
        active=[i for i in active if row_index-nodes[i]['row']<=4]
        current=[]
        for seed_left,seed_right in seeds:
            left=float(rim_peak(ridge_image[row_index],seed_left,-1))+x
            right=float(rim_peak(ridge_image[row_index],seed_right,1))+x
            support=0
            if search:
                if not values and body:continue
                attached=[[],[]]
                for value in values:
                    distances=[abs(value-(seed_left+x)),abs(value-(seed_right+x))]
                    side=int(np.argmin(distances))
                    if distances[side]<=19 and in_bands(value,search):attached[side].append(value)
                # All simultaneously traced branches must describe this same
                # cavity, not independent walls belonging to two trenches.
                support=sum(map(len,attached))
                if values and support<len(values):continue
                if strict and body and not all(attached):continue
                for side,choices in enumerate(attached):
                    if choices:
                        value=min(choices,key=lambda p:abs(p-(left if side==0 else right)))
                        if side==0:left=value
                        else:right=value
                if strict:
                    if not (in_bands(left,search) and in_bands(right,search)):continue
                elif not (in_bands(left,search) or in_bands(right,search)):continue
            if right-left<=2:continue
            reward=1+3*min(2,support)
            best=0.;parent=None
            for i in active:
                previous=nodes[i];gap=row_index-previous['row']
                overlap=min(right,previous['right'])-max(left,previous['left'])
                jump=abs((left+right-previous['left']-previous['right'])/2)
                if overlap<=0 or jump>max(6,width*.2)*gap:continue
                movement=((left-previous['left'])**2+(right-previous['right'])**2)/gap**2
                score=previous['score']-min(.5,.02*movement)-.3*(gap-1)
                if score>best:best,parent=score,i
            nodes.append(dict(row=row_index,left=left,right=right,score=best+reward,parent=parent,
                              support=support+(nodes[parent]['support'] if parent is not None else 0)))
            current.append(len(nodes)-1)
        active.extend(current)
    eligible=[i for i,n in enumerate(nodes) if n['support']>=4]
    if not eligible:raise ValueError('가이드의 연결된 윤곽과 같은 Trench 영역을 찾지 못했습니다. 범위 안의 실제 경계와 곡선 위치를 확인하세요.')
    selected=max(eligible,key=lambda i:nodes[i]['score'])
    profile=[]
    while selected is not None:
        node=nodes[selected];profile.append((node['row'],node['left']-x,node['right']-x));selected=node['parent']
    profile=np.array(profile[::-1],float)
    expected_height=min(height,max(hi for lo,hi in spans)-min(lo for lo,hi in spans)+1) if strict else height
    if len(profile)<max(12,expected_height*.2):raise ValueError('가이드 범위 안에서 양쪽 측벽의 충분히 긴 연결 윤곽을 찾지 못했습니다.')
    return profile
