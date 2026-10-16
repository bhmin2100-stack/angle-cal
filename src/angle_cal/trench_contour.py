"""Ordered, side-independent image contour tracing and cavity association.

Guides supply a search corridor, never measured coordinates. Trace evidence
along curve normals first; classify the connected cavity's walls afterwards.
"""
from __future__ import annotations

import math
import cv2
import numpy as np

from .trench_guides import sweep_samples, in_bands


def trace_guide(image, guide, threshold):
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
            peaks=supported&(b>=np.maximum(a,c))&(b>np.minimum(a,c))&(prominence>=5)
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
                found.append((offsets[k],float(.8*gradient[j]+.65*contrast[j]),'gradient',k))
        ranked=[]
        ridges=[offsets[j] for j in ridge_indices]
        dark=np.flatnonzero((np.convolve(profile,np.ones(3)/3,'same')<=threshold+8)&valid[i])
        for offset,strength,kind,k in sorted(found,key=lambda p:p[1],reverse=True):
            # Accept the first resolved boundary connected to nearby cavity
            # pixels. A farther/brighter material line behind another rim is
            # not the trench, even when it falls inside a wide guide corridor.
            nearest=[]
            before=dark[dark<k];after=dark[dark>k]
            if len(before):nearest.append(int(before[-1]))
            if len(after):nearest.append(int(after[0]))
            connected=any(abs(seed-k)<=24 and profile[k]-profile[seed]>=12
                          and not any(min(seed,k)<j<max(seed,k) for j in ridge_indices)
                          for seed in nearest)
            if not connected:continue
            # A resolved SEM rim measures at its bright crest, rather than
            # either steep flank. Suppress only nearby flanks, not all other
            # gradients elsewhere in the search corridor.
            if kind=='gradient' and any(abs(offset-ridge)<4 for ridge in ridges):continue
            if all(abs(offset-old[0])>=1.5 for old in ranked):ranked.append((offset,strength))
            if len(ranked)==6:break
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
    for i,point in sorted(chosen.items()):
        if not segments or i!=segments[-1][-1][0]+1:segments.append([])
        segments[-1].append((i,point))
    coverage=len(chosen)/len(centers)
    return dict(side='contour',requested_points=len(centers),detected_points=len(chosen),
                trace_coverage=coverage,status='valid' if coverage>=.95 else 'partial',
                reason='곡선 방향의 연결된 명암 경계' if coverage>=.95 else '명암 근거가 없는 구간은 윤곽을 만들지 않았습니다.',
                segments=[[point for _,point in segment] for segment in segments],
                points=[point for _,point in sorted(chosen.items())])


def contour_row_targets(traces):
    rows={}
    for trace in traces:
        for segment in trace['segments']:
            for a,b in zip(segment,segment[1:]):
                if abs(b[1]-a[1])<1e-8:continue
                for y in range(math.ceil(min(a[1],b[1])),math.floor(max(a[1],b[1]))+1):
                    xx=a[0]+(b[0]-a[0])*(y-a[1])/(b[1]-a[1])
                    values=rows.setdefault(y,[])
                    if not any(abs(xx-old)<.5 for old in values):values.append(float(xx))
    return rows


def guided_cavity_profile(seed_rows, ridge_image, guides, bands, traces, roi, rim_peak):
    """Associate the ordered contour with one complete, connected cavity.

    Each node is an already paired dark-space interval, rather than an
    independently selected left and right edge. Guide side labels are ignored.
    """
    x,top,width,height=roi
    targets=contour_row_targets(traces)
    spans=[(float(sweep_samples(g)[:,1].min()),float(sweep_samples(g)[:,1].max())) for g in guides]
    nodes=[];active=[]
    for row_index,seeds in enumerate(seed_rows):
        yy=top+row_index
        search=bands.get((row_index,'contour'),[])
        values=targets.get(yy,[])
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
                for side,choices in enumerate(attached):
                    if choices:
                        value=min(choices,key=lambda p:abs(p-(left if side==0 else right)))
                        if side==0:left=value
                        else:right=value
                if not (in_bands(left,search) or in_bands(right,search)):continue
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
    if len(profile)<max(12,height*.2):raise ValueError('가이드 주변에서 충분히 긴 연결 윤곽을 찾지 못했습니다.')
    return profile
