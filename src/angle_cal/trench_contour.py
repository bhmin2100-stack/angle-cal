"""Ordered, side-independent image contour tracing and cavity association.

Guides supply a search corridor, never measured coordinates. Trace evidence
along curve normals first; classify the connected cavity's walls afterwards.
"""
from __future__ import annotations

import math
import cv2
import numpy as np

from .trench_guides import sweep_samples, in_bands, points_in_guides
from .image_ops import _boundary_profile_index
from .trench_rim import inset_rim
from .trench_tem import interface_candidates, void_normal_sign
from .trench_boundary_sample import prepare_templates, profile_candidates


def _consensus_path(indices,candidates,distance,centers,normals,radius):
    """Optimize a supported path with an explicit missing-observation option.

    A contaminated row must not force the entire path onto a competing edge.
    Skipped observations are retained separately from subsequent short joins.
    """
    costs=[];parents=[]
    for j,i in enumerate(indices):
        current=np.asarray(candidates[i],float)
        unary=1-current[:,1]/max(64.,float(current[:,1].max()))+.15*(current[:,0]/radius)**2
        best=np.full(len(current),np.inf);parent=[None]*len(current)
        if j==0:best=unary
        for prev_j in range(j-1,max(-1,j-5),-1):
            prev_i=indices[prev_j]
            gap=i-prev_i
            if gap>4:break
            previous=np.asarray(candidates[prev_i],float)
            ds=max(1.,distance[i]-distance[prev_i])
            # Saturation permits a sustained real notch/corner; an unbounded
            # straightness penalty would prefer weak noise over that feature.
            movement=.3*((previous[:,0,None]-current[None,:,0])/ds)**2
            transition=costs[prev_j][:,None]+np.minimum(6.,movement)+1.25*(gap-1)
            selected=np.argmin(transition,axis=0)
            score=unary+transition[selected,np.arange(len(current))]
            for k in np.flatnonzero(score<best):
                best[k]=score[k];parent[k]=(prev_j,int(selected[k]))
        costs.append(best);parents.append(parent)
    tail=[]
    for j in range(max(0,len(indices)-4),len(indices)):
        k=int(np.argmin(costs[j]));tail.append((costs[j][k]+1.25*(indices[-1]-indices[j]),j,k))
    _,j,k=min(tail)
    chosen={}
    while True:
        i=indices[j];offset=candidates[i][k][0]
        chosen[i]=(centers[i]+normals[i]*offset).tolist()
        parent=parents[j][k]
        if parent is None:break
        j,k=parent
    return chosen


def _connect_short_gaps(chosen, distance, guide, image_shape):
    """Join small holes between supported endpoints, never extrapolate ends.

    Cubic endpoint tangents come from observed contour coordinates, not the
    user's guide. Check the entire sampled join against the search corridor.
    Long holes, incompatible branches and out-of-image joins remain unknown.
    """
    connected=dict(chosen);joins=[]
    indices=sorted(chosen)
    for a,b in zip(indices,indices[1:]):
        if b==a+1 or distance[b]-distance[a]>12:continue
        before_indices=[k for k in range(max(0,a-6),a) if k in chosen][-2:]
        after_indices=[k for k in range(b+1,b+7) if k in chosen][:2]
        if len(before_indices)<2 or len(after_indices)<2:continue
        p,q=np.asarray(chosen[a]),np.asarray(chosen[b])
        chord=q-p;length=float(np.linalg.norm(chord))
        if length<.1 or length>max(12.,1.8*(distance[b]-distance[a])):continue
        before=p-np.asarray(chosen[before_indices[0]]);after=np.asarray(chosen[after_indices[-1]])-q
        sizes=np.linalg.norm([before,after],axis=1)
        if np.any(sizes<.1):continue
        u,v=before/sizes[0],after/sizes[1]
        if np.dot(u,v)<-.5:continue
        # A local crest displacement can point one endpoint tangent slightly
        # backward. Limit that tangent to the chord to prevent a loop.
        if np.dot(u,chord)/length<.1:u=chord/length
        if np.dot(v,chord)/length<.1:v=chord/length
        def curve(t):
            t=np.asarray(t)[:,None]
            return ((2*t**3-3*t**2+1)*p+(t**3-2*t**2+t)*length*u
                    +(-2*t**3+3*t**2)*q+(t**3-t**2)*length*v)
        dense=curve(np.linspace(0,1,max(8,int(np.ceil(length*4))+1)))
        h,w=image_shape[:2]
        if (np.any(dense<0) or np.any(dense[:,0]>w-1) or np.any(dense[:,1]>h-1)
                or not points_in_guides(dense,(guide,)).all()):continue
        missing=np.arange(a+1,b)
        points=curve((distance[missing]-distance[a])/(distance[b]-distance[a]))
        if not points_in_guides(points,(guide,)).all():continue
        connected.update({int(k):point.tolist() for k,point in zip(missing,points)})
        joins.append(dict(start_index=a,end_index=b,length_px=float(distance[b]-distance[a]),
                          points=[p.tolist(),*points.tolist(),q.tolist()],method='endpoint_tangent_cubic'))
    return connected,joins


def _connect_region_gaps(chosen, distance, centers, guide, image_shape, corner_turns=False):
    """Complete interior holes of ONE guide, with explicitly inferred geometry.

    Prefer observed endpoint tangents. For curved missing intervals, transport
    endpoint residuals along the guide instead of cutting across its corridor.
    Never extend unsupported ends or combine separate guides. If no in-image,
    in-corridor route exists, the caller must reject a disconnected result.
    """
    connected, joins = _connect_short_gaps(chosen, distance, guide, image_shape)
    indices = sorted(connected)
    h, w = image_shape[:2]
    for a, b in zip(indices, indices[1:]):
        if b == a+1:
            continue
        p, q = np.asarray(connected[a]), np.asarray(connected[b])
        span = distance[b]-distance[a]
        residual = np.array([p-centers[a], q-centers[b]])
        # An abrupt transverse switch is a competing branch, not a blank rim.
        if np.linalg.norm(residual[1]-residual[0]) > max(6., .75*span):
            # At a sharp TEM field corner the normal can rotate across a few
            # samples. A short join is allowed only at that actual guide turn;
            # a transverse jump along a straight wall is still rejected.
            u=centers[a]-centers[max(0,a-4)]
            v=centers[min(len(centers)-1,b+4)]-centers[b]
            turning=np.dot(u,v)<.85*np.linalg.norm(u)*np.linalg.norm(v)
            if not (corner_turns and turning and span<=12 and np.linalg.norm(q-p)<=12):
                continue
        ds = np.linspace(distance[a], distance[b], max(8, int(np.ceil(span*4))+1))
        base = np.column_stack([np.interp(ds, distance[a:b+1], centers[a:b+1, axis]) for axis in (0, 1)])
        t = ((ds-distance[a])/span)[:, None]
        blend = t*t*(3-2*t)
        # Smooth endpoint residual interpolation keeps a long missing floor or
        # entrance on the drawn route, not on a chord across the empty trench.
        dense = base + (1-blend)*residual[0] + blend*residual[1]
        if (np.any(dense < 0) or np.any(dense[:, 0] > w-1) or np.any(dense[:, 1] > h-1)
                or not points_in_guides(dense, (guide,)).all()):
            continue
        missing = np.arange(a+1, b)
        points = np.column_stack([np.interp(distance[missing], ds, dense[:, axis]) for axis in (0, 1)])
        if not points_in_guides(points, (guide,)).all():
            continue
        connected.update({int(i): point.tolist() for i, point in zip(missing, points)})
        joins.append(dict(start_index=a, end_index=b, length_px=float(span),
                          points=[p.tolist(), *points.tolist(), q.tolist()],
                          method='guide_corridor_residual_interpolation', source='inferred'))
    return connected, sorted(joins, key=lambda join: join['start_index'])


def trace_guide(image, guide, threshold, boundary_mode='inner_rim', local_contrast=False, material_level=None, rim_inset_px=1., continuous=False, boundary_sample=None, sample_inverted=False):
    if boundary_sample is not None:boundary_mode='sample_profile'
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
    context=64 if boundary_mode=='rim_inset' else 48 if boundary_mode=='tem_interface' else 24
    templates=prepare_templates(boundary_sample,sample_inverted) if boundary_sample is not None else None
    if templates is not None:context=max(context,len(templates[0][0])+2)
    offsets=np.arange(-math.ceil(radius)-context,math.ceil(radius)+context+1,dtype=float)
    xy=centers[:,None,:]+normals[:,None,:]*offsets[None,:,None]
    valid=(xy[:,:,0]>=0)&(xy[:,:,0]<image.shape[1]-1)&(xy[:,:,1]>=0)&(xy[:,:,1]<image.shape[0]-1)
    profiles=cv2.remap(image.astype(np.float32,copy=False),xy[:,:,0].astype(np.float32),
                       xy[:,:,1].astype(np.float32),cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
    raw_profiles=profiles
    if local_contrast:
        # Work in arc-length/normal coordinates. Tangential median pooling
        # suppresses isolated SEM grains and transverse dimension lines without
        # pooling the opposite wall or fitting either wall to a straight line.
        sample_step=float(distance[1]-distance[0])
        median_half=max(1,int(round(4/sample_step)))
        tangent_sigma=max(.5,1.5/sample_step)
        padded=np.pad(profiles,((median_half,median_half),(0,0)),mode='edge')
        windows=np.lib.stride_tricks.sliding_window_view(padded,2*median_half+1,axis=0)
        profiles=np.empty_like(raw_profiles)
        # Bound the temporary median workspace for long, wide guides.
        for start in range(0,len(profiles),512):
            profiles[start:start+512]=np.median(windows[start:start+512],axis=-1)
        profiles=cv2.GaussianBlur(profiles,(0,0),.8,sigmaY=tangent_sigma,borderType=cv2.BORDER_REPLICATE)
    if material_level is None:
        material=image[image>threshold+20]
        material_level=float(np.percentile(material,75)) if len(material) else threshold+20
    candidates=[]
    tem_sign=void_normal_sign(centers) if boundary_mode=='tem_interface' else None
    ks=np.arange(6,len(offsets)-6)
    far=(ks>=15)&(ks+15<len(offsets))
    for i,profile in enumerate(profiles):
        if templates is not None:
            matches=profile_candidates(profile,offsets,valid[i],radius,templates)
            # Tangential pooling cannot turn a blank cross-section into a
            # measured boundary. Such holes use the explicit join provenance.
            evidence=profile_candidates(raw_profiles[i],offsets,valid[i],radius,templates)
            candidates.append([c for c in matches if any(abs(c[0]-p[0])<=3 for p in evidence)])
            continue
        if boundary_mode=='tem_interface':
            candidates.append(interface_candidates(profile,offsets,valid[i],radius,tem_sign))
            continue
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
            if local_contrast:
                # Roof edges can have similar gray levels on both sides.
                # Their crest prominence is evidence even when step contrast
                # cancels; this is common on shaded SEM sidewalls.
                supported|=(invalid[ks+7]==invalid[ks-6])&(b-np.minimum(before,after)>=8)
            # A broad material-layer plateau is not a resolved bright rim.
            # Require a local crest before suppressing its gradient flanks.
            peaks=supported&(b>=np.maximum(a,c))&(b-(a+c)/2>=(.05 if local_contrast else .5))&(prominence>=5)
            if boundary_mode=='rim_inset':
                # A broad white roof has little curvature/prominence at +/-6
                # pixels. Require a true two-sided crest over wider context,
                # without admitting a monotonic shading ramp as an edge.
                lo=np.clip(ks-20,0,len(profile));li=np.clip(ks-10,0,len(profile))
                ri=np.clip(ks+11,0,len(profile));hi=np.clip(ks+21,0,len(profile))
                wide_before=(prefix[li]-prefix[lo])/np.maximum(1,li-lo)
                wide_after=(prefix[hi]-prefix[ri])/np.maximum(1,hi-ri)
                prominence=b-np.maximum(wide_before,wide_after)
                supported=(ks>=20)&(ks+21<=len(profile))&(invalid[hi]==invalid[lo])
                peaks=supported&(b>=np.maximum(a,c))&(b-(a+c)/2>=.005)&(prominence>=3)
            ridge_indices=ks[peaks].tolist()
            for j in np.flatnonzero(peaks):
                k=ks[j];denom=float(a[j]-2*b[j]+c[j])
                sub=float(np.clip(.5*(float(a[j])-float(c[j]))/denom,-.5,.5)) if denom<-1e-6 else 0.
                offset=offsets[k]+sub
                if abs(offset)<=radius:
                    found.append((offset,float(1.25*prominence[j]+.65*contrast[j]),'ridge',k))
            gradient=np.abs(c-a)/2
            # First differences contain the real broad edge slope. Estimate
            # residual grain from second differences instead, so a smooth but
            # low-gradient flank does not disappear in favor of its crest.
            gradient_floor=max(1.,min(4.,2*float(np.median(np.abs(np.diff(profile,n=2)))))) if local_contrast else 8
            steep=supported&(np.abs(offsets[ks])<=radius)&(gradient>=gradient_floor)
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
        dark_level=threshold+8
        if local_contrast and valid[i].any():
            local=profile[valid[i]]
            # A shaded SEM cavity may stay much brighter than the global void
            # cutoff near one wall. Use the darker local side of a resolved rim,
            # while retaining ridge barriers and material-background rejection.
            dark_level=max(dark_level,min(material_level-8,float(np.percentile(local,10)+.5*np.ptp(local))))
        dark=np.flatnonzero((np.convolve(profile,np.ones(3)/3,'same')<=dark_level)&valid[i])
        barriers=ridge_indices[:]
        # A bright rim must rise above the material background as well as
        # the void. A finite bright layer plateau is otherwise mistaken for
        # a crest halfway inside the solid. Other local peaks still block
        # borrowing a cavity through an intervening real boundary.
        resolved=[k for k in ridge_indices if profile[k]>=material_level+5 or
                  (local_contrast and profile[k]-max(float(np.mean(profile[k-6:k-2])),float(np.mean(profile[k+3:k+7])))>=12)]
        if boundary_mode=='rim_inset':resolved=ridge_indices[:]
        if local_contrast:barriers=resolved
        ridge_indices=resolved
        found=[candidate for candidate in found if candidate[2]!='ridge' or candidate[3] in resolved]
        if boundary_mode=='rim_inset':
            # A broad gradient flank is never a substitute for a bright crest.
            found=[candidate for candidate in found if candidate[2]=='ridge']
        ridges=[offsets[j] for j in ridge_indices]
        for offset,strength,kind,k in sorted(found,key=lambda p:p[1],reverse=True):
            # Accept the first resolved boundary connected to nearby cavity
            # pixels. A farther/brighter material line behind another rim is
            # not the trench, even when it falls inside a wide guide corridor.
            nearest=[]
            before=dark[dark<k];after=dark[dark>k]
            if len(before):nearest.append(int(before[-1]))
            if len(after):nearest.append(int(after[0]))
            if local_contrast:
                # On a soft SEM edge the first dark-threshold crossing is
                # necessarily close in intensity. Look farther along the same
                # unobstructed normal for contrast, not only at that crossing.
                connection_range=64 if boundary_mode=='rim_inset' else 24
                nearest=dark[(np.abs(dark-k)<=connection_range)&(profile[k]-profile[dark]>=12)].tolist()
            else:connection_range=24
            connected=any(abs(seed-k)<=connection_range and profile[k]-profile[seed]>=12
                          and not any(min(seed,k)<j<max(seed,k) for j in barriers)
                          for seed in nearest)
            if not connected:continue
            # A resolved SEM rim measures at its bright crest, rather than
            # either steep flank. Suppress only nearby flanks, not all other
            # gradients elsewhere in the search corridor.
            if boundary_mode=='inner_rim' and kind=='gradient' and any(abs(offset-ridge)<4 for ridge in ridges):continue
            if all(abs(offset-old[0])>=1.5 for old in ranked):ranked.append((offset,strength))
            if len(ranked)==6:break
        if ranked and boundary_mode=='rim_inset':
            levels=np.interp([c[0] for c in ranked],offsets,profile)
            low=float(np.min(profile));span=max(1.,float(np.max(profile))-low)
            ranked=[(offset,96*(levels[j]-low)/span+.1*strength)
                    for j,(offset,strength) in enumerate(ranked)]
        elif ranked and boundary_mode!='inner_rim':
            positions=np.array([c[0] for c in ranked])
            levels=np.interp(positions,offsets,profile)
            slopes=np.interp(positions,offsets,np.abs(np.gradient(profile)))
            selected=_boundary_profile_index(positions,levels,slopes,boundary_mode)
            if local_contrast and boundary_mode=='max_gradient':
                # Score the requested edge criterion continuously. A binary
                # winner bonus and unrelated roof prominence can otherwise
                # make a weak shading ripple beat the actual gradient flank.
                ranked=[(offset,64*slopes[index]/max(1.,float(slopes.max()))+.15*strength)
                        for index,(offset,strength) in enumerate(ranked)]
            else:
                ranked=[(offset,strength+((12 if local_contrast else 96) if index==selected else 0)) for index,(offset,strength) in enumerate(ranked)]
        if local_contrast:
            # Neighborhood support cannot manufacture an edge through a blank
            # image strip. Retain independent contrast evidence on this normal.
            evidence_radius=20 if boundary_mode=='rim_inset' else 6
            ranked=[(offset,strength) for offset,strength in ranked
                    if np.ptp(raw_profiles[i,max(0,int(round(offset-offsets[0]))-evidence_radius):
                                               min(len(offsets),int(round(offset-offsets[0]))+evidence_radius+1)])>=8]
        candidates.append(ranked)
    # Viterbi along the ordered guide. Keep multiple candidates until the
    # complete local path is evaluated; a strong isolated peak cannot win a
    # row independently. Interior holes are connected afterwards according to
    # the requested bounded or continuous-region policy.
    chosen={}
    blocks=[]
    supported=[i for i,c in enumerate(candidates) if c]
    for i in supported:
        if not blocks or i-blocks[-1][-1]>4:blocks.append([])
        blocks[-1].append(i)
    for indices in blocks:
        if local_contrast:
            chosen.update(_consensus_path(indices,candidates,distance,centers,normals,radius))
            continue
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
        # Report the original image's edge spread, not the pooled profile's
        # apparent blur or an inferred dimensional confidence interval.
        profile=raw_profiles[i];derivative=np.abs(np.gradient(profile))
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
    measured=dict(chosen)
    peak_points={};inward_normals={}
    spread_by_index=dict(zip(sorted(measured),spreads))
    if boundary_mode=='rim_inset' and templates is None:
        chosen,peak_points,inward_normals=inset_rim(chosen,centers,normals,profiles,offsets,guide,image.shape,rim_inset_px)
        measured=dict(chosen)
    joins=[]
    if continuous:
        chosen,joins=_connect_region_gaps(chosen,distance,centers,guide,image.shape,corner_turns=boundary_mode=='tem_interface' or templates is not None)
    elif local_contrast:chosen,joins=_connect_short_gaps(chosen,distance,guide,image.shape)
    spreads=[spread_by_index.get(i) if i in measured else None for i in sorted(chosen)]
    for i,point in sorted(chosen.items()):
        if not segments or i!=segments[-1][-1][0]+1:segments.append([])
        segments[-1].append((i,point))
    coverage=len(measured)/len(centers)
    return dict(side='contour',requested_points=len(centers),detected_points=len(measured),
                method='sampled_profile_landmark' if templates is not None else
                       'tem_gray_material_interface' if boundary_mode=='tem_interface' else
                       'bright_crest_inward_offset' if boundary_mode=='rim_inset' else
                       'tangent_consensus_contour' if local_contrast else 'ordered_normal_contour',
                rim_inset_px=rim_inset_px if boundary_mode=='rim_inset' else None,
                rim_peak_points=[peak_points.get(i) for i in sorted(chosen)] if boundary_mode=='rim_inset' else [],
                inward_normals=[inward_normals.get(i) for i in sorted(chosen)] if boundary_mode=='rim_inset' else [],
                preprocessing=(dict(tangent_median_samples=2*median_half+1,tangent_sample_step_px=sample_step,
                                    tangent_gaussian_sigma_px=tangent_sigma*sample_step,normal_gaussian_sigma_px=.8,
                                    missing_policy='continuous_region_interpolation' if continuous else 'bounded_short_gap_connection',
                                    max_join_length_px=None if continuous else 12,
                                    measurement_scope='guide_corridor') if local_contrast else None),
                trace_coverage=coverage,status='valid' if coverage>=.95 else 'partial',
                connected_coverage=len(chosen)/len(centers),interpolated_points=len(chosen)-len(measured),
                continuity_requested=continuous,continuity_complete=bool(segments) and len(segments)==1,
                joins=joins,point_sources=['measured' if i in measured else 'interpolated' for i in sorted(chosen)],
                reason=('검출 경계 사이의 누락을 인식 범위 안에서 보간해 연결했습니다.' if joins else '곡선 방향의 연결된 명암 경계')
                       if len(segments)<=1 else '연결할 근거가 부족한 구간이 남아 있습니다.',
                segments=[[point for _,point in segment] for segment in segments],
                points=[point for _,point in sorted(chosen.items())],edge_spreads_px=spreads)


def contour_row_targets(traces):
    rows={}
    for trace in traces:
        points=np.asarray(trace.get('points',[p for s in trace['segments'] for p in s]),float).reshape(-1,2)
        low,high=(float(points[:,1].min()),float(points[:,1].max())) if len(points) else (0.,0.)
        for segment in trace['segments']:
            # A subpixel displacement at an open trace endpoint must not
            # create a false missing integer row (e.g. y=30.005 vs row 30).
            for point in (segment[:1]+segment[-1:]):
                y=round(point[1])
                if abs(point[1]-y)<=.25:
                    values=rows.setdefault(y,[])
                    if not any(abs(point[0]-old)<.5 for old in values):values.append(float(point[0]))
            for index,(a,b) in enumerate(zip(segment,segment[1:])):
                # Horizontal Field supports and floor tangents are not wall
                # targets. They remain in the ordered trace for surface fits.
                floor_transition=high-low>50 and min(a[1],b[1])>=high-max(12,(high-low)*.15)
                if not floor_transition and abs(b[1]-a[1])<.25*max(1e-8,np.linalg.norm(np.subtract(b,a))):
                    # One noisy shallow step on a vertical wall is still part
                    # of that wall. Classify it using its local contour span.
                    span=np.subtract(segment[min(len(segment)-1,index+4)],segment[max(0,index-3)])
                    if abs(span[1])<.25*max(1e-8,np.linalg.norm(span)):continue
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
        active=[i for i in active if row_index-nodes[i]['row']<=4]
        if strict and not search:continue
        if strict and len(values)>=2:
            ordered=np.sort(values);split=int(np.argmax(np.diff(ordered)))+1
            branches=(ordered[:split],ordered[split:])
            a,b=(float(np.median(branch)) for branch in branches)
            # Classify an intensity gradient across one cavity from the actual
            # paired trace, rather than moving its bright wall to a global dark
            # threshold crossing. The central span must stay darker than BOTH
            # external material supports; this excludes a pillar between two
            # different trenches. Missing trace branches cannot use this path.
            ia,ib=int(round(a-x)),int(round(b-x))
            trim=max(2,int((ib-ia)*.15))
            left_context=ridge_image[row_index,max(0,ia-20):max(0,ia-6)]
            right_context=ridge_image[row_index,min(width,ib+6):min(width,ib+20)]
            interior=ridge_image[row_index,max(0,ia+trim):min(width,ib-trim)]
            compact=all(np.ptp(branch)<=max(6,2*max(g['half_width_px'] for g in guides)) for branch in branches)
            if compact and b-a>8 and len(left_context)>=5 and len(right_context)>=5 and len(interior)>=5:
                material=min(float(np.median(left_context)),float(np.median(right_context)))
                contrast_ok=np.percentile(interior,90)<material-8
                continued=any(nodes[i].get('local_cavity',False)
                              and abs(nodes[i]['left']-a)<=max(4,2*(row_index-nodes[i]['row']))
                              and abs(nodes[i]['right']-b)<=max(4,2*(row_index-nodes[i]['row'])) for i in active)
                if (contrast_ok or continued) and in_bands(a,search) and in_bands(b,search):
                    seeds=[*seeds,(a-x,b-x)]
        else:
            a=b=None
        body=any(lo<=yy<=hi for lo,hi in spans)
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
                              support=support+(nodes[parent]['support'] if parent is not None else 0),
                              local_cavity=bool(strict and a is not None and abs(left-a)<6 and abs(right-b)<6)))
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
