"""Guide-first workflow with automatic image bounds and no user ROI."""
from dataclasses import replace
import math
import numpy as np
from .trench_analysis import analyze_trench
from .trench_guides import normalize_guides, sweep_samples


def recognize_from_guides(image,options):
    options=replace(options,strict_guides=True,continuous_guides=True)
    guides=normalize_guides(options.wall_guides,image.shape)
    if not guides:raise ValueError('먼저 곡선을 그리거나 구조를 불러온 뒤 인식을 누르세요.')
    h,w=image.shape[:2];candidates=[]
    # SEM footers and dimension labels outside the requested contour must not
    # set its intensity range. This context box is automatic; final candidates
    # still have to belong to the exact swept guide corridor.
    lower=[];upper=[];guided_top=[]
    for guide in guides:
        samples=sweep_samples(guide);margin=guide['half_width_px']+32
        lower.append(samples.min(axis=0)-margin);upper.append(samples.max(axis=0)+margin)
        guided_top.append(samples[:,1].min()-guide['half_width_px'])
    x0,y0=np.maximum(0,np.floor(np.min(lower,axis=0)).astype(int))
    y0=max(y0,int(math.floor(min(guided_top))))
    x1,y1=np.minimum([w,h],np.ceil(np.max(upper,axis=0)).astype(int)+1)
    context=(int(x0),int(y0),int(x1-x0),int(y1-y0))
    # Try both contrast directions internally. The user need not know whether
    # the specimen/void is bright. Observations require image evidence;
    # explicitly labeled interior joins may follow the selected guide path.
    for bright in (False,True):
        try:
            # Smoothing must not change polarity selection or move the entrance
            # reference plane. Establish both from the same uncorrected evidence.
            result=analyze_trench(image,replace(options,roi=context,step_px=max(options.step_px,context[3]/10000),
                                              bright_trench=bright,wall_guides=guides,contour_strength=0))
            support=sum(t['detected_points'] for t in result.guide_traces)
            candidates.append((result.depth_px is not None,support,len(result.y),bright,result))
        except ValueError:continue
    if not candidates:
        if options.boundary_mode=='rim_inset':
            raise ValueError('곡선 범위에서 양쪽의 밝은 테두리를 찾지 못했습니다. 범위·이동량을 확인하세요. 밝은 띠가 없는 영상은 「밝은 테두리 기준」을 끄고 기존 방식으로 인식할 수 있습니다.')
        raise ValueError('곡선 주변에서 연결된 Trench 경계를 찾지 못했습니다. 곡선을 옮기거나 휠로 인식 폭을 조절하세요.')
    _,_,_,bright,preliminary=max(candidates,key=lambda c:c[:3])
    # The topmost supported cavity row becomes the entrance depth reference.
    # Keep the complete requested contour and its local intensity context.
    top=max(0,int(math.floor(preliminary.y[0])))
    fields=preliminary.field_angles
    if len(fields)==2 and all(item.get('status')=='valid' and item.get('source')=='guide' for item in fields):
        levels=[float(np.polyval(item['line'],edge[0])) for item,edge in zip(fields,(preliminary.left,preliminary.right))]
        if max(levels)-min(levels)<=options.field_search_px and abs(np.mean(levels)-top)<=options.field_search_px:
            top=max(0,int(math.floor(np.mean(levels))))
    return analyze_trench(image,replace(options,roi=(context[0],top,context[2],int(y1-top)),bright_trench=bright,wall_guides=guides),
                          field_reference=(preliminary.field_angles,preliminary.corners))
