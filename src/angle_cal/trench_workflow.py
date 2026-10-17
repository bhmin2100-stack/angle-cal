"""Guide-first workflow with automatic image bounds and no user ROI."""
from dataclasses import replace
import math
import numpy as np
from .trench_analysis import analyze_trench
from .trench_guides import normalize_guides


def recognize_from_guides(image,options):
    guides=normalize_guides(options.wall_guides,image.shape)
    if not guides:raise ValueError('먼저 곡선을 그리거나 구조를 불러온 뒤 인식을 누르세요.')
    h,w=image.shape[:2];candidates=[]
    # Try both contrast directions internally. The user need not know whether
    # the specimen/void is bright. Never substitute drawn coordinates.
    for bright in (False,True):
        try:
            result=analyze_trench(image,replace(options,roi=(0,0,w,h),step_px=max(options.step_px,h/10000),bright_trench=bright,wall_guides=guides))
            support=sum(t['detected_points'] for t in result.guide_traces)
            candidates.append((result.depth_px is not None,support,len(result.y),bright,result))
        except ValueError:continue
    if not candidates:raise ValueError('곡선 주변에서 연결된 Trench 경계를 찾지 못했습니다. 곡선을 옮기거나 휠로 인식 폭을 조절하세요.')
    _,_,_,bright,preliminary=max(candidates,key=lambda c:c[:3])
    # The topmost supported cavity row becomes the entrance depth reference.
    # Keep complete image width, including opposite wall and optional Field.
    top=max(0,int(math.floor(preliminary.y[0])))
    return analyze_trench(image,replace(options,roi=(0,top,w,h-top),bright_trench=bright,wall_guides=guides))
