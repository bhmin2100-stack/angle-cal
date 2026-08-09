from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
import cv2
import numpy as np

class StitchingError(RuntimeError): pass
class StitchingCancelled(StitchingError): pass
class StitchingNeedsManual(StitchingError):
    def __init__(self, message, suggested_pair=None): super().__init__(message); self.suggested_pair=suggested_pair

@dataclass(frozen=True)
class StitchOptions: overlay_crop_fraction: float=0.; min_matches: int=8; ratio_test: float=.75; ransac_threshold: float=3.
@dataclass
class StitchPlacement: path: str; transform: np.ndarray; mode: str; inlier_count: int=0; reprojection_error: float=0.
@dataclass
class StitchResult: image: np.ndarray; valid_mask: np.ndarray; placements: list[StitchPlacement]; output_size: tuple[int,int]

def read_raw_image(path):
    data=np.fromfile(str(path),np.uint8); image=cv2.imdecode(data,cv2.IMREAD_UNCHANGED)
    if image is None: raise StitchingError(f"이미지를 읽을 수 없습니다: {path}")
    return image

def _gray8(image):
    gray=cv2.cvtColor(image[...,:3],cv2.COLOR_BGR2GRAY) if image.ndim==3 else image
    if gray.dtype==np.uint8:return gray
    lo,hi=np.percentile(gray,(1,99)); return np.zeros(gray.shape,np.uint8) if hi<=lo else np.clip((gray-lo)*255/(hi-lo),0,255).astype(np.uint8)

def detect_bottom_overlay_fraction(images):
    found=[]
    for image in images:
        gray=_gray8(image); h=gray.shape[0]; std=gray.std(1)
        for y in range(h-2,int(h*.55),-1):
            if std[y:].mean()<max(4.,std[max(0,y-h//15):y].mean()*.4): found.append((h-y)/h); break
    return float(np.median(found)) if found else 0.

def alignment_from_manual_points(source,target):
    if len(source)!=len(target) or len(source)<4: raise StitchingError("수동 기준점은 양쪽에 같은 개수로 4개 이상 필요합니다.")
    matrix,_=cv2.findHomography(np.float32(source),np.float32(target),0)
    if matrix is None: raise StitchingError("기준점 변환을 계산하지 못했습니다.")
    return matrix

def stitch_paths(paths,options=StitchOptions(),*,manual_links=None,progress=None,cancelled=None):
    if not 2<=len(paths)<=20: raise StitchingError("이미지는 2~20장을 선택할 수 있습니다.")
    images=[read_raw_image(p) for p in paths]
    formats={(x.dtype.str,x.ndim,x.shape[2] if x.ndim==3 else 1) for x in images}
    if len(formats)!=1: raise StitchingError("픽셀 보존을 위해 비트 깊이와 채널 수가 같아야 합니다.")
    sift=cv2.SIFT_create(nfeatures=8000); features=[]
    for i,image in enumerate(images):
        if cancelled and cancelled(): raise StitchingCancelled()
        gray=_gray8(image)[:max(1,int(image.shape[0]*(1-options.overlay_crop_fraction)))]
        features.append(sift.detectAndCompute(gray,None))
        if progress: progress("features",i,len(images),f"특징점 추출 {i+1}/{len(images)}")
    edges=[]; manual_links=manual_links or {}
    for i in range(len(images)):
      for j in range(i+1,len(images)):
        if (i,j) in manual_links: matrix=alignment_from_manual_points(*manual_links[(i,j)]); edges.append((99,i,j,matrix,0.,"manual Lanczos")); continue
        ka,da=features[i]; kb,db=features[j]
        if da is None or db is None: continue
        good=[a for a,b in cv2.BFMatcher().knnMatch(da,db,k=2) if a.distance<options.ratio_test*b.distance]
        if len(good)<options.min_matches: continue
        src=np.float32([ka[m.queryIdx].pt for m in good]); dst=np.float32([kb[m.trainIdx].pt for m in good])
        matrix,mask=cv2.findHomography(src,dst,cv2.RANSAC,options.ransac_threshold)
        if matrix is None or mask is None or mask.sum()<options.min_matches: continue
        keep=mask.ravel().astype(bool); projected=cv2.perspectiveTransform(src[keep,None],matrix)[:,0]
        error=float(np.linalg.norm(projected-dst[keep],axis=1).mean()); mode="Lanczos perspective"
        if np.allclose(matrix[:2,:2]/matrix[2,2],np.eye(2),atol=.005) and np.allclose(matrix[2,:2],0,atol=1e-5):
            delta=np.median(dst[keep]-src[keep],0); matrix=np.array([[1.,0.,round(float(delta[0]))],[0.,1.,round(float(delta[1]))],[0.,0.,1.]]); mode="exact translation"
        edges.append((int(mask.sum()),i,j,matrix,error,mode))
    transforms={0:np.eye(3)}; meta={0:(0,0.,"anchor")}
    while len(transforms)<len(images):
        candidates=[]
        for count,i,j,matrix,error,mode in edges:
            if i in transforms and j not in transforms:candidates.append((count,j,transforms[i]@np.linalg.inv(matrix),error,mode))
            elif j in transforms and i not in transforms:candidates.append((count,i,transforms[j]@matrix,error,mode))
        if not candidates:
            missing=next(i for i in range(len(images)) if i not in transforms); raise StitchingNeedsManual("자동 정렬 실패: 공통 기준점 4개 이상이 필요합니다.",(0,missing))
        count,node,matrix,error,mode=max(candidates,key=lambda x:x[0]); transforms[node]=matrix; meta[node]=(count,error,mode)
    corners=[]
    for i,image in enumerate(images):
        h,w=image.shape[:2]; corners.append(cv2.perspectiveTransform(np.float32([[[0,0],[w,0],[w,h],[0,h]]]),transforms[i])[0])
    allc=np.concatenate(corners); low=np.floor(allc.min(0)).astype(int); high=np.ceil(allc.max(0)).astype(int); width,height=(high-low).tolist()
    if width*height>1_200_000_000: raise StitchingError("결과 캔버스가 너무 큽니다.")
    shift=np.array([[1.,0.,-low[0]],[0.,1.,-low[1]],[0.,0.,1.]])
    output=np.zeros((height,width)+images[0].shape[2:],images[0].dtype); valid=np.zeros((height,width),np.uint8); placements=[]
    for i,(path,image) in enumerate(zip(paths,images)):
        matrix=shift@transforms[i]; exact=np.allclose(matrix[:2,:2],np.eye(2)) and np.allclose(matrix[2],[0,0,1]) and np.allclose(matrix[:2,2],np.round(matrix[:2,2]))
        if exact:
            x,y=np.round(matrix[:2,2]).astype(int); warped=np.zeros_like(output); mask=np.zeros_like(valid); warped[y:y+image.shape[0],x:x+image.shape[1]]=image; mask[y:y+image.shape[0],x:x+image.shape[1]]=255
        else:
            warped=cv2.warpPerspective(image,matrix,(width,height),flags=cv2.INTER_LANCZOS4); mask=cv2.warpPerspective(np.full(image.shape[:2],255,np.uint8),matrix,(width,height),flags=cv2.INTER_NEAREST)
        take=(mask>0)&(valid==0); output[take]=warped[take]; valid[mask>0]=255
        count,error,mode=meta[i]; placements.append(StitchPlacement(path,matrix,mode,count,error))
        if progress:progress("compose",i,len(images),f"원본 픽셀 배치 {i+1}/{len(images)}")
    return StitchResult(output,valid,placements,(width,height))

def save_stitch_result(path,result):
    output=Path(path); output=output if output.suffix.lower() in ('.tif','.tiff','.png') else output.with_suffix('.tif')
    image=result.image
    if image.ndim==2: encoded=np.dstack((image,image,image,result.valid_mask.astype(image.dtype) * (np.iinfo(image.dtype).max // 255)))
    elif image.shape[2]==3: encoded=np.dstack((image,result.valid_mask))
    else: encoded=image.copy(); encoded[...,3]=result.valid_mask.astype(image.dtype) * (np.iinfo(image.dtype).max // 255)
    ok,data=cv2.imencode(output.suffix,encoded)
    if not ok:raise StitchingError("결과를 저장하지 못했습니다.")
    data.tofile(str(output)); mask=output.with_name(output.stem+'.mask.png'); cv2.imencode('.png',result.valid_mask)[1].tofile(str(mask))
    report=output.with_suffix('.stitch.json'); report.write_text(json.dumps({'version':1,'scale_status':'recalibration_required','output_size':result.output_size,'sources':[{'path':p.path,'mode':p.mode,'inliers':p.inlier_count,'error':p.reprojection_error,'transform':p.transform.tolist()} for p in result.placements]},ensure_ascii=False,indent=2),encoding='utf-8')
    return output,mask,report
