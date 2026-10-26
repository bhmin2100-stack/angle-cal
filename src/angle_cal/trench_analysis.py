"""Pixel-coordinate trench measurements. No inferred physical calibration.

An ROI contains one trench, with its upper edge on the entrance reference plane.
The image must already be aligned so depth increases downwards. Thresholds are
relative to the ROI's 1st--99th intensity percentiles, including for 16-bit input.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np

from .image_ops import to_gray
from .trench_corner import measure_entrance_corner
from .trench_surface import measure_field_angles, connect_entrance
from .trench_guides import normalize_guides, prepare_guide_bands, in_bands, sweep_samples, points_in_guides
from .trench_contour import trace_guide, guided_cavity_profile
from .trench_guided_surface import measure_guided_surfaces
from .trench_smoothing import smooth_traces, smoothing_parameters, METHODS
from .trench_field_metrics import measure_field_metrics


@dataclass(frozen=True)
class TrenchOptions:
    roi: tuple[int, int, int, int]
    step_px: float = 20.0
    threshold: float = 0.0  # zero: automatic; otherwise normalized 1..254
    bright_trench: bool = False
    smooth_px: int = 3
    bow_reference: str = "entrance"
    corner_window_px: int = 24
    negative_slope_deg: float = 1.0
    corner_rois: tuple = (None, None)
    corner_spans: tuple = (None, None)
    field_search_px: int = 32
    angle_span: tuple = (40, 100)
    wall_guides: tuple = ()
    boundary_mode: str = 'inner_rim'
    strict_guides: bool = False
    rim_inset_px: float = 1.0
    contour_smoothing: str = 'robust'
    contour_strength: float = 0.0
    continuous_guides: bool = False
    imaging_mode: str = 'vsem'


@dataclass
class TrenchResult:
    options: TrenchOptions
    y: np.ndarray
    left: np.ndarray
    right: np.ndarray
    sample_y: np.ndarray
    sample_left: np.ndarray
    sample_right: np.ndarray
    left_reference: np.ndarray
    right_reference: np.ndarray
    depth_px: float | None
    observed_depth_px: float
    left_bowing_px: float
    right_bowing_px: float
    left_bowing_y: float
    right_bowing_y: float
    left_angle_deg: float | None
    right_angle_deg: float | None
    corners: list[dict | None]
    threshold: float
    warnings: list[str]
    local_bowing: list[dict]
    field_angles: list[dict] = field(default_factory=list)
    guide_diagnostics: list[dict] = field(default_factory=list)
    guide_traces: list[dict] = field(default_factory=list)
    field_measurements: dict = field(default_factory=dict)
    imaging: dict = field(default_factory=dict)

    @property
    def depth_origin_px(self):
        value = self.field_measurements.get('reference_y_px')
        return self.options.roi[1] if value is None else value

    def summary(self) -> dict:
        return {
            "depth_px": self.depth_px,
            "observed_depth_px": self.observed_depth_px,
            "left_bowing_px": self.left_bowing_px,
            "right_bowing_px": self.right_bowing_px,
            "left_bowing_depth_px": self.left_bowing_y - self.options.roi[1],
            "right_bowing_depth_px": self.right_bowing_y - self.options.roi[1],
            "left_angle_deg": self.left_angle_deg,
            "right_angle_deg": self.right_angle_deg,
            "left_radius_px": self.corners[0]["radius_px"] if self.corners[0] else None,
            "right_radius_px": self.corners[1]["radius_px"] if self.corners[1] else None,
            "min_cd_px": float(np.min(self.right - self.left)),
            "max_cd_px": float(np.max(self.right - self.left)),
            "threshold": self.threshold,
            "tcd_px": self.field_measurements.get('tcd_px'),
            "field_depth_px": self.field_measurements.get('depth_px'),
            "field_reference_y_px": self.field_measurements.get('reference_y_px'),
            **{f"{item['side']}_field_{key}":item.get(key) for item in self.field_angles
               for key in ("length_px","tilt_deg","rms_px")},
        }

    def to_dict(self, nm_per_px: float | None = None) -> dict:
        if nm_per_px is not None and (not math.isfinite(nm_per_px) or nm_per_px <= 0):
            raise ValueError("nm/px 보정값은 양수여야 합니다.")
        exchange=None;exchange_error=None
        if nm_per_px is not None:
            try:exchange=trench_profile_exchange(self,nm_per_px)
            except ValueError as exc:exchange_error=str(exc)
        return {
            "format": "anglecal.trench.v1", "options": asdict(self.options),
            "imaging": self.imaging,
            "wall_boundary": "tem_gray_material_interface" if self.options.boundary_mode=='tem_interface' else "bright_crest_plus_inward_normal_offset" if self.options.boundary_mode=='rim_inset' else "connected_inner_rim_peak_or_gradient",
            "nm_per_px": nm_per_px, "summary": self.summary(),
            "warnings": self.warnings, "corners": self.corners,
            "corner_length_units": {unit: [
                {key.replace("_px", "_" + unit): c.get(key) * factor if c.get(key) is not None else None
                 for key in ("radius_px", "rms_px", "edge_spread_px")} for c in self.corners]
                for unit, factor in (("nm", nm_per_px), ("angstrom", (nm_per_px or 0) * 10)) if factor},
            "local_bowing": self.local_bowing,
            "field_angles": self.field_angles,
            "guide_diagnostics": self.guide_diagnostics,
            "guide_traces": self.guide_traces,
            "field_measurements": self.field_measurements,
            "profile_exchange":exchange,"profile_exchange_error":exchange_error,
            "field_measurement_units": {unit: {key.removesuffix('_px')+'_'+unit: self.field_measurements.get(key)*factor
                if self.field_measurements.get(key) is not None else None for key in ('tcd_px','depth_px','left_depth_px','right_depth_px')}
                for unit,factor in (("um",(nm_per_px or 0)*.001),("nm",nm_per_px),("angstrom",(nm_per_px or 0)*10)) if factor},
            "guide_method": "ordered_normal_contour_then_cavity_classification" if self.guide_traces else "automatic",
            "field_length_units":{unit:[{key.replace("_px","_"+unit):item.get(key)*factor if item.get(key) is not None else None
                                        for key in ("length_px","rms_px","wall_rms_px")} for item in self.field_angles]
                                  for unit,factor in (("nm",nm_per_px),("angstrom",(nm_per_px or 0)*10)) if factor},
            "profile": {key: getattr(self, key).tolist() for key in
                        ("y", "left", "right", "sample_y", "sample_left", "sample_right",
                         "left_reference", "right_reference")},
        }


def _legacy_gfe_tsv(result: TrenchResult, nm_per_px: float) -> str:
    """GFE surface path: right Field, right wall, floor, left wall, left Field.

    GFE expects two numeric columns in angstroms, with depth negative. Field is
    measured from the image; short gaps to the wall/floor are straight joins,
    not additional measured contour points. Keep the full measured wall profile.
    """
    if not math.isfinite(nm_per_px) or nm_per_px <= 0:
        raise ValueError("GFE는 Å 좌표를 사용합니다. 먼저 nm/px 스케일을 보정하세요.")
    y, left, right = result.y, result.left, result.right
    if (len(y) < 2 or len(left) != len(y) or len(right) != len(y)
            or not np.isfinite(np.r_[y, left, right]).all()
            or np.any(left >= right) or np.any(np.diff(y) <= 0)):
        raise ValueError("좌우 윤곽 좌표가 유효하지 않습니다. 다시 분석하세요.")
    if np.any(np.diff(y)>1.5):
        raise ValueError("윤곽에 미검출 구간이 있어 GFE 형상을 연결할 수 없습니다. 가이드 위치·폭과 끊긴 경계를 확인하세요.")
    x0, top, width, _ = result.options.roi
    floor = float(y[-1]) if result.depth_px is None else top + result.depth_px
    if not math.isfinite(floor) or y[0] < top or y[-1] > floor or floor <= top:
        raise ValueError("입구·바닥 기준과 윤곽 좌표가 일치하지 않습니다.")
    if x0 >= float(np.min(left)) or x0 + width <= float(np.max(right)):
        raise ValueError("Field가 양쪽 측벽 바깥에 오도록 분석 영역을 넓히세요.")
    if len(result.field_angles)!=2 or any(item["status"]!="valid" for item in result.field_angles):
        if result.options.strict_guides:
            raise ValueError('좌우 Field가 미확정입니다. 상단 Field까지 곡선 인식 범위에 포함해 다시 인식하세요.')
        raise ValueError("좌우 Field가 미확정입니다. 표면 위쪽과 양쪽 바깥 재료를 포함하거나 Field 검색 범위를 조정하세요.")
    lf,rf=result.field_angles
    center = (float(left[0]) + float(right[0])) / 2
    reference = (np.polyval(lf["line"],left[0])+np.polyval(rf["line"],right[0]))/2
    rc,lc=rf.get('connection',{}),lf.get('connection',{})
    right_connected=rc.get('status')=='valid';left_connected=lc.get('status')=='valid'
    points = [tuple(point) for point in rf["points"][::-1]
              if not right_connected or point[0]>=rc['field_tangent'][0]]
    if right_connected:
        points.extend(map(tuple,rc.get('field_join_points',[])))
        points.extend(map(tuple,rc['arc_points']))
    elif result.corners[1]:
        points.extend(sorted((tuple(point) for point in result.corners[1].get("trace_points",[]) if point[1]<y[0]),key=lambda p:p[1]))
    points.extend((xx,yy) for xx,yy in zip(right,y) if not right_connected or yy>rc['wall_tangent'][1])
    points.extend([(float(right[-1]), floor), (float(left[-1]), floor)])
    points.extend((xx,yy) for xx,yy in zip(left[::-1],y[::-1]) if not left_connected or yy>lc['wall_tangent'][1])
    if left_connected:
        points.extend(map(tuple,lc['arc_points'][::-1]))
        points.extend(map(tuple,lc.get('field_join_points',[])[::-1]))
    elif result.corners[0]:
        points.extend(sorted((tuple(point) for point in result.corners[0].get("trace_points",[]) if point[1]<y[0]),key=lambda p:p[1],reverse=True))
    points.extend(tuple(point) for point in lf["points"][::-1]
                  if not left_connected or point[0]<=lc['field_tangent'][0])
    if result.options.strict_guides and not points_in_guides(points,result.options.wall_guides).all():
        raise ValueError('GFE 연결 형상이 곡선 인식 범위를 벗어납니다. Field·입구·바닥을 포함하도록 곡선을 보정하세요.')
    factor = nm_per_px * 10
    lines = ["X\tY"]
    previous = None
    for x, yy in reversed(points):
        line = f"{(x - center) * factor:.9f}\t{(reference - yy) * factor:.9f}"
        if line != previous:
            lines.append(line)
            previous = line
    return "\n".join(lines) + "\n"


def _join_export_bottom(result, lf, rf):
    """Join only the terminal bottom gap; never repair an interior wall gap.

    The two supported paths retain their exact point order and coordinates.
    The user-authorized export segment may cross outside the search corridors.
    It is not image evidence and must not alter measured depth/CD or traces.
    """
    segments=[np.asarray(s,float) for t in result.guide_traces for s in t.get('segments',[]) if len(s)]
    if len(segments)!=2:return None
    if any(p.ndim!=2 or p.shape[1]!=2 or len(p)<2 or not np.isfinite(p).all() for p in segments):return None
    if any(not points_in_guides(p,result.options.wall_guides).all() for p in segments):return None

    def from_field(p, field, side):
        fp=np.asarray(field['points'],float)
        for path in (p,p[::-1]):
            start,end=path[0],path[-1]
            outer=start[0]<np.max(fp[:,0]) if side=='left' else start[0]>np.min(fp[:,0])
            if (outer and np.min(np.linalg.norm(fp-start,axis=1))<=8
                    and end[1]>start[1]+2 and end[1]>=np.max(path[:,1])-2):
                return path
        return None

    matches=[]
    for a,b in ((segments[0],segments[1]),(segments[1],segments[0])):
        left=from_field(a,lf,'left');right=from_field(b,rf,'right')
        if left is None or right is None:continue
        if left[-1,0]>right[-1,0]+1e-9:continue
        matches.append((np.vstack((left,right[::-1])),np.array([left[-1],right[-1]])))
    return matches[0] if len(matches)==1 else None


def trench_profile_exchange(result: TrenchResult, nm_per_px: float) -> dict:
    """Gapseam open surface: ordered original contour, Å, positive Y upward.

    Never sort/reconstruct a guided surface from wall rows: doing so destroys
    overhangs and rounded floors. Only reverse the complete ordered sequence.
    """
    if not math.isfinite(nm_per_px) or nm_per_px<=0:
        raise ValueError('GFE는 Å 좌표를 사용합니다. 먼저 nm/px 스케일을 보정하세요.')
    if len(result.field_angles)!=2 or any(item.get('status')!='valid' for item in result.field_angles):
        raise ValueError('좌우 Field가 미확정입니다. 양쪽 상단 Field까지 곡선에 포함해 다시 인식하세요.')
    lf,rf=result.field_angles
    metrics=result.field_measurements
    anchors=metrics.get('anchors',{})
    center=(anchors['left'][0]+anchors['right'][0])/2 if len(anchors)==2 else (float(result.left[0])+float(result.right[0]))/2
    reference=metrics.get('reference_y_px')
    if reference is None:reference=float((np.polyval(lf['line'],result.left[0])+np.polyval(rf['line'],result.right[0]))/2)
    bottom_join=None
    if result.options.strict_guides:
        candidates=[]
        for trace in result.guide_traces:
            if len(trace['segments'])!=1:continue
            p=np.asarray(trace['segments'][0],float)
            if len(p)<4 or not np.isfinite(p).all():continue
            if p[0,0]>p[-1,0]:p=p[::-1]
            # Endpoints must actually reach opposite Fields, not arbitrary
            # wall endpoints joined across the mouth or a second trench.
            if (p[0,0]>=max(q[0] for q in lf['points']) or p[-1,0]<=min(q[0] for q in rf['points'])
                    or np.min(np.linalg.norm(np.asarray(lf['points'])-p[0],axis=1))>8
                    or np.min(np.linalg.norm(np.asarray(rf['points'])-p[-1],axis=1))>8
                    or np.max(p[:,1])<result.y[-1]-2):continue
            if not points_in_guides(p,result.options.wall_guides).all():continue
            candidates.append(p)
        if len(candidates)==1:
            p=candidates[0]
        elif not candidates and (joined := _join_export_bottom(result,lf,rf)) is not None:
            p,bottom_join=joined
        else:
            raise ValueError('GFE 윤곽을 연결할 수 없습니다. 양쪽 Field와 벽을 확인하세요. 자동 연결은 맨 아래 두 끝점에만 적용됩니다.')
        converted=np.column_stack(((p[:,0]-center)*nm_per_px*10,(reference-p[:,1])*nm_per_px*10))
        kept=[converted[0].tolist()]
        for point in converted[1:]:
            if np.max(np.abs(point-kept[-1]))>1e-10:kept.append(point.tolist())
        points=kept
        source='ordered_guide_contour'
    else:
        points=np.loadtxt(_legacy_gfe_tsv(result,nm_per_px).splitlines()[1:]).tolist()
        source='legacy_wall_field_reconstruction'
        center=(float(result.left[0])+float(result.right[0]))/2
        reference=float((np.polyval(lf['line'],result.left[0])+np.polyval(rf['line'],result.right[0]))/2)
        if result.depth_px is None:
            bottom_join=np.array([[result.left[-1],result.y[-1]],[result.right[-1],result.y[-1]]])
    if len(points)<2 or points[-1][0]-points[0][0]<=1e-12:
        raise ValueError('열린 표면의 양쪽 Field 끝점을 확인하세요.')
    connection=None
    if bottom_join is not None:
        connection=dict(method='straight_terminal_join',measured=False,points_pixel=bottom_join.tolist(),
                        points=np.column_stack(((bottom_join[:,0]-center)*nm_per_px*10,
                                                (reference-bottom_join[:,1])*nm_per_px*10)).tolist())
    return dict(schema='trench-profile/v1',units='A',coordinate_system=dict(x_positive='right',y_positive='up'),
                surface_order='left_to_right',surface_closed=False,field_y=0.,points=points,initial_voids=[],
                origin_pixel=[float(center),float(reference)],nm_per_px=float(nm_per_px),source=source,
                interpolation_included=any(t.get('interpolated_points',0)>0 for t in result.guide_traces),
                initial_voids_source='not_analyzed',bottom_detected=result.depth_px is not None,
                bottom_connection=connection)


def trench_gfe_tsv(result: TrenchResult, nm_per_px: float) -> str:
    return trench_profile_tsv(trench_profile_exchange(result,nm_per_px))


def trench_profile_tsv(payload: dict) -> str:
    return 'X\tY\n'+''.join(f'{x:.9f}\t{y:.9f}\n' for x,y in payload['points'])


def _runs(row: np.ndarray, threshold: float) -> list[tuple[float, float]]:
    mask = row < threshold
    changes = np.diff(np.r_[False, mask, False].astype(np.int8))
    starts, ends = np.where(changes == 1)[0], np.where(changes == -1)[0] - 1
    runs = []
    for a, b in zip(starts, ends):
        if a < 1 or b >= len(row) - 1 or b - a < 2:
            continue  # both sidewalls must be inside the ROI
        left = a - 1 + (row[a - 1] - threshold) / max(1e-9, row[a - 1] - row[a])
        right = b + (threshold - row[b]) / max(1e-9, row[b + 1] - row[b])
        runs.append((float(left), float(right)))
    return runs


def _rim_peak(row: np.ndarray, seed: float, direction: int) -> float:
    """Use the first prominent ridge connected to the dark-space boundary.

    Search outwards from the threshold seed, never across the trench or for
    the brightest pixel in the entire material. A monotone step has no ridge
    and retains the threshold boundary. Preserve unsaturated intensities.
    """
    start = int(math.floor(seed) if direction < 0 else math.ceil(seed))
    positions = start + direction * np.arange(19)
    positions = positions[(positions >= 1) & (positions < len(row) - 1)]
    if len(positions) < 7:
        return seed
    values = row[positions]
    for j in range(1, len(values) - 4):
        k = int(positions[j])
        a, b, c = float(row[k - 1]), float(row[k]), float(row[k + 1])
        if b < max(a, c) or b <= min(a, c):
            continue
        outside = float(np.median(values[j + 1:j + 5]))
        if b - outside < 5 or b - float(values[0]) < 12:
            continue
        denominator = a - 2 * b + c
        offset = .5 * (a - c) / denominator if denominator < -1e-6 else 0
        return float(k + np.clip(offset, -.5, .5))
    # Without a ridge use the strongest connected intensity transition, not
    # an arbitrary dark-space threshold level. This aligns cliff and Field.
    lo=max(0,int(seed)-3);hi=min(len(row)-1,int(seed)+4)
    gradient=np.diff(row[lo:hi+1])*direction
    # Material direction reverses the sign of an x derivative on the left.
    if len(gradient) and float(gradient.max())>=8:
        return float(lo+int(np.argmax(gradient))+.5)
    return seed


def _smooth(values: np.ndarray, window: int) -> np.ndarray:
    radius = min(max(0, int(window) // 2), (len(values) - 1) // 2)
    if not radius:
        return values.copy()
    return np.convolve(np.pad(values, radius, mode="edge"),
                       np.ones(radius * 2 + 1) / (radius * 2 + 1), "valid")


def _profile_blocks(y):
    return np.split(np.arange(len(y)),np.flatnonzero(np.diff(y)>1.5)+1)


def _smooth_profile(values,y,window):
    result=values.copy()
    for block in _profile_blocks(y):result[block]=_smooth(values[block],window)
    return result


def find_local_bowing(y: np.ndarray, left: np.ndarray, right: np.ndarray,
                     smooth_px: int = 7, minimum_angle_deg: float = 1.0) -> list[dict]:
    """Re-entrant intervals: positive outward displacement as depth increases.

    Local bow is measured from each interval's starting vertical line. Also
    report excess over the chord through its surrounding valley/return, so a
    straight negative taper can be distinguished from a localized bulge.
    """
    if len(y) < 8:
        return []
    features = []
    for side, edge, sign in (("left", left, -1), ("right", right, 1)):
        outward = _smooth(sign * edge, max(5, smooth_px))
        slope = np.gradient(outward, y)
        flags = slope > math.tan(math.radians(minimum_angle_deg))
        changes = np.diff(np.r_[False, flags, False].astype(np.int8))
        for a, b in zip(np.where(changes == 1)[0], np.where(changes == -1)[0] - 1):
            if b - a < max(2, smooth_px // 2):
                continue
            start = int(a)
            while start > 0 and slope[start - 1] > 0:
                start -= 1
            peak = int(b)
            while peak < len(y) - 1 and slope[peak + 1] > 0:
                peak += 1
            if features and features[-1]["side"] == side and features[-1]["peak_y"] == float(y[peak]):
                continue
            end = peak
            while end < len(y) - 1 and (slope[end + 1] <= 0 or end == peak):
                end += 1
                if outward[end] <= outward[start]:
                    break
            amount = float(outward[peak] - outward[start])
            if amount < .5:
                continue
            chord = np.interp(y[start:end + 1], [y[start], y[end]], [outward[start], outward[end]])
            residual = outward[start:end + 1] - chord
            k = int(np.argmax(residual)) + start
            features.append({"side": side, "start_y": float(y[start]),
                             "negative_end_y": float(y[peak]), "end_y": float(y[end]),
                             "peak_y": float(y[peak]), "start_x": float(edge[start]),
                             "peak_x": float(edge[peak]), "end_x": float(edge[end]),
                             "bowing_px": amount, "chord_bowing_px": max(0.0, float(residual[k - start])),
                             "chord_peak_y": float(y[k]),
                             "max_negative_angle_deg": math.degrees(math.atan(float(np.max(slope[a:b + 1])))),
                             "return_confirmed": bool(end < len(y) - 1 and slope[end] <= 0)})
    return sorted(features, key=lambda item: (item["start_y"], item["side"]))


def analyze_trench(image: np.ndarray, options: TrenchOptions, *, field_reference=None) -> TrenchResult:
    if image is None or image.size == 0:
        raise ValueError("먼저 이미지를 불러오세요.")
    x, top, width, height = (int(v) for v in options.roi)
    if x < 0 or top < 0 or width < 12 or height < 16 or x + width > image.shape[1] or top + height > image.shape[0]:
        raise ValueError("입구와 양쪽 벽을 포함하는 분석 영역을 이미지 안에 지정하세요 (최소 12×16 px).")
    if not math.isfinite(options.step_px) or options.step_px <= 0:
        raise ValueError("CD 깊이 간격은 양수여야 합니다.")
    if not math.isfinite(options.rim_inset_px) or not 0<=options.rim_inset_px<=10:
        raise ValueError('빈 공간 쪽 이동은 0~10 px여야 합니다.')
    smoothing_parameters(options.contour_strength)
    if options.contour_smoothing not in METHODS:
        raise ValueError('지원하지 않는 윤곽 보정 방식입니다.')
    if options.contour_strength and not options.wall_guides:
        raise ValueError('윤곽 보정은 곡선 가이드 인식에서 사용할 수 있습니다.')
    if options.bow_reference not in ("entrance", "chord"):
        raise ValueError("지원하지 않는 Bowing 기준입니다.")
    if not math.isfinite(options.negative_slope_deg) or not 0 <= options.negative_slope_deg < 89:
        raise ValueError("Nega slope 기준 각도는 0 이상 89도 미만이어야 합니다.")
    if not math.isfinite(options.threshold) or not 0 <= options.threshold <= 254:
        raise ValueError("경계값은 0(자동) 또는 1~254여야 합니다.")
    if not 6 <= options.corner_window_px <= 200:
        raise ValueError("입구 검색 크기는 6~200 px여야 합니다.")
    guides = normalize_guides(options.wall_guides, image.shape)
    bands, per_guide = prepare_guide_bands(guides,top,height) if guides else ({},[])
    for guide,rows in zip(guides,per_guide):
        if not any(hi>=x and lo<=x+width-1 for intervals in rows.values() for lo,hi in intervals) and not (
                np.min(np.abs(sweep_samples(guide)[:,1]-top))<=options.field_search_px):
            raise ValueError('가이드 인식 띠가 분석 영역과 겹치지 않습니다. 영역 또는 가이드를 옮겨주세요.')
    for values, length in ((options.corner_rois, 4), (options.corner_spans, 2)):
        if len(values) != 2 or any(value is not None and (len(value) != length or not np.isfinite(value).all()) for value in values):
            raise ValueError("좌우 입구 검색 영역 또는 원호 선택 값이 올바르지 않습니다.")
    gray = to_gray(image[top:top + height, x:x + width])
    if not np.all(np.isfinite(gray)):
        raise ValueError("이미지에 유효하지 않은 픽셀 값이 있습니다.")
    low, high = np.percentile(gray, (1, 99))
    if high - low < 1e-6:
        raise ValueError("영역 안에 구분할 수 있는 명암 경계가 없습니다.")
    normalized = np.clip((gray - low) * (255.0 / (high - low)), 0, 255).astype(np.float32)
    if options.bright_trench:
        normalized = 255 - normalized
    normalized = cv2.GaussianBlur(normalized, (3, 3), 0)
    # Clipping used for segmentation must not flatten the brightest SEM rim.
    ridge_image = (gray - low) * (255.0 / (high - low))
    if options.bright_trench:
        ridge_image = 255 - ridge_image
    ridge_image = cv2.GaussianBlur(ridge_image.astype(np.float32), (3, 3), 0)
    threshold = float(options.threshold or (np.percentile(normalized, 5) +
                      .28 * (np.percentile(normalized, 90) - np.percentile(normalized, 5))))
    traces=[];wall_indices=[];guided_walls=False
    if guides:
        full_gray=to_gray(image)
        full_ridge=(full_gray-low)*(255.0/(high-low))
        if options.bright_trench:full_ridge=255-full_ridge
        full_ridge=cv2.GaussianBlur(full_ridge.astype(np.float32),(3,3),0)
        mode=options.boundary_mode
        if options.bright_trench and mode in ('brightest','darkest'):
            mode='darkest' if mode=='brightest' else 'brightest'
        material=ridge_image[ridge_image>threshold+20]
        local_material=float(np.percentile(material,75)) if len(material) else threshold+20
        traces=[trace_guide(full_ridge,g,threshold,mode,local_contrast=options.strict_guides,
                            material_level=local_material if options.strict_guides else None,
                            rim_inset_px=options.rim_inset_px,continuous=options.continuous_guides) for g in guides]
        if options.continuous_guides and any(len(t['segments'])>1 for t in traces):
            raise ValueError('같은 구간의 경계를 인식 범위 안에서 연결할 수 없습니다. 끊긴 부분의 곡선 위치·폭을 보정하세요.')
        traces = smooth_traces(traces, guides, options.contour_smoothing,
                               options.contour_strength, image.shape)
        for i,g in enumerate(guides):
            samples=sweep_samples(g);direction=np.gradient(samples,axis=0)
            vertical=np.abs(direction[:,1])>=.75*np.abs(direction[:,0])
            if np.ptp(samples[:,1])>=5 and np.any(vertical&(samples[:,1]>top+12)):
                wall_indices.append(i)
        if any(traces[i]['trace_coverage']<.3 or traces[i]['detected_points']<4 for i in wall_indices):
            raise ValueError('가이드 범위에서 이어지는 명암 경계를 찾지 못했습니다. 실제 윤곽을 포함하도록 위치·폭을 확인하세요.')
        if wall_indices:
            wall_guides=tuple(guides[i] for i in wall_indices)
            wall_bands,_=prepare_guide_bands(wall_guides,top,height)
            track=guided_cavity_profile([_runs(row,threshold) for row in normalized],ridge_image,
                                        wall_guides,wall_bands,[traces[i] for i in wall_indices],
                                        (x,top,width,height),_rim_peak,strict=options.strict_guides)
            candidates=[track]
            guided_walls=True
    if options.strict_guides and not guided_walls:
        raise ValueError('곡선 인식 범위 안에 양쪽 측벽이 포함되도록 곡선을 그려주세요.')
    if not guides or not wall_indices:
        # Link overlapping intervals on successive rows. An unrelated trench cannot
        # silently bridge a missing row or replace one of the selected walls.
        tracks: list[list[tuple[int, float, float]]] = []
        active: list[int] = []
        for row_index, row in enumerate(normalized):
            candidates = _runs(row, threshold)
            next_active: list[int] = []
            used: set[int] = set()
            for left, right in candidates:
                choices = []
                for track_id in active:
                    if track_id in used:
                        continue
                    _, old_left, old_right = tracks[track_id][-1]
                    overlap = min(right, old_right) - max(left, old_left)
                    jump = abs((left + right - old_left - old_right) / 2)
                    if overlap > 0 and jump < max(4, width * .08):
                        choices.append((jump, track_id))
                if choices:
                    track_id = min(choices)[1]
                    tracks[track_id].append((row_index, left, right))
                    used.add(track_id)
                else:
                    track_id = len(tracks)
                    tracks.append([(row_index, left, right)])
                next_active.append(track_id)
            active = next_active
        candidates = [track for track in tracks if len(track) >= max(12, height * .2)]
        if not candidates:
            raise ValueError('가이드 인식 띠 안에서 연속된 양쪽 경계를 찾지 못했습니다. 선의 위치·폭과 좌우를 확인하세요.'
                             if guides else "연속된 양쪽 측벽을 찾지 못했습니다. 한 Trench만 포함하도록 영역 또는 경계값을 조절하세요.")
        def track_rank(track):
            guided_count = sum(bool(bands.get((p[0],side))) for p in track for side in ('left','right'))
            center = np.median([(p[1]+p[2])/2 for p in track])
            return guided_count, len(track)*(1-.6*abs(center-width/2)/(width/2))
        candidates.sort(key=track_rank, reverse=True)
        track = np.asarray(candidates[0], dtype=float)
    # A rounded, shaded entrance can first appear as a tiny isolated dark tip.
    # Start the sidewall profile only when a usable full-width interval exists.
    widths = track[:, 2] - track[:, 1]
    # Use nearby entrance support, not the bowed body: a genuinely narrow
    # opening must not be discarded just because the trench widens at depth.
    entrance_rows = min(48, max(8, int(len(widths) * .05)))
    body_width_hint = float(np.median(widths[:entrance_rows]))
    usable_start = np.flatnonzero(widths >= body_width_hint * .65)
    if len(usable_start):
        track = track[int(usable_start[0]):]
    y = track[:, 0] + top
    peak_left = np.array([l if guided_walls else _rim_peak(ridge_image[int(row)], l, -1) for row, l, r in track])
    peak_right = np.array([r if guided_walls else _rim_peak(ridge_image[int(row)], r, 1) for row, l, r in track])
    left = _smooth_profile(peak_left + x,y, options.smooth_px)
    right = _smooth_profile(peak_right + x,y, options.smooth_px)
    for edge, raw, side in ((left,peak_left,'left'),(right,peak_right,'right')):
        for i, row in enumerate(track[:,0].astype(int)):
            search = bands.get((row,'contour'),[])
            if search and in_bands(raw[i]+x,search) and not in_bands(edge[i],search):
                edge[i] = raw[i]+x  # smoothing must not move outside a guide
    crossed = left>=right
    if crossed.any():
        left[crossed],right[crossed]=peak_left[crossed]+x,peak_right[crossed]+x
    if np.any(left>=right):
        raise ValueError('좌우 가이드 또는 경계가 교차합니다. 가이드의 좌우와 인식 폭을 확인하세요.')
    warnings: list[str] = []
    if options.contour_strength:
        warnings.append(f'윤곽 보정: {METHODS[options.contour_smoothing]} {options.contour_strength:g}/100. '
                        '보정 전 윤곽을 함께 저장합니다. Depth의 바닥 폐쇄 판단은 기존 밝기 기준입니다.')
    joined=sum(len(t.get('joins',[])) for t in traces)
    if joined:
        warnings.append(f'경계 누락 {joined}곳을 인식 범위 안에서 보간해 연결했습니다. 연결 구간은 추정 형상이며 저장 결과에 기록됩니다.')
    if np.any(np.diff(y)>1.5):
        warnings.append('일부 깊이에 양쪽 측벽 교차점이 없어 해당 CD를 제외합니다. GFE는 연속된 전체 표면이 있으면 그 좌표를 사용합니다.'
                        if options.continuous_guides else '윤곽에 미검출 구간이 있습니다. 해당 구간은 측정·CD 보간에서 제외하며 GFE 복사를 제한합니다.')
    diagnostics = []
    for index, guide in enumerate(guides):
        requested = [yy for yy,intervals in per_guide[index].items() if intervals]
        used = [[float(xx),float(yy)] for yy,ll,rr in zip(y,left,right) for xx in (ll,rr)
                if in_bands(xx,per_guide[index].get(int(yy),[]))]
        detected_rows=len({p[1] for p in used})
        coverage=detected_rows/max(1,len(requested))
        role='wall_and_surface' if index in wall_indices else 'surface'
        if role=='surface':coverage=traces[index]['trace_coverage'];used=traces[index]['points']
        diagnostics.append(dict(index=index,side='contour',requested_rows=len(requested),
                                detected_rows=detected_rows,coverage=coverage,points=used,
                                role=role,trace_coverage=traces[index]['trace_coverage'],trace_status=traces[index]['status']))
        if coverage<.8 or traces[index]['status']!='valid':
            message='보간 구간을 확인하세요.' if traces[index].get('continuity_complete') else '끊긴 구간을 확인하세요.'
            warnings.append(f"윤곽 가이드 {index+1}: 범위 내 검출 {coverage:.0%}, 곡선 추적 {traces[index]['trace_coverage']:.0%}. {message}")
    if len(candidates) > 1 and len(candidates[1]) >= len(track) * .6:
        warnings.append("긴 경계 후보가 여러 개입니다. 표시된 경계를 확인하거나 한 Trench로 영역을 좁히세요.")
    if y[0] - top > max(3, height * .03):
        warnings.append("입구 부근 경계가 끊겨 있습니다. 첫 CD는 검출된 깊이부터 표시합니다.")
    observed_depth = float(y[-1] - top)
    # A terminated trace alone is not a floor. Require narrowing AND a bright
    # center below the trace; otherwise report only a lower bound / visible span.
    end_row = int(track[-1, 0])
    tail_width = float(np.median(right[-min(4, len(right)):] - left[-min(4, len(left)):]))
    body_width = float(np.median((right - left)[:max(3, int(len(y) * .8))]))
    center = int(round((left[-1] + right[-1]) / 2 - x))
    center_column = normalized[:, max(0, center - 1):min(width, center + 2)].mean(axis=1)
    floor_candidates = np.flatnonzero(center_column[end_row + 1:] >= threshold) + end_row + 1
    floor_row = int(floor_candidates[0]) if len(floor_candidates) else height
    below = center_column[floor_row:]
    closed = (floor_row < height - 2 and floor_row - end_row <= 5 and len(below) >= 3
              and np.mean(below >= threshold) >= .9
              and (tail_width < body_width * .6 or len(below) >= 8))
    depth = None
    if closed:
        before, after = center_column[floor_row - 1], center_column[floor_row]
        depth = float(floor_row - 1 + np.clip((threshold - before) / max(1e-9, after - before), 0, 1))
        if options.strict_guides and not points_in_guides([[(left[-1]+right[-1])/2,top+depth]],guides)[0]:
            depth=None;closed=False
    if not closed:
        warnings.append("바닥 폐쇄를 확인하지 못했습니다. 전체 Depth는 미확정이며 관찰 깊이만 표시합니다.")
    # Exclude the final 10% from sidewall/bowing fits to avoid fitting the floor.
    last_wall = max(2, int(len(y) * .9) - 1)
    wall_y = y[:last_wall + 1]
    l0, r0 = float(left[0]), float(right[0])
    if options.bow_reference == "chord":
        left_ref = np.interp(y, [y[0], wall_y[-1]], [l0, left[last_wall]])
        right_ref = np.interp(y, [y[0], wall_y[-1]], [r0, right[last_wall]])
    else:
        left_ref, right_ref = np.full_like(y, l0), np.full_like(y, r0)
    left_bow = left_ref[:last_wall + 1] - left[:last_wall + 1]
    right_bow = right[:last_wall + 1] - right_ref[:last_wall + 1]
    li, ri = int(np.argmax(left_bow)), int(np.argmax(right_bow))
    if options.field_search_px < 4 or not (0 <= options.angle_span[0] < options.angle_span[1]):
        raise ValueError("Field 검색 범위와 각도 시작/끝 깊이를 확인하세요.")
    field_angles = measure_field_angles(image, options.roi, y, left, right,
                                       options.field_search_px, options.angle_span, options.bright_trench)
    angles = [item["angle_deg"] for item in field_angles]
    corners = [measure_entrance_corner(image, side, top, y, edge, options.corner_window_px,
                                      options.corner_rois[i], options.corner_spans[i], options.bright_trench)
               for i, (side, edge) in enumerate((("left", left), ("right", right)))]
    if guides:
        if options.strict_guides:
            for item in field_angles:
                item.update(status='unconfirmed',angle_status='unconfirmed',angle_deg=None,points=[],
                            source='guide',reason='곡선 인식 범위에 Field 윤곽이 포함되지 않았습니다.')
            for corner in corners:
                corner.update(status='unconfirmed',radius_px=None,center=None,trace_points=[],fit_points=[],arc_points=[],
                              reason='곡선 인식 범위에 입구 전이 윤곽이 포함되지 않았습니다.')
        field_angles,corners=measure_guided_surfaces(guides,traces,options,y,left,right,field_angles,corners,image.shape)
        if options.strict_guides:
            for item in field_angles:
                if item['points'] and not points_in_guides(item['points'],guides).all():
                    item.update(status='unconfirmed',angle_status='unconfirmed',angle_deg=None,points=[],
                                reason='Field 맞춤 윤곽이 곡선 인식 범위를 벗어나 미확정입니다.')
        angles=[item['angle_deg'] for item in field_angles]
    if any(item['angle_status']!='valid' for item in field_angles):
        warnings.append("일부 Field 또는 각도가 미확정입니다. Field / 각도 탭에서 측정 구간과 이유를 확인하세요.")
    for item, corner in zip(field_angles, corners):
        connect_entrance(item, corner)
    if any(corner["status"] != "valid" for corner in corners):
        warnings.append("일부 입구 R은 미확정입니다. 입구 곡률 탭에서 윤곽과 판정 이유를 확인하세요.")
    reference_fields,reference_corners = field_reference if field_reference is not None else (field_angles,corners)
    field_metrics = measure_field_metrics(reference_fields,traces,guides,y,left,right,closed,reference_corners) if options.strict_guides else {}
    if field_metrics:field_metrics['field_reference_scope']='unsmoothed_image_contour' if field_reference is not None else 'current_contour'
    reference_y = field_metrics.get('reference_y_px')
    if reference_y is None: reference_y = top
    first_sample = max(0, math.ceil((y[0] - reference_y) / options.step_px)) * options.step_px
    if (observed_depth-first_sample)/options.step_px>100000:
        raise ValueError('CD 깊이 간격을 늘려주세요. 한 번에 최대 100,000개 지점을 측정합니다.')
    sample_y = np.arange(first_sample, y[-1]-reference_y + 1e-8, options.step_px) + reference_y
    if len(sample_y) and np.any(np.diff(y)>1.5):
        high_index=np.clip(np.searchsorted(y,sample_y),1,len(y)-1)
        missing=(y[high_index]-y[high_index-1]>1.5)&(sample_y>y[high_index-1]+1e-7)&(sample_y<y[high_index]-1e-7)
        sample_y=sample_y[~missing]
    if options.strict_guides and len(sample_y):
        sample_left=np.interp(sample_y,y,left);sample_right=np.interp(sample_y,y,right)
        keep=points_in_guides(np.column_stack((sample_left,sample_y)),guides)&points_in_guides(np.column_stack((sample_right,sample_y)),guides)
        sample_y=sample_y[keep]
    if not len(sample_y):
        warnings.append("CD 간격이 관찰 범위보다 큽니다. 간격을 줄여주세요.")
    local=[]
    for block in _profile_blocks(wall_y):
        if len(block)>=3:
            local.extend(find_local_bowing(wall_y[block],left[block],right[block],
                                          max(7,options.smooth_px),options.negative_slope_deg))
    return TrenchResult(options, y, left, right, sample_y,
                        np.interp(sample_y, y, left), np.interp(sample_y, y, right),
                        left_ref, right_ref, depth, observed_depth,
                        max(0.0, float(left_bow[li])), max(0.0, float(right_bow[ri])),
                        float(y[li]), float(y[ri]), angles[0], angles[1], corners, threshold, warnings,
                        local, field_angles, diagnostics, traces, field_metrics)


def export_trench_csv(path: str | Path, result: TrenchResult, nm_per_px: float | None = None) -> None:
    if nm_per_px is not None and (not math.isfinite(nm_per_px) or nm_per_px <= 0):
        raise ValueError("nm/px 보정값은 양수여야 합니다.")
    with open(path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["section", "name", "value", "unit"])
        writer.writerow(["metadata", "nm_per_px", nm_per_px or "", "nm/px"])
        writer.writerow(["metadata", "bow_reference", result.options.bow_reference, ""])
        writer.writerow(["metadata", "boundary_mode", result.options.boundary_mode, ""])
        writer.writerow(["metadata", "imaging_mode_requested", result.options.imaging_mode, ""])
        for key,value in result.imaging.items():
            writer.writerow(['imaging',key,value,''])
        writer.writerow(["metadata", "contour_smoothing", result.options.contour_smoothing, ""])
        writer.writerow(["metadata", "contour_strength", result.options.contour_strength, "0..100"])
        if result.options.boundary_mode=='rim_inset':
            writer.writerow(["metadata", "rim_inset_px", result.options.rim_inset_px, "px"])
        writer.writerow(["metadata", "entrance_y_px", result.options.roi[1], "px"])
        writer.writerow(["metadata", "sidewall_end", 90, "% of detected profile"])
        writer.writerow(["metadata", "wall_guides", json.dumps(list(result.options.wall_guides),ensure_ascii=False), "px"])
        writer.writerow(["metadata","guide_method","ordered_normal_contour_then_cavity_classification" if result.guide_traces else "automatic",""])
        writer.writerow(["metadata","continuous_guides",result.options.continuous_guides,""])
        writer.writerow(["metadata","depth_reference_y_px",result.depth_origin_px,"px"])
        writer.writerow(["field_measurements","result",json.dumps(result.field_measurements,ensure_ascii=False,allow_nan=False),"px"])
        if nm_per_px:
            try:
                exchange=trench_profile_exchange(result,nm_per_px)
                writer.writerow(['profile_exchange','bottom_connection',json.dumps(exchange['bottom_connection'],allow_nan=False),'A / px'])
            except ValueError as exc:
                writer.writerow(['profile_exchange','error',str(exc),''])
        for index,trace in enumerate(result.guide_traces):
            writer.writerow(["contour_trace",index+1,json.dumps(trace,ensure_ascii=False,allow_nan=False),"px"])
        for item in result.guide_diagnostics:
            writer.writerow(["guide",str(item['index']+1),json.dumps(item,ensure_ascii=False,allow_nan=False),"px"])
        for item in result.field_angles:
            for key, value in item.items():
                writer.writerow(["field_angle", item["side"] + "_" + key,
                                 json.dumps(value, ensure_ascii=False, allow_nan=False),
                                 "px" if key.endswith("_px") else "deg" if key == "angle_deg" else ""])
        for key, value in result.summary().items():
            unit = "px" if key.endswith("_px") else "deg" if key.endswith("_deg") else ""
            writer.writerow(["summary", key, "" if value is None else value, unit])
            if unit == "px" and nm_per_px is not None:
                writer.writerow(["summary", key.removesuffix("_px") + "_nm", "" if value is None else value * nm_per_px, "nm"])
                if key in ('tcd_px','field_depth_px'):
                    for suffix,factor,label in (('um',.001,'µm'),('angstrom',10,'Å')):
                        writer.writerow(['summary',key.removesuffix('_px')+'_'+suffix,'' if value is None else value*nm_per_px*factor,label])
        for warning in result.warnings:
            writer.writerow(["warning", warning, "", ""])
        for corner in result.corners:
            for key in ("status", "reason", "source", "roi", "selection", "center", "radius_px", "rms_px", "edge_spread_px", "arc_angle_deg", "stability_fraction"):
                value = corner.get(key)
                unit = "px" if key.endswith("_px") else "deg" if key.endswith("_deg") else ""
                writer.writerow(["corner_" + corner["side"], key,
                                 json.dumps(value, ensure_ascii=False) if isinstance(value, list) else value, unit])
                if unit == "px" and nm_per_px:
                    for name, factor in (("nm", nm_per_px), ("angstrom", nm_per_px * 10)):
                        writer.writerow(["corner_" + corner["side"], key.replace("_px", "_" + name),
                                         value * factor if value is not None else "", name])
        writer.writerow([])
        writer.writerow(["corner_side", "point_kind", "point_index", "x_px", "y_px"])
        for corner in result.corners:
            for kind in ("trace_points", "fit_points", "arc_points"):
                for index, (px, py) in enumerate(corner.get(kind, [])):
                    writer.writerow([corner["side"], kind, index, px, py])
        writer.writerow([])
        writer.writerow(["side", "negative_start_depth_px", "negative_end_depth_px", "local_end_depth_px",
                         "local_bowing_px", "chord_bowing_px", "max_negative_angle_deg", "return_confirmed",
                         "local_bowing_nm", "chord_bowing_nm"])
        for item in result.local_bowing:
            writer.writerow([item["side"], item["start_y"] - result.options.roi[1],
                             item["negative_end_y"] - result.options.roi[1], item["end_y"] - result.options.roi[1],
                             item["bowing_px"], item["chord_bowing_px"], item["max_negative_angle_deg"],
                             item["return_confirmed"], item["bowing_px"] * nm_per_px if nm_per_px else "",
                             item["chord_bowing_px"] * nm_per_px if nm_per_px else ""])
        writer.writerow([])
        writer.writerow(["depth_px", "left_x_px", "right_x_px", "cd_px", "depth_nm", "cd_nm"])
        for y, left, right in zip(result.sample_y, result.sample_left, result.sample_right):
            depth, cd = float(y - result.depth_origin_px), float(right - left)
            writer.writerow([depth, float(left), float(right), cd,
                             depth * nm_per_px if nm_per_px else "", cd * nm_per_px if nm_per_px else ""])
