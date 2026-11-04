"""Experimental, material-agnostic connected-region segmentation (BGR input)."""
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time

import cv2
import numpy as np


@dataclass(frozen=True)
class RegionOptions:
    detail: int = 50  # higher = finer regions, not a requested region count
    min_size: int = 48  # analysis pixels; large values can remove thin films
    smoothing: float = 1.0
    texture: float = 0.0
    max_dimension: int = 1600  # 0 = original resolution


@dataclass
class RegionResult:
    labels: np.ndarray  # original dimensions, positive contiguous IDs
    analysis_shape: tuple
    options: RegionOptions
    elapsed: float

    @property
    def count(self):
        return int(self.labels.max())


def normalized_rgb(image):
    """Global linear normalization preserves channel ratios and 16-bit contrast."""
    a = np.asarray(image)
    if a.ndim not in (2, 3) or min(a.shape[:2]) < 2:
        raise ValueError("가로·세로 2픽셀 이상의 이미지가 필요합니다.")
    if a.ndim == 3 and a.shape[2] not in (1, 3, 4):
        raise ValueError("지원하지 않는 이미지 채널 수입니다.")
    if not np.issubdtype(a.dtype, np.number) or not np.isfinite(a).all():
        raise ValueError("이미지에는 유효한 숫자 픽셀만 포함되어야 합니다.")
    if a.ndim == 2:
        a = np.repeat(a[..., None], 3, axis=2)
    elif a.shape[2] == 1:
        a = np.repeat(a, 3, axis=2)
    else:
        a = a[..., :3][..., ::-1]
    rgb = a.astype(np.float32)
    if image.dtype == np.uint8:
        rgb /= 255.0
    else:
        lo, hi = float(rgb.min()), float(rgb.max())
        rgb = (rgb - lo) / (hi - lo) if hi > lo else np.zeros_like(rgb)
    return np.ascontiguousarray(rgb)


def segment_regions(image, options=RegionOptions(), cancelled=lambda: False):
    from skimage.segmentation import felzenszwalb

    if not 0 <= options.detail <= 100 or not 1 <= options.min_size <= 100000:
        raise ValueError("분할 세밀도 또는 최소 영역 크기가 범위를 벗어났습니다.")
    if not 0 <= options.smoothing <= 5 or not 0 <= options.texture <= 2:
        raise ValueError("노이즈 완화 또는 질감 비중이 범위를 벗어났습니다.")
    if options.max_dimension != 0 and not 64 <= options.max_dimension <= 4096:
        raise ValueError("분석 해상도는 원본 또는 64~4096이어야 합니다.")
    started = time.perf_counter()
    rgb = normalized_rgb(image)
    h, w = rgb.shape[:2]
    ratio = min(1.0, options.max_dimension / max(h, w)) if options.max_dimension else 1.0
    shape = (max(2, round(h * ratio)), max(2, round(w * ratio)))
    if shape[0] * shape[1] > 8_000_000:
        raise ValueError("원본 분석은 800만 픽셀까지 지원합니다. 분석 해상도를 낮춰 주세요.")
    work = cv2.resize(rgb, (shape[1], shape[0]), interpolation=cv2.INTER_AREA) if shape != (h, w) else rgb
    if cancelled():
        raise InterruptedError("분할을 취소했습니다.")
    # A local standard-deviation feature exposes texture transitions. Encode it
    # in three channels for skimage's RGB interface; color mode stays unchanged.
    if options.texture:
        gray = cv2.cvtColor(work, cv2.COLOR_RGB2GRAY)
        mean = cv2.GaussianBlur(gray, (0, 0), 2)
        variance = np.maximum(cv2.GaussianBlur(gray * gray, (0, 0), 2) - mean * mean, 0)
        std = np.sqrt(variance) * options.texture * 2
        work = np.dstack((mean, mean, std)).astype(np.float32)
    scale = 12.0 * (500.0 ** ((100 - options.detail) / 100))
    labels = felzenszwalb(work, scale=scale, sigma=options.smoothing,
                          min_size=options.min_size, channel_axis=-1).astype(np.int32) + 1
    if cancelled():
        raise InterruptedError("분할을 취소했습니다.")
    if shape != (h, w):
        labels = cv2.resize(labels, (w, h), interpolation=cv2.INTER_NEAREST)
    return RegionResult(labels, shape, options, time.perf_counter() - started)


def boundaries(labels):
    edge = np.zeros(labels.shape, bool)
    edge[1:] |= labels[1:] != labels[:-1]
    edge[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    return edge


def colorize(labels):
    # Deterministic colors, stable while switching display modes.
    colors = np.random.default_rng(421).integers(50, 240, (int(labels.max()) + 1, 3), dtype=np.uint8)
    return colors[labels]


def render_regions(rgb, result, mode="overlay", opacity=0.5, selected=0):
    base = np.uint8(np.clip(rgb * 255, 0, 255))
    if mode == "original":
        return base
    colors = colorize(result.labels)
    if mode == "colors":
        shown = colors
    elif mode == "boundaries":
        shown = base.copy()
        shown[boundaries(result.labels)] = (255, 215, 40)
    else:
        shown = np.uint8(base * (1 - opacity) + colors * opacity)
    if selected:
        mask = result.labels == selected
        shown = np.uint8(shown * 0.35)
        shown[mask] = np.uint8(base[mask] * 0.3 + np.array([255, 220, 50]) * 0.7)
    return np.ascontiguousarray(shown)


def save_regions(directory, source, rgb, result):
    """Save an immutable run directory; never overwrite another experiment."""
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(target / "labels.npz", labels=result.labels)
    for name, pixels in (("regions", colorize(result.labels)),
                         ("overlay", render_regions(rgb, result)),
                         ("boundaries", render_regions(rgb, result, "boundaries"))):
        ok, data = cv2.imencode(".png", pixels[..., ::-1])
        if not ok:
            raise OSError("결과 이미지 인코딩에 실패했습니다.")
        data.tofile(str(target / f"{name}.png"))
    counts = np.bincount(result.labels.ravel())[1:]
    metadata = dict(format="anglecal.regions.v1", source=str(source),
                    original_shape=list(result.labels.shape), analysis_shape=list(result.analysis_shape),
                    options=asdict(result.options), region_count=result.count,
                    elapsed_seconds=result.elapsed, area_pixels=counts.tolist(),
                    label_ids="1-based; area_pixels[id-1]", material_identification=False,
                    algorithm="felzenszwalb", upsampling="nearest" if result.analysis_shape != result.labels.shape else None)
    (target / "result.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return target
