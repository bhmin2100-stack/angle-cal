"""Generate deterministic low-contrast VNAND stitching regression samples."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "samples" / "photo_merge" / "reproducibility"


def write(path: Path, image: np.ndarray) -> None:
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError(f"Could not encode {path}")
    encoded.tofile(str(path))


def scene_image() -> np.ndarray:
    height, width = 1080, 520
    yy, xx = np.mgrid[:height, :width]
    layer = 112.0 + 7.5 * np.sin(2.0 * np.pi * yy / 31.0)
    layer += 2.5 * np.sin(2.0 * np.pi * yy / 8.0)
    vignette = -8.0 * ((xx - width / 2.0) / width) ** 2
    image = layer + vignette

    normalized_y = yy[:, 0] / (height - 1)
    half_width = 42.0 + 16.0 * np.sin(np.pi * normalized_y) ** 2
    half_width += 4.0 * np.sin(2.0 * np.pi * normalized_y + 0.35)
    distance = np.abs(xx - width / 2.0)
    trench = distance <= half_width[:, None]
    edge = np.abs(distance - half_width[:, None]) <= 3.0
    image[trench] = 38.0 + 3.0 * np.sin(yy[trench] / 17.0)
    image[edge] = 184.0

    # Sparse, deterministic details break the periodic mold-layer ambiguity.
    cv2.circle(image, (235, 344), 7, 154, -1)
    cv2.circle(image, (286, 701), 6, 72, -1)
    cv2.line(image, (207, 522), (231, 522), 165, 3)
    image += np.random.default_rng(20260928).normal(0.0, 4.0, image.shape)
    return np.clip(image, 0, 255).astype(np.uint8)


def footer(width: int, height: int = 60) -> np.ndarray:
    band = np.zeros((height, width), np.uint8)
    cv2.line(band, (0, 0), (width - 1, 0), 145, 2)
    cv2.line(band, (300, 28), (438, 28), 245, 4)
    cv2.line(band, (300, 20), (300, 36), 245, 3)
    cv2.line(band, (438, 20), (438, 36), 245, 3)
    cv2.putText(band, "1 um", (447, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.45, 235, 1, cv2.LINE_AA)
    return band


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    scene = scene_image()
    write(OUTPUT / "00_expected_full_scene_not_input.png", scene)
    starts = (0, 310, 620)
    gains = (0.91, 1.08, 0.82)
    offsets = (12.0, -7.0, 24.0)
    blur_sigmas = (0.0, 0.75, 1.15)
    for index, (start, gain, offset, sigma) in enumerate(zip(starts, gains, offsets, blur_sigmas), 1):
        body = scene[start : start + 460].astype(np.float32) * gain + offset
        body = np.clip(body, 0, 255).astype(np.uint8)
        if sigma:
            body = cv2.GaussianBlur(body, (0, 0), sigma)
        capture = np.vstack((body, footer(scene.shape[1])))
        write(OUTPUT / f"vnand_low_contrast_capture_{index}.png", capture)


if __name__ == "__main__":
    main()
