"""Small, reproducible CPU benchmark for AngleCal's photo-stitching engine."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import time

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from angle_cal.stitching import StitchLayoutHint, StitchingNeedsManual, stitch_paths  # noqa: E402


def _write(path: Path, image: np.ndarray) -> None:
    ok, encoded = cv2.imencode(path.suffix, image)
    if not ok:
        raise RuntimeError(f"Could not encode {path}")
    encoded.tofile(str(path))


def _hint(path: Path, x: float, y: float = 0.0) -> StitchLayoutHint:
    return StitchLayoutHint(
        str(path),
        np.array([[1.0, 0.0, x], [0.0, 1.0, y], [0.0, 0.0, 1.0]], dtype=np.float64),
    )


def _run_case(name, paths, hints, expected_size=None, expect_rejection=False):
    started = time.perf_counter()
    rejected = False
    result = None
    try:
        result = stitch_paths([str(path) for path in paths], layout_hints=hints)
    except StitchingNeedsManual:
        rejected = True
    elapsed = time.perf_counter() - started
    passed = rejected if expect_rejection else bool(result and (expected_size is None or result.output_size == expected_size))
    return {
        "case": name,
        "seconds": round(elapsed, 3),
        "passed": passed,
        "rejected": rejected,
        "output_size": list(result.output_size) if result else None,
        "modes": [placement.mode for placement in result.placements] if result else [],
    }


def main() -> int:
    rng = np.random.default_rng(20260811)
    records = []
    with tempfile.TemporaryDirectory(prefix="anglecal-stitch-bench-") as temp:
        root = Path(temp)

        scene = rng.integers(0, 65536, (900, 1800), dtype=np.uint16)
        left, right = root / "translation-left.tif", root / "translation-right.tif"
        _write(left, scene[:, :1200])
        _write(right, scene[:, 600:])
        records.append(
            _run_case(
                "16-bit integer translation",
                [left, right],
                [_hint(left, 0), _hint(right, 600)],
                expected_size=(1800, 900),
            )
        )

        tile = rng.integers(0, 256, (120, 120), dtype=np.uint8)
        repeated = np.tile(tile, (6, 15))
        repeated[80:100, 310:330] = 255
        repeated[500:525, 1180:1210] = 0
        repeat_left, repeat_right = root / "repeat-left.png", root / "repeat-right.png"
        _write(repeat_left, repeated[:, :1200])
        _write(repeat_right, repeated[:, 600:])
        records.append(
            _run_case(
                "repetitive SEM-like texture with board prior",
                [repeat_left, repeat_right],
                [_hint(repeat_left, 0), _hint(repeat_right, 600)],
                expected_size=(1800, 720),
            )
        )

        affine_scene = rng.integers(0, 256, (720, 1000), dtype=np.uint8)
        rotation = cv2.getRotationMatrix2D((500, 360), 2.0, 1.0)
        rotation[:, 2] += (12.0, -8.0)
        affine_moved = cv2.warpAffine(affine_scene, rotation, (1000, 720))
        affine_a, affine_b = root / "affine-a.png", root / "affine-b.png"
        _write(affine_a, affine_scene)
        _write(affine_b, affine_moved)
        records.append(
            _run_case(
                "small rotation and translation",
                [affine_a, affine_b],
                [_hint(affine_a, 0), _hint(affine_b, 0)],
            )
        )

        unrelated_a, unrelated_b = root / "unrelated-a.png", root / "unrelated-b.png"
        _write(unrelated_a, rng.integers(0, 256, (720, 1000), dtype=np.uint8))
        _write(unrelated_b, rng.integers(0, 256, (720, 1000), dtype=np.uint8))
        records.append(
            _run_case(
                "unrelated-image false-positive rejection",
                [unrelated_a, unrelated_b],
                [_hint(unrelated_a, 0), _hint(unrelated_b, 500)],
                expect_rejection=True,
            )
        )

    summary = {
        "opencv": cv2.__version__,
        "all_passed": all(record["passed"] for record in records),
        "total_seconds": round(sum(record["seconds"] for record in records), 3),
        "cases": records,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
