from pathlib import Path
import cv2
import numpy as np

from PySide6.QtWidgets import QApplication
from angle_cal.app import MainWindow
from angle_cal.stitching import StitchOptions, save_stitch_result, stitch_paths


def _write(path: Path, image: np.ndarray) -> None:
    cv2.imencode(path.suffix, image)[1].tofile(str(path))


def test_integer_translation_preserves_original_uint16_pixels(tmp_path):
    rng = np.random.default_rng(42)
    scene = rng.integers(0, 65535, (180, 260), dtype=np.uint16)
    left, right = scene[:, :180], scene[:, 80:]
    a, b = tmp_path / "a.tif", tmp_path / "b.tif"
    _write(a, left); _write(b, right)
    result = stitch_paths([str(a), str(b)], StitchOptions(min_matches=6, ratio_test=.8))
    assert result.output_size == (260, 180)
    assert np.array_equal(result.image, scene)
    assert {p.mode for p in result.placements} == {"anchor", "exact translation"}


def test_saved_result_has_alpha_mask_and_report(tmp_path):
    image = np.random.default_rng(7).integers(0, 65535, (150, 150), dtype=np.uint16)
    a, b = tmp_path / "a.tif", tmp_path / "b.tif"
    _write(a, image); _write(b, image)
    result = stitch_paths([str(a), str(b)], StitchOptions(min_matches=4, ratio_test=.9))
    output, mask, report = save_stitch_result(tmp_path / "merged.tif", result)
    loaded = cv2.imdecode(np.fromfile(output, np.uint8), cv2.IMREAD_UNCHANGED)
    assert loaded.dtype == np.uint16 and loaded.shape[2] == 4
    assert mask.exists() and report.exists()
    assert "recalibration_required" in report.read_text(encoding="utf-8")


def test_multiple_addons_can_be_enabled_and_disabled():
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    try:
        assert window.ribbon_tabs.count() == 5
        window.addon_actions["photo_merge"].setChecked(True)
        window.addon_actions["trench_analyzer"].setChecked(True)
        assert window.ribbon_tabs.count() == 7
        assert window.ribbon_tabs.tabText(5) == "사진 합치기"
        assert window.ribbon_tabs.tabText(6) == "Trench 자동분석기"
        window.addon_actions["photo_merge"].setChecked(False)
        assert window.ribbon_tabs.count() == 6
        assert window.ribbon_tabs.tabText(5) == "Trench 자동분석기"
    finally:
        window.close()
