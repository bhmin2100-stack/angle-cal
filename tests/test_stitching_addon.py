from pathlib import Path
import time
import cv2
import numpy as np

from PySide6.QtWidgets import QApplication
from angle_cal.app import MainWindow
from angle_cal.photo_merge import PhotoMergeBoard
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


def test_photo_merge_button_starts_worker_and_finishes(tmp_path):
    app = QApplication.instance() or QApplication([])
    scene = np.random.default_rng(31).integers(0, 256, (220, 420), dtype=np.uint8)
    left_path, right_path = tmp_path / "left.png", tmp_path / "right.png"
    _write(left_path, scene[:, :300])
    _write(right_path, scene[:, 120:])
    dialog = PhotoMergeBoard()
    dialog.add_paths([str(left_path), str(right_path)])
    try:
        dialog.align_button.click()
        assert dialog.status.text() == "보드 이미지 자동 정렬 준비 중…"
        deadline = time.monotonic() + 5.0
        captured = []
        dialog.result_ready.connect(captured.append)
        while not captured and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(0.01)
        assert captured
        assert captured[0].output_size == (420, 220)
        assert dialog.status.text().startswith("합치기 완료:")
    finally:
        dialog.close()
        app.processEvents()


def test_photo_merge_addon_reuses_thumbnail_dock_and_central_board(tmp_path):
    app = QApplication.instance() or QApplication([])
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    _write(first, np.full((80, 120, 3), 70, dtype=np.uint8))
    _write(second, np.full((80, 120, 3), 170, dtype=np.uint8))
    window = MainWindow()
    try:
        window.browser_root = tmp_path
        window.browser_image_paths = [str(first), str(second)]
        window.selected_thumbnail_paths = {str(first), str(second)}
        window._populate_thumbnails()
        window.addon_actions["photo_merge"].setChecked(True)

        assert window.photo_merge_board is not None
        assert window.workspace_stack.currentWidget() is window.photo_merge_board
        window.open_photo_merge_dialog()
        assert len(window.photo_merge_board.view.items_in_board()) == 2

        item = window.photo_merge_board.view.items_in_board()[0]
        item.setPos(115, 75)
        item.setOpacity(0.44)
        item.setSelected(True)
        window.photo_merge_board.view.delete_selected()
        assert len(window.photo_merge_board.view.items_in_board()) == 1
        assert window.photo_merge_board.count_label.text() == "보드 이미지 1장"
        assert not window.photo_merge_board.align_button.isEnabled()
        app.processEvents()
    finally:
        window.close()


def test_thumbnail_width_fills_viewport_for_each_column_count(tmp_path):
    app = QApplication.instance() or QApplication([])
    image_path = tmp_path / "wide.png"
    _write(image_path, np.full((60, 180, 3), 180, dtype=np.uint8))
    window = MainWindow()
    try:
        window.browser_root = tmp_path
        window.browser_image_paths = [str(image_path)]
        window.resize(1280, 820)
        window.show()
        app.processEvents()
        for columns in (1, 2, 3):
            window.thumbnail_columns = columns
            thumb_width, _, _, _ = window._thumbnail_dimensions()
            margins = window.thumbnail_layout.contentsMargins()
            occupied = thumb_width * columns + window.thumbnail_layout.horizontalSpacing() * (columns - 1)
            available = window.thumbnail_scroll.viewport().width() - margins.left() - margins.right()
            assert 0 <= available - occupied < columns

        window.thumbnail_columns = 3
        window.thumbnail_scroll.resize(90, 300)
        app.processEvents()
        window._populate_thumbnails()
        button = window.thumbnail_buttons[str(image_path)]
        narrow_width, _, narrow_icon_width, _ = window._thumbnail_dimensions()
        assert button.width() == narrow_width
        assert narrow_width * 3 + window.thumbnail_layout.horizontalSpacing() * 2 <= (
            window.thumbnail_scroll.viewport().width() - margins.left() - margins.right()
        )
        assert max(size.width() for size in button.icon().availableSizes()) <= narrow_icon_width
    finally:
        window.close()
