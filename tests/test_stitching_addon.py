from pathlib import Path
import time
import cv2
import numpy as np
import pytest

from PySide6.QtWidgets import QApplication
from angle_cal.app import MainWindow
from angle_cal.photo_merge import PhotoMergeBoard
from angle_cal.stitching import (
    StitchLayoutHint,
    StitchOptions,
    StitchingNeedsManual,
    detect_bottom_overlay_fraction,
    save_stitch_result,
    stitch_paths,
)


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


def test_board_hint_prevents_false_full_overlay(tmp_path):
    image = np.random.default_rng(91).integers(0, 256, (220, 300), dtype=np.uint8)
    a, b = tmp_path / "same-a.png", tmp_path / "same-b.png"
    _write(a, image)
    _write(b, image)
    hints = [
        StitchLayoutHint(str(a), np.eye(3)),
        StitchLayoutHint(str(b), np.array([[1.0, 0.0, 270.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])),
    ]

    with pytest.raises(StitchingNeedsManual):
        stitch_paths([str(a), str(b)], layout_hints=hints)


def test_unrelated_images_are_rejected_instead_of_composited(tmp_path):
    first = np.random.default_rng(123).integers(0, 256, (220, 300), dtype=np.uint8)
    second = np.random.default_rng(456).integers(0, 256, (220, 300), dtype=np.uint8)
    a, b = tmp_path / "unrelated-a.png", tmp_path / "unrelated-b.png"
    _write(a, first)
    _write(b, second)

    with pytest.raises(StitchingNeedsManual):
        stitch_paths([str(a), str(b)])


def test_repeated_sem_footer_is_excluded_from_alignment(tmp_path):
    rng = np.random.default_rng(888)
    scene = rng.integers(0, 256, (180, 420), dtype=np.uint8)
    footer = np.zeros((40, 300), dtype=np.uint8)
    cv2.line(footer, (0, 0), (299, 0), 180, 2)
    cv2.line(footer, (24, 18), (124, 18), 255, 4)
    cv2.putText(footer, "500 nm  15.0 kV", (145, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.38, 230, 1, cv2.LINE_AA)
    left = np.vstack((scene[:, :300], footer))
    right = np.vstack((scene[:, 120:], footer))
    a, b = tmp_path / "sem-left.png", tmp_path / "sem-right.png"
    _write(a, left)
    _write(b, right)

    detected = detect_bottom_overlay_fraction([left, right])
    assert detected == pytest.approx(40 / 220, abs=0.035)
    hints = [
        StitchLayoutHint(str(a), np.eye(3), (0.0, 0.0, 1.0, 180 / 220)),
        StitchLayoutHint(
            str(b),
            np.array([[1.0, 0.0, 120.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
            (0.0, 0.0, 1.0, 180 / 220),
        ),
    ]
    result = stitch_paths([str(a), str(b)], layout_hints=hints)

    assert result.output_size == (420, 180)
    assert np.array_equal(result.image, scene)


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
        assert not item.acceptHoverEvents()
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


def test_photo_merge_board_crop_regions_support_individual_and_common_scope(tmp_path):
    app = QApplication.instance() or QApplication([])
    first_path = tmp_path / "crop-first.png"
    second_path = tmp_path / "crop-second.png"
    _write(first_path, np.full((120, 180), 70, dtype=np.uint8))
    _write(second_path, np.full((120, 180), 170, dtype=np.uint8))
    board = PhotoMergeBoard()
    board.add_paths([str(first_path), str(second_path)])
    try:
        items = sorted(board.view.items_in_board(), key=lambda item: item.path)
        first, second = items
        first.setSelected(True)
        board.crop_button.click()
        assert board.crop_button.isChecked()
        assert first.crop_overlay.isVisible()
        assert not second.crop_overlay.isVisible()

        first.set_match_rect((0.05, 0.08, 0.9, 0.72))
        assert first.match_rect == (0.05, 0.08, 0.9, 0.72)
        assert second.match_rect is None

        board.crop_scope.setCurrentIndex(1)
        first.set_match_rect((0.1, 0.0, 0.8, 0.8))
        assert second.match_rect == first.match_rect
        assert all(item.crop_overlay.isVisible() for item in items)
        assert [hint.match_rect for hint in board.view.layout_hints()] == [
            (0.1, 0.0, 0.8, 0.8),
            (0.1, 0.0, 0.8, 0.8),
        ]

        board.crop_reset_button.click()
        assert all(item.match_rect is None for item in items)
    finally:
        board.close()
        app.processEvents()


def test_thumbnail_width_fills_viewport_for_each_column_count(tmp_path):
    app = QApplication.instance() or QApplication([])
    long_folder = tmp_path.joinpath(*(["very-long-folder-title"] * 4))
    long_folder.mkdir(parents=True)
    image_path = long_folder / "wide.png"
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
        assert window.thumbnail_scroll.horizontalScrollBar().maximum() == 0
        window.thumbnail_hover_path = str(image_path)
        window.thumbnail_hover_button = button
        window._show_thumbnail_hover_preview()
        assert window.thumbnail_hover_popup.isVisible()
        assert window.thumbnail_hover_popup.pixmap().width() > button.iconSize().width()
        window._hide_thumbnail_hover_preview()
        assert not window.thumbnail_hover_popup.isVisible()
    finally:
        window.close()
