from pathlib import Path
import time
import cv2
import numpy as np
import pytest

from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication
from angle_cal.app import MainWindow
from angle_cal.photo_merge import PhotoMergeBoard
from angle_cal.stitching import (
    StitchLayoutHint,
    StitchOptions,
    StitchingNeedsManual,
    detect_bottom_overlay_fraction,
    save_stitch_result,
    save_stitch_result_auto,
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
    image = np.random.default_rng(7).integers(0, 65535, (150, 220), dtype=np.uint16)
    a, b = tmp_path / "a.tif", tmp_path / "b.tif"
    _write(a, image[:, :160]); _write(b, image[:, 60:])
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

    assert result.output_size == (420, 220)
    assert np.array_equal(result.image[:180], scene)
    assert result.scale_bar_source in {str(a), str(b)}
    assert np.count_nonzero(result.image[180:]) > 0


def test_bright_mold_layers_without_dark_footer_are_not_cropped():
    height, width = 300, 420
    yy, _ = np.mgrid[:height, :width]
    image = np.uint8(np.clip(125 + 55 * (np.sin(yy / 5.0) > 0), 0, 255))
    assert detect_bottom_overlay_fraction([image]) == 0.0


def test_multiple_addons_open_independent_checked_windows():
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    try:
        assert window.ribbon_tabs.count() == 5
        window.addon_actions["photo_merge"].setChecked(True)
        window.addon_actions["trench_analyzer"].setChecked(True)
        app.processEvents()

        assert window.ribbon_tabs.count() == 5
        assert window.addon_actions["photo_merge"].isChecked()
        assert window.addon_actions["trench_analyzer"].isChecked()
        assert window.addon_windows["photo_merge"].isVisible()
        assert window.addon_windows["trench_analyzer"].isVisible()
        assert window.addon_windows["photo_merge"].windowTitle() == "AngleCal 애드온 - 사진 합치기"

        window.addon_actions["photo_merge"].setChecked(False)
        app.processEvents()
        assert not window.addon_windows["photo_merge"].isVisible()
        assert window.addon_windows["trench_analyzer"].isVisible()

        window.addon_windows["trench_analyzer"].close()
        app.processEvents()
        assert not window.addon_actions["trench_analyzer"].isChecked()
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
        assert dialog.status.text().startswith("합치기 완료")
        assert "align_" in Path(captured[0].saved_path).name
        assert "2imgs" in Path(captured[0].saved_path).name
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
        assert window.addon_windows["photo_merge"].isVisible()
        assert window.workspace_stack.currentWidget() is window.canvas
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


def test_photo_merge_board_crop_hides_excluded_area_after_completion(tmp_path):
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

        first.set_match_rect((0.1, 0.0, 0.8, 0.8))
        assert second.match_rect is None
        assert [hint.match_rect for hint in board.view.layout_hints()] == [
            (0.1, 0.0, 0.8, 0.8),
            None,
        ]

        board.crop_button.click()
        assert not board.crop_button.isChecked()
        image = first.pixmap().toImage()
        assert image.pixelColor(0, image.height() // 2).alpha() == 0
        assert image.pixelColor(image.width() // 2, image.height() // 2).alpha() == 255

        first.setSelected(True)
        board.crop_reset_button.click()
        assert first.match_rect is None
        assert second.match_rect is None
    finally:
        board.close()
        app.processEvents()


def test_registration_is_reproducible_after_arbitrary_board_moves(tmp_path):
    height, width = 240, 560
    yy, xx = np.mgrid[:height, :width]
    scene = 104 + 18 * np.sin(yy / 7.0) + 8 * np.sin(yy / 2.5)
    scene += 32 * np.exp(-((xx - 280) / 18.0) ** 2)
    scene += np.random.default_rng(902).normal(0, 8.0, scene.shape)
    scene[:, 258:264] -= 28
    scene[:, 296:302] += 24
    cv2.circle(scene, (278, 82), 12, 52, -1)
    scene = np.clip(scene, 0, 255).astype(np.uint8)
    left = scene[:, :380]
    right = cv2.GaussianBlur(scene[:, 180:], (3, 3), 0)
    right = np.clip(right.astype(np.float32) * 0.82 + 19, 0, 255).astype(np.uint8)
    a, b = tmp_path / "repro-left.png", tmp_path / "repro-right.png"
    _write(a, left)
    _write(b, right)

    first_hints = [
        StitchLayoutHint(str(a), np.eye(3)),
        StitchLayoutHint(str(b), np.array([[1, 0, 180], [0, 1, 0], [0, 0, 1]], np.float64)),
    ]
    moved_hints = [
        StitchLayoutHint(str(a), np.array([[1, 0, -700], [0, 1, 420], [0, 0, 1]], np.float64)),
        StitchLayoutHint(str(b), np.array([[1, 0, 900], [0, 1, -360], [0, 0, 1]], np.float64)),
    ]
    first = stitch_paths([str(a), str(b)], layout_hints=first_hints)
    moved = stitch_paths([str(a), str(b)], layout_hints=moved_hints)

    assert first.output_size == moved.output_size
    assert first.confidence == pytest.approx(moved.confidence, abs=1e-9)
    assert all(
        np.allclose(a_placement.transform, b_placement.transform)
        for a_placement, b_placement in zip(first.placements, moved.placements)
    )


def test_checked_in_low_contrast_vnand_samples_preserve_one_scale_bar():
    sample_dir = Path(__file__).resolve().parents[1] / "samples" / "photo_merge" / "reproducibility"
    paths = [str(sample_dir / f"vnand_low_contrast_capture_{index}.png") for index in (1, 2, 3)]
    result = stitch_paths(paths)

    assert result.output_size == (520, 1140)
    assert result.confidence >= 0.90
    assert result.scale_bar_source is not None
    assert [placement.mode for placement in result.placements] == [
        "anchor",
        "exact translation",
        "exact translation",
    ]


def test_photo_merge_board_has_band_panel_zoom_and_delete_shortcuts(tmp_path):
    app = QApplication.instance() or QApplication([])
    scene = np.random.default_rng(55).integers(0, 256, (180, 360), dtype=np.uint8)
    first_path, second_path = tmp_path / "band-a.png", tmp_path / "band-b.png"
    _write(first_path, scene[:, :260])
    _write(second_path, scene[:, 100:])
    board = PhotoMergeBoard()
    board.add_paths([str(first_path), str(second_path)])
    try:
        assert not hasattr(board, "crop_scope")
        assert "이미지 불러오기" not in {button.text() for button in board.findChildren(type(board.align_button))}
        items = board.view.items_in_board()
        items[0].setSelected(True)
        board.view._emit_band_profiles()
        assert board.band_graph.selected_name == Path(items[0].path).name
        assert board.band_graph.profiles

        before = board.view.transform().m11()
        board.view.zoom_by(120)
        assert board.view.transform().m11() > before

        board.view.setFocus()
        QTest.keyClick(board.view, Qt.Key.Key_Delete)
        app.processEvents()
        assert len(board.view.items_in_board()) == 1
    finally:
        board.close()
        app.processEvents()


def test_saved_merge_is_added_to_thumbnails_and_combined_tab(tmp_path):
    app = QApplication.instance() or QApplication([])
    scene = np.random.default_rng(77).integers(0, 256, (180, 420), dtype=np.uint8)
    first, second = tmp_path / "source-left.png", tmp_path / "source-right.png"
    _write(first, scene[:, :300])
    _write(second, scene[:, 120:])
    result = stitch_paths([str(first), str(second)])
    saved = save_stitch_result_auto(result)
    window = MainWindow()
    try:
        window.browser_root = tmp_path
        window.browser_image_paths = [str(first), str(second)]
        window._populate_thumbnails()
        window._show_photo_merge_result(result)
        resolved = str(saved.resolve())
        assert window.image_path == resolved
        assert resolved in window.browser_image_paths
        assert resolved in window.thumbnail_buttons
        assert resolved in window.favorite_image_paths
        assert window.favorite_image_groups[resolved] == "Combined Pictures"
        assert "align_" in saved.name and "2imgs" in saved.name
    finally:
        window.close()
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
