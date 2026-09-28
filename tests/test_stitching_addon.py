from pathlib import Path
import time
import cv2
import numpy as np
import pytest

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QImage, QPainter
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication
from angle_cal.app import MainWindow
from angle_cal.band_registration import band_profiles, full_overlap_score, translation_candidates
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


def test_band_image_strips_match_graph_samples_and_source_regions():
    yy, xx = np.mgrid[:90, :120]
    first = (xx + yy).astype(np.uint8)
    second = first + 12
    valid = np.zeros(first.shape, dtype=bool)
    valid[9:81, 12:108] = True
    details = band_profiles(first, second, valid, with_regions=True)
    assert len(details) == 6
    assert {entry[4]["axis"] for entry in details} == {0, 1}
    for name, a, b, score, detail in details:
        x, y, width, height = detail["region"]
        expected = first[y:y + height, x:x + width]
        if detail["axis"] == 1:
            expected = expected.T
        assert np.array_equal(detail["first_strip"], expected)
        assert np.allclose(detail["first_strip"].mean(axis=0), a)
        assert np.allclose(detail["second_strip"].mean(axis=0), b)
    # Invalid holes must not contribute to either the displayed strip or curve.
    valid[20:35, 25:40] = False
    for _, a, b, _, detail in band_profiles(first, second, valid, with_regions=True):
        mask = detail["strip_mask"]
        counts = mask.sum(axis=0)
        assert np.allclose((detail["first_strip"] * mask).sum(axis=0) / counts, a)
        assert np.allclose((detail["second_strip"] * mask).sum(axis=0) / counts, b)


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


def test_full_overlap_distinguishes_images_with_identical_band_means():
    rng = np.random.default_rng(920)
    yy, xx = np.mgrid[:120, :120]
    base = 110 + 12 * np.sin(yy / 10) + 12 * np.sin(xx / 9)
    # Each 2x2 texture has zero row/column sums. All six averaged bands
    # match exactly even though the actual two-dimensional texture is inverted.
    texture = np.kron(rng.choice([-1, 1], (60, 60)), np.array([[45, -45], [-45, 45]]))
    first, wrong = base + texture, base - texture
    valid = np.ones(first.shape, bool)
    exposure = first.astype(np.float32) * 1.2 + 13
    assert full_overlap_score(first, exposure, valid) > .999
    assert all(score > .999 for _, _, _, score in band_profiles(first, wrong, valid))
    assert full_overlap_score(first, wrong, valid) < .2


def test_whole_area_registration_resolves_periodic_layers_with_local_texture(tmp_path):
    yy, xx = np.mgrid[:260, :420]
    scene = 110 + 30 * np.sin(2 * np.pi * yy / 20) + 10 * np.sin(xx / 15)
    scene += np.random.default_rng(808).normal(0, 12, scene.shape)
    scene = np.clip(scene, 0, 255).astype(np.uint8)
    first, second = tmp_path / "area-a.png", tmp_path / "area-b.png"
    _write(first, scene[:180])
    _write(second, np.uint8(scene[80:].astype(np.float32) * .8 + 15))
    result = stitch_paths([str(first), str(second)], StitchOptions(preserve_scale_bar=False))
    assert result.output_size == (420, 260)
    assert result.placements[1].transform[1, 2] == pytest.approx(80, abs=1)


def test_tiny_perfect_overlap_is_not_a_translation_candidate():
    scene = np.random.default_rng(391).integers(0, 255, (120, 380), dtype=np.uint8)
    first, second = scene[:, :200], scene[:, 180:]
    mask = np.full(first.shape, 255, np.uint8)
    peaks = translation_candidates(first, second, mask, mask)
    assert not any(abs(dx + 180) <= 2 and abs(dy) <= 2 for _, dx, dy in peaks)


def test_truly_periodic_whole_area_merges_with_warning(tmp_path):
    tile = np.random.default_rng(18).integers(30, 200, (20, 30), dtype=np.uint8)
    repeated = np.tile(tile, (8, 8))
    first, second = tmp_path / "repeat-a.png", tmp_path / "repeat-b.png"
    _write(first, repeated)
    _write(second, np.roll(repeated, 7, axis=0))
    result = stitch_paths([str(first), str(second)])
    assert result.confidence > .9999
    assert result.warnings


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


def test_identical_images_choose_full_overlay_with_warning_despite_board_hint(tmp_path):
    image = np.random.default_rng(91).integers(0, 256, (220, 300), dtype=np.uint8)
    a, b = tmp_path / "same-a.png", tmp_path / "same-b.png"
    _write(a, image)
    _write(b, image)
    hints = [
        StitchLayoutHint(str(a), np.eye(3)),
        StitchLayoutHint(str(b), np.array([[1.0, 0.0, 270.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])),
    ]

    result = stitch_paths([str(a), str(b)], layout_hints=hints)
    assert result.output_size == (300, 220)
    assert result.confidence == pytest.approx(1.0)
    assert result.warnings


def test_nearly_identical_uint16_repetition_uses_sub_8bit_differences(tmp_path):
    rng = np.random.default_rng(510)
    tile = rng.integers(5000, 55000, (20, 30), dtype=np.uint16)
    scene = np.tile(tile, (14, 8)) + rng.integers(0, 5, (280, 240), dtype=np.uint16)
    a, b = tmp_path / "precise-a.tif", tmp_path / "precise-b.tif"
    _write(a, scene[:200])
    _write(b, scene[80:])
    result = stitch_paths([str(a), str(b)], StitchOptions(preserve_scale_bar=False))
    assert result.output_size == (240, 280)
    assert result.placements[1].transform[1, 2] == 80
    assert result.confidence > .9999
    assert result.warnings  # Near ties warn but do not override the best score.
    assert np.array_equal(result.image, scene)


def test_board_saves_precise_score_and_shows_nonblocking_warning(tmp_path):
    import json
    from angle_cal.stitching import StitchResult, StitchPlacement
    app = QApplication.instance() or QApplication([])
    source = tmp_path / "input.png"
    image = np.arange(400, dtype=np.uint8).reshape(20, 20)
    _write(source, image)
    result = StitchResult(image, np.full(image.shape, 255, np.uint8),
                          [StitchPlacement(str(source), np.eye(3), "anchor", 0, 0)],
                          (20, 20), .99994, warnings=["Near tied candidates"])
    board = PhotoMergeBoard()
    received = []
    board.result_ready.connect(received.append)
    board._on_finished(result)
    app.processEvents()
    assert received == [result]
    assert "99.99%" in board.status.text()
    saved = Path(result.saved_path)
    assert saved.exists() and "99.99pct" in saved.name
    report = json.loads(saved.with_suffix(".stitch.json").read_text(encoding="utf-8"))
    assert report["confidence"] == .99994
    assert report["warnings"] == result.warnings
    assert board.alignment_warning.isVisible()
    assert not board.alignment_warning.isModal()
    board.alignment_warning.close()
    board.close()


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


def test_multiple_addons_open_menu_tabs_and_preserve_board():
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    try:
        assert window.ribbon_tabs.count() == 5
        window.addon_actions["photo_merge"].setChecked(True)
        window.addon_actions["trench_analyzer"].setChecked(True)
        app.processEvents()

        assert window.ribbon_tabs.count() == 7
        assert window.addon_actions["photo_merge"].isChecked()
        assert window.addon_actions["trench_analyzer"].isChecked()
        assert window.ribbon_tabs.tabText(5) == "사진 합치기"
        assert window.ribbon_tabs.tabText(6) == "Trench 자동분석기"
        board = window.photo_merge_board
        assert board.window() is window
        window.ribbon_tabs.setCurrentIndex(5)
        assert window.workspace_stack.currentWidget() is board
        window.ribbon_tabs.setCurrentIndex(0)
        assert window.workspace_stack.currentWidget() is window.canvas
        window.ribbon_tabs.setCurrentIndex(5)
        assert window.workspace_stack.currentWidget() is board

        window.addon_actions["photo_merge"].setChecked(False)
        app.processEvents()
        assert window.ribbon_tabs.count() == 6
        assert window.workspace_stack.currentWidget() is window.canvas
        assert window.ribbon_tabs.tabText(5) == "Trench 자동분석기"
        window.addon_actions["photo_merge"].setChecked(True)
        assert window.ribbon_tabs.count() == 7
        assert window.photo_merge_board is board
        assert window.workspace_stack.currentWidget() is board
        window.addon_actions["photo_merge"].setChecked(False)
        window.addon_actions["trench_analyzer"].setChecked(False)
        assert window.ribbon_tabs.count() == 5
    finally:
        window.close()


@pytest.mark.parametrize("focus_target", ["view", "crop_button", "band_scroll", "ribbon_tabs"])
def test_delete_key_in_main_window_removes_only_board_selection(tmp_path, monkeypatch, focus_target):
    app = QApplication.instance() or QApplication([])
    path = tmp_path / "board-delete.png"
    _write(path, np.full((80, 120), 100, dtype=np.uint8))
    window = MainWindow()
    canvas_calls = []
    monkeypatch.setattr(window.canvas, "selected_line_ids", lambda: canvas_calls.append(True) or [])
    try:
        window.addon_actions["photo_merge"].setChecked(True)
        board = window.photo_merge_board
        board.add_paths([str(path)])
        item = board.view.items_in_board()[0]
        item.setSelected(True)
        item.set_match_rect((.1, .1, .8, .8))
        window.show()
        window.activateWindow()
        app.processEvents()
        target = window.ribbon_tabs if focus_target == "ribbon_tabs" else getattr(board, focus_target)
        target.setFocus()
        app.processEvents()
        QTest.keyClick(target, Qt.Key.Key_Delete)
        app.processEvents()
        assert not board.view.items_in_board()
        assert board.count_label.text() == "보드 이미지 0장"
        assert not canvas_calls
        assert path.exists()
        # Returning to the analysis tab restores its original Delete action.
        window.ribbon_tabs.setCurrentIndex(0)
        window.delete_action.trigger()
        assert canvas_calls
    finally:
        window.close()
        app.processEvents()


def test_blank_board_click_finishes_crop_and_keeps_keyboard_selection(tmp_path):
    app = QApplication.instance() or QApplication([])
    path = tmp_path / "crop-finish.png"
    _write(path, np.full((100, 150), 100, dtype=np.uint8))
    window = MainWindow()
    try:
        window.addon_actions["photo_merge"].setChecked(True)
        board = window.photo_merge_board
        board.add_paths([str(path)])
        item = board.view.items_in_board()[0]
        item.setPos(0, 0)
        window.show()
        window.activateWindow()
        app.processEvents()
        board.view.centerOn(item.sceneBoundingRect().center())
        center = board.view.mapFromScene(item.sceneBoundingRect().center())
        QTest.mouseClick(board.view.viewport(), Qt.MouseButton.LeftButton, pos=center)
        assert item.isSelected()
        QTest.mouseClick(board.crop_button, Qt.MouseButton.LeftButton)
        assert board.view.crop_mode
        assert board.view.hasFocus()
        item.set_match_rect((.2, .2, .6, .6))
        # Clicking the image itself must not finish editing.
        QTest.mouseClick(board.view.viewport(), Qt.MouseButton.LeftButton, pos=center)
        assert board.crop_button.isChecked()
        blank = board.view.mapFromScene(QPointF(-30, -30))
        assert board.view.itemAt(blank) is None
        QTest.mouseClick(board.view.viewport(), Qt.MouseButton.LeftButton, pos=blank)
        app.processEvents()
        assert not board.crop_button.isChecked()
        assert not board.view.crop_mode
        assert not item.crop_overlay.isVisible()
        assert item.isSelected()
        assert item.pixmap().width() < item.original_pixmap.width()
        QTest.keyClick(board.view, Qt.Key.Key_Right)
        assert item.pos() == QPointF(10, 0)
        assert item.match_rect == (.2, .2, .6, .6)
    finally:
        window.close()
        app.processEvents()


def test_board_arrow_keys_move_selected_group_with_ctrl_fine_steps(tmp_path):
    app = QApplication.instance() or QApplication([])
    paths = [tmp_path / f"move-{index}.png" for index in range(3)]
    for path in paths:
        _write(path, np.full((100, 150), 100, dtype=np.uint8))
    window = MainWindow()
    try:
        window.addon_actions["photo_merge"].setChecked(True)
        board = window.photo_merge_board
        board.add_paths([str(path) for path in paths])
        items = sorted(board.view.items_in_board(), key=lambda item: item.path)
        items[0].setSelected(True)
        items[1].setSelected(True)
        items[0].set_match_rect((.1, .1, .8, .8))
        window.show()
        window.activateWindow()
        app.processEvents()
        board.view.setFocus()
        board.view.scale(2, 2)
        unchanged = items[2].pos()
        for modifiers, step in ((Qt.KeyboardModifier.NoModifier, 10),
                                (Qt.KeyboardModifier.ControlModifier, 1)):
            for key, dx, dy in ((Qt.Key.Key_Left, -1, 0), (Qt.Key.Key_Right, 1, 0),
                               (Qt.Key.Key_Up, 0, -1), (Qt.Key.Key_Down, 0, 1)):
                before = [item.pos() for item in items[:2]]
                QTest.keyClick(board.view, key, modifiers)
                for item, position in zip(items[:2], before):
                    assert item.pos() == position + QPointF(dx * step, dy * step)
                assert items[2].pos() == unchanged
                assert board.view.profile_timer.isActive()
        assert items[0].match_rect == (.1, .1, .8, .8)
    finally:
        window.close()
        app.processEvents()


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
        assert window.ribbon_tabs.currentWidget() is window.addon_pages["photo_merge"]
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
        full = QRectF(first.original_pixmap.rect())
        visible = QRectF(full.width() * .1, 0, full.width() * .8, full.height() * .8)
        assert first.boundingRect() == visible.adjusted(-.5, -.5, .5, .5)
        image = first.pixmap().toImage()
        assert image.width() == int(visible.width())
        assert image.height() == int(visible.height())
        assert image.pixelColor(0, image.height() // 2).alpha() == 255
        assert image.pixelColor(image.width() // 2, image.height() // 2).alpha() == 255
        assert not first.contains(QPointF(1, full.height() / 2))
        transform = board.view.layout_hints()[0].source_to_board.copy()
        position = first.pos()

        # Render the scene: excluded pixels must show the background, including
        # while the cropped item is selected (no original-size selection box).
        second.setVisible(False)
        rendered = QImage(int(full.width()), int(full.height()), QImage.Format.Format_ARGB32)
        rendered.fill(Qt.GlobalColor.magenta)
        painter = QPainter(rendered)
        board.view.scene().render(painter, full, first.original_scene_rect())
        painter.end()
        assert rendered.pixelColor(1, rendered.height() // 2).name() == "#ff00ff"
        assert rendered.pixelColor(rendered.width() // 2, rendered.height() // 2).name() == "#464646"

        board.crop_button.click()
        assert first.boundingRect() == full.adjusted(-.5, -.5, .5, .5)
        assert first.crop_overlay.crop_rect() == visible
        assert first.pos() == position
        assert np.array_equal(board.view.layout_hints()[0].source_to_board, transform)
        board.crop_button.click()
        assert first.boundingRect() == visible.adjusted(-.5, -.5, .5, .5)

        first.setSelected(True)
        board.crop_reset_button.click()
        assert first.match_rect is None
        assert second.match_rect is None
        assert first.boundingRect() == full.adjusted(-.5, -.5, .5, .5)
        assert first.offset() == QPointF()
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
        for entry in board.band_graph.profiles:
            assert entry["first_strip_image"].width() == len(entry["first"])
            assert entry["second_strip_image"].width() == len(entry["second"])
            assert not entry["first_image"].isNull()
            x, y, width, height = entry["first_region"]
            assert x >= 0 and y >= 0
            assert x + width <= entry["first_image"].width()
            assert y + height <= entry["first_image"].height()

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
