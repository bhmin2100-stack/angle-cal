import os
import json
import math
from pathlib import Path
import zipfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QKeyEvent, QKeySequence, QMouseEvent, QWheelEvent
from PySide6.QtWidgets import QApplication, QDialog, QGraphicsItem, QGraphicsPathItem, QGraphicsTextItem, QGroupBox, QLabel, QMessageBox, QPushButton

import angle_cal.app as app_module
from angle_cal.app import DataExportOptions, LineRecord, MainWindow, ScalePreset, StructureTemplate, record_points, structure_template_from_dict, structure_template_to_dict
from angle_cal.image_ops import segment_brightness_profile, snap_polyline_to_gradient


def _app():
    return QApplication.instance() or QApplication([])


def _window_with_edge_image():
    _app()
    window = MainWindow()
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    image[:, 82:] = 255
    window.image_bgr = image
    window._show_image()
    return window


def _select_edges(window: MainWindow) -> list[LineRecord]:
    window.canvas.redraw_lines(list(window.records.values()))
    edges = [record for record in window.records.values() if record.kind == "edge"]
    for edge in edges:
        window.canvas.line_items[edge.id].setSelected(True)
    return edges


def test_ctrl_s_shortcut_saves_project():
    window = _window_with_edge_image()
    try:
        shortcut = window.smart_save_action.shortcut()

        assert shortcut.matches(QKeySequence(QKeySequence.StandardKey.Save)) == QKeySequence.SequenceMatch.ExactMatch
        assert window.smart_save_action.shortcutContext() == Qt.ShortcutContext.ApplicationShortcut
        assert window.smart_save_action in window.actions()
        assert window.save_project_action.shortcut().isEmpty()
    finally:
        window.close()


def test_tooltips_use_readable_contrast():
    window = _window_with_edge_image()
    try:
        stylesheet = QApplication.instance().styleSheet()

        assert "QToolTip" in stylesheet
        assert "color: #f8fafc" in stylesheet
        assert "background-color: #111827" in stylesheet
    finally:
        window.close()


def test_recognize_segments_line_mode_by_segment_size():
    window = _window_with_edge_image()
    try:
        window.curve_sensitivity_spin.setValue(10)
        window._create_edge_line((70.0, 20.0), (70.0, 100.0))
        _select_edges(window)

        window.recognize_edges()

        edge = next(record for record in window.records.values() if record.kind == "edge")
        assert edge.edge_mode == "line"
        assert edge.points is not None
        assert len(edge.points) > 2
        assert edge.edge_segmented is True
        assert 80 <= (edge.start[0] + edge.end[0]) / 2 <= 83
    finally:
        window.close()


def test_repeated_recognition_uses_stable_original_edge_path():
    window = _window_with_edge_image()
    try:
        window.curve_sensitivity_spin.setValue(10)
        window._create_edge_line((70.0, 20.0), (70.0, 100.0))
        _select_edges(window)

        window.recognize_edges()
        edge = next(record for record in window.records.values() if record.kind == "edge")
        first_points = list(edge.points)

        window.recognize_edges()
        second_points = list(edge.points)

        assert first_points == second_points
        assert edge.recognition_points == [(70.0, 20.0), (70.0, 100.0)]
    finally:
        window.close()


def test_moving_segmented_edge_shifts_recognition_path():
    window = _window_with_edge_image()
    try:
        window.curve_sensitivity_spin.setValue(10)
        window._create_edge_line((70.0, 20.0), (70.0, 100.0))
        _select_edges(window)
        window.recognize_edges()
        edge = next(record for record in window.records.values() if record.kind == "edge")
        assert edge.edge_segmented is True
        original_recognition_points = list(edge.recognition_points)

        item = window.canvas.line_items[edge.id]
        item.moveBy(24.0, 3.0)
        window._sync_records_from_canvas()

        assert edge.recognition_points == [(point[0] + 24.0, point[1] + 3.0) for point in original_recognition_points]
    finally:
        window.close()


def test_connected_line_edges_recognize_independently(monkeypatch):
    calls: list[list[tuple[float, float]]] = []

    class Result:
        def __init__(self, points):
            self.points = points

    def fake_snap(_gray, points, *args):
        source_points = [tuple(point) for point in points]
        calls.append(source_points)
        return Result([(point[0] + 1.0, point[1]) for point in source_points])

    monkeypatch.setattr(app_module, "snap_polyline_to_gradient", fake_snap)
    window = _window_with_edge_image()
    try:
        window._create_edge_line((70.0, 20.0), (70.0, 60.0))
        window._create_edge_line((70.0, 60.0), (70.0, 100.0))
        _select_edges(window)

        window.recognize_edges()

        edges = [record for record in window.records.values() if record.kind == "edge"]
        assert len(edges) == 2
        assert calls == [
            [(70.0, 20.0), (70.0, 60.0)],
            [(70.0, 60.0), (70.0, 100.0)],
        ]
        assert edges[0].edge_mode == "line"
        assert edges[1].edge_mode == "line"
        assert edges[0].end == (71.0, 60.0)
        assert edges[1].start == (71.0, 60.0)
    finally:
        window.close()


def test_recognize_only_selected_edges():
    window = _window_with_edge_image()
    try:
        window._create_edge_line((70.0, 20.0), (70.0, 100.0))
        window._create_edge_line((45.0, 20.0), (45.0, 100.0))
        window.canvas.redraw_lines(list(window.records.values()))
        edges = [record for record in window.records.values() if record.kind == "edge"]
        window.canvas.line_items[edges[0].id].setSelected(True)

        window.recognize_edges()

        assert 80 <= window.records[edges[0].id].start[0] <= 83
        assert window.records[edges[1].id].start == (45.0, 20.0)
    finally:
        window.close()


def test_recognize_all_edges_when_no_edge_is_selected():
    window = _window_with_edge_image()
    try:
        window._create_edge_line((70.0, 20.0), (70.0, 100.0))
        window._create_edge_line((45.0, 20.0), (45.0, 100.0))
        window.canvas.redraw_lines(list(window.records.values()))

        window.recognize_edges()

        edges = [record for record in window.records.values() if record.kind == "edge"]
        assert all(80 <= edge.start[0] <= 83 for edge in edges)
    finally:
        window.close()


def test_enter_recognizes_selected_edge_only():
    window = _window_with_edge_image()
    try:
        window._create_edge_line((70.0, 20.0), (70.0, 100.0))
        window._create_edge_line((45.0, 20.0), (45.0, 100.0))
        window.canvas.redraw_lines(list(window.records.values()))
        edges = [record for record in window.records.values() if record.kind == "edge"]
        window.canvas.line_items[edges[0].id].setSelected(True)

        press = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Return, Qt.KeyboardModifier.NoModifier)
        window.canvas.keyPressEvent(press)

        assert 80 <= window.records[edges[0].id].start[0] <= 83
        assert window.records[edges[1].id].start == (45.0, 20.0)
    finally:
        window.close()


def test_enter_without_selection_recognizes_all_edges():
    window = _window_with_edge_image()
    try:
        window._create_edge_line((70.0, 20.0), (70.0, 100.0))
        window._create_edge_line((75.0, 20.0), (75.0, 100.0))
        window.canvas.redraw_lines(list(window.records.values()))
        edges = [record for record in window.records.values() if record.kind == "edge"]

        press = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Return, Qt.KeyboardModifier.NoModifier)
        window.canvas.keyPressEvent(press)

        assert all(80 <= window.records[edge.id].start[0] <= 83 for edge in edges)
    finally:
        window.close()


def test_recognize_all_edges_over_ten_requires_confirmation(monkeypatch):
    calls: list[str] = []

    def _question(*args, **kwargs):
        calls.append("question")
        return QMessageBox.StandardButton.No

    monkeypatch.setattr(app_module.QMessageBox, "question", _question)
    window = _window_with_edge_image()
    try:
        for idx in range(10):
            window._create_edge_line((30.0 + idx, 20.0), (30.0 + idx, 100.0))
        window.canvas.redraw_lines(list(window.records.values()))

        window.recognize_edges()

        assert calls == ["question"]
        edges = [record for record in window.records.values() if record.kind == "edge"]
        assert edges[0].start == (30.0, 20.0)
    finally:
        window.close()


def test_selected_edge_detection_controls_apply_per_edge():
    window = _window_with_edge_image()
    try:
        window._create_edge_line((70.0, 20.0), (70.0, 100.0))
        window._create_edge_line((45.0, 20.0), (45.0, 100.0))
        window.canvas.redraw_lines(list(window.records.values()))
        edges = [record for record in window.records.values() if record.kind == "edge"]
        window.canvas.line_items[edges[0].id].setSelected(True)

        window.search_radius_spin.setValue(22)
        window.curve_sensitivity_spin.setValue(7)

        assert window.records[edges[0].id].search_radius_px == 22
        assert window.records[edges[0].id].segment_size_px == 7
        assert window.records[edges[1].id].search_radius_px == 35
        assert window.records[edges[1].id].segment_size_px == 9
    finally:
        window.close()


def test_split_search_range_controls_apply_per_selected_edge():
    window = _window_with_edge_image()
    try:
        window._create_edge_line((70.0, 20.0), (70.0, 100.0))
        window._create_edge_line((45.0, 20.0), (45.0, 100.0))
        window.canvas.redraw_lines(list(window.records.values()))
        edges = [record for record in window.records.values() if record.kind == "edge"]
        window.canvas.line_items[edges[0].id].setSelected(True)

        window.split_search_range_checkbox.setChecked(True)
        window.search_radius_left_spin.setValue(12)
        window.search_radius_right_spin.setValue(28)

        selected = window.records[edges[0].id]
        unselected = window.records[edges[1].id]
        assert selected.search_radius_split is True
        assert selected.search_radius_left_px == 12
        assert selected.search_radius_right_px == 28
        assert unselected.search_radius_split is False
    finally:
        window.close()


def test_split_search_range_wheel_adjusts_left_and_shift_adjusts_right():
    window = _window_with_edge_image()
    try:
        window._create_edge_line((70.0, 20.0), (70.0, 100.0))
        window.canvas.redraw_lines(list(window.records.values()))
        edge = next(record for record in window.records.values() if record.kind == "edge")
        window.canvas.line_items[edge.id].setSelected(True)
        window.split_search_range_checkbox.setChecked(True)

        window.adjust_search_range_by_wheel(1, "left")
        window.adjust_search_range_by_wheel(2, "right")

        assert window.records[edge.id].search_radius_left_px == 36
        assert window.records[edge.id].search_radius_right_px == 37
    finally:
        window.close()


def test_segment_size_change_preserves_mixed_split_search_ranges():
    window = _window_with_edge_image()
    try:
        window._create_edge_line((70.0, 20.0), (70.0, 100.0))
        window._create_edge_line((45.0, 20.0), (45.0, 100.0))
        window.canvas.redraw_lines(list(window.records.values()))
        edges = [record for record in window.records.values() if record.kind == "edge"]
        edges[0].search_radius_split = True
        edges[0].search_radius_left_px = 12
        edges[0].search_radius_right_px = 28
        edges[1].search_radius_split = True
        edges[1].search_radius_left_px = 20
        edges[1].search_radius_right_px = 44
        for edge in edges:
            window.canvas.line_items[edge.id].setSelected(True)

        window.curve_sensitivity_spin.setValue(13)

        assert window.records[edges[0].id].search_radius_left_px == 12
        assert window.records[edges[0].id].search_radius_right_px == 28
        assert window.records[edges[1].id].search_radius_left_px == 20
        assert window.records[edges[1].id].search_radius_right_px == 44
        assert all(window.records[edge.id].segment_size_px == 13 for edge in edges)
    finally:
        window.close()


def test_image_adjustment_updates_background_without_dropping_annotations():
    window = _window_with_edge_image()
    try:
        window._create_edge_line((70.0, 20.0), (70.0, 100.0))
        window.canvas.redraw_lines(list(window.records.values()))
        edge = next(record for record in window.records.values() if record.kind == "edge")

        window.image_brightness_spin.setValue(45)

        assert window.image_brightness == 45
        assert edge.id in window.canvas.line_items
        assert window.canvas.line_items[edge.id].scene() is window.canvas.scene
        assert window.canvas.pixmap_item is not None
    finally:
        window.close()


def test_recognition_uses_adjusted_image(monkeypatch):
    captured: dict[str, float] = {}

    def fake_snap(gray, *args, **kwargs):
        captured["mean"] = float(gray.mean())
        return None

    monkeypatch.setattr(app_module, "snap_polyline_to_gradient", fake_snap)
    window = _window_with_edge_image()
    try:
        original_mean = float(app_module.to_gray(window.image_bgr).mean())
        window._create_edge_line((70.0, 20.0), (70.0, 100.0))
        window.image_brightness_spin.setValue(35)

        window.recognize_edges()

        assert captured["mean"] > original_mean
    finally:
        window.close()


def test_recognition_passes_selected_boundary_snap_mode(monkeypatch):
    captured: dict[str, object] = {}

    class Result:
        points = [(81.0, 20.0), (81.0, 100.0)]

    def fake_snap(_gray, points, radius, segment_size, left_radius, right_radius, boundary_mode):
        captured["boundary_mode"] = boundary_mode
        return Result()

    monkeypatch.setattr(app_module, "snap_polyline_to_gradient", fake_snap)
    window = _window_with_edge_image()
    try:
        window._create_edge_line((70.0, 20.0), (70.0, 100.0))
        edge = next(record for record in window.records.values() if record.kind == "edge")
        window.canvas.redraw_lines(list(window.records.values()))
        window.canvas.line_items[edge.id].setSelected(True)
        window.boundary_snap_combo.setCurrentIndex(window.boundary_snap_combo.findData("darkest"))

        window.recognize_edges()

        assert edge.boundary_snap_mode == "darkest"
        assert captured["boundary_mode"] == "darkest"
    finally:
        window.close()


def test_boundary_offset_uses_image_axis_direction():
    vertical_top_down = MainWindow._apply_boundary_offset([(70.0, 20.0), (70.0, 100.0)], 5)
    vertical_bottom_up = MainWindow._apply_boundary_offset([(70.0, 100.0), (70.0, 20.0)], 5)
    horizontal_left_right = MainWindow._apply_boundary_offset([(20.0, 60.0), (120.0, 60.0)], 5)
    horizontal_right_left = MainWindow._apply_boundary_offset([(120.0, 60.0), (20.0, 60.0)], 5)
    vertical_negative = MainWindow._apply_boundary_offset([(70.0, 20.0), (70.0, 100.0)], -5)
    horizontal_negative = MainWindow._apply_boundary_offset([(20.0, 60.0), (120.0, 60.0)], -5)

    assert vertical_top_down == [(75.0, 20.0), (75.0, 100.0)]
    assert vertical_bottom_up == [(75.0, 100.0), (75.0, 20.0)]
    assert horizontal_left_right == [(20.0, 55.0), (120.0, 55.0)]
    assert horizontal_right_left == [(120.0, 55.0), (20.0, 55.0)]
    assert vertical_negative == [(65.0, 20.0), (65.0, 100.0)]
    assert horizontal_negative == [(20.0, 65.0), (120.0, 65.0)]


def test_recognition_applies_boundary_offset_after_snap(monkeypatch):
    class Result:
        points = [(81.0, 20.0), (81.0, 100.0)]

    def fake_snap(_gray, _points, _radius, _segment_size, _left_radius, _right_radius, _boundary_mode):
        return Result()

    monkeypatch.setattr(app_module, "snap_polyline_to_gradient", fake_snap)
    window = _window_with_edge_image()
    try:
        window._create_edge_line((70.0, 20.0), (70.0, 100.0))
        edge = next(record for record in window.records.values() if record.kind == "edge")
        window.canvas.redraw_lines(list(window.records.values()))
        window.canvas.line_items[edge.id].setSelected(True)
        window.boundary_offset_spin.setValue(4)

        window.recognize_edges()

        assert edge.boundary_offset_px == 4
        assert edge.start == (85.0, 20.0)
        assert edge.end == (85.0, 100.0)
    finally:
        window.close()


def test_segment_size_bulk_change_does_not_overwrite_boundary_snap_modes():
    window = _window_with_edge_image()
    try:
        window._create_edge_line((70.0, 20.0), (70.0, 100.0))
        window._create_edge_line((45.0, 20.0), (45.0, 100.0))
        window.canvas.redraw_lines(list(window.records.values()))
        edges = [record for record in window.records.values() if record.kind == "edge"]
        edges[0].boundary_snap_mode = "brightest"
        edges[1].boundary_snap_mode = "darkest"
        edges[0].boundary_offset_px = 2
        edges[1].boundary_offset_px = -3
        for edge in edges:
            window.canvas.line_items[edge.id].setSelected(True)

        window.curve_sensitivity_spin.setValue(17)
        window._edge_detection_settings_changed(force_all=True)

        assert [edge.segment_size_px for edge in edges] == [17, 17]
        assert [edge.boundary_snap_mode for edge in edges] == ["brightest", "darkest"]
        assert [edge.boundary_offset_px for edge in edges] == [2, -3]
    finally:
        window.close()


def test_mouse_wheel_adjusts_search_range_for_selected_edge():
    window = _window_with_edge_image()
    try:
        window._create_edge_line((70.0, 20.0), (70.0, 100.0))
        window._create_edge_line((45.0, 20.0), (45.0, 100.0))
        window.canvas.redraw_lines(list(window.records.values()))
        edges = [record for record in window.records.values() if record.kind == "edge"]
        window.canvas.line_items[edges[0].id].setSelected(True)
        assert window.records[edges[0].id].search_radius_px == 35

        event = QWheelEvent(
            QPointF(10.0, 10.0),
            QPointF(10.0, 10.0),
            QPoint(0, 0),
            QPoint(0, 120),
            Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.NoModifier,
            Qt.ScrollPhase.NoScrollPhase,
            False,
        )
        window.canvas.wheelEvent(event)

        assert window.records[edges[0].id].search_radius_px == 36
        assert window.records[edges[1].id].search_radius_px == 35
    finally:
        window.close()


def test_mouse_drag_does_not_adjust_search_range():
    window = _window_with_edge_image()
    try:
        window._create_edge_line((70.0, 20.0), (70.0, 100.0))
        window.canvas.redraw_lines(list(window.records.values()))
        edge = next(record for record in window.records.values() if record.kind == "edge")
        window.canvas.line_items[edge.id].setSelected(True)
        window.split_search_range_checkbox.setChecked(True)
        view_pos = window.canvas.mapFromScene(QPointF(90.0, 60.0))

        assert window.canvas._search_range_drag_candidate(view_pos) is not None
        assert not window.canvas._begin_search_range_drag(view_pos, Qt.KeyboardModifier.NoModifier)
        assert window.canvas._search_range_drag_segment is None
    finally:
        window.close()


def test_grouped_split_search_range_has_no_drag_candidate():
    window = _window_with_edge_image()
    try:
        window._create_edge_line((70.0, 20.0), (70.0, 100.0))
        window._create_edge_line((45.0, 20.0), (45.0, 100.0))
        edges = [record for record in window.records.values() if record.kind == "edge"]
        for edge in edges:
            edge.object_group = "G1"
            edge.search_radius_split = True
            edge.search_radius_left_px = 12
            edge.search_radius_right_px = 28
        window.split_search_range_checkbox.setChecked(True)
        window.canvas.redraw_lines(list(window.records.values()))
        for edge in edges:
            window.canvas.line_items[edge.id].setSelected(True)
        view_pos = window.canvas.mapFromScene(QPointF(90.0, 60.0))

        assert window.canvas._search_range_drag_candidate(view_pos) is None
        assert not window.canvas._begin_search_range_drag(view_pos, Qt.KeyboardModifier.NoModifier)
    finally:
        window.close()


def test_mouse_wheel_adjusts_last_edge_and_next_edge_default_when_none_selected():
    window = _window_with_edge_image()
    try:
        window._create_edge_line((70.0, 20.0), (70.0, 100.0))
        window._create_edge_line((45.0, 20.0), (45.0, 100.0))
        edges = [record for record in window.records.values() if record.kind == "edge"]
        assert edges[0].search_radius_px == 35
        assert edges[1].search_radius_px == 35

        event = QWheelEvent(
            QPointF(10.0, 10.0),
            QPointF(10.0, 10.0),
            QPoint(0, 0),
            QPoint(0, 120),
            Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.NoModifier,
            Qt.ScrollPhase.NoScrollPhase,
            False,
        )
        window.canvas.wheelEvent(event)

        assert window.records[edges[0].id].search_radius_px == 35
        assert window.records[edges[1].id].search_radius_px == 36
        assert window.search_radius_spin.value() == 36

        window._create_edge_line((30.0, 20.0), (30.0, 100.0))
        new_edge = [record for record in window.records.values() if record.kind == "edge"][-1]
        assert new_edge.search_radius_px == 36
    finally:
        window.close()


def test_edge_shape_control_is_removed_and_edges_stay_straight():
    window = _window_with_edge_image()
    try:
        assert not hasattr(window, "edge_mode_combo")

        window._create_edge_line((70.0, 20.0), (70.0, 100.0), [(70.0, 20.0), (70.0, 60.0), (70.0, 100.0)])

        edge = next(record for record in window.records.values() if record.kind == "edge")
        assert edge.edge_mode == "line"
        assert edge.edge_segmented is False
        assert edge.points is None
        assert edge.recognition_points == [(70.0, 20.0), (70.0, 100.0)]
    finally:
        window.close()


def test_segmented_edge_does_not_show_reference_angles_after_recognition():
    window = _window_with_edge_image()
    try:
        window._create_reference_line((10.0, 10.0), (100.0, 10.0))
        window._create_edge_line((70.0, 20.0), (70.0, 100.0))
        _select_edges(window)
        window.recognize_edges()

        window.calculate_angles()

        assert len(window.canvas.angle_items) == 0
        assert not [row for row in window.last_measurements if row["kind"] == "edge_to_reference"]
        assert not [row for row in window.last_measurements if row["kind"] == "edge_segment_to_reference"]
        assert not [row for row in window.last_measurements if row["kind"] == "edge_guide_intersection"]
    finally:
        window.close()


def test_segmented_edge_shows_only_guide_intersection_angles():
    window = _window_with_edge_image()
    try:
        window._create_reference_line((10.0, 10.0), (100.0, 10.0))
        window._create_edge_line((70.0, 20.0), (70.0, 100.0))
        window.records["G99"] = LineRecord(
            id="G99",
            kind="guide",
            start=(10.0, 50.0),
            end=(140.0, 50.0),
            label="guide",
            axis="horizontal",
        )
        window.canvas.redraw_lines(list(window.records.values()))
        _select_edges(window)
        window.recognize_edges()

        window.calculate_angles()

        assert not [row for row in window.last_measurements if row["kind"] == "edge_segment_to_reference"]
        guide_rows = [row for row in window.last_measurements if row["kind"] == "edge_guide_intersection"]
        assert len(guide_rows) == 1
        assert len(window.canvas.angle_items) >= 2
    finally:
        window.close()


def test_straight_edge_still_shows_single_reference_angle():
    window = _window_with_edge_image()
    try:
        window._create_reference_line((10.0, 10.0), (100.0, 10.0))
        window._create_edge_line((70.0, 20.0), (70.0, 100.0))
        window.canvas.redraw_lines(list(window.records.values()))

        window.calculate_angles()

        assert len(window.canvas.angle_items) == 1
        assert len([row for row in window.last_measurements if row["kind"] == "edge_to_reference"]) == 1
    finally:
        window.close()


def test_deleted_angle_annotation_stays_hidden_until_manual_recalculate():
    window = _window_with_edge_image()
    try:
        window._create_reference_line((10.0, 10.0), (100.0, 10.0))
        window._create_edge_line((70.0, 20.0), (70.0, 100.0))
        window.canvas.redraw_lines(list(window.records.values()))
        window.set_visibility("line_angle", True)

        window.calculate_angles(reset_hidden=True)
        assert len(window.canvas.angle_items) == 1

        window.canvas.angle_items[0].setSelected(True)
        window.delete_selected()
        assert len(window.canvas.angle_items) == 0
        assert window.hidden_angle_measurements

        window._create_edge_line((90.0, 20.0), (90.0, 100.0))
        added_edge_id = max(record.id for record in window.records.values() if record.kind == "edge")
        window.canvas.redraw_lines(list(window.records.values()))
        window.canvas.line_items[added_edge_id].setSelected(True)
        window.delete_selected()

        assert len(window.canvas.angle_items) == 0
        assert len([row for row in window.last_measurements if row["kind"] == "edge_to_reference"]) == 1

        window.calculate_angles(reset_hidden=True)
        assert len(window.canvas.angle_items) == 1
    finally:
        window.close()


def test_search_range_visibility_only_draws_band_without_number_label():
    window = _window_with_edge_image()
    try:
        window._create_edge_line((70.0, 20.0), (70.0, 100.0))
        window.canvas.redraw_lines(list(window.records.values()))

        window.set_visibility("range", False)
        assert len(window.canvas.search_range_band_items) == 0
        assert len(window.canvas.search_range_label_items) == 0

        window.set_visibility("range", True)
        assert len(window.canvas.search_range_band_items) == 1
        assert len(window.canvas.search_range_label_items) == 0
    finally:
        window.close()


def test_search_range_overlay_gets_more_transparent_when_zoomed():
    window = _window_with_edge_image()
    try:
        window._create_edge_line((70.0, 20.0), (70.0, 100.0))
        window.canvas.redraw_lines(list(window.records.values()))
        window._update_search_range_overlay()
        normal_alpha = window.canvas.search_range_band_items[0].brush().color().alpha()

        window.canvas.scale(4.0, 4.0)
        window._update_search_range_overlay()
        zoomed_alpha = window.canvas.search_range_band_items[0].brush().color().alpha()

        assert zoomed_alpha < normal_alpha
    finally:
        window.close()


def test_measurements_dock_is_hidden_by_default():
    window = _window_with_edge_image()
    try:
        assert window.measurements_dock.isHidden()
    finally:
        window.close()


def test_detection_preview_updates_measurements_dock_not_canvas():
    window = _window_with_edge_image()
    try:
        window.search_radius_spin.setValue(120)
        window.curve_sensitivity_spin.setValue(9)
        window.canvas.clear_detection_preview()
        window._show_detection_preview()

        assert not window.canvas.detection_preview_items
        assert "경계인식 범위" in window.detection_preview_label.text()
        assert "120" in window.detection_preview_label.text()
        assert "세그먼트 크기: 9px" in window.detection_preview_label.text()
        assert not window.measurements_dock.isHidden()

        window.clear_detection_preview()
        assert window.detection_preview_label.text() == "경계인식 범위: -"
    finally:
        window.close()


def test_segment_selection_tool_shows_profile_in_measurement_dock():
    window = _window_with_edge_image()
    try:
        edge = LineRecord(
            id="E1",
            kind="edge",
            start=(70.0, 20.0),
            end=(70.0, 100.0),
            label="edge",
            points=[(70.0, 20.0), (70.0, 60.0), (70.0, 100.0)],
            recognition_points=[(70.0, 20.0), (70.0, 60.0), (70.0, 100.0)],
            edge_segmented=True,
            search_radius_px=25,
        )
        window.records[edge.id] = edge
        window.canvas.redraw_lines(list(window.records.values()))
        assert "segment" in window.tool_buttons

        window.set_current_tool("segment")
        view_pos = QPointF(window.canvas.mapFromScene(QPointF(70.0, 40.0)))
        press = QMouseEvent(
            QEvent.Type.MouseButtonPress,
            view_pos,
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        release = QMouseEvent(
            QEvent.Type.MouseButtonRelease,
            view_pos,
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.NoModifier,
        )
        window.canvas.mousePressEvent(press)
        window.canvas.mouseReleaseEvent(release)

        assert window.selected_segment == ("E1", 0)
        assert window.canvas.selected_segment_item is not None
        assert window.measurements_dock.windowTitle() == "세그먼트 밝기"
        assert not window.measurements_dock.isHidden()
        assert not window.segment_profile_label.isHidden()
        assert window.measurement_table.isHidden()
        assert window.segment_profile_label.pixmap() is not None
        assert not window.segment_profile_label.pixmap().isNull()
    finally:
        window.close()


def test_segment_selection_accepts_search_range_band_click():
    window = _window_with_edge_image()
    try:
        edge = LineRecord(
            id="E1",
            kind="edge",
            start=(70.0, 20.0),
            end=(70.0, 100.0),
            label="edge",
            points=[(70.0, 20.0), (70.0, 60.0), (70.0, 100.0)],
            recognition_points=[(70.0, 20.0), (70.0, 60.0), (70.0, 100.0)],
            edge_segmented=True,
            search_radius_px=25,
        )
        window.records[edge.id] = edge
        window.canvas.redraw_lines(list(window.records.values()))
        window._update_search_range_overlay()

        window.set_current_tool("segment")
        view_pos = QPointF(window.canvas.mapFromScene(QPointF(90.0, 40.0)))
        press = QMouseEvent(
            QEvent.Type.MouseButtonPress,
            view_pos,
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        release = QMouseEvent(
            QEvent.Type.MouseButtonRelease,
            view_pos,
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.NoModifier,
        )
        window.canvas.mousePressEvent(press)
        window.canvas.mouseReleaseEvent(release)

        assert window.selected_segment == ("E1", 0)
        assert window.canvas.selected_segment_item is not None
    finally:
        window.close()


def test_segment_profile_display_orders_follow_image_axes():
    offsets = np.asarray([-2, -1, 0, 1, 2], dtype=np.float32)

    vertical_top_down = MainWindow._segment_profile_offset_display_order((70.0, 20.0), (70.0, 100.0), offsets)
    horizontal_left_right = MainWindow._segment_profile_offset_display_order((20.0, 60.0), (120.0, 60.0), offsets)
    vertical_distance = MainWindow._segment_profile_distance_display_order((70.0, 100.0), (70.0, 20.0), 5)
    horizontal_distance = MainWindow._segment_profile_distance_display_order((120.0, 60.0), (20.0, 60.0), 5)

    assert list(vertical_top_down) == [4, 3, 2, 1, 0]
    assert list(horizontal_left_right) == [0, 1, 2, 3, 4]
    assert list(vertical_distance) == [4, 3, 2, 1, 0]
    assert list(horizontal_distance) == [4, 3, 2, 1, 0]


def test_r_hold_drag_box_selects_segment():
    window = _window_with_edge_image()
    try:
        edge = LineRecord(
            id="E1",
            kind="edge",
            start=(70.0, 20.0),
            end=(70.0, 100.0),
            label="edge",
            points=[(70.0, 20.0), (70.0, 60.0), (70.0, 100.0)],
            recognition_points=[(70.0, 20.0), (70.0, 60.0), (70.0, 100.0)],
            edge_segmented=True,
            search_radius_px=25,
        )
        window.records[edge.id] = edge
        window.canvas.redraw_lines(list(window.records.values()))

        press_r = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_R, Qt.KeyboardModifier.NoModifier)
        window.canvas.keyPressEvent(press_r)
        assert window.canvas.current_tool == "segment"

        start_pos = QPointF(window.canvas.mapFromScene(QPointF(58.0, 24.0)))
        end_pos = QPointF(window.canvas.mapFromScene(QPointF(82.0, 56.0)))
        press = QMouseEvent(
            QEvent.Type.MouseButtonPress,
            start_pos,
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        move = QMouseEvent(
            QEvent.Type.MouseMove,
            end_pos,
            Qt.MouseButton.NoButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        release = QMouseEvent(
            QEvent.Type.MouseButtonRelease,
            end_pos,
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.NoModifier,
        )
        window.canvas.mousePressEvent(press)
        window.canvas.mouseMoveEvent(move)
        window.canvas.mouseReleaseEvent(release)

        assert window.selected_segment == ("E1", 0)
        assert window.canvas.selected_segment_item is not None
        assert window.canvas._segment_drag_origin is None
        assert not window.canvas._segment_rubber_band.isVisible()
    finally:
        window.close()


def test_segment_profile_samples_every_pixel_along_segment_length():
    gray = np.zeros((80, 80), dtype=np.float32)
    gray[:, 32:] = 255.0

    result = segment_brightness_profile(gray, (24.0, 10.0), (24.0, 33.0), 12)

    assert result is not None
    assert result.distances.size == 24
    assert math.isclose(float(result.distances[0]), 0.0)
    assert math.isclose(float(result.distances[-1]), 23.0)
    assert result.sample_grid.shape[1] == 24
    assert result.sample_counts.shape == result.sample_grid.shape
    assert int(np.nansum(result.sample_counts)) > result.distances.size
    assert math.isclose(result.best_offset_px, -7.0, abs_tol=1.5)


def test_segment_profile_uses_actual_bgr_image_pixels_as_luma():
    image = np.zeros((80, 80, 3), dtype=np.uint8)
    image[:, 32:] = (255, 255, 255)

    result = segment_brightness_profile(image, (24.0, 10.0), (24.0, 33.0), 12)

    assert result is not None
    assert result.sample_grid.shape == (25, 24)
    assert int(np.nansum(result.sample_counts)) > 300
    assert math.isclose(result.best_offset_px, -7.0, abs_tol=1.5)


def test_segment_profile_boundary_modes_pick_peak_valley_and_side_gradients():
    bright_peak = np.zeros((80, 80), dtype=np.float32)
    for x in range(80):
        bright_peak[:, x] = max(0.0, 255.0 - abs(x - 40) * 35.0)
    dark_valley = np.full((80, 80), 255.0, dtype=np.float32) - bright_peak

    brightest = segment_brightness_profile(bright_peak, (40.0, 10.0), (40.0, 40.0), 12, boundary_mode="brightest")
    darkest = segment_brightness_profile(dark_valley, (40.0, 10.0), (40.0, 40.0), 12, boundary_mode="darkest")
    left = segment_brightness_profile(bright_peak, (40.0, 10.0), (40.0, 40.0), 12, boundary_mode="left_gradient")
    right = segment_brightness_profile(bright_peak, (40.0, 10.0), (40.0, 40.0), 12, boundary_mode="right_gradient")

    assert brightest is not None
    assert darkest is not None
    assert left is not None
    assert right is not None
    assert math.isclose(brightest.best_offset_px, 0.0, abs_tol=1.0)
    assert math.isclose(darkest.best_offset_px, 0.0, abs_tol=1.0)
    assert left.best_offset_px < 0.0
    assert right.best_offset_px > 0.0


def test_snap_polyline_uses_full_segment_area_pixels():
    gray = np.zeros((80, 80), dtype=np.float32)
    gray[:, 32:] = 255.0

    result = snap_polyline_to_gradient(gray, [(24.0, 10.0), (24.0, 33.0)], 12, 30)

    assert result is not None
    assert len(result.points) == 2
    assert 30.0 <= result.points[0][0] <= 33.5
    assert 30.0 <= result.points[1][0] <= 33.5


def test_r_key_temporarily_activates_segment_selection_tool():
    window = _window_with_edge_image()
    try:
        window.set_current_tool("select")
        press = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_R, Qt.KeyboardModifier.NoModifier)
        window.canvas.keyPressEvent(press)

        assert window.canvas.current_tool == "segment"

        release = QKeyEvent(QEvent.Type.KeyRelease, Qt.Key.Key_R, Qt.KeyboardModifier.NoModifier)
        window.canvas.keyReleaseEvent(release)

        assert window.canvas.current_tool == "select"
    finally:
        window.close()


def test_e_key_restores_angle_label_selection_filter():
    window = _window_with_edge_image()
    try:
        press = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_E, Qt.KeyboardModifier.NoModifier)
        window.canvas.keyPressEvent(press)

        assert window.canvas._selection_filter == "angle_label"

        release = QKeyEvent(QEvent.Type.KeyRelease, Qt.Key.Key_E, Qt.KeyboardModifier.NoModifier)
        window.canvas.keyReleaseEvent(release)

        assert window.canvas._selection_filter is None
        assert window.canvas.current_tool == "select"
    finally:
        window.close()


def test_annotation_visual_sizes_use_screen_units():
    window = _window_with_edge_image()
    try:
        window._create_edge_line((70.0, 20.0), (70.0, 100.0))
        window.canvas.redraw_lines(list(window.records.values()))
        edge = next(record for record in window.records.values() if record.kind == "edge")
        line_item = window.canvas.line_items[edge.id]

        assert line_item.pen().isCosmetic()

        line_item.setSelected(True)
        window.canvas.refresh_point_handles()
        assert window.canvas.point_handle_items
        assert window.canvas.point_handle_items[0].flags() & QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations
    finally:
        window.close()


def test_scale_line_shift_constraints():
    window = _window_with_edge_image()
    try:
        start = QPointF(20.0, 30.0)
        end = QPointF(90.0, 80.0)

        horizontal = window.canvas._scale_line_end_for_modifiers(start, end, Qt.KeyboardModifier.ShiftModifier)
        vertical = window.canvas._scale_line_end_for_modifiers(
            start,
            end,
            Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier,
        )

        assert horizontal.x() == 90.0
        assert horizontal.y() == 30.0
        assert vertical.x() == 20.0
        assert vertical.y() == 80.0
    finally:
        window.close()


def test_guide_line_shift_constraints_with_mouse_drag():
    window = _window_with_edge_image()
    try:
        window.set_current_tool("guide")
        start = QPointF(20.0, 30.0)
        end = QPointF(90.0, 80.0)

        def drag(modifiers: Qt.KeyboardModifier) -> LineRecord:
            before = {record.id for record in window.records.values() if record.kind == "guide"}
            start_pos = QPointF(window.canvas.mapFromScene(start))
            end_pos = QPointF(window.canvas.mapFromScene(end))
            press = QMouseEvent(
                QEvent.Type.MouseButtonPress,
                start_pos,
                Qt.MouseButton.LeftButton,
                Qt.MouseButton.LeftButton,
                modifiers,
            )
            move = QMouseEvent(
                QEvent.Type.MouseMove,
                end_pos,
                Qt.MouseButton.NoButton,
                Qt.MouseButton.LeftButton,
                modifiers,
            )
            release = QMouseEvent(
                QEvent.Type.MouseButtonRelease,
                end_pos,
                Qt.MouseButton.LeftButton,
                Qt.MouseButton.NoButton,
                modifiers,
            )
            window.canvas.mousePressEvent(press)
            window.canvas.mouseMoveEvent(move)
            window.canvas.mouseReleaseEvent(release)
            created = [record for record in window.records.values() if record.kind == "guide" and record.id not in before]
            assert len(created) == 1
            return created[0]

        horizontal = drag(Qt.KeyboardModifier.ShiftModifier)
        vertical = drag(Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier)

        assert math.isclose(horizontal.start[1], horizontal.end[1], abs_tol=0.001)
        assert horizontal.axis == "horizontal"
        assert math.isclose(vertical.start[0], vertical.end[0], abs_tol=0.001)
        assert vertical.axis == "vertical"
        assert window.canvas._panning is False
    finally:
        window.close()


def test_scale_tool_magnifier_updates_near_cursor():
    window = _window_with_edge_image()
    try:
        window.set_current_tool("scale")
        window.canvas.resize(320, 240)

        window.canvas._update_scale_magnifier(QPoint(40, 40))

        assert not window.canvas._magnifier_label.isHidden()
        assert window.canvas._magnifier_label.pixmap() is not None
        assert not window.canvas._magnifier_label.pixmap().isNull()
    finally:
        window.close()


def test_canvas_right_click_requests_image_context_menu():
    window = _window_with_edge_image()
    try:
        captured: list[QPoint] = []
        window.canvas.image_context_requested.connect(captured.append)
        view_pos = QPointF(window.canvas.mapFromScene(QPointF(20.0, 20.0)))
        event = QMouseEvent(
            QEvent.Type.MouseButtonPress,
            view_pos,
            Qt.MouseButton.RightButton,
            Qt.MouseButton.RightButton,
            Qt.KeyboardModifier.NoModifier,
        )

        window.canvas.mousePressEvent(event)

        assert len(captured) == 1
        assert window.canvas._panning is False
    finally:
        window.close()


def test_space_temporarily_switches_to_edge_tool():
    window = _window_with_edge_image()
    try:
        window.set_current_tool("select")

        press = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Space, Qt.KeyboardModifier.NoModifier)
        window.canvas.keyPressEvent(press)

        assert window.canvas.current_tool == "edge"
        assert window.current_tool == "edge"

        release = QKeyEvent(QEvent.Type.KeyRelease, Qt.Key.Key_Space, Qt.KeyboardModifier.NoModifier)
        window.canvas.keyReleaseEvent(release)

        assert window.canvas.current_tool == "select"
        assert window.current_tool == "select"
    finally:
        window.close()


def test_ctrl_tab_shows_shortcut_overlay_only_while_held():
    window = _window_with_edge_image()
    try:
        press = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Tab, Qt.KeyboardModifier.ControlModifier)
        window.canvas.keyPressEvent(press)

        assert window.canvas._shortcut_overlay_visible is True
        assert len(window.canvas.shortcut_overlay_items) == 2
        text = next(item for item in window.canvas.shortcut_overlay_items if isinstance(item, QGraphicsTextItem))
        panel = next(item for item in window.canvas.shortcut_overlay_items if not isinstance(item, QGraphicsTextItem))
        assert "Q + 드래그" in text.toPlainText()
        assert "E + 드래그" in text.toPlainText()
        assert "R 누르고 클릭" in text.toPlainText()
        assert panel.brush().color().alpha() == 150

        release = QKeyEvent(QEvent.Type.KeyRelease, Qt.Key.Key_Tab, Qt.KeyboardModifier.ControlModifier)
        window.canvas.keyReleaseEvent(release)

        assert window.canvas._shortcut_overlay_visible is False
        assert len(window.canvas.shortcut_overlay_items) == 0
    finally:
        window.close()


def test_copy_paste_duplicates_parent_edge_without_angle_children():
    window = _window_with_edge_image()
    try:
        window._create_reference_line((10.0, 10.0), (100.0, 10.0))
        window._create_edge_line((70.0, 20.0), (70.0, 100.0))
        window.canvas.redraw_lines(list(window.records.values()))
        edge_id = next(record.id for record in window.records.values() if record.kind == "edge")
        window.canvas.line_items[edge_id].setSelected(True)
        window.calculate_angles()
        assert len(window.canvas.angle_items) > 0

        window.copy_selected_parent_objects()
        window.paste_parent_objects()

        edges = [record for record in window.records.values() if record.kind == "edge"]
        assert len(edges) == 2
        assert len(window.record_clipboard) == 1
        assert len(window.canvas.selected_line_ids()) == 1
        assert all(record.kind == "edge" for record in window.record_clipboard)
    finally:
        window.close()


def test_ctrl_c_without_selection_copies_annotated_image_at_original_size():
    window = _window_with_edge_image()
    try:
        window.set_current_tool("select")
        window._create_edge_line((70.0, 20.0), (70.0, 100.0))
        window.canvas.redraw_lines(list(window.records.values()))
        window.canvas.scene.clearSelection()
        QApplication.clipboard().clear()

        window.copy_selected_parent_objects()

        image = QApplication.clipboard().image()
        assert not image.isNull()
        assert image.width() == 160
        assert image.height() == 120
        assert window.clipboard_mode is None
    finally:
        window.close()


def test_ctrl_c_with_selected_object_still_copies_parent_object():
    window = _window_with_edge_image()
    try:
        window.set_current_tool("select")
        window._create_edge_line((70.0, 20.0), (70.0, 100.0))
        window.canvas.redraw_lines(list(window.records.values()))
        edge = next(record for record in window.records.values() if record.kind == "edge")
        window.canvas.line_items[edge.id].setSelected(True)

        window.copy_selected_parent_objects()

        assert len(window.record_clipboard) == 1
        assert window.clipboard_mode == "object"
    finally:
        window.close()


def test_selected_edge_angle_display_sector_controls_measurement_angle():
    window = _window_with_edge_image()
    try:
        window._create_edge_line((20.0, 60.0), (120.0, 60.0))
        edge = next(record for record in window.records.values() if record.kind == "edge")
        edge.angle_sector = 1
        edge.angle_arc_radius = 44.0
        edge.angle_label_side = "bottom_right"
        edge.angle_label_gap = 8.0
        window.records["G99"] = LineRecord(
            id="G99",
            kind="guide",
            start=(40.0, 42.679),
            end=(100.0, 77.321),
            label="guide",
            axis="custom",
        )

        window.calculate_angles()

        guide_rows = [row for row in window.last_measurements if row["kind"] == "edge_guide_intersection"]
        assert len(guide_rows) == 1
        assert 29 <= guide_rows[0]["angle_deg"] <= 31
        assert len(window.canvas.angle_items) >= 2
    finally:
        window.close()


def test_angle_sector_numbers_follow_visual_quadrants():
    sectors = [
        app_module.angle_sector_geometry(0.0, 30.0, idx)
        for idx in range(4)
    ]
    midpoints = [math.radians((start + span / 2.0) % 360.0) for start, _end, span in sectors]
    vectors = [(math.cos(angle), math.sin(angle)) for angle in midpoints]

    assert vectors[0][0] > 0 and vectors[0][1] < 0
    assert vectors[1][0] < 0 and vectors[1][1] < 0
    assert vectors[2][0] < 0 and vectors[2][1] > 0
    assert vectors[3][0] > 0 and vectors[3][1] > 0


def test_angle_label_degree_positions():
    center = (100.0, 100.0)
    positions = [
        app_module.angle_label_position_for_sector(center, 0.0, 90.0, 20.0, angle, 5.0)
        for angle in (0, 90, 180, 270)
    ]

    assert [(round(x, 3), round(y, 3)) for x, y in positions] == [
        (125.0, 100.0),
        (100.0, 75.0),
        (75.0, 100.0),
        (100.0, 125.0),
    ]
    assert app_module.normalize_angle_label_side("bottom_left") == "225"


def test_angle_display_edit_without_selection_applies_to_all_edges_and_default(monkeypatch):
    observed_live_values: list[list[str]] = []

    class _Value:
        def __init__(self, value):
            self._value = value

        def value(self):
            return self._value

        def currentData(self):
            return self._value

    class _Dialog:
        def __init__(self, *args, **kwargs):
            self.sector_combo = _Value(2)
            self.arc_radius_spin = _Value(55.0)
            self.label_position_spin = _Value(225)
            self.label_gap_spin = _Value(9.0)
            self.label_font_size_spin = _Value(16.0)
            self._on_changed = kwargs.get("on_changed")

        def exec(self):
            if self._on_changed is not None:
                self._on_changed()
                observed_live_values.append(
                    [record.angle_label_side for record in window.records.values() if record.kind == "edge"]
                )
            return QDialog.DialogCode.Accepted

    monkeypatch.setattr(app_module, "AngleDisplaySettingsDialog", _Dialog)
    window = _window_with_edge_image()
    try:
        window._create_edge_line((20.0, 20.0), (20.0, 80.0))
        window._create_edge_line((60.0, 20.0), (60.0, 80.0))
        window.canvas.redraw_lines(list(window.records.values()))

        window.edit_angle_display_for_selected_edges()

        edges = [record for record in window.records.values() if record.kind == "edge"]
        assert observed_live_values == [["225", "225"]]
        assert all(edge.angle_sector == 2 for edge in edges)
        assert all(edge.angle_arc_radius == 55.0 for edge in edges)
        assert all(edge.angle_label_side == "225" for edge in edges)
        assert all(edge.angle_label_gap == 9.0 for edge in edges)
        assert all(edge.angle_label_font_size == 16.0 for edge in edges)
        assert window.default_angle_sector == 2
        assert window.default_angle_label_font_size == 16.0

        window._create_edge_line((90.0, 20.0), (90.0, 80.0))
        newest = list(window.records.values())[-1]
        assert newest.angle_sector == 2
        assert newest.angle_arc_radius == 55.0
        assert newest.angle_label_side == "225"
        assert newest.angle_label_gap == 9.0
        assert newest.angle_label_font_size == 16.0
    finally:
        window.close()


def test_angle_label_font_size_is_rendered():
    window = _window_with_edge_image()
    try:
        window._create_reference_line((10.0, 10.0), (100.0, 10.0))
        window._create_edge_line((70.0, 20.0), (70.0, 100.0))
        edge = next(record for record in window.records.values() if record.kind == "edge")
        edge.angle_label_font_size = 18.0
        window.canvas.redraw_lines(list(window.records.values()))

        window.calculate_angles(reset_hidden=True)

        label = next(item for item in window.canvas.angle_items if isinstance(item, QGraphicsTextItem))
        assert "font-size:18pt" in label.toHtml()
    finally:
        window.close()


def test_cd_length_modes_measure_adjacent_edge_intersections():
    window = _window_with_edge_image()
    try:
        for x in (40.0, 80.0, 120.0):
            window._create_edge_line((x, 10.0), (x, 110.0))
        window.records["G99"] = LineRecord(
            id="G99",
            kind="guide",
            start=(0.0, 60.0),
            end=(150.0, 60.0),
            label="guide",
            axis="horizontal",
        )
        window.canvas.redraw_lines(list(window.records.values()))

        window.cd_segment_combo.setCurrentIndex(window.cd_segment_combo.findData("all"))
        window.calculate_cd_lengths()
        cd_rows = [row for row in window.last_measurements if row["kind"] == "cd_length"]
        assert len(cd_rows) == 2
        assert [round(row["cd_length_px"], 2) for row in cd_rows] == [40.0, 40.0]
        assert len(window.canvas.cd_items) == 4

        window.cd_segment_combo.setCurrentIndex(window.cd_segment_combo.findData("odd"))
        window.calculate_cd_lengths()
        cd_rows = [row for row in window.last_measurements if row["kind"] == "cd_length"]
        assert len(cd_rows) == 1
        assert "_1_" in cd_rows[0]["measurement"]

        window.cd_segment_combo.setCurrentIndex(window.cd_segment_combo.findData("even"))
        window.calculate_cd_lengths()
        cd_rows = [row for row in window.last_measurements if row["kind"] == "cd_length"]
        assert len(cd_rows) == 1
        assert "_2_" in cd_rows[0]["measurement"]
    finally:
        window.close()


def test_guide_display_ids_use_main_guide_as_zero_with_signed_positions():
    horizontal_guides = [
        LineRecord("G_top2", "guide", (0.0, 20.0), (160.0, 20.0), axis="horizontal"),
        LineRecord("G_top1", "guide", (0.0, 40.0), (160.0, 40.0), axis="horizontal"),
        LineRecord("G_main", "guide", (0.0, 60.0), (160.0, 60.0), axis="horizontal", is_main_guide=True),
        LineRecord("G_bottom1", "guide", (0.0, 80.0), (160.0, 80.0), axis="horizontal"),
        LineRecord("G_bottom2", "guide", (0.0, 100.0), (160.0, 100.0), axis="horizontal"),
    ]
    vertical_guides = [
        LineRecord("G_left2", "guide", (20.0, 0.0), (20.0, 120.0), axis="vertical"),
        LineRecord("G_left1", "guide", (40.0, 0.0), (40.0, 120.0), axis="vertical"),
        LineRecord("G_main", "guide", (60.0, 0.0), (60.0, 120.0), axis="vertical", is_main_guide=True),
        LineRecord("G_right1", "guide", (80.0, 0.0), (80.0, 120.0), axis="vertical"),
    ]

    assert app_module.guide_display_ids(horizontal_guides) == {
        "G_top2": "G-2",
        "G_top1": "G-1",
        "G_main": "G0",
        "G_bottom1": "G1",
        "G_bottom2": "G2",
    }
    assert app_module.guide_display_numbers(horizontal_guides) == {
        "G_top2": -2,
        "G_top1": -1,
        "G_main": 0,
        "G_bottom1": 1,
        "G_bottom2": 2,
    }
    assert app_module.guide_display_ids(vertical_guides) == {
        "G_left2": "G-2",
        "G_left1": "G-1",
        "G_main": "G0",
        "G_right1": "G1",
    }
    assert app_module.guide_display_numbers(vertical_guides) == {
        "G_left2": -2,
        "G_left1": -1,
        "G_main": 0,
        "G_right1": 1,
    }


def test_measurement_rows_use_signed_main_guide_display_ids():
    window = _window_with_edge_image()
    try:
        window._create_edge_line((80.0, 10.0), (80.0, 110.0))
        edge = next(record for record in window.records.values() if record.kind == "edge")
        window.records["G_top"] = LineRecord("G_top", "guide", (0.0, 30.0), (150.0, 30.0), axis="horizontal")
        window.records["G_main"] = LineRecord("G_main", "guide", (0.0, 60.0), (150.0, 60.0), axis="horizontal", is_main_guide=True)
        window.records["G_bottom"] = LineRecord("G_bottom", "guide", (0.0, 90.0), (150.0, 90.0), axis="horizontal")

        window.calculate_angles(reset_hidden=False)

        rows = sorted(
            [row for row in window.last_measurements if row["kind"] == "edge_guide_intersection"],
            key=lambda row: float(row["y_px"]),
        )
        assert [row["guide_id"] for row in rows] == ["G-1", "G0", "G1"]
        assert [row["guide_number"] for row in rows] == [-1, 0, 1]
        assert [row["measurement"] for row in rows] == [f"{edge.id}_x_G-1", f"{edge.id}_x_G0", f"{edge.id}_x_G1"]
    finally:
        window.close()


def test_data_export_rows_use_signed_main_guide_display_ids():
    window = _window_with_edge_image()
    try:
        edge = LineRecord("E1", "edge", (80.0, 10.0), (80.0, 110.0))
        top = LineRecord("G_top", "guide", (0.0, 30.0), (150.0, 30.0), axis="horizontal")
        main = LineRecord("G_main", "guide", (0.0, 60.0), (150.0, 60.0), axis="horizontal", is_main_guide=True)
        bottom = LineRecord("G_bottom", "guide", (0.0, 90.0), (150.0, 90.0), axis="horizontal")
        records = [edge, bottom, main, top]

        rows = window._export_rows_for_records(
            "image.png",
            records,
            None,
            window._export_group_info(records, "y"),
            "y",
        )["intersection_angle"]
        assert [row["가이드ID"] for row in rows] == ["G-1", "G0", "G1"]
        assert [row["가이드번호"] for row in rows] == [-1, 0, 1]
        assert [row["측정ID"] for row in rows] == ["E1_x_G-1", "E1_x_G0", "E1_x_G1"]
    finally:
        window.close()


def test_data_export_left_to_right_sorts_by_objects_before_guides():
    window = _window_with_edge_image()
    try:
        left_edge = LineRecord("E_left", "edge", (40.0, 10.0), (40.0, 110.0))
        right_edge = LineRecord("E_right", "edge", (100.0, 10.0), (100.0, 110.0))
        top = LineRecord("G_top", "guide", (0.0, 30.0), (150.0, 30.0), axis="horizontal")
        main = LineRecord("G_main", "guide", (0.0, 60.0), (150.0, 60.0), axis="horizontal", is_main_guide=True)
        bottom = LineRecord("G_bottom", "guide", (0.0, 90.0), (150.0, 90.0), axis="horizontal")
        records = [right_edge, left_edge, bottom, main, top]

        rows = window._export_rows_for_records(
            "image.png",
            records,
            None,
            window._export_group_info(records, "x"),
            "x",
        )["intersection_angle"]

        assert [row["경계ID"] for row in rows] == ["E_left", "E_left", "E_left", "E_right", "E_right", "E_right"]
        assert [row["가이드번호"] for row in rows] == [-1, 0, 1, -1, 0, 1]
        assert [row["개체"] for row in rows] == [
            "E_left|G-1",
            "E_left|G0",
            "E_left|G1",
            "E_right|G-1",
            "E_right|G0",
            "E_right|G1",
        ]
    finally:
        window.close()


def test_cd_label_text_and_position_can_be_edited(monkeypatch):
    observed_live_values: list[str] = []

    class _Value:
        def __init__(self, value):
            self._value = value

        def value(self):
            return self._value

        def currentData(self):
            return self._value

    class _Dialog:
        def __init__(self, *args, **kwargs):
            self.label_position_spin = _Value(270)
            self.label_gap_spin = _Value(22.0)
            self.label_font_size_spin = _Value(17.0)
            self._on_changed = kwargs.get("on_changed")

        def exec(self):
            if self._on_changed is not None:
                self._on_changed()
                observed_live_values.append(window.cd_label_side)
            return QDialog.DialogCode.Accepted

    monkeypatch.setattr(app_module, "CdDisplaySettingsDialog", _Dialog)
    window = _window_with_edge_image()
    try:
        for x in (40.0, 80.0):
            window._create_edge_line((x, 10.0), (x, 110.0))
        window.records["G99"] = LineRecord(
            id="G99",
            kind="guide",
            start=(0.0, 60.0),
            end=(150.0, 60.0),
            label="guide",
            axis="horizontal",
        )
        window.canvas.redraw_lines(list(window.records.values()))

        window.calculate_cd_lengths()
        label = next(item for item in window.canvas.cd_items if isinstance(item, QGraphicsTextItem))
        assert "CD" not in label.toPlainText()
        assert abs(label.sceneBoundingRect().center().x() - 60.0) < 0.001
        assert label.sceneBoundingRect().center().y() < 60.0

        window.edit_cd_display()
        label = next(item for item in window.canvas.cd_items if isinstance(item, QGraphicsTextItem))
        assert observed_live_values == ["270"]
        assert window.cd_label_side == "270"
        assert window.cd_label_gap == 22.0
        assert window.cd_label_font_size == 17.0
        assert abs(label.sceneBoundingRect().center().x() - 60.0) < 0.001
        assert label.sceneBoundingRect().center().y() > 60.0
        assert "font-size:17" in label.toHtml()
    finally:
        window.close()


def test_data_export_sorts_by_group_item_and_position(tmp_path):
    window = _window_with_edge_image()
    try:
        window._create_edge_line((100.0, 10.0), (100.0, 110.0))
        window._create_edge_line((40.0, 10.0), (40.0, 110.0))
        edges = [record for record in window.records.values() if record.kind == "edge"]
        edges[0].object_group = "right_group"
        edges[1].object_group = "left_group"
        window.records["G99"] = LineRecord(
            id="G99",
            kind="guide",
            start=(0.0, 60.0),
            end=(150.0, 60.0),
            axis="horizontal",
        )
        options = DataExportOptions(
            scope="current",
            selected_items={"line_angle", "intersection_angle", "cd_length", "edge_length"},
            order_priority="x",
        )

        sheets = window._build_export_sheets(options)

        assert list(sheets) == ["선각도", "교점각도", "CD길이", "경계길이"]
        assert [row["그룹"] for row in sheets["선각도"]] == ["G1", "G2"]
        assert [row["경계ID"] for row in sheets["선각도"]] == [edges[1].id, edges[0].id]
        assert [row["개체"] for row in sheets["선각도"]] == [edges[1].id, edges[0].id]
        assert sheets["교점각도"][0]["개체"] == f"{edges[1].id}|G1"
        assert sheets["CD길이"][0]["개체"] == f"{edges[1].id}|{edges[0].id}|G1"
        assert len(sheets["교점각도"]) == 2
        assert len(sheets["CD길이"]) == 1
        assert sheets["CD길이"][0]["길이_px"] == 60.0

        export_path = tmp_path / "export.xlsx"
        app_module.write_xlsx(str(export_path), sheets)
        with zipfile.ZipFile(export_path) as archive:
            names = set(archive.namelist())
            assert "xl/workbook.xml" in names
            assert "xl/worksheets/sheet1.xml" in names
            assert "<t>개체</t>" in archive.read("xl/worksheets/sheet1.xml").decode("utf-8")
    finally:
        window.close()


def test_data_export_dialog_has_open_after_export_option():
    _app()
    dialog = app_module.DataExportDialog(False)
    try:
        assert dialog.options().open_after_export is False
        dialog.open_after_export_checkbox.setChecked(True)
        assert dialog.options().open_after_export is True
    finally:
        dialog.close()


def test_data_export_opens_file_when_option_is_checked(monkeypatch, tmp_path):
    class _Dialog:
        def __init__(self, *args, **kwargs):
            pass

        def exec(self):
            return QDialog.DialogCode.Accepted

        def options(self):
            return DataExportOptions(scope="current", selected_items={"edge_length"}, order_priority="y", open_after_export=True)

    export_path = tmp_path / "export.xlsx"
    opened: list[str] = []
    written: list[str] = []
    monkeypatch.setattr(app_module, "DataExportDialog", _Dialog)
    monkeypatch.setattr(app_module.QFileDialog, "getSaveFileName", lambda *args, **kwargs: (str(export_path), "Excel Workbook (*.xlsx)"))
    monkeypatch.setattr(app_module, "write_xlsx", lambda path, sheets: written.append(path))
    monkeypatch.setattr(app_module.MainWindow, "_open_export_file", staticmethod(lambda path: opened.append(path)))

    window = _window_with_edge_image()
    try:
        window._create_edge_line((40.0, 10.0), (40.0, 110.0))

        window.export_data_xlsx()

        assert written == [str(export_path)]
        assert opened == [str(export_path)]
    finally:
        window.close()


def test_data_export_project_scope_includes_saved_image_states():
    window = _window_with_edge_image()
    try:
        window.image_path = "/tmp/current.png"
        window._create_edge_line((20.0, 10.0), (20.0, 110.0))
        other = LineRecord("E_other", "edge", (80.0, 10.0), (80.0, 110.0))
        window.image_states["/tmp/other.png"] = {
            "records": [app_module.asdict(other)],
            "counter": 2,
            "nm_per_px": None,
            "hidden_angle_measurements": [],
        }
        options = DataExportOptions(scope="project", selected_items={"edge_length"}, order_priority="y")

        sheets = window._build_export_sheets(options)

        image_names = {row["이미지"] for row in sheets["경계길이"]}
        assert image_names == {"current.png", "other.png"}
    finally:
        window.close()


def test_structure_template_paste_skips_guides_when_current_image_has_guides():
    window = _window_with_edge_image()
    try:
        template = StructureTemplate(
            name="Line Space",
            cd_segment_mode="odd",
            records=[
                LineRecord("E_old_1", "edge", (40.0, 10.0), (40.0, 110.0)),
                LineRecord("E_old_2", "edge", (80.0, 10.0), (80.0, 110.0)),
                LineRecord("G_old", "guide", (0.0, 60.0), (150.0, 60.0), axis="horizontal"),
            ],
        )
        window.structure_templates = [template]
        window._refresh_structure_combo(0)
        window.records["G_existing"] = LineRecord(
            "G_existing",
            "guide",
            (0.0, 50.0),
            (150.0, 50.0),
            axis="horizontal",
        )
        window.canvas.redraw_lines(list(window.records.values()))

        window.paste_selected_structure_template()

        edges = [record for record in window.records.values() if record.kind == "edge"]
        guides = [record for record in window.records.values() if record.kind == "guide"]
        assert len(edges) == 2
        assert len(guides) == 1
        assert window.cd_segment_combo.currentData() == "odd"
        assert not [row for row in window.last_measurements if row["kind"] == "cd_length"]

        window.calculate_cd_lengths()
        assert len([row for row in window.last_measurements if row["kind"] == "cd_length"]) == 1
    finally:
        window.close()


def test_guides_draw_select_move_and_delete():
    window = _window_with_edge_image()
    try:
        window.guide_orientation_combo.setCurrentIndex(window.guide_orientation_combo.findData("horizontal"))
        window.guide_spacing_spin.setValue(40)

        window.add_guides()

        guides = [record for record in window.records.values() if record.kind == "guide"]
        assert len(guides) == 4
        guide = guides[0]
        original_y = guide.start[1]
        item = window.canvas.line_items[guide.id]
        assert item.flags() & QGraphicsItem.GraphicsItemFlag.ItemIsSelectable
        assert item.flags() & QGraphicsItem.GraphicsItemFlag.ItemIsMovable

        item.setSelected(True)
        assert window.canvas._nudge_selected_items(Qt.Key.Key_Down, Qt.KeyboardModifier.ControlModifier)
        window._sync_records_from_canvas()
        assert window.records[guide.id].start[1] == original_y + 1.0

        window.delete_selected()
        assert guide.id not in window.records
    finally:
        window.close()


def test_guide_tool_creates_manual_guide_line():
    window = _window_with_edge_image()
    try:
        window._handle_line_created("guide", (10.0, 30.0), (120.0, 30.0), None)

        guides = [record for record in window.records.values() if record.kind == "guide"]
        assert len(guides) == 1
        assert guides[0].axis == "horizontal"
        assert guides[0].id in window.canvas.line_items
    finally:
        window.close()


def test_reference_line_creates_main_guide_on_same_line():
    window = _window_with_edge_image()
    try:
        window.axis_combo.setCurrentIndex(window.axis_combo.findData("horizontal"))
        window._create_reference_line((10.0, 20.0), (120.0, 20.0))

        guides = [record for record in window.records.values() if record.kind == "guide"]
        assert len(guides) == 1
        main_guide = guides[0]
        assert main_guide.is_main_guide is True
        assert main_guide.start == (10.0, 20.0)
        assert main_guide.end == (120.0, 20.0)
        assert main_guide.axis == "horizontal"

        window.axis_combo.setCurrentIndex(window.axis_combo.findData("vertical"))
        window._create_reference_line((40.0, 10.0), (40.0, 100.0))

        guides = [record for record in window.records.values() if record.kind == "guide"]
        assert len(guides) == 1
        assert guides[0].id == main_guide.id
        assert guides[0].is_main_guide is True
        assert guides[0].start == (40.0, 10.0)
        assert guides[0].end == (40.0, 100.0)
        assert guides[0].axis == "vertical"
    finally:
        window.close()


def test_reference_tool_returns_to_select_after_drawing():
    window = _window_with_edge_image()
    try:
        window.set_current_tool("reference")

        window._handle_line_created("reference", (10.0, 20.0), (120.0, 20.0), None)

        assert window.current_tool == "select"
        assert window.canvas.current_tool == "select"
        assert window.tool_buttons["select"].isChecked()
        assert len([record for record in window.records.values() if record.kind == "reference"]) == 1
    finally:
        window.close()


def test_align_to_reference_minimizes_first_rotation_then_toggles_180(monkeypatch):
    captured: list[float] = []

    def fake_rotate(image, points, angle):
        captured.append(float(angle))
        if len(captured) == 1:
            return image, [(float(point[0]), 10.0) for point in points]
        return image, list(points)

    monkeypatch.setattr(app_module, "rotate_image_and_points", fake_rotate)
    window = _window_with_edge_image()
    try:
        window.axis_combo.setCurrentIndex(window.axis_combo.findData("horizontal"))
        end_y = 10.0 + math.tan(math.radians(179.0)) * 80.0
        window._create_reference_line((10.0, 10.0), (90.0, end_y))

        window.align_to_reference()
        window.align_to_reference()
        window.align_to_reference()

        assert -1.1 <= captured[0] <= -0.9
        assert captured[1:] == [180.0, 180.0]
    finally:
        window.close()


def test_rotate_current_image_saves_rotation_state(tmp_path):
    image_path = tmp_path / "a.png"
    cv2.imwrite(str(image_path), np.zeros((20, 40, 3), dtype=np.uint8))

    _app()
    window = MainWindow()
    try:
        window.browser_root = tmp_path
        window.browser_image_paths = [str(image_path)]
        window._load_image_path(str(image_path), preserve_calibration=False)
        window._create_edge_line((5.0, 5.0), (30.0, 5.0))
        edge = next(record for record in window.records.values() if record.kind == "edge")

        window.rotate_current_image(90.0, "테스트 회전")

        assert window.image_bgr.shape[:2] == (40, 20)
        assert window.image_rotation_degrees == 90.0
        assert window.image_states[str(image_path)]["image_rotation_degrees"] == 90.0
        assert window.records[edge.id].start != (5.0, 5.0)
        assert "테스트 회전: 90.000°" in window.rotation_status_label.text()

        window._load_image_path(str(image_path), preserve_calibration=True)

        assert window.image_bgr.shape[:2] == (40, 20)
        assert window.image_rotation_degrees == 90.0
        assert window.records[edge.id].start != (5.0, 5.0)
    finally:
        window.close()


def test_rotate_selected_thumbnails_updates_per_image_state_without_overwriting_files(tmp_path):
    path_a = tmp_path / "a.png"
    path_b = tmp_path / "b.png"
    cv2.imwrite(str(path_a), np.zeros((20, 40, 3), dtype=np.uint8))
    cv2.imwrite(str(path_b), np.full((30, 50, 3), 255, dtype=np.uint8))

    _app()
    window = MainWindow()
    try:
        window.browser_root = tmp_path
        window.browser_image_paths = [str(path_a), str(path_b)]
        window.selected_thumbnail_paths = {str(path_a), str(path_b)}
        edge = LineRecord("E1", "edge", (5.0, 5.0), (25.0, 5.0))
        window.image_states[str(path_b)] = {
            "records": [app_module.asdict(edge)],
            "counter": 2,
            "nm_per_px": None,
            "hidden_angle_measurements": [],
            "image_adjustments": {},
            "image_rotation_degrees": 0.0,
        }

        window.apply_image_rotation(90.0, "90° 회전")

        assert window.image_states[str(path_a)]["image_rotation_degrees"] == 90.0
        assert window.image_states[str(path_b)]["image_rotation_degrees"] == 90.0
        rotated_edge = app_module.line_record_from_dict(window.image_states[str(path_b)]["records"][0])
        assert rotated_edge.start != edge.start
        assert cv2.imread(str(path_a)).shape[:2] == (20, 40)
        assert cv2.imread(str(path_b)).shape[:2] == (30, 50)
    finally:
        window.close()


def test_structure_template_round_trip_dict():
    template = StructureTemplate(
        name="CD pair",
        cd_segment_mode="even",
        records=[LineRecord("E1", "edge", (1.0, 2.0), (3.0, 4.0), angle_sector=2)],
    )

    loaded = structure_template_from_dict(structure_template_to_dict(template))

    assert loaded.name == "CD pair"
    assert loaded.cd_segment_mode == "even"
    assert loaded.records[0].angle_sector == 2


def test_selection_filter_keeps_only_requested_object_type():
    window = _window_with_edge_image()
    try:
        window._create_edge_line((20.0, 60.0), (120.0, 60.0))
        edge = next(record for record in window.records.values() if record.kind == "edge")
        window.records["G99"] = LineRecord(
            id="G99",
            kind="guide",
            start=(40.0, 42.679),
            end=(100.0, 77.321),
            label="guide",
            axis="custom",
        )
        window.canvas.redraw_lines(list(window.records.values()))
        window.calculate_angles()

        for item in [window.canvas.line_items[edge.id], *window.canvas.angle_items]:
            item.setSelected(True)
        window.canvas._selection_filter = "edge"
        window.canvas._apply_selection_filter()
        assert window.canvas.selected_line_ids() == [edge.id]

        window.canvas._selection_filter = "angle_arc"
        window.canvas.scene.clearSelection()
        for item in window.canvas.angle_items:
            item.setSelected(True)
        window.canvas._apply_selection_filter()
        assert all(isinstance(item, QGraphicsPathItem) for item in window.canvas.scene.selectedItems())

        window.canvas._selection_filter = "angle_label"
        window.canvas.scene.clearSelection()
        for item in window.canvas.angle_items:
            item.setSelected(True)
        window.canvas._apply_selection_filter()
        assert all(isinstance(item, QGraphicsTextItem) for item in window.canvas.scene.selectedItems())
    finally:
        window.close()


def test_edge_length_overlay_uses_calibration_and_visibility():
    window = _window_with_edge_image()
    try:
        window._create_edge_line((20.0, 60.0), (120.0, 60.0))
        window.nm_per_px = 2.0
        window.canvas.redraw_lines(list(window.records.values()))
        window.set_visibility("edge_length", True)

        window._update_edge_length_overlay()

        assert len(window.canvas.edge_length_items) == 1
        assert "200" in window.canvas.edge_length_items[0].toHtml()
        center = window.canvas.edge_length_items[0].sceneBoundingRect().center()
        assert abs(center.x() - 70.0) < 1.0
        assert abs(center.y() - 60.0) < 1.0

        window.set_visibility("edge_length", False)
        assert len(window.canvas.edge_length_items) == 0
    finally:
        window.close()


def test_edge_length_label_is_deletable_child_object():
    window = _window_with_edge_image()
    try:
        window._create_edge_line((20.0, 60.0), (120.0, 60.0))
        edge = next(record for record in window.records.values() if record.kind == "edge")
        window.canvas.redraw_lines(list(window.records.values()))
        window.set_visibility("edge_length", True)
        window._update_edge_length_overlay()

        assert len(window.canvas.edge_length_items) == 1
        label = window.canvas.edge_length_items[0]
        label.setSelected(True)

        window.delete_selected()

        assert len(window.canvas.edge_length_items) == 0
        assert window.records[edge.id].show_edge_length is False

        window._update_edge_length_overlay()
        assert len(window.canvas.edge_length_items) == 0
    finally:
        window.close()


def test_edge_length_label_position_can_be_moved_and_persists():
    window = _window_with_edge_image()
    try:
        window._create_edge_line((20.0, 60.0), (120.0, 60.0))
        edge = next(record for record in window.records.values() if record.kind == "edge")
        window.canvas.redraw_lines(list(window.records.values()))
        window.set_visibility("edge_length", True)
        window._update_edge_length_overlay()

        label = window.canvas.edge_length_items[0]
        label.setPos(42.0, 37.0)
        window._sync_records_from_canvas()

        assert window.records[edge.id].edge_length_label_pos == (42.0, 37.0)

        window._update_edge_length_overlay()

        assert len(window.canvas.edge_length_items) == 1
        assert window.canvas.edge_length_items[0].pos() == QPointF(42.0, 37.0)
    finally:
        window.close()


def test_edge_length_overlay_skips_segmented_edges():
    window = _window_with_edge_image()
    try:
        window._create_edge_line((40.0, 20.0), (70.0, 60.0))
        edge = next(record for record in window.records.values() if record.kind == "edge")
        edge.points = [(40.0, 20.0), (55.0, 40.0), (70.0, 60.0)]
        edge.recognition_points = list(edge.points)
        edge.edge_segmented = True
        window.canvas.redraw_lines(list(window.records.values()))

        window._update_edge_length_overlay()

        assert len(window.canvas.edge_length_items) == 0
    finally:
        window.close()


def test_scale_preset_apply_restores_scale_bar():
    window = _window_with_edge_image()
    try:
        window.scale_presets = [
            ScalePreset("100 nm", nm_per_px=2.0, bar_px=50.0, bar_nm=100.0, start=(25.0, 35.0), end=(75.0, 35.0))
        ]

        window.apply_scale_preset(0)

        scales = [record for record in window.records.values() if record.kind == "scale"]
        assert len(scales) == 1
        assert scales[0].start == (25.0, 35.0)
        assert scales[0].end == (75.0, 35.0)
        assert scales[0].value_nm == 100.0
        assert scales[0].label == "100 nm"
        assert round(scales[0].end[0] - scales[0].start[0], 3) == 50.0
        assert window.nm_per_px == 2.0
    finally:
        window.close()


def test_switching_images_restores_per_image_annotations(tmp_path):
    path_a = tmp_path / "a.png"
    path_b = tmp_path / "b.png"
    cv2.imwrite(str(path_a), np.zeros((80, 120, 3), dtype=np.uint8))
    cv2.imwrite(str(path_b), np.full((80, 120, 3), 255, dtype=np.uint8))

    _app()
    window = MainWindow()
    try:
        window.browser_image_paths = [str(path_a), str(path_b)]
        window.current_browser_index = 0
        window._load_image_path(str(path_a), preserve_calibration=False)
        window._create_edge_line((20.0, 20.0), (90.0, 20.0))
        edge = next(record for record in window.records.values() if record.kind == "edge")

        window._load_image_path(str(path_b), preserve_calibration=True)
        window.scale_presets = [ScalePreset("100 nm", nm_per_px=2.0, bar_px=50.0, bar_nm=100.0)]
        window.apply_scale_preset(0)
        assert [record for record in window.records.values() if record.kind == "scale"]

        window._load_image_path(str(path_a), preserve_calibration=True)

        assert edge.id in window.records
        assert window.records[edge.id].start == (20.0, 20.0)
        assert edge.id in window.canvas.line_items
        assert window.canvas.line_items[edge.id].isVisible()
        assert not [record for record in window.records.values() if record.kind == "scale"]
    finally:
        window.close()


def test_scan_folder_images_uses_natural_numeric_order(tmp_path):
    for name in ["sem1.png", "sem10.png", "sem2.png", "sem11.png", "sem3.png"]:
        cv2.imwrite(str(tmp_path / name), np.zeros((20, 20, 3), dtype=np.uint8))
    (tmp_path / "sem4.txt").write_text("not an image", encoding="utf-8")

    _app()
    window = MainWindow()
    try:
        paths = window._scan_folder_images(tmp_path)

        assert [path.name for path in paths] == ["sem1.png", "sem2.png", "sem3.png", "sem10.png", "sem11.png"]
    finally:
        window.close()


def test_scan_folder_images_uses_natural_numeric_order_for_nested_folders(tmp_path):
    paths = [
        tmp_path / "000" / "1" / "a.png",
        tmp_path / "000" / "10" / "a.png",
        tmp_path / "000" / "2" / "a.png",
    ]
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), np.zeros((20, 20, 3), dtype=np.uint8))

    _app()
    window = MainWindow()
    try:
        scanned = window._scan_folder_images(tmp_path)

        assert [str(path.relative_to(tmp_path)) for path in scanned] == [
            "000/1/a.png",
            "000/2/a.png",
            "000/10/a.png",
        ]
    finally:
        window.close()


def test_favorite_images_show_as_tabs_and_switch_images(tmp_path):
    path_a = tmp_path / "a.png"
    path_b = tmp_path / "b.png"
    cv2.imwrite(str(path_a), np.zeros((80, 120, 3), dtype=np.uint8))
    cv2.imwrite(str(path_b), np.full((80, 120, 3), 255, dtype=np.uint8))

    _app()
    window = MainWindow()
    try:
        window.browser_root = tmp_path
        window.browser_image_paths = [str(path_a), str(path_b)]
        window.current_browser_index = 0
        window._load_image_path(str(path_a), preserve_calibration=False)

        window.add_favorite_image(str(path_a))
        window.add_favorite_image(str(path_b))

        assert not window.favorite_tab_bar.isHidden()
        assert window.favorite_tab_bar.count() == 2
        assert window.favorite_tab_bar.tabData(0) == str(path_a)
        assert window.favorite_tab_bar.tabText(1) == "b.png"

        window.favorite_tab_bar.setCurrentIndex(1)

        assert window.image_path == str(path_b)
        assert window.current_browser_index == 1

        window.remove_favorite_image(str(path_b))

        assert window.favorite_tab_bar.count() == 1
        assert window.favorite_image_paths == [str(path_a)]
    finally:
        window.close()


def test_thumbnail_context_menu_adds_and_removes_favorite_without_switching_image(tmp_path):
    path_a = tmp_path / "a.png"
    path_b = tmp_path / "b.png"
    cv2.imwrite(str(path_a), np.zeros((80, 120, 3), dtype=np.uint8))
    cv2.imwrite(str(path_b), np.full((80, 120, 3), 255, dtype=np.uint8))

    _app()
    window = MainWindow()
    try:
        window.browser_root = tmp_path
        window.browser_image_paths = [str(path_a), str(path_b)]
        window.current_browser_index = 0
        window._load_image_path(str(path_a), preserve_calibration=False)
        window._populate_thumbnails()

        thumbnail = window.thumbnail_buttons[str(path_b)]
        assert thumbnail.contextMenuPolicy() == Qt.ContextMenuPolicy.CustomContextMenu

        menu = window._favorite_menu_for_path(str(path_b))
        assert menu.actions()[0].text() == "즐겨찾기에 저장"
        menu.actions()[0].trigger()

        assert window.image_path == str(path_a)
        assert window.favorite_image_paths == [str(path_b)]
        assert window.favorite_tab_bar.count() == 1
        assert window.favorite_tab_bar.tabData(0) == str(path_b)

        menu = window._favorite_menu_for_path(str(path_b))
        assert menu.actions()[0].text() == "즐겨찾기에서 제거"
        menu.actions()[0].trigger()

        assert window.favorite_image_paths == []
        assert window.favorite_tab_bar.count() == 0
    finally:
        window.close()


def test_favorite_tabs_can_be_renamed_and_grouped(tmp_path, monkeypatch):
    path_a = tmp_path / "a.png"
    path_b = tmp_path / "b.png"
    cv2.imwrite(str(path_a), np.zeros((80, 120, 3), dtype=np.uint8))
    cv2.imwrite(str(path_b), np.full((80, 120, 3), 255, dtype=np.uint8))
    monkeypatch.setattr(app_module.QInputDialog, "getText", lambda *args, **kwargs: ("Top SEM", True))

    _app()
    window = MainWindow()
    try:
        window.browser_root = tmp_path
        window.browser_image_paths = [str(path_a), str(path_b)]
        window.current_browser_index = 0
        window._load_image_path(str(path_a), preserve_calibration=False)
        window.add_favorite_image(str(path_a))
        window.add_favorite_image(str(path_b))

        window.rename_favorite_tab(0)

        assert window.favorite_image_labels[str(path_a)] == "Top SEM"
        assert window.favorite_tab_bar.tabText(0) == "Top SEM"

        window.create_favorite_group_for_path(str(path_b), "Review")

        assert window.favorite_image_groups[str(path_b)] == "Review"
        assert not window.favorite_group_bar.isHidden()
        assert window.current_favorite_group == "Review"
        assert window.favorite_tab_bar.count() == 1
        assert window.favorite_tab_bar.tabData(0) == str(path_b)

        default_index = next(
            index
            for index in range(window.favorite_group_bar.count())
            if window.favorite_group_bar.tabData(index) == app_module.FAVORITE_DEFAULT_GROUP
        )
        window.favorite_group_bar.setCurrentIndex(default_index)

        assert window.current_favorite_group == app_module.FAVORITE_DEFAULT_GROUP
        assert window.favorite_tab_bar.count() == 1
        assert window.favorite_tab_bar.tabData(0) == str(path_a)
        assert window.favorite_tab_bar.tabText(0) == "Top SEM"
    finally:
        window.close()


def test_export_favorite_images_writes_visible_annotated_pngs(tmp_path, monkeypatch):
    path_a = tmp_path / "a.png"
    path_b = tmp_path / "b.png"
    export_dir = tmp_path / "exports"
    export_dir.mkdir()
    cv2.imwrite(str(path_a), np.zeros((80, 120, 3), dtype=np.uint8))
    cv2.imwrite(str(path_b), np.full((80, 120, 3), 255, dtype=np.uint8))
    monkeypatch.setattr(app_module.QFileDialog, "getExistingDirectory", lambda *args, **kwargs: str(export_dir))

    _app()
    window = MainWindow()
    try:
        window.browser_root = tmp_path
        window.browser_image_paths = [str(path_a), str(path_b)]
        window.current_browser_index = 0
        window._load_image_path(str(path_a), preserve_calibration=False)
        window._create_edge_line((20.0, 20.0), (90.0, 20.0))
        window.add_favorite_image(str(path_a))
        window.favorite_image_labels[str(path_a)] = "First SEM"

        window._load_image_path(str(path_b), preserve_calibration=True)
        window._create_edge_line((30.0, 30.0), (100.0, 30.0))
        window.add_favorite_image(str(path_b))
        window.favorite_image_labels[str(path_b)] = "Second SEM"

        window.export_favorite_images()

        first = export_dir / "First SEM.png"
        second = export_dir / "Second SEM.png"
        assert first.exists()
        assert second.exists()
        assert first.stat().st_size > 0
        assert second.stat().st_size > 0
        assert window.image_path == str(path_b)
        assert window.save_notification_label.text() == "즐겨찾기 이미지 내보내기 완료"
    finally:
        window.close()


def test_save_project_includes_all_loaded_image_states(tmp_path, monkeypatch):
    path_a = tmp_path / "a.png"
    path_b = tmp_path / "b.png"
    project_path = tmp_path / "folder_project.anglecal.json"
    cv2.imwrite(str(path_a), np.zeros((80, 120, 3), dtype=np.uint8))
    cv2.imwrite(str(path_b), np.full((80, 120, 3), 255, dtype=np.uint8))
    monkeypatch.setattr(app_module.QFileDialog, "getSaveFileName", lambda *args, **kwargs: (str(project_path), "Angle Cal Project (*.anglecal.json)"))

    _app()
    window = MainWindow()
    try:
        window.browser_root = tmp_path
        window.browser_image_paths = [str(path_a), str(path_b)]
        window.favorite_image_paths = [str(path_a)]
        window.favorite_image_labels = {str(path_a): "Top SEM"}
        window.favorite_image_groups = {str(path_a): "Review"}
        window.favorite_group_order = [app_module.FAVORITE_DEFAULT_GROUP, "Review"]
        window.current_favorite_group = "Review"
        window.scale_presets = [
            ScalePreset("50 nm", nm_per_px=1.25, bar_px=40.0, bar_nm=50.0, start=(12.0, 18.0), end=(52.0, 18.0))
        ]
        window.structure_templates = [
            StructureTemplate(
                name="Gate CD",
                cd_segment_mode="odd",
                records=[LineRecord("T1", "edge", (10.0, 20.0), (80.0, 20.0), angle_sector=3)],
            )
        ]
        window.current_browser_index = 0
        window._load_image_path(str(path_a), preserve_calibration=False)
        window._create_edge_line((20.0, 20.0), (90.0, 20.0))

        window._load_image_path(str(path_b), preserve_calibration=True)
        window._create_edge_line((30.0, 30.0), (100.0, 30.0))

        window.save_project()

        payload = json.loads(project_path.read_text(encoding="utf-8"))
        assert payload["project_format_version"] == 2
        assert payload["image_path"] == str(path_b)
        assert payload["browser_root"] == str(tmp_path)
        assert payload["browser_image_paths"] == [str(path_a), str(path_b)]
        assert payload["favorite_image_paths"] == [str(path_a)]
        assert payload["favorite_image_labels"] == {str(path_a): "Top SEM"}
        assert payload["favorite_image_groups"] == {str(path_a): "Review"}
        assert payload["favorite_group_order"] == ["Review"]
        assert payload["current_favorite_group"] == "Review"
        assert payload["scale_presets"] == [
            {"name": "50 nm", "nm_per_px": 1.25, "bar_px": 40.0, "bar_nm": 50.0, "start": [12.0, 18.0], "end": [52.0, 18.0]}
        ]
        assert payload["structure_templates"][0]["name"] == "Gate CD"
        assert payload["structure_templates"][0]["cd_segment_mode"] == "odd"
        assert payload["structure_templates"][0]["records"][0]["id"] == "T1"
        assert payload["structure_templates"][0]["records"][0]["angle_sector"] == 3
        assert set(payload["image_states"]) == {str(path_a), str(path_b)}
        assert len(payload["image_states"][str(path_a)]["records"]) == 1
        assert len(payload["image_states"][str(path_b)]["records"]) == 1
        assert payload["records"] == payload["image_states"][str(path_b)]["records"]
    finally:
        window.close()


def test_save_project_as_new_prompts_even_with_existing_project_path(tmp_path, monkeypatch):
    image_path = tmp_path / "a.png"
    old_project_path = tmp_path / "old_project.anglecal.json"
    new_project_path = tmp_path / "new_project.anglecal.json"
    cv2.imwrite(str(image_path), np.zeros((80, 120, 3), dtype=np.uint8))
    dialogs: list[str] = []

    def save_dialog(_parent, title, *args, **kwargs):
        dialogs.append(title)
        return (str(new_project_path), "Angle Cal Project (*.anglecal.json)")

    monkeypatch.setattr(app_module.QFileDialog, "getSaveFileName", save_dialog)
    _app()
    window = MainWindow()
    try:
        window.browser_root = tmp_path
        window.browser_image_paths = [str(image_path)]
        window.current_browser_index = 0
        window.project_path = str(old_project_path)
        window._load_image_path(str(image_path), preserve_calibration=False)
        window._create_edge_line((20.0, 20.0), (90.0, 20.0))

        window.save_project_as_new()

        assert dialogs == ["새 프로젝트로 저장"]
        assert window.project_path == str(new_project_path)
        assert not old_project_path.exists()
        payload = json.loads(new_project_path.read_text(encoding="utf-8"))
        assert payload["project_format_version"] == 2
        assert len(payload["image_states"][str(image_path)]["records"]) == 1
    finally:
        window.close()


def test_switching_images_auto_saves_image_format_sidecar(tmp_path):
    path_a = tmp_path / "a.png"
    path_b = tmp_path / "b.png"
    cv2.imwrite(str(path_a), np.zeros((80, 120, 3), dtype=np.uint8))
    cv2.imwrite(str(path_b), np.full((80, 120, 3), 255, dtype=np.uint8))

    _app()
    window = MainWindow()
    try:
        window.browser_root = tmp_path
        window.browser_image_paths = [str(path_a), str(path_b)]
        window.current_browser_index = 0
        window._load_image_path(str(path_a), preserve_calibration=False)
        window._create_edge_line((20.0, 20.0), (90.0, 20.0))

        window._load_image_path(str(path_b), preserve_calibration=True)

        format_path = window._image_format_path(str(path_a))
        payload = json.loads(format_path.read_text(encoding="utf-8"))
        assert payload["angle_cal_format_version"] == 1
        assert payload["image_path"] == str(path_a)
        assert len(payload["image_state"]["records"]) == 1
    finally:
        window.close()


def test_loading_image_restores_image_format_sidecar(tmp_path):
    path = tmp_path / "a.png"
    cv2.imwrite(str(path), np.zeros((80, 120, 3), dtype=np.uint8))
    edge = LineRecord("E1", "edge", (20.0, 20.0), (90.0, 20.0))
    format_path = MainWindow._image_format_path(str(path))
    format_path.write_text(
        json.dumps(
            {
                "angle_cal_format_version": 1,
                "image_path": str(path),
                "image_state": {
                    "records": [app_module.asdict(edge)],
                    "counter": 2,
                    "nm_per_px": 1.25,
                    "hidden_angle_measurements": [],
                    "image_adjustments": {},
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    _app()
    window = MainWindow()
    try:
        window._load_image_path(str(path), preserve_calibration=False)

        assert "E1" in window.records
        assert window.nm_per_px == 1.25
        assert window.image_states[str(path)]["counter"] == 2
    finally:
        window.close()


def test_smart_save_writes_image_format_without_project(tmp_path, monkeypatch):
    path = tmp_path / "a.png"
    cv2.imwrite(str(path), np.zeros((80, 120, 3), dtype=np.uint8))

    def fail_save_dialog(*args, **kwargs):
        raise AssertionError("smart save without project should not ask for a project path")

    monkeypatch.setattr(app_module.QFileDialog, "getSaveFileName", fail_save_dialog)
    _app()
    window = MainWindow()
    try:
        window._load_image_path(str(path), preserve_calibration=False)
        window._create_edge_line((20.0, 20.0), (90.0, 20.0))

        window.smart_save()

        format_path = window._image_format_path(str(path))
        payload = json.loads(format_path.read_text(encoding="utf-8"))
        assert len(payload["image_state"]["records"]) == 1
        assert window.project_path is None
        assert window.save_notification_label.text() == "이미지 저장 완료"
        assert not window.save_notification_label.isHidden()
    finally:
        window.close()


def test_smart_save_uses_existing_project_path(tmp_path):
    image_path = tmp_path / "a.png"
    project_path = tmp_path / "folder_project.anglecal.json"
    cv2.imwrite(str(image_path), np.zeros((80, 120, 3), dtype=np.uint8))

    _app()
    window = MainWindow()
    try:
        window.browser_root = tmp_path
        window.browser_image_paths = [str(image_path)]
        window.current_browser_index = 0
        window.project_path = str(project_path)
        window._load_image_path(str(image_path), preserve_calibration=False)
        window._create_edge_line((20.0, 20.0), (90.0, 20.0))

        window.smart_save()

        payload = json.loads(project_path.read_text(encoding="utf-8"))
        assert payload["project_format_version"] == 2
        assert payload["browser_image_paths"] == [str(image_path)]
        assert len(payload["image_states"][str(image_path)]["records"]) == 1
        assert window.save_notification_label.text() == "프로젝트 저장 완료"
        assert not window.save_notification_label.isHidden()
    finally:
        window.close()


def test_open_project_restores_all_loaded_image_states(tmp_path, monkeypatch):
    path_a = tmp_path / "a.png"
    path_b = tmp_path / "b.png"
    project_path = tmp_path / "folder_project.anglecal.json"
    cv2.imwrite(str(path_a), np.zeros((80, 120, 3), dtype=np.uint8))
    cv2.imwrite(str(path_b), np.full((80, 120, 3), 255, dtype=np.uint8))
    edge_a = LineRecord("E_a", "edge", (20.0, 20.0), (90.0, 20.0))
    edge_b = LineRecord("E_b", "edge", (30.0, 30.0), (100.0, 30.0))
    payload = {
        "project_format_version": 2,
        "image_path": str(path_b),
        "browser_root": str(tmp_path),
        "browser_image_paths": [str(path_a), str(path_b)],
        "favorite_image_paths": [str(path_a)],
        "favorite_image_labels": {str(path_a): "Top SEM"},
        "favorite_image_groups": {str(path_a): "Review"},
        "favorite_group_order": [app_module.FAVORITE_DEFAULT_GROUP, "Review"],
        "current_favorite_group": "Review",
        "scale_presets": [
            {"name": "50 nm", "nm_per_px": 1.25, "bar_px": 40.0, "bar_nm": 50.0, "start": [12.0, 18.0], "end": [52.0, 18.0]}
        ],
        "structure_templates": [
            {
                "name": "Gate CD",
                "cd_segment_mode": "even",
                "records": [app_module.asdict(LineRecord("T1", "edge", (10.0, 20.0), (80.0, 20.0), angle_sector=3))],
            }
        ],
        "current_browser_index": 1,
        "image_states": {
            str(path_a): {"records": [app_module.asdict(edge_a)], "counter": 2, "nm_per_px": 1.5, "hidden_angle_measurements": [], "image_adjustments": {}},
            str(path_b): {"records": [app_module.asdict(edge_b)], "counter": 2, "nm_per_px": 2.0, "hidden_angle_measurements": [], "image_adjustments": {}},
        },
        "records": [app_module.asdict(edge_b)],
        "counter": 2,
        "nm_per_px": 2.0,
    }
    project_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(app_module.QFileDialog, "getOpenFileName", lambda *args, **kwargs: (str(project_path), "Angle Cal Project (*.anglecal.json)"))

    _app()
    window = MainWindow()
    try:
        window.open_project()

        assert window.project_path == str(project_path)
        assert window.browser_root == tmp_path
        assert window.browser_image_paths == [str(path_a), str(path_b)]
        assert window.favorite_image_paths == [str(path_a)]
        assert window.favorite_image_labels == {str(path_a): "Top SEM"}
        assert window.favorite_image_groups == {str(path_a): "Review"}
        assert window.favorite_group_order == [app_module.FAVORITE_DEFAULT_GROUP, "Review"]
        assert window.current_favorite_group == "Review"
        assert window.favorite_group_bar.count() == 1
        assert window.favorite_group_bar.tabText(0) == "Review"
        assert window.favorite_tab_bar.count() == 1
        assert window.favorite_tab_bar.tabData(0) == str(path_a)
        assert window.favorite_tab_bar.tabText(0) == "Top SEM"
        assert window.current_browser_index == 1
        assert len(window.scale_presets) == 1
        assert window.scale_presets[0].name == "50 nm"
        assert window.scale_presets[0].start == (12.0, 18.0)
        assert window.scale_presets[0].end == (52.0, 18.0)
        assert window.scale_preset_table.rowCount() == 1
        assert len(window.structure_templates) == 1
        assert window.structure_templates[0].name == "Gate CD"
        assert window.structure_templates[0].cd_segment_mode == "even"
        assert window.structure_templates[0].records[0].angle_sector == 3
        assert window.structure_combo.itemText(1) == "Gate CD"
        assert set(window.image_states) == {str(path_a), str(path_b)}
        assert "E_b" in window.records
        assert window.nm_per_px == 2.0

        window.load_browser_image(str(path_a))

        assert window.project_path == str(project_path)
        assert window.current_browser_index == 0
        assert "E_a" in window.records
        assert window.nm_per_px == 1.5
    finally:
        window.close()


def test_open_project_sorts_nested_browser_paths_naturally(tmp_path, monkeypatch):
    path_1 = tmp_path / "000" / "1" / "a.png"
    path_10 = tmp_path / "000" / "10" / "a.png"
    path_2 = tmp_path / "000" / "2" / "a.png"
    for path in [path_1, path_10, path_2]:
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), np.zeros((30, 30, 3), dtype=np.uint8))
    project_path = tmp_path / "folder_project.anglecal.json"
    payload = {
        "project_format_version": 2,
        "image_path": str(path_10),
        "browser_root": str(tmp_path),
        "browser_image_paths": [str(path_1), str(path_10), str(path_2)],
        "current_browser_index": 1,
        "image_states": {
            str(path_1): {"records": [], "counter": 1, "nm_per_px": None, "hidden_angle_measurements": [], "image_adjustments": {}},
            str(path_10): {"records": [], "counter": 1, "nm_per_px": None, "hidden_angle_measurements": [], "image_adjustments": {}},
            str(path_2): {"records": [], "counter": 1, "nm_per_px": None, "hidden_angle_measurements": [], "image_adjustments": {}},
        },
        "records": [],
        "counter": 1,
        "nm_per_px": None,
    }
    project_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(app_module.QFileDialog, "getOpenFileName", lambda *args, **kwargs: (str(project_path), "Angle Cal Project (*.anglecal.json)"))

    _app()
    window = MainWindow()
    try:
        window.open_project()

        assert [str(Path(path).relative_to(tmp_path)) for path in window.browser_image_paths] == [
            "000/1/a.png",
            "000/2/a.png",
            "000/10/a.png",
        ]
        assert window.image_path == str(path_10)
        assert window.current_browser_index == 2
    finally:
        window.close()


def test_selected_object_visibility_can_be_mixed_and_applied():
    window = _window_with_edge_image()
    try:
        window._create_reference_line((10.0, 10.0), (100.0, 10.0))
        window._create_edge_line((40.0, 20.0), (40.0, 100.0))
        window._create_edge_line((80.0, 20.0), (80.0, 100.0))
        edges = [record for record in window.records.values() if record.kind == "edge"]
        edges[0].show_line_angle = True
        edges[1].show_line_angle = False
        window.canvas.redraw_lines(list(window.records.values()))
        for edge in edges:
            window.canvas.line_items[edge.id].setSelected(True)

        window._update_object_visibility_controls()

        checkbox = window.object_visibility_checkboxes["show_line_angle"]
        assert checkbox.checkState() == Qt.CheckState.PartiallyChecked

        checkbox.setCheckState(Qt.CheckState.Checked)
        assert all(edge.show_line_angle for edge in edges)
        window.calculate_angles(reset_hidden=True)
        assert len(window.canvas.angle_items) == 2

        checkbox.setCheckState(Qt.CheckState.Unchecked)
        assert not any(edge.show_line_angle for edge in edges)
        window.calculate_angles(reset_hidden=True)
        assert len(window.canvas.angle_items) == 0
        assert len([row for row in window.last_measurements if row["kind"] == "edge_to_reference"]) == 2
    finally:
        window.close()


def test_selected_object_visibility_can_hide_selected_edge_line():
    window = _window_with_edge_image()
    try:
        window._create_edge_line((40.0, 20.0), (40.0, 100.0))
        edge = next(record for record in window.records.values() if record.kind == "edge")
        window.canvas.redraw_lines(list(window.records.values()))
        item = window.canvas.line_items[edge.id]
        item.setSelected(True)

        checkbox = window.object_visibility_checkboxes["show_line"]
        checkbox.setCheckState(Qt.CheckState.Unchecked)

        assert edge.show_line is False
        assert item.isVisible() is False
    finally:
        window.close()


def test_angle_visibility_splits_line_intersection_and_arc():
    window = _window_with_edge_image()
    try:
        window._create_edge_line((70.0, 20.0), (70.0, 100.0))
        window._create_guide_line((10.0, 60.0), (120.0, 60.0))
        window.canvas.redraw_lines(list(window.records.values()))
        window.calculate_angles(reset_hidden=True)

        line_labels = [
            item
            for item in window.canvas.angle_items
            if isinstance(item, QGraphicsTextItem) and item.data(app_module.ANGLE_TYPE_KEY) == "line"
        ]
        intersection_labels = [
            item
            for item in window.canvas.angle_items
            if isinstance(item, QGraphicsTextItem) and item.data(app_module.ANGLE_TYPE_KEY) == "intersection"
        ]
        arcs = [item for item in window.canvas.angle_items if isinstance(item, QGraphicsPathItem)]
        assert len(line_labels) == 1
        assert len(intersection_labels) == 1
        assert len(arcs) == 1

        window.set_visibility("line_angle", False)
        assert not line_labels[0].isVisible()
        assert intersection_labels[0].isVisible()
        assert arcs[0].isVisible()

        window.set_visibility("intersection_angle", False)
        assert not intersection_labels[0].isVisible()
        assert arcs[0].isVisible()

        window.set_visibility("angle_arc", False)
        assert not arcs[0].isVisible()
    finally:
        window.close()


def test_selected_object_visibility_can_hide_angle_arc_only():
    window = _window_with_edge_image()
    try:
        window._create_edge_line((70.0, 20.0), (70.0, 100.0))
        window._create_guide_line((10.0, 60.0), (120.0, 60.0))
        edge = next(record for record in window.records.values() if record.kind == "edge")
        window.canvas.redraw_lines(list(window.records.values()))
        window.canvas.line_items[edge.id].setSelected(True)

        checkbox = window.object_visibility_checkboxes["show_angle_arc"]
        checkbox.setCheckState(Qt.CheckState.Unchecked)
        window.calculate_angles(reset_hidden=True)

        assert edge.show_angle_arc is False
        assert not [item for item in window.canvas.angle_items if isinstance(item, QGraphicsPathItem)]
        assert [
            item
            for item in window.canvas.angle_items
            if isinstance(item, QGraphicsTextItem) and item.data(app_module.ANGLE_TYPE_KEY) == "intersection"
        ]
    finally:
        window.close()


def test_delete_line_does_not_create_new_angle_annotations():
    window = _window_with_edge_image()
    try:
        window._create_reference_line((10.0, 10.0), (120.0, 10.0))
        window._create_edge_line((40.0, 20.0), (40.0, 100.0))
        window._create_edge_line((80.0, 20.0), (80.0, 100.0))
        edges = [record for record in window.records.values() if record.kind == "edge"]
        window.canvas.redraw_lines(list(window.records.values()))
        window.canvas.line_items[edges[0].id].setSelected(True)

        window.delete_selected()

        assert edges[0].id not in window.records
        assert edges[1].id in window.records
        assert len(window.canvas.angle_items) == 0
        assert len([row for row in window.last_measurements if row["kind"] == "edge_to_reference"]) == 1
    finally:
        window.close()


def test_delete_line_preserves_only_already_visible_angle_annotations():
    window = _window_with_edge_image()
    try:
        window._create_reference_line((10.0, 10.0), (120.0, 10.0))
        window._create_edge_line((40.0, 20.0), (40.0, 100.0))
        window._create_edge_line((80.0, 20.0), (80.0, 100.0))
        edges = [record for record in window.records.values() if record.kind == "edge"]
        window.canvas.redraw_lines(list(window.records.values()))
        window.calculate_angles(reset_hidden=True)
        assert len(window.canvas.angle_items) == 2

        window.canvas.line_items[edges[0].id].setSelected(True)
        window.delete_selected()

        remaining_measurement_ids = {
            str(item.data(app_module.ANGLE_MEASUREMENT_KEY)) for item in window.canvas.angle_items
        }
        assert remaining_measurement_ids == {f"{edges[1].id}_to_{window._reference_record().id}"}
    finally:
        window.close()


def test_style_toolbar_applies_to_selected_line_objects_and_new_defaults():
    window = _window_with_edge_image()
    try:
        window._create_edge_line((40.0, 20.0), (40.0, 100.0))
        window._create_guide_line((10.0, 60.0), (120.0, 60.0))
        window.canvas.redraw_lines(list(window.records.values()))
        edge = next(record for record in window.records.values() if record.kind == "edge")
        guide = next(record for record in window.records.values() if record.kind == "guide")
        window.canvas.line_items[edge.id].setSelected(True)
        window.canvas.line_items[guide.id].setSelected(True)

        window.stroke_color_combo.setCurrentIndex(window.stroke_color_combo.findData("#ffd166"))
        window.stroke_width_spin.setValue(4.0)

        assert window.records[edge.id].stroke_color == "#ffd166"
        assert window.records[guide.id].stroke_color == "#ffd166"
        assert window.records[edge.id].stroke_width == 4.0
        assert window.canvas.line_items[guide.id].pen().widthF() == 4.0

        window.canvas.scene.clearSelection()
        window.stroke_color_combo.setCurrentIndex(window.stroke_color_combo.findData("#4cc9f0"))
        window.stroke_width_spin.setValue(3.0)
        window._create_edge_line((80.0, 20.0), (80.0, 100.0))
        new_edge = max((record for record in window.records.values() if record.kind == "edge"), key=lambda record: record.id)

        assert new_edge.stroke_color == "#4cc9f0"
        assert new_edge.stroke_width == 3.0
    finally:
        window.close()


def test_top_controls_are_grouped_in_ribbon_tabs():
    window = _window_with_edge_image()
    try:
        assert window.ribbon_tabs.count() == 5
        assert [window.ribbon_tabs.tabText(idx) for idx in range(window.ribbon_tabs.count())] == ["파일", "경계", "가이드/측정", "이미지/표시/서식", "구조"]
        ribbon_layout = window.menuWidget().layout()
        assert ribbon_layout.itemAt(0).widget() is window.ribbon_tabs
        assert ribbon_layout.itemAt(1).layout() is not None
        assert window.tool_buttons["edge"].text() == "경계선"
        assert window.structure_combo.itemText(0) == "구조 선택"
        edge_groups = window.ribbon_tabs.widget(1).findChildren(QGroupBox)
        assert [group.title() for group in edge_groups] == ["경계 인식"]
        edge_labels = [label.text() for label in edge_groups[0].findChildren(QLabel)]
        assert "경계 형태" not in edge_labels
        assert [window.boundary_snap_combo.itemText(idx) for idx in range(window.boundary_snap_combo.count())] == [
            "기울기 최대",
            "밝은 꼭대기",
            "어두운 골",
            "좌측 급경사",
            "우측 급경사",
        ]
        assert window.boundary_snap_combo.currentData() == "max_gradient"
        display_groups = window.ribbon_tabs.widget(3).findChildren(QGroupBox)
        assert [group.title() for group in display_groups][:2] == ["기준선", "이미지 회전"]
        rotation_group = next(group for group in display_groups if group.title() == "이미지 회전")
        assert [button.text() for button in rotation_group.findChildren(QPushButton)] == ["90° 회전", "회전"]
        assert window.thumbnail_columns_combo.findData(3) >= 0
        file_groups = window.ribbon_tabs.widget(0).findChildren(QGroupBox)
        file_group_titles = [group.title() for group in file_groups]
        assert file_group_titles == ["불러오기 / 저장", "내보내기"]
        export_group = next(group for group in file_groups if group.title() == "내보내기")
        assert [button.text() for button in export_group.findChildren(QPushButton)] == ["Data Export", "즐겨찾기 이미지 내보내기"]
    finally:
        window.close()


def test_line_angle_and_edge_length_visibility_are_off_by_default():
    window = _window_with_edge_image()
    try:
        assert window.visibility["line_angle"] is False
        assert window.visibility["edge_length"] is False
        assert window.visibility_checkboxes["line_angle"].isChecked() is False
        assert window.visibility_checkboxes["edge_length"].isChecked() is False
    finally:
        window.close()


def test_copy_format_then_ctrl_v_pastes_style_to_selected_objects():
    window = _window_with_edge_image()
    try:
        window._create_edge_line((40.0, 20.0), (40.0, 100.0))
        window._create_edge_line((80.0, 20.0), (80.0, 100.0))
        window.canvas.redraw_lines(list(window.records.values()))
        edges = [record for record in window.records.values() if record.kind == "edge"]
        source, target = edges[0], edges[1]
        source.stroke_color = "#ffd166"
        source.stroke_width = 4.0
        source.angle_sector = 2
        source.angle_label_font_size = 18.0
        target.stroke_color = "#4cc9f0"
        target.stroke_width = 2.0
        window.canvas.redraw_lines(list(window.records.values()))

        window.canvas.line_items[source.id].setSelected(True)
        window.copy_selected_format()
        window.canvas.scene.clearSelection()
        window.canvas.line_items[target.id].setSelected(True)
        window.paste_from_clipboard()

        assert window.records[target.id].stroke_color == "#ffd166"
        assert window.records[target.id].stroke_width == 4.0
        assert window.records[target.id].angle_sector == 2
        assert window.records[target.id].angle_label_font_size == 18.0
        assert window.canvas.line_items[target.id].pen().widthF() == 4.0
    finally:
        window.close()


def test_ctrl_pan_restore_returns_to_edge_cursor():
    window = _window_with_edge_image()
    try:
        window.set_current_tool("edge")
        window.canvas._start_pan(window.canvas.viewport().rect().center())
        assert window.canvas.cursor().shape() == Qt.CursorShape.ClosedHandCursor

        window.canvas._panning = False
        window.canvas._restore_tool_cursor()

        assert window.canvas.cursor().shape() == Qt.CursorShape.ArrowCursor
        assert window.canvas.current_tool == "edge"
    finally:
        window.close()


def test_grouped_objects_select_and_move_together():
    window = _window_with_edge_image()
    try:
        window._create_edge_line((20.0, 20.0), (20.0, 80.0))
        window._create_edge_line((60.0, 20.0), (60.0, 80.0))
        edges = [record for record in window.records.values() if record.kind == "edge"]
        window.canvas.redraw_lines(list(window.records.values()))
        for edge in edges:
            window.canvas.line_items[edge.id].setSelected(True)

        window.group_selected_objects()

        assert edges[0].object_group
        assert edges[0].object_group == edges[1].object_group
        assert len(window.canvas.group_box_items) == 1
        original_box = window.canvas.group_box_items[0].rect()

        window.canvas.scene.clearSelection()
        window.canvas.line_items[edges[0].id].setSelected(True)

        assert set(window.canvas.selected_line_ids()) == {edges[0].id, edges[1].id}

        assert window.canvas._nudge_selected_items(Qt.Key.Key_Right, Qt.KeyboardModifier.NoModifier)
        window._sync_records_from_canvas()

        assert window.records[edges[0].id].start == (30.0, 20.0)
        assert window.records[edges[1].id].start == (70.0, 20.0)
        moved_box = window.canvas.group_box_items[0].rect()
        assert moved_box.left() > original_box.left()
    finally:
        window.close()


def test_group_box_drag_moves_grouped_objects():
    window = _window_with_edge_image()
    try:
        window._create_edge_line((20.0, 20.0), (20.0, 80.0))
        window._create_edge_line((60.0, 20.0), (60.0, 80.0))
        edges = [record for record in window.records.values() if record.kind == "edge"]
        window.canvas.redraw_lines(list(window.records.values()))
        for edge in edges:
            window.canvas.line_items[edge.id].setSelected(True)
        window.group_selected_objects()
        window.canvas.scene.clearSelection()
        box = window.canvas.group_box_items[0]
        view_pos = window.canvas.mapFromScene(box.rect().center())

        assert set(window.canvas._group_box_record_ids_at(view_pos)) == {edge.id for edge in edges}
        assert window.canvas._begin_group_box_drag(view_pos, Qt.KeyboardModifier.NoModifier)
        assert set(window.canvas.selected_line_ids()) == {edge.id for edge in edges}

        window.canvas._apply_object_drag_delta(QPointF(15.0, 5.0))
        window.canvas._object_drag_moved = True
        window.canvas._finish_object_drag(Qt.KeyboardModifier.NoModifier)

        assert window.records[edges[0].id].start == (35.0, 25.0)
        assert window.records[edges[1].id].start == (75.0, 25.0)
    finally:
        window.close()


def test_axis_locked_drag_delta_uses_dominant_direction():
    horizontal = app_module.AngleCanvas._axis_locked_delta(QPointF(24.0, 6.0))
    vertical = app_module.AngleCanvas._axis_locked_delta(QPointF(4.0, -18.0))

    assert horizontal == QPointF(24.0, 0.0)
    assert vertical == QPointF(0.0, -18.0)


def test_drag_copy_duplicates_selected_edge_group_and_guide():
    window = _window_with_edge_image()
    try:
        window._create_edge_line((20.0, 20.0), (20.0, 80.0))
        window._create_edge_line((60.0, 20.0), (60.0, 80.0))
        window._create_guide_line((0.0, 50.0), (120.0, 50.0))
        window.canvas.redraw_lines(list(window.records.values()))
        edges = [record for record in window.records.values() if record.kind == "edge"]
        guide = next(record for record in window.records.values() if record.kind == "guide")
        for record in [*edges, guide]:
            window.canvas.line_items[record.id].setSelected(True)

        window.group_selected_objects()
        original_group = edges[0].object_group
        assert original_group

        window.duplicate_dragged_objects([edges[0].id, edges[1].id, guide.id], 12.0, 0.0)

        copied_edges = [
            record
            for record in window.records.values()
            if record.kind == "edge" and record.id not in {edge.id for edge in edges}
        ]
        copied_guides = [record for record in window.records.values() if record.kind == "guide" and record.id != guide.id]
        copied_group_ids = {record.object_group for record in copied_edges + copied_guides}
        copied_edge_starts = sorted(record.start for record in copied_edges)

        assert len(copied_edges) == 2
        assert len(copied_guides) == 1
        assert copied_edge_starts == [(32.0, 20.0), (72.0, 20.0)]
        assert copied_guides[0].start == (12.0, 50.0)
        assert len(copied_group_ids) == 1
        assert None not in copied_group_ids
        assert copied_group_ids != {original_group}
        assert {record.id for record in copied_edges + copied_guides} == set(window.canvas.selected_line_ids())
    finally:
        window.close()


def test_ungroup_selected_objects_stops_group_selection():
    window = _window_with_edge_image()
    try:
        window._create_edge_line((20.0, 20.0), (20.0, 80.0))
        window._create_edge_line((60.0, 20.0), (60.0, 80.0))
        edges = [record for record in window.records.values() if record.kind == "edge"]
        window.canvas.redraw_lines(list(window.records.values()))
        for edge in edges:
            window.canvas.line_items[edge.id].setSelected(True)
        window.group_selected_objects()

        window.ungroup_selected_objects()
        assert not any(edge.object_group for edge in edges)
        assert len(window.canvas.group_box_items) == 0

        window.canvas.scene.clearSelection()
        window.canvas.line_items[edges[0].id].setSelected(True)

        assert window.canvas.selected_line_ids() == [edges[0].id]
    finally:
        window.close()


def test_ctrl_rubberband_restores_previous_selection_as_additive():
    window = _window_with_edge_image()
    try:
        window._create_edge_line((20.0, 20.0), (20.0, 80.0))
        window._create_edge_line((60.0, 20.0), (60.0, 80.0))
        edges = [record for record in window.records.values() if record.kind == "edge"]
        window.canvas.redraw_lines(list(window.records.values()))
        first_item = window.canvas.line_items[edges[0].id]
        second_item = window.canvas.line_items[edges[1].id]
        first_item.setSelected(True)

        window.canvas._additive_rubberband_items = {first_item}
        window.canvas.scene.clearSelection()
        second_item.setSelected(True)
        window.canvas._restore_additive_rubberband_selection()

        assert set(window.canvas.selected_line_ids()) == {edges[0].id, edges[1].id}
    finally:
        window.close()


def test_arrow_keys_nudge_selected_edge_and_ctrl_uses_one_pixel():
    window = _window_with_edge_image()
    try:
        window._create_edge_line((20.0, 60.0), (120.0, 60.0))
        window.canvas.redraw_lines(list(window.records.values()))
        edge = next(record for record in window.records.values() if record.kind == "edge")
        window.canvas.line_items[edge.id].setSelected(True)

        assert window.canvas._nudge_selected_items(Qt.Key.Key_Right, Qt.KeyboardModifier.NoModifier)
        window._sync_records_from_canvas()
        assert window.records[edge.id].start == (30.0, 60.0)
        assert window.records[edge.id].end == (130.0, 60.0)

        assert window.canvas._nudge_selected_items(Qt.Key.Key_Up, Qt.KeyboardModifier.ControlModifier)
        window._sync_records_from_canvas()
        assert window.records[edge.id].start == (30.0, 59.0)
        assert window.records[edge.id].end == (130.0, 59.0)
    finally:
        window.close()


def test_nudging_selected_line_keeps_all_point_handles_attached():
    window = _window_with_edge_image()
    try:
        window._create_edge_line((20.0, 60.0), (120.0, 60.0))
        edge = next(record for record in window.records.values() if record.kind == "edge")
        window.canvas.redraw_lines(list(window.records.values()))
        line_item = window.canvas.line_items[edge.id]
        line_item.setSelected(True)
        window.canvas.refresh_point_handles()
        window.canvas.point_handle_items[0].setSelected(True)

        assert window.canvas._nudge_selected_items(Qt.Key.Key_Right, Qt.KeyboardModifier.NoModifier)
        window._sync_records_from_canvas()

        handle_positions = [(handle.pos().x(), handle.pos().y()) for handle in window.canvas.point_handle_items]
        assert window.records[edge.id].start == (30.0, 60.0)
        assert window.records[edge.id].end == (130.0, 60.0)
        assert handle_positions == [(30.0, 60.0), (130.0, 60.0)]
    finally:
        window.close()


def test_split_segmented_edge_selects_independent_edge_for_detection_settings():
    window = _window_with_edge_image()
    try:
        points = [(20.0, 20.0), (40.0, 40.0), (60.0, 40.0), (80.0, 60.0)]
        window._create_edge_line((20.0, 20.0), (80.0, 60.0))
        original = next(record for record in window.records.values() if record.kind == "edge")
        original.points = points
        original.recognition_points = list(points)
        original.edge_segmented = True
        window.canvas.redraw_lines(list(window.records.values()))

        window.split_edge_segment_for_selection(original.id, 1)

        edges = [record for record in window.records.values() if record.kind == "edge"]
        selected_ids = window.canvas.selected_line_ids()
        assert len(edges) == 3
        assert len(selected_ids) == 1
        selected = window.records[selected_ids[0]]
        assert record_points(selected) == [(40.0, 40.0), (60.0, 40.0)]
        assert selected.edge_mode == "line"

        window.search_radius_spin.setValue(18)
        window.curve_sensitivity_spin.setValue(5)

        assert selected.search_radius_px == 18
        assert selected.segment_size_px == 5
        assert all(edge.search_radius_px != 18 for edge in edges if edge.id != selected.id)
    finally:
        window.close()


def test_point_handles_edit_line_endpoints_and_delete_segmented_edge_points():
    window = _window_with_edge_image()
    try:
        window._create_edge_line((20.0, 60.0), (120.0, 60.0))
        line_edge = next(record for record in window.records.values() if record.kind == "edge")
        window.canvas.redraw_lines(list(window.records.values()))
        window.canvas.line_items[line_edge.id].setSelected(True)
        window.canvas.refresh_point_handles()

        assert len(window.canvas.point_handle_items) == 2
        window.canvas.point_handle_items[1].setPos(130.0, 66.0)
        window._sync_records_from_canvas()
        assert window.records[line_edge.id].end == (130.0, 66.0)

        segmented_points = [(40.0, 20.0), (55.0, 40.0), (70.0, 60.0)]
        window._create_edge_line((40.0, 20.0), (70.0, 60.0))
        poly_edge = max((record for record in window.records.values() if record.kind == "edge"), key=lambda record: record.id)
        poly_edge.points = segmented_points
        poly_edge.recognition_points = list(segmented_points)
        poly_edge.edge_segmented = True
        window.canvas.redraw_lines(list(window.records.values()))
        window.canvas.scene.clearSelection()
        window.canvas.line_items[poly_edge.id].setSelected(True)
        window.canvas.refresh_point_handles()

        assert len(window.canvas.point_handle_items) == 3
        window.canvas.point_handle_items[1].setSelected(True)
        assert window.canvas.delete_selected_point_handles()
        window._sync_records_from_canvas()
        assert len(window.records[poly_edge.id].points) == 2
    finally:
        window.close()


def test_point_handles_follow_line_move_and_can_be_hidden():
    window = _window_with_edge_image()
    try:
        window._create_edge_line((20.0, 60.0), (120.0, 60.0))
        edge = next(record for record in window.records.values() if record.kind == "edge")
        window.canvas.redraw_lines(list(window.records.values()))
        window.canvas.line_items[edge.id].setSelected(True)
        window.canvas.refresh_point_handles()

        assert len(window.canvas.point_handle_items) == 2
        assert window.canvas.point_handle_items[0].pos() == QPointF(20.0, 60.0)

        assert window.canvas._nudge_selected_items(Qt.Key.Key_Right, Qt.KeyboardModifier.NoModifier)
        assert len(window.canvas.point_handle_items) == 2
        assert window.canvas.point_handle_items[0].pos() == QPointF(30.0, 60.0)

        window.set_visibility("point_handle", False)
        assert len(window.canvas.point_handle_items) == 0

        window.set_visibility("point_handle", True)
        assert len(window.canvas.point_handle_items) == 2
    finally:
        window.close()


def test_deleting_line_clears_point_handles():
    window = _window_with_edge_image()
    try:
        window._create_edge_line((20.0, 60.0), (120.0, 60.0))
        edge = next(record for record in window.records.values() if record.kind == "edge")
        window.canvas.redraw_lines(list(window.records.values()))
        window.canvas.line_items[edge.id].setSelected(True)
        window.canvas.refresh_point_handles()

        assert len(window.canvas.point_handle_items) == 2

        window.delete_selected()

        assert edge.id not in window.records
        assert len(window.canvas.point_handle_items) == 0
        assert not [item for item in window.canvas.scene.items() if item.__class__.__name__ == "PointHandleItem"]
    finally:
        window.close()


def test_main_guide_generates_guides_from_selected_baseline():
    window = _window_with_edge_image()
    try:
        window.records["G_main"] = LineRecord(
            id="G_main",
            kind="guide",
            start=(0.0, 60.0),
            end=(150.0, 60.0),
            axis="horizontal",
        )
        window.canvas.redraw_lines(list(window.records.values()))
        window.set_main_guide("G_main")
        window.guide_spacing_spin.setValue(20)
        window.guide_spacing_unit.setCurrentIndex(window.guide_spacing_unit.findData("px"))
        window.guide_direction_combo.setCurrentIndex(window.guide_direction_combo.findData("both"))
        window.guide_count_spin.setValue(2)

        window.add_guides()

        guides = sorted(
            [record for record in window.records.values() if record.kind == "guide"],
            key=lambda record: record.start[1],
        )
        assert [round(guide.start[1], 2) for guide in guides] == [20.0, 40.0, 60.0, 80.0, 100.0]
        assert sum(1 for guide in guides if guide.is_main_guide) == 1
        assert next(guide for guide in guides if guide.is_main_guide).id == "G_main"
    finally:
        window.close()


def test_cd_only_appears_after_measure_button_then_refreshes_on_move():
    window = _window_with_edge_image()
    try:
        for x in (40.0, 80.0):
            window._create_edge_line((x, 10.0), (x, 110.0))
        window.records["G99"] = LineRecord(
            id="G99",
            kind="guide",
            start=(0.0, 60.0),
            end=(150.0, 60.0),
            axis="horizontal",
        )
        window.canvas.redraw_lines(list(window.records.values()))
        guide_item = window.canvas.line_items["G99"]
        guide_item.setSelected(True)

        guide_item.moveBy(0.0, 5.0)
        window._handle_scene_changed()

        assert not [row for row in window.last_measurements if row["kind"] == "cd_length"]
        assert not window.canvas.cd_items

        window.calculate_cd_lengths()
        assert len([row for row in window.last_measurements if row["kind"] == "cd_length"]) == 1
        assert len(window.canvas.cd_items) == 2

        guide_item.moveBy(0.0, 5.0)
        window._handle_scene_changed()
        cd_rows = [row for row in window.last_measurements if row["kind"] == "cd_length"]
        assert len(cd_rows) == 1
        assert len(window.canvas.cd_items) == 2

        window.set_visibility("cd", False)
        guide_item.moveBy(0.0, 5.0)
        window._handle_scene_changed()

        assert not window.canvas.cd_items
    finally:
        window.close()


def test_delete_prefers_parent_line_over_selected_point_handle():
    window = _window_with_edge_image()
    try:
        points = [(20.0, 20.0), (50.0, 50.0), (80.0, 80.0)]
        window._create_edge_line((20.0, 20.0), (80.0, 80.0))
        edge = next(record for record in window.records.values() if record.kind == "edge")
        edge.points = points
        edge.recognition_points = list(points)
        edge.edge_segmented = True
        window.canvas.redraw_lines(list(window.records.values()))
        window.canvas.line_items[edge.id].setSelected(True)
        window.canvas.refresh_point_handles()
        window.canvas.point_handle_items[1].setSelected(True)

        window.delete_selected()

        assert edge.id not in window.records
        assert len(window.canvas.point_handle_items) == 0
    finally:
        window.close()


def test_delete_multiple_parents_even_when_point_handles_are_selected():
    window = _window_with_edge_image()
    try:
        window._create_edge_line((20.0, 20.0), (20.0, 80.0))
        window._create_edge_line((60.0, 20.0), (60.0, 80.0))
        edges = [record for record in window.records.values() if record.kind == "edge"]
        window.canvas.redraw_lines(list(window.records.values()))
        for edge in edges:
            window.canvas.line_items[edge.id].setSelected(True)
        window.canvas.refresh_point_handles()
        window.canvas.point_handle_items[0].setSelected(True)

        window.delete_selected()

        assert not [record for record in window.records.values() if record.kind == "edge"]
        assert len(window.canvas.point_handle_items) == 0
    finally:
        window.close()


def test_undo_restores_keyboard_move_and_delete():
    window = _window_with_edge_image()
    try:
        window._create_edge_line((20.0, 60.0), (120.0, 60.0))
        window.canvas.redraw_lines(list(window.records.values()))
        edge = next(record for record in window.records.values() if record.kind == "edge")
        window.canvas.line_items[edge.id].setSelected(True)

        assert window.canvas._nudge_selected_items(Qt.Key.Key_Right, Qt.KeyboardModifier.NoModifier)
        window._sync_records_from_canvas()
        assert window.records[edge.id].start == (30.0, 60.0)

        window.undo()
        assert window.records[edge.id].start == (20.0, 60.0)
        assert window.records[edge.id].end == (120.0, 60.0)

        window.canvas.line_items[edge.id].setSelected(True)
        window.delete_selected()
        assert edge.id not in window.records

        window.undo()
        assert edge.id in window.records
        assert window.records[edge.id].start == (20.0, 60.0)
        assert window.records[edge.id].end == (120.0, 60.0)
    finally:
        window.close()


def test_undo_preserves_current_view_transform_and_center():
    _app()
    window = MainWindow()
    try:
        window.resize(420, 320)
        window.show()
        image = np.zeros((800, 1000, 3), dtype=np.uint8)
        image[:, 500:] = 255
        window.image_bgr = image
        window._show_image()
        QApplication.processEvents()

        window.canvas.scale(3.0, 3.0)
        target_center = QPointF(420.0, 360.0)
        window.canvas.centerOn(target_center)
        QApplication.processEvents()
        before_transform = window.canvas.transform()
        before_center = window.canvas.mapToScene(window.canvas.viewport().rect().center())

        window.save_undo_snapshot()
        window._create_edge_line((20.0, 20.0), (80.0, 20.0))
        assert window.records

        window.undo()
        after_transform = window.canvas.transform()
        after_center = window.canvas.mapToScene(window.canvas.viewport().rect().center())

        assert not window.records
        assert math.isclose(after_transform.m11(), before_transform.m11(), rel_tol=1e-6)
        assert math.isclose(after_transform.m22(), before_transform.m22(), rel_tol=1e-6)
        assert math.isclose(after_center.x(), before_center.x(), abs_tol=1.0)
        assert math.isclose(after_center.y(), before_center.y(), abs_tol=1.0)
    finally:
        window.close()


def test_point_handle_drag_undo_is_coalesced_to_one_step():
    window = _window_with_edge_image()
    try:
        window._create_edge_line((20.0, 60.0), (120.0, 60.0))
        edge = next(record for record in window.records.values() if record.kind == "edge")
        window.canvas.redraw_lines(list(window.records.values()))
        window.canvas.line_items[edge.id].setSelected(True)
        window.canvas.refresh_point_handles()
        handle = window.canvas.point_handle_items[0]

        window.canvas.edit_started.emit()
        handle.setPos(22.0, 61.0)
        handle.setPos(25.0, 65.0)
        window._sync_records_from_canvas()

        assert len(window.undo_stack) == 1
        assert window.records[edge.id].start == (25.0, 65.0)

        window.undo()

        assert window.records[edge.id].start == (20.0, 60.0)
        assert window.records[edge.id].end == (120.0, 60.0)
    finally:
        window.close()


def test_undo_restores_angle_label_move():
    window = _window_with_edge_image()
    try:
        window._create_reference_line((10.0, 10.0), (100.0, 10.0))
        window._create_edge_line((70.0, 20.0), (70.0, 100.0))
        window.canvas.redraw_lines(list(window.records.values()))
        window.set_visibility("line_angle", True)
        window.calculate_angles(reset_hidden=True)
        label = next(item for item in window.canvas.angle_items if item.__class__.__name__ == "QGraphicsTextItem")
        original_pos = label.pos()
        label.setSelected(True)

        assert window.canvas._nudge_selected_items(Qt.Key.Key_Right, Qt.KeyboardModifier.NoModifier)
        assert label.pos().x() == original_pos.x() + 10.0

        window.undo()
        restored_label = next(item for item in window.canvas.angle_items if item.__class__.__name__ == "QGraphicsTextItem")
        assert restored_label.pos() == original_pos
    finally:
        window.close()
