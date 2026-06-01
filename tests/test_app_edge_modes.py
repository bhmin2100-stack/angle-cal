import os
import math
import zipfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QKeyEvent, QWheelEvent
from PySide6.QtWidgets import QApplication, QDialog, QGraphicsItem, QGraphicsPathItem, QGraphicsTextItem, QMessageBox

import angle_cal.app as app_module
from angle_cal.app import DataExportOptions, LineRecord, MainWindow, ScalePreset, StructureTemplate, record_points, structure_template_from_dict, structure_template_to_dict


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


def test_recognize_segments_line_mode_by_segment_size():
    window = _window_with_edge_image()
    try:
        window.edge_mode_combo.setCurrentIndex(window.edge_mode_combo.findData("line"))
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
        window.edge_mode_combo.setCurrentIndex(window.edge_mode_combo.findData("line"))
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
        window.edge_mode_combo.setCurrentIndex(window.edge_mode_combo.findData("line"))
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


def test_connected_line_edges_recognize_as_joined_chain():
    window = _window_with_edge_image()
    try:
        window.edge_mode_combo.setCurrentIndex(window.edge_mode_combo.findData("line"))
        window._create_edge_line((70.0, 20.0), (70.0, 60.0))
        window._create_edge_line((70.0, 60.0), (70.0, 100.0))
        _select_edges(window)

        window.recognize_edges()

        edges = [record for record in window.records.values() if record.kind == "edge"]
        assert len(edges) == 2
        assert abs(edges[0].end[0] - edges[1].start[0]) < 0.001
        assert abs(edges[0].end[1] - edges[1].start[1]) < 0.001
        assert 80 <= edges[0].end[0] <= 83
    finally:
        window.close()


def test_recognize_only_selected_edges():
    window = _window_with_edge_image()
    try:
        window.edge_mode_combo.setCurrentIndex(window.edge_mode_combo.findData("line"))
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
        window.edge_mode_combo.setCurrentIndex(window.edge_mode_combo.findData("line"))
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
        window.edge_mode_combo.setCurrentIndex(window.edge_mode_combo.findData("line"))
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
        window.edge_mode_combo.setCurrentIndex(window.edge_mode_combo.findData("line"))
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
        window.edge_mode_combo.setCurrentIndex(window.edge_mode_combo.findData("line"))
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


def test_recognize_converts_polyline_mode_to_connected_segments():
    window = _window_with_edge_image()
    try:
        window.edge_mode_combo.setCurrentIndex(window.edge_mode_combo.findData("polyline"))
        window._create_edge_line((70.0, 20.0), (70.0, 100.0), [(70.0, 20.0), (70.0, 60.0), (70.0, 100.0)])
        _select_edges(window)

        window.recognize_edges()

        edge = next(record for record in window.records.values() if record.kind == "edge")
        assert edge.edge_mode == "polyline"
        assert edge.edge_segmented is True
        assert edge.points is not None
        assert len(edge.points) > 2
        assert 80 <= (edge.start[0] + edge.end[0]) / 2 <= 83
        item = window.canvas.line_items[edge.id]
        assert item.path().elementCount() == len(edge.points)
    finally:
        window.close()


def test_segmented_edge_does_not_show_reference_angles_after_recognition():
    window = _window_with_edge_image()
    try:
        window._create_reference_line((10.0, 10.0), (100.0, 10.0))
        window.edge_mode_combo.setCurrentIndex(window.edge_mode_combo.findData("polyline"))
        window._create_edge_line((70.0, 20.0), (70.0, 100.0), [(70.0, 20.0), (70.0, 60.0), (70.0, 100.0)])
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
        window.edge_mode_combo.setCurrentIndex(window.edge_mode_combo.findData("polyline"))
        window._create_edge_line((70.0, 20.0), (70.0, 100.0), [(70.0, 20.0), (70.0, 60.0), (70.0, 100.0)])
        window.records["G99"] = LineRecord(
            id="G99",
            kind="guide",
            start=(10.0, 50.0),
            end=(140.0, 50.0),
            label="guide",
            axis="horizontal",
        )
        window.canvas.redraw_lines(list(window.records.values()))

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
        window.edge_mode_combo.setCurrentIndex(window.edge_mode_combo.findData("line"))
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


def test_detection_preview_uses_screen_sized_panel_when_zoomed():
    window = _window_with_edge_image()
    try:
        window.canvas.scale(4.0, 4.0)
        window.canvas.show_detection_preview(120, 9, "120 px")
        panel = window.canvas.detection_preview_items[0]
        screen_width = panel.sceneBoundingRect().width() * window.canvas.transform().m11()

        assert 170.0 <= screen_width <= 210.0
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
        assert "Q + 드래그" in text.toPlainText()

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


def test_angle_label_quadrant_positions():
    center = (100.0, 100.0)
    assert app_module.angle_label_position_for_sector(center, 0.0, 90.0, 20.0, "top_right", 5.0) == (125.0, 75.0)
    assert app_module.angle_label_position_for_sector(center, 0.0, 90.0, 20.0, "top_left", 5.0) == (75.0, 75.0)
    assert app_module.angle_label_position_for_sector(center, 0.0, 90.0, 20.0, "bottom_right", 5.0) == (125.0, 125.0)
    assert app_module.angle_label_position_for_sector(center, 0.0, 90.0, 20.0, "bottom_left", 5.0) == (75.0, 125.0)


def test_angle_display_edit_without_selection_applies_to_all_edges_and_default(monkeypatch):
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
            self.label_side_combo = _Value("bottom_left")
            self.label_gap_spin = _Value(9.0)
            self.label_font_size_spin = _Value(16.0)

        def exec(self):
            return QDialog.DialogCode.Accepted

    monkeypatch.setattr(app_module, "AngleDisplaySettingsDialog", _Dialog)
    window = _window_with_edge_image()
    try:
        window._create_edge_line((20.0, 20.0), (20.0, 80.0))
        window._create_edge_line((60.0, 20.0), (60.0, 80.0))
        window.canvas.redraw_lines(list(window.records.values()))

        window.edit_angle_display_for_selected_edges()

        edges = [record for record in window.records.values() if record.kind == "edge"]
        assert all(edge.angle_sector == 2 for edge in edges)
        assert all(edge.angle_arc_radius == 55.0 for edge in edges)
        assert all(edge.angle_label_side == "bottom_left" for edge in edges)
        assert all(edge.angle_label_gap == 9.0 for edge in edges)
        assert all(edge.angle_label_font_size == 16.0 for edge in edges)
        assert window.default_angle_sector == 2
        assert window.default_angle_label_font_size == 16.0

        window._create_edge_line((90.0, 20.0), (90.0, 80.0))
        newest = list(window.records.values())[-1]
        assert newest.angle_sector == 2
        assert newest.angle_arc_radius == 55.0
        assert newest.angle_label_side == "bottom_left"
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


def test_cd_label_text_and_position_can_be_edited(monkeypatch):
    class _Value:
        def __init__(self, value):
            self._value = value

        def value(self):
            return self._value

        def currentData(self):
            return self._value

    class _Dialog:
        def __init__(self, *args, **kwargs):
            self.label_side_combo = _Value("below")
            self.label_gap_spin = _Value(22.0)
            self.label_font_size_spin = _Value(17.0)

        def exec(self):
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
        assert window.cd_label_side == "below"
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
        assert len(sheets["교점각도"]) == 2
        assert len(sheets["CD길이"]) == 1
        assert sheets["CD길이"][0]["길이_px"] == 60.0

        export_path = tmp_path / "export.xlsx"
        app_module.write_xlsx(str(export_path), sheets)
        with zipfile.ZipFile(export_path) as archive:
            names = set(archive.namelist())
            assert "xl/workbook.xml" in names
            assert "xl/worksheets/sheet1.xml" in names
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


def test_edge_length_overlay_skips_polyline_edges():
    window = _window_with_edge_image()
    try:
        window.edge_mode_combo.setCurrentIndex(window.edge_mode_combo.findData("polyline"))
        window._create_edge_line((40.0, 20.0), (70.0, 60.0), [(40.0, 20.0), (55.0, 40.0), (70.0, 60.0)])
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
        assert [window.ribbon_tabs.tabText(idx) for idx in range(window.ribbon_tabs.count())] == ["파일", "경계", "가이드/측정", "표시/서식", "구조"]
        ribbon_layout = window.menuWidget().layout()
        assert ribbon_layout.itemAt(0).widget() is window.ribbon_tabs
        assert ribbon_layout.itemAt(1).layout() is not None
        assert window.tool_buttons["edge"].text() == "경계선"
        assert window.edge_mode_combo.currentData() == "line"
        assert window.structure_combo.itemText(0) == "구조 선택"
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


def test_split_polyline_segment_selects_independent_edge_for_detection_settings():
    window = _window_with_edge_image()
    try:
        window.edge_mode_combo.setCurrentIndex(window.edge_mode_combo.findData("polyline"))
        window._create_edge_line(
            (20.0, 20.0),
            (80.0, 60.0),
            [(20.0, 20.0), (40.0, 40.0), (60.0, 40.0), (80.0, 60.0)],
        )
        original = next(record for record in window.records.values() if record.kind == "edge")
        window.canvas.redraw_lines(list(window.records.values()))

        window.split_edge_segment_for_selection(original.id, 1)

        edges = [record for record in window.records.values() if record.kind == "edge"]
        selected_ids = window.canvas.selected_line_ids()
        assert len(edges) == 3
        assert len(selected_ids) == 1
        selected = window.records[selected_ids[0]]
        assert record_points(selected) == [(40.0, 40.0), (60.0, 40.0)]

        window.search_radius_spin.setValue(18)
        window.curve_sensitivity_spin.setValue(5)

        assert selected.search_radius_px == 18
        assert selected.segment_size_px == 5
        assert all(edge.search_radius_px != 18 for edge in edges if edge.id != selected.id)
    finally:
        window.close()


def test_point_handles_edit_line_endpoints_and_delete_polyline_points():
    window = _window_with_edge_image()
    try:
        window.edge_mode_combo.setCurrentIndex(window.edge_mode_combo.findData("line"))
        window._create_edge_line((20.0, 60.0), (120.0, 60.0))
        line_edge = next(record for record in window.records.values() if record.kind == "edge")
        window.canvas.redraw_lines(list(window.records.values()))
        window.canvas.line_items[line_edge.id].setSelected(True)
        window.canvas.refresh_point_handles()

        assert len(window.canvas.point_handle_items) == 2
        window.canvas.point_handle_items[1].setPos(130.0, 66.0)
        window._sync_records_from_canvas()
        assert window.records[line_edge.id].end == (130.0, 66.0)

        window.edge_mode_combo.setCurrentIndex(window.edge_mode_combo.findData("polyline"))
        window._create_edge_line((40.0, 20.0), (70.0, 60.0), [(40.0, 20.0), (55.0, 40.0), (70.0, 60.0)])
        poly_edge = max((record for record in window.records.values() if record.kind == "edge"), key=lambda record: record.id)
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


def test_moving_guide_refreshes_cd_and_respects_cd_visibility():
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
        cd_rows = [row for row in window.last_measurements if row["kind"] == "cd_length"]
        assert len(cd_rows) == 1
        assert len(window.canvas.cd_items) == 2

        window.set_visibility("cd", False)
        window.canvas.clear_cd_items()
        guide_item.moveBy(0.0, 5.0)
        window._handle_scene_changed()

        assert not window.canvas.cd_items
    finally:
        window.close()


def test_delete_prefers_parent_line_over_selected_point_handle():
    window = _window_with_edge_image()
    try:
        window.edge_mode_combo.setCurrentIndex(window.edge_mode_combo.findData("polyline"))
        window._create_edge_line((20.0, 20.0), (80.0, 80.0), [(20.0, 20.0), (50.0, 50.0), (80.0, 80.0)])
        edge = next(record for record in window.records.values() if record.kind == "edge")
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
