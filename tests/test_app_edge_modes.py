import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtCore import QPointF, Qt
from PySide6.QtWidgets import QApplication, QGraphicsItem, QGraphicsPathItem, QGraphicsTextItem

from angle_cal.app import LineRecord, MainWindow, ScalePreset, StructureTemplate, structure_template_from_dict, structure_template_to_dict


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


def test_recognize_segments_line_mode_by_segment_size():
    window = _window_with_edge_image()
    try:
        window.edge_mode_combo.setCurrentIndex(window.edge_mode_combo.findData("line"))
        window.curve_sensitivity_spin.setValue(10)
        window._create_edge_line((70.0, 20.0), (70.0, 100.0))

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

        window.recognize_edges()
        edge = next(record for record in window.records.values() if record.kind == "edge")
        first_points = list(edge.points)

        window.recognize_edges()
        second_points = list(edge.points)

        assert first_points == second_points
        assert edge.recognition_points == [(70.0, 20.0), (70.0, 100.0)]
    finally:
        window.close()


def test_connected_line_edges_recognize_as_joined_chain():
    window = _window_with_edge_image()
    try:
        window.edge_mode_combo.setCurrentIndex(window.edge_mode_combo.findData("line"))
        window._create_edge_line((70.0, 20.0), (70.0, 60.0))
        window._create_edge_line((70.0, 60.0), (70.0, 100.0))
        window.canvas.redraw_lines(list(window.records.values()))

        window.recognize_edges()

        edges = [record for record in window.records.values() if record.kind == "edge"]
        assert len(edges) == 2
        assert abs(edges[0].end[0] - edges[1].start[0]) < 0.001
        assert abs(edges[0].end[1] - edges[1].start[1]) < 0.001
        assert 80 <= edges[0].end[0] <= 83
    finally:
        window.close()


def test_recognize_converts_polyline_mode_to_connected_segments():
    window = _window_with_edge_image()
    try:
        window.edge_mode_combo.setCurrentIndex(window.edge_mode_combo.findData("polyline"))
        window._create_edge_line((70.0, 20.0), (70.0, 100.0), [(70.0, 20.0), (70.0, 60.0), (70.0, 100.0)])

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


def test_segmented_edge_shows_segment_reference_angles_after_recognition():
    window = _window_with_edge_image()
    try:
        window._create_reference_line((10.0, 10.0), (100.0, 10.0))
        window.edge_mode_combo.setCurrentIndex(window.edge_mode_combo.findData("polyline"))
        window._create_edge_line((70.0, 20.0), (70.0, 100.0), [(70.0, 20.0), (70.0, 60.0), (70.0, 100.0)])
        window.recognize_edges()

        window.calculate_angles()

        edge = next(record for record in window.records.values() if record.kind == "edge")
        assert len(window.canvas.angle_items) == len(edge.points) - 1
        assert not [row for row in window.last_measurements if row["kind"] == "edge_to_reference"]
        assert [row for row in window.last_measurements if row["kind"] == "edge_segment_to_reference"]
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


def test_search_range_band_and_label_visibility_are_independent():
    window = _window_with_edge_image()
    try:
        window._create_edge_line((70.0, 20.0), (70.0, 100.0))
        window.canvas.redraw_lines(list(window.records.values()))

        window.set_visibility("range", False)
        window.set_visibility("range_label", True)
        assert len(window.canvas.search_range_band_items) == 0
        assert len(window.canvas.search_range_label_items) == 1

        window.set_visibility("range", True)
        window.set_visibility("range_label", False)
        assert len(window.canvas.search_range_band_items) == 1
        assert len(window.canvas.search_range_label_items) == 0
    finally:
        window.close()


def test_space_temporarily_shows_hidden_edges():
    window = _window_with_edge_image()
    try:
        window._create_edge_line((70.0, 20.0), (70.0, 100.0))
        window.canvas.redraw_lines(list(window.records.values()))
        edge = next(record for record in window.records.values() if record.kind == "edge")
        item = window.canvas.line_items[edge.id]

        window.set_visibility("edge", False)
        assert not item.isVisible()

        window.set_edge_peek(True)
        assert item.isVisible()

        window.set_edge_peek(False)
        assert not item.isVisible()
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
        edge.angle_label_side = "inside"
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
        assert 149 <= guide_rows[0]["angle_deg"] <= 151
        assert len(window.canvas.angle_items) >= 2
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

        window._update_edge_length_overlay()

        assert len(window.canvas.edge_length_items) == 1
        assert "200" in window.canvas.edge_length_items[0].toHtml()

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
        window.scale_presets = [ScalePreset("100 nm", nm_per_px=2.0, bar_px=50.0, bar_nm=100.0)]

        window.apply_scale_preset(0)

        scales = [record for record in window.records.values() if record.kind == "scale"]
        assert len(scales) == 1
        assert scales[0].value_nm == 100.0
        assert scales[0].label == "100 nm"
        assert round(scales[0].end[0] - scales[0].start[0], 3) == 50.0
        assert window.nm_per_px == 2.0
    finally:
        window.close()


def test_selected_object_visibility_can_be_mixed_and_applied():
    window = _window_with_edge_image()
    try:
        window._create_reference_line((10.0, 10.0), (100.0, 10.0))
        window._create_edge_line((40.0, 20.0), (40.0, 100.0))
        window._create_edge_line((80.0, 20.0), (80.0, 100.0))
        edges = [record for record in window.records.values() if record.kind == "edge"]
        edges[0].show_angle = True
        edges[1].show_angle = False
        window.canvas.redraw_lines(list(window.records.values()))
        for edge in edges:
            window.canvas.line_items[edge.id].setSelected(True)

        window._update_object_visibility_controls()

        checkbox = window.object_visibility_checkboxes["show_angle"]
        assert checkbox.checkState() == Qt.CheckState.PartiallyChecked

        checkbox.setCheckState(Qt.CheckState.Checked)
        assert all(edge.show_angle for edge in edges)
        window.calculate_angles(reset_hidden=True)
        assert len(window.canvas.angle_items) == 2

        checkbox.setCheckState(Qt.CheckState.Unchecked)
        assert not any(edge.show_angle for edge in edges)
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


def test_undo_restores_angle_label_move():
    window = _window_with_edge_image()
    try:
        window._create_reference_line((10.0, 10.0), (100.0, 10.0))
        window._create_edge_line((70.0, 20.0), (70.0, 100.0))
        window.canvas.redraw_lines(list(window.records.values()))
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
