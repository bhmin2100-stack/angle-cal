import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtWidgets import QApplication

from angle_cal.app import LineRecord, MainWindow


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


def test_recognize_keeps_line_mode_straight():
    window = _window_with_edge_image()
    try:
        window.edge_mode_combo.setCurrentIndex(window.edge_mode_combo.findData("line"))
        window._create_edge_line((70.0, 20.0), (70.0, 100.0))

        window.recognize_edges()

        edge = next(record for record in window.records.values() if record.kind == "edge")
        assert edge.edge_mode == "line"
        assert edge.points is None
        assert 80 <= (edge.start[0] + edge.end[0]) / 2 <= 83
    finally:
        window.close()


def test_recognize_converts_curve_mode_to_points():
    window = _window_with_edge_image()
    try:
        window.edge_mode_combo.setCurrentIndex(window.edge_mode_combo.findData("curve"))
        window._create_edge_line((70.0, 20.0), (70.0, 100.0), [(70.0, 20.0), (70.0, 60.0), (70.0, 100.0)])

        window.recognize_edges()

        edge = next(record for record in window.records.values() if record.kind == "edge")
        assert edge.edge_mode == "curve"
        assert edge.points is not None
        assert len(edge.points) > 2
        assert 80 <= (edge.start[0] + edge.end[0]) / 2 <= 83
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
