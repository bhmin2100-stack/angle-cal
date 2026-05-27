import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtWidgets import QApplication

from angle_cal.app import MainWindow


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
