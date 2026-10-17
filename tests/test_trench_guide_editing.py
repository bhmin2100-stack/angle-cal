import json

import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest, QSignalSpy
from PySide6.QtWidgets import QApplication, QGraphicsView

from angle_cal.trench_guides import curve_samples
from angle_cal.trench_panel import TrenchPanel


@pytest.fixture
def panel():
    app = QApplication.instance() or QApplication([])
    p = TrenchPanel();p.resize(1200, 850);p.show()
    p.set_image(np.full((300, 600), 100, np.uint8), 'edit.png', .5)
    p.load_structure_payload(dict(wall_guides=[dict(points=[[50, 60], [200, 100], [350, 60], [500, 150]], half_width_px=10)]))
    app.processEvents();p.view.fit_image()
    yield p
    p.close()


@pytest.mark.parametrize('zoom', [.6, 1.7])
def test_click_curved_span_inserts_in_order_and_drags_only_new_point(panel, zoom):
    p = panel;p.view.scale(zoom, zoom)
    original = p.guide_editor.guides[0]['points']
    samples = curve_samples(p.guide_editor.guides[0])
    # A point on the second curved span, not its straight control polygon.
    count = max(3, int(np.ceil(np.linalg.norm(np.subtract(original[1], original[0]))*1.5)))
    count2 = max(3, int(np.ceil(np.linalg.norm(np.subtract(original[2], original[1]))*1.5)))
    middle = samples[count+count2//2]
    a = p.view.mapFromScene(*middle);b = p.view.mapFromScene(middle[0]+8, middle[1]+12)
    changed = QSignalSpy(p.guide_editor.changed)
    QTest.mousePress(p.view.viewport(), Qt.MouseButton.LeftButton, pos=a)
    assert len(p.guide_editor.guides[0]['points']) == 5
    QTest.mouseMove(p.view.viewport(), b)
    QTest.mouseRelease(p.view.viewport(), Qt.MouseButton.LeftButton, pos=b)
    points = p.guide_editor.guides[0]['points']
    np.testing.assert_array_equal(points[:2]+points[3:], original)
    np.testing.assert_allclose(points[2], middle+[8, 12], atol=2/zoom)
    assert changed.count() == 1
    state = json.loads(json.dumps(p.state()))
    p.set_image(p.image, 'restored.png', .5, state)
    assert p.guide_editor.guides[0]['points'] == points


def test_existing_node_edits_without_duplicate_and_alt_drag_moves_whole_curve(panel):
    p = panel
    original = np.array(p.guide_editor.guides[0]['points'])
    a = p.view.mapFromScene(*original[1]);b = p.view.mapFromScene(*(original[1]+[5, 8]))
    QTest.mousePress(p.view.viewport(), Qt.MouseButton.LeftButton, pos=a)
    QTest.mouseMove(p.view.viewport(), b)
    QTest.mouseRelease(p.view.viewport(), Qt.MouseButton.LeftButton, pos=b)
    assert len(p.guide_editor.guides[0]['points']) == len(original)
    original = np.array(p.guide_editor.guides[0]['points'])
    samples = curve_samples(p.guide_editor.guides[0]);point = samples[len(samples)//5]
    a = p.view.mapFromScene(*point);b = p.view.mapFromScene(*(point+[4, 6]))
    QTest.mousePress(p.view.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.AltModifier, a)
    QTest.mouseMove(p.view.viewport(), b)
    QTest.mouseRelease(p.view.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.AltModifier, b)
    np.testing.assert_allclose(np.array(p.guide_editor.guides[0]['points'])-original, np.tile([4, 6], (4, 1)), atol=1)


def test_point_limit_does_not_start_whole_curve_drag(panel):
    p = panel
    points = [[20, 100]]+[[float(x), 100] for x in np.linspace(200, 580, 199)]
    p.load_structure_payload(dict(wall_guides=[dict(points=points, half_width_px=10)]))
    message = QSignalSpy(p.guide_editor.message)
    a = p.view.mapFromScene(100, 100);b = p.view.mapFromScene(100, 120)
    QTest.mousePress(p.view.viewport(), Qt.MouseButton.LeftButton, pos=a)
    QTest.mouseMove(p.view.viewport(), b)
    QTest.mouseRelease(p.view.viewport(), Qt.MouseButton.LeftButton, pos=b)
    assert p.guide_editor.guides[0]['points'] == points and message.count() == 1


def test_curve_drawing_uses_arrow_cursor_and_never_hand_pans(panel):
    p = panel;p.start_contour_guide()
    assert p.view.cursor().shape() == Qt.CursorShape.ArrowCursor
    assert p.view.viewport().cursor().shape() == Qt.CursorShape.ArrowCursor
    assert p.view.dragMode() == QGraphicsView.DragMode.NoDrag
    a = p.view.mapFromScene(50, 50)
    QTest.mousePress(p.view.viewport(), Qt.MouseButton.LeftButton, pos=a)
    assert p.view.viewport().cursor().shape() == Qt.CursorShape.ArrowCursor
    QTest.mouseRelease(p.view.viewport(), Qt.MouseButton.LeftButton, pos=a)
    QTest.mouseDClick(p.view.viewport(), Qt.MouseButton.LeftButton, pos=p.view.mapFromScene(150, 80))
    assert not p.guide_editor.drawing and len(p.guide_editor.guides[-1]['points']) == 2
    assert p.view.viewport().cursor().shape() == Qt.CursorShape.ArrowCursor
