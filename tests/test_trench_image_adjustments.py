import cv2
import numpy as np
import pytest
from PySide6.QtWidgets import QApplication, QFileDialog, QGraphicsPixmapItem

from angle_cal.app import MainWindow
from angle_cal.image_ops import adjust_image_bgr
from test_trench_analysis import trench_image
from test_trench_simple_workflow import guides, finish


@pytest.fixture
def window(tmp_path):
    app = QApplication.instance() or QApplication([])
    paths = [tmp_path / 'one.png', tmp_path / 'two.png']
    for path in paths:
        cv2.imencode('.png', trench_image(bulb=12))[1].tofile(str(path))
    w = MainWindow()
    w.browser_image_paths = list(map(str, paths))
    w._populate_thumbnails()
    w._load_image_path(str(paths[0]))
    yield w, paths, app
    w.close()


def assert_display(panel, expected):
    np.testing.assert_array_equal(panel.image, expected)
    for view in (panel.view, panel.overview):
        item = next(i for i in view.scene().items() if isinstance(i, QGraphicsPixmapItem))
        image = item.pixmap().toImage()
        for x, y in ((10, 10), (80, 80), (100, 220)):
            assert image.pixelColor(x, y).getRgb()[:3] == tuple(int(v) for v in expected[y, x, ::-1])


@pytest.mark.parametrize('adjustments', [(25, 100, 0), (0, 130, 0), (0, 100, 120), (15, 110, 60)])
def test_adjusted_pixels_display_and_analysis_survive_tab_switch(window, adjustments):
    w, _, app = window
    original = w.image_bgr.copy()
    w._set_image_adjustments(*adjustments)
    expected = adjust_image_bgr(original, *adjustments)
    w.nm_per_px = .5
    w.addon_actions['trench_analyzer'].setChecked(True)
    p = w.trench_panel
    assert_display(p, expected)
    p.rim_reference.setChecked(False)
    p.load_structure_payload(dict(wall_guides=guides()))
    p.start_analysis()
    np.testing.assert_array_equal(p.worker.image, expected)
    finish(p, app)
    result, image, revision = p.result, p.image, p.revision
    w.ribbon_tabs.setCurrentIndex(0)
    w.ribbon_tabs.setCurrentWidget(w.addon_pages['trench_analyzer'])
    assert p.result is result and p.image is image and p.revision == revision
    assert p.state()['wall_guides'] == guides()
    assert p.scale_spin.value() == .5
    np.testing.assert_array_equal(w.image_bgr, original)


def test_changed_adjustments_reset_result_and_roundtrip_per_image(window, monkeypatch, tmp_path):
    w, paths, app = window
    w.addon_actions['trench_analyzer'].setChecked(True)
    p = w.trench_panel
    p.rim_reference.setChecked(False)
    p.load_structure_payload(dict(wall_guides=guides()))
    p.start_analysis()
    finish(p, app)
    revision = p.revision
    w.image_brightness_spin.setValue(30)
    assert p.result is None and not p.export_button.isEnabled() and p.revision > revision
    assert p.state()['wall_guides'] == guides()
    expected = adjust_image_bgr(w.image_bgr, 30)
    assert_display(p, expected)
    w.load_browser_image(str(paths[1]))
    w.image_brightness_spin.setValue(-10)
    assert_display(p, adjust_image_bgr(w.image_bgr, -10))
    w.load_browser_image(str(paths[0]))
    assert_display(p, expected)
    assert p.state()['wall_guides'] == guides()
    project = tmp_path / 'adjusted.anglecal.json'
    monkeypatch.setattr(w, '_show_save_notification', lambda *a: None)
    w._save_project_to_path(str(project))
    reader = MainWindow()
    try:
        monkeypatch.setattr(QFileDialog, 'getOpenFileName', lambda *a, **k: (str(project), ''))
        reader.open_project()
        reader.addon_actions['trench_analyzer'].setChecked(True)
        assert_display(reader.trench_panel, expected)
        assert reader.trench_panel.state()['wall_guides'] == guides()
        reader.reset_image_adjustments()
        assert_display(reader.trench_panel, adjust_image_bgr(reader.image_bgr))
    finally:
        reader.close()
