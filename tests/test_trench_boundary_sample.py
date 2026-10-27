import json
import time
from dataclasses import replace

import cv2
import numpy as np
import pytest
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from angle_cal.trench_analysis import TrenchOptions
from angle_cal.trench_boundary_sample import make_boundary_sample, sample_point, validate_boundary_sample
from angle_cal.trench_boundary_dialog import BoundarySampleDialog
from angle_cal.trench_contour import trace_guide
from angle_cal.trench_panel import TrenchPanel
from angle_cal.trench_workflow import recognize_from_guides
from test_trench_corner import rounded_trench
from test_trench_guided_surface import surface_guide


def edge_image():
    x = np.arange(210, dtype=float)
    values = 40 + 120/(1+np.exp(np.clip((x-100)/1.5, -100, 100))) + 110*np.exp(-((x-98)/2)**2)
    return np.tile(values.astype(np.float32), (200, 1))


def normalized(image):
    lo, hi = np.percentile(image, (1, 99))
    return cv2.GaussianBlur(((image-lo)*255/(hi-lo)).astype(np.float32), (3, 3), 0)


@pytest.mark.parametrize('marker', [97.5, 100., 103.25])
@pytest.mark.parametrize('variant', ['normal', 'reverse_guide', 'reverse_sample', '16bit', 'inverted', 'gain_offset'])
def test_selected_landmark_moves_detection_and_handles_direction_and_intensity(marker, variant):
    image = edge_image()
    if variant == '16bit':image = (image*200).astype(np.uint16)
    if variant == 'inverted':image = 300-image
    sample = make_boundary_sample(image, [82, 100], [118, 100], anchor=marker-82)
    if variant == 'reverse_sample':sample = make_boundary_sample(image, [118, 100], [82, 100], anchor=118-marker)
    if variant == 'gain_offset':image = image*.6+80
    points = [[100, 25], [100, 175]]
    if variant == 'reverse_guide':points.reverse()
    trace = trace_guide(normalized(image), dict(points=points, half_width_px=9), 65,
                        'sample_profile', local_contrast=True, boundary_sample=sample)
    assert trace['method'] == 'sampled_profile_landmark'
    assert trace['trace_coverage'] > .95
    x = np.asarray(trace['points'])[:, 0]
    assert np.max(abs(x-marker)) < .65


def test_matching_never_selects_an_outside_landmark_or_blank_band():
    image = edge_image(); sample = make_boundary_sample(image, [82, 100], [118, 100], anchor=18)
    for im, center in ((image, 80), (np.full_like(image, 40), 100)):
        trace = trace_guide(cv2.GaussianBlur(im, (3, 3), 0), dict(points=[[center, 25], [center, 175]], half_width_px=5),
                            65, 'sample_profile', local_contrast=True, boundary_sample=sample, continuous=True)
        assert trace['detected_points'] == 0 and not trace['points']


def test_blank_cross_sections_are_explicitly_interpolated_not_measured():
    image = edge_image(); sample = make_boundary_sample(image, [82, 100], [118, 100], anchor=18)
    image[65:77] = 70
    trace = trace_guide(normalized(image), dict(points=[[100, 25], [100, 175]], half_width_px=9), 65,
                        'sample_profile', local_contrast=True, boundary_sample=sample, continuous=True)
    assert trace['continuity_complete'] and trace['interpolated_points'] > 0
    for point, source in zip(trace['points'], trace['point_sources']):
        if 68 < point[1] < 74:assert source == 'interpolated'


@pytest.mark.parametrize('bad', ['blank', 'short', 'long', 'outside'])
def test_invalid_sample_rejected(bad):
    image = edge_image(); start, end = [82, 100], [118, 100]
    if bad == 'blank':image[:] = 40
    if bad == 'short':end = [85, 100]
    if bad == 'long':start, end = [10, 100], [190, 100]
    if bad == 'outside':start, end = [82, 0], [118, 0]
    with pytest.raises(ValueError):make_boundary_sample(image, start, end)


@pytest.mark.parametrize('variant', ['normal', 'bright_tem', '16bit', 'reverse'])
def test_guided_workflow_uses_sample_on_both_walls_fields_and_floor(variant):
    image = rounded_trench(15, 30, blur=.6); guide = surface_guide(15, 30)
    if variant == 'bright_tem':image = 255-image
    if variant == '16bit':image = image.astype(np.uint16)*250
    if variant == 'reverse':guide = dict(guide, points=guide['points'][::-1])
    sample = make_boundary_sample(image, [145, 200], [175, 200], anchor=15)
    opts = TrenchOptions((0, 0, 420, 400), wall_guides=(guide,), boundary_sample=sample,
                         imaging_mode='tem' if variant == 'bright_tem' else 'vsem')
    r = recognize_from_guides(image, opts)
    assert r.imaging['boundary_method'] == 'sampled_profile_landmark'
    assert r.options.boundary_mode == 'sample_profile'
    assert r.depth_px == pytest.approx(305, abs=3)
    assert all(f['status'] == 'valid' for f in r.field_angles)
    assert np.interp(200, r.y, r.left) == pytest.approx(160, abs=.8)
    assert np.interp(200, r.y, r.right) == pytest.approx(260, abs=1.2)
    if variant == 'normal':
        # The same material-to-void marker moves inward on BOTH walls.
        shifted = recognize_from_guides(image, replace(opts, boundary_sample=dict(sample, anchor_px=17)))
        assert np.interp(200, shifted.y, shifted.left)-np.interp(200, r.y, r.left) == pytest.approx(2, abs=.3)
        assert np.interp(200, shifted.y, shifted.right)-np.interp(200, r.y, r.right) == pytest.approx(-2, abs=.3)


@pytest.fixture
def app():return QApplication.instance() or QApplication([])


def test_dialog_image_band_graph_and_position_are_linked_in_original_pixels(app):
    d = BoundarySampleDialog(edge_image()); d.show(); app.processEvents()
    # Actual drag through the visible image; graph and enlarged image share it.
    a, b = d.view.mapFromScene(82, 100), d.view.mapFromScene(118, 100)
    QTest.mousePress(d.view.viewport(), Qt.MouseButton.LeftButton, pos=a)
    QTest.mouseMove(d.view.viewport(), b)
    QTest.mouseRelease(d.view.viewport(), Qt.MouseButton.LeftButton, pos=b); app.processEvents()
    assert d.apply_button.isEnabled() and not d.view.drawing
    rect = d.graph.plot_rect()
    QTest.mouseClick(d.graph, Qt.MouseButton.LeftButton, pos=QPoint(round(rect.left()+rect.width()*.7), round(rect.center().y())))
    assert d.sample['anchor_px']/d.sample['distances_px'][-1] == pytest.approx(.7, abs=.01)
    d.detail.zoom_sample(d.sample); app.processEvents()
    pos = d.detail.mapFromScene(99, 100)
    QTest.mouseClick(d.detail.viewport(), Qt.MouseButton.LeftButton, pos=pos)
    assert sample_point(d.sample)[0] == pytest.approx(99, abs=.15)
    d.band.setValue(15); assert d.sample['half_width_px'] == 7
    d.position.setValue(19.25)
    sample = d.selected_sample(); assert sample['anchor_px'] == 19.25
    assert '평균 밝기' in d.info.text() and '샘플 A' in d.info.text()
    d.close()
    restored = BoundarySampleDialog(edge_image(), sample); assert restored.sample['anchor_px'] == 19.25
    restored.close()


def test_panel_persists_sample_exports_and_toggle_reanalyzes(app, tmp_path):
    image = rounded_trench(15, 30, blur=.6)
    sample = make_boundary_sample(image, [145, 200], [175, 200], anchor=15)
    state = dict(wall_guides=[surface_guide(15, 30)], boundary_sample=sample, boundary_sample_enabled=True)
    p = TrenchPanel(); p.set_image(image, 'sample', .5, state)
    assert p.sample_enabled.isChecked() and not p.rim_reference.isEnabled()
    p.start_analysis()
    def finish():
        deadline = time.monotonic()+30
        while (p.worker is not None or p.smoothing_timer.isActive()) and time.monotonic()<deadline:
            app.processEvents(); time.sleep(.01)
        assert p.result is not None, p.summary_label.text()
    finish()
    for suffix in ('csv', 'json', 'png'):p.save_result(str(tmp_path/f'result.{suffix}'))
    saved = json.loads((tmp_path/'result.json').read_text(encoding='utf-8'))
    assert saved['settings']['boundary_sample'] == sample
    assert saved['options']['boundary_sample'] == sample
    assert 'boundary_sample' in (tmp_path/'result.csv').read_text(encoding='utf-8-sig')
    assert (tmp_path/'result.png').stat().st_size>1000
    p.rim_reference.setChecked(False)
    p.sample_enabled.setChecked(False); finish()
    assert p.result.options.boundary_sample is None
    assert p.rim_reference.isEnabled() and p.state()['boundary_sample'] == sample
    p.sample_enabled.setChecked(True); finish()
    assert p.result.options.boundary_sample == sample
    restored = TrenchPanel(); restored.set_image(image.copy(), 'restored', .5, p.state())
    assert restored.options().boundary_sample == sample
    restored.reset_boundary_sample(); assert restored.options().boundary_sample is None
    assert restored.state()['boundary_sample'] is None
    p.close(); restored.close()


def test_old_or_invalid_saved_sample_defaults_safely(app):
    p = TrenchPanel(); im = edge_image(); p.set_image(im, 'old')
    assert not p.sample_enabled.isChecked() and p.options().boundary_sample is None
    p.set_image(im.copy(), 'bad', state=dict(boundary_sample={'version': 1}))
    assert not p.sample_enabled.isChecked() and '사용하지 못했습니다' in p.sample_note.text()
    p.close()
    sample = make_boundary_sample(im, [82, 100], [118, 100])
    with pytest.raises(ValueError):validate_boundary_sample(sample, (10, 10))
    with pytest.raises(ValueError):validate_boundary_sample(dict(sample, anchor_px=float('nan')))
