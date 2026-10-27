"""Independent image landmarks on one connected trench surface."""
import copy
import json
import time
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from angle_cal.trench_analysis import TrenchOptions
from angle_cal.trench_boundary_dialog import BoundarySamplesDialog
from angle_cal.trench_boundary_sample import make_boundary_sample, validate_boundary_samples, wall_sample_scopes
from angle_cal.trench_contour import trace_guide
from angle_cal.trench_guides import sweep_samples, points_in_guides
from angle_cal.trench_panel import TrenchPanel
from angle_cal.trench_workflow import recognize_from_guides
from test_trench_boundary_sample import normalized
from test_trench_corner import rounded_trench
from test_trench_guided_surface import surface_guide
from test_trench_tem import tem_image, tem_guide
from test_trench_rim import roof_image


def wall_samples(image):
    return dict(left=make_boundary_sample(image, [145, 200], [175, 200], anchor=15),
                right=make_boundary_sample(image, [245, 200], [275, 200], anchor=15))


def asymmetric_image():
    image = rounded_trench(15, 30, blur=.6).astype(np.float32)
    x = np.arange(image.shape[1], dtype=float)
    left = 40+120/(1+np.exp(np.clip((x-160)/1.5, -100, 100)))+100*np.exp(-((x-158)/2)**2)
    right = 40+120/(1+np.exp(np.clip((260-x)/2, -100, 100)))-30*np.exp(-((x-257)/2)**2)
    image[95:310] = np.where(x<210, left, right)
    return image


@pytest.mark.parametrize('variant', ['normal', 'reverse_guide', 'reverse_samples', '16bit', 'inverted'])
def test_distinct_wall_profiles_follow_independent_landmarks(variant):
    image = asymmetric_image(); guide = surface_guide(15, 30)
    if variant == '16bit':image = (image*200).astype(np.uint16)
    if variant == 'inverted':image = 300-image
    samples = wall_samples(image)
    if variant == 'reverse_guide':guide = dict(guide, points=guide['points'][::-1])
    if variant == 'reverse_samples':
        samples = {s:make_boundary_sample(image, v['end'], v['start'], anchor=15) for s,v in samples.items()}
    def trace(settings):
        return trace_guide(normalized(image), guide, 65, 'max_gradient', local_contrast=True,
                           boundary_samples=settings, continuous=True)
    baseline = trace(samples)
    assert baseline['method'] == 'side_sampled_profile_landmarks'
    assert baseline['continuity_complete']
    assert baseline['sample_scope_counts']['left'] > 150
    assert baseline['sample_scope_counts']['right'] > 150
    for side, expected in [('left', 160), ('right', 260)]:
        points = np.asarray(baseline['points']); scopes = np.asarray(baseline['point_boundary_scopes'])
        wall = points[(scopes==side) & (points[:,1]>150) & (points[:,1]<250)]
        assert np.max(abs(wall[:,0]-expected)) < .7
    shifted_samples = copy.deepcopy(samples); shifted_samples['left']['anchor_px'] += 2
    shifted = trace(shifted_samples)
    for side in ('left', 'right'):
        def wall_x(result):
            points = np.asarray(result['points']); scopes = np.asarray(result['point_boundary_scopes'])
            return np.median(points[(scopes==side) & (points[:,1]>150) & (points[:,1]<250), 0])
        expected = (-2 if variant=='reverse_samples' else 2) if side=='left' else 0
        assert wall_x(shifted)-wall_x(baseline) == pytest.approx(expected, abs=.15)


@pytest.mark.parametrize('common', [False, True])
def test_unilateral_override_preserves_other_wall_and_surface_defaults(common):
    image = rounded_trench(15, 30, blur=.6); guide = surface_guide(15, 30)
    samples = wall_samples(image)
    options = TrenchOptions((0, 0, 420, 400), wall_guides=(guide,), boundary_mode='max_gradient',
                           boundary_sample=samples['left'] if common else None)
    base = recognize_from_guides(image, options)
    left = dict(samples['left'], anchor_px=17)
    changed = recognize_from_guides(image, replace(options, boundary_samples={'left':left}))
    assert np.interp(200, changed.y, changed.right) == pytest.approx(np.interp(200, base.y, base.right), abs=.1)
    assert np.interp(200, changed.y, changed.left) == pytest.approx(162, abs=.8)
    assert changed.depth_px == pytest.approx(base.depth_px, abs=.15)
    trace = changed.guide_traces[0]
    assert trace['continuity_complete']
    assert trace['sample_scope_counts']['common' if common else 'automatic'] > 100
    assert 'right' not in trace['point_boundary_scopes']


def test_tem_wall_samples_keep_material_interface_on_field_and_floor():
    image = tem_image()
    samples = dict(left=make_boundary_sample(image, [65,180], [100,180], anchor=16.5),
                   right=make_boundary_sample(image, [200,180], [235,180], anchor=18.5))
    options = TrenchOptions((0, 0, 320, 360), wall_guides=(tem_guide(),), imaging_mode='tem')
    base = recognize_from_guides(image, options)
    r = recognize_from_guides(image, replace(options, boundary_samples=samples))
    assert r.options.boundary_mode == 'tem_interface'
    assert r.imaging['boundary_method'] == 'side_sampled_profile_landmarks'
    assert r.depth_px == pytest.approx(base.depth_px, abs=.1)
    assert all(f['status']=='valid' for f in r.field_angles)
    assert r.guide_traces[0]['sample_scope_counts']['automatic'] > 100
    assert np.interp(180, r.y, r.left) == pytest.approx(81.5, abs=.8)
    assert np.interp(180, r.y, r.right) == pytest.approx(218.5, abs=.8)


def test_sample_marker_has_priority_over_rim_inset_on_its_wall_only():
    image = roof_image()
    sample = make_boundary_sample(image, [75,150], [115,150], anchor=15)
    guide = dict(points=[[90,80],[90,240],[100,270],[220,270],[230,240],[230,80]], half_width_px=18)
    a, b = [trace_guide(image, guide, 65, 'rim_inset', True, 150, rim_inset_px=amount,
                        boundary_samples={'left':sample}, continuous=True) for amount in (.5,2.5)]
    for side, delta in [('left',0), ('right',-2)]:
        positions = []
        for t in (a,b):
            points = np.asarray(t['points']); mid = (points[:,1]>120)&(points[:,1]<200)
            mid &= (points[:,0]<160) if side=='left' else (points[:,0]>160)
            assert mid.sum()>50
            positions.append(np.median(points[mid,0]))
        assert positions[1]-positions[0] == pytest.approx(delta, abs=.1)


@pytest.mark.parametrize('reverse', [False, True])
def test_short_tem_sample_does_not_flip_to_opposite_flank(reverse):
    image = tem_image()
    a,b=[212,180],[228,180]
    sample = make_boundary_sample(image, b if reverse else a, a if reverse else b,
                                  anchor=9.5 if reverse else 6.5)
    r = recognize_from_guides(image, TrenchOptions((0,0,320,360),wall_guides=(tem_guide(reverse),),
        imaging_mode='tem',boundary_samples={'right':sample}))
    assert np.interp(180,r.y,r.right) == pytest.approx(218.5,abs=.8)


def test_tem_reference_selected_flanks_join_the_field_with_explicit_interpolation():
    root = Path(__file__).resolve().parents[1]
    image = cv2.imdecode(np.fromfile(root/'samples/trench-tem-reference.png',np.uint8),-1)
    guide = json.loads((root/'samples/trench-tem-reference-right.anglecal.structure.json').read_text())['wall_guides'][0]
    # Material-facing positions chosen in this fixture, not the opposite film flank.
    samples = {side:make_boundary_sample(image,[x-12,400],[x+12,400],anchor=12)
               for side,x in [('left',830.14),('right',988.65)]}
    r = recognize_from_guides(image, TrenchOptions((0,0,1536,1024),wall_guides=(guide,),
        imaging_mode='tem',boundary_samples=samples,contour_strength=45))
    assert np.interp(400,r.y,r.left) == pytest.approx(830.14,abs=2)
    assert np.interp(400,r.y,r.right) == pytest.approx(988.65,abs=2)
    trace=r.guide_traces[0]
    assert trace['continuity_complete'] and trace['interpolated_points']>0
    assert points_in_guides(trace['points'],(guide,)).all()
    assert r.field_measurements['depth_status']=='valid'


def test_path_order_handles_bowing_across_midline_and_reversed_guides():
    points = np.array([[20,10],[30,10],[35,30],[85,70],[40,110],[40,130],
                       [60,130],[60,110],[110,70],[65,30],[70,10],[90,10]], float)
    normals = np.tile([1.,0.], (len(points),1)); normals[[0,1,5,6,10,11]] = [0,1]
    expected = ['surface','surface','left','left','left','surface','surface','right','right','right','surface','surface']
    assert list(wall_sample_scopes(points, normals, {'half_width_px':10})) == expected
    assert list(wall_sample_scopes(points[::-1], normals[::-1], {'half_width_px':10})) == expected[::-1]


def test_separate_wall_guides_ignore_legacy_side_labels():
    guides = [dict(points=[[160,80],[160,310]], half_width_px=10, side='right'),
              dict(points=[[260,310],[260,80]], half_width_px=10, side='left')]
    for guide, side in zip(guides, ('left','right')):
        points = sweep_samples(guide)
        normals = np.tile([1.,0.], (len(points),1))
        assert set(wall_sample_scopes(points, normals, guide, guides)) == {side}


@pytest.fixture
def app():return QApplication.instance() or QApplication([])


def test_dialog_tabs_preserve_independent_sample_bands_and_markers(app):
    image = asymmetric_image(); d = BoundarySamplesDialog(image)
    d.show(); app.processEvents()
    for index, side, start, end, marker in [(0,'left',145,175,17.25),(1,'right',245,275,12.75)]:
        d.tabs.setCurrentIndex(index); app.processEvents(); app.processEvents()
        e = d.editors[side]; a, b = e.view.mapFromScene(start,200), e.view.mapFromScene(end,200)
        QTest.mousePress(e.view.viewport(), Qt.MouseButton.LeftButton, pos=a)
        QTest.mouseMove(e.view.viewport(), b)
        QTest.mouseRelease(e.view.viewport(), Qt.MouseButton.LeftButton, pos=b); app.processEvents()
        assert e.sample is not None, e.status.text()
        e.position.setValue(marker); e.band.setValue(11 if side=='left' else 7)
    saved = d.selected_samples()
    assert saved['left']['anchor_px']==17.25 and saved['left']['half_width_px']==5
    assert saved['right']['anchor_px']==12.75 and saved['right']['half_width_px']==3
    d.tabs.setCurrentIndex(0); app.processEvents(); app.processEvents()
    rect = d.editors['left'].graph.plot_rect()
    QTest.mouseClick(d.editors['left'].graph, Qt.MouseButton.LeftButton,
                    pos=QPoint(round(rect.left()+.4*rect.width()), round(rect.center().y())))
    assert d.selected_samples()['right'] == saved['right']
    assert d.selected_samples()['left']['anchor_px'] != saved['left']['anchor_px']
    d.editors['left'].clear_button.click()
    assert set(d.selected_samples()) == {'right'} and d.apply_button.isEnabled()
    # A bad newly drawn line must not silently discard an existing setting.
    e = d.editors['right']; e.line = ([200,200],[202,200]); e._resample()
    assert not d.apply_button.isEnabled() and '확인 필요' in d.tabs.tabText(1)
    e.clear_sample(); assert d.apply_button.isEnabled() and d.selected_samples()=={}
    d.close()


def test_legacy_common_sample_survives_tab_switch_and_cancel(app):
    image = asymmetric_image(); samples = wall_samples(image); original = copy.deepcopy(samples)
    common = make_boundary_sample(image, [145,200], [175,200], anchor=14.25)
    d = BoundarySamplesDialog(image, common, samples); d.show(); app.processEvents()
    app.processEvents()
    assert d.height() <= 820
    assert d.apply_button.mapTo(d, QPoint(0,d.apply_button.height())).y() <= d.height()
    assert d.editors['left'].graph.mapTo(d, QPoint(0,d.editors['left'].graph.height())).y() < d.height()
    d.tabs.setCurrentIndex(1); d.editors['right'].position.setValue(17)
    QTest.keyClick(d.editors['right'].view, Qt.Key.Key_Escape)
    assert not d.isVisible() and d.result()==d.DialogCode.Rejected
    assert samples == original and common['anchor_px']==14.25
    assert d.selected_common_sample()['anchor_px']==14.25
    d.close()


def test_panel_roundtrip_exports_and_disable_preserve_both_samples(app, tmp_path):
    image = rounded_trench(15, 30, blur=.6); samples = wall_samples(image)
    state = dict(wall_guides=[surface_guide(15,30)], boundary_samples=samples, boundary_sample_enabled=True,
                 rim_reference=False)
    p = TrenchPanel(); p.set_image(image, 'dual', .5, state)
    assert p.sample_enabled.isChecked() and p.rim_reference.isEnabled()
    assert '왼쪽:' in p.sample_note.text() and '오른쪽:' in p.sample_note.text()
    p.start_analysis()
    def finish():
        deadline = time.monotonic()+30
        while (p.worker is not None or p.smoothing_timer.isActive()) and time.monotonic()<deadline:
            app.processEvents(); time.sleep(.01)
        assert p.worker is None and p.result is not None, p.summary_label.text()
    finish()
    for suffix in ('csv','json','png'):p.save_result(str(tmp_path/f'dual.{suffix}'))
    saved = json.loads((tmp_path/'dual.json').read_text(encoding='utf-8'))
    assert saved['settings']['boundary_samples']==samples
    assert saved['options']['boundary_samples']==samples
    assert 'boundary_samples' in (tmp_path/'dual.csv').read_text(encoding='utf-8-sig')
    assert (tmp_path/'dual.png').stat().st_size>1000
    scopes = saved['guide_traces'][0]['sample_scope_counts']
    assert scopes['left']>100 and scopes['right']>100 and scopes['automatic']>100
    p.sample_enabled.setChecked(False); finish()
    assert p.result.options.boundary_samples is None and p.state()['boundary_samples']==samples
    restored = TrenchPanel(); restored.set_image(image.copy(), 'restored', .5, p.state())
    assert not restored.sample_enabled.isChecked() and restored.boundary_samples==samples
    restored.sample_enabled.setChecked(True)
    assert restored.options().boundary_samples==samples
    restored.reset_boundary_sample()
    assert restored.state()['boundary_samples']=={} and restored.options().boundary_samples is None
    p.close(); restored.close()


@pytest.mark.parametrize('bad', [[], {'wrong':{}}, {'left':{'version':1}}, {'right':{'version':1}}])
def test_invalid_side_settings_are_rejected(bad):
    with pytest.raises(ValueError):validate_boundary_samples(bad)
