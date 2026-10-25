import copy
import json
import time
import numpy as np
import pytest
from PySide6.QtWidgets import QApplication
from PySide6.QtTest import QTest

from angle_cal.trench_smoothing import METHODS, _pentadiagonal, smooth_segment, smooth_traces, smoothing_parameters
from angle_cal.trench_guides import points_in_guides
from angle_cal.trench_panel import TrenchPanel
from angle_cal.trench_plot import shape_series
from test_trench_analysis import trench_image
from test_trench_simple_workflow import guides, finish


def fixture_curve():
    y = np.arange(20., 241.)
    x = 80 + 7*np.sin((y-20)/220*np.pi) + .03*y
    points = np.column_stack((x + .8*np.sin(y*2), y))
    guide = dict(points=np.column_stack((x, y))[::10].tolist(), half_width_px=10)
    return points, x, guide


def test_banded_solver_matches_dense_reference():
    rng = np.random.default_rng(42)
    second = rng.normal(size=28); first = rng.normal(size=29)
    matrix = np.diag(np.full(30, 15.)) + np.diag(first, 1) + np.diag(first, -1) + np.diag(second, 2) + np.diag(second, -2)
    rhs = rng.normal(size=(30, 2))
    np.testing.assert_allclose(_pentadiagonal(matrix.diagonal(), first, second, rhs), np.linalg.solve(matrix, rhs), atol=1e-10)


@pytest.mark.parametrize('method', METHODS)
def test_smoothing_reduces_noise_preserves_bow_endpoints_and_corridor(method):
    p, truth, guide = fixture_curve()
    result = smooth_segment(p, method, 65, guide, (300, 300))
    np.testing.assert_array_equal(smooth_segment(p, method, 0, guide, (300, 300)), p)
    # Assess against the underlying sloped, bowed wall at corrected Y positions.
    expected = np.interp(result[:, 1], p[:, 1], truth)
    assert np.std(result[25:-25, 0]-expected[25:-25]) < .3
    assert abs(np.mean(result[25:-25, 0]-expected[25:-25])) < .25
    np.testing.assert_array_equal(result[[0, -1]], p[[0, -1]])
    assert np.linalg.norm(result-p, axis=1).max() <= smoothing_parameters(65)[1]+1e-8
    assert points_in_guides(result, (guide,)).all()
    # Reversing the drawing direction must not change the smoothed geometry.
    reverse = smooth_segment(p[::-1], method, 65, guide, (300, 300))
    np.testing.assert_allclose(reverse[::-1], result, atol=1e-7)


@pytest.mark.parametrize('method', METHODS)
def test_straight_field_and_cliff_are_not_shifted(method):
    for p in (np.column_stack((np.arange(20., 180.), np.full(160, 50.))),
              np.column_stack((np.full(160, 50.), np.arange(20., 180.)))):
        guide = dict(points=p[::20].tolist()+[p[-1].tolist()], half_width_px=8)
        fit = smooth_segment(p, method, 90, guide, (220, 220))
        fixed_axis = 1 if np.ptp(p[:, 1]) == 0 else 0
        np.testing.assert_allclose(fit[:, fixed_axis], p[:, fixed_axis], atol=1e-8)


def test_robust_spline_limits_spikes_and_does_not_bridge_missing_segments():
    p, truth, guide = fixture_curve()
    p[90, 0] += 6
    trace = dict(points=p.tolist(), segments=[p[:100].tolist(), p[100:].tolist()], rim_peak_points=p.tolist())
    before = copy.deepcopy(trace)
    corrected = smooth_traces([trace], [guide], 'robust', 100, (300, 300))[0]
    assert trace == before
    assert corrected['raw_points'] == before['points']
    assert list(map(len, corrected['segments'])) == [100, len(p)-100]
    assert corrected['segments'][0][-1] == before['segments'][0][-1]
    assert corrected['segments'][1][0] == before['segments'][1][0]
    assert abs(corrected['points'][90][0]-truth[90]) < abs(p[90, 0]-truth[90])
    assert corrected['smoothing']['max_displacement_px'] <= 4+1e-8


@pytest.mark.parametrize('method', METHODS)
def test_sharp_corner_correction_stays_bounded(method):
    p = np.vstack((np.column_stack((np.arange(20., 81.), np.full(61, 50.))),
                   np.column_stack((np.full(80, 80.), np.arange(51., 131.)))))
    guide = dict(points=[[20, 50], [80, 50], [80, 130]], half_width_px=12)
    fit = smooth_segment(p, method, 100, guide, (200, 200))
    assert np.linalg.norm(fit[60]-p[60]) < .25


@pytest.fixture
def panel():
    app = QApplication.instance() or QApplication([])
    p = TrenchPanel(); p.resize(1200, 900); p.show()
    p.set_image(trench_image(bulb=12), 'sample', .5)
    p.rim_reference.setChecked(False)
    p.load_structure_payload(dict(wall_guides=guides()))
    yield p, app
    p.wait_for_worker(); app.processEvents(); p.close()


def settle(p, app):
    deadline = time.monotonic()+30
    while time.monotonic() < deadline:
        app.processEvents()
        if not p.smoothing_timer.isActive() and p.worker is None:
            break
        time.sleep(.01)  # Let the Python analysis worker acquire the GIL.
    assert p.result is not None, p.summary_label.text()


def test_controls_reanalyze_keep_amounts_export_and_reset(panel, tmp_path):
    p, app = panel
    p.start_analysis(); finish(p, app)
    baseline = p.result.to_dict(.5)
    controls = p.smoothing_controls
    for index, method in enumerate(METHODS):
        controls.method.setCurrentIndex(controls.method.findData(method))
        controls.sliders[method].setValue(40+index*10)
        settle(p, app)
        assert p.result.options.contour_smoothing == method
        assert p.result.options.contour_strength == 40+index*10
        assert p.result.options.roi[1] == baseline['options']['roi'][1]
        assert p.result.guide_traces[0]['raw_points'] == baseline['guide_traces'][0]['points']
        assert len(shape_series(p.result, True)) > len(shape_series(p.result))
    assert [s.value() for s in controls.sliders.values()] == [40, 50, 60, 70]
    state = p.state()
    for suffix in ('json', 'csv', 'png'):
        path = tmp_path/f'corrected.{suffix}'; p.save_result(str(path)); assert path.stat().st_size > 100
    payload = json.loads((tmp_path/'corrected.json').read_text(encoding='utf-8'))
    assert payload['settings']['contour_smoothing'] == state['contour_smoothing']
    assert payload['options']['contour_strength'] == 70
    assert 'contour_strength,70' in (tmp_path/'corrected.csv').read_text(encoding='utf-8-sig')
    p.set_image(trench_image(bulb=12), 'reload', .5, state)
    assert p.state()['contour_smoothing'] == state['contour_smoothing']
    assert p.state()['wall_guides'] == guides() and p.scale_spin.value() == .5
    p.start_analysis(); finish(p, app)
    controls.reset(); settle(p, app)
    assert all(s.value() == 0 for s in controls.sliders.values())
    assert p.result.options.contour_strength == 0
    np.testing.assert_array_equal(p.result.left, baseline['profile']['left'])
    np.testing.assert_array_equal(p.result.right, baseline['profile']['right'])
    p.set_image(trench_image(), 'old-project', .5, dict(wall_guides=guides()))
    assert p.smoothing_controls.strength == 0


def test_drag_debounced_image_switch_cancels_pending_work(panel):
    p, app = panel
    p.start_analysis(); finish(p, app)
    slider = p.smoothing_controls.sliders['robust']
    slider.setSliderDown(True); slider.setValue(60)
    QTest.qWait(350)
    assert p.worker is None and not p.smoothing_timer.isActive()
    slider.setSliderDown(False)
    assert p.smoothing_timer.isActive()
    p.set_image(trench_image(), 'different-image')
    QTest.qWait(350)
    assert p.worker is None and p.result is None and not p.smoothing_timer.isActive()
