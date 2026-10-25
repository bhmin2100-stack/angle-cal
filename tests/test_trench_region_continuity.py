"""One selected region yields one contour; inferred portions keep provenance."""
import json
import numpy as np
import pytest

from angle_cal.trench_contour import trace_guide, _connect_region_gaps
from angle_cal.trench_guides import points_in_guides, sweep_samples
from angle_cal.trench_analysis import TrenchOptions, export_trench_csv
from angle_cal.trench_workflow import recognize_from_guides
from angle_cal.trench_smoothing import smooth_traces
from test_trench_continuity import sem_wall
from test_trench_analysis import trench_image
from test_trench_simple_workflow import guides


@pytest.mark.parametrize('kind', ['straight', 'bow', 'taper'])
def test_long_missing_wall_is_continuous_without_claiming_detected_points(kind):
    image, _ = sem_wall(kind, True)
    image[135:175] = 155
    g = dict(points=[[90, 30], [90, 290]], half_width_px=30)
    raw = trace_guide(image, g, 75, 'max_gradient', True, 155)
    t = trace_guide(image, g, 75, 'max_gradient', True, 155, continuous=True)
    assert len(raw['segments']) >= 2 and len(t['segments']) == 1
    assert t['detected_points'] == raw['detected_points']
    assert t['interpolated_points'] >= 35 and t['continuity_complete']
    assert t['preprocessing']['missing_policy'] == 'continuous_region_interpolation'
    assert points_in_guides(t['points'], (g,)).all()
    assert t['points'][0] == raw['points'][0] and t['points'][-1] == raw['points'][-1]
    assert all(spread is None for spread, source in zip(t['edge_spreads_px'], t['point_sources']) if source == 'interpolated')
    reverse = dict(g, points=g['points'][::-1])
    backward = trace_guide(image, reverse, 75, 'max_gradient', True, 155, continuous=True)
    np.testing.assert_allclose(t['points'], backward['points'][::-1], atol=1)
    for method in ('mean', 'gaussian', 'savgol', 'robust'):
        corrected = smooth_traces([t], [g], method, 70, image.shape)[0]
        assert len(corrected['segments']) == 1 and corrected['point_sources'] == t['point_sources']


def test_missing_curved_floor_follows_corridor_not_cross_trench_chord():
    g = dict(points=[[30, 20], [30, 80], [50, 100], [80, 80], [80, 20]], half_width_px=3)
    centers = sweep_samples(g)
    distance = np.r_[0., np.cumsum(np.linalg.norm(np.diff(centers, axis=0), axis=1))]
    measured = {i: p.tolist() for i, p in enumerate(centers) if p[1] < 70}
    connected, joins = _connect_region_gaps(measured, distance, centers, g, (130, 120))
    assert len(connected) == len(centers) and len(joins) == 1
    assert joins[0]['length_px'] > 40
    assert np.max(np.array(list(connected.values()))[:, 1]) > 98
    assert points_in_guides(list(connected.values()), (g,)).all()
    assert all(connected[i] == p for i, p in measured.items())


def test_no_extrapolation_no_unobserved_contour_and_no_transverse_branch_jump():
    centers = np.column_stack((np.full(30, 30.), np.arange(30.)))
    distance = np.arange(30.)
    g = dict(points=[[30, 0], [30, 29]], half_width_px=15)
    measured = {i: [30. if i < 15 else 43., float(i)] for i in range(3, 27) if i not in (12, 13, 14)}
    connected, joins = _connect_region_gaps(measured, distance, centers, g, (50, 60))
    assert connected == measured and not joins
    assert _connect_region_gaps({}, distance, centers, g, (50, 60)) == ({}, [])
    flat = trace_guide(np.full((50, 60), 155., np.float32), g, 75,
                       'max_gradient', True, 155, continuous=True)
    assert not flat['points'] and not flat['continuity_complete']


@pytest.mark.parametrize('variant', ['normal', 'inverted', '16bit'])
def test_auto_region_measurements_and_export_keep_both_walls_across_blank_band(variant, tmp_path):
    image = trench_image(bulb=12)
    image[130:160] = 205
    if variant == 'inverted': image = 255-image
    if variant == '16bit': image = image.astype(np.uint16)*200
    result = recognize_from_guides(image, TrenchOptions((0, 0, 200, 260), step_px=1,
                                   wall_guides=tuple(guides()), boundary_mode='max_gradient'))
    assert result.options.continuous_guides
    assert all(t['continuity_complete'] for t in result.guide_traces)
    assert np.all(np.diff(result.y) <= 1.01)
    assert result.y[0] < 50 and result.y[-1] > 200
    assert np.all(np.isin(np.arange(130, 160), result.sample_y))
    assert any(t['interpolated_points'] > 40 for t in result.guide_traces)
    assert points_in_guides(np.column_stack((result.left, result.y)), guides()).all()
    assert points_in_guides(np.column_stack((result.right, result.y)), guides()).all()
    payload = json.loads(json.dumps(result.to_dict(.5)))
    assert payload['options']['continuous_guides']
    assert any('interpolated' in t['point_sources'] for t in payload['guide_traces'])
    export_trench_csv(str(tmp_path/'connected.csv'), result, .5)
    assert 'guide_corridor_residual_interpolation' in (tmp_path/'connected.csv').read_text(encoding='utf-8-sig')


def test_bright_rim_inset_keeps_gap_provenance_and_inward_offset():
    from test_trench_rim import roof_image
    image = roof_image()
    image[135:175] = 150
    g = dict(points=[[90, 80], [90, 240], [100, 270], [220, 270], [230, 240], [230, 80]], half_width_px=18)
    r = recognize_from_guides(image, TrenchOptions((0, 0, 320, 320), wall_guides=(g,), boundary_mode='rim_inset'))
    assert len(r.guide_traces[0]['segments']) == 1 and np.max(np.diff(r.y)) <= 1
    t = r.guide_traces[0]
    assert t['interpolated_points'] > 60
    for point, peak, normal, source in zip(t['points'], t['rim_peak_points'], t['inward_normals'], t['point_sources']):
        if source == 'interpolated':
            assert peak is None and normal is None
        else:
            assert np.linalg.norm(np.subtract(point, peak)) == pytest.approx(1)


def test_ui_graph_saved_result_and_reload_use_connected_contour(tmp_path):
    from PySide6.QtWidgets import QApplication
    from angle_cal.trench_panel import TrenchPanel
    from angle_cal.trench_plot import shape_series
    from test_trench_simple_workflow import finish
    app = QApplication.instance() or QApplication([])
    image = trench_image(bulb=12); image[130:160] = 205
    panel = TrenchPanel()
    try:
        panel.set_image(image, 'gap-test', .5)
        panel.rim_reference.setChecked(False)
        panel.load_structure_payload(dict(wall_guides=guides()))
        panel.start_analysis(); finish(panel, app)
        assert '경계 누락' in panel.summary_label.text()
        assert '끊겼습니다' not in panel.summary_label.text()
        contour_series = [s for s in shape_series(panel.result) if s[2] == '#8f45a4' and not s[3]]
        assert len(contour_series) == 1
        for suffix in ('json', 'csv', 'png'):
            panel.save_result(str(tmp_path/f'connected.{suffix}'))
            assert (tmp_path/f'connected.{suffix}').stat().st_size > 100
        payload = json.loads((tmp_path/'connected.json').read_text(encoding='utf-8'))
        assert payload['guide_traces'][0]['continuity_complete']
        assert payload['guide_traces'][0]['interpolated_points'] > 40
        state = panel.state()
        panel.set_image(image, 'reloaded', .5, state)
        panel.start_analysis(); finish(panel, app)
        assert panel.result.guide_traces[0]['continuity_complete']
    finally:
        panel.wait_for_worker(); panel.close(); app.processEvents()
