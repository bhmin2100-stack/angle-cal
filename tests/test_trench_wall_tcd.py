"""Analytic wall/Field intersections, independent of entrance fillet width."""
import json
import numpy as np
import pytest

from angle_cal.trench_field_metrics import measure_field_metrics, wall_field_intersection, contour_floor
from angle_cal.trench_analysis import export_trench_csv
from test_trench_field_metrics import analyze
from test_trench_corner import rounded_trench
from test_trench_guided_surface import surface_guide


def geometry(bottom_source='measured'):
    y=np.arange(21.,101.)
    left=30+.1*(y-20);right=90-.05*(y-20)
    fields=[dict(source='guide',status='valid',side=s,points=p,line=[0.,20.]) for s,p in (
        ('left',[[0,20],[20,20]]),('right',[[100,20],[120,20]]))]
    p=[[0.,20.],[20.,20.]]+list(map(list,zip(left,y)))+[[60.,104.]]+list(map(list,zip(right[::-1],y[::-1])))+[[100.,20.],[120.,20.]]
    sources=[bottom_source if v[1]>=90 else 'measured' for v in p]
    trace=dict(points=p,segments=[p],point_sources=sources)
    guide=dict(points=[[0,20],[30,20],[38,100],[60,104],[86,100],[90,20],[120,20]],half_width_px=15)
    return fields,[trace],(guide,),y,left,right


@pytest.mark.parametrize('scale,span',[(.1,100),(.5,20),(1.,10),(2.,6),(32.,6),(None,10)])
def test_100_angstrom_interval_uses_calibration_and_exposes_pixel_limit(scale,span):
    y=np.arange(20.,250.);edge=30+.1*(y-20)
    field=dict(line=[0.,20.],points=[[0,20],[20,20]])
    fit=wall_field_intersection(field,y,edge,scale)
    assert fit['status']=='valid'
    assert fit['intersection']==pytest.approx([30.,20.])
    assert fit['actual_span_px']==span
    assert fit['resolution_limited']==bool(scale and scale>10/6)
    assert fit['actual_span_angstrom']==(pytest.approx(span*scale*10) if scale else None)


def test_tapered_walls_intersect_fields_not_field_transition_ends():
    args=geometry();m=measure_field_metrics(*args,False,nm_per_px=1)
    assert m['tcd_status']==m['depth_status']=='valid'
    assert m['tcd_px']==pytest.approx(60.)  # Field endpoints themselves are 80 px apart.
    assert m['tcd_anchors']['left']==pytest.approx([30,20])
    assert m['tcd_anchors']['right']==pytest.approx([90,20])
    assert m['depth_px']==84
    assert m['bottom_basis']=='connected_contour'
    # Lengthening outward Field display geometry must not change TCD.
    args[0][0]['extension']=dict(points_pixel=[[-100,20],[0,20]])
    assert measure_field_metrics(*args,False,nm_per_px=1)['tcd_px']==pytest.approx(60.)


def test_tilted_fields_use_line_intersection_not_nearest_sample():
    field=dict(line=[.05,18.5],points=[[0,18.5],[20,19.5]])
    y=np.arange(21.,250.);edge=30+.2*(y-20)
    fit=wall_field_intersection(field,y,edge,1.)
    assert fit['intersection']==pytest.approx([30,20])
    assert fit['line'][0]==pytest.approx(.2)


def test_missing_or_interpolated_floor_does_not_disappear_or_become_a_measurement():
    args=geometry('interpolated');m=measure_field_metrics(*args,False,nm_per_px=1)
    assert m['depth_status']=='estimated' and m['depth_px']==84
    assert m['bottom_source']=='interpolated'
    # Two separate wall tips are not a closed contour, even if y is identical.
    trace=args[1][0];p=trace['points'];trace['segments']=[p[:82],p[83:]]
    assert contour_floor(args[1],*args[3:]) is None
    m=measure_field_metrics(*args,False,nm_per_px=1)
    assert m['depth_px'] is None and m['observed_depth_px']==80


@pytest.mark.parametrize('reverse',[False,True])
def test_connected_floor_with_dark_layer_below_keeps_depth(reverse):
    image=rounded_trench(15,30);image[365:,:]=35
    guide=surface_guide(15,30)
    if reverse:guide['points']=guide['points'][::-1]
    r=analyze(image,guide,nm_per_px=.5)
    assert r.depth_px is not None
    assert r.field_measurements['depth_status']=='valid'
    assert r.field_measurements['depth_px']==pytest.approx(305,abs=2)


def test_unobserved_wall_cannot_supply_a_tcd_fit():
    args=geometry();args[1][0]['point_sources']=['interpolated']*len(args[1][0]['points'])
    m=measure_field_metrics(*args,False,nm_per_px=1)
    assert m['tcd_px'] is None
    assert m['depth_status']=='estimated'


def test_saved_results_describe_new_definition_and_used_interval(tmp_path):
    r=analyze(nm_per_px=.5);m=r.field_measurements
    payload=r.to_dict(.5)
    assert payload['options']['nm_per_px']==.5
    assert payload['field_measurements']['tcd_basis']=='upper_wall_field_intersections'
    for fit in m['wall_fits'].values():
        assert fit['actual_span_angstrom']==100
    path=tmp_path/'tcd.csv';export_trench_csv(path,r,.5)
    assert 'upper_wall_field_intersections' in path.read_text(encoding='utf-8-sig')
    json.dumps(payload,allow_nan=False)


def test_primary_metrics_stay_visible_and_scale_change_remeasures():
    from PySide6.QtWidgets import QApplication
    from angle_cal.trench_panel import TrenchPanel
    from test_trench_simple_workflow import finish
    app=QApplication.instance() or QApplication([])
    panel=TrenchPanel();panel.resize(1100,800);panel.show()
    try:
        panel.set_image(rounded_trench(15,30),'metric-visibility',.5)
        panel.rim_reference.setChecked(False)
        panel.load_structure_payload(dict(wall_guides=[surface_guide(15,30)]))
        panel.start_analysis();finish(panel,app)
        assert panel.metric_label.isVisible()
        assert '152.' in panel.metric_label.text() and 'TCD' in panel.metric_label.text()
        assert panel.result.options.nm_per_px==.5
        assert panel.result.field_measurements['wall_fits']['left']['actual_span_px']==20
        panel.scale_spin.setValue(1.)
        panel.smoothing_timer.stop();panel._apply_smoothing();finish(panel,app)
        assert panel.result.options.nm_per_px==1.
        assert panel.result.field_measurements['wall_fits']['left']['actual_span_px']==10
        assert '305.' in panel.metric_label.text()
    finally:panel.wait_for_worker();panel.close();app.processEvents()
