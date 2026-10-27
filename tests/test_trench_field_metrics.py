import json
import numpy as np
import pytest

from angle_cal.trench_analysis import TrenchOptions
from angle_cal.trench_workflow import recognize_from_guides
from angle_cal.trench_field_metrics import measure_field_metrics
from test_trench_corner import rounded_trench
from test_trench_guided_surface import surface_guide
from angle_cal.trench_analysis import trench_profile_exchange


def analyze(image=None, guide=None, **kwargs):
    return recognize_from_guides(rounded_trench(15,30) if image is None else image,
        TrenchOptions((0,0,420,400),step_px=2,wall_guides=(surface_guide(15,30) if guide is None else guide,),
                      boundary_mode='max_gradient',**kwargs))


@pytest.mark.parametrize('left_radius,right_radius', [(8,8),(15,30),(30,30),(60,60)])
def test_known_top_opening_and_depth_are_measured_from_flat_fields(left_radius,right_radius):
    r=analyze(rounded_trench(left_radius,right_radius),surface_guide(left_radius,right_radius))
    m=r.field_measurements
    assert m['field_status']==m['tcd_status']==m['depth_status']=='valid',m
    assert m['reference_y_px']==pytest.approx(45,abs=1)
    assert m['tcd_px']==pytest.approx(100+left_radius+right_radius,abs=3)
    assert m['depth_px']==pytest.approx(305,abs=1.5)
    assert m['bottom_point'][1]==pytest.approx(350,abs=1)
    # TCD is the top opening, not the 100 px straight-cliff separation.
    assert m['tcd_px'] > 110
    samples=(r.sample_y-r.depth_origin_px)/2
    np.testing.assert_allclose(samples,np.round(samples),atol=1e-8)


@pytest.mark.parametrize('variant',['reversed','inverted','16bit','gap'])
def test_field_metrics_survive_direction_polarity_bitdepth_and_wall_gap(variant):
    im=rounded_trench(15,30);g=surface_guide(15,30)
    if variant=='reversed':g['points']=g['points'][::-1]
    if variant=='inverted':im=255-im
    if variant=='16bit':im=im.astype(np.uint16)*200
    if variant=='gap':im[150:180]=205
    m=analyze(im,g).field_measurements
    assert m['tcd_px']==pytest.approx(145,abs=3)
    assert m['depth_px']==pytest.approx(305,abs=1.5)


def test_one_field_gives_depth_but_cannot_invent_tcd_or_open_bottom_depth():
    g=surface_guide(15,30);g['points']=g['points'][:-4]
    m=analyze(guide=g).field_measurements
    assert m['field_status']=='valid' and len(m['anchors'])==1
    assert m['depth_px']==pytest.approx(305,abs=2) and m['tcd_px'] is None
    im=rounded_trench(15,30);im[345:,161:260]=35
    g=surface_guide(15,30)
    open_guides=(dict(g,points=g['points'][:6]),dict(g,points=g['points'][10:]))
    m=recognize_from_guides(im,TrenchOptions((0,0,420,400),wall_guides=open_guides,boundary_mode='max_gradient')).field_measurements
    assert m['tcd_status']=='valid'
    assert m['depth_status']=='unconfirmed' and m['depth_px'] is None


def test_different_field_heights_have_explicit_mean_and_individual_depths():
    fields=[dict(source='guide',status='valid',side='left',points=[[10,20],[30,20]],line=[0,20]),
            dict(source='guide',status='valid',side='right',points=[[70,24],[90,24]],line=[0,24])]
    g=dict(points=[[10,20],[30,20],[30,100],[70,100],[70,24],[90,24]],half_width_px=10)
    traces=[dict(points=[[30,80],[50,100],[70,80]],point_sources=['measured']*3)]
    m=measure_field_metrics(fields,traces,(g,),np.arange(25,100),np.full(75,30),np.full(75,70),True)
    assert m['tcd_px']==40 and m['reference_y_px']==22
    assert m['depth_px']==78 and m['left_depth_px']==80 and m['right_depth_px']==76


def test_smoothing_does_not_move_the_field_reference_or_top_opening():
    baseline=analyze().field_measurements
    for method in ('mean','gaussian','savgol','robust'):
        m=analyze(contour_smoothing=method,contour_strength=75).field_measurements
        assert m['anchors']==baseline['anchors']
        assert m['reference_y_px']==baseline['reference_y_px']
        assert m['tcd_px']==baseline['tcd_px']
        assert abs(m['depth_px']-baseline['depth_px'])<=3


@pytest.mark.parametrize('method', ['mean','gaussian','savgol','robust'])
def test_short_confirmed_fields_remain_exportable_after_strong_smoothing(method):
    guide=surface_guide(15,30)
    guide['points']=guide['points'][2:-2]
    guide['points'][0][0]-=18
    guide['points'][-1][0]+=18
    baseline=analyze(guide=guide)
    result=analyze(guide=guide,contour_smoothing=method,contour_strength=100)
    assert all(f['status']=='valid' for f in baseline.field_angles)
    assert all(f['status']=='valid' for f in result.field_angles)
    assert result.field_measurements['anchors']==baseline.field_measurements['anchors']
    assert result.field_measurements['tcd_status']=='valid'
    for before,after in zip(baseline.field_angles,result.field_angles):
        assert before['points']==after['points']
    payload=trench_profile_exchange(result,.5)
    points=np.asarray(result.guide_traces[0]['points'])
    np.testing.assert_allclose(payload['points'],np.column_stack((
        (points[:,0]-payload['origin_pixel'][0])*5,(payload['origin_pixel'][1]-points[:,1])*5)))


@pytest.mark.parametrize('width', [32,48])
def test_wide_search_corridor_uses_fields_at_actual_entrance(width):
    guide=surface_guide(15,30);guide['half_width_px']=width
    result=analyze(guide=guide)
    assert all(f['status']=='valid' for f in result.field_angles)
    assert result.field_measurements['tcd_status']=='valid'
    assert result.field_measurements['tcd_px']==pytest.approx(145,abs=3)
    assert result.field_measurements['depth_px']==pytest.approx(305,abs=1.5)


def test_smoothing_does_not_manufacture_missing_short_fields():
    guide=surface_guide(15,30);guide['points']=guide['points'][2:-2]
    guide['points'][0][0]-=4;guide['points'][-1][0]+=4
    result=analyze(guide=guide,contour_smoothing='mean',contour_strength=100)
    assert all(f['status']=='unconfirmed' for f in result.field_angles)
    assert result.field_measurements['tcd_px'] is None
    with pytest.raises(ValueError,match='Field 미확정'):
        trench_profile_exchange(result,.5)


@pytest.mark.parametrize('short_smoothed',[False,True])
def test_ui_field_tcd_depth_labels_units_saved_results_and_gfe(tmp_path,short_smoothed):
    from PySide6.QtWidgets import QApplication
    from angle_cal.trench_panel import TrenchPanel
    from test_trench_simple_workflow import finish
    app=QApplication.instance() or QApplication([])
    p=TrenchPanel();p.resize(1250,900);p.show()
    try:
        p.set_image(rounded_trench(15,30),'field-test',.5)
        p.rim_reference.setChecked(False)
        guide=surface_guide(15,30)
        if short_smoothed:
            guide['points']=guide['points'][2:-2]
            guide['points'][0][0]-=18;guide['points'][-1][0]+=18
            p.smoothing_controls.method.setCurrentIndex(p.smoothing_controls.method.findData('mean'))
            p.smoothing_controls.sliders['mean'].setValue(100)
        p.load_structure_payload(dict(wall_guides=[guide]))
        p.start_analysis();finish(p,app)
        assert 'Field → 최하단 깊이' in p.summary_label.text()
        assert 'TCD (Field 입구 폭)' in p.summary_label.text()
        assert 'Field 자동 인식' in p.summary_label.text()
        assert '좌측 Field 확인' in p.summary_label.text() and '우측 Field 확인' in p.summary_label.text()
        assert p.profile_table.rowCount()>0
        p.graph.grab();p.grab().save(str(tmp_path/'workspace.png'))
        texts=[item.text() for item in p.overlays if hasattr(item,'text')]
        assert sum(text=='Field' for text in texts)==2
        assert any(text.startswith('TCD ') for text in texts)
        for ext in ('csv','json','png'):p.save_result(str(tmp_path/f'metrics.{ext}'))
        data=json.loads((tmp_path/'metrics.json').read_text(encoding='utf-8'))
        assert data['field_measurement_units']['nm']['tcd_nm']==pytest.approx(data['summary']['tcd_px']*.5)
        assert data['field_measurement_units']['angstrom']['depth_angstrom']==pytest.approx(data['summary']['field_depth_px']*5)
        assert data['settings']['field_measurements']==data['field_measurements']
        assert 'field_depth_nm' in (tmp_path/'metrics.csv').read_text(encoding='utf-8-sig')
        p.copy_gfe_coordinates();assert app.clipboard().text().startswith('X\tY\n'),p.hint.text()
    finally:
        p.wait_for_worker();p.close();app.processEvents()
