from dataclasses import replace
import csv
import json

import numpy as np
import pytest

from angle_cal.trench_analysis import trench_profile_exchange, export_trench_csv
from angle_cal.trench_auto_field import recover_fields, extend_export_surface
from angle_cal.trench_plot import shape_series
from test_trench_field_metrics import analyze
from test_trench_guided_surface import surface_guide
from test_trench_corner import rounded_trench


def short_guide():
    guide=surface_guide(15,30);guide['points']=guide['points'][2:-2]
    guide['points'][0][0]-=12;guide['points'][-1][0]+=12
    return guide


@pytest.mark.parametrize('variant',['plain','reverse','inverted','16bit','smooth','strong'])
def test_short_horizontal_ends_create_fields_and_export_ordered_extensions(variant):
    image=rounded_trench(15,30);guide=short_guide()
    if variant=='reverse':guide['points']=guide['points'][::-1]
    if variant=='inverted':image=255-image
    if variant=='16bit':image=image.astype(np.uint16)*200
    result=analyze(image,guide,contour_strength={'smooth':100,'strong':300}.get(variant,0),contour_smoothing='mean')
    assert all(f['status']=='valid' and f['automatic_field'] for f in result.field_angles)
    assert result.field_measurements['automatic_fields']==['left','right']
    assert result.field_measurements['tcd_status']=='valid'
    assert result.field_measurements['tcd_basis']=='segmented_transition_estimate'
    assert result.field_measurements['tcd_px']==pytest.approx(145,abs=6)
    assert result.field_measurements['reference_y_px']==pytest.approx(45,abs=1)
    assert result.field_measurements['depth_px']==pytest.approx(305,abs=3)
    payload=trench_profile_exchange(result,.5)
    assert len(payload['field_extensions'])==2 and not payload['surface_closed']
    p=np.asarray(result.guide_traces[0]['points'])
    if p[0,0]>p[-1,0]:p=p[::-1]
    expected=np.column_stack(((p[:,0]-payload['origin_pixel'][0])*5,(payload['origin_pixel'][1]-p[:,1])*5))
    nleft=len(payload['field_extensions'][0]['points'])-1
    np.testing.assert_allclose(np.asarray(payload['points'])[nleft:nleft+len(p)],expected)
    for field,extension in zip(result.field_angles,payload['field_extensions']):
        assert extension['measured'] is False
        q=np.asarray(extension['points_pixel'])
        assert np.all((q[:,0]>=0)&(q[:,0]<image.shape[1])&(q[:,1]>=0)&(q[:,1]<image.shape[0]))
        assert len(q)>10
        # A terminal continuation cannot cross the opening or close the mouth.
        assert np.all(np.diff(q[:,0])<0) if field['side']=='left' else np.all(np.diff(q[:,0])>0)
    assert sum(color=='#268b48' and dashed for _,_,color,dashed in shape_series(result))==2


def test_automatic_fields_save_provenance_and_do_not_accumulate(tmp_path):
    result=analyze(guide=short_guide())
    payload=trench_profile_exchange(result,1)
    assert trench_profile_exchange(result,1)==payload
    saved=json.loads(json.dumps(result.to_dict(1),allow_nan=False))
    assert saved['profile_exchange']['field_extensions']==payload['field_extensions']
    export_trench_csv(tmp_path/'field.csv',result,1)
    with (tmp_path/'field.csv').open(encoding='utf-8-sig',newline='') as handle:rows=list(csv.reader(handle))
    row=next(row for row in rows if row[:2]==['profile_exchange','field_extensions'])
    assert json.loads(row[2])==payload['field_extensions']


def test_existing_fields_are_not_extended_or_replaced():
    result=analyze()
    assert not any(f.get('automatic_field') for f in result.field_angles)
    assert trench_profile_exchange(result,1)['field_extensions']==[]


@pytest.mark.parametrize('variant',['wall','floor','disconnected','interpolated','steep','ambiguous'])
def test_no_field_is_invented_without_unique_upper_measured_horizontal_support(variant):
    result=analyze(guide=short_guide())
    # Remove a Field then provide unsuitable evidence without changing the
    # independently measured walls. The original support is never a fallback.
    fields=[dict(f,status='unconfirmed',points=[]) for f in result.field_angles]
    if variant=='wall':path=[[160,y] for y in range(90,240)]
    else:path=[[x,45] for x in range(120,161)]+[[160,y] for y in range(46,241)]
    if variant=='floor':path=[[x,y+280] for x,y in path]
    if variant=='disconnected':path=path[:40]
    if variant=='steep':path=[[x,y+(x-120)*.8] for x,y in path]
    traces=[dict(segments=[path],point_sources=['interpolated' if variant=='interpolated' else 'measured']*len(path))]
    if variant=='ambiguous':traces*=2
    guides=(dict(points=[[120,45],[160,45],[160,240]],half_width_px=20),)
    fields=recover_fields(fields,traces,replace(result.options,wall_guides=guides),result.y,result.left,result.right,(400,420))
    assert all(f['status']=='unconfirmed' for f in fields)


def test_extension_must_attach_to_the_actual_contour_endpoint():
    result=analyze(guide=short_guide())
    p=np.asarray(result.guide_traces[0]['points'])
    with pytest.raises(ValueError,match='끝점'):
        extend_export_surface(p+5,result.field_angles)


def test_auto_fields_ui_clipboard_project_state_and_png(tmp_path):
    from PySide6.QtWidgets import QApplication
    from angle_cal.trench_panel import TrenchPanel
    from test_trench_simple_workflow import finish
    app=QApplication.instance() or QApplication([])
    panel=TrenchPanel();panel.resize(1250,900);panel.show()
    try:
        panel.set_image(rounded_trench(15,30),'automatic-field',.5)
        panel.rim_reference.setChecked(False)
        panel.load_structure_payload(dict(wall_guides=[short_guide()]))
        panel.start_analysis();finish(panel,app)
        assert '좌측 Field 자동 구성' in panel.summary_label.text()
        assert '초록 점선' in panel.summary_label.text()
        assert sum(getattr(item,'text',lambda:None)()=='Field 자동' for item in panel.overlays)==2
        panel.copy_gfe_coordinates()
        assert 'Field 자동 연장 포함' in panel.hint.text()
        copied=app.clipboard().text()
        assert copied.startswith('X\tY\n')
        assert panel.profile_table.rowCount()==len(copied.splitlines())-1
        state=panel.state()
        assert all(f['extension']['measured'] is False for f in state['field_angles'])
        panel.save_result(str(tmp_path/'auto-field.png'))
        panel.set_image(rounded_trench(15,30),'restored',.5,state)
        panel.start_analysis();finish(panel,app);panel.copy_gfe_coordinates()
        assert app.clipboard().text()==copied
    finally:
        panel.wait_for_worker();panel.close();app.processEvents()
