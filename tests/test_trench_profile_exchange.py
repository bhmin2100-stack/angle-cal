from dataclasses import replace
import json
import numpy as np
import pytest

from angle_cal.trench_analysis import trench_profile_exchange, trench_gfe_tsv
from test_trench_field_metrics import analyze


@pytest.fixture(scope='module')
def result():return analyze()


def example_result(result, physical, reverse=False):
    points=np.column_stack((np.asarray(physical)[:,0]/10+200,10-np.asarray(physical)[:,1]/10)).tolist()
    if reverse:points=points[::-1]
    g=dict(points=points,half_width_px=3)
    fields=[dict(side='left',status='valid',source='guide',points=[[50,10],[175,10]],line=[0,10]),
            dict(side='right',status='valid',source='guide',points=[[225,10],[350,10]],line=[0,10])]
    m=dict(reference_y_px=10,anchors=dict(left=[175,10],right=[225,10]))
    return replace(result,options=replace(result.options,wall_guides=(g,)),depth_px=400,
                   y=np.array([10.,410.]),left=np.array([175.,187.5]),right=np.array([225.,212.5]),
                   field_angles=fields,field_measurements=m,
                   guide_traces=[dict(points=points,segments=[points],interpolated_points=0)])


@pytest.mark.parametrize('reverse',[False,True])
def test_documented_six_points_units_order_normals_and_open_surface(result,reverse):
    expected=[[-1500,0],[-250,0],[-125,-4000],[125,-4000],[250,0],[1500,0]]
    r=example_result(result,expected,reverse)
    payload=trench_profile_exchange(r,1)
    assert payload['points']==expected
    assert payload['schema']=='trench-profile/v1' and payload['units']=='A'
    assert payload['coordinate_system']==dict(x_positive='right',y_positive='up')
    assert payload['surface_order']=='left_to_right' and not payload['surface_closed']
    assert payload['initial_voids']==[] and payload['initial_voids_source']=='not_analyzed'
    assert payload['points'][0]!=payload['points'][-1]
    assert payload['origin_pixel']==[200,10]
    assert json.loads(json.dumps(payload))==payload
    np.testing.assert_allclose(np.loadtxt(trench_gfe_tsv(r,1).splitlines()[1:]),expected)
    d=np.diff(payload['points'],axis=0);normal=np.column_stack((-d[:,1],d[:,0]))
    assert normal[0,1]>0 and normal[1,0]>0 and normal[3,0]<0


def test_overhang_and_curved_floor_order_is_never_sorted_or_resampled(result):
    expected=[[-1500,0],[-250,0],[-100,-900],[-300,-1600],[-125,-3900],
              [-50,-3990],[0,-4000],[50,-3990],[125,-3900],[300,-1600],[100,-900],[250,0],[1500,0]]
    r=example_result(result,expected)
    assert trench_profile_exchange(r,1)['points']==expected
    reverse=example_result(result,expected,True)
    assert trench_profile_exchange(reverse,1)['points']==expected
    assert np.any(np.diff(np.asarray(expected)[:,0])<0)


def test_real_trace_coordinates_use_same_field_origin_and_bottom_as_metrics(result):
    payload=trench_profile_exchange(result,.5)
    p=np.asarray(result.guide_traces[0]['points'])
    expected=np.column_stack(((p[:,0]-payload['origin_pixel'][0])*5,(payload['origin_pixel'][1]-p[:,1])*5))
    np.testing.assert_allclose(payload['points'],expected)
    assert -np.min(np.asarray(payload['points'])[:,1])==pytest.approx(result.field_measurements['depth_px']*5)
    assert result.to_dict(.5)['profile_exchange']==payload


def test_missing_field_unscaled_or_broken_surface_cannot_export_fake_connections(result):
    for scale in (0,-1,float('nan')):
        with pytest.raises(ValueError):trench_profile_exchange(result,scale)
    with pytest.raises(ValueError,match='Field'):
        trench_profile_exchange(replace(result,field_angles=[]),.5)
    t=dict(result.guide_traces[0]);p=t['points'];t['segments']=[p[:100],p[120:]]
    with pytest.raises(ValueError,match='연결'):
        trench_profile_exchange(replace(result,guide_traces=[t]),.5)
