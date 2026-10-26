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


def test_depth_confirmation_is_not_required_to_copy_an_existing_contour(result):
    unconfirmed=replace(result,depth_px=None,
                        field_measurements=dict(result.field_measurements,depth_px=None,depth_status='unconfirmed'))
    payload=trench_profile_exchange(unconfirmed,.5)
    assert payload['points']==trench_profile_exchange(result,.5)['points']
    assert payload['bottom_detected'] is False
    assert unconfirmed.depth_px is None and unconfirmed.field_measurements['depth_px'] is None
    assert trench_gfe_tsv(unconfirmed,.5)==trench_gfe_tsv(result,.5)


def test_missing_field_unscaled_or_broken_surface_cannot_export_fake_connections(result):
    for scale in (0,-1,float('nan')):
        with pytest.raises(ValueError):trench_profile_exchange(result,scale)
    with pytest.raises(ValueError,match='Field'):
        trench_profile_exchange(replace(result,field_angles=[]),.5)
    t=dict(result.guide_traces[0]);p=t['points'];t['segments']=[p[:100],p[120:]]
    with pytest.raises(ValueError,match='연결'):
        trench_profile_exchange(replace(result,guide_traces=[t]),.5)


def split_bottom_result(result, separate=False, reverse=False, unequal=False):
    expected=[[-1500,0],[-250,0],[-100,-900],[-300,-1600],[-125,-4000],
              [125,-4100 if unequal else -4000],[300,-1600],[250,0],[1500,0]]
    r=example_result(result,expected)
    p=r.guide_traces[0]['points'];parts=[p[:5],p[5:]]
    if reverse:parts=[s[::-1] for s in parts[::-1]]
    guides=tuple(dict(points=s,half_width_px=3) for s in parts)
    traces=([dict(points=s,segments=[s],interpolated_points=0) for s in parts] if separate else
            [dict(points=sum(parts,[]),segments=parts,interpolated_points=0)])
    return replace(r,depth_px=None,options=replace(r.options,wall_guides=guides),guide_traces=traces),expected


@pytest.mark.parametrize('separate',[False,True])
@pytest.mark.parametrize('reverse',[False,True])
@pytest.mark.parametrize('unequal',[False,True])
def test_only_terminal_bottom_gap_is_joined_without_moving_points(result,separate,reverse,unequal):
    r,expected=split_bottom_result(result,separate,reverse,unequal)
    before=json.dumps(r.guide_traces)
    payload=trench_profile_exchange(r,1)
    assert payload['points']==expected
    connection=payload['bottom_connection']
    assert connection['points']==expected[4:6]
    assert connection['method']=='straight_terminal_join' and connection['measured'] is False
    assert r.depth_px is None and json.dumps(r.guide_traces)==before
    assert payload['surface_closed'] is False and payload['bottom_detected'] is False
    assert r.to_dict(1)['profile_exchange']==payload


def test_actual_open_bottom_image_exports_both_walls_and_marks_connection(result,tmp_path):
    import csv
    from angle_cal.trench_analysis import TrenchOptions,export_trench_csv
    from angle_cal.trench_workflow import recognize_from_guides
    from test_trench_corner import rounded_trench
    from test_trench_guided_surface import surface_guide
    image=rounded_trench(15,30);image[345:,161:260]=35
    g=surface_guide(15,30)
    guides=(dict(g,points=g['points'][:6]),dict(g,points=g['points'][10:]))
    r=recognize_from_guides(image,TrenchOptions((0,0,420,400),wall_guides=guides,boundary_mode='max_gradient'))
    payload=trench_profile_exchange(r,.5)
    assert payload['bottom_connection'] and not payload['bottom_connection']['measured']
    assert r.depth_px is None and r.field_measurements['depth_px'] is None
    assert len(payload['points'])==sum(len(t['points']) for t in r.guide_traces)
    path=tmp_path/'open.csv';export_trench_csv(path,r,.5)
    with path.open(encoding='utf-8-sig',newline='') as f:rows=list(csv.reader(f))
    row=next(row for row in rows if row[:2]==['profile_exchange','bottom_connection'])
    assert json.loads(row[2])==payload['bottom_connection']


def test_additional_wall_break_is_not_repaired_as_bottom(result):
    r,_=split_bottom_result(result,separate=True)
    traces=[dict(t) for t in r.guide_traces]
    traces[0]['segments']=[traces[0]['points'][:2],traces[0]['points'][3:]]
    with pytest.raises(ValueError,match='맨 아래'):
        trench_profile_exchange(replace(r,guide_traces=traces),1)
