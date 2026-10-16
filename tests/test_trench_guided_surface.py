import json
import cv2
import numpy as np
import pytest

from angle_cal.trench_analysis import TrenchOptions,analyze_trench,trench_gfe_tsv,export_trench_csv
from angle_cal.trench_guides import normalize_guides
from test_trench_corner import rounded_trench


def surface_guide(radius=30,right_radius=None):
    rr=radius if right_radius is None else right_radius
    return dict(points=[[65,45],[160-radius-15,45],[160-radius,45],
                        [160-radius*.3,45+radius*.3],[160,45+radius],
                        [160,310],[160,345],[190,350],[230,350],[260,345],[260,310],
                        [260,45+rr],[260+rr*.3,45+rr*.3],[260+rr,45],[260+rr+15,45],[355,45]],
                half_width_px=18)


def analyze(im,g,**kw):
    return analyze_trench(im,TrenchOptions((60,45,300,335),wall_guides=(g,),angle_span=(100,160),**kw))


@pytest.mark.parametrize('radius',[8,15,30,60])
def test_guided_field_arc_cliff_share_actual_trace_known_radii(radius):
    r=analyze(rounded_trench(radius),surface_guide(radius))
    evidence=np.asarray(r.guide_traces[0]['points'])
    for f,c in zip(r.field_angles,r.corners):
        assert f['source']=='guide' and f['status']=='valid'
        assert f['angle_deg']==pytest.approx(90,abs=1)
        assert c['source']=='guide' and c['status']=='valid',c['reason']
        assert c['radius_px']==pytest.approx(radius,abs=max(1,radius*.1))
        assert f['connection']['status']=='valid',f['connection']['reason']
        # Field and arc are resampled along measured evidence, never on the
        # clicked curve. Interpolation may displace points by at most 1px.
        for point in np.asarray(f['points']+c['fit_points'])[::3]:
            assert np.linalg.norm(evidence-point,axis=1).min()<1.1


@pytest.mark.parametrize('variant',['reverse','noise','blur','rim','layers','bright','16bit'])
def test_asymmetric_guided_surface_variants(variant):
    im=rounded_trench(15,30,noise=2 if variant=='noise' else 0,blur=.6 if variant=='blur' else 0,
                      rim=variant=='rim',layers=variant=='layers')
    g=surface_guide(15,30)
    if variant=='reverse':g=dict(g,points=g['points'][::-1])
    if variant=='bright':im=255-im
    if variant=='16bit':im=im.astype(np.uint16)*201
    r=analyze(im,g,bright_trench=variant=='bright')
    for f,c,radius in zip(r.field_angles,r.corners,(15,30)):
        assert f['status']=='valid' and f['source']=='guide'
        assert f['angle_deg']==pytest.approx(90,abs=2)
        assert c['radius_px']==pytest.approx(radius,abs=max(1.5,radius*.12)),c['reason']


def curved_field_fixture():
    s=4;yy,xx=np.mgrid[:400*s,:420*s].astype(float)/s
    d=np.clip(yy-45,0,30)
    left=160-(30-np.sqrt(np.maximum(0,900-(30-d)**2)))
    right=260+(30-np.sqrt(np.maximum(0,900-(30-d)**2)))
    field=45+.0025*np.maximum(130-xx,0)**2+.0025*np.maximum(xx-290,0)**2
    void=(yy<field)|((xx>left)&(xx<right)&(yy<350))
    image=cv2.resize(np.where(void,35.,205.).astype(np.float32),(420,400),interpolation=cv2.INTER_AREA).astype(np.uint8)
    g=surface_guide();g['half_width_px']=12
    g['points'][:3]=[[65,55.5],[95,48],[130,45]]
    g['points'][-3:]=[[290,45],[325,48],[355,55.5]]
    return image,g


def test_curved_field_keeps_contour_and_distinguishes_local_tangent_from_straight_reference():
    im,g=curved_field_fixture();r=analyze(im,g)
    for f,c in zip(r.field_angles,r.corners):
        assert f['curved'] and f['angle_basis']=='local_tangent'
        assert f['local_angle_deg']==pytest.approx(90,abs=2)
        assert abs(f['straight_angle_deg']-f['local_angle_deg'])>4
        assert np.ptp(np.asarray(f['points'])[:,1])>8
        assert c['radius_px']==pytest.approx(30,abs=3)
        assert f['connection']['status']=='valid'
        assert len(f['connection']['field_join_points'])>=2
        for point in f['connection']['field_join_points']:
            assert np.linalg.norm(np.asarray(r.guide_traces[0]['points'])-point,axis=1).min()<1.1
    coordinates=np.loadtxt(trench_gfe_tsv(r,.5).splitlines()[1:])
    assert np.ptp(coordinates[:30,1])>8  # No flattened Field in GFE.


def test_field_only_horizontal_guides_and_legacy_wall_only_guides():
    im=rounded_trench(30)
    guides=normalize_guides((dict(points=[[65,45],[120,45]],half_width_px=8),
                             dict(points=[[300,45],[355,45]],half_width_px=8)))
    r=analyze_trench(im,TrenchOptions((60,45,300,335),wall_guides=guides,angle_span=(100,160)))
    assert all(f['source']=='guide' and f['status']=='valid' for f in r.field_angles)
    assert all(c['status']=='unconfirmed' for c in r.corners)  # No guided arc evidence.
    automatic=analyze_trench(im,TrenchOptions((60,45,300,335),angle_span=(100,160)))
    for key in ('y','left','right','sample_left','sample_right'):
        assert np.array_equal(getattr(r,key),getattr(automatic,key))
    legacy=analyze(im,dict(points=[[160,110],[160,300]],half_width_px=8))
    assert all(f.get('source')!='guide' for f in legacy.field_angles)


def test_field_guide_does_not_fall_back_to_unrelated_automatic_layer():
    im=rounded_trench(30);im[:60,:125]=205
    r=analyze(im,surface_guide())
    assert r.field_angles[0]['source']=='guide'
    assert r.field_angles[0]['status']=='unconfirmed'
    assert r.left_angle_deg is None
    assert r.corners[0]['radius_px'] is None


@pytest.mark.parametrize('sigma',[0,3])
def test_sharp_guided_corner_has_no_resolved_radius(sigma):
    im=rounded_trench(0,blur=sigma)
    g=surface_guide(8)
    r=analyze(im,g)
    assert all(c['radius_px'] is None for c in r.corners)


def test_arc_selection_is_independent_of_guided_angle_and_measurements(tmp_path):
    im=rounded_trench(30);g=surface_guide()
    a=analyze(im,g);b=analyze(im,g,corner_spans=((.4,.6),None))
    assert b.corners[0]['radius_px'] is None
    assert a.left_angle_deg==b.left_angle_deg
    for key in ('y','left','right','sample_left','sample_right'):
        assert np.array_equal(getattr(a,key),getattr(b,key))
    assert a.depth_px==b.depth_px and a.left_bowing_px==b.left_bowing_px
    payload=json.loads(json.dumps(b.to_dict(.5),allow_nan=False))
    assert payload['field_angles'][0]['source']=='guide'
    assert payload['corners'][0]['source']=='guide'
    csv=tmp_path/'surface.csv';export_trench_csv(csv,b,.5)
    assert 'left_angle_basis' in csv.read_text(encoding='utf-8-sig')


def test_curved_field_angle_is_independent_of_radius_selection_rectangle_and_window():
    im,g=curved_field_fixture();a=analyze(im,g)
    b=analyze(im,g,corner_spans=((.4,.6),(.4,.6)),corner_rois=((140,50,30,30),(250,50,30,30)),corner_window_px=48)
    assert [f['angle_deg'] for f in a.field_angles]==[f['angle_deg'] for f in b.field_angles]
    assert all(c['status']=='unconfirmed' for c in b.corners)


def test_guided_sloped_field_and_asymmetric_cliff_angles():
    from test_trench_surface import fixture
    import math
    guide=dict(points=[[45,40.55],[90,43.7],[129.6,46.5],[136,100],[155,280],
                       [160,305],[175,310],[191,305],[194,290],[222.5,100],
                       [229.5,53.5],[270,56.3],[315,59.45]],half_width_px=10)
    r=analyze_trench(fixture(),TrenchOptions((40,50,280,300),wall_guides=(guide,)))
    for f,slope,sign in zip(r.field_angles,(.12,-.15),(-1,1)):
        assert f.get('source')=='guide' and f['status']=='valid'
        assert f['line'][0]==pytest.approx(.07,abs=.012)
        expected=math.degrees(math.acos(sign*(slope+.07)/(math.sqrt(1+.07**2)*math.sqrt(1+slope**2))))
        assert f['angle_deg']==pytest.approx(expected,abs=1)


def test_curved_field_only_guide_is_not_mistaken_for_a_wall():
    im,_=curved_field_fixture()
    guide=dict(points=[[30,70],[60,57.25],[95,48.06],[125,45.06]],half_width_px=8)
    r=analyze(im,guide)
    assert r.field_angles[0]['source']=='guide' and r.field_angles[0]['status']=='valid'
    assert r.corners[0]['status']=='unconfirmed'
    assert r.depth_px is not None


def test_guided_radius_search_size_restricts_only_radius_not_field_angle():
    im=rounded_trench(60);g=surface_guide(60)
    a=analyze(im,g);b=analyze(im,g,corner_window_px=6)
    assert all(c['status']=='valid' for c in a.corners)
    assert all(c['status']=='unconfirmed' for c in b.corners)
    assert [f['angle_deg'] for f in a.field_angles]==[f['angle_deg'] for f in b.field_angles]
