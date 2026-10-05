import math
import cv2
import numpy as np
import pytest
from angle_cal.trench_analysis import TrenchOptions,analyze_trench,trench_gfe_tsv
from angle_cal.trench_surface import connect_entrance


@pytest.mark.parametrize('radius',[8,15,30,60])
@pytest.mark.parametrize('side',['left','right'])
def test_field_arc_cliff_join_has_exact_tangency(radius,side):
    sign=-1 if side=='left' else 1
    vertex=np.array([100.,20.])
    center=vertex+radius*np.array([sign,1.])
    theta=np.linspace(-np.pi/2,0 if side=='left' else -np.pi,100)
    points=center+radius*np.column_stack((np.cos(theta),np.sin(theta)))
    item=dict(side=side,angle_status='valid',line=[0.,20.],wall_line=[0.,100.],vertex=vertex.tolist())
    connect_entrance(item,dict(fit_points=points.tolist(),edge_spread_px=.5))
    connection=item['connection']
    assert connection['status']=='valid',connection
    assert connection['radius_px']==pytest.approx(radius,abs=.01)
    assert connection['field_tangent'][1]==pytest.approx(20)
    assert connection['wall_tangent'][0]==pytest.approx(100)
    assert connection['arc_points'][0]==pytest.approx(connection['field_tangent'])
    assert connection['arc_points'][-1]==pytest.approx(connection['wall_tangent'])
    # At either join the circle radius is perpendicular to its support.
    assert (np.array(connection['field_tangent'])-connection['center'])[0]==pytest.approx(0)
    assert (np.array(connection['wall_tangent'])-connection['center'])[1]==pytest.approx(0)


def test_connection_does_not_invent_radius_for_missing_contour():
    item=dict(angle_status='valid')
    connect_entrance(item,dict(fit_points=[]))
    assert item['connection']['status']=='unconfirmed'
    assert item['connection']['radius_px'] is None


@pytest.mark.parametrize('radius',[8,15,30,60])
def test_image_to_tangent_connection_known_radius(radius):
    from test_trench_corner import rounded_trench
    result=analyze_trench(rounded_trench(radius),TrenchOptions((60,45,300,335),angle_span=(100,160)))
    for item in result.field_angles:
        connection=item['connection']
        assert connection['status']=='valid',connection['reason']
        assert connection['radius_px']==pytest.approx(radius,abs=max(1,radius*.1))


def fixture(fm=.07,lm=.12,rm=-.15,noise=0,blur=0,rim=False,layers=False):
    scale=4
    yy,xx=np.mgrid[:380*scale,:360*scale].astype(float)/scale
    void=(yy<50+fm*(xx-180))|((xx>130+lm*(yy-50))&(xx<230+rm*(yy-50))&(yy<310))
    im=np.where(void,35.,205.).astype(np.float32)
    if layers:
        im[(~void)&(yy>69)&(yy<72)] = 40
    if rim:
        solid=(~void).astype(np.uint8)
        im[(solid-cv2.erode(solid,np.ones((7,7),np.uint8)))>0]=250
    im=cv2.resize(im,(360,380),interpolation=cv2.INTER_AREA)
    if blur:im=cv2.GaussianBlur(im,(0,0),blur)
    if noise:im+=np.random.default_rng(6).normal(0,noise,im.shape)
    return np.clip(im,0,255).astype(np.uint8)


@pytest.mark.parametrize('variant',['clean','noise','blur','rim','layers','bright','16bit'])
def test_known_field_tilt_and_sidewall_angles(variant):
    im=fixture(noise=3 if variant=='noise' else 0,blur=1 if variant=='blur' else 0,rim=variant=='rim',layers=variant=='layers')
    if variant=='bright':im=255-im
    if variant=='16bit':im=im.astype(np.uint16)*200
    r=analyze_trench(im,TrenchOptions((40,50,280,300),bright_trench=variant=='bright',smooth_px=3))
    for f,wall_slope in zip(r.field_angles,(.12,-.15)):
        assert f['status']=='valid',f['reason']
        assert f['angle_status']=='valid',f['angle_reason']
        assert f['line'][0]==pytest.approx(.07,abs=.012)
        sign=-1 if f['side']=='left' else 1
        expected=math.degrees(math.acos(sign*(wall_slope+.07)/(math.sqrt(1+.07**2)*math.sqrt(1+wall_slope**2))))
        assert f['angle_deg']==pytest.approx(expected,abs=1)
        assert f['rms_px']<1.5
        assert f['wall_rms_px']<1.5
    assert r.depth_px is not None
    points=np.loadtxt(trench_gfe_tsv(r,.5).splitlines()[1:])
    assert abs(points[0,1]-points[-1,1])>10 # Measured sloped Field survives export.


def test_missing_field_does_not_invent_angle_or_export_plane():
    im=fixture(fm=0,lm=0,rm=0)
    im[:50]=205 # No material/space boundary outside the mouth.
    r=analyze_trench(im,TrenchOptions((40,50,280,300)))
    assert all(f['status']=='unconfirmed' for f in r.field_angles)
    assert r.left_angle_deg is None and r.right_angle_deg is None
    with pytest.raises(ValueError,match='Field'):
        trench_gfe_tsv(r,.5)


def test_surface_controls_are_independent_of_cd_depth_bowing_and_corner_range():
    im=fixture()
    a=analyze_trench(im,TrenchOptions((40,50,280,300)))
    b=analyze_trench(im,TrenchOptions((40,50,280,300),field_search_px=40,angle_span=(55,120),corner_window_px=48))
    for name in ('y','left','right','sample_left','sample_right'):
        assert np.array_equal(getattr(a,name),getattr(b,name))
    assert a.depth_px==b.depth_px
    assert a.left_bowing_px==b.left_bowing_px
    assert a.right_bowing_px==b.right_bowing_px


def test_angle_short_or_curved_support_is_unconfirmed():
    im=fixture()
    r=analyze_trench(im,TrenchOptions((40,50,280,300),angle_span=(40,45)))
    assert all(f['angle_status']=='unconfirmed' for f in r.field_angles)
    assert all(f['status']=='valid' for f in r.field_angles)


def test_curved_wall_is_not_reported_as_single_entrance_angle():
    yy,xx=np.mgrid[:380,:360]
    left=130+8*np.sin((yy-50)/10)
    void=(yy<50)|((xx>left)&(xx<230)&(yy<310))
    im=np.where(void,35,205).astype(np.uint8)
    r=analyze_trench(im,TrenchOptions((40,50,280,300),smooth_px=1))
    assert r.field_angles[0]['status']=='valid'
    assert r.field_angles[0]['angle_status']=='unconfirmed'
    assert r.field_angles[0]['angle_deg'] is None
    assert r.field_angles[1]['angle_status']=='valid'
