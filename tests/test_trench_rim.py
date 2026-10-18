import json
import cv2
import numpy as np
import pytest
from PySide6.QtCore import Qt,QPoint
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication
from angle_cal.trench_contour import trace_guide
from angle_cal.trench_guides import points_in_guides
from angle_cal.trench_panel import TrenchPanel


def roof_image():
    yy,xx=np.mgrid[:320,:320]
    inside=(xx>90)&(xx<230)&(yy>45)&(yy<270)
    im=np.where(inside,45.,150.)
    # Symmetric roof-shaped crest; a broad cavity-side halo is deliberately
    # asymmetric and must not become the picked gradient flank.
    for x in (90,230):im+=92*np.exp(-((xx-x)/3.)**2)
    im+=65*np.exp(-((xx-104)/12.)**2)*(xx>90)*(xx<160)
    im+=92*np.exp(-((yy-270)/3.)**2)
    im+=92*np.exp(-((yy-45)/3.)**2)
    return cv2.GaussianBlur(im.astype(np.float32),(3,3),.6)


@pytest.mark.parametrize('points,direction',[
    ([[90,80],[90,230]],[1,0]),([[230,80],[230,230]],[-1,0]),
    ([[120,45],[200,45]],[0,1]),([[120,270],[200,270]],[0,-1])])
@pytest.mark.parametrize('reverse',[False,True])
def test_peak_then_equal_inset_follows_void_on_all_sides(points,direction,reverse):
    g=dict(points=points[::-1] if reverse else points,half_width_px=15)
    image=roof_image()
    zero=trace_guide(image,g,65,'rim_inset',True,150,rim_inset_px=0)
    result=trace_guide(image,g,65,'rim_inset',True,150,rim_inset_px=1.25)
    assert result['trace_coverage']>.95
    p=np.asarray(result['points']);peak=np.asarray(result['rim_peak_points'])
    np.testing.assert_allclose(np.linalg.norm(p-peak,axis=1),1.25,atol=1e-8)
    assert np.median((p-peak)@np.asarray(direction))>1.2
    np.testing.assert_allclose(peak,np.asarray(zero['points']),atol=1e-8)
    assert points_in_guides(p,(g,)).all()
    # Reject the halo as an alternative boundary far inside the void.
    axis=0 if direction[0] else 1
    assert np.median(abs(peak[:,axis]-points[0][axis]))<3


def test_monotonic_brightness_ramp_is_not_a_white_rim():
    x=np.arange(260)
    image=np.tile(30+160/(1+np.exp((x-120)/9)),(200,1)).astype(np.float32)
    g=dict(points=[[120,30],[120,170]],half_width_px=30)
    t=trace_guide(image,g,70,'rim_inset',True,180)
    assert not t['points']


def test_shaded_void_does_not_reverse_the_inset_direction():
    xx=np.arange(320,dtype=float)
    row=np.where(xx<=90,150.,35+275*np.exp(-np.maximum(xx-90,0)/20))
    row+=70*np.exp(-((xx-90)/3)**2)
    image=np.tile(row,(240,1)).astype(np.float32)
    g=dict(points=[[90,40],[90,200]],half_width_px=15)
    t=trace_guide(image,g,65,'rim_inset',True,150)
    assert t['trace_coverage']>.95
    for point,peak in zip(t['points'],t['rim_peak_points']):
        assert point[0]>peak[0]+.99


@pytest.mark.parametrize('inverted,dtype',[(False,np.uint8),(True,np.uint8),(False,np.uint16)])
def test_full_analysis_polarity_bitdepth_and_exported_reference(inverted,dtype):
    from angle_cal.trench_analysis import TrenchOptions
    from angle_cal.trench_workflow import recognize_from_guides
    image=np.clip(roof_image(),0,255).astype(np.uint8)
    if inverted:image=255-image
    if dtype==np.uint16:image=image.astype(dtype)*200
    g=dict(points=[[90,80],[90,240],[100,270],[220,270],[230,240],[230,80]],half_width_px=18)
    r=recognize_from_guides(image,TrenchOptions((0,0,320,320),wall_guides=(g,),boundary_mode='rim_inset'))
    assert r.options.bright_trench==inverted
    data=r.to_dict(.5)
    assert data['wall_boundary']=='bright_crest_plus_inward_normal_offset'
    assert data['options']['rim_inset_px']==1
    for t in r.guide_traces:
        for point,peak in zip(t['points'],t['rim_peak_points']):
            if peak is None:continue
            assert np.linalg.norm(np.subtract(point,peak))==pytest.approx(1)


def test_inset_cannot_escape_corridor():
    image=roof_image();g=dict(points=[[89,80],[89,230]],half_width_px=1)
    t=trace_guide(image,g,65,'rim_inset',True,150,rim_inset_px=10)
    assert not t['points']


def test_inset_setting_and_reference_round_trip():
    app=QApplication.instance() or QApplication([])
    p=TrenchPanel();p.set_image(roof_image(),'rim',.5)
    assert p.options().boundary_mode=='rim_inset' and p.options().rim_inset_px==1
    p.rim_inset_spin.setValue(1.75)
    state=json.loads(json.dumps(p.state()))
    p.set_image(roof_image(),'rim',.5,state)
    assert p.options().rim_inset_px==1.75
    p.unit_combo.setCurrentIndex(p.unit_combo.findData('angstrom'))
    assert p.options().rim_inset_px==1.75
    p.rim_reference.setChecked(False)
    assert p.options().boundary_mode==p.boundary_mode
    p.close()


@pytest.mark.parametrize('drawing',[False,True])
def test_ctrl_drag_over_guide_pans_without_editing(drawing):
    app=QApplication.instance() or QApplication([])
    p=TrenchPanel();p.resize(1000,750);p.show();p.set_image(roof_image(),'pan')
    g=dict(points=[[90,80],[90,230]],half_width_px=15)
    p.load_structure_payload(dict(wall_guides=[g]));app.processEvents()
    p.view.scale(3,3);p.view.centerOn(90,150)
    if drawing:p.start_contour_guide()
    old=json.dumps(p.guide_editor.guides);draft=p.guide_editor.draft[:];revision=p.revision
    a=p.view.mapFromScene(90,150);b=a+QPoint(35,20)
    bars=(p.view.horizontalScrollBar(),p.view.verticalScrollBar());before=[v.value() for v in bars]
    QTest.mousePress(p.view.viewport(),Qt.MouseButton.LeftButton,Qt.KeyboardModifier.ControlModifier,pos=a)
    QTest.mouseMove(p.view.viewport(),b)
    QTest.mouseRelease(p.view.viewport(),Qt.MouseButton.LeftButton,pos=b)
    assert [v.value() for v in bars]!=before
    assert p.view.ctrl_pan_last is None
    assert json.dumps(p.guide_editor.guides)==old and p.guide_editor.draft==draft and p.revision==revision
    assert p.guide_editor.drawing==drawing
    p.close()
