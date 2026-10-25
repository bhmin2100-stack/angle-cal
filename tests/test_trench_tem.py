import json
from pathlib import Path
import cv2
import numpy as np
import pytest
from angle_cal.trench_analysis import TrenchOptions, trench_profile_exchange
from angle_cal.trench_workflow import recognize_from_guides
from angle_cal.trench_guides import points_in_guides

ROOT=Path(__file__).resolve().parents[1]


def tem_image(width=8, noise=0, blur=.7):
    yy,xx=np.mgrid[:360,:320]
    void=(yy<40)|((xx>=90)&(xx<=210)&(yy<=240))|(((xx-150)**2+(yy-240)**2<=60**2)&(yy>=240))
    distance=cv2.distanceTransform((~void).astype(np.uint8),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
    film=(~void)&(distance<=width)
    image=np.where(void,240.,np.where(film,15.,145+12*np.sin(yy/3)))
    image+=np.random.default_rng(12).normal(0,noise,image.shape)
    image=cv2.GaussianBlur(image,(0,0),blur)
    return np.uint8(np.clip(image,0,255))


def tem_guide(reverse=False):
    points=[[40,39],[75,39],[90,42],[90,130],[90,235],[98,270],[120,292],
            [150,300],[180,292],[202,270],[210,235],[210,130],[210,42],[235,39],[270,39]]
    return dict(points=points[::-1] if reverse else points,half_width_px=25)


@pytest.mark.parametrize('width,reverse,noise,blur,bit16,invert',[
    (3,False,0,.5,False,False),(8,True,0,.7,False,False),
    (14,False,4,1.2,True,False),(8,False,3,.8,False,True)])
def test_tem_measures_gray_material_end_not_dark_band_inner_side(width,reverse,noise,blur,bit16,invert):
    image=tem_image(width,noise,blur)
    if invert:image=255-image
    if bit16:image=image.astype(np.uint16)*211
    guide=tem_guide(reverse)
    r=recognize_from_guides(image,TrenchOptions((0,0,320,360),step_px=2,
        wall_guides=(guide,),boundary_mode='rim_inset',imaging_mode='tem' if invert else 'auto'))
    assert r.imaging['selected']=='tem'
    assert r.options.boundary_mode=='tem_interface'
    mid=(r.y>100)&(r.y<220)
    assert np.median(r.left[mid])==pytest.approx(89.5-width,abs=1.5)
    assert np.median(r.right[mid])==pytest.approx(210.5+width,abs=1.5)
    assert r.field_measurements['reference_y_px']==pytest.approx(39.5+width,abs=2)
    assert r.field_measurements['depth_px']==pytest.approx(261,abs=2)
    trace=r.guide_traces[0]
    assert trace['continuity_complete'] and points_in_guides(trace['points'],(guide,)).all()
    assert trace['method']=='tem_gray_material_interface'
    exported=trench_profile_exchange(r,1)
    assert exported['units']=='A' and not exported['surface_closed']
    assert exported['points'][0][0]<exported['points'][-1][0]


@pytest.mark.parametrize('side',['left','right'])
def test_user_tem_reference_complete_and_material_side(side):
    image=cv2.imdecode(np.fromfile(ROOT/'samples/trench-tem-reference.png',np.uint8),-1)
    guide=json.loads((ROOT/f'samples/trench-tem-reference-{side}.anglecal.structure.json').read_text())['wall_guides'][0]
    r=recognize_from_guides(image,TrenchOptions((0,0,1536,1024),wall_guides=(guide,),
          boundary_mode='rim_inset',imaging_mode='auto',contour_strength=45))
    assert r.imaging['selected']=='tem'
    trace=r.guide_traces[0]
    assert trace['trace_coverage']>.98 and len(trace['segments'])==1
    assert points_in_guides(trace['points'],(guide,)).all()
    assert r.field_measurements['field_status']=='valid'
    assert r.field_measurements['tcd_status']=='valid'
    assert r.field_measurements['depth_status']=='valid'
    # Independent image-only reference: the outer limits of the dark band on
    # horizontal mid-wall slices. Do not use the tracer's fitted coordinates.
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) if image.ndim==3 else image
    for y in (200,400,600,780):
        row=np.median(gray[y-2:y+3],axis=0)
        window=(500,720) if side=='left' else (790,1020)
        band=np.flatnonzero(row[window[0]:window[1]]<80)+window[0]
        assert abs(np.interp(y,r.y,r.left)-(band.min()-.5))<3
        assert abs(np.interp(y,r.y,r.right)-(band.max()+.5))<3
    assert len(trench_profile_exchange(r,1)['points'])>1700


def test_auto_preserves_existing_vsem_reference_exactly():
    from dataclasses import replace
    image=cv2.imdecode(np.fromfile(ROOT/'samples/trench-reference.png',np.uint8),-1)
    g=json.loads((ROOT/'samples/trench-reference-left.anglecal.structure.json').read_text())['wall_guides'][0]
    options=TrenchOptions((0,0,1536,1024),wall_guides=(g,),boundary_mode='rim_inset',contour_strength=45)
    previous=recognize_from_guides(image,options)
    automatic=recognize_from_guides(image,replace(options,imaging_mode='auto'))
    assert automatic.imaging['selected']=='vsem'
    np.testing.assert_array_equal(previous.guide_traces[0]['points'],automatic.guide_traces[0]['points'])
    np.testing.assert_array_equal(previous.sample_left,automatic.sample_left)


def test_tem_ui_persistence_and_export(tmp_path):
    from PySide6.QtWidgets import QApplication
    from angle_cal.trench_panel import TrenchPanel
    from test_trench_simple_workflow import finish
    app=QApplication.instance() or QApplication([])
    p=TrenchPanel()
    try:
        p.set_image(tem_image(),'tem',1)
        assert p.imaging_mode.currentData()=='auto'
        p.rim_reference.setChecked(False)  # VSEM-only controls cannot alter TEM.
        p.load_structure_payload(dict(wall_guides=[tem_guide()]))
        p.start_analysis();finish(p,app)
        assert p.result is not None,p.summary_label.text()
        assert 'TEM' in p.summary_label.text() and '회색 재료 끝' in p.summary_label.text()
        for suffix in ('json','csv','png'):p.save_result(str(tmp_path/f'tem.{suffix}'))
        assert json.loads((tmp_path/'tem.json').read_text(encoding='utf-8'))['imaging']['selected']=='tem'
        assert 'tem_gray_material_interface' in (tmp_path/'tem.csv').read_text(encoding='utf-8-sig')
        p.copy_gfe_coordinates();assert app.clipboard().text().startswith('X\tY\n')
        p.imaging_mode.setCurrentIndex(p.imaging_mode.findData('tem'))
        assert not p.rim_inset_spin.isEnabled()
        state=p.state();p.set_image(tem_image(),'restored',1,state)
        assert p.imaging_mode.currentData()=='tem'
        assert p.state()['wall_guides']
    finally:p.wait_for_worker();p.close();app.processEvents()
