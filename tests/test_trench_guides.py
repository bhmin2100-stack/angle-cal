import json
import math
import numpy as np
import pytest
from PySide6.QtCore import QPoint, QPointF, QRectF, Qt
from PySide6.QtGui import QWheelEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from angle_cal.trench_analysis import TrenchOptions, analyze_trench, export_trench_csv
from angle_cal.trench_guides import normalize_guides, curve_samples, sweep_samples, row_bands, in_bands, guided_edges, prepare_guide_bands
from angle_cal.trench_guide_view import guide_sweep_path
from angle_cal.trench_panel import TrenchPanel
from test_trench_analysis import trench_image, finish_analysis


def curve(side, points, width=5):
    return dict(side=side,points=points,half_width_px=width)


def bowed_guides():
    yy=[23,60,85,110,135,165,195,218]
    half=lambda y:30+12*math.exp(-((y-110)/28)**2)
    return (curve('left',[[100-half(y),y] for y in yy]),
            curve('right',[[100+half(y),y] for y in yy]))


def test_curve_passes_all_clicked_points_and_is_smooth():
    guide=curve('left',[[50,20],[45,55],[30,120],[51,200],[48,215]])
    samples=curve_samples(guide)
    for p in guide['points']:
        index=int(np.argmin(np.linalg.norm(samples-p,axis=1)))
        assert samples[index]==pytest.approx(p)
        if 0<index<len(samples)-1:
            before=samples[index]-samples[index-1];after=samples[index+1]-samples[index]
            assert np.dot(before,after)/np.linalg.norm(before)/np.linalg.norm(after)>.99
    assert np.isfinite(samples).all()
    assert len(samples)>200


def distance_to_sampled_curve(points, samples):
    a,b=samples[:-1],samples[1:]
    delta=b-a
    along=np.clip(np.sum((points[:,None]-a)*delta,axis=2)/np.maximum(np.sum(delta*delta,axis=1),1e-20),0,1)
    return np.min(np.linalg.norm(points[:,None]-(a+along[:,:,None]*delta),axis=2),axis=1)


@pytest.mark.parametrize('points',[
    [[50,20],[50,100]],
    [[20,20],[100,100]],
    [[50,20],[25,50],[75,52],[50,100]],  # tight S bend
    [[40,20],[40,80],[47,85],[47,25]],  # overlapping return/hairpin
])
@pytest.mark.parametrize('radius',[1,5,24,75,300])
def test_sweep_search_and_display_equal_distance_to_curve(points,radius):
    guide=curve('left',points,radius)
    samples=sweep_samples(guide)
    rng=np.random.default_rng(317)
    probes=rng.uniform(samples.min(0)-radius-2,samples.max(0)+radius+2,(350,2))
    distances=distance_to_sampled_curve(probes,samples)
    displayed=guide_sweep_path(guide)
    for point,distance in zip(probes,distances):
        # Qt approximates round arcs when flattening the union's outline.
        # Exclude only that subpixel boundary tolerance from the UI assertion.
        assert in_bands(point[0],row_bands((guide,),point[1],'left'))==(distance<=radius)
        if abs(distance-radius)>.25:
            assert displayed.contains(QPointF(*point))==(distance<radius)


def test_sweep_centerline_preserves_knots_and_subpixel_curve_shape():
    guide=curve('left',[[100,30],[70,300],[100,700],[80,1200]],15)
    samples=sweep_samples(guide)
    for point in guide['points']:
        assert np.min(np.linalg.norm(samples-point,axis=1))<1e-10
    assert np.max(distance_to_sampled_curve(curve_samples(guide),samples))<.021


def test_wide_sweep_has_no_numerical_holes_at_overlapping_joins():
    guide=curve('left',[[60,35],[25,105],[95,145],[25,185],[65,300]],75)
    # Known Qt stroke-union slivers; all are strictly inside the distance band.
    points=np.array([[107.655964,213.364350],[110.397337,218.967102],
                     [48.359382,87.735533],[44.698971,200.387123]])
    assert np.all(distance_to_sampled_curve(points,sweep_samples(guide))<74)
    path=guide_sweep_path(guide)
    assert all(path.contains(QPointF(*point)) for point in points)


def test_sweep_keeps_a_real_hole_when_a_curve_loops_around_it():
    guide=curve('left',[[20,20],[80,20],[80,80],[20,80],[20,20]],5)
    path=guide_sweep_path(guide)
    assert not path.contains(QPointF(50,50))
    assert not in_bands(50,row_bands((guide,),50,'left'))
    assert path.contains(QPointF(20,20))


def test_sweep_round_caps_and_merged_overlaps_have_no_clipped_or_cancelled_band():
    guide=curve('left',[[50,20],[50,100]],10)
    assert row_bands((guide,),14,'left')==pytest.approx([(42,58)])
    assert row_bands((guide,),106,'left')==pytest.approx([(42,58)])
    assert row_bands((guide,),60,'left')==pytest.approx([(40,60)])
    assert not row_bands((guide,),111,'left')
    neighbor=curve('left',[[55,20],[55,100]],10)
    bands,per_guide=prepare_guide_bands((guide,neighbor),0,120)
    assert bands[60,'left']==[(40,65)]
    assert per_guide[0][60]==[(40,60)]
    assert per_guide[1][60]==[(45,65)]


@pytest.mark.parametrize('bright,dtype',[(False,np.uint8),(True,np.uint8),(False,np.uint16)])
def test_guided_bowing_measurements_stay_inside_curved_band(bright,dtype):
    image=trench_image(bulb=12,noise=2,bright=bright)
    if dtype==np.uint16:image=image.astype(dtype)*200
    guides=bowed_guides()
    result=analyze_trench(image,TrenchOptions((30,20,140,225),step_px=5,bright_trench=bright,smooth_px=25,wall_guides=guides))
    for side,edge in [('left',result.left),('right',result.right)]:
        for xx,yy in zip(edge,result.y):
            bands=row_bands(guides,yy,side)
            assert not bands or in_bands(xx,bands)
    assert max(result.sample_right-result.sample_left)==pytest.approx(84,abs=2)
    assert all(d['coverage']>.95 for d in result.guide_diagnostics)
    assert result.depth_px==pytest.approx(201,abs=2)


def test_manual_curves_choose_smaller_neighbor_instead_of_larger_trench():
    image=np.full((280,360),205,np.uint8);image[:20]=35
    image[20:261,50:165]=35;image[20:238,255:305]=35
    options=dict(roi=(20,20,320,250),smooth_px=3)
    auto=analyze_trench(image,TrenchOptions(**options))
    guides=(curve('left',[[257,20],[256,130],[257,234]],7),
            curve('right',[[303,20],[304,130],[303,234]],7))
    manual=analyze_trench(image,TrenchOptions(**options,wall_guides=guides))
    assert np.median(auto.left)<100
    assert np.median(manual.left)==pytest.approx(254.5,abs=1)
    assert np.median(manual.right)==pytest.approx(304.5,abs=1)
    assert manual.depth_px==pytest.approx(217,abs=2)


def test_empty_band_does_not_fall_back_to_unrelated_boundary():
    guides=(curve('left',[[40,20],[40,100],[40,240]],2),
            curve('right',[[160,20],[160,100],[160,240]],2))
    with pytest.raises(ValueError,match='가이드'):
        analyze_trench(trench_image(),TrenchOptions((30,20,140,225),wall_guides=guides))
    assert guided_edges(np.full(100,170.),[(35,45)],-1)==[]


def test_partial_single_side_guide_and_legacy_line_load():
    guide=dict(side='left',start=[69,50],end=[69,180],half_width_px=5)
    result=analyze_trench(trench_image(),TrenchOptions((30,20,140,225),wall_guides=(guide,)))
    assert result.y[0]<=22 and result.y[-1]>210
    assert np.median(result.right-result.left)==pytest.approx(60,abs=1)
    assert normalize_guides([guide])[0]['points']==[[69,50],[69,180]]


def test_oblique_curve_end_resumes_automatic_tracking_without_a_gap():
    guides=(curve('left',[[68,25],[67,110],[70,180]],5),
            curve('right',[[132,25],[133,110],[129,180]],5))
    result=analyze_trench(trench_image(),TrenchOptions((30,20,140,225),wall_guides=guides))
    assert result.y[-1]>215
    assert result.depth_px==pytest.approx(201,abs=2)
    assert all(item['coverage']>.98 for item in result.guide_diagnostics)


def test_guide_coordinates_and_coverage_exported(tmp_path):
    guides=bowed_guides()
    result=analyze_trench(trench_image(bulb=12),TrenchOptions((30,20,140,225),wall_guides=guides))
    payload=json.loads(json.dumps(result.to_dict(.5),allow_nan=False))
    assert payload['options']['wall_guides']==list(guides)
    assert payload['guide_diagnostics'][0]['detected_rows']>180
    target=tmp_path/'guided.csv';export_trench_csv(target,result,.5)
    assert 'wall_guides' in target.read_text(encoding='utf-8-sig')
    assert 'half_width_px' in target.read_text(encoding='utf-8-sig')


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


def draw_curve(panel,app,side,points):
    panel.guide_button.setChecked(True);app.processEvents()
    for p in points[:-1]:
        QTest.mouseClick(panel.view.viewport(),Qt.MouseButton.LeftButton,pos=panel.view.mapFromScene(*p))
    QTest.mouseDClick(panel.view.viewport(),Qt.MouseButton.LeftButton,pos=panel.view.mapFromScene(*points[-1]))
    app.processEvents()


def wheel(panel,point,modifiers=Qt.KeyboardModifier.NoModifier):
    local=panel.view.mapFromScene(*point);global_pos=panel.view.viewport().mapToGlobal(local)
    event=QWheelEvent(QPointF(local),QPointF(global_pos),QPoint(),QPoint(0,120),
                      Qt.MouseButton.NoButton,modifiers,Qt.ScrollPhase.NoScrollPhase,False)
    QApplication.sendEvent(panel.view.viewport(),event)


def test_real_click_curve_double_click_wheel_edit_recognition_and_restore(app):
    p=TrenchPanel();p.resize(1250,880);p.show()
    p.set_image(trench_image(bulb=12),'curved.png',.5)
    p.set_roi(QRectF(30,20,140,225));app.processEvents();p.view.fit_image()
    for guide in bowed_guides():draw_curve(p,app,guide['side'],guide['points'])
    assert len(p.guide_editor.guides)==2
    assert len(p.guide_editor.guides[-1]['points'])==8  # last double-click adds only once
    assert not p.guide_editor.drawing and not p.guide_button.isChecked()
    index=p.guide_editor.selected;before=p.guide_editor.guides[index]['half_width_px']
    transform=p.view.transform().m11();wheel(p,bowed_guides()[1]['points'][3])
    assert p.guide_editor.guides[index]['half_width_px']==before+1
    assert p.view.transform().m11()==transform
    wheel(p,bowed_guides()[1]['points'][3],Qt.KeyboardModifier.ControlModifier)
    assert p.view.transform().m11()>transform
    assert p.guide_editor.guides[index]['half_width_px']==before+1
    p.guide_analyze.click();finish_analysis(p,app)
    assert len(p.result.guide_diagnostics)==2
    # Drag one existing knot; editing invalidates all old/exportable results.
    old=p.guide_editor.guides[index]['points'][3]
    a=p.view.mapFromScene(*old);b=p.view.mapFromScene(old[0]+2,old[1])
    QTest.mousePress(p.view.viewport(),Qt.MouseButton.LeftButton,pos=a)
    QTest.mouseMove(p.view.viewport(),b)
    QTest.mouseRelease(p.view.viewport(),Qt.MouseButton.LeftButton,pos=b)
    assert p.result is None and not p.export_button.isEnabled()
    assert p.guide_editor.guides[index]['points'][3][0]>old[0]+1
    state=json.loads(json.dumps(p.state()))
    p.set_image(trench_image(),'other.png')
    assert not p.guide_editor.guides
    p.set_image(trench_image(bulb=12),'curved.png',.5,state)
    assert p.state()['wall_guides']==state['wall_guides']
    p.guide_editor.select(0);p.guide_delete.click()
    assert len(p.guide_editor.guides)==1
    p.guide_clear.click();assert not p.guide_editor.guides
    p.close()


def test_escape_cancels_draft_without_changing_completed_curves(app):
    p=TrenchPanel();p.resize(1000,800);p.show();p.set_image(trench_image(),'one')
    p.set_roi(QRectF(30,20,140,225));app.processEvents();p.view.fit_image()
    p.guide_button.setChecked(True)
    QTest.mouseClick(p.view.viewport(),Qt.MouseButton.LeftButton,pos=p.view.mapFromScene(70,30))
    assert p.guide_editor.draft
    QTest.keyClick(p.view,Qt.Key.Key_Escape)
    assert not p.guide_editor.draft and not p.guide_editor.guides
    assert not p.guide_button.isChecked()
    p.close()


def test_project_roundtrip_and_rotation_clear_curve_guides(app,tmp_path,monkeypatch):
    import cv2
    from angle_cal.app import MainWindow
    image_path=tmp_path/'sample.png'
    cv2.imencode('.png',trench_image(bulb=12))[1].tofile(str(image_path))
    w=MainWindow();monkeypatch.setattr(w,'_show_save_notification',lambda *_:None)
    w._load_image_path(str(image_path));w.addon_actions['trench_analyzer'].setChecked(True)
    p=w.trench_panel;p.set_roi(QRectF(30,20,140,225))
    p.guide_editor.restore(bowed_guides());p._guides_changed()
    target=tmp_path/'curves.anglecal.json';w._save_project_to_path(str(target))
    saved=json.loads(target.read_text(encoding='utf-8'))
    assert saved['image_states'][str(image_path)]['trench']['wall_guides']==list(bowed_guides())
    state=w._normalize_image_state(saved['image_states'][str(image_path)])
    assert state['trench']['wall_guides']==list(bowed_guides())
    w.rotate_current_image(90)
    assert w.trench_state['wall_guides']==[]
    assert not p.guide_editor.guides
    w.close()


def test_ribbon_starts_side_independent_contour_from_other_analysis_tab(app, tmp_path):
    import cv2
    from PySide6.QtWidgets import QPushButton
    from angle_cal.app import MainWindow
    image_path=tmp_path/'sample.png'
    cv2.imencode('.png',trench_image())[1].tofile(str(image_path))
    w=MainWindow();w.resize(1280,850);w.show()
    w.addon_actions['trench_analyzer'].setChecked(True);app.processEvents()
    panel=w.trench_panel
    button=w.findChild(QPushButton,'trench_contour_guide')
    assert button.isVisible()
    button.click()
    assert not panel.guide_editor.drawing
    w._load_image_path(str(image_path));app.processEvents()
    panel.image_tabs.setCurrentIndex(2)
    button.click();app.processEvents()
    assert panel.image_tabs.currentIndex()==0
    assert panel.guide_editor.drawing and panel.guide_editor.side=='contour'
    QTest.keyClick(panel.view,Qt.Key.Key_Escape)
    assert not panel.guide_editor.drawing
    w.close()
