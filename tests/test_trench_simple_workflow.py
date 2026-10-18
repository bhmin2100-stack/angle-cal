import json
import time
import cv2
import numpy as np
import pytest
from PySide6.QtCore import Qt,QRectF
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication,QInputDialog,QFileDialog
from angle_cal.app import MainWindow,StructureTemplate,structure_template_to_dict,structure_template_from_dict
from angle_cal.trench_panel import TrenchPanel
from angle_cal.trench_workflow import recognize_from_guides
from angle_cal.trench_analysis import TrenchOptions
from test_trench_analysis import trench_image


def guides():
    return [dict(side='contour',points=[[69,27],[67,80],[57,110],[67,160],[69,210],[78,222],
                         [122,222],[130,210],[133,160],[143,110],[133,80],[130,27]],half_width_px=12)]


@pytest.fixture
def app():return QApplication.instance() or QApplication([])


def finish(panel,app):
    limit=time.monotonic()+20
    while panel.worker is not None and panel.worker.isRunning() and time.monotonic()<limit:
        app.processEvents();time.sleep(.01)
    panel.wait_for_worker();app.processEvents()
    assert panel.result is not None,panel.summary_label.text()


@pytest.mark.parametrize('inverted',[False,True])
def test_no_roi_auto_entrance_polarity_and_one_nm_step(inverted):
    im=trench_image(bulb=12)
    if inverted:im=255-im
    r=recognize_from_guides(im,TrenchOptions((0,0,200,260),step_px=2,wall_guides=tuple(guides()),boundary_mode='max_gradient'))
    assert r.options.roi[1]==pytest.approx(20,abs=1)
    assert r.options.bright_trench==inverted
    assert r.depth_px==pytest.approx(201,abs=2)
    assert r.summary()['max_cd_px']==pytest.approx(84,abs=2)
    assert np.allclose(np.diff(r.sample_y),2)


def test_default_workspace_has_only_simple_controls(app):
    p=TrenchPanel();p.show();p.set_image(trench_image(),'simple',.5);app.processEvents()
    assert p.automatic_area and p.roi is None and not p.view.select_region
    assert p.step_spin.value()==1 and p.options().step_px==2
    assert p.image_tabs.count()==1
    for widget in (p.roi_button,p.threshold_spin,p.bright_check,p.bow_combo,p.negative_spin,
                   p.corner_spin,p.field_spin,p.angle_start,p.angle_end,p.graph_combo,*p.corner_editors,*p.surface_editors):
        assert not widget.isVisible()
    p.start_analysis();app.processEvents();assert '곡선' in p.summary_label.text()
    p.close()


def test_thumbnail_curve_recognize_scale_and_project_shared(app,tmp_path,monkeypatch):
    paths=[tmp_path/'one.png',tmp_path/'two.png']
    for path in paths:cv2.imencode('.png',trench_image(bulb=12))[1].tofile(str(path))
    w=MainWindow();w.resize(1450,950);w.show()
    w.browser_image_paths=list(map(str,paths));w._populate_thumbnails()
    w._load_image_path(str(paths[0]));w.nm_per_px=.5
    w.addon_actions['trench_analyzer'].setChecked(True);app.processEvents();p=w.trench_panel
    assert w.trench_quick_tools.isVisible() and not w.normal_quick_tools.isVisible()
    assert p.scale_spin.value()==.5
    p.rim_reference.setChecked(False)  # This fixture is a step with no bright crest.
    p.view.fit_image();p.guide_width.setValue(12);p.start_contour_guide()
    for point in guides()[0]['points'][:-1]:QTest.mouseClick(p.view.viewport(),Qt.MouseButton.LeftButton,pos=p.view.mapFromScene(*point))
    QTest.mouseDClick(p.view.viewport(),Qt.MouseButton.LeftButton,pos=p.view.mapFromScene(*guides()[0]['points'][-1]))
    assert p.roi is None
    p.guide_analyze.click();finish(p,app)
    assert np.allclose(np.diff(p.result.sample_y),2)
    assert not p.graph.show_entrance
    assert all(not text in p.summary_label.text() for text in ('입구 각도','입구 R','경계값','Nega'))
    # The scale tool stays in Trench and updates main records/calibration.
    monkeypatch.setattr(QInputDialog,'getDouble',lambda *a,**k:(25.,True))
    p.tool_buttons['scale'].click();assert w.workspace_stack.currentWidget() is p
    a=p.view.mapFromScene(30,245);b=p.view.mapFromScene(80,245)
    QTest.mousePress(p.view.viewport(),Qt.MouseButton.LeftButton,pos=a);QTest.mouseMove(p.view.viewport(),b);QTest.mouseRelease(p.view.viewport(),Qt.MouseButton.LeftButton,pos=b)
    assert p.scale_bar is not None
    assert w.nm_per_px==pytest.approx(.5,abs=.02)
    assert any(r.kind=='scale' for r in w.records.values())
    saved=p.state()['wall_guides'];p.step_spin.setValue(1)
    w.load_browser_image(str(paths[1]));app.processEvents()
    assert not p.guide_editor.guides and p.roi is None and p.scale_bar is None
    w.load_browser_image(str(paths[0]));app.processEvents()
    assert p.state()['wall_guides']==saved and p.scale_bar
    assert p.scale_spin.value()==w.nm_per_px
    monkeypatch.setattr(w,'_show_save_notification',lambda *a:None)
    target=tmp_path/'project.anglecal.json';w._save_project_to_path(str(target))
    payload=json.loads(target.read_text(encoding='utf-8'))
    assert payload['image_states'][str(paths[0])]['trench']['step_nm']==1
    w.ribbon_tabs.setCurrentIndex(0);app.processEvents()
    assert w.normal_quick_tools.isVisible() and not w.trench_quick_tools.isVisible()
    w.close()


def test_load_existing_structure_curve_resize_and_exports(app,tmp_path,monkeypatch):
    p=TrenchPanel();p.resize(1200,850);p.show();p.set_image(trench_image(bulb=12),'curve',.5);app.processEvents();p.view.fit_image()
    p.rim_reference.setChecked(False)  # Keep testing the supported legacy step-edge mode.
    # New curves round-trip through the existing structure library too.
    template=StructureTemplate('곡선',[],wall_guides=guides())
    payload=structure_template_to_dict(structure_template_from_dict(structure_template_to_dict(template)))
    p.load_structure_payload(payload);assert p.guide_editor.guides==guides()
    p.set_tool('resize');app.processEvents();old=np.asarray(p.guide_editor.guides[0]['points'])
    corner=p.guide_editor.resize_corners[2]
    a=p.view.mapFromScene(*corner);b=p.view.mapFromScene(corner[0]+3,corner[1])
    QTest.mousePress(p.view.viewport(),Qt.MouseButton.LeftButton,pos=a);QTest.mouseMove(p.view.viewport(),b);QTest.mouseRelease(p.view.viewport(),Qt.MouseButton.LeftButton,pos=b)
    assert np.ptp(np.asarray(p.guide_editor.guides[0]['points'])[:,0])>np.ptp(old[:,0])
    p.load_structure_payload(payload);p.guide_analyze.click();finish(p,app)
    y=p.result.sample_y.copy()
    p.unit_combo.setCurrentIndex(p.unit_combo.findData('angstrom'));p.start_analysis();finish(p,app)
    assert np.array_equal(y,p.result.sample_y) # Display unit cannot alter 1nm spacing.
    for ext in ('csv','json','png'):p.save_result(str(tmp_path/f'simple.{ext}'))
    QApplication.clipboard().clear();p.copy_gfe_coordinates()
    assert not QApplication.clipboard().text() and 'Field' in p.hint.text()
    p.close()


def test_legacy_structure_edges_can_be_loaded_as_guides(app):
    p=TrenchPanel();p.set_image(trench_image(),'legacy')
    p.load_structure_payload(dict(records=[dict(kind='edge',start=[69,27],end=[69,210],search_radius_px=8)]))
    assert p.guide_editor.guides[0]['points']==[[69.,27.],[69.,210.]]
    p.close()


def test_existing_guide_search_settings_are_inherited(app,tmp_path):
    path=tmp_path/'settings.png';cv2.imencode('.png',trench_image())[1].tofile(str(path))
    w=MainWindow();w._load_image_path(str(path));w.search_radius_spin.setValue(9)
    w.boundary_snap_combo.setCurrentIndex(w.boundary_snap_combo.findData('brightest'))
    w.addon_actions['trench_analyzer'].setChecked(True)
    p=w.trench_panel
    assert p.guide_editor.radius==9
    assert p.options().boundary_mode=='rim_inset'
    p.rim_reference.setChecked(False)
    assert p.options().boundary_mode=='brightest'
    w.close()


def test_one_nm_deep_measurement_does_not_hit_old_twenty_thousand_limit():
    r=recognize_from_guides(trench_image(bulb=12),TrenchOptions((0,0,200,260),step_px=1/128,
                           wall_guides=tuple(guides()),boundary_mode='max_gradient'))
    assert len(r.sample_y)>20000
    assert np.allclose(np.diff(r.sample_y)*128,1)


def test_removed_legacy_settings_do_not_leak_into_simple_recognition(app):
    p=TrenchPanel();p.set_image(trench_image(),'old-project',.5,dict(threshold=200,bright=True,
        smooth=99,bow_reference='chord',corner=200,field_search=100,angle_start=5,angle_end=10))
    options=p.options()
    assert options.threshold==0 and options.smooth_px==7 and options.bow_reference=='entrance'
    assert options.corner_window_px==24 and options.field_search_px==32 and options.angle_span==(40,100)
    assert options.step_px==2
    p.close()


def test_saved_curve_structure_survives_library_and_project_reopen(app,tmp_path,monkeypatch):
    path=tmp_path/'image.png';cv2.imencode('.png',trench_image(bulb=12))[1].tofile(str(path))
    w=MainWindow();w._load_image_path(str(path));w.addon_actions['trench_analyzer'].setChecked(True)
    w.trench_panel.load_structure_payload(dict(wall_guides=guides()))
    structure=tmp_path/'curve.anglecal.structure.json'
    monkeypatch.setattr(QFileDialog,'getSaveFileName',lambda *a,**k:(str(structure),''))
    w.trench_panel.save_structure_dialog()
    assert w.structure_templates[0].wall_guides==guides()
    monkeypatch.setattr(w,'_show_save_notification',lambda *a:None)
    project=tmp_path/'library.anglecal.json';w._save_project_to_path(str(project))
    reader=MainWindow();monkeypatch.setattr(QFileDialog,'getOpenFileName',lambda *a,**k:(str(project),''))
    reader.open_project()
    assert reader.structure_templates[0].wall_guides==guides()
    reader.close();w.close()
