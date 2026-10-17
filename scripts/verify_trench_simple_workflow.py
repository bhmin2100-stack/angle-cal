"""Actual thumbnail -> curve/structure -> recognize flow, without a user ROI."""
from pathlib import Path
import json
import time
import uuid
import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont,QFontDatabase
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication,QInputDialog
from angle_cal.app import MainWindow
from angle_cal import __version__


def sample(seed):
    yy,xx=np.mgrid[:480,:700]
    image=np.full(xx.shape,160.,np.float32)
    void=yy<40
    for center in (250,450):
        bulb=12*np.exp(-((yy-205)/70)**2)
        void|=(xx>center-40-bulb)&(xx<center+40+bulb)&(yy<430)
    image[void]=30
    image=cv2.GaussianBlur(image,(3,3),.6)
    image+=np.random.default_rng(seed).normal(0,1.5,image.shape)
    image[464:467,50:151]=245
    return np.clip(image,0,255).astype(np.uint8)


def finish(panel,app):
    limit=time.monotonic()+30
    while panel.worker is not None and panel.worker.isRunning() and time.monotonic()<limit:
        app.processEvents();time.sleep(.01)
    panel.wait_for_worker();app.processEvents()
    assert panel.result is not None,panel.summary_label.text()


def main():
    out=Path(__file__).resolve().parents[1]/'build'/'simple_workflow_verification';out.mkdir(parents=True,exist_ok=True)
    app=QApplication.instance() or QApplication([]);QFontDatabase.addApplicationFont('C:/Windows/Fonts/malgun.ttf');app.setFont(QFont('Malgun Gothic',9))
    inputs=out/'inputs'/uuid.uuid4().hex[:8];inputs.mkdir(parents=True)
    paths=[inputs/'sample_one.png',inputs/'sample_two.png']
    for i,path in enumerate(paths):cv2.imwrite(str(path),sample(i))
    w=MainWindow();w.resize(1450,1000);w.show()
    w.browser_image_paths=list(map(str,paths));w._populate_thumbnails()
    w.addon_actions['trench_analyzer'].setChecked(True);app.processEvents();p=w.trench_panel
    QTest.mouseClick(w.thumbnail_buttons[str(paths[0])],Qt.MouseButton.LeftButton);app.processEvents()
    assert p.image_path==str(paths[0]) and p.roi is None,(p.image_path,p.roi,w.image_path,int(QApplication.keyboardModifiers().value),w.size())
    p.view.fit_image();p.tool_buttons['scale'].click()
    original=QInputDialog.getDouble;QInputDialog.getDouble=lambda *a,**k:(50.,True)
    a=p.view.mapFromScene(50,465);b=p.view.mapFromScene(150,465)
    QTest.mousePress(p.view.viewport(),Qt.MouseButton.LeftButton,pos=a);QTest.mouseMove(p.view.viewport(),b);QTest.mouseRelease(p.view.viewport(),Qt.MouseButton.LeftButton,pos=b)
    QInputDialog.getDouble=original
    assert abs(w.nm_per_px-.5)<.01 and p.scale_bar
    points=[[150,40],[210,40],[210,65],[207,120],[198,205],[206,300],[210,400],
            [215,427],[240,430],[280,427],[290,400],[294,300],[302,205],[293,120],[290,65],[290,40],[350,40]]
    p.guide_width.setValue(14);p.start_contour_guide()
    for point in points[:-1]:QTest.mouseClick(p.view.viewport(),Qt.MouseButton.LeftButton,pos=p.view.mapFromScene(*point))
    QTest.mouseDClick(p.view.viewport(),Qt.MouseButton.LeftButton,pos=p.view.mapFromScene(*points[-1]))
    assert len(p.guide_editor.guides)==1 and p.roi is None
    p.guide_analyze.click();finish(p,app)
    assert p.result.depth_px is not None and p.result.depth_px>380
    assert len(p.result.sample_y)>180 and np.allclose(np.diff(p.result.sample_y)*w.nm_per_px,1)
    structure=dict(name='Trench 곡선',wall_guides=p.state()['wall_guides'])
    (out/'curve.anglecal.structure.json').write_text(json.dumps(structure,ensure_ascii=False,indent=2),encoding='utf-8')
    for ext in ('csv','json','png'):p.save_result(str(out/f'result.{ext}'))
    p.copy_gfe_coordinates();table=app.clipboard().text()
    assert table.startswith('X\tY\n'),p.hint.text()
    (out/'gfe.tsv').write_text(table,encoding='utf-8')
    w._save_project_to_path(str(out/'simple.anglecal.json'));app.processEvents()
    p.graph.grab()
    trace_y=[point[1] for trace in p.result.guide_traces for segment in trace['segments'] for point in segment]
    assert p.graph.plot_y_bounds[0]<=min(trace_y) and p.graph.plot_y_bounds[1]>=max(trace_y)
    w.grab().save(str(out/'workspace.png'))
    y=p.result.sample_y.copy();scale=p.scale_spin.value()
    QTest.mouseClick(w.thumbnail_buttons[str(paths[1])],Qt.MouseButton.LeftButton);app.processEvents()
    assert p.roi is None and not p.guide_editor.guides and p.scale_bar is None
    QTest.mouseClick(w.thumbnail_buttons[str(paths[0])],Qt.MouseButton.LeftButton);app.processEvents()
    assert p.scale_spin.value()==scale and p.state()['wall_guides']==structure['wall_guides']
    p.load_structure_payload(structure);p.guide_analyze.click();finish(p,app)
    assert np.array_equal(y,p.result.sample_y)
    summary=dict(version=__version__,graph_includes_field_and_floor=True,guide_points=len(points),roi_automatic=True,nm_per_px=scale,step_nm=p.step_spin.value(),
                 depth_nm=p.result.depth_px*scale,max_cd_nm=p.result.summary()['max_cd_px']*scale,
                 cd_rows=len(y),gfe_rows=len(table.splitlines())-1,thumbnail_roundtrip=True,structure_roundtrip=True)
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps(summary))
    w.close()


if __name__=='__main__':main()
