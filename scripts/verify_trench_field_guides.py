"""Qt click-through verification using newly generated known-shape samples."""
from pathlib import Path
import json
import time

import cv2
import numpy as np
from PySide6.QtCore import QRectF,Qt
from PySide6.QtGui import QFont,QFontDatabase
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from angle_cal.app import MainWindow


def sample(curved=False):
    s=4;yy,xx=np.mgrid[:400*s,:420*s].astype(float)/s
    lr,rr=(30,30) if curved else (15,30)
    boundaries=[]
    for radius,wall,sign in ((lr,160,-1),(rr,260,1)):
        d=np.clip(yy-45,0,radius)
        boundaries.append(wall+sign*(radius-np.sqrt(np.maximum(0,radius**2-(radius-d)**2))))
    field=np.full_like(xx,45)
    if curved:field+=.0025*np.maximum(130-xx,0)**2+.0025*np.maximum(xx-290,0)**2
    void=(yy<field)|((xx>boundaries[0])&(xx<boundaries[1])&(yy<350))
    image=cv2.resize(np.where(void,35.,205.).astype(np.float32),(420,400),interpolation=cv2.INTER_AREA).astype(np.uint8)
    points=[[65,45],[160-lr-15,45],[160-lr,45],[160-lr*.3,45+lr*.3],[160,45+lr],
            [160,310],[160,345],[190,350],[230,350],[260,345],[260,310],
            [260,45+rr],[260+rr*.3,45+rr*.3],[260+rr,45],[260+rr+15,45],[355,45]]
    if curved:points[:3]=[[65,55.5],[95,48],[130,45]];points[-3:]=[[290,45],[325,48],[355,55.5]]
    return image,points


def main():
    out=Path(__file__).resolve().parents[1]/'build'/'field_guide_verification';out.mkdir(parents=True,exist_ok=True)
    print('Creating Qt workspace',flush=True)
    app=QApplication.instance() or QApplication([])
    QFontDatabase.addApplicationFont('C:/Windows/Fonts/malgun.ttf');app.setFont(QFont('Malgun Gothic',9))
    window=MainWindow();window.resize(1600,1040);window.show();summary={}
    for curved in (False,True):
        name='curved_field' if curved else 'asymmetric_field'
        image,points=sample(curved);path=out/(name+'.png');cv2.imwrite(str(path),image)
        print('Loading '+name,flush=True)
        window._load_image_path(str(path));window.nm_per_px=.5
        window.addon_actions['trench_analyzer'].setChecked(True)
        panel=window.trench_panel;panel.guide_editor.clear();panel.set_roi(QRectF(60,45,300,335))
        panel.scale_spin.setValue(.5);panel.unit_combo.setCurrentIndex(panel.unit_combo.findData('px'))
        panel.angle_start.setValue(100);panel.angle_end.setValue(160)
        panel.view.fitInView(QRectF(0,0,420,400),Qt.AspectRatioMode.KeepAspectRatio)
        app.processEvents();panel.guide_button.setChecked(True);panel.guide_width.setValue(18)
        for point in points[:-1]:
            QTest.mouseClick(panel.view.viewport(),Qt.MouseButton.LeftButton,pos=panel.view.mapFromScene(*point))
        QTest.mouseDClick(panel.view.viewport(),Qt.MouseButton.LeftButton,pos=panel.view.mapFromScene(*points[-1]))
        assert not panel.guide_editor.drawing and len(panel.guide_editor.guides)==1
        panel.guide_analyze.click();deadline=time.monotonic()+25
        while panel.worker and panel.worker.isRunning() and time.monotonic()<deadline:
            app.processEvents();time.sleep(.01)
        panel.wait_for_worker();app.processEvents();r=panel.result
        assert r is not None,panel.summary_label.text()
        assert all(f.get('source')=='guide' and f['status']=='valid' for f in r.field_angles)
        assert all(c['status']=='valid' for c in r.corners),r.corners
        print('Measured '+name,flush=True)
        angles=[f['angle_deg'] for f in r.field_angles]
        for unit in ('nm','angstrom','px'):
            print('Unit '+unit,flush=True)
            panel.unit_combo.setCurrentIndex(panel.unit_combo.findData(unit));app.processEvents()
            # Unit also defines the CD sampling interval, so existing UI
            # invalidates results and requires the normal analysis action.
            panel.guide_analyze.click();deadline=time.monotonic()+25
            while panel.worker and panel.worker.isRunning() and time.monotonic()<deadline:
                app.processEvents();time.sleep(.01)
            panel.wait_for_worker();app.processEvents()
            assert panel.result is not None,panel.summary_label.text()
            assert [f['angle_deg'] for f in panel.result.field_angles]==angles
            assert ('Å' if unit=='angstrom' else unit) in panel.surface_editors[0].label.text()
        for ext in ('csv','json','png'):panel.save_result(str(out/(name+'_result.'+ext)))
        panel.copy_gfe_coordinates();table=app.clipboard().text()
        assert table.startswith('X\tY\n') and len(table.splitlines())>500,panel.hint.text()
        (out/(name+'_gfe.tsv')).write_text(table,encoding='utf-8')
        project=out/(name+'.anglecal.json');window._save_project_to_path(str(project))
        payload=json.loads(project.read_text(encoding='utf-8'))
        assert 'local_angle_deg' in json.dumps(payload)
        panel.image_tabs.setCurrentIndex(2);app.processEvents();window.grab().save(str(out/(name+'_workspace.png')))
        summary[name]=dict(radii=[c['radius_px'] for c in r.corners],angles=angles,
                           field_sources=[f['source'] for f in r.field_angles],
                           straight_references=[f['straight_angle_deg'] for f in r.field_angles],
                           trace_coverage=r.guide_traces[0]['trace_coverage'],gfe_rows=len(table.splitlines())-1)
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps(summary))
    window.close()


if __name__=='__main__':main()
