"""Exercise PPT-style curve clicks, bounded recognition and saved artifacts.

Run with QT_QPA_PLATFORM=offscreen. The SEM fixture is a visual simulation;
known-shape accuracy and interactions are separately covered by pytest.
"""
from pathlib import Path
import json
import shutil
import time

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from angle_cal.app import MainWindow
from angle_cal.trench_guides import normalize_guides, row_bands, in_bands


def main():
    root=Path(__file__).resolve().parents[1]
    out=root/'build'/'curve_guide_verification';out.mkdir(parents=True,exist_ok=True)
    source=out/'bowing_sample.png'
    shutil.copyfile(root/'output/imagegen/vsem-bowing-20261005/bowing_sample.png',source)
    app=QApplication.instance() or QApplication([])
    QFontDatabase.addApplicationFont('C:/Windows/Fonts/malgun.ttf');app.setFont(QFont('Malgun Gothic',9))
    window=MainWindow();window.resize(1600,1040);window.show()
    window._load_image_path(str(source));window.nm_per_px=10000/312
    window.addon_actions['trench_analyzer'].setChecked(True)
    panel=window.trench_panel
    panel.guide_editor.clear()
    panel.set_roi(QRectF(780,76,250,860));panel.scale_spin.setValue(10000/312)
    panel.smooth_spin.setValue(9);panel.negative_spin.setValue(2)
    window._save_project_to_path(str(out/'before_guides.anglecal.json'))
    app.processEvents();QTest.qWait(50)
    panel.view.fitInView(QRectF(740,35,330,920),Qt.AspectRatioMode.KeepAspectRatio)
    app.processEvents()
    controls=[('left',[[889,90],[879,170],[850,360],[829,495],[855,660],[882,835],[884,887]]),
              ('right',[[940,90],[940,170],[970,360],[983,490],[951,680],[931,835],[930,890]])]
    for side,points in controls:
        panel.guide_side.setCurrentIndex(0 if side=='left' else 1)
        panel.guide_button.setChecked(True);panel.guide_width.setValue(12)
        for point in points[:-1]:
            QTest.mouseClick(panel.view.viewport(),Qt.MouseButton.LeftButton,pos=panel.view.mapFromScene(*point))
        QTest.mouseDClick(panel.view.viewport(),Qt.MouseButton.LeftButton,pos=panel.view.mapFromScene(*points[-1]))
        app.processEvents()
    assert len(panel.guide_editor.guides)==2,(len(panel.guide_editor.guides),panel.guide_editor.draft,panel.guide_hint.text(),panel.view.image_rect)
    assert not panel.guide_editor.drawing
    panel.guide_analyze.click()
    deadline=time.monotonic()+20
    while panel.worker and panel.worker.isRunning() and time.monotonic()<deadline:
        app.processEvents();QTest.qWait(10)
    panel.wait_for_worker();app.processEvents()
    assert panel.result is not None,panel.summary_label.text()
    assert panel.result.depth_px is not None and panel.result.depth_px>800,(panel.result.y[0],panel.result.y[-1],panel.result.warnings)
    assert all(d['coverage']>.95 for d in panel.result.guide_diagnostics)
    guides=normalize_guides(panel.guide_editor.guides)
    for side,edge in [('left',panel.result.left),('right',panel.result.right)]:
        for xx,yy in zip(edge,panel.result.y):
            bands=row_bands(guides,yy,side)
            assert not bands or in_bands(xx,bands)
    for extension in ('csv','json','png'):panel.save_result(str(out/f'guided.{extension}'))
    panel.copy_gfe_coordinates();(out/'guided_gfe.tsv').write_text(app.clipboard().text(),encoding='utf-8')
    assert len(app.clipboard().text().splitlines())>1000,panel.hint.text()
    window._save_project_to_path(str(out/'curve_guides.anglecal.json'))
    window.grab().save(str(out/'workspace.png'))
    summary=dict(guides=len(guides),control_points=[len(g['points']) for g in guides],
                 coverage=[d['coverage'] for d in panel.result.guide_diagnostics],
                 depth_px=panel.result.depth_px,max_cd_px=panel.result.summary()['max_cd_px'],
                 clipboard_rows=len(app.clipboard().text().splitlines())-1)
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary))
    window.close()


if __name__=='__main__':main()
