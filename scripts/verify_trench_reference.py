from pathlib import Path
import json
import time
import cv2
import numpy as np
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QFont, QFontDatabase
from angle_cal.trench_panel import TrenchPanel
from angle_cal.trench_analysis import trench_profile_exchange

root=Path(__file__).resolve().parents[1];out=root/'build/region-continuity-verification/reference';out.mkdir(parents=True,exist_ok=True)
app=QApplication.instance() or QApplication([])
QFontDatabase.addApplicationFont('C:/Windows/Fonts/malgun.ttf');app.setFont(QFont('Malgun Gothic',9))
image=cv2.imdecode(np.fromfile(root/'samples/trench-reference.png',dtype=np.uint8),cv2.IMREAD_UNCHANGED)
results=[]
for side in ('left','right'):
    p=TrenchPanel();p.resize(1400,950);p.show()
    try:
        p.set_image(image,str(root/'samples/trench-reference.png'),10000/313)
        structure=json.loads((root/f'samples/trench-reference-{side}.anglecal.structure.json').read_text(encoding='utf-8'))
        p.load_structure_payload(structure)
        p.smoothing_controls.sliders['robust'].setValue(45)
        p.start_analysis()
        end=time.monotonic()+90
        while p.worker is not None and p.worker.isRunning() and time.monotonic()<end:
            app.processEvents();time.sleep(.01)
        p.wait_for_worker();app.processEvents()
        assert p.result is not None,p.summary_label.text()
        r=p.result
        assert len(r.guide_traces)==1 and r.guide_traces[0]['continuity_complete']
        assert r.field_measurements['tcd_status']=='valid',r.field_measurements
        assert r.field_measurements['depth_status']=='valid',r.field_measurements
        for suffix in ('json','csv','png'):p.save_result(str(out/f'{side}.{suffix}'))
        p.copy_gfe_coordinates();text=app.clipboard().text()
        assert text.startswith('X\tY\n'),p.hint.text()
        (out/f'{side}.tsv').write_text(text,encoding='utf-8')
        payload=trench_profile_exchange(r,p.scale_spin.value())
        (out/f'{side}-trench-profile.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
        for _ in range(3):app.processEvents()
        assert p.export_button.geometry().top()>=p.summary_label.geometry().bottom()
        p.graph.grab();p.grab().save(str(out/f'{side}-workspace.png'))
        results.append(dict(side=side,summary=r.summary(),field=r.field_measurements,
                            segments=len(r.guide_traces[0]['segments']),detected_points=r.guide_traces[0]['detected_points'],
                            interpolated_points=r.guide_traces[0]['interpolated_points'],gfe_points=len(payload['points'])))
        print(side,json.dumps(dict(summary=r.summary(),segments=len(r.guide_traces[0]['segments']),gfe_points=len(payload['points']))),flush=True)
    finally:p.wait_for_worker();p.close();app.processEvents()
(out/'verification.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
