import json
import numpy as np
import pytest
from angle_cal.trench_units import format_metric_length


@pytest.mark.parametrize('pixels,scale,unit,expected',[
    (26000,1,'auto','26 µm'),(400,1,'auto','400 nm'),(.3,1,'auto','3 Å'),
    (1000,1,'auto','1 µm'),(1,1,'auto','1 nm'),(0,1,'auto','0 nm'),
    (2000,.5,'um','1 µm'),(2000,.5,'nm','1000 nm'),(2000,.5,'angstrom','10000 Å'),
    (20,0,'um','20 px'),(None,1,'auto','미확정')])
def test_metric_display_unit_conversions(pixels,scale,unit,expected):
    assert format_metric_length(pixels,scale,unit)==expected


def test_independent_units_redraw_without_reanalysis_and_keep_gfe_angstroms(tmp_path):
    from PySide6.QtWidgets import QApplication
    from angle_cal.trench_panel import TrenchPanel
    from angle_cal.trench_analysis import trench_gfe_tsv
    from test_trench_corner import rounded_trench
    from test_trench_guided_surface import surface_guide
    from test_trench_simple_workflow import finish
    app=QApplication.instance() or QApplication([])
    p=TrenchPanel();p.resize(1300,950);p.show()
    try:
        image=rounded_trench(15,30)
        p.set_image(image,'unit-test',10)
        p.rim_reference.setChecked(False)
        p.load_structure_payload(dict(wall_guides=[surface_guide(15,30)]))
        p.start_analysis();finish(p,app)
        result=p.result;revision=p.revision
        coordinates=trench_gfe_tsv(result,10)
        assert 'µm' in p.summary_label.text().splitlines()[0]
        assert 'µm' in p.summary_label.text().splitlines()[1]
        for key,choice in (('depth','nm'),('tcd','angstrom')):
            p.metric_units[key].setCurrentIndex(p.metric_units[key].findData(choice))
        assert p.result is result and p.revision==revision and p.worker is None
        assert 'nm' in p.summary_label.text().splitlines()[0]
        assert 'Å' in p.summary_label.text().splitlines()[1] and 'Å' in p.graph.tcd_text
        p.copy_gfe_coordinates();assert app.clipboard().text()==coordinates
        p.save_result(str(tmp_path/'units.json'));p.save_result(str(tmp_path/'units.csv'))
        data=json.loads((tmp_path/'units.json').read_text(encoding='utf-8'))
        units=data['field_measurement_units']
        assert units['um']['depth_um']==pytest.approx(units['nm']['depth_nm']/1000)
        assert units['angstrom']['tcd_angstrom']==pytest.approx(units['nm']['tcd_nm']*10)
        csv=(tmp_path/'units.csv').read_text(encoding='utf-8-sig')
        assert 'field_depth_um' in csv and 'tcd_angstrom' in csv
        state=p.state();p.set_image(image,'reloaded',10,state)
        assert p.metric_units['depth'].currentData()=='nm' and p.metric_units['tcd'].currentData()=='angstrom'
        p.unit_combo.setCurrentIndex(p.unit_combo.findData('um'))
        assert p._factor()==(.01,'µm')
        assert p.options().step_px==.1  # Still 1 nm spacing at 10 nm/px.
        p.set_image(image,'old-project',10,dict(wall_guides=[surface_guide(15,30)]))
        assert all(c.currentData()=='auto' for c in p.metric_units.values())
    finally:p.wait_for_worker();p.close();app.processEvents()
