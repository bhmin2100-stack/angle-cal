from dataclasses import replace
import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication
from angle_cal.trench_panel import TrenchPanel
from angle_cal.trench_analysis import trench_gfe_tsv
from test_trench_profile_exchange import example_result, split_bottom_result
from test_trench_field_metrics import analyze


@pytest.fixture(scope='module')
def result():return analyze()


def test_default_lower_table_matches_gapseam_coordinates_and_actual_keyboard_copy(result):
    app=QApplication.instance() or QApplication([])
    p=TrenchPanel();p.resize(1200,850);p.show()
    try:
        expected=[[-1500,0],[-250,0],[-100,-900],[-300,-1600],[-125,-4000],
                  [125,-4000],[250,0],[1500,0]]
        p.scale_spin.setValue(1)
        p.result=example_result(result,expected,reverse=True)
        p._render_profile_coordinates();app.processEvents()
        table=p.profile_table
        assert p.tables.currentWidget() is p.profile_page
        assert not p.tables.tabBar().isHidden()
        assert p.tables.widget(1) is p.cd_table
        assert [table.horizontalHeaderItem(i).text() for i in (0,1)]==['X (Å)','Y (Å)']
        assert table.rowCount()==len(expected) and table.columnCount()==2
        assert [[float(table.item(row,col).text()) for col in (0,1)] for row in range(table.rowCount())]==expected
        assert not table.isSortingEnabled()
        assert not table.item(0,0).flags() & Qt.ItemFlag.ItemIsEditable
    finally:p.close();app.processEvents()


def test_profile_copy_preserves_exact_export_values_and_full_order(result):
    app=QApplication.instance() or QApplication([])
    p=TrenchPanel();p.resize(1200,850);p.show()
    try:
        p.scale_spin.setValue(.5);p.result=result;p._render_profile_coordinates();app.processEvents()
        table=p.profile_table;table.setFocus();table.selectAll()
        QTest.keyClick(table,Qt.Key.Key_C,Qt.KeyboardModifier.ControlModifier)
        app.processEvents()
        expected='\n'.join(trench_gfe_tsv(result,.5).splitlines()[1:])+'\n'
        assert app.clipboard().text()==expected
        assert 'Field' in p.profile_status.text() and '깊이 음수' in p.profile_status.text()
        p.metric_units['depth'].setCurrentIndex(p.metric_units['depth'].findData('angstrom'))
        assert table.item(0,0).text()==expected.splitlines()[0].split('\t')[0]
        p._clear_result();assert table.rowCount()==0
    finally:p.close();app.processEvents()


@pytest.mark.parametrize('missing',['scale','field'])
def test_unconfirmed_profile_does_not_present_pixels_as_angstroms(result,missing):
    app=QApplication.instance() or QApplication([])
    p=TrenchPanel()
    try:
        p.scale_spin.setValue(1);p.result=result;p._render_profile_coordinates()
        assert p.profile_table.rowCount()>0
        if missing=='scale':p.scale_spin.setValue(0);p.result=result
        else:p.result=replace(result,field_angles=[])
        p._render_profile_coordinates()
        assert p.profile_table.rowCount()==0
        assert '미확정' in p.profile_status.text()
    finally:p.close();app.processEvents()


def test_bottom_unconfirmed_still_shows_and_copies_coordinates(result):
    app=QApplication.instance() or QApplication([])
    p=TrenchPanel()
    try:
        p.scale_spin.setValue(.5)
        p.result=replace(result,depth_px=None)
        p._render_profile_coordinates()
        assert p.profile_table.rowCount()>0
        assert '바닥/전체 깊이 미확정' in p.profile_status.text()
        p.copy_gfe_coordinates()
        assert app.clipboard().text()==trench_gfe_tsv(result,.5)
        assert '복사 완료' in p.hint.text() and '바닥 미확정' in p.hint.text()
    finally:p.close();app.processEvents()


def test_bottom_connection_is_visible_in_table_and_export_copy(result):
    app=QApplication.instance() or QApplication([])
    p=TrenchPanel()
    try:
        p.scale_spin.setValue(1);p.result,expected=split_bottom_result(result,separate=True)
        p._render_profile_coordinates()
        assert p.profile_table.rowCount()==len(expected)
        assert '하단 끝점 자동 연결' in p.profile_status.text()
        assert p.graph.export_bottom_connection['points']==expected[4:6]
        p.copy_gfe_coordinates()
        assert app.clipboard().text()==trench_gfe_tsv(p.result,1)
        assert '하단 끝점 자동 연결' in p.hint.text()
    finally:p.close();app.processEvents()
