import csv
import json
import time

import cv2
import numpy as np
import pytest
from PySide6.QtCore import QPoint, QRectF, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from angle_cal.app import MainWindow
from angle_cal.trench_analysis import TrenchOptions, analyze_trench, export_trench_csv, find_local_bowing
from angle_cal.trench_panel import TrenchPanel


def trench_image(*, bulb=0, bright=False, noise=0, open_bottom=False):
    image = np.full((260, 200), 205, np.float32)
    image[:20] = 35
    for y in range(20, 260 if open_bottom else 222):
        distance = y - 20
        half = 30 + bulb * np.exp(-((distance - 90) / 28) ** 2)
        image[y, int(round(100 - half)):int(round(100 + half))] = 35
    if noise:
        image += np.random.default_rng(31).normal(0, noise, image.shape)
    image = np.clip(image, 0, 255).astype(np.uint8)
    return 255 - image if bright else image


def test_straight_flat_floor_cd_depth_and_angles():
    result = analyze_trench(trench_image(), TrenchOptions((30, 20, 140, 225), 10))
    assert result.depth_px == pytest.approx(201, abs=1.5)
    assert np.median(result.sample_right - result.sample_left) == pytest.approx(60, abs=1.5)
    assert result.left_angle_deg == pytest.approx(90, abs=.1)
    assert result.right_angle_deg == pytest.approx(90, abs=.1)
    assert result.left_bowing_px < 1
    assert result.right_bowing_px < 1
    assert not result.local_bowing


@pytest.mark.parametrize('inverted,dtype', [(False,np.uint8),(True,np.uint8),(False,np.uint16)])
def test_wall_tracks_bright_ridge_instead_of_dark_threshold(inverted, dtype):
    xx = np.arange(240)
    row = np.where((xx > 80) & (xx < 160), 25., 130.)
    row += 120 * np.exp(-((xx-79)/2.2)**2) + 120 * np.exp(-((xx-161)/2.2)**2)
    # A brighter unrelated material line must not replace the connected rim.
    row += 125 * np.exp(-((xx-65)/1.2)**2)
    image = np.tile(row, (240,1))
    image[:20] = 25
    image[220:] = 130
    image += np.random.default_rng(19).normal(0,1,image.shape)
    image = np.clip(image,0,255).astype(np.uint8)
    if inverted:
        image = 255-image
    if dtype == np.uint16:
        image = image.astype(dtype)*200
    result = analyze_trench(image,TrenchOptions((40,20,160,215),bright_trench=inverted))
    middle=(result.y>40)&(result.y<190)
    assert np.median(result.left[middle]) == pytest.approx(79,abs=1)
    assert np.median(result.right[middle]) == pytest.approx(161,abs=1)
    assert np.median(result.right[middle]-result.left[middle]) == pytest.approx(82,abs=1.5)


def test_strong_bowing_preserves_narrow_entrance():
    result = analyze_trench(trench_image(bulb=55), TrenchOptions((5, 20, 190, 225)))
    assert result.y[0] <= 22
    assert result.left_bowing_px > 50
    assert result.right_bowing_px > 50


def test_closed_and_cropped_trench_are_not_conflated():
    result = analyze_trench(trench_image(open_bottom=True), TrenchOptions((30, 20, 140, 240)))
    assert result.depth_px is None
    assert result.observed_depth_px >= 238
    result = analyze_trench(trench_image(), TrenchOptions((30, 20, 140, 110)))
    assert result.depth_px is None


@pytest.mark.parametrize("bright,dtype", [(False, np.uint8), (True, np.uint8), (False, np.uint16)])
def test_bowing_with_noise_polarity_and_16bit(bright, dtype):
    image = trench_image(bulb=12, bright=bright, noise=3)
    if dtype == np.uint16:
        image = image.astype(dtype) * 200
    result = analyze_trench(image, TrenchOptions((30, 20, 140, 225), bright_trench=bright, smooth_px=9))
    assert result.left_bowing_px == pytest.approx(12, abs=1.5)
    assert result.right_bowing_px == pytest.approx(12, abs=1.5)
    assert any(item["side"] == "left" and item["bowing_px"] > 9 for item in result.local_bowing)
    assert any(item["side"] == "right" and item["bowing_px"] > 9 for item in result.local_bowing)


def test_multiple_local_bulges_on_tapered_wall_have_distinct_regions():
    y = np.arange(400, dtype=float)
    outward = -.025 * y + 9 * np.exp(-((y - 110) / 20) ** 2) + 6 * np.exp(-((y - 275) / 14) ** 2)
    features = find_local_bowing(y, 60 - outward, 140 + outward, 9, 1)
    for side in ("left", "right"):
        regions = [f for f in features if f["side"] == side]
        assert len(regions) == 2
        assert regions[0]["peak_y"] == pytest.approx(110, abs=3)
        assert regions[1]["peak_y"] == pytest.approx(275, abs=3)
        assert regions[0]["chord_bowing_px"] > 7
        assert regions[1]["chord_bowing_px"] > 4
        assert all(f["return_confirmed"] for f in regions)


def test_straight_negative_taper_has_expansion_but_no_local_chord_bow():
    y = np.arange(200, dtype=float)
    features = find_local_bowing(y, 60 - .1 * y, 140 + .1 * y)
    assert len(features) == 2
    assert all(f["bowing_px"] > 18 for f in features)
    assert all(f["chord_bowing_px"] < .2 for f in features)
    assert not any(f["return_confirmed"] for f in features)


@pytest.mark.parametrize("roi,step", [((-1, 10, 80, 150), 10), ((0, 0, 3, 4), 10), ((0, 0, 200, 260), 0), ((0, 0, 200, 260), .0001)])
def test_reject_invalid_measurement_ranges(roi, step):
    with pytest.raises(ValueError):
        analyze_trench(trench_image(), TrenchOptions(roi, step))


def test_uniform_image_has_no_fabricated_measurements():
    with pytest.raises(ValueError):
        analyze_trench(np.full((100, 100), 100, np.uint8), TrenchOptions((0, 0, 100, 100)))


def test_rounded_entrance_radius_matches_known_geometry():
    image = np.full((260, 200), 205, np.uint8)
    image[:30] = 35
    for y in range(30, 220):
        d = y - 30
        half = 30 + (15 - np.sqrt(max(0, 225 - (15 - d) ** 2)) if d < 15 else 0)
        image[y, int(100 - half):int(100 + half)] = 35
    result = analyze_trench(image, TrenchOptions((30, 30, 140, 210), corner_window_px=25))
    assert all(corner is not None for corner in result.corners)
    assert result.corners[0]["radius_px"] == pytest.approx(15, abs=1.5)
    assert result.corners[1]["radius_px"] == pytest.approx(15, abs=1.5)


def test_csv_keeps_pixels_and_only_adds_calibrated_lengths(tmp_path):
    result = analyze_trench(trench_image(bulb=10), TrenchOptions((30, 20, 140, 225)))
    path = tmp_path / "분석.csv"
    export_trench_csv(path, result, .5)
    rows = list(csv.reader(path.open(encoding="utf-8-sig")))
    header = rows.index(["depth_px", "left_x_px", "right_x_px", "cd_px", "depth_nm", "cd_nm"])
    row = [float(v) for v in rows[header + 1]]
    assert row[4] == pytest.approx(row[0] * .5)
    assert row[5] == pytest.approx(row[3] * .5)
    export_trench_csv(path, result)
    rows = list(csv.reader(path.open(encoding="utf-8-sig")))
    assert rows[-1][-2:] == ["", ""]
    assert result.to_dict()["summary"]["depth_px"] is not None


@pytest.fixture
def qt_app():
    return QApplication.instance() or QApplication([])


def finish_analysis(panel, app):
    deadline = time.monotonic() + 12
    while panel.worker is not None and panel.worker.isRunning() and time.monotonic() < deadline:
        app.processEvents()
        QTest.qWait(10)
    panel.wait_for_worker()
    app.processEvents()
    assert panel.result is not None, panel.summary_label.text()


def test_workspace_real_mouse_roi_worker_navigation_and_exports(qt_app, tmp_path):
    panel = TrenchPanel()
    panel.resize(1150, 820)
    panel.show()
    panel.set_image(trench_image(bulb=12), "synthetic.png", .5)
    panel.unit_combo.setCurrentIndex(0)
    qt_app.processEvents()
    panel.view.fit_image()
    a = panel.view.mapFromScene(30, 20)
    b = panel.view.mapFromScene(170, 245)
    QTest.mousePress(panel.view.viewport(), Qt.MouseButton.LeftButton, pos=a)
    QTest.mouseMove(panel.view.viewport(), b)
    QTest.mouseRelease(panel.view.viewport(), Qt.MouseButton.LeftButton, pos=b)
    assert panel.roi is not None
    QTest.mouseClick(panel.analyze_button, Qt.MouseButton.LeftButton)
    finish_analysis(panel, qt_app)
    assert panel.cd_table.rowCount() >= 9
    assert panel.local_table.rowCount() >= 2
    panel._local_selected(0, 0)
    assert panel.selected_band.rect().height() > 0
    panel.graph.depth_selected.emit(120)
    assert panel.graph.selected_y == 120
    for suffix in ("csv", "json", "png"):
        path = tmp_path / f"결과.{suffix}"
        panel.save_result(str(path))
        assert path.stat().st_size > 100
    data = json.loads((tmp_path / "결과.json").read_text(encoding="utf-8"))
    assert data["local_bowing"]
    assert data["nm_per_px"] == .5
    panel.threshold_spin.setValue(100)
    assert panel.result is None and not panel.export_button.isEnabled()
    panel.close()


def test_unit_conversion_and_uncalibrated_guard(qt_app):
    panel = TrenchPanel()
    panel.set_image(trench_image(), "one", .5)
    panel.set_roi(QRectF(30, 20, 140, 225))
    panel.unit_combo.setCurrentIndex(1)  # nm
    panel.step_spin.setValue(10)
    assert panel.options().step_px == 20
    panel.unit_combo.setCurrentIndex(2)  # angstrom
    assert panel.options().step_px == 2
    panel.scale_spin.setValue(0)
    with pytest.raises(ValueError):
        panel.options()
    panel.close()


def test_image_switch_clears_corner_endpoint_mode_and_focus(qt_app):
    from PySide6.QtWidgets import QPushButton

    panel = TrenchPanel()
    panel.resize(1150, 820)
    panel.show()
    panel.set_image(trench_image(), "one.png")
    panel.set_roi(QRectF(30, 20, 140, 225))
    panel.start_analysis()
    finish_analysis(panel, qt_app)
    panel.image_tabs.setCurrentIndex(1)
    editor = panel.corner_editors[0]
    assert not editor.focus_rect.isEmpty()
    button = next(button for button in editor.findChildren(QPushButton) if button.text() == "시작점 찍기")
    QTest.mouseClick(button, Qt.MouseButton.LeftButton)
    assert editor.view.pick_endpoint == "start"

    panel.set_image(trench_image(), "two.png")
    qt_app.processEvents()
    assert editor.view.pick_endpoint is None
    assert editor.focus_rect.isEmpty()
    assert editor.corner is None
    assert not editor.region_button.isChecked()
    assert editor.view.cursor().shape() == Qt.CursorShape.OpenHandCursor
    selected = []
    editor.view.endpoint_selected.connect(selected.append)
    QTest.mouseClick(editor.view.viewport(), Qt.MouseButton.LeftButton,
                     pos=editor.view.viewport().rect().center())
    assert not selected
    panel.close()


def test_app_image_settings_survive_project_roundtrip(qt_app, tmp_path, monkeypatch):
    first, second = tmp_path / "one.png", tmp_path / "two.png"
    for path in (first, second):
        cv2.imencode(".png", trench_image())[1].tofile(str(path))
    window = MainWindow()
    monkeypatch.setattr(window, "_show_save_notification", lambda *_args: None)
    window._load_image_path(str(first))
    window.addon_actions["trench_analyzer"].setChecked(True)
    panel = window.trench_panel
    assert window.workspace_stack.currentWidget() is panel
    panel.set_roi(QRectF(30, 20, 140, 225))
    panel.step_spin.setValue(17)
    window._load_image_path(str(second))
    assert panel.roi is None
    window._load_image_path(str(first))
    assert panel.roi == (30, 20, 140, 225)
    assert panel.step_spin.value() == 17
    project = tmp_path / "test.anglecal.json"
    window._save_project_to_path(str(project))
    payload = json.loads(project.read_text(encoding="utf-8"))
    assert payload["image_states"][str(first)]["trench"]["step"] == 17
    assert window._normalize_image_state(payload["image_states"][str(first)])["trench"]["roi"] == [30, 20, 140, 225]
    assert not window.delete_action.isEnabled()
    window.addon_actions["trench_analyzer"].setChecked(False)
    assert window.workspace_stack.currentWidget() is window.canvas
    assert window.delete_action.isEnabled()
    window.close()


def test_rotation_clears_stale_roi_and_docks_restore(qt_app, tmp_path):
    from PySide6.QtWidgets import QDockWidget
    path = tmp_path / "sample.png"
    cv2.imencode(".png", trench_image())[1].tofile(str(path))
    window = MainWindow()
    window._load_image_path(str(path))
    before = {dock: not dock.isHidden() for dock in window.findChildren(QDockWidget)}
    window.addon_actions["trench_analyzer"].setChecked(True)
    panel = window.trench_panel
    panel.set_roi(QRectF(30, 20, 140, 225))
    panel._corner_changed(0, (35, 5, 80, 120), (.05, .95))
    finish_analysis(panel, qt_app)
    assert window.trench_state["corner_rois"][0] is not None
    assert window.trench_state["corner_spans"][0] is not None
    assert window.trench_state["corner_results"] is not None
    window.rotate_current_image(90)
    assert panel.roi is None
    assert panel.corner_rois == [None, None]
    assert panel.corner_spans == [None, None]
    assert panel.result is None
    assert window.trench_state["corner_rois"] == [None, None]
    assert window.trench_state["corner_spans"] == [None, None]
    assert window.trench_state["corner_results"] is None
    window.ribbon_tabs.setCurrentIndex(0)
    assert all((not dock.isHidden()) == visible for dock, visible in before.items())
    window.close()


def test_settings_change_discards_inflight_result(qt_app):
    panel = TrenchPanel()
    panel.set_image(trench_image(), "one")
    panel.set_roi(QRectF(30, 20, 140, 225))
    old_revision = panel.revision
    old_result = analyze_trench(panel.image, panel.options())
    panel.step_spin.setValue(19)
    panel._analysis_completed(old_result, "", old_revision)
    assert panel.result is None
    panel.close()


def test_gfe_path_units_field_order_and_full_profile():
    from angle_cal.trench_analysis import trench_gfe_tsv
    result = analyze_trench(trench_image(bulb=12), TrenchOptions((30, 20, 140, 225), 40))
    points = np.loadtxt(trench_gfe_tsv(result, .5).splitlines()[1:])
    center = (result.left[0] + result.right[0]) / 2
    lf,rf=result.field_angles
    reference=(np.polyval(lf["line"],result.left[0])+np.polyval(rf["line"],result.right[0]))/2
    assert points[0] == pytest.approx([(rf["points"][-1][0]-center)*5,(reference-rf["points"][-1][1])*5])
    assert points[-1] == pytest.approx([(lf["points"][0][0]-center)*5,(reference-lf["points"][0][1])*5])
    assert points[:, 1].min() == pytest.approx((reference-20-result.depth_px) * 5)
    assert len(points) >= 2 * len(result.y)
    assert len(points) > 2 * len(result.sample_y)
    # Every measured wall point survives, including the bowing section.
    center = (result.left[0] + result.right[0]) / 2
    for x, y in zip(result.right, result.y):
        assert np.min(np.linalg.norm(points - [(x-center)*5, (reference-y)*5], axis=1)) < 1e-7
    assert points[0,0]>0 and points[-1,0]<0


def test_gfe_clipboard_export_and_failure_preserves_clipboard(qt_app):
    from angle_cal.trench_analysis import trench_gfe_tsv
    panel = TrenchPanel()
    panel.result = analyze_trench(trench_image(), TrenchOptions((30, 20, 140, 225)))
    panel.scale_spin.setValue(.5)
    # Setting calibration invalidates measurements; simulate a completed analysis.
    panel.result = analyze_trench(trench_image(), TrenchOptions((30, 20, 140, 225)))
    panel.export_button.setEnabled(True)
    panel.export_button.click()
    copied = qt_app.clipboard().text()
    assert copied == trench_gfe_tsv(panel.result, .5)
    assert '복사 완료' in panel.hint.text()
    assert any('파일 저장' in a.text() for a in panel.export_button.menu().actions())
    panel.scale_spin.setValue(0)
    panel.result = analyze_trench(trench_image(), TrenchOptions((30, 20, 140, 225)))
    panel.copy_gfe_coordinates()
    assert qt_app.clipboard().text() == copied
    assert '스케일' in panel.hint.text()
    panel.scale_spin.setValue(.5)
    panel.result = analyze_trench(trench_image(open_bottom=True), TrenchOptions((30, 20, 140, 240)))
    panel.copy_gfe_coordinates()
    assert qt_app.clipboard().text() == copied
    assert '바닥' in panel.hint.text()
    panel.close()


def test_default_graph_shows_both_walls_with_shared_cd_scale(qt_app):
    panel = TrenchPanel()
    assert panel.graph_combo.currentData() == 'shape'
    graph = panel.graph
    graph.result = analyze_trench(trench_image(), TrenchOptions((30,20,140,225)))
    graph.resize(440,360)
    graph.show()
    qt_app.processEvents()
    image = graph.grab().toImage()
    rect = graph.plot_rect
    py = round(rect.top() + rect.height() * .75)
    left_pixels = [x for x in range(image.width()) if image.pixelColor(x,py).name() == '#007d92']
    right_pixels = [x for x in range(image.width()) if image.pixelColor(x,py).name() == '#bb4b1e']
    assert left_pixels and right_pixels
    # One shared horizontal axis preserves the measured wall separation.
    r = graph.result
    xs=np.r_[r.left,r.right,*[np.asarray(f["points"])[:,0] for f in r.field_angles if f["status"]=="valid"]]
    span = float(xs.max()-xs.min())
    low,high=graph.plot_y_bounds
    depth = low + .75*(high-low)
    cd = float(np.interp(depth,r.y,r.right)-np.interp(depth,r.y,r.left))
    assert np.mean(right_pixels)-np.mean(left_pixels) == pytest.approx(rect.width()*cd/(span+2*max(1,.1*span)),abs=2)
    graph.selected_y = 120
    graph.factor = 5
    graph.unit = 'Å'
    graph.update()
    qt_app.processEvents()
    assert not graph.grab().isNull()
    graph.close()
    panel.close()


def test_field_angle_tab_units_persistence_and_exports(qt_app,tmp_path):
    panel=TrenchPanel()
    panel.set_image(trench_image(),'field-sample',.5)
    panel.set_roi(QRectF(30,20,140,225))
    panel.start_analysis();finish_analysis(panel,qt_app)
    assert panel.result is not None
    assert all(item['status']=='valid' for item in panel.result.field_angles)
    panel.image_tabs.setCurrentIndex(2)
    qt_app.processEvents()
    for editor in panel.surface_editors:
        assert 'Field 지지 길이' in editor.label.text()
        assert '입구 내각' in editor.label.text()
        assert editor.items
    panel.save_result(str(tmp_path/'fields.json'))
    payload=json.loads((tmp_path/'fields.json').read_text(encoding='utf-8'))
    assert len(payload['field_angles'])==2
    assert payload['field_length_units']['angstrom'][0]['length_angstrom']==pytest.approx(payload['field_angles'][0]['length_px']*5)
    panel.save_result(str(tmp_path/'fields.csv'))
    assert 'field_angle' in (tmp_path/'fields.csv').read_text(encoding='utf-8-sig')
    panel.angle_start.setValue(55);panel.angle_end.setValue(110);panel.field_spin.setValue(40)
    state=panel.state()
    restored=TrenchPanel();restored.set_image(trench_image(),'restored',.5,state)
    assert restored.options().angle_span==(55,110)
    assert restored.options().field_search_px==40
    panel.close();restored.close()
