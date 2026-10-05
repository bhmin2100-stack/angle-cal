"""Known geometry is the radius oracle; generated SEMs are only UI fixtures."""
import cv2
import numpy as np
import pytest

from angle_cal.trench_analysis import TrenchOptions, analyze_trench, export_trench_csv
from angle_cal.trench_corner import fit_circle, measure_entrance_corner


def rounded_trench(left_radius=30, right_radius=None, noise=0, blur=0, rim=False, layers=False):
    right_radius = left_radius if right_radius is None else right_radius
    scale = 4
    yy, xx = np.mgrid[:400 * scale, :420 * scale].astype(float) / scale
    left = np.full_like(yy, 160.)
    right = np.full_like(yy, 260.)
    for radius, boundary, sign in ((left_radius, left, -1), (right_radius, right, 1)):
        if radius:
            d = np.clip(yy - 45, 0, radius)
            boundary += sign * (radius - np.sqrt(np.maximum(0, radius ** 2 - (radius - d) ** 2)))
    void = (yy < 45) | ((xx > left) & (xx < right) & (yy < 350))
    im = np.where(void, 35., 205.).astype(np.float32)
    if layers:
        im[~void] -= 55 * ((yy[~void] // 17) % 2)
    if rim:
        solid = (~void).astype(np.uint8)
        boundary = solid - cv2.erode(solid, np.ones((9, 9), np.uint8))
        im[boundary > 0] = 250
    im = cv2.resize(im, (420, 400), interpolation=cv2.INTER_AREA)
    if blur:
        im = cv2.GaussianBlur(im, (0, 0), blur)
    if noise:
        im += np.random.default_rng(913).normal(0, noise, im.shape)
    return np.clip(im, 0, 255).astype(np.uint8)


def analyze(image, **kwargs):
    return analyze_trench(image, TrenchOptions((60, 45, 300, 335), **kwargs))


@pytest.mark.parametrize("radius", [8, 15, 30, 60])
def test_known_radii(radius):
    for c in analyze(rounded_trench(radius)).corners:
        assert c["status"] == "valid", c["reason"]
        assert c["radius_px"] == pytest.approx(radius, abs=max(1, radius * .1))
        assert c["arc_angle_deg"] >= 45


@pytest.mark.parametrize("variant", ["noise", "blur", "rim", "layers", "bright", "16bit"])
def test_asymmetric_radii_with_image_variants(variant):
    im = rounded_trench(15, 30, noise=3 if variant == "noise" else 0,
                         blur=.6 if variant == "blur" else 0,
                         rim=variant == "rim", layers=variant == "layers")
    if variant == "bright":
        im = 255 - im
    if variant == "16bit":
        im = im.astype(np.uint16) * 201
    result = analyze(im, bright_trench=variant == "bright")
    for c, radius in zip(result.corners, (15, 30)):
        assert c["radius_px"] == pytest.approx(radius, abs=max(1.5, radius * .12)), c["reason"]


def test_sharp_and_cropped_entrances_are_unconfirmed():
    assert all(c["radius_px"] is None for c in analyze(rounded_trench(0)).corners)
    image = rounded_trench(30)[55:]
    result = analyze_trench(image, TrenchOptions((60, 0, 300, 320)))
    assert all(c["radius_px"] is None for c in result.corners)


def test_blurred_sharp_corner_is_not_reported_as_a_resolved_radius():
    for sigma in (3, 5):
        result = analyze(rounded_trench(0, blur=sigma))
        for corner in result.corners:
            assert corner["radius_px"] is None
            assert corner["status"] == "unconfirmed"
            assert corner["edge_spread_px"] > 5
            assert "흐림" in corner["reason"]


def test_search_changes_do_not_change_other_metrics():
    image = rounded_trench(30)
    results = [analyze(image, corner_window_px=w) for w in (20, 24, 32)]
    for result in results:
        assert result.left_angle_deg == results[0].left_angle_deg
        assert result.left_bowing_px == results[0].left_bowing_px
        assert np.array_equal(result.sample_left, results[0].sample_left)
        for c in result.corners:
            assert c["radius_px"] == pytest.approx(30, abs=3)


def test_short_manual_span_is_rejected_and_preserved_in_exports(tmp_path):
    result = analyze(rounded_trench(), corner_spans=((.4, .6), None))
    assert result.corners[0]["radius_px"] is None
    assert result.corners[1]["radius_px"] is not None
    payload = result.to_dict(.5)
    assert payload["options"]["corner_spans"][0] == (.4, .6)
    assert payload["corners"][0]["reason"]
    assert payload["corners"][1]["fit_points"]
    assert payload["corner_length_units"]["angstrom"][1]["radius_angstrom"] == pytest.approx(result.corners[1]["radius_px"] * 5)
    path = tmp_path / "corner.csv"
    export_trench_csv(path, result, .5)
    text = path.read_text(encoding="utf-8-sig")
    assert "corner_left,status,unconfirmed" in text
    assert "corner_right,radius_angstrom" in text
    assert "right,arc_points,0" in text


def test_circle_fit_resists_outliers_and_large_image_coordinates():
    theta = np.linspace(-np.pi / 2, 0, 70)
    points = np.array([100000, 200000]) + 30 * np.column_stack((np.cos(theta), np.sin(theta)))
    points[20] += (2, -2)
    fit = fit_circle(points)
    assert fit[1] == pytest.approx(30, abs=.5)


def test_manual_roi_cannot_measure_a_lower_layer_as_an_entrance():
    result = analyze(rounded_trench(30, layers=True), corner_rois=((95, 100, 100, 100), None))
    assert result.corners[0]["radius_px"] is None
    assert result.corners[1]["radius_px"] == pytest.approx(30, abs=3)


def test_corner_editor_mouse_roi_spans_reset_and_unit_exports(tmp_path):
    import json
    import time
    from PySide6.QtCore import QRectF, Qt
    from PySide6.QtGui import QImage
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication
    from angle_cal.trench_panel import TrenchPanel

    app = QApplication.instance() or QApplication([])
    panel = TrenchPanel()
    panel.resize(1400, 1050)
    panel.show()
    image = rounded_trench(30)
    panel.set_image(image, "known-radius.png", .5)
    panel.set_roi(QRectF(60, 45, 300, 335))
    panel.image_tabs.setCurrentIndex(1)

    def finish():
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            app.processEvents()
            if panel.worker is None:
                break
            QTest.qWait(10)
        assert panel.worker is None
        assert panel.result is not None, panel.summary_label.text()

    panel.start_analysis()
    finish()
    original_right = panel.result.corners[1]["radius_px"]
    editor = panel.corner_editors[0]
    editor.start_spin.setValue(5)
    editor.end_spin.setValue(95)
    QTest.mouseClick(editor.apply_button, Qt.MouseButton.LeftButton)
    finish()
    assert panel.result.corners[0]["source"] == "manual"
    assert panel.result.corners[0]["radius_px"] == pytest.approx(30, abs=3)
    assert panel.result.corners[1]["radius_px"] == original_right
    editor.view.fit_image()
    QTest.mouseClick(editor.region_button, Qt.MouseButton.LeftButton)
    app.processEvents()
    a = editor.view.mapFromScene(85, 20)
    b = editor.view.mapFromScene(195, 145)
    QTest.mousePress(editor.view.viewport(), Qt.MouseButton.LeftButton, pos=a)
    QTest.mouseMove(editor.view.viewport(), b)
    QTest.mouseRelease(editor.view.viewport(), Qt.MouseButton.LeftButton, pos=b)
    finish()
    assert panel.corner_rois[0] is not None
    assert panel.corner_spans[0] is None
    assert panel.result.corners[0]["radius_px"] == pytest.approx(30, abs=3)
    saved = panel.state()
    assert saved["corner_results"][0]["status"] == "valid"
    panel.set_image(image.copy(), "restored.png", .5, saved)
    panel.start_analysis()
    finish()
    assert panel.corner_rois[0] == saved["corner_rois"][0]
    panel.unit_combo.setCurrentIndex(2)
    panel.start_analysis()
    finish()
    assert "Å" in editor.label.text()
    for suffix in ("csv", "json", "png"):
        panel.save_result(str(tmp_path / ("corner." + suffix)))
    data = json.loads((tmp_path / "corner.json").read_text(encoding="utf-8"))
    assert data["corners"][0]["roi"] == list(panel.corner_rois[0])
    assert data["corners"][0]["status"] == "valid"
    assert QImage(str(tmp_path / "corner.png")).height() == image.shape[0] + 360
    QTest.mouseClick(editor.reset_button, Qt.MouseButton.LeftButton)
    finish()
    assert panel.corner_rois == [None, None]
    assert panel.corner_spans == [None, None]
    assert panel.result.corners[0]["source"] == "auto"
    panel.close()
