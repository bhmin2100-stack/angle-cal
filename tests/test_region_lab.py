import json
import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("ANGLECAL_DISABLE_BROWSER_SESSION", "1")

import cv2
import numpy as np
import pytest
pytest.importorskip("skimage")
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from angle_cal.region_segmentation import (RegionOptions, boundaries, normalized_rgb,
    render_regions, save_regions, segment_regions)
from angle_cal.region_lab import RegionLab


def layered_image():
    image = np.full((160, 240), 120, np.uint8)
    image[25:135, 30:80] = 35
    image[25:135, 150:200] = 35
    image[60:64, :] = 220  # four-pixel film across the image
    return image


def test_full_coverage_disconnected_objects_and_thin_film():
    source = layered_image()
    result = segment_regions(source, RegionOptions(min_size=1, smoothing=0, max_dimension=0))
    labels = result.labels
    assert labels.dtype == np.int32 and labels.shape == source.shape
    assert np.array_equal(np.unique(labels), np.arange(1, result.count + 1))
    assert len({labels[40, 40], labels[40, 170], labels[40, 100], labels[61, 100]}) == 4
    assert len(np.unique(labels[60:64])) == 1
    assert np.all(boundaries(labels)[60])
    assert np.array_equal(labels, segment_regions(source, result.options).labels)


def test_uniform_image_is_one_region_and_input_not_modified():
    source = np.full((65, 70, 3), 89, np.uint8)
    before = source.copy()
    result = segment_regions(source)
    assert result.count == 1
    assert not boundaries(result.labels).any()
    assert np.array_equal(source, before)


def test_16bit_native_contrast_is_not_truncated():
    source = np.full((80, 100), 30000, np.uint16)
    source[:, 50:] = 30100
    result = segment_regions(source, RegionOptions(smoothing=0, min_size=1))
    assert result.count == 2
    assert result.labels[40, 30] != result.labels[40, 70]


def test_downsample_coordinates_and_export(tmp_path):
    source = layered_image()
    result = segment_regions(source, RegionOptions(max_dimension=120))
    assert result.analysis_shape == (80, 120)
    assert result.labels.shape == source.shape
    rgb = normalized_rgb(source)
    saved = save_regions(tmp_path / "한글 결과", "실험.tif", rgb, result)
    assert np.array_equal(np.load(saved / "labels.npz")["labels"], result.labels)
    meta = json.loads((saved / "result.json").read_text(encoding="utf-8"))
    assert sum(meta["area_pixels"]) == source.size
    assert meta["analysis_shape"] == [80, 120]
    assert meta["material_identification"] is False
    for name in ("regions", "overlay", "boundaries"):
        img = cv2.imdecode(np.fromfile(saved / f"{name}.png", np.uint8), cv2.IMREAD_COLOR)
        assert img.shape == (*source.shape, 3)
    with pytest.raises(FileExistsError):
        save_regions(saved, "", rgb, result)


def test_color_and_texture_modes_and_invalid_input():
    image = np.zeros((80, 100, 3), np.uint8)
    image[:, :50] = (200, 20, 20)
    image[:, 50:] = (20, 20, 200)
    r = segment_regions(image, RegionOptions(smoothing=0, min_size=1))
    assert r.count == 2
    rgb = normalized_rgb(image)
    assert np.array_equal(render_regions(rgb, r, "original"), image[..., ::-1])
    assert render_regions(rgb, r, "colors", selected=1).shape == image.shape
    assert segment_regions(image, RegionOptions(texture=1)).count >= 1
    with pytest.raises(ValueError):
        segment_regions(np.full((5, 5), np.nan))
    with pytest.raises(InterruptedError):
        segment_regions(image, cancelled=lambda: True)


def wait_worker(app, window):
    deadline = time.monotonic() + 20
    while window.worker is not None and time.monotonic() < deadline:
        app.processEvents()
        QTest.qWait(10)
    assert window.worker is None


def test_window_run_selection_stale_settings_and_image_change():
    app = QApplication.instance() or QApplication([])
    window = RegionLab(image=layered_image(), source="test.png")
    window.show()
    app.processEvents()
    QTest.mouseClick(window.run_button, Qt.MouseButton.LeftButton)
    wait_worker(app, window)
    assert window.result is not None and window.save_button.isEnabled()
    window.select_region(40, 40)
    assert window.selected > 0 and "면적" in window.selection_label.text()
    window.mode.setCurrentIndex(2)
    window.detail.setValue(80)
    assert "이전 설정" in window.status.text()
    window.set_image(np.full((70, 80), 80, np.uint8))
    assert window.result is None and not window.save_button.isEnabled()
    assert window.selected == 0
    window.close()


def test_cancel_and_close_waits_for_worker():
    app = QApplication.instance() or QApplication([])
    window = RegionLab(image=layered_image())
    window.show()
    window.start()
    window.reject()
    wait_worker(app, window)
    assert window.result is None
    assert not window.isVisible()


def test_addon_available_without_changing_existing_definitions():
    from angle_cal.app import ADDON_DEFINITIONS, MainWindow
    ids = {definition.addon_id for definition in ADDON_DEFINITIONS}
    assert {"region_lab", "trench_analyzer", "photo_merge", "cliff_angle_analyzer"} <= ids
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.set_addon_enabled("region_lab", True)
    assert window.ribbon_tabs.currentWidget() is window.addon_pages["region_lab"]
    window.close()
