"""Render the actual Qt workspace and export a synthetic sample analysis.

Run with QT_QPA_PLATFORM=offscreen for headless verification. The bundled SEM
sample is synthetic; these exports demonstrate the UI, not calibrated metrology.
"""
from pathlib import Path
import time

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from angle_cal.app import MainWindow


def main():
    root = Path(__file__).resolve().parents[1]
    out = root / "build" / "trench_verification"
    out.mkdir(parents=True, exist_ok=True)
    app = QApplication.instance() or QApplication([])
    font_path = Path("C:/Windows/Fonts/malgun.ttf")
    if font_path.exists():
        QFontDatabase.addApplicationFont(str(font_path))
        app.setFont(QFont("Malgun Gothic", 9))
    window = MainWindow()
    window.resize(1600, 1050)
    window.show()
    window._load_image_path(str(root / "samples/trench_analyzer/01_symmetric_mild_bowing_scale.png"))
    window.addon_actions["trench_analyzer"].setChecked(True)
    panel = window.trench_panel
    panel.set_roi(QRectF(625, 20, 300, 920))
    panel.smooth_spin.setValue(17)
    panel.negative_spin.setValue(2)
    app.processEvents()
    panel.view.fitInView(QRectF(570, 0, 420, 945), Qt.AspectRatioMode.KeepAspectRatio)
    panel.view.notify_viewport()
    panel.start_analysis()
    deadline = time.monotonic() + 20
    while panel.worker is not None and panel.worker.isRunning() and time.monotonic() < deadline:
        app.processEvents()
        QTest.qWait(10)
    panel.wait_for_worker()
    app.processEvents()
    if panel.result is None:
        raise RuntimeError(panel.summary_label.text())
    panel.graph_combo.setCurrentIndex(1)
    panel.tables.setCurrentIndex(1)
    if panel.result.local_bowing:
        panel._local_selected(0, 0)
    app.processEvents()
    window.grab().save(str(out / "workspace.png"))
    panel.image_tabs.setCurrentIndex(1)
    app.processEvents()
    for editor in panel.corner_editors:
        editor.fit_corner()
    window.grab().save(str(out / "entrance_corners.png"))
    for extension in ("csv", "json", "png"):
        panel.save_result(str(out / f"synthetic_trench.{extension}"))
    print({"sample": "synthetic", "cd_rows": panel.cd_table.rowCount(),
           "local_rows": panel.local_table.rowCount(), "depth_px": panel.result.depth_px,
           "left_bow_px": panel.result.left_bowing_px, "right_bow_px": panel.result.right_bowing_px,
           "output": str(out)})
    window.close()


if __name__ == "__main__":
    main()
