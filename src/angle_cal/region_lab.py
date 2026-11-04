"""Independent region-segmentation experiment window; no measurement state writes."""
from datetime import datetime
from pathlib import Path
import sys
import threading

import cv2
import numpy as np
from PySide6.QtCore import Qt, Signal, QThread
from PySide6.QtGui import QImage, QPixmap, QPainter
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDialog,
    QDoubleSpinBox, QFileDialog, QFormLayout, QGraphicsScene, QGraphicsView,
    QHBoxLayout, QLabel, QMessageBox, QPushButton, QSlider, QSpinBox,
    QSplitter, QVBoxLayout, QWidget)

from .region_segmentation import (RegionOptions, normalized_rgb, render_regions,
                                  save_regions, segment_regions)


class RegionView(QGraphicsView):
    picked = Signal(int, int)
    moved = Signal()

    def __init__(self):
        super().__init__()
        self.setScene(QGraphicsScene(self))
        self.item = self.scene().addPixmap(QPixmap())
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setBackgroundBrush(Qt.GlobalColor.black)
        self.horizontalScrollBar().valueChanged.connect(lambda _: self.moved.emit())
        self.verticalScrollBar().valueChanged.connect(lambda _: self.moved.emit())
        self._start = None

    def set_pixels(self, rgb, fit=False):
        a = np.ascontiguousarray(rgb, dtype=np.uint8)
        image = QImage(a.data, a.shape[1], a.shape[0], a.strides[0], QImage.Format.Format_RGB888).copy()
        self.item.setPixmap(QPixmap.fromImage(image))
        self.scene().setSceneRect(self.item.boundingRect())
        if fit:
            self.fit_image()

    def fit_image(self):
        self.fitInView(self.item, Qt.AspectRatioMode.KeepAspectRatio)
        self.moved.emit()

    def wheelEvent(self, event):
        factor = 1.2 if event.angleDelta().y() > 0 else 1 / 1.2
        if .02 <= self.transform().m11() * factor <= 40:
            self.scale(factor, factor)
            self.moved.emit()
        event.accept()

    def mousePressEvent(self, event):
        self._start = event.position().toPoint()
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        point = event.position().toPoint()
        if self._start is not None and (point - self._start).manhattanLength() < 4 and event.button() == Qt.MouseButton.LeftButton:
            pos = self.mapToScene(point)
            if self.item.boundingRect().contains(pos):
                self.picked.emit(int(pos.x()), int(pos.y()))
        super().mouseReleaseEvent(event)
        self.moved.emit()


class RegionWorker(QThread):
    ready = Signal(object)
    failed = Signal(str)

    def __init__(self, image, options, parent):
        super().__init__(parent)
        self.image = image
        self.options = options
        self.cancelled = threading.Event()

    def run(self):
        try:
            self.ready.emit(segment_regions(self.image, self.options, self.cancelled.is_set))
        except ImportError as exc:
            self.failed.emit(f'영역 분할 패키지 로딩 실패: {exc}. 개발 환경은 pip install -e ".[regions]"로 설치하세요.')
        except Exception as exc:
            self.failed.emit(str(exc))


class RegionLab(QDialog):
    def __init__(self, parent=None, image=None, source=""):
        super().__init__(parent)
        self.setWindowTitle("AngleCal · 영역 분할 실험실")
        self.resize(1280, 840)
        self.setMinimumSize(900, 620)
        self.image = self.rgb = self.result = self.worker = None
        self.source = ""
        self.selected = 0
        self._syncing = False
        self._closing = False
        root = QVBoxLayout(self)
        title = QLabel("영역 분할 실험실")
        title.setStyleSheet("font-size:22px;font-weight:700")
        root.addWidget(title)
        root.addWidget(QLabel("재료 구분 없이 전체 이미지를 연결된 영역으로 나눕니다.  휠: 확대 · 드래그: 이동 · 클릭: 영역 선택"))
        tools = QHBoxLayout()
        self.open_button = QPushButton("이미지 열기")
        self.open_button.clicked.connect(self.choose_image)
        tools.addWidget(self.open_button)
        self.file_label = QLabel("이미지를 열어 시작하세요.")
        self.file_label.setWordWrap(True)
        tools.addWidget(self.file_label, 1)
        fit = QPushButton("화면 맞춤")
        fit.clicked.connect(self.fit_views)
        tools.addWidget(fit)
        root.addLayout(tools)

        body = QSplitter()
        settings = QWidget()
        settings.setMinimumWidth(235)
        settings.setMaximumWidth(300)
        controls = QVBoxLayout(settings)
        form = QFormLayout()
        self.detail = QSlider(Qt.Orientation.Horizontal)
        self.detail.setRange(0, 100)
        self.detail.setValue(50)
        self.detail_label = QLabel()
        self.detail.valueChanged.connect(self.settings_changed)
        form.addRow("분할 세밀도", self.detail)
        form.addRow(self.detail_label)
        self.min_size = QSpinBox()
        self.min_size.setRange(1, 100000)
        self.min_size.setValue(48)
        self.min_size.setSuffix(" px")
        self.min_size.valueChanged.connect(self.settings_changed)
        form.addRow("최소 영역 면적", self.min_size)
        self.smoothing = QDoubleSpinBox()
        self.smoothing.setRange(0, 5)
        self.smoothing.setSingleStep(.1)
        self.smoothing.setValue(1.0)
        self.smoothing.valueChanged.connect(self.settings_changed)
        form.addRow("노이즈 완화", self.smoothing)
        self.texture = QDoubleSpinBox()
        self.texture.setRange(0, 2)
        self.texture.setSingleStep(.1)
        self.texture.setValue(0)
        self.texture.valueChanged.connect(self.settings_changed)
        form.addRow("질감 비중", self.texture)
        self.resolution = QComboBox()
        for label, value in [("빠른 비교 · 긴 변 1000", 1000), ("기본 · 긴 변 1600", 1600), ("정밀 · 긴 변 2400", 2400), ("원본 (최대 800만 px)", 0)]:
            self.resolution.addItem(label, value)
        self.resolution.setCurrentIndex(1)
        self.resolution.currentIndexChanged.connect(self.settings_changed)
        form.addRow("분석 해상도", self.resolution)
        controls.addLayout(form)
        hint = QLabel("세밀도 ↑: 더 잘게 분리\n최소 면적·노이즈 완화 ↑: 작은 영역 병합\n질감 > 0: 명암·국소 질감 사용 (색상 제외)\n\n얇은 막은 최소 면적과 노이즈 완화를 낮추고 원본 해상도로 비교하세요.")
        hint.setWordWrap(True)
        controls.addWidget(hint)
        self.run_button = QPushButton("전체 영역 분할")
        self.run_button.setMinimumHeight(40)
        self.run_button.clicked.connect(self.start)
        self.run_button.setEnabled(False)
        controls.addWidget(self.run_button)
        self.cancel_button = QPushButton("분할 취소")
        self.cancel_button.clicked.connect(self.cancel)
        self.cancel_button.setEnabled(False)
        controls.addWidget(self.cancel_button)
        controls.addSpacing(15)
        controls.addWidget(QLabel("결과 표시"))
        self.mode = QComboBox()
        for label, mode in [("색상 겹쳐 보기", "overlay"), ("영역 색상만", "colors"), ("경계선만", "boundaries"), ("원본", "original")]:
            self.mode.addItem(label, mode)
        self.mode.currentIndexChanged.connect(self.redraw)
        controls.addWidget(self.mode)
        self.opacity = QSlider(Qt.Orientation.Horizontal)
        self.opacity.setRange(0, 100)
        self.opacity.setValue(50)
        self.opacity.valueChanged.connect(self.redraw)
        controls.addWidget(QLabel("색상 불투명도"))
        controls.addWidget(self.opacity)
        self.link_views = QCheckBox("확대·이동 동기화")
        self.link_views.setChecked(True)
        controls.addWidget(self.link_views)
        clear = QPushButton("영역 선택 해제")
        clear.clicked.connect(self.clear_selection)
        controls.addWidget(clear)
        self.selection_label = QLabel("결과에서 영역을 클릭하세요.")
        self.selection_label.setWordWrap(True)
        controls.addWidget(self.selection_label)
        controls.addStretch(1)
        self.save_button = QPushButton("결과 저장…")
        self.save_button.setEnabled(False)
        self.save_button.clicked.connect(self.save)
        controls.addWidget(self.save_button)
        body.addWidget(settings)
        views = QSplitter()
        self.original_view = RegionView()
        self.result_view = RegionView()
        for text, view in [("원본", self.original_view), ("영역 분할 결과", self.result_view)]:
            panel = QWidget()
            layout = QVBoxLayout(panel)
            layout.setContentsMargins(2, 0, 2, 0)
            layout.addWidget(QLabel(text))
            layout.addWidget(view)
            views.addWidget(panel)
            view.picked.connect(self.select_region)
        self.original_view.moved.connect(lambda: self.sync_view(self.original_view, self.result_view))
        self.result_view.moved.connect(lambda: self.sync_view(self.result_view, self.original_view))
        body.addWidget(views)
        body.setStretchFactor(1, 1)
        body.setSizes([280, 960])
        views.setSizes([480, 480])
        root.addWidget(body, 1)
        self.status = QLabel("대기 중 · 자동 분할은 재료 판별이나 검증된 측정 결과가 아닙니다.")
        self.status.setWordWrap(True)
        root.addWidget(self.status)
        self.settings_changed()
        if image is not None:
            self.set_image(image, source)

    def options(self):
        return RegionOptions(self.detail.value(), self.min_size.value(), self.smoothing.value(),
                             self.texture.value(), self.resolution.currentData())

    def settings_changed(self, *_):
        self.detail_label.setText(f"크게 묶기  ←  {self.detail.value()}  →  세밀하게")
        if self.result is not None and hasattr(self, 'status'):
            if self.options() != self.result.options:
                self.status.setText("설정이 변경되었습니다. 현재 결과는 이전 설정입니다. ‘전체 영역 분할’을 눌러 적용하세요.")
            else:
                self.result_status()

    def choose_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "분할할 이미지", "", "Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp)")
        if path:
            self.open_path(path)

    def open_path(self, path):
        try:
            image = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_UNCHANGED)
            if image is None:
                raise ValueError("이미지 파일을 읽을 수 없습니다.")
            self.set_image(image, path)
        except Exception as exc:
            QMessageBox.warning(self, "이미지 열기", str(exc))

    def set_image(self, image, source=""):
        if self.worker is not None:
            raise RuntimeError("분할이 끝난 뒤 이미지를 변경하세요.")
        rgb = normalized_rgb(image)
        self.image, self.rgb = image.copy(), rgb
        self.source, self.result, self.selected = source, None, 0
        self.file_label.setText(f"{Path(source).name or '현재 이미지 복사본'} · {image.shape[1]} × {image.shape[0]} px · {image.dtype}")
        self.original_view.set_pixels(rgb * 255, True)
        self.result_view.set_pixels(rgb * 255, True)
        self.selection_label.setText("결과에서 영역을 클릭하세요.")
        self.status.setText("이미지 준비 완료. 전체 영역 분할을 실행하세요.")
        self.run_button.setEnabled(True)
        self.save_button.setEnabled(False)

    def fit_views(self):
        self._syncing = True
        self.original_view.fit_image()
        self.result_view.fit_image()
        scale = min(self.original_view.transform().m11(), self.result_view.transform().m11())
        for view in (self.original_view, self.result_view):
            view.resetTransform()
            view.scale(scale, scale)
            view.centerOn(view.item.boundingRect().center())
        self._syncing = False

    def sync_view(self, source, target):
        if self._syncing or not self.link_views.isChecked():
            return
        self._syncing = True
        target.setTransform(source.transform())
        target.centerOn(source.mapToScene(source.viewport().rect().center()))
        self._syncing = False

    def start(self):
        if self.image is None or self.worker is not None:
            return
        self.worker = RegionWorker(self.image, self.options(), self)
        self.worker.ready.connect(self.finished_result)
        self.worker.failed.connect(self.failed)
        self.worker.finished.connect(self.worker_finished)
        for control in self.analysis_controls():
            control.setEnabled(False)
        self.save_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.status.setText("전체 영역을 분할하는 중… 이전 결과가 있다면 완료될 때까지 유지됩니다.")
        self.worker.start()

    def analysis_controls(self):
        return (self.open_button, self.run_button, self.detail, self.min_size, self.smoothing, self.texture, self.resolution)

    def cancel(self):
        if self.worker:
            self.worker.cancelled.set()
            self.cancel_button.setEnabled(False)
            self.status.setText("취소 요청됨 · 현재 계산 단계가 끝나면 정리합니다.")

    def finished_result(self, result):
        if self.worker.cancelled.is_set():
            self.failed("분할을 취소했습니다.")
            return
        self.result = result
        self.selected = 0
        self.selection_label.setText("결과에서 영역을 클릭하세요.")
        self.redraw()
        self.result_status()

    def result_status(self):
        r = self.result
        h, w = r.analysis_shape
        note = "원본 분석" if r.analysis_shape == r.labels.shape else "축소 분석 → 원본 크기로 최근접 확대 (세부 경계 손실 가능)"
        self.status.setText(f"{r.count:,}개 영역 · {r.elapsed:.2f}초 · 분석 {w} × {h} px · {note}")

    def failed(self, message):
        suffix = " · 화면과 저장은 이전 성공 결과입니다." if self.result is not None else ""
        self.status.setText(message + suffix)

    def worker_finished(self):
        self.worker.deleteLater()
        self.worker = None
        for control in self.analysis_controls():
            control.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.save_button.setEnabled(self.result is not None)
        if self._closing:
            self.reject()

    def redraw(self, *_):
        if self.result is not None:
            self.result_view.set_pixels(render_regions(self.rgb, self.result, self.mode.currentData(), self.opacity.value() / 100, self.selected))

    def select_region(self, x, y):
        if self.result is None or not (0 <= y < self.result.labels.shape[0] and 0 <= x < self.result.labels.shape[1]):
            return
        self.selected = int(self.result.labels[y, x])
        area = int(np.count_nonzero(self.result.labels == self.selected))
        self.selection_label.setText(f"영역 {self.selected}\n면적 {area:,} px ({100 * area / self.result.labels.size:.2f}%)\n좌표 ({x}, {y}) · 원본 픽셀 기준")
        self.redraw()

    def clear_selection(self):
        self.selected = 0
        self.selection_label.setText("결과에서 영역을 클릭하세요.")
        self.redraw()

    def save(self):
        if self.result is None:
            return
        directory = QFileDialog.getExistingDirectory(self, "결과를 저장할 상위 폴더")
        if directory:
            try:
                name = "regions-" + datetime.now().strftime("%Y%m%d-%H%M%S-%f")
                path = save_regions(Path(directory) / name, self.source, self.rgb, self.result)
                self.status.setText(f"저장 완료: {path} · 색상/겹침/경계 PNG, labels.npz, result.json")
            except Exception as exc:
                QMessageBox.warning(self, "저장 실패", str(exc))

    def reject(self):
        if self.worker is not None:
            self._closing = True
            self.cancel()
            return
        super().reject()

    def closeEvent(self, event):
        if self.worker is not None:
            self._closing = True
            self.cancel()
            event.ignore()
        else:
            super().closeEvent(event)


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--self-test":
        import json
        import traceback
        report = {}
        try:
            sample = np.full((96, 160), 100, np.uint16)
            sample[:, 50:54] = 50000
            sample[:, 100:] = 30000
            result = segment_regions(sample, RegionOptions(min_size=1, smoothing=0))
            assert result.count == 4
            assert result.labels.shape == sample.shape
            report = {"ok": True, "regions": result.count, "shape": list(result.labels.shape)}
        except Exception:
            report = {"ok": False, "error": traceback.format_exc()}
        Path(sys.argv[2]).write_text(json.dumps(report, indent=2), encoding="utf-8")
        sys.exit(0 if report["ok"] else 1)
    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyle("Fusion")
    window = RegionLab()
    window.show()
    if len(sys.argv) > 1:
        window.open_path(sys.argv[1])
    window.fit_views()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
