"""Linked image, sampling band and brightness-profile landmark editor."""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal, QTimer
from PySide6.QtGui import QColor, QImage, QPainter, QPainterPath, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import (QDialog, QDialogButtonBox, QDoubleSpinBox, QGraphicsScene,
    QGraphicsView, QHBoxLayout, QLabel, QPushButton, QSizePolicy, QSpinBox, QSplitter, QTabWidget, QVBoxLayout, QWidget)

from .image_ops import bgr_to_rgb8_for_display
from .trench_boundary_sample import make_boundary_sample, sample_point, sample_polygon, validate_boundary_sample, validate_boundary_samples
from .trench_guides import sweep_samples


def _pen(color, width=1.5):
    pen = QPen(QColor(color), width); pen.setCosmetic(True)
    return pen


class SampleImageView(QGraphicsView):
    line_drawn = Signal(object, object)
    point_picked = Signal(object)

    def __init__(self, image, guides=(), parent=None):
        super().__init__(parent)
        # A zoomed scene must not inflate the surrounding tab/dialog size hint.
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.setScene(QGraphicsScene(self)); self.setBackgroundBrush(QColor('#20262e'))
        rgb = np.ascontiguousarray(bgr_to_rgb8_for_display(image))
        pixmap = QPixmap.fromImage(QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format.Format_RGB888).copy())
        self.scene().addPixmap(pixmap); self.setSceneRect(0, 0, pixmap.width(), pixmap.height())
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.drawing = True; self.origin = None; self.pan_origin = None; self.items = []
        self.setCursor(Qt.CursorShape.CrossCursor)
        for guide in guides:
            points = sweep_samples(guide)
            path = QPainterPath(QPointF(*points[0]))
            for p in points[1:]:path.lineTo(QPointF(*p))
            item = self.scene().addPath(path, _pen('#83a5c4', 1)); item.setOpacity(.45)

    def fit_image(self):
        self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def show_sample(self, sample):
        for item in self.items:self.scene().removeItem(item)
        self.items = []
        if sample is None:return
        polygon = QPolygonF([QPointF(*p) for p in sample_polygon(sample)])
        fill = QColor('#00d6df'); fill.setAlpha(40)
        self.items.append(self.scene().addPolygon(polygon, _pen('#00d6df'), fill))
        a, b = sample['start'], sample['end']
        self.items.append(self.scene().addLine(*a, *b, _pen('#00d6df')))
        for label, p in (('A', a), ('B', b)):
            item = self.scene().addSimpleText(label); item.setBrush(QColor('#00ebef'))
            item.setFlag(item.GraphicsItemFlag.ItemIgnoresTransformations, True)
            item.setPos(*p); self.items.append(item)
        p = sample_point(sample); a, b = np.asarray(a), np.asarray(b)
        normal = np.array([-(b-a)[1], (b-a)[0]])/np.linalg.norm(b-a)
        delta = normal*max(3, sample['half_width_px']+2)
        self.items.append(self.scene().addLine(*(p-delta), *(p+delta), _pen('#ffb04f', 2)))
        self.items.append(self.scene().addEllipse(p[0]-1, p[1]-1, 2, 2, _pen('#ffb04f', 2)))

    def zoom_sample(self, sample):
        points = sample_polygon(sample)
        low, high = points.min(axis=0)-12, points.max(axis=0)+12
        self.fitInView(QRectF(*low, *(high-low)), Qt.AspectRatioMode.KeepAspectRatio)

    def wheelEvent(self, event):
        factor = 1.2 if event.angleDelta().y()>0 else 1/1.2
        if .02 <= self.transform().m11()*factor <= 100:self.scale(factor, factor)
        event.accept()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
                self.pan_origin = event.position(); self.setCursor(Qt.CursorShape.ClosedHandCursor)
            elif self.drawing:
                self.origin = self.mapToScene(event.position().toPoint())
            else:
                self.point_picked.emit(self.mapToScene(event.position().toPoint()))
            event.accept(); return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.pan_origin is not None:
            delta = event.position()-self.pan_origin; self.pan_origin = event.position()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value()-round(delta.x()))
            self.verticalScrollBar().setValue(self.verticalScrollBar().value()-round(delta.y()))
            event.accept(); return
        if self.origin is not None:
            end = self.mapToScene(event.position().toPoint())
            if self.items:
                for item in self.items:self.scene().removeItem(item)
            self.items = [self.scene().addLine(self.origin.x(), self.origin.y(), end.x(), end.y(), _pen('#00d6df', 2))]
            event.accept(); return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            if self.pan_origin is not None:
                self.pan_origin = None; self.setCursor(Qt.CursorShape.CrossCursor)
            elif self.origin is not None:
                start = self.origin; self.origin = None
                self.line_drawn.emit(start, self.mapToScene(event.position().toPoint()))
            event.accept(); return
        super().mouseReleaseEvent(event)


class BrightnessProfile(QWidget):
    position_picked = Signal(float)

    def __init__(self, parent=None):
        super().__init__(parent); self.sample = None
        self.setMinimumSize(330, 260); self.setMouseTracking(True)
        self.setAccessibleName('밝기 변화 그래프 · 클릭하여 경계 지점 선택')

    def plot_rect(self):return QRectF(52, 30, max(10, self.width()-74), self.height()*.48)

    def set_sample(self, sample):self.sample = sample; self.update()

    def paintEvent(self, event):
        p = QPainter(self); p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor('#18212b')); p.setPen(QColor('#d8e2ee'))
        p.drawText(12, 19, '평균 밝기 (원본값)'); rect = self.plot_rect()
        if self.sample is None:
            p.drawText(rect, Qt.AlignmentFlag.AlignCenter, '이미지에서 경계를 가로질러 드래그하세요.'); return
        s = self.sample; d = np.asarray(s['distances_px']); values = np.asarray(s['intensities'])
        lo, hi = float(values.min()), float(values.max()); span = max(hi-lo, 1e-6)
        xs = rect.left()+d/d[-1]*rect.width()
        gradient = np.gradient(values, d)
        lower = QRectF(rect.left(), rect.bottom()+43, rect.width(), max(30, self.height()-rect.bottom()-75))
        p.drawText(12, round(lower.top()-10), '밝기 변화율 / px')
        for box, levels, bottom, top, color in ((rect, values, lo, hi, '#5edcea'),
                (lower, gradient, -max(abs(gradient)), max(abs(gradient)), '#adacd7')):
            p.setPen(_pen('#45515d', 1)); p.drawRect(box)
            ys = box.bottom()-(levels-bottom)/max(top-bottom, 1e-6)*box.height()
            path = QPainterPath(QPointF(xs[0], ys[0]))
            for x, y in zip(xs[1:], ys[1:]):path.lineTo(x, y)
            p.setPen(_pen(color, 1.8)); p.drawPath(path)
            p.setPen(QColor('#b5c3d3')); p.drawText(2, round(box.top()+9), f'{top:.3g}')
            p.drawText(2, round(box.bottom()), f'{bottom:.3g}')
        x = rect.left()+s['anchor_px']/d[-1]*rect.width()
        level = float(np.interp(s['anchor_px'], d, values))
        y = rect.bottom()-(level-lo)/span*rect.height()
        p.setPen(_pen('#ffb04f', 1.5)); p.drawLine(QPointF(x, rect.top()), QPointF(x, lower.bottom()))
        p.setBrush(QColor('#ffb04f')); p.drawEllipse(QPointF(x, y), 4, 4)
        p.setPen(QColor('#d8e2ee')); p.drawText(round(rect.left()), self.height()-8, 'A · 0 px')
        p.drawText(round(rect.right()-80), self.height()-8, f'B · {d[-1]:.1f} px')

    def mousePressEvent(self, event):
        if self.sample is not None and event.button() == Qt.MouseButton.LeftButton:
            rect = self.plot_rect()
            fraction = np.clip((event.position().x()-rect.left())/rect.width(), 0, 1)
            self.position_picked.emit(float(fraction*self.sample['distances_px'][-1]))


class BoundarySampleDialog(QDialog):
    sample_changed = Signal()

    def __init__(self, image, sample=None, guides=(), parent=None, *, embedded=False):
        super().__init__(parent); self.image = image; self.sample = None; self.line = None
        self.embedded = embedded
        self.sample_error = False
        if embedded:self.setWindowFlags(Qt.WindowType.Widget)
        self.setWindowTitle('인식지점 지정 · 이미지와 밝기 변화'); self.resize(1120, 740)
        root = QVBoxLayout(self)
        note = QLabel('① 경계에 수직으로 짧게 드래그 → ② 밝기 그래프 또는 확대 이미지에서 기준점 클릭 → ③ 적용\n'
                      '청록색 띠: 밝기 평균을 내는 샘플 구역 · 주황색: 경계로 사용할 위치 · 휠: 확대 · Ctrl+드래그: 이동')
        note.setWordWrap(True); root.addWidget(note)
        row = QHBoxLayout(); self.draw_button = QPushButton('샘플 구역 다시 그리기'); self.draw_button.setCheckable(True)
        self.draw_button.setChecked(True); row.addWidget(self.draw_button)
        self.band = QSpinBox(); self.band.setRange(1, 41); self.band.setSingleStep(2); self.band.setValue(7)
        self.band.setSuffix(' px'); self.band.setAccessibleName('샘플 띠 폭'); row.addWidget(QLabel('평균 띠 폭')); row.addWidget(self.band)
        fit = QPushButton('전체 보기'); row.addWidget(fit); zoom = QPushButton('샘플 확대'); row.addWidget(zoom)
        self.clear_button = QPushButton('이 샘플 지우기'); row.addWidget(self.clear_button)
        self.clear_button.clicked.connect(self.clear_sample)
        row.addStretch(); root.addLayout(row)
        split = QSplitter(); self.view = SampleImageView(image, guides); split.addWidget(self.view)
        right = QWidget(); right_layout = QVBoxLayout(right); right_layout.setContentsMargins(3, 0, 0, 0)
        right_layout.addWidget(QLabel('샘플 확대 · 클릭으로 기준점 이동'))
        self.detail = SampleImageView(image); self.detail.drawing = False; self.detail.setMinimumHeight(170)
        right_layout.addWidget(self.detail, 1); self.graph = BrightnessProfile(); right_layout.addWidget(self.graph, 1)
        split.addWidget(right); split.setSizes([640, 440]); root.addWidget(split, 1)
        marker_row = QHBoxLayout(); marker_row.addWidget(QLabel('A에서 선택점까지'))
        self.position = QDoubleSpinBox(); self.position.setDecimals(2); self.position.setSingleStep(.1); self.position.setSuffix(' px')
        self.position.setAccessibleName('경계 인식지점 거리'); marker_row.addWidget(self.position)
        self.info = QLabel(''); self.info.setWordWrap(True); marker_row.addWidget(self.info, 1); root.addLayout(marker_row)
        self.status = QLabel('대표적인 경계 하나를 가로질러 8~160 px 길이로 지정하세요.'); self.status.setWordWrap(True); root.addWidget(self.status)
        description = QLabel('선택한 밝기 변화 모양을 곡선의 인식 범위 안에서 찾아 같은 위치를 경계로 사용합니다.\n'
                             '전체 밝기가 달라도 변화 모양을 비교하며, 맞지 않는 구간은 기존 연속 연결 규칙을 따릅니다.')
        description.setWordWrap(True); root.addWidget(description)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Apply | QDialogButtonBox.StandardButton.Cancel)
        self.apply_button = buttons.button(QDialogButtonBox.StandardButton.Apply); self.apply_button.setText('이 지점을 기준으로 적용')
        self.apply_button.setEnabled(False); self.apply_button.clicked.connect(self.accept); buttons.rejected.connect(self.reject); root.addWidget(buttons)
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText('취소')
        if embedded:buttons.hide()
        self.view.line_drawn.connect(self._new_line); self.view.point_picked.connect(self._pick_image)
        self.detail.point_picked.connect(self._pick_image); self.graph.position_picked.connect(self.position.setValue)
        self.position.valueChanged.connect(self._select); self.band.valueChanged.connect(self._resample)
        self.draw_button.toggled.connect(lambda checked: setattr(self.view, 'drawing', checked))
        fit.clicked.connect(self.view.fit_image); zoom.clicked.connect(self._zoom)
        if sample is not None:
            try:
                old = validate_boundary_sample(sample, image.shape)
                self.band.blockSignals(True); self.band.setValue(round(2*old['half_width_px']+1)); self.band.blockSignals(False)
                self.line = (old['start'], old['end']); self._show_profile(old)
            except ValueError as exc:self.sample_error = True; self.status.setText(str(exc))
        QTimer.singleShot(0, self.view, self.view.fit_image)
        # Fit after the dialog/splitter has its visible size, including when a
        # saved sample is restored before the dialog is first shown.
        QTimer.singleShot(0, self, self._fit_detail)

    def keyPressEvent(self, event):
        # Embedded tab pages must not consume Escape/Enter as their own dialog.
        if self.embedded:QWidget.keyPressEvent(self, event)
        else:super().keyPressEvent(event)

    def clear_sample(self):
        self.sample = None; self.line = None; self.sample_error = False
        self.position.blockSignals(True); self.position.setValue(0); self.position.blockSignals(False)
        self.view.show_sample(None); self.detail.show_sample(None); self.graph.set_sample(None)
        self.draw_button.setChecked(True); self.apply_button.setEnabled(False); self.info.clear()
        self.status.setText('샘플 미지정 · 이미지에서 경계를 가로질러 드래그하세요.')
        self.sample_changed.emit()

    def _fit_detail(self):
        if self.sample is None:self.detail.fit_image()
        else:self.detail.zoom_sample(self.sample)

    def _new_line(self, a, b):
        self.line = ([a.x(), a.y()], [b.x(), b.y()]); self._resample()

    def _resample(self, *_args, anchor=None):
        if self.line is None:return
        if anchor is None and self.sample is not None and self.line == (self.sample['start'], self.sample['end']):
            anchor = self.sample['anchor_px']
        try:
            self.sample = make_boundary_sample(self.image, *self.line, (self.band.value()-1)/2, anchor)
        except ValueError as exc:
            self.sample = None; self.graph.set_sample(None); self.detail.show_sample(None)
            self.sample_error = True; self.apply_button.setEnabled(False); self.info.clear(); self.status.setText(str(exc))
            self.sample_changed.emit(); return
        self._show_profile(self.sample)

    def _show_profile(self, sample):
        self.sample = sample
        self.sample_error = False
        self.position.blockSignals(True); self.position.setRange(0, self.sample['distances_px'][-1])
        self.position.setValue(self.sample['anchor_px']); self.position.blockSignals(False)
        self.draw_button.setChecked(False); self.apply_button.setEnabled(True)
        self.status.setText('그래프 또는 확대 이미지에서 주황색 기준점을 선택하세요. 샘플은 현재 이미지에 저장됩니다.')
        self._select(sample['anchor_px']); self.detail.zoom_sample(self.sample)

    def _select(self, distance):
        if self.sample is None:return
        self.sample['anchor_px'] = float(distance)
        point = sample_point(self.sample)
        level = np.interp(distance, self.sample['distances_px'], self.sample['intensities'])
        self.info.setText(f'선택점 X {point[0]:.2f} · Y {point[1]:.2f} px  |  평균 밝기 {level:.2f}\n'
                          f"샘플 A ({self.sample['start'][0]:.1f}, {self.sample['start'][1]:.1f}) → "
                          f"B ({self.sample['end'][0]:.1f}, {self.sample['end'][1]:.1f})")
        self.view.show_sample(self.sample); self.detail.show_sample(self.sample); self.graph.set_sample(self.sample)
        self.sample_changed.emit()

    def _pick_image(self, point):
        if self.sample is None:return
        a, b = np.asarray(self.sample['start']), np.asarray(self.sample['end']); direction = b-a
        distance = np.dot(np.array([point.x(), point.y()])-a, direction)/np.linalg.norm(direction)
        self.position.setValue(float(np.clip(distance, 0, self.sample['distances_px'][-1])))

    def _zoom(self):
        if self.sample is not None:self.view.zoom_sample(self.sample)

    def selected_sample(self):return validate_boundary_sample(self.sample, self.image.shape)


class BoundarySamplesDialog(QDialog):
    """Independent editors prevent tab switches from overwriting the other wall."""
    def __init__(self, image, common=None, samples=None, guides=(), parent=None):
        super().__init__(parent)
        self.setWindowTitle('좌우 인식지점 지정 · 이미지와 밝기 변화'); self.resize(1180, 820)
        root = QVBoxLayout(self)
        note = QLabel('왼쪽·오른쪽 측벽에서 각각 샘플 구역을 그리고 밝기 그래프의 기준점을 선택하세요.\n'
                      '미지정 측벽과 Field·바닥은 공통 샘플을 사용하며, 공통 샘플도 없으면 기존 인식 기준을 사용합니다.')
        note.setWordWrap(True); root.addWidget(note)
        samples = validate_boundary_samples(samples, image.shape)
        self.tabs = QTabWidget(); self.tabs.setObjectName('boundary_sample_sides'); root.addWidget(self.tabs, 1)
        self.editors = {}; self.labels = {'left':'왼쪽 측벽', 'right':'오른쪽 측벽', 'common':'공통 · 기존 설정'}
        for side, label in self.labels.items():
            editor = BoundarySampleDialog(image, common if side=='common' else samples.get(side), guides, self, embedded=True)
            editor.setObjectName('boundary_sample_'+side)
            editor.sample_changed.connect(self._refresh)
            self.editors[side] = editor; self.tabs.addTab(editor, label)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Apply | QDialogButtonBox.StandardButton.Cancel)
        self.apply_button = buttons.button(QDialogButtonBox.StandardButton.Apply)
        self.apply_button.setText('좌우 설정 적용'); self.apply_button.clicked.connect(self.accept)
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText('취소'); buttons.rejected.connect(self.reject)
        root.addWidget(buttons)
        self.tabs.currentChanged.connect(self._fit_tab)
        self._refresh()
        QTimer.singleShot(0, self, self._fit_tab)

    def showEvent(self, event):
        super().showEvent(event)
        available = self.screen().availableGeometry()
        self.resize(min(1180, available.width()-30), min(820, available.height()-45))
        QTimer.singleShot(0, self, self._fit_tab)

    def _refresh(self):
        self.apply_button.setEnabled(not any(e.sample_error for e in self.editors.values()))
        for i, (side, editor) in enumerate(self.editors.items()):
            state = '확인 필요' if editor.sample_error else '지정됨' if editor.sample is not None else '미지정'
            self.tabs.setTabText(i, self.labels[side]+' · '+state)

    def _fit_tab(self, *_args):
        editor = self.tabs.currentWidget()
        QTimer.singleShot(0, editor.view, editor.view.fit_image); QTimer.singleShot(0, editor, editor._fit_detail)

    def selected_samples(self):
        return {s:e.selected_sample() for s,e in self.editors.items() if s!='common' and e.sample is not None}

    def selected_common_sample(self):
        editor = self.editors['common']
        return editor.selected_sample() if editor.sample is not None else None
