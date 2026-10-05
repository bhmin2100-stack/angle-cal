"""Trench workspace: linked image, navigator, profiles and measurement tables."""
from __future__ import annotations

from dataclasses import asdict
import json
import math
from pathlib import Path

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, QThread, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView, QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QMenu, QToolButton,
    QFormLayout, QGraphicsItem, QGraphicsScene, QGraphicsView, QGroupBox, QHBoxLayout,
    QHeaderView, QLabel, QPushButton, QScrollArea, QSpinBox, QSplitter,
    QSizePolicy, QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget,
)

from .image_ops import bgr_to_rgb8_for_display
from .trench_analysis import TrenchOptions, TrenchResult, analyze_trench, export_trench_csv, trench_gfe_tsv


def _pen(color: str, width: float = 1.5, dashed: bool = False) -> QPen:
    pen = QPen(QColor(color), width)
    pen.setCosmetic(True)
    if dashed:
        pen.setStyle(Qt.PenStyle.DashLine)
    return pen


class TrenchView(QGraphicsView):
    region_selected = Signal(QRectF)
    viewport_changed = Signal(QRectF)
    navigate = Signal(QPointF)

    def __init__(self, overview: bool = False):
        super().__init__()
        self.setScene(QGraphicsScene(self))
        self.setBackgroundBrush(QColor("#20262e"))
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.overview = overview
        self.select_region = False
        self.origin = None
        self.rubber = None
        self.image_rect = QRectF()
        self.viewport_box = None
        self.setMinimumSize(100, 120)
        self.horizontalScrollBar().valueChanged.connect(self.notify_viewport)
        self.verticalScrollBar().valueChanged.connect(self.notify_viewport)

    def show_image(self, pixmap: QPixmap) -> None:
        self.scene().clear()
        self.rubber = self.viewport_box = None
        self.scene().addPixmap(pixmap)
        self.image_rect = QRectF(pixmap.rect())
        self.scene().setSceneRect(self.image_rect)
        if self.overview:
            self.viewport_box = self.scene().addRect(QRectF(), _pen("#ffc857", 2))
        self.fit_image()

    def fit_image(self) -> None:
        if not self.image_rect.isEmpty():
            self.fitInView(self.image_rect, Qt.AspectRatioMode.KeepAspectRatio)
            self.notify_viewport()

    def notify_viewport(self, *_args) -> None:
        if not self.overview:
            self.viewport_changed.emit(self.mapToScene(self.viewport().rect()).boundingRect())

    def set_viewport_box(self, rect: QRectF) -> None:
        if self.viewport_box is not None:
            self.viewport_box.setRect(rect.intersected(self.image_rect))

    def set_select_region(self, selected: bool) -> None:
        self.select_region = selected
        self.setDragMode(QGraphicsView.DragMode.NoDrag if selected else QGraphicsView.DragMode.ScrollHandDrag)
        self.setCursor(Qt.CursorShape.CrossCursor if selected else Qt.CursorShape.OpenHandCursor)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self.overview:
            self.navigate.emit(self.mapToScene(event.position().toPoint()))
            return
        if event.button() == Qt.MouseButton.LeftButton and self.select_region and not self.image_rect.isEmpty():
            self.origin = self.mapToScene(event.position().toPoint())
            self.rubber = self.scene().addRect(QRectF(self.origin, self.origin), _pen("#ffc857", 2, True))
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self.origin is not None:
            self.rubber.setRect(QRectF(self.origin, self.mapToScene(event.position().toPoint())).normalized().intersected(self.image_rect))
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if self.origin is not None:
            rect = self.rubber.rect()
            self.scene().removeItem(self.rubber)
            self.rubber = self.origin = None
            if rect.width() >= 12 and rect.height() >= 16:
                self.region_selected.emit(rect)
            return
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event) -> None:
        if self.overview:
            event.ignore()
            return
        factor = 1.2 if event.angleDelta().y() > 0 else 1 / 1.2
        if .01 < self.transform().m11() * factor < 40:
            self.scale(factor, factor)
            self.notify_viewport()
        event.accept()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self.overview:
            self.fit_image()
        self.notify_viewport()


def draw_corner(scene, corner, factor=1.0, unit="px"):
    items = []
    if not corner:
        return items
    for key, color, width in (("trace_points", "#86bfc7", 1), ("arc_points", "#df91ff", 2)):
        points = corner.get(key, [])
        if points:
            path = QPainterPath(QPointF(*points[0]))
            for point in points[1:]:
                path.lineTo(QPointF(*point))
            items.append(scene.addPath(path, _pen(color, width)))
    for x, y in corner.get("fit_points", []):
        items.append(scene.addEllipse(x - .4, y - .4, .8, .8, _pen("#ffe59a", 1)))
    if corner.get("radius_px") is not None:
        cx, cy = corner["center"]
        px, py = corner["arc_points"][len(corner["arc_points"]) // 2]
        items.append(scene.addLine(cx, cy, px, py, _pen("#df91ff", 1)))
        items.append(scene.addLine(cx - 2, cy, cx + 2, cy, _pen("#df91ff", 1)))
        items.append(scene.addLine(cx, cy - 2, cx, cy + 2, _pen("#df91ff", 1)))
        label = scene.addSimpleText(f"{'L' if corner['side'] == 'left' else 'R'} R {corner['radius_px'] * factor:.4g} {unit}")
        label.setBrush(QColor("#df91ff"))
        label.setPos(cx + 4, cy + 4)
        label.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations)
        items.append(label)
    return items


class CornerView(TrenchView):
    endpoint_selected = Signal(QPointF)

    def __init__(self):
        super().__init__()
        self.pick_endpoint = None

    def mousePressEvent(self, event):
        if self.pick_endpoint and event.button() == Qt.MouseButton.LeftButton:
            self.endpoint_selected.emit(self.mapToScene(event.position().toPoint()))
            return
        super().mousePressEvent(event)


class CornerEditor(QGroupBox):
    changed = Signal(object, object)

    def __init__(self, title):
        super().__init__(title)
        self.roi_override = None
        self.corner = None
        self.items = []
        self.focus_rect = QRectF()
        layout = QVBoxLayout(self)
        self.label = QLabel("분석 후 입구 윤곽과 대표 R이 표시됩니다.")
        self.label.setWordWrap(True)
        self.label.setMinimumHeight(60)
        self.label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        layout.addWidget(self.label)
        self.view = CornerView()
        layout.addWidget(self.view, 1)
        row = QHBoxLayout()
        self.region_button = QPushButton("검색 영역 지정")
        self.region_button.setCheckable(True)
        self.region_button.toggled.connect(self._region_mode)
        self.reset_button = QPushButton("자동 초기화")
        self.reset_button.clicked.connect(self.reset_selection)
        self.fit_button = QPushButton("확대 맞춤")
        self.fit_button.clicked.connect(self.fit_corner)
        for button in (self.region_button, self.fit_button, self.reset_button):
            row.addWidget(button)
        layout.addLayout(row)
        row = QHBoxLayout()
        self.start_spin, self.end_spin = QDoubleSpinBox(), QDoubleSpinBox()
        for title, spin in (("시작", self.start_spin), ("끝", self.end_spin)):
            spin.setRange(0, 100)
            spin.setDecimals(1)
            spin.setSuffix(" %")
            row.addWidget(QLabel(title))
            row.addWidget(spin)
            spin.valueChanged.connect(self._preview_endpoints)
        self.end_spin.setValue(100)
        self.apply_button = QPushButton("구간 적용")
        self.apply_button.clicked.connect(self.apply_selection)
        row.addWidget(self.apply_button)
        layout.addLayout(row)
        row = QHBoxLayout()
        for title, endpoint in (("시작점 찍기", "start"), ("끝점 찍기", "end")):
            button = QPushButton(title)
            button.clicked.connect(lambda _checked=False, name=endpoint: self._pick_mode(name))
            row.addWidget(button)
        layout.addLayout(row)
        self.view.region_selected.connect(self._region_selected)
        self.view.endpoint_selected.connect(self._point_selected)

    def set_image(self, pixmap):
        self.items = []
        self.corner = None
        self.focus_rect = QRectF()
        self.view.pick_endpoint = None
        self.view.origin = None
        self.region_button.setChecked(False)
        self.view.set_select_region(False)
        self.view.show_image(pixmap)

    def restore(self, roi=None, span=None):
        self.roi_override = roi
        self.start_spin.setValue((span or (0, 1))[0] * 100)
        self.end_spin.setValue((span or (0, 1))[1] * 100)

    def _region_mode(self, enabled):
        self.view.pick_endpoint = None
        self.view.set_select_region(enabled)

    def _pick_mode(self, endpoint):
        self.region_button.setChecked(False)
        self.view.pick_endpoint = endpoint
        self.view.setCursor(Qt.CursorShape.CrossCursor)

    def _point_selected(self, point):
        if self.corner and self.corner.get("trace_points"):
            points = np.asarray(self.corner["trace_points"])
            index = int(np.argmin(np.sum((points - [point.x(), point.y()]) ** 2, axis=1)))
            spin = self.start_spin if self.view.pick_endpoint == "start" else self.end_spin
            spin.setValue(100 * index / max(1, len(points) - 1))
        self.view.pick_endpoint = None
        self.view.setCursor(Qt.CursorShape.OpenHandCursor)

    def _region_selected(self, rect):
        self.roi_override = tuple(int(v) for v in (rect.x(), rect.y(), rect.width(), rect.height()))
        self.region_button.setChecked(False)
        self.restore(self.roi_override)
        self.changed.emit(self.roi_override, None)

    def apply_selection(self):
        if self.start_spin.value() >= self.end_spin.value():
            self.label.setText("시작은 끝보다 앞에 있어야 합니다.")
            return
        self.changed.emit(self.roi_override, (self.start_spin.value() / 100, self.end_spin.value() / 100))

    def reset_selection(self):
        self.region_button.setChecked(False)
        self.view.pick_endpoint = None
        self.view.set_select_region(False)
        self.restore()
        self.changed.emit(None, None)

    def fit_corner(self):
        if not self.focus_rect.isEmpty():
            self.view.fitInView(self.focus_rect, Qt.AspectRatioMode.KeepAspectRatio)

    def show_result(self, corner, factor=1, unit="px"):
        self.corner = corner
        for item in self.items:
            self.view.scene().removeItem(item)
        self.items = []
        if corner is None:
            self.focus_rect = QRectF()
            self.view.pick_endpoint = None
            self.view.setCursor(Qt.CursorShape.OpenHandCursor)
            self.label.setText("설정 후 분석을 실행하세요.")
            return
        roi = QRectF(*corner["roi"])
        self.items = draw_corner(self.view.scene(), corner, factor, unit)
        self.items.append(self.view.scene().addRect(roi, _pen("#ffc857", 1, True)))
        trace = corner.get("trace_points", [])
        if trace:
            points = np.asarray(trace)
            minimum, maximum = points.min(axis=0), points.max(axis=0)
            self.focus_rect = QRectF(QPointF(*minimum), QPointF(*maximum)).adjusted(-15, -15, 15, 15)
        else:
            self.focus_rect = roi
        radius, error = corner.get("radius_px"), corner.get("rms_px")
        text = "R 미확정" if radius is None else f"대표 R  {radius * factor:.4g} {unit}"
        if error is not None:
            text += f"  ·  맞춤 RMS {error * factor:.3g} {unit}"
        self.label.setText(text + "\n" + corner["reason"])
        self.label.setToolTip("RMS는 윤곽과 원호의 맞춤 오차이며 실제 치수의 정확도 또는 신뢰 확률이 아닙니다.")
        self._preview_endpoints()
        self.fit_corner()

    def _preview_endpoints(self, *_args):
        for item in list(self.items):
            if item.data(0) == "endpoint":
                self.view.scene().removeItem(item)
                self.items.remove(item)
        if self.corner and self.corner.get("trace_points"):
            points = self.corner["trace_points"]
            for spin, color in ((self.start_spin, "#56efb2"), (self.end_spin, "#ff75a6")):
                x, y = points[round(spin.value() * (len(points) - 1) / 100)]
                item = self.view.scene().addEllipse(x - 1, y - 1, 2, 2, _pen(color, 2))
                item.setData(0, "endpoint")
                self.items.append(item)


def draw_field_angle(scene, item, factor=1., unit="px"):
    drawn = []
    points = np.asarray(item.get("points") or [])
    if len(points):
        color="#63ef90" if item["status"]=="valid" else "#f7a03d"
        path = QPainterPath(QPointF(*points[0]))
        for point in points[1:]:
            path.lineTo(QPointF(*point))
        drawn.append(scene.addPath(path, _pen(color, 2)))
        m, c = item["line"]
        drawn.append(scene.addLine(points[0,0], m*points[0,0]+c, points[-1,0], m*points[-1,0]+c, _pen(color, 1, True)))
        prefix="Field" if item["status"]=="valid" else "Field 후보"
        text = scene.addSimpleText(f"{prefix} RMS {item['rms_px'] * factor:.3g} {unit}")
        text.setBrush(QColor(color))
        text.setPos(points[0,0], points[:,1].min()-24)
        text.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True)
        drawn.append(text)
    wall = np.asarray(item.get("wall_points") or [])
    for x, y in wall[::max(1,len(wall)//24)]:
        drawn.append(scene.addEllipse(x-1,y-1,2,2,_pen("#fff275",1)))
    if item.get("angle_status") == "valid":
        connection=item.get('connection',{})
        if connection.get('status')=='valid':
            ft=connection['field_tangent'];wt=connection['wall_tangent']
            field_start=points[0] if item['side']=='left' else points[-1]
            drawn.append(scene.addLine(*field_start,*ft,_pen('#63ef90',2)))
            path=QPainterPath(QPointF(*ft))
            for p in connection['arc_points'][1:]:path.lineTo(QPointF(*p))
            drawn.append(scene.addPath(path,_pen('#df91ff',3)))
            wm,wc=item['wall_line'];end_y=wall[-1,1]
            drawn.append(scene.addLine(*wt,wm*end_y+wc,end_y,_pen('#fff275',2)))
            center=connection['center'];mid=connection['arc_points'][len(connection['arc_points'])//2]
            drawn.append(scene.addLine(*center,*mid,_pen('#df91ff',1,True)))
            drawn.append(scene.addEllipse(center[0]-1.5,center[1]-1.5,3,3,_pen('#df91ff')))
            label=scene.addSimpleText(f"R {connection['radius_px']*factor:.3g} {unit} · {item['angle_deg']:.2f}°")
            label.setBrush(QColor('#df91ff'));label.setPos(mid[0]+8,mid[1]-22)
            label.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations,True)
            drawn.append(label)
            return drawn
        vx, vy = item["vertex"]
        wm, wc = item["wall_line"]
        end_y = wall[-1,1]
        drawn.append(scene.addLine(vx,vy,wm*end_y+wc,end_y,_pen("#fff275",2,True)))
        sign = -1 if item["side"] == "left" else 1
        fm, _ = item["line"]
        a = math.atan2(sign*fm,sign)
        b = math.atan2(1,wm)
        delta = (b-a+math.pi)%(2*math.pi)-math.pi
        length = 40
        drawn.append(scene.addLine(vx,vy,vx+sign*length,vy+sign*length*fm,_pen("#63ef90",2,True)))
        arc = QPainterPath()
        for i,t in enumerate(np.linspace(a,a+delta,40)):
            point=QPointF(vx+18*math.cos(t),vy+18*math.sin(t))
            arc.moveTo(point) if i==0 else arc.lineTo(point)
        # Physical entrance arcs are drawn only for validated tangent fits.
        drawn.append(scene.addEllipse(vx-2,vy-2,4,4,_pen("#ff75cf",1)))
        text=scene.addSimpleText(f"{item['angle_deg']:.2f}° · R 미확정")
        text.setBrush(QColor("#ff75cf")); text.setPos(vx+sign*22,vy+18)
        text.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations,True)
        drawn.append(text)
    return drawn


class SurfaceEditor(QWidget):
    def __init__(self, title):
        super().__init__()
        layout=QVBoxLayout(self)
        layout.addWidget(QLabel(title + " · 초록 Field / 노랑 측벽 / 분홍 각도"))
        self.label=QLabel("분석 후 검출 Field와 각도 구간을 확인하세요.")
        self.label.setWordWrap(True); layout.addWidget(self.label)
        self.view=TrenchView();layout.addWidget(self.view,1)
        self.items=[]
        self.focus_rect=None
        button=QPushButton("확대 맞춤");button.clicked.connect(self.fit_result);layout.addWidget(button)

    def set_image(self,pixmap):
        self.items=[];self.focus_rect=None;self.view.show_image(pixmap)

    def fit_result(self):
        if self.focus_rect is not None:
            self.view.fitInView(self.focus_rect,Qt.AspectRatioMode.KeepAspectRatio)

    def show_result(self,item=None,factor=1.,unit="px"):
        for obj in self.items:
            self.view.scene().removeItem(obj)
        self.items=[]
        if item is None:
            self.focus_rect=None
            self.label.setText("분석 후 검출 Field와 각도 구간을 확인하세요.");return
        field = "Field 미확정" if item['status']!='valid' else f"Field 지지 길이 {item['length_px']*factor:.4g} {unit} · 기울기 {item['tilt_deg']:.2f}° · RMS {item['rms_px']*factor:.3g} {unit}"
        angle = "각도 미확정" if item['angle_deg'] is None else f"입구 내각 {item['angle_deg']:.2f}° · 측벽 RMS {item['wall_rms_px']*factor:.3g} {unit}"
        connection=item.get('connection',{})
        rounding=(f"접선 R {connection['radius_px']*factor:.4g} {unit} · 원호 RMS {connection['rms_px']*factor:.3g} {unit}"
                  if connection.get('status')=='valid' else '입구 연결 원호 미확정')
        self.label.setText(field+"\n"+item['reason']+"\n"+angle+"\n"+item['angle_reason']+"\n"+rounding+" · "+connection.get('reason',''))
        self.items=draw_field_angle(self.view.scene(),item,factor,unit)
        points=item.get('points',[])+item.get('wall_points',[])+connection.get('arc_points',[])
        if points:
            minimum,maximum=np.min(points,axis=0),np.max(points,axis=0)
            self.focus_rect=QRectF(QPointF(*minimum),QPointF(*maximum)).adjusted(-25,-35,25,25)
            self.fit_result()


class TrenchGraph(QWidget):
    depth_selected = Signal(float)

    def __init__(self):
        super().__init__()
        self.result = None
        self.factor = 1.0
        self.unit = "px"
        self.mode = "shape"
        self.selected_y = None
        self.plot_rect = QRectF()
        self.setMinimumSize(230, 160)

    def mousePressEvent(self, event) -> None:
        if self.result is not None and self.plot_rect.contains(event.position()):
            t = (event.position().y() - self.plot_rect.top()) / self.plot_rect.height()
            low,high=getattr(self,"plot_y_bounds",(self.result.y[0],self.result.y[-1]))
            y = low + t * (high-low)
            self.depth_selected.emit(float(y))

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#f7f9fc"))
        painter.setPen(QColor("#24364c"))
        if self.result is None:
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "분석 후 깊이별 그래프가 표시됩니다")
            return
        r = self.result
        rect = QRectF(60, 28, max(30, self.width() - 85), max(30, self.height() - 60))
        self.plot_rect = rect
        if self.mode == "shape":
            series = [(r.left, r.y, "#007d92"), (r.right, r.y, "#bb4b1e")]
            for item in r.field_angles:
                if item["status"] == "valid":
                    points=np.asarray(item["points"])
                    series.append((points[:,0],points[:,1],"#268b48"))
            title = "좌우 윤곽 X"
        elif self.mode == "bow":
            series = [(r.left_reference - r.left, r.y, "#007d92"), (r.right - r.right_reference, r.y, "#bb4b1e")]
            title = "좌 / 우 벌어짐 (+ 바깥)"
        else:
            series = [(r.right - r.left, r.y, "#285fe1")]
            title = "CD"
        all_x = np.concatenate([values for values, _, _ in series])
        xmin, xmax = float(np.min(all_x)), float(np.max(all_x))
        padding = max(1, (xmax - xmin) * .1)
        xmin, xmax = xmin - padding, xmax + padding
        ymin = float(min(np.min(ys) for _,ys,_ in series))
        ymax = float(r.y[-1])
        self.plot_y_bounds=(ymin,ymax)
        def point(x, y):
            return QPointF(rect.left() + (x - xmin) / (xmax - xmin) * rect.width(),
                           rect.top() + (y - ymin) / max(1, ymax - ymin) * rect.height())
        for i in range(5):
            t = i / 4
            y = ymin + t * (ymax - ymin)
            py = point(xmin, y).y()
            painter.setPen(QColor("#d9e1eb"))
            painter.drawLine(QPointF(rect.left(), py), QPointF(rect.right(), py))
            painter.setPen(QColor("#425368"))
            painter.drawText(QRectF(0, py - 10, 55, 20), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                             f"{(y - r.options.roi[1]) * self.factor:.3g}")
        for item in r.local_bowing:
            a, b = point(xmin, item["start_y"]).y(), point(xmax, item["negative_end_y"]).y()
            painter.fillRect(QRectF(rect.left(), a, rect.width(), max(1, b - a)), QColor(255, 171, 64, 22))
        if self.mode == "shape":
            # Both walls share one X axis, so their separation is the measured
            # CD. Do not normalize or center each wall independently.
            cavity = QPainterPath()
            cavity.moveTo(point(r.left[0], r.y[0]))
            for x, y in zip(r.left[1:], r.y[1:]):
                cavity.lineTo(point(x, y))
            for x, y in zip(r.right[::-1], r.y[::-1]):
                cavity.lineTo(point(x, y))
            cavity.closeSubpath()
            painter.fillPath(cavity, QColor(0, 125, 146, 25))
        for values, ys, color in series:
            path = QPainterPath()
            stride = max(1, len(ys) // 2000)
            for j, index in enumerate(range(0, len(ys), stride)):
                if j == 0:
                    path.moveTo(point(values[index], ys[index]))
                else:
                    path.lineTo(point(values[index], ys[index]))
            painter.setPen(QPen(QColor(color), 2))
            painter.drawPath(path)
        if self.mode == "shape":
            for item in r.field_angles:
                if item["angle_status"] == "valid":
                    vx,vy=item["vertex"]
                    wm,wc=item["wall_line"]
                    end_y=item["wall_points"][-1][1]
                    connection=item.get('connection',{})
                    if connection.get('status')=='valid':
                        path=QPainterPath(point(*connection['arc_points'][0]))
                        for p in connection['arc_points'][1:]:path.lineTo(point(*p))
                        painter.setPen(_pen('#8f45a4',2))
                        painter.drawPath(path)
                    painter.setPen(_pen("#8f45a4",1,True))
                    painter.drawLine(point(vx,vy),point(wm*end_y+wc,end_y))
                    painter.drawText(QRectF(point(vx,vy).x()-35,rect.top()+19,70,20),
                                     Qt.AlignmentFlag.AlignCenter,f"{item['angle_deg']:.2f}°")
            y = float(np.clip(self.selected_y if self.selected_y is not None else
                              (ymin + ymax) / 2, ymin, ymax))
            left = float(np.interp(y, r.y, r.left))
            right = float(np.interp(y, r.y, r.right))
            a, b = point(left, y), point(right, y)
            painter.setPen(_pen("#c13d85", 1.5))
            painter.drawLine(a, b)
            for p in (a, b):
                painter.drawLine(p + QPointF(0, -4), p + QPointF(0, 4))
            painter.drawText(QRectF(rect.left(), max(rect.top(), a.y() - 23), rect.width(), 20),
                             Qt.AlignmentFlag.AlignCenter, f"CD {(right - left) * self.factor:.4g} {self.unit}")
            painter.setPen(QColor("#007d92"))
            painter.drawText(QRectF(rect.left(), rect.top(), rect.width() / 2, 20),
                             Qt.AlignmentFlag.AlignLeft, "왼쪽 벽")
            painter.setPen(QColor("#bb4b1e"))
            painter.drawText(QRectF(rect.center().x(), rect.top(), rect.width() / 2, 20),
                             Qt.AlignmentFlag.AlignRight, "오른쪽 벽")
        if self.selected_y is not None:
            painter.setPen(_pen("#c13d85", 1, True))
            painter.drawLine(point(xmin, self.selected_y), point(xmax, self.selected_y))
        painter.setPen(QColor("#24364c"))
        painter.drawText(QRectF(5, 2, self.width() - 10, 23), Qt.AlignmentFlag.AlignCenter, f"{title} ({self.unit}) · 세로: 깊이 ({self.unit})")
        painter.drawText(QRectF(rect.left(), rect.bottom() + 5, rect.width(), 22), Qt.AlignmentFlag.AlignLeft,
                         f"{xmin * self.factor:.3g}")
        painter.drawText(QRectF(rect.left(), rect.bottom() + 5, rect.width(), 22), Qt.AlignmentFlag.AlignRight,
                         f"{xmax * self.factor:.3g}")


class TrenchWorker(QThread):
    completed = Signal(object, str, int)

    def __init__(self, image, options, revision, parent):
        super().__init__(parent)
        self.image, self.options, self.revision = image, options, revision

    def run(self):
        try:
            self.completed.emit(analyze_trench(self.image, self.options), "", self.revision)
        except Exception as exc:
            self.completed.emit(None, str(exc), self.revision)


class TrenchPanel(QWidget):
    state_changed = Signal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.image = None
        self.image_path = ""
        self.source_key = None
        self.source_calibration = None
        self.roi = None
        self.result = None
        self.worker = None
        self.revision = 0
        self.loading = False
        self.corner_rois = [None, None]
        self.corner_spans = [None, None]
        self.reanalysis_requested = False
        self.overlays = []
        self.selected_line = None
        self.selected_band = None
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 4, 6, 4)
        self.hint = QLabel("① 기존 화면에서 수평 정렬·스케일 보정 → ② 입구 높이부터 한 Trench를 드래그 → ③ 분석")
        self.hint.setWordWrap(True)
        root.addWidget(self.hint)
        body = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(body, 1)
        left = QSplitter(Qt.Orientation.Vertical)
        body.addWidget(left)
        self.view = TrenchView()
        self.image_tabs = QTabWidget()
        self.image_tabs.addTab(self.view, "Trench 이미지")
        self.corner_editors = [CornerEditor("왼쪽 입구"), CornerEditor("오른쪽 입구")]
        corner_page = QWidget()
        corner_layout = QHBoxLayout(corner_page)
        corner_layout.setContentsMargins(0, 0, 0, 0)
        for index, editor in enumerate(self.corner_editors):
            corner_layout.addWidget(editor)
            editor.changed.connect(lambda roi, span, i=index: self._corner_changed(i, roi, span))
        self.image_tabs.addTab(corner_page, "입구 곡률")
        self.surface_editors=[SurfaceEditor("왼쪽 Field / 각도"),SurfaceEditor("오른쪽 Field / 각도")]
        surface_page=QWidget();surface_layout=QHBoxLayout(surface_page)
        for editor in self.surface_editors:
            surface_layout.addWidget(editor)
        self.image_tabs.addTab(surface_page,"Field / 각도")
        self.image_tabs.currentChanged.connect(lambda _index:[editor.fit_result() for editor in self.surface_editors])
        self.image_tabs.currentChanged.connect(lambda _index: [editor.fit_corner() for editor in self.corner_editors])
        left.addWidget(self.image_tabs)
        lower = QSplitter(Qt.Orientation.Horizontal)
        left.addWidget(lower)
        self.graph = TrenchGraph()
        lower.addWidget(self.graph)
        self.tables = QTabWidget()
        lower.addWidget(self.tables)
        self.cd_table = self._table(["깊이", "CD", "왼쪽 X", "오른쪽 X"])
        self.local_table = self._table(["측벽", "시작 깊이", "Nega 끝", "국부 Bow", "현 대비", "역경사°"])
        self.tables.addTab(self.cd_table, "깊이별 CD")
        self.tables.addTab(self.local_table, "국부 Bowing / Nega slope")
        self.cd_table.cellClicked.connect(self._cd_selected)
        self.local_table.cellClicked.connect(self._local_selected)
        left.setSizes([550, 245])
        lower.setSizes([320, 500])
        sidebar_scroll = QScrollArea()
        sidebar_scroll.setWidgetResizable(True)
        sidebar_scroll.setMinimumWidth(280)
        sidebar_scroll.setMaximumWidth(330)
        sidebar_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        body.addWidget(sidebar_scroll)
        sidebar = QWidget()
        sidebar_scroll.setWidget(sidebar)
        controls = QVBoxLayout(sidebar)
        controls.setContentsMargins(6, 4, 6, 4)
        controls.addWidget(QLabel("전체 위치 · 노란 상자 = 확대 화면"))
        self.overview = TrenchView(overview=True)
        self.overview.setFixedHeight(180)
        controls.addWidget(self.overview)
        self.view.viewport_changed.connect(self.overview.set_viewport_box)
        self.overview.navigate.connect(self._navigate)
        self.view.region_selected.connect(self.set_roi)
        self.graph.depth_selected.connect(self.select_depth)
        actions = QHBoxLayout()
        self.roi_button = QPushButton("영역 지정")
        self.roi_button.setCheckable(True)
        self.roi_button.toggled.connect(self.view.set_select_region)
        actions.addWidget(self.roi_button)
        fit = QPushButton("전체 보기")
        fit.clicked.connect(self.view.fit_image)
        actions.addWidget(fit)
        controls.addLayout(actions)
        form = QFormLayout()
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        controls.addLayout(form)
        self.roi_label = QLabel("미지정")
        self.roi_label.setWordWrap(True)
        form.addRow("분석 영역", self.roi_label)
        self.scale_spin = QDoubleSpinBox()
        self.scale_spin.setRange(0, 1e9)
        self.scale_spin.setDecimals(8)
        self.scale_spin.setSpecialValueText("미보정 (px)")
        self.scale_spin.setToolTip("기존 스케일 보정값을 가져옵니다. 0이면 물리 길이를 계산하지 않습니다.")
        form.addRow("nm / px", self.scale_spin)
        self.unit_combo = QComboBox()
        for title, data in (("px", "px"), ("nm", "nm"), ("Å", "angstrom")):
            self.unit_combo.addItem(title, data)
        form.addRow("표시·간격 단위", self.unit_combo)
        self.step_spin = QDoubleSpinBox()
        self.step_spin.setRange(.001, 1e7)
        self.step_spin.setDecimals(3)
        self.step_spin.setValue(20)
        form.addRow("CD 깊이 간격", self.step_spin)
        self.threshold_spin = QSpinBox()
        self.threshold_spin.setRange(0, 254)
        self.threshold_spin.setSpecialValueText("자동")
        self.threshold_spin.setToolTip("Trench 후보 영역을 찾는 상대 명암값 (1~254). 밝은 테두리는 연결된 첫 밝기 정점으로 보정하며, 정점이 없으면 명암 경계를 사용합니다.")
        form.addRow("경계값", self.threshold_spin)
        self.bright_check = QCheckBox("밝은 Trench")
        form.addRow("명암", self.bright_check)
        self.smooth_spin = QSpinBox()
        self.smooth_spin.setRange(1, 101)
        self.smooth_spin.setSingleStep(2)
        self.smooth_spin.setValue(7)
        form.addRow("평활 범위 (px)", self.smooth_spin)
        self.bow_combo = QComboBox()
        self.bow_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.bow_combo.setMinimumContentsLength(10)
        self.bow_combo.addItem("상단 검출점 수직선", "entrance")
        self.bow_combo.addItem("상단–90% 깊이 연결선", "chord")
        form.addRow("전체 Bow 기준", self.bow_combo)
        self.negative_spin = QDoubleSpinBox()
        self.negative_spin.setRange(0, 45)
        self.negative_spin.setValue(1)
        self.negative_spin.setSuffix(" °")
        self.negative_spin.setToolTip("수직선 대비 바깥 방향 역경사. 미세 잡음을 제외할 최소 각도입니다.")
        form.addRow("Nega 최소 각도", self.negative_spin)
        self.corner_spin = QSpinBox()
        self.corner_spin.setRange(6, 200)
        self.corner_spin.setValue(24)
        self.corner_spin.setToolTip("자동 입구 검색의 기준 크기입니다. 각도·CD·Bowing 계산에는 영향을 주지 않습니다.")
        form.addRow("입구 검색 크기 px", self.corner_spin)
        self.field_spin=QSpinBox();self.field_spin.setRange(4,300);self.field_spin.setValue(32)
        self.field_spin.setToolTip("ROI 입구 높이 위아래에서 실제 Field를 찾는 범위입니다. 곡률 검색과 별개입니다.")
        form.addRow("Field 검색 ±px",self.field_spin)
        self.angle_start=QSpinBox();self.angle_start.setRange(0,10000);self.angle_start.setValue(40)
        self.angle_end=QSpinBox();self.angle_end.setRange(1,10000);self.angle_end.setValue(100)
        form.addRow("각도 측벽 시작 px",self.angle_start)
        form.addRow("각도 측벽 끝 px",self.angle_end)
        self.analyze_button = QPushButton("Trench 분석")
        self.analyze_button.setMinimumHeight(32)
        self.analyze_button.clicked.connect(self.start_analysis)
        controls.addWidget(self.analyze_button)
        self.summary_label = QLabel("분석 대기")
        self.summary_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.summary_label.setWordWrap(True)
        self.summary_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        controls.addWidget(self.summary_label)
        self.graph_combo = QComboBox()
        for text, data in (("CD / 깊이", "cd"), ("통합 윤곽", "shape"), ("좌우 Bowing", "bow")):
            self.graph_combo.addItem(text, data)
        self.graph_combo.currentIndexChanged.connect(self._graph_changed)
        self.graph_combo.setCurrentIndex(1)  # Show both walls by default.
        controls.addWidget(self.graph_combo)
        self.export_button = QToolButton()
        self.export_button.setText("Export · GFE 좌표 복사")
        self.export_button.setPopupMode(QToolButton.ToolButtonPopupMode.MenuButtonPopup)
        self.export_button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.export_button.setToolTip("클릭: 검출 Field와 좌우 측벽을 GFE X/Y 표로 복사 (Å)\n화살표: CSV / JSON / PNG 저장\nY 기준 = 좌우 Field의 입구 높이 평균, X = 입구 중심\n윤곽 간격은 직선 연결이며 Field가 미확정이면 복사하지 않습니다.")
        export_menu = QMenu(self.export_button)
        export_menu.addAction("GFE 좌표 복사 (클립보드)", self.copy_gfe_coordinates)
        export_menu.addAction("파일 저장 (CSV / JSON / PNG)…", self.export_dialog)
        self.export_button.setMenu(export_menu)
        self.export_button.clicked.connect(self.copy_gfe_coordinates)
        self.export_button.setEnabled(False)
        controls.addWidget(self.export_button)
        definitions = QLabel("측정 기준\n• ROI 위쪽 = Depth 기준 높이\n• Depth = 입구~검출 바닥 (이미지 수직)\n• 전체 Bow = 기준선 대비 최대 외측 변위\n• 국부 Bow = Nega 시작 수직선 대비 벌어짐\n• 현 대비 = 국부 양끝 연결선 대비 휨\n• Nega = 내려갈수록 바깥으로 벌어짐\n• Field = 검출된 실제 재료/공간 경계\n• 각도 = 검출 Field와 선택 측벽의 내각\n• 입구 R = 둥근 전이 구간의 대표 원 맞춤\n• RMS = 윤곽 맞춤 오차 (치수 정확도 아님)\n• 측벽 분석은 검출 깊이의 90%까지\n\n입구 곡률 탭: 노란 점 = 사용 윤곽 · 보라 = 원호\n초록/분홍 = 원호 시작/끝\n그래프·표 클릭으로 해당 깊이를 확인하세요.")
        definitions.setWordWrap(True)
        definitions.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        definitions.setStyleSheet("color: #666; font-size: 11px;")
        controls.addWidget(definitions)
        controls.addStretch()
        body.setSizes([850, 270])
        for spin in (self.scale_spin, self.step_spin, self.threshold_spin, self.smooth_spin, self.negative_spin, self.corner_spin,self.field_spin,self.angle_start,self.angle_end):
            spin.valueChanged.connect(self._settings_changed)
        self.unit_combo.currentIndexChanged.connect(self._settings_changed)
        self.bow_combo.currentIndexChanged.connect(self._settings_changed)
        self.bright_check.toggled.connect(self._settings_changed)

    @staticmethod
    def _table(headers):
        table = QTableWidget(0, len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        table.verticalHeader().hide()
        return table

    def state(self) -> dict:
        return {"roi": self.roi, "nm_per_px": self.scale_spin.value(),
                "unit": self.unit_combo.currentData(), "step": self.step_spin.value(),
                "threshold": self.threshold_spin.value(), "bright": self.bright_check.isChecked(),
                "smooth": self.smooth_spin.value(), "bow_reference": self.bow_combo.currentData(),
                "negative_angle": self.negative_spin.value(), "corner": self.corner_spin.value(),
                "field_search":self.field_spin.value(),"angle_start":self.angle_start.value(),"angle_end":self.angle_end.value(),
                "field_angles":self.result.field_angles if self.result is not None else None,
                "corner_rois": list(self.corner_rois), "corner_spans": list(self.corner_spans),
                "corner_results": self.result.corners if self.result is not None else None}

    def set_image(self, image, path: str, nm_per_px=None, state=None) -> None:
        key = (id(image), path)
        if key == self.source_key:
            if nm_per_px != self.source_calibration:
                self.source_calibration = nm_per_px
                self.scale_spin.setValue(nm_per_px or 0)
            return
        self.loading = True
        self.reanalysis_requested = False
        self.source_key, self.source_calibration = key, nm_per_px
        self.image, self.image_path = image, path or ""
        self.revision += 1
        self.roi = None
        self.corner_rois, self.corner_spans = [None, None], [None, None]
        self.overlays = []
        self.selected_line = self.selected_band = None
        self._clear_result()
        if image is not None:
            rgb = np.ascontiguousarray(bgr_to_rgb8_for_display(image))
            pixmap = QPixmap.fromImage(QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format.Format_RGB888).copy())
            self.view.show_image(pixmap)
            self.overview.show_image(pixmap)
            for editor in self.corner_editors:
                editor.set_image(pixmap)
            for editor in self.surface_editors:
                editor.set_image(pixmap)
            self.view.notify_viewport()
        state = state or {}
        for key_name in ("corner_rois", "corner_spans"):
            values = state.get(key_name)
            if isinstance(values, (list, tuple)) and len(values) == 2:
                setattr(self, key_name, [tuple(value) if value is not None else None for value in values])
        for i, editor in enumerate(self.corner_editors):
            editor.restore(self.corner_rois[i], self.corner_spans[i])
        self.scale_spin.setValue(float(state.get("nm_per_px", nm_per_px or 0)))
        for control, key_name, default in ((self.step_spin, "step", 20), (self.threshold_spin, "threshold", 0),
                                          (self.smooth_spin, "smooth", 7), (self.negative_spin, "negative_angle", 1),
                                          (self.corner_spin, "corner", 24),(self.field_spin,"field_search",32),
                                          (self.angle_start,"angle_start",40),(self.angle_end,"angle_end",100)):
            control.setValue(state.get(key_name, default))
        self.unit_combo.setCurrentIndex(max(0, self.unit_combo.findData(state.get("unit", "nm" if nm_per_px else "px"))))
        self.bow_combo.setCurrentIndex(max(0, self.bow_combo.findData(state.get("bow_reference", "entrance"))))
        self.bright_check.setChecked(bool(state.get("bright", False)))
        if state.get("roi") and image is not None:
            roi = QRectF(*state["roi"]).intersected(self.view.image_rect)
            if roi.width() >= 12 and roi.height() >= 16:
                self.roi = (int(roi.x()), int(roi.y()), int(roi.width()), int(roi.height()))
        self.roi_button.setChecked(self.roi is None)
        self.loading = False
        self._draw_overlay()

    def set_roi(self, rect: QRectF) -> None:
        rect = rect.normalized().intersected(self.view.image_rect)
        self.roi = (int(rect.x()), int(rect.y()), int(rect.width()), int(rect.height()))
        self.roi_button.setChecked(False)
        self.corner_rois, self.corner_spans = [None, None], [None, None]
        for editor in self.corner_editors:
            editor.restore()
        self._settings_changed()

    def _corner_changed(self, index, roi, span):
        self.corner_rois[index], self.corner_spans[index] = roi, span
        self._settings_changed()
        if self.worker is not None and self.worker.isRunning():
            self.reanalysis_requested = True
        else:
            self.start_analysis()

    def _clear_result(self):
        self.result = None
        self.graph.result = None
        self.graph.selected_y = None
        self.graph.update()
        self.cd_table.setRowCount(0)
        self.local_table.setRowCount(0)
        self.export_button.setEnabled(False)
        self.summary_label.setText("설정 후 분석을 실행하세요.")
        for editor in self.corner_editors:
            editor.show_result(None)
        for editor in self.surface_editors:
            editor.show_result(None)

    def _settings_changed(self, *_args):
        if self.loading:
            return
        self.revision += 1
        self._clear_result()
        self._draw_overlay()
        self.state_changed.emit(self.state())

    def _factor(self) -> tuple[float, str]:
        unit = self.unit_combo.currentData()
        if unit == "px":
            return 1.0, "px"
        if self.scale_spin.value() <= 0:
            raise ValueError("nm 또는 Å 단위에는 스케일 보정값(nm/px)이 필요합니다.")
        return self.scale_spin.value() * (10 if unit == "angstrom" else 1), "Å" if unit == "angstrom" else "nm"

    def options(self) -> TrenchOptions:
        if self.roi is None:
            raise ValueError("먼저 입구 높이부터 한 Trench를 포함하는 영역을 지정하세요.")
        factor, _ = self._factor()
        return TrenchOptions(self.roi, self.step_spin.value() / factor, self.threshold_spin.value(),
                             self.bright_check.isChecked(), self.smooth_spin.value(), self.bow_combo.currentData(),
                             self.corner_spin.value(), self.negative_spin.value(),
                             tuple(self.corner_rois), tuple(self.corner_spans),self.field_spin.value(),
                             (self.angle_start.value(),self.angle_end.value()))

    def start_analysis(self):
        if self.worker is not None and self.worker.isRunning():
            return
        try:
            if self.image is None:
                raise ValueError("파일 탭에서 이미지를 먼저 불러오세요.")
            options = self.options()
        except ValueError as exc:
            self.summary_label.setText(str(exc))
            return
        self._clear_result()
        self.summary_label.setText("측벽과 국부 Bowing을 분석하고 있습니다…")
        self.analyze_button.setEnabled(False)
        self.worker = TrenchWorker(self.image, options, self.revision, self)
        self.worker.completed.connect(self._analysis_completed)
        self.worker.finished.connect(lambda worker=self.worker: self._worker_finished(worker))
        self.worker.start()

    def _worker_finished(self, worker):
        if self.worker is worker:
            self.worker = None
        worker.deleteLater()
        self.analyze_button.setEnabled(True)
        if self.reanalysis_requested:
            self.reanalysis_requested = False
            self.start_analysis()

    def wait_for_worker(self):
        self.reanalysis_requested = False
        if self.worker is not None:
            self.worker.wait()

    def _analysis_completed(self, result, error: str, revision: int):
        if revision != self.revision:
            return
        if error:
            self.summary_label.setText(error)
            return
        self.result = result
        self._render_result()
        self.state_changed.emit(self.state())

    def _render_result(self):
        r = self.result
        factor, unit = self._factor()
        fmt = lambda value: "미확정" if value is None else f"{value * factor:.4g} {unit}"
        angle = lambda value: "미확정" if value is None else f"{value:.2f}°"
        summary = r.summary()
        lines = [f"Depth  {fmt(r.depth_px)}", f"관찰 깊이  {fmt(r.observed_depth_px)}",
                 f"전체 Bow 좌 / 우  {fmt(r.left_bowing_px)} / {fmt(r.right_bowing_px)}",
                 f"입구 각도 좌 / 우  {angle(r.left_angle_deg)} / {angle(r.right_angle_deg)}",
                 f"입구 R 좌 / 우  {fmt(summary['left_radius_px'])} / {fmt(summary['right_radius_px'])}",
                 f"국부 Nega 구간 {len(r.local_bowing)}개 · 경계값 {r.threshold:.1f}"]
        if self.step_spin.value() / factor < 1:
            lines.append("1 px 미만 간격은 경계 보간값입니다.")
        if r.warnings:
            lines += ["", *r.warnings]
        self.summary_label.setText("\n".join(lines))
        self.cd_table.setHorizontalHeaderLabels([f"깊이 ({unit})", f"CD ({unit})", "왼쪽 X (px)", "오른쪽 X (px)"])
        self.cd_table.setRowCount(len(r.sample_y))
        for row, (y, l, right) in enumerate(zip(r.sample_y, r.sample_left, r.sample_right)):
            for col, value in enumerate(((y - r.options.roi[1]) * factor, (right - l) * factor, l, right)):
                self.cd_table.setItem(row, col, QTableWidgetItem(f"{value:.5g}"))
        self.local_table.setHorizontalHeaderLabels(["측벽", f"시작 ({unit})", f"Nega 끝 ({unit})", f"국부 Bow ({unit})", f"현 대비 ({unit})", "역경사°"])
        self.local_table.setRowCount(len(r.local_bowing))
        for row, item in enumerate(r.local_bowing):
            values = ["좌" if item["side"] == "left" else "우",
                      f"{(item['start_y'] - r.options.roi[1]) * factor:.4g}",
                      f"{(item['negative_end_y'] - r.options.roi[1]) * factor:.4g}",
                      f"{item['bowing_px'] * factor:.4g}", f"{item['chord_bowing_px'] * factor:.4g}",
                      f"{item['max_negative_angle_deg']:.2f}"]
            for col, value in enumerate(values):
                cell = QTableWidgetItem(value)
                cell.setToolTip("국부 종료/복귀 확인" if item["return_confirmed"] else "분석 범위 끝까지 복귀 미확인")
                self.local_table.setItem(row, col, cell)
        self.graph.result, self.graph.factor, self.graph.unit = r, factor, unit
        self.graph.update()
        self.export_button.setEnabled(True)
        for editor, corner in zip(self.corner_editors, r.corners):
            editor.show_result(corner, factor, unit)
        for editor, item in zip(self.surface_editors,r.field_angles):
            editor.show_result(item,factor,unit)
        self._draw_overlay()

    def _draw_overlay(self):
        scene = self.view.scene()
        for item in self.overlays:
            scene.removeItem(item)
        self.overlays = []
        self.selected_line = self.selected_band = None
        if self.roi is None:
            self.roi_label.setText("미지정")
            return
        x, y, w, h = self.roi
        self.roi_label.setText(f"({x}, {y}) · {w}×{h} px")
        self.overlays.append(scene.addRect(QRectF(x, y, w, h), _pen("#ffc857", 1, True)))
        self.overlays.append(scene.addLine(x, y, x + w, y, _pen("#ffc857", 1, True)))
        r = self.result
        if r is None:
            return
        factor,unit=self._factor()
        for item in r.field_angles:
            self.overlays.extend(draw_field_angle(scene,item,factor,unit))
        for edge, color in ((r.left, "#39ded7"), (r.right, "#ffab62"),
                            (r.left_reference, "#9cdfcc"), (r.right_reference, "#efcfa3")):
            path = QPainterPath(QPointF(edge[0], r.y[0]))
            for xx, yy in zip(edge[1:], r.y[1:]):
                path.lineTo(float(xx), float(yy))
            self.overlays.append(scene.addPath(path, _pen(color, 1.5, color in ("#9cdfcc", "#efcfa3"))))
        for yy, ll, rr in list(zip(r.sample_y, r.sample_left, r.sample_right))[::max(1, len(r.sample_y) // 160)]:
            self.overlays.append(scene.addLine(ll, yy, rr, yy, _pen("#a0c5ff", .7)))
        for item in r.local_bowing:
            edge = r.left if item["side"] == "left" else r.right
            mask = (r.y >= item["start_y"]) & (r.y <= item["negative_end_y"])
            path = QPainterPath()
            for index, (xx, yy) in enumerate(zip(edge[mask], r.y[mask])):
                if index == 0:
                    path.moveTo(xx, yy)
                else:
                    path.lineTo(xx, yy)
            self.overlays.append(scene.addPath(path, _pen("#ff7b21", 4)))
        factor, unit = self._factor()
        for corner in r.corners:
            self.overlays.extend(draw_corner(scene, corner, factor, unit))
        for yy, amount, edge, baseline in ((r.left_bowing_y, r.left_bowing_px, r.left, r.left_reference),
                                           (r.right_bowing_y, r.right_bowing_px, r.right, r.right_reference)):
            xx, ref = np.interp(yy, r.y, edge), np.interp(yy, r.y, baseline)
            self.overlays.append(scene.addLine(xx, yy, ref, yy, _pen("#ffed69", 3)))
            label = scene.addText(f"Bow {amount * factor:.3g} {unit}")
            label.setDefaultTextColor(QColor("#ffed69"))
            label.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations)
            label.setPos(xx - (150 if edge is r.left else -20), yy)
            self.overlays.append(label)
        if r.depth_px is not None:
            depth_x = x + w * .96
            self.overlays.append(scene.addLine(depth_x, y, depth_x, y + r.depth_px, _pen("#e6edfa", 1.5, True)))
            for end_y in (y, y + r.depth_px):
                self.overlays.append(scene.addLine(depth_x - 6, end_y, depth_x + 6, end_y, _pen("#e6edfa")))
            label = scene.addText(f"Depth {r.depth_px * factor:.4g} {unit}")
            label.setDefaultTextColor(QColor("#ffffff"))
            label.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations)
            label.setPos(depth_x + 10, y + r.depth_px * .55)
            self.overlays.append(label)
        self.selected_line = scene.addLine(0, 0, 0, 0, _pen("#fb4fc5", 2))
        self.selected_band = scene.addRect(QRectF(), _pen("#fb4fc5", 1, True))
        self.overlays += [self.selected_line, self.selected_band]

    def _navigate(self, point):
        self.view.centerOn(point)
        self.view.notify_viewport()

    def select_depth(self, y: float):
        if self.result is None:
            return
        r = self.result
        y = float(np.clip(y, r.y[0], r.y[-1]))
        l, right = np.interp(y, r.y, r.left), np.interp(y, r.y, r.right)
        self.selected_line.setLine(l, y, right, y)
        self.graph.selected_y = y
        self.graph.update()
        self._navigate(QPointF((l + right) / 2, y))

    def _cd_selected(self, row, _column):
        if self.result is not None and row < len(self.result.sample_y):
            self.select_depth(float(self.result.sample_y[row]))

    def _local_selected(self, row, _column):
        if self.result is None or row >= len(self.result.local_bowing):
            return
        item = self.result.local_bowing[row]
        self.selected_band.setRect(QRectF(self.roi[0], item["start_y"], self.roi[2], max(1, item["end_y"] - item["start_y"])))
        self.select_depth(item["peak_y"])

    def _graph_changed(self):
        self.graph.mode = self.graph_combo.currentData()
        self.graph.update()

    def save_result(self, path: str):
        if self.result is None:
            raise ValueError("먼저 분석을 실행하세요.")
        scale = self.scale_spin.value() or None
        suffix = Path(path).suffix.lower()
        if suffix == ".csv":
            export_trench_csv(path, self.result, scale)
        elif suffix == ".json":
            payload = self.result.to_dict(scale)
            payload["image_path"] = self.image_path
            payload["settings"] = self.state()
            Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        elif suffix == ".png":
            # Full image plus independent corner insets, unaffected by pan/zoom.
            width = max(800, int(self.view.image_rect.width()))
            height = int(self.view.image_rect.height())
            image = QImage(width, height + 360, QImage.Format.Format_RGB32)
            image.fill(QColor("white"))
            painter = QPainter(image)
            self.view.scene().render(painter, QRectF(0, 0, self.view.image_rect.width(), height), self.view.image_rect)
            painter.setFont(QFont("Malgun Gothic", 10))
            for index, editor in enumerate(self.corner_editors):
                x = index * width / 2
                painter.setPen(QColor("#24364c"))
                painter.drawText(QRectF(x + 8, height + 6, width / 2 - 16, 80),
                                 Qt.TextFlag.TextWordWrap, editor.title() + " · " + editor.label.text())
                target = QRectF(x + 8, height + 88, width / 2 - 16, 230)
                painter.fillRect(target, QColor("#20262e"))
                editor.view.scene().render(painter, target, editor.focus_rect)
            painter.setPen(QColor("#425368"))
            painter.drawText(QRectF(8, height + 326, width - 16, 30),
                             "노란 점: 사용 윤곽 · 보라: 적합 원호 / 반지름 · RMS: 윤곽 맞춤 오차 (실제 치수의 정확도 아님)")
            painter.end()
            if not image.save(path):
                raise OSError("PNG 저장에 실패했습니다.")
        else:
            raise ValueError("CSV, JSON 또는 PNG 파일을 선택하세요.")

    def copy_gfe_coordinates(self):
        if self.result is None:
            self.hint.setText("먼저 Trench를 분석하세요.")
            return
        try:
            table = trench_gfe_tsv(self.result, self.scale_spin.value())
        except ValueError as exc:
            self.hint.setText(f"GFE 복사 실패: {exc}")
            return
        QApplication.clipboard().setText(table)
        self.hint.setText(f"GFE 좌표 {len(table.splitlines()) - 1}개 복사 완료 (Å). 실제 검출 Field 포함. GFE 표에서 Ctrl+V. Y=좌우 입구 Field 평균 기준, 검출 윤곽 사이 간격은 직선 연결입니다.")

    def export_dialog(self):
        stem = Path(self.image_path).stem if self.image_path else "trench"
        path, selected = QFileDialog.getSaveFileName(self, "Trench 분석 결과 저장", f"{stem}_trench.csv",
                                                    "CSV (*.csv);;JSON (*.json);;PNG (*.png)")
        if not path:
            return
        if not Path(path).suffix:
            path += ".json" if selected.startswith("JSON") else ".png" if selected.startswith("PNG") else ".csv"
        try:
            self.save_result(path)
            self.hint.setText(f"저장 완료: {path}")
        except (OSError, ValueError) as exc:
            self.hint.setText(f"저장 실패: {exc}")
