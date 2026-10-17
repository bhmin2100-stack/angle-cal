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
    QHeaderView, QLabel, QPushButton, QScrollArea, QSpinBox, QSplitter, QInputDialog,
    QSizePolicy, QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget,
)

from .image_ops import bgr_to_rgb8_for_display
from .trench_analysis import TrenchOptions, TrenchResult, analyze_trench, export_trench_csv, trench_gfe_tsv
from .trench_guides import normalize_guides
from .trench_guide_view import TrenchGuideEditor
from .trench_workflow import recognize_from_guides
from .trench_structure import structure_guides


def _pen(color: str, width: float = 1.5, dashed: bool = False) -> QPen:
    pen = QPen(QColor(color), width)
    pen.setCosmetic(True)
    if dashed:
        pen.setStyle(Qt.PenStyle.DashLine)
    return pen


class TrenchView(QGraphicsView):
    scale_drawn = Signal(object,object)
    tool_cancelled=Signal()
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
        self.scale_mode=False
        self.scale_origin=None
        self.scale_preview=None
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
        if self.scale_mode and event.button()==Qt.MouseButton.LeftButton and not self.image_rect.isEmpty():
            p=self.mapToScene(event.position().toPoint())
            self.scale_origin=QPointF(np.clip(p.x(),0,self.image_rect.width()-1),np.clip(p.y(),0,self.image_rect.height()-1))
            self.scale_preview=self.scene().addLine(p.x(),p.y(),p.x(),p.y(),_pen('#ffffff',2))
            return
        if event.button() == Qt.MouseButton.LeftButton and self.overview:
            self.navigate.emit(self.mapToScene(event.position().toPoint()))
            return
        if event.button() == Qt.MouseButton.LeftButton and self.select_region and not self.image_rect.isEmpty():
            self.origin = self.mapToScene(event.position().toPoint())
            self.rubber = self.scene().addRect(QRectF(self.origin, self.origin), _pen("#ffc857", 2, True))
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self.scale_origin is not None:
            p=self._scale_point(event);self.scale_preview.setLine(self.scale_origin.x(),self.scale_origin.y(),p.x(),p.y());return
        if self.origin is not None:
            self.rubber.setRect(QRectF(self.origin, self.mapToScene(event.position().toPoint())).normalized().intersected(self.image_rect))
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if self.scale_origin is not None:
            start=self.scale_origin;end=self._scale_point(event)
            self.scene().removeItem(self.scale_preview);self.scale_origin=self.scale_preview=None
            if (end-start).manhattanLength()>=2:self.scale_drawn.emit((start.x(),start.y()),(end.x(),end.y()))
            return
        if self.origin is not None:
            rect = self.rubber.rect()
            self.scene().removeItem(self.rubber)
            self.rubber = self.origin = None
            if rect.width() >= 12 and rect.height() >= 16:
                self.region_selected.emit(rect)
            return
        super().mouseReleaseEvent(event)

    def _scale_point(self,event):
        p=self.mapToScene(event.position().toPoint())
        p=QPointF(np.clip(p.x(),0,self.image_rect.width()-1),np.clip(p.y(),0,self.image_rect.height()-1))
        if event.modifiers()&Qt.KeyboardModifier.ShiftModifier:
            if abs(p.x()-self.scale_origin.x())>=abs(p.y()-self.scale_origin.y()):p.setY(self.scale_origin.y())
            else:p.setX(self.scale_origin.x())
        return p

    def keyPressEvent(self,event):
        if event.key()==Qt.Key.Key_Escape and self.scale_mode:
            if self.scale_preview is not None:self.scene().removeItem(self.scale_preview)
            self.scale_origin=self.scale_preview=None;self.tool_cancelled.emit();return
        super().keyPressEvent(event)

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
        support=np.asarray(item.get('tangent_points') or points.tolist()) if item.get('curved') else points
        drawn.append(scene.addLine(support[0,0], m*support[0,0]+c, support[-1,0], m*support[-1,0]+c, _pen(color, 1, True)))
        prefix="Field" if item["status"]=="valid" else "Field 후보"
        basis=' · 입구 접선' if item.get('curved') else ''
        text = scene.addSimpleText(f"{prefix}{basis} RMS {item['rms_px'] * factor:.3g} {unit}")
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
            if not item.get('curved'):
                drawn.append(scene.addLine(*field_start,*ft,_pen('#63ef90',2)))
            elif connection.get('field_join_points'):
                join=connection['field_join_points']
                path=QPainterPath(QPointF(*join[0]))
                for p in join[1:]+[ft]:path.lineTo(QPointF(*p))
                drawn.append(scene.addPath(path,_pen('#63ef90',2)))
            path=QPainterPath(QPointF(*ft))
            for p in connection['arc_points'][1:]:path.lineTo(QPointF(*p))
            drawn.append(scene.addPath(path,_pen('#df91ff',3)))
            wm,wc=item['wall_line'];end_y=wall[-1,1]
            drawn.append(scene.addLine(*wt,wm*end_y+wc,end_y,_pen('#fff275',2)))
            center=connection['center'];mid=connection['arc_points'][len(connection['arc_points'])//2]
            drawn.append(scene.addLine(*center,*mid,_pen('#df91ff',1,True)))
            drawn.append(scene.addEllipse(center[0]-1.5,center[1]-1.5,3,3,_pen('#df91ff')))
            basis='접선 ' if item.get('curved') else ''
            label=scene.addSimpleText(f"R {connection['radius_px']*factor:.3g} {unit} · {basis}{item['angle_deg']:.2f}°")
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
        if item.get('curved'):
            reference=item.get('straight_angle_deg')
            angle=(f"입구 접선 내각 {item['angle_deg']:.2f}°" if item['angle_deg'] is not None else '입구 접선 각도 미확정')
            angle+=' · 전체 직선 맞춤 참고 '+(f'{reference:.2f}°' if reference is not None else '미확정')
            field=field.replace('기울기','입구 접선 기울기')
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
            for item in (r.field_angles if getattr(self,"show_entrance",True) else []):
                if item["status"] == "valid":
                    points=np.asarray(item["points"])
                    series.append((points[:,0],points[:,1],"#268b48"))
            title = "좌우 경계 X"
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
        for item in (r.local_bowing if getattr(self,"show_entrance",True) else []):
            a, b = point(xmin, item["start_y"]).y(), point(xmax, item["negative_end_y"]).y()
            painter.fillRect(QRectF(rect.left(), a, rect.width(), max(1, b - a)), QColor(255, 171, 64, 22))
        if self.mode == "shape":
            # Both walls share one X axis, so their separation is the measured
            # CD. Do not normalize or center each wall independently.
            for block in np.split(np.arange(len(r.y)),np.flatnonzero(np.diff(r.y)>1.5)+1):
                cavity = QPainterPath(point(r.left[block[0]],r.y[block[0]]))
                for i in block[1:]:cavity.lineTo(point(r.left[i],r.y[i]))
                for i in block[::-1]:cavity.lineTo(point(r.right[i],r.y[i]))
                cavity.closeSubpath()
                painter.fillPath(cavity, QColor(0, 125, 146, 25))
        for values, ys, color in series:
            path = QPainterPath()
            stride = max(1, len(ys) // 2000)
            for j, index in enumerate(range(0, len(ys), stride)):
                if j == 0 or (ys is r.y and np.any(np.diff(ys[index-stride:index+1])>1.5)):
                    path.moveTo(point(values[index], ys[index]))
                else:
                    path.lineTo(point(values[index], ys[index]))
            painter.setPen(QPen(QColor(color), 2))
            painter.drawPath(path)
        if self.mode == "shape":
            for item in (r.field_angles if getattr(self,"show_entrance",True) else []):
                if item["angle_status"] == "valid":
                    vx,vy=item["vertex"]
                    wm,wc=item["wall_line"]
                    end_y=item["wall_points"][-1][1]
                    connection=item.get('connection',{})
                    if connection.get('status')=='valid':
                        join=connection.get('field_join_points',[])
                        if join:
                            path=QPainterPath(point(*join[0]))
                            for p in join[1:]+[connection['field_tangent']]:path.lineTo(point(*p))
                            painter.setPen(_pen('#268b48',2));painter.drawPath(path)
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

    def __init__(self, image, options, revision, parent, automatic_area=False):
        super().__init__(parent)
        self.image, self.options, self.revision = image, options, revision
        self.automatic_area=automatic_area

    def run(self):
        try:
            result=recognize_from_guides(self.image,self.options) if self.automatic_area else analyze_trench(self.image,self.options)
            self.completed.emit(result, "", self.revision)
        except Exception as exc:
            self.completed.emit(None, str(exc), self.revision)


class TrenchPanel(QWidget):
    state_changed = Signal(dict)
    calibration_changed = Signal(float)
    scale_bar_changed = Signal(dict)

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
        self.automatic_area=True
        self.boundary_mode='max_gradient'
        self.scale_bar=None
        root=QVBoxLayout(self);root.setContentsMargins(6,4,6,4)
        self.hint=QLabel('① 썸네일 선택 → ② 곡선 그리기 또는 구조 불러오기 → ③ 인식')
        self.hint.setWordWrap(True);root.addWidget(self.hint)
        self.view=TrenchView();self.guide_editor=TrenchGuideEditor(self.view)
        self.guide_bar=QWidget();tools=QHBoxLayout(self.guide_bar);tools.setContentsMargins(0,0,0,0);tools.setSpacing(5)
        self.tool_buttons={}
        for key,title in (('select','선택'),('pan','이동'),('resize','크기 조절'),('scale','스케일바')):
            button=QPushButton(title);button.setCheckable(True)
            button.clicked.connect(lambda checked=False,mode=key:self.set_tool(mode))
            self.tool_buttons[key]=button;tools.addWidget(button)
        self.guide_button=QPushButton('곡선 그리기');self.guide_button.setCheckable(True)
        self.guide_button.setObjectName('trench_contour_guide');tools.addWidget(self.guide_button)
        self.load_structure_button=QToolButton();self.load_structure_button.setText('구조 불러오기')
        self.load_structure_button.setPopupMode(QToolButton.ToolButtonPopupMode.MenuButtonPopup)
        self.structure_menu=QMenu(self.load_structure_button);self.load_structure_button.setMenu(self.structure_menu)
        self.load_structure_button.clicked.connect(self.load_structure_dialog);tools.addWidget(self.load_structure_button)
        self.save_structure_button=QPushButton('구조 저장');self.save_structure_button.clicked.connect(self.save_structure_dialog);tools.addWidget(self.save_structure_button)
        self.guide_analyze=QPushButton('인식');self.guide_analyze.clicked.connect(self.start_analysis);tools.addWidget(self.guide_analyze)
        self.guide_selection=QComboBox();self.guide_selection.setMinimumWidth(100);tools.addWidget(self.guide_selection)
        tools.addWidget(QLabel('인식 폭 ±px'));self.guide_width=QDoubleSpinBox();self.guide_width.setRange(1,300);self.guide_width.setValue(12);tools.addWidget(self.guide_width)
        self.guide_delete=QPushButton('삭제');self.guide_delete.clicked.connect(self.guide_editor.remove_selected);tools.addWidget(self.guide_delete)
        self.guide_clear=QPushButton('모두 지우기');self.guide_clear.clicked.connect(self.guide_editor.clear)
        tools.addStretch();root.addWidget(self.guide_bar)
        self.guide_hint=QLabel('점을 찍고 마지막 더블클릭으로 완료 · 점/곡선 드래그로 보정 · 휠: 인식 폭 · Ctrl+휠: 확대')
        root.addWidget(self.guide_hint)
        self.guide_bar.setEnabled(False)
        self.guide_button.toggled.connect(self._guide_mode)
        self.guide_width.valueChanged.connect(self.guide_editor.set_radius)
        self.guide_selection.currentIndexChanged.connect(lambda i:self.guide_editor.select(i-1))
        self.guide_editor.changed.connect(self._guides_changed)
        self.guide_editor.selection_changed.connect(self._guide_selection_changed)
        self.guide_editor.mode_changed.connect(self._guide_mode_changed)
        self.guide_editor.message.connect(self.guide_hint.setText)
        self.view.scale_drawn.connect(self._calibrate_scale)
        self.view.tool_cancelled.connect(lambda:self.set_tool('select'))
        body=QSplitter(Qt.Orientation.Horizontal);root.addWidget(body,1)
        left=QSplitter(Qt.Orientation.Vertical);body.addWidget(left)
        self.image_tabs=QTabWidget();self.image_tabs.addTab(self.view,'이미지');self.image_tabs.tabBar().hide();left.addWidget(self.image_tabs)
        lower=QSplitter(Qt.Orientation.Horizontal);left.addWidget(lower)
        self.graph=TrenchGraph();lower.addWidget(self.graph)
        self.tables=QTabWidget();self.cd_table=self._table(['깊이','CD','왼쪽 X','오른쪽 X'])
        self.local_table=self._table(['측벽','시작 깊이','Nega 끝','국부 Bow','현 대비','역경사°'])
        self.tables.addTab(self.cd_table,'깊이별 CD');self.tables.tabBar().hide();lower.addWidget(self.tables)
        self.cd_table.cellClicked.connect(self._cd_selected);self.graph.depth_selected.connect(self.select_depth)
        left.setSizes([560,230]);lower.setSizes([320,500])
        sidebar=QWidget();sidebar.setMinimumWidth(220);sidebar.setMaximumWidth(270);body.addWidget(sidebar)
        controls=QVBoxLayout(sidebar);controls.setContentsMargins(6,4,6,4)
        controls.addWidget(QLabel('스케일'))
        form=QFormLayout();controls.addLayout(form)
        self.scale_spin=QDoubleSpinBox();self.scale_spin.setRange(0,1e9);self.scale_spin.setDecimals(8);self.scale_spin.setSpecialValueText('미보정')
        form.addRow('nm / px',self.scale_spin)
        self.unit_combo=QComboBox()
        for title,data in (('px','px'),('nm','nm'),('Å','angstrom')):self.unit_combo.addItem(title,data)
        form.addRow('표시 단위',self.unit_combo)
        self.step_spin=QDoubleSpinBox();self.step_spin.setRange(.001,1e7);self.step_spin.setDecimals(3);self.step_spin.setValue(1)
        form.addRow('CD 깊이 간격 (nm)',self.step_spin)
        self.calibration_note=QLabel('스케일바를 그리면 입력한 길이를 기준으로 보정합니다.');self.calibration_note.setWordWrap(True);controls.addWidget(self.calibration_note)
        self.analyze_button=QPushButton('인식');self.analyze_button.setMinimumHeight(32);self.analyze_button.clicked.connect(self.start_analysis);controls.addWidget(self.analyze_button)
        fit=QPushButton('전체 보기');fit.clicked.connect(self.view.fit_image);controls.addWidget(fit)
        self.summary_label=QLabel('곡선을 그리거나 구조를 불러오세요.');self.summary_label.setWordWrap(True);self.summary_label.setSizePolicy(QSizePolicy.Policy.Ignored,QSizePolicy.Policy.Preferred);controls.addWidget(self.summary_label)
        self.export_button=QToolButton();self.export_button.setText('Export · GFE 좌표 복사');self.export_button.setPopupMode(QToolButton.ToolButtonPopupMode.MenuButtonPopup)
        menu=QMenu(self.export_button);menu.addAction('GFE 좌표 복사',self.copy_gfe_coordinates);menu.addAction('파일 저장 (CSV / JSON / PNG)…',self.export_dialog)
        self.export_button.setMenu(menu);self.export_button.clicked.connect(self.copy_gfe_coordinates);self.export_button.setEnabled(False);controls.addWidget(self.export_button);controls.addStretch()
        # Compatibility objects for old projects and the independent backend.
        # These are not part of the simplified workspace or image overlays.
        self.overview=TrenchView(overview=True);self.overview.setParent(self);self.overview.hide()
        self.roi_label=QLabel(self);self.roi_label.hide()
        self.roi_button=QPushButton(self);self.roi_button.setCheckable(True);self.roi_button.toggled.connect(self._roi_mode);self.roi_button.hide()
        self.view.region_selected.connect(self.set_roi)
        self.threshold_spin=QSpinBox(self);self.threshold_spin.setRange(0,254)
        self.bright_check=QCheckBox(self)
        self.smooth_spin=QSpinBox(self);self.smooth_spin.setRange(1,101);self.smooth_spin.setValue(7)
        self.bow_combo=QComboBox(self);self.bow_combo.addItem('','entrance');self.bow_combo.addItem('','chord')
        self.negative_spin=QDoubleSpinBox(self);self.negative_spin.setRange(0,45);self.negative_spin.setValue(1)
        self.corner_spin=QSpinBox(self);self.corner_spin.setRange(6,200);self.corner_spin.setValue(24)
        self.field_spin=QSpinBox(self);self.field_spin.setRange(4,300);self.field_spin.setValue(32)
        self.angle_start=QSpinBox(self);self.angle_start.setRange(0,10000);self.angle_start.setValue(40)
        self.angle_end=QSpinBox(self);self.angle_end.setRange(1,10000);self.angle_end.setValue(100)
        self.graph_combo=QComboBox(self);self.graph_combo.addItem('','cd');self.graph_combo.addItem('','shape');self.graph_combo.addItem('','bow');self.graph_combo.setCurrentIndex(1);self.graph_combo.currentIndexChanged.connect(self._graph_changed);self.graph.mode='shape'
        self.corner_editors=[CornerEditor('왼쪽 입구'),CornerEditor('오른쪽 입구')]
        for i,editor in enumerate(self.corner_editors):editor.changed.connect(lambda roi,span,index=i:self._corner_changed(index,roi,span))
        self.surface_editors=[SurfaceEditor('왼쪽'),SurfaceEditor('오른쪽')]
        for widget in [self.guide_clear,self.local_table,self.threshold_spin,self.bright_check,self.smooth_spin,self.bow_combo,self.negative_spin,self.corner_spin,self.field_spin,self.angle_start,self.angle_end,self.graph_combo,*self.corner_editors,*self.surface_editors]:
            widget.setParent(self);widget.hide()
        for spin in (self.step_spin,self.threshold_spin,self.smooth_spin,self.negative_spin,self.corner_spin,self.field_spin,self.angle_start,self.angle_end):spin.valueChanged.connect(self._settings_changed)
        self.scale_spin.valueChanged.connect(self._scale_changed)
        self.unit_combo.currentIndexChanged.connect(self._settings_changed)
        self.bow_combo.currentIndexChanged.connect(self._settings_changed);self.bright_check.toggled.connect(self._settings_changed)
        self.set_tool('select')

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
                "workflow_version":2,"automatic_area":self.automatic_area,"step_nm":self.step_spin.value(),
                "boundary_mode":self.boundary_mode,"scale_bar":self.scale_bar,
                "unit": self.unit_combo.currentData(), "step": self.step_spin.value(),
                "threshold": self.threshold_spin.value(), "bright": self.bright_check.isChecked(),
                "smooth": self.smooth_spin.value(), "bow_reference": self.bow_combo.currentData(),
                "negative_angle": self.negative_spin.value(), "corner": self.corner_spin.value(),
                "field_search":self.field_spin.value(),"angle_start":self.angle_start.value(),"angle_end":self.angle_end.value(),
                "field_angles":self.result.field_angles if self.result is not None else None,
                "corner_rois": list(self.corner_rois), "corner_spans": list(self.corner_spans),
                "corner_results": self.result.corners if self.result is not None else None,
                "wall_guides":list(normalize_guides(self.guide_editor.guides)),
                "guide_traces":self.result.guide_traces if self.result is not None else []}

    def _roi_mode(self, enabled):
        if enabled:self.guide_editor.set_drawing(False)
        self.view.set_select_region(enabled)

    def _guide_mode(self, enabled):
        if enabled:
            self.set_tool('curve',from_button=True)
            self.image_tabs.setCurrentIndex(0)
            self.roi_button.setChecked(False)
        self.guide_editor.set_drawing(enabled)

    def _guide_mode_changed(self, enabled):
        self.guide_button.blockSignals(True);self.guide_button.setChecked(enabled);self.guide_button.blockSignals(False)
        if not enabled and self.guide_editor.interaction_mode=='curve':
            self.guide_editor.interaction_mode='select';self.tool_buttons['select'].setChecked(True)

    def start_contour_guide(self):
        """Open the same curve editor from the Trench ribbon."""
        if self.image is None:
            self.hint.setText('먼저 Trench 이미지를 불러오세요.')
            return
        self.guide_button.setChecked(True)
        self.image_tabs.setCurrentIndex(0)
        self.view.setFocus()

    def _guide_selection_changed(self):
        editor=self.guide_editor
        self.guide_selection.blockSignals(True)
        self.guide_selection.clear();self.guide_selection.addItem('곡선 선택')
        for i,g in enumerate(editor.guides):
            self.guide_selection.addItem(f"윤곽 {i+1} · {len(g['points'])}점")
        self.guide_selection.setCurrentIndex(editor.selected+1)
        self.guide_selection.blockSignals(False)
        self.guide_width.blockSignals(True)
        self.guide_width.setValue(editor.guides[editor.selected]['half_width_px'] if editor.selected>=0 else editor.radius)
        self.guide_width.blockSignals(False)
        self.guide_delete.setEnabled(editor.selected>=0)
        self.guide_clear.setEnabled(bool(editor.guides) or editor.drawing)

    def _guides_changed(self):
        self._guide_selection_changed()
        self._settings_changed()
        self.guide_hint.setText(f'곡선 {len(self.guide_editor.guides)}개 · 점/곡선을 드래그로 보정 · 휠로 인식 폭 조절 → 인식')

    def set_tool(self,mode,from_button=False):
        if not from_button:self.guide_button.setChecked(False)
        self.view.scale_mode=mode=='scale'
        self.view.set_select_region(False)
        self.guide_editor.interaction_mode=mode
        if not from_button:self.guide_editor.set_drawing(False)
        self.view.setDragMode(QGraphicsView.DragMode.NoDrag if mode in ('scale','curve','resize') else QGraphicsView.DragMode.ScrollHandDrag)
        self.view.setCursor(Qt.CursorShape.CrossCursor if mode in ('scale','curve') else Qt.CursorShape.OpenHandCursor)
        for key,button in self.tool_buttons.items():button.setChecked(key==mode)
        self.guide_editor.render()

    def _scale_changed(self,value):
        if self.loading:return
        if self.scale_bar:
            length=np.linalg.norm(np.subtract(self.scale_bar['end'],self.scale_bar['start']))
            self.scale_bar['length_nm']=float(value*length)
        self._settings_changed();self.calibration_changed.emit(value)
        if self.scale_bar:self.scale_bar_changed.emit(self.scale_bar)
        self.calibration_note.setText('기존 스케일 또는 스케일바 보정값을 사용합니다.' if value>0 else '미보정: px로 표시합니다. 스케일바를 그리면 1nm 간격으로 측정합니다.')

    def _calibrate_scale(self,start,end):
        length=float(np.linalg.norm(np.subtract(end,start)))
        value,ok=QInputDialog.getDouble(self,'스케일바 길이',f'그은 선 길이: {length:.2f} px\n실제 길이(nm)를 입력하세요.',100.,.0001,1_000_000.,4)
        if not ok:return
        self.scale_bar=dict(start=list(start),end=list(end),length_nm=value)
        self.scale_spin.setValue(value/length)
        self.scale_bar_changed.emit(self.scale_bar)
        self.unit_combo.setCurrentIndex(self.unit_combo.findData('nm'))
        self.set_tool('select');self._draw_overlay();self.state_changed.emit(self.state())

    def set_scale_record(self,record=None):
        if record is not None:
            self.scale_bar=dict(start=list(record.start),end=list(record.end),length_nm=record.value_nm or 0)
        self._draw_overlay()

    def load_structure_payload(self,payload):
        guides=structure_guides(payload,self.image.shape if self.image is not None else None)
        self.guide_editor.restore(guides);self._guides_changed();self.set_tool('select')
        self.guide_editor.select(0);self.hint.setText('구조를 불러왔습니다. 위치·크기를 보정하거나 바로 인식을 누르세요.')

    def load_structure_dialog(self):
        path,_=QFileDialog.getOpenFileName(self,'구조 불러오기','','AngleCal 구조 (*.anglecal.structure.json *.json)')
        if not path:return
        try:self.load_structure_payload(json.loads(Path(path).read_text(encoding='utf-8')))
        except (ValueError,OSError,TypeError,KeyError) as exc:self.hint.setText(f'구조를 불러오지 못했습니다: {exc}')

    def save_structure_dialog(self):
        if not self.guide_editor.guides:self.hint.setText('먼저 저장할 곡선을 그리세요.');return
        path,_=QFileDialog.getSaveFileName(self,'구조 저장','Trench.anglecal.structure.json','AngleCal 구조 (*.anglecal.structure.json)')
        if not path:return
        if not path.endswith('.json'):path+='.anglecal.structure.json'
        payload=dict(format='anglecal.trench.structure.v1',name=Path(path).stem,wall_guides=list(normalize_guides(self.guide_editor.guides)))
        try:Path(path).write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8');self.hint.setText('곡선 구조를 저장했습니다.')
        except OSError as exc:self.hint.setText(f'구조를 저장하지 못했습니다: {exc}')

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
        self.scale_bar=None
        self.automatic_area=True
        self.revision += 1
        self.guide_editor.restore()
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
        try:
            guides=normalize_guides(state.get('wall_guides',()),image.shape if image is not None else None)
            self.guide_editor.restore(guides)
        except ValueError as exc:
            self.guide_hint.setText(f'저장된 가이드를 열지 못했습니다: {exc}')
        self.guide_bar.setEnabled(image is not None)
        for key_name in ("corner_rois", "corner_spans"):
            values = state.get(key_name)
            if isinstance(values, (list, tuple)) and len(values) == 2:
                setattr(self, key_name, [tuple(value) if value is not None else None for value in values])
        for i, editor in enumerate(self.corner_editors):
            editor.restore(self.corner_rois[i], self.corner_spans[i])
        self.automatic_area=bool(state.get('automatic_area',True))
        self.boundary_mode=state.get('boundary_mode',self.boundary_mode)
        self.scale_bar=state.get('scale_bar')
        self.scale_spin.setValue(float(nm_per_px if nm_per_px is not None else state.get('nm_per_px',0)))
        for control, key_name, default in ((self.step_spin, "step_nm", 1), (self.threshold_spin, "threshold", 0),
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
        self.roi_button.setChecked(False)
        self.view.set_select_region(False)
        self.set_tool('select')
        self.loading = False
        self.calibration_note.setText('기존 스케일 또는 스케일바 보정값을 사용합니다.' if self.scale_spin.value()>0 else '미보정: px로 표시합니다. 스케일바를 그리면 1nm 간격으로 측정합니다.')
        self._draw_overlay()

    def set_roi(self, rect: QRectF) -> None:
        self.automatic_area=False  # Legacy projects/API; never a visible workflow step.
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
        self.summary_label.setText("곡선을 그리거나 불러온 뒤 인식을 누르세요.")
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
        if self.roi is None and not self.automatic_area:
            raise ValueError("먼저 입구 높이부터 한 Trench를 포함하는 영역을 지정하세요.")
        factor, _ = self._factor()
        roi=self.roi if not self.automatic_area else (0,0,self.image.shape[1],self.image.shape[0])
        step=self.step_spin.value()/self.scale_spin.value() if self.scale_spin.value()>0 else self.step_spin.value()
        return TrenchOptions(roi, step, self.threshold_spin.value(),
                             self.bright_check.isChecked(), self.smooth_spin.value(), self.bow_combo.currentData(),
                             self.corner_spin.value(), self.negative_spin.value(),
                             tuple(self.corner_rois), tuple(self.corner_spans),self.field_spin.value(),
                             (self.angle_start.value(),self.angle_end.value()),normalize_guides(self.guide_editor.guides),
                             self.boundary_mode if self.automatic_area else 'inner_rim')

    def start_analysis(self):
        if self.worker is not None and self.worker.isRunning():
            return
        if self.guide_editor.drawing and self.guide_editor.draft:
            self.summary_label.setText('곡선의 마지막 점을 더블클릭해 그리기를 완료한 뒤 경계를 인식하세요.')
            return
        try:
            if self.image is None:
                raise ValueError("파일 탭에서 이미지를 먼저 불러오세요.")
            options = self.options()
            if self.automatic_area and not options.wall_guides:
                raise ValueError('먼저 곡선을 그리거나 구조를 불러온 뒤 인식을 누르세요.')
        except ValueError as exc:
            self.summary_label.setText(str(exc))
            return
        self._clear_result()
        self.summary_label.setText("곡선 주변 경계를 인식하고 있습니다…")
        self.analyze_button.setEnabled(False)
        self.guide_analyze.setEnabled(False)
        self.worker = TrenchWorker(self.image, options, self.revision, self, self.automatic_area)
        self.worker.completed.connect(self._analysis_completed)
        self.worker.finished.connect(lambda worker=self.worker: self._worker_finished(worker))
        self.worker.start()

    def _worker_finished(self, worker):
        if self.worker is worker:
            self.worker = None
        worker.deleteLater()
        self.analyze_button.setEnabled(True)
        self.guide_analyze.setEnabled(True)
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
        self.guide_editor.select(-1)
        self.roi=result.options.roi
        self._render_result()
        self.state_changed.emit(self.state())

    def _render_result(self):
        r = self.result
        factor, unit = self._factor()
        fmt = lambda value: "미확정" if value is None else f"{value * factor:.4g} {unit}"
        angle = lambda value: "미확정" if value is None else f"{value:.2f}°"
        summary = r.summary()
        lines=[f"깊이  {fmt(r.depth_px)}",f"CD  {fmt(summary['min_cd_px'])} ~ {fmt(summary['max_cd_px'])}",
               f"Bowing 좌 / 우  {fmt(r.left_bowing_px)} / {fmt(r.right_bowing_px)}"]
        if self.scale_spin.value()<=0:lines.append('미보정: px 표시 · 스케일바로 보정하세요.')
        if self.step_spin.value()/max(self.scale_spin.value(),1e-9)<1:lines.append('1px 미만 간격은 윤곽 보간값입니다.')
        if r.depth_px is None:lines.append('바닥이 보이지 않아 전체 깊이는 미확정입니다.')
        if any(t['trace_coverage']<.8 for t in r.guide_traces):lines.append('일부 경계가 끊겼습니다. 곡선 위치·인식 폭을 확인하세요.')
        self.summary_label.setText("\n".join(lines))
        self.summary_label.setMinimumHeight(self.summary_label.fontMetrics().lineSpacing()*(len(lines)+1))
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
        self.graph.show_entrance=False
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
        if self.scale_bar:
            a,b=self.scale_bar['start'],self.scale_bar['end']
            self.overlays.append(scene.addLine(*a,*b,_pen('#ffffff',2)))
            label=scene.addSimpleText(f"{self.scale_bar['length_nm']:g} nm")
            label.setBrush(QColor('#ffffff'));label.setPos(a[0],a[1]-22);label.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations,True);self.overlays.append(label)
        if self.roi is None:
            self.roi_label.setText("미지정")
            return
        x, y, w, h = self.roi
        self.roi_label.setText(f"({x}, {y}) · {w}×{h} px")
        # Analysis bounds are automatic and are not an editable/visible ROI.

        r = self.result
        if r is None:
            return
        factor,unit=self._factor()
        for trace in r.guide_traces:
            for segment in trace['segments']:
                if len(segment)<2:continue
                path=QPainterPath(QPointF(*segment[0]))
                for point in segment[1:]:path.lineTo(QPointF(*point))
                self.overlays.append(scene.addPath(path,_pen('#ffe38a',1.2)))

        for edge,color in ((r.left,'#39ded7'),(r.right,'#ffab62')):
            path = QPainterPath(QPointF(edge[0], r.y[0]))
            for previous_y,xx,yy in zip(r.y[:-1],edge[1:],r.y[1:]):
                if yy-previous_y>1.5:path.moveTo(float(xx),float(yy))
                else:path.lineTo(float(xx),float(yy))
            self.overlays.append(scene.addPath(path, _pen(color, 1.5, color in ("#9cdfcc", "#efcfa3"))))
        if r.depth_px is not None:
            depth_x = float(np.max(r.right))+max(12,w*.015)
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
            width=max(800,int(self.view.image_rect.width()));height=int(self.view.image_rect.height())
            image=QImage(width,height+95,QImage.Format.Format_RGB32);image.fill(QColor('white'))
            painter=QPainter(image)
            self.view.scene().render(painter,QRectF(0,0,self.view.image_rect.width(),height),self.view.image_rect)
            painter.setFont(QFont('Malgun Gothic',10));painter.setPen(QColor('#24364c'))
            painter.drawText(QRectF(8,height+5,width-16,85),Qt.TextFlag.TextWordWrap,self.summary_label.text())
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
        self.hint.setText(f"GFE 좌표 {len(table.splitlines()) - 1}개 복사 완료 (Å). GFE 표에서 Ctrl+V로 붙여넣으세요.")

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
