from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import json
import math
from pathlib import Path
import sys
from typing import Optional

import cv2
import numpy as np
from PySide6.QtCore import QPoint, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QAction, QBrush, QColor, QImage, QPainter, QPainterPath, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDockWidget,
    QFormLayout,
    QGraphicsItem,
    QFileDialog,
    QGraphicsLineItem,
    QGraphicsPathItem,
    QGraphicsPixmapItem,
    QGraphicsPolygonItem,
    QGraphicsScene,
    QGraphicsTextItem,
    QGraphicsView,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QStatusBar,
    QTableWidget,
    QTableWidgetItem,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from .image_ops import (
    Point,
    acute_angle_difference,
    angle_to_axis,
    bgr_to_rgb8_for_display,
    intersection,
    line_angle_degrees,
    line_length,
    normal_for_line,
    read_image,
    rotate_image_and_points,
    snap_line_to_gradient,
    to_gray,
)


@dataclass
class LineRecord:
    id: str
    kind: str
    start: Point
    end: Point
    label: str = ""
    axis: str = "horizontal"
    value_nm: Optional[float] = None


class AnnotationLineItem(QGraphicsLineItem):
    def __init__(self, record: LineRecord, pen: QPen):
        super().__init__(record.start[0], record.start[1], record.end[0], record.end[1])
        self.record_id = record.id
        self.kind = record.kind
        self.setPen(pen)
        self.setZValue(10 if record.kind != "guide" else 4)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        if record.kind != "guide":
            self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)


class AngleCanvas(QGraphicsView):
    line_created = Signal(str, tuple, tuple)
    scene_changed = Signal()

    def __init__(self):
        super().__init__()
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.BoundingRectViewportUpdate)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)

        self.pixmap_item: Optional[QGraphicsPixmapItem] = None
        self.line_items: dict[str, AnnotationLineItem] = {}
        self.angle_items: list[QGraphicsPathItem | QGraphicsTextItem] = []
        self.search_range_items: list[QGraphicsPolygonItem] = []
        self.search_range_radius_px = 35
        self.show_search_range = True
        self.current_tool = "select"
        self._drawing_start: Optional[QPointF] = None
        self._temp_line: Optional[QGraphicsLineItem] = None
        self._panning = False
        self._pan_last = QPoint()

    def set_tool(self, tool: str) -> None:
        self.current_tool = tool
        if tool == "pan":
            self.setCursor(Qt.CursorShape.OpenHandCursor)
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
        else:
            self.unsetCursor()
            self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)

    def set_image(self, pixmap: QPixmap) -> None:
        self.scene.clear()
        self.line_items.clear()
        self.angle_items.clear()
        self.search_range_items.clear()
        self.pixmap_item = self.scene.addPixmap(pixmap)
        self.pixmap_item.setZValue(0)
        self.scene.setSceneRect(QRectF(0, 0, pixmap.width(), pixmap.height()))
        self.resetTransform()
        self.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def redraw_lines(self, records: list[LineRecord]) -> None:
        for item in list(self.line_items.values()):
            self.scene.removeItem(item)
        self.line_items.clear()
        for record in records:
            item = AnnotationLineItem(record, self._pen_for_record(record))
            self.scene.addItem(item)
            self.line_items[record.id] = item

    def set_search_range(self, radius_px: int, visible: bool, records: list[LineRecord]) -> None:
        self.search_range_radius_px = radius_px
        self.show_search_range = visible
        self.update_search_range_overlay(records)

    def update_search_range_overlay(self, records: list[LineRecord]) -> None:
        for item in self.search_range_items:
            self.scene.removeItem(item)
        self.search_range_items.clear()
        if not self.show_search_range or self.pixmap_item is None:
            return
        radius = float(self.search_range_radius_px)
        if radius <= 0:
            return
        for record in records:
            if record.kind != "edge":
                continue
            polygon = self._search_range_polygon(record, radius)
            if polygon is None:
                continue
            item = QGraphicsPolygonItem(polygon)
            item.setPen(QPen(QColor(255, 209, 102, 210), 1.2, Qt.PenStyle.DashLine))
            item.setBrush(QBrush(QColor(255, 209, 102, 42)))
            item.setZValue(3)
            item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, False)
            self.scene.addItem(item)
            self.search_range_items.append(item)

    def clear_angle_items(self) -> None:
        for item in self.angle_items:
            self.scene.removeItem(item)
        self.angle_items.clear()

    def add_angle_arc(self, center: Point, angle_a: float, angle_b: float, radius: float = 28.0) -> None:
        path = self._arc_path(center, angle_a, angle_b, radius)
        item = QGraphicsPathItem(path)
        item.setPen(QPen(QColor("#ffd166"), 2.0))
        item.setZValue(20)
        self.scene.addItem(item)
        self.angle_items.append(item)

    def add_angle_label(self, text: str, pos: Point) -> QGraphicsTextItem:
        item = QGraphicsTextItem()
        item.setHtml(
            "<div style='background-color:rgba(24,24,24,185);"
            "color:white;padding:2px 5px;border-radius:3px;'>"
            f"{text}</div>"
        )
        item.setDefaultTextColor(QColor("white"))
        item.setPos(pos[0], pos[1])
        item.setZValue(30)
        item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        self.scene.addItem(item)
        self.angle_items.append(item)
        return item

    def selected_line_ids(self) -> list[str]:
        ids: list[str] = []
        for item in self.scene.selectedItems():
            if isinstance(item, AnnotationLineItem):
                ids.append(item.record_id)
        return ids

    def export_scene_png(self, path: str) -> None:
        rect = self.scene.sceneRect()
        image = QImage(int(rect.width()), int(rect.height()), QImage.Format.Format_ARGB32)
        image.fill(Qt.GlobalColor.transparent)
        painter = QPainter(image)
        self.scene.render(painter, QRectF(image.rect()), rect)
        painter.end()
        image.save(path)

    def wheelEvent(self, event):  # noqa: N802
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            factor = 1.2 if event.angleDelta().y() > 0 else 1 / 1.2
            self.scale(factor, factor)
            event.accept()
            return
        super().wheelEvent(event)

    def mousePressEvent(self, event):  # noqa: N802
        if event.button() in (Qt.MouseButton.MiddleButton, Qt.MouseButton.RightButton):
            self._start_pan(event.pos())
            event.accept()
            return

        if event.button() == Qt.MouseButton.LeftButton and self.current_tool == "pan":
            self._start_pan(event.pos())
            event.accept()
            return

        if (
            event.button() == Qt.MouseButton.LeftButton
            and self.current_tool in {"scale", "reference", "edge"}
            and self.pixmap_item is not None
        ):
            self._drawing_start = self._clamp_to_image(self.mapToScene(event.pos()))
            self._temp_line = QGraphicsLineItem(
                self._drawing_start.x(),
                self._drawing_start.y(),
                self._drawing_start.x(),
                self._drawing_start.y(),
            )
            self._temp_line.setPen(QPen(QColor("#4cc9f0"), 2.0, Qt.PenStyle.DashLine))
            self._temp_line.setZValue(40)
            self.scene.addItem(self._temp_line)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):  # noqa: N802
        if self._panning:
            delta = event.pos() - self._pan_last
            self._pan_last = event.pos()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            event.accept()
            return

        if self._temp_line is not None and self._drawing_start is not None:
            end = self._clamp_to_image(self.mapToScene(event.pos()))
            self._temp_line.setLine(
                self._drawing_start.x(),
                self._drawing_start.y(),
                end.x(),
                end.y(),
            )
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):  # noqa: N802
        if self._panning and event.button() in (
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.MiddleButton,
            Qt.MouseButton.RightButton,
        ):
            self._panning = False
            if self.current_tool == "pan":
                self.setCursor(Qt.CursorShape.OpenHandCursor)
            event.accept()
            return

        if self._temp_line is not None and self._drawing_start is not None:
            end = self._clamp_to_image(self.mapToScene(event.pos()))
            start = self._drawing_start
            self.scene.removeItem(self._temp_line)
            self._temp_line = None
            self._drawing_start = None
            if math.hypot(end.x() - start.x(), end.y() - start.y()) > 3:
                self.line_created.emit(
                    self.current_tool,
                    (float(start.x()), float(start.y())),
                    (float(end.x()), float(end.y())),
                )
            event.accept()
            return
        super().mouseReleaseEvent(event)
        self.scene_changed.emit()

    def _start_pan(self, pos: QPoint) -> None:
        self._panning = True
        self._pan_last = pos
        self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def _clamp_to_image(self, point: QPointF) -> QPointF:
        rect = self.scene.sceneRect()
        return QPointF(
            min(max(point.x(), rect.left()), rect.right()),
            min(max(point.y(), rect.top()), rect.bottom()),
        )

    @staticmethod
    def _arc_path(center: Point, angle_a: float, angle_b: float, radius: float) -> QPainterPath:
        delta = ((angle_b - angle_a + 90.0) % 180.0) - 90.0
        steps = max(8, int(abs(delta) / 4))
        path = QPainterPath()
        for idx in range(steps + 1):
            angle = math.radians(angle_a + delta * idx / steps)
            x = center[0] + math.cos(angle) * radius
            y = center[1] + math.sin(angle) * radius
            if idx == 0:
                path.moveTo(x, y)
            else:
                path.lineTo(x, y)
        return path

    @staticmethod
    def _pen_for_record(record: LineRecord) -> QPen:
        colors = {
            "scale": "#4cc9f0",
            "reference": "#06d6a0",
            "edge": "#ff6b6b",
            "guide": "#f7fff7",
        }
        width = 1.2 if record.kind == "guide" else 2.2
        pen = QPen(QColor(colors.get(record.kind, "#ffffff")), width)
        if record.kind == "guide":
            pen.setStyle(Qt.PenStyle.DotLine)
        return pen

    @staticmethod
    def _search_range_polygon(record: LineRecord, radius: float) -> Optional[QPolygonF]:
        nx, ny = normal_for_line(record.start, record.end)
        if nx == 0 and ny == 0:
            return None
        sx, sy = record.start
        ex, ey = record.end
        return QPolygonF(
            [
                QPointF(sx + nx * radius, sy + ny * radius),
                QPointF(ex + nx * radius, ey + ny * radius),
                QPointF(ex - nx * radius, ey - ny * radius),
                QPointF(sx - nx * radius, sy - ny * radius),
            ]
        )


class EdgeDetectionSettingsDialog(QDialog):
    def __init__(self, radius_px: int, show_overlay: bool, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("인식 설정")
        self.setModal(True)

        self.radius_spin = QSpinBox()
        self.radius_spin.setRange(2, 300)
        self.radius_spin.setValue(radius_px)
        self.radius_spin.setSuffix(" px")

        self.overlay_checkbox = QCheckBox("이미지 위에 탐색 범위 표시")
        self.overlay_checkbox.setChecked(show_overlay)

        layout = QVBoxLayout(self)
        form = QFormLayout()
        form.addRow("명도 탐색 반경", self.radius_spin)
        layout.addLayout(form)
        layout.addWidget(self.overlay_checkbox)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Angle Cal - SEM Angle Measurement")
        self.resize(1280, 820)
        self.image_bgr: Optional[np.ndarray] = None
        self.image_path: Optional[str] = None
        self.project_path: Optional[str] = None
        self.nm_per_px: Optional[float] = None
        self.records: dict[str, LineRecord] = {}
        self._counter = 1
        self.last_measurements: list[dict[str, str | float]] = []

        self.canvas = AngleCanvas()
        self.setCentralWidget(self.canvas)
        self.canvas.line_created.connect(self._handle_line_created)
        self.canvas.scene_changed.connect(self._handle_scene_changed)

        self._build_actions()
        self._build_toolbar()
        self._build_measurements_dock()
        self.setStatusBar(QStatusBar())
        self._set_status("이미지를 불러오면 시작할 수 있습니다.")

    def _build_actions(self) -> None:
        self.open_action = QAction("이미지 열기", self)
        self.open_action.triggered.connect(self.open_image)
        self.save_project_action = QAction("프로젝트 저장", self)
        self.save_project_action.triggered.connect(self.save_project)
        self.open_project_action = QAction("프로젝트 열기", self)
        self.open_project_action.triggered.connect(self.open_project)
        self.export_png_action = QAction("주석 PNG 내보내기", self)
        self.export_png_action.triggered.connect(self.export_annotated_png)
        self.export_csv_action = QAction("CSV 내보내기", self)
        self.export_csv_action.triggered.connect(self.export_csv)

        file_menu = self.menuBar().addMenu("파일")
        file_menu.addAction(self.open_action)
        file_menu.addAction(self.open_project_action)
        file_menu.addAction(self.save_project_action)
        file_menu.addSeparator()
        file_menu.addAction(self.export_png_action)
        file_menu.addAction(self.export_csv_action)

    def _build_toolbar(self) -> None:
        toolbar = QToolBar("Tools")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)
        toolbar.addAction(self.open_action)

        self.tool_combo = QComboBox()
        self.tool_combo.addItem("선택", "select")
        self.tool_combo.addItem("이동", "pan")
        self.tool_combo.addItem("스케일바", "scale")
        self.tool_combo.addItem("기준선", "reference")
        self.tool_combo.addItem("경계선", "edge")
        self.tool_combo.currentIndexChanged.connect(self._tool_changed)
        toolbar.addWidget(QLabel(" 도구 "))
        toolbar.addWidget(self.tool_combo)

        self.axis_combo = QComboBox()
        self.axis_combo.addItem("수평 기준", "horizontal")
        self.axis_combo.addItem("수직 기준", "vertical")
        toolbar.addWidget(QLabel(" 기준 "))
        toolbar.addWidget(self.axis_combo)

        align_button = QPushButton("이미지 맞춤")
        align_button.clicked.connect(self.align_to_reference)
        toolbar.addWidget(align_button)

        self.search_radius_spin = QSpinBox()
        self.search_radius_spin.setRange(2, 300)
        self.search_radius_spin.setValue(35)
        self.search_radius_spin.setSuffix(" px")
        self.search_radius_spin.valueChanged.connect(self._edge_detection_settings_changed)
        toolbar.addWidget(QLabel(" 탐색 "))
        toolbar.addWidget(self.search_radius_spin)

        self.show_search_range_checkbox = QCheckBox("범위 표시")
        self.show_search_range_checkbox.setChecked(True)
        self.show_search_range_checkbox.toggled.connect(self._edge_detection_settings_changed)
        toolbar.addWidget(self.show_search_range_checkbox)

        settings_button = QPushButton("인식 설정")
        settings_button.clicked.connect(self.open_edge_detection_settings)
        toolbar.addWidget(settings_button)

        recognize_button = QPushButton("인식")
        recognize_button.clicked.connect(self.recognize_edges)
        toolbar.addWidget(recognize_button)

        self.guide_orientation_combo = QComboBox()
        self.guide_orientation_combo.addItem("수평선", "horizontal")
        self.guide_orientation_combo.addItem("수직선", "vertical")
        self.guide_spacing_spin = QSpinBox()
        self.guide_spacing_spin.setRange(1, 100000)
        self.guide_spacing_spin.setValue(50)
        self.guide_spacing_unit = QComboBox()
        self.guide_spacing_unit.addItem("px", "px")
        self.guide_spacing_unit.addItem("nm", "nm")
        self.guide_offset_spin = QSpinBox()
        self.guide_offset_spin.setRange(0, 100000)
        self.guide_offset_spin.setValue(0)
        self.guide_offset_spin.setSuffix(" px")
        toolbar.addSeparator()
        toolbar.addWidget(QLabel(" 가이드 "))
        toolbar.addWidget(self.guide_orientation_combo)
        toolbar.addWidget(self.guide_spacing_spin)
        toolbar.addWidget(self.guide_spacing_unit)
        toolbar.addWidget(QLabel(" 시작 "))
        toolbar.addWidget(self.guide_offset_spin)

        add_guides_button = QPushButton("그리기")
        add_guides_button.clicked.connect(self.add_guides)
        clear_guides_button = QPushButton("가이드 지우기")
        clear_guides_button.clicked.connect(self.clear_guides)
        angle_button = QPushButton("각도 계산")
        angle_button.clicked.connect(self.calculate_angles)
        toolbar.addWidget(add_guides_button)
        toolbar.addWidget(clear_guides_button)
        toolbar.addWidget(angle_button)

    def _build_measurements_dock(self) -> None:
        dock = QDockWidget("측정값", self)
        dock.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)
        container = QWidget()
        layout = QVBoxLayout(container)
        self.calibration_label = QLabel("Calibration: -")
        self.measurement_table = QTableWidget(0, 5)
        self.measurement_table.setHorizontalHeaderLabels(["ID", "종류", "길이(px)", "길이(nm)", "각도"])
        self.measurement_table.verticalHeader().setVisible(False)
        layout.addWidget(self.calibration_label)
        layout.addWidget(self.measurement_table)

        controls = QHBoxLayout()
        delete_button = QPushButton("선택 삭제")
        delete_button.clicked.connect(self.delete_selected)
        reset_button = QPushButton("화면 맞춤")
        reset_button.clicked.connect(lambda: self.canvas.fitInView(self.canvas.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio))
        controls.addWidget(delete_button)
        controls.addWidget(reset_button)
        layout.addLayout(controls)
        dock.setWidget(container)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)

    def open_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "이미지 열기",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;All files (*.*)",
        )
        if not path:
            return
        image = read_image(path)
        if image is None:
            QMessageBox.warning(self, "열기 실패", "이미지를 읽을 수 없습니다.")
            return
        self.image_bgr = image
        self.image_path = path
        self.project_path = None
        self.nm_per_px = None
        self.records.clear()
        self._counter = 1
        self._show_image()
        self._refresh_table()
        self._update_search_range_overlay()
        self._set_status(f"이미지 로드: {Path(path).name} ({image.shape[1]} x {image.shape[0]} px)")

    def open_project(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "프로젝트 열기", "", "Angle Cal Project (*.anglecal.json)")
        if not path:
            return
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        image_path = payload.get("image_path")
        if not image_path or not Path(image_path).exists():
            QMessageBox.warning(self, "프로젝트 열기", "프로젝트에 기록된 이미지 경로를 찾을 수 없습니다.")
            return
        image = read_image(image_path)
        if image is None:
            QMessageBox.warning(self, "프로젝트 열기", "이미지를 읽을 수 없습니다.")
            return
        self.image_bgr = image
        self.image_path = image_path
        self.project_path = path
        self.nm_per_px = payload.get("nm_per_px")
        edge_detection = payload.get("edge_detection", {})
        self.search_radius_spin.setValue(int(edge_detection.get("search_radius_px", self.search_radius_spin.value())))
        self.show_search_range_checkbox.setChecked(bool(edge_detection.get("show_search_range", True)))
        self.records = {
            item["id"]: LineRecord(
                id=item["id"],
                kind=item["kind"],
                start=tuple(item["start"]),
                end=tuple(item["end"]),
                label=item.get("label", ""),
                axis=item.get("axis", "horizontal"),
                value_nm=item.get("value_nm"),
            )
            for item in payload.get("records", [])
        }
        self._counter = payload.get("counter", len(self.records) + 1)
        self._show_image()
        self.canvas.redraw_lines(list(self.records.values()))
        self._refresh_table()
        self._update_search_range_overlay()
        self._set_status(f"프로젝트 로드: {Path(path).name}")

    def save_project(self) -> None:
        if self.image_bgr is None:
            return
        self._sync_records_from_canvas()
        path = self.project_path
        if not path:
            path, _ = QFileDialog.getSaveFileName(self, "프로젝트 저장", "", "Angle Cal Project (*.anglecal.json)")
            if not path:
                return
            if not path.endswith(".anglecal.json"):
                path += ".anglecal.json"
        payload = {
            "image_path": self.image_path,
            "nm_per_px": self.nm_per_px,
            "edge_detection": {
                "search_radius_px": self.search_radius_spin.value(),
                "show_search_range": self.show_search_range_checkbox.isChecked(),
            },
            "counter": self._counter,
            "records": [asdict(record) for record in self.records.values()],
        }
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        self.project_path = path
        self._set_status(f"프로젝트 저장: {Path(path).name}")

    def export_annotated_png(self) -> None:
        if self.image_bgr is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "주석 PNG 내보내기", "", "PNG Image (*.png)")
        if not path:
            return
        if not path.lower().endswith(".png"):
            path += ".png"
        self.canvas.export_scene_png(path)
        self._set_status(f"PNG 저장: {Path(path).name}")

    def export_csv(self) -> None:
        if self.image_bgr is None:
            return
        self.calculate_angles()
        path, _ = QFileDialog.getSaveFileName(self, "CSV 내보내기", "", "CSV (*.csv)")
        if not path:
            return
        if not path.lower().endswith(".csv"):
            path += ".csv"
        with open(path, "w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "measurement",
                    "edge_id",
                    "guide_id",
                    "kind",
                    "x_px",
                    "y_px",
                    "angle_deg",
                    "edge_length_px",
                    "edge_length_nm",
                    "nm_per_px",
                ],
            )
            writer.writeheader()
            for row in self.last_measurements:
                writer.writerow(row)
        self._set_status(f"CSV 저장: {Path(path).name}")

    def _handle_line_created(self, tool: str, start: Point, end: Point) -> None:
        if tool == "scale":
            self._create_scale_line(start, end)
        elif tool == "reference":
            self._create_reference_line(start, end)
        elif tool == "edge":
            self._create_edge_line(start, end)
        self.canvas.redraw_lines(list(self.records.values()))
        self._refresh_table()
        self._update_search_range_overlay()

    def _create_scale_line(self, start: Point, end: Point) -> None:
        length_px = line_length(start, end)
        value, ok = QInputDialog.getDouble(
            self,
            "스케일바 길이",
            f"그은 선 길이: {length_px:.2f} px\n실제 길이(nm)를 입력하세요.",
            100.0,
            0.0001,
            1_000_000.0,
            4,
        )
        if not ok:
            return
        for record_id in [rid for rid, record in self.records.items() if record.kind == "scale"]:
            del self.records[record_id]
        record = LineRecord(
            id=self._next_id("S"),
            kind="scale",
            start=start,
            end=end,
            label=f"{value:g} nm",
            value_nm=float(value),
        )
        self.records[record.id] = record
        self.nm_per_px = float(value) / length_px
        self._set_status(f"Calibration: {self.nm_per_px:.6g} nm/px")

    def _create_reference_line(self, start: Point, end: Point) -> None:
        for record_id in [rid for rid, record in self.records.items() if record.kind == "reference"]:
            del self.records[record_id]
        axis = self.axis_combo.currentData()
        record = LineRecord(
            id=self._next_id("R"),
            kind="reference",
            start=start,
            end=end,
            axis=axis,
            label="horizontal" if axis == "horizontal" else "vertical",
        )
        self.records[record.id] = record
        self._set_status("기준선을 만들었습니다. 필요하면 '이미지 맞춤'을 누르세요.")

    def _create_edge_line(self, start: Point, end: Point) -> None:
        record = LineRecord(
            id=self._next_id("E"),
            kind="edge",
            start=start,
            end=end,
            label="edge",
            axis=self.axis_combo.currentData(),
        )
        self.records[record.id] = record
        self._set_status("경계선을 추가했습니다.")

    def align_to_reference(self) -> None:
        if self.image_bgr is None:
            return
        self._sync_records_from_canvas()
        reference = self._reference_record()
        if reference is None:
            QMessageBox.information(self, "이미지 맞춤", "먼저 기준선을 그려주세요.")
            return
        angle = line_angle_degrees(reference.start, reference.end)
        target = 0.0 if reference.axis == "horizontal" else 90.0
        rotate_by = angle - target
        lines = list(self.records.values())
        points = []
        for record in lines:
            points.extend([record.start, record.end])
        rotated, transformed = rotate_image_and_points(self.image_bgr, points, rotate_by)
        for idx, record in enumerate(lines):
            record.start = transformed[idx * 2]
            record.end = transformed[idx * 2 + 1]
        self.image_bgr = rotated
        self._show_image(keep_view=False)
        self.canvas.redraw_lines(list(self.records.values()))
        self.calculate_angles()
        self._update_search_range_overlay()
        self._set_status(f"이미지를 {rotate_by:.3f}도 회전해 기준을 맞췄습니다.")

    def recognize_edges(self) -> None:
        if self.image_bgr is None:
            return
        self._sync_records_from_canvas()
        selected_ids = set(self.canvas.selected_line_ids())
        edge_records = [record for record in self.records.values() if record.kind == "edge"]
        if selected_ids:
            edge_records = [record for record in edge_records if record.id in selected_ids]
        if not edge_records:
            QMessageBox.information(self, "인식", "인식할 경계선이 없습니다.")
            return
        gray = to_gray(self.image_bgr)
        radius = self.search_radius_spin.value()
        moved = 0
        for record in edge_records:
            result = snap_line_to_gradient(gray, record.start, record.end, radius)
            if result is not None:
                record.start = result.start
                record.end = result.end
                moved += 1
        self.canvas.redraw_lines(list(self.records.values()))
        self.calculate_angles()
        self._update_search_range_overlay()
        self._set_status(f"{moved}/{len(edge_records)}개 경계선을 명도 변화 최대 위치로 이동했습니다.")

    def add_guides(self) -> None:
        if self.image_bgr is None:
            return
        self._sync_records_from_canvas()
        orientation = self.guide_orientation_combo.currentData()
        spacing = float(self.guide_spacing_spin.value())
        if self.guide_spacing_unit.currentData() == "nm":
            if not self.nm_per_px:
                QMessageBox.information(self, "가이드", "nm 간격을 쓰려면 먼저 스케일바를 캘리브레이션하세요.")
                return
            spacing = spacing / self.nm_per_px
        if spacing < 1:
            QMessageBox.information(self, "가이드", "간격이 너무 작습니다.")
            return
        self.clear_guides(redraw=False)
        height, width = self.image_bgr.shape[:2]
        offset = float(self.guide_offset_spin.value())
        count = int((height if orientation == "horizontal" else width) / spacing) + 2
        if count > 500:
            QMessageBox.information(self, "가이드", "가이드가 너무 많습니다. 간격을 키워주세요.")
            return
        for idx in range(count):
            pos = offset + idx * spacing
            if orientation == "horizontal":
                if pos > height:
                    break
                start, end = (0.0, pos), (float(width), pos)
            else:
                if pos > width:
                    break
                start, end = (pos, 0.0), (pos, float(height))
            record = LineRecord(
                id=self._next_id("G"),
                kind="guide",
                start=start,
                end=end,
                label=f"{orientation} guide",
                axis=orientation,
            )
            self.records[record.id] = record
        self.canvas.redraw_lines(list(self.records.values()))
        self.calculate_angles()
        self._update_search_range_overlay()
        self._set_status(f"{orientation} 가이드 {count}개를 만들었습니다.")

    def clear_guides(self, redraw: bool = True) -> None:
        for record_id in [rid for rid, record in self.records.items() if record.kind == "guide"]:
            del self.records[record_id]
        self.canvas.clear_angle_items()
        if redraw:
            self.canvas.redraw_lines(list(self.records.values()))
            self._refresh_table()
            self._update_search_range_overlay()
            self._set_status("가이드를 지웠습니다.")

    def calculate_angles(self) -> None:
        if self.image_bgr is None:
            return
        self._sync_records_from_canvas()
        self.canvas.clear_angle_items()
        self.last_measurements = []

        reference = self._reference_record()
        if reference is not None:
            reference_angle = line_angle_degrees(reference.start, reference.end)
            reference_name = reference.id
        else:
            axis = self.axis_combo.currentData()
            reference_angle = 0.0 if axis == "horizontal" else 90.0
            reference_name = axis

        edges = [record for record in self.records.values() if record.kind == "edge"]
        guides = [record for record in self.records.values() if record.kind == "guide"]
        for edge in edges:
            edge_angle = line_angle_degrees(edge.start, edge.end)
            angle = acute_angle_difference(edge_angle, reference_angle)
            midpoint = ((edge.start[0] + edge.end[0]) / 2.0, (edge.start[1] + edge.end[1]) / 2.0)
            label_pos = self._label_position(midpoint, edge_angle, reference_angle, 34.0)
            self.canvas.add_angle_label(f"{angle:.2f}°", label_pos)
            length_px = line_length(edge.start, edge.end)
            self.last_measurements.append(
                {
                    "measurement": f"{edge.id}_to_{reference_name}",
                    "edge_id": edge.id,
                    "guide_id": "",
                    "kind": "edge_to_reference",
                    "x_px": midpoint[0],
                    "y_px": midpoint[1],
                    "angle_deg": angle,
                    "edge_length_px": length_px,
                    "edge_length_nm": length_px * self.nm_per_px if self.nm_per_px else "",
                    "nm_per_px": self.nm_per_px if self.nm_per_px else "",
                }
            )

        for edge in edges:
            edge_angle = line_angle_degrees(edge.start, edge.end)
            for guide in guides:
                cross = intersection((edge.start, edge.end), (guide.start, guide.end))
                if cross is None:
                    continue
                guide_angle = line_angle_degrees(guide.start, guide.end)
                angle = acute_angle_difference(edge_angle, guide_angle)
                self.canvas.add_angle_arc(cross, edge_angle, guide_angle)
                label_pos = self._label_position(cross, edge_angle, guide_angle, 42.0)
                self.canvas.add_angle_label(f"{angle:.2f}°", label_pos)
                length_px = line_length(edge.start, edge.end)
                self.last_measurements.append(
                    {
                        "measurement": f"{edge.id}_x_{guide.id}",
                        "edge_id": edge.id,
                        "guide_id": guide.id,
                        "kind": "edge_guide_intersection",
                        "x_px": cross[0],
                        "y_px": cross[1],
                        "angle_deg": angle,
                        "edge_length_px": length_px,
                        "edge_length_nm": length_px * self.nm_per_px if self.nm_per_px else "",
                        "nm_per_px": self.nm_per_px if self.nm_per_px else "",
                    }
                )
        self._refresh_table()
        self._set_status(f"각도 {len(self.last_measurements)}개를 계산했습니다. 숫자 라벨은 선택해서 옮길 수 있습니다.")

    def delete_selected(self) -> None:
        selected = set(self.canvas.selected_line_ids())
        if not selected:
            return
        for record_id in selected:
            self.records.pop(record_id, None)
        self.canvas.redraw_lines(list(self.records.values()))
        self.calculate_angles()
        self._update_search_range_overlay()

    def _show_image(self, keep_view: bool = False) -> None:
        if self.image_bgr is None:
            return
        transform = self.canvas.transform() if keep_view else None
        h, w = self.image_bgr.shape[:2]
        rgb = bgr_to_rgb8_for_display(self.image_bgr)
        qimage = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format.Format_RGB888).copy()
        self.canvas.set_image(QPixmap.fromImage(qimage))
        if keep_view and transform is not None:
            self.canvas.setTransform(transform)

    def _tool_changed(self) -> None:
        self.canvas.set_tool(self.tool_combo.currentData())

    def open_edge_detection_settings(self) -> None:
        dialog = EdgeDetectionSettingsDialog(
            self.search_radius_spin.value(),
            self.show_search_range_checkbox.isChecked(),
            self,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        self.search_radius_spin.setValue(dialog.radius_spin.value())
        self.show_search_range_checkbox.setChecked(dialog.overlay_checkbox.isChecked())
        self._edge_detection_settings_changed()

    def _edge_detection_settings_changed(self) -> None:
        self._update_search_range_overlay()
        radius = self.search_radius_spin.value()
        if self.show_search_range_checkbox.isChecked():
            self._set_status(f"명도 인식 탐색 범위: 경계선 양쪽 {radius}px")
        else:
            self._set_status(f"명도 인식 탐색 범위: 경계선 양쪽 {radius}px, 표시 꺼짐")

    def _update_search_range_overlay(self) -> None:
        self._sync_records_from_canvas()
        self.canvas.set_search_range(
            self.search_radius_spin.value(),
            self.show_search_range_checkbox.isChecked(),
            list(self.records.values()),
        )

    def _handle_scene_changed(self) -> None:
        self._refresh_table()
        self._update_search_range_overlay()

    def _sync_records_from_canvas(self) -> None:
        for record_id, item in self.canvas.line_items.items():
            if record_id not in self.records:
                continue
            line = item.line()
            p1 = item.mapToScene(line.p1())
            p2 = item.mapToScene(line.p2())
            self.records[record_id].start = (float(p1.x()), float(p1.y()))
            self.records[record_id].end = (float(p2.x()), float(p2.y()))

    def _reference_record(self) -> Optional[LineRecord]:
        for record in self.records.values():
            if record.kind == "reference":
                return record
        return None

    def _refresh_table(self) -> None:
        self._sync_records_from_canvas()
        rows = [record for record in self.records.values() if record.kind != "guide"]
        self.measurement_table.setRowCount(len(rows))
        for row_idx, record in enumerate(rows):
            length_px = line_length(record.start, record.end)
            length_nm = length_px * self.nm_per_px if self.nm_per_px else None
            if record.kind == "edge":
                reference = self._reference_record()
                if reference is not None:
                    angle = acute_angle_difference(
                        line_angle_degrees(record.start, record.end),
                        line_angle_degrees(reference.start, reference.end),
                    )
                else:
                    angle = angle_to_axis(record.start, record.end, self.axis_combo.currentData())
                angle_text = f"{angle:.2f}°"
            elif record.kind == "reference":
                angle_text = f"{line_angle_degrees(record.start, record.end):.2f}°"
            else:
                angle_text = ""
            values = [
                record.id,
                record.kind,
                f"{length_px:.2f}",
                f"{length_nm:.2f}" if length_nm is not None else "",
                angle_text,
            ]
            for col_idx, value in enumerate(values):
                self.measurement_table.setItem(row_idx, col_idx, QTableWidgetItem(value))
        self.measurement_table.resizeColumnsToContents()
        self.calibration_label.setText(
            f"Calibration: {self.nm_per_px:.6g} nm/px" if self.nm_per_px else "Calibration: -"
        )

    def _next_id(self, prefix: str) -> str:
        value = f"{prefix}{self._counter}"
        self._counter += 1
        return value

    @staticmethod
    def _label_position(center: Point, angle_a: float, angle_b: float, distance: float) -> Point:
        delta = ((angle_b - angle_a + 90.0) % 180.0) - 90.0
        bisector = math.radians(angle_a + delta / 2.0)
        x = center[0] + math.cos(bisector) * distance
        y = center[1] + math.sin(bisector) * distance
        return (x, y)

    def _set_status(self, text: str) -> None:
        self.statusBar().showMessage(text, 8000)


def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName("Angle Cal")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
