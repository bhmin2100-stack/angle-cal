from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import json
import math
from pathlib import Path
import sys
from typing import Optional

import numpy as np
from PySide6.QtCore import QPoint, QPointF, QRectF, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QAction, QBrush, QColor, QIcon, QImage, QKeySequence, QPainter, QPainterPath, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDockWidget,
    QFormLayout,
    QGraphicsItem,
    QGridLayout,
    QFileDialog,
    QFrame,
    QGraphicsEllipseItem,
    QGraphicsLineItem,
    QGraphicsPathItem,
    QGraphicsPixmapItem,
    QGraphicsPolygonItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsTextItem,
    QGraphicsView,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QDoubleSpinBox,
    QPushButton,
    QScrollArea,
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
    bgr_to_rgb8_for_display,
    intersection,
    line_angle_degrees,
    line_length,
    normal_for_line,
    read_image,
    rotate_image_and_points,
    snap_polyline_to_gradient,
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
    points: Optional[list[Point]] = None
    recognition_points: Optional[list[Point]] = None
    edge_mode: str = "line"
    search_radius_px: Optional[int] = None
    segment_size_px: Optional[int] = None
    angle_sector: int = 0
    angle_arc_radius: float = 28.0
    angle_label_side: str = "outside"
    angle_label_gap: float = 14.0
    edge_segmented: bool = False
    object_group: Optional[str] = None
    show_line: bool = True
    show_angle: bool = True
    show_range: bool = True
    show_range_label: bool = True
    show_edge_length: bool = True
    edge_length_label_pos: Optional[Point] = None
    stroke_color: Optional[str] = None
    stroke_width: Optional[float] = None


@dataclass
class ScalePreset:
    name: str
    nm_per_px: float
    bar_px: float = 100.0
    bar_nm: Optional[float] = None
    start: Optional[Point] = None
    end: Optional[Point] = None


@dataclass
class StructureTemplate:
    name: str
    records: list[LineRecord]
    cd_segment_mode: str = "all"


ANGLE_GROUP_KEY = 1
ANGLE_MEASUREMENT_KEY = 2
LENGTH_GROUP_KEY = 3
LENGTH_PARENT_KEY = 4
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


class AnnotationLineItem(QGraphicsLineItem):
    def __init__(self, record: LineRecord, pen: QPen):
        super().__init__(record.start[0], record.start[1], record.end[0], record.end[1])
        self.record_id = record.id
        self.kind = record.kind
        self.setPen(pen)
        self.setZValue(10 if record.kind != "guide" else 4)
        if record.kind in {"edge", "scale", "guide"}:
            self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
            self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)


class AnnotationCurveItem(QGraphicsPathItem):
    def __init__(self, record: LineRecord, pen: QPen):
        super().__init__(path_from_points(record.points or [record.start, record.end], smooth=False))
        self.record_id = record.id
        self.kind = record.kind
        self.anchor_points = list(record.points or [record.start, record.end])
        self.setPen(pen)
        self.setZValue(10)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)


class PointHandleItem(QGraphicsEllipseItem):
    def __init__(self, owner: AnnotationLineItem | AnnotationCurveItem, point_index: int, pos: Point, canvas: "AngleCanvas"):
        radius = 3.0
        super().__init__(-radius, -radius, radius * 2, radius * 2)
        self.owner = owner
        self.point_index = point_index
        self.canvas = canvas
        self.setPos(pos[0], pos[1])
        self.setBrush(QBrush(QColor("#ffffff")))
        self.setPen(QPen(QColor("#ffb703"), 1.2))
        self.setZValue(55)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges, True)
        self.setCursor(Qt.CursorShape.CrossCursor)

    def itemChange(self, change, value):  # noqa: N802
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            self.canvas.update_owner_point_from_handle(self)
        return super().itemChange(change, value)


def path_from_points(points: list[Point], smooth: bool = True) -> QPainterPath:
    path = QPainterPath()
    if not points:
        return path
    path.moveTo(points[0][0], points[0][1])
    if len(points) == 2 or not smooth:
        for point in points[1:]:
            path.lineTo(point[0], point[1])
        return path
    for idx in range(len(points) - 1):
        p0 = points[max(0, idx - 1)]
        p1 = points[idx]
        p2 = points[idx + 1]
        p3 = points[min(len(points) - 1, idx + 2)]
        c1 = (p1[0] + (p2[0] - p0[0]) / 6.0, p1[1] + (p2[1] - p0[1]) / 6.0)
        c2 = (p2[0] - (p3[0] - p1[0]) / 6.0, p2[1] - (p3[1] - p1[1]) / 6.0)
        path.cubicTo(c1[0], c1[1], c2[0], c2[1], p2[0], p2[1])
    return path


def points_from_path_item(item: QGraphicsPathItem | QGraphicsLineItem) -> list[Point]:
    if isinstance(item, QGraphicsLineItem):
        line = item.line()
        p1 = item.mapToScene(line.p1())
        p2 = item.mapToScene(line.p2())
        return [(float(p1.x()), float(p1.y())), (float(p2.x()), float(p2.y()))]
    anchor_points = getattr(item, "anchor_points", None)
    if anchor_points:
        return [
            (float(item.mapToScene(QPointF(point[0], point[1])).x()), float(item.mapToScene(QPointF(point[0], point[1])).y()))
            for point in anchor_points
        ]
    path = item.path()
    points: list[Point] = []
    for idx in range(path.elementCount()):
        element = path.elementAt(idx)
        scene_point = item.mapToScene(QPointF(element.x, element.y))
        points.append((float(scene_point.x()), float(scene_point.y())))
    return points


class AngleCanvas(QGraphicsView):
    line_created = Signal(str, tuple, tuple, object)
    segment_split_requested = Signal(str, int)
    scene_changed = Signal()
    scale_requested = Signal(float)
    edit_started = Signal()
    temporary_edge_tool_changed = Signal(bool)

    def __init__(self):
        super().__init__()
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.BoundingRectViewportUpdate)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)

        self.pixmap_item: Optional[QGraphicsPixmapItem] = None
        self.line_items: dict[str, AnnotationLineItem | AnnotationCurveItem] = {}
        self.angle_items: list[QGraphicsPathItem | QGraphicsTextItem] = []
        self.cd_items: list[QGraphicsItem] = []
        self.edge_length_items: list[QGraphicsItem] = []
        self.group_box_items: list[QGraphicsItem] = []
        self.search_range_band_items: list[QGraphicsItem] = []
        self.search_range_label_items: list[QGraphicsItem] = []
        self.detection_preview_items: list[QGraphicsItem] = []
        self.point_handle_items: list[PointHandleItem] = []
        self.angle_groups: dict[str, list[QGraphicsItem]] = {}
        self.angle_group_parents: dict[str, str] = {}
        self.angle_group_measurements: dict[str, str] = {}
        self.edge_length_groups: dict[str, list[QGraphicsItem]] = {}
        self.edge_length_group_parents: dict[str, str] = {}
        self._angle_counter = 1
        self.search_range_radius_px = 35
        self.show_search_range = True
        self.show_search_range_band = True
        self.show_point_handles = True
        self.current_tool = "select"
        self._drawing_start: Optional[QPointF] = None
        self._temp_line: Optional[QGraphicsLineItem] = None
        self.edge_draw_mode = "line"
        self._curve_points: list[QPointF] = []
        self._temp_curve: Optional[QGraphicsPathItem] = None
        self._panning = False
        self._pan_last = QPoint()
        self._resizing = False
        self._resize_last = QPoint()
        self._selection_filter: Optional[str] = None
        self._filtering_selection = False
        self._expanding_angle_selection = False
        self._refreshing_point_handles = False
        self._updating_from_point_handle = False
        self._space_edge_previous_tool: Optional[str] = None
        self.scene.selectionChanged.connect(self._expand_angle_group_selection)
        self.scene.selectionChanged.connect(self.refresh_point_handles)

    def set_tool(self, tool: str) -> None:
        self.current_tool = tool
        if tool != "edge" or self.edge_draw_mode != "polyline":
            self._clear_curve_preview()
        if tool == "pan":
            self.setCursor(Qt.CursorShape.OpenHandCursor)
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
        elif tool == "resize":
            self.setCursor(Qt.CursorShape.SizeFDiagCursor)
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
        else:
            self.unsetCursor()
            self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)

    def set_edge_draw_mode(self, mode: str) -> None:
        self.edge_draw_mode = mode
        self.cancel_interaction()

    def cancel_interaction(self) -> None:
        if self._temp_line is not None:
            self.scene.removeItem(self._temp_line)
            self._temp_line = None
        self._clear_curve_preview()
        self._drawing_start = None
        self._panning = False
        self._resizing = False
        self._restore_tool_cursor()

    def _restore_tool_cursor(self) -> None:
        if self.current_tool == "pan":
            self.setCursor(Qt.CursorShape.OpenHandCursor)
        elif self.current_tool == "resize":
            self.setCursor(Qt.CursorShape.SizeFDiagCursor)
        else:
            self.unsetCursor()

    def set_image(self, pixmap: QPixmap) -> None:
        self.scene.clear()
        self.line_items.clear()
        self.angle_items.clear()
        self.cd_items.clear()
        self.edge_length_items.clear()
        self.group_box_items.clear()
        self.search_range_band_items.clear()
        self.search_range_label_items.clear()
        self.detection_preview_items.clear()
        self.point_handle_items.clear()
        self.angle_groups.clear()
        self.angle_group_parents.clear()
        self.angle_group_measurements.clear()
        self.edge_length_groups.clear()
        self.edge_length_group_parents.clear()
        self.pixmap_item = self.scene.addPixmap(pixmap)
        self.pixmap_item.setZValue(0)
        self.scene.setSceneRect(QRectF(0, 0, pixmap.width(), pixmap.height()))
        self.resetTransform()
        self.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def redraw_lines(self, records: list[LineRecord]) -> None:
        self.clear_point_handles()
        for item in list(self.line_items.values()):
            self.scene.removeItem(item)
        self.line_items.clear()
        for record in records:
            if record.kind == "edge" and record.points and len(record.points) >= 2:
                item = AnnotationCurveItem(record, self._pen_for_record(record))
            else:
                item = AnnotationLineItem(record, self._pen_for_record(record))
            self.scene.addItem(item)
            self.line_items[record.id] = item
        self.update_group_boxes(records)
        self.refresh_point_handles()

    def clear_group_boxes(self) -> None:
        for item in self.group_box_items:
            self.scene.removeItem(item)
        self.group_box_items.clear()

    def update_group_boxes(self, records: list[LineRecord]) -> None:
        self.clear_group_boxes()
        grouped: dict[str, list[LineRecord]] = {}
        for record in records:
            if record.object_group and record.id in self.line_items:
                grouped.setdefault(record.object_group, []).append(record)
        for group_records in grouped.values():
            if len(group_records) < 2:
                continue
            rect: Optional[QRectF] = None
            for record in group_records:
                item = self.line_items.get(record.id)
                if item is None or not item.isVisible():
                    continue
                item_rect = item.sceneBoundingRect()
                rect = item_rect if rect is None else rect.united(item_rect)
            if rect is None:
                continue
            rect = rect.adjusted(-3.0, -3.0, 3.0, 3.0)
            box = QGraphicsRectItem(rect)
            box.setPen(QPen(QColor(255, 214, 102, 175), 0.8, Qt.PenStyle.DashLine))
            box.setBrush(QBrush(Qt.BrushStyle.NoBrush))
            box.setZValue(9)
            box.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, False)
            self.scene.addItem(box)
            self.group_box_items.append(box)

    def clear_point_handles(self) -> None:
        handle_items = list(self.point_handle_items)
        handle_items.extend(item for item in self.scene.items() if isinstance(item, PointHandleItem) and item not in handle_items)
        for item in handle_items:
            self.scene.removeItem(item)
        self.point_handle_items.clear()

    def refresh_point_handles(self) -> None:
        if self._refreshing_point_handles:
            return
        self._refreshing_point_handles = True
        try:
            if hasattr(self, "show_point_handles") and not self.show_point_handles:
                self.clear_point_handles()
                return
            if self.selected_point_handles():
                return
            selected_items = [
                item
                for item in self.scene.selectedItems()
                if isinstance(item, (AnnotationLineItem, AnnotationCurveItem)) and item.kind in {"edge", "scale", "reference", "guide"}
            ]
            self.clear_point_handles()
            for item in selected_items:
                for idx, point in enumerate(points_from_path_item(item)):
                    handle = PointHandleItem(item, idx, point, self)
                    self.scene.addItem(handle)
                    self.point_handle_items.append(handle)
        finally:
            self._refreshing_point_handles = False

    def set_point_handles_visible(self, visible: bool) -> None:
        self.show_point_handles = visible
        if visible:
            self.refresh_point_handles()
        else:
            self.clear_point_handles()

    def update_owner_point_from_handle(self, handle: PointHandleItem) -> None:
        if self._updating_from_point_handle:
            return
        self._updating_from_point_handle = True
        try:
            scene_pos = handle.scenePos()
            owner = handle.owner
            if isinstance(owner, AnnotationLineItem):
                line = owner.line()
                local_pos = owner.mapFromScene(scene_pos)
                if handle.point_index == 0:
                    line.setP1(local_pos)
                else:
                    line.setP2(local_pos)
                owner.setLine(line)
            elif isinstance(owner, AnnotationCurveItem):
                points = list(owner.anchor_points)
                if 0 <= handle.point_index < len(points):
                    local_pos = owner.mapFromScene(scene_pos)
                    points[handle.point_index] = (float(local_pos.x()), float(local_pos.y()))
                    owner.anchor_points = points
                    owner.setPath(path_from_points(points, smooth=False))
            self.scene_changed.emit()
        finally:
            self._updating_from_point_handle = False

    def selected_point_handles(self) -> list[PointHandleItem]:
        return [item for item in self.scene.selectedItems() if isinstance(item, PointHandleItem)]

    def delete_selected_point_handles(self) -> bool:
        handles = sorted(self.selected_point_handles(), key=lambda item: item.point_index, reverse=True)
        if not handles:
            return False
        self.edit_started.emit()
        changed = False
        for handle in handles:
            owner = handle.owner
            if not isinstance(owner, AnnotationCurveItem):
                continue
            points = list(owner.anchor_points)
            if len(points) <= 2 or not (0 <= handle.point_index < len(points)):
                continue
            del points[handle.point_index]
            owner.anchor_points = points
            owner.setPath(path_from_points(points, smooth=False))
            changed = True
        if changed:
            self.refresh_point_handles()
            self.scene_changed.emit()
        return changed

    def set_search_range(
        self,
        radius_px: int,
        visible: bool,
        band_visible: bool,
        records: list[LineRecord],
    ) -> None:
        self.search_range_radius_px = radius_px
        self.show_search_range = visible
        self.show_search_range_band = band_visible
        self.update_search_range_overlay(records)

    def show_detection_preview(self, radius_px: int, segment_value: int, range_label: str) -> None:
        self.clear_detection_preview()
        if self.pixmap_item is None:
            return
        rect = self.scene.sceneRect()
        if rect.width() <= 0 or rect.height() <= 0:
            return
        width = min(190.0, max(120.0, rect.width() * 0.22))
        height = 72.0
        visible_center = self.mapToScene(self.viewport().rect().center())
        left = float(visible_center.x()) - width / 2.0
        top = float(visible_center.y()) - height / 2.0
        left = min(max(left, rect.left() + 8.0), rect.right() - width - 8.0)
        top = min(max(top, rect.top() + 8.0), rect.bottom() - height - 8.0)
        panel = QGraphicsPolygonItem(
            QPolygonF(
                [
                    QPointF(left, top),
                    QPointF(left + width, top),
                    QPointF(left + width, top + height),
                    QPointF(left, top + height),
                ]
            )
        )
        panel.setBrush(QBrush(QColor(18, 24, 28, 185)))
        panel.setPen(QPen(QColor(76, 201, 240, 190), 1.0))
        panel.setZValue(60)
        self.scene.addItem(panel)
        self.detection_preview_items.append(panel)

        center_y = top + height * 0.56
        line_start = (left + 22.0, center_y)
        line_end = (left + width - 22.0, center_y)
        radius = min(float(radius_px), height * 0.32)
        band = QGraphicsPolygonItem(
            QPolygonF(
                [
                    QPointF(line_start[0], center_y - radius),
                    QPointF(line_end[0], center_y - radius),
                    QPointF(line_end[0], center_y + radius),
                    QPointF(line_start[0], center_y + radius),
                ]
            )
        )
        band.setBrush(QBrush(QColor(0, 220, 110, 54)))
        band.setPen(QPen(QColor(0, 220, 110, 210), 1.0, Qt.PenStyle.DashLine))
        band.setZValue(61)
        self.scene.addItem(band)
        self.detection_preview_items.append(band)

        segment_count = max(3, min(24, int(round((line_end[0] - line_start[0]) / max(2, segment_value)))))
        segment_width = (line_end[0] - line_start[0]) / segment_count
        for idx in range(segment_count + 1):
            x = line_start[0] + idx * segment_width
            tick = QGraphicsLineItem(x, center_y - radius - 4.0, x, center_y + radius + 4.0)
            tick.setPen(QPen(QColor("#ffd166"), 1.0))
            tick.setZValue(62)
            self.scene.addItem(tick)
            self.detection_preview_items.append(tick)

        label = QGraphicsTextItem()
        label.setHtml(
            "<div style='color:white; font-size:10pt;'>"
            f"경계인식 범위<br>세그먼트 크기 {segment_value}px</div>"
        )
        label.setPos(left + 8.0, top + 5.0)
        label.setZValue(63)
        self.scene.addItem(label)
        self.detection_preview_items.append(label)

    def clear_detection_preview(self) -> None:
        for item in self.detection_preview_items:
            self.scene.removeItem(item)
        self.detection_preview_items.clear()

    def update_search_range_overlay(self, records: list[LineRecord]) -> None:
        for item in self.search_range_band_items + self.search_range_label_items:
            self.scene.removeItem(item)
        self.search_range_band_items.clear()
        self.search_range_label_items.clear()
        if not self.show_search_range or self.pixmap_item is None:
            return
        radius = float(self.search_range_radius_px)
        if radius <= 0:
            return
        for record in records:
            if record.kind != "edge":
                continue
            radius = float(record.search_radius_px or self.search_range_radius_px)
            if radius <= 0:
                continue
            polygons = self._search_range_polygons(record, radius)
            if self.show_search_range_band and record.show_range:
                for polygon in polygons:
                    item = QGraphicsPolygonItem(polygon)
                    item.setPen(QPen(QColor(0, 220, 110, 220), 1.2, Qt.PenStyle.DashLine))
                    item.setBrush(QBrush(QColor(0, 220, 110, 46)))
                    item.setZValue(3)
                    item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, False)
                    self.scene.addItem(item)
                    self.search_range_band_items.append(item)

    def clear_angle_items(self) -> None:
        for item in self.angle_items:
            self.scene.removeItem(item)
        self.angle_items.clear()
        self.angle_groups.clear()
        self.angle_group_parents.clear()
        self.angle_group_measurements.clear()

    def clear_cd_items(self) -> None:
        for item in self.cd_items:
            self.scene.removeItem(item)
        self.cd_items.clear()

    def clear_edge_length_items(self) -> None:
        for item in self.edge_length_items:
            self.scene.removeItem(item)
        self.edge_length_items.clear()
        self.edge_length_groups.clear()
        self.edge_length_group_parents.clear()

    def update_edge_length_overlay(
        self,
        records: list[LineRecord],
        nm_per_px: Optional[float],
        visible: bool,
    ) -> None:
        self.clear_edge_length_items()
        if not visible or self.pixmap_item is None:
            return
        for record in records:
            if record.kind != "edge":
                continue
            if record.edge_mode == "polyline" or (record.points and len(record.points) > 2):
                continue
            if not record.show_edge_length:
                continue
            points = record_points(record)
            if len(points) < 2:
                continue
            length_px = record_length(record)
            if nm_per_px:
                text = f"L {length_px * nm_per_px:.3g} nm"
            else:
                text = f"L {length_px:.2f} px"
            midpoint = ((points[0][0] + points[-1][0]) / 2.0, (points[0][1] + points[-1][1]) / 2.0)
            label = QGraphicsTextItem()
            label.setHtml(
                "<div style='background-color:rgba(35,20,45,185);"
                "color:#f5ddff;padding:2px 5px;border-radius:3px;'>"
                f"{text}</div>"
            )
            label_rect = label.boundingRect()
            if record.edge_length_label_pos is not None:
                label.setPos(record.edge_length_label_pos[0], record.edge_length_label_pos[1])
            else:
                label.setPos(midpoint[0] - label_rect.width() / 2.0, midpoint[1] - label_rect.height() / 2.0)
            label.setZValue(28)
            label.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
            label.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
            group_id = f"L_{record.id}"
            label.setData(LENGTH_GROUP_KEY, group_id)
            label.setData(LENGTH_PARENT_KEY, record.id)
            self.scene.addItem(label)
            self.edge_length_items.append(label)
            self.edge_length_groups[group_id] = [label]
            self.edge_length_group_parents[group_id] = record.id

    def add_cd_measurement(self, start: Point, end: Point, text: str, label_pos: Point) -> list[QGraphicsItem]:
        items: list[QGraphicsItem] = []
        line = QGraphicsLineItem(start[0], start[1], end[0], end[1])
        line.setPen(QPen(QColor("#8ecae6"), 2.0))
        line.setZValue(18)
        line.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, False)
        self.scene.addItem(line)
        self.cd_items.append(line)
        items.append(line)

        label = QGraphicsTextItem()
        label.setHtml(
            "<div style='background-color:rgba(3,37,65,185);"
            "color:#d9f6ff;padding:2px 5px;border-radius:3px;'>"
            f"{text}</div>"
        )
        label.setPos(label_pos[0], label_pos[1])
        label.setZValue(31)
        label.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, False)
        self.scene.addItem(label)
        self.cd_items.append(label)
        items.append(label)
        return items

    def add_angle_annotation(
        self,
        text: str,
        label_pos: Point,
        center: Optional[Point] = None,
        angle_a: Optional[float] = None,
        angle_b: Optional[float] = None,
        radius: float = 28.0,
        parent_record_id: Optional[str] = None,
        measurement_id: Optional[str] = None,
    ) -> list[QGraphicsItem]:
        group_id = f"A{self._angle_counter}"
        self._angle_counter += 1
        measurement_id = measurement_id or group_id
        items: list[QGraphicsItem] = []
        if center is not None and angle_a is not None and angle_b is not None:
            items.append(self._create_angle_arc(center, angle_a, angle_b, radius, group_id, measurement_id))
        items.append(self._create_angle_label(text, label_pos, group_id, measurement_id))
        self.angle_groups[group_id] = items
        self.angle_group_measurements[group_id] = measurement_id
        if parent_record_id is not None:
            self.angle_group_parents[group_id] = parent_record_id
        return items

    def _create_angle_arc(
        self,
        center: Point,
        angle_start: float,
        angle_end: float,
        radius: float,
        group_id: str,
        measurement_id: str,
    ) -> QGraphicsPathItem:
        path = self._arc_path(center, angle_start, angle_end, radius)
        item = QGraphicsPathItem(path)
        item.setPen(QPen(QColor("#ffd166"), 2.0))
        item.setZValue(20)
        item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        item.setData(ANGLE_GROUP_KEY, group_id)
        item.setData(ANGLE_MEASUREMENT_KEY, measurement_id)
        self.scene.addItem(item)
        self.angle_items.append(item)
        return item

    def _create_angle_label(self, text: str, pos: Point, group_id: str, measurement_id: str) -> QGraphicsTextItem:
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
        item.setData(ANGLE_GROUP_KEY, group_id)
        item.setData(ANGLE_MEASUREMENT_KEY, measurement_id)
        self.scene.addItem(item)
        self.angle_items.append(item)
        return item

    def selected_line_ids(self) -> list[str]:
        ids: list[str] = []
        for item in self.scene.selectedItems():
            if isinstance(item, (AnnotationLineItem, AnnotationCurveItem)):
                ids.append(item.record_id)
        return ids

    def selected_angle_items(self) -> list[QGraphicsPathItem | QGraphicsTextItem]:
        selected: list[QGraphicsPathItem | QGraphicsTextItem] = []
        group_ids = {item.data(ANGLE_GROUP_KEY) for item in self.scene.selectedItems() if item.data(ANGLE_GROUP_KEY)}
        for group_id in group_ids:
            for item in self.angle_groups.get(str(group_id), []):
                if item in self.angle_items:
                    selected.append(item)
        for item in self.scene.selectedItems():
            if item in self.angle_items and item not in selected:
                selected.append(item)
        return selected

    def selected_edge_length_items(self) -> list[QGraphicsItem]:
        selected: list[QGraphicsItem] = []
        group_ids = {item.data(LENGTH_GROUP_KEY) for item in self.scene.selectedItems() if item.data(LENGTH_GROUP_KEY)}
        for group_id in group_ids:
            for item in self.edge_length_groups.get(str(group_id), []):
                if item in self.edge_length_items and item not in selected:
                    selected.append(item)
        for item in self.scene.selectedItems():
            if item in self.edge_length_items and item not in selected:
                selected.append(item)
        return selected

    def selected_edge_length_parent_ids(self) -> set[str]:
        parent_ids: set[str] = set()
        group_ids = {item.data(LENGTH_GROUP_KEY) for item in self.scene.selectedItems() if item.data(LENGTH_GROUP_KEY)}
        for group_id in group_ids:
            parent_id = self.edge_length_group_parents.get(str(group_id))
            if parent_id:
                parent_ids.add(parent_id)
        for item in self.scene.selectedItems():
            parent_id = item.data(LENGTH_PARENT_KEY)
            if parent_id:
                parent_ids.add(str(parent_id))
        return parent_ids

    def selected_angle_measurement_ids(self) -> set[str]:
        measurement_ids: set[str] = set()
        group_ids = {item.data(ANGLE_GROUP_KEY) for item in self.scene.selectedItems() if item.data(ANGLE_GROUP_KEY)}
        for group_id in group_ids:
            measurement_id = self.angle_group_measurements.get(str(group_id))
            if measurement_id:
                measurement_ids.add(measurement_id)
        for item in self.scene.selectedItems():
            measurement_id = item.data(ANGLE_MEASUREMENT_KEY)
            if measurement_id:
                measurement_ids.add(str(measurement_id))
        return measurement_ids

    def remove_angle_items(self, items: list[QGraphicsPathItem | QGraphicsTextItem]) -> None:
        for item in items:
            if item in self.angle_items:
                self.angle_items.remove(item)
            self.scene.removeItem(item)
        self.angle_groups = {
            group_id: [item for item in group_items if item in self.angle_items]
            for group_id, group_items in self.angle_groups.items()
        }
        self.angle_groups = {group_id: items for group_id, items in self.angle_groups.items() if items}
        self.angle_group_parents = {
            group_id: parent_id
            for group_id, parent_id in self.angle_group_parents.items()
            if group_id in self.angle_groups
        }
        self.angle_group_measurements = {
            group_id: measurement_id
            for group_id, measurement_id in self.angle_group_measurements.items()
            if group_id in self.angle_groups
        }

    def remove_edge_length_items(self, items: list[QGraphicsItem]) -> None:
        for item in items:
            if item in self.edge_length_items:
                self.edge_length_items.remove(item)
            self.scene.removeItem(item)
        self.edge_length_groups = {
            group_id: [item for item in group_items if item in self.edge_length_items]
            for group_id, group_items in self.edge_length_groups.items()
        }
        self.edge_length_groups = {group_id: items for group_id, items in self.edge_length_groups.items() if items}
        self.edge_length_group_parents = {
            group_id: parent_id
            for group_id, parent_id in self.edge_length_group_parents.items()
            if group_id in self.edge_length_groups
        }

    def _expand_angle_group_selection(self) -> None:
        if self._filtering_selection:
            return
        if self._expanding_angle_selection:
            return
        if self._selection_filter is not None:
            self._apply_selection_filter()
            return
        selected_group_ids = {
            item.data(ANGLE_GROUP_KEY)
            for item in self.scene.selectedItems()
            if item.data(ANGLE_GROUP_KEY)
        }
        if not selected_group_ids:
            return
        self._expanding_angle_selection = True
        try:
            for group_id in selected_group_ids:
                for item in self.angle_groups.get(str(group_id), []):
                    item.setSelected(True)
        finally:
            self._expanding_angle_selection = False

    def _apply_selection_filter(self) -> None:
        if self._selection_filter is None or self._filtering_selection:
            return
        self._filtering_selection = True
        try:
            for item in self.scene.selectedItems():
                if not self._matches_selection_filter(item):
                    item.setSelected(False)
        finally:
            self._filtering_selection = False

    def _matches_selection_filter(self, item: QGraphicsItem) -> bool:
        if self._selection_filter == "edge":
            return isinstance(item, (AnnotationLineItem, AnnotationCurveItem)) and item.kind == "edge"
        if self._selection_filter == "angle_arc":
            return isinstance(item, QGraphicsPathItem) and item in self.angle_items
        if self._selection_filter == "angle_label":
            return isinstance(item, QGraphicsTextItem) and item in self.angle_items
        return True

    def selected_persistent_bounds(self) -> Optional[QRectF]:
        items = [item for item in self.scene.selectedItems() if isinstance(item, (AnnotationLineItem, AnnotationCurveItem))]
        if not items:
            return None
        rect = items[0].sceneBoundingRect()
        for item in items[1:]:
            rect = rect.united(item.sceneBoundingRect())
        return rect

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
        self.setFocus(Qt.FocusReason.MouseFocusReason)
        if event.button() in (Qt.MouseButton.MiddleButton, Qt.MouseButton.RightButton):
            self._start_pan(event.pos())
            event.accept()
            return

        if (
            event.button() == Qt.MouseButton.LeftButton
            and event.modifiers() & Qt.KeyboardModifier.ControlModifier
            and self.current_tool == "select"
        ):
            segment = self._segment_at(event.pos())
            if segment is not None:
                self.segment_split_requested.emit(segment[0], segment[1])
                event.accept()
                return

        if (
            event.button() == Qt.MouseButton.LeftButton
            and (self.current_tool == "pan" or event.modifiers() & Qt.KeyboardModifier.ControlModifier)
        ):
            self._start_pan(event.pos())
            event.accept()
            return

        if event.button() == Qt.MouseButton.LeftButton and self.current_tool == "resize":
            self.edit_started.emit()
            self._resizing = True
            self._resize_last = event.pos()
            event.accept()
            return

        if (
            event.button() == Qt.MouseButton.LeftButton
            and self.current_tool in {"scale", "reference", "edge", "guide"}
            and self.pixmap_item is not None
        ):
            if self.current_tool == "edge" and self.edge_draw_mode == "polyline":
                self._append_curve_point(self._clamp_to_image(self.mapToScene(event.pos())))
                event.accept()
                return
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
        if event.button() == Qt.MouseButton.LeftButton:
            item = self.itemAt(event.pos())
            if item is not None and item.flags() & QGraphicsItem.GraphicsItemFlag.ItemIsMovable:
                self.edit_started.emit()
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

        if self._temp_curve is not None and self._curve_points:
            preview_point = self._clamp_to_image(self.mapToScene(event.pos()))
            self._update_curve_preview(preview_point)
            event.accept()
            return

        if self._resizing:
            delta = event.pos() - self._resize_last
            self._resize_last = event.pos()
            factor = 1.0 + (delta.x() + delta.y()) / 240.0
            factor = max(0.2, min(5.0, factor))
            self.scale_requested.emit(float(factor))
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
            self._restore_tool_cursor()
            event.accept()
            return

        if self._resizing and event.button() == Qt.MouseButton.LeftButton:
            self._resizing = False
            event.accept()
            self.scene_changed.emit()
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
                    None,
                )
            event.accept()
            return
        super().mouseReleaseEvent(event)
        self._apply_selection_filter()
        self.scene_changed.emit()

    def _segment_at(self, view_pos: QPoint) -> Optional[tuple[str, int]]:
        scene_pos = self.mapToScene(view_pos)
        click = (float(scene_pos.x()), float(scene_pos.y()))
        tolerance = max(6.0, 7.0 / max(0.2, self.transform().m11()))
        best: Optional[tuple[float, str, int]] = None
        for item in self.items(view_pos):
            if not isinstance(item, AnnotationCurveItem) or item.kind != "edge":
                continue
            points = points_from_path_item(item)
            for idx, (start, end) in enumerate(zip(points, points[1:])):
                distance = point_to_segment_distance(click, start, end)
                if distance <= tolerance and (best is None or distance < best[0]):
                    best = (distance, item.record_id, idx)
        if best is None:
            return None
        return (best[1], best[2])

    def mouseDoubleClickEvent(self, event):  # noqa: N802
        if (
            event.button() == Qt.MouseButton.LeftButton
            and self.current_tool == "edge"
            and self.edge_draw_mode == "polyline"
            and self.pixmap_item is not None
        ):
            self._append_curve_point(self._clamp_to_image(self.mapToScene(event.pos())))
            self._finish_curve()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def keyPressEvent(self, event):  # noqa: N802
        if not event.isAutoRepeat() and event.key() == Qt.Key.Key_Space:
            if self.current_tool != "edge":
                self._space_edge_previous_tool = self.current_tool
                self.set_tool("edge")
            else:
                self._space_edge_previous_tool = None
            self.temporary_edge_tool_changed.emit(True)
            event.accept()
            return
        if event.key() in (Qt.Key.Key_Backspace, Qt.Key.Key_Delete):
            if not self.selected_line_ids() and self.delete_selected_point_handles():
                event.accept()
                return
        if event.key() in (Qt.Key.Key_Left, Qt.Key.Key_Right, Qt.Key.Key_Up, Qt.Key.Key_Down):
            if self._nudge_selected_items(event.key(), event.modifiers()):
                event.accept()
                return
        if not event.isAutoRepeat() and event.key() in (Qt.Key.Key_Q, Qt.Key.Key_W, Qt.Key.Key_E):
            self._selection_filter = {
                Qt.Key.Key_Q: "edge",
                Qt.Key.Key_W: "angle_arc",
                Qt.Key.Key_E: "angle_label",
            }[event.key()]
            self._apply_selection_filter()
            event.accept()
            return
        if self.current_tool == "edge" and self.edge_draw_mode == "polyline":
            if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                self._finish_curve()
                event.accept()
                return
            if event.key() == Qt.Key.Key_Escape and self._curve_points:
                self.cancel_interaction()
                event.accept()
                return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event):  # noqa: N802
        if not event.isAutoRepeat() and event.key() == Qt.Key.Key_Space:
            previous_tool = self._space_edge_previous_tool
            self._space_edge_previous_tool = None
            if previous_tool is not None:
                self.set_tool(previous_tool)
            self.temporary_edge_tool_changed.emit(False)
            event.accept()
            return
        if not event.isAutoRepeat() and event.key() in (Qt.Key.Key_Q, Qt.Key.Key_W, Qt.Key.Key_E):
            self._selection_filter = None
            event.accept()
            return
        super().keyReleaseEvent(event)

    def _append_curve_point(self, point: QPointF) -> None:
        if self._curve_points:
            last = self._curve_points[-1]
            if math.hypot(point.x() - last.x(), point.y() - last.y()) < 2:
                return
        self._curve_points.append(point)
        if self._temp_curve is None:
            self._temp_curve = QGraphicsPathItem()
            self._temp_curve.setPen(QPen(QColor("#4cc9f0"), 2.0, Qt.PenStyle.DashLine))
            self._temp_curve.setZValue(40)
            self.scene.addItem(self._temp_curve)
        self._update_curve_preview()

    def _update_curve_preview(self, preview_point: Optional[QPointF] = None) -> None:
        if self._temp_curve is None:
            return
        points = [(float(point.x()), float(point.y())) for point in self._curve_points]
        if preview_point is not None:
            points.append((float(preview_point.x()), float(preview_point.y())))
        self._temp_curve.setPath(path_from_points(points, smooth=False))

    def _finish_curve(self) -> None:
        if len(self._curve_points) < 2:
            self._clear_curve_preview()
            return
        points = [(float(point.x()), float(point.y())) for point in self._curve_points]
        start = points[0]
        end = points[-1]
        self._clear_curve_preview()
        if record_length(LineRecord("_preview", "edge", start, end, points=points, edge_mode="polyline")) > 3:
            self.line_created.emit(self.current_tool, start, end, points)

    def _clear_curve_preview(self) -> None:
        if self._temp_curve is not None:
            self.scene.removeItem(self._temp_curve)
        self._temp_curve = None
        self._curve_points = []

    def _nudge_selected_items(self, key: int, modifiers: Qt.KeyboardModifier) -> bool:
        selected = self.scene.selectedItems()
        if not selected:
            return False
        step = 1.0 if modifiers & Qt.KeyboardModifier.ControlModifier else 10.0
        dx = 0.0
        dy = 0.0
        if key == Qt.Key.Key_Left:
            dx = -step
        elif key == Qt.Key.Key_Right:
            dx = step
        elif key == Qt.Key.Key_Up:
            dy = -step
        elif key == Qt.Key.Key_Down:
            dy = step
        else:
            return False
        self.edit_started.emit()
        for item in selected:
            if item.flags() & QGraphicsItem.GraphicsItemFlag.ItemIsMovable:
                item.moveBy(dx, dy)
        self.scene_changed.emit()
        return True

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
    def _arc_path(center: Point, angle_start: float, angle_end: float, radius: float) -> QPainterPath:
        delta = (angle_end - angle_start) % 360.0
        if delta <= 0:
            delta = 360.0
        steps = max(8, int(abs(delta) / 4))
        path = QPainterPath()
        for idx in range(steps + 1):
            angle = math.radians(angle_start + delta * idx / steps)
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
        width = float(record.stroke_width) if record.stroke_width is not None else (1.2 if record.kind == "guide" else 2.2)
        color = QColor(record.stroke_color or colors.get(record.kind, "#ffffff"))
        if record.kind == "reference" and record.stroke_color is None:
            color.setAlpha(128)
        pen = QPen(color, width)
        if record.kind == "guide":
            pen.setStyle(Qt.PenStyle.DotLine)
        return pen

    @staticmethod
    def _search_range_polygons(record: LineRecord, radius: float) -> list[QPolygonF]:
        points = record_points(record)
        polygons: list[QPolygonF] = []
        for start, end in zip(points, points[1:]):
            nx, ny = normal_for_line(start, end)
            if nx == 0 and ny == 0:
                continue
            sx, sy = start
            ex, ey = end
            polygons.append(
                QPolygonF(
                    [
                        QPointF(sx + nx * radius, sy + ny * radius),
                        QPointF(ex + nx * radius, ey + ny * radius),
                        QPointF(ex - nx * radius, ey - ny * radius),
                        QPointF(sx - nx * radius, sy - ny * radius),
                    ]
                )
            )
        return polygons


def record_points(record: LineRecord) -> list[Point]:
    if record.points and len(record.points) >= 2:
        return record.points
    return [record.start, record.end]


def recognition_points(record: LineRecord) -> list[Point]:
    if record.recognition_points and len(record.recognition_points) >= 2:
        return record.recognition_points
    return record_points(record)


def translated_points(points: list[Point], dx: float, dy: float) -> list[Point]:
    return [(point[0] + dx, point[1] + dy) for point in points]


def uniform_translation_delta(old_points: list[Point], new_points: list[Point], tolerance: float = 0.01) -> Optional[Point]:
    if len(old_points) != len(new_points) or not old_points:
        return None
    dx = new_points[0][0] - old_points[0][0]
    dy = new_points[0][1] - old_points[0][1]
    for old_point, new_point in zip(old_points[1:], new_points[1:]):
        if abs((new_point[0] - old_point[0]) - dx) > tolerance:
            return None
        if abs((new_point[1] - old_point[1]) - dy) > tolerance:
            return None
    if abs(dx) <= tolerance and abs(dy) <= tolerance:
        return (0.0, 0.0)
    return (dx, dy)


def point_to_segment_distance(point: Point, start: Point, end: Point) -> float:
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length_sq = dx * dx + dy * dy
    if length_sq <= 0:
        return line_length(point, start)
    t = ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / length_sq
    t = max(0.0, min(1.0, t))
    projection = (start[0] + t * dx, start[1] + t * dy)
    return line_length(point, projection)


def record_length(record: LineRecord) -> float:
    points = record_points(record)
    return sum(line_length(start, end) for start, end in zip(points, points[1:]))


def record_angle(record: LineRecord) -> float:
    points = record_points(record)
    return line_angle_degrees(points[0], points[-1])


def has_segmented_edge_angle(record: LineRecord) -> bool:
    return record.kind == "edge" and bool(record.points and len(record.points) > 2)


def scale_point(point: Point, center: Point, factor: float) -> Point:
    return (
        center[0] + (point[0] - center[0]) * factor,
        center[1] + (point[1] - center[1]) * factor,
    )


def cumulative_lengths(points: list[Point]) -> list[float]:
    lengths = [0.0]
    for start, end in zip(points, points[1:]):
        lengths.append(lengths[-1] + line_length(start, end))
    return lengths


def point_at_polyline_distance(points: list[Point], distance: float) -> Point:
    if not points:
        return (0.0, 0.0)
    if len(points) == 1:
        return points[0]
    remaining = max(0.0, distance)
    for start, end in zip(points, points[1:]):
        segment_length = line_length(start, end)
        if segment_length <= 0:
            continue
        if remaining <= segment_length:
            fraction = remaining / segment_length
            return (
                start[0] + (end[0] - start[0]) * fraction,
                start[1] + (end[1] - start[1]) * fraction,
            )
        remaining -= segment_length
    return points[-1]


def polyline_slice(points: list[Point], start_distance: float, end_distance: float) -> list[Point]:
    if end_distance < start_distance:
        start_distance, end_distance = end_distance, start_distance
    lengths = cumulative_lengths(points)
    sliced = [point_at_polyline_distance(points, start_distance)]
    for point, distance in zip(points[1:-1], lengths[1:-1]):
        if start_distance < distance < end_distance:
            sliced.append(point)
    sliced.append(point_at_polyline_distance(points, end_distance))
    return sliced


def points_close(a: Point, b: Point, tolerance: float = 6.0) -> bool:
    return line_length(a, b) <= tolerance


def legacy_sensitivity_to_segment_size_px(value: int | float) -> int:
    return int(round(max(2.0, min(80.0, 18.0 - float(value) * 0.14))))


def offset_point(point: Point, dx: float, dy: float) -> Point:
    return (point[0] + dx, point[1] + dy)


def clone_record(record: LineRecord) -> LineRecord:
    return LineRecord(
        id=record.id,
        kind=record.kind,
        start=tuple(record.start),
        end=tuple(record.end),
        label=record.label,
        axis=record.axis,
        value_nm=record.value_nm,
        points=[tuple(point) for point in record.points] if record.points else None,
        recognition_points=[tuple(point) for point in record.recognition_points] if record.recognition_points else None,
        edge_mode=record.edge_mode,
        search_radius_px=record.search_radius_px,
        segment_size_px=record.segment_size_px,
        angle_sector=record.angle_sector,
        angle_arc_radius=record.angle_arc_radius,
        angle_label_side=record.angle_label_side,
        angle_label_gap=record.angle_label_gap,
        edge_segmented=record.edge_segmented,
        object_group=record.object_group,
        show_line=record.show_line,
        show_angle=record.show_angle,
        show_range=record.show_range,
        show_range_label=record.show_range_label,
        show_edge_length=record.show_edge_length,
        edge_length_label_pos=tuple(record.edge_length_label_pos) if record.edge_length_label_pos else None,
        stroke_color=record.stroke_color,
        stroke_width=record.stroke_width,
    )


def line_record_from_dict(item: dict) -> LineRecord:
    raw_points = item.get("points") or []
    raw_recognition_points = item.get("recognition_points") or []
    record_edge_mode = item.get("edge_mode")
    if record_edge_mode == "curve":
        record_edge_mode = "polyline"
    if record_edge_mode not in {"line", "polyline"}:
        record_edge_mode = "polyline" if raw_points else "line"
    return LineRecord(
        id=item["id"],
        kind=item["kind"],
        start=tuple(item["start"]),
        end=tuple(item["end"]),
        label=item.get("label", ""),
        axis=item.get("axis", "horizontal"),
        value_nm=item.get("value_nm"),
        points=[tuple(point) for point in raw_points] or None,
        recognition_points=[tuple(point) for point in raw_recognition_points] or None,
        edge_mode=record_edge_mode,
        search_radius_px=int(item["search_radius_px"]) if item.get("search_radius_px") is not None else None,
        segment_size_px=int(item["segment_size_px"]) if item.get("segment_size_px") is not None else None,
        angle_sector=int(item.get("angle_sector", 0)),
        angle_arc_radius=float(item.get("angle_arc_radius", 28.0)),
        angle_label_side=item.get("angle_label_side", "outside"),
        angle_label_gap=float(item.get("angle_label_gap", 14.0)),
        edge_segmented=bool(item.get("edge_segmented", bool(raw_points and len(raw_points) > 6))),
        object_group=item.get("object_group"),
        show_line=bool(item.get("show_line", True)),
        show_angle=bool(item.get("show_angle", True)),
        show_range=bool(item.get("show_range", True)),
        show_range_label=bool(item.get("show_range_label", True)),
        show_edge_length=bool(item.get("show_edge_length", True)),
        edge_length_label_pos=tuple(item["edge_length_label_pos"]) if item.get("edge_length_label_pos") is not None else None,
        stroke_color=item.get("stroke_color"),
        stroke_width=float(item["stroke_width"]) if item.get("stroke_width") is not None else None,
    )


def structure_template_from_dict(item: dict) -> StructureTemplate:
    records = [
        line_record_from_dict(record)
        for record in item.get("records", [])
        if record.get("kind") in {"edge", "guide"}
    ]
    return StructureTemplate(
        name=item.get("name", "구조"),
        records=records,
        cd_segment_mode=item.get("cd_segment_mode", "all"),
    )


def structure_template_to_dict(template: StructureTemplate) -> dict:
    return {
        "name": template.name,
        "cd_segment_mode": template.cd_segment_mode,
        "records": [asdict(record) for record in template.records],
    }


def angle_sector_geometry(angle_a: float, angle_b: float, sector_index: int) -> tuple[float, float, float]:
    rays = sorted(
        [
            angle_a % 360.0,
            (angle_a + 180.0) % 360.0,
            angle_b % 360.0,
            (angle_b + 180.0) % 360.0,
        ]
    )
    sectors: list[tuple[float, float, float]] = []
    for idx, start in enumerate(rays):
        end = rays[(idx + 1) % len(rays)]
        span = (end - start) % 360.0
        if span > 0.01:
            sectors.append((start, start + span, span))
    if not sectors:
        return angle_a % 360.0, (angle_a + 180.0) % 360.0, 180.0
    return sectors[sector_index % len(sectors)]


def angle_label_position_for_sector(
    center: Point,
    start_angle: float,
    span: float,
    radius: float,
    side: str,
    gap: float,
) -> Point:
    if side == "inside":
        distance = max(6.0, radius - gap)
    elif side == "on_arc":
        distance = radius
    else:
        distance = radius + gap
    angle = math.radians(start_angle + span / 2.0)
    return (center[0] + math.cos(angle) * distance, center[1] + math.sin(angle) * distance)


def polyline_intersections(edge: LineRecord, guide_line: tuple[Point, Point]) -> list[tuple[Point, float]]:
    crosses: list[tuple[Point, float]] = []
    points = record_points(edge)
    for start, end in zip(points, points[1:]):
        cross = intersection((start, end), guide_line)
        if cross is not None:
            crosses.append((cross, line_angle_degrees(start, end)))
    return crosses


def line_fraction(point: Point, line: tuple[Point, Point]) -> float:
    start, end = line
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length_sq = dx * dx + dy * dy
    if length_sq <= 0:
        return 0.0
    return ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / length_sq


def cd_segment_allowed(index: int, mode: str) -> bool:
    segment_number = index + 1
    if mode == "odd":
        return segment_number % 2 == 1
    if mode == "even":
        return segment_number % 2 == 0
    return True


class EdgeDetectionSettingsDialog(QDialog):
    def __init__(
        self,
        radius_px: int,
        segment_size_px: int,
        show_overlay: bool,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("인식 설정")
        self.setModal(True)

        self.radius_spin = QSpinBox()
        self.radius_spin.setRange(2, 300)
        self.radius_spin.setValue(radius_px)
        self.radius_spin.setSuffix(" px")

        self.sensitivity_spin = QSpinBox()
        self.sensitivity_spin.setRange(2, 80)
        self.sensitivity_spin.setValue(segment_size_px)
        self.sensitivity_spin.setSuffix(" px")

        self.overlay_checkbox = QCheckBox("이미지 위에 경계인식 범위 표시")
        self.overlay_checkbox.setChecked(show_overlay)

        layout = QVBoxLayout(self)
        form = QFormLayout()
        form.addRow("경계인식 범위", self.radius_spin)
        form.addRow("세그먼트 크기", self.sensitivity_spin)
        layout.addLayout(form)
        layout.addWidget(self.overlay_checkbox)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)


class AngleDisplaySettingsDialog(QDialog):
    def __init__(
        self,
        sector: int,
        arc_radius: float,
        label_side: str,
        label_gap: float,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("각도 표시 편집")
        self.setModal(True)

        self.sector_combo = QComboBox()
        for idx in range(4):
            self.sector_combo.addItem(f"각도 위치 {idx + 1}", idx)
        self.sector_combo.setCurrentIndex(max(0, min(3, int(sector))))

        self.arc_radius_spin = QDoubleSpinBox()
        self.arc_radius_spin.setRange(6.0, 300.0)
        self.arc_radius_spin.setValue(float(arc_radius))
        self.arc_radius_spin.setSuffix(" px")

        self.label_side_combo = QComboBox()
        for label, value in [
            ("호 바깥쪽", "outside"),
            ("호 위", "on_arc"),
            ("호 안쪽", "inside"),
        ]:
            self.label_side_combo.addItem(label, value)
        side_index = self.label_side_combo.findData(label_side)
        self.label_side_combo.setCurrentIndex(side_index if side_index >= 0 else 0)

        self.label_gap_spin = QDoubleSpinBox()
        self.label_gap_spin.setRange(0.0, 300.0)
        self.label_gap_spin.setValue(float(label_gap))
        self.label_gap_spin.setSuffix(" px")

        layout = QVBoxLayout(self)
        form = QFormLayout()
        form.addRow("각도 호 위치", self.sector_combo)
        form.addRow("각도 호 크기", self.arc_radius_spin)
        form.addRow("숫자 위치", self.label_side_combo)
        form.addRow("숫자 거리", self.label_gap_spin)
        layout.addLayout(form)
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
        self.browser_root: Optional[Path] = None
        self.browser_image_paths: list[str] = []
        self.current_browser_index = -1
        self.thumbnail_buttons: dict[str, QPushButton] = {}
        self.thumbnail_columns = 2
        self.current_tool = "select"
        self.scale_presets: list[ScalePreset] = []
        self.structure_templates: list[StructureTemplate] = []
        self.record_clipboard: list[LineRecord] = []
        self._paste_offset_steps = 0
        self.default_angle_sector = 0
        self.default_angle_arc_radius = 28.0
        self.default_angle_label_side = "outside"
        self.default_angle_label_gap = 14.0
        self.default_stroke_color = "#ff6b6b"
        self.default_stroke_width = 2.2
        self.hidden_angle_measurements: set[str] = set()
        self.undo_stack: list[dict] = []
        self._restoring_undo = False
        self._updating_object_visibility_controls = False
        self._updating_detection_controls = False
        self._expanding_object_group_selection = False
        self.visibility: dict[str, bool] = {
            "scale": True,
            "reference": True,
            "edge": True,
            "guide": True,
            "angle": True,
            "cd": True,
            "edge_length": True,
            "range": True,
            "point_handle": True,
        }

        self.canvas = AngleCanvas()
        self.setCentralWidget(self.canvas)
        self.canvas.line_created.connect(self._handle_line_created)
        self.canvas.segment_split_requested.connect(self.split_edge_segment_for_selection)
        self.canvas.scene_changed.connect(self._handle_scene_changed)
        self.canvas.scale_requested.connect(self.scale_selected_objects)
        self.canvas.edit_started.connect(self.save_undo_snapshot)
        self.canvas.temporary_edge_tool_changed.connect(self.set_temporary_edge_tool)
        self.detection_preview_timer = QTimer(self)
        self.detection_preview_timer.setSingleShot(True)
        self.detection_preview_timer.timeout.connect(self.canvas.clear_detection_preview)

        self._build_actions()
        self._build_toolbar()
        self._build_measurements_dock()
        self._build_visibility_dock()
        self._build_scale_preset_dock()
        self._build_thumbnail_dock()
        self.canvas.scene.selectionChanged.connect(self._expand_object_group_selection)
        self.canvas.scene.selectionChanged.connect(self._update_object_visibility_controls)
        self.setStatusBar(QStatusBar())
        self._set_status("이미지를 불러오면 시작할 수 있습니다.")

    def _build_actions(self) -> None:
        self.open_action = QAction("이미지 열기", self)
        self.open_action.triggered.connect(self.open_image)
        self.open_folder_action = QAction("폴더 열기", self)
        self.open_folder_action.triggered.connect(self.open_folder)
        self.save_project_action = QAction("프로젝트 저장", self)
        self.save_project_action.triggered.connect(self.save_project)
        self.open_project_action = QAction("프로젝트 열기", self)
        self.open_project_action.triggered.connect(self.open_project)
        self.export_png_action = QAction("주석 PNG 내보내기", self)
        self.export_png_action.triggered.connect(self.export_annotated_png)
        self.export_csv_action = QAction("CSV 내보내기", self)
        self.export_csv_action.triggered.connect(self.export_csv)
        self.select_tool_action = QAction("선택 도구", self)
        self.select_tool_action.setShortcut(QKeySequence(Qt.Key.Key_Escape))
        self.select_tool_action.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
        self.select_tool_action.triggered.connect(self.activate_select_tool)
        self.delete_action = QAction("선택 삭제", self)
        self.delete_action.setShortcut(QKeySequence(Qt.Key.Key_Delete))
        self.delete_action.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
        self.delete_action.triggered.connect(self.delete_selected)
        self.undo_action = QAction("되돌리기", self)
        self.undo_action.setShortcut(QKeySequence.StandardKey.Undo)
        self.undo_action.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
        self.undo_action.triggered.connect(self.undo)
        self.copy_action = QAction("선택 복사", self)
        self.copy_action.setShortcut(QKeySequence.StandardKey.Copy)
        self.copy_action.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
        self.copy_action.triggered.connect(self.copy_selected_parent_objects)
        self.paste_action = QAction("붙여넣기", self)
        self.paste_action.setShortcut(QKeySequence.StandardKey.Paste)
        self.paste_action.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
        self.paste_action.triggered.connect(self.paste_parent_objects)
        self.save_structure_action = QAction("구조 저장", self)
        self.save_structure_action.setShortcut(QKeySequence("Ctrl+Shift+C"))
        self.save_structure_action.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
        self.save_structure_action.triggered.connect(self.save_current_structure_template)
        self.paste_structure_action = QAction("구조 붙여넣기", self)
        self.paste_structure_action.setShortcut(QKeySequence("Ctrl+Shift+V"))
        self.paste_structure_action.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
        self.paste_structure_action.triggered.connect(self.paste_selected_structure_template)
        self.group_action = QAction("그룹화", self)
        self.group_action.setShortcut(QKeySequence("Ctrl+G"))
        self.group_action.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
        self.group_action.triggered.connect(self.group_selected_objects)
        self.ungroup_action = QAction("그룹 해제", self)
        self.ungroup_action.setShortcut(QKeySequence("Ctrl+Shift+G"))
        self.ungroup_action.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
        self.ungroup_action.triggered.connect(self.ungroup_selected_objects)
        self.previous_image_action = QAction("이전 이미지", self)
        self.previous_image_action.setShortcut(QKeySequence(Qt.Key.Key_PageUp))
        self.previous_image_action.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
        self.previous_image_action.triggered.connect(lambda: self.load_relative_browser_image(-1))
        self.next_image_action = QAction("다음 이미지", self)
        self.next_image_action.setShortcut(QKeySequence(Qt.Key.Key_PageDown))
        self.next_image_action.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
        self.next_image_action.triggered.connect(lambda: self.load_relative_browser_image(1))
        self.addAction(self.select_tool_action)
        self.addAction(self.delete_action)
        self.addAction(self.undo_action)
        self.addAction(self.copy_action)
        self.addAction(self.paste_action)
        self.addAction(self.save_structure_action)
        self.addAction(self.paste_structure_action)
        self.addAction(self.group_action)
        self.addAction(self.ungroup_action)
        self.addAction(self.previous_image_action)
        self.addAction(self.next_image_action)
        self.scale_preset_actions: list[QAction] = []
        for idx in range(9):
            action = QAction(f"스케일 프리셋 {idx + 1}", self)
            action.setShortcut(QKeySequence(str(idx + 1)))
            action.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
            action.triggered.connect(lambda checked=False, slot=idx: self.apply_scale_preset(slot))
            self.scale_preset_actions.append(action)
            self.addAction(action)

    def _build_toolbar(self) -> None:
        file_toolbar = self._new_toolbar("파일")
        for action in [self.open_action, self.open_folder_action, self.open_project_action, self.save_project_action]:
            file_toolbar.addWidget(self._button_for_action(action))

        export_toolbar = self._new_toolbar("내보내기")
        for action in [self.export_png_action, self.export_csv_action]:
            export_toolbar.addWidget(self._button_for_action(action))

        self.addToolBarBreak(Qt.ToolBarArea.TopToolBarArea)
        tool_toolbar = self._new_toolbar("도구")

        self.tool_buttons: dict[str, QPushButton] = {}
        self.tool_button_group = QButtonGroup(self)
        self.tool_button_group.setExclusive(True)
        for label, tool in [
            ("선택", "select"),
            ("이동", "pan"),
            ("크기 조절", "resize"),
            ("스케일바", "scale"),
            ("기준선", "reference"),
            ("경계선", "edge"),
            ("가이드선", "guide"),
        ]:
            button = QPushButton(label)
            button.setCheckable(True)
            button.clicked.connect(lambda checked=False, selected_tool=tool: self.set_current_tool(selected_tool))
            self.tool_button_group.addButton(button)
            self.tool_buttons[tool] = button
            tool_toolbar.addWidget(button)
        self.tool_buttons["select"].setChecked(True)

        reference_toolbar = self._new_toolbar("기준")

        self.axis_combo = QComboBox()
        self.axis_combo.addItem("수평기준선", "horizontal")
        self.axis_combo.addItem("수직기준선", "vertical")
        self.axis_combo.currentIndexChanged.connect(self._axis_changed)
        reference_toolbar.addWidget(self.axis_combo)

        align_button = QPushButton("이미지 맞춤")
        align_button.clicked.connect(self.align_to_reference)
        reference_toolbar.addWidget(align_button)

        self.addToolBarBreak(Qt.ToolBarArea.TopToolBarArea)
        detect_toolbar = self._new_toolbar("인식")

        self.edge_mode_combo = QComboBox()
        self.edge_mode_combo.addItem("직선", "line")
        self.edge_mode_combo.addItem("이어진 직선", "polyline")
        self.edge_mode_combo.currentIndexChanged.connect(self._edge_mode_changed)
        detect_toolbar.addWidget(QLabel("경계 형태"))
        detect_toolbar.addWidget(self.edge_mode_combo)
        self.canvas.set_edge_draw_mode(self.edge_mode_combo.currentData())

        self.search_radius_spin = QSpinBox()
        self.search_radius_spin.setRange(2, 300)
        self.search_radius_spin.setValue(35)
        self.search_radius_spin.setSuffix(" px")
        self.search_radius_spin.valueChanged.connect(self._edge_detection_settings_changed)
        detect_toolbar.addWidget(QLabel("경계인식 범위"))
        detect_toolbar.addWidget(self.search_radius_spin)

        self.curve_sensitivity_spin = QSpinBox()
        self.curve_sensitivity_spin.setRange(2, 80)
        self.curve_sensitivity_spin.setValue(9)
        self.curve_sensitivity_spin.setSuffix(" px")
        self.curve_sensitivity_spin.valueChanged.connect(self._edge_detection_settings_changed)
        detect_toolbar.addWidget(QLabel("세그먼트 크기"))
        detect_toolbar.addWidget(self.curve_sensitivity_spin)

        self.show_search_range_checkbox = QCheckBox("범위 표시")
        self.show_search_range_checkbox.setChecked(True)
        self.show_search_range_checkbox.toggled.connect(self._edge_detection_settings_changed)
        detect_toolbar.addWidget(self.show_search_range_checkbox)

        settings_button = QPushButton("인식 설정")
        settings_button.clicked.connect(self.open_edge_detection_settings)
        detect_toolbar.addWidget(settings_button)

        recognize_button = QPushButton("인식")
        recognize_button.clicked.connect(self.recognize_edges)
        detect_toolbar.addWidget(recognize_button)

        style_toolbar = self._new_toolbar("양식")
        self.stroke_color_combo = QComboBox()
        for label, color in [
            ("빨강", "#ff6b6b"),
            ("노랑", "#ffd166"),
            ("초록", "#06d6a0"),
            ("파랑", "#4cc9f0"),
            ("흰색", "#f7fff7"),
            ("검정", "#111111"),
        ]:
            self.stroke_color_combo.addItem(label, color)
        self.stroke_color_combo.currentIndexChanged.connect(self.apply_selected_style)
        self.stroke_width_spin = QDoubleSpinBox()
        self.stroke_width_spin.setRange(0.4, 12.0)
        self.stroke_width_spin.setSingleStep(0.2)
        self.stroke_width_spin.setValue(self.default_stroke_width)
        self.stroke_width_spin.setSuffix(" px")
        self.stroke_width_spin.valueChanged.connect(self.apply_selected_style)
        style_toolbar.addWidget(QLabel("선 색"))
        style_toolbar.addWidget(self.stroke_color_combo)
        style_toolbar.addWidget(QLabel("선 두께"))
        style_toolbar.addWidget(self.stroke_width_spin)

        self.addToolBarBreak(Qt.ToolBarArea.TopToolBarArea)
        guide_toolbar = self._new_toolbar("가이드")

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
        self.cd_segment_combo = QComboBox()
        self.cd_segment_combo.addItem("CD 전체", "all")
        self.cd_segment_combo.addItem("CD 홀수번째", "odd")
        self.cd_segment_combo.addItem("CD 짝수번째", "even")
        guide_toolbar.addWidget(self.guide_orientation_combo)
        guide_toolbar.addWidget(self.guide_spacing_spin)
        guide_toolbar.addWidget(self.guide_spacing_unit)
        guide_toolbar.addWidget(QLabel("시작"))
        guide_toolbar.addWidget(self.guide_offset_spin)
        guide_toolbar.addWidget(self.cd_segment_combo)

        add_guides_button = QPushButton("그리기")
        add_guides_button.clicked.connect(self.add_guides)
        clear_guides_button = QPushButton("가이드 지우기")
        clear_guides_button.clicked.connect(self.clear_guides)
        angle_button = QPushButton("각도 계산")
        angle_button.clicked.connect(lambda: self.calculate_angles(reset_hidden=True))
        angle_settings_button = QPushButton("각도 표시 편집")
        angle_settings_button.clicked.connect(self.edit_angle_display_for_selected_edges)
        cd_button = QPushButton("CD 길이")
        cd_button.clicked.connect(self.calculate_cd_lengths)
        guide_toolbar.addWidget(add_guides_button)
        guide_toolbar.addWidget(clear_guides_button)
        guide_toolbar.addWidget(angle_button)
        guide_toolbar.addWidget(angle_settings_button)
        guide_toolbar.addWidget(cd_button)

        structure_toolbar = self._new_toolbar("구조")
        self.structure_combo = QComboBox()
        self.structure_combo.addItem("구조 선택", -1)
        structure_toolbar.addWidget(self.structure_combo)
        structure_toolbar.addWidget(self._button_for_action(self.save_structure_action))
        structure_paste_button = self._button_for_action(self.paste_structure_action)
        structure_toolbar.addWidget(structure_paste_button)
        structure_export_button = QPushButton("구조 공유")
        structure_export_button.clicked.connect(self.export_selected_structure_template)
        structure_import_button = QPushButton("구조 가져오기")
        structure_import_button.clicked.connect(self.import_structure_template)
        structure_delete_button = QPushButton("구조 삭제")
        structure_delete_button.clicked.connect(self.delete_selected_structure_template)
        structure_toolbar.addWidget(structure_export_button)
        structure_toolbar.addWidget(structure_import_button)
        structure_toolbar.addWidget(structure_delete_button)

    def _new_toolbar(self, title: str) -> QToolBar:
        toolbar = QToolBar(title)
        toolbar.setMovable(True)
        toolbar.setFloatable(True)
        toolbar.setAllowedAreas(Qt.ToolBarArea.AllToolBarAreas)
        toolbar.setStyleSheet(
            "QToolBar { spacing: 4px; padding: 2px; } "
            "QToolBar::handle { background: #5a5a5a; } "
            "QToolBar::handle:horizontal { width: 8px; margin: 4px 2px; } "
            "QToolBar::handle:vertical { height: 8px; margin: 2px 4px; }"
        )
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, toolbar)
        return toolbar

    def _button_for_action(self, action: QAction) -> QPushButton:
        button = QPushButton(action.text())
        button.clicked.connect(action.trigger)
        return button

    def save_undo_snapshot(self) -> None:
        if self._restoring_undo or self.image_bgr is None:
            return
        self._sync_records_from_canvas()
        snapshot = {
            "image_bgr": self.image_bgr.copy() if self.image_bgr is not None else None,
            "records": [asdict(record) for record in self.records.values()],
            "counter": self._counter,
            "nm_per_px": self.nm_per_px,
            "hidden_angle_measurements": list(self.hidden_angle_measurements),
            "angle_item_states": self._angle_item_states(),
        }
        self.undo_stack.append(snapshot)
        if len(self.undo_stack) > 30:
            self.undo_stack.pop(0)

    def undo(self) -> None:
        if not self.undo_stack:
            self._set_status("되돌릴 작업이 없습니다.")
            return
        snapshot = self.undo_stack.pop()
        self._restoring_undo = True
        try:
            image = snapshot.get("image_bgr")
            if image is not None:
                self.image_bgr = image.copy()
            self.records = {item["id"]: line_record_from_dict(item) for item in snapshot.get("records", [])}
            self._counter = int(snapshot.get("counter", len(self.records) + 1))
            self.nm_per_px = snapshot.get("nm_per_px")
            self.hidden_angle_measurements = set(snapshot.get("hidden_angle_measurements", []))
            self.canvas.clear_point_handles()
            self._show_image(keep_view=True)
            self.canvas.redraw_lines(list(self.records.values()))
            self.calculate_angles(reset_hidden=False)
            self._restore_angle_item_states(snapshot.get("angle_item_states", []))
            self._update_search_range_overlay()
            self._update_edge_length_overlay(sync=False)
            self._apply_visibility()
            self._refresh_table()
            self._set_status("되돌렸습니다.")
        finally:
            self._restoring_undo = False

    def _angle_item_states(self) -> list[dict]:
        states: list[dict] = []
        for item in self.canvas.angle_items:
            pos = item.pos()
            states.append(
                {
                    "measurement_id": item.data(ANGLE_MEASUREMENT_KEY),
                    "group_id": item.data(ANGLE_GROUP_KEY),
                    "kind": "label" if isinstance(item, QGraphicsTextItem) else "arc",
                    "x": float(pos.x()),
                    "y": float(pos.y()),
                    "scale": float(item.scale()),
                }
            )
        return states

    def _restore_angle_item_states(self, states: list[dict]) -> None:
        pending = list(states)
        for item in self.canvas.angle_items:
            item_kind = "label" if isinstance(item, QGraphicsTextItem) else "arc"
            match_index = next(
                (
                    idx
                    for idx, state in enumerate(pending)
                    if state.get("measurement_id") == item.data(ANGLE_MEASUREMENT_KEY)
                    and state.get("kind") == item_kind
                ),
                None,
            )
            if match_index is None:
                continue
            state = pending.pop(match_index)
            item.setPos(float(state.get("x", 0.0)), float(state.get("y", 0.0)))
            item.setScale(float(state.get("scale", 1.0)))

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

    def _build_visibility_dock(self) -> None:
        dock = QDockWidget("표시", self)
        dock.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)
        container = QWidget()
        layout = QVBoxLayout(container)
        visibility_layout = QGridLayout()
        visibility_layout.setContentsMargins(0, 0, 0, 0)
        visibility_layout.setHorizontalSpacing(10)
        global_visibility_layout = QVBoxLayout()
        object_visibility_layout = QVBoxLayout()
        global_visibility_layout.setContentsMargins(0, 0, 0, 0)
        object_visibility_layout.setContentsMargins(0, 0, 0, 0)
        global_visibility_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        object_visibility_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        global_title = QLabel("전체 표시")
        object_title = QLabel("선택 개체 표시")
        global_title.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        object_title.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        global_visibility_layout.addWidget(global_title)
        object_visibility_layout.addWidget(object_title)
        self.visibility_checkboxes: dict[str, QCheckBox] = {}
        for key, label in [
            ("scale", "스케일바"),
            ("reference", "기준선"),
            ("edge", "경계/이어진 직선"),
            ("guide", "가이드"),
            ("angle", "각도 숫자/호"),
            ("cd", "CD 길이"),
            ("edge_length", "경계 길이"),
            ("range", "인식 범위 영역"),
            ("point_handle", "편집점"),
        ]:
            checkbox = QCheckBox(label)
            checkbox.setChecked(self.visibility[key])
            checkbox.toggled.connect(lambda checked, item_key=key: self.set_visibility(item_key, checked))
            self.visibility_checkboxes[key] = checkbox
            global_visibility_layout.addWidget(checkbox)

        self.object_visibility_checkboxes: dict[str, QCheckBox] = {}
        for key, label in [
            ("show_line", "경계선"),
            ("show_angle", "각도 숫자/호"),
            ("show_edge_length", "경계 길이"),
            ("show_range", "인식 범위 영역"),
        ]:
            checkbox = QCheckBox(label)
            checkbox.setTristate(True)
            checkbox.stateChanged.connect(lambda state, item_key=key: self.set_selected_object_visibility(item_key, state))
            self.object_visibility_checkboxes[key] = checkbox
            object_visibility_layout.addWidget(checkbox)

        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.VLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        visibility_layout.addLayout(global_visibility_layout, 0, 0, Qt.AlignmentFlag.AlignTop)
        visibility_layout.addWidget(separator, 0, 1)
        visibility_layout.addLayout(object_visibility_layout, 0, 2, Qt.AlignmentFlag.AlignTop)
        visibility_layout.setColumnStretch(0, 1)
        visibility_layout.setColumnStretch(2, 1)
        layout.addLayout(visibility_layout)
        layout.addStretch(1)
        self._update_object_visibility_controls()
        dock.setWidget(container)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)

    def _build_scale_preset_dock(self) -> None:
        dock = QDockWidget("스케일 프리셋", self)
        dock.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)
        container = QWidget()
        layout = QVBoxLayout(container)
        self.scale_preset_table = QTableWidget(0, 3)
        self.scale_preset_table.setHorizontalHeaderLabels(["키", "이름", "nm/px"])
        self.scale_preset_table.verticalHeader().setVisible(False)
        self.scale_preset_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.scale_preset_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        layout.addWidget(self.scale_preset_table)

        controls = QHBoxLayout()
        add_button = QPushButton("현재 등록")
        add_button.clicked.connect(self.add_current_scale_preset)
        edit_button = QPushButton("편집")
        edit_button.clicked.connect(self.edit_selected_scale_preset)
        delete_button = QPushButton("삭제")
        delete_button.clicked.connect(self.delete_selected_scale_preset)
        up_button = QPushButton("위")
        up_button.clicked.connect(lambda: self.move_selected_scale_preset(-1))
        down_button = QPushButton("아래")
        down_button.clicked.connect(lambda: self.move_selected_scale_preset(1))
        for button in [add_button, edit_button, delete_button, up_button, down_button]:
            controls.addWidget(button)
        layout.addLayout(controls)
        dock.setWidget(container)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)

    def _build_thumbnail_dock(self) -> None:
        dock = QDockWidget("썸네일", self)
        dock.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)
        container = QWidget()
        container_layout = QVBoxLayout(container)
        controls = QHBoxLayout()
        controls.addWidget(QLabel("열"))
        self.thumbnail_columns_combo = QComboBox()
        self.thumbnail_columns_combo.addItem("1열", 1)
        self.thumbnail_columns_combo.addItem("2열", 2)
        self.thumbnail_columns_combo.setCurrentIndex(1)
        self.thumbnail_columns_combo.currentIndexChanged.connect(self._thumbnail_columns_changed)
        controls.addWidget(self.thumbnail_columns_combo)
        controls.addStretch(1)
        container_layout.addLayout(controls)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self.thumbnail_container = QWidget()
        self.thumbnail_layout = QGridLayout(self.thumbnail_container)
        self.thumbnail_layout.setContentsMargins(6, 6, 6, 6)
        self.thumbnail_layout.setHorizontalSpacing(6)
        self.thumbnail_layout.setVerticalSpacing(8)
        self.thumbnail_empty_label = QLabel("폴더를 열면 이미지가 표시됩니다.")
        self.thumbnail_layout.addWidget(self.thumbnail_empty_label, 0, 0, 1, self.thumbnail_columns)
        scroll.setWidget(self.thumbnail_container)
        container_layout.addWidget(scroll)
        dock.setWidget(container)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock)

    def open_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "이미지 열기",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;All files (*.*)",
        )
        if not path:
            return
        self.browser_root = None
        self.browser_image_paths = [path]
        self.current_browser_index = 0
        self._populate_thumbnails()
        self._load_image_path(path, preserve_calibration=False)

    def open_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "폴더 열기", "")
        if not folder:
            return
        root = Path(folder)
        image_paths = self._scan_folder_images(root)
        if not image_paths:
            QMessageBox.information(self, "폴더 열기", "폴더 안에서 지원되는 이미지 파일을 찾지 못했습니다.")
            return
        self.browser_root = root
        self.browser_image_paths = [str(path) for path in image_paths]
        self.current_browser_index = 0
        self._populate_thumbnails()
        self._load_image_path(self.browser_image_paths[0], preserve_calibration=True)
        self._set_status(f"폴더 로드: {root.name}, 이미지 {len(self.browser_image_paths)}개")

    def _load_image_path(self, path: str, preserve_calibration: bool = False) -> None:
        image = read_image(path)
        if image is None:
            QMessageBox.warning(self, "열기 실패", "이미지를 읽을 수 없습니다.")
            return
        previous_nm_per_px = self.nm_per_px
        self.image_bgr = image
        self.image_path = path
        self.project_path = None
        self.nm_per_px = previous_nm_per_px if preserve_calibration else None
        self.records.clear()
        self.undo_stack.clear()
        self.hidden_angle_measurements.clear()
        self._counter = 1
        self._show_image()
        self._refresh_table()
        self._update_search_range_overlay()
        self._apply_visibility()
        self._select_thumbnail(path)
        calibration_text = f", calibration 유지: {self.nm_per_px:.6g} nm/px" if self.nm_per_px else ""
        self._set_status(f"이미지 로드: {Path(path).name} ({image.shape[1]} x {image.shape[0]} px){calibration_text}")

    def load_browser_image(self, path: str) -> None:
        if path not in self.browser_image_paths:
            return
        self.current_browser_index = self.browser_image_paths.index(path)
        self._load_image_path(path, preserve_calibration=True)

    def load_relative_browser_image(self, delta: int) -> None:
        if not self.browser_image_paths:
            return
        if self.current_browser_index < 0:
            self.current_browser_index = 0
        index = max(0, min(len(self.browser_image_paths) - 1, self.current_browser_index + delta))
        if index == self.current_browser_index and self.image_path:
            return
        self.current_browser_index = index
        self._load_image_path(self.browser_image_paths[index], preserve_calibration=True)

    def _scan_folder_images(self, root: Path) -> list[Path]:
        paths = [
            path
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ]
        return sorted(paths, key=lambda path: (str(path.parent.relative_to(root)).lower(), path.name.lower()))

    def _populate_thumbnails(self) -> None:
        self._clear_thumbnail_layout()
        self.thumbnail_buttons.clear()
        if not self.browser_image_paths:
            self.thumbnail_layout.addWidget(self.thumbnail_empty_label, 0, 0, 1, self.thumbnail_columns)
            self.thumbnail_empty_label.show()
            return

        self.thumbnail_empty_label.hide()
        row = 0
        current_folder: Optional[Path] = None
        col = 0
        thumb_width, thumb_height, icon_width, icon_height = self._thumbnail_dimensions()
        for image_path in self.browser_image_paths:
            path = Path(image_path)
            folder = path.parent
            if folder != current_folder:
                if col != 0:
                    row += 1
                    col = 0
                header_text = self._thumbnail_folder_label(folder)
                header = QLabel(header_text)
                header.setStyleSheet(
                    "font-weight:700; font-size:13px; padding:8px 8px; "
                    "margin-top:8px; background:#203040; color:#ffffff; "
                    "border-left:4px solid #4cc9f0; border-radius:2px;"
                )
                self.thumbnail_layout.addWidget(header, row, 0, 1, self.thumbnail_columns)
                row += 1
                current_folder = folder
                col = 0

            button = QPushButton()
            button.setCheckable(True)
            button.setIcon(self._thumbnail_icon(path))
            button.setIconSize(QSize(icon_width, icon_height))
            button.setFixedSize(thumb_width, thumb_height)
            button.setToolTip(str(path))
            button.clicked.connect(lambda checked=False, selected_path=str(path): self.load_browser_image(selected_path))
            self.thumbnail_layout.addWidget(button, row, col)
            self.thumbnail_buttons[str(path)] = button
            col += 1
            if col >= self.thumbnail_columns:
                row += 1
                col = 0
        self.thumbnail_layout.setRowStretch(row + 1, 1)
        self._select_thumbnail(self.image_path)

    def _thumbnail_columns_changed(self) -> None:
        self.thumbnail_columns = int(self.thumbnail_columns_combo.currentData())
        self._populate_thumbnails()

    def _thumbnail_dimensions(self) -> tuple[int, int, int, int]:
        if self.thumbnail_columns == 1:
            return (188, 136, 176, 124)
        return (92, 78, 84, 66)

    def _clear_thumbnail_layout(self) -> None:
        while self.thumbnail_layout.count():
            item = self.thumbnail_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)

    def _thumbnail_folder_label(self, folder: Path) -> str:
        if self.browser_root is None:
            return folder.name or str(folder)
        try:
            relative = folder.relative_to(self.browser_root)
        except ValueError:
            return str(folder)
        if str(relative) == ".":
            return self.browser_root.name
        return f"{self.browser_root.name} / {relative}"

    def _thumbnail_icon(self, path: Path) -> QIcon:
        image = read_image(path)
        if image is None:
            _, _, icon_width, icon_height = self._thumbnail_dimensions()
            pixmap = QPixmap(icon_width, icon_height)
            pixmap.fill(QColor("#333333"))
            return QIcon(pixmap)
        rgb = bgr_to_rgb8_for_display(image)
        h, w = rgb.shape[:2]
        qimage = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format.Format_RGB888).copy()
        _, _, icon_width, icon_height = self._thumbnail_dimensions()
        pixmap = QPixmap.fromImage(qimage).scaled(
            icon_width,
            icon_height,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        return QIcon(pixmap)

    def _select_thumbnail(self, path: Optional[str]) -> None:
        for button_path, button in self.thumbnail_buttons.items():
            checked = path is not None and button_path == path
            button.setChecked(checked)
            button.setStyleSheet(
                "border:2px solid #4cc9f0; background:#263642;"
                if checked
                else ""
            )

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
        edge_mode = edge_detection.get("edge_mode", self.edge_mode_combo.currentData())
        if edge_mode == "curve":
            edge_mode = "polyline"
        mode_index = self.edge_mode_combo.findData(edge_mode)
        if mode_index >= 0:
            self.edge_mode_combo.setCurrentIndex(mode_index)
        self.search_radius_spin.setValue(int(edge_detection.get("search_radius_px", self.search_radius_spin.value())))
        segment_size_px = edge_detection.get("segment_size_px")
        if segment_size_px is None and "curve_sensitivity" in edge_detection:
            segment_size_px = legacy_sensitivity_to_segment_size_px(edge_detection["curve_sensitivity"])
        self.curve_sensitivity_spin.setValue(int(segment_size_px or self.curve_sensitivity_spin.value()))
        self.show_search_range_checkbox.setChecked(bool(edge_detection.get("show_search_range", True)))
        cd_mode = payload.get("cd_segment_mode")
        cd_index = self.cd_segment_combo.findData(cd_mode)
        if cd_index >= 0:
            self.cd_segment_combo.setCurrentIndex(cd_index)
        visibility = payload.get("visibility", {})
        for key, visible in visibility.items():
            if key in self.visibility:
                self.visibility[key] = bool(visible)
                if hasattr(self, "visibility_checkboxes") and key in self.visibility_checkboxes:
                    self.visibility_checkboxes[key].setChecked(bool(visible))
        self.scale_presets = [
            ScalePreset(
                name=item.get("name", f"Preset {idx + 1}"),
                nm_per_px=float(item["nm_per_px"]),
                bar_px=float(item.get("bar_px", 100.0)),
                bar_nm=float(item["bar_nm"]) if item.get("bar_nm") is not None else None,
                start=tuple(item["start"]) if item.get("start") is not None else None,
                end=tuple(item["end"]) if item.get("end") is not None else None,
            )
            for idx, item in enumerate(payload.get("scale_presets", []))
            if "nm_per_px" in item
        ][:9]
        self._refresh_scale_preset_table()
        self.structure_templates = [
            template
            for template in (structure_template_from_dict(item) for item in payload.get("structure_templates", []))
            if template.records
        ]
        self._refresh_structure_combo()
        self.records = {}
        self.hidden_angle_measurements.clear()
        self.undo_stack.clear()
        for item in payload.get("records", []):
            self.records[item["id"]] = line_record_from_dict(item)
        self._counter = payload.get("counter", len(self.records) + 1)
        self._show_image()
        self.canvas.redraw_lines(list(self.records.values()))
        self._refresh_table()
        self._update_search_range_overlay()
        self._apply_visibility()
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
                "edge_mode": self.edge_mode_combo.currentData(),
                "search_radius_px": self.search_radius_spin.value(),
                "segment_size_px": self.curve_sensitivity_spin.value(),
                "show_search_range": self.show_search_range_checkbox.isChecked(),
            },
            "visibility": self.visibility,
            "cd_segment_mode": self.cd_segment_combo.currentData(),
            "scale_presets": [asdict(preset) for preset in self.scale_presets],
            "structure_templates": [structure_template_to_dict(template) for template in self.structure_templates],
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
        if self.canvas.cd_items:
            self.calculate_cd_lengths()
        else:
            self.calculate_angles(reset_hidden=False)
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
                    "cd_length_px",
                    "cd_length_nm",
                    "nm_per_px",
                ],
            )
            writer.writeheader()
            for row in self.last_measurements:
                writer.writerow(row)
        self._set_status(f"CSV 저장: {Path(path).name}")

    def _handle_line_created(self, tool: str, start: Point, end: Point, points: Optional[list[Point]]) -> None:
        self.save_undo_snapshot()
        if tool == "scale":
            self._create_scale_line(start, end)
        elif tool == "reference":
            self._create_reference_line(start, end)
        elif tool == "edge":
            self._create_edge_line(start, end, points)
        elif tool == "guide":
            self._create_guide_line(start, end)
        self.canvas.redraw_lines(list(self.records.values()))
        self._refresh_table()
        self._update_search_range_overlay()
        self._apply_visibility()

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

    def add_current_scale_preset(self) -> None:
        if not self.nm_per_px:
            QMessageBox.information(self, "스케일 프리셋", "먼저 스케일바를 캘리브레이션하세요.")
            return
        if len(self.scale_presets) >= 9:
            QMessageBox.information(self, "스케일 프리셋", "숫자키 1~9 슬롯까지만 등록할 수 있습니다.")
            return
        default_name = f"Scale {len(self.scale_presets) + 1}"
        name, ok = QInputDialog.getText(self, "스케일 프리셋", "이름", text=default_name)
        if not ok:
            return
        name = name.strip() or default_name
        scale_records = [record for record in self.records.values() if record.kind == "scale"]
        if scale_records:
            scale_record = scale_records[-1]
            bar_px = record_length(scale_record)
            bar_nm = float(scale_record.value_nm) if scale_record.value_nm is not None else bar_px * float(self.nm_per_px)
            start = tuple(scale_record.start)
            end = tuple(scale_record.end)
        else:
            bar_px = 100.0
            bar_nm = bar_px * float(self.nm_per_px)
            start = None
            end = None
        self.scale_presets.append(
            ScalePreset(name=name, nm_per_px=float(self.nm_per_px), bar_px=bar_px, bar_nm=bar_nm, start=start, end=end)
        )
        self._refresh_scale_preset_table()
        self._set_status(f"{len(self.scale_presets)}번 스케일 프리셋 등록: {name}, {self.nm_per_px:.6g} nm/px, 바 {bar_px:.2f}px")

    def edit_selected_scale_preset(self) -> None:
        row = self._selected_scale_preset_row()
        if row is None:
            return
        preset = self.scale_presets[row]
        name, ok = QInputDialog.getText(self, "스케일 프리셋 이름", "이름", text=preset.name)
        if not ok:
            return
        value, ok = QInputDialog.getDouble(
            self,
            "스케일 프리셋 값",
            "nm/px",
            preset.nm_per_px,
            0.0000001,
            1_000_000.0,
            8,
        )
        if not ok:
            return
        preset.name = name.strip() or preset.name
        preset.nm_per_px = float(value)
        preset.bar_nm = preset.bar_px * preset.nm_per_px
        self._refresh_scale_preset_table(select_row=row)

    def delete_selected_scale_preset(self) -> None:
        row = self._selected_scale_preset_row()
        if row is None:
            return
        del self.scale_presets[row]
        self._refresh_scale_preset_table(select_row=min(row, len(self.scale_presets) - 1))

    def move_selected_scale_preset(self, delta: int) -> None:
        row = self._selected_scale_preset_row()
        if row is None:
            return
        new_row = row + delta
        if not (0 <= new_row < len(self.scale_presets)):
            return
        self.scale_presets[row], self.scale_presets[new_row] = self.scale_presets[new_row], self.scale_presets[row]
        self._refresh_scale_preset_table(select_row=new_row)

    def apply_scale_preset(self, index: int) -> None:
        if not (0 <= index < len(self.scale_presets)):
            return
        preset = self.scale_presets[index]
        self.save_undo_snapshot()
        self.nm_per_px = preset.nm_per_px
        for record_id in [rid for rid, record in self.records.items() if record.kind == "scale"]:
            del self.records[record_id]
        self._create_scale_record_from_preset(preset)
        self.canvas.redraw_lines(list(self.records.values()))
        self._refresh_table()
        self._update_search_range_overlay()
        self._apply_visibility()
        self._set_status(f"{index + 1}번 스케일 적용: {preset.name}, {preset.nm_per_px:.6g} nm/px")

    def _create_scale_record_from_preset(self, preset: ScalePreset) -> None:
        if self.image_bgr is None:
            return
        height, width = self.image_bgr.shape[:2]
        bar_px = max(5.0, min(float(preset.bar_px), max(5.0, width * 0.8)))
        if preset.start is not None and preset.end is not None:
            start, end = self._scale_preset_points_in_image(preset.start, preset.end, width, height)
        else:
            margin = max(12.0, min(width, height) * 0.06)
            start_x = min(max(margin, 0.0), max(0.0, width - bar_px - margin))
            y = max(margin, height - margin)
            start = (float(start_x), float(y))
            end = (float(start_x + bar_px), float(y))
        bar_nm = float(preset.bar_nm) if preset.bar_nm is not None else bar_px * preset.nm_per_px
        record_id = self._next_id("S")
        self.records[record_id] = LineRecord(
            id=record_id,
            kind="scale",
            start=start,
            end=end,
            label=f"{bar_nm:g} nm",
            value_nm=bar_nm,
        )

    @staticmethod
    def _scale_preset_points_in_image(start: Point, end: Point, width: int, height: int) -> tuple[Point, Point]:
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        clamped_start_x = min(max(float(start[0]), 0.0), max(0.0, float(width)))
        clamped_start_y = min(max(float(start[1]), 0.0), max(0.0, float(height)))
        clamped_end_x = clamped_start_x + dx
        clamped_end_y = clamped_start_y + dy
        shift_x = 0.0
        shift_y = 0.0
        if clamped_end_x < 0.0:
            shift_x = -clamped_end_x
        elif clamped_end_x > width:
            shift_x = float(width) - clamped_end_x
        if clamped_end_y < 0.0:
            shift_y = -clamped_end_y
        elif clamped_end_y > height:
            shift_y = float(height) - clamped_end_y
        return (
            (clamped_start_x + shift_x, clamped_start_y + shift_y),
            (clamped_end_x + shift_x, clamped_end_y + shift_y),
        )

    def _selected_scale_preset_row(self) -> Optional[int]:
        rows = self.scale_preset_table.selectionModel().selectedRows()
        if not rows:
            return None
        row = rows[0].row()
        if 0 <= row < len(self.scale_presets):
            return row
        return None

    def _refresh_scale_preset_table(self, select_row: Optional[int] = None) -> None:
        self.scale_preset_table.setRowCount(len(self.scale_presets))
        for row, preset in enumerate(self.scale_presets):
            values = [str(row + 1), preset.name, f"{preset.nm_per_px:.8g}"]
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.scale_preset_table.setItem(row, col, item)
        self.scale_preset_table.resizeColumnsToContents()
        if select_row is not None and 0 <= select_row < len(self.scale_presets):
            self.scale_preset_table.selectRow(select_row)

    def _refresh_structure_combo(self, select_index: Optional[int] = None) -> None:
        if not hasattr(self, "structure_combo"):
            return
        self.structure_combo.blockSignals(True)
        self.structure_combo.clear()
        self.structure_combo.addItem("구조 선택", -1)
        for idx, template in enumerate(self.structure_templates):
            self.structure_combo.addItem(template.name, idx)
        if select_index is not None and 0 <= select_index < len(self.structure_templates):
            self.structure_combo.setCurrentIndex(select_index + 1)
        self.structure_combo.blockSignals(False)

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
            label=self._reference_label(axis),
        )
        self.records[record.id] = record
        self._set_status("기준선을 만들었습니다. 기준 토글을 바꾼 뒤 '이미지 맞춤'으로 수평/수직 전환할 수 있습니다.")

    def _create_edge_line(self, start: Point, end: Point, points: Optional[list[Point]] = None) -> None:
        edge_mode = self.edge_mode_combo.currentData()
        if edge_mode != "polyline":
            points = None
        else:
            points = points or [start, end]
        record = LineRecord(
            id=self._next_id("E"),
            kind="edge",
            start=start,
            end=end,
            label="edge",
            axis=self.axis_combo.currentData(),
            points=points,
            recognition_points=list(points or [start, end]),
            edge_mode=edge_mode,
            search_radius_px=self.search_radius_spin.value() if hasattr(self, "search_radius_spin") else None,
            segment_size_px=self.curve_sensitivity_spin.value() if hasattr(self, "curve_sensitivity_spin") else None,
            edge_segmented=edge_mode == "polyline",
            angle_sector=self.default_angle_sector,
            angle_arc_radius=self.default_angle_arc_radius,
            angle_label_side=self.default_angle_label_side,
            angle_label_gap=self.default_angle_label_gap,
            stroke_color=self.default_stroke_color,
            stroke_width=self.default_stroke_width,
        )
        self.records[record.id] = record
        self._set_status(f"{'이어진 직선' if edge_mode == 'polyline' else '직선'} 경계선을 추가했습니다.")

    def split_edge_segment_for_selection(self, record_id: str, segment_index: int) -> None:
        self._sync_records_from_canvas()
        record = self.records.get(record_id)
        if record is None or record.kind != "edge":
            return
        points = record_points(record)
        if len(points) < 3 or not (0 <= segment_index < len(points) - 1):
            return
        self.save_undo_snapshot()
        original_group = record.object_group
        parts: list[tuple[list[Point], bool]] = []
        before = points[: segment_index + 1]
        selected = points[segment_index : segment_index + 2]
        after = points[segment_index + 1 :]
        if len(before) >= 2:
            parts.append((before, False))
        parts.append((selected, True))
        if len(after) >= 2:
            parts.append((after, False))

        del self.records[record_id]
        selected_id = ""
        for part_points, is_selected in parts:
            new_record = clone_record(record)
            new_record.id = self._next_id("E")
            new_record.start = part_points[0]
            new_record.end = part_points[-1]
            new_record.points = part_points if len(part_points) > 2 else None
            new_record.recognition_points = part_points
            new_record.edge_mode = "polyline" if len(part_points) > 2 else "line"
            new_record.edge_segmented = len(part_points) > 2
            new_record.object_group = None if is_selected else original_group
            new_record.edge_length_label_pos = None
            self.records[new_record.id] = new_record
            if is_selected:
                selected_id = new_record.id

        self.canvas.redraw_lines(list(self.records.values()))
        if selected_id:
            self._select_record_ids({selected_id})
        self.calculate_angles(reset_hidden=False)
        self._update_search_range_overlay()
        self._apply_visibility()
        self._set_status("선택한 세그먼트를 별도 경계선으로 잘라 선택했습니다.")

    def _create_guide_line(self, start: Point, end: Point) -> None:
        axis = "horizontal" if abs(end[0] - start[0]) >= abs(end[1] - start[1]) else "vertical"
        record = LineRecord(
            id=self._next_id("G"),
            kind="guide",
            start=start,
            end=end,
            label=f"{axis} guide",
            axis=axis,
        )
        self.records[record.id] = record
        self._set_status("가이드선을 추가했습니다. 선택 후 방향키로 이동하거나 Delete로 삭제할 수 있습니다.")

    def align_to_reference(self) -> None:
        if self.image_bgr is None:
            return
        self._sync_records_from_canvas()
        reference = self._reference_record()
        if reference is None:
            QMessageBox.information(self, "이미지 맞춤", "먼저 기준선을 그려주세요.")
            return
        self.save_undo_snapshot()
        reference.axis = self.axis_combo.currentData()
        reference.label = self._reference_label(reference.axis)
        angle = line_angle_degrees(reference.start, reference.end)
        target = 0.0 if reference.axis == "horizontal" else 90.0
        rotate_by = angle - target
        lines = list(self.records.values())
        points: list[Point] = []
        point_counts: list[int] = []
        for record in lines:
            current_points = record_points(record)
            points.extend(current_points)
            point_counts.append(len(current_points))
            if record.recognition_points:
                points.extend(record.recognition_points)
                point_counts.append(-len(record.recognition_points))
        rotated, transformed = rotate_image_and_points(self.image_bgr, points, rotate_by)
        cursor = 0
        for record in lines:
            count = point_counts.pop(0)
            next_cursor = cursor + count
            record_transformed = transformed[cursor:next_cursor]
            record.start = record_transformed[0]
            record.end = record_transformed[-1]
            if record.points:
                record.points = record_transformed
            cursor = next_cursor
            if point_counts and point_counts[0] < 0:
                recognition_count = -point_counts.pop(0)
                next_cursor = cursor + recognition_count
                record.recognition_points = transformed[cursor:next_cursor]
                cursor = next_cursor
        self.image_bgr = rotated
        self._show_image(keep_view=False)
        self.canvas.redraw_lines(list(self.records.values()))
        self.calculate_angles(reset_hidden=False)
        self._update_search_range_overlay()
        self._apply_visibility()
        self._set_status(f"이미지를 {rotate_by:.3f}도 회전해 기준을 맞췄습니다.")

    def recognize_edges(self) -> None:
        if self.image_bgr is None:
            return
        self._sync_records_from_canvas()
        selected_ids = set(self.canvas.selected_line_ids())
        edge_records = [record for record in self.records.values() if record.kind == "edge" and record.id in selected_ids]
        if not edge_records:
            QMessageBox.information(self, "인식", "인식할 경계선을 먼저 선택하세요.")
            return
        self.save_undo_snapshot()
        gray = to_gray(self.image_bgr)
        moved = 0
        for chain in self._connected_edge_chains(edge_records):
            source_points = self._chain_points(chain)
            radius = self._edge_search_radius(chain[0][0])
            segment_size_px = self._edge_segment_size(chain[0][0])
            result = snap_polyline_to_gradient(gray, source_points, radius, segment_size_px)
            if result is None:
                continue
            self._apply_snapped_chain(chain, source_points, result.points)
            moved += len(chain)
        self.canvas.redraw_lines(list(self.records.values()))
        self._select_record_ids({record.id for record in edge_records})
        self.calculate_angles(reset_hidden=False)
        self._update_search_range_overlay()
        self._apply_visibility()
        self._set_status(f"{moved}/{len(edge_records)}개 경계선을 선택한 형태로 명도 변화 최대 위치에 맞췄습니다.")

    def _edge_search_radius(self, record: LineRecord) -> int:
        return int(record.search_radius_px or self.search_radius_spin.value())

    def _edge_segment_size(self, record: LineRecord) -> int:
        return int(record.segment_size_px or self.curve_sensitivity_spin.value())

    def _connected_edge_chains(self, records: list[LineRecord]) -> list[list[tuple[LineRecord, bool]]]:
        remaining = list(records)
        chains: list[list[tuple[LineRecord, bool]]] = []
        while remaining:
            record = remaining.pop(0)
            chain: list[tuple[LineRecord, bool]] = [(record, False)]
            changed = True
            while changed:
                changed = False
                chain_start = self._oriented_points(chain[0][0], chain[0][1])[0]
                chain_end = self._oriented_points(chain[-1][0], chain[-1][1])[-1]
                for candidate in list(remaining):
                    points = recognition_points(candidate)
                    candidate_start = points[0]
                    candidate_end = points[-1]
                    if points_close(chain_end, candidate_start):
                        chain.append((candidate, False))
                    elif points_close(chain_end, candidate_end):
                        chain.append((candidate, True))
                    elif points_close(chain_start, candidate_end):
                        chain.insert(0, (candidate, False))
                    elif points_close(chain_start, candidate_start):
                        chain.insert(0, (candidate, True))
                    else:
                        continue
                    remaining.remove(candidate)
                    changed = True
                    break
            chains.append(chain)
        return chains

    @staticmethod
    def _oriented_points(record: LineRecord, reverse: bool) -> list[Point]:
        points = list(recognition_points(record))
        return list(reversed(points)) if reverse else points

    def _chain_points(self, chain: list[tuple[LineRecord, bool]]) -> list[Point]:
        points: list[Point] = []
        for record, reverse in chain:
            record_chain_points = self._oriented_points(record, reverse)
            if points and points_close(points[-1], record_chain_points[0]):
                points.extend(record_chain_points[1:])
            else:
                points.extend(record_chain_points)
        return points

    def _apply_snapped_chain(
        self,
        chain: list[tuple[LineRecord, bool]],
        source_points: list[Point],
        snapped_points: list[Point],
    ) -> None:
        source_total = record_length(LineRecord("_chain", "edge", source_points[0], source_points[-1], points=source_points))
        snapped_total = record_length(LineRecord("_snapped", "edge", snapped_points[0], snapped_points[-1], points=snapped_points))
        if source_total <= 0 or snapped_total <= 0:
            return
        source_cursor = 0.0
        snapped_cursor = 0.0
        for record, reverse in chain:
            original_points = self._oriented_points(record, reverse)
            original_length = record_length(LineRecord("_part", "edge", original_points[0], original_points[-1], points=original_points))
            snapped_next = snapped_total * min(1.0, (source_cursor + original_length) / source_total)
            snapped_slice = polyline_slice(snapped_points, snapped_cursor, snapped_next)
            if reverse:
                snapped_slice = list(reversed(snapped_slice))
            if record.edge_mode in {"curve", "polyline"} or len(snapped_slice) > 2:
                record.points = snapped_slice
                record.edge_segmented = True
                record.start = snapped_slice[0]
                record.end = snapped_slice[-1]
            else:
                record.points = None
                record.edge_segmented = False
                record.start = snapped_slice[0]
                record.end = snapped_slice[-1]
            source_cursor += original_length
            snapped_cursor = snapped_next

    def add_guides(self) -> None:
        if self.image_bgr is None:
            return
        self._sync_records_from_canvas()
        self.save_undo_snapshot()
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
        created = 0
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
            created += 1
        self.canvas.redraw_lines(list(self.records.values()))
        self.calculate_angles(reset_hidden=False)
        self._update_search_range_overlay()
        self._apply_visibility()
        self._set_status(f"{orientation} 가이드 {created}개를 만들었습니다.")

    def clear_guides(self, redraw: bool = True) -> None:
        if redraw:
            self.save_undo_snapshot()
        for record_id in [rid for rid, record in self.records.items() if record.kind == "guide"]:
            del self.records[record_id]
        self.canvas.clear_angle_items()
        self.canvas.clear_cd_items()
        if redraw:
            self.canvas.redraw_lines(list(self.records.values()))
            self._refresh_table()
            self._update_search_range_overlay()
            self._apply_visibility()
            self._set_status("가이드를 지웠습니다.")

    def calculate_cd_lengths(self) -> None:
        if self.image_bgr is None:
            return
        self._sync_records_from_canvas()
        edges = [record for record in self.records.values() if record.kind == "edge"]
        guides = [record for record in self.records.values() if record.kind == "guide"]
        if len(edges) < 2 or not guides:
            QMessageBox.information(self, "CD 길이", "CD 길이를 재려면 경계선 2개 이상과 가이드선이 필요합니다.")
            return
        self.calculate_angles(reset_hidden=False)
        self._sync_records_from_canvas()
        edges = [record for record in self.records.values() if record.kind == "edge"]
        guides = [record for record in self.records.values() if record.kind == "guide"]

        mode = self.cd_segment_combo.currentData()
        created = 0
        for guide in guides:
            guide_line = (guide.start, guide.end)
            crosses: list[tuple[float, Point, str]] = []
            for edge in edges:
                for cross, _edge_angle in polyline_intersections(edge, guide_line):
                    fraction = line_fraction(cross, guide_line)
                    if -0.0001 <= fraction <= 1.0001:
                        crosses.append((fraction, cross, edge.id))
            crosses.sort(key=lambda item: item[0])
            filtered: list[tuple[float, Point, str]] = []
            for item in crosses:
                if filtered and abs(item[0] - filtered[-1][0]) < 0.0005 and item[2] == filtered[-1][2]:
                    continue
                filtered.append(item)
            for idx in range(len(filtered) - 1):
                if not cd_segment_allowed(idx, str(mode)):
                    continue
                _fraction_a, point_a, edge_a = filtered[idx]
                _fraction_b, point_b, edge_b = filtered[idx + 1]
                if edge_a == edge_b:
                    continue
                length_px = line_length(point_a, point_b)
                if length_px <= 0:
                    continue
                midpoint = ((point_a[0] + point_b[0]) / 2.0, (point_a[1] + point_b[1]) / 2.0)
                nx, ny = normal_for_line(point_a, point_b)
                label_pos = (midpoint[0] + nx * 16.0, midpoint[1] + ny * 16.0)
                if self.nm_per_px:
                    text = f"CD {length_px * self.nm_per_px:.3g} nm"
                    cd_length_nm = length_px * self.nm_per_px
                else:
                    text = f"CD {length_px:.2f} px"
                    cd_length_nm = ""
                self.canvas.add_cd_measurement(point_a, point_b, text, label_pos)
                self.last_measurements.append(
                    {
                        "measurement": f"CD_{guide.id}_{idx + 1}_{edge_a}_{edge_b}",
                        "edge_id": f"{edge_a}|{edge_b}",
                        "guide_id": guide.id,
                        "kind": "cd_length",
                        "x_px": midpoint[0],
                        "y_px": midpoint[1],
                        "angle_deg": "",
                        "edge_length_px": "",
                        "edge_length_nm": "",
                        "cd_length_px": length_px,
                        "cd_length_nm": cd_length_nm,
                        "nm_per_px": self.nm_per_px if self.nm_per_px else "",
                    }
                )
                created += 1
        self._apply_visibility()
        self._set_status(f"CD 길이 {created}개를 표시했습니다.")

    def save_current_structure_template(self) -> None:
        self._sync_records_from_canvas()
        selected_ids = set(self.canvas.selected_line_ids())
        selected_edges = [
            clone_record(record)
            for record_id, record in self.records.items()
            if record.kind == "edge" and (not selected_ids or record_id in selected_ids)
        ]
        guides = [clone_record(record) for record in self.records.values() if record.kind == "guide"]
        if not selected_edges:
            QMessageBox.information(self, "구조 저장", "저장할 경계선이 없습니다. 경계선을 선택하거나 먼저 그려주세요.")
            return
        current_index = self.structure_combo.currentData() if hasattr(self, "structure_combo") else -1
        default_name = ""
        if isinstance(current_index, int) and 0 <= current_index < len(self.structure_templates):
            default_name = self.structure_templates[current_index].name
        if not default_name:
            default_name = f"구조 {len(self.structure_templates) + 1}"
        name, ok = QInputDialog.getText(self, "구조 저장", "구조 이름", text=default_name)
        if not ok:
            return
        name = name.strip()
        if not name:
            return
        template = StructureTemplate(
            name=name,
            records=selected_edges + guides,
            cd_segment_mode=str(self.cd_segment_combo.currentData()),
        )
        replace_index = next((idx for idx, item in enumerate(self.structure_templates) if item.name == name), None)
        if replace_index is None:
            self.structure_templates.append(template)
            replace_index = len(self.structure_templates) - 1
        else:
            self.structure_templates[replace_index] = template
        self._refresh_structure_combo(replace_index)
        self._set_status(f"구조 저장: {name} (경계 {len(selected_edges)}개, 가이드 {len(guides)}개)")

    def paste_selected_structure_template(self) -> None:
        if self.image_bgr is None:
            return
        template = self._selected_structure_template()
        if template is None:
            QMessageBox.information(self, "구조 붙여넣기", "먼저 구조 드롭다운에서 불러올 구조를 선택하세요.")
            return
        self.save_undo_snapshot()
        self._sync_records_from_canvas()
        has_guides = any(record.kind == "guide" for record in self.records.values())
        new_ids: list[str] = []
        added_edges = 0
        added_guides = 0
        for source in template.records:
            if source.kind == "guide" and has_guides:
                continue
            record = clone_record(source)
            prefix = "G" if record.kind == "guide" else "E"
            record.id = self._next_id(prefix)
            self.records[record.id] = record
            new_ids.append(record.id)
            if record.kind == "guide":
                added_guides += 1
            elif record.kind == "edge":
                added_edges += 1
        mode_index = self.cd_segment_combo.findData(template.cd_segment_mode)
        if mode_index >= 0:
            self.cd_segment_combo.setCurrentIndex(mode_index)
        self.canvas.redraw_lines(list(self.records.values()))
        self.canvas.scene.clearSelection()
        for record_id in new_ids:
            item = self.canvas.line_items.get(record_id)
            if item is not None and self.records.get(record_id, None) and self.records[record_id].kind == "edge":
                item.setSelected(True)
        if any(record.kind == "guide" for record in self.records.values()) and len(
            [record for record in self.records.values() if record.kind == "edge"]
        ) >= 2:
            self.calculate_cd_lengths()
        else:
            self.calculate_angles(reset_hidden=False)
        self._update_search_range_overlay()
        self._apply_visibility()
        skipped = " 기존 가이드를 유지했습니다." if has_guides else ""
        self._set_status(f"구조 붙여넣기: {template.name} (경계 {added_edges}개, 가이드 {added_guides}개).{skipped}")

    def export_selected_structure_template(self) -> None:
        template = self._selected_structure_template()
        if template is None:
            QMessageBox.information(self, "구조 공유", "공유할 구조를 먼저 선택하세요.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "구조 공유", f"{template.name}.anglecal.structure.json", "Angle Cal Structure (*.anglecal.structure.json)")
        if not path:
            return
        if not path.endswith(".anglecal.structure.json"):
            path += ".anglecal.structure.json"
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(structure_template_to_dict(template), handle, ensure_ascii=False, indent=2)
        self._set_status(f"구조 공유 파일 저장: {Path(path).name}")

    def import_structure_template(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "구조 가져오기", "", "Angle Cal Structure (*.anglecal.structure.json *.json)")
        if not path:
            return
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        template = structure_template_from_dict(payload)
        if not template.records:
            QMessageBox.warning(self, "구조 가져오기", "구조 파일에 경계선이나 가이드가 없습니다.")
            return
        replace_index = next((idx for idx, item in enumerate(self.structure_templates) if item.name == template.name), None)
        if replace_index is None:
            self.structure_templates.append(template)
            replace_index = len(self.structure_templates) - 1
        else:
            self.structure_templates[replace_index] = template
        self._refresh_structure_combo(replace_index)
        self._set_status(f"구조 가져오기: {template.name}")

    def delete_selected_structure_template(self) -> None:
        index = self.structure_combo.currentData() if hasattr(self, "structure_combo") else -1
        if not isinstance(index, int) or not (0 <= index < len(self.structure_templates)):
            return
        name = self.structure_templates[index].name
        del self.structure_templates[index]
        self._refresh_structure_combo()
        self._set_status(f"구조 삭제: {name}")

    def _selected_structure_template(self) -> Optional[StructureTemplate]:
        index = self.structure_combo.currentData() if hasattr(self, "structure_combo") else -1
        if isinstance(index, int) and 0 <= index < len(self.structure_templates):
            return self.structure_templates[index]
        return None

    def edit_angle_display_for_selected_edges(self) -> None:
        self._sync_records_from_canvas()
        selected_ids = self.canvas.selected_line_ids()
        edges = [
            self.records[record_id]
            for record_id in selected_ids
            if record_id in self.records and self.records[record_id].kind == "edge"
        ]
        editing_all = not edges
        if editing_all:
            edges = [record for record in self.records.values() if record.kind == "edge"]
        base = edges[0] if edges else None
        dialog = AngleDisplaySettingsDialog(
            base.angle_sector if base is not None else self.default_angle_sector,
            base.angle_arc_radius if base is not None else self.default_angle_arc_radius,
            base.angle_label_side if base is not None else self.default_angle_label_side,
            base.angle_label_gap if base is not None else self.default_angle_label_gap,
            self,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        self.save_undo_snapshot()
        sector = int(dialog.sector_combo.currentData())
        arc_radius = float(dialog.arc_radius_spin.value())
        label_side = str(dialog.label_side_combo.currentData())
        label_gap = float(dialog.label_gap_spin.value())
        if editing_all:
            self.default_angle_sector = sector
            self.default_angle_arc_radius = arc_radius
            self.default_angle_label_side = label_side
            self.default_angle_label_gap = label_gap
        for edge in edges:
            edge.angle_sector = sector
            edge.angle_arc_radius = arc_radius
            edge.angle_label_side = label_side
            edge.angle_label_gap = label_gap
        self.calculate_angles(reset_hidden=False)
        target_text = "전체 경계선" if editing_all else "선택한 경계선"
        self._set_status(f"{target_text} {len(edges)}개의 각도 표시 설정을 바꿨습니다.")

    def calculate_angles(self, reset_hidden: bool = True) -> None:
        if self.image_bgr is None:
            return
        if reset_hidden:
            self.hidden_angle_measurements.clear()
        self._sync_records_from_canvas()
        self.canvas.clear_angle_items()
        self.canvas.clear_cd_items()
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
            if has_segmented_edge_angle(edge):
                continue
            edge_angle = record_angle(edge)
            angle = acute_angle_difference(edge_angle, reference_angle)
            midpoint = ((edge.start[0] + edge.end[0]) / 2.0, (edge.start[1] + edge.end[1]) / 2.0)
            label_pos = self._label_position(midpoint, edge_angle, reference_angle, 34.0)
            measurement_id = f"{edge.id}_to_{reference_name}"
            if edge.show_angle and measurement_id not in self.hidden_angle_measurements:
                self.canvas.add_angle_annotation(
                    f"{angle:.2f}°",
                    label_pos,
                    parent_record_id=edge.id,
                    measurement_id=measurement_id,
                )
            length_px = record_length(edge)
            self.last_measurements.append(
                {
                    "measurement": measurement_id,
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
            for guide in guides:
                guide_line = (guide.start, guide.end)
                guide_angle = line_angle_degrees(guide.start, guide.end)
                crosses = polyline_intersections(edge, guide_line)
                for cross_idx, (cross, edge_angle) in enumerate(crosses, start=1):
                    arc_start, arc_end, angle = angle_sector_geometry(edge_angle, guide_angle, edge.angle_sector)
                    label_pos = angle_label_position_for_sector(
                        cross,
                        arc_start,
                        angle,
                        edge.angle_arc_radius,
                        edge.angle_label_side,
                        edge.angle_label_gap,
                    )
                    length_px = record_length(edge)
                    suffix = f"_{cross_idx}" if len(crosses) > 1 else ""
                    measurement_id = f"{edge.id}_x_{guide.id}{suffix}"
                    if edge.show_angle and measurement_id not in self.hidden_angle_measurements:
                        self.canvas.add_angle_annotation(
                            f"{angle:.2f}°",
                            label_pos,
                            center=cross,
                            angle_a=arc_start,
                            angle_b=arc_end,
                            radius=edge.angle_arc_radius,
                            parent_record_id=edge.id,
                            measurement_id=measurement_id,
                        )
                    self.last_measurements.append(
                        {
                            "measurement": measurement_id,
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
        self._apply_visibility()
        self._set_status(f"각도 {len(self.last_measurements)}개를 계산했습니다. 숫자 라벨은 선택해서 옮길 수 있습니다.")

    def delete_selected(self) -> None:
        selected = set(self.canvas.selected_line_ids())
        selected_angle_items = self.canvas.selected_angle_items()
        selected_edge_length_items = self.canvas.selected_edge_length_items()
        selected_point_handles = self.canvas.selected_point_handles()
        if not selected and not selected_angle_items and not selected_edge_length_items and not selected_point_handles:
            return
        if not selected and not selected_angle_items and not selected_edge_length_items and selected_point_handles:
            self.save_undo_snapshot()
            if self.canvas.delete_selected_point_handles():
                self._sync_records_from_canvas()
                self.calculate_angles(reset_hidden=False)
                self._update_search_range_overlay()
                self._apply_visibility()
            self._set_status("선택한 편집점을 삭제했습니다.")
            return
        self.save_undo_snapshot()
        self.hidden_angle_measurements.update(self.canvas.selected_angle_measurement_ids())
        for parent_id in self.canvas.selected_edge_length_parent_ids():
            record = self.records.get(parent_id)
            if record is not None:
                record.show_edge_length = False
        if selected:
            self.canvas.clear_point_handles()
        for record_id in selected:
            self.records.pop(record_id, None)
        if selected:
            self.canvas.redraw_lines(list(self.records.values()))
            self.calculate_angles(reset_hidden=False)
            self._update_search_range_overlay()
            self._apply_visibility()
        if selected_angle_items and not selected:
            self.canvas.remove_angle_items(selected_angle_items)
            self._refresh_table()
        if selected_edge_length_items and not selected:
            self.canvas.remove_edge_length_items(selected_edge_length_items)
            self._update_object_visibility_controls()
        deleted_count = len(selected) + len(selected_angle_items) + len(selected_edge_length_items)
        self._set_status(f"{deleted_count}개 개체를 삭제했습니다.")

    def group_selected_objects(self) -> None:
        self._sync_records_from_canvas()
        selected_ids = [
            record_id
            for record_id in self.canvas.selected_line_ids()
            if record_id in self.records and self.records[record_id].kind in {"edge", "scale", "guide"}
        ]
        selected_ids = list(dict.fromkeys(selected_ids))
        if len(selected_ids) < 2:
            self._set_status("그룹화하려면 개체를 2개 이상 선택하세요.")
            return
        self.save_undo_snapshot()
        group_id = self._next_object_group_id()
        for record_id in selected_ids:
            self.records[record_id].object_group = group_id
        self.canvas.update_group_boxes(list(self.records.values()))
        self._select_record_ids(set(selected_ids))
        self._set_status(f"{len(selected_ids)}개 개체를 그룹화했습니다.")

    def ungroup_selected_objects(self) -> None:
        self._sync_records_from_canvas()
        selected_ids = [
            record_id
            for record_id in self.canvas.selected_line_ids()
            if record_id in self.records and self.records[record_id].object_group
        ]
        if not selected_ids:
            self._set_status("그룹 해제할 개체를 선택하세요.")
            return
        group_ids = {self.records[record_id].object_group for record_id in selected_ids}
        self.save_undo_snapshot()
        changed = 0
        for record in self.records.values():
            if record.object_group in group_ids:
                record.object_group = None
                changed += 1
        self.canvas.update_group_boxes(list(self.records.values()))
        self._set_status(f"{changed}개 개체의 그룹을 해제했습니다.")

    def _next_object_group_id(self, reserved: Optional[set[str]] = None) -> str:
        used = {record.object_group for record in self.records.values() if record.object_group}
        if reserved:
            used.update(reserved)
        idx = 1
        while f"OG{idx}" in used:
            idx += 1
        return f"OG{idx}"

    def _expand_object_group_selection(self) -> None:
        if self._expanding_object_group_selection:
            return
        if getattr(self.canvas, "_selection_filter", None) is not None:
            return
        selected_ids = self.canvas.selected_line_ids()
        group_ids = {
            self.records[record_id].object_group
            for record_id in selected_ids
            if record_id in self.records and self.records[record_id].object_group
        }
        if not group_ids:
            return
        target_ids = {
            record.id
            for record in self.records.values()
            if record.object_group in group_ids
        }
        if target_ids.issubset(set(selected_ids)):
            return
        self._select_record_ids(target_ids)

    def _select_record_ids(self, record_ids: set[str]) -> None:
        self._expanding_object_group_selection = True
        try:
            for item in self.canvas.line_items.values():
                item.setSelected(item.record_id in record_ids)
        finally:
            self._expanding_object_group_selection = False

    def copy_selected_parent_objects(self) -> None:
        self._sync_records_from_canvas()
        selected_ids = self.canvas.selected_line_ids()
        copied = [
            clone_record(self.records[record_id])
            for record_id in selected_ids
            if record_id in self.records and self.records[record_id].kind in {"edge", "scale"}
        ]
        if not copied:
            self._set_status("복사할 상위개체를 선택하세요. 각도 숫자/호는 복사 대상이 아닙니다.")
            return
        self.record_clipboard = copied
        self._paste_offset_steps = 0
        QApplication.clipboard().setText(json.dumps([asdict(record) for record in copied], ensure_ascii=False))
        self._set_status(f"상위개체 {len(copied)}개를 복사했습니다. Ctrl+V로 붙여넣을 수 있습니다.")

    def paste_parent_objects(self) -> None:
        if self.image_bgr is None:
            return
        if not self.record_clipboard:
            self._set_status("붙여넣을 상위개체가 없습니다. 먼저 Ctrl+C로 복사하세요.")
            return
        self.save_undo_snapshot()
        self._sync_records_from_canvas()
        self._paste_offset_steps += 1
        offset = 14.0 * self._paste_offset_steps
        new_ids: list[str] = []
        group_map: dict[str, str] = {}
        for source in self.record_clipboard:
            record = clone_record(source)
            prefix = "S" if record.kind == "scale" else "E"
            record.id = self._next_id(prefix)
            if record.object_group:
                if record.object_group not in group_map:
                    group_map[record.object_group] = self._next_object_group_id(set(group_map.values()))
                record.object_group = group_map[record.object_group]
            record.start = offset_point(record.start, offset, offset)
            record.end = offset_point(record.end, offset, offset)
            if record.points:
                record.points = [offset_point(point, offset, offset) for point in record.points]
                record.start = record.points[0]
                record.end = record.points[-1]
            if record.recognition_points:
                record.recognition_points = [offset_point(point, offset, offset) for point in record.recognition_points]
            self.records[record.id] = record
            new_ids.append(record.id)
        self.canvas.redraw_lines(list(self.records.values()))
        self.canvas.scene.clearSelection()
        for record_id in new_ids:
            item = self.canvas.line_items.get(record_id)
            if item is not None:
                item.setSelected(True)
        self.calculate_angles(reset_hidden=False)
        self._update_search_range_overlay()
        self._apply_visibility()
        self._set_status(f"상위개체 {len(new_ids)}개를 붙여넣었습니다. 하위 각도 표시들은 새로 계산됩니다.")

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

    def set_current_tool(self, tool: str) -> None:
        self.current_tool = tool
        self.canvas.set_tool(tool)
        if hasattr(self, "tool_buttons") and tool in self.tool_buttons:
            self.tool_buttons[tool].setChecked(True)

    def _edge_mode_changed(self) -> None:
        mode = self.edge_mode_combo.currentData()
        if self._updating_detection_controls:
            self.canvas.set_edge_draw_mode(mode)
            return
        self.canvas.set_edge_draw_mode(mode)
        selected_ids = set(self.canvas.selected_line_ids())
        changed = 0
        if selected_ids:
            self._sync_records_from_canvas()
            self.save_undo_snapshot()
            for record_id in selected_ids:
                record = self.records.get(record_id)
                if record is None or record.kind != "edge":
                    continue
                record.edge_mode = mode
                if mode == "line":
                    record.points = None
                    record.edge_segmented = False
                    record.recognition_points = [record.start, record.end]
                elif mode == "polyline":
                    record.recognition_points = record_points(record)
                changed += 1
            if changed:
                self.canvas.redraw_lines(list(self.records.values()))
                for record_id in selected_ids:
                    item = self.canvas.line_items.get(record_id)
                    if item is not None:
                        item.setSelected(True)
                self.calculate_angles(reset_hidden=False)
                self._update_search_range_overlay()
                self._apply_visibility()
        mode_label = "이어진 직선" if mode == "polyline" else "직선"
        if changed:
            self._set_status(f"선택한 경계선 {changed}개를 {mode_label} 모드로 바꿨습니다.")
        elif mode == "polyline":
            self._set_status("경계 형태: 이어진 직선. 경계선 도구에서 점을 찍고 더블클릭 또는 Enter로 확정합니다.")
        else:
            self._set_status("경계 형태: 직선. 경계선 도구에서 드래그로 선분을 긋습니다.")

    def apply_selected_style(self) -> None:
        if not hasattr(self, "stroke_color_combo"):
            return
        color = str(self.stroke_color_combo.currentData())
        width = float(self.stroke_width_spin.value())
        selected_ids = [
            record_id
            for record_id in self.canvas.selected_line_ids()
            if record_id in self.records and self.records[record_id].kind in {"scale", "reference", "edge", "guide"}
        ]
        if not selected_ids:
            self.default_stroke_color = color
            self.default_stroke_width = width
            self._set_status(f"기본 선 양식: 색 {color}, 두께 {width:.1f}px")
            return
        self._sync_records_from_canvas()
        for record_id in selected_ids:
            record = self.records[record_id]
            record.stroke_color = color
            record.stroke_width = width
            item = self.canvas.line_items.get(record_id)
            if item is not None:
                item.setPen(self.canvas._pen_for_record(record))
        self.canvas.update_group_boxes(list(self.records.values()))
        self._set_status(f"선택 개체 {len(selected_ids)}개의 선 양식을 바꿨습니다.")

    def _axis_changed(self) -> None:
        if self.current_tool != "reference":
            return
        reference = self._reference_record()
        if reference is None:
            return
        reference.axis = self.axis_combo.currentData()
        reference.label = self._reference_label(reference.axis)
        self._refresh_table()
        self._set_status("기준선 축을 바꿨습니다. '이미지 맞춤'을 누르면 새 축 기준으로 정렬됩니다.")

    def activate_select_tool(self) -> None:
        self.canvas.cancel_interaction()
        self.set_current_tool("select")
        self._set_status("선택 도구: 드래그 박스로 개체를 선택하고, Delete로 삭제할 수 있습니다.")

    def scale_selected_objects(self, factor: float) -> None:
        self._sync_records_from_canvas()
        selected_ids = set(self.canvas.selected_line_ids())
        selected_angle_items = self.canvas.selected_angle_items()
        if not selected_ids and not selected_angle_items:
            return

        bounds = self.canvas.selected_persistent_bounds()
        if bounds is None and selected_angle_items:
            bounds = selected_angle_items[0].sceneBoundingRect()
            for item in selected_angle_items[1:]:
                bounds = bounds.united(item.sceneBoundingRect())
        if bounds is None or bounds.width() <= 0 and bounds.height() <= 0:
            return

        center = (bounds.center().x(), bounds.center().y())
        if selected_ids:
            for record_id in selected_ids:
                record = self.records.get(record_id)
                if record is None:
                    continue
                points = [scale_point(point, center, factor) for point in record_points(record)]
                record.start = points[0]
                record.end = points[-1]
                if record.points:
                    record.points = points
                if record.recognition_points:
                    record.recognition_points = [scale_point(point, center, factor) for point in record.recognition_points]
            self.canvas.redraw_lines(list(self.records.values()))
            for record_id in selected_ids:
                item = self.canvas.line_items.get(record_id)
                if item is not None:
                    item.setSelected(True)

        for item in selected_angle_items:
            origin = item.mapFromScene(QPointF(center[0], center[1]))
            item.setTransformOriginPoint(origin)
            item.setScale(max(0.05, item.scale() * factor))
            item.setSelected(True)

        self._refresh_table()
        self._update_search_range_overlay()
        self._apply_visibility()

    def open_edge_detection_settings(self) -> None:
        dialog = EdgeDetectionSettingsDialog(
            self.search_radius_spin.value(),
            self.curve_sensitivity_spin.value(),
            self.show_search_range_checkbox.isChecked(),
            self,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        self.search_radius_spin.setValue(dialog.radius_spin.value())
        self.curve_sensitivity_spin.setValue(dialog.sensitivity_spin.value())
        self.show_search_range_checkbox.setChecked(dialog.overlay_checkbox.isChecked())
        self._edge_detection_settings_changed()

    def _edge_detection_settings_changed(self) -> None:
        if self._updating_detection_controls:
            return
        selected_edges = self._selected_edge_records()
        if selected_edges:
            radius = self.search_radius_spin.value()
            segment_size_px = self.curve_sensitivity_spin.value()
            for edge in selected_edges:
                edge.search_radius_px = radius
                edge.segment_size_px = segment_size_px
        self._update_search_range_overlay()
        if self.sender() in {self.curve_sensitivity_spin, None}:
            self._show_detection_preview()
        radius = self.search_radius_spin.value()
        segment_size_px = self.curve_sensitivity_spin.value()
        target_text = f"선택 경계선 {len(selected_edges)}개" if selected_edges else "기본값"
        if self.show_search_range_checkbox.isChecked():
            self._set_status(f"{target_text} 경계인식 범위: 양쪽 {radius}px, 세그먼트 크기 {segment_size_px}px")
        else:
            self._set_status(f"{target_text} 경계인식 범위: 양쪽 {radius}px, 세그먼트 크기 {segment_size_px}px, 표시 꺼짐")

    def _update_search_range_overlay(self) -> None:
        self._sync_records_from_canvas()
        self.canvas.set_search_range(
            self.search_radius_spin.value(),
            self.show_search_range_checkbox.isChecked() and self.visibility.get("range", True),
            self.visibility.get("range", True),
            list(self.records.values()),
        )
        self._update_edge_length_overlay(sync=False)
        self._apply_visibility()

    def _update_edge_length_overlay(self, sync: bool = True) -> None:
        if sync:
            self._sync_records_from_canvas()
        self.canvas.update_edge_length_overlay(
            list(self.records.values()),
            self.nm_per_px,
            self.visibility.get("edge_length", True),
        )

    def set_visibility(self, key: str, visible: bool) -> None:
        self.visibility[key] = visible
        if key == "range":
            self._update_search_range_overlay()
        elif key == "edge_length":
            self._update_edge_length_overlay()
            self._apply_visibility()
        elif key == "point_handle":
            self.canvas.set_point_handles_visible(visible)
        else:
            self._apply_visibility()

    def set_temporary_edge_tool(self, active: bool) -> None:
        if active:
            self.current_tool = "edge"
            if hasattr(self, "tool_buttons") and "edge" in self.tool_buttons:
                self.tool_buttons["edge"].setChecked(True)
            return
        self.current_tool = self.canvas.current_tool
        if hasattr(self, "tool_buttons") and self.current_tool in self.tool_buttons:
            self.tool_buttons[self.current_tool].setChecked(True)

    def _selected_edge_records(self) -> list[LineRecord]:
        return [
            self.records[record_id]
            for record_id in self.canvas.selected_line_ids()
            if record_id in self.records and self.records[record_id].kind == "edge"
        ]

    def _update_object_visibility_controls(self) -> None:
        if not hasattr(self, "object_visibility_checkboxes"):
            return
        selected_edges = self._selected_edge_records()
        self._update_detection_controls_from_selection(selected_edges)
        self._updating_object_visibility_controls = True
        try:
            for key, checkbox in self.object_visibility_checkboxes.items():
                checkbox.setEnabled(bool(selected_edges))
                if not selected_edges:
                    checkbox.setCheckState(Qt.CheckState.Unchecked)
                    continue
                values = [bool(getattr(record, key)) for record in selected_edges]
                if all(values):
                    checkbox.setCheckState(Qt.CheckState.Checked)
                elif not any(values):
                    checkbox.setCheckState(Qt.CheckState.Unchecked)
                else:
                    checkbox.setCheckState(Qt.CheckState.PartiallyChecked)
        finally:
            self._updating_object_visibility_controls = False

    def set_selected_object_visibility(self, key: str, state: int) -> None:
        if self._updating_object_visibility_controls:
            return
        check_state = Qt.CheckState(state)
        if check_state == Qt.CheckState.PartiallyChecked:
            return
        selected_edges = self._selected_edge_records()
        if not selected_edges:
            self._update_object_visibility_controls()
            return
        visible = check_state == Qt.CheckState.Checked
        for record in selected_edges:
            setattr(record, key, visible)
        if key == "show_angle":
            self.calculate_angles(reset_hidden=False)
        elif key == "show_edge_length":
            self._update_edge_length_overlay()
            self._apply_visibility()
        elif key == "show_range":
            self._update_search_range_overlay()
            self._apply_visibility()
        elif key == "show_line":
            self._apply_visibility()
        self._update_object_visibility_controls()
        self._set_status(f"선택 경계선 {len(selected_edges)}개의 표시 기준을 바꿨습니다.")

    def _update_detection_controls_from_selection(self, selected_edges: Optional[list[LineRecord]] = None) -> None:
        if not hasattr(self, "search_radius_spin") or self._updating_detection_controls:
            return
        selected_edges = selected_edges if selected_edges is not None else self._selected_edge_records()
        if not selected_edges:
            return
        radius_values = [self._edge_search_radius(edge) for edge in selected_edges]
        segment_values = [self._edge_segment_size(edge) for edge in selected_edges]
        first_radius = radius_values[0]
        first_segment = segment_values[0]
        self._updating_detection_controls = True
        try:
            if all(value == first_radius for value in radius_values):
                self.search_radius_spin.setValue(first_radius)
            if all(value == first_segment for value in segment_values):
                self.curve_sensitivity_spin.setValue(first_segment)
            mode_values = [edge.edge_mode for edge in selected_edges]
            if all(value == mode_values[0] for value in mode_values):
                mode_index = self.edge_mode_combo.findData(mode_values[0])
                if mode_index >= 0:
                    self.edge_mode_combo.setCurrentIndex(mode_index)
        finally:
            self._updating_detection_controls = False

    def _apply_visibility(self) -> None:
        for record_id, item in self.canvas.line_items.items():
            record = self.records.get(record_id)
            if record is not None:
                visible = self.visibility.get(record.kind, True) and getattr(record, "show_line", True)
                item.setVisible(visible)
        self.canvas.update_group_boxes(list(self.records.values()))
        for item in self.canvas.angle_items:
            item.setVisible(self.visibility.get("angle", True))
        for item in self.canvas.cd_items:
            item.setVisible(self.visibility.get("cd", True))
        for item in self.canvas.edge_length_items:
            item.setVisible(self.visibility.get("edge_length", True))
        for item in self.canvas.search_range_band_items:
            item.setVisible(self.visibility.get("range", True))
        self.canvas.set_point_handles_visible(self.visibility.get("point_handle", True))

    def _search_range_label(self) -> str:
        radius = self.search_radius_spin.value()
        width = radius * 2
        if self.nm_per_px:
            return f"±{radius}px / {width}px ({width * self.nm_per_px:.3g} nm)"
        return f"±{radius}px / {width}px"

    def _show_detection_preview(self) -> None:
        self.canvas.show_detection_preview(
            self.search_radius_spin.value(),
            self.curve_sensitivity_spin.value(),
            self._search_range_label(),
        )
        self.detection_preview_timer.start(1300)

    def _handle_scene_changed(self) -> None:
        self._refresh_table()
        self._update_search_range_overlay()
        self.canvas.update_group_boxes(list(self.records.values()))
        self._update_object_visibility_controls()
        self.canvas.refresh_point_handles()

    def _sync_records_from_canvas(self) -> None:
        for item in self.canvas.edge_length_items:
            parent_id = item.data(LENGTH_PARENT_KEY)
            if parent_id and str(parent_id) in self.records:
                pos = item.pos()
                self.records[str(parent_id)].edge_length_label_pos = (float(pos.x()), float(pos.y()))
        for record_id, item in self.canvas.line_items.items():
            if record_id not in self.records:
                continue
            record = self.records[record_id]
            if isinstance(item, AnnotationLineItem):
                line = item.line()
                p1 = item.mapToScene(line.p1())
                p2 = item.mapToScene(line.p2())
                record.start = (float(p1.x()), float(p1.y()))
                record.end = (float(p2.x()), float(p2.y()))
                if record.kind != "edge":
                    record.points = None
                elif not record.edge_segmented:
                    record.recognition_points = [record.start, record.end]
            elif isinstance(item, AnnotationCurveItem):
                points = points_from_path_item(item)
                if len(points) >= 2:
                    previous_points = record_points(record)
                    if record.kind == "edge" and record.edge_segmented:
                        delta = uniform_translation_delta(previous_points, points)
                        if delta is not None:
                            if record.recognition_points:
                                record.recognition_points = translated_points(record.recognition_points, delta[0], delta[1])
                        else:
                            record.recognition_points = points
                    record.points = points
                    record.start = points[0]
                    record.end = points[-1]
                    if record.kind == "edge" and not record.edge_segmented:
                        record.recognition_points = points

    def _reference_record(self) -> Optional[LineRecord]:
        for record in self.records.values():
            if record.kind == "reference":
                return record
        return None

    @staticmethod
    def _reference_label(axis: str) -> str:
        return "수평기준선" if axis == "horizontal" else "수직기준선"

    def _refresh_table(self) -> None:
        self._sync_records_from_canvas()
        rows = [record for record in self.records.values() if record.kind != "guide"]
        self.measurement_table.setRowCount(len(rows))
        for row_idx, record in enumerate(rows):
            length_px = record_length(record)
            length_nm = length_px * self.nm_per_px if self.nm_per_px else None
            if record.kind == "edge":
                reference = self._reference_record()
                if reference is not None:
                    angle = acute_angle_difference(
                        record_angle(record),
                        line_angle_degrees(reference.start, reference.end),
                    )
                else:
                    angle = acute_angle_difference(
                        record_angle(record),
                        0.0 if self.axis_combo.currentData() == "horizontal" else 90.0,
                    )
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
