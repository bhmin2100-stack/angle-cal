from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
from html import escape as xml_escape
import json
import math
from pathlib import Path
import sys
from typing import Optional
import zipfile

import numpy as np
from PySide6.QtCore import QPoint, QPointF, QRect, QRectF, QSize, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QAction, QBrush, QColor, QDesktopServices, QIcon, QImage, QKeySequence, QPainter, QPainterPath, QPen, QPixmap, QPolygonF
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
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QDoubleSpinBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QStatusBar,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
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
    angle_label_side: str = "top_right"
    angle_label_gap: float = 14.0
    angle_label_font_size: float = 10.0
    edge_segmented: bool = False
    object_group: Optional[str] = None
    show_line: bool = True
    show_angle: bool = True
    show_line_angle: bool = True
    show_intersection_angle: bool = True
    show_angle_arc: bool = True
    show_range: bool = True
    show_range_label: bool = True
    show_edge_length: bool = True
    edge_length_label_pos: Optional[Point] = None
    stroke_color: Optional[str] = None
    stroke_width: Optional[float] = None
    is_main_guide: bool = False


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


@dataclass
class DataExportOptions:
    scope: str
    selected_items: set[str]
    order_priority: str
    open_after_export: bool = False


ANGLE_GROUP_KEY = 1
ANGLE_MEASUREMENT_KEY = 2
LENGTH_GROUP_KEY = 3
LENGTH_PARENT_KEY = 4
ANGLE_TYPE_KEY = 5
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def cosmetic_pen(color: QColor | str, width: float = 1.0, style: Qt.PenStyle = Qt.PenStyle.SolidLine) -> QPen:
    pen = QPen(QColor(color), width, style)
    pen.setCosmetic(True)
    return pen


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
        self.setPen(cosmetic_pen(QColor("#ffb703"), 1.2))
        self.setZValue(55)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True)
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
    search_range_wheel_requested = Signal(int)
    copy_drag_requested = Signal(object, float, float)
    edit_started = Signal()
    temporary_edge_tool_changed = Signal(bool)
    guide_context_requested = Signal(str, QPoint)
    recognize_requested = Signal()

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
        self.shortcut_overlay_items: list[QGraphicsItem] = []
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
        self._additive_rubberband_items: Optional[set[QGraphicsItem]] = None
        self._shortcut_overlay_visible = False
        self._object_drag_items: list[AnnotationLineItem | AnnotationCurveItem] = []
        self._object_drag_record_ids: list[str] = []
        self._object_drag_start_scene: Optional[QPointF] = None
        self._object_drag_start_positions: dict[QGraphicsItem, QPointF] = {}
        self._object_drag_copy = False
        self._object_drag_constrain = False
        self._object_drag_moved = False
        self._object_drag_last_delta = QPointF(0.0, 0.0)
        self._magnifier_label = QLabel(self.viewport())
        self._magnifier_label.setFixedSize(168, 168)
        self._magnifier_label.setStyleSheet("border: 1px solid rgba(255, 209, 102, 210); background: #101418;")
        self._magnifier_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self._magnifier_label.hide()
        self.scene.selectionChanged.connect(self._expand_angle_group_selection)
        self.scene.selectionChanged.connect(self.refresh_point_handles)

    def set_tool(self, tool: str) -> None:
        self.current_tool = tool
        if tool != "scale":
            self._hide_scale_magnifier()
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
        self._hide_scale_magnifier()
        self._panning = False
        self._resizing = False
        self._clear_object_drag(restore=True)
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
        self.shortcut_overlay_items.clear()
        self.point_handle_items.clear()
        self._hide_scale_magnifier()
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
            box.setPen(cosmetic_pen(QColor(255, 214, 102, 175), 0.8, Qt.PenStyle.DashLine))
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

    def sync_point_handles_to_owners(self) -> None:
        if not self.point_handle_items:
            return
        self._updating_from_point_handle = True
        try:
            for handle in self.point_handle_items:
                owner = handle.owner
                if owner.scene() is None:
                    continue
                points = points_from_path_item(owner)
                if 0 <= handle.point_index < len(points):
                    point = points[handle.point_index]
                    handle.setPos(point[0], point[1])
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
        width = self.screen_to_scene_length(190.0)
        height = self.screen_to_scene_length(72.0)
        margin = self.screen_to_scene_length(8.0)
        visible_center = self.mapToScene(self.viewport().rect().center())
        left = float(visible_center.x()) - width / 2.0
        top = float(visible_center.y()) - height / 2.0
        left = min(max(left, rect.left() + margin), rect.right() - width - margin)
        top = min(max(top, rect.top() + margin), rect.bottom() - height - margin)
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
        panel.setBrush(QBrush(QColor(18, 24, 28, 135)))
        panel.setPen(cosmetic_pen(QColor(76, 201, 240, 140), 1.0))
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
        band.setBrush(QBrush(QColor(0, 220, 110, self._search_range_fill_alpha())))
        band.setPen(cosmetic_pen(QColor(0, 220, 110, self._search_range_pen_alpha()), 1.0, Qt.PenStyle.DashLine))
        band.setZValue(61)
        self.scene.addItem(band)
        self.detection_preview_items.append(band)

        segment_count = max(3, min(24, int(round((line_end[0] - line_start[0]) / max(2, segment_value)))))
        segment_width = (line_end[0] - line_start[0]) / segment_count
        for idx in range(segment_count + 1):
            x = line_start[0] + idx * segment_width
            tick = QGraphicsLineItem(x, center_y - radius - 4.0, x, center_y + radius + 4.0)
            tick.setPen(cosmetic_pen(QColor("#ffd166"), 1.0))
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

    def screen_to_scene_length(self, pixels: float) -> float:
        scale_x = abs(self.transform().m11())
        scale_y = abs(self.transform().m22())
        scale = (scale_x + scale_y) / 2.0
        if scale <= 0:
            return float(pixels)
        return float(pixels) / scale

    def _view_scale(self) -> float:
        scale_x = abs(self.transform().m11())
        scale_y = abs(self.transform().m22())
        return max(1.0, (scale_x + scale_y) / 2.0)

    def _search_range_fill_alpha(self) -> int:
        return max(8, int(24 / (self._view_scale() ** 0.55)))

    def _search_range_pen_alpha(self) -> int:
        return max(70, int(155 / (self._view_scale() ** 0.35)))

    def _scale_line_end_for_modifiers(self, start: QPointF, end: QPointF, modifiers: Qt.KeyboardModifier) -> QPointF:
        if not (modifiers & Qt.KeyboardModifier.ShiftModifier):
            return end
        if modifiers & Qt.KeyboardModifier.ControlModifier:
            return QPointF(start.x(), end.y())
        return QPointF(end.x(), start.y())

    def _scale_tool_scene_point(self, view_pos: QPoint) -> QPointF:
        return self._clamp_to_image(self.mapToScene(view_pos))

    def _update_scale_magnifier(self, view_pos: QPoint) -> None:
        if self.current_tool != "scale" or self.pixmap_item is None:
            self._hide_scale_magnifier()
            return
        scene_point = self._scale_tool_scene_point(view_pos)
        pixmap = self.pixmap_item.pixmap()
        if pixmap.isNull():
            self._hide_scale_magnifier()
            return
        image_x = int(round(scene_point.x()))
        image_y = int(round(scene_point.y()))
        radius = 10
        source = QRect(image_x - radius, image_y - radius, radius * 2 + 1, radius * 2 + 1).intersected(pixmap.rect())
        if source.isEmpty():
            self._hide_scale_magnifier()
            return
        magnified = pixmap.copy(source).scaled(
            160,
            160,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.FastTransformation,
        )
        painter = QPainter(magnified)
        painter.setPen(cosmetic_pen(QColor(255, 209, 102, 230), 1.0))
        center_x = magnified.width() / 2.0
        center_y = magnified.height() / 2.0
        painter.drawLine(QPointF(center_x, 0.0), QPointF(center_x, float(magnified.height())))
        painter.drawLine(QPointF(0.0, center_y), QPointF(float(magnified.width()), center_y))
        painter.end()

        self._magnifier_label.setPixmap(magnified)
        label_size = self._magnifier_label.size()
        pos = view_pos + QPoint(18, 18)
        if pos.x() + label_size.width() > self.viewport().width():
            pos.setX(view_pos.x() - label_size.width() - 18)
        if pos.y() + label_size.height() > self.viewport().height():
            pos.setY(view_pos.y() - label_size.height() - 18)
        pos.setX(max(4, pos.x()))
        pos.setY(max(4, pos.y()))
        self._magnifier_label.move(pos)
        self._magnifier_label.show()
        self._magnifier_label.raise_()

    def _hide_scale_magnifier(self) -> None:
        self._magnifier_label.hide()

    def show_shortcut_overlay(self) -> None:
        self.clear_shortcut_overlay()
        visible_rect = self.mapToScene(self.viewport().rect()).boundingRect()
        width = 430.0
        left = visible_rect.center().x() - width / 2.0
        top = visible_rect.top() + max(18.0, visible_rect.height() * 0.08)
        text = QGraphicsTextItem()
        text.setHtml(
            "<div style='color:white; font-size:10pt; line-height:1.35;'>"
            "<b>단축키</b><br>"
            "Q + 드래그: 경계선만 선택<br>"
            "W + 드래그: 각도 호만 선택<br>"
            "E + 드래그: 각도 숫자만 선택<br>"
            "선택 개체 Ctrl + 드래그: 복사 이동<br>"
            "선택 개체 Shift + 드래그: 수평/수직 이동 고정<br>"
            "빈 화면 Ctrl + 드래그: 선택 추가 / 화면 이동<br>"
            "Space 누르고 있기: 경계선 그리기<br>"
            "Ctrl + 휠: 확대/축소<br>"
            "Ctrl+Shift+C: 서식 복사, Ctrl+V: 서식/개체 붙여넣기<br>"
            "방향키: 선택 개체 이동, Ctrl + 방향키: 1px 이동<br>"
            "Delete: 선택 삭제, Ctrl+Z: 되돌리기"
            "</div>"
        )
        text.setTextWidth(width - 28.0)
        text_rect = text.boundingRect()
        panel = QGraphicsRectItem(0, 0, width, text_rect.height() + 24.0)
        panel.setBrush(QBrush(QColor(18, 24, 32, 205)))
        panel.setPen(cosmetic_pen(QColor(255, 255, 255, 95), 1.0))
        panel.setZValue(95)
        text.setZValue(96)
        panel.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True)
        text.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True)
        panel.setPos(left, top)
        text.setPos(left + 14.0, top + 10.0)
        self.scene.addItem(panel)
        self.scene.addItem(text)
        self.shortcut_overlay_items.extend([panel, text])
        self._shortcut_overlay_visible = True

    def clear_shortcut_overlay(self) -> None:
        for item in self.shortcut_overlay_items:
            if item.scene() is self.scene:
                self.scene.removeItem(item)
        self.shortcut_overlay_items.clear()
        self._shortcut_overlay_visible = False

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
                    item.setPen(cosmetic_pen(QColor(0, 220, 110, self._search_range_pen_alpha()), 1.2, Qt.PenStyle.DashLine))
                    item.setBrush(QBrush(QColor(0, 220, 110, self._search_range_fill_alpha())))
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
            label.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True)
            group_id = f"L_{record.id}"
            label.setData(LENGTH_GROUP_KEY, group_id)
            label.setData(LENGTH_PARENT_KEY, record.id)
            self.scene.addItem(label)
            self.edge_length_items.append(label)
            self.edge_length_groups[group_id] = [label]
            self.edge_length_group_parents[group_id] = record.id

    def add_cd_measurement(self, start: Point, end: Point, text: str, label_center: Point, font_size: float = 10.0) -> list[QGraphicsItem]:
        items: list[QGraphicsItem] = []
        line = QGraphicsLineItem(start[0], start[1], end[0], end[1])
        line.setPen(cosmetic_pen(QColor("#8ecae6"), 2.0))
        line.setZValue(18)
        line.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, False)
        self.scene.addItem(line)
        self.cd_items.append(line)
        items.append(line)

        label = QGraphicsTextItem()
        label.setHtml(
            "<div style='background-color:rgba(3,37,65,185);"
            f"color:#d9f6ff;font-size:{float(font_size):.1f}pt;padding:2px 5px;border-radius:3px;'>"
            f"{text}</div>"
        )
        label_rect = label.boundingRect()
        label.setPos(label_center[0] - label_rect.width() / 2.0, label_center[1] - label_rect.height() / 2.0)
        label.setZValue(31)
        label.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, False)
        label.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True)
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
        angle_type: str = "line",
        show_label: bool = True,
        show_arc: bool = True,
        label_font_size: float = 10.0,
    ) -> list[QGraphicsItem]:
        group_id = f"A{self._angle_counter}"
        self._angle_counter += 1
        measurement_id = measurement_id or group_id
        items: list[QGraphicsItem] = []
        if show_arc and center is not None and angle_a is not None and angle_b is not None:
            items.append(self._create_angle_arc(center, angle_a, angle_b, radius, group_id, measurement_id, angle_type))
        if show_label:
            items.append(self._create_angle_label(text, label_pos, group_id, measurement_id, angle_type, label_font_size))
        if not items:
            return []
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
        angle_type: str,
    ) -> QGraphicsPathItem:
        path = self._arc_path(center, angle_start, angle_end, radius)
        item = QGraphicsPathItem(path)
        item.setPen(cosmetic_pen(QColor("#ffd166"), 2.0))
        item.setZValue(20)
        item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        item.setData(ANGLE_GROUP_KEY, group_id)
        item.setData(ANGLE_MEASUREMENT_KEY, measurement_id)
        item.setData(ANGLE_TYPE_KEY, angle_type)
        self.scene.addItem(item)
        self.angle_items.append(item)
        return item

    def _create_angle_label(
        self,
        text: str,
        pos: Point,
        group_id: str,
        measurement_id: str,
        angle_type: str,
        font_size: float,
    ) -> QGraphicsTextItem:
        item = QGraphicsTextItem()
        item.setHtml(
            "<div style='background-color:rgba(24,24,24,185);"
            f"color:white;font-size:{float(font_size):.1f}pt;padding:2px 5px;border-radius:3px;'>"
            f"{text}</div>"
        )
        item.setDefaultTextColor(QColor("white"))
        item.setPos(pos[0], pos[1])
        item.setZValue(30)
        item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True)
        item.setData(ANGLE_GROUP_KEY, group_id)
        item.setData(ANGLE_MEASUREMENT_KEY, measurement_id)
        item.setData(ANGLE_TYPE_KEY, angle_type)
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

    def _selected_draggable_line_at(self, view_pos: QPoint) -> Optional[AnnotationLineItem | AnnotationCurveItem]:
        for item in self.items(view_pos):
            if (
                isinstance(item, (AnnotationLineItem, AnnotationCurveItem))
                and item.kind in {"edge", "guide"}
                and item.isSelected()
            ):
                return item
        return None

    def _begin_object_drag(self, view_pos: QPoint, modifiers: Qt.KeyboardModifier) -> bool:
        clicked_item = self._selected_draggable_line_at(view_pos)
        if clicked_item is None:
            return False
        selected_items = [
            item
            for item in self.scene.selectedItems()
            if isinstance(item, (AnnotationLineItem, AnnotationCurveItem)) and item.kind in {"edge", "guide"}
        ]
        if clicked_item not in selected_items:
            selected_items.append(clicked_item)
        if not selected_items:
            return False
        self._object_drag_items = selected_items
        self._object_drag_record_ids = [item.record_id for item in selected_items]
        self._object_drag_start_scene = self.mapToScene(view_pos)
        self._object_drag_start_positions = {item: QPointF(item.pos()) for item in selected_items}
        self._object_drag_copy = bool(modifiers & Qt.KeyboardModifier.ControlModifier)
        self._object_drag_constrain = bool(modifiers & Qt.KeyboardModifier.ShiftModifier)
        self._object_drag_moved = False
        self._object_drag_last_delta = QPointF(0.0, 0.0)
        self.setCursor(Qt.CursorShape.SizeAllCursor)
        return True

    @staticmethod
    def _axis_locked_delta(delta: QPointF) -> QPointF:
        if abs(delta.x()) >= abs(delta.y()):
            return QPointF(delta.x(), 0.0)
        return QPointF(0.0, delta.y())

    def _current_object_drag_delta(self, view_pos: QPoint) -> QPointF:
        if self._object_drag_start_scene is None:
            return QPointF(0.0, 0.0)
        raw_delta = self.mapToScene(view_pos) - self._object_drag_start_scene
        if self._object_drag_constrain:
            return self._axis_locked_delta(raw_delta)
        return raw_delta

    def _apply_object_drag_delta(self, delta: QPointF) -> None:
        for item in self._object_drag_items:
            start_pos = self._object_drag_start_positions.get(item)
            if start_pos is None or item.scene() is not self.scene:
                continue
            item.setPos(start_pos + delta)
        self.sync_point_handles_to_owners()
        self._object_drag_last_delta = delta

    def _finish_object_drag(self, modifiers: Qt.KeyboardModifier) -> None:
        delta = self._object_drag_last_delta
        record_ids = list(self._object_drag_record_ids)
        should_copy = self._object_drag_copy and bool(modifiers & Qt.KeyboardModifier.ControlModifier)
        moved = self._object_drag_moved
        if should_copy:
            self._restore_object_drag_positions()
        self._clear_object_drag()
        if not moved:
            return
        if should_copy:
            self.copy_drag_requested.emit(record_ids, float(delta.x()), float(delta.y()))
        else:
            self.scene_changed.emit()

    def _restore_object_drag_positions(self) -> None:
        for item, start_pos in self._object_drag_start_positions.items():
            if item.scene() is self.scene:
                item.setPos(start_pos)
        self.sync_point_handles_to_owners()

    def _clear_object_drag(self, restore: bool = False) -> None:
        if restore:
            self._restore_object_drag_positions()
        self._object_drag_items = []
        self._object_drag_record_ids = []
        self._object_drag_start_scene = None
        self._object_drag_start_positions = {}
        self._object_drag_copy = False
        self._object_drag_constrain = False
        self._object_drag_moved = False
        self._object_drag_last_delta = QPointF(0.0, 0.0)

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
        steps = int(event.angleDelta().y() / 120)
        if steps:
            self.search_range_wheel_requested.emit(steps)
            event.accept()
            return
        super().wheelEvent(event)

    def mousePressEvent(self, event):  # noqa: N802
        self.setFocus(Qt.FocusReason.MouseFocusReason)
        if event.button() == Qt.MouseButton.RightButton:
            item = self.itemAt(event.pos())
            if isinstance(item, AnnotationLineItem) and item.kind == "guide":
                self.guide_context_requested.emit(item.record_id, self.mapToGlobal(event.pos()))
                event.accept()
                return
        if event.button() in (Qt.MouseButton.MiddleButton, Qt.MouseButton.RightButton):
            self._start_pan(event.pos())
            event.accept()
            return

        if (
            event.button() == Qt.MouseButton.LeftButton
            and event.modifiers() & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier)
            and self._begin_object_drag(event.pos(), event.modifiers())
        ):
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

            if self.scene.selectedItems():
                self._additive_rubberband_items = set(self.scene.selectedItems())
            else:
                self._additive_rubberband_items = None

        if (
            event.button() == Qt.MouseButton.LeftButton
            and (self.current_tool == "pan" or event.modifiers() & Qt.KeyboardModifier.ControlModifier)
            and not (
                self.current_tool == "scale"
                and event.modifiers() & Qt.KeyboardModifier.ShiftModifier
            )
            and self._additive_rubberband_items is None
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
            if self.current_tool == "scale":
                self._update_scale_magnifier(event.pos())
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
            self._temp_line.setPen(cosmetic_pen(QColor("#4cc9f0"), 2.0, Qt.PenStyle.DashLine))
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
        if self.current_tool == "scale" and self.pixmap_item is not None:
            self._update_scale_magnifier(event.pos())
        else:
            self._hide_scale_magnifier()

        if self._object_drag_start_scene is not None:
            delta = self._current_object_drag_delta(event.pos())
            if not self._object_drag_moved and (abs(delta.x()) + abs(delta.y())) > 0.01:
                self.edit_started.emit()
                self._object_drag_moved = True
            self._apply_object_drag_delta(delta)
            event.accept()
            return

        if self._panning:
            delta = event.pos() - self._pan_last
            self._pan_last = event.pos()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            event.accept()
            return

        if self._temp_line is not None and self._drawing_start is not None:
            end = self._clamp_to_image(self.mapToScene(event.pos()))
            if self.current_tool == "scale":
                end = self._scale_line_end_for_modifiers(self._drawing_start, end, event.modifiers())
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
        if self._object_drag_start_scene is not None and event.button() == Qt.MouseButton.LeftButton:
            self._finish_object_drag(event.modifiers())
            self._restore_tool_cursor()
            event.accept()
            return

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
            if self.current_tool == "scale":
                end = self._scale_line_end_for_modifiers(start, end, event.modifiers())
            self.scene.removeItem(self._temp_line)
            self._temp_line = None
            self._drawing_start = None
            self._hide_scale_magnifier()
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
        self._restore_additive_rubberband_selection()
        self._apply_selection_filter()
        self.scene_changed.emit()

    def leaveEvent(self, event):  # noqa: N802
        self._hide_scale_magnifier()
        super().leaveEvent(event)

    def _restore_additive_rubberband_selection(self) -> None:
        if self._additive_rubberband_items is None:
            return
        previous_items = self._additive_rubberband_items
        self._additive_rubberband_items = None
        for item in previous_items:
            if item.scene() is self.scene:
                item.setSelected(True)

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
        if (
            not event.isAutoRepeat()
            and event.key() == Qt.Key.Key_Tab
            and event.modifiers() & Qt.KeyboardModifier.ControlModifier
        ):
            self.show_shortcut_overlay()
            event.accept()
            return
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
        if not event.isAutoRepeat() and event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self.recognize_requested.emit()
            event.accept()
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event):  # noqa: N802
        if not event.isAutoRepeat() and event.key() in (Qt.Key.Key_Tab, Qt.Key.Key_Control):
            self.clear_shortcut_overlay()
            event.accept()
            return
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
            self._temp_curve.setPen(cosmetic_pen(QColor("#4cc9f0"), 2.0, Qt.PenStyle.DashLine))
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
        selected_line_items = {
            item
            for item in selected
            if isinstance(item, (AnnotationLineItem, AnnotationCurveItem))
        }
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
            if isinstance(item, PointHandleItem) and item.owner in selected_line_items:
                continue
            if item.flags() & QGraphicsItem.GraphicsItemFlag.ItemIsMovable:
                item.moveBy(dx, dy)
        self.sync_point_handles_to_owners()
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
            "guide": "#ffd166" if record.is_main_guide else "#f7fff7",
        }
        guide_width = 2.0 if record.is_main_guide else 1.2
        width = float(record.stroke_width) if record.stroke_width is not None else (guide_width if record.kind == "guide" else 2.2)
        color = QColor(record.stroke_color or colors.get(record.kind, "#ffffff"))
        if record.kind == "reference" and record.stroke_color is None:
            color.setAlpha(128)
        pen = cosmetic_pen(color, width)
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


def normalize_angle_label_side(value: str) -> str:
    if value in {"top_right", "top_left", "bottom_right", "bottom_left"}:
        return value
    legacy_map = {
        "outside": "top_right",
        "on_arc": "top_right",
        "inside": "bottom_right",
        "right": "top_right",
        "left": "top_left",
        "up": "top_right",
        "down": "bottom_right",
    }
    return legacy_map.get(value, "top_right")


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
        angle_label_side=normalize_angle_label_side(record.angle_label_side),
        angle_label_gap=record.angle_label_gap,
        angle_label_font_size=record.angle_label_font_size,
        edge_segmented=record.edge_segmented,
        object_group=record.object_group,
        show_line=record.show_line,
        show_angle=record.show_angle,
        show_line_angle=record.show_line_angle,
        show_intersection_angle=record.show_intersection_angle,
        show_angle_arc=record.show_angle_arc,
        show_range=record.show_range,
        show_range_label=record.show_range_label,
        show_edge_length=record.show_edge_length,
        edge_length_label_pos=tuple(record.edge_length_label_pos) if record.edge_length_label_pos else None,
        stroke_color=record.stroke_color,
        stroke_width=record.stroke_width,
        is_main_guide=record.is_main_guide,
    )


def line_record_from_dict(item: dict) -> LineRecord:
    raw_points = item.get("points") or []
    raw_recognition_points = item.get("recognition_points") or []
    record_edge_mode = item.get("edge_mode")
    if record_edge_mode == "curve":
        record_edge_mode = "polyline"
    if record_edge_mode not in {"line", "polyline"}:
        record_edge_mode = "polyline" if raw_points else "line"
    legacy_show_angle = bool(item.get("show_angle", True))
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
        angle_label_side=normalize_angle_label_side(str(item.get("angle_label_side", "top_right"))),
        angle_label_gap=float(item.get("angle_label_gap", 14.0)),
        angle_label_font_size=float(item.get("angle_label_font_size", 10.0)),
        edge_segmented=bool(item.get("edge_segmented", bool(raw_points and len(raw_points) > 6))),
        object_group=item.get("object_group"),
        show_line=bool(item.get("show_line", True)),
        show_angle=legacy_show_angle,
        show_line_angle=bool(item.get("show_line_angle", legacy_show_angle)),
        show_intersection_angle=bool(item.get("show_intersection_angle", legacy_show_angle)),
        show_angle_arc=bool(item.get("show_angle_arc", legacy_show_angle)),
        show_range=bool(item.get("show_range", True)),
        show_range_label=bool(item.get("show_range_label", True)),
        show_edge_length=bool(item.get("show_edge_length", True)),
        edge_length_label_pos=tuple(item["edge_length_label_pos"]) if item.get("edge_length_label_pos") is not None else None,
        stroke_color=item.get("stroke_color"),
        stroke_width=float(item["stroke_width"]) if item.get("stroke_width") is not None else None,
        is_main_guide=bool(item.get("is_main_guide", False)),
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
    quadrant_sector = sector_for_visual_quadrant(sectors, sector_index)
    if quadrant_sector is not None:
        return quadrant_sector
    return sectors[sector_index % len(sectors)]


def sector_for_visual_quadrant(
    sectors: list[tuple[float, float, float]],
    quadrant_index: int,
) -> Optional[tuple[float, float, float]]:
    target_quadrant = quadrant_index % 4
    for sector in sectors:
        start, _end, span = sector
        midpoint = math.radians((start + span / 2.0) % 360.0)
        dx = math.cos(midpoint)
        dy = math.sin(midpoint)
        if dx >= 0 and dy <= 0:
            visual_quadrant = 0
        elif dx < 0 and dy <= 0:
            visual_quadrant = 1
        elif dx < 0 and dy > 0:
            visual_quadrant = 2
        else:
            visual_quadrant = 3
        if visual_quadrant == target_quadrant:
            return sector
    return None


def angle_label_position_for_sector(
    center: Point,
    start_angle: float,
    span: float,
    radius: float,
    side: str,
    gap: float,
) -> Point:
    side = normalize_angle_label_side(side)
    distance = radius + gap
    dx = distance if side.endswith("right") else -distance
    dy = -distance if side.startswith("top") else distance
    return (center[0] + dx, center[1] + dy)


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


def position_key(point: Point, priority: str) -> tuple[float, float]:
    if priority == "x":
        return (point[0], point[1])
    return (point[1], point[0])


def cd_segment_allowed(index: int, mode: str) -> bool:
    segment_number = index + 1
    if mode == "odd":
        return segment_number % 2 == 1
    if mode == "even":
        return segment_number % 2 == 0
    return True


def cd_label_center(start: Point, end: Point, side: str, gap: float) -> Point:
    midpoint = ((start[0] + end[0]) / 2.0, (start[1] + end[1]) / 2.0)
    nx, ny = normal_for_line(start, end)
    if abs(ny) < 0.0001:
        nx, ny = 0.0, -1.0
    if side == "above" and ny > 0:
        nx, ny = -nx, -ny
    elif side == "below" and ny < 0:
        nx, ny = -nx, -ny
    return (midpoint[0] + nx * gap, midpoint[1] + ny * gap)


def record_center(record: LineRecord) -> Point:
    points = record_points(record)
    if not points:
        return (0.0, 0.0)
    return (
        sum(point[0] for point in points) / len(points),
        sum(point[1] for point in points) / len(points),
    )


def group_bounds_center(records: list[LineRecord]) -> Point:
    points: list[Point] = []
    for record in records:
        points.extend(record_points(record))
    if not points:
        return (0.0, 0.0)
    min_x = min(point[0] for point in points)
    max_x = max(point[0] for point in points)
    min_y = min(point[1] for point in points)
    max_y = max(point[1] for point in points)
    return ((min_x + max_x) / 2.0, (min_y + max_y) / 2.0)


EXPORT_COLUMNS = ["이미지", "그룹", "그룹번호", "순서", "항목", "측정ID", "경계ID", "가이드ID", "x_px", "y_px", "각도_deg", "길이_px", "길이_nm"]


def xlsx_col_name(index: int) -> str:
    name = ""
    while index:
        index, remainder = divmod(index - 1, 26)
        name = chr(65 + remainder) + name
    return name


def xlsx_cell_xml(row_idx: int, col_idx: int, value: object) -> str:
    ref = f"{xlsx_col_name(col_idx)}{row_idx}"
    if value is None or value == "":
        return f'<c r="{ref}"/>'
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f'<c r="{ref}"><v>{value}</v></c>'
    return f'<c r="{ref}" t="inlineStr"><is><t>{xml_escape(str(value))}</t></is></c>'


def xlsx_sheet_xml(rows: list[dict[str, object]]) -> str:
    all_rows = [dict(zip(EXPORT_COLUMNS, EXPORT_COLUMNS))]
    all_rows.extend(rows)
    row_xml: list[str] = []
    for row_idx, row in enumerate(all_rows, start=1):
        cells = "".join(xlsx_cell_xml(row_idx, col_idx, row.get(column, "")) for col_idx, column in enumerate(EXPORT_COLUMNS, start=1))
        row_xml.append(f'<row r="{row_idx}">{cells}</row>')
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        '<sheetViews><sheetView workbookViewId="0"/></sheetViews>'
        '<sheetFormatPr defaultRowHeight="15"/>'
        f'<sheetData>{"".join(row_xml)}</sheetData>'
        '</worksheet>'
    )


def write_xlsx(path: str, sheets: dict[str, list[dict[str, object]]]) -> None:
    sheet_items = list(sheets.items())
    workbook_sheets = "".join(
        f'<sheet name="{xml_escape(name[:31])}" sheetId="{idx}" r:id="rId{idx}"/>'
        for idx, (name, _rows) in enumerate(sheet_items, start=1)
    )
    workbook_rels = "".join(
        f'<Relationship Id="rId{idx}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{idx}.xml"/>'
        for idx, _item in enumerate(sheet_items, start=1)
    )
    workbook_rels += '<Relationship Id="rIdStyles" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>'
    content_types = "".join(
        f'<Override PartName="/xl/worksheets/sheet{idx}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        for idx, _item in enumerate(sheet_items, start=1)
    )
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "[Content_Types].xml",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
            '<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>'
            f"{content_types}</Types>",
        )
        archive.writestr(
            "_rels/.rels",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
            '</Relationships>',
        )
        archive.writestr(
            "xl/workbook.xml",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
            'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
            f'<sheets>{workbook_sheets}</sheets></workbook>',
        )
        archive.writestr(
            "xl/_rels/workbook.xml.rels",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            f"{workbook_rels}</Relationships>",
        )
        archive.writestr(
            "xl/styles.xml",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
            '<fonts count="1"><font><sz val="11"/><name val="Calibri"/></font></fonts>'
            '<fills count="1"><fill><patternFill patternType="none"/></fill></fills>'
            '<borders count="1"><border/></borders>'
            '<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>'
            '<cellXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/></cellXfs>'
            '</styleSheet>',
        )
        for idx, (_name, rows) in enumerate(sheet_items, start=1):
            archive.writestr(f"xl/worksheets/sheet{idx}.xml", xlsx_sheet_xml(rows))


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
        label_font_size: float,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("각도 표시 편집")
        self.setModal(True)

        self.sector_combo = QComboBox()
        for idx in range(4):
            self.sector_combo.addItem(f"{idx + 1}사분면", idx)
        self.sector_combo.setCurrentIndex(max(0, min(3, int(sector))))

        self.arc_radius_spin = QDoubleSpinBox()
        self.arc_radius_spin.setRange(6.0, 300.0)
        self.arc_radius_spin.setValue(float(arc_radius))
        self.arc_radius_spin.setSuffix(" px")

        self.label_side_combo = QComboBox()
        for label, value in [
            ("호 오른쪽위", "top_right"),
            ("호 왼쪽위", "top_left"),
            ("호 오른쪽아래", "bottom_right"),
            ("호 왼쪽아래", "bottom_left"),
        ]:
            self.label_side_combo.addItem(label, value)
        side_index = self.label_side_combo.findData(normalize_angle_label_side(label_side))
        self.label_side_combo.setCurrentIndex(side_index if side_index >= 0 else 0)

        self.label_gap_spin = QDoubleSpinBox()
        self.label_gap_spin.setRange(0.0, 300.0)
        self.label_gap_spin.setValue(float(label_gap))
        self.label_gap_spin.setSuffix(" px")

        self.label_font_size_spin = QDoubleSpinBox()
        self.label_font_size_spin.setRange(6.0, 72.0)
        self.label_font_size_spin.setSingleStep(1.0)
        self.label_font_size_spin.setValue(float(label_font_size))
        self.label_font_size_spin.setSuffix(" pt")

        layout = QVBoxLayout(self)
        form = QFormLayout()
        form.addRow("각도 호 위치", self.sector_combo)
        form.addRow("각도 호 크기", self.arc_radius_spin)
        form.addRow("숫자 위치", self.label_side_combo)
        form.addRow("숫자 거리", self.label_gap_spin)
        form.addRow("숫자 크기", self.label_font_size_spin)
        layout.addLayout(form)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)


class CdDisplaySettingsDialog(QDialog):
    def __init__(
        self,
        label_side: str,
        label_gap: float,
        label_font_size: float,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("CD 표시 편집")
        self.setModal(True)

        self.label_side_combo = QComboBox()
        self.label_side_combo.addItem("선분 위", "above")
        self.label_side_combo.addItem("선분 아래", "below")
        side_index = self.label_side_combo.findData(label_side)
        self.label_side_combo.setCurrentIndex(side_index if side_index >= 0 else 0)

        self.label_gap_spin = QDoubleSpinBox()
        self.label_gap_spin.setRange(0.0, 300.0)
        self.label_gap_spin.setValue(float(label_gap))
        self.label_gap_spin.setSuffix(" px")

        self.label_font_size_spin = QDoubleSpinBox()
        self.label_font_size_spin.setRange(6.0, 72.0)
        self.label_font_size_spin.setSingleStep(1.0)
        self.label_font_size_spin.setValue(float(label_font_size))
        self.label_font_size_spin.setSuffix(" pt")

        layout = QVBoxLayout(self)
        form = QFormLayout()
        form.addRow("숫자 위치", self.label_side_combo)
        form.addRow("숫자 거리", self.label_gap_spin)
        form.addRow("글씨 크기", self.label_font_size_spin)
        layout.addLayout(form)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)


class DataExportDialog(QDialog):
    def __init__(self, has_multiple_images: bool, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Data Export")
        self.setModal(True)

        self.scope_combo = QComboBox()
        self.scope_combo.addItem("현재 보이는 이미지만", "current")
        self.scope_combo.addItem("현재 작업 프로젝트 전부", "project")

        self.item_checkboxes: dict[str, QCheckBox] = {}
        for key, label in [
            ("line_angle", "선각도"),
            ("intersection_angle", "교점각도"),
            ("cd_length", "CD 길이"),
            ("edge_length", "경계길이"),
        ]:
            checkbox = QCheckBox(label)
            checkbox.setChecked(True)
            self.item_checkboxes[key] = checkbox

        self.order_combo = QComboBox()
        self.order_combo.addItem("위쪽에서 아래 우선", "y")
        self.order_combo.addItem("왼쪽에서 오른쪽 우선", "x")

        self.open_after_export_checkbox = QCheckBox("Data Export 후 파일 열기")
        self.open_after_export_checkbox.setChecked(False)

        layout = QVBoxLayout(self)
        form = QFormLayout()
        form.addRow("내보낼 범위", self.scope_combo)
        form.addRow("정렬 우선순위", self.order_combo)
        layout.addLayout(form)
        layout.addWidget(QLabel("내보낼 항목"))
        for checkbox in self.item_checkboxes.values():
            layout.addWidget(checkbox)
        layout.addWidget(self.open_after_export_checkbox)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def options(self) -> DataExportOptions:
        return DataExportOptions(
            scope=str(self.scope_combo.currentData()),
            selected_items={key for key, checkbox in self.item_checkboxes.items() if checkbox.isChecked()},
            order_priority=str(self.order_combo.currentData()),
            open_after_export=self.open_after_export_checkbox.isChecked(),
        )


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
        self.image_states: dict[str, dict] = {}
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
        self.format_clipboard: Optional[dict[str, object]] = None
        self.clipboard_mode: Optional[str] = None
        self._paste_offset_steps = 0
        self.default_angle_sector = 0
        self.default_angle_arc_radius = 28.0
        self.default_angle_label_side = "top_right"
        self.default_angle_label_gap = 14.0
        self.default_angle_label_font_size = 10.0
        self.cd_label_side = "above"
        self.cd_label_gap = 14.0
        self.cd_label_font_size = 10.0
        self.default_stroke_color = "#ff6b6b"
        self.default_stroke_width = 2.2
        self.hidden_angle_measurements: set[str] = set()
        self.undo_stack: list[dict] = []
        self._restoring_undo = False
        self._updating_object_visibility_controls = False
        self._updating_detection_controls = False
        self._expanding_object_group_selection = False
        self.last_edge_record_id: Optional[str] = None
        self.visibility: dict[str, bool] = {
            "scale": True,
            "reference": True,
            "edge": True,
            "guide": True,
            "line_angle": False,
            "intersection_angle": True,
            "angle_arc": True,
            "cd": True,
            "edge_length": False,
            "range": True,
            "point_handle": True,
        }

        self.canvas = AngleCanvas()
        self.setCentralWidget(self.canvas)
        self.canvas.line_created.connect(self._handle_line_created)
        self.canvas.segment_split_requested.connect(self.split_edge_segment_for_selection)
        self.canvas.scene_changed.connect(self._handle_scene_changed)
        self.canvas.scale_requested.connect(self.scale_selected_objects)
        self.canvas.search_range_wheel_requested.connect(self.adjust_search_range_by_wheel)
        self.canvas.copy_drag_requested.connect(self.duplicate_dragged_objects)
        self.canvas.edit_started.connect(self.save_undo_snapshot)
        self.canvas.temporary_edge_tool_changed.connect(self.set_temporary_edge_tool)
        self.canvas.guide_context_requested.connect(self.open_guide_context_menu)
        self.canvas.recognize_requested.connect(self.recognize_edges)
        self.detection_preview_timer = QTimer(self)
        self.detection_preview_timer.setSingleShot(True)
        self.detection_preview_timer.timeout.connect(self.clear_detection_preview)

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
        self.export_data_action = QAction("Data Export", self)
        self.export_data_action.triggered.connect(self.export_data_xlsx)
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
        self.paste_action.triggered.connect(self.paste_from_clipboard)
        self.copy_format_action = QAction("서식 복사", self)
        self.copy_format_action.setShortcut(QKeySequence("Ctrl+Shift+C"))
        self.copy_format_action.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
        self.copy_format_action.triggered.connect(self.copy_selected_format)
        self.save_structure_action = QAction("구조 저장", self)
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
        self.addAction(self.copy_format_action)
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
        ribbon = QWidget()
        ribbon_layout = QVBoxLayout(ribbon)
        ribbon_layout.setContentsMargins(4, 4, 4, 2)
        ribbon_layout.setSpacing(2)

        def page() -> QWidget:
            widget = QWidget()
            layout = QHBoxLayout(widget)
            layout.setContentsMargins(6, 6, 6, 6)
            layout.setSpacing(8)
            layout.addStretch(1)
            return widget

        def group(parent_page: QWidget, title: str) -> QHBoxLayout:
            box = QGroupBox(title)
            layout = QHBoxLayout(box)
            layout.setContentsMargins(8, 12, 8, 8)
            layout.setSpacing(5)
            parent_page.layout().insertWidget(parent_page.layout().count() - 1, box)
            return layout

        quick_row = QHBoxLayout()
        quick_row.setContentsMargins(6, 0, 6, 0)
        quick_row.setSpacing(5)
        quick_row.addWidget(QLabel("도구"))
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
            quick_row.addWidget(button)
        self.tool_buttons["select"].setChecked(True)
        quick_recognize_button = QPushButton("인식")
        quick_recognize_button.clicked.connect(self.recognize_edges)
        quick_row.addWidget(quick_recognize_button)
        quick_row.addStretch(1)

        self.ribbon_tabs = QTabWidget()
        self.ribbon_tabs.setDocumentMode(True)
        self.ribbon_tabs.setStyleSheet(
            "QTabWidget::pane { border: 1px solid #c7c7c7; } "
            "QTabBar::tab { padding: 5px 14px; } "
            "QGroupBox { font-weight: 600; margin-top: 8px; border: 1px solid #d0d0d0; border-radius: 4px; } "
            "QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 3px; }"
        )
        ribbon_layout.addWidget(self.ribbon_tabs)
        ribbon_layout.addLayout(quick_row)
        self.setMenuWidget(ribbon)

        file_page = page()
        file_group = group(file_page, "불러오기 / 저장")
        for action in [self.open_action, self.open_folder_action, self.open_project_action, self.save_project_action]:
            file_group.addWidget(self._button_for_action(action))
        export_group = group(file_page, "내보내기")
        for action in [self.export_png_action, self.export_csv_action, self.export_data_action]:
            export_group.addWidget(self._button_for_action(action))
        self.ribbon_tabs.addTab(file_page, "파일")

        edge_page = page()
        reference_group = group(edge_page, "기준선")

        self.axis_combo = QComboBox()
        self.axis_combo.addItem("수평기준선", "horizontal")
        self.axis_combo.addItem("수직기준선", "vertical")
        self.axis_combo.currentIndexChanged.connect(self._axis_changed)
        reference_group.addWidget(self.axis_combo)

        align_button = QPushButton("이미지 맞춤")
        align_button.clicked.connect(self.align_to_reference)
        reference_group.addWidget(align_button)

        detect_group = group(edge_page, "경계 인식")

        self.edge_mode_combo = QComboBox()
        self.edge_mode_combo.addItem("직선", "line")
        self.edge_mode_combo.addItem("이어진 직선", "polyline")
        self.edge_mode_combo.currentIndexChanged.connect(self._edge_mode_changed)
        detect_group.addWidget(QLabel("경계 형태"))
        detect_group.addWidget(self.edge_mode_combo)
        self.canvas.set_edge_draw_mode(self.edge_mode_combo.currentData())

        self.search_radius_spin = QSpinBox()
        self.search_radius_spin.setRange(2, 300)
        self.search_radius_spin.setValue(35)
        self.search_radius_spin.setSuffix(" px")
        self.search_radius_spin.valueChanged.connect(self._edge_detection_settings_changed)
        detect_group.addWidget(QLabel("경계인식 범위"))
        detect_group.addWidget(self.search_radius_spin)

        self.curve_sensitivity_spin = QSpinBox()
        self.curve_sensitivity_spin.setRange(2, 80)
        self.curve_sensitivity_spin.setValue(9)
        self.curve_sensitivity_spin.setSuffix(" px")
        self.curve_sensitivity_spin.valueChanged.connect(self._edge_detection_settings_changed)
        detect_group.addWidget(QLabel("세그먼트 크기"))
        detect_group.addWidget(self.curve_sensitivity_spin)

        self.show_search_range_checkbox = QCheckBox("범위 표시")
        self.show_search_range_checkbox.setChecked(True)
        self.show_search_range_checkbox.toggled.connect(self._edge_detection_settings_changed)
        detect_group.addWidget(self.show_search_range_checkbox)

        settings_button = QPushButton("인식 설정")
        settings_button.clicked.connect(self.open_edge_detection_settings)
        detect_group.addWidget(settings_button)
        self.ribbon_tabs.addTab(edge_page, "경계")

        guide_page = page()
        guide_group = group(guide_page, "가이드 생성")

        self.guide_orientation_combo = QComboBox()
        self.guide_orientation_combo.addItem("수평선", "horizontal")
        self.guide_orientation_combo.addItem("수직선", "vertical")
        self.guide_spacing_spin = QSpinBox()
        self.guide_spacing_spin.setRange(1, 100000)
        self.guide_spacing_spin.setValue(50)
        self.guide_spacing_unit = QComboBox()
        self.guide_spacing_unit.addItem("px", "px")
        self.guide_spacing_unit.addItem("nm", "nm")
        self.guide_direction_combo = QComboBox()
        self.guide_direction_combo.addItem("아래/오른쪽", "positive")
        self.guide_direction_combo.addItem("위/왼쪽", "negative")
        self.guide_direction_combo.addItem("양쪽", "both")
        self.guide_count_spin = QSpinBox()
        self.guide_count_spin.setRange(1, 500)
        self.guide_count_spin.setValue(3)
        self.guide_offset_spin = QSpinBox()
        self.guide_offset_spin.setRange(0, 100000)
        self.guide_offset_spin.setValue(0)
        self.guide_offset_spin.setSuffix(" px")
        guide_group.addWidget(self.guide_orientation_combo)
        guide_group.addWidget(self.guide_spacing_spin)
        guide_group.addWidget(self.guide_spacing_unit)
        guide_group.addWidget(QLabel("방향"))
        guide_group.addWidget(self.guide_direction_combo)
        guide_group.addWidget(QLabel("개수/쪽"))
        guide_group.addWidget(self.guide_count_spin)
        guide_group.addWidget(QLabel("시작"))
        guide_group.addWidget(self.guide_offset_spin)

        add_guides_button = QPushButton("그리기")
        add_guides_button.clicked.connect(self.add_guides)
        clear_guides_button = QPushButton("가이드 지우기")
        clear_guides_button.clicked.connect(self.clear_guides)
        guide_group.addWidget(add_guides_button)
        guide_group.addWidget(clear_guides_button)

        measurement_group = group(guide_page, "측정")
        self.cd_segment_combo = QComboBox()
        self.cd_segment_combo.addItem("CD 전체", "all")
        self.cd_segment_combo.addItem("CD 홀수번째", "odd")
        self.cd_segment_combo.addItem("CD 짝수번째", "even")
        angle_button = QPushButton("각도 계산")
        angle_button.clicked.connect(lambda: self.calculate_angles(reset_hidden=True))
        cd_button = QPushButton("CD 측정")
        cd_button.clicked.connect(self.calculate_cd_lengths)
        measurement_group.addWidget(self.cd_segment_combo)
        measurement_group.addWidget(angle_button)
        measurement_group.addWidget(cd_button)
        self.ribbon_tabs.addTab(guide_page, "가이드/측정")

        display_page = page()
        style_group = group(display_page, "선 서식")
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
        style_group.addWidget(QLabel("선 색"))
        style_group.addWidget(self.stroke_color_combo)
        style_group.addWidget(QLabel("선 두께"))
        style_group.addWidget(self.stroke_width_spin)
        style_group.addWidget(self._button_for_action(self.copy_format_action))

        display_edit_group = group(display_page, "표시 편집")
        angle_settings_button = QPushButton("각도 표시 편집")
        angle_settings_button.clicked.connect(self.edit_angle_display_for_selected_edges)
        cd_settings_button = QPushButton("CD 표시 편집")
        cd_settings_button.clicked.connect(self.edit_cd_display)
        display_edit_group.addWidget(angle_settings_button)
        display_edit_group.addWidget(cd_settings_button)
        self.ribbon_tabs.addTab(display_page, "표시/서식")

        structure_page = page()
        structure_group = group(structure_page, "구조")
        self.structure_combo = QComboBox()
        self.structure_combo.addItem("구조 선택", -1)
        structure_group.addWidget(self.structure_combo)
        structure_group.addWidget(self._button_for_action(self.save_structure_action))
        structure_paste_button = self._button_for_action(self.paste_structure_action)
        structure_group.addWidget(structure_paste_button)
        structure_export_button = QPushButton("구조 공유")
        structure_export_button.clicked.connect(self.export_selected_structure_template)
        structure_import_button = QPushButton("구조 가져오기")
        structure_import_button.clicked.connect(self.import_structure_template)
        structure_delete_button = QPushButton("구조 삭제")
        structure_delete_button.clicked.connect(self.delete_selected_structure_template)
        structure_group.addWidget(structure_export_button)
        structure_group.addWidget(structure_import_button)
        structure_group.addWidget(structure_delete_button)
        self.ribbon_tabs.addTab(structure_page, "구조")

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

    def _visible_angle_measurement_ids(self) -> set[str]:
        measurement_ids: set[str] = set()
        for item in self.canvas.angle_items:
            measurement_id = item.data(ANGLE_MEASUREMENT_KEY)
            if measurement_id:
                measurement_ids.add(str(measurement_id))
        return measurement_ids

    def _build_measurements_dock(self) -> None:
        dock = QDockWidget("측정값", self)
        dock.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)
        container = QWidget()
        layout = QVBoxLayout(container)
        self.calibration_label = QLabel("Calibration: -")
        self.detection_preview_label = QLabel("경계인식 범위: -")
        self.detection_preview_label.setWordWrap(True)
        self.detection_preview_label.setStyleSheet(
            "QLabel {"
            "background: rgba(0, 170, 90, 35);"
            "border: 1px solid rgba(0, 140, 80, 90);"
            "border-radius: 4px;"
            "padding: 6px;"
            "color: #1f2d25;"
            "}"
        )
        self.measurement_table = QTableWidget(0, 5)
        self.measurement_table.setHorizontalHeaderLabels(["ID", "종류", "길이(px)", "길이(nm)", "각도"])
        self.measurement_table.verticalHeader().setVisible(False)
        layout.addWidget(self.calibration_label)
        layout.addWidget(self.detection_preview_label)
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
        self.measurements_dock = dock
        dock.hide()

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
            ("line_angle", "선 각도"),
            ("intersection_angle", "교점 각도"),
            ("angle_arc", "각도 호"),
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
            ("show_line", "선/개체"),
            ("show_line_angle", "선 각도"),
            ("show_intersection_angle", "교점 각도"),
            ("show_angle_arc", "각도 호"),
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
        self._save_current_image_state()
        self.image_states.clear()
        self.image_path = None
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
        self._save_current_image_state()
        self.image_states.clear()
        self.image_path = None
        self.browser_root = root
        self.browser_image_paths = [str(path) for path in image_paths]
        self.current_browser_index = 0
        self._populate_thumbnails()
        self._load_image_path(self.browser_image_paths[0], preserve_calibration=True)
        self._set_status(f"폴더 로드: {root.name}, 이미지 {len(self.browser_image_paths)}개")

    def _load_image_path(self, path: str, preserve_calibration: bool = False) -> None:
        if self.image_path and self.image_path != path:
            self._save_current_image_state()
        image = read_image(path)
        if image is None:
            QMessageBox.warning(self, "열기 실패", "이미지를 읽을 수 없습니다.")
            return
        previous_nm_per_px = self.nm_per_px
        state = self.image_states.get(path)
        self.image_bgr = image
        self.image_path = path
        self.project_path = None
        if state is not None:
            self._restore_image_state(state)
        else:
            self.nm_per_px = previous_nm_per_px if preserve_calibration else None
            self.records.clear()
            self._counter = 1
        self.undo_stack.clear()
        if state is None:
            self.hidden_angle_measurements.clear()
        self._show_image()
        self.canvas.redraw_lines(list(self.records.values()))
        self._refresh_table()
        self._update_search_range_overlay()
        self._apply_visibility()
        self._select_thumbnail(path)
        calibration_text = f", calibration 유지: {self.nm_per_px:.6g} nm/px" if self.nm_per_px else ""
        self._set_status(f"이미지 로드: {Path(path).name} ({image.shape[1]} x {image.shape[0]} px){calibration_text}")

    def _save_current_image_state(self) -> None:
        if not self.image_path:
            return
        self._sync_records_from_canvas()
        self.image_states[self.image_path] = {
            "records": [asdict(record) for record in self.records.values()],
            "counter": self._counter,
            "nm_per_px": self.nm_per_px,
            "hidden_angle_measurements": list(self.hidden_angle_measurements),
        }

    def _restore_image_state(self, state: dict) -> None:
        self.records = {item["id"]: line_record_from_dict(item) for item in state.get("records", [])}
        self._counter = int(state.get("counter", len(self.records) + 1))
        self.nm_per_px = state.get("nm_per_px")
        self.hidden_angle_measurements = set(state.get("hidden_angle_measurements", []))

    def _current_image_state_dict(self) -> dict:
        self._sync_records_from_canvas()
        return {
            "records": [asdict(record) for record in self.records.values()],
            "counter": self._counter,
            "nm_per_px": self.nm_per_px,
            "hidden_angle_measurements": list(self.hidden_angle_measurements),
        }

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
        cd_display = payload.get("cd_display", {})
        self.cd_label_side = str(cd_display.get("label_side", self.cd_label_side))
        if self.cd_label_side not in {"above", "below"}:
            self.cd_label_side = "above"
        self.cd_label_gap = float(cd_display.get("label_gap", self.cd_label_gap))
        self.cd_label_font_size = float(cd_display.get("label_font_size", self.cd_label_font_size))
        guide_generation = payload.get("guide_generation", {})
        guide_direction = guide_generation.get("direction")
        direction_index = self.guide_direction_combo.findData(guide_direction)
        if direction_index >= 0:
            self.guide_direction_combo.setCurrentIndex(direction_index)
        if "count_per_side" in guide_generation:
            self.guide_count_spin.setValue(int(guide_generation["count_per_side"]))
        visibility = payload.get("visibility", {})
        if "angle" in visibility:
            legacy_angle_visible = bool(visibility["angle"])
            visibility.setdefault("line_angle", legacy_angle_visible)
            visibility.setdefault("intersection_angle", legacy_angle_visible)
            visibility.setdefault("angle_arc", legacy_angle_visible)
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
            "cd_display": {
                "label_side": self.cd_label_side,
                "label_gap": self.cd_label_gap,
                "label_font_size": self.cd_label_font_size,
            },
            "guide_generation": {
                "direction": self.guide_direction_combo.currentData(),
                "count_per_side": self.guide_count_spin.value(),
            },
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

    def export_data_xlsx(self) -> None:
        if self.image_bgr is None:
            return
        self._save_current_image_state()
        dialog = DataExportDialog(bool(self.image_states), self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        options = dialog.options()
        if not options.selected_items:
            QMessageBox.information(self, "Data Export", "내보낼 항목을 하나 이상 선택하세요.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Data Export", "", "Excel Workbook (*.xlsx)")
        if not path:
            return
        if not path.lower().endswith(".xlsx"):
            path += ".xlsx"
        sheets = self._build_export_sheets(options)
        if not any(rows for rows in sheets.values()):
            QMessageBox.information(self, "Data Export", "내보낼 측정 데이터가 없습니다.")
            return
        write_xlsx(path, sheets)
        if options.open_after_export:
            self._open_export_file(path)
        self._set_status(f"Data Export 저장: {Path(path).name}")

    @staticmethod
    def _open_export_file(path: str) -> None:
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(path).resolve())))

    def _export_image_states(self, scope: str) -> list[tuple[str, dict]]:
        current_path = self.image_path or "current"
        current_state = self._current_image_state_dict()
        if scope == "current":
            return [(current_path, current_state)]
        states = dict(self.image_states)
        states[current_path] = current_state
        ordered_paths = self.browser_image_paths or list(states.keys())
        result: list[tuple[str, dict]] = []
        seen: set[str] = set()
        for path in ordered_paths:
            state = states.get(path)
            if state is not None:
                result.append((path, state))
                seen.add(path)
        for path, state in states.items():
            if path not in seen:
                result.append((path, state))
        return result

    def _build_export_sheets(self, options: DataExportOptions) -> dict[str, list[dict[str, object]]]:
        sheets = {
            "선각도": [],
            "교점각도": [],
            "CD길이": [],
            "경계길이": [],
        }
        for image_index, (image_path, state) in enumerate(self._export_image_states(options.scope), start=1):
            records = [line_record_from_dict(item) for item in state.get("records", [])]
            nm_per_px = state.get("nm_per_px")
            image_name = Path(image_path).name if image_path else f"image_{image_index}"
            group_info = self._export_group_info(records, options.order_priority)
            rows_by_item = self._export_rows_for_records(image_name, records, nm_per_px, group_info, options.order_priority)
            for item_key, sheet_name in [
                ("line_angle", "선각도"),
                ("intersection_angle", "교점각도"),
                ("cd_length", "CD길이"),
                ("edge_length", "경계길이"),
            ]:
                if item_key in options.selected_items:
                    sheets[sheet_name].extend(rows_by_item[item_key])
        return {name: rows for name, rows in sheets.items() if rows or name in self._selected_sheet_names(options)}

    def _selected_sheet_names(self, options: DataExportOptions) -> set[str]:
        names = {
            "line_angle": "선각도",
            "intersection_angle": "교점각도",
            "cd_length": "CD길이",
            "edge_length": "경계길이",
        }
        return {names[key] for key in options.selected_items}

    def _export_group_info(self, records: list[LineRecord], priority: str) -> dict[Optional[str], dict[str, object]]:
        grouped: dict[str, list[LineRecord]] = {}
        for record in records:
            if record.object_group:
                grouped.setdefault(record.object_group, []).append(record)
        ordered_groups = sorted(
            grouped.items(),
            key=lambda item: position_key(group_bounds_center(item[1]), priority),
        )
        info: dict[Optional[str], dict[str, object]] = {
            None: {"label": "미그룹", "number": 0, "source": ""}
        }
        for index, (group_id, group_records) in enumerate(ordered_groups, start=1):
            info[group_id] = {"label": f"G{index}", "number": index, "source": group_id}
        return info

    def _export_base_row(
        self,
        image_name: str,
        item_type: str,
        records: list[LineRecord],
        group_info: dict[Optional[str], dict[str, object]],
    ) -> dict[str, object]:
        group_id = next((record.object_group for record in records if record.object_group), None)
        info = group_info.get(group_id, group_info[None])
        return {
            "이미지": image_name,
            "그룹": info["label"],
            "그룹번호": info["number"],
            "항목": item_type,
        }

    def _export_rows_for_records(
        self,
        image_name: str,
        records: list[LineRecord],
        nm_per_px: Optional[float],
        group_info: dict[Optional[str], dict[str, object]],
        priority: str,
    ) -> dict[str, list[dict[str, object]]]:
        edges = [record for record in records if record.kind == "edge"]
        guides = [record for record in records if record.kind == "guide"]
        reference = next((record for record in records if record.kind == "reference"), None)
        reference_angle = line_angle_degrees(reference.start, reference.end) if reference else 0.0
        reference_id = reference.id if reference else "horizontal"
        rows = {"line_angle": [], "intersection_angle": [], "cd_length": [], "edge_length": []}

        for edge in edges:
            center = record_center(edge)
            length_px = record_length(edge)
            if not has_segmented_edge_angle(edge):
                angle = acute_angle_difference(record_angle(edge), reference_angle)
                row = self._export_base_row(image_name, "선각도", [edge], group_info)
                row.update(
                    {
                        "측정ID": f"{edge.id}_to_{reference_id}",
                        "경계ID": edge.id,
                        "가이드ID": "",
                        "x_px": center[0],
                        "y_px": center[1],
                        "각도_deg": angle,
                        "길이_px": "",
                        "길이_nm": "",
                    }
                )
                rows["line_angle"].append(row)
            length_row = self._export_base_row(image_name, "경계길이", [edge], group_info)
            length_row.update(
                {
                    "측정ID": f"{edge.id}_length",
                    "경계ID": edge.id,
                    "가이드ID": "",
                    "x_px": center[0],
                    "y_px": center[1],
                    "각도_deg": "",
                    "길이_px": length_px,
                    "길이_nm": length_px * nm_per_px if nm_per_px else "",
                }
            )
            rows["edge_length"].append(length_row)

        for edge in edges:
            for guide in guides:
                crosses = polyline_intersections(edge, (guide.start, guide.end))
                guide_angle = line_angle_degrees(guide.start, guide.end)
                for cross_idx, (cross, edge_angle) in enumerate(crosses, start=1):
                    _arc_start, _arc_end, angle = angle_sector_geometry(edge_angle, guide_angle, edge.angle_sector)
                    suffix = f"_{cross_idx}" if len(crosses) > 1 else ""
                    row = self._export_base_row(image_name, "교점각도", [edge, guide], group_info)
                    row.update(
                        {
                            "측정ID": f"{edge.id}_x_{guide.id}{suffix}",
                            "경계ID": edge.id,
                            "가이드ID": guide.id,
                            "x_px": cross[0],
                            "y_px": cross[1],
                            "각도_deg": angle,
                            "길이_px": "",
                            "길이_nm": "",
                        }
                    )
                    rows["intersection_angle"].append(row)

        mode = str(self.cd_segment_combo.currentData())
        for guide in guides:
            crosses: list[tuple[float, Point, str, LineRecord]] = []
            guide_line = (guide.start, guide.end)
            for edge in edges:
                for cross, _edge_angle in polyline_intersections(edge, guide_line):
                    fraction = line_fraction(cross, guide_line)
                    if -0.0001 <= fraction <= 1.0001:
                        crosses.append((fraction, cross, edge.id, edge))
            crosses.sort(key=lambda item: item[0])
            filtered: list[tuple[float, Point, str, LineRecord]] = []
            for item in crosses:
                if filtered and abs(item[0] - filtered[-1][0]) < 0.0005 and item[2] == filtered[-1][2]:
                    continue
                filtered.append(item)
            for idx in range(len(filtered) - 1):
                if not cd_segment_allowed(idx, mode):
                    continue
                _fa, point_a, edge_a_id, edge_a = filtered[idx]
                _fb, point_b, edge_b_id, edge_b = filtered[idx + 1]
                if edge_a_id == edge_b_id:
                    continue
                length_px = line_length(point_a, point_b)
                midpoint = ((point_a[0] + point_b[0]) / 2.0, (point_a[1] + point_b[1]) / 2.0)
                row = self._export_base_row(image_name, "CD길이", [edge_a, edge_b, guide], group_info)
                row.update(
                    {
                        "측정ID": f"CD_{guide.id}_{idx + 1}_{edge_a_id}_{edge_b_id}",
                        "경계ID": f"{edge_a_id}|{edge_b_id}",
                        "가이드ID": guide.id,
                        "x_px": midpoint[0],
                        "y_px": midpoint[1],
                        "각도_deg": "",
                        "길이_px": length_px,
                        "길이_nm": length_px * nm_per_px if nm_per_px else "",
                    }
                )
                rows["cd_length"].append(row)

        for item_rows in rows.values():
            item_rows.sort(key=lambda row: (int(row["그룹번호"]), *position_key((float(row["x_px"]), float(row["y_px"])), priority), str(row["측정ID"])))
            for order, row in enumerate(item_rows, start=1):
                row["순서"] = order
        return rows

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
            angle_label_font_size=self.default_angle_label_font_size,
            stroke_color=self.default_stroke_color,
            stroke_width=self.default_stroke_width,
        )
        self.records[record.id] = record
        self.last_edge_record_id = record.id
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
        all_edge_records = [record for record in self.records.values() if record.kind == "edge"]
        selected_edge_ids = {record.id for record in all_edge_records if record.id in selected_ids}
        edge_records = [record for record in all_edge_records if record.id in selected_edge_ids]
        if not edge_records:
            edge_records = all_edge_records
            if not edge_records:
                QMessageBox.information(self, "인식", "인식할 경계선이 없습니다.")
                return
            if len(edge_records) >= 10:
                result = QMessageBox.question(
                    self,
                    "전체 경계선 인식",
                    f"선택된 경계선이 없습니다. 전체 경계선 {len(edge_records)}개를 인식할까요?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No,
                )
                if result != QMessageBox.StandardButton.Yes:
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
        target_text = "선택한" if selected_edge_ids else "전체"
        self._set_status(f"{moved}/{len(edge_records)}개 {target_text} 경계선을 선택한 형태로 명도 변화 최대 위치에 맞췄습니다.")

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
        main_guide = self._main_guide_record()
        orientation = main_guide.axis if main_guide is not None else self.guide_orientation_combo.currentData()
        spacing = float(self.guide_spacing_spin.value())
        if self.guide_spacing_unit.currentData() == "nm":
            if not self.nm_per_px:
                QMessageBox.information(self, "가이드", "nm 간격을 쓰려면 먼저 스케일바를 캘리브레이션하세요.")
                return
            spacing = spacing / self.nm_per_px
        if spacing < 1:
            QMessageBox.information(self, "가이드", "간격이 너무 작습니다.")
            return
        height, width = self.image_bgr.shape[:2]
        if main_guide is not None:
            self._add_guides_from_main(main_guide, orientation, spacing, width, height)
            return
        self.clear_guides(redraw=False)
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
        self._refresh_guide_measurements()
        self._update_search_range_overlay()
        self._apply_visibility()
        self._set_status(f"{orientation} 가이드 {created}개를 만들었습니다.")

    def _main_guide_record(self) -> Optional[LineRecord]:
        for record in self.records.values():
            if record.kind == "guide" and record.is_main_guide:
                return record
        return None

    def open_guide_context_menu(self, record_id: str, global_pos: QPoint) -> None:
        record = self.records.get(record_id)
        if record is None or record.kind != "guide":
            return
        menu = QMenu(self)
        if record.is_main_guide:
            action = menu.addAction("메인가이드 해제")
            action.triggered.connect(lambda checked=False: self.set_main_guide(None))
        else:
            action = menu.addAction("메인가이드로 지정")
            action.triggered.connect(lambda checked=False, guide_id=record_id: self.set_main_guide(guide_id))
        menu.exec(global_pos)

    def set_main_guide(self, record_id: Optional[str]) -> None:
        self.save_undo_snapshot()
        for guide_id, record in self.records.items():
            if record.kind == "guide":
                record.is_main_guide = guide_id == record_id
        self.canvas.redraw_lines(list(self.records.values()))
        self._refresh_guide_measurements()
        self._apply_visibility()
        if record_id is None:
            self._set_status("메인가이드를 해제했습니다.")
        else:
            self._set_status("메인가이드로 지정했습니다.")

    def _add_guides_from_main(self, main_guide: LineRecord, orientation: str, spacing: float, width: int, height: int) -> None:
        direction = str(self.guide_direction_combo.currentData())
        count_per_side = int(self.guide_count_spin.value())
        if direction == "both":
            offsets = [idx * spacing for idx in range(-count_per_side, count_per_side + 1)]
        elif direction == "negative":
            offsets = [-idx * spacing for idx in range(0, count_per_side + 1)]
        else:
            offsets = [idx * spacing for idx in range(0, count_per_side + 1)]
        if len(offsets) > 501:
            QMessageBox.information(self, "가이드", "가이드가 너무 많습니다. 개수를 줄여주세요.")
            return
        for record_id in [
            rid
            for rid, record in self.records.items()
            if record.kind == "guide" and rid != main_guide.id
        ]:
            del self.records[record_id]
        base_pos = (main_guide.start[1] + main_guide.end[1]) / 2.0 if orientation == "horizontal" else (main_guide.start[0] + main_guide.end[0]) / 2.0
        main_guide.start, main_guide.end = self._guide_points(orientation, base_pos, width, height)
        main_guide.axis = orientation
        main_guide.is_main_guide = True
        created = 1
        for offset in offsets:
            if abs(offset) < 0.0001:
                continue
            pos = base_pos + offset
            if orientation == "horizontal" and not (0.0 <= pos <= float(height)):
                continue
            if orientation == "vertical" and not (0.0 <= pos <= float(width)):
                continue
            start, end = self._guide_points(orientation, pos, width, height)
            record_id = self._next_id("G")
            self.records[record_id] = LineRecord(
                id=record_id,
                kind="guide",
                start=start,
                end=end,
                label=f"{orientation} guide",
                axis=orientation,
            )
            created += 1
        self.canvas.redraw_lines(list(self.records.values()))
        self._refresh_guide_measurements()
        self._update_search_range_overlay()
        self._apply_visibility()
        self._set_status(f"메인가이드 기준 {orientation} 가이드 {created}개를 만들었습니다.")

    @staticmethod
    def _guide_points(orientation: str, pos: float, width: int, height: int) -> tuple[Point, Point]:
        if orientation == "horizontal":
            return (0.0, pos), (float(width), pos)
        return (pos, 0.0), (pos, float(height))

    def _refresh_guide_measurements(self) -> None:
        edges = [record for record in self.records.values() if record.kind == "edge"]
        guides = [record for record in self.records.values() if record.kind == "guide"]
        if not edges:
            self.canvas.clear_angle_items()
            self.canvas.clear_cd_items()
            return
        self.calculate_angles(reset_hidden=False, update_status=False)
        if guides and self.visibility.get("cd", True):
            self.calculate_cd_lengths(silent=True, update_angles=False, update_status=False)
        elif not guides:
            self.canvas.clear_cd_items()

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

    def calculate_cd_lengths(self, silent: bool = False, update_angles: bool = True, update_status: bool = True) -> None:
        if self.image_bgr is None:
            return
        self._sync_records_from_canvas()
        edges = [record for record in self.records.values() if record.kind == "edge"]
        guides = [record for record in self.records.values() if record.kind == "guide"]
        if len(edges) < 2 or not guides:
            self.canvas.clear_cd_items()
            if not silent:
                QMessageBox.information(self, "CD 길이", "CD 길이를 재려면 경계선 2개 이상과 가이드선이 필요합니다.")
            return
        if update_angles:
            self.calculate_angles(reset_hidden=False, update_status=False)
        else:
            self.canvas.clear_cd_items()
        self._sync_records_from_canvas()
        edges = [record for record in self.records.values() if record.kind == "edge"]
        guides = [record for record in self.records.values() if record.kind == "guide"]

        mode = self.cd_segment_combo.currentData()
        cd_label_gap = self.canvas.screen_to_scene_length(self.cd_label_gap)
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
                label_pos = cd_label_center(point_a, point_b, self.cd_label_side, cd_label_gap)
                if self.nm_per_px:
                    text = f"{length_px * self.nm_per_px:.3g} nm"
                    cd_length_nm = length_px * self.nm_per_px
                else:
                    text = f"{length_px:.2f} px"
                    cd_length_nm = ""
                self.canvas.add_cd_measurement(point_a, point_b, text, label_pos, self.cd_label_font_size)
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
        if update_status:
            self._set_status(f"CD 길이 {created}개를 표시했습니다.")

    def edit_cd_display(self) -> None:
        dialog = CdDisplaySettingsDialog(self.cd_label_side, self.cd_label_gap, self.cd_label_font_size, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        self.cd_label_side = str(dialog.label_side_combo.currentData())
        if self.cd_label_side not in {"above", "below"}:
            self.cd_label_side = "above"
        self.cd_label_gap = float(dialog.label_gap_spin.value())
        self.cd_label_font_size = float(dialog.label_font_size_spin.value())
        if self.canvas.cd_items:
            self.calculate_cd_lengths()
        self._set_status("CD 표시 설정을 바꿨습니다.")

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
            base.angle_label_font_size if base is not None else self.default_angle_label_font_size,
            self,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        self.save_undo_snapshot()
        sector = int(dialog.sector_combo.currentData())
        arc_radius = float(dialog.arc_radius_spin.value())
        label_side = normalize_angle_label_side(str(dialog.label_side_combo.currentData()))
        label_gap = float(dialog.label_gap_spin.value())
        label_font_size = float(dialog.label_font_size_spin.value())
        if editing_all:
            self.default_angle_sector = sector
            self.default_angle_arc_radius = arc_radius
            self.default_angle_label_side = label_side
            self.default_angle_label_gap = label_gap
            self.default_angle_label_font_size = label_font_size
        for edge in edges:
            edge.angle_sector = sector
            edge.angle_arc_radius = arc_radius
            edge.angle_label_side = label_side
            edge.angle_label_gap = label_gap
            edge.angle_label_font_size = label_font_size
        self.calculate_angles(reset_hidden=False)
        target_text = "전체 경계선" if editing_all else "선택한 경계선"
        self._set_status(f"{target_text} {len(edges)}개의 각도 표시 설정을 바꿨습니다.")

    def calculate_angles(
        self,
        reset_hidden: bool = True,
        visible_measurement_ids: Optional[set[str]] = None,
        update_status: bool = True,
    ) -> None:
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
            label_pos = self._label_position(midpoint, edge_angle, reference_angle, self.canvas.screen_to_scene_length(34.0))
            measurement_id = f"{edge.id}_to_{reference_name}"
            should_draw_angle = edge.show_line_angle and measurement_id not in self.hidden_angle_measurements
            if visible_measurement_ids is not None:
                should_draw_angle = should_draw_angle and measurement_id in visible_measurement_ids
            if should_draw_angle:
                self.canvas.add_angle_annotation(
                    f"{angle:.2f}°",
                    label_pos,
                    parent_record_id=edge.id,
                    measurement_id=measurement_id,
                    angle_type="line",
                    label_font_size=edge.angle_label_font_size,
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
                    arc_radius = self.canvas.screen_to_scene_length(edge.angle_arc_radius)
                    label_gap = self.canvas.screen_to_scene_length(edge.angle_label_gap)
                    label_pos = angle_label_position_for_sector(
                        cross,
                        arc_start,
                        angle,
                        arc_radius,
                        edge.angle_label_side,
                        label_gap,
                    )
                    length_px = record_length(edge)
                    suffix = f"_{cross_idx}" if len(crosses) > 1 else ""
                    measurement_id = f"{edge.id}_x_{guide.id}{suffix}"
                    should_draw_label = edge.show_intersection_angle and measurement_id not in self.hidden_angle_measurements
                    should_draw_arc = edge.show_angle_arc and measurement_id not in self.hidden_angle_measurements
                    if visible_measurement_ids is not None:
                        should_draw_label = should_draw_label and measurement_id in visible_measurement_ids
                        should_draw_arc = should_draw_arc and measurement_id in visible_measurement_ids
                    if should_draw_label or should_draw_arc:
                        self.canvas.add_angle_annotation(
                            f"{angle:.2f}°",
                            label_pos,
                            center=cross,
                            angle_a=arc_start,
                            angle_b=arc_end,
                            radius=arc_radius,
                            parent_record_id=edge.id,
                            measurement_id=measurement_id,
                            angle_type="intersection",
                            show_label=should_draw_label,
                            show_arc=should_draw_arc,
                            label_font_size=edge.angle_label_font_size,
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
        if update_status:
            self._set_status(f"각도 {len(self.last_measurements)}개를 계산했습니다. 숫자 라벨은 선택해서 옮길 수 있습니다.")

    def delete_selected(self) -> None:
        selected = set(self.canvas.selected_line_ids())
        selected_angle_items = self.canvas.selected_angle_items()
        selected_edge_length_items = self.canvas.selected_edge_length_items()
        selected_point_handles = self.canvas.selected_point_handles()
        if not selected and not selected_angle_items and not selected_edge_length_items and not selected_point_handles:
            return
        if not selected and not selected_angle_items and not selected_edge_length_items and selected_point_handles:
            visible_angle_ids = self._visible_angle_measurement_ids()
            self.save_undo_snapshot()
            if self.canvas.delete_selected_point_handles():
                self._sync_records_from_canvas()
                self.calculate_angles(reset_hidden=False, visible_measurement_ids=visible_angle_ids)
                self._update_search_range_overlay()
                self._apply_visibility()
            self._set_status("선택한 편집점을 삭제했습니다.")
            return
        visible_angle_ids = self._visible_angle_measurement_ids()
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
            self.calculate_angles(reset_hidden=False, visible_measurement_ids=visible_angle_ids)
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
        self.clipboard_mode = "object"
        self._paste_offset_steps = 0
        QApplication.clipboard().setText(json.dumps([asdict(record) for record in copied], ensure_ascii=False))
        self._set_status(f"상위개체 {len(copied)}개를 복사했습니다. Ctrl+V로 붙여넣을 수 있습니다.")

    def copy_selected_format(self) -> None:
        self._sync_records_from_canvas()
        for record_id in self.canvas.selected_line_ids():
            record = self.records.get(record_id)
            if record is None or record.kind not in {"scale", "reference", "edge", "guide"}:
                continue
            self.format_clipboard = {
                "stroke_color": record.stroke_color,
                "stroke_width": record.stroke_width,
                "angle_sector": record.angle_sector,
                "angle_arc_radius": record.angle_arc_radius,
                "angle_label_side": record.angle_label_side,
                "angle_label_gap": record.angle_label_gap,
                "angle_label_font_size": record.angle_label_font_size,
                "show_line_angle": record.show_line_angle,
                "show_intersection_angle": record.show_intersection_angle,
                "show_angle_arc": record.show_angle_arc,
                "show_edge_length": record.show_edge_length,
                "show_range": record.show_range,
                "show_range_label": record.show_range_label,
            }
            self.clipboard_mode = "format"
            self._set_status("선택 개체의 서식을 복사했습니다. 적용할 개체를 선택하고 Ctrl+V를 누르세요.")
            return
        self._set_status("서식을 복사할 선 개체를 선택하세요.")

    def paste_from_clipboard(self) -> None:
        if self.clipboard_mode == "format":
            self.paste_selected_format()
            return
        self.paste_parent_objects()

    def paste_selected_format(self) -> None:
        if not self.format_clipboard:
            self._set_status("붙여넣을 서식이 없습니다. 먼저 Ctrl+Shift+C로 서식을 복사하세요.")
            return
        target_ids = [
            record_id
            for record_id in self.canvas.selected_line_ids()
            if record_id in self.records and self.records[record_id].kind in {"scale", "reference", "edge", "guide"}
        ]
        if not target_ids:
            self._set_status("서식을 붙여넣을 선 개체를 선택하세요.")
            return
        self.save_undo_snapshot()
        self._sync_records_from_canvas()
        for record_id in target_ids:
            record = self.records[record_id]
            record.stroke_color = self.format_clipboard.get("stroke_color")  # type: ignore[assignment]
            stroke_width = self.format_clipboard.get("stroke_width")
            record.stroke_width = float(stroke_width) if stroke_width is not None else None
            if record.kind == "edge":
                record.angle_sector = int(self.format_clipboard.get("angle_sector", record.angle_sector))
                record.angle_arc_radius = float(self.format_clipboard.get("angle_arc_radius", record.angle_arc_radius))
                record.angle_label_side = normalize_angle_label_side(str(self.format_clipboard.get("angle_label_side", record.angle_label_side)))
                record.angle_label_gap = float(self.format_clipboard.get("angle_label_gap", record.angle_label_gap))
                record.angle_label_font_size = float(self.format_clipboard.get("angle_label_font_size", record.angle_label_font_size))
                record.show_line_angle = bool(self.format_clipboard.get("show_line_angle", record.show_line_angle))
                record.show_intersection_angle = bool(self.format_clipboard.get("show_intersection_angle", record.show_intersection_angle))
                record.show_angle_arc = bool(self.format_clipboard.get("show_angle_arc", record.show_angle_arc))
                record.show_edge_length = bool(self.format_clipboard.get("show_edge_length", record.show_edge_length))
                record.show_range = bool(self.format_clipboard.get("show_range", record.show_range))
                record.show_range_label = bool(self.format_clipboard.get("show_range_label", record.show_range_label))
            item = self.canvas.line_items.get(record_id)
            if item is not None:
                item.setPen(self.canvas._pen_for_record(record))
        self.canvas.update_group_boxes(list(self.records.values()))
        self.calculate_angles(reset_hidden=False)
        self._update_edge_length_overlay()
        self._update_search_range_overlay()
        self._apply_visibility()
        self._set_status(f"선택 개체 {len(target_ids)}개에 서식을 붙여넣었습니다.")

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

    def duplicate_dragged_objects(self, record_ids: object, dx: float, dy: float) -> None:
        if self.image_bgr is None:
            return
        if abs(dx) + abs(dy) <= 0.01:
            return
        source_ids: list[str] = []
        seen: set[str] = set()
        iterable_ids = record_ids if isinstance(record_ids, (list, tuple, set)) else []
        for raw_id in iterable_ids:
            record_id = str(raw_id)
            record = self.records.get(record_id)
            if record_id in seen or record is None or record.kind not in {"edge", "guide"}:
                continue
            seen.add(record_id)
            source_ids.append(record_id)
        if not source_ids:
            return

        self._sync_records_from_canvas()
        group_map: dict[str, str] = {}
        new_ids: list[str] = []
        for source_id in source_ids:
            source = self.records[source_id]
            record = clone_record(source)
            record.id = self._next_id("G" if record.kind == "guide" else "E")
            if record.object_group:
                if record.object_group not in group_map:
                    group_map[record.object_group] = self._next_object_group_id(set(group_map.values()))
                record.object_group = group_map[record.object_group]
            record.start = offset_point(record.start, dx, dy)
            record.end = offset_point(record.end, dx, dy)
            if record.points:
                record.points = [offset_point(point, dx, dy) for point in record.points]
                record.start = record.points[0]
                record.end = record.points[-1]
            if record.recognition_points:
                record.recognition_points = [offset_point(point, dx, dy) for point in record.recognition_points]
            if record.edge_length_label_pos:
                record.edge_length_label_pos = offset_point(record.edge_length_label_pos, dx, dy)
            if record.kind == "guide":
                record.is_main_guide = False
            self.records[record.id] = record
            new_ids.append(record.id)

        self.canvas.redraw_lines(list(self.records.values()))
        self.canvas.scene.clearSelection()
        for record_id in new_ids:
            item = self.canvas.line_items.get(record_id)
            if item is not None:
                item.setSelected(True)
        self.calculate_angles(reset_hidden=False)
        self._refresh_guide_measurements()
        self._update_edge_length_overlay()
        self._update_search_range_overlay()
        self._apply_visibility()
        self._set_status(f"선택 개체 {len(new_ids)}개를 드래그 위치에 복사했습니다.")

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

    def adjust_search_range_by_wheel(self, steps: int) -> None:
        if not hasattr(self, "search_radius_spin") or steps == 0:
            return
        delta = 1 if steps > 0 else -1
        new_value = self.search_radius_spin.value() + delta * abs(steps)
        new_value = max(self.search_radius_spin.minimum(), min(self.search_radius_spin.maximum(), new_value))
        if new_value == self.search_radius_spin.value():
            return

        selected_edges = self._selected_edge_records()
        target_edges = selected_edges or self._last_edge_records()

        self._updating_detection_controls = True
        try:
            self.search_radius_spin.setValue(new_value)
        finally:
            self._updating_detection_controls = False

        for edge in target_edges:
            edge.search_radius_px = new_value
        self._update_search_range_overlay()
        self._show_detection_preview()

        if selected_edges:
            target_text = f"선택 경계선 {len(selected_edges)}개"
        elif target_edges:
            target_text = "마지막 경계선"
        else:
            target_text = "기본값"
        self._set_status(f"{target_text} 경계인식 범위: 양쪽 {new_value}px")

    def _last_edge_records(self) -> list[LineRecord]:
        if self.last_edge_record_id in self.records:
            record = self.records[self.last_edge_record_id]
            if record.kind == "edge":
                return [record]
        for record in reversed(list(self.records.values())):
            if record.kind == "edge":
                self.last_edge_record_id = record.id
                return [record]
        return []

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

    def _angle_item_visible(self, item: QGraphicsPathItem | QGraphicsTextItem) -> bool:
        group_id = item.data(ANGLE_GROUP_KEY)
        parent_id = self.canvas.angle_group_parents.get(str(group_id)) if group_id else None
        parent = self.records.get(parent_id) if parent_id else None
        angle_type = str(item.data(ANGLE_TYPE_KEY) or "line")
        if isinstance(item, QGraphicsPathItem):
            return self.visibility.get("angle_arc", True) and (parent.show_angle_arc if parent else True)
        if angle_type == "intersection":
            return self.visibility.get("intersection_angle", True) and (
                parent.show_intersection_angle if parent else True
            )
        return self.visibility.get("line_angle", True) and (parent.show_line_angle if parent else True)

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

    def _selected_visibility_records(self) -> list[LineRecord]:
        return [
            self.records[record_id]
            for record_id in self.canvas.selected_line_ids()
            if record_id in self.records
        ]

    def _records_for_visibility_key(self, key: str, records: list[LineRecord]) -> list[LineRecord]:
        if key == "show_line":
            return records
        if key in {"show_line_angle", "show_intersection_angle", "show_angle_arc", "show_edge_length", "show_range"}:
            return [record for record in records if record.kind == "edge"]
        return []

    def _update_object_visibility_controls(self) -> None:
        if not hasattr(self, "object_visibility_checkboxes"):
            return
        selected_edges = self._selected_edge_records()
        selected_records = self._selected_visibility_records()
        self._update_detection_controls_from_selection(selected_edges)
        self._updating_object_visibility_controls = True
        try:
            for key, checkbox in self.object_visibility_checkboxes.items():
                target_records = self._records_for_visibility_key(key, selected_records)
                checkbox.setEnabled(bool(target_records))
                if not target_records:
                    checkbox.setCheckState(Qt.CheckState.Unchecked)
                    continue
                values = [bool(getattr(record, key)) for record in target_records]
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
        selected_records = self._selected_visibility_records()
        target_records = self._records_for_visibility_key(key, selected_records)
        if not target_records:
            self._update_object_visibility_controls()
            return
        visible = check_state == Qt.CheckState.Checked
        for record in target_records:
            setattr(record, key, visible)
        if key in {"show_line_angle", "show_intersection_angle", "show_angle_arc"}:
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
        self._set_status(f"선택 개체 {len(target_records)}개의 표시 기준을 바꿨습니다.")

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
            item.setVisible(self._angle_item_visible(item))
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
        self.canvas.clear_detection_preview()
        segment_size_px = self.curve_sensitivity_spin.value()
        self.detection_preview_label.setText(
            f"경계인식 범위: {self._search_range_label()}\n세그먼트 크기: {segment_size_px}px"
        )
        self.measurements_dock.show()
        self.measurements_dock.raise_()
        self.detection_preview_timer.start(1300)

    def clear_detection_preview(self) -> None:
        self.canvas.clear_detection_preview()
        if hasattr(self, "detection_preview_label"):
            self.detection_preview_label.setText("경계인식 범위: -")

    def _handle_scene_changed(self) -> None:
        self._refresh_table()
        moved_measurement_parent = any(
            record_id in self.records and self.records[record_id].kind in {"edge", "guide"}
            for record_id in self.canvas.selected_line_ids()
        ) or any(
            handle.owner.record_id in self.records and self.records[handle.owner.record_id].kind in {"edge", "guide"}
            for handle in self.canvas.selected_point_handles()
        )
        if moved_measurement_parent:
            self._refresh_guide_measurements()
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
