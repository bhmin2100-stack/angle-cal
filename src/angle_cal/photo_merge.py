from __future__ import annotations
from pathlib import Path
import threading
import cv2
import numpy as np
from PySide6.QtCore import QObject,QPointF,QRectF,Qt,QThread,QTimer,Signal,Slot
from PySide6.QtGui import QColor,QImage,QKeyEvent,QPainter,QPainterPath,QPen,QPixmap,QPolygonF,QWheelEvent
from PySide6.QtWidgets import QDialog,QDoubleSpinBox,QFileDialog,QFormLayout,QGraphicsItem,QGraphicsPixmapItem,QGraphicsScene,QGraphicsView,QHBoxLayout,QLabel,QListWidget,QListWidgetItem,QMessageBox,QProgressBar,QPushButton,QSplitter,QTableWidget,QTableWidgetItem,QVBoxLayout,QWidget
from .band_registration import band_profiles
from .stitching import StitchLayoutHint,StitchOptions,StitchResult,StitchingCancelled,StitchingNeedsManual,detect_bottom_overlay_fraction,read_raw_image,save_stitch_result,save_stitch_result_auto,stitch_paths

def preview_pixmap(image):
    shown=image
    if shown.dtype!=np.uint8:
        lo,hi=np.percentile(shown,(1,99)); shown=np.zeros(shown.shape,np.uint8) if hi<=lo else np.clip((shown-lo)*255/(hi-lo),0,255).astype(np.uint8)
    if shown.ndim==2:shown=cv2.cvtColor(shown,cv2.COLOR_GRAY2RGB)
    elif shown.shape[2]==4:shown=cv2.cvtColor(shown,cv2.COLOR_BGRA2RGB)
    else:shown=cv2.cvtColor(shown,cv2.COLOR_BGR2RGB)
    shown=np.ascontiguousarray(shown); return QPixmap.fromImage(QImage(shown.data,shown.shape[1],shown.shape[0],shown.strides[0],QImage.Format.Format_RGB888).copy())

class StitchWorker(QObject):
    progress=Signal(int,str); finished=Signal(object); failed=Signal(str); manual=Signal(str)
    def __init__(self,paths,options,layout_hints=None):super().__init__();self.paths=paths;self.options=options;self.layout_hints=layout_hints;self.cancel_event=threading.Event()
    @Slot()
    def run(self):
        try:self.finished.emit(stitch_paths(self.paths,self.options,layout_hints=self.layout_hints,progress=self.on_progress,cancelled=self.cancel_event.is_set))
        except StitchingNeedsManual as exc:self.manual.emit(str(exc))
        except StitchingCancelled:self.failed.emit("사진 합치기를 취소했습니다.")
        except Exception as exc:self.failed.emit(str(exc))
    def on_progress(self,stage,current,total,message):self.progress.emit(int(100*(current+1)/max(total,1)),message)

class PhotoMergeDialog(QDialog):
    result_saved=Signal(str)
    def __init__(self,initial_paths=None,parent=None):
        super().__init__(parent);self.setWindowTitle("사진 합치기");self.resize(1080,700);self.result=None;self.thread=None;self.worker=None
        root=QVBoxLayout(self);split=QSplitter();left=QWidget();ll=QVBoxLayout(left);ll.addWidget(QLabel("입력 이미지 (2~20장, 순서 무관)"));self.list=QListWidget();ll.addWidget(self.list,1)
        row=QHBoxLayout();add=QPushButton("파일 추가");add.clicked.connect(self.choose);remove=QPushButton("선택 제거");remove.clicked.connect(self.remove);row.addWidget(add);row.addWidget(remove);ll.addLayout(row)
        form=QFormLayout();self.crop=QDoubleSpinBox();self.crop.setRange(0,45);self.crop.setSuffix(" %");form.addRow("하단 장비 정보 제거",self.crop);ll.addLayout(form);detect=QPushButton("장비 정보 띠 자동 감지");detect.clicked.connect(self.detect);ll.addWidget(detect)
        right=QWidget();rl=QVBoxLayout(right);self.preview=QLabel("이미지를 추가하고 자동 합치기를 실행하세요.");self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter);self.preview.setMinimumSize(500,350);self.preview.setStyleSheet("background:#202124;color:#ddd");rl.addWidget(self.preview,1)
        self.table=QTableWidget(0,4);self.table.setHorizontalHeaderLabels(["이미지","처리","인라이어","오차(px)"]);self.table.setMaximumHeight(180);rl.addWidget(self.table);split.addWidget(left);split.addWidget(right);split.setStretchFactor(1,1);root.addWidget(split,1)
        self.status=QLabel("대기 중");self.progress=QProgressBar();root.addWidget(self.status);root.addWidget(self.progress);actions=QHBoxLayout();self.start=QPushButton("자동 합치기");self.start.clicked.connect(self.start_stitch);self.cancel=QPushButton("취소");self.cancel.setEnabled(False);self.cancel.clicked.connect(self.cancel_stitch);self.save=QPushButton("저장 후 AngleCal에서 열기");self.save.setEnabled(False);self.save.clicked.connect(self.save_result);close=QPushButton("닫기");close.clicked.connect(self.reject)
        actions.addWidget(self.start);actions.addWidget(self.cancel);actions.addStretch(1);actions.addWidget(self.save);actions.addWidget(close);root.addLayout(actions);self.add_paths(initial_paths or [])
    def paths(self):return [self.list.item(i).data(Qt.ItemDataRole.UserRole) for i in range(self.list.count())]
    def add_paths(self,paths):
        existing={p.casefold() for p in self.paths()}
        for raw in paths:
            path=str(Path(raw).resolve())
            if Path(path).is_file() and path.casefold() not in existing:
                item=QListWidgetItem(Path(path).name);item.setData(Qt.ItemDataRole.UserRole,path);item.setToolTip(path);self.list.addItem(item);existing.add(path.casefold())
    def choose(self):paths,_=QFileDialog.getOpenFileNames(self,"합칠 이미지 추가","","Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)");self.add_paths(paths)
    def remove(self):
        for item in self.list.selectedItems():self.list.takeItem(self.list.row(item))
    def detect(self):
        try:self.crop.setValue(detect_bottom_overlay_fraction([read_raw_image(p) for p in self.paths()])*100)
        except Exception as exc:QMessageBox.warning(self,"장비 정보 감지",str(exc))
    def start_stitch(self):
        if len(self.paths())<2:QMessageBox.information(self,"사진 합치기","이미지를 2장 이상 추가하세요.");return
        self.result=None;self.save.setEnabled(False);self.start.setEnabled(False);self.cancel.setEnabled(True);self.status.setText("합치기 준비 중…");self.progress.setValue(0);self.thread=QThread(self);self.worker=StitchWorker(self.paths(),StitchOptions(self.crop.value()/100));self.worker.moveToThread(self.thread);self.thread.started.connect(self.worker.run);self.worker.progress.connect(self.on_progress);self.worker.finished.connect(self._on_stitch_finished);self.worker.failed.connect(self._on_stitch_failed);self.worker.manual.connect(self._on_manual_required)
        for signal in (self.worker.finished,self.worker.failed,self.worker.manual):signal.connect(self.thread.quit)
        self.thread.finished.connect(self.worker.deleteLater);self.thread.finished.connect(self.thread_finished);self.thread.start()
    def cancel_stitch(self):
        if self.worker:self.worker.cancel_event.set();self.status.setText("취소 중…")
    def on_progress(self,value,text):self.progress.setValue(value);self.status.setText(text)
    def _on_stitch_finished(self,result:StitchResult):
        self.result=result;self.progress.setValue(100);self.status.setText(f"완료: {result.output_size[0]} × {result.output_size[1]} px · 스케일 재보정 필요");self.save.setEnabled(True);self.preview.setPixmap(preview_pixmap(result.image).scaled(self.preview.size(),Qt.AspectRatioMode.KeepAspectRatio,Qt.TransformationMode.SmoothTransformation));self.table.setRowCount(len(result.placements))
        for r,p in enumerate(result.placements):
            for c,v in enumerate((Path(p.path).name,p.mode,p.inlier_count,f"{p.reprojection_error:.3f}")):self.table.setItem(r,c,QTableWidgetItem(str(v)))
    def _on_stitch_failed(self,message):self.status.setText(message);QMessageBox.warning(self,"사진 합치기",message)
    def _on_manual_required(self,message):self.status.setText("수동 보정 필요");QMessageBox.information(self,"수동 정렬 필요",message+"\n수동 기준점 편집기는 다음 업데이트에서 연결됩니다.")
    def thread_finished(self):self.thread.deleteLater();self.thread=None;self.worker=None;self.start.setEnabled(True);self.cancel.setEnabled(False)
    def save_result(self):
        if self.result is None:return
        path,_=QFileDialog.getSaveFileName(self,"합친 이미지 저장","merged.tif","TIFF (*.tif *.tiff);;PNG (*.png)")
        if path:
            try:output,_,_=save_stitch_result(path,self.result)
            except Exception as exc:QMessageBox.warning(self,"사진 합치기",str(exc));return
            self.result_saved.emit(str(output));self.accept()
    def reject(self):self.cancel_stitch();super().reject()


class CropOverlayItem(QGraphicsItem):
    HANDLE_SIZE = 9.0
    MIN_SIZE = 24.0

    def __init__(self, parent: "MergeBoardItem") -> None:
        super().__init__(parent)
        self.owner = parent
        self.active_handle: str | None = None
        self.drag_start = QPointF()
        self.start_rect = QRectF()
        self.setZValue(1000)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresParentOpacity, True)
        self.setAcceptHoverEvents(True)
        self.setAcceptedMouseButtons(Qt.MouseButton.LeftButton)
        self.hide()

    def boundingRect(self) -> QRectF:  # noqa: N802
        return self.owner.boundingRect()

    def crop_rect(self) -> QRectF:
        full = self.boundingRect()
        region = self.owner.match_rect
        if region is None:
            return QRectF(full)
        x, y, width, height = region
        return QRectF(full.left() + x * full.width(), full.top() + y * full.height(), width * full.width(), height * full.height())

    def _handle_rects(self, crop: QRectF) -> dict[str, QRectF]:
        size = self.HANDLE_SIZE
        half = size / 2.0
        points = {
            "top_left": crop.topLeft(),
            "top": QPointF(crop.center().x(), crop.top()),
            "top_right": crop.topRight(),
            "right": QPointF(crop.right(), crop.center().y()),
            "bottom_right": crop.bottomRight(),
            "bottom": QPointF(crop.center().x(), crop.bottom()),
            "bottom_left": crop.bottomLeft(),
            "left": QPointF(crop.left(), crop.center().y()),
        }
        return {name: QRectF(point.x() - half, point.y() - half, size, size) for name, point in points.items()}

    def _handle_at(self, position: QPointF) -> str | None:
        for name, rect in self._handle_rects(self.crop_rect()).items():
            if rect.adjusted(-4, -4, 4, 4).contains(position):
                return name
        return None

    def paint(self, painter: QPainter, option, widget=None) -> None:
        full = self.boundingRect()
        crop = self.crop_rect()
        outside = QPainterPath()
        outside.addRect(full)
        inside = QPainterPath()
        inside.addRect(crop)
        painter.fillPath(outside.subtracted(inside), QColor(0, 0, 0, 150))
        painter.setPen(QPen(QColor(255, 255, 255), 2.0, Qt.PenStyle.DashLine))
        painter.drawRect(crop)
        painter.setPen(QPen(QColor(255, 255, 255, 130), 1.0, Qt.PenStyle.DotLine))
        painter.drawLine(QPointF(crop.left() + crop.width() / 3, crop.top()), QPointF(crop.left() + crop.width() / 3, crop.bottom()))
        painter.drawLine(QPointF(crop.left() + crop.width() * 2 / 3, crop.top()), QPointF(crop.left() + crop.width() * 2 / 3, crop.bottom()))
        painter.drawLine(QPointF(crop.left(), crop.top() + crop.height() / 3), QPointF(crop.right(), crop.top() + crop.height() / 3))
        painter.drawLine(QPointF(crop.left(), crop.top() + crop.height() * 2 / 3), QPointF(crop.right(), crop.top() + crop.height() * 2 / 3))
        painter.setPen(QPen(QColor(20, 20, 20), 1.0))
        painter.setBrush(QColor(255, 255, 255))
        for rect in self._handle_rects(crop).values():
            painter.drawRect(rect)

    def hoverMoveEvent(self, event) -> None:  # noqa: N802
        handle = self._handle_at(event.pos())
        cursors = {
            "top_left": Qt.CursorShape.SizeFDiagCursor,
            "bottom_right": Qt.CursorShape.SizeFDiagCursor,
            "top_right": Qt.CursorShape.SizeBDiagCursor,
            "bottom_left": Qt.CursorShape.SizeBDiagCursor,
            "top": Qt.CursorShape.SizeVerCursor,
            "bottom": Qt.CursorShape.SizeVerCursor,
            "left": Qt.CursorShape.SizeHorCursor,
            "right": Qt.CursorShape.SizeHorCursor,
        }
        self.setCursor(cursors.get(handle, Qt.CursorShape.ArrowCursor))
        super().hoverMoveEvent(event)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        handle = self._handle_at(event.pos())
        if handle is None:
            event.ignore()
            return
        self.owner.setSelected(True)
        self.active_handle = handle
        self.drag_start = event.pos()
        self.start_rect = self.crop_rect()
        event.accept()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self.active_handle is None:
            event.ignore()
            return
        delta = event.pos() - self.drag_start
        crop = QRectF(self.start_rect)
        if "left" in self.active_handle:
            crop.setLeft(min(crop.right() - self.MIN_SIZE, crop.left() + delta.x()))
        if "right" in self.active_handle:
            crop.setRight(max(crop.left() + self.MIN_SIZE, crop.right() + delta.x()))
        if "top" in self.active_handle:
            crop.setTop(min(crop.bottom() - self.MIN_SIZE, crop.top() + delta.y()))
        if "bottom" in self.active_handle:
            crop.setBottom(max(crop.top() + self.MIN_SIZE, crop.bottom() + delta.y()))
        full = self.boundingRect()
        crop.setLeft(max(full.left(), crop.left()))
        crop.setTop(max(full.top(), crop.top()))
        crop.setRight(min(full.right(), crop.right()))
        crop.setBottom(min(full.bottom(), crop.bottom()))
        self.owner.set_match_rect_from_display(crop)
        event.accept()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        self.active_handle = None
        event.accept()


class MergeBoardItem(QGraphicsPixmapItem):
    def __init__(self, path: str, pixmap: QPixmap, source_size: tuple[int, int], gray_preview: np.ndarray, view: "MergeBoardView") -> None:
        super().__init__(pixmap)
        self.path = path
        self.source_size = source_size
        self.original_pixmap = QPixmap(pixmap)
        self.gray_preview = gray_preview
        self.view = view
        self.match_rect: tuple[float, float, float, float] | None = None
        self.setFlags(
            QGraphicsPixmapItem.GraphicsItemFlag.ItemIsMovable
            | QGraphicsPixmapItem.GraphicsItemFlag.ItemIsSelectable
            | QGraphicsPixmapItem.GraphicsItemFlag.ItemSendsGeometryChanges
        )
        self.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        self.setToolTip(Path(path).name)
        self.crop_overlay = CropOverlayItem(self)

    def set_match_rect_from_display(self, rect: QRectF) -> None:
        full = self.boundingRect()
        normalized = (
            (rect.left() - full.left()) / max(1.0, full.width()),
            (rect.top() - full.top()) / max(1.0, full.height()),
            rect.width() / max(1.0, full.width()),
            rect.height() / max(1.0, full.height()),
        )
        self.set_match_rect(normalized)

    def set_match_rect(self, region: tuple[float, float, float, float] | None, *, notify: bool = True) -> None:
        self.match_rect = region
        self.refresh_display(self.view.crop_mode)
        self.crop_overlay.update()
        if notify:
            self.view.crop_region_changed(self)

    def refresh_display(self, crop_active: bool) -> None:
        if crop_active or self.match_rect is None:
            self.setPixmap(self.original_pixmap)
            return
        x, y, width, height = self.match_rect
        full = QRectF(self.original_pixmap.rect())
        visible = QRectF(
            full.left() + x * full.width(),
            full.top() + y * full.height(),
            width * full.width(),
            height * full.height(),
        )
        cropped = QPixmap(self.original_pixmap.size())
        cropped.fill(Qt.GlobalColor.transparent)
        painter = QPainter(cropped)
        painter.drawPixmap(visible, self.original_pixmap, visible)
        painter.end()
        self.setPixmap(cropped)

    def itemChange(self, change, value):  # noqa: N802
        result = super().itemChange(change, value)
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged and self.scene() is not None:
            self.view.request_profile_update()
        return result

    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self.view.zoom_by(event.delta())
            event.accept()
            return
        delta = 0.08 if event.delta() > 0 else -0.08
        self.setOpacity(max(0.12, min(1.0, self.opacity() + delta)))
        self.view.opacity_changed.emit(round(self.opacity() * 100))
        event.accept()


class BandProfileGraph(QWidget):
    """Lightweight overlap-profile plot without an extra plotting dependency."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.selected_name = ""
        self.profiles: list[dict[str, object]] = []
        self.setMinimumWidth(300)
        self.setMinimumHeight(320)

    def set_profiles(self, selected_name: str, profiles: list[dict[str, object]]) -> None:
        self.selected_name = selected_name
        self.profiles = profiles[:4]
        self.update()

    @staticmethod
    def _points(values: np.ndarray, rect: QRectF, low: float, high: float) -> QPolygonF:
        values = np.asarray(values, np.float64).ravel()
        if not len(values):
            return QPolygonF()
        span = max(1e-9, high - low)
        denominator = max(1, len(values) - 1)
        return QPolygonF([
            QPointF(
                rect.left() + rect.width() * index / denominator,
                rect.bottom() - rect.height() * float(np.clip((value - low) / span, 0.0, 1.0)),
            )
            for index, value in enumerate(values)
        ])

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#f7f9fb"))
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setPen(QColor("#24292f"))
        painter.drawText(12, 22, f"선택: {self.selected_name or '없음'}")
        if not self.profiles:
            painter.setPen(QColor("#6e7781"))
            painter.drawText(QRectF(12, 42, self.width() - 24, 90), Qt.TextFlag.TextWordWrap,
                             "이미지를 선택하고 다른 이미지와 겹치면 밝기 밴드 비교가 표시됩니다.")
            return
        top = 38.0
        block_height = max(74.0, min(128.0, (self.height() - top - 12.0) / len(self.profiles)))
        for index, entry in enumerate(self.profiles):
            block_top = top + index * block_height
            title = str(entry["title"])
            score = float(entry["score"])
            painter.setPen(QColor("#24292f"))
            painter.drawText(QRectF(12, block_top, self.width() - 24, 20),
                             Qt.AlignmentFlag.AlignLeft, f"{title}  ·  일치 {max(0, score) * 100:.0f}%")
            plot = QRectF(14, block_top + 24, self.width() - 28, block_height - 32)
            painter.fillRect(plot, QColor("#ffffff"))
            painter.setPen(QPen(QColor("#d0d7de"), 1.0))
            painter.drawRect(plot)
            first = np.asarray(entry["first"], np.float64)
            second = np.asarray(entry["second"], np.float64)
            if not len(first) or not len(second):
                continue
            low = float(min(first.min(), second.min()))
            high = float(max(first.max(), second.max()))
            painter.setPen(QPen(QColor("#0969da"), 1.6))
            painter.drawPolyline(self._points(first, plot, low, high))
            painter.setPen(QPen(QColor("#cf222e"), 1.6))
            painter.drawPolyline(self._points(second, plot, low, high))



class MergeBoardView(QGraphicsView):
    paths_changed = Signal(int)
    opacity_changed = Signal(int)
    crop_changed = Signal(str)
    band_profiles_changed = Signal(str, object)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.setAcceptDrops(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setRenderHints(self.renderHints())
        self.setStyleSheet("QGraphicsView { background:#24282d; border:1px solid #8b949e; }")
        self.setSceneRect(-2500, -1800, 5000, 3600)
        self.crop_mode = False
        self.profile_timer = QTimer(self)
        self.profile_timer.setSingleShot(True)
        self.profile_timer.setInterval(120)
        self.profile_timer.timeout.connect(self._emit_band_profiles)
        self.scene().selectionChanged.connect(self._selection_changed)

    def items_in_board(self) -> list[MergeBoardItem]:
        return [item for item in self.scene().items() if isinstance(item, MergeBoardItem)]

    def paths(self) -> list[str]:
        return [item.path for item in sorted(self.items_in_board(), key=lambda item: (item.pos().y(), item.pos().x()))]

    def layout_hints(self) -> list[StitchLayoutHint]:
        hints = []
        for item in sorted(self.items_in_board(), key=lambda candidate: (candidate.pos().y(), candidate.pos().x())):
            source_width, source_height = item.source_size
            scale_x = item.pixmap().width() / max(1, source_width)
            scale_y = item.pixmap().height() / max(1, source_height)
            position = item.scenePos()
            source_to_board = np.array(
                [[scale_x, 0.0, position.x()], [0.0, scale_y, position.y()], [0.0, 0.0, 1.0]],
                dtype=np.float64,
            )
            hints.append(StitchLayoutHint(item.path, source_to_board, item.match_rect))
        return hints

    def set_crop_mode(self, enabled: bool) -> None:
        self.crop_mode = enabled
        self._sync_crop_overlays()

    def _sync_crop_overlays(self) -> None:
        for item in self.items_in_board():
            item.refresh_display(self.crop_mode)
            item.crop_overlay.setVisible(self.crop_mode and item.isSelected())

    def _selection_changed(self) -> None:
        self._sync_crop_overlays()
        self.request_profile_update()

    def crop_region_changed(self, source: MergeBoardItem) -> None:
        region = source.match_rect
        if region is None:
            self.crop_changed.emit("정합 영역이 원본 전체로 초기화되었습니다.")
            return
        x, y, width, height = region
        self.crop_changed.emit(
            f"정합 사용 영역: 왼쪽 {x:.0%}, 위 {y:.0%}, 너비 {width:.0%}, 높이 {height:.0%}"
        )

    def reset_crop_regions(self) -> None:
        targets = [item for item in self.items_in_board() if item.isSelected()]
        for item in targets:
            item.set_match_rect(None, notify=False)
        self._sync_crop_overlays()
        if targets:
            self.crop_changed.emit("선택 이미지의 정합 영역을 초기화했습니다.")
        else:
            self.crop_changed.emit("자르기 영역을 초기화할 이미지를 먼저 선택하세요.")

    def add_paths(self, paths: list[str], drop_position: QPointF | None = None) -> None:
        existing = {item.path.casefold() for item in self.items_in_board()}
        added = 0
        current_items = self.items_in_board()
        if drop_position is not None:
            origin = drop_position
        elif current_items:
            rightmost = max(item.sceneBoundingRect().right() for item in current_items)
            origin = QPointF(rightmost + 24, min(item.sceneBoundingRect().top() for item in current_items))
        else:
            origin = self.mapToScene(self.viewport().rect().center())
        cursor_x = origin.x()
        for raw_path in paths:
            path = str(Path(raw_path).resolve())
            if path.casefold() in existing or not Path(path).is_file():
                continue
            try:
                image = read_raw_image(path)
            except Exception:
                continue
            full = preview_pixmap(image)
            board_pixmap = full.scaled(360, 280, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            gray = cv2.cvtColor(image[..., :3], cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
            if gray.dtype != np.uint8:
                lo, hi = np.percentile(gray, (1, 99))
                gray = np.zeros(gray.shape, np.uint8) if hi <= lo else np.clip((gray-lo)*255/(hi-lo), 0, 255).astype(np.uint8)
            gray_preview = cv2.resize(gray, (board_pixmap.width(), board_pixmap.height()), interpolation=cv2.INTER_AREA)
            item = MergeBoardItem(path, board_pixmap, (image.shape[1], image.shape[0]), gray_preview, self)
            item.setPos(QPointF(cursor_x, origin.y()))
            item.setZValue(len(self.items_in_board()) + 1)
            self.scene().addItem(item)
            cursor_x += max(48.0, board_pixmap.width() * 0.62)
            existing.add(path.casefold())
            added += 1
        if added:
            self.paths_changed.emit(len(self.items_in_board()))
            self._sync_crop_overlays()
            self.request_profile_update()

    def clear_board(self) -> None:
        self.scene().clear()
        self.paths_changed.emit(0)

    def delete_selected(self) -> None:
        removed = False
        for item in list(self.scene().selectedItems()):
            if isinstance(item, MergeBoardItem):
                self.scene().removeItem(item)
                removed = True
        if removed:
            self.paths_changed.emit(len(self.items_in_board()))
            self.request_profile_update()

    def mousePressEvent(self, event) -> None:  # noqa: N802
        self.setFocus(Qt.FocusReason.MouseFocusReason)
        super().mousePressEvent(event)

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802
        if event.key() == Qt.Key.Key_Delete:
            self.delete_selected()
            event.accept()
            return
        super().keyPressEvent(event)

    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self.zoom_by(event.angleDelta().y())
            event.accept()
            return
        super().wheelEvent(event)

    def zoom_by(self, delta: int) -> None:
        current = float(self.transform().m11())
        factor = 1.15 if delta > 0 else 1.0 / 1.15
        target = current * factor
        if 0.20 <= target <= 5.0:
            self.scale(factor, factor)

    def request_profile_update(self) -> None:
        self.profile_timer.start()

    @staticmethod
    def _region_mask(item: MergeBoardItem, rect: QRectF, width: int, height: int) -> np.ndarray:
        mask = np.ones((height, width), dtype=bool)
        if item.match_rect is None:
            return mask
        x, y, region_width, region_height = item.match_rect
        bounds = item.sceneBoundingRect()
        allowed = QRectF(
            bounds.left() + x * bounds.width(),
            bounds.top() + y * bounds.height(),
            region_width * bounds.width(),
            region_height * bounds.height(),
        )
        yy, xx = np.mgrid[:height, :width]
        scene_x = rect.left() + xx + 0.5
        scene_y = rect.top() + yy + 0.5
        return (scene_x >= allowed.left()) & (scene_x <= allowed.right()) & (scene_y >= allowed.top()) & (scene_y <= allowed.bottom())

    @staticmethod
    def _overlap_array(item: MergeBoardItem, rect: QRectF) -> np.ndarray:
        bounds = item.sceneBoundingRect()
        left = max(0, int(round(rect.left() - bounds.left())))
        top = max(0, int(round(rect.top() - bounds.top())))
        right = min(item.gray_preview.shape[1], left + int(round(rect.width())))
        bottom = min(item.gray_preview.shape[0], top + int(round(rect.height())))
        return item.gray_preview[top:bottom, left:right]

    def _emit_band_profiles(self) -> None:
        selected = [item for item in self.items_in_board() if item.isSelected()]
        if not selected:
            self.band_profiles_changed.emit("", [])
            return
        primary = selected[-1]
        entries: list[dict[str, object]] = []
        for other in self.items_in_board():
            if other is primary:
                continue
            overlap = primary.sceneBoundingRect().intersected(other.sceneBoundingRect())
            if overlap.width() < 14 or overlap.height() < 14:
                continue
            first = self._overlap_array(primary, overlap)
            second = self._overlap_array(other, overlap)
            height = min(first.shape[0], second.shape[0])
            width = min(first.shape[1], second.shape[1])
            if height < 14 or width < 14:
                continue
            first, second = first[:height, :width], second[:height, :width]
            valid = self._region_mask(primary, overlap, width, height)
            valid &= self._region_mask(other, overlap, width, height)
            profiles = band_profiles(first, second, valid)
            for name, profile_a, profile_b, score in profiles[:2]:
                entries.append({
                    "title": f"{Path(other.path).name} · {name}",
                    "first": profile_a,
                    "second": profile_b,
                    "score": score,
                })
        entries.sort(key=lambda entry: float(entry["score"]), reverse=True)
        self.band_profiles_changed.emit(Path(primary.path).name, entries[:4])

    def dragEnterEvent(self, event) -> None:  # noqa: N802
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dragMoveEvent(self, event) -> None:  # noqa: N802
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragMoveEvent(event)

    def dropEvent(self, event) -> None:  # noqa: N802
        paths = [url.toLocalFile() for url in event.mimeData().urls() if url.isLocalFile()]
        if paths:
            self.add_paths(paths, self.mapToScene(event.position().toPoint()))
            event.acceptProposedAction()
            return
        super().dropEvent(event)


class PhotoMergeBoard(QWidget):
    result_ready = Signal(object)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.thread: QThread | None = None
        self.worker: StitchWorker | None = None
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        toolbar = QHBoxLayout()
        title = QLabel("사진 합치기 보드")
        title.setStyleSheet("font-size:16px;font-weight:700")
        toolbar.addWidget(title)
        self.count_label = QLabel("보드 이미지 0장")
        toolbar.addWidget(self.count_label)
        toolbar.addStretch(1)
        clear_button = QPushButton("보드 비우기")
        clear_button.clicked.connect(self.clear_board)
        toolbar.addWidget(clear_button)
        self.align_button = QPushButton("맞추기")
        self.align_button.setEnabled(False)
        self.align_button.setStyleSheet("font-weight:700;padding:7px 18px")
        self.align_button.clicked.connect(self.start_alignment)
        toolbar.addWidget(self.align_button)
        root.addLayout(toolbar)
        crop_toolbar = QHBoxLayout()
        self.crop_button = QPushButton("정합 영역 자르기")
        self.crop_button.setCheckable(True)
        self.crop_button.setEnabled(False)
        self.crop_button.setToolTip("PowerPoint 자르기처럼 손잡이를 끌어 정합과 완성 결과에 사용할 영역을 지정합니다.")
        self.crop_button.toggled.connect(self._toggle_crop_mode)
        crop_toolbar.addWidget(self.crop_button)
        self.crop_reset_button = QPushButton("자르기 초기화")
        self.crop_reset_button.setEnabled(False)
        self.crop_reset_button.clicked.connect(self._reset_crop_regions)
        crop_toolbar.addWidget(self.crop_reset_button)
        crop_help = QLabel("어두운 부분은 정합·완성 결과에서 제외 · 사용 영역의 원본 픽셀은 유지")
        crop_help.setStyleSheet("color:#586069")
        crop_toolbar.addWidget(crop_help)
        crop_toolbar.addStretch(1)
        root.addLayout(crop_toolbar)
        hint = QLabel("왼쪽 썸네일을 끌어 놓으세요  ·  드래그: 위치 이동  ·  휠: 투명도  ·  Ctrl+휠: 확대/축소  ·  Delete: 제거")
        hint.setStyleSheet("color:#586069;padding:2px")
        root.addWidget(hint)
        workspace = QSplitter(Qt.Orientation.Horizontal)
        self.view = MergeBoardView()
        workspace.addWidget(self.view)
        side = QWidget()
        side_layout = QVBoxLayout(side)
        side_layout.setContentsMargins(8, 6, 6, 6)
        side_title = QLabel("정합 밴드 비교")
        side_title.setStyleSheet("font-size:15px;font-weight:700")
        side_layout.addWidget(side_title)
        side_help = QLabel("선택 이미지와 겹친 이미지의 밝기·경계 밴드를 비교합니다. 파랑은 선택 이미지, 빨강은 맞닿은 이미지입니다.")
        side_help.setWordWrap(True)
        side_help.setStyleSheet("color:#586069")
        side_layout.addWidget(side_help)
        self.band_graph = BandProfileGraph()
        side_layout.addWidget(self.band_graph, 1)
        workspace.addWidget(side)
        workspace.setStretchFactor(0, 1)
        workspace.setStretchFactor(1, 0)
        workspace.setSizes([720, 320])
        root.addWidget(workspace, 1)
        bottom = QHBoxLayout()
        self.status = QLabel("이미지를 보드에 배치한 뒤 맞추기를 누르세요.")
        self.progress = QProgressBar()
        self.progress.setMaximumWidth(260)
        self.progress.setValue(0)
        bottom.addWidget(self.status, 1)
        bottom.addWidget(self.progress)
        root.addLayout(bottom)
        self.view.paths_changed.connect(self._update_count)
        self.view.opacity_changed.connect(lambda value: self.status.setText(f"선택 이미지 투명도 {value}%"))
        self.view.crop_changed.connect(self.status.setText)
        self.view.band_profiles_changed.connect(self.band_graph.set_profiles)

    def add_paths(self, paths: list[str]) -> None:
        self.view.add_paths(paths)

    def clear_board(self) -> None:
        if self.thread is None:
            self.view.clear_board()

    def _update_count(self, count: int) -> None:
        self.count_label.setText(f"보드 이미지 {count}장")
        self.align_button.setEnabled(count >= 2 and self.thread is None)
        crop_enabled = count >= 1 and self.thread is None
        self.crop_button.setEnabled(crop_enabled)
        self.crop_reset_button.setEnabled(crop_enabled)

    def _toggle_crop_mode(self, enabled: bool) -> None:
        self.view.set_crop_mode(enabled)
        self.crop_button.setText("자르기 완료" if enabled else "정합 영역 자르기")
        if enabled:
            self.status.setText("이미지를 선택하고 흰색 자르기 손잡이를 끌어 정합 사용 영역을 지정하세요.")
        else:
            self.status.setText("자르기 바깥 영역을 숨기고 정합 설정에 반영했습니다.")

    def _reset_crop_regions(self) -> None:
        self.view.reset_crop_regions()

    def start_alignment(self) -> None:
        paths = self.view.paths()
        if len(paths) < 2 or self.thread is not None:
            return
        self.crop_button.setChecked(False)
        self.status.setText("보드 이미지 자동 정렬 준비 중…")
        self.progress.setValue(0)
        self.align_button.setEnabled(False)
        self.crop_button.setEnabled(False)
        self.crop_reset_button.setEnabled(False)
        self.thread = QThread(self)
        self.worker = StitchWorker(paths, StitchOptions(), self.view.layout_hints())
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_finished)
        self.worker.failed.connect(self._on_failed)
        self.worker.manual.connect(self._on_failed)
        for signal in (self.worker.finished, self.worker.failed, self.worker.manual):
            signal.connect(self.thread.quit)
        self.thread.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self._thread_finished)
        self.thread.start()

    def _on_progress(self, value: int, text: str) -> None:
        self.progress.setValue(value)
        self.status.setText(text)

    def _on_finished(self, result: StitchResult) -> None:
        try:
            saved = save_stitch_result_auto(result)
        except Exception as exc:
            self._on_failed(f"합친 이미지를 자동 저장하지 못했습니다: {exc}")
            return
        self.progress.setValue(100)
        self.status.setText(
            f"합치기 완료 · 정합 {result.confidence * 100:.0f}% · {len(result.placements)}장 · {saved.name}"
        )
        self.result_ready.emit(result)

    def _on_failed(self, message: str) -> None:
        self.status.setText(message)
        QMessageBox.warning(self, "사진 합치기", message)

    def _thread_finished(self) -> None:
        if self.thread is not None:
            self.thread.deleteLater()
        self.thread = None
        self.worker = None
        self._update_count(len(self.view.items_in_board()))
