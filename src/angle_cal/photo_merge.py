from __future__ import annotations
from pathlib import Path
import threading
import cv2
import numpy as np
from PySide6.QtCore import QObject,QPointF,Qt,QThread,Signal,Slot
from PySide6.QtGui import QImage,QKeyEvent,QPixmap,QWheelEvent
from PySide6.QtWidgets import QDialog,QDoubleSpinBox,QFileDialog,QFormLayout,QGraphicsPixmapItem,QGraphicsScene,QGraphicsView,QHBoxLayout,QLabel,QListWidget,QListWidgetItem,QMessageBox,QProgressBar,QPushButton,QSplitter,QTableWidget,QTableWidgetItem,QVBoxLayout,QWidget
from .stitching import StitchOptions,StitchResult,StitchingCancelled,StitchingNeedsManual,detect_bottom_overlay_fraction,read_raw_image,save_stitch_result,stitch_paths

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
    def __init__(self,paths,options):super().__init__();self.paths=paths;self.options=options;self.cancel_event=threading.Event()
    @Slot()
    def run(self):
        try:self.finished.emit(stitch_paths(self.paths,self.options,progress=self.on_progress,cancelled=self.cancel_event.is_set))
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


class MergeBoardItem(QGraphicsPixmapItem):
    def __init__(self, path: str, pixmap: QPixmap, view: "MergeBoardView") -> None:
        super().__init__(pixmap)
        self.path = path
        self.view = view
        self.setFlags(QGraphicsPixmapItem.GraphicsItemFlag.ItemIsMovable | QGraphicsPixmapItem.GraphicsItemFlag.ItemIsSelectable)
        self.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        self.setToolTip(Path(path).name)

    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802
        delta = 0.08 if event.delta() > 0 else -0.08
        self.setOpacity(max(0.12, min(1.0, self.opacity() + delta)))
        self.view.opacity_changed.emit(round(self.opacity() * 100))
        event.accept()


class MergeBoardView(QGraphicsView):
    paths_changed = Signal(int)
    opacity_changed = Signal(int)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.setAcceptDrops(True)
        self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        self.setRenderHints(self.renderHints())
        self.setStyleSheet("QGraphicsView { background:#24282d; border:1px solid #8b949e; }")
        self.setSceneRect(-2500, -1800, 5000, 3600)

    def items_in_board(self) -> list[MergeBoardItem]:
        return [item for item in self.scene().items() if isinstance(item, MergeBoardItem)]

    def paths(self) -> list[str]:
        return [item.path for item in sorted(self.items_in_board(), key=lambda item: (item.pos().y(), item.pos().x()))]

    def add_paths(self, paths: list[str], drop_position: QPointF | None = None) -> None:
        existing = {item.path.casefold() for item in self.items_in_board()}
        added = 0
        origin = drop_position or self.mapToScene(self.viewport().rect().center())
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
            item = MergeBoardItem(path, board_pixmap, self)
            item.setPos(origin + QPointF(34 * added, 28 * added))
            item.setZValue(len(self.items_in_board()) + 1)
            self.scene().addItem(item)
            existing.add(path.casefold())
            added += 1
        if added:
            self.paths_changed.emit(len(self.items_in_board()))

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

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802
        if event.key() == Qt.Key.Key_Delete:
            self.delete_selected()
            event.accept()
            return
        super().keyPressEvent(event)

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
        load_button = QPushButton("이미지 불러오기")
        load_button.clicked.connect(self.choose_files)
        toolbar.addWidget(load_button)
        clear_button = QPushButton("보드 비우기")
        clear_button.clicked.connect(self.clear_board)
        toolbar.addWidget(clear_button)
        self.align_button = QPushButton("맞추기")
        self.align_button.setEnabled(False)
        self.align_button.setStyleSheet("font-weight:700;padding:7px 18px")
        self.align_button.clicked.connect(self.start_alignment)
        toolbar.addWidget(self.align_button)
        root.addLayout(toolbar)
        hint = QLabel("왼쪽 썸네일을 끌어 놓으세요  ·  드래그: 위치 이동  ·  휠: 투명도  ·  Delete: 보드에서 제거  ·  잠시 올려두기: 크게 보기")
        hint.setStyleSheet("color:#586069;padding:2px")
        root.addWidget(hint)
        self.view = MergeBoardView()
        root.addWidget(self.view, 1)
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

    def add_paths(self, paths: list[str]) -> None:
        self.view.add_paths(paths)

    def choose_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(self, "합칠 이미지 불러오기", "", "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)")
        self.add_paths(paths)

    def clear_board(self) -> None:
        if self.thread is None:
            self.view.clear_board()

    def _update_count(self, count: int) -> None:
        self.count_label.setText(f"보드 이미지 {count}장")
        self.align_button.setEnabled(count >= 2 and self.thread is None)

    def start_alignment(self) -> None:
        paths = self.view.paths()
        if len(paths) < 2 or self.thread is not None:
            return
        self.status.setText("보드 이미지 자동 정렬 준비 중…")
        self.progress.setValue(0)
        self.align_button.setEnabled(False)
        self.thread = QThread(self)
        self.worker = StitchWorker(paths, StitchOptions())
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
        self.progress.setValue(100)
        self.status.setText(f"합치기 완료: {result.output_size[0]} × {result.output_size[1]} px")
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
