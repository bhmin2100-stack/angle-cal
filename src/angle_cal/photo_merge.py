from __future__ import annotations
from pathlib import Path
import threading
import cv2
import numpy as np
from PySide6.QtCore import QObject,Qt,QThread,Signal,Slot
from PySide6.QtGui import QImage,QPixmap
from PySide6.QtWidgets import QDialog,QDoubleSpinBox,QFileDialog,QFormLayout,QHBoxLayout,QLabel,QListWidget,QListWidgetItem,QMessageBox,QProgressBar,QPushButton,QSplitter,QTableWidget,QTableWidgetItem,QVBoxLayout,QWidget
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
        self.result=None;self.save.setEnabled(False);self.start.setEnabled(False);self.cancel.setEnabled(True);self.thread=QThread(self);self.worker=StitchWorker(self.paths(),StitchOptions(self.crop.value()/100));self.worker.moveToThread(self.thread);self.thread.started.connect(self.worker.run);self.worker.progress.connect(self.on_progress);self.worker.finished.connect(self.finished);self.worker.failed.connect(self.failed);self.worker.manual.connect(self.manual_required)
        for signal in (self.worker.finished,self.worker.failed,self.worker.manual):signal.connect(self.thread.quit)
        self.thread.finished.connect(self.worker.deleteLater);self.thread.finished.connect(self.thread_finished);self.thread.start()
    def cancel_stitch(self):
        if self.worker:self.worker.cancel_event.set();self.status.setText("취소 중…")
    def on_progress(self,value,text):self.progress.setValue(value);self.status.setText(text)
    def finished(self,result:StitchResult):
        self.result=result;self.progress.setValue(100);self.status.setText(f"완료: {result.output_size[0]} × {result.output_size[1]} px · 스케일 재보정 필요");self.save.setEnabled(True);self.preview.setPixmap(preview_pixmap(result.image).scaled(self.preview.size(),Qt.AspectRatioMode.KeepAspectRatio,Qt.TransformationMode.SmoothTransformation));self.table.setRowCount(len(result.placements))
        for r,p in enumerate(result.placements):
            for c,v in enumerate((Path(p.path).name,p.mode,p.inlier_count,f"{p.reprojection_error:.3f}")):self.table.setItem(r,c,QTableWidgetItem(str(v)))
    def failed(self,message):self.status.setText(message);QMessageBox.warning(self,"사진 합치기",message)
    def manual_required(self,message):self.status.setText("수동 보정 필요");QMessageBox.information(self,"수동 정렬 필요",message+"\n수동 기준점 편집기는 다음 업데이트에서 연결됩니다.")
    def thread_finished(self):self.thread.deleteLater();self.thread=None;self.worker=None;self.start.setEnabled(True);self.cancel.setEnabled(False)
    def save_result(self):
        if self.result is None:return
        path,_=QFileDialog.getSaveFileName(self,"합친 이미지 저장","merged.tif","TIFF (*.tif *.tiff);;PNG (*.png)")
        if path:
            try:output,_,_=save_stitch_result(path,self.result)
            except Exception as exc:QMessageBox.warning(self,"사진 합치기",str(exc));return
            self.result_saved.emit(str(output));self.accept()
    def reject(self):self.cancel_stitch();super().reject()
