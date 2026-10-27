"""Small per-method smoothing controls; all amounts are original-pixel based."""
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QGroupBox, QVBoxLayout, QComboBox,
                              QStackedWidget, QWidget, QSlider, QLabel, QPushButton, QCheckBox)
from .trench_smoothing import METHODS, MAX_STRENGTH, smoothing_parameters


class SmoothingControls(QGroupBox):
    changed = Signal()
    released = Signal()
    comparison_changed = Signal(bool)

    def __init__(self, parent=None):
        super().__init__('윤곽 보정', parent)
        self.restoring = False
        layout = QVBoxLayout(self)
        self.method = QComboBox()
        self.pages = QStackedWidget()
        self.sliders, self.labels = {}, {}
        layout.addWidget(self.method); layout.addWidget(self.pages)
        for key, title in METHODS.items():
            self.method.addItem(title, key)
            page = QWidget(); row = QVBoxLayout(page); row.setContentsMargins(0, 0, 0, 0)
            slider = QSlider(Qt.Orientation.Horizontal); slider.setRange(0, MAX_STRENGTH)
            slider.setObjectName('trench_smoothing_' + key)
            slider.setAccessibleName(title + ' 보정 강도')
            slider.setToolTip('0: 추가 보정 없음 · 100: 기존 최대 · 300: 더 강하게. 마우스를 놓으면 자동 적용합니다.')
            label = QLabel(); label.setWordWrap(True)
            row.addWidget(slider); row.addWidget(label); self.pages.addWidget(page)
            self.sliders[key], self.labels[key] = slider, label
            slider.valueChanged.connect(self._changed)
            slider.sliderReleased.connect(self.released.emit)
        self.compare = QCheckBox('보정 전 경계 겹쳐 보기'); self.compare.setChecked(True)
        self.compare.toggled.connect(self.comparison_changed.emit); layout.addWidget(self.compare)
        reset = QPushButton('보정 초기화'); reset.clicked.connect(self.reset)
        layout.addWidget(reset)
        note = QLabel('방법별 강도 기억 · 한 방법씩 적용\n100: 기존 최대 · 101~300: 강한 보정\n0은 기존 인식 결과를 유지합니다.')
        note.setWordWrap(True); layout.addWidget(note)
        self.method.setCurrentIndex(self.method.findData('robust'))
        self.method.currentIndexChanged.connect(self._changed)
        self._changed()

    @property
    def strength(self):
        return self.sliders[self.method.currentData()].value()

    def is_dragging(self):
        return any(slider.isSliderDown() for slider in self.sliders.values())

    def _changed(self, *_):
        self.pages.setCurrentIndex(self.method.currentIndex())
        for key, slider in self.sliders.items():
            value = slider.value(); window, limit = smoothing_parameters(value)
            self.labels[key].setText(f'0 / {MAX_STRENGTH} · 추가 보정 없음' if not value else
                                    f'{value} / {MAX_STRENGTH} · 범위 약 {window} px\n점 이동 제한 {limit:.2g} px')
        if not self.restoring:
            self.changed.emit()

    def state(self):
        return dict(method=self.method.currentData(), strengths={k: s.value() for k, s in self.sliders.items()},
                    compare=self.compare.isChecked())

    def restore(self, state=None):
        state = state or {}
        self.restoring = True
        try:
            self.method.setCurrentIndex(max(0, self.method.findData(state.get('method', 'robust'))))
            strengths = state.get('strengths') or {}
            for key, slider in self.sliders.items():
                slider.setValue(int(strengths.get(key, 0)))
            self.compare.setChecked(bool(state.get('compare', True)))
            self._changed()
        finally:
            self.restoring = False

    def reset(self):
        self.restore(dict(method=self.method.currentData(), compare=self.compare.isChecked()))
        self.changed.emit()
