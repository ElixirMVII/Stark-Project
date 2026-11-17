"""แผง Live tuning ในหน้าต่าง debug (spec v1.1 M5) — slider ของ :data:`aether.tuning.TUNABLES`

ขยับ slider แล้วแอปใช้ค่าใหม่ทันที (ไม่ต้องโหลด config ใหม่ มือที่ตามอยู่ไม่หลุด) กด "Save to config"
เพื่อเขียนลง config.toml (คอมเมนต์เดิมอยู่ครบ) หรือ "Reset" กลับไปค่าตอนเปิดแอป
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from .tuning import TUNABLES, Tunable

__all__ = ["TuningPanel"]

PANEL_WIDTH = 300


class TuningPanel(QWidget):
    changed = Signal(object)  # dict[str, float] ค่าทั้งหมดที่ต่างจากค่าตั้งต้น
    save_requested = Signal()
    reset_requested = Signal()

    def __init__(self, values: dict[str, float], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAutoFillBackground(True)
        self.setFixedWidth(PANEL_WIDTH)
        self._base = dict(values)
        self._values = dict(values)
        self._sliders: dict[str, QSlider] = {}
        self._labels: dict[str, QLabel] = {}

        grid = QGridLayout()
        grid.setColumnStretch(1, 1)
        for row, tunable in enumerate(TUNABLES):
            name = QLabel(tunable.label)
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(0, tunable.steps)
            slider.setValue(tunable.to_slider(values[tunable.key]))
            slider.valueChanged.connect(lambda pos, t=tunable: self._moved(t, pos))
            value = QLabel(tunable.text(values[tunable.key]))
            value.setMinimumWidth(48)
            value.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            grid.addWidget(name, row * 2, 0, 1, 3)
            grid.addWidget(slider, row * 2 + 1, 0, 1, 2)
            grid.addWidget(value, row * 2 + 1, 2)
            self._sliders[tunable.key] = slider
            self._labels[tunable.key] = value

        self._save = QPushButton("Save to config")
        self._save.clicked.connect(self.save_requested)
        self._reset = QPushButton("Reset")
        self._reset.clicked.connect(self.reset_requested)
        buttons = QHBoxLayout()
        buttons.addWidget(self._reset)
        buttons.addStretch(1)
        buttons.addWidget(self._save)
        self._status = QLabel()
        self._status.setWordWrap(True)

        title = QLabel("LIVE TUNING  (กด T เพื่อซ่อน)")
        font = title.font()
        font.setBold(True)
        title.setFont(font)
        layout = QVBoxLayout(self)
        layout.addWidget(title)
        layout.addLayout(grid)
        layout.addStretch(1)
        layout.addLayout(buttons)
        layout.addWidget(self._status)

    # ---- API ------------------------------------------------------------ #

    @property
    def values(self) -> dict[str, float]:
        return dict(self._values)

    def overrides(self) -> dict[str, float]:
        """ค่าที่ต่างจากค่าตั้งต้น"""
        return {k: v for k, v in self._values.items() if v != self._base.get(k)}

    def set_value(self, key: str, value: float) -> None:
        """ตั้งค่า slider จากโค้ด (ส่ง ``changed`` เหมือนผู้ใช้เลื่อนเอง)"""
        self._sliders[key].setValue(next(t for t in TUNABLES if t.key == key).to_slider(value))

    def set_values(self, values: dict[str, float], *, base: bool = False) -> None:
        """ตั้งทุก slider โดยไม่ส่ง ``changed`` (เช่นหลัง Reset หรือ Save) — ``base`` = ตั้งเป็นค่าตั้งต้นด้วย"""
        if base:
            self._base = dict(values)
        for tunable in TUNABLES:
            value = values[tunable.key]
            slider = self._sliders[tunable.key]
            blocked = slider.blockSignals(True)
            slider.setValue(tunable.to_slider(value))
            slider.blockSignals(blocked)
            self._values[tunable.key] = value
            self._labels[tunable.key].setText(tunable.text(value))

    def show_status(self, text: str, *, error: bool = False) -> None:
        self._status.setText(text)
        self._status.setStyleSheet("color: #FF3D57;" if error else "")

    # ---- internal ------------------------------------------------------- #

    def _moved(self, tunable: Tunable, position: int) -> None:
        value = tunable.from_slider(position)
        self._values[tunable.key] = value
        self._labels[tunable.key].setText(tunable.text(value))
        self._status.setText("")
        self.changed.emit(self.overrides())
