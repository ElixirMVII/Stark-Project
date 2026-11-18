"""หน้าต่าง AETHER Studio: แถบหัว (ชื่อ + แท็บ + สถานะมือ) / หน้าปัจจุบัน / แถบควบคุม / คำแนะนำท่ามือ

เปิดจาก tray → "Studio…" หรือ ``python -m aether --studio`` ระหว่างที่เปิด มือ **ไม่คุมเมาส์** (เหมือน wizard)
ใช้กับ ``--source mouse`` หรือเมาส์จริงบนผืนผ้าใบได้ ถ้ายังไม่มีกล้อง
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QStackedWidget,
    QTabBar,
    QVBoxLayout,
    QWidget,
)

from .. import ui_style
from ..config import Config
from ..tracker import HandFrame
from .chem_page import ChemistryPage, chemistry_controls
from .input import StudioInput
from .pages import (
    DrawPage,
    LabPage,
    SandboxPage,
    StudioPage,
    ThereminPage,
    draw_controls,
    lab_controls,
    page_shortcuts,
    sandbox_controls,
    theremin_controls,
)

__all__ = ["StudioWindow"]

TICK_MS = 16


class StudioWindow(QWidget):
    """เรียก :meth:`show_frame` จาก UI thread ทุกเฟรม (แบบเดียวกับ wizard)"""

    finished = Signal(bool)  # False เสมอ: Studio ไม่ได้เปลี่ยน config/ท่าที่สอนไว้

    def __init__(
        self,
        config: Config,
        *,
        samples_path: Path | None = None,
        save_dir: Path | None = None,
        audio: bool = True,
        clock: Callable[[], float] = time.perf_counter,
        claude=None,
        on_answer: Callable[[str, str], None] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        """``claude`` = :class:`aether.assistant.tasks.ClaudeBridge` (หน้า Draw/Elements ใช้ถาม Claude)"""
        super().__init__(parent)
        self.setWindowTitle("AETHER Studio")
        self.setMinimumSize(900, 600)
        self.resize(1180, 760)
        self._clock = clock
        self._reported = False
        self.input = StudioInput(
            config.pointer.active_region,
            pinch_on=config.pinch.on,
            pinch_off=config.pinch.off,
        )

        self.draw = DrawPage(clock=clock, save_dir=save_dir)
        self.sandbox = SandboxPage(clock=clock)
        self.theremin = ThereminPage(clock=clock, audio=audio)
        self.lab = LabPage(clock=clock, samples_path=samples_path)
        self.elements = ChemistryPage(clock=clock)
        for page in (self.draw, self.elements):
            page.claude = claude
            page.on_answer = on_answer
        self.pages: list[StudioPage] = [
            self.draw,
            self.sandbox,
            self.theremin,
            self.elements,
            self.lab,
        ]
        controls = [
            draw_controls(self.draw),
            sandbox_controls(self.sandbox),
            theremin_controls(self.theremin),
            chemistry_controls(self.elements),
            lab_controls(self.lab),
        ]
        self.lab_label = controls[4].label_edit  # type: ignore[attr-defined]

        title = QLabel("Studio")
        title.setStyleSheet("font-size: 17px; font-weight: 600;")
        self.tabs = QTabBar()
        self.tabs.setDrawBase(False)
        self.tabs.setExpanding(False)
        for page in self.pages:
            self.tabs.addTab(page.title)
        self.status = QLabel()
        self.status.setStyleSheet(f"color: {ui_style.TEXT_DIM};")
        header = QHBoxLayout()
        header.addWidget(title)
        header.addSpacing(24)
        header.addWidget(self.tabs)
        header.addStretch(1)
        header.addWidget(self.status)

        self.stack = QStackedWidget()
        self.controls = QStackedWidget()
        for page, bar in zip(self.pages, controls, strict=True):
            self.stack.addWidget(page)
            self.controls.addWidget(bar)
        self.hint = QLabel()
        self.hint.setWordWrap(True)
        self.hint.setStyleSheet(f"color: {ui_style.TEXT_DIM};")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 14, 20, 14)
        layout.setSpacing(12)
        layout.addLayout(header)
        layout.addWidget(self.stack, 1)
        layout.addWidget(self.controls)
        layout.addWidget(self.hint)

        self.tabs.currentChanged.connect(self._page_changed)
        page_shortcuts(
            self,
            {
                "1": lambda: self.tabs.setCurrentIndex(0),
                "2": lambda: self.tabs.setCurrentIndex(1),
                "3": lambda: self.tabs.setCurrentIndex(2),
                "4": lambda: self.tabs.setCurrentIndex(3),
                "5": lambda: self.tabs.setCurrentIndex(4),
                "Ctrl+Z": self.draw.undo,
                "Ctrl+S": lambda: self.draw.save_png(),
                "Escape": self.close,
            },
        )

        self._timer = QTimer(self)
        self._timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._timer.setInterval(TICK_MS)
        self._timer.timeout.connect(self._tick)
        self._timer.start()
        self._page_changed(0)
        self._update_status()

    # ---- API ------------------------------------------------------------ #

    @property
    def current(self) -> StudioPage:
        return self.pages[self.tabs.currentIndex()]

    def set_page(self, index: int) -> None:
        self.tabs.setCurrentIndex(index)

    def show_frame(self, frame: HandFrame) -> None:
        hands = self.input.update(frame)
        now = self._clock()
        self.current.set_hands(hands, now)
        if self.current is self.lab:
            self.lab.show_hand_frame(frame)
        self._update_status()

    # ---- internal ------------------------------------------------------- #

    def _page_changed(self, index: int) -> None:
        self.stack.setCurrentIndex(index)
        self.controls.setCurrentIndex(index)
        page = self.pages[index]
        self.hint.setText(page.hint)
        if page is self.theremin:
            self.theremin.start_audio()
        else:
            self.theremin.stop_audio()
        for other in self.pages:  # หน้าที่ไม่ได้ดู: ลืมมือ (ไม่มีลูกบอลค้างในมือ / เส้นค้าง)
            if other is not page:
                other.set_hands((), self._clock())
        page.setFocus()

    def _update_status(self) -> None:
        hands = self.input.hands
        if not hands:
            self.status.setText("No hand in view — the mouse works too")
            return
        parts = []
        for hand in hands:
            state = (
                "pinch"
                if hand.pinch
                else (hand.gesture.replace("_", " ").lower() if hand.gesture != "None" else "open")
            )
            parts.append(f"{hand.handedness.capitalize()} · {state}")
        self.status.setText("   ".join(parts))

    def _tick(self) -> None:
        self.current.tick(self._clock())

    def closeEvent(self, event) -> None:
        self._timer.stop()
        self.theremin.stop_audio()
        if not self._reported:
            self._reported = True
            self.finished.emit(False)
        super().closeEvent(event)
