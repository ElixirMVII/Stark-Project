"""หน้าต่าง calibration wizard — แสดงขั้นตอนของ :class:`aether.calibration.CalibrationSession`

รับ ``HandFrame`` จากกล้องทีละเฟรม (ผ่าน :meth:`CalibrationWizard.show_frame`) จบแล้วเขียน
``calibration.json`` และส่ง signal ``finished(True)`` ให้แอปโหลด config ใหม่
ระหว่างเปิด wizard แอปจะอยู่ในสถานะ DISARMED (มือไม่คุมเมาส์)
"""

from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from . import ui_style
from .calibration import CalibrationSession, save_calibration
from .config import Config
from .tracker import Hand, HandFrame, hand_connections

__all__ = ["CalibrationWizard", "HandPreview"]

log = logging.getLogger(__name__)

_BACKGROUND = QColor("#0B0F14")
_DIM = QColor("#7A8B99")
_WARN = ui_style.WARN


class HandPreview(QWidget):
    """โครงมือ 21 จุดของมือข้างถนัด (ซูมให้เต็มกรอบ) — ให้ผู้ใช้เห็นว่ากล้องจับมือได้"""

    def __init__(self, color: QColor, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._color = color
        self._hand: Hand | None = None
        self._aspect = 4.0 / 3.0
        self._connections = hand_connections()
        self.setMinimumSize(220, 180)

    def set_hand(self, hand: Hand | None, width: int, height: int) -> None:
        self._hand = hand
        if height > 0:
            self._aspect = width / height
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), _BACKGROUND)
        if self._hand is None:
            painter.setPen(_DIM)
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "ไม่เห็นมือข้างถนัด")
            return
        area = QRectF(self.rect()).adjusted(18, 18, -18, -18)
        xs = [p.x * self._aspect for p in self._hand.landmarks]
        ys = [p.y for p in self._hand.landmarks]
        span = max(max(xs) - min(xs), max(ys) - min(ys), 1e-6)
        scale = min(area.width(), area.height()) / span
        mid_x, mid_y = (max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2
        points = [
            QPointF(
                area.center().x() + (x - mid_x) * scale, area.center().y() + (y - mid_y) * scale
            )
            for x, y in zip(xs, ys)
        ]
        painter.setPen(QPen(self._color, 2))
        for start, end in self._connections:
            painter.drawLine(points[start], points[end])
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(self._color)
        for point in points:
            painter.drawEllipse(point, 3, 3)


class CalibrationWizard(QWidget):
    """เรียก :meth:`show_frame` จาก UI thread ทุกเฟรม"""

    finished = Signal(bool)  # True = บันทึก calibration.json แล้ว, False = ยกเลิก

    def __init__(self, config: Config, path: Path, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._config = config
        self._path = path
        self._session = CalibrationSession(config.tracking.dominant_hand)
        self._saved = False
        self._reported = False

        self.setWindowTitle("AETHER — Calibration")
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint)
        self.setMinimumWidth(460)

        title_font = QFont(self.font())
        title_font.setPointSize(title_font.pointSize() + 4)
        title_font.setBold(True)
        self._title = QLabel()
        self._title.setFont(title_font)
        self._instruction = QLabel()
        self._instruction.setWordWrap(True)
        self._preview = HandPreview(QColor(config.hud.palette.primary))
        self._status = QLabel()
        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setTextVisible(False)
        self._hint = QLabel()
        self._hint.setWordWrap(True)
        self._hint.setStyleSheet(f"color: {_WARN};")

        self._restart_button = QPushButton("เริ่มใหม่")
        self._restart_button.clicked.connect(self.restart)
        self._close_button = QPushButton("ยกเลิก")
        self._close_button.clicked.connect(self.close)
        buttons = QHBoxLayout()
        buttons.addWidget(self._restart_button)
        buttons.addStretch(1)
        buttons.addWidget(self._close_button)

        # จองความสูงให้ข้อความครบทุกบรรทัด (สถานะ 3 บรรทัด, คำแนะนำ 2 บรรทัด) ภาพมือได้พื้นที่ที่เหลือ
        line = self.fontMetrics().lineSpacing()
        self._instruction.setMinimumHeight(line * 2 + 4)
        self._status.setWordWrap(True)
        self._status.setMinimumHeight(line * 3 + 4)
        self._hint.setMinimumHeight(line * 2 + 4)

        layout = QVBoxLayout(self)
        layout.addWidget(self._title)
        layout.addWidget(self._instruction)
        layout.addWidget(self._preview, 1)
        layout.addWidget(self._status)
        layout.addWidget(self._progress)
        layout.addWidget(self._hint)
        layout.addLayout(buttons)
        self._refresh()

    @property
    def session(self) -> CalibrationSession:
        return self._session

    @property
    def saved(self) -> bool:
        return self._saved

    def restart(self) -> None:
        self._session.restart()
        self._saved = False
        self._refresh()

    def show_frame(self, frame: HandFrame) -> None:
        """ป้อนเฟรมล่าสุดจากกล้อง"""
        if self._saved:
            return
        self._session.update(frame)
        hand = frame.hand(self._config.tracking.dominant_hand)
        self._preview.set_hand(hand, frame.width, frame.height)
        if self._session.done:
            self._save()
        self._refresh()

    def closeEvent(self, event) -> None:
        if not self._reported:
            self._reported = True
            self.finished.emit(self._saved)
        super().closeEvent(event)

    # ---- internal ------------------------------------------------------- #

    def _save(self) -> None:
        result = self._session.result
        assert result is not None
        try:
            save_calibration(self._path, result, self._config.tracking.dominant_hand)
        except OSError as exc:
            log.error("cannot write %s: %s", self._path, exc)
            self._hint.setText(f"บันทึก {self._path} ไม่ได้: {exc}")
            return
        self._saved = True
        log.info(
            "calibration saved to %s (pinch on=%.3f off=%.3f, region=%s)",
            self._path,
            result.pinch_on,
            result.pinch_off,
            result.active_region,
        )

    def _refresh(self) -> None:
        session = self._session
        self._title.setText(session.title)
        self._instruction.setText(session.instruction)
        self._progress.setValue(round(session.progress * 100))
        if self._saved and session.result is not None:
            result = session.result
            x0, y0, x1, y1 = result.active_region
            self._status.setText(
                f"pinch.on = {result.pinch_on:.2f}   pinch.off = {result.pinch_off:.2f}\n"
                f"active region = [{x0:.2f}, {y0:.2f}, {x1:.2f}, {y1:.2f}]\n"
                f"บันทึกแล้ว: {self._path.name} (ชี้ค้างเพื่อดูตำแหน่งไฟล์)"
            )
            self._status.setToolTip(str(self._path))
            self._hint.setText("")
            self._close_button.setText("ปิด")
            return
        self._close_button.setText("ยกเลิก")
        if session.error:
            self._hint.setText(session.error)
        elif not session.done:
            self._hint.setText("")

        if not session.hand_visible:
            self._status.setText("ยกมือข้างถนัดให้กล้องเห็น")
        elif session.sampling:
            ratio = (
                f"   (ระยะนิ้ว {session.live_ratio:.2f})" if session.live_ratio is not None else ""
            )
            self._status.setText(f"กำลังวัด… ค้างท่าไว้{ratio}")
        else:
            self._status.setText(f"เตรียมตัว… เริ่มวัดใน {session.countdown_ms / 1000:.1f} วินาที")
