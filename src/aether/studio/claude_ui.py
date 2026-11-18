"""ชิ้นส่วนร่วมของ Studio สำหรับคำตอบจาก Claude: สถานะคำตอบ + การ์ดที่วาดบนผืนผ้าใบ + แปลงภาพเป็น PNG"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

from PySide6.QtCore import QBuffer, QByteArray, QIODevice, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QFontMetricsF, QImage, QPainter, QPen

from .. import ui_style
from ..hud.theme import sans_family, with_alpha

__all__ = ["StudioAnswer", "png_bytes"]

_FLAGS = Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop | Qt.TextFlag.TextWordWrap
DANGER = QColor(ui_style.DANGER)
PRIMARY = QColor("#F2F4F7")  # เท่ากับ pages.PRIMARY (pages import โมดูลนี้ จึง import กลับไม่ได้)
ACCENT = QColor(ui_style.ACCENT)


def _font(pixel_size: float, weight: int = 400) -> QFont:
    font = QFont(sans_family())
    font.setPixelSize(max(1, round(pixel_size)))
    font.setWeight(QFont.Weight(weight))
    return font


def png_bytes(image: QImage) -> bytes:
    data = QByteArray()
    buffer = QBuffer(data)
    buffer.open(QIODevice.OpenModeFlag.WriteOnly)
    image.save(buffer, "PNG")
    buffer.close()
    return bytes(data.data())


class StudioAnswer:
    """คำตอบหนึ่งรายการ: ``idle`` → ``thinking`` → ``streaming`` → ``done`` / ``error``"""

    def __init__(self, clock: Callable[[], float]) -> None:
        self._clock = clock
        self.state = "idle"
        self.title = ""
        self.text = ""
        self.started = 0.0

    @property
    def visible(self) -> bool:
        return self.state != "idle"

    @property
    def busy(self) -> bool:
        return self.state in ("thinking", "streaming")

    def begin(self, title: str) -> None:
        self.state, self.title, self.text = "thinking", title, ""
        self.started = self._clock()

    def append(self, piece: str) -> None:
        if self.busy:
            self.state = "streaming"
            self.text += piece

    def finish(self, text: Any = None) -> None:
        if isinstance(text, str):
            self.text = text
        self.state = "done"

    def fail(self, message: str) -> None:
        self.state, self.text = "error", message

    def dismiss(self) -> None:
        self.state = "idle"

    def paint(self, painter: QPainter, rect: QRectF, *, card: bool = True) -> None:
        """การ์ด (หรือแค่ข้อความถ้า ``card=False``) — ข้อความยาวเกินกรอบจะถูกตัด"""
        if not self.visible or rect.isEmpty():
            return
        painter.save()
        if card:
            painter.setPen(QPen(QColor(255, 255, 255, 26), 1.0))
            painter.setBrush(QColor(24, 27, 32, 238))
            painter.drawRoundedRect(rect, 12, 12)
            inner = rect.adjusted(16, 12, -16, -12)
        else:
            inner = QRectF(rect)
        color = DANGER if self.state == "error" else ACCENT
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(color)
        painter.drawEllipse(QRectF(inner.left(), inner.top() + 6, 7, 7))
        painter.setFont(_font(12, 600))
        painter.setPen(PRIMARY)
        painter.drawText(
            QRectF(inner.left() + 14, inner.top(), inner.width() - 14, 20),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            self.title or "Claude",
        )
        body = inner.adjusted(0, 26, 0, 0)
        if self.state == "thinking":
            painter.setPen(Qt.PenStyle.NoPen)
            for i in range(3):
                lift = 0.3 + 0.7 * max(
                    0.0, math.sin((self._clock() - self.started) * 3.0 - i * 0.5)
                )
                painter.setBrush(with_alpha(PRIMARY, lift))
                painter.drawEllipse(QRectF(body.left() + i * 13, body.top() + 4, 6, 6))
        else:
            painter.setFont(_font(13, 400))
            painter.setPen(DANGER if self.state == "error" else with_alpha(PRIMARY, 0.92))
            painter.setClipRect(body)
            painter.drawText(body, _FLAGS, self.text)
        painter.restore()

    def height_for(self, width: float) -> float:
        """ความสูงของการ์ดที่พอดีข้อความ (กว้าง ``width``)"""
        if self.state == "thinking" or not self.text:
            return 24 + 26 + 18
        metrics = QFontMetricsF(_font(13, 400))
        bounds = metrics.boundingRect(QRectF(0, 0, width - 32, 100000), int(_FLAGS), self.text)
        return 24 + 26 + bounds.height() + 4
