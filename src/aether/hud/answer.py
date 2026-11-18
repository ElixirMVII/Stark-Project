"""การ์ดคำตอบของ Claude บน HUD (กลางจอด้านบน) — ขึ้นเมื่อถามด้วยท่ามือ ``ask_claude``

สถานะ: ``thinking`` (จุดวิ่ง) → ``streaming`` (ข้อความค่อยๆ ขึ้น) → ``done`` / ``error`` แล้วจางหายเอง
ตามความยาวข้อความ ข้อความยาวเกินการ์ดจะถูกตัดท้าย (ดูเต็มได้ในหน้าต่างผู้ช่วย Claude)

ถ้ามีสรุปความคิดส่งมา (``[claude] show_thinking``) ใต้หัวการ์ดมี "โครงข่ายความคิด" ขนาดเล็กงอกระหว่างที่คิด
"""

from __future__ import annotations

import math

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QBrush, QColor, QFontMetricsF, QPainter, QPainterPath, QPen

from ..assistant.thought_view import ThoughtStyle, paint_graph
from ..assistant.thoughts import ThoughtGraph
from .theme import Theme, chamfered_rect, with_alpha

__all__ = ["AnswerCard"]

TOP = 28.0
WIDTH = 560.0
PAD = 18.0
TITLE_H = 22.0
MAX_TEXT_H = 260.0
FADE = 0.6
MIN_HOLD = 10.0
MAX_HOLD = 45.0
CHARS_PER_SECOND = 14.0  # เวลาอ่านโดยประมาณ (ภาษาไทยปนอังกฤษ)
ERROR_HOLD = 6.0
GRAPH_H = 92.0  # แถบโครงข่ายความคิด
GRAPH_NODES = 28

_FLAGS = Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop | Qt.TextFlag.TextWordWrap


class AnswerCard:
    def __init__(self, theme: Theme) -> None:
        self._theme = theme
        self._viewport = QRectF()
        self.state = "hidden"  # hidden | thinking | streaming | done | error
        self.title = ""
        self.text = ""
        self._started = 0.0
        self._ended = 0.0
        self._text_height = 0.0
        self.graph = ThoughtGraph(max_nodes=GRAPH_NODES)
        self._graph_style = ThoughtStyle(
            theme.primary, theme.accent, theme.secondary, theme.text_family
        )
        self._last_step = 0.0
        self._energy = 0.0

    # ---- API ------------------------------------------------------------ #

    def set_viewport(self, viewport: QRectF) -> None:
        self._viewport = QRectF(viewport)

    @property
    def visible(self) -> bool:
        return self.state != "hidden"

    def begin(self, title: str, now: float) -> None:
        self.state, self.title, self.text = "thinking", title, ""
        self._started = self._last_step = now
        self.graph.reset("", now)
        self._measure()

    def add_thinking(self, piece: str, now: float) -> None:
        """ชิ้นสรุปความคิด → โหนดใหม่ในโครงข่ายเล็ก"""
        if self.state in ("hidden", "error", "done"):
            return
        self.graph.feed(piece, now)

    def append(self, piece: str, now: float) -> None:
        if self.state in ("hidden", "error"):
            return
        self.state = "streaming"
        self.text += piece
        self._measure()

    def finish(self, text: str | None, now: float) -> None:
        if self.state == "hidden":
            return
        if text is not None:
            self.text = text
        self.state, self._ended = "done", now
        if self.graph.thoughts:
            self.graph.add_answer(self.text, now)
        self._measure()

    def fail(self, message: str, now: float) -> None:
        self.state, self.text, self._ended = "error", message, now
        if not self.title:
            self.title = "Claude"
        self._measure()

    def dismiss(self) -> None:
        self.state = "hidden"

    def hold_seconds(self) -> float:
        if self.state == "error":
            return ERROR_HOLD
        return min(MAX_HOLD, max(MIN_HOLD, len(self.text) / CHARS_PER_SECOND))

    def alpha(self, now: float) -> float:
        if self.state not in ("done", "error"):
            return 1.0
        left = self._ended + self.hold_seconds() - now
        return max(0.0, min(1.0, left / FADE))

    def advance(self, now: float) -> bool:
        """``True`` = ต้องวาดใหม่ (จุดวิ่ง/ข้อความใหม่/กำลังจาง)"""
        if self.state == "hidden":
            return False
        self._energy = self.graph.step(now - self._last_step)
        self._last_step = now
        moving = self.graph.active(now, self._energy)
        if self.state in ("done", "error"):
            if self.alpha(now) <= 0.0:
                self.state = "hidden"
                return True
            return moving or now >= self._ended + self.hold_seconds() - FADE
        return True

    @property
    def has_graph(self) -> bool:
        return bool(self.graph.thoughts)

    def rect(self) -> QRectF:
        if self.state == "hidden" or self._viewport.isEmpty():
            return QRectF()
        width = min(WIDTH, self._viewport.width() - 48.0)
        body = self._text_height if self.text else 18.0
        height = PAD + TITLE_H + 6.0 + body + PAD + (GRAPH_H if self.has_graph else 0.0)
        left = self._viewport.center().x() - width / 2.0
        return QRectF(left, self._viewport.top() + TOP, width, height)

    # ---- วาด ------------------------------------------------------------- #

    def _font(self):
        return self._theme.text_font(15, 400)

    def _measure(self) -> None:
        if not self.text:
            self._text_height = 0.0
            return
        width = min(WIDTH, max(200.0, self._viewport.width() - 48.0)) - 2 * PAD
        metrics = QFontMetricsF(self._font())
        bounds = metrics.boundingRect(QRectF(0, 0, width, 10000.0), int(_FLAGS), self.text)
        self._text_height = min(MAX_TEXT_H, math.ceil(bounds.height()))

    def paint(self, painter: QPainter, now: float) -> None:
        rect = self.rect()
        if rect.isEmpty():
            return
        theme = self._theme
        alpha = self.alpha(now)
        painter.setOpacity(painter.opacity() * alpha)
        if theme.minimal:
            painter.setPen(QPen(with_alpha(theme.primary, 0.10), 1.0))
            painter.setBrush(QBrush(QColor(16, 18, 22, round(255 * 0.88))))
            painter.drawRoundedRect(rect.adjusted(0.5, 0.5, -0.5, -0.5), 14.0, 14.0)
        else:
            path: QPainterPath = chamfered_rect(rect, 12.0)
            painter.setPen(QPen(with_alpha(theme.primary, 0.8), 1.2))
            painter.setBrush(QBrush(QColor(4, 10, 16, round(255 * 0.78))))
            painter.drawPath(path)

        accent = theme.danger if self.state == "error" else theme.accent
        dot_x, dot_y = rect.left() + PAD + 4.0, rect.top() + PAD + TITLE_H / 2.0
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(accent))
        painter.drawEllipse(QRectF(dot_x - 4.0, dot_y - 4.0, 8.0, 8.0))
        painter.setPen(theme.primary)
        painter.setFont(theme.text_font(13, 600))
        title_rect = QRectF(
            rect.left() + PAD + 16.0, rect.top() + PAD, rect.width() - 2 * PAD - 16.0, TITLE_H
        )
        painter.drawText(
            title_rect,
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            theme.label(self.title or "Claude"),
        )

        top = rect.top() + PAD + TITLE_H + 6.0
        if self.has_graph:
            strip = QRectF(rect.left() + PAD, top, rect.width() - 2 * PAD, GRAPH_H - 8.0)
            painter.save()
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(with_alpha(theme.primary, 0.04)))
            painter.drawRoundedRect(strip, 8.0, 8.0)
            paint_graph(
                painter,
                self.graph,
                strip.adjusted(8, 6, -8, -6),
                now,
                self._graph_style,
                compact=True,
            )
            painter.setFont(theme.text_font(10, 500))
            painter.setPen(theme.secondary)
            painter.drawText(
                strip.adjusted(8, 4, -8, -4),
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignBottom,
                f"{len(self.graph.thoughts)} thoughts",
            )
            painter.restore()
            top += GRAPH_H
        body = QRectF(rect.left() + PAD, top, rect.width() - 2 * PAD, rect.bottom() - PAD - top)
        if self.state == "thinking" or not self.text:
            painter.setPen(Qt.PenStyle.NoPen)
            for i in range(3):  # จุดสามจุดกระเพื่อม
                phase = (now - self._started) * 3.0 - i * 0.5
                lift = 0.35 + 0.65 * max(0.0, math.sin(phase))
                painter.setBrush(QBrush(with_alpha(theme.primary, lift)))
                painter.drawEllipse(QRectF(body.left() + i * 14.0, body.top() + 5.0, 7.0, 7.0))
            return
        painter.setPen(theme.danger if self.state == "error" else with_alpha(theme.primary, 0.92))
        painter.setFont(self._font())
        painter.save()
        painter.setClipRect(body)
        painter.drawText(body, int(_FLAGS), self.text)
        painter.restore()
