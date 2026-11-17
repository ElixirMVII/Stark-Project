"""แอนิเมชันตอนเปิดแอป (Boot v2, spec v1.1 M7.4): วงแหวนขยายแบบเหลื่อมเวลา + ข้อความ "ถอดรหัส"
(ตัวอักษรสุ่มค่อยๆ กลายเป็น ``SYSTEM ONLINE``) ~1.5 วินาที — แผง telemetry สไลด์เข้าพร้อมกัน (overlay สั่ง)

``reduce_motion=True``: ไม่มีวงแหวน/การถอดรหัส เหลือแค่ข้อความที่ fade เข้าแล้ว fade ออก
"""

from __future__ import annotations

import math
import random

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QPainter, QPainterPath, QPen

from .theme import Theme, arc_path, glow_stroke, with_alpha

__all__ = ["BootSequence", "decoded_text"]

DURATION = 1.5
FADE_OUT = 0.3  # วินาทีสุดท้ายที่ทั้งหมดจางหาย
TEXT = "SYSTEM ONLINE"
RING_RADII = (96.0, 176.0, 264.0)
RING_STAGGER = 0.12  # แต่ละวงเริ่มช้ากว่าวงก่อนหน้า (สัดส่วนของ DURATION)
TICK_RADIUS = 214.0
TICKS = 48
HALF_EXTENT = RING_RADII[-1] + 24.0
DECODE_START = 0.18  # สัดส่วนของ DURATION ที่ตัวอักษรแรกเริ่มถอดรหัส
DECODE_STEP = 0.035  # ตัวถัดไปถอดรหัสช้ากว่าตัวก่อนเท่านี้
SCRAMBLE_HZ = 24.0  # ตัวอักษรสุ่มเปลี่ยนกี่ครั้งต่อวินาที
GLYPHS = "ABCDEFGHJKLMNPQRSTUVWXYZ0123456789#%&*+<>/\\"


def _ease_out(t: float) -> float:
    t = min(max(t, 0.0), 1.0)
    return 1.0 - (1.0 - t) ** 3


def decoded_text(text: str, progress: float, frame: int, seed: int = 11) -> str:
    """ข้อความ ณ ``progress`` (0–1 ของ DURATION): ตัวที่ถึงเวลาแล้วเป็นตัวจริง ตัวที่ยังไม่ถึงเป็นตัวสุ่ม

    ตัวที่ยังไม่เริ่ม (ก่อน DECODE_START) เป็นช่องว่าง — ``frame`` เปลี่ยนตัวสุ่มตามเวลา (deterministic)
    """
    rng = random.Random(seed * 1_000_003 + frame)
    out = []
    for i, char in enumerate(text):
        resolve_at = DECODE_START + 0.12 + i * DECODE_STEP
        appear_at = DECODE_START + i * DECODE_STEP * 0.5
        if char == " " or progress >= resolve_at:
            out.append(char)
        elif progress >= appear_at:
            out.append(rng.choice(GLYPHS))
        else:
            out.append(" ")
    return "".join(out)


class BootSequence:
    def __init__(self, theme: Theme, *, reduce_motion: bool = False) -> None:
        self._theme = theme
        self._reduce_motion = reduce_motion
        self._started: float | None = None
        self._center = QPointF()

    def start(self, now: float, center: QPointF) -> None:
        self._started = now
        self._center = QPointF(center)

    def active(self, now: float) -> bool:
        return self._started is not None and now - self._started < DURATION

    def advance(self, now: float) -> bool:
        if self._started is not None and now - self._started >= DURATION:
            self._started = None
        return self._started is not None

    def rect(self) -> QRectF:
        if self._started is None:
            return QRectF()
        c = self._center
        return QRectF(c.x() - HALF_EXTENT, c.y() - HALF_EXTENT, HALF_EXTENT * 2, HALF_EXTENT * 2)

    def paint(self, painter: QPainter, now: float) -> None:
        if self._started is None:
            return
        elapsed = now - self._started
        if not 0.0 <= elapsed < DURATION:
            return
        t = elapsed / DURATION
        fade = min(1.0, (DURATION - elapsed) / FADE_OUT)
        theme = self._theme
        color = theme.primary
        cx, cy = self._center.x(), self._center.y()
        if self._reduce_motion:
            self._paint_text(painter, TEXT, cx, cy, min(1.0, elapsed / FADE_OUT) * fade, 1.0)
            return

        # วงแหวนขยายออกจากกลางจอ ทีละวง
        for index, target in enumerate(RING_RADII):
            local = (t - index * RING_STAGGER) / 0.55
            if local <= 0.0:
                continue
            radius = target * _ease_out(local)
            ring = QPainterPath()
            ring.addEllipse(self._center, radius, radius)
            glow_stroke(
                painter, ring, color, 1.6 if index else 2.2, alpha=fade * (0.9 - 0.2 * index)
            )

        # ส่วนโค้งสีอำพันกวาดรอบวงกลาง
        sweep = 300.0 * _ease_out((t - 0.1) / 0.6)
        if sweep > 0.0:
            arc = arc_path(cx, cy, RING_RADII[1] - 14.0, -150.0, sweep)
            glow_stroke(painter, arc, theme.accent, 2.0, strength=0.8, alpha=fade * 0.9)

        # tick รอบวง ทยอยปรากฏตามเข็มนาฬิกา
        shown = int(TICKS * _ease_out((t - 0.15) / 0.55))
        painter.setPen(QPen(with_alpha(color, 0.7 * fade), 1.2))
        for i in range(shown):
            angle = math.radians(i * 360.0 / TICKS)
            dx, dy = math.sin(angle), -math.cos(angle)
            length = 12.0 if i % 4 == 0 else 6.0
            painter.drawLine(
                QPointF(cx + dx * TICK_RADIUS, cy + dy * TICK_RADIUS),
                QPointF(cx + dx * (TICK_RADIUS + length), cy + dy * (TICK_RADIUS + length)),
            )

        # ข้อความ "ถอดรหัส": ตัวสุ่มค่อยๆ กลายเป็นตัวจริง ขนาบด้วยเส้นแนวนอนที่ยืดออก
        if t >= DECODE_START:
            text = decoded_text(TEXT, t, int(elapsed * SCRAMBLE_HZ))
            reveal = _ease_out((t - DECODE_START) / 0.45)
            self._paint_text(painter, text, cx, cy, fade, reveal)

    def _paint_text(
        self, painter: QPainter, text: str, cx: float, cy: float, alpha: float, reveal: float
    ) -> None:
        theme = self._theme
        color = theme.primary
        painter.setFont(theme.display_font(30, 800, spacing=7.0))
        metrics = painter.fontMetrics()
        full_width = metrics.horizontalAdvance(TEXT)
        text_rect = QRectF(cx - full_width / 2, cy - 24.0, full_width + 8.0, 48.0)
        painter.setPen(with_alpha(color, alpha))
        flags = Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        painter.drawText(text_rect, flags, text)

        half = (full_width / 2 + 18.0) * reveal
        line = QPainterPath()
        for y in (cy - 30.0, cy + 30.0):
            line.moveTo(cx - half, y)
            line.lineTo(cx + half, y)
        glow_stroke(painter, line, color, 1.2, strength=0.7, alpha=alpha * 0.8)
