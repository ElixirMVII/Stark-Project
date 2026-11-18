"""กรอบล็อกเป้า (spec v1.1 M6.3): มุม 4 มุมที่ spring ออกจาก reticle ไปล้อมปุ่มใต้เคอร์เซอร์

- เป้าใหม่ → มุมเด้งย้ายไปหา (preset ``bouncy``) ไม่กระโดด, ไม่มีเป้า → หุบกลับเข้า reticle แล้วจาง
- pinch fill ใกล้เต็ม → กรอบบีบเข้าหากลางปุ่ม, คลิก → กระพริบ
- ป้ายฟอนต์ mono ``BUTTON · Save`` (ตัดสั้น) — ปิดได้ด้วย ``targeting.show_label``
"""

from __future__ import annotations

import re

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QFontMetrics, QPainter, QPainterPath

from ..gestures import EngineSnapshot
from .anim import spring
from .theme import Theme, draw_label, glow_stroke

__all__ = ["TargetLock", "kind_text"]

CORNER = 10.0  # ความยาวขาของมุม (logical px)
PAD = 6.0  # ระยะเผื่อรอบปุ่ม
SQUEEZE = 0.35  # บีบเข้าหากลางปุ่มสูงสุดเท่านี้ของระยะเผื่อ+ขอบ เมื่อ pinch fill เต็ม
FADE_SECONDS = 0.15
FLASH_SECONDS = 0.3
LABEL_MAX = 26  # ตัวอักษร
SEED = 14.0  # ขนาดสี่เหลี่ยมเริ่มต้นรอบ reticle
MINIMAL_RADIUS = 7.0  # minimal: มุมมนของกรอบรอบปุ่ม


def kind_text(control_type: str) -> str:
    """ชนิดของปุ่มแบบอ่านง่าย: ``MenuItem`` → ``Menu item``"""
    return re.sub(r"(?<=[a-z])([A-Z])", r" \1", control_type).capitalize()


class TargetLock:
    def __init__(self, theme: Theme, *, show_label: bool = True) -> None:
        self._theme = theme
        self._show_label = show_label
        self._rect_spring = spring("bouncy", (0.0, 0.0, 0.0, 0.0))
        self._goal: QRectF | None = None
        self._label = ""
        self._alpha = 0.0
        self._squeeze = 0.0
        self._flash: float | None = None
        self._last_time: float | None = None
        self._origin: QPointF | None = None
        self._painted_rect = QRectF()
        self._target_key: tuple | None = None

    @property
    def alpha(self) -> float:
        return self._alpha

    @property
    def label(self) -> str:
        return self._label

    def set_state(
        self, rect: QRectF | None, reticle: QPointF | None, snapshot: EngineSnapshot
    ) -> None:
        """``rect`` = สี่เหลี่ยมของปุ่ม (logical px ของ overlay นี้) หรือ None"""
        target = snapshot.target
        if reticle is not None:
            self._origin = QPointF(reticle)
        self._squeeze = snapshot.pinch_level if rect is not None else 0.0
        if rect is None or target is None:
            self._goal = None
            self._target_key = None
            return
        key = (target.rect, target.control_type, target.name)
        if self._goal is None and self._origin is not None and self._alpha <= 0.01:
            # เริ่มจาก reticle: มุมพุ่งออกไปล้อมปุ่ม
            o = self._origin
            self._rect_spring.reset((o.x() - SEED, o.y() - SEED, o.x() + SEED, o.y() + SEED))
        self._goal = QRectF(rect)
        if key != self._target_key:
            self._target_key = key
            name = target.name.strip().replace("\n", " ")
            if len(name) > LABEL_MAX:
                name = name[: LABEL_MAX - 1] + "…"
            kind = self._theme.label(kind_text(target.control_type))
            self._label = f"{kind} · {name}" if name else kind

    def flash(self, now: float) -> None:
        """คลิกบนปุ่มที่ล็อกอยู่"""
        if self._goal is not None:
            self._flash = now

    def advance(self, now: float) -> bool:
        dt = 0.0 if self._last_time is None else max(0.0, min(now - self._last_time, 0.1))
        self._last_time = now
        visible = self._goal is not None
        step = dt / FADE_SECONDS
        self._alpha = min(1.0, self._alpha + step) if visible else max(0.0, self._alpha - step)
        if self._flash is not None and now - self._flash >= FLASH_SECONDS:
            self._flash = None
        goal = self._goal
        if goal is None and self._origin is not None:  # หุบกลับเข้า reticle ระหว่างจาง
            o = self._origin
            goal = QRectF(o.x() - SEED, o.y() - SEED, SEED * 2, SEED * 2)
        if goal is not None and dt > 0:
            self._rect_spring.update(dt, (goal.left(), goal.top(), goal.right(), goal.bottom()))
        return self._alpha > 0.0

    def _current(self) -> QRectF:
        left, top, right, bottom = self._rect_spring.value  # type: ignore[misc]
        rect = QRectF(QPointF(left, top), QPointF(right, bottom)).normalized()
        pad = PAD * (1.0 - SQUEEZE * self._squeeze * 2.0)
        return rect.adjusted(-pad, -pad, pad, pad)

    def rect(self) -> QRectF:
        if self._alpha <= 0.0:
            return QRectF()
        frame = self._current()
        # เผื่อพื้นที่ป้ายใต้กรอบ (กว้างสุด ~ 26 ตัวอักษร)
        return frame.adjusted(-8.0, -8.0, max(8.0, 230.0 - frame.width()), 30.0)

    def paint(self, painter: QPainter, now: float) -> None:
        if self._alpha <= 0.0:
            return
        theme = self._theme
        frame = self._current()
        flash = 0.0 if self._flash is None else 1.0 - (now - self._flash) / FLASH_SECONDS
        color = theme.accent if self._squeeze > 0.6 or flash > 0 else theme.primary
        if theme.minimal:  # กรอบมนบางรอบปุ่ม
            path = QPainterPath()
            radius = min(MINIMAL_RADIUS, frame.width() / 2.0, frame.height() / 2.0)
            path.addRoundedRect(frame, radius, radius)
            theme.stroke(painter, path, color, 1.3 + 0.6 * max(flash, 0.0), alpha=self._alpha)
            self._paint_label(painter, frame, color, pill=True, font=theme.text_font(11, 500))
            return
        length = min(CORNER, frame.width() / 2.5, frame.height() / 2.5)
        path = QPainterPath()
        for cx, cy, sx, sy in (
            (frame.left(), frame.top(), 1, 1),
            (frame.right(), frame.top(), -1, 1),
            (frame.left(), frame.bottom(), 1, -1),
            (frame.right(), frame.bottom(), -1, -1),
        ):
            path.moveTo(cx + sx * length, cy)
            path.lineTo(cx, cy)
            path.lineTo(cx, cy + sy * length)
        glow_stroke(painter, path, color, 2.0, strength=1.0 + flash, alpha=self._alpha)
        self._paint_label(painter, frame, color, pill=False, font=theme.mono_font(11))

    def _paint_label(self, painter: QPainter, frame: QRectF, color, *, pill: bool, font) -> None:
        if not (self._show_label and self._label and self._goal is not None):
            return
        painter.setFont(font)
        width = QFontMetrics(font).horizontalAdvance(self._label) + (20.0 if pill else 12.0)
        box = QRectF(
            frame.left(), frame.bottom() + (6.0 if pill else 4.0), width, 22.0 if pill else 20.0
        )
        draw_label(
            painter,
            box,
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            self._label,
            color,
            self._alpha,
            pill=pill,
        )
