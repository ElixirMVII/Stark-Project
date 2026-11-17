"""Reticle ที่ลอยตามเคอร์เซอร์ + shockwave ตอนคลิก + เส้น trail ตอนลาก

องค์ประกอบ (จากนอกเข้าใน): วงนอกบางพร้อม tick → ส่วนโค้ง 3 ช่องที่หมุนช้าๆ →
วงแหวนรอ "ค้างเพื่อลาก" → วงในที่เติมเต็มตามระดับการจีบ → จุดกลาง
"""

from __future__ import annotations

import itertools
import math
from collections import deque
from dataclasses import dataclass

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QBrush, QColor, QPainter, QPainterPath, QPen, QRadialGradient

from ..gestures import EngineSnapshot, State
from .theme import LayerCache, Theme, arc_path, glow_stroke, with_alpha

__all__ = ["Reticle"]

OUTER_RADIUS = 46.0
ARC_RADIUS = 37.0
HOLD_RADIUS = 30.0
PINCH_RADIUS = 23.0
HALF_EXTENT = 64.0  # ครึ่งด้านของกรอบที่ต้องวาดใหม่รอบ reticle (รวม tick + glow)
BACKDROP_RADIUS = 60.0  # ฉากหลังเข้มโปร่งแสง: ให้เส้นสว่างอ่านออกบนหน้าต่างสีขาว
BACKDROP_ALPHA = 0.42

FADE_SECONDS = 0.15
SPIN_DEG_PER_S = 40.0
SPIN_DRAG_DEG_PER_S = 170.0

SHOCKWAVE_SECONDS = 0.38
SHOCKWAVE_START = 12.0
SHOCKWAVE_END = 74.0
TRAIL_SECONDS = 0.45
TRAIL_POINTS = 28
_DRAG_STATES = (State.DRAGGING, State.WINDOW_DRAG)  # ลากด้วยการจีบ หรือถือหน้าต่างด้วยกำมือ


@dataclass(slots=True)
class _Shockwave:
    center: QPointF
    started: float
    color: QColor


class Reticle:
    """เก็บสถานะและวาด reticle — ตำแหน่งทั้งหมดเป็น logical px ของ overlay"""

    def __init__(self, theme: Theme) -> None:
        self._theme = theme
        self._pos: QPointF | None = None
        self._snapshot = EngineSnapshot()
        self._visible = False
        self._alpha = 0.0
        self._angle = 0.0
        self._last_time: float | None = None
        self._shockwaves: list[_Shockwave] = []
        self._trail: deque[tuple[QPointF, float]] = deque(maxlen=TRAIL_POINTS)
        self._layers = LayerCache()

    # ---- state ---------------------------------------------------------- #

    @property
    def position(self) -> QPointF | None:
        return self._pos

    @property
    def alpha(self) -> float:
        return self._alpha

    def set_state(self, position: QPointF | None, snapshot: EngineSnapshot, now: float) -> None:
        """อัปเดตจากเฟรมล่าสุด: ``position`` = ตำแหน่งเคอร์เซอร์ (logical px) หรือ ``None``"""
        self._snapshot = snapshot
        self._visible = position is not None and snapshot.hand_present
        if position is not None:
            self._pos = position
            if snapshot.state in _DRAG_STATES:
                self._trail.append((QPointF(position), now))

    def add_shockwave(self, center: QPointF, color: QColor, now: float) -> None:
        self._shockwaves.append(_Shockwave(QPointF(center), now, QColor(color)))

    def advance(self, now: float) -> bool:
        """เดินแอนิเมชันถึงเวลา ``now`` คืน ``True`` ถ้ายังมีอะไรต้องวาดต่อ"""
        dt = 0.0 if self._last_time is None else max(0.0, min(now - self._last_time, 0.1))
        self._last_time = now

        target = 1.0 if self._visible else 0.0
        step = dt / FADE_SECONDS
        if self._alpha < target:
            self._alpha = min(target, self._alpha + step)
        elif self._alpha > target:
            self._alpha = max(target, self._alpha - step)

        if self._snapshot.armed:
            dragging = self._snapshot.state in _DRAG_STATES
            speed = SPIN_DRAG_DEG_PER_S if dragging else SPIN_DEG_PER_S
            self._angle = (self._angle + speed * dt) % 360.0

        self._shockwaves = [s for s in self._shockwaves if now - s.started < SHOCKWAVE_SECONDS]
        while self._trail and now - self._trail[0][1] > TRAIL_SECONDS:
            self._trail.popleft()
        return self._alpha > 0.0 or bool(self._shockwaves) or bool(self._trail)

    def rect(self) -> QRectF:
        """กรอบที่ครอบทุกอย่างที่ reticle วาดอยู่ตอนนี้ (ว่าง = ไม่มีอะไรต้องวาด)"""
        rect = QRectF()
        if self._pos is not None and self._alpha > 0.0:
            rect = _square(self._pos, HALF_EXTENT)
        for wave in self._shockwaves:
            rect = rect.united(_square(wave.center, SHOCKWAVE_END + 10.0))
        for point, _ in self._trail:
            rect = rect.united(_square(point, 8.0))
        return rect

    # ---- paint ---------------------------------------------------------- #

    def paint(self, painter: QPainter, now: float) -> None:
        self._paint_trail(painter, now)
        self._paint_shockwaves(painter, now)
        if self._pos is not None and self._alpha > 0.0:
            self._paint_reticle(painter, self._pos, self._alpha)

    def _paint_trail(self, painter: QPainter, now: float) -> None:
        points = list(self._trail)
        for (a, _), (b, t) in itertools.pairwise(points):
            alpha = max(0.0, 1.0 - (now - t) / TRAIL_SECONDS)
            if alpha <= 0.0:
                continue
            # เส้นจางกว้าง + เส้นสว่างบาง ต่อช่วง (ถูกกว่า glow เต็มรูปแบบซึ่งต้องวาดหลายชั้นต่อช่วง)
            pen = QPen(with_alpha(self._theme.accent, 0.18 * alpha), 6.0)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(pen)
            painter.drawLine(a, b)
            pen = QPen(with_alpha(self._theme.accent, 0.85 * alpha), 2.0)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(pen)
            painter.drawLine(a, b)

    def _paint_shockwaves(self, painter: QPainter, now: float) -> None:
        for wave in self._shockwaves:
            progress = (now - wave.started) / SHOCKWAVE_SECONDS
            if not 0.0 <= progress < 1.0:
                continue
            eased = 1.0 - (1.0 - progress) ** 3  # ease-out: พุ่งเร็วแล้วช้าลง
            radius = SHOCKWAVE_START + (SHOCKWAVE_END - SHOCKWAVE_START) * eased
            path = QPainterPath()
            path.addEllipse(wave.center, radius, radius)
            glow_stroke(painter, path, wave.color, 2.0, alpha=(1.0 - progress) ** 1.5)

    def _paint_reticle(self, painter: QPainter, center: QPointF, alpha: float) -> None:
        theme = self._theme
        snapshot = self._snapshot
        armed = snapshot.armed
        dragging = snapshot.state in _DRAG_STATES
        pinched = snapshot.pinch_level >= 1.0
        base = theme.state_color(armed)
        hot = theme.accent if armed else theme.disarmed
        cx, cy = center.x(), center.y()

        # ส่วนที่ไม่ขยับ (ฉากหลัง วงนอก tick รางวงใน) วาดครั้งเดียวต่อสี แล้วใช้ซ้ำทุกเฟรม
        opacity = painter.opacity()
        painter.setOpacity(opacity * alpha)
        self._layers.draw(
            painter,
            ("static", base.rgba()),
            QPointF(cx - HALF_EXTENT, cy - HALF_EXTENT),
            HALF_EXTENT * 2.0,
            HALF_EXTENT * 2.0,
            lambda layer: self._render_static(layer, base),
        )
        painter.setOpacity(opacity)

        # ส่วนโค้ง 3 ช่องที่หมุนช้าๆ (นิ่งและจางเมื่อ DISARMED, หมุนเร็วสีอำพันเมื่อลาก)
        arcs = QPainterPath()
        for k in range(3):
            arcs.addPath(arc_path(cx, cy, ARC_RADIUS, self._angle + k * 120.0, 72.0))
        glow_stroke(
            painter,
            arcs,
            hot if dragging else base,
            2.0,
            strength=1.0 if armed else 0.0,
            alpha=alpha * (1.0 if armed else 0.6),
        )

        if snapshot.state is State.SCROLL:
            self._paint_chevrons(painter, cx, cy, base, alpha, vertical=True)
        elif snapshot.state is State.TWO_HAND_ZOOM:
            self._paint_chevrons(painter, cx, cy, hot, alpha, vertical=False)

        # วงแหวนรอให้การจีบกลายเป็นการลาก
        if snapshot.state is State.PINCH_PENDING and snapshot.pinch_button == "left":
            sweep = 360.0 * snapshot.hold_progress
            if sweep > 0.0:
                glow_stroke(
                    painter, arc_path(cx, cy, HOLD_RADIUS, 0.0, sweep), hot, 2.0, alpha=alpha
                )

        # วงใน: ส่วนที่เติมตามระดับการจีบ (เต็มวง = จีบแล้ว) — รางอยู่ในชั้นที่ cache ไว้
        level = 1.0 if dragging else snapshot.pinch_level
        if level > 0.0:
            fill = arc_path(cx, cy, PINCH_RADIUS, 0.0, 359.99 * level)
            glow_stroke(painter, fill, hot if (pinched or dragging) else base, 3.0, alpha=alpha)

        # จุดกลาง
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(with_alpha(hot if (pinched or dragging) else base, alpha)))
        radius = 4.0 if dragging else 2.5
        painter.drawEllipse(center, radius, radius)

    @staticmethod
    def _render_static(painter: QPainter, color: QColor) -> None:
        """ฉากหลังเข้มโปร่งแสง + วงนอก + tick ทุก 30° + รางของวงใน (จุดกำเนิด = มุมซ้ายบนของชั้น)"""
        center = QPointF(HALF_EXTENT, HALF_EXTENT)
        backdrop = QRadialGradient(center, BACKDROP_RADIUS)
        backdrop.setColorAt(0.0, QColor(2, 8, 14, round(255 * BACKDROP_ALPHA)))
        backdrop.setColorAt(0.75, QColor(2, 8, 14, round(255 * BACKDROP_ALPHA * 0.8)))
        backdrop.setColorAt(1.0, QColor(2, 8, 14, 0))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(backdrop))
        painter.drawEllipse(center, BACKDROP_RADIUS, BACKDROP_RADIUS)

        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(with_alpha(color, 0.45), 1.0))
        painter.drawEllipse(center, OUTER_RADIUS, OUTER_RADIUS)
        painter.setPen(QPen(with_alpha(color, 0.25), 1.0))
        painter.drawEllipse(center, PINCH_RADIUS, PINCH_RADIUS)
        painter.setPen(QPen(with_alpha(color, 0.8), 1.2))
        for step in range(12):
            angle = math.radians(step * 30.0)
            length = 8.0 if step % 3 == 0 else 4.0  # ยาวขึ้นที่ 4 ทิศหลัก
            dx, dy = math.sin(angle), -math.cos(angle)
            painter.drawLine(
                QPointF(HALF_EXTENT + dx * OUTER_RADIUS, HALF_EXTENT + dy * OUTER_RADIUS),
                QPointF(
                    HALF_EXTENT + dx * (OUTER_RADIUS + length),
                    HALF_EXTENT + dy * (OUTER_RADIUS + length),
                ),
            )

    def _paint_chevrons(
        self,
        painter: QPainter,
        cx: float,
        cy: float,
        color: QColor,
        alpha: float,
        *,
        vertical: bool,
    ) -> None:
        """ลูกศรคู่ชี้ออกจากกลาง: แนวตั้ง = โหมดสกรอล, แนวนอน = ซูมสองมือ"""
        path = QPainterPath()
        for direction in (-1.0, 1.0):
            tip, base = direction * 15.0, direction * 9.0
            if vertical:
                path.moveTo(cx - 6.0, cy + base)
                path.lineTo(cx, cy + tip)
                path.lineTo(cx + 6.0, cy + base)
            else:
                path.moveTo(cx + base, cy - 6.0)
                path.lineTo(cx + tip, cy)
                path.lineTo(cx + base, cy + 6.0)
        glow_stroke(painter, path, color, 1.8, strength=0.7, alpha=alpha)


def _square(center: QPointF, half: float) -> QRectF:
    return QRectF(center.x() - half, center.y() - half, half * 2.0, half * 2.0)
