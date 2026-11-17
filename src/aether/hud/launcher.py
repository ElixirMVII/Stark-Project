"""เมนูวงกลมเปิดแอป (state ``LAUNCHER``): ช่องรอบ reticle ตามรายการใน ``[launcher].items``

ช่อง 0 อยู่บนสุด นับตามเข็มนาฬิกา (ตรงกับ :func:`aether.gestures.launcher_slice`) ช่องที่เลือกอยู่
เติมสีเน้น พอเปิดรายการแล้วช่องนั้นวาบก่อนเมนูจางหาย
"""

from __future__ import annotations

from collections.abc import Sequence
from math import cos, radians, sin

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QBrush, QColor, QFontMetrics, QPainter, QPainterPath

from ..gestures import EngineSnapshot, State
from .theme import Theme, glow_stroke, with_alpha

__all__ = ["LauncherMenu", "sector_path"]

INNER_RADIUS = 48.0
OUTER_RADIUS = 124.0
GAP_DEG = 3.0  # ช่องว่างระหว่างช่อง
HALF_EXTENT = OUTER_RADIUS + 16.0
FADE_SECONDS = 0.12
FLASH_SECONDS = 0.35
_BACKDROP = QColor(2, 8, 14)


def sector_path(
    center: QPointF, inner: float, outer: float, start_deg: float, sweep_deg: float
) -> QPainterPath:
    """ช่องรูปวงแหวน — มุมแบบเข็มนาฬิกา: 0° = บนสุด"""
    qt_start = 90.0 - start_deg  # Qt: 0° = ขวา, ค่าบวก = ทวนเข็ม
    outer_rect = QRectF(center.x() - outer, center.y() - outer, outer * 2, outer * 2)
    inner_rect = QRectF(center.x() - inner, center.y() - inner, inner * 2, inner * 2)
    path = QPainterPath()
    path.arcMoveTo(outer_rect, qt_start)
    path.arcTo(outer_rect, qt_start, -sweep_deg)
    path.arcTo(inner_rect, qt_start - sweep_deg, sweep_deg)
    path.closeSubpath()
    return path


class LauncherMenu:
    def __init__(self, theme: Theme, labels: Sequence[str]) -> None:
        self._theme = theme
        self._labels = tuple(labels)
        self._center: QPointF | None = None
        self._visible = False
        self._selection: int | None = None
        self._armed = True
        self._alpha = 0.0
        self._flash: tuple[int, float] | None = None  # (ช่อง, เวลาเริ่ม)
        self._last_time: float | None = None

    @property
    def alpha(self) -> float:
        return self._alpha

    @property
    def selection(self) -> int | None:
        return self._selection

    def set_state(self, center: QPointF | None, snapshot: EngineSnapshot) -> None:
        self._visible = (
            center is not None and snapshot.state is State.LAUNCHER and bool(self._labels)
        )
        self._armed = snapshot.armed
        if self._visible:
            self._center = QPointF(center)
            self._selection = snapshot.launcher_selection

    def flash(self, index: int, now: float) -> None:
        """ช่องที่เพิ่งเปิดวาบ (เรียกเมื่อเกิด event launch_N)"""
        if 0 <= index < len(self._labels):
            self._flash = (index, now)
            self._selection = index

    def advance(self, now: float) -> bool:
        dt = 0.0 if self._last_time is None else max(0.0, min(now - self._last_time, 0.1))
        self._last_time = now
        if self._flash is not None and now - self._flash[1] >= FLASH_SECONDS:
            self._flash = None
        showing = self._visible or self._flash is not None
        step = dt / FADE_SECONDS
        self._alpha = min(1.0, self._alpha + step) if showing else max(0.0, self._alpha - step)
        return self._alpha > 0.0

    def rect(self) -> QRectF:
        if self._center is None or self._alpha <= 0.0:
            return QRectF()
        c = self._center
        return QRectF(c.x() - HALF_EXTENT, c.y() - HALF_EXTENT, HALF_EXTENT * 2, HALF_EXTENT * 2)

    def paint(self, painter: QPainter, now: float) -> None:
        if self._center is None or self._alpha <= 0.0 or not self._labels:
            return
        theme = self._theme
        alpha = self._alpha
        base = theme.state_color(self._armed)
        count = len(self._labels)
        width = 360.0 / count
        flash_index = self._flash[0] if self._flash is not None else None
        flash = 0.0 if self._flash is None else 1.0 - (now - self._flash[1]) / FLASH_SECONDS

        font = theme.display_font(11, 700, spacing=1.5)
        painter.setFont(font)
        metrics = QFontMetrics(font)
        middle = (INNER_RADIUS + OUTER_RADIUS) / 2.0
        for index, label in enumerate(self._labels):
            start = index * width - width / 2.0 + GAP_DEG / 2.0
            sector = sector_path(self._center, INNER_RADIUS, OUTER_RADIUS, start, width - GAP_DEG)
            selected = index == self._selection
            color = theme.accent if selected else base

            fill = with_alpha(_BACKDROP, 0.6 * alpha)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(fill))
            painter.drawPath(sector)
            if selected:
                boost = 0.25 + 0.45 * max(flash, 0.0) if index == flash_index else 0.25
                painter.setBrush(QBrush(with_alpha(theme.accent, boost * alpha)))
                painter.drawPath(sector)
            glow_stroke(
                painter,
                sector,
                color,
                2.0 if selected else 1.2,
                strength=1.0 if selected else 0.4,
                alpha=alpha * (1.0 if selected else 0.7),
            )

            # ป้ายกลางช่อง (ตัดให้พอดีความกว้างของช่องตรงรัศมีกลาง)
            angle = radians(index * width)
            point = QPointF(
                self._center.x() + sin(angle) * middle, self._center.y() - cos(angle) * middle
            )
            span = max(36.0, min(2.0 * middle * sin(radians(width / 2.0)) - 8.0, 110.0))
            text = metrics.elidedText(label, Qt.TextElideMode.ElideRight, int(span))
            box = QRectF(point.x() - span / 2.0, point.y() - 10.0, span, 20.0)
            painter.setPen(with_alpha(theme.accent if selected else base, alpha))
            painter.drawText(box, Qt.AlignmentFlag.AlignCenter, text)
