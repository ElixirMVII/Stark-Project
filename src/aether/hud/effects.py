"""Effect สั้นๆ ของ HUD (spec v1.1 M7.4): อนุภาคตอนคลิก, แสงพุ่งตอนปัด, กรอบหน้าต่างยุบตอนย่อ

- :class:`ParticlePool` — object pool ไม่เกิน ``MAX_PARTICLES`` ตัว สร้างครั้งเดียวตอนเริ่ม ใช้ซ้ำตลอด
  (ไม่สร้าง object ใหม่ทุกเฟรม) แต่ละตัวอยู่ ``PARTICLE_LIFE`` วินาที
- :class:`SwipeStreak` — แถบแสงพุ่งผ่านจอตามทิศที่ปัด
- :class:`MinimizeCollapse` — กรอบของหน้าต่างที่เพิ่งย่อ ยุบลงไปทาง taskbar

ทุกชิ้นมี ``advance(now) -> bool`` / ``rect()`` / ``paint(painter, now)`` แบบเดียวกับชิ้นอื่นของ overlay
"""

from __future__ import annotations

import math
import random

from PySide6.QtCore import QLineF, QPointF, QRectF, Qt
from PySide6.QtGui import QBrush, QColor, QLinearGradient, QPainter, QPainterPath, QPen

from .theme import Theme, glow_stroke, with_alpha

__all__ = [
    "MAX_PARTICLES",
    "MinimizeCollapse",
    "ParticlePool",
    "SwipeStreak",
]

MAX_PARTICLES = 150
PARTICLE_LIFE = 0.4
PARTICLES_PER_BURST = 14  # ครึ่งหนึ่งเมื่อ hud.quality = "low"
PARTICLE_SPEED = (90.0, 260.0)  # logical px/s
PARTICLE_DRAG = 3.5  # ชะลอแบบ exponential ต่อวินาที

STREAK_SECONDS = 0.42
STREAK_WIDTH = 220.0  # ความหนาของแถบแสง (logical px)

COLLAPSE_SECONDS = 0.38


class _Particle:
    __slots__ = ("alive", "born", "color", "vx", "vy", "x", "y")

    def __init__(self) -> None:
        self.alive = False
        self.x = self.y = self.vx = self.vy = 0.0
        self.born = 0.0
        self.color = QColor()


class ParticlePool:
    def __init__(
        self, theme: Theme, *, per_burst: int = PARTICLES_PER_BURST, seed: int = 7
    ) -> None:
        self._theme = theme
        self._pool = [_Particle() for _ in range(MAX_PARTICLES)]  # สร้างครั้งเดียว
        self._per_burst = per_burst
        self._rng = random.Random(seed)
        self._last_time: float | None = None
        self._bounds = QRectF()

    @property
    def alive(self) -> int:
        return sum(1 for p in self._pool if p.alive)

    @property
    def capacity(self) -> int:
        return len(self._pool)

    def burst(self, center: QPointF, color: QColor, now: float) -> None:
        spawned = 0
        for particle in self._pool:
            if spawned >= self._per_burst:
                break
            if particle.alive:
                continue
            angle = self._rng.uniform(0.0, math.tau)
            speed = self._rng.uniform(*PARTICLE_SPEED)
            particle.alive = True
            particle.x, particle.y = center.x(), center.y()
            particle.vx, particle.vy = math.cos(angle) * speed, math.sin(angle) * speed
            particle.born = now
            particle.color = color  # QColor ของ theme (ไม่คัดลอก — ไม่สร้าง object ใหม่)
            spawned += 1

    def advance(self, now: float) -> bool:
        dt = 0.0 if self._last_time is None else max(0.0, min(now - self._last_time, 0.1))
        self._last_time = now
        drag = math.exp(-PARTICLE_DRAG * dt)
        left = top = math.inf
        right = bottom = -math.inf
        any_alive = False
        for p in self._pool:
            if not p.alive:
                continue
            if now - p.born >= PARTICLE_LIFE:
                p.alive = False
                continue
            p.x += p.vx * dt
            p.y += p.vy * dt
            p.vx *= drag
            p.vy *= drag
            any_alive = True
            left, top = min(left, p.x), min(top, p.y)
            right, bottom = max(right, p.x), max(bottom, p.y)
        self._bounds = (
            QRectF(left - 8, top - 8, right - left + 16, bottom - top + 16)
            if any_alive
            else QRectF()
        )
        return any_alive

    def rect(self) -> QRectF:
        return QRectF(self._bounds)

    def paint(self, painter: QPainter, now: float) -> None:
        for p in self._pool:
            if not p.alive:
                continue
            life = 1.0 - (now - p.born) / PARTICLE_LIFE
            if life <= 0.0:
                continue
            tail = QPointF(p.x - p.vx * 0.03, p.y - p.vy * 0.03)  # เส้นสั้นตามทิศที่พุ่ง
            pen = QPen(with_alpha(p.color, life), 1.6)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(pen)
            painter.drawLine(QLineF(tail, QPointF(p.x, p.y)))


class SwipeStreak:
    """แถบแสงพุ่งผ่าน viewport ตามทิศที่ปัด"""

    def __init__(self, theme: Theme) -> None:
        self._theme = theme
        self._viewport = QRectF()
        self._direction: tuple[int, int] | None = None
        self._started = 0.0

    def set_viewport(self, viewport: QRectF) -> None:
        self._viewport = QRectF(viewport)

    def start(self, direction: str, now: float) -> None:
        self._direction = {"left": (-1, 0), "right": (1, 0), "up": (0, -1), "down": (0, 1)}.get(
            direction
        )
        self._started = now

    def _progress(self, now: float) -> float | None:
        if self._direction is None:
            return None
        progress = (now - self._started) / STREAK_SECONDS
        return progress if 0.0 <= progress < 1.0 else None

    def advance(self, now: float) -> bool:
        if self._direction is not None and self._progress(now) is None and now > self._started:
            self._direction = None
        return self._direction is not None

    def _band(self, progress: float) -> QRectF:
        v = self._viewport
        eased = 1.0 - (1.0 - progress) ** 2
        dx, dy = self._direction or (0, 0)
        if dx:
            span = v.width() + STREAK_WIDTH
            start = v.left() - STREAK_WIDTH if dx > 0 else v.right()
            x = start + dx * span * eased
            return QRectF(x, v.top(), STREAK_WIDTH, v.height())
        span = v.height() + STREAK_WIDTH
        start = v.top() - STREAK_WIDTH if dy > 0 else v.bottom()
        y = start + dy * span * eased
        return QRectF(v.left(), y, v.width(), STREAK_WIDTH)

    def rect(self) -> QRectF:
        if self._direction is None:
            return QRectF()
        return QRectF(self._viewport)  # แถบเคลื่อนทั้งจอ: วาดใหม่ทั้ง viewport ตลอด ~0.4 วินาที

    def paint(self, painter: QPainter, now: float) -> None:
        progress = self._progress(now)
        if progress is None:
            return
        band = self._band(progress)
        dx, dy = self._direction or (0, 0)
        if dx:
            gradient = QLinearGradient(band.left(), 0, band.right(), 0)
        else:
            gradient = QLinearGradient(0, band.top(), 0, band.bottom())
        strength = 0.22 * (1.0 - progress)
        lead, tail = (1.0, 0.0) if (dx > 0 or dy > 0) else (0.0, 1.0)
        gradient.setColorAt(tail, with_alpha(self._theme.primary, 0.0))
        gradient.setColorAt(
            0.5 * (lead + tail) + 0.35 * (lead - tail), with_alpha(self._theme.primary, strength)
        )
        gradient.setColorAt(lead, with_alpha(self._theme.primary, 0.0))
        painter.fillRect(band, QBrush(gradient))


class MinimizeCollapse:
    """กรอบหน้าต่าง (logical px) ยุบลงไปหาจุดบน taskbar"""

    def __init__(self, theme: Theme) -> None:
        self._theme = theme
        self._start: QRectF | None = None
        self._goal = QPointF()
        self._started = 0.0

    def start(self, window: QRectF, goal: QPointF, now: float) -> None:
        self._start = QRectF(window)
        self._goal = QPointF(goal)
        self._started = now

    def _current(self, now: float) -> tuple[QRectF, float] | None:
        if self._start is None:
            return None
        progress = (now - self._started) / COLLAPSE_SECONDS
        if not 0.0 <= progress < 1.0:
            return None
        eased = progress * progress * (3.0 - 2.0 * progress)
        s = self._start
        width = s.width() * (1.0 - 0.94 * eased)
        height = s.height() * (1.0 - 0.97 * eased)
        cx = s.center().x() + (self._goal.x() - s.center().x()) * eased
        cy = s.center().y() + (self._goal.y() - s.center().y()) * eased
        return QRectF(cx - width / 2, cy - height / 2, width, height), 1.0 - progress

    def advance(self, now: float) -> bool:
        if self._start is not None and self._current(now) is None and now > self._started:
            self._start = None
        return self._start is not None

    def rect(self) -> QRectF:
        if self._start is None:
            return QRectF()
        s = self._start
        return s.united(QRectF(self._goal.x() - 20, self._goal.y() - 20, 40, 40)).adjusted(
            -12, -12, 12, 12
        )

    def paint(self, painter: QPainter, now: float) -> None:
        current = self._current(now)
        if current is None:
            return
        rect, alpha = current
        path = QPainterPath()
        path.addRect(rect)
        glow_stroke(painter, path, self._theme.accent, 1.6, strength=0.8, alpha=alpha)
