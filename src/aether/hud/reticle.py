"""Reticle ที่ลอยตามเคอร์เซอร์ (spec v1.1 M7.2) — นิยามเป็นชั้นของเส้นโค้ง แล้ว "แปลงร่าง" ระหว่างโหมด

แต่ละชั้น (:class:`Ring`) มี ``radius, thickness, span, alpha, dash, spin`` แต่ละโหมดเป็นแค่ชุดค่าเป้าหมาย
(:data:`MODES`) ตอนเปลี่ยนโหมดค่าทุกตัววิ่งเข้าหาค่าใหม่ด้วย spring เดียวกัน ภาพจึงเปลี่ยนต่อเนื่องไม่ตัด:

- ``POINTER`` วงแหวน 3 ชั้น หมุนช้าๆ · ``DRAGGING`` บีบแน่น + เส้น trail · ``SCROLL`` ยืดเป็นรางแนวตั้ง มีลูกศร
  ขึ้นลงที่สว่างตามทิศ · ``GESTURE`` กางเป็น 4 ทิศ · ``CHARGING`` วงนอกกลายเป็นวงชาร์จ (สี accent) ·
  ``DISARMED`` วงเส้นประสีเทา หมุนช้ามาก

เพิ่มเติม: ขนาดตามระยะมือ (hand_scale → 0.85–1.25×, M7.3), หายใจ (alpha ±8% รอบ 3 วินาที) และ tick กระพริบ
(M7.4), shockwave ตอนคลิก, trail 300 ms ตอนลาก — ``reduce_motion`` ปิดการหายใจและ tick กระพริบ

สไตล์ minimal (:data:`MINIMAL_MODES`) ใช้กลไกแปลงร่างเดียวกัน แต่เหลือวงแหวนบางวงเดียว (ไม่หมุน ไม่มี tick
ไม่หายใจ) ระดับการจีบเป็นเส้นโค้งสีเน้นที่ค่อยๆ วิ่งรอบวง ลากอยู่ = วงหดเล็กลงเป็นสีเน้น
"""

from __future__ import annotations

import itertools
import math
import random
from collections import deque
from dataclasses import astuple, dataclass

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QBrush, QColor, QPainter, QPainterPath, QPen, QRadialGradient

from ..gestures import EngineSnapshot, EventKind, State
from .anim import SecondOrderDynamics, spring
from .theme import LayerCache, Theme, arc_path, glow_stroke, with_alpha

__all__ = ["MINIMAL_MODES", "MODES", "MORPH", "Reticle", "Ring", "depth_scale", "reticle_mode"]

PULSE_OVERSHOOT = 1.15  # จังหวะคลิกเด้งเกินขนาดปกติได้ราวนี้
BACKDROP_RADIUS = 58.0
BACKDROP_ALPHA = 0.42
FADE_SECONDS = 0.15

SHOCKWAVE_SECONDS = 0.38
SHOCKWAVE_START = 12.0
SHOCKWAVE_END = 74.0
TRAIL_SECONDS = 0.30
TRAIL_POINTS = 24

BREATH_PERIOD = 3.0
BREATH_DEPTH = 0.08
TICK_COUNT = 12
TICK_FLICKER_EVERY = 0.12  # วินาที: สุ่ม tick ให้กระพริบ
TICK_FLICKER_FOR = 0.25

DEPTH_RANGE = (0.10, 0.22)  # hand_size (ขนาดมือ ÷ ความกว้างเฟรม): ไกล … ใกล้กล้อง
SCALE_RANGE = (0.85, 1.25)

# spring ของการแปลงร่าง: เร็วกว่า "snappy" เพื่อให้จบใน ≤ 250 ms (spec M7 ✅)
MORPH = (6.5, 0.9)

_DRAG_STATES = (State.DRAGGING, State.WINDOW_DRAG)
_DASHES = 14  # จำนวนช่วงของเส้นประ (ต่อรอบ)


@dataclass(frozen=True, slots=True)
class Ring:
    radius: float
    thickness: float
    span: float  # องศาต่อช่วง (360 = วงเต็ม)
    alpha: float
    dash: float = 0.0  # 0 = เส้นทึบ, 1 = เส้นประ (ค่ากลางผสมกัน)
    spin: float = 0.0  # องศาต่อวินาที


@dataclass(frozen=True, slots=True)
class Mode:
    outer: Ring  # วงนอก (มี tick)
    arcs: Ring  # ส่วนโค้ง 4 ช่อง
    inner: Ring  # วงใน (รางของ pinch fill)
    stretch: tuple[float, float] = (1.0, 1.0)  # ยืดแกน x, y (รางสกรอล)
    hot: float = 0.0  # 0 = สีหลัก, 1 = สีเน้น (accent)


MODES: dict[str, Mode] = {
    "POINTER": Mode(
        Ring(46, 1.0, 360, 0.45), Ring(37, 2.0, 62, 1.0, spin=40), Ring(23, 1.0, 360, 0.25)
    ),
    "DRAGGING": Mode(
        Ring(36, 1.2, 360, 0.55),
        Ring(29, 2.4, 80, 1.0, spin=170),
        Ring(18, 1.2, 360, 0.35),
        hot=1.0,
    ),
    "SCROLL": Mode(
        Ring(44, 1.0, 360, 0.35),
        Ring(40, 2.2, 28, 1.0, spin=0),
        Ring(16, 1.0, 360, 0.25),
        stretch=(0.55, 1.4),
    ),
    "ZOOM": Mode(
        Ring(44, 1.0, 360, 0.35),
        Ring(40, 2.2, 28, 1.0, spin=0),
        Ring(16, 1.0, 360, 0.25),
        stretch=(1.4, 0.55),
        hot=1.0,
    ),
    "GESTURE": Mode(
        Ring(58, 1.0, 360, 0.30), Ring(50, 2.2, 34, 1.0, spin=0), Ring(14, 1.0, 360, 0.2)
    ),
    "CHARGING": Mode(
        Ring(56, 1.0, 360, 0.0),  # วงนอกขยายไปเป็นวงชาร์จ (ChargeRing วาดส่วนที่เติม + ป้าย)
        Ring(38, 1.8, 50, 0.55, spin=20),
        Ring(22, 1.0, 360, 0.25),
        hot=1.0,
    ),
    "DISARMED": Mode(
        Ring(44, 1.0, 360, 0.30),
        Ring(37, 1.6, 360, 0.60, dash=1.0, spin=8),
        Ring(22, 1.0, 360, 0.15),
    ),
}


# minimal: วงแหวนบางวงเดียว (outer) — arcs ใช้แค่ GESTURE (ขีดสั้น 4 ทิศ), inner = รางของ pinch fill (มองไม่เห็น)
MINIMAL_MODES: dict[str, Mode] = {
    "POINTER": Mode(Ring(16, 1.3, 360, 0.9), Ring(16, 1.8, 18, 0.0), Ring(16, 2.0, 360, 0.0)),
    "DRAGGING": Mode(
        Ring(11, 1.6, 360, 1.0), Ring(11, 1.8, 18, 0.0), Ring(11, 2.0, 360, 0.0), hot=1.0
    ),
    "SCROLL": Mode(
        Ring(14, 1.3, 360, 0.85),
        Ring(14, 1.8, 18, 0.0),
        Ring(14, 2.0, 360, 0.0),
        stretch=(0.75, 1.3),
    ),
    "ZOOM": Mode(
        Ring(14, 1.3, 360, 0.85),
        Ring(14, 1.8, 18, 0.0),
        Ring(14, 2.0, 360, 0.0),
        stretch=(1.3, 0.75),
        hot=1.0,
    ),
    "GESTURE": Mode(Ring(22, 1.0, 360, 0.45), Ring(22, 1.8, 16, 0.95), Ring(22, 2.0, 360, 0.0)),
    "CHARGING": Mode(
        Ring(10, 1.2, 360, 0.6), Ring(10, 1.8, 18, 0.0), Ring(10, 2.0, 360, 0.0), hot=1.0
    ),
    "DISARMED": Mode(Ring(12, 1.2, 360, 0.6), Ring(12, 1.8, 18, 0.0), Ring(12, 2.0, 360, 0.0)),
}
MINIMAL_BACKDROP_RADIUS = 30.0  # เงานุ่มๆ ใต้ reticle ให้เห็นบนพื้นสว่าง
MINIMAL_BACKDROP_ALPHA = 0.10
MINIMAL_SHOCKWAVE = (10.0, 30.0)  # รัศมีเริ่ม → จบ ของวงคลิก
MINIMAL_CHEVRON = (30.0, 25.0, 5.0)  # ปลาย, โคน, ครึ่งความกว้าง ของลูกศรสกรอล/ซูม

# ครึ่งด้านของกรอบที่ต้องวาดใหม่: ใหญ่สุดของทุกโหมด (วงนอก + tick, ยืด, ระยะมือใกล้สุด, เด้งตอนคลิก) + glow
HALF_EXTENT = (
    max((m.outer.radius + 8.0) * max(m.stretch) for m in MODES.values())
    * SCALE_RANGE[1]
    * PULSE_OVERSHOOT
    + 8.0
)


def reticle_mode(snapshot: EngineSnapshot) -> str:
    if not snapshot.armed:
        return "DISARMED"
    state = snapshot.state
    if state in _DRAG_STATES:
        return "DRAGGING"
    if state is State.SCROLL:
        return "SCROLL"
    if state is State.TWO_HAND_ZOOM:
        return "ZOOM"
    if state in (State.GESTURE, State.KNOB):  # ปุ่มหมุน: หน้าปัด (KnobDial) วาดรอบ reticle ที่กางอยู่
        return "GESTURE"
    if state is State.CHARGING:
        return "CHARGING"
    return "POINTER"


def depth_scale(hand_size: float | None) -> float:
    """ขนาดมือในภาพ → ขนาดของ reticle (มือใกล้กล้อง = reticle ใหญ่) — ไม่ใช้ค่า z ของ landmark"""
    if hand_size is None:
        return 1.0
    near, far = DEPTH_RANGE[1], DEPTH_RANGE[0]
    t = min(max((hand_size - far) / (near - far), 0.0), 1.0)
    return SCALE_RANGE[0] + t * (SCALE_RANGE[1] - SCALE_RANGE[0])


def _vector(mode: Mode) -> tuple[float, ...]:
    return (
        *astuple(mode.outer),
        *astuple(mode.arcs),
        *astuple(mode.inner),
        *mode.stretch,
        mode.hot,
    )


@dataclass(slots=True)
class _Shockwave:
    center: QPointF
    started: float
    color: QColor


class Reticle:
    """เก็บสถานะและวาด reticle — ตำแหน่งทั้งหมดเป็น logical px ของ overlay"""

    def __init__(self, theme: Theme, *, reduce_motion: bool = False, seed: int = 3) -> None:
        self._theme = theme
        self._minimal = theme.minimal
        self._modes = MINIMAL_MODES if theme.minimal else MODES
        self._reduce_motion = reduce_motion
        self._pos: QPointF | None = None
        self._snapshot = EngineSnapshot()
        self._visible = False
        self._alpha = 0.0
        self._mode = "DISARMED"
        self._morph = SecondOrderDynamics(*MORPH, x0=_vector(self._modes["DISARMED"]))
        self._scale = spring("soft", 1.0)
        self._pulse = spring("bouncy", 1.0)  # หดแล้วเด้งกลับตอนคลิก
        self._fill = spring("snappy", 0.0)  # ระดับการจีบที่แสดง (นุ่มกว่าค่าดิบรายเฟรม)
        self._angles = [0.0, 0.0, 0.0]  # มุมหมุนของแต่ละชั้น
        self._scroll_dir = 0  # +1 ขึ้น, -1 ลง (ลูกศรที่สว่าง)
        self._scroll_at = -1e9
        self._last_time: float | None = None
        self._clock_now = 0.0
        self._shockwaves: list[_Shockwave] = []
        self._trail: deque[tuple[QPointF, float]] = deque(maxlen=TRAIL_POINTS)
        self._layers = LayerCache()
        self._rng = random.Random(seed)
        self._flicker: dict[int, float] = {}  # tick → เวลาที่เริ่มกระพริบ
        self._next_flicker = 0.0

    # ---- state ---------------------------------------------------------- #

    @property
    def position(self) -> QPointF | None:
        return self._pos

    @property
    def alpha(self) -> float:
        return self._alpha

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def scale(self) -> float:
        """ขนาดรวม: ตามระยะมือ (soft) × จังหวะคลิก (bouncy)"""
        return float(self._scale.value) * float(self._pulse.value)  # type: ignore[arg-type]

    def morph_settled(self, tolerance: float = 0.5) -> bool:
        return self._morph.settled(_vector(self._modes[self._mode]), tolerance)

    def set_state(self, position: QPointF | None, snapshot: EngineSnapshot, now: float) -> None:
        """อัปเดตจากเฟรมล่าสุด: ``position`` = ตำแหน่งเคอร์เซอร์ (logical px) หรือ ``None``"""
        self._snapshot = snapshot
        self._visible = position is not None and snapshot.hand_present
        self._mode = reticle_mode(snapshot)
        if position is not None:
            self._pos = position
            if snapshot.state in _DRAG_STATES:
                self._trail.append((QPointF(position), now))

    def handle_event(self, kind: EventKind, dy: int, now: float) -> None:
        """ลูกศรของรางสกรอลสว่างตามทิศของการสกรอลล่าสุด"""
        if kind is EventKind.SCROLL and dy:
            self._scroll_dir = 1 if dy > 0 else -1
            self._scroll_at = now

    def add_shockwave(self, center: QPointF, color: QColor, now: float) -> None:
        self._shockwaves.append(_Shockwave(QPointF(center), now, QColor(color)))
        self._pulse.reset(0.82)  # คลิก: reticle หดแล้วเด้งกลับ (preset bouncy)

    def advance(self, now: float) -> bool:
        """เดินแอนิเมชันถึงเวลา ``now`` คืน ``True`` ถ้ายังมีอะไรต้องวาดต่อ"""
        dt = 0.0 if self._last_time is None else max(0.0, min(now - self._last_time, 0.1))
        self._last_time = now
        self._clock_now = now

        target = 1.0 if self._visible else 0.0
        step = dt / FADE_SECONDS
        self._alpha = (
            min(target, self._alpha + step)
            if self._alpha < target
            else max(target, self._alpha - step)
        )
        if dt > 0:
            values = self._morph.update(dt, _vector(self._modes[self._mode]))
            self._scale.update(dt, depth_scale(self._snapshot.hand_size))
            self._pulse.update(dt, 1.0)
            dragging = self._snapshot.state in _DRAG_STATES
            self._fill.update(dt, 1.0 if dragging else self._snapshot.pinch_level)
            spins = (values[5], values[11], values[17])  # type: ignore[index]
            self._angles = [(a + s * dt) % 360.0 for a, s in zip(self._angles, spins)]
        if not (self._reduce_motion or self._minimal) and now >= self._next_flicker:
            self._next_flicker = now + TICK_FLICKER_EVERY
            if self._rng.random() < 0.5:
                self._flicker[self._rng.randrange(TICK_COUNT)] = now
        self._flicker = {k: t for k, t in self._flicker.items() if now - t < TICK_FLICKER_FOR}

        self._shockwaves = [s for s in self._shockwaves if now - s.started < SHOCKWAVE_SECONDS]
        while self._trail and now - self._trail[0][1] > TRAIL_SECONDS:
            self._trail.popleft()
        return self._alpha > 0.0 or bool(self._shockwaves) or bool(self._trail)

    def rect(self) -> QRectF:
        """กรอบที่ครอบทุกอย่างที่ reticle วาดอยู่ตอนนี้ (ว่าง = ไม่มีอะไรต้องวาด)"""
        rect = QRectF()
        if self._pos is not None and self._alpha > 0.0:
            rect = _square(self._pos, self._extent())
        for wave in self._shockwaves:
            rect = rect.united(_square(wave.center, SHOCKWAVE_END + 10.0))
        for point, _ in self._trail:
            rect = rect.united(_square(point, 8.0))
        return rect

    def _extent(self) -> float:
        """ครึ่งด้านของสิ่งที่วาดอยู่จริงตอนนี้ (เล็กกว่า HALF_EXTENT ซึ่งเป็นกรณีใหญ่สุดของทุกโหมด)"""
        values = self._morph.value
        stretch = max(values[18], values[19], 1.0)  # type: ignore[index]
        if self._minimal:
            rings = max(values[0] + 6.0, values[6] + 4.0, MINIMAL_BACKDROP_RADIUS)  # type: ignore[index]
            if self._mode in ("SCROLL", "ZOOM"):
                rings = max(rings, MINIMAL_CHEVRON[0] + 4.0)
            return min(rings * stretch * max(self.scale, 1.0) + 6.0, HALF_EXTENT)
        rings = max(values[0] + 8.0, values[6] + 4.0, 30.0)  # type: ignore[index]
        if self._mode in ("SCROLL", "ZOOM"):
            rings = max(rings, 62.0 + 4.0)  # ลูกศรของรางสกรอล/ซูม
        return min(rings * stretch * max(self.scale, 1.0) + 10.0, HALF_EXTENT)

    # ---- paint ---------------------------------------------------------- #

    def paint(self, painter: QPainter, now: float) -> None:
        self._paint_trail(painter, now)
        self._paint_shockwaves(painter, now)
        if self._pos is not None and self._alpha > 0.0:
            if self._minimal:
                self._paint_minimal(painter, self._pos, self._alpha)
            else:
                self._paint_reticle(painter, self._pos, self._alpha, now)

    def _paint_trail(self, painter: QPainter, now: float) -> None:
        points = list(self._trail)
        for (a, _), (b, t) in itertools.pairwise(points):
            alpha = max(0.0, 1.0 - (now - t) / TRAIL_SECONDS)
            if alpha <= 0.0:
                continue
            if self._minimal:  # เส้นบางเส้นเดียว
                pen = QPen(with_alpha(self._theme.accent, 0.6 * alpha), 1.5)
                pen.setCapStyle(Qt.PenCapStyle.RoundCap)
                painter.setPen(pen)
                painter.drawLine(a, b)
                continue
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
            eased = 1.0 - (1.0 - progress) ** 3
            if self._minimal:  # วงบางๆ ขยายออกเล็กน้อยแล้วจาง
                start, end = MINIMAL_SHOCKWAVE
                radius = start + (end - start) * eased
                path = QPainterPath()
                path.addEllipse(wave.center, radius, radius)
                self._theme.stroke(painter, path, wave.color, 1.2, alpha=(1.0 - progress) ** 2)
                continue
            radius = SHOCKWAVE_START + (SHOCKWAVE_END - SHOCKWAVE_START) * eased
            path = QPainterPath()
            path.addEllipse(wave.center, radius, radius)
            glow_stroke(painter, path, wave.color, 2.0, alpha=(1.0 - progress) ** 1.5)

    def _paint_reticle(self, painter: QPainter, center: QPointF, alpha: float, now: float) -> None:
        theme = self._theme
        snapshot = self._snapshot
        armed = snapshot.armed
        values = self._morph.value
        outer, arcs, inner = (Ring(*values[i : i + 6]) for i in (0, 6, 12))  # type: ignore[index, misc]
        stretch_x, stretch_y, hot = values[18], values[19], values[20]  # type: ignore[index]
        base = theme.state_color(armed)
        accent = theme.accent if armed else theme.disarmed
        arc_color = _mix(base, accent, min(max(hot, 0.0), 1.0))
        scale = self.scale
        if not self._reduce_motion:
            alpha *= 1.0 + BREATH_DEPTH * math.sin(math.tau * now / BREATH_PERIOD)
        alpha = min(alpha, 1.0)
        cx, cy = center.x(), center.y()

        # ฉากหลังเข้มโปร่งแสง (ชั้นที่ cache ไว้ — ไม่ขึ้นกับโหมด)
        opacity = painter.opacity()
        painter.setOpacity(opacity * min(self._alpha, 1.0))
        self._layers.draw(
            painter,
            "backdrop",
            QPointF(cx - BACKDROP_RADIUS, cy - BACKDROP_RADIUS),
            BACKDROP_RADIUS * 2.0,
            BACKDROP_RADIUS * 2.0,
            _render_backdrop,
        )
        painter.setOpacity(opacity)

        painter.save()
        painter.translate(cx, cy)
        painter.scale(scale * stretch_x, scale * stretch_y)
        glow = min(scale, 1.2)
        self._paint_ring(painter, outer, base, alpha, self._angles[0], segments=1, glow=0.0)
        self._paint_ticks(painter, outer.radius, base, alpha * min(outer.alpha * 2.0, 1.0), now)
        self._paint_ring(painter, arcs, arc_color, alpha, self._angles[1], segments=4, glow=glow)
        self._paint_ring(painter, inner, base, alpha, 0.0, segments=1, glow=0.0)
        painter.restore()

        if self._mode in ("SCROLL", "ZOOM"):
            self._paint_chevrons(painter, cx, cy, arc_color, alpha, vertical=self._mode == "SCROLL")

        # วงแหวนรอให้การจีบกลายเป็นการลาก
        r = scale
        if snapshot.state is State.PINCH_PENDING and snapshot.pinch_button == "left":
            sweep = 360.0 * snapshot.hold_progress
            if sweep > 0.0:
                glow_stroke(
                    painter, arc_path(cx, cy, 30.0 * r, 0.0, sweep), accent, 2.0, alpha=alpha
                )

        # วงใน: เติมตามระดับการจีบ (เต็มวง = จีบแล้ว)
        dragging = snapshot.state in _DRAG_STATES
        pinched = snapshot.pinch_level >= 1.0
        level = min(max(float(self._fill.value), 0.0), 1.0)  # type: ignore[arg-type]
        if level > 0.01 and armed:
            fill = arc_path(cx, cy, inner.radius * r, 0.0, 359.99 * level)
            glow_stroke(painter, fill, accent if (pinched or dragging) else base, 3.0, alpha=alpha)

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(with_alpha(accent if (pinched or dragging) else base, alpha)))
        radius = 4.0 if dragging else 2.5
        painter.drawEllipse(center, radius, radius)

    def _paint_minimal(self, painter: QPainter, center: QPointF, alpha: float) -> None:
        """วงแหวนบางวงเดียว + เส้นโค้งสีเน้นตามระดับการจีบ + จุดกลาง"""
        theme = self._theme
        snapshot = self._snapshot
        armed = snapshot.armed
        values = self._morph.value
        outer, arcs, inner = (Ring(*values[i : i + 6]) for i in (0, 6, 12))  # type: ignore[index, misc]
        stretch_x, stretch_y, hot = values[18], values[19], values[20]  # type: ignore[index]
        base = theme.state_color(armed)
        accent = theme.accent if armed else theme.disarmed
        ring_color = _mix(base, accent, min(max(hot, 0.0), 1.0))
        scale = self.scale
        cx, cy = center.x(), center.y()

        opacity = painter.opacity()
        painter.setOpacity(opacity * min(alpha, 1.0))
        self._layers.draw(
            painter,
            "backdrop-minimal",
            QPointF(cx - MINIMAL_BACKDROP_RADIUS, cy - MINIMAL_BACKDROP_RADIUS),
            MINIMAL_BACKDROP_RADIUS * 2.0,
            MINIMAL_BACKDROP_RADIUS * 2.0,
            _render_minimal_backdrop,
        )
        painter.setOpacity(opacity)

        painter.save()
        painter.translate(cx, cy)
        painter.scale(scale * stretch_x, scale * stretch_y)
        self._paint_ring(painter, outer, ring_color, alpha, 0.0, segments=1, glow=0.0, theme=theme)
        self._paint_ring(painter, arcs, ring_color, alpha, 0.0, segments=4, glow=0.0, theme=theme)
        painter.restore()

        if self._mode in ("SCROLL", "ZOOM"):
            self._paint_minimal_chevrons(painter, cx, cy, ring_color, alpha)

        # รอให้การจีบกลายเป็นการลาก: เส้นโค้งบางรอบนอก
        if snapshot.state is State.PINCH_PENDING and snapshot.pinch_button == "left":
            sweep = 360.0 * snapshot.hold_progress
            if sweep > 0.0:
                arc = arc_path(cx, cy, (outer.radius + 6.0) * scale, 0.0, sweep)
                theme.stroke(painter, arc, accent, 1.4, alpha=alpha)

        # ระดับการจีบ: เส้นโค้งสีเน้นบนวงแหวน (เต็มวง = จีบแล้ว)
        dragging = snapshot.state in _DRAG_STATES
        level = min(max(float(self._fill.value), 0.0), 1.0)  # type: ignore[arg-type]
        if level > 0.01 and armed and not dragging:
            fill = arc_path(cx, cy, inner.radius * scale, 0.0, 359.99 * level)
            theme.stroke(painter, fill, accent, 2.0, alpha=alpha)

        pinched = snapshot.pinch_level >= 1.0
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(with_alpha(accent if (pinched or dragging) else base, alpha)))
        radius = 3.0 if dragging else 1.8
        painter.drawEllipse(center, radius, radius)

    def _paint_minimal_chevrons(
        self, painter: QPainter, cx: float, cy: float, color: QColor, alpha: float
    ) -> None:
        recent = self._clock_now - self._scroll_at < 0.4
        vertical = self._mode == "SCROLL"
        tip_at, base_at, half = MINIMAL_CHEVRON
        for direction in (-1.0, 1.0):
            path = QPainterPath()
            tip, base = direction * tip_at * self.scale, direction * base_at * self.scale
            if vertical:
                path.moveTo(cx - half, cy + base)
                path.lineTo(cx, cy + tip)
                path.lineTo(cx + half, cy + base)
                lit = recent and self._scroll_dir == (1 if direction < 0 else -1)
            else:
                path.moveTo(cx + base, cy - half)
                path.lineTo(cx + tip, cy)
                path.lineTo(cx + base, cy + half)
                lit = False
            self._theme.stroke(
                painter,
                path,
                self._theme.accent if lit else color,
                1.4,
                alpha=alpha * (1.0 if lit else 0.55),
            )

    @staticmethod
    def _paint_ring(
        painter: QPainter,
        ring: Ring,
        color: QColor,
        alpha: float,
        angle: float,
        *,
        segments: int,
        glow: float,
        theme: Theme | None = None,
    ) -> None:
        """ชั้นหนึ่ง (พิกัดรอบจุดกำเนิด) — ``segments`` ช่องเท่าๆ กัน, เส้นประเมื่อ ``dash`` > 0.5

        ``theme`` = วาดด้วยเส้นตามสไตล์ของ theme (minimal) แทนเส้นเรืองแสง
        """
        ring_alpha = alpha * max(ring.alpha, 0.0)
        if ring_alpha <= 0.01 or ring.radius <= 1.0 or ring.span <= 0.5:
            return
        path = QPainterPath()
        span = min(ring.span, 360.0)
        if ring.dash > 0.5 and span >= 359.0:  # วงเส้นประ
            step = 360.0 / _DASHES
            for k in range(_DASHES):
                path.addPath(arc_path(0.0, 0.0, ring.radius, angle + k * step, step * 0.55))
        else:
            gap = 360.0 / segments
            for k in range(segments):
                start = angle + k * gap - span / 2.0 if segments > 1 else angle
                path.addPath(
                    arc_path(0.0, 0.0, ring.radius, start, span if segments > 1 else 359.99)
                )
        if theme is not None:
            theme.stroke(painter, path, color, max(ring.thickness, 0.5), alpha=ring_alpha)
            return
        glow_stroke(painter, path, color, max(ring.thickness, 0.5), strength=glow, alpha=ring_alpha)

    def _paint_ticks(
        self, painter: QPainter, radius: float, color: QColor, alpha: float, now: float
    ) -> None:
        if alpha <= 0.01:
            return
        for step in range(TICK_COUNT):
            flicker = self._flicker.get(step)
            tick_alpha = alpha * 0.8
            if flicker is not None:
                tick_alpha *= 0.25 + 0.75 * abs(math.cos((now - flicker) * math.tau * 4))
            painter.setPen(QPen(with_alpha(color, tick_alpha), 1.2))
            angle = math.radians(step * 30.0)
            length = 8.0 if step % 3 == 0 else 4.0
            dx, dy = math.sin(angle), -math.cos(angle)
            painter.drawLine(
                QPointF(dx * radius, dy * radius),
                QPointF(dx * (radius + length), dy * (radius + length)),
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
        """ลูกศรคู่: แนวตั้ง = รางสกรอล (ทิศที่สกรอลล่าสุดสว่าง), แนวนอน = ซูมสองมือ"""
        recent = self._clock_now - self._scroll_at < 0.4
        for direction in (-1.0, 1.0):
            path = QPainterPath()
            tip, base = direction * 62.0 * self.scale, direction * 54.0 * self.scale
            if vertical:
                path.moveTo(cx - 7.0, cy + base)
                path.lineTo(cx, cy + tip)
                path.lineTo(cx + 7.0, cy + base)
                lit = recent and self._scroll_dir == (1 if direction < 0 else -1)  # ขึ้น = ลูกศรบน
            else:
                path.moveTo(cx + base, cy - 7.0)
                path.lineTo(cx + tip, cy)
                path.lineTo(cx + base, cy + 7.0)
                lit = False
            glow_stroke(
                painter,
                path,
                self._theme.accent if lit else color,
                2.2 if lit else 1.6,
                strength=1.2 if lit else 0.5,
                alpha=alpha * (1.0 if lit else 0.6),
            )


def _render_backdrop(painter: QPainter) -> None:
    center = QPointF(BACKDROP_RADIUS, BACKDROP_RADIUS)
    backdrop = QRadialGradient(center, BACKDROP_RADIUS)
    backdrop.setColorAt(0.0, QColor(2, 8, 14, round(255 * BACKDROP_ALPHA)))
    backdrop.setColorAt(0.75, QColor(2, 8, 14, round(255 * BACKDROP_ALPHA * 0.8)))
    backdrop.setColorAt(1.0, QColor(2, 8, 14, 0))
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QBrush(backdrop))
    painter.drawEllipse(center, BACKDROP_RADIUS, BACKDROP_RADIUS)


def _render_minimal_backdrop(painter: QPainter) -> None:
    radius = MINIMAL_BACKDROP_RADIUS
    center = QPointF(radius, radius)
    shadow = QRadialGradient(center, radius)
    shadow.setColorAt(0.0, QColor(6, 8, 11, round(255 * MINIMAL_BACKDROP_ALPHA)))
    shadow.setColorAt(0.6, QColor(6, 8, 11, round(255 * MINIMAL_BACKDROP_ALPHA * 0.6)))
    shadow.setColorAt(1.0, QColor(6, 8, 11, 0))
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QBrush(shadow))
    painter.drawEllipse(center, radius, radius)


def _mix(a: QColor, b: QColor, t: float) -> QColor:
    return QColor.fromRgbF(
        a.redF() + (b.redF() - a.redF()) * t,
        a.greenF() + (b.greenF() - a.greenF()) * t,
        a.blueF() + (b.blueF() - a.blueF()) * t,
        a.alphaF() + (b.alphaF() - a.alphaF()) * t,
    )


def _square(center: QPointF, half: float) -> QRectF:
    return QRectF(center.x() - half, center.y() - half, half * 2.0, half * 2.0)
