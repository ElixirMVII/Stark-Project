"""ชิ้นส่วน HUD ของท่าพิเศษ: เข็มทิศ 4 ทิศ (โหมด GESTURE) และวงแหวนชาร์จ (ค้างท่า ✊ / 🤟)

ทั้งสองชิ้นวาดรอบตำแหน่ง reticle (ซึ่งหยุดนิ่งระหว่างโหมดเหล่านี้)
"""

from __future__ import annotations

import math
from collections.abc import Mapping

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QPainter, QPainterPath

from ..gestures import (
    ACTION_KNOB_LEFT,
    ACTION_KNOB_RIGHT,
    CHARGE_ARM,
    CHARGE_FIST,
    CHARGE_KNOB,
    CHARGE_LAUNCH,
    EngineSnapshot,
    State,
)
from ..hotkeys import Hotkey, parse_action
from ..poses import POSE_PREFIX
from .theme import Theme, arc_path, draw_label, glow_stroke, with_alpha

__all__ = ["DIRECTIONS", "ChargeRing", "KnobDial", "Radial", "action_label", "knob_label"]

DIRECTIONS = ("left", "right", "up", "down")

FADE_SECONDS = 0.15
FLASH_SECONDS = 0.5  # ทิศที่ปัดสว่างค้างนานเท่านี้

RADIAL_INNER = 74.0  # ระยะจากศูนย์กลางถึงหัวลูกศร
RADIAL_HALF_WIDTH = 190.0  # ครึ่งความกว้างของกรอบ (ป้ายซ้าย/ขวายื่นออกไป)
RADIAL_HALF_HEIGHT = 128.0

CHARGE_RADIUS = 56.0
CHARGE_HALF_WIDTH = 84.0
CHARGE_TOP = 66.0
CHARGE_BOTTOM = 92.0  # มีป้ายใต้วงแหวน

MINIMAL_RADIAL = 46.0  # minimal: ระยะถึงหัวลูกศร (ชิด reticle กว่า)
MINIMAL_CHARGE_RADIUS = 30.0

# ป้ายของ action ค่าเริ่มต้น (ตามลำดับการกดปุ่ม) — action อื่นแสดงเป็นชื่อปุ่มลัด
_KNOWN_LABELS = {
    ("ctrl", "win", "left"): "Desk",
    ("ctrl", "win", "right"): "Desk",
    ("win", "tab"): "Tasks",
    ("win", "d"): "Desktop",
    ("shift", "win", "s"): "Snip",
    ("media_play_pause",): "Play/Pause",
    ("media_next",): "Next track",
    ("media_prev",): "Previous track",
    ("media_stop",): "Stop",
    ("volume_up",): "Volume up",
    ("volume_down",): "Volume down",
    ("volume_mute",): "Mute",
}


def _key_label(name: str) -> str:
    return name.upper() if len(name) == 1 else name.replace("_", " ").capitalize()


def action_label(spec: str) -> str:
    """ป้ายสั้นๆ ของ action ใน config สำหรับแสดงบน HUD (ตัวพิมพ์ปกติ — holo แปลงเป็นตัวใหญ่ตอนวาด)"""
    try:
        action = parse_action(spec)
    except ValueError:
        return "?"
    if action == "none":
        return "—"
    if action == "minimize_foreground":
        return "Minimize"
    if isinstance(action, Hotkey):
        known = _KNOWN_LABELS.get(action.keys)
        return known or "+".join(_key_label(key) for key in action.keys)
    return str(action).replace("_", " ").capitalize()


def _fade(alpha: float, visible: bool, dt: float) -> float:
    step = dt / FADE_SECONDS
    return min(1.0, alpha + step) if visible else max(0.0, alpha - step)


class Radial:
    """เข็มทิศ 4 ทิศรอบ reticle ระหว่างโหมด GESTURE — ทิศที่ปัดจะสว่างขึ้นตอนเกิด event"""

    def __init__(self, theme: Theme, actions: Mapping[str, str]) -> None:
        self._theme = theme
        self._labels = {d: action_label(actions.get(f"swipe_{d}", "none")) for d in DIRECTIONS}
        self._center: QPointF | None = None
        self._visible = False
        self._ready = False
        self._armed = False
        self._alpha = 0.0
        self._flash: dict[str, float] = {}  # ทิศ → เวลาที่เริ่มสว่าง
        self._last_time: float | None = None

    @property
    def labels(self) -> dict[str, str]:
        return dict(self._labels)

    @property
    def alpha(self) -> float:
        return self._alpha

    def set_state(self, center: QPointF | None, snapshot: EngineSnapshot) -> None:
        self._visible = center is not None and snapshot.state is State.GESTURE
        self._ready = snapshot.swipe_ready
        self._armed = snapshot.armed
        if center is not None and self._visible:
            self._center = QPointF(center)

    def flash(self, direction: str, now: float) -> None:
        """ทำให้ทิศ ``direction`` สว่าง (เรียกเมื่อเกิด event ปัด)"""
        if direction in DIRECTIONS:
            self._flash[direction] = now

    def advance(self, now: float) -> bool:
        dt = 0.0 if self._last_time is None else max(0.0, min(now - self._last_time, 0.1))
        self._last_time = now
        flashing = bool(self._flash)
        self._flash = {d: t for d, t in self._flash.items() if now - t < FLASH_SECONDS}
        # ทิศที่เพิ่งปัดยังต้องเห็นจนสว่างครบ แม้ engine จะออกจากโหมดไปแล้ว
        self._alpha = _fade(self._alpha, self._visible or bool(self._flash), dt)
        return self._alpha > 0.0 or flashing

    def rect(self) -> QRectF:
        if self._center is None or self._alpha <= 0.0:
            return QRectF()
        c = self._center
        return QRectF(
            c.x() - RADIAL_HALF_WIDTH,
            c.y() - RADIAL_HALF_HEIGHT,
            RADIAL_HALF_WIDTH * 2.0,
            RADIAL_HALF_HEIGHT * 2.0,
        )

    def paint(self, painter: QPainter, now: float) -> None:
        if self._center is None or self._alpha <= 0.0:
            return
        theme = self._theme
        cx, cy = self._center.x(), self._center.y()
        base = theme.state_color(self._armed)
        idle_alpha = self._alpha * (0.95 if self._ready else 0.4)  # ยังไม่พร้อมปัด = จาง
        if theme.minimal:
            self._paint_minimal(painter, now, cx, cy, base, idle_alpha)
            return

        # วงแหวนประบางๆ บอกขอบเขตของเข็มทิศ
        ring = QPainterPath()
        for k in range(4):
            ring.addPath(arc_path(cx, cy, RADIAL_INNER - 10.0, 45.0 + k * 90.0 + 12.0, 66.0))
        glow_stroke(painter, ring, base, 1.2, strength=0.5, alpha=idle_alpha * 0.7)

        painter.setFont(theme.display_font(12, 700, spacing=2.0))
        for direction in DIRECTIONS:
            started = self._flash.get(direction)
            flash = 0.0 if started is None else max(0.0, 1.0 - (now - started) / FLASH_SECONDS)
            color = theme.accent if flash > 0.0 else base
            alpha = max(idle_alpha, self._alpha * flash) if flash > 0.0 else idle_alpha
            self._paint_direction(painter, direction, cx, cy, color, alpha, 1.0 + 0.6 * flash)

    def _paint_direction(
        self,
        painter: QPainter,
        direction: str,
        cx: float,
        cy: float,
        color,
        alpha: float,
        strength: float,
    ) -> None:
        dx, dy = {"left": (-1, 0), "right": (1, 0), "up": (0, -1), "down": (0, 1)}[direction]
        tip = QPointF(cx + dx * (RADIAL_INNER + 14.0), cy + dy * (RADIAL_INNER + 14.0))
        # ลูกศรสองชั้นชี้ออกจากศูนย์กลาง
        chevrons = QPainterPath()
        for offset in (0.0, 9.0):
            head = QPointF(tip.x() - dx * offset, tip.y() - dy * offset)
            back_x, back_y = head.x() - dx * 9.0, head.y() - dy * 9.0
            chevrons.moveTo(back_x - dy * 9.0, back_y - dx * 9.0)
            chevrons.lineTo(head)
            chevrons.lineTo(back_x + dy * 9.0, back_y + dx * 9.0)
        glow_stroke(painter, chevrons, color, 2.0, strength=strength, alpha=alpha)

        label = self._theme.label(self._labels[direction])
        if dx:
            box = QRectF(tip.x() + (8.0 if dx > 0 else -100.0), tip.y() - 13.0, 92.0, 26.0)
            horizontal = Qt.AlignmentFlag.AlignLeft if dx > 0 else Qt.AlignmentFlag.AlignRight
            flags = horizontal | Qt.AlignmentFlag.AlignVCenter
        else:
            box = QRectF(tip.x() - 84.0, tip.y() + (6.0 if dy > 0 else -32.0), 168.0, 26.0)
            flags = Qt.AlignmentFlag.AlignCenter
        draw_label(painter, box, flags, label, color, alpha)

    def _paint_minimal(
        self, painter: QPainter, now: float, cx: float, cy: float, base, idle_alpha: float
    ) -> None:
        """ลูกศรบางทิศละอัน + ป้ายแคปซูล — ทิศที่ปัดเปลี่ยนเป็นสีเน้นแล้วค่อยๆ จาง"""
        theme = self._theme
        painter.setFont(theme.text_font(12, 500))
        for direction in DIRECTIONS:
            started = self._flash.get(direction)
            flash = 0.0 if started is None else max(0.0, 1.0 - (now - started) / FLASH_SECONDS)
            color = theme.accent if flash > 0.0 else base
            alpha = max(idle_alpha, self._alpha * flash)
            dx, dy = {"left": (-1, 0), "right": (1, 0), "up": (0, -1), "down": (0, 1)}[direction]
            tip = QPointF(cx + dx * MINIMAL_RADIAL, cy + dy * MINIMAL_RADIAL)
            back_x, back_y = tip.x() - dx * 6.0, tip.y() - dy * 6.0
            chevron = QPainterPath()
            chevron.moveTo(back_x - dy * 6.0, back_y - dx * 6.0)
            chevron.lineTo(tip)
            chevron.lineTo(back_x + dy * 6.0, back_y + dx * 6.0)
            theme.stroke(painter, chevron, color, 1.5, alpha=alpha)
            if dx:
                box = QRectF(tip.x() + (8.0 if dx > 0 else -108.0), tip.y() - 13.0, 100.0, 26.0)
                horizontal = Qt.AlignmentFlag.AlignLeft if dx > 0 else Qt.AlignmentFlag.AlignRight
                flags = horizontal | Qt.AlignmentFlag.AlignVCenter
            else:
                box = QRectF(tip.x() - 84.0, tip.y() + (6.0 if dy > 0 else -32.0), 168.0, 26.0)
                flags = Qt.AlignmentFlag.AlignCenter
            draw_label(painter, box, flags, self._labels[direction], color, alpha, pill=True)


def knob_label(actions: Mapping[str, str]) -> str:
    """ชื่อของปุ่มหมุน: ผูกกับปุ่มเสียงอยู่ = "Volume" ไม่งั้นบอกปุ่มลัดของทั้งสองทาง"""
    right, left = actions.get(ACTION_KNOB_RIGHT, "none"), actions.get(ACTION_KNOB_LEFT, "none")
    if (right.strip().lower(), left.strip().lower()) == ("volume_up", "volume_down"):
        return "Volume"
    return f"{action_label(left)} / {action_label(right)}"


class ChargeRing:
    """วงแหวนที่ค่อยๆ เติมเต็มระหว่างค้างท่า ✊ / 🤟 / 👍 / 👎 / ☝️ / ท่าที่สอนเอง + ป้ายบอกผล"""

    def __init__(self, theme: Theme, actions: Mapping[str, str]) -> None:
        self._theme = theme
        # ชนิดของการนับที่ตรงกับชื่อ action (fist_hold, thumb_down_hold) → ป้ายของปุ่มลัดที่ผูกไว้
        self._labels = {kind: action_label(spec) for kind, spec in actions.items()}
        self._labels[CHARGE_KNOB] = knob_label(actions)
        self._fist_label = self._labels.get(CHARGE_FIST, "—")
        self._center: QPointF | None = None
        self._charge = 0.0
        self._kind: str | None = None
        self._armed = False

    @property
    def charge(self) -> float:
        return self._charge

    def label(self) -> str:
        """ป้ายใต้วงแหวน (ตามสไตล์: holo = ตัวพิมพ์ใหญ่)"""
        if self._kind == CHARGE_ARM:
            text = "Disarm" if self._armed else "Arm"
        elif self._kind == CHARGE_LAUNCH:
            text = "Apps"
        elif self._kind is not None and self._kind.startswith(POSE_PREFIX):
            text = self._kind.removeprefix(POSE_PREFIX)  # ชื่อท่าที่สอนเอง
        elif self._kind in self._labels:
            text = self._labels[self._kind]
        else:
            text = self._fist_label
        return self._theme.label(text)

    def set_state(self, center: QPointF | None, snapshot: EngineSnapshot) -> bool:
        """คืน ``True`` ถ้าสิ่งที่แสดงเปลี่ยน"""
        charge = snapshot.charge if center is not None and snapshot.charge_kind else 0.0
        changed = charge != self._charge or (charge > 0.0 and center != self._center)
        self._charge = charge
        self._kind = snapshot.charge_kind
        self._armed = snapshot.armed
        if center is not None:
            self._center = QPointF(center)
        return changed

    def advance(self, now: float) -> bool:
        return False  # ไม่มีแอนิเมชันของตัวเอง: เปลี่ยนตามเฟรมของกล้องเท่านั้น

    def rect(self) -> QRectF:
        if self._center is None or self._charge <= 0.0:
            return QRectF()
        c = self._center
        return QRectF(
            c.x() - CHARGE_HALF_WIDTH,
            c.y() - CHARGE_TOP,
            CHARGE_HALF_WIDTH * 2.0,
            CHARGE_TOP + CHARGE_BOTTOM,
        )

    def paint(self, painter: QPainter, now: float) -> None:
        if self._center is None or self._charge <= 0.0:
            return
        theme = self._theme
        cx, cy = self._center.x(), self._center.y()
        if theme.minimal:  # วงบางชิด reticle: รางจางๆ + เส้นสีเน้นที่เติมจนเต็มวง
            radius = MINIMAL_CHARGE_RADIUS
            track = QPainterPath()
            track.addEllipse(self._center, radius, radius)
            theme.stroke(painter, track, theme.accent, 1.0, alpha=0.28)
            fill = arc_path(cx, cy, radius, 0.0, 359.99 * min(self._charge, 1.0))
            theme.stroke(painter, fill, theme.accent, 2.2)
            painter.setFont(theme.text_font(12, 600))
            box = QRectF(cx - CHARGE_HALF_WIDTH, cy + radius + 8.0, CHARGE_HALF_WIDTH * 2.0, 26.0)
            draw_label(
                painter,
                box,
                Qt.AlignmentFlag.AlignCenter,
                self.label(),
                theme.accent,
                1.0,
                pill=True,
            )
            return
        track = QPainterPath()
        track.addEllipse(self._center, CHARGE_RADIUS, CHARGE_RADIUS)
        glow_stroke(painter, track, theme.accent, 1.0, strength=0.0, alpha=0.3)
        fill = arc_path(cx, cy, CHARGE_RADIUS, 0.0, 359.99 * min(self._charge, 1.0))
        glow_stroke(painter, fill, theme.accent, 3.5)

        painter.setFont(theme.display_font(12, 700, spacing=2.0))
        box = QRectF(
            cx - CHARGE_HALF_WIDTH, cy + CHARGE_RADIUS + 8.0, CHARGE_HALF_WIDTH * 2.0, 26.0
        )
        draw_label(painter, box, Qt.AlignmentFlag.AlignCenter, self.label(), theme.accent, 1.0)


KNOB_RADIUS = 40.0  # minimal (holo ใช้ KNOB_RADIUS × 1.5)
KNOB_SWEEP = 270.0  # หน้าปัดกว้างเท่านี้ (องศา) เริ่มที่ −135°
KNOB_HALF = 120.0
KNOB_FLASH_SECONDS = 0.25


class KnobDial:
    """หน้าปัดรอบ reticle ระหว่าง state ``KNOB``: เข็มชี้มุมที่หมุนไป + ขีดทุก step + ป้าย (เช่น Volume)

    เข็มหมุนตามมือจริง (มุมของมือ × 1) แต่วนอยู่ในหน้าปัด ส่วนขีดสว่างวาบทุกครั้งที่นับเป็นหนึ่ง step
    """

    def __init__(self, theme: Theme, actions: Mapping[str, str], step_deg: float = 18.0) -> None:
        self._theme = theme
        self._label = knob_label(actions)
        self._step = step_deg
        self._center: QPointF | None = None
        self._angle: float | None = None
        self._alpha = 0.0
        self._flash: tuple[int, float] | None = None  # (+1/−1, เวลา)
        self._last_time: float | None = None

    @property
    def label(self) -> str:
        return self._theme.label(self._label)

    @property
    def alpha(self) -> float:
        return self._alpha

    def set_state(self, center: QPointF | None, snapshot: EngineSnapshot) -> bool:
        angle = snapshot.knob if center is not None and snapshot.state is State.KNOB else None
        changed = angle != self._angle
        self._angle = angle
        if center is not None and angle is not None:
            self._center = QPointF(center)
        return changed

    def step(self, direction: int, now: float) -> None:
        self._flash = (direction, now)

    def advance(self, now: float) -> bool:
        dt = 0.0 if self._last_time is None else max(0.0, min(now - self._last_time, 0.1))
        self._last_time = now
        self._alpha = _fade(self._alpha, self._angle is not None, dt)
        if self._flash is not None and now - self._flash[1] > KNOB_FLASH_SECONDS:
            self._flash = None
        return self._alpha > 0.0

    def _radius(self) -> float:
        return KNOB_RADIUS if self._theme.minimal else KNOB_RADIUS * 1.5

    def rect(self) -> QRectF:
        if self._center is None or self._alpha <= 0.0:
            return QRectF()
        c = self._center
        half = self._radius() + 14.0
        return QRectF(
            c.x() - KNOB_HALF, c.y() - half, KNOB_HALF * 2.0, half + self._radius() + 44.0
        )

    def paint(self, painter: QPainter, now: float) -> None:
        if self._center is None or self._alpha <= 0.0:
            return
        theme = self._theme
        cx, cy = self._center.x(), self._center.y()
        radius = self._radius()
        alpha = self._alpha
        start = -KNOB_SWEEP / 2.0
        track = arc_path(cx, cy, radius, start, KNOB_SWEEP)
        theme.stroke(painter, track, theme.primary, 1.0, strength=0.4, alpha=0.35 * alpha)
        # ขีดทุก step
        ticks = QPainterPath()
        count = int(KNOB_SWEEP / self._step)
        for i in range(count + 1):
            angle = math.radians(start + i * self._step)
            dx, dy = math.sin(angle), -math.cos(angle)
            ticks.moveTo(cx + dx * (radius + 4.0), cy + dy * (radius + 4.0))
            ticks.lineTo(cx + dx * (radius + 8.0), cy + dy * (radius + 8.0))
        theme.stroke(painter, ticks, theme.primary, 1.0, strength=0.3, alpha=0.4 * alpha)
        # เข็ม: มุมที่หมุนไป (วนอยู่ในหน้าปัด)
        turned = self._angle or 0.0
        needle = max(-KNOB_SWEEP / 2.0, min(KNOB_SWEEP / 2.0, turned))
        sweep = arc_path(cx, cy, radius, 0.0, needle) if abs(needle) > 0.5 else None
        if sweep is not None:
            theme.stroke(painter, sweep, theme.accent, 2.2, alpha=alpha)
        angle = math.radians(needle)
        tip = QPointF(cx + math.sin(angle) * radius, cy - math.cos(angle) * radius)
        flash = 0.0
        if self._flash is not None:
            flash = max(0.0, 1.0 - (now - self._flash[1]) / KNOB_FLASH_SECONDS)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(with_alpha(theme.accent, alpha))
        painter.drawEllipse(tip, 3.5 + 2.5 * flash, 3.5 + 2.5 * flash)
        painter.setFont(
            theme.text_font(12, 600) if theme.minimal else theme.display_font(12, 700, 2.0)
        )
        box = QRectF(cx - KNOB_HALF, cy + radius + 12.0, KNOB_HALF * 2.0, 26.0)
        sign = "+" if self._flash and self._flash[0] > 0 else ("−" if self._flash else "")
        text = f"{self.label} {sign}".strip()
        draw_label(
            painter,
            box,
            Qt.AlignmentFlag.AlignCenter,
            text,
            theme.accent,
            alpha,
            pill=theme.minimal,
        )
