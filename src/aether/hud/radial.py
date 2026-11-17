"""ชิ้นส่วน HUD ของท่าพิเศษ: เข็มทิศ 4 ทิศ (โหมด GESTURE) และวงแหวนชาร์จ (ค้างท่า ✊ / 🤟)

ทั้งสองชิ้นวาดรอบตำแหน่ง reticle (ซึ่งหยุดนิ่งระหว่างโหมดเหล่านี้)
"""

from __future__ import annotations

from collections.abc import Mapping

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QPainter, QPainterPath

from ..gestures import CHARGE_ARM, CHARGE_FIST, EngineSnapshot, State
from ..hotkeys import Hotkey, parse_action
from .theme import Theme, arc_path, draw_label, glow_stroke

__all__ = ["DIRECTIONS", "ChargeRing", "Radial", "action_label"]

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

# ป้ายของ action ค่าเริ่มต้น (ตามลำดับการกดปุ่ม) — action อื่นแสดงเป็นชื่อปุ่มลัด
_KNOWN_LABELS = {
    ("ctrl", "win", "left"): "DESK",
    ("ctrl", "win", "right"): "DESK",
    ("win", "tab"): "TASKS",
    ("win", "d"): "DESKTOP",
}


def action_label(spec: str) -> str:
    """ป้ายสั้นๆ ของ action ใน config สำหรับแสดงบน HUD"""
    try:
        action = parse_action(spec)
    except ValueError:
        return "?"
    if action == "none":
        return "—"
    if action == "minimize_foreground":
        return "MINIMIZE"
    if isinstance(action, Hotkey):
        return _KNOWN_LABELS.get(action.keys, str(action).upper())
    return str(action).upper()


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

        label = self._labels[direction]
        if dx:
            box = QRectF(tip.x() + (8.0 if dx > 0 else -100.0), tip.y() - 13.0, 92.0, 26.0)
            horizontal = Qt.AlignmentFlag.AlignLeft if dx > 0 else Qt.AlignmentFlag.AlignRight
            flags = horizontal | Qt.AlignmentFlag.AlignVCenter
        else:
            box = QRectF(tip.x() - 84.0, tip.y() + (6.0 if dy > 0 else -32.0), 168.0, 26.0)
            flags = Qt.AlignmentFlag.AlignCenter
        draw_label(painter, box, flags, label, color, alpha)


class ChargeRing:
    """วงแหวนที่ค่อยๆ เติมเต็มระหว่างค้างท่า ✊ (ย่อหน้าต่าง) หรือ 🤟 (สลับ ARM) + ป้ายบอกผล"""

    def __init__(self, theme: Theme, actions: Mapping[str, str]) -> None:
        self._theme = theme
        self._fist_label = action_label(actions.get(CHARGE_FIST, "none"))
        self._center: QPointF | None = None
        self._charge = 0.0
        self._kind: str | None = None
        self._armed = False

    @property
    def charge(self) -> float:
        return self._charge

    def label(self) -> str:
        if self._kind == CHARGE_ARM:
            return "DISARM" if self._armed else "ARM"
        return self._fist_label

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
