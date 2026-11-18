"""หน้า Experiments ของ Studio — เครื่องมือทดลองที่ใช้กับสารจากหน้า Elements

- **Molecule** โมเลกุล 3 มิติ (จีบลาก = หมุน) รูปร่าง VSEPR มุมพันธะ ชนิดพันธะ · ปฏิกิริยาดูได้ทั้งก่อน/หลัง
- **Matter** สถานะของสสารตามอุณหภูมิ (จีบแล้วยกมือขึ้น = ร้อนขึ้น) จากจุดหลอมเหลว/จุดเดือดจริง
- **Air** องค์ประกอบของอากาศ + ลูกโป่งที่เติมสารนี้ลอยหรือจม (เทียบมวลโมเลกุลกับอากาศ)
- **Gravity** น้ำหนักของก้อนสาร 1 ลิตรบนดาวต่างๆ + ปล่อยให้ตก (จีบยกก้อนแล้วปล่อย)

สาร (subject) มาจากหน้า Elements: ธาตุที่เลือก, สารที่เพิ่งผสมได้, หรือของในขวดที่ยังไม่ตรงสูตร — และกด
"Mix in flask" เพื่อกลับไปผสมสารนั้นในขวดได้
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QBrush, QColor, QPainter, QPen, QRadialGradient
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QPushButton, QWidget

from ..hud.theme import with_alpha
from .chemistry import ATOM_COLORS, CATEGORIES, by_symbol, formula_text
from .input import SwipeDetector
from .matter import MatterSim, slider_from_temperature, temperature_from_slider
from .molecule import Structure, structure_for
from .pages import ACCENT, DIM, PRIMARY, StudioPage, _font
from .science import (
    AIR,
    AIR_MOLAR_MASS,
    EARTH_G,
    PLANETS,
    Subject,
    balloon_for,
    cube_mass_kg,
    diffusion_speed,
    fall_time,
    kelvin,
    phase_at,
    phase_data,
    room_state,
)

__all__ = ["QUICK_PICKS", "TOOLS", "ExperimentsPage", "experiments_controls"]

TOOLS = (
    ("molecule", "Molecule · โมเลกุล"),
    ("matter", "Matter · สถานะสสาร"),
    ("air", "Air · อากาศ"),
    ("gravity", "Gravity · แรงโน้มถ่วง"),
)
QUICK_PICKS = ("H2O", "CO2", "CH4", "NaCl", "He", "Fe", "C", "NH3")
PAD = 14.0
ROW = 30.0
VIEW_SHARE = 0.63
DROP_HEIGHT_M = 2.0
PHASE_TEXT = {
    "solid": "ของแข็ง · solid",
    "liquid": "ของเหลว · liquid",
    "gas": "แก๊ส · gas",
    "unknown": "ไม่ทราบ",
}
KIND_TEXT = {
    "molecule": "โมเลกุล (พันธะโคเวเลนต์)",
    "ionic": "สารประกอบไอออนิก",
    "metallic": "โลหะ (พันธะโลหะ)",
    "network": "ผลึกร่างตาข่าย",
    "atomic": "อะตอมเดี่ยว",
    "reaction": "ปฏิกิริยาเคมี",
}
Zone = tuple[QRectF, Callable[[], None]]


def _atom_color(symbol: str) -> QColor:
    if symbol in ATOM_COLORS:
        return QColor(ATOM_COLORS[symbol])
    return QColor(CATEGORIES[by_symbol(symbol).category][2])


def _charge_text(charge: int) -> str:
    if not charge:
        return ""
    sign = "+" if charge > 0 else "−"
    return sign if abs(charge) == 1 else f"{abs(charge)}{sign}"


class ExperimentsPage(StudioPage):
    title = "Experiments"
    hint = (
        "Pick a tool and a substance (from Elements or the chips)  ·  pinch + drag: rotate molecule / "
        "raise to heat / lift the balloon or block  ·  open palm + flick = next tool"
    )

    def __init__(
        self,
        parent=None,
        *,
        clock: Callable[[], float] = time.perf_counter,
        sources: Callable[[], list[Subject]] | None = None,
    ) -> None:
        super().__init__(parent, clock=clock)
        self.sources = sources or list
        self.on_mix: Callable[[Subject], None] | None = None
        self.tool = "molecule"
        self.subject = Subject.from_formula("H2O", "Water", "น้ำ")
        self.recent: list[Subject] = []
        self.swipe = SwipeDetector()
        self._zones: list[Zone] = []
        self._pinched: dict[str, bool] = {}
        self._drag: tuple[str, str, float, float] | None = None  # (มือ, อะไร, x, y เริ่ม)
        self._last_tick = clock()
        # molecule
        self._structures: dict[tuple[str, int], Structure] = {}
        self.side = 0
        self.yaw, self.pitch = 0.6, -0.35
        # matter
        self.matter = MatterSim()
        self.celsius = 25.0
        # air
        self.balloon_y = 0.55
        self.balloon_v = 0.0
        self.balloon_held = False
        # gravity
        self.planet = next(i for i, p in enumerate(PLANETS) if p[0] == "Earth")
        self.cube_height = DROP_HEIGHT_M
        self.falling_since: float | None = None
        self.fall_from = DROP_HEIGHT_M
        self.last_fall: float | None = None
        self.set_subject(self.subject)

    # ---- สาร ------------------------------------------------------------- #

    def set_subject(self, subject: Subject) -> None:
        self.subject = subject
        self.recent = [subject] + [s for s in self.recent if s.key != subject.key][:5]
        self.side = 0
        self._apply_temperature(self.celsius)
        self.balloon_y, self.balloon_v, self.balloon_held = 0.55, 0.0, False
        self.cube_height, self.falling_since, self.last_fall = DROP_HEIGHT_M, None, None
        self.update()

    def subject_choices(self) -> list[Subject]:
        """ชิปเลือกสาร: จาก Elements ก่อน แล้วสารล่าสุด แล้วตัวอย่าง (ไม่ซ้ำกัน)"""
        seen: set[str] = set()
        out: list[Subject] = []
        quick = [Subject.from_formula(f) for f in QUICK_PICKS]
        for subject in [*self.sources(), *self.recent, *quick]:
            if subject.key not in seen:
                seen.add(subject.key)
                out.append(subject)
        return out

    def structure(self) -> Structure:
        key = (
            self.subject.key + ("" if not self.subject.is_reaction else self.subject.formula),
            self.side,
        )
        if key not in self._structures:
            self._structures[key] = structure_for(self.subject, side=self.side)
        return self._structures[key]

    def set_tool(self, tool: str) -> None:
        self.tool = tool
        self.update()

    def next_tool(self, step: int = 1) -> None:
        keys = [k for k, _ in TOOLS]
        self.set_tool(keys[(keys.index(self.tool) + step) % len(keys)])

    def toggle_side(self) -> None:
        if self.subject.is_reaction:
            self.side = 1 - self.side
            self.update()

    def mix(self) -> None:
        if self.on_mix is not None:
            self.on_mix(self.subject)

    # ---- matter ---------------------------------------------------------- #

    def set_temperature(self, celsius: float) -> None:
        self._apply_temperature(max(-273.15, celsius))
        self.update()

    def _apply_temperature(self, celsius: float) -> None:
        self.celsius = celsius
        phase = phase_at(self.subject, celsius)
        if phase is None:
            state = room_state(self.subject)
            phase = state if state in ("solid", "liquid", "gas") else None
        self.matter.set_state(phase, celsius)

    def preset(self, which: str) -> None:
        data = phase_data(self.subject)
        if which == "room":
            self.set_temperature(25.0)
        elif data is not None and which == "freeze" and data.melt is not None:
            self.set_temperature(data.melt - max(20.0, abs(data.melt) * 0.1))
        elif data is not None and which == "boil" and data.boil is not None:
            self.set_temperature(data.boil + max(20.0, abs(data.boil) * 0.1))

    # ---- air / gravity ---------------------------------------------------- #

    def release_balloon(self) -> None:
        self.balloon_y, self.balloon_v, self.balloon_held = 0.55, 0.0, False
        self.update()

    def set_planet(self, index: int) -> None:
        self.planet = index % len(PLANETS)
        self.cube_height, self.falling_since, self.last_fall = DROP_HEIGHT_M, None, None
        self.update()

    def drop(self) -> None:
        if self.falling_since is None:
            self.fall_from = self.cube_height
            self.falling_since = self._clock()
            self.last_fall = None

    # ---- layout ------------------------------------------------------------ #

    def view_rect(self) -> QRectF:
        top = PAD + 2 * ROW + 16
        return QRectF(PAD, top, self.width() * VIEW_SHARE - 1.5 * PAD, self.height() - top - PAD)

    def panel_rect(self) -> QRectF:
        view = self.view_rect()
        return QRectF(
            view.right() + PAD, view.top(), self.width() - view.right() - 2 * PAD, view.height()
        )

    def _tab_rects(self) -> list[tuple[QRectF, str, str]]:
        rects, x = [], PAD
        for key, label in TOOLS:
            width = len(label) * 7.0 + 26
            rects.append((QRectF(x, PAD, width, ROW), key, label))
            x += width + 6
        return rects

    def _subject_rects(self) -> list[tuple[QRectF, Subject]]:
        rects, x = [], PAD + 70
        y = PAD + ROW + 8
        for subject in self.subject_choices():
            label = subject.chip
            width = max(44.0, len(label) * 8.0 + 22)
            if x + width > self.width() - PAD:
                break
            rects.append((QRectF(x, y, width, ROW - 4), subject))
            x += width + 6
        return rects

    def _panel_buttons(self) -> list[tuple[QRectF, str, Callable[[], None]]]:
        panel = self.panel_rect()
        buttons: list[tuple[str, Callable[[], None]]] = []
        if self.tool == "molecule" and self.subject.is_reaction:
            buttons.append(("After ›" if self.side == 0 else "‹ Before", self.toggle_side))
        if self.tool == "matter":
            buttons += [("Freeze", lambda: self.preset("freeze")), ("Room 25 °C", lambda: self.preset("room")),
                        ("Boil", lambda: self.preset("boil"))]  # fmt: skip
        if self.tool == "air":
            buttons.append(("Release balloon", self.release_balloon))
        if self.tool == "gravity":
            buttons.append(("Drop", self.drop))
        if self.on_mix is not None:
            buttons.append(("Mix in flask ›", self.mix))
        rects, x, y = [], panel.left(), panel.bottom() - ROW
        for label, action in buttons:
            width = len(label) * 7.2 + 26
            if x + width > panel.right():
                x, y = panel.left(), y - ROW - 6
            rects.append((QRectF(x, y, width, ROW - 4), label, action))
            x += width + 6
        return rects

    def _planet_rects(self) -> list[tuple[QRectF, int]]:
        panel = self.panel_rect()
        top = panel.top() + 118
        return [
            (QRectF(panel.left(), top + i * 26, panel.width(), 24), i) for i in range(len(PLANETS))
        ]

    def _build_zones(self) -> list[Zone]:
        zones: list[Zone] = [(r, lambda k=k: self.set_tool(k)) for r, k, _ in self._tab_rects()]
        zones += [(r, lambda s=s: self.set_subject(s)) for r, s in self._subject_rects()]
        zones += [(r, action) for r, _label, action in self._panel_buttons()]
        if self.tool == "gravity":
            zones += [(r, lambda i=i: self.set_planet(i)) for r, i in self._planet_rects()]
        self._zones = zones
        return zones

    def zones(self) -> list[Zone]:
        return self._build_zones()

    # ---- มือ / เมาส์ -------------------------------------------------------- #

    def on_hands(self, now: float) -> None:
        self._build_zones()
        view = self.view_rect()
        present = set()
        for hand in self.hands:
            present.add(hand.handedness)
            point = QPointF(hand.x * self.width(), hand.y * self.height())
            was = self._pinched.get(hand.handedness, False)
            if hand.pinch and not was:
                for rect, action in reversed(self._zones):
                    if rect.contains(point):
                        action()
                        break
                else:
                    if view.contains(point):
                        self._start_drag(hand.handedness, point)
            elif hand.pinch and self._drag is not None and self._drag[0] == hand.handedness:
                self._continue_drag(point)
            elif (
                not hand.pinch
                and was
                and self._drag is not None
                and self._drag[0] == hand.handedness
            ):
                self._end_drag()
            self._pinched[hand.handedness] = hand.pinch
        if self._drag is not None and self._drag[0] not in present:
            self._end_drag()
        for name in list(self._pinched):
            if name not in present:
                del self._pinched[name]
        main = self.hands[0] if self.hands else None
        if main is not None and self._drag is None:
            direction = self.swipe.update(main, now)
            if direction:
                self.next_tool(-direction)

    def _start_drag(self, hand: str, point: QPointF) -> None:
        what = self.tool
        if self.tool == "air":
            balloon = self._balloon_center()
            if math.dist((point.x(), point.y()), (balloon.x(), balloon.y())) > 60:
                return
            self.balloon_held = True
        if self.tool == "gravity":
            cube = self._cube_rect()
            if not cube.adjusted(-30, -30, 30, 30).contains(point):
                return
            self.falling_since = None
        self._drag = (hand, what, point.x(), point.y())
        self._continue_drag(point)

    def _continue_drag(self, point: QPointF) -> None:
        assert self._drag is not None
        hand, what, x0, y0 = self._drag
        view = self.view_rect()
        if what == "molecule":
            self.yaw += (point.x() - x0) / max(view.width(), 1) * 4.0
            self.pitch = max(
                -1.5, min(1.5, self.pitch + (point.y() - y0) / max(view.height(), 1) * 4.0)
            )
            self._drag = (hand, what, point.x(), point.y())
        elif what == "matter":
            value = 1.0 - (point.y() - view.top()) / max(view.height(), 1)
            self._apply_temperature(temperature_from_slider(value))
        elif what == "air":
            self.balloon_y = min(max((point.y() - view.top()) / max(view.height(), 1), 0.08), 0.92)
            self.balloon_v = 0.0
        elif what == "gravity":
            ground, top = self._ground_y(), view.top() + 60
            fraction = (ground - point.y()) / max(ground - top, 1)
            self.cube_height = max(0.0, min(1.0, fraction)) * DROP_HEIGHT_M
        self.update()

    def _end_drag(self) -> None:
        if self._drag is not None:
            what = self._drag[1]
            if what == "air":
                self.balloon_held = False
            elif what == "gravity" and self.cube_height > 0.01:
                self.drop()
        self._drag = None

    def wheelEvent(self, event) -> None:
        step = 1 if event.angleDelta().y() > 0 else -1
        if self.tool == "matter":
            self.set_temperature(
                temperature_from_slider(slider_from_temperature(self.celsius) + step * 0.015)
            )
        elif self.tool == "gravity":
            self.set_planet(self.planet + step)
        elif self.tool == "molecule":
            self.yaw += step * 0.2
            self.update()

    # ---- เวลา ---------------------------------------------------------------- #

    def tick(self, now: float) -> None:
        dt = min(max(now - self._last_tick, 0.0), 0.05)
        self._last_tick = now
        if self.tool == "molecule" and self._drag is None:
            self.yaw += 0.35 * dt
        elif self.tool == "matter":
            self.matter.step(dt)
        elif self.tool == "air" and not self.balloon_held:
            balloon = balloon_for(self.subject)
            accel = -balloon.lift * 1.4 if balloon.gas else 2.5
            self.balloon_v = (self.balloon_v + accel * dt) * 0.97
            self.balloon_y += self.balloon_v * dt
            if not 0.08 <= self.balloon_y <= 0.92:
                self.balloon_y = min(max(self.balloon_y, 0.08), 0.92)
                self.balloon_v = 0.0
        elif self.tool == "gravity" and self.falling_since is not None:
            g = PLANETS[self.planet][2]
            elapsed = now - self.falling_since
            height = self.fall_from - 0.5 * g * elapsed * elapsed
            if height <= 0.0:
                self.cube_height = 0.0
                self.last_fall = fall_time(self.fall_from, g)
                self.falling_since = None
            else:
                self.cube_height = height
        super().tick(now)
        self.update()

    # ---- geometry ของภาพ ------------------------------------------------------ #

    def _ground_y(self) -> float:
        return self.view_rect().bottom() - 40

    def _cube_rect(self) -> QRectF:
        view = self.view_rect()
        ground, top = self._ground_y(), view.top() + 60
        size = 46.0
        y = ground - (ground - top) * (self.cube_height / DROP_HEIGHT_M) - size
        return QRectF(view.center().x() - size / 2, y, size, size)

    def _balloon_center(self) -> QPointF:
        view = self.view_rect()
        return QPointF(
            view.left() + view.width() * 0.7, view.top() + view.height() * self.balloon_y
        )

    def project(self, structure: Structure, rect: QRectF) -> list[tuple[float, float, float]]:
        """(x, y, ความลึก) บนจอของทุกอะตอม — ความลึกมาก = ใกล้ผู้ดู"""
        cy, sy = math.cos(self.yaw), math.sin(self.yaw)
        cp, sp = math.cos(self.pitch), math.sin(self.pitch)
        scale = min(rect.width(), rect.height()) / (2.0 * max(structure.radius(), 1.0) * 1.1)
        out = []
        for atom in structure.atoms:
            x1 = atom.x * cy + atom.z * sy
            z1 = -atom.x * sy + atom.z * cy
            y1 = atom.y * cp - z1 * sp
            z2 = atom.y * sp + z1 * cp
            out.append((rect.center().x() + x1 * scale, rect.center().y() + y1 * scale, z2))
        return out

    # ---- วาด ----------------------------------------------------------------- #

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.paint_background(painter, grid=False)
        self._build_zones()
        self._paint_header(painter)
        view = self.view_rect()
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(255, 255, 255, 6))
        painter.drawRoundedRect(view, 14, 14)
        painter.save()
        painter.setClipRect(view)
        {
            "molecule": self._paint_molecule,
            "matter": self._paint_matter,
            "air": self._paint_air,
            "gravity": self._paint_gravity,
        }[self.tool](painter, view)
        painter.restore()
        self._paint_panel(painter)
        self.paint_cursors(painter)
        self.paint_toast(painter)
        painter.end()

    def _chip(
        self, painter: QPainter, rect: QRectF, label: str, active: bool, size: float = 12
    ) -> None:
        painter.setPen(QPen(ACCENT if active else QColor(255, 255, 255, 40), 1.0))
        painter.setBrush(with_alpha(ACCENT, 0.18) if active else QColor(255, 255, 255, 8))
        painter.drawRoundedRect(rect, rect.height() / 2, rect.height() / 2)
        painter.setFont(_font(size, 600 if active else 500))
        painter.setPen(ACCENT if active else PRIMARY)
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, label)

    def _paint_header(self, painter: QPainter) -> None:
        for rect, key, label in self._tab_rects():
            self._chip(painter, rect, label, key == self.tool)
        painter.setFont(_font(11, 500))
        painter.setPen(DIM)
        painter.drawText(
            QRectF(PAD, PAD + ROW + 8, 66, ROW - 4),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            "Substance",
        )
        for rect, subject in self._subject_rects():
            active = subject.key == self.subject.key and subject.formula == self.subject.formula
            self._chip(painter, rect, subject.chip, active, 12)

    # ---- Molecule ----------------------------------------------------------- #

    def _paint_molecule(self, painter: QPainter, view: QRectF) -> None:
        structure = self.structure()
        area = view.adjusted(30, 30, -30, -50)
        points = self.project(structure, area)
        scale = min(area.width(), area.height()) / (2.0 * max(structure.radius(), 1.0) * 1.1)
        depths = [p[2] for p in points] or [0.0]
        near, far = max(depths), min(depths)

        def fog(depth: float) -> float:
            return 0.45 + 0.55 * ((depth - far) / (near - far) if near > far else 1.0)

        items: list[tuple[float, int, int]] = [(p[2], 1, i) for i, p in enumerate(points)]
        items += [
            ((points[b.a][2] + points[b.b][2]) / 2 - 0.01, 0, k)
            for k, b in enumerate(structure.bonds)
        ]
        if structure.kind == "metallic":
            self._paint_electron_sea(painter, points)
        for _depth, kind, index in sorted(items):
            if kind == 0:
                bond = structure.bonds[index]
                a, b = points[bond.a], points[bond.b]
                alpha = fog((a[2] + b[2]) / 2)
                dx, dy = b[0] - a[0], b[1] - a[1]
                length = math.hypot(dx, dy) or 1.0
                nx, ny = -dy / length, dx / length
                pen = QPen(QColor(200, 205, 214, round(255 * alpha)), max(2.0, scale * 0.07))
                pen.setCapStyle(Qt.PenCapStyle.RoundCap)
                painter.setPen(pen)
                offsets = {1: (0.0,), 2: (-1.0, 1.0), 3: (-1.6, 0.0, 1.6)}[bond.order]
                gap = max(2.5, scale * 0.06)
                for offset in offsets:
                    painter.drawLine(
                        QPointF(a[0] + nx * gap * offset, a[1] + ny * gap * offset),
                        QPointF(b[0] + nx * gap * offset, b[1] + ny * gap * offset),
                    )
            else:
                atom = structure.atoms[index]
                x, y, depth = points[index]
                radius = max(
                    5.0,
                    atom.radius
                    * scale
                    * (0.55 if structure.kind in ("molecule", "reaction") else 0.32),
                )
                color = _atom_color(atom.symbol)
                alpha = fog(depth)
                gradient = QRadialGradient(
                    QPointF(x - radius * 0.35, y - radius * 0.35), radius * 1.3
                )
                gradient.setColorAt(0.0, with_alpha(color.lighter(150), alpha))
                gradient.setColorAt(0.6, with_alpha(color, alpha))
                gradient.setColorAt(1.0, with_alpha(color.darker(170), alpha))
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QBrush(gradient))
                painter.drawEllipse(QPointF(x, y), radius, radius)
                if atom.charge:
                    ring = QColor("#E5795E") if atom.charge > 0 else QColor("#5B8DEF")
                    painter.setBrush(Qt.BrushStyle.NoBrush)
                    painter.setPen(QPen(with_alpha(ring, alpha), 1.6))
                    painter.drawEllipse(QPointF(x, y), radius + 2.5, radius + 2.5)
                if radius >= 9:
                    painter.setFont(_font(max(9.0, radius * 0.75), 600))
                    painter.setPen(QColor(20, 22, 26) if color.lightness() > 150 else PRIMARY)
                    painter.drawText(
                        QRectF(x - radius, y - radius, 2 * radius, 2 * radius),
                        Qt.AlignmentFlag.AlignCenter,
                        atom.symbol,
                    )
                if atom.charge:
                    painter.setFont(_font(11, 700))
                    painter.setPen(PRIMARY)
                    painter.drawText(
                        QPointF(x + radius * 0.7, y - radius * 0.7), _charge_text(atom.charge)
                    )
        if structure.labels:
            painter.setFont(_font(13, 600))
            painter.setPen(ACCENT)
            cy, sy = math.cos(self.yaw), math.sin(self.yaw)
            for lx, ly, lz, text in structure.labels:
                sx = area.center().x() + (lx * cy + lz * sy) * scale
                sy_px = area.center().y() + ly * scale * math.cos(self.pitch)
                painter.drawText(
                    QRectF(sx - 80, sy_px, 160, 22), Qt.AlignmentFlag.AlignCenter, text
                )
        painter.setFont(_font(11, 400))
        painter.setPen(DIM)
        painter.drawText(
            view.adjusted(12, 0, -12, -10),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignBottom,
            "Pinch + drag to rotate · ball-and-stick model, sizes ∝ covalent radius",
        )

    def _paint_electron_sea(self, painter: QPainter, points) -> None:
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        if not xs:
            return
        left, right, top, bottom = min(xs), max(xs), min(ys), max(ys)
        now = self._clock()
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(127, 216, 255, 160))
        for i in range(48):
            fx = 0.5 + 0.5 * math.sin(now * (0.6 + i % 7 * 0.13) + i * 2.1)
            fy = 0.5 + 0.5 * math.sin(now * (0.5 + i % 5 * 0.17) + i * 1.3)
            painter.drawEllipse(
                QPointF(left + (right - left) * fx, top + (bottom - top) * fy), 2.2, 2.2
            )

    # ---- Matter --------------------------------------------------------------- #

    def _paint_matter(self, painter: QPainter, view: QRectF) -> None:
        box = QRectF(view.left() + 40, view.top() + 40, view.width() - 150, view.height() - 80)
        painter.setPen(QPen(QColor(255, 255, 255, 60), 1.5))
        painter.setBrush(QColor(255, 255, 255, 5))
        painter.drawRoundedRect(box, 10, 10)
        color = _atom_color(self.subject.main_symbol)
        radius = max(4.0, box.width() * 0.022 * 0.9)
        painter.setPen(Qt.PenStyle.NoPen)
        heat = min(1.0, max(0.0, slider_from_temperature(self.celsius)))
        glow = QColor(color).lighter(100 + round(60 * heat))
        for p in self.matter.particles:
            center = QPointF(box.left() + p.x * box.width(), box.top() + p.y * box.height())
            gradient = QRadialGradient(center - QPointF(radius * 0.3, radius * 0.3), radius * 1.3)
            gradient.setColorAt(0.0, glow.lighter(140))
            gradient.setColorAt(1.0, glow.darker(160))
            painter.setBrush(QBrush(gradient))
            painter.drawEllipse(center, radius, radius)
        self._paint_thermometer(painter, QRectF(box.right() + 40, box.top(), 30, box.height()))

    def _paint_thermometer(self, painter: QPainter, rect: QRectF) -> None:
        painter.setPen(QPen(QColor(255, 255, 255, 60), 1.0))
        painter.setBrush(QColor(255, 255, 255, 8))
        painter.drawRoundedRect(rect, rect.width() / 2, rect.width() / 2)
        level = slider_from_temperature(self.celsius)
        fill = QRectF(
            rect.left() + 6,
            rect.bottom() - 6 - (rect.height() - 12) * level,
            rect.width() - 12,
            (rect.height() - 12) * level,
        )
        warm = QColor("#E5484D") if self.celsius > 25 else QColor("#5B8DEF")
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(warm)
        painter.drawRoundedRect(fill, 6, 6)
        data = phase_data(self.subject)
        painter.setFont(_font(10, 500))
        marks = []
        if data is not None:
            if data.melt is not None and not data.sublimes:
                marks.append((data.melt, "mp"))
            if data.boil is not None:
                marks.append((data.boil, "sub" if data.sublimes else "bp"))
        marks.append((25.0, "room"))
        placed: list[float] = []
        for celsius, label in sorted(marks, key=lambda m: -m[0]):  # บนลงล่าง ป้ายไม่ทับกัน
            y = rect.bottom() - 6 - (rect.height() - 12) * slider_from_temperature(celsius)
            text_y = max(y, placed[-1] + 13) if placed else y
            placed.append(text_y)
            painter.setPen(QPen(ACCENT if label != "room" else DIM, 1.0))
            painter.drawLine(QPointF(rect.left() - 8, y), QPointF(rect.right() + 8, y))
            text = label if label == "room" else f"{label} {celsius:g}°"
            painter.drawText(QPointF(rect.right() + 10, text_y + 4), text)

    # ---- Air -------------------------------------------------------------------- #

    def _paint_air(self, painter: QPainter, view: QRectF) -> None:
        now = self._clock()
        painter.setPen(Qt.PenStyle.NoPen)
        weights = [(formula, percent, color) for formula, percent, color in AIR]
        index = 0
        for formula, percent, color in weights:  # โมเลกุลในอากาศ ~ ตามสัดส่วนจริง (50 โมเลกุล)
            count = max(0, round(percent / 2))
            for _ in range(count):
                fx = 0.5 + 0.48 * math.sin(now * (0.3 + index % 7 * 0.05) + index * 2.3)
                fy = 0.5 + 0.48 * math.sin(now * (0.27 + index % 5 * 0.06) + index * 1.7)
                center = QPointF(view.left() + view.width() * fx, view.top() + view.height() * fy)
                painter.setBrush(with_alpha(QColor(color), 0.75))
                if formula in ("N2", "O2"):
                    painter.drawEllipse(center + QPointF(-3.5, 0), 4.0, 4.0)
                    painter.drawEllipse(center + QPointF(3.5, 0), 4.0, 4.0)
                else:
                    painter.drawEllipse(center, 4.5, 4.5)
                index += 1
        balloon = balloon_for(self.subject)
        center = self._balloon_center()
        if balloon.gas:
            color = _atom_color(self.subject.main_symbol)
            painter.setPen(QPen(QColor(255, 255, 255, 120), 1.0))
            painter.drawLine(center + QPointF(0, 40), center + QPointF(math.sin(now * 2) * 6, 110))
            gradient = QRadialGradient(center - QPointF(12, 14), 50)
            gradient.setColorAt(0.0, color.lighter(150))
            gradient.setColorAt(1.0, color.darker(140))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(gradient))
            painter.drawEllipse(center, 34, 40)
            painter.setFont(_font(13, 700))
            painter.setPen(QColor(20, 22, 26) if color.lightness() > 150 else PRIMARY)
            painter.drawText(
                QRectF(center.x() - 34, center.y() - 40, 68, 80),
                Qt.AlignmentFlag.AlignCenter,
                self.subject.display,
            )
        else:
            box = QRectF(center.x() - 30, view.bottom() - 70, 60, 50)
            painter.setPen(QPen(QColor(255, 255, 255, 80), 1.0))
            painter.setBrush(QColor(255, 255, 255, 12))
            painter.drawRoundedRect(box, 8, 8)
            painter.setFont(_font(12, 600))
            painter.setPen(PRIMARY)
            painter.drawText(box, Qt.AlignmentFlag.AlignCenter, self.subject.display)
        painter.setFont(_font(11, 400))
        painter.setPen(DIM)
        painter.drawText(
            view.adjusted(12, 0, -12, -10),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignBottom,
            "Blue pairs N₂ · red pairs O₂ · purple Ar — about 1 in 2,500 molecules is CO₂",
        )

    # ---- Gravity ------------------------------------------------------------------ #

    def _paint_gravity(self, painter: QPainter, view: QRectF) -> None:
        name, thai, g = PLANETS[self.planet]
        ground = self._ground_y()
        painter.setPen(QPen(QColor(255, 255, 255, 70), 1.5))
        painter.drawLine(QPointF(view.left() + 20, ground), QPointF(view.right() - 20, ground))
        painter.setFont(_font(12, 600))
        painter.setPen(DIM)
        painter.drawText(
            QRectF(view.left() + 20, ground + 6, 300, 22),
            Qt.AlignmentFlag.AlignLeft,
            f"{name} · {thai} · g = {g:g} m/s²",
        )
        top = view.top() + 60
        painter.setPen(QPen(QColor(255, 255, 255, 40), 1.0, Qt.PenStyle.DashLine))
        painter.drawLine(QPointF(view.left() + 60, top), QPointF(view.left() + 60, ground))
        painter.setFont(_font(10, 500))
        painter.drawText(QPointF(view.left() + 66, top + 4), f"{DROP_HEIGHT_M:g} m")
        cube = self._cube_rect()
        color = _atom_color(self.subject.main_symbol)
        painter.setPen(QPen(color.lighter(130), 1.4))
        painter.setBrush(with_alpha(color, 0.75))
        painter.drawRoundedRect(cube, 6, 6)
        painter.setFont(_font(11, 700))
        painter.setPen(QColor(20, 22, 26) if color.lightness() > 150 else PRIMARY)
        painter.drawText(cube, Qt.AlignmentFlag.AlignCenter, self.subject.display)
        # ขนนกตกข้างๆ ในสุญญากาศ: ถึงพื้นพร้อมกัน (กาลิเลโอ / Apollo 15)
        feather = QRectF(cube.left() - 120, cube.top() + 16, 30, 12)
        painter.setPen(QPen(QColor(230, 230, 230, 180), 1.2))
        painter.setBrush(QColor(230, 230, 230, 60))
        painter.drawEllipse(feather)
        painter.setFont(_font(12, 600))
        painter.setPen(ACCENT)
        if self.last_fall is not None:
            text = f"Landed in {self.last_fall:.2f} s — the feather lands at the same time (vacuum)"
        elif self.falling_since is not None:
            text = f"Falling… {self._clock() - self.falling_since:.2f} s"
        else:
            text = "Pinch the block, lift it and let go (or press Drop)"
        painter.drawText(
            QRectF(view.left() + 20, view.top() + 14, view.width() - 40, 22),
            Qt.AlignmentFlag.AlignLeft,
            text,
        )

    # ---- แผงข้อมูลด้านขวา -------------------------------------------------------------- #

    def _paint_panel(self, painter: QPainter) -> None:
        panel = self.panel_rect()
        subject = self.subject
        wrap = Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop | Qt.TextFlag.TextWordWrap
        left = Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        painter.setFont(_font(24, 600))
        painter.setPen(ACCENT)
        title = formula_text(subject.compound.equation) if subject.is_reaction else subject.display
        painter.setFont(_font(15 if subject.is_reaction else 24, 600))
        painter.drawText(QRectF(panel.left(), panel.top(), panel.width(), 40), wrap, title)
        painter.setFont(_font(13, 600))
        painter.setPen(PRIMARY)
        name = subject.name + (f" · {subject.thai}" if subject.thai else "")
        painter.drawText(QRectF(panel.left(), panel.top() + 42, panel.width(), 20), left, name)
        painter.setFont(_font(11, 400))
        painter.setPen(DIM)
        kind = {"element": "ธาตุ", "compound": "สารประกอบ", "mix": "ของในขวด (ยังไม่ตรงสูตรที่รู้จัก)"}[
            subject.kind
        ]
        if subject.is_reaction:
            kind = "ปฏิกิริยาเคมี — กด After › เพื่อดูผลิตภัณฑ์"
        mass = f" · {subject.molar_mass:.2f} g/mol" if not subject.is_reaction else ""
        painter.drawText(
            QRectF(panel.left(), panel.top() + 62, panel.width(), 18), left, kind + mass
        )
        body = QRectF(
            panel.left(), panel.top() + 90, panel.width(), panel.height() - 90 - 2 * ROW - 12
        )
        lines = {
            "molecule": self._molecule_lines,
            "matter": self._matter_lines,
            "air": self._air_lines,
            "gravity": self._gravity_lines,
        }[self.tool]()
        if self.tool == "gravity":
            self._paint_planets(painter)
            body = QRectF(
                panel.left(),
                panel.top() + 118 + len(PLANETS) * 26 + 8,
                panel.width(),
                body.bottom() - (panel.top() + 118 + len(PLANETS) * 26 + 8),
            )
        y = body.top()
        for text, emphasis in lines:
            painter.setFont(_font(13 if emphasis else 12, 600 if emphasis else 400))
            painter.setPen(PRIMARY if emphasis else with_alpha(PRIMARY, 0.82))
            bounds = painter.boundingRect(QRectF(body.left(), y, body.width(), 400), wrap, text)
            if y + bounds.height() > body.bottom():
                break
            painter.drawText(QRectF(body.left(), y, body.width(), bounds.height()), wrap, text)
            y += bounds.height() + 6
        for rect, label, _action in self._panel_buttons():
            self._chip(painter, rect, label, label.startswith("Mix"), 12)

    def _molecule_lines(self) -> list[tuple[str, bool]]:
        structure = self.structure()
        lines = [(KIND_TEXT.get(structure.kind, structure.kind), True)]
        if structure.shape:
            lines.append((f"รูปร่าง: {structure.shape}", True))
            lines.append((f"มุมพันธะตามทฤษฎี VSEPR: {structure.angle}", False))
        counts = {1: 0, 2: 0, 3: 0}
        for bond in structure.bonds:
            counts[bond.order] += 1
        if structure.bonds:
            parts = [
                f"{name} {counts[o]}"
                for o, name in ((1, "เดี่ยว"), (2, "คู่"), (3, "สาม"))
                if counts[o]
            ]
            lines.append(("พันธะ: " + " · ".join(parts), False))
        lone = sum(a.lone_pairs for a in structure.atoms)
        if lone and structure.kind in ("molecule", "reaction"):
            lines.append((f"อิเล็กตรอนคู่โดดเดี่ยว {lone} คู่ — ผลักแรงกว่าคู่พันธะ ทำให้มุมพันธะแคบลง", False))
        if structure.note:
            lines.append((structure.note, False))
        if structure.approximate:
            lines.append(("⚠ โครงสร้างโดยประมาณ — ไม่ใช่สารที่บันทึกโครงสร้างไว้", False))
        return lines

    def _matter_lines(self) -> list[tuple[str, bool]]:
        data = phase_data(self.subject)
        phase = phase_at(self.subject, self.celsius)
        lines = [(f"{self.celsius:.0f} °C  ·  {kelvin(self.celsius):.0f} K", True)]
        if phase is not None:
            lines.append((f"สถานะ: {PHASE_TEXT[phase]}", True))
        else:
            room = room_state(self.subject)
            lines.append(
                (
                    f"ไม่มีข้อมูลจุดหลอมเหลว/จุดเดือด — แสดงสถานะที่อุณหภูมิห้อง ({PHASE_TEXT.get(room, room)})",
                    True,
                )
            )
        if data is not None:
            if data.sublimes and data.melt is not None:
                lines.append((f"ระเหิดที่ {data.melt:g} °C (ของแข็ง → แก๊ส ไม่ผ่านของเหลว)", False))
            else:
                if data.melt is not None:
                    lines.append((f"จุดหลอมเหลว {data.melt:g} °C", False))
                if data.boil is not None:
                    lines.append((f"จุดเดือด {data.boil:g} °C", False))
                elif not data.never_freezes:
                    lines.append(("จุดเดือด: ไม่มีข้อมูล", False))
            if data.note:
                lines.append((data.note, False))
        lines.append(("ร้อนขึ้น = อนุภาคเคลื่อนที่เร็วขึ้น (พลังงานจลน์ ∝ อุณหภูมิเคลวิน)", False))
        lines.append(("จีบในภาชนะแล้วยกมือขึ้น/ลง = เพิ่ม/ลดอุณหภูมิ · ล้อเมาส์ก็ได้", False))
        return lines

    def _air_lines(self) -> list[tuple[str, bool]]:
        balloon = balloon_for(self.subject)
        lines = [("อากาศแห้ง (โดยปริมาตร)", True)]
        lines += [(f"{formula_text(f)}  {p:g}%", False) for f, p, _ in AIR]
        lines.append((f"มวลโมเลกุลเฉลี่ยของอากาศ {AIR_MOLAR_MASS:.2f} g/mol", False))
        lines.append((balloon.text, True))
        if balloon.gas:
            lines.append(
                (
                    f"แพร่ในอากาศเร็วกว่าอากาศ {diffusion_speed(self.subject):.2f} เท่า (กฎของเกรแฮม ∝ 1/√M)",
                    False,
                )
            )
        return lines

    def _gravity_lines(self) -> list[tuple[str, bool]]:
        mass = cube_mass_kg(self.subject)
        g = PLANETS[self.planet][2]
        lines = []
        if mass is None:
            lines.append(("ไม่มีข้อมูลความหนาแน่น — คำนวณน้ำหนักไม่ได้ (เวลาตกยังเท่าเดิม: ไม่ขึ้นกับมวล)", True))
        lines.append(
            (
                f"ตกจาก {DROP_HEIGHT_M:g} m ใช้เวลา {fall_time(DROP_HEIGHT_M, g):.2f} s บน{PLANETS[self.planet][1]} — ทุกอย่างตกเร็วเท่ากันในสุญญากาศ",
                False,
            )
        )
        return lines

    def _paint_planets(self, painter: QPainter) -> None:
        panel = self.panel_rect()
        mass = cube_mass_kg(self.subject)
        painter.setFont(_font(12, 600))
        painter.setPen(PRIMARY)
        state = room_state(self.subject)
        what = "กล่อง 1 ลิตรที่บรรจุแก๊สนี้" if state == "gas" else "ก้อน 10×10×10 ซม. (1 ลิตร)"
        mass_text = (
            f" = {mass * 1000:.2f} g"
            if mass is not None and mass < 1
            else (f" = {mass:.2f} kg" if mass is not None else "")
        )
        painter.drawText(
            QRectF(panel.left(), panel.top() + 92, panel.width(), 22),
            Qt.AlignmentFlag.AlignLeft,
            what + mass_text,
        )
        for rect, index in self._planet_rects():
            _name, thai, g = PLANETS[index]
            active = index == self.planet
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(with_alpha(ACCENT, 0.16) if active else QColor(255, 255, 255, 6))
            painter.drawRoundedRect(rect, 6, 6)
            painter.setFont(_font(12, 600 if active else 500))
            painter.setPen(ACCENT if active else PRIMARY)
            painter.drawText(
                rect.adjusted(10, 0, 0, 0),
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                f"{thai}",
            )
            if mass is not None:
                weight = mass * g
                feels = mass * g / EARTH_G
                shown = f"{weight:,.0f}" if weight >= 1000 else f"{weight:.3g}"
                text = f"{shown} N  ·  ≈ {feels:.3g} kg บนตาชั่งโลก"
            else:
                text = f"g = {g:g}"
            painter.setFont(_font(11, 500))
            painter.drawText(
                rect.adjusted(0, 0, -10, 0),
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                text,
            )


def experiments_controls(page: ExperimentsPage) -> QWidget:
    bar = QWidget()
    layout = QHBoxLayout(bar)
    layout.setContentsMargins(0, 0, 0, 0)
    for key, label in TOOLS:
        button = QPushButton(label.split(" · ")[0])
        button.clicked.connect(lambda _=False, k=key: page.set_tool(k))
        layout.addWidget(button)
    layout.addStretch(1)
    picker = QComboBox()
    picker.setToolTip("Substance — or pick one in Elements and press 'Open in Experiments'")
    for formula in QUICK_PICKS:
        picker.addItem(formula_text(formula), formula)
    picker.activated.connect(
        lambda index: page.set_subject(Subject.from_formula(picker.itemData(index)))
    )
    mix = QPushButton("Mix in flask ›")
    mix.clicked.connect(page.mix)
    layout.addWidget(picker)
    layout.addWidget(mix)
    bar.picker = picker  # type: ignore[attr-defined]
    return bar
