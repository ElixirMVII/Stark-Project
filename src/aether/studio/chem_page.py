"""หน้า Elements ของ Studio: ตารางธาตุแบบลิ้นชัก (เปิด/ปิดได้) + ขวดผสม + การ์ดข้อมูลธาตุ + ผลการผสม"""

from __future__ import annotations

import math
import time
from collections.abc import Callable

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QBrush, QColor, QPainter, QPainterPath, QPen, QRadialGradient
from PySide6.QtWidgets import QHBoxLayout, QPushButton, QWidget

from ..hud.theme import hairline_stroke, with_alpha
from .chemistry import CATEGORIES, ELEMENTS, ChemLab, Compound, Element, by_symbol
from .pages import ACCENT, DIM, PRIMARY, StudioPage, _font

__all__ = ["ChemistryPage", "chemistry_controls"]

TABLE_SHARE = 0.56  # สัดส่วนความสูงของผืนผ้าใบที่ตารางใช้ (ตอนเปิด)
TABLE_ROWS = 9.4  # 7 คาบ + ช่องว่าง 0.4 + แลนทาไนด์ + แอกทิไนด์
MARGIN = 14.0
DRAWER_SPEED = 9.0  # ต่อวินาที (เลื่อนเปิด/ปิดแบบ exponential)
FLASH_SECONDS = 0.9
DROP_PAD = 36.0  # ปล่อยใกล้ขวดเท่านี้ (px) ก็นับว่าลงขวด


class ChemistryPage(StudioPage):
    title = "Elements"
    hint = (
        "Pinch an element to pick an atom, drop it in the flask  ·  hold ✌ remove last atom  ·  "
        "hold ✊ empty  ·  hold 👍 show/hide the table"
    )

    def __init__(self, parent=None, *, clock: Callable[[], float] = time.perf_counter) -> None:
        super().__init__(parent, clock=clock)
        self.lab = ChemLab()
        self.table_open = True
        self._drawer = 1.0  # 0 = ปิด, 1 = เปิด (ค่าที่กำลังแสดง)
        self._last_tick: float | None = None
        self._flash: tuple[float, Compound] | None = None
        self.focus: Element | None = by_symbol("H")  # การ์ดข้อมูล: ธาตุที่ชี้/หยิบล่าสุด
        self.matches: list[tuple[str, int]] = []  # สารที่เกิดในรอบนี้ (ตามลำดับ)

    # ---- geometry --------------------------------------------------------- #

    def cell_size(self) -> float:
        width = self.width() - 2 * MARGIN
        height = self.height() * TABLE_SHARE - MARGIN
        return max(8.0, min(width / 18.0, height / TABLE_ROWS))

    def table_rect(self) -> QRectF:
        cell = self.cell_size()
        width, height = cell * 18.0, cell * TABLE_ROWS
        top = MARGIN - (1.0 - self._drawer) * (height + MARGIN + 4.0)
        return QRectF((self.width() - width) / 2.0, top, width, height)

    def cell_rect(self, element: Element) -> QRectF:
        table, cell = self.table_rect(), self.cell_size()
        row = element.row if element.row < 7 else element.row + 0.4
        return QRectF(
            table.left() + element.column * cell, table.top() + row * cell, cell, cell
        ).adjusted(1.0, 1.0, -1.0, -1.0)

    def cell_at(self, x: float, y: float) -> Element | None:
        """ธาตุใต้จุด (หน่วยผืนผ้าใบ 0–1) — ตารางปิดอยู่ = ไม่มี"""
        if self._drawer < 0.6:
            return None
        point = QPointF(x * self.width(), y * self.height())
        table = self.table_rect()
        if not table.adjusted(-4, -4, 4, 4).contains(point):
            return None
        return next(
            (e for e in ELEMENTS if self.cell_rect(e).adjusted(-1, -1, 1, 1).contains(point)), None
        )

    def flask_rect(self) -> QRectF:
        top = self.table_rect().bottom() + 24.0 if self._drawer > 0.0 else MARGIN
        top = max(top, MARGIN + 40.0)
        height = min(self.height() - top - 28.0, 300.0)
        width = min(240.0, max(150.0, height * 0.85))
        return QRectF((self.width() - width) / 2.0, self.height() - 28.0 - height, width, height)

    def in_flask(self, x: float, y: float) -> bool:
        point = QPointF(x * self.width(), y * self.height())
        return self.flask_rect().adjusted(-DROP_PAD, -DROP_PAD, DROP_PAD, DROP_PAD).contains(point)

    # ---- actions ---------------------------------------------------------- #

    def toggle_table(self) -> None:
        self.table_open = not self.table_open
        self.update()

    def undo(self) -> None:
        if self.lab.flask.undo():
            self.lab.check()
            self.toast("Removed")
            self.update()

    def empty(self) -> None:
        if self.lab.flask.atoms:
            self.lab.flask.clear()
            self.lab.check()
            self.toast("Emptied")
            self.update()

    def drop_atom(self, symbol: str) -> None:
        """ใส่อะตอมหนึ่งตัวลงขวด (ใช้กับเมาส์/ทดสอบ)"""
        if self.lab.flask.add(symbol):
            for event in self.lab.check():
                self._on_match(event.compound, event.multiple)

    def on_hands(self, now: float) -> None:
        events = self.lab.update(self.hands, now, cell_at=self.cell_at, in_flask=self.in_flask)
        if self.lab.hover is not None:
            self.focus = self.lab.hover
        for event in events:
            if event.kind == "pick":
                self.focus = by_symbol(event.symbol)
            elif event.kind == "full":
                self.toast("The flask is full")
            elif event.kind == "undo":
                self.toast(f"Removed {event.symbol}")
            elif event.kind == "empty":
                self.toast("Emptied")
            elif event.kind == "table":
                self.toggle_table()
            elif event.kind == "match" and event.compound is not None:
                self._on_match(event.compound, event.multiple)

    def _on_match(self, compound: Compound, multiple: int) -> None:
        """เกิดสาร: วงวาบรอบขวด (การ์ดด้านขวาแสดงชื่อ/ข้อมูลอยู่แล้ว ไม่ต้องมีข้อความลอยซ้ำ)"""
        self._flash = (self._clock(), compound)
        self.matches.append((compound.formula, multiple))

    def tick(self, now: float) -> None:
        dt = 0.0 if self._last_tick is None else min(now - self._last_tick, 0.1)
        self._last_tick = now
        target = 1.0 if self.table_open else 0.0
        self._drawer += (target - self._drawer) * (1.0 - math.exp(-DRAWER_SPEED * dt))
        if abs(target - self._drawer) < 0.002:
            self._drawer = target
        if self._flash is not None and now - self._flash[0] > FLASH_SECONDS:
            self._flash = None
        super().tick(now)
        self.update()

    # ---- paint ------------------------------------------------------------ #

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.paint_background(painter, grid=False)
        now = self._clock()
        if self._drawer > 0.01:
            self._paint_table(painter)
        self._paint_flask(painter, now)
        self._paint_info(painter)
        self._paint_result(painter)
        self._paint_held(painter)
        self.paint_cursors(painter, self.lab.hold_progress)
        self.paint_toast(painter)
        painter.end()

    def _paint_table(self, painter: QPainter) -> None:
        cell = self.cell_size()
        hover = self.lab.hover
        held = set(self.lab.held.values())
        small = cell >= 34.0
        symbol_font = _font(cell * 0.36, 600)
        number_font = _font(max(cell * 0.19, 7.0), 500, tabular=True)
        for element in ELEMENTS:
            rect = self.cell_rect(element)
            color = QColor(CATEGORIES[element.category][2])
            lit = element is hover or element.symbol in held
            painter.setPen(QPen(ACCENT if lit else with_alpha(color, 0.35), 1.4 if lit else 1.0))
            painter.setBrush(with_alpha(color, 0.42 if lit else 0.16))
            painter.drawRoundedRect(rect, 4.0, 4.0)
            painter.setPen(PRIMARY if lit else with_alpha(PRIMARY, 0.88))
            painter.setFont(symbol_font)
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, element.symbol)
            if small:
                painter.setFont(number_font)
                painter.setPen(with_alpha(PRIMARY, 0.5))
                painter.drawText(
                    rect.adjusted(3, 1, -2, -2),
                    Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop,
                    str(element.number),
                )
        # ช่องหมู่ 3 ของคาบ 6–7 ชี้ไปแถวล่าง
        table = self.table_rect()
        painter.setFont(_font(max(cell * 0.2, 7.0), 500))
        painter.setPen(DIM)
        for row, text in ((5, "57–71"), (6, "89–103")):
            rect = QRectF(table.left() + 2 * cell, table.top() + row * cell, cell, cell)
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, text)

    def _flask_path(self, rect: QRectF) -> QPainterPath:
        """ขวดรูปชมพู่: คอแคบด้านบน ฐานกว้าง มุมมน"""
        neck = rect.width() * 0.26
        cx = rect.center().x()
        neck_bottom = rect.top() + rect.height() * 0.28
        path = QPainterPath()
        path.moveTo(cx - neck / 2, rect.top())
        path.lineTo(cx - neck / 2, neck_bottom)
        path.lineTo(rect.left() + 10, rect.bottom() - 14)
        path.quadTo(rect.left(), rect.bottom(), rect.left() + 16, rect.bottom())
        path.lineTo(rect.right() - 16, rect.bottom())
        path.quadTo(rect.right(), rect.bottom(), rect.right() - 10, rect.bottom() - 14)
        path.lineTo(cx + neck / 2, neck_bottom)
        path.lineTo(cx + neck / 2, rect.top())
        return path

    def _paint_flask(self, painter: QPainter, now: float) -> None:
        rect = self.flask_rect()
        path = self._flask_path(rect)
        match = self.lab.last_match
        atoms = self.lab.flask.atoms
        # ของเหลว: สีของสารที่เกิด (ยังไม่เกิด = ใสอมเทา)
        level = min(0.15 + 0.03 * len(atoms), 0.62) if atoms else 0.0
        if level > 0.0:
            liquid = QRectF(
                rect.left(), rect.bottom() - rect.height() * level, rect.width(), rect.height()
            )
            clip = QPainterPath()
            clip.addRect(liquid)
            color = QColor(match[0].color) if match else QColor(200, 210, 222)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(with_alpha(color, 0.30 if match else 0.12))
            painter.drawPath(path.intersected(clip))
        # อะตอมในขวด: ลูกบอลสีตามธาตุ ลอยขยับเบาๆ
        radius = 11.0
        base_y = rect.bottom() - radius - 8.0
        columns = max(1, int((rect.width() - 30) // (radius * 2.3)))
        for index, symbol in enumerate(atoms):
            col, row = index % columns, index // columns
            x = rect.center().x() + (col - (columns - 1) / 2) * radius * 2.3
            y = base_y - row * radius * 2.1
            x += 1.6 * math.sin(now * 2.1 + index * 1.7)
            y += 1.6 * math.cos(now * 1.7 + index * 2.3)
            self._paint_atom(painter, QPointF(x, y), symbol, radius)
        hairline_stroke(painter, path, PRIMARY, 1.4, alpha=0.75)
        # วงวาบตอนเกิดสาร (สารที่ปล่อยพลังงานมาก = วงใหญ่ + แสงขาว)
        if self._flash is not None:
            start, compound = self._flash
            progress = min((now - start) / FLASH_SECONDS, 1.0)
            grow = (1.0 - (1.0 - progress) ** 3) * (90.0 if compound.energetic else 50.0)
            ring = QPainterPath()
            center = QPointF(rect.center().x(), rect.bottom() - rect.height() * 0.3)
            ring.addEllipse(center, 30.0 + grow, 30.0 + grow)
            color = QColor("#FFFFFF") if compound.energetic else QColor(compound.color)
            hairline_stroke(painter, ring, color, 2.0, alpha=(1.0 - progress) ** 1.5)
            if compound.energetic:
                glow = QRadialGradient(center, 60.0 + grow)
                glow.setColorAt(0.0, with_alpha(QColor("#FFFFFF"), 0.35 * (1.0 - progress)))
                glow.setColorAt(1.0, QColor(255, 255, 255, 0))
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QBrush(glow))
                painter.drawEllipse(center, 60.0 + grow, 60.0 + grow)
        # สูตรของในขวด
        painter.setFont(_font(20, 600))
        painter.setPen(ACCENT if match else PRIMARY)
        text = self.lab.flask.formula() if atoms else "Empty flask"
        if match and match[1] > 1:
            text = f"{match[1]} × {match[0].display}"
        elif match:
            text = match[0].display
        label = QRectF(rect.left() - 60, rect.bottom() + 2, rect.width() + 120, 26)
        painter.drawText(label, Qt.AlignmentFlag.AlignCenter, text)

    def _paint_atom(self, painter: QPainter, center: QPointF, symbol: str, radius: float) -> None:
        color = QColor(by_symbol(symbol).color)
        shade = QRadialGradient(center + QPointF(-radius * 0.35, -radius * 0.4), radius * 1.5)
        shade.setColorAt(0.0, color.lighter(125))
        shade.setColorAt(1.0, color.darker(140))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(shade))
        painter.drawEllipse(center, radius, radius)
        painter.setFont(_font(radius * 0.95, 700))
        painter.setPen(QColor(20, 22, 26) if color.lightness() > 150 else PRIMARY)
        painter.drawText(
            QRectF(center.x() - radius, center.y() - radius, radius * 2, radius * 2),
            Qt.AlignmentFlag.AlignCenter,
            symbol,
        )

    def _paint_held(self, painter: QPainter) -> None:
        for hand in self.hands:
            symbol = self.lab.held.get(hand.handedness)
            if symbol:
                self._paint_atom(
                    painter, self.to_px(hand.x, hand.y) + QPointF(0, -26), symbol, 16.0
                )

    def _side_area(self, left: bool) -> QRectF:
        flask = self.flask_rect()
        top = flask.top()
        if left:
            return QRectF(MARGIN + 8, top, flask.left() - MARGIN - 40, self.height() - top - 20)
        return QRectF(
            flask.right() + 32,
            top,
            self.width() - flask.right() - MARGIN - 40,
            self.height() - top - 20,
        )

    def _paint_info(self, painter: QPainter) -> None:
        element = self.focus
        area = self._side_area(left=True)
        if element is None or area.width() < 120:
            return
        name, thai, color = CATEGORIES[element.category]
        tile = QRectF(area.left(), area.top(), 84, 84)
        painter.setPen(QPen(with_alpha(QColor(color), 0.6), 1.0))
        painter.setBrush(with_alpha(QColor(color), 0.22))
        painter.drawRoundedRect(tile, 10, 10)
        painter.setPen(PRIMARY)
        painter.setFont(_font(34, 600))
        painter.drawText(tile, Qt.AlignmentFlag.AlignCenter, element.symbol)
        painter.setFont(_font(11, 500, tabular=True))
        painter.setPen(with_alpha(PRIMARY, 0.6))
        painter.drawText(
            tile.adjusted(6, 4, -4, -4),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop,
            str(element.number),
        )
        text_left = tile.right() + 14
        width = area.right() - text_left
        left = Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        painter.setPen(PRIMARY)
        painter.setFont(_font(18, 600))
        painter.drawText(QRectF(text_left, tile.top(), width, 26), left, element.name)
        painter.setFont(_font(13, 400))
        painter.setPen(DIM)
        if element.thai:
            painter.drawText(QRectF(text_left, tile.top() + 26, width, 20), left, element.thai)
        painter.drawText(QRectF(text_left, tile.top() + 48, width, 20), left, f"{thai} · {name}")
        painter.setFont(_font(13, 500, tabular=True))
        painter.setPen(PRIMARY)
        position = f"period {element.period}" + (
            f" · group {element.group}" if element.group else ""
        )
        painter.drawText(
            QRectF(area.left(), tile.bottom() + 10, area.width(), 20),
            left,
            f"{element.mass_text} u  ·  {position}",
        )

    def _paint_result(self, painter: QPainter) -> None:
        area = self._side_area(left=False)
        if area.width() < 140:
            return
        left = Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop
        match = self.lab.last_match
        if match is not None:
            compound, multiple = match
            painter.setPen(ACCENT)
            painter.setFont(_font(34, 600))
            prefix = f"{multiple} × " if multiple > 1 else ""
            painter.drawText(
                QRectF(area.left(), area.top(), area.width(), 44), left, prefix + compound.display
            )
            painter.setPen(PRIMARY)
            painter.setFont(_font(16, 600))
            painter.drawText(
                QRectF(area.left(), area.top() + 46, area.width(), 22), left, compound.name
            )
            painter.setFont(_font(14, 400))
            painter.setPen(DIM)
            painter.drawText(
                QRectF(area.left(), area.top() + 70, area.width(), 20),
                left,
                f"{compound.thai} · {compound.molar_mass:.2f} g/mol",
            )
            painter.setPen(with_alpha(PRIMARY, 0.85))
            painter.drawText(
                QRectF(area.left(), area.top() + 96, area.width(), area.height() - 96),
                left | Qt.TextFlag.TextWordWrap,
                compound.fact,
            )
            return
        hints = self.lab.flask.hints()
        painter.setFont(_font(13, 500))
        painter.setPen(DIM)
        title = "Almost there" if hints else "Drop atoms in the flask to mix them"
        painter.drawText(QRectF(area.left(), area.top(), area.width(), 20), left, title)
        painter.setFont(_font(15, 500))
        for index, hint in enumerate(hints):
            painter.setPen(PRIMARY)
            painter.drawText(
                QRectF(area.left(), area.top() + 28 + index * 46, area.width(), 22), left, hint.text
            )
            painter.setPen(DIM)
            painter.setFont(_font(12, 400))
            painter.drawText(
                QRectF(area.left(), area.top() + 50 + index * 46, area.width(), 18),
                left,
                f"{hint.compound.name} · {hint.compound.thai}",
            )
            painter.setFont(_font(15, 500))


def chemistry_controls(page: ChemistryPage) -> QWidget:
    bar = QWidget()
    layout = QHBoxLayout(bar)
    layout.setContentsMargins(0, 0, 0, 0)
    table = QPushButton("Periodic table")
    table.setCheckable(True)
    table.setChecked(page.table_open)
    table.toggled.connect(lambda on: (setattr(page, "table_open", on), page.update()))
    undo = QPushButton("Remove last atom")
    undo.clicked.connect(page.undo)
    empty = QPushButton("Empty flask")
    empty.clicked.connect(page.empty)
    layout.addWidget(table)
    layout.addWidget(undo)
    layout.addWidget(empty)
    layout.addStretch(1)
    bar.table_button = table  # type: ignore[attr-defined]
    return bar
