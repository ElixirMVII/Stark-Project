"""หน้า Elements ของ Studio — ซ้าย: ตารางธาตุ / การ์ดธาตุทีละหมวด (ปัดมือเปลี่ยนหน้า) · ขวา: แท็บ Element
(แบบจำลองอะตอม + รายละเอียด) / Guide (คำแนะนำการผสม) · ล่างขวา: ขวดผสม + ผลลัพธ์

ปุ่มทั้งหมดบนผืนผ้าใบ (ลูกศร แท็บ ตัวกรอง แถวคำแนะนำ) กดได้ด้วยการจีบหรือคลิกเมาส์ — เป็น "โซน" ที่คำนวณใหม่ทุกครั้ง
ที่วาด/รับมือ จึงทดสอบได้โดยไม่ต้องมีปุ่ม Qt
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QBrush, QColor, QPainter, QPainterPath, QPen, QRadialGradient
from PySide6.QtWidgets import QHBoxLayout, QPushButton, QWidget

from ..hud.theme import hairline_stroke, with_alpha
from .atoms import AtomInfo, atom_info
from .chemistry import (
    CATEGORIES,
    COMPOUNDS,
    DANGER_LEVELS,
    ELEMENTS,
    ChemLab,
    Compound,
    Element,
    by_symbol,
    compounds_with,
)
from .input import SwipeDetector
from .pages import ACCENT, DIM, PRIMARY, StudioPage, _font

__all__ = ["GUIDE_FILTERS", "ChemistryPage", "browser_pages", "chemistry_controls", "guide_entries"]

PAD = 14.0
BROWSER_SHARE = 0.60
HEADER = 34.0
TABS = 32.0
TABLE_ROWS = 9.4
CARDS_PER_PAGE = 12
CATEGORY_ORDER = (
    "alkali",
    "alkaline",
    "transition",
    "post",
    "metalloid",
    "nonmetal",
    "halogen",
    "noble",
    "lanthanide",
    "actinide",
)
GUIDE_FILTERS = ("all", "fun", "everyday", "danger", "reaction")
FILTER_LABELS = {
    "all": "All",
    "fun": "Fun",
    "everyday": "Everyday",
    "danger": "Dangerous",
    "reaction": "Reactions",
}
GUIDE_ROW = 50.0
FILL_INTERVAL = 0.08  # วินาทีต่ออะตอมตอน "ลองผสม" จากคำแนะนำ
FLASH_SECONDS = 1.0
SHAKE_SECONDS = 0.45
PAGE_ANIM = 0.25


def browser_pages() -> list[tuple[str, int, int, list[Element]]]:
    """หน้าของฝั่งซ้าย: ``(หมวด, หน้าที่, จำนวนหน้าของหมวด, ธาตุ)`` — หน้าแรกคือตารางธาตุ (หมวด ``"table"``)"""
    pages: list[tuple[str, int, int, list[Element]]] = [("table", 1, 1, [])]
    for category in CATEGORY_ORDER:
        members = [e for e in ELEMENTS if e.category == category]
        count = math.ceil(len(members) / CARDS_PER_PAGE)
        size = math.ceil(len(members) / count)  # แบ่งหน้าให้เท่าๆ กัน
        for index in range(count):
            pages.append((category, index + 1, count, members[index * size : (index + 1) * size]))
    return pages


def guide_entries(kind: str) -> list[Compound]:
    if kind == "all":
        return list(COMPOUNDS)
    if kind == "danger":
        return sorted((c for c in COMPOUNDS if c.danger >= 2), key=lambda c: -c.danger)
    if kind == "reaction":
        return [c for c in COMPOUNDS if c.is_reaction]
    return [c for c in COMPOUNDS if c.kind == kind]


Zone = tuple[QRectF, Callable[[], None]]


class ChemistryPage(StudioPage):
    title = "Elements"
    hint = (
        "Pinch an element and drop it in the flask  ·  open palm + flick left/right = next/previous page  ·  "
        "Guide tab: pinch a recipe to try it  ·  hold ✌ remove · ✊ empty · 👍 table"
    )

    def __init__(self, parent=None, *, clock: Callable[[], float] = time.perf_counter) -> None:
        super().__init__(parent, clock=clock)
        self.lab = ChemLab()
        self.pages = browser_pages()
        self.page_index = 0
        self._last_category_page = 1
        self.panel = "element"  # "element" | "guide"
        self.guide_filter = "all"
        self.guide_page = 0
        self.focus: Element = by_symbol("H")
        self.matches: list[tuple[str, int]] = []
        self.swipe = SwipeDetector()
        self._zones: list[Zone] = []
        self._pinched: dict[str, bool] = {}
        self._fill_queue: list[str] = []
        self._fill_next = 0.0
        self._flash: tuple[float, Compound] | None = None
        self._shake_until = 0.0
        self._page_anim: tuple[float, int] | None = None
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    # ---- layout ------------------------------------------------------------ #

    @property
    def table_open(self) -> bool:
        return self.page_index == 0

    def left_rect(self) -> QRectF:
        return QRectF(PAD, PAD, self.width() * BROWSER_SHARE - 1.5 * PAD, self.height() - 2 * PAD)

    def right_rect(self) -> QRectF:
        left = self.left_rect()
        return QRectF(
            left.right() + PAD, PAD, self.width() - left.right() - 2 * PAD, self.height() - 2 * PAD
        )

    def content_rect(self) -> QRectF:
        return self.left_rect().adjusted(0, HEADER + 6, 0, 0)

    def panel_rect(self) -> QRectF:
        right = self.right_rect()
        height = (right.height() - TABS) * 0.58
        return QRectF(right.left(), right.top() + TABS + 6, right.width(), height - 6)

    def flask_rect(self) -> QRectF:
        right = self.right_rect()
        panel = self.panel_rect()
        top = panel.bottom() + 14
        height = max(right.bottom() - top - 26, 60.0)
        width = min(150.0, max(110.0, height * 0.75))
        return QRectF(right.left() + 8, top, width, height)

    def result_rect(self) -> QRectF:
        flask, right = self.flask_rect(), self.right_rect()
        return QRectF(
            flask.right() + 22, flask.top(), right.right() - flask.right() - 22, flask.height() + 20
        )

    def table_cell(self) -> float:
        content = self.content_rect()
        return max(8.0, min(content.width() / 18.0, content.height() / TABLE_ROWS))

    def cell_rect(self, element: Element) -> QRectF:
        """ช่องของธาตุในหน้าปัจจุบัน (ตารางหรือการ์ด) — ไม่อยู่ในหน้านี้ = สี่เหลี่ยมว่าง"""
        category = self.pages[self.page_index][0]
        content = self.content_rect()
        if category == "table":
            cell = self.table_cell()
            left = content.left() + (content.width() - cell * 18) / 2
            row = element.row if element.row < 7 else element.row + 0.4
            rect = QRectF(left + element.column * cell, content.top() + row * cell, cell, cell)
            return rect.adjusted(1, 1, -1, -1)
        members = self.pages[self.page_index][3]
        if element not in members:
            return QRectF()
        index = members.index(element)
        columns = 4 if content.width() > 560 else 3
        rows = math.ceil(len(members) / columns)
        gap = 10.0
        width = (content.width() - gap * (columns - 1)) / columns
        height = min(150.0, (content.height() - gap * (rows - 1)) / rows)
        col, row = index % columns, index // columns
        return QRectF(
            content.left() + col * (width + gap),
            content.top() + row * (height + gap),
            width,
            height,
        )

    def visible_elements(self) -> list[Element]:
        category, _, _, members = self.pages[self.page_index]
        return list(ELEMENTS) if category == "table" else members

    def cell_at(self, x: float, y: float) -> Element | None:
        point = QPointF(x * self.width(), y * self.height())
        if not self.content_rect().adjusted(-4, -4, 4, 4).contains(point):
            return None
        return next((e for e in self.visible_elements() if self.cell_rect(e).contains(point)), None)

    def in_flask(self, x: float, y: float) -> bool:
        point = QPointF(x * self.width(), y * self.height())
        return self.flask_rect().adjusted(-40, -40, 40, 40).contains(point)

    def guide_rows(self) -> tuple[list[Compound], int, int, QRectF]:
        """(แถวในหน้านี้, หน้าที่, จำนวนหน้า, กรอบรายการ)"""
        panel = self.panel_rect()
        area = panel.adjusted(0, 36, 0, -30)
        per_page = max(1, int(area.height() // GUIDE_ROW))
        entries = guide_entries(self.guide_filter)
        pages = max(1, math.ceil(len(entries) / per_page))
        self.guide_page = min(self.guide_page, pages - 1)
        start = self.guide_page * per_page
        return entries[start : start + per_page], self.guide_page, pages, area

    # ---- actions ------------------------------------------------------------ #

    def set_page(self, index: int) -> None:
        index %= len(self.pages)
        if index == self.page_index:
            return
        direction = 1 if index > self.page_index else -1
        self.page_index = index
        if index:
            self._last_category_page = index
        self._page_anim = (self._clock(), direction)
        self.update()

    def next_page(self) -> None:
        self.set_page(self.page_index + 1)

    def previous_page(self) -> None:
        self.set_page(self.page_index - 1)

    def toggle_table(self) -> None:
        self.set_page(self._last_category_page if self.page_index == 0 else 0)

    def show_category(self, category: str) -> None:
        self.set_page(next(i for i, page in enumerate(self.pages) if page[0] == category))

    def set_panel(self, panel: str) -> None:
        self.panel = panel
        self.update()

    def set_filter(self, kind: str) -> None:
        self.guide_filter, self.guide_page = kind, 0
        self.update()

    def guide_next(self, step: int = 1) -> None:
        _, page, pages, _ = self.guide_rows()
        self.guide_page = (page + step) % pages
        self.update()

    def try_recipe(self, compound: Compound) -> None:
        """เทขวด แล้วเติมอะตอมตามสูตรทีละตัว (เห็นสารค่อยๆ ก่อตัว)"""
        self.lab.flask.clear()
        self.lab.check()
        self._fill_queue = [s for s, n in compound.counts.items() for _ in range(n)]
        self._fill_next = self._clock()
        self.toast(f"Mixing {compound.name}")
        self.update()

    def undo(self) -> None:
        if self.lab.flask.undo():
            self.lab.check()
            self.toast("Removed")
            self.update()

    def empty(self) -> None:
        self._fill_queue.clear()
        if self.lab.flask.atoms:
            self.lab.flask.clear()
            self.lab.check()
            self.toast("Emptied")
            self.update()

    def drop_atom(self, symbol: str) -> None:
        if self.lab.flask.add(symbol):
            for event in self.lab.check():
                self._on_match(event.compound, event.multiple)

    # ---- input --------------------------------------------------------------- #

    def on_hands(self, now: float) -> None:
        self._build_zones()
        present = {h.handedness for h in self.hands}
        for hand in self.hands:  # ปุ่มบนผืนผ้าใบ: จีบ/คลิกครั้งแรกในโซน
            was = self._pinched.get(hand.handedness, False)
            if hand.pinch and not was and hand.handedness not in self.lab.held:
                point = QPointF(hand.x * self.width(), hand.y * self.height())
                for rect, action in reversed(self._zones):
                    if rect.contains(point):
                        action()
                        break
            self._pinched[hand.handedness] = hand.pinch
        for name in list(self._pinched):
            if name not in present:
                del self._pinched[name]

        main = self.hands[0] if self.hands else None
        if main is not None and not self.lab.held:
            direction = self.swipe.update(main, now)
            if direction:  # สะบัดไปซ้าย = หน้าถัดไป (เหมือนเลื่อนหน้ากระดาษ)
                over_guide = (
                    self.panel == "guide" and main.x * self.width() > self.right_rect().left()
                )
                if over_guide:
                    self.guide_next(-direction)
                else:
                    self.set_page(self.page_index - direction)

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
                self._fill_queue.clear()
                self.toast("Emptied")
            elif event.kind == "table":
                self.toggle_table()
            elif event.kind == "match" and event.compound is not None:
                self._on_match(event.compound, event.multiple)

    def _on_match(self, compound: Compound, multiple: int) -> None:
        now = self._clock()
        self._flash = (now, compound)
        if compound.danger >= 3:
            self._shake_until = now + SHAKE_SECONDS
        self.matches.append((compound.formula, multiple))

    def keyPressEvent(self, event) -> None:
        key = event.key()
        if key == Qt.Key.Key_Right:
            self.next_page()
        elif key == Qt.Key.Key_Left:
            self.previous_page()
        elif key == Qt.Key.Key_Down and self.panel == "guide":
            self.guide_next(1)
        elif key == Qt.Key.Key_Up and self.panel == "guide":
            self.guide_next(-1)
        else:
            super().keyPressEvent(event)

    def wheelEvent(self, event) -> None:
        step = -1 if event.angleDelta().y() > 0 else 1
        if self.panel == "guide" and event.position().x() > self.right_rect().left():
            self.guide_next(step)
        else:
            self.set_page(self.page_index + step)

    def tick(self, now: float) -> None:
        if self._fill_queue and now >= self._fill_next:
            self._fill_next = now + FILL_INTERVAL
            self.lab.flask.add(self._fill_queue.pop(0))
            if not self._fill_queue:
                for event in self.lab.check():
                    self._on_match(event.compound, event.multiple)
        if self._flash is not None and now - self._flash[0] > FLASH_SECONDS:
            self._flash = None
        if self._page_anim is not None and now - self._page_anim[0] > PAGE_ANIM:
            self._page_anim = None
        super().tick(now)
        self.update()

    # ---- zones (ปุ่มบนผืนผ้าใบ) ---------------------------------------------- #

    def _build_zones(self) -> list[Zone]:
        zones: list[Zone] = []
        left = self.left_rect()
        header = QRectF(left.left(), left.top(), left.width(), HEADER)
        zones.append((QRectF(header.right() - 70, header.top(), 32, HEADER), self.previous_page))
        zones.append((QRectF(header.right() - 34, header.top(), 32, HEADER), self.next_page))
        right = self.right_rect()
        half = right.width() / 2
        zones.append(
            (QRectF(right.left(), right.top(), half, TABS), lambda: self.set_panel("element"))
        )
        zones.append(
            (QRectF(right.left() + half, right.top(), half, TABS), lambda: self.set_panel("guide"))
        )
        panel = self.panel_rect()
        if self.panel == "guide":
            for rect, kind in self._filter_rects(panel):
                zones.append((rect, lambda kind=kind: self.set_filter(kind)))
            rows, _, _, area = self.guide_rows()
            for index, compound in enumerate(rows):
                rect = QRectF(
                    area.left(), area.top() + index * GUIDE_ROW, area.width(), GUIDE_ROW - 4
                )
                zones.append((rect, lambda c=compound: self.try_recipe(c)))
            pager = QRectF(panel.right() - 70, panel.bottom() - 26, 32, 26)
            zones.append((pager, lambda: self.guide_next(-1)))
            zones.append((pager.translated(36, 0), lambda: self.guide_next(1)))
        else:
            for rect, compound in self._makes_rects(panel):
                zones.append((rect, lambda c=compound: self.try_recipe(c)))
        self._zones = zones
        return zones

    def zones(self) -> list[Zone]:
        return self._build_zones()

    def _filter_rects(self, panel: QRectF) -> list[tuple[QRectF, str]]:
        rects, x = [], panel.left()
        for kind in GUIDE_FILTERS:
            width = len(FILTER_LABELS[kind]) * 7.5 + 22
            rects.append((QRectF(x, panel.top(), width, 26), kind))
            x += width + 6
        return rects

    def _makes_rects(self, panel: QRectF) -> list[tuple[QRectF, Compound]]:
        """ชิป "ผสมได้เป็น" ใต้รายละเอียดธาตุ (กดแล้วลองผสมสารนั้น)"""
        rects, x = [], panel.left()
        y = panel.bottom() - 28
        for compound in compounds_with(self.focus.symbol)[:8]:
            label = compound.display if not compound.is_reaction else compound.name
            width = len(label) * 7.5 + 24
            if x + width > panel.right():
                break
            rects.append((QRectF(x, y, width, 24), compound))
            x += width + 6
        return rects

    # ---- paint ----------------------------------------------------------------- #

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.paint_background(painter, grid=False)
        now = self._clock()
        self._build_zones()
        if now < self._shake_until:  # อันตรายมาก: จอสั่น
            strength = (self._shake_until - now) / SHAKE_SECONDS * 6.0
            painter.translate(math.sin(now * 91.0) * strength, math.cos(now * 77.0) * strength)
        self._paint_browser(painter, now)
        self._paint_tabs(painter)
        if self.panel == "guide":
            self._paint_guide(painter)
        else:
            self._paint_element(painter, now)
        self._paint_flask(painter, now)
        self._paint_result(painter)
        self._paint_held(painter)
        self.paint_cursors(painter, self.lab.hold_progress)
        self.paint_toast(painter)
        painter.end()

    def _arrow(self, painter: QPainter, rect: QRectF, left: bool) -> None:
        path = QPainterPath()
        cx, cy = rect.center().x(), rect.center().y()
        d = -1 if left else 1
        path.moveTo(cx - 4 * d, cy - 7)
        path.lineTo(cx + 4 * d, cy)
        path.lineTo(cx - 4 * d, cy + 7)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        pen = QPen(PRIMARY, 1.6)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        painter.drawPath(path)

    def _paint_browser(self, painter: QPainter, now: float) -> None:
        left = self.left_rect()
        category, part, parts, members = self.pages[self.page_index]
        header = QRectF(left.left(), left.top(), left.width(), HEADER)
        if category == "table":
            title, thai, color = "Periodic table", "ตารางธาตุ 118 ธาตุ", QColor(ACCENT)
        else:
            title, thai, color_hex = CATEGORIES[category]
            color = QColor(color_hex)
            if parts > 1:
                title = f"{title} ({part}/{parts})"
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(color)
        painter.drawEllipse(QPointF(header.left() + 6, header.center().y()), 5, 5)
        title_font = _font(17, 600)
        painter.setFont(title_font)
        painter.setPen(PRIMARY)
        painter.drawText(header.adjusted(20, 0, -150, 0), Qt.AlignmentFlag.AlignVCenter, title)
        offset = painter.fontMetrics().horizontalAdvance(title) + 34
        painter.setFont(_font(13, 400))
        painter.setPen(DIM)
        painter.drawText(header.adjusted(offset, 0, -150, 0), Qt.AlignmentFlag.AlignVCenter, thai)
        painter.setFont(_font(12, 500, tabular=True))
        painter.drawText(
            QRectF(header.right() - 150, header.top(), 76, HEADER),
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
            f"{self.page_index + 1} / {len(self.pages)}",
        )
        self._arrow(painter, QRectF(header.right() - 70, header.top(), 32, HEADER), True)
        self._arrow(painter, QRectF(header.right() - 34, header.top(), 32, HEADER), False)

        painter.save()
        if self._page_anim is not None:  # เลื่อนเข้าเบาๆ ตามทิศที่เปลี่ยนหน้า
            start, direction = self._page_anim
            t = min((now - start) / PAGE_ANIM, 1.0)
            eased = 1.0 - (1.0 - t) ** 3
            painter.translate(direction * 40.0 * (1.0 - eased), 0)
            painter.setOpacity(0.4 + 0.6 * eased)
        if category == "table":
            self._paint_table(painter)
        else:
            self._paint_cards(painter, members)
        painter.restore()

    def _paint_table(self, painter: QPainter) -> None:
        cell = self.table_cell()
        hover = self.lab.hover
        held = set(self.lab.held.values())
        symbol_font = _font(cell * 0.36, 600)
        number_font = _font(max(cell * 0.19, 7.0), 500, tabular=True)
        top_left = Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop
        for element in ELEMENTS:
            rect = self.cell_rect(element)
            color = QColor(CATEGORIES[element.category][2])
            lit = element is hover or element.symbol in held or element is self.focus
            painter.setPen(QPen(ACCENT if lit else with_alpha(color, 0.35), 1.4 if lit else 1.0))
            painter.setBrush(with_alpha(color, 0.42 if lit else 0.16))
            painter.drawRoundedRect(rect, 4.0, 4.0)
            painter.setPen(PRIMARY if lit else with_alpha(PRIMARY, 0.88))
            painter.setFont(symbol_font)
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, element.symbol)
            if cell >= 30:
                painter.setFont(number_font)
                painter.setPen(with_alpha(PRIMARY, 0.5))
                painter.drawText(rect.adjusted(3, 1, -2, -2), top_left, str(element.number))
        content = self.content_rect()
        left = content.left() + (content.width() - cell * 18) / 2
        painter.setFont(_font(max(cell * 0.2, 7.0), 500))
        painter.setPen(DIM)
        for row, text in ((5, "57–71"), (6, "89–103")):
            box = QRectF(left + 2 * cell, content.top() + row * cell, cell, cell)
            painter.drawText(box, Qt.AlignmentFlag.AlignCenter, text)

    def _paint_cards(self, painter: QPainter, members: list[Element]) -> None:
        hover = self.lab.hover
        held = set(self.lab.held.values())
        for element in members:
            rect = self.cell_rect(element)
            color = QColor(CATEGORIES[element.category][2])
            lit = element is hover or element.symbol in held or element is self.focus
            painter.setPen(QPen(ACCENT if lit else with_alpha(color, 0.4), 1.4 if lit else 1.0))
            painter.setBrush(with_alpha(color, 0.30 if lit else 0.13))
            painter.drawRoundedRect(rect, 10, 10)
            info = atom_info(element)
            corner = rect.adjusted(10, 6, -10, 0)
            painter.setPen(with_alpha(PRIMARY, 0.55))
            painter.setFont(_font(12, 500, tabular=True))
            top = Qt.AlignmentFlag.AlignTop
            painter.drawText(corner, Qt.AlignmentFlag.AlignLeft | top, str(element.number))
            painter.drawText(corner, Qt.AlignmentFlag.AlignRight | top, element.mass_text)
            painter.setPen(PRIMARY)
            painter.setFont(_font(min(rect.height() * 0.32, 40), 600))
            symbol_box = rect.adjusted(0, -rect.height() * 0.12, 0, 0)
            painter.drawText(symbol_box, Qt.AlignmentFlag.AlignCenter, element.symbol)
            painter.setFont(_font(13, 600))
            name_box = QRectF(rect.left(), rect.bottom() - 46, rect.width(), 20)
            painter.drawText(name_box, Qt.AlignmentFlag.AlignCenter, element.name)
            painter.setFont(_font(11, 400))
            painter.setPen(DIM)
            sub_box = QRectF(rect.left(), rect.bottom() - 26, rect.width(), 18)
            painter.drawText(sub_box, Qt.AlignmentFlag.AlignCenter, element.thai or info.state_thai)

    def _paint_tabs(self, painter: QPainter) -> None:
        right = self.right_rect()
        half = right.width() / 2
        tabs = (("element", "Element · ธาตุ"), ("guide", "Guide · คำแนะนำการผสม"))
        for index, (key, label) in enumerate(tabs):
            rect = QRectF(right.left() + index * half, right.top(), half, TABS)
            active = self.panel == key
            painter.setPen(PRIMARY if active else DIM)
            painter.setFont(_font(13, 600 if active else 500))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, label)
            if active:
                painter.setPen(QPen(ACCENT, 2.0))
                painter.drawLine(
                    QPointF(rect.left() + 20, rect.bottom()),
                    QPointF(rect.right() - 20, rect.bottom()),
                )
        painter.setPen(QPen(QColor(255, 255, 255, 18), 1.0))
        painter.drawLine(
            QPointF(right.left(), right.top() + TABS), QPointF(right.right(), right.top() + TABS)
        )

    # ---- แท็บ Element: แบบจำลองอะตอม + รายละเอียด ------------------------------ #

    def _paint_element(self, painter: QPainter, now: float) -> None:
        panel = self.panel_rect()
        info = atom_info(self.focus)
        size = min(panel.height() - 64, panel.width() * 0.46)
        model = QRectF(panel.left(), panel.top(), size, size)
        paint_atom_model(painter, model, info, now)
        details = QRectF(model.right() + 14, panel.top(), panel.right() - model.right() - 14, size)
        self._paint_details(painter, details, info)
        note_top = model.bottom() + 18
        painter.setFont(_font(12, 400))
        painter.setPen(with_alpha(PRIMARY, 0.85))
        painter.drawText(
            QRectF(panel.left(), note_top, panel.width(), panel.bottom() - 32 - note_top),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop | Qt.TextFlag.TextWordWrap,
            info.note,
        )
        painter.setFont(_font(12, 600))
        for rect, compound in self._makes_rects(panel):
            level = QColor(DANGER_LEVELS[compound.danger][2])
            painter.setPen(QPen(with_alpha(level, 0.7), 1.0))
            painter.setBrush(with_alpha(level, 0.12))
            painter.drawRoundedRect(rect, 12, 12)
            painter.setPen(PRIMARY)
            label = compound.display if not compound.is_reaction else compound.name
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, label)

    def _paint_details(self, painter: QPainter, area: QRectF, info: AtomInfo) -> None:
        element = info.element
        name, thai, color = CATEGORIES[element.category]
        left = Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        y = area.top()

        def line(text: str, size: float, weight: int, pen, height: float) -> None:
            nonlocal y
            painter.setFont(_font(size, weight, tabular=True))
            painter.setPen(pen)
            painter.drawText(QRectF(area.left(), y, area.width(), height), left, text)
            y += height

        line(f"{element.symbol}  {element.name}", 19, 600, PRIMARY, 28)
        if element.thai:
            line(element.thai, 13, 400, DIM, 20)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(color))
        painter.drawEllipse(QPointF(area.left() + 5, y + 10), 4, 4)
        painter.setFont(_font(12, 500))
        painter.setPen(QColor(color))
        painter.drawText(
            QRectF(area.left() + 14, y, area.width() - 14, 20), left, f"{thai} · {name}"
        )
        y += 26
        line(f"Atomic number {element.number}  ·  {element.mass_text} u", 12, 500, PRIMARY, 20)
        line(f"p⁺ {info.protons}   n⁰ {info.neutrons}   e⁻ {info.electrons}", 12, 500, PRIMARY, 20)
        line("Shells " + " · ".join(str(n) for n in info.shells), 12, 500, PRIMARY, 20)
        line(info.configuration, 12, 500, ACCENT, 20)
        valence = "–" if info.valence is None else str(info.valence)
        line(f"{info.state_thai} · {info.block}-block · valence {valence}", 12, 400, DIM, 20)
        group = f" · group {element.group}" if element.group else ""
        line(f"period {element.period}{group}", 12, 400, DIM, 20)

    # ---- แท็บ Guide ------------------------------------------------------------- #

    def _paint_guide(self, painter: QPainter) -> None:
        panel = self.panel_rect()
        painter.setFont(_font(12, 500))
        for rect, kind in self._filter_rects(panel):
            active = kind == self.guide_filter
            painter.setPen(QPen(ACCENT if active else QColor(255, 255, 255, 40), 1.0))
            painter.setBrush(with_alpha(ACCENT, 0.18) if active else QColor(255, 255, 255, 8))
            painter.drawRoundedRect(rect, 13, 13)
            painter.setPen(ACCENT if active else DIM)
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, FILTER_LABELS[kind])
        rows, page, pages, area = self.guide_rows()
        hovered = None
        for hand in self.hands:
            point = QPointF(hand.x * self.width(), hand.y * self.height())
            if area.contains(point):
                hovered = int((point.y() - area.top()) // GUIDE_ROW)
        left_v = Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        for index, compound in enumerate(rows):
            rect = QRectF(area.left(), area.top() + index * GUIDE_ROW, area.width(), GUIDE_ROW - 4)
            _, level_thai, level_hex = DANGER_LEVELS[compound.danger]
            level = QColor(level_hex)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(255, 255, 255, 16 if index == hovered else 7))
            painter.drawRoundedRect(rect, 8, 8)
            painter.setBrush(level)
            painter.drawRoundedRect(
                QRectF(rect.left(), rect.top() + 8, 3, rect.height() - 16), 1.5, 1.5
            )
            badge = QRectF(rect.right() - 92, rect.top() + 6, 86, 18)
            painter.setPen(QPen(with_alpha(level, 0.8), 1.0))
            painter.setBrush(with_alpha(level, 0.14))
            painter.drawRoundedRect(badge, 9, 9)
            painter.setFont(_font(10, 600))
            painter.setPen(level)
            painter.drawText(badge, Qt.AlignmentFlag.AlignCenter, level_thai)
            text_left = rect.left() + 12
            painter.setFont(_font(12, 600))
            painter.setPen(PRIMARY)
            recipe = f"{compound.recipe}  →  {compound.name}"
            width = int(rect.width() - 112)
            elided = painter.fontMetrics().elidedText(recipe, Qt.TextElideMode.ElideRight, width)
            painter.drawText(QRectF(text_left, rect.top() + 4, width, 20), left_v, elided)
            painter.setFont(_font(11, 400))
            painter.setPen(DIM)
            detail = f"{compound.thai} — {compound.fact}"
            width = int(rect.width() - 20)
            elided = painter.fontMetrics().elidedText(detail, Qt.TextElideMode.ElideRight, width)
            painter.drawText(QRectF(text_left, rect.top() + 24, width, 18), left_v, elided)
        painter.setFont(_font(12, 500, tabular=True))
        painter.setPen(DIM)
        count = len(guide_entries(self.guide_filter))
        footer = QRectF(panel.left(), panel.bottom() - 26, panel.width() - 150, 26)
        painter.drawText(footer, left_v, f"{count} recipes · pinch one to mix it")
        painter.drawText(
            QRectF(panel.right() - 140, panel.bottom() - 26, 66, 26),
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
            f"{page + 1}/{pages}",
        )
        self._arrow(painter, QRectF(panel.right() - 70, panel.bottom() - 26, 32, 26), True)
        self._arrow(painter, QRectF(panel.right() - 34, panel.bottom() - 26, 32, 26), False)

    # ---- ขวด + ผลลัพธ์ ------------------------------------------------------------ #

    def _flask_path(self, rect: QRectF) -> QPainterPath:
        neck = rect.width() * 0.28
        cx = rect.center().x()
        neck_bottom = rect.top() + rect.height() * 0.3
        path = QPainterPath()
        path.moveTo(cx - neck / 2, rect.top())
        path.lineTo(cx - neck / 2, neck_bottom)
        path.lineTo(rect.left() + 8, rect.bottom() - 12)
        path.quadTo(rect.left(), rect.bottom(), rect.left() + 14, rect.bottom())
        path.lineTo(rect.right() - 14, rect.bottom())
        path.quadTo(rect.right(), rect.bottom(), rect.right() - 8, rect.bottom() - 12)
        path.lineTo(cx + neck / 2, neck_bottom)
        path.lineTo(cx + neck / 2, rect.top())
        return path

    def _paint_flask(self, painter: QPainter, now: float) -> None:
        rect = self.flask_rect()
        path = self._flask_path(rect)
        match = self.lab.last_match
        atoms = self.lab.flask.atoms
        level = min(0.15 + 0.025 * len(atoms), 0.62) if atoms else 0.0
        if level > 0.0:
            clip = QPainterPath()
            liquid_top = rect.bottom() - rect.height() * level
            clip.addRect(QRectF(rect.left(), liquid_top, rect.width(), rect.height()))
            color = QColor(match[0].color) if match else QColor(200, 210, 222)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(with_alpha(color, 0.32 if match else 0.12))
            painter.drawPath(path.intersected(clip))
        radius = 8.0
        columns = max(1, int((rect.width() - 20) // (radius * 2.2)))
        for index, symbol in enumerate(atoms):
            col, row = index % columns, index // columns
            x = rect.center().x() + (col - (columns - 1) / 2) * radius * 2.2
            y = rect.bottom() - radius - 6 - row * radius * 2.0
            x += 1.4 * math.sin(now * 2.1 + index * 1.7)
            y += 1.4 * math.cos(now * 1.7 + index * 2.3)
            paint_atom(painter, QPointF(x, y), symbol, radius)
        hairline_stroke(painter, path, PRIMARY, 1.4, alpha=0.75)
        if self._flash is not None:
            self._paint_flash(painter, rect, now)
        painter.setFont(_font(15, 600))
        painter.setPen(ACCENT if match else PRIMARY)
        if match:
            compound, multiple = match
            text = (f"{multiple} × " if multiple > 1 else "") + (
                compound.name if compound.is_reaction else compound.display
            )
        else:
            text = self.lab.flask.formula() if atoms else "Empty"
        label = QRectF(rect.left() - 30, rect.bottom() + 2, rect.width() + 60, 22)
        painter.drawText(label, Qt.AlignmentFlag.AlignCenter, text)

    def _paint_flash(self, painter: QPainter, rect: QRectF, now: float) -> None:
        """วงวาบตอนเกิดสาร — อันตรายมากขึ้น วงใหญ่ขึ้น (ระดับ 3 = แสงขาว + แสงไฟส้ม + จอสั่น)"""
        start, compound = self._flash  # type: ignore[misc]
        progress = min((now - start) / FLASH_SECONDS, 1.0)
        reach = (1.0 - (1.0 - progress) ** 3) * (40.0 + 30.0 * compound.danger)
        center = QPointF(rect.center().x(), rect.bottom() - rect.height() * 0.3)
        if compound.danger >= 3:
            color = QColor("#FFFFFF")
        elif compound.danger:
            color = QColor(DANGER_LEVELS[compound.danger][2])
        else:
            color = QColor(compound.color)
        ring = QPainterPath()
        ring.addEllipse(center, 24.0 + reach, 24.0 + reach)
        hairline_stroke(painter, ring, color, 2.0, alpha=(1.0 - progress) ** 1.5)
        if compound.energetic or compound.danger >= 2:
            glow = QRadialGradient(center, 50.0 + reach)
            tint = QColor("#FF6B4A") if compound.danger >= 3 else QColor("#FFFFFF")
            glow.setColorAt(0.0, with_alpha(tint, 0.4 * (1.0 - progress)))
            glow.setColorAt(1.0, QColor(255, 255, 255, 0))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(glow))
            painter.drawEllipse(center, 50.0 + reach, 50.0 + reach)

    def _paint_result(self, painter: QPainter) -> None:
        area = self.result_rect()
        if area.width() < 120:
            return
        top_left = Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop
        wrap = top_left | Qt.TextFlag.TextWordWrap
        match = self.lab.last_match
        if match is not None:
            compound, multiple = match
            _, level_thai, level_hex = DANGER_LEVELS[compound.danger]
            level = QColor(level_hex)
            painter.setPen(ACCENT)
            title = (f"{multiple} × " if multiple > 1 else "") + compound.display
            painter.setFont(_font(15 if compound.is_reaction else 24, 600))
            height = 44 if compound.is_reaction else 34
            painter.drawText(QRectF(area.left(), area.top(), area.width(), height), wrap, title)
            y = area.top() + height + 2
            painter.setPen(PRIMARY)
            painter.setFont(_font(14, 600))
            name = f"{compound.name} · {compound.thai}"
            painter.drawText(QRectF(area.left(), y, area.width(), 20), top_left, name)
            y += 22
            badge = QRectF(area.left(), y, 96, 20)
            painter.setPen(QPen(level, 1.0))
            painter.setBrush(with_alpha(level, 0.15))
            painter.drawRoundedRect(badge, 10, 10)
            painter.setFont(_font(11, 600))
            painter.setPen(level)
            painter.drawText(badge, Qt.AlignmentFlag.AlignCenter, level_thai)
            painter.setFont(_font(11, 400))
            painter.setPen(DIM)
            extra = "reaction" if compound.is_reaction else f"{compound.molar_mass:.2f} g/mol"
            extra_box = QRectF(badge.right() + 8, y, area.width() - 104, 20)
            painter.drawText(extra_box, Qt.AlignmentFlag.AlignVCenter, extra)
            y += 26
            painter.setFont(_font(12, 400))
            painter.setPen(with_alpha(PRIMARY, 0.88))
            painter.drawText(
                QRectF(area.left(), y, area.width(), area.bottom() - y), wrap, compound.fact
            )
            return
        hints = self.lab.flask.hints()
        painter.setFont(_font(12, 500))
        painter.setPen(DIM)
        title = "Almost there" if hints else "Drop atoms into the flask — or open the Guide tab"
        painter.drawText(QRectF(area.left(), area.top(), area.width(), 34), wrap, title)
        for index, hint in enumerate(hints):
            y = area.top() + 30 + index * 40
            painter.setFont(_font(13, 600))
            painter.setPen(PRIMARY)
            painter.drawText(QRectF(area.left(), y, area.width(), 20), top_left, hint.text)
            painter.setFont(_font(11, 400))
            painter.setPen(DIM)
            name = f"{hint.compound.name} · {hint.compound.thai}"
            painter.drawText(QRectF(area.left(), y + 18, area.width(), 18), top_left, name)

    def _paint_held(self, painter: QPainter) -> None:
        for hand in self.hands:
            symbol = self.lab.held.get(hand.handedness)
            if symbol:
                paint_atom(painter, self.to_px(hand.x, hand.y) + QPointF(0, -26), symbol, 16.0)


# --------------------------------------------------------------------------- #
# ตัวช่วยวาด
# --------------------------------------------------------------------------- #

PROTON = QColor("#E5484D")
NEUTRON = QColor("#8B9098")
ELECTRON = QColor("#7FD8FF")
MAX_NUCLEONS = 60  # ธาตุหนักวาดนิวคลีออนเท่านี้ (สัดส่วนโปรตอน:นิวตรอนเท่าของจริง)


def paint_atom(painter: QPainter, center: QPointF, symbol: str, radius: float) -> None:
    color = QColor(by_symbol(symbol).color)
    shade = QRadialGradient(center + QPointF(-radius * 0.35, -radius * 0.4), radius * 1.5)
    shade.setColorAt(0.0, color.lighter(125))
    shade.setColorAt(1.0, color.darker(140))
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QBrush(shade))
    painter.drawEllipse(center, radius, radius)
    painter.setFont(_font(radius * 0.95, 700))
    painter.setPen(QColor(20, 22, 26) if color.lightness() > 150 else PRIMARY)
    box = QRectF(center.x() - radius, center.y() - radius, radius * 2, radius * 2)
    painter.drawText(box, Qt.AlignmentFlag.AlignCenter, symbol)


def model_layout(info: AtomInfo, rect: QRectF) -> tuple[float, list[float], int]:
    """(รัศมีนิวเคลียส, รัศมีของแต่ละชั้นอิเล็กตรอน, จำนวนนิวคลีออนที่วาด)"""
    outer = min(rect.width(), rect.height()) / 2 - 6
    shown = min(info.protons + info.neutrons, MAX_NUCLEONS)
    nucleus = 7.0 + 2.2 * math.sqrt(shown)
    first = nucleus + 10
    count = len(info.shells)
    if count == 1:
        radii = [(first + outer) / 2]
    else:
        step = (outer - first) / (count - 1)
        radii = [first + step * i for i in range(count)]
    return nucleus, radii, shown


def paint_atom_model(painter: QPainter, rect: QRectF, info: AtomInfo, now: float) -> None:
    """แบบจำลองของโบร์: นิวเคลียส (โปรตอนแดง นิวตรอนเทา) + ชั้นอิเล็กตรอนที่หมุนรอบ"""
    center = rect.center()
    nucleus, radii, shown = model_layout(info, rect)
    for index, (count, radius) in enumerate(zip(info.shells, radii, strict=True)):
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(QColor(255, 255, 255, 34), 1.0))
        painter.drawEllipse(center, radius, radius)
        speed = (0.9 / (index + 1)) * (1 if index % 2 == 0 else -1)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(ELECTRON)
        for k in range(count):
            angle = now * speed + k * math.tau / count + index * 0.4
            point = QPointF(
                center.x() + math.cos(angle) * radius, center.y() + math.sin(angle) * radius
            )
            painter.drawEllipse(point, 2.6, 2.6)
    nucleons = info.protons + info.neutrons
    protons_shown = round(shown * info.protons / max(nucleons, 1))
    ball = max(2.4, min(4.5, nucleus / math.sqrt(max(shown, 1)) * 1.05))
    golden = math.pi * (3 - math.sqrt(5))  # เรียงแบบเกลียวทานตะวัน
    for k in range(shown):
        r = (nucleus - ball) * math.sqrt((k + 0.5) / shown)
        theta = k * golden
        point = QPointF(center.x() + math.cos(theta) * r, center.y() + math.sin(theta) * r)
        is_proton = (k * protons_shown) // shown != ((k + 1) * protons_shown) // shown
        color = PROTON if is_proton else NEUTRON
        shade = QRadialGradient(point + QPointF(-ball * 0.3, -ball * 0.3), ball * 1.4)
        shade.setColorAt(0.0, color.lighter(130))
        shade.setColorAt(1.0, color.darker(130))
        painter.setBrush(QBrush(shade))
        painter.drawEllipse(point, ball, ball)
    painter.setFont(_font(11, 500))
    painter.setPen(DIM)
    caption = "Bohr model"
    if nucleons > MAX_NUCLEONS:
        caption += f" · nucleus {shown}/{nucleons}"
    painter.drawText(
        QRectF(rect.left(), rect.bottom() - 2, rect.width(), 18),
        Qt.AlignmentFlag.AlignCenter,
        caption,
    )


def chemistry_controls(page: ChemistryPage) -> QWidget:
    bar = QWidget()
    layout = QHBoxLayout(bar)
    layout.setContentsMargins(0, 0, 0, 0)
    buttons = (
        ("‹ Previous", page.previous_page),
        ("Next ›", page.next_page),
        ("Periodic table", page.toggle_table),
        ("Guide", lambda: page.set_panel("guide" if page.panel == "element" else "element")),
        ("Remove last atom", page.undo),
        ("Empty flask", page.empty),
    )
    for text, slot in buttons:
        button = QPushButton(text)
        button.clicked.connect(lambda _=False, s=slot: s())
        layout.addWidget(button)
    layout.addStretch(1)
    return bar
