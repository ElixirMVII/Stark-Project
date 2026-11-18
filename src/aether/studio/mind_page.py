"""หน้า Mind ของ Studio — ถาม Claude แล้วดู "โครงข่ายความคิด" งอกขึ้นเต็มจอระหว่างที่คิด

- โหนด = ความคิดหนึ่งประโยคจากสรุปความคิดที่ API ส่งมา (``thinking.display = "summarized"``) — ไม่ใช่เซลล์ประสาทจริง
- จีบที่โหนด = หยิบลากได้ (โครงข่ายดีดกลับเหมือนสปริง) ปล่อย = เลือกโหนดนั้น อ่านข้อความเต็มในการ์ด
- ชี้ค้างไว้ (ไม่จีบ) = ไฮไลต์โหนดกับความคิดที่เชื่อมกัน · ✊ ค้าง = ล้าง
"""

from __future__ import annotations

import time
from collections.abc import Callable

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QLineEdit, QPushButton, QWidget

from ..assistant import prompts
from ..assistant.thought_view import (
    HIT_RADIUS_PX,
    ThoughtStyle,
    graph_caption,
    paint_graph,
    paint_thought_card,
)
from ..assistant.thoughts import ThoughtGraph
from .claude_ui import StudioAnswer
from .input import HoldDetector
from .pages import DIM, PRIMARY, StudioPage, _font

__all__ = ["DEPTHS", "MindPage", "mind_controls"]

# (ชื่อปุ่ม, effort, max_tokens) — คิดลึก = โครงข่ายใหญ่ขึ้น แต่ช้าและแพงขึ้น
DEPTHS = (
    ("Quick · low", "low", 4096),
    ("Think · medium", "medium", 8000),
    ("Deep · high", "high", 16000),
)
GRAPH_SHARE = 0.64
PAD = 16.0


class MindPage(StudioPage):
    title = "Mind"
    hint = (
        "Ask Claude below and watch its thinking grow into a network  ·  pinch a node to drag it, "
        "release to read it  ·  point at a node to light up its links  ·  hold ✊ to clear"
    )

    def __init__(self, parent=None, *, clock: Callable[[], float] = time.perf_counter) -> None:
        super().__init__(parent, clock=clock)
        self.claude = None
        self.on_answer: Callable[[str, str], None] | None = None
        self.graph = ThoughtGraph()
        self.answer = StudioAnswer(clock)
        self.style_ = ThoughtStyle.default()
        self.depth = 1
        self.hovered: int | None = None
        self.selected: int | None = None
        self.question = ""
        self._grabbing: str | None = None  # มือที่กำลังลากโหนด
        self._was_pinching: dict[str, bool] = {}
        self._last_tick = clock()
        self._energy = 0.0
        self.clear_hold = HoldDetector(1.0)

    # ---- layout ---------------------------------------------------------- #

    def graph_rect(self) -> QRectF:
        return QRectF(
            PAD, PAD, self.width() * GRAPH_SHARE - 1.5 * PAD, self.height() - 2 * PAD - 22
        )

    def side_rect(self) -> QRectF:
        graph = self.graph_rect()
        return QRectF(
            graph.right() + PAD,
            PAD,
            self.width() - graph.right() - 2 * PAD,
            self.height() - 2 * PAD,
        )

    def _area(self) -> tuple[float, float, float, float]:
        rect = self.graph_rect().adjusted(20, 20, -20, -20)
        return rect.left(), rect.top(), rect.width(), rect.height()

    def screen_point(self, node_id: int) -> QPointF:
        node = self.graph.nodes[node_id]
        return QPointF(*self.graph.to_screen(node.x, node.y, *self._area()))

    def node_at_px(self, x: float, y: float):
        left, top, width, height = self._area()
        gx, gy = self.graph.from_screen(x, y, left, top, width, height)
        scale = self.graph.scales(width, height)[0]
        return self.graph.nearest(gx, gy, HIT_RADIUS_PX / scale)

    # ---- Claude ---------------------------------------------------------- #

    def set_depth(self, index: int) -> None:
        self.depth = max(0, min(index, len(DEPTHS) - 1))

    def ask(self, question: str) -> bool:
        question = question.strip()
        if not question:
            self.toast("Type a question first")
            return False
        if self.claude is None:
            self.toast("Claude is not available")
            return False
        if self.answer.busy:
            self.toast("Claude is still thinking")
            return False
        now = self._clock()
        self.question = question
        self.graph.reset(question, now)
        self.hovered = self.selected = None
        _label, effort, max_tokens = DEPTHS[self.depth]
        self.answer.begin("Claude")
        system, content = prompts.free_question(question)

        def thinking(piece: str) -> None:
            self.graph.feed(piece, self._clock())

        def piece(text: str) -> None:
            if self.answer.state == "thinking":
                self.graph.flush(self._clock())  # เริ่มตอบ = คิดเสร็จแล้ว
            self.answer.append(text)

        def done(text: str) -> None:
            self.graph.add_answer(text, self._clock())
            self.answer.finish(text)
            if not self.graph.thoughts:
                self.toast("Claude answered without extended thinking — try Think or Deep")
            if self.on_answer is not None:
                self.on_answer(f"[Mind] {question}", text)
            self.update()

        def failed(message: str) -> None:
            self.answer.fail(message)
            self.update()

        self.claude.stream(
            system,
            content,
            on_text=piece,
            on_thinking=thinking,
            on_result=done,
            on_error=failed,
            effort=effort,
            max_tokens=max_tokens,
        )
        self.update()
        return True

    def clear(self) -> None:
        if self.answer.busy:
            return
        self.graph.reset("", self._clock())
        self.answer.dismiss()
        self.question = ""
        self.hovered = self.selected = None
        self.update()

    # ---- มือ / เมาส์ ------------------------------------------------------- #

    def on_hands(self, now: float) -> None:
        main = self.hands[0] if self.hands else None
        if self.clear_hold.update(main is not None and main.gesture == "Closed_Fist", now):
            self.clear()
            self.toast("Cleared")
        present = set()
        hovered = None
        for hand in self.hands:
            present.add(hand.handedness)
            x, y = hand.x * self.width(), hand.y * self.height()
            was = self._was_pinching.get(hand.handedness, False)
            if self._grabbing == hand.handedness:
                if hand.pinch:
                    left, top, width, height = self._area()
                    self.graph.drag_to(*self.graph.from_screen(x, y, left, top, width, height))
                else:
                    released = self.graph.release()
                    self._grabbing = None
                    self.selected = released
            elif hand.pinch and not was and self._grabbing is None:
                node = self.node_at_px(x, y)
                if node is not None:
                    self.graph.grab(node.id)
                    self._grabbing = hand.handedness
                    self.selected = node.id
                elif self.side_rect().contains(QPointF(x, y)) is False:
                    self.selected = None  # จีบที่ว่าง = เลิกเลือก
            if not hand.pinch and hovered is None:
                node = self.node_at_px(x, y)
                hovered = node.id if node is not None else None
            self._was_pinching[hand.handedness] = hand.pinch
        if self._grabbing is not None and self._grabbing not in present:
            self.graph.release()  # มือหายระหว่างลาก
            self._grabbing = None
        for name in list(self._was_pinching):
            if name not in present:
                del self._was_pinching[name]
        self.hovered = hovered

    def tick(self, now: float) -> None:
        self._energy = self.graph.step(now - self._last_tick)
        self._last_tick = now
        super().tick(now)
        if self.graph.nodes or self.answer.busy:
            self.update()

    # ---- paint ----------------------------------------------------------- #

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.paint_background(painter, grid=False)
        now = self._clock()
        graph_rect = self.graph_rect()
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(255, 255, 255, 6))
        painter.drawRoundedRect(graph_rect, 14, 14)
        flags = int(Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap)
        if not self.graph.nodes:
            painter.setFont(_font(14, 400))
            painter.setPen(DIM)
            reason = self.claude.ready_reason() if self.claude is not None else None
            if self.claude is None:
                reason = "Claude is not connected in this window"
            painter.drawText(
                graph_rect.adjusted(40, 40, -40, -40),
                flags,
                reason
                or "Ask Claude something below — its thinking grows here as a network of ideas.\n"
                "Bigger nodes are topics, dashed lines join thoughts about the same thing.",
            )
        focus = self.selected if self.selected is not None else self.hovered
        left, top, width, height = self._area()
        paint_graph(
            painter, self.graph, QRectF(left, top, width, height), now, self.style_, focus=focus
        )
        if focus is not None and focus < len(self.graph.nodes):
            paint_thought_card(
                painter, self.graph.nodes[focus], self.screen_point(focus), graph_rect, self.style_
            )
        painter.setFont(_font(11, 400))
        painter.setPen(DIM)
        painter.drawText(
            QRectF(graph_rect.left() + 4, graph_rect.bottom() + 4, graph_rect.width(), 18),
            int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
            graph_caption(self.graph),
        )
        self._paint_side(painter)
        self.paint_cursors(painter, self.clear_hold.progress)
        self.paint_toast(painter)
        painter.end()

    def _paint_side(self, painter: QPainter) -> None:
        side = self.side_rect()
        painter.setFont(_font(12, 600))
        painter.setPen(DIM)
        painter.drawText(
            QRectF(side.left(), side.top(), side.width(), 20),
            int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
            f"Depth: {DEPTHS[self.depth][0]}",
        )
        top = side.top() + 28
        if self.question:
            painter.setFont(_font(14, 600))
            painter.setPen(PRIMARY)
            flags = int(
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop | Qt.TextFlag.TextWordWrap
            )
            bounds = painter.boundingRect(
                QRectF(side.left(), top, side.width(), 200), flags, self.question
            )
            painter.drawText(
                QRectF(side.left(), top, side.width(), min(bounds.height(), 120)),
                flags,
                self.question,
            )
            top += min(bounds.height(), 120) + 14
        self.answer.paint(painter, QRectF(side.left(), top, side.width(), side.bottom() - top))


def mind_controls(page: MindPage) -> QWidget:
    bar = QWidget()
    layout = QHBoxLayout(bar)
    layout.setContentsMargins(0, 0, 0, 0)
    question = QLineEdit()
    question.setPlaceholderText("Ask Claude anything — e.g. why is the sky blue?")
    depth = QComboBox()
    for label, _effort, _tokens in DEPTHS:
        depth.addItem(label)
    depth.setCurrentIndex(page.depth)
    depth.setToolTip("Deeper = bigger network, slower and costs more")
    depth.currentIndexChanged.connect(page.set_depth)
    ask = QPushButton("Ask")
    clear = QPushButton("Clear")

    def send() -> None:
        if page.ask(question.text()):
            question.clear()

    ask.clicked.connect(send)
    question.returnPressed.connect(send)
    clear.clicked.connect(page.clear)
    layout.addWidget(question, 1)
    layout.addWidget(depth)
    layout.addWidget(ask)
    layout.addWidget(clear)
    bar.question = question  # type: ignore[attr-defined]
    bar.depth = depth  # type: ignore[attr-defined]
    bar.ask = ask  # type: ignore[attr-defined]
    return bar
