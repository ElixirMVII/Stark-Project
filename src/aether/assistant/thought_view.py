"""วาดโครงข่ายความคิด (:class:`ThoughtGraph`) ด้วย QPainter — ใช้ร่วมกันใน HUD, Studio (หน้า Mind) และหน้าต่างผู้ช่วย

- เส้นค่อยๆ งอกจากความคิดเดิมไปหาความคิดใหม่ มีประกายวิ่งที่ปลายเส้น
- โหนดเกิดใหม่มีวงแสงขยายออก หัวข้อ (hub) ใหญ่กว่า คำตอบเป็นสีเน้น
- ชี้/เลือกโหนด = โหนดนั้นกับเพื่อนบ้านสว่าง ที่เหลือจางลง + การ์ดข้อความเต็ม
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass

from PySide6.QtCore import QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import QBrush, QColor, QFont, QFontMetricsF, QPainter, QPen
from PySide6.QtWidgets import QWidget

from .. import ui_style
from ..hud.theme import sans_family
from .thoughts import BIRTH_SECONDS, Thought, ThoughtGraph

__all__ = ["ThoughtStyle", "ThoughtView", "paint_graph", "paint_thought_card"]

GROW_SECONDS = 0.45  # เวลาที่เส้นใหม่งอกจนถึงปลาย
HIT_RADIUS_PX = 26.0


@dataclass(frozen=True)
class ThoughtStyle:
    primary: QColor
    accent: QColor
    dim: QColor
    family: str = ""

    @classmethod
    def default(cls) -> ThoughtStyle:
        return cls(QColor("#F2F4F7"), QColor(ui_style.ACCENT), QColor(ui_style.TEXT_DIM))

    def font(self, size: float, weight: int = 400) -> QFont:
        font = QFont(self.family or sans_family())
        font.setPixelSize(max(1, round(size)))
        font.setWeight(QFont.Weight(weight))
        return font


def _alpha(color: QColor, alpha: float) -> QColor:
    out = QColor(color)
    out.setAlphaF(max(0.0, min(1.0, alpha)))
    return out


def _elide(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def paint_graph(
    painter: QPainter,
    graph: ThoughtGraph,
    rect: QRectF,
    now: float,
    style: ThoughtStyle,
    *,
    labels: bool = True,
    focus: int | None = None,
    compact: bool = False,
) -> None:
    """วาดโครงข่ายลงใน ``rect`` (ไม่วาดพื้นหลัง) — ``focus`` = id โหนดที่ชี้/เลือกอยู่"""
    if not graph.nodes or rect.isEmpty():
        return
    painter.save()
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    area = (rect.left(), rect.top(), rect.width(), rect.height())
    points = {
        n.id: QPointF(*graph.to_screen(n.x, n.y, *area, stretch=compact)) for n in graph.nodes
    }
    lit = {focus, *graph.neighbors(focus)} if focus is not None else None
    size = 0.7 if compact else 1.0
    newest = graph.nodes[-1].id

    for edge in graph.edges:
        a, b = points[edge.a], points[edge.b]
        grow = min(1.0, max(0.0, (now - edge.born) / GROW_SECONDS))
        if grow <= 0.0:
            continue
        tip = a + (b - a) * grow
        faded = lit is not None and not (edge.a in lit and edge.b in lit)
        if edge.kind == "link":
            pen = QPen(_alpha(style.accent, 0.10 if faded else 0.32), 1.0 * size)
            pen.setStyle(Qt.PenStyle.DashLine)
        elif edge.kind == "answer":
            pen = QPen(_alpha(style.accent, 0.15 if faded else 0.6), 1.4 * size)
        elif edge.kind == "hub":
            pen = QPen(_alpha(style.primary, 0.15 if faded else 0.5), 1.6 * size)
        else:
            pen = QPen(_alpha(style.primary, 0.10 if faded else 0.3), 1.2 * size)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        painter.drawLine(a, tip)
        if grow < 1.0:  # ประกายวิ่งที่ปลายเส้นที่กำลังงอก
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(_alpha(style.accent, 0.9))
            painter.drawEllipse(tip, 2.6 * size, 2.6 * size)

    for node in graph.nodes:
        point = points[node.id]
        age = now - node.born
        appear = min(1.0, max(0.0, age / 0.25))
        if appear <= 0.0:
            continue
        radius = (3.5 + 6.0 * node.weight) * size * (0.6 + 0.4 * appear)
        faded = lit is not None and node.id not in lit
        alpha = (0.35 if faded else 1.0) * appear
        if age < BIRTH_SECONDS:  # วงแสงตอนเกิด
            t = age / BIRTH_SECONDS
            painter.setPen(QPen(_alpha(style.accent, (1.0 - t) * 0.7), 1.2))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(point, radius + 14.0 * t * size, radius + 14.0 * t * size)
        if node.kind == "answer":
            glow = 0.25 + 0.1 * math.sin(now * 3.0)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(_alpha(style.accent, glow * alpha))
            painter.drawEllipse(point, radius * 2.0, radius * 2.0)
            painter.setBrush(_alpha(style.accent, alpha))
            painter.drawEllipse(point, radius, radius)
        elif node.kind == "question":
            painter.setBrush(_alpha(QColor(20, 22, 26), alpha))
            painter.setPen(QPen(_alpha(style.primary, 0.9 * alpha), 1.6 * size))
            painter.drawEllipse(point, radius, radius)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(_alpha(style.primary, 0.9 * alpha))
            painter.drawEllipse(point, radius * 0.35, radius * 0.35)
        else:
            base = 0.95 if node.kind == "hub" else 0.62
            if node.id == newest and node.kind != "answer":
                base = min(1.0, base + 0.25 * (0.5 + 0.5 * math.sin(now * 6.0)))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(_alpha(style.primary, base * alpha))
            painter.drawEllipse(point, radius, radius)
        if node.id == focus:
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(_alpha(style.accent, 0.95), 1.4))
            painter.drawEllipse(point, radius + 5.0, radius + 5.0)

    if labels and not compact:
        _paint_labels(painter, graph, points, rect, style, lit, newest)
    painter.restore()


def _paint_labels(painter, graph, points, rect, style, lit, newest) -> None:
    painter.setFont(style.font(11, 500))
    metrics = painter.fontMetrics()
    for node in graph.nodes:
        show = node.kind in ("hub", "question", "answer") or node.id == newest
        if not show or (lit is not None and node.id not in lit):
            continue
        if node.kind == "answer":
            text = "Answer"
        elif node.kind == "question":
            text = _elide(node.text, 36)
        else:
            text = _elide(node.text, 30 if node.kind == "hub" else 42)
        point = points[node.id]
        width = metrics.horizontalAdvance(text) + 16
        box = QRectF(point.x() - width / 2, point.y() + 12 + 6.0 * node.weight, width, 20)
        box.moveLeft(max(rect.left(), min(box.left(), rect.right() - width)))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(18, 20, 24, 200))
        painter.drawRoundedRect(box, 10, 10)
        color = style.accent if node.kind == "answer" else style.primary
        painter.setPen(_alpha(color, 0.92 if node.kind != "thought" else 0.75))
        painter.drawText(box, Qt.AlignmentFlag.AlignCenter, text)


def paint_thought_card(
    painter: QPainter, node: Thought, anchor: QPointF, bounds: QRectF, style: ThoughtStyle
) -> QRectF:
    """การ์ดข้อความเต็มของโหนดข้างจุด ``anchor`` (ไม่ล้น ``bounds``) คืนกรอบที่วาด"""
    label = {"question": "Question", "hub": "Topic", "thought": "Thought", "answer": "Answer"}
    width = min(300.0, bounds.width() - 16)
    flags = int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop | Qt.TextFlag.TextWordWrap)
    metrics = QFontMetricsF(style.font(12, 400))
    text_rect = metrics.boundingRect(QRectF(0, 0, width - 24, 1000), flags, node.text)
    height = min(220.0, text_rect.height() + 40)
    x = anchor.x() + 16 if anchor.x() + 16 + width < bounds.right() else anchor.x() - 16 - width
    y = max(bounds.top() + 8, min(anchor.y() - height / 2, bounds.bottom() - height - 8))
    card = QRectF(max(bounds.left() + 8, x), y, width, height)
    painter.save()
    painter.setPen(QPen(QColor(255, 255, 255, 30), 1.0))
    painter.setBrush(QColor(24, 27, 32, 242))
    painter.drawRoundedRect(card, 10, 10)
    painter.setFont(style.font(10, 600))
    painter.setPen(style.accent if node.kind == "answer" else style.dim)
    painter.drawText(
        card.adjusted(12, 8, -12, 0), int(Qt.AlignmentFlag.AlignLeft), label[node.kind]
    )
    painter.setFont(style.font(12, 400))
    painter.setPen(style.primary)
    body = card.adjusted(12, 26, -12, -10)
    painter.setClipRect(body)
    painter.drawText(body, flags, node.text)
    painter.restore()
    return card


class ThoughtView(QWidget):
    """วิดเจ็ตโครงข่ายความคิด: ชี้เมาส์ = อ่านความคิด, ลากโหนดได้ — ขยับเองด้วย timer ระหว่างที่ยังมีชีวิต"""

    def __init__(
        self, parent: QWidget | None = None, *, clock: Callable[[], float] = time.perf_counter
    ) -> None:
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setMinimumSize(240, 200)
        self._clock = clock
        self.graph = ThoughtGraph()
        self.style_ = ThoughtStyle.default()
        self.hovered: int | None = None
        self._last = clock()
        self._energy = 0.0
        self._timer = QTimer(self)
        self._timer.setInterval(16)
        self._timer.timeout.connect(self.tick)

    # ---- API ------------------------------------------------------------ #

    def begin(self, question: str) -> None:
        self.graph.reset(question, self._clock())
        self._wake()

    def add_thinking(self, piece: str) -> None:
        self.graph.feed(piece, self._clock())
        self._wake()

    def finish(self, answer: str) -> None:
        self.graph.add_answer(answer, self._clock())
        self._wake()

    def clear(self) -> None:
        self.graph.reset("", self._clock())
        self.update()

    def tick(self) -> None:
        now = self._clock()
        self._energy = self.graph.step(now - self._last)
        self._last = now
        self.update()
        if not self.graph.active(now, self._energy):
            self._timer.stop()

    # ---- internal ------------------------------------------------------- #

    def _wake(self) -> None:
        self._last = self._clock()
        if not self._timer.isActive():
            self._timer.start()
        self.update()

    def _area(self) -> QRectF:
        return QRectF(self.rect()).adjusted(12, 12, -12, -28)

    def node_at(self, x: float, y: float) -> Thought | None:
        area = self._area()
        gx, gy = self.graph.from_screen(x, y, area.left(), area.top(), area.width(), area.height())
        scale = self.graph.scales(area.width(), area.height())[0]
        return self.graph.nearest(gx, gy, HIT_RADIUS_PX / scale)

    def mouseMoveEvent(self, event) -> None:
        position = event.position()
        if self.graph.dragging is not None:
            area = self._area()
            self.graph.drag_to(
                *self.graph.from_screen(
                    position.x(), position.y(), area.left(), area.top(), area.width(), area.height()
                )
            )
            self._wake()
            return
        node = self.node_at(position.x(), position.y())
        hovered = node.id if node is not None else None
        if hovered != self.hovered:
            self.hovered = hovered
            self.update()

    def mousePressEvent(self, event) -> None:
        node = self.node_at(event.position().x(), event.position().y())
        if node is not None:
            self.graph.grab(node.id)
            self.hovered = node.id
            self._wake()

    def mouseReleaseEvent(self, event) -> None:
        self.graph.release()
        self._wake()

    def leaveEvent(self, event) -> None:
        self.hovered = None
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor(ui_style.SURFACE)))
        painter.drawRoundedRect(QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5), 8, 8)
        area = self._area()
        now = self._clock()
        if not self.graph.nodes:
            painter.setPen(self.style_.dim)
            painter.setFont(self.style_.font(12))
            painter.drawText(
                area,
                int(Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap),
                "โครงข่ายความคิดของ Claude จะงอกที่นี่ระหว่างที่คิด",
            )
        roomy = area.height() >= 300 and area.width() >= 360  # เล็กกว่านี้ป้ายจะทับกัน: ชี้เพื่ออ่านแทน
        paint_graph(painter, self.graph, area, now, self.style_, focus=self.hovered, labels=roomy)
        if self.hovered is not None and self.hovered < len(self.graph.nodes):
            node = self.graph.nodes[self.hovered]
            point = QPointF(
                *self.graph.to_screen(
                    node.x, node.y, area.left(), area.top(), area.width(), area.height()
                )
            )
            paint_thought_card(painter, node, point, QRectF(self.rect()), self.style_)
        painter.setFont(self.style_.font(10))
        painter.setPen(self.style_.dim)
        painter.drawText(
            QRectF(12, self.height() - 24, self.width() - 24, 18),
            int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
            graph_caption(self.graph),
        )
        painter.end()


def graph_caption(graph: ThoughtGraph) -> str:
    thoughts = graph.thoughts
    if not thoughts:
        return "สรุปความคิดจาก Claude API · เส้นประ = ความคิดที่พูดถึงเรื่องเดียวกัน"
    hubs = sum(1 for n in thoughts if n.kind == "hub")
    links = sum(1 for e in graph.edges if e.kind == "link")
    return f"{len(thoughts)} ความคิด · {hubs} หัวข้อ · {links} เส้นเชื่อมเรื่องเดียวกัน · ชี้เพื่ออ่าน ลากได้"
