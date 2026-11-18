"""โครงข่ายความคิดของ Claude: แบ่งความคิด/คำสำคัญ/เส้นเชื่อม/จัดวาง, stream สรุปความคิดจาก API,
หน้า Mind ของ Studio, แถบโครงข่ายบนการ์ด HUD และหน้าต่างผู้ช่วย"""

from __future__ import annotations

import math

import pytest
from PySide6.QtCore import QPointF, QRectF
from PySide6.QtGui import QGuiApplication
from test_claude import FakeClient, FakeService, make_service

from aether.assistant.service import ClaudeError
from aether.assistant.tasks import ClaudeBridge
from aether.assistant.thoughts import ThoughtGraph, keywords, split_thoughts
from aether.config import ClaudeConfig, Config
from aether.gestures import EventKind, GestureEvent
from aether.studio.input import StudioHand
from aether.studio.mind_page import DEPTHS
from aether.studio.window import StudioWindow

THINKING = """**Understanding the question**
The user asks why the sky is blue. This is about sunlight and the atmosphere.

**Recalling Rayleigh scattering**
Air molecules scatter short wavelengths through Rayleigh scattering.
Rayleigh scattering grows with one over wavelength to the fourth power, so blue light scatters more.

**Planning the answer**
Explain it in Thai with a simple analogy."""


def chunks(text: str, size: int = 17) -> list[str]:
    return [text[i : i + size] for i in range(0, len(text), size)]


# ---- แบ่งความคิด / คำสำคัญ ------------------------------------------------------- #


def test_split_titles_short_and_long_sentences():
    parts = split_thoughts("**Plan**\nOk. Short one here.\n- " + "word " * 80)
    assert parts[0] == ("Plan", True)
    assert parts[1] == ("Ok. Short one here.", False)  # สั้นเกินไปรวมกับประโยคถัดไป
    long_parts = [text for text, title in parts[2:]]
    assert len(long_parts) >= 2 and all(len(t) <= 221 for t in long_parts)
    assert not long_parts[0].startswith("-")  # ตัด bullet markdown ออก


def test_keywords_skip_common_words_and_keep_codes():
    words = keywords("The Windows errors 0x80070005 should mention NaCl and H2O permissions")
    assert {"window", "error", "0x80070005", "nacl", "h2o", "permission", "mention"} <= words
    assert "the" not in words and "should" not in words
    assert keywords("ท้องฟ้าเป็นสีฟ้า") == frozenset()  # ภาษาไทยล้วน: ไม่มีคำละติน


# ---- สร้างโครงข่าย ---------------------------------------------------------------- #


def build(text: str = THINKING, size: int = 17) -> ThoughtGraph:
    graph = ThoughtGraph()
    graph.reset("Why is the sky blue?", 0.0)
    for piece in chunks(text, size):
        graph.feed(piece, 0.0)
    graph.flush(0.0)
    return graph


def test_streamed_chunks_build_the_same_graph_as_one_piece():
    a, b = build(size=5), build(size=10_000)
    assert [(n.kind, n.text) for n in a.nodes] == [(n.kind, n.text) for n in b.nodes]
    assert a.thinking == THINKING


def test_hubs_chains_and_links():
    graph = build()
    kinds = [n.kind for n in graph.nodes]
    assert kinds[0] == "question" and kinds.count("hub") == 3
    assert [n.text for n in graph.nodes if n.kind == "hub"] == [
        "Understanding the question",
        "Recalling Rayleigh scattering",
        "Planning the answer",
    ]
    edge_kinds = {(e.a, e.b): e.kind for e in graph.edges}
    hubs = [n.id for n in graph.nodes if n.kind == "hub"]
    assert edge_kinds[(0, hubs[0])] == "hub" and edge_kinds[(hubs[0], hubs[1])] == "hub"
    rayleigh = [n.id for n in graph.thoughts if n.kind == "thought" and "rayleigh" in n.keywords]
    assert len(rayleigh) == 2
    assert edge_kinds.get((rayleigh[0], rayleigh[1])) in ("chain", "link")
    links = [e for e in graph.edges if e.kind == "link"]
    for edge in links:  # เส้นประต้องมีคำสำคัญร่วมกันจริง
        a, b = graph.nodes[edge.a], graph.nodes[edge.b]
        shared = a.keywords & b.keywords
        assert len(shared) >= 2 or any(len(w) >= 7 for w in shared)
    assert graph.neighbors(hubs[1]) >= {hubs[0], hubs[2]}


def test_answer_node_once_and_cap_on_nodes():
    graph = build()
    answer = graph.add_answer("Rayleigh scattering makes the sky blue\nmore lines", 1.0)
    assert answer is not None and answer.kind == "answer" and answer.text.startswith("Rayleigh")
    assert any(e.kind == "answer" and e.b == answer.id for e in graph.edges)
    assert graph.add_answer("again", 2.0) is None and graph.answer is answer
    small = ThoughtGraph(max_nodes=3)
    small.reset("", 0.0)
    small.feed("".join(f"Thought number {i} about something long enough. " for i in range(10)), 0)
    small.flush(0)
    assert len(small.thoughts) == 3 and "number 9" in small.thoughts[-1].text
    assert small.root is None


def test_layout_settles_fits_view_and_maps_back():
    graph = build()
    graph.add_answer("Rayleigh scattering", 0.0)
    first = graph.settle(60)
    last = graph.settle(400)
    assert last < first and last < 0.01
    assert all(math.isfinite(n.x) and math.isfinite(n.y) for n in graph.nodes)
    assert graph.root.x == 0.0 and graph.root.y == 0.0  # คำถามตรึงไว้กลาง
    left, top, right, bottom = graph.bounds()
    assert graph.view.span_x >= right - left and graph.view.span_y >= bottom - top
    for node in graph.nodes:  # ทุกโหนดอยู่ในกรอบที่วาด
        x, y = graph.to_screen(node.x, node.y, 0, 0, 800, 600)
        assert -1 <= x <= 801 and -1 <= y <= 601
        back = graph.from_screen(x, y, 0, 0, 800, 600)
        assert back == pytest.approx((node.x, node.y), abs=1e-6)
    sx, sy = graph.scales(500, 80, stretch=True)
    assert sx >= sy  # แถบเล็กบน HUD: ยืดแนวนอนได้


def test_drag_pins_node_and_nearest_and_active():
    graph = build()
    node = graph.thoughts[2]
    assert graph.nearest(node.x + 0.01, node.y, 0.2) is node
    assert graph.nearest(99, 99, 0.2) is None
    graph.grab(node.id)
    graph.drag_to(3.0, -2.0)
    graph.step(0.03)
    assert (node.x, node.y) == (3.0, -2.0) and graph.dragging == node.id
    assert graph.active(100.0)
    assert graph.release() == node.id and graph.dragging is None
    graph.settle(600)
    assert not graph.active(100.0, 0.0)
    assert graph.active(0.1)  # เพิ่งเกิด


# ---- stream สรุปความคิดจาก API ----------------------------------------------------- #


def test_service_requests_summarized_thinking_and_delivers_it():
    client = FakeClient(thinking=("Plan. ", "Think."))
    service, _, _ = make_service(client)
    got: list[str] = []
    text = service.stream_text("s", "q", on_thinking=got.append, effort="high")
    assert text == "สวัสดี" and got == ["Plan. ", "Think."]
    kwargs = client.calls[-1][1]
    assert kwargs["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert kwargs["output_config"] == {"effort": "high"}
    service.stream_text("s", "q")
    assert "thinking" not in client.calls[-1][1]  # ไม่ขอ = ไม่ส่งพารามิเตอร์
    with pytest.raises(ClaudeError):
        service.stream_text("s", "q", effort="ultra")
    off, client_off, _ = make_service(
        FakeClient(thinking=("x. ",)), ClaudeConfig(enabled=True, show_thinking=False)
    )
    got.clear()
    off.stream_text("s", "q", on_thinking=got.append)
    assert got == [] and "thinking" not in client_off.calls[-1][1]
    assert Config().claude.show_thinking is True


def test_bridge_passes_thinking_and_effort(qapp):
    service = FakeService(thinking=("a. ", "b."))
    bridge = ClaudeBridge(service, sync=True)
    got: list[str] = []
    bridge.stream("s", "q", on_thinking=got.append, effort="medium")
    assert got == ["a. ", "b."] and service.efforts == ["medium"]
    bridge.stream("s", "q")
    assert service.efforts[-1] is None


# ---- Studio › Mind ------------------------------------------------------------------ #


@pytest.fixture
def mind(qapp):
    clock = {"t": 0.0}
    service = FakeService(thinking=chunks(THINKING), pieces=("ฟ้า", "เป็นสีฟ้า"))
    history: list[tuple[str, str]] = []
    window = StudioWindow(
        Config(),
        audio=False,
        clock=lambda: clock["t"],
        claude=ClaudeBridge(service, sync=True),
        on_answer=lambda q, a: history.append((q, a)),
    )
    window.resize(1180, 780)
    window.show()
    window.set_page(window.pages.index(window.mind))
    yield window, service, clock, history
    window.close()


def hand(page, x_px: float, y_px: float, *, pinch=False, gesture="None") -> StudioHand:
    return StudioHand(
        "right", x_px / page.width(), y_px / page.height(), pinch, 1.0 if pinch else 0.0, gesture
    )


def advance(page, clock, seconds: float) -> None:
    for _ in range(int(seconds * 60)):
        clock["t"] += 1 / 60
        page.tick(clock["t"])


def test_mind_page_grows_network_from_thinking(mind):
    window, service, clock, history = mind
    page = window.mind
    assert window.current is page and "network" in window.hint.text()
    window.grab()
    assert not page.ask("   ")
    page.set_depth(2)
    assert page.ask("Why is the sky blue?")
    assert service.efforts[-1] == DEPTHS[2][1]
    assert page.graph.root.text == "Why is the sky blue?"
    assert len([n for n in page.graph.nodes if n.kind == "hub"]) == 3
    assert page.graph.answer is not None and page.answer.text == "ฟ้าเป็นสีฟ้า"
    assert history == [("[Mind] Why is the sky blue?", "ฟ้าเป็นสีฟ้า")]
    advance(page, clock, 2.0)
    window.grab()
    bar = window.controls.widget(window.pages.index(page))
    bar.depth.setCurrentIndex(0)
    assert page.depth == 0
    bar.question.setText("และตอนเย็นล่ะ?")
    bar.ask.click()
    assert page.question == "และตอนเย็นล่ะ?" and service.efforts[-1] == "low"


def test_mind_page_pinch_drags_and_selects_nodes(mind):
    window, _service, clock, _ = mind
    page = window.mind
    page.ask("Why is the sky blue?")
    advance(page, clock, 3.0)
    target = page.graph.thoughts[1]
    point = page.screen_point(target.id)
    page.set_hands((hand(page, point.x(), point.y()),), clock["t"])
    assert page.hovered == target.id  # ชี้ = ไฮไลต์
    page.set_hands((hand(page, point.x(), point.y(), pinch=True),), clock["t"])
    assert page.graph.dragging == target.id
    page.set_hands((hand(page, point.x() + 80, point.y() + 40, pinch=True),), clock["t"])
    moved = page.screen_point(target.id)
    assert moved.x() == pytest.approx(point.x() + 80, abs=1.0)
    page.set_hands((hand(page, moved.x(), moved.y()),), clock["t"])
    assert page.graph.dragging is None and page.selected == target.id
    window.grab()  # การ์ดข้อความเต็มของโหนดที่เลือก
    page.set_hands((), clock["t"])
    for step in range(70):  # ✊ ค้าง = ล้าง
        page.set_hands((hand(page, 50, 50, gesture="Closed_Fist"),), clock["t"] + step / 60)
    assert not page.graph.nodes and not page.answer.visible


def test_mind_page_without_thinking_or_claude(qapp):
    clock = {"t": 0.0}
    window = StudioWindow(
        Config(),
        audio=False,
        clock=lambda: clock["t"],
        claude=ClaudeBridge(FakeService(), sync=True),
    )
    try:
        page = window.mind
        window.resize(1180, 780)
        page.ask("hi")
        assert not page.graph.thoughts and "without extended thinking" in page._toast[0]
        window.grab()
        page.claude = None
        page.clear()
        assert not page.ask("hi")
        window.grab()
    finally:
        window.close()


# ---- HUD ----------------------------------------------------------------------------- #


def test_answer_card_grows_a_thought_strip(qapp):
    from aether.hud.answer import GRAPH_H, AnswerCard
    from aether.hud.theme import Theme

    card = AnswerCard(Theme.from_config(Config().hud))
    card.set_viewport(QRectF(0, 0, 1920, 1080))
    card.begin("Claude", 0.0)
    plain = card.rect().height()
    for piece in chunks(THINKING):
        card.add_thinking(piece, 0.1)
    assert card.has_graph and card.rect().height() == pytest.approx(plain + GRAPH_H)
    card.append("คำตอบ", 0.5)
    card.finish(None, 1.0)
    assert card.graph.answer is not None
    count = len(card.graph.nodes)
    card.add_thinking("late. more.", 1.1)  # ตอบจบแล้ว: ไม่รับเพิ่ม
    assert len(card.graph.nodes) == count
    assert card.advance(1.2)  # โครงข่ายยังขยับ
    card.begin("Claude", 50.0)
    assert not card.has_graph  # คำถามใหม่ล้างโครงข่ายเดิม


def test_gesture_answer_card_shows_thinking(qapp):
    from test_daily_use import make_app

    app, _ = make_app(qapp, {"actions": {"thumb_down_hold": "ask_claude"}}, hud=True)
    try:
        app.set_claude(ClaudeBridge(FakeService(thinking=chunks(THINKING)), sync=True))
        QGuiApplication.clipboard().setText("why is the sky blue")
        app.gesture_events.emit((GestureEvent(EventKind.ACTION, name="thumb_down_hold"),))
        qapp.processEvents()
        card = getattr(app.hud, "primary", app.hud).answer
        assert card.has_graph and card.state == "done"
    finally:
        app.shutdown()


# ---- หน้าต่างผู้ช่วย ---------------------------------------------------------------------- #


def test_assistant_ask_draws_thought_view(qapp, tmp_path):
    from aether.assistant.dialog import AssistantWindow, KeyStore

    service = FakeService(thinking=chunks(THINKING))
    window = AssistantWindow(
        ClaudeBridge(service, sync=True),
        config_path=tmp_path / "config.toml",
        log_directory=tmp_path,
        incident_directory=tmp_path,
        keys=KeyStore(tmp_path / "k"),
    )
    try:
        window.show()
        view = window.ask_thoughts
        clock = {"t": 0.0}
        view._clock = lambda: clock["t"]
        window.grab()  # ว่าง: ข้อความชวนถาม
        window.ask_edit.setPlainText("Why is the sky blue?")
        window.ask()
        graph = view.graph
        assert graph.root is not None and graph.answer is not None and len(graph.thoughts) >= 5
        for _ in range(240):
            clock["t"] += 1 / 60
            view.tick()
        node = graph.thoughts[0]
        area = view._area()
        point = QPointF(
            *graph.to_screen(node.x, node.y, area.left(), area.top(), area.width(), area.height())
        )
        assert view.node_at(point.x(), point.y()) is node
        view.hovered = node.id
        window.grab()  # การ์ดข้อความเต็ม
        view.clear()
        assert not graph.nodes
    finally:
        window.close()
