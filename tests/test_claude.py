"""ฟีเจอร์ Claude: เก็บ key, คำขอที่ส่งไป API (ของปลอม ไม่ต่ออินเทอร์เน็ต), แก้ config ตามข้อเสนอ, log,
HUD การ์ดคำตอบ, ท่ามือ ask_claude, Studio (Draw/Elements) และหน้าต่างผู้ช่วย"""

from __future__ import annotations

import base64
import sys
import tomllib
from types import SimpleNamespace

import anthropic
import httpx2 as httpx  # anthropic 2.x ใช้ httpx2
import pytest
from PySide6.QtCore import QRectF
from PySide6.QtGui import QGuiApplication

from aether.assistant import configedit, diagnostics, prompts, secret
from aether.assistant.dialog import AssistantWindow, KeyStore
from aether.assistant.schemas import (
    ConfigChange,
    ConfigProposal,
    Quiz,
    QuizQuestion,
    RecipeIdea,
    RecipeIdeas,
)
from aether.assistant.service import FALLBACK_BETA, ClaudeError, ClaudeService, supports_fallback
from aether.assistant.tasks import ClaudeBridge
from aether.config import ClaudeConfig, Config, ConfigError, parse_config
from aether.gestures import EventKind, GestureEvent
from aether.hotkeys import parse_action
from aether.studio.chemistry import (
    COMPOUNDS,
    Flask,
    all_compounds,
    clear_extra_compounds,
    compound_from_idea,
)
from aether.studio.window import StudioWindow

ON = ClaudeConfig(enabled=True)


# ---- ของปลอม ------------------------------------------------------------------- #


class FakeStream:
    def __init__(self, pieces, message):
        self._pieces, self._message = pieces, message

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    @property
    def text_stream(self):
        return iter(self._pieces)

    def get_final_message(self):
        return self._message


class FakeClient:
    """แทน ``anthropic.Anthropic`` — จดทุกคำขอ"""

    def __init__(self, *, pieces=("สวัส", "ดี"), stop="end_turn", parsed=None, error=None):
        self.calls: list[tuple[str, dict]] = []
        self.pieces, self.stop, self.parsed, self.error = pieces, stop, parsed, error
        self.beta = SimpleNamespace(
            messages=SimpleNamespace(stream=self._stream, parse=self._parse)
        )

    def _stream(self, **kwargs):
        self.calls.append(("stream", kwargs))
        if self.error is not None:
            raise self.error
        text = "".join(self.pieces)
        message = SimpleNamespace(
            stop_reason=self.stop, content=[SimpleNamespace(type="text", text=text)]
        )
        return FakeStream(self.pieces, message)

    def _parse(self, **kwargs):
        self.calls.append(("parse", kwargs))
        if self.error is not None:
            raise self.error
        return SimpleNamespace(stop_reason=self.stop, parsed_output=self.parsed, content=[])


def make_service(client: FakeClient | None = None, config: ClaudeConfig = ON, key="sk-test"):
    client = client or FakeClient()
    keys: list[str | None] = []

    def factory(api_key):
        keys.append(api_key)
        return client

    service = ClaudeService(config, key_loader=lambda: key, client_factory=factory)
    return service, client, keys


class FakeService:
    """แทน :class:`ClaudeService` ทั้งตัว (สำหรับเทส UI)"""

    def __init__(self, *, reason=None, pieces=("คำ", "ตอบ"), parsed=None, error=None):
        self.config = ON
        self.reason, self.pieces, self.parsed, self.error = reason, pieces, parsed, error
        self.calls: list[tuple] = []
        self.resets = 0

    @property
    def enabled(self):
        return self.config.enabled

    def ready_reason(self):
        if not self.config.enabled:
            return "ฟีเจอร์ Claude ปิดอยู่"
        return self.reason

    def has_key(self):
        return True

    def update_config(self, config):
        self.config = config

    def reset_client(self):
        self.resets += 1

    def stream_text(self, system, content, on_text=None, *, max_tokens=None):
        self.calls.append(("stream", system, content))
        if self.error:
            raise ClaudeError(self.error)
        for piece in self.pieces:
            if on_text is not None:
                on_text(piece)
        return "".join(self.pieces)

    def parse(self, system, content, model_cls, *, max_tokens=None):
        self.calls.append(("parse", system, content, model_cls))
        if self.error:
            raise ClaudeError(self.error)
        return self.parsed


@pytest.fixture(autouse=True)
def _no_extra_compounds():
    clear_extra_compounds()
    yield
    clear_extra_compounds()


# ---- config ---------------------------------------------------------------------- #


def test_claude_is_off_by_default_and_validated():
    config = Config()
    assert config.claude.enabled is False
    assert config.claude.model == "claude-opus-5-5" and config.claude.effort == "low"
    with open("config.toml", "rb") as handle:  # ไฟล์ที่แจกไป: ปิดไว้เช่นกัน
        assert tomllib.load(handle)["claude"]["enabled"] is False
    with pytest.raises(ConfigError):
        parse_config({"claude": {"effort": "turbo"}})
    with pytest.raises(ConfigError):
        parse_config({"claude": {"max_tokens": 10}})
    assert parse_action("ask_claude") == "ask_claude"


# ---- API key ----------------------------------------------------------------------- #


def test_api_key_round_trip_and_mask(tmp_path):
    path = tmp_path / "sub" / "claude.key"
    assert secret.load_api_key(path) is None
    secret.save_api_key("  sk-ant-api03-abcdefghijklmnop  ", path)
    assert secret.load_api_key(path) == "sk-ant-api03-abcdefghijklmnop"
    assert secret.mask("sk-ant-api03-abcdefghijklmnop") == "sk-ant-…mnop"
    assert secret.mask(None) == "—"
    with pytest.raises(ValueError):
        secret.save_api_key("   ", path)
    secret.clear_api_key(path)
    secret.clear_api_key(path)  # ลบซ้ำได้
    assert secret.load_api_key(path) is None
    assert "claude.key" in str(secret.key_path())


@pytest.mark.skipif(sys.platform != "win32", reason="DPAPI มีเฉพาะ Windows")
def test_dpapi_round_trip():
    sealed = secret._dpapi(b"sk-ant-secret", protect=True)
    assert sealed != b"sk-ant-secret"
    assert secret._dpapi(sealed, protect=False) == b"sk-ant-secret"


# ---- คำขอไป API ---------------------------------------------------------------------- #


def test_stream_request_uses_config_effort_and_server_fallback():
    service, client, keys = make_service(config=ClaudeConfig(enabled=True, effort="high"))
    pieces: list[str] = []
    assert service.stream_text("sys", "hello", pieces.append) == "สวัสดี"
    assert pieces == ["สวัส", "ดี"] and keys == ["sk-test"]
    kind, kwargs = client.calls[0]
    assert kind == "stream"
    assert kwargs["model"] == "claude-opus-5-5"
    assert kwargs["max_tokens"] == 4096
    assert kwargs["system"] == "sys"
    assert kwargs["messages"] == [{"role": "user", "content": "hello"}]
    assert kwargs["output_config"] == {"effort": "high"}
    assert kwargs["betas"] == [FALLBACK_BETA] and kwargs["fallbacks"] == "default"
    service.stream_text("sys", "again")
    assert keys == ["sk-test"]  # client สร้างครั้งเดียว
    service.reset_client()
    service.stream_text("sys", "again", max_tokens=512)
    assert keys == ["sk-test", "sk-test"] and client.calls[-1][1]["max_tokens"] == 512


def test_parse_request_and_haiku_has_no_fallback():
    ideas = RecipeIdeas(ideas=[])
    client = FakeClient(parsed=ideas)
    service, _, _ = make_service(client, ClaudeConfig(enabled=True, model="claude-haiku-5-5"))
    assert service.parse("sys", "x", RecipeIdeas) is ideas
    kind, kwargs = client.calls[0]
    assert kind == "parse" and kwargs["output_format"] is RecipeIdeas
    assert "fallbacks" not in kwargs and "betas" not in kwargs
    assert supports_fallback("claude-opus-5-5") and not supports_fallback("claude-haiku-5-5")
    client.parsed = None
    with pytest.raises(ClaudeError, match="รูปแบบ"):
        service.parse("sys", "x", RecipeIdeas)


def test_key_is_read_once_until_reset():
    reads: list[int] = []

    def loader():
        reads.append(1)
        return "sk-test"

    service = ClaudeService(ON, key_loader=loader, client_factory=lambda key: FakeClient())
    for _ in range(5):
        assert service.ready_reason() is None
    assert len(reads) == 1
    service.reset_client()
    service.ready_reason()
    assert len(reads) == 2


def test_not_ready_and_refusal_raise_thai_errors(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    service, client, _ = make_service(config=ClaudeConfig())
    assert "ปิด" in service.ready_reason()
    with pytest.raises(ClaudeError, match="ปิด"):
        service.stream_text("s", "x")
    service, client, _ = make_service(key=None)
    assert "API key" in service.ready_reason() and not service.has_key()
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-env")
    assert service.has_key() and service.ready_reason() is None
    assert client.calls == []  # ไม่เคยส่งอะไรออกไปตอนยังไม่พร้อม
    service, client, _ = make_service(FakeClient(stop="refusal"))
    with pytest.raises(ClaudeError, match="ปฏิเสธ"):
        service.stream_text("s", "x")


def test_sdk_errors_become_thai_messages():
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    error = anthropic.APIConnectionError(message="boom", request=request)
    service, _, _ = make_service(FakeClient(error=error))
    with pytest.raises(ClaudeError, match="อินเทอร์เน็ต"):
        service.stream_text("s", "x")
    response = httpx.Response(401, request=request)
    auth = anthropic.AuthenticationError("bad key", response=response, body=None)
    service, _, keys = make_service(FakeClient(error=auth))
    with pytest.raises(ClaudeError, match="API key"):
        service.stream_text("s", "x")
    with pytest.raises(ClaudeError):
        service.stream_text("s", "x")
    assert len(keys) == 2  # key ผิด: สร้าง client ใหม่รอบหน้า (เผื่อเปลี่ยน key แล้ว)


def test_bridge_reports_reason_and_busy(qapp):
    bridge = ClaudeBridge(FakeService(reason="ปิดอยู่"), sync=True)
    errors: list[str] = []
    assert bridge.stream("s", "x", on_error=errors.append) is None
    assert errors == ["ปิดอยู่"]
    service = FakeService()
    bridge = ClaudeBridge(service, sync=True)
    results, texts = [], []
    bridge.stream("s", "x", on_text=texts.append, on_result=results.append)
    assert texts == ["คำ", "ตอบ"] and results == ["คำตอบ"] and not bridge.busy
    bridge.busy = True
    bridge.parse("s", "x", Quiz, on_error=errors.append)
    assert "รอสักครู่" in errors[-1]
    bridge.busy = False
    service.error = "พัง"
    bridge.stream("s", "x", on_error=errors.append)
    assert errors[-1] == "พัง" and not bridge.busy


def test_threaded_task_delivers_on_ui_thread(qapp):
    from test_m4 import wait

    bridge = ClaudeBridge(FakeService())
    results: list[str] = []
    bridge.stream("s", "x", on_result=results.append)
    assert wait(qapp, lambda: results == ["คำตอบ"]) and wait(qapp, lambda: not bridge.busy)


# ---- prompts ----------------------------------------------------------------------- #


def test_prompts_never_include_camera_and_trim_long_text():
    system, content = prompts.gesture_question("x" * 20000)
    assert "Thai" in system and "never see the camera" in system
    assert "12,000" in content and len(content) < 13000
    png = b"\x89PNG\r\n\x1a\nfake"
    _, blocks = prompts.drawing_request(png)
    assert blocks[0]["type"] == "image" and blocks[0]["source"]["media_type"] == "image/png"
    assert base64.b64decode(blocks[0]["source"]["data"]) == png
    assert "finger" in blocks[1]["text"]
    system, content = prompts.recipe_request(["H2O", "NaCl", "H2O"], "Na")
    assert "H2O, NaCl" in content and "contain Na" in content
    assert "step-by-step" in system  # ไม่สอนวิธีทำของอันตรายจริง
    _, content = prompts.config_request("[pinch]\non = 0.3\n", "  คลิกยาก  ", ["note"])
    assert "<config_toml>" in content and content.endswith("คลิกยาก") and "note" in content
    _, content = prompts.log_request("tail", [])
    assert "no session statistics" in content


# ---- แก้ config ---------------------------------------------------------------------- #


def test_editable_keys_and_preview_rejects_bad_changes():
    keys = configedit.editable_keys()
    assert "pinch.on" in keys and "pointer.one_euro.beta" in keys
    assert not any(key.startswith("claude.") for key in keys)
    assert "launcher.items" not in keys  # โปรแกรมที่สั่งเปิดได้: แก้ผ่านผู้ช่วยไม่ได้
    with open("config.toml", encoding="utf-8") as handle:
        text = handle.read()
    new = configedit.preview_changes(
        text, [("pinch.on", "0.31"), ("pointer.one_euro.beta", "0.05")]
    )
    config = parse_config(tomllib.loads(new))
    assert config.pinch.on == pytest.approx(0.31)
    assert config.pointer.one_euro.beta == pytest.approx(0.05)
    for bad, match in (
        (("claude.enabled", "true"), "claude"),
        (("nope.key", "1"), "ไม่มีคีย์"),
        (("pinch.on", '"high"'), "number"),
        (("pinch.on", "{{"), "TOML"),
        (("pinch.on", "9.0"), "pinch"),  # ชนิดถูกแต่ค่าผิดกติกา → parse_config ไม่ผ่าน
    ):
        with pytest.raises(ConfigError, match=match):
            configedit.preview_changes(text, [bad])


def test_apply_changes_and_toggle_claude(tmp_path):
    path = tmp_path / "config.toml"
    template = tmp_path / "template.toml"
    template.write_text("[pinch]\non = 0.3 # คอมเมนต์\noff = 0.45\n", encoding="utf-8")
    configedit.apply_changes(path, [("pinch.on", "0.28")], template=template)
    text = path.read_text(encoding="utf-8")
    assert "on = 0.28 # คอมเมนต์" in text
    with pytest.raises(ConfigError):
        configedit.apply_changes(path, [("pinch.on", "0.2"), ("bad.key", "1")])
    assert path.read_text(encoding="utf-8") == text  # ผิดข้อเดียว = ไม่แก้อะไรเลย
    configedit.set_claude_enabled(path, True)
    assert parse_config(tomllib.loads(path.read_text(encoding="utf-8"))).claude.enabled
    configedit.set_claude_enabled(path, False)
    assert "enabled = false" in path.read_text(encoding="utf-8")


# ---- log ------------------------------------------------------------------------------- #


def test_diagnostics_scrub_home_and_tail(tmp_path, monkeypatch):
    home = tmp_path / "home" / "somchai"
    monkeypatch.setattr(diagnostics.Path, "home", staticmethod(lambda: home))
    monkeypatch.setenv("USERNAME", "somchai")
    monkeypatch.setenv("USER", "somchai")
    logs = tmp_path / "logs"
    logs.mkdir()
    lines = [f"line {i} {home}/aether/config.toml" for i in range(300)]
    (logs / "aether.log").write_text("\n".join(lines), encoding="utf-8")
    tail = diagnostics.log_tail(logs)
    assert tail.count("\n") <= diagnostics.MAX_LINES - 1 and len(tail) <= diagnostics.MAX_CHARS
    assert "somchai" not in tail and "~/aether/config.toml" in tail and "line 299" in tail
    assert diagnostics.report_lines(logs, tmp_path / "none") == []
    assert "ยังไม่มี" in diagnostics.log_tail(tmp_path / "empty")


# ---- สูตรผสมจาก Claude ---------------------------------------------------------------- #


def test_compound_ideas_are_validated_and_mixable():
    from aether.studio.chemistry import add_compounds

    assert compound_from_idea("NaCl", "Salt", "เกลือ", "x", "fun", 0) is None  # มีแล้ว
    assert compound_from_idea("Xx2", "?", "", "", "fun", 0) is None  # ไม่มีธาตุนี้
    assert compound_from_idea("C30H62", "?", "", "", "fun", 0) is None  # อะตอมเกิน
    assert compound_from_idea("He", "?", "", "", "fun", 0) is None  # อะตอมเดียว
    borax = compound_from_idea("B4O7", "Tetraborate", "บอเรต", "ใช้ทำสไลม์", "weird", 9)
    assert borax is not None and borax.by_claude and borax.kind == "everyday" and borax.danger == 3
    assert add_compounds([borax, borax]) == [borax]
    assert len(all_compounds()) == len(COMPOUNDS) + 1
    flask = Flask()
    for symbol in ["B"] * 4 + ["O"] * 7:
        flask.add(symbol)
    assert flask.match() == (borax, 1)


# ---- HUD ----------------------------------------------------------------------------- #


def test_answer_card_states_and_fade(qapp):
    from aether.hud.answer import MIN_HOLD, AnswerCard
    from aether.hud.theme import Theme

    card = AnswerCard(Theme.from_config(Config().hud))
    card.set_viewport(QRectF(0, 0, 1920, 1080))
    assert card.rect().isEmpty() and not card.advance(0.0)
    card.begin("Claude", 0.0)
    thinking = card.rect()
    assert not thinking.isEmpty() and card.advance(0.1)
    assert thinking.center().x() == pytest.approx(960.0)
    card.append("สวัสดีครับ " * 40, 0.5)
    assert card.state == "streaming" and card.rect().height() > thinking.height()
    card.finish(None, 1.0)
    assert card.alpha(1.0 + MIN_HOLD - 1.0) == 1.0
    assert not card.advance(2.0)  # ค้างไว้ให้อ่าน ไม่ต้องวาดใหม่
    assert card.advance(1.0 + card.hold_seconds() - 0.3)  # กำลังจาง
    card.advance(1.0 + card.hold_seconds() + 0.1)
    assert not card.visible
    card.fail("ต่อไม่ได้", 5.0)
    assert card.state == "error" and card.hold_seconds() < MIN_HOLD


def _gesture_app(qapp, overrides=None):
    from test_daily_use import make_app

    app, _ = make_app(
        qapp, {"actions": {"thumb_down_hold": "ask_claude"}, **(overrides or {})}, hud=True
    )
    return app


def test_ask_claude_gesture_streams_clipboard_to_hud(qapp):
    app = _gesture_app(qapp)
    try:
        service = FakeService()
        history: list[tuple[str, str]] = []
        app.set_claude(ClaudeBridge(service, sync=True), lambda q, a: history.append((q, a)))
        QGuiApplication.clipboard().setText("Error 0x80070005 access denied")
        app.gesture_events.emit((GestureEvent(EventKind.ACTION, name="thumb_down_hold"),))
        qapp.processEvents()
        card = getattr(app.hud, "primary", app.hud).answer
        assert card.state == "done" and card.text == "คำตอบ"
        assert "0x80070005" in service.calls[0][2]
        assert history == [("Error 0x80070005 access denied", "คำตอบ")]
        QGuiApplication.clipboard().setText("")
        assert not app.ask_claude_clipboard()  # clipboard ว่าง
        app.set_claude(ClaudeBridge(FakeService(reason="ฟีเจอร์ Claude ปิดอยู่"), sync=True))
        assert not app.ask_claude_clipboard("hello")
        assert card.state == "error" and "ปิด" in card.text
    finally:
        app.shutdown()


# ---- Studio --------------------------------------------------------------------------- #


@pytest.fixture
def claude_studio(qapp):
    clock = {"t": 0.0}
    service = FakeService()
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
    yield window, service, history
    window.close()


def test_draw_sends_only_the_drawing(claude_studio):
    window, service, history = claude_studio
    page = window.draw
    assert not page.ask_claude()  # ยังไม่ได้วาด
    assert service.calls == []
    page.drawing.begin(0.2, 0.2)
    page.drawing.extend(0.6, 0.5)
    page.drawing.end()
    assert page.ask_claude("นี่รูปอะไร")
    _, _, content = service.calls[0]
    image = base64.b64decode(content[0]["source"]["data"])
    assert image.startswith(b"\x89PNG") and "นี่รูปอะไร" in content[1]["text"]
    assert page.answer.state == "done" and history == [("[Draw] นี่รูปอะไร", "คำตอบ")]
    assert not page.answer_rect().isEmpty()
    window.grab()
    page.drawing.begin(0.1, 0.8)  # เริ่มวาดเส้นใหม่ = ปิดการ์ด
    page.drawing.end()
    page.on_hands(0.0)
    assert not page.answer.visible
    bar = window.controls.widget(0)
    bar.claude_question.setText("ให้คะแนนหน่อย")
    bar.claude_button.click()
    assert "ให้คะแนนหน่อย" in service.calls[-1][2][1]["text"] and bar.claude_question.text() == ""


def test_elements_claude_tab_explain_ideas_and_quiz(claude_studio):
    window, service, history = claude_studio
    window.set_page(window.pages.index(window.elements))
    page = window.elements
    page.set_panel("claude")
    window.grab()
    assert page.ask()  # ขวดว่าง → ถามเรื่องธาตุที่เลือก
    assert '"symbol": "H"' in service.calls[-1][2] and page.answer.text == "คำตอบ"
    for symbol in ("Na", "Cl"):
        page.drop_atom(symbol)
    assert page.ask("ทำไมเค็ม")
    assert "Na×1, Cl×1" in service.calls[-1][2] and "Sodium chloride" in service.calls[-1][2]
    assert history[-1] == ("[Elements] ทำไมเค็ม", "คำตอบ")
    window.grab()

    service.parsed = RecipeIdeas(
        ideas=[
            RecipeIdea(
                formula="B4O7",
                name="Tetraborate",
                thai="บอเรต",
                fact="ทำสไลม์",
                kind="fun",
                danger=0,
            ),
            RecipeIdea(formula="NaCl", name="dup", thai="", fact="", kind="fun", danger=0),
            RecipeIdea(formula="Qq", name="bad", thai="", fact="", kind="fun", danger=0),
        ]
    )
    assert page.suggest_recipes()
    assert page.claude_view == "ideas" and [c.formula for c in page.ideas] == ["B4O7"]
    assert "2 skipped" in page.answer.text
    window.grab()
    _rect, action = page._claude_rows(page.panel_rect())[0]
    action()  # จีบแถว = ลองผสม
    for step in range(20):
        page.tick(1.0 + step)
    assert page.matches[-1] == ("B4O7", 1)

    service.parsed = Quiz(
        questions=[
            QuizQuestion(
                question="H คือ?", choices=["ไฮโดรเจน", "ฮีเลียม"], answer=0, explanation="เลขอะตอม 1"
            ),
            QuizQuestion(question="bad", choices=["a"], answer=3, explanation=""),
        ]
    )
    assert page.start_quiz()
    assert page.claude_view == "quiz" and len(page.quiz.questions) == 1
    window.grab()
    page.choose(1)
    page.choose(0)  # ตอบแล้ว เปลี่ยนไม่ได้
    assert page.quiz_choice == 1 and page.quiz_score == 0
    window.grab()
    page.next_question()
    assert page.quiz is None and "0/1" in page.answer.text


def test_elements_shows_reason_when_claude_is_off(qapp):
    window = StudioWindow(
        Config(), audio=False, claude=ClaudeBridge(FakeService(reason="ปิดอยู่"), sync=True)
    )
    try:
        window.resize(1180, 780)
        page = window.elements
        page.set_panel("claude")
        window.grab()
        assert page.ask_element()  # เริ่มแล้วแจ้งเหตุผลบนการ์ด
        assert page.answer.state == "error" and page.answer.text == "ปิดอยู่"
        window.grab()
    finally:
        window.close()


# ---- หน้าต่างผู้ช่วย --------------------------------------------------------------------- #


@pytest.fixture
def assistant(qapp, tmp_path):
    service = FakeService()
    reloads: list[bool] = []
    window = AssistantWindow(
        ClaudeBridge(service, sync=True),
        config_path=tmp_path / "config.toml",
        template=None,
        log_directory=tmp_path / "logs",
        incident_directory=tmp_path / "incidents",
        on_config_changed=lambda: reloads.append(True),
        history=[("เก่า", "คำตอบเก่า")],
        keys=KeyStore(tmp_path / "claude.key"),
    )
    window.show()
    yield window, service, reloads, tmp_path
    window.close()


def test_assistant_toggle_key_and_ask(assistant):
    window, service, _, tmp_path = assistant
    assert window.history_list.count() == 1
    window.enabled_box.setChecked(False)
    assert not service.config.enabled and "ปิด" in window.state_label.text()
    assert not window.ask_button.isEnabled() or service.reason is None
    assert "enabled = false" in (tmp_path / "config.toml").read_text(encoding="utf-8")
    window.enabled_box.setChecked(True)
    assert service.config.enabled
    window.key_edit.setText("sk-ant-api03-1234567890abcd")
    window.save_key()
    assert (tmp_path / "claude.key").exists() and window.key_edit.text() == ""
    assert "abcd" in window.key_label.text() and service.resets == 1
    window.clear_key()
    assert not (tmp_path / "claude.key").exists() and service.resets == 2
    window.ask_edit.setPlainText("ท่าไหนดี")
    window.ask()
    assert window.ask_answer.toPlainText() == "คำตอบ" and window.history_list.count() == 2
    window.test_connection()
    assert "ต่อได้แล้ว" in window.status.text()


def test_assistant_logs_preview_then_analyze(assistant):
    window, service, _, tmp_path = assistant
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "aether.log").write_text("WARNING camera lost\n", encoding="utf-8")
    window.collect_logs()
    assert "camera lost" in window.logs_preview.toPlainText()
    window.logs_question.setText("กล้องหลุดบ่อย")
    window.analyze_logs()
    assert "camera lost" in service.calls[-1][2] and "กล้องหลุดบ่อย" in service.calls[-1][2]
    assert window.logs_answer.toPlainText() == "คำตอบ"


def test_assistant_config_proposal_validates_then_applies(assistant):
    window, service, reloads, tmp_path = assistant
    service.parsed = ConfigProposal(
        summary="ลดการสั่น",
        changes=[
            ConfigChange(key="pointer.one_euro.min_cutoff", value="0.6", reason="นิ่งขึ้น"),
            ConfigChange(key="claude.enabled", value="false", reason="แอบปิด"),
            ConfigChange(key="pinch.on", value='"x"', reason="ผิดชนิด"),
        ],
    )
    window.tune_edit.setText("เคอร์เซอร์สั่น")
    window.suggest_config()
    assert service.calls[-1][0] == "parse" and service.calls[-1][3] is ConfigProposal
    assert window.tune_summary.text() == "ลดการสั่น" and window.tune_list.count() == 3
    assert window.selected_changes() == [("pointer.one_euro.min_cutoff", "0.6")]
    assert window.tune_apply.isEnabled()
    assert window.apply_selected() and reloads == [True]
    config = parse_config(tomllib.loads((tmp_path / "config.toml").read_text(encoding="utf-8")))
    assert config.pointer.one_euro.min_cutoff == pytest.approx(0.6)
    assert not config.claude.enabled
    window.tune_edit.setText("")
    window.suggest_config()
    assert "บอกก่อน" in window.status.text()


def test_tray_opens_assistant_and_reload_keeps_bridge(qapp, tmp_path):
    from test_m4 import Factory, Loader

    from aether.app import AppController, Options

    factory, loader = Factory(), Loader()
    controller = AppController(
        Options(no_hud=True, config_path=tmp_path / "config.toml"),
        app_factory=factory,
        config_loader=loader,
        calibration_file=tmp_path / "calibration.json",
        poses_file=tmp_path / "poses.json",
        samples_file=tmp_path / "samples.jsonl",
        use_tray=True,
    )
    try:
        controller.start(loader())
        bridge = controller.claude
        assert bridge is not None and not bridge.service.enabled
        controller.tray.actions()["assistant"].trigger()
        window = controller.assistant
        assert isinstance(window, AssistantWindow) and window.isVisible()
        assert controller.wizard is None  # ไม่ใช่ wizard: มือยังคุมเมาส์ได้
        controller.remember_answer("q", "a")
        assert window.history_list.count() == 1
        loader.next = parse_config({"hud": {"sounds": False}, "claude": {"enabled": True}})
        assert controller.reload()
        assert controller.claude is bridge and bridge.service.enabled
        window.close()
        qapp.processEvents()
        assert controller.assistant is None
    finally:
        controller.shutdown()
