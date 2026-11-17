"""แหล่งเฟรมจำลอง (spec 4.1): มือจากเมาส์ และ replay — ป้อนเข้า gesture engine จริง ไม่ใช้กล้อง"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from handsim import OPEN, PINCHED, make_frame, make_hand

from aether.config import Config, parse_config
from aether.geometry import pinch_ratio, pointer_anchor
from aether.gestures import (
    DEFAULT_DOUBLE_CLICK_MS,
    EventKind,
    GestureEngine,
    GestureEvent,
    State,
)
from aether.mapping import ScreenGeometry, map_to_screen, screen_to_camera
from aether.pipeline import Pipeline
from aether.recording import FrameWriter
from aether.sources import (
    MouseHandModel,
    MouseSimSource,
    NullTracker,
    ReplaySource,
    SourceSpec,
    key_name,
    parse_source,
    synthetic_hand,
)
from aether.tracker import LM

FHD = ScreenGeometry(0, 0, 1920, 1080)
FRAME_MS = 33


class Sim:
    """MouseHandModel + GestureEngine ที่ ARM แล้ว — ขยับ/กดเมาส์แล้วดู event"""

    def __init__(self, config: Config | None = None, *, armed: bool = True) -> None:
        self.config = config or Config()
        self.model = MouseHandModel(self.config, FHD)
        self.engine = GestureEngine(self.config, FHD, double_click_ms=DEFAULT_DOUBLE_CLICK_MS)
        if armed:
            self.engine.set_armed(True)
        self.t = 1000
        self.mouse = (960.0, 540.0)
        self.events: list[GestureEvent] = []
        self.run(300)
        self.events.clear()

    def run(self, ms: int, mouse: tuple[float, float] | None = None) -> list[GestureEvent]:
        if mouse is not None:
            self.mouse = mouse
        out: list[GestureEvent] = []
        for _ in range(max(1, round(ms / FRAME_MS))):
            _, events = self.engine.update(self.model.frame(self.t, self.mouse))
            out.extend(events)
            self.t += FRAME_MS
        self.events.extend(out)
        return out

    def of(self, *kinds: EventKind) -> list[GestureEvent]:
        return [e for e in self.events if e.kind in kinds]

    def actions(self) -> list[str]:
        return [e.name for e in self.of(EventKind.ACTION)]


# ---- spec / helpers ----------------------------------------------------- #


@pytest.mark.parametrize(
    ("text", "kind", "path"),
    [
        ("camera", "camera", None),
        ("Mouse", "mouse", None),
        ("replay:tests/fixtures/a.jsonl", "replay", "tests/fixtures/a.jsonl"),
        ('replay:"C:\\rec\\x.jsonl"', "replay", "C:\\rec\\x.jsonl"),
    ],
)
def test_parse_source(text, kind, path):
    spec = parse_source(text)
    assert spec.kind == kind
    assert (str(spec.path) if spec.path else None) == (str(path) if path else None) or (
        spec.path is not None and spec.path.name.endswith("x.jsonl")
    )
    assert spec.simulated == (kind != "camera")


@pytest.mark.parametrize("text", ["webcam", "replay:", ""])
def test_parse_source_rejects_unknown(text):
    with pytest.raises(ValueError):
        parse_source(text)


@pytest.mark.parametrize(
    "point", [(0, 0), (1919, 1079), (960, 540), (123, 987), (-50, 2000)]
)  # fmt: skip
def test_screen_to_camera_inverts_map_to_screen(point):
    region = Config().pointer.active_region
    nx, ny = screen_to_camera(*point, region, FHD)
    x, y = map_to_screen(nx, ny, region, FHD)
    expected = (min(max(point[0], 0), 1919), min(max(point[1], 0), 1079))
    assert (x, y) == pytest.approx(expected, abs=1e-6)


def test_synthetic_hand_measures_as_requested():
    for side in ("right", "left"):
        hand = synthetic_hand((0.4, 0.3), 640, 480, side=side, pinch_index=0.2, pinch_middle=0.9)
        assert pointer_anchor(hand) == pytest.approx((0.4, 0.3), abs=1e-9)
        assert pinch_ratio(hand, 640, 480, LM.INDEX_TIP) == pytest.approx(0.2, abs=1e-6)
        assert pinch_ratio(hand, 640, 480, LM.MIDDLE_TIP) == pytest.approx(0.9, abs=1e-6)


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        (SimpleNamespace(vk=0x31, char="1"), "1"),
        (SimpleNamespace(vk=0x35, char="ๅ"), "5"),  # แป้นไทย: ตัวอักษรเป็นภาษาไทย แต่รหัสปุ่มยังเป็น 5
        (SimpleNamespace(vk=0x62, char=None), "2"),  # numpad
        (SimpleNamespace(vk=None, char="3"), "3"),
        (SimpleNamespace(name="left"), "left"),
        (SimpleNamespace(name="up"), "up"),
        (SimpleNamespace(vk=0x36, char="6"), None),
        (SimpleNamespace(name="shift"), None),
    ],
)
def test_key_name(key, expected):
    assert key_name(key) == expected


# ---- mouse → engine ----------------------------------------------------- #


def test_cursor_follows_the_mouse_exactly():
    sim = Sim()
    for point in ((200.0, 150.0), (1700.0, 900.0), (960.0, 540.0)):
        sim.run(400, point)
        assert sim.engine.snapshot.cursor == pytest.approx(point, abs=2)


def test_left_button_click_is_a_left_click():
    sim = Sim()
    sim.model.press("left")
    sim.run(100)
    sim.model.release("left")
    sim.run(100)
    clicks = sim.of(EventKind.CLICK)
    assert len(clicks) == 1 and clicks[0].button == "left"
    assert (clicks[0].x, clicks[0].y) == pytest.approx((960, 540), abs=3)


def test_left_button_held_and_moved_is_a_drag():
    sim = Sim()
    sim.model.press("left")
    sim.run(400)
    assert sim.engine.state is State.DRAGGING
    for i in range(10):
        sim.run(33, (960 + 30 * i, 540))
    sim.model.release("left")
    sim.run(100)
    downs, ups = sim.of(EventKind.BUTTON_DOWN), sim.of(EventKind.BUTTON_UP)
    assert len(downs) == 1 and len(ups) == 1 and ups[0].x > downs[0].x + 100


def test_right_button_is_a_right_click():
    sim = Sim()
    sim.model.press("right")
    sim.run(100)
    sim.model.release("right")
    sim.run(100)
    clicks = sim.of(EventKind.CLICK)
    assert [c.button for c in clicks] == ["right"]


@pytest.mark.parametrize(("notches", "sign"), [(3, 1), (-4, -1)])
def test_wheel_scrolls_in_the_same_direction(notches, sign):
    sim = Sim()
    sim.model.wheel(notches, sim.t)
    sim.run(700)
    scrolled = sum(e.dy for e in sim.of(EventKind.SCROLL))
    assert scrolled * sign == abs(notches)  # 1 notch ของล้อ = สกรอล 1 notch
    assert all(e.dy * sign > 0 for e in sim.of(EventKind.SCROLL))
    assert sim.of(EventKind.CLICK, EventKind.BUTTON_DOWN) == []


def test_after_scrolling_the_cursor_is_back_on_the_mouse():
    sim = Sim()
    sim.model.wheel(5, sim.t)
    sim.run(1400)  # สกรอล → มือหายครู่หนึ่ง → กลับมา
    assert sim.engine.state is State.POINTER
    sim.run(200, (1000.0, 600.0))
    assert sim.engine.snapshot.cursor == pytest.approx((1000, 600), abs=2)


def test_fist_key_held_minimizes():
    sim = Sim()
    sim.model.key("3", sim.t)
    sim.run(900)
    assert sim.actions() == ["fist_hold"]
    sim.model.key("1", sim.t)
    sim.run(200)
    assert sim.engine.state is State.POINTER


def test_fist_key_and_moving_the_mouse_grabs_a_window():
    sim = Sim()
    sim.model.key("3", sim.t)
    sim.run(100)
    for i in range(8):
        sim.run(33, (960 + 40 * i, 540))
    assert sim.engine.state is State.WINDOW_DRAG
    assert sim.of(EventKind.WINDOW_GRAB)


@pytest.mark.parametrize("direction", ["left", "right", "up", "down"])
def test_arrow_key_swipes(direction):
    sim = Sim()
    sim.model.key(direction, sim.t)
    sim.run(1500)
    assert sim.actions() == [f"swipe_{direction}"]
    assert sim.engine.state is State.POINTER  # กลับมาชี้ต่อหลังปัด
    assert sim.engine.snapshot.cursor == pytest.approx((960, 540), abs=3)


def test_palm_key_enters_gesture_mode():
    sim = Sim()
    sim.model.key("2", sim.t)
    sim.run(300)
    assert sim.engine.state is State.GESTURE


def test_victory_key_enters_scroll_mode():
    sim = Sim()
    sim.model.key("4", sim.t)
    sim.run(300)
    assert sim.engine.state is State.SCROLL


def test_love_key_held_arms_from_disarmed():
    sim = Sim(armed=False)
    sim.model.key("5", sim.t)
    sim.run(1300)
    assert sim.engine.armed
    assert sim.of(EventKind.ARMED)


def test_left_handed_config_simulates_the_left_hand():
    sim = Sim(parse_config({"tracking": {"dominant_hand": "left"}}))
    frame = sim.model.frame(sim.t, (500.0, 500.0))
    assert frame.hands[0].handedness == "left"
    sim.model.press("left")
    sim.run(100)
    sim.model.release("left")
    sim.run(100)
    assert len(sim.of(EventKind.CLICK)) == 1


def test_mouse_source_paces_frames_and_reads_position():
    clock = [10.0]
    sleeps: list[float] = []

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        clock[0] += seconds

    source = MouseSimSource(
        Config(),
        FHD,
        position=lambda: (300.0, 200.0),
        clock=lambda: clock[0],
        sleep=sleep,
        listen=False,
    )
    assert source.open() == "mouse simulation"
    frames = [source.read() for _ in range(4)]
    assert all(len(f.hands) == 1 for f in frames)
    assert sleeps and sleeps[0] == pytest.approx(1 / 30, abs=1e-6)
    stamps = [f.timestamp_ms for f in frames]
    assert stamps == sorted(stamps) and stamps[-1] - stamps[0] == pytest.approx(100, abs=2)
    source._on_click(0, 0, SimpleNamespace(name="left"), True)
    assert pinch_ratio(source.read().hands[0], 640, 480) < 0.25
    source._on_key(SimpleNamespace(vk=0x33, char="3"))
    assert source.model.gesture == "Closed_Fist"
    source.close()


# ---- replay ------------------------------------------------------------- #


def _recording(tmp_path, count: int = 30):
    path = tmp_path / "rec.jsonl"
    with FrameWriter(path, note="test") as writer:
        for i in range(count):
            pinch = PINCHED if 10 <= i < 13 else OPEN
            writer.write(make_frame(5000 + i * 33, make_hand(pinch=pinch)))
    return path


def test_replay_plays_frames_at_their_original_pace(tmp_path):
    clock = [100.0]
    sleeps: list[float] = []

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        clock[0] += seconds

    source = ReplaySource(_recording(tmp_path), clock=lambda: clock[0], sleep=sleep)
    description = source.open()
    assert "30 เฟรม" in description
    frames = [source.read() for _ in range(30)]
    assert all(len(f.hands) == 1 for f in frames)
    assert clock[0] - 100.0 == pytest.approx(29 * 0.033, abs=1e-3)
    assert not source.finished
    tail = source.read()
    assert source.finished and tail.hands == ()


def test_replay_through_the_pipeline_gives_the_recorded_click(tmp_path):
    source = ReplaySource(_recording(tmp_path), sleep=lambda s: None)
    source.open()
    engine = GestureEngine(Config(), FHD)
    engine.set_armed(True)

    class Executor:
        def __init__(self) -> None:
            self.events: list[GestureEvent] = []

        def execute(self, events) -> None:
            self.events.extend(events)

        def release_all(self) -> None:
            pass

    executor = Executor()
    pipeline = Pipeline(NullTracker().process, engine, executor)  # type: ignore[arg-type]
    for i in range(40):
        pipeline.process(source.read(), 90_000 + i * 33)  # เวลาใหม่ของ app ไม่ใช่เวลาในไฟล์
    clicks = [e for e in executor.events if e.kind is EventKind.CLICK]
    assert len(clicks) == 1


def test_broken_replay_file_is_a_camera_error(tmp_path):
    from aether.camera import CameraError

    bad = tmp_path / "bad.jsonl"
    bad.write_text("{nope\n", encoding="utf-8")
    with pytest.raises(CameraError):
        ReplaySource(bad).open()
    with pytest.raises(CameraError):
        ReplaySource(tmp_path / "missing.jsonl").open()


def test_pipeline_rebases_simulated_frames_and_skips_the_tracker():
    engine = GestureEngine(Config(), FHD)
    calls: list[int] = []

    class Executor:
        def execute(self, events) -> None:
            pass

        def release_all(self) -> None:
            pass

    def track(image, ts):
        calls.append(ts)
        return make_frame(ts)

    pipeline = Pipeline(track, engine, Executor())  # type: ignore[arg-type]
    result = pipeline.process(make_frame(5, make_hand()), 777)
    assert result.hands.timestamp_ms == 777 and calls == []
    pipeline.process(np.zeros((4, 4, 3), dtype=np.uint8), 800)
    assert calls == [800]


def test_null_tracker_refuses_images():
    with pytest.raises(TypeError):
        NullTracker().process(np.zeros((1, 1, 3)), 0)


def test_source_spec_text():
    assert str(SourceSpec("replay", parse_source("replay:a/b.jsonl").path)) == "replay:a/b.jsonl"
    assert str(SourceSpec("mouse")) == "mouse"


# ---- app ---------------------------------------------------------------- #


def test_app_with_mouse_source_is_always_dry_run_and_runs_without_images(qapp):
    from test_actions import RecordingBackend
    from test_hotkeys import FakeApi
    from test_m4 import wait

    from aether.app import AetherApp, Options
    from aether.hotkeys import HotkeyManager

    config = parse_config({"hud": {"sounds": False}})
    source = MouseSimSource(config, FHD, position=lambda: (700.0, 400.0), listen=False)
    app = AetherApp(
        config,
        Options(source=parse_source("mouse"), dry_run=False, debug=True, no_hud=True),
        NullTracker(),
        source=source,
        backend=RecordingBackend(),
        screen=FHD,
        hotkeys=HotkeyManager({config.hotkeys.toggle_arm: lambda: None}, api=FakeApi()),
    )
    app.quit_requested.disconnect()
    try:
        assert app.dry_run
        app.start(boot=False)
        assert wait(qapp, lambda: app.latest_packet is not None)
        packet = app.latest_packet
        assert packet.image is None and len(packet.result.hands.hands) == 1
        app.toggle_armed()
        assert wait(qapp, lambda: app.pipeline.snapshot.cursor is not None)
        assert wait(qapp, lambda: app.pipeline.snapshot.cursor == (700, 400))
        app.debug_view.repaint()  # วาดได้แม้ไม่มีภาพกล้อง
        assert app.debug_view.grab().width() > 0
    finally:
        app.shutdown()


def test_cli_source_option(monkeypatch):
    from aether import __main__ as cli

    captured = {}
    monkeypatch.setattr("aether.app.run", lambda options: captured.setdefault("o", options) and 0)
    cli.main(["--source", "replay:x.jsonl"])
    assert captured["o"].source.kind == "replay" and captured["o"].source.path.name == "x.jsonl"
    captured.clear()
    cli.main([])
    assert captured["o"].source.kind == "camera"
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["--source", "webcam"])
