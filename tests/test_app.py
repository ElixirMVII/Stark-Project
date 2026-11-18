"""การเชื่อมทั้งแอป (Qt + worker thread + hotkey) ด้วยกล้อง/tracker/backend/hotkey ปลอม"""

from __future__ import annotations

import time

import numpy as np
import pytest
from handsim import OPEN, PINCHED, make_frame, make_hand
from test_actions import RecordingBackend
from test_hotkeys import FakeApi

from aether.app import AetherApp, Options
from aether.config import parse_config
from aether.gestures import State
from aether.hotkeys import HotkeyManager
from aether.mapping import ScreenGeometry

FHD = ScreenGeometry(0, 0, 1920, 1080)


class ScriptedTracker:
    """แทน HandTracker: คืนมือตามท่าที่ test ตั้งไว้ (ไม่ใช้ mediapipe)"""

    def __init__(self) -> None:
        self.hand = make_hand()
        self.closed = False

    def process(self, frame, timestamp_ms):
        hand = self.hand
        return make_frame(timestamp_ms, *(() if hand is None else (hand,)))

    def close(self) -> None:
        self.closed = True


class BlankCamera:
    def __init__(self, interval: float = 0.005) -> None:
        self.interval = interval
        self.stalled = False

    def open(self) -> str:
        return "640x480 @ fake"

    def read(self):
        while self.stalled:  # จำลอง driver กล้องที่ค้าง
            time.sleep(0.01)
        time.sleep(self.interval)
        return np.zeros((480, 640, 3), dtype=np.uint8)

    def close(self) -> None:
        self.stalled = False


class Harness:
    def __init__(self, qapp, *, debug: bool = False, hud: bool = False) -> None:
        self.qapp = qapp
        self.tracker = ScriptedTracker()
        self.camera = BlankCamera()
        self.backend = RecordingBackend()
        self.api = FakeApi()
        config = parse_config({"hud": {"sounds": False}})
        self.quit_requests: list[int] = []
        self.app = AetherApp(
            config,
            Options(debug=debug, no_hud=not hud),
            self.tracker,
            source=self.camera,
            backend=self.backend,
            screen=FHD,
            hotkeys=HotkeyManager(
                {
                    config.hotkeys.toggle_arm: lambda: self.app.toggle_armed(),
                    config.hotkeys.quit: lambda: self.app.quit_requested.emit(),
                },
                api=self.api,
            ),
        )
        # ไม่ให้ test ปิด QApplication ของทั้ง session
        self.app.quit_requested.disconnect()
        self.app.quit_requested.connect(lambda: self.quit_requests.append(1))

    def wait(self, condition, timeout: float = 5.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.qapp.processEvents()
            if condition():
                return True
            time.sleep(0.005)
        return False

    def state(self) -> State:
        return self.app.pipeline.snapshot.state

    def calls(self, name: str) -> list[tuple]:
        return [c for c in list(self.backend.calls) if c[0] == name]


@pytest.fixture
def harness(qapp):
    h = Harness(qapp)
    h.app.start()
    yield h
    h.camera.stalled = False
    h.app.shutdown()


def test_starts_disarmed_and_sends_no_input(harness):
    assert harness.wait(lambda: harness.app.latest_packet is not None)
    harness.tracker.hand = make_hand(pinch=PINCHED)
    time.sleep(0.4)
    harness.qapp.processEvents()
    assert harness.state() is State.DISARMED
    assert harness.backend.calls == []


def test_hotkey_arms_then_hand_moves_and_clicks(harness):
    armed_signals: list[bool] = []
    harness.app.armed_changed.connect(armed_signals.append)

    harness.api.fire(1)  # Ctrl+Alt+G
    assert harness.wait(lambda: harness.state() is State.POINTER)
    assert harness.wait(lambda: armed_signals == [True])
    assert harness.wait(lambda: harness.calls("move") != [])
    assert harness.calls("move")[0] == ("move", 960, 540)

    time.sleep(0.15)  # ให้ engine เห็นนิ้วห่างก่อน
    harness.tracker.hand = make_hand(pinch=PINCHED)
    assert harness.wait(lambda: harness.state() is State.PINCH_PENDING)
    time.sleep(0.08)
    harness.tracker.hand = make_hand(pinch=OPEN)
    assert harness.wait(lambda: harness.calls("press") == [("press", "left")])
    assert harness.wait(lambda: harness.calls("release") == [("release", "left")])

    harness.api.fire(1)
    assert harness.wait(lambda: harness.state() is State.DISARMED)
    assert harness.wait(lambda: armed_signals == [True, False])


def test_quit_hotkey_requests_quit(harness):
    harness.api.fire(2)  # Ctrl+Alt+Q
    assert harness.wait(lambda: harness.quit_requests == [1])


def test_camera_stall_during_drag_releases_button(harness):
    """กล้องค้างขณะกดเมาส์ค้าง: watchdog ใน UI thread ต้องปล่อยปุ่มให้"""
    harness.api.fire(1)
    assert harness.wait(lambda: harness.state() is State.POINTER)
    time.sleep(0.15)
    harness.tracker.hand = make_hand(pinch=PINCHED)
    assert harness.wait(lambda: harness.state() is State.DRAGGING)
    assert harness.calls("press") == [("press", "left")]

    harness.camera.stalled = True
    assert harness.wait(lambda: harness.calls("release") == [("release", "left")], timeout=3.0)
    assert harness.state() is State.IDLE


def test_shutdown_releases_and_closes_tracker(qapp):
    h = Harness(qapp)
    h.app.start()
    h.api.fire(1)
    assert h.wait(lambda: h.state() is State.POINTER)
    time.sleep(0.15)
    h.tracker.hand = make_hand(pinch=PINCHED)
    assert h.wait(lambda: h.state() is State.DRAGGING)
    h.app.shutdown()
    assert h.calls("release") == [("release", "left")]
    assert not h.app.pipeline.armed
    assert h.tracker.closed
    assert h.api.unregistered  # ถอน hotkey แล้ว


def test_pipeline_crash_releases_button_and_stops(qapp):
    h = Harness(qapp)
    h.app.start()
    try:
        h.api.fire(1)
        assert h.wait(lambda: h.state() is State.POINTER)
        time.sleep(0.15)
        h.tracker.hand = make_hand(pinch=PINCHED)
        assert h.wait(lambda: h.state() is State.DRAGGING)

        def crash(frame, timestamp_ms):
            raise RuntimeError("tracker crashed (test)")

        h.app.pipeline._track = crash
        assert h.wait(lambda: h.calls("release") == [("release", "left")])
        assert h.wait(lambda: not h.app.pipeline.armed)
    finally:
        h.app.shutdown()


def test_debug_view_shows_armed_state(qapp):
    h = Harness(qapp, debug=True)
    h.app.start()
    try:
        view = h.app.debug_view
        assert view is not None and view.isVisible()
        assert h.wait(lambda: h.app.latest_packet is not None)
        h.api.fire(1)
        assert h.wait(lambda: view._armed is True)
        image = view.grab().toImage()  # วาดได้จริงโดยไม่ error
        assert image.width() > 0
    finally:
        h.app.shutdown()


# ---- HUD ----------------------------------------------------------------- #


def test_hud_is_created_by_default_and_follows_the_pipeline(qapp):
    h = Harness(qapp, hud=True)
    hud = h.app.hud
    assert hud is not None
    h.app.start()
    try:
        assert hud.isVisible()
        assert h.wait(lambda: h.app.latest_packet is not None)
        assert h.wait(lambda: hud._reticle.position is not None)  # ตามมือแม้ยัง DISARMED
        assert hud._badge._armed is False

        h.api.fire(1)  # arm ด้วย hotkey
        assert h.wait(lambda: hud._badge._armed is True)
        assert h.wait(lambda: hud._badge._mode == "POINTER")

        time.sleep(0.15)
        h.tracker.hand = make_hand(pinch=PINCHED)
        assert h.wait(lambda: h.state() is State.PINCH_PENDING)
        time.sleep(0.08)
        h.tracker.hand = make_hand(pinch=OPEN)
        assert h.wait(lambda: len(hud._reticle._shockwaves) > 0)  # คลิก → shockwave
        position = hud._reticle.position
        dpr = qapp.primaryScreen().devicePixelRatio()
        assert (position.x(), position.y()) == pytest.approx((960 / dpr, 540 / dpr), abs=3)
    finally:
        h.app.shutdown()
    assert not hud.isVisible()


def test_hud_can_be_disabled_by_flag_or_config(qapp):
    by_flag = Harness(qapp, hud=False)
    assert by_flag.app.hud is None
    by_flag.app.shutdown()

    tracker = ScriptedTracker()
    by_config = AetherApp(
        parse_config({"hud": {"enabled": False}}),
        Options(),
        tracker,
        source=BlankCamera(),
        backend=RecordingBackend(),
        screen=FHD,
        hotkeys=HotkeyManager({"<ctrl>+<alt>+g": lambda: None}, api=FakeApi()),
    )
    assert by_config.hud is None
    by_config.shutdown()


def test_camera_error_shows_alert_on_hud(qapp):
    h = Harness(qapp, hud=True)

    class DeadCamera(BlankCamera):
        def open(self) -> str:
            from aether.camera import CameraError

            raise CameraError("เปิดกล้องไม่ได้ (test)")

    h.app._thread._source = DeadCamera()
    h.app.start()
    try:
        assert h.wait(lambda: h.app.hud._badge._alert == "Camera lost")
    finally:
        h.app.shutdown()


# ---- M3: ท่าพิเศษผ่านทั้งแอป ------------------------------------------------- #


def test_swipe_and_fist_hold_reach_the_backend(harness):
    harness.api.fire(1)
    assert harness.wait(lambda: harness.state() is State.POINTER)

    palm = {"gesture": "Open_Palm"}
    harness.tracker.hand = make_hand(0.5, 0.4, **palm)
    assert harness.wait(lambda: harness.app.pipeline.snapshot.swipe_ready)

    for i in range(1, 11):  # ปัดขวา ~0.3 ของเฟรมใน ~150ms
        harness.tracker.hand = make_hand(0.5 + 0.03 * i, 0.4, **palm)
        time.sleep(0.015)
    keys = ["ctrl", "win", "right"]
    expected = [("key_down", k) for k in keys] + [("key_up", k) for k in reversed(keys)]
    assert harness.wait(
        lambda: [c for c in harness.backend.calls if c[0].startswith("key")] == expected
    )

    harness.tracker.hand = make_hand(0.8, 0.4)
    assert harness.wait(lambda: harness.state() is State.POINTER)
    harness.tracker.hand = make_hand(0.8, 0.4, fist=True, gesture="Closed_Fist")
    assert harness.wait(lambda: harness.calls("minimize") == [("minimize",)], timeout=3.0)
    assert harness.calls("press") == []  # ไม่มีคลิกแถม


def test_iloveyou_gesture_arms_and_updates_listeners(harness):
    armed: list[bool] = []
    harness.app.armed_changed.connect(armed.append)
    harness.tracker.hand = make_hand(gesture="ILoveYou")
    assert harness.wait(lambda: armed == [True], timeout=4.0)
    assert harness.app.pipeline.armed
