"""การเชื่อมทั้งแอป (Qt + worker thread + hotkey) ด้วยกล้อง/tracker/backend/hotkey ปลอม"""

from __future__ import annotations

import time

import numpy as np
import pytest
from handsim import OPEN, PINCHED, make_frame, make_hand
from test_actions import RecordingBackend
from test_hotkeys import FakeApi

from aether.app import AetherApp, Options
from aether.config import Config
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
    def __init__(self, qapp, *, debug: bool = False) -> None:
        self.qapp = qapp
        self.tracker = ScriptedTracker()
        self.camera = BlankCamera()
        self.backend = RecordingBackend()
        self.api = FakeApi()
        config = Config()
        self.quit_requests: list[int] = []
        self.app = AetherApp(
            config,
            Options(debug=debug),
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
