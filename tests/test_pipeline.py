"""Pipeline: tracker → engine → executor พร้อม lock, watchdog และการปล่อยปุ่มเมื่อมีปัญหา"""

from __future__ import annotations

import threading

import pytest
from handsim import FRAME_MS, OPEN, PINCHED, make_frame, make_hand
from test_actions import RecordingBackend

from aether.actions import ActionExecutor
from aether.config import Config
from aether.gestures import EventKind, GestureEngine, State
from aether.mapping import ScreenGeometry
from aether.pipeline import FrameResult, Pipeline, clock_ms

FHD = ScreenGeometry(0, 0, 1920, 1080)


class Rig:
    """Pipeline ที่ใช้ tracker ปลอม (คืนมือที่ test กำหนด) และนาฬิกาที่ test คุมเอง"""

    def __init__(self) -> None:
        self.now = 10_000
        self.hand = None
        self.backend = RecordingBackend()
        self.executor = ActionExecutor(self.backend)
        self.engine = GestureEngine(Config(), FHD)
        self.notified: list[tuple] = []
        self.pipeline = Pipeline(
            self._track,
            self.engine,
            self.executor,
            on_events=self.notified.append,
            clock=lambda: self.now,
        )

    def _track(self, frame, timestamp_ms):
        hands = () if self.hand is None else (self.hand,)
        return make_frame(timestamp_ms, *hands)

    def frame(self, hand=None) -> FrameResult:
        self.hand = hand
        result = self.pipeline.process(None, self.now)
        self.now += FRAME_MS
        return result

    def hold(self, hand, ms: int) -> FrameResult:
        result = None
        for _ in range(max(1, round(ms / FRAME_MS))):
            result = self.frame(hand)
        return result

    def start_drag(self) -> None:
        self.pipeline.set_armed(True)
        self.hold(make_hand(), 200)
        self.hold(make_hand(pinch=PINCHED), 330)
        assert self.pipeline.snapshot.state is State.DRAGGING
        assert self.executor.pressed == frozenset({"left"})

    def calls(self, name: str) -> list[tuple]:
        return [c for c in self.backend.calls if c[0] == name]

    def notified_kinds(self) -> list[EventKind]:
        return [e.kind for batch in self.notified for e in batch]


@pytest.fixture
def rig() -> Rig:
    return Rig()


def test_starts_disarmed_and_sends_nothing(rig):
    assert not rig.pipeline.armed
    result = rig.hold(make_hand(pinch=PINCHED), 600)
    assert rig.backend.calls == []
    assert result.snapshot.state is State.DISARMED and result.events == ()
    assert len(result.hands.hands) == 1


def test_process_runs_full_chain(rig):
    rig.pipeline.set_armed(True)
    result = rig.frame(make_hand())
    assert result.snapshot.state is State.POINTER
    assert [e.kind for e in result.events] == [EventKind.MOVE]
    assert rig.backend.calls == [("move", 960, 540)]

    rig.hold(make_hand(), 200)
    rig.hold(make_hand(pinch=PINCHED), 100)
    rig.frame(make_hand(pinch=OPEN))
    assert rig.calls("press") == [("press", "left")]
    assert rig.calls("release") == [("release", "left")]


def test_listener_gets_discrete_events_but_not_moves(rig):
    rig.pipeline.set_armed(True)
    rig.hold(make_hand(), 200)
    rig.hold(make_hand(0.6, 0.4), 200)  # มีแต่ MOVE
    rig.hold(make_hand(0.6, 0.4, pinch=PINCHED), 100)
    rig.frame(make_hand(0.6, 0.4, pinch=OPEN))
    assert rig.notified_kinds() == [EventKind.ARMED, EventKind.CLICK]


def test_toggle_armed(rig):
    assert rig.pipeline.toggle_armed() is True
    assert rig.pipeline.armed
    assert rig.pipeline.toggle_armed() is False
    assert rig.notified_kinds() == [EventKind.ARMED, EventKind.DISARMED]
    assert rig.pipeline.set_armed(False) is False  # ซ้ำ: ไม่มี event เพิ่ม
    assert len(rig.notified) == 2


def test_disarm_during_drag_releases_immediately_without_a_frame(rig):
    """Hotkey disarm ต้องปล่อยปุ่มทันที ไม่ต้องรอเฟรมถัดไปจากกล้อง"""
    rig.start_drag()
    rig.pipeline.toggle_armed()
    assert rig.executor.pressed == frozenset()
    assert rig.calls("release") == [("release", "left")]
    assert rig.notified_kinds()[-2:] == [EventKind.BUTTON_UP, EventKind.DISARMED]
    count = len(rig.backend.calls)
    rig.hold(make_hand(pinch=PINCHED), 600)
    assert len(rig.backend.calls) == count  # disarm แล้วไม่มี input อีก


def test_watchdog_releases_when_camera_stalls(rig):
    rig.start_drag()
    rig.now += 200
    rig.pipeline.check_stall()
    assert rig.executor.pressed == frozenset({"left"})  # ยังไม่ถึง 300ms
    rig.now += 200
    rig.pipeline.check_stall()
    assert rig.executor.pressed == frozenset()
    assert rig.calls("release") == [("release", "left")]
    assert rig.pipeline.snapshot.state is State.IDLE
    assert rig.pipeline.armed  # ยัง arm อยู่ แค่ปล่อยปุ่ม
    rig.pipeline.check_stall()
    assert len(rig.calls("release")) == 1


def test_watchdog_does_nothing_before_first_frame_or_while_frames_flow(rig):
    rig.pipeline.check_stall()  # ยังไม่เคยมีเฟรม
    rig.start_drag()
    for _ in range(20):
        rig.frame(make_hand(pinch=PINCHED))
        rig.pipeline.check_stall()
    assert rig.executor.pressed == frozenset({"left"})


def test_late_frame_after_watchdog_tick_is_handled(rig):
    """เฟรมที่ถ่ายก่อน watchdog ทำงานแต่ประมวลผลเสร็จทีหลัง (timestamp เก่ากว่า) ต้องไม่ทำให้พัง"""
    rig.start_drag()
    stale_timestamp = rig.now
    rig.now += 500
    rig.pipeline.check_stall()
    rig.hand = make_hand(pinch=PINCHED)
    result = rig.pipeline.process(None, stale_timestamp)
    assert result.snapshot.state is State.POINTER
    assert rig.executor.pressed == frozenset()  # ไม่กดซ้ำเองหลังปล่อย


def test_shutdown_disarms_and_releases(rig):
    rig.start_drag()
    rig.pipeline.shutdown()
    assert rig.executor.pressed == frozenset()
    assert not rig.pipeline.armed
    rig.pipeline.shutdown()  # เรียกซ้ำได้


def test_engine_failure_releases_buttons_and_propagates(rig, monkeypatch):
    rig.start_drag()

    def explode(frame):
        raise RuntimeError("engine bug (test)")

    monkeypatch.setattr(rig.engine, "update", explode)
    with pytest.raises(RuntimeError, match="engine bug"):
        rig.frame(make_hand(pinch=PINCHED))
    assert rig.executor.pressed == frozenset()
    assert rig.calls("release") == [("release", "left")]


def test_tracker_failure_propagates_and_later_shutdown_releases(rig):
    rig.start_drag()

    def broken(frame, timestamp_ms):
        raise ValueError("tracker bug (test)")

    rig.pipeline._track = broken
    with pytest.raises(ValueError):
        rig.pipeline.process(None, rig.now)
    rig.pipeline.shutdown()  # แอปเรียกเมื่อ worker แจ้งว่าพัง
    assert rig.executor.pressed == frozenset()


def test_listener_exception_does_not_break_pipeline(rig):
    def bad_listener(events):
        raise RuntimeError("listener failed (test)")

    rig.pipeline._on_events = bad_listener
    assert rig.pipeline.toggle_armed() is True
    assert rig.frame(make_hand()).snapshot.state is State.POINTER


def test_toggle_from_another_thread_while_frames_are_processed(rig):
    """hotkey thread กับ worker thread ใช้ engine พร้อมกัน: ต้องไม่พัง และจบแบบไม่มีปุ่มค้าง"""
    rig.pipeline.set_armed(True)
    stop = threading.Event()
    errors: list[BaseException] = []

    def toggler():
        try:
            while not stop.is_set():
                rig.pipeline.toggle_armed()
                rig.pipeline.check_stall()
        except BaseException as exc:  # noqa: BLE001 — เก็บทุกอย่างที่ thread โยน
            errors.append(exc)

    thread = threading.Thread(target=toggler)
    thread.start()
    try:
        for i in range(600):
            rig.frame(make_hand(0.3 + (i % 50) / 100, 0.4, pinch=PINCHED if i % 40 < 25 else OPEN))
    finally:
        stop.set()
        thread.join(5.0)
    assert errors == []
    rig.pipeline.shutdown()
    assert rig.executor.pressed == frozenset()
    presses, releases = len(rig.calls("press")), len(rig.calls("release"))
    assert presses == releases


def test_clock_ms_is_monotonic_milliseconds():
    a = clock_ms()
    b = clock_ms()
    assert isinstance(a, int) and b >= a
