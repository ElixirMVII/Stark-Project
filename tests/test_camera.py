from __future__ import annotations

import time

import numpy as np
import pytest

from aether import camera
from aether.camera import CameraError, CameraThread, FpsMeter
from aether.gestures import EngineSnapshot
from aether.pipeline import FrameResult
from aether.tracker import HandFrame

# ---- FpsMeter ----------------------------------------------------------- #


def test_fps_meter_needs_two_samples():
    meter = FpsMeter()
    assert meter.fps == 0.0
    assert meter.tick(1.0) == 0.0


def test_fps_meter_steady_rate():
    meter = FpsMeter(window=30)
    for i in range(60):
        fps = meter.tick(i / 30.0)
    assert fps == pytest.approx(30.0)


def test_fps_meter_uses_only_recent_window():
    meter = FpsMeter(window=5)
    for i in range(10):  # ช้า 10 fps
        meter.tick(i * 0.1)
    start = 1.0
    for i in range(5):  # แล้วเร็วขึ้นเป็น 50 fps
        fps = meter.tick(start + i * 0.02)
    assert fps == pytest.approx(50.0)


def test_fps_meter_identical_timestamps_do_not_divide_by_zero():
    meter = FpsMeter()
    meter.tick(1.0)
    assert meter.tick(1.0) == 0.0


def test_fps_meter_reset_and_validation():
    meter = FpsMeter()
    meter.tick(0.0)
    meter.tick(0.1)
    meter.reset()
    assert meter.fps == 0.0
    with pytest.raises(ValueError):
        FpsMeter(window=1)


# ---- CameraThread (กล้องปลอม) -------------------------------------------- #


class FakeSource:
    """ส่งเฟรมดำ 640x480 ตามจำนวนที่กำหนด แล้วอ่านไม่ได้ (เหมือนกล้องหลุด)"""

    def __init__(self, frames: int = 10_000, fail_open: int = 0, interval: float = 0.002):
        self.frames_left = frames
        self.fail_open = fail_open
        self.interval = interval
        self.opened = 0
        self.closed = 0

    def open(self) -> str:
        if self.fail_open > 0:
            self.fail_open -= 1
            raise CameraError("เปิดกล้องไม่ได้ (test)")
        self.opened += 1
        return "640x480 @ fake"

    def read(self):
        if self.frames_left <= 0:
            return None
        self.frames_left -= 1
        time.sleep(self.interval)
        return np.zeros((480, 640, 3), dtype=np.uint8)

    def close(self) -> None:
        self.closed += 1


def _empty_frame(frame, timestamp_ms: int) -> FrameResult:
    """processor ปลอม: ไม่เจอมือ engine อยู่สถานะเริ่มต้น"""
    return FrameResult(HandFrame(timestamp_ms, frame.shape[1], frame.shape[0]), EngineSnapshot())


def _wait_until(qapp, condition, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        qapp.processEvents()
        if condition():
            return True
        time.sleep(0.005)
    return False


@pytest.fixture
def fast_retry(monkeypatch):
    """ย่นเวลารอเปิดกล้องใหม่ให้ test เร็ว"""
    monkeypatch.setattr(camera, "REOPEN_INTERVAL_S", 0.05)
    monkeypatch.setattr(camera, "READ_FAIL_LIMIT", 3)
    monkeypatch.setattr(camera, "READ_RETRY_MS", 1)


def test_thread_delivers_packets_with_increasing_timestamps(qapp):
    timestamps: list[int] = []

    def processor(frame, timestamp_ms):
        timestamps.append(timestamp_ms)
        return _empty_frame(frame, timestamp_ms)

    source = FakeSource()
    thread = CameraThread(source, processor)
    opened: list[str] = []
    packets = []
    thread.camera_opened.connect(opened.append)
    thread.frame_ready.connect(lambda: packets.append(thread.latest()))
    thread.start()
    try:
        assert _wait_until(qapp, lambda: len(packets) >= 5)
    finally:
        assert thread.stop()

    assert opened == ["640x480 @ fake"]
    assert source.closed == 1
    assert timestamps == sorted(timestamps)
    packet = packets[-1]
    assert packet.image.shape == (480, 640, 3)
    assert (packet.result.hands.width, packet.result.hands.height) == (640, 480)
    assert packet.process_ms >= 0.0
    assert packet.fps > 0.0


def test_ui_is_not_flooded_when_it_does_not_consume(qapp):
    """ถ้า UI ยังไม่หยิบ packet ต้องไม่มี signal ใหม่มาต่อคิว (เฟรมเก่าถูกทิ้ง)"""
    processed = []

    def processor(frame, timestamp_ms):
        processed.append(timestamp_ms)
        return _empty_frame(frame, timestamp_ms)

    thread = CameraThread(FakeSource(), processor)
    notifications = []
    thread.frame_ready.connect(lambda: notifications.append(1))  # ไม่เรียก latest()
    thread.start()
    try:
        assert _wait_until(qapp, lambda: len(processed) >= 20)
        qapp.processEvents()
    finally:
        assert thread.stop()
    qapp.processEvents()

    assert len(notifications) == 1
    assert thread.latest().result.hands.timestamp_ms == processed[-1]  # ยังได้เฟรมล่าสุดเสมอ


def test_open_failure_is_reported_once_then_recovers(qapp, fast_retry):
    source = FakeSource(fail_open=3)
    thread = CameraThread(source, _empty_frame)
    errors: list[str] = []
    opened: list[str] = []
    thread.camera_error.connect(errors.append)
    thread.camera_opened.connect(opened.append)
    thread.start()
    try:
        assert _wait_until(qapp, lambda: bool(opened))
    finally:
        assert thread.stop()

    assert errors == ["เปิดกล้องไม่ได้ (test)"]  # ลองใหม่ 3 รอบแต่แจ้งครั้งเดียว
    assert source.opened == 1


def test_camera_dropout_triggers_reopen(qapp, fast_retry):
    source = FakeSource(frames=5)
    thread = CameraThread(source, _empty_frame)
    errors: list[str] = []
    thread.camera_error.connect(errors.append)
    thread.start()
    try:
        assert _wait_until(qapp, lambda: source.opened >= 2)
    finally:
        assert thread.stop()

    assert errors and "กล้องหยุดส่งภาพ" in errors[0]
    assert source.closed == source.opened  # ปิดทุกครั้งที่เปิด


def test_processor_exception_stops_thread_and_reports(qapp):
    def broken(frame, timestamp_ms):
        raise RuntimeError("boom")

    source = FakeSource()
    thread = CameraThread(source, broken)
    failures: list[str] = []
    thread.failed.connect(failures.append)
    thread.start()
    assert _wait_until(qapp, lambda: bool(failures))
    assert thread.wait(3000)

    assert failures == ["RuntimeError: boom"]
    assert source.closed == 1


def test_stop_interrupts_reopen_wait(qapp):
    """สั่งหยุดระหว่างรอเปิดกล้องใหม่ (2 วินาที) ต้องจบเร็ว ไม่ค้าง"""
    thread = CameraThread(FakeSource(fail_open=1000), _empty_frame)
    thread.start()
    time.sleep(0.1)
    started = time.monotonic()
    assert thread.stop(timeout_ms=1500)
    assert time.monotonic() - started < 1.0
