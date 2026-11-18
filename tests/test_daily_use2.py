"""ใช้ทุกวันได้ (2): ประหยัดแบต และเปิดพร้อม Windows (registry ปลอม — ของจริงทดสอบใน CI บน Windows)"""

from __future__ import annotations

import time

import pytest

from aether.autostart import AutoStart, launch_command
from aether.camera import FrameThrottle
from aether.config import ConfigError, parse_config


def test_throttle_limits_processed_frames():
    throttle = FrameThrottle(15)
    processed = sum(throttle.should_process(i / 30) for i in range(90))  # กล้อง 30 fps 3 วินาที
    assert 44 <= processed <= 46  # ≈ 15 fps (ทุก 2 เฟรม)
    throttle.set_max_fps(None)
    assert all(throttle.should_process(10 + i / 30) for i in range(30))
    assert throttle.max_fps is None


def test_throttle_tolerates_uneven_camera_timing():
    throttle = FrameThrottle(15)
    times = [0.0, 0.031, 0.069, 0.098, 0.135, 0.166, 0.201]  # กล้อง ~30 fps ที่ส่งไม่สม่ำเสมอ
    assert [throttle.should_process(t) for t in times] == [
        True,
        False,
        True,
        False,
        True,
        False,
        True,
    ]


def test_battery_fps_config():
    assert parse_config({}).camera.battery_fps == 15
    assert parse_config({"camera": {"battery_fps": 0}}).camera.battery_fps == 0
    with pytest.raises(ConfigError):
        parse_config({"camera": {"battery_fps": 2}})


def test_app_switches_battery_saver_with_the_power_state(qapp):
    from test_daily_use import make_app
    from test_m4 import wait

    app, _ = make_app(qapp)
    power = {"battery": False}
    app._battery_check = lambda: power["battery"]
    app.start(boot=False)
    try:
        assert not app.battery_saver and app._thread.max_fps is None
        power["battery"] = True
        app.check_power()
        assert app.battery_saver and app._thread.max_fps == pytest.approx(15)
        before = app.latest_packet
        assert wait(qapp, lambda: app.latest_packet is not before)  # ยังได้เฟรม (แค่น้อยลง)
        app.set_camera_paused(True)
        app.set_camera_paused(False)  # thread ใหม่ต้องได้ค่าประหยัดแบตด้วย
        assert app._thread.max_fps == pytest.approx(15)
        power["battery"] = False
        app.check_power()
        assert not app.battery_saver and app._thread.max_fps is None
    finally:
        app.shutdown()


def test_battery_saver_off_when_disabled(qapp):
    from test_daily_use import make_app

    app, _ = make_app(qapp, {"camera": {"battery_fps": 0}})
    app._battery_check = lambda: True
    app.check_power()
    try:
        assert not app.battery_saver
    finally:
        app.shutdown()


def test_frames_really_slow_down_when_throttled(qapp):
    from test_daily_use import make_app
    from test_m4 import wait

    app, _ = make_app(qapp)
    app._battery_check = lambda: True
    app.start(boot=False)
    try:
        assert wait(qapp, lambda: app.latest_packet is not None)
        start = app.stats.summary()["frames"]
        time.sleep(1.0)
        processed = app.stats.summary()["frames"] - start
        assert 8 <= processed <= 20  # ~15 fps (กล้องปลอมส่งเร็วกว่านั้นมาก)
    finally:
        app.shutdown()


# ---- เปิดพร้อม Windows ---------------------------------------------------- #


class FakeRegistry:
    def __init__(self) -> None:
        self.values: dict[str, str] = {}

    def get(self, name):
        return self.values.get(name)

    def set(self, name, value):
        self.values[name] = value

    def delete(self, name):
        self.values.pop(name, None)


def test_launch_command():
    assert launch_command(frozen=True, executable="C:\\Apps\\aether\\aether.exe") == (
        '"C:\\Apps\\aether\\aether.exe"'
    )
    cmd = launch_command(frozen=False, executable="C:\\Py\\python.exe")
    assert cmd.endswith('pythonw.exe" -m aether')
    assert launch_command(frozen=False, executable="/usr/bin/python3").endswith(
        'python3" -m aether'
    )


def test_autostart_enable_disable_with_a_fake_registry():
    registry = FakeRegistry()
    auto = AutoStart(registry, name="AETHER-test")
    assert auto.available and not auto.enabled()
    auto.set_enabled(True)
    assert registry.values["AETHER-test"] == launch_command() and auto.enabled()
    registry.values["AETHER-test"] = '"D:\\old\\aether.exe"'  # ย้ายโฟลเดอร์ไปแล้ว: ถือว่าไม่ได้เปิด
    assert not auto.enabled()
    auto.set_enabled(False)
    assert "AETHER-test" not in registry.values and not auto.enabled()


def test_autostart_unavailable_off_windows(monkeypatch):
    import sys

    monkeypatch.setattr(sys, "platform", "linux")
    auto = AutoStart()
    assert not auto.available and not auto.enabled()
    with pytest.raises(OSError):
        auto.set_enabled(True)


def test_tray_autostart_toggle(qapp, tmp_path):
    from test_m4 import Factory, Loader

    from aether.app import AppController, Options

    registry = FakeRegistry()
    factory, loader = Factory(), Loader()
    controller = AppController(
        Options(no_hud=True),
        app_factory=factory,
        config_loader=loader,
        calibration_file=tmp_path / "c.json",
        poses_file=tmp_path / "p.json",
        use_tray=True,
        autostart=AutoStart(registry, name="AETHER-test"),
    )
    controller.start(loader())
    try:
        action = controller.tray.actions()["autostart"]
        assert action.isVisible() and not action.isChecked()
        action.trigger()
        assert registry.values.get("AETHER-test") == launch_command() and action.isChecked()
        action.trigger()
        assert "AETHER-test" not in registry.values and not action.isChecked()
    finally:
        controller.shutdown()
