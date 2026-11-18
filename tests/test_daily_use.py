"""ใช้ทุกวันได้ (backlog v1.2): DISARM อัตโนมัติ, พักกล้อง, โปรแกรมที่ใช้กล้องอยู่, เตือนแสงน้อย"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from handsim import Driver, make_frame, make_hand

from aether.config import Config, ConfigError, parse_config
from aether.gestures import EventKind, GestureEngine, State
from aether.health import (
    ConsentEntry,
    LightMonitor,
    consent_store_users,
    frame_brightness,
    on_battery,
)
from aether.mapping import ScreenGeometry

FHD = ScreenGeometry(0, 0, 1920, 1080)
MINUTE = 60_000


# ---- DISARM อัตโนมัติ ----------------------------------------------------- #


def engine(minutes: float = 10.0) -> GestureEngine:
    return GestureEngine(parse_config({"safety": {"auto_disarm_minutes": minutes}}), FHD)


def idle(e: GestureEngine, start: int, minutes: float, step: int = 1000) -> list:
    events = []
    for t in range(start, start + int(minutes * MINUTE) + 1, step):
        events.extend(e.update(make_frame(t))[1])
    return events


def test_disarms_after_the_idle_time_without_a_hand():
    e = engine(10.0)
    e.set_armed(True)
    events = idle(e, 0, 9.9)
    assert e.armed and not events
    events = idle(e, int(9.9 * MINUTE) + 1000, 0.2)
    disarmed = [ev for ev in events if ev.kind is EventKind.DISARMED]
    assert len(disarmed) == 1 and disarmed[0].reason == "idle"
    assert not e.armed and e.state is State.DISARMED


def test_seeing_the_hand_restarts_the_idle_clock():
    e = engine(1.0)
    e.set_armed(True)
    idle(e, 0, 0.9)
    e.update(make_frame(int(0.9 * MINUTE) + 500, make_hand()))  # เห็นมือแวบเดียว
    idle(e, int(0.9 * MINUTE) + 1000, 0.9)
    assert e.armed


def test_arming_restarts_the_idle_clock():
    e = engine(1.0)
    idle(e, 0, 5.0)  # ไม่เห็นมือมานาน ตอน DISARMED ไม่นับ
    e.set_armed(True)
    events = e.update(make_frame(5 * MINUTE + 1000))[1]
    assert e.armed and not events


def test_zero_disables_auto_disarm():
    e = engine(0.0)
    e.set_armed(True)
    idle(e, 0, 30.0, step=10_000)
    assert e.armed


def test_camera_stall_ticks_also_count():
    e = engine(1.0)
    e.set_armed(True)
    events = []
    for t in range(0, MINUTE + 2000, 1000):
        events.extend(e.tick(t))  # watchdog เรียกแทนเฟรม (กล้องค้าง)
    assert not e.armed and any(ev.reason == "idle" for ev in events)


def test_disarm_releases_a_held_drag_first():
    e = engine(1.0)
    d = Driver(e)
    e.set_armed(True)
    d.hold(make_hand(), 200)
    d.hold(make_hand(pinch=0.1), 400)
    assert e.state is State.DRAGGING
    d.hold(None, 400)  # มือหายเกิน 300ms → ปล่อยปุ่มตามปกติ
    events = idle(e, d.t, 1.1)
    assert not e.armed
    ups = [ev for ev in d.events + events if ev.kind is EventKind.BUTTON_UP]
    assert len(ups) == 1


def test_safety_config_range():
    assert Config().safety.auto_disarm_minutes == 10.0
    with pytest.raises(ConfigError):
        parse_config({"safety": {"auto_disarm_minutes": -1}})


# ---- แสงน้อย -------------------------------------------------------------- #


def test_frame_brightness():
    assert frame_brightness(np.zeros((480, 640, 3), dtype=np.uint8)) == 0.0
    assert frame_brightness(np.full((480, 640, 3), 200, dtype=np.uint8)) == pytest.approx(200.0)


def test_light_monitor_warns_after_three_dark_seconds_with_hysteresis():
    monitor = LightMonitor(40)
    changes = [monitor.update(20, t / 10) for t in range(29)]
    assert changes.count("low") == 0
    assert monitor.update(20, 3.0) == "low" and monitor.low
    # สว่างขึ้นนิดเดียว (ยังไม่เกินเกณฑ์ + 10) ไม่นับว่าหาย
    assert all(monitor.update(45, 3.0 + t / 10) is None for t in range(1, 40))
    assert monitor.low
    results = [monitor.update(80, 7.0 + t / 10) for t in range(21)]
    assert results.count("ok") == 1 and not monitor.low


def test_brief_darkness_does_not_warn():
    monitor = LightMonitor(40)
    for t in range(20):
        monitor.update(20 if t % 5 else 90, t / 10)  # มืดสลับสว่าง (เช่นมือบังกล้อง)
    assert not monitor.low


def test_light_monitor_disabled_or_without_images():
    assert LightMonitor(0).update(5, 100.0) is None
    monitor = LightMonitor(40)
    assert all(monitor.update(None, t) is None for t in range(10))


# ---- โปรแกรมที่ใช้กล้อง --------------------------------------------------- #


def test_consent_store_users():
    entries = [
        ConsentEntry("Microsoft.WindowsCamera_8wekyb3d8bbwe", True, 1000, 0),  # ใช้อยู่
        ConsentEntry("C:#Program Files#Zoom#bin#Zoom.exe", False, 2000, 0),  # ใช้อยู่
        ConsentEntry("C:#Users#me#AppData#Local#Microsoft#Teams#current#Teams.exe", False, 5, 9),
        ConsentEntry("C:#Tools#aether#aether.exe", False, 3000, 0),  # ตัวเราเอง
        ConsentEntry("C:#Old#never.exe", False, 0, 0),  # ไม่เคยใช้
        ConsentEntry("C:#Other#Zoom.exe", False, 10, 0),  # ชื่อซ้ำ
    ]
    users = consent_store_users(entries, exclude="c:\\tools\\aether\\aether.exe")
    assert users == ["WindowsCamera", "Zoom.exe"]


def test_on_battery(monkeypatch):
    import psutil

    monkeypatch.setattr(psutil, "sensors_battery", lambda: SimpleNamespace(power_plugged=False))
    assert on_battery() is True
    monkeypatch.setattr(psutil, "sensors_battery", lambda: SimpleNamespace(power_plugged=True))
    assert on_battery() is False
    monkeypatch.setattr(psutil, "sensors_battery", lambda: None)  # เดสก์ท็อป: ไม่มีแบต
    assert on_battery() is None


# ---- app ---------------------------------------------------------------- #


def make_app(qapp, overrides: dict | None = None, *, hud: bool = False):
    from test_actions import RecordingBackend
    from test_app import BlankCamera, ScriptedTracker
    from test_hotkeys import FakeApi

    from aether.app import AetherApp, Options
    from aether.hotkeys import HotkeyManager

    config = parse_config({"hud": {"sounds": False}, **(overrides or {})})
    tracker = ScriptedTracker()
    app = AetherApp(
        config,
        Options(no_hud=not hud),
        tracker,
        source=BlankCamera(),
        backend=RecordingBackend(),
        screen=FHD,
        hotkeys=HotkeyManager({config.hotkeys.toggle_arm: lambda: None}, api=FakeApi()),
    )
    app.quit_requested.disconnect()
    return app, tracker


def test_pause_releases_the_camera_and_resume_brings_it_back(qapp):
    from test_m4 import wait

    app, _tracker = make_app(qapp)
    app.start(boot=False)
    try:
        assert wait(qapp, lambda: app.latest_packet is not None)
        app.toggle_armed()
        changes: list[bool] = []
        app.camera_paused_changed.connect(changes.append)
        app.set_camera_paused(True)
        assert app.camera_paused and not app.pipeline.armed and changes == [True]
        assert app._thread.isFinished()
        assert app.toggle_armed() is False and not app.pipeline.armed  # พักอยู่: ARM ไม่ได้
        before = app.latest_packet
        app.set_camera_paused(False)
        assert wait(qapp, lambda: app.latest_packet is not before)  # ได้เฟรมใหม่จาก thread ใหม่
        assert app.toggle_armed() and changes == [True, False]
    finally:
        app.shutdown()


def test_camera_error_names_the_app_using_the_camera(qapp, monkeypatch):
    from aether import app as app_module

    monkeypatch.setattr(app_module, "webcam_users", lambda: ["Teams.exe"])
    app, _ = make_app(qapp)
    try:
        app._on_camera_error("เปิดกล้อง index 0 ไม่ได้")
        assert "Teams.exe" in app._camera_message
    finally:
        app.shutdown()


def test_low_light_alert_on_the_hud(qapp):
    from aether.hud.overlay import HudOverlay

    app, _ = make_app(qapp, hud=True)
    try:
        assert isinstance(app.hud, HudOverlay)
        for i in range(40):
            packet = SimpleNamespace(brightness=15.0, captured_at=100.0 + i / 10)
            app._check_light(packet)
        assert app.light.low and app.hud._badge._alert == "Low light"
        for i in range(30):
            app._check_light(SimpleNamespace(brightness=120.0, captured_at=110.0 + i / 10))
        assert not app.light.low and app.hud._badge._alert is None
    finally:
        app.shutdown()


def test_tray_pause_toggles_the_app(qapp, tmp_path):
    from test_m4 import Factory, Loader

    from aether.app import AppController, Options

    factory, loader = Factory(), Loader()
    controller = AppController(
        Options(no_hud=True),
        app_factory=factory,
        config_loader=loader,
        calibration_file=tmp_path / "c.json",
        poses_file=tmp_path / "p.json",
        use_tray=True,
    )
    controller.start(loader())
    try:
        controller.tray.actions()["pause"].trigger()
        assert controller.app.camera_paused
        controller.tray.actions()["pause"].trigger()
        assert not controller.app.camera_paused
        controller.app.set_camera_paused(True)  # จากที่อื่น → เมนูติ๊กตาม
        assert controller.tray.actions()["pause"].isChecked()
    finally:
        controller.shutdown()
