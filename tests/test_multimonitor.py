"""หลายจอ (``pointer.screens = "all"``): Desktop, การ map เคอร์เซอร์, HUD หนึ่ง overlay ต่อจอ"""

from __future__ import annotations

import math

import numpy as np
import pytest
from handsim import make_frame, make_hand
from PySide6.QtCore import QRect
from test_actions import RecordingBackend
from test_app import BlankCamera, ScriptedTracker
from test_hotkeys import FakeApi
from test_hud import FakeClock, FakeSampler, FakeSounds, opaque_pixels, packet, render, snapshot

from aether.app import AetherApp, Options
from aether.camera import FramePacket
from aether.config import ConfigError, parse_config
from aether.gestures import EventKind, GestureEngine, GestureEvent, State
from aether.hotkeys import HotkeyManager
from aether.hud.overlay import HudGroup, HudOverlay
from aether.hud.theme import Theme
from aether.mapping import (
    Desktop,
    Monitor,
    PointerTracker,
    ScreenGeometry,
    list_monitors,
    map_to_screen,
    match_qt_screens,
)
from aether.pipeline import FrameResult

LEFT = Monitor("\\\\.\\DISPLAY1", ScreenGeometry(0, 0, 1920, 1080), primary=True)
RIGHT = Monitor("\\\\.\\DISPLAY2", ScreenGeometry(1920, 0, 1920, 1080))
# จอที่สองอยู่ทางขวาและต่ำกว่า: มุมขวาบนของกรอบรวมเป็นช่องว่าง (ไม่มีจอ)
LOWER_RIGHT = Monitor("\\\\.\\DISPLAY2", ScreenGeometry(1920, 400, 1280, 1024))
REGION = (0.15, 0.10, 0.85, 0.70)


# ---- ScreenGeometry ------------------------------------------------------ #


def test_contains_and_union():
    a = ScreenGeometry(0, 0, 100, 50)
    assert a.contains(0, 0) and a.contains(99, 49)
    assert not a.contains(100, 10) and not a.contains(10, 50) and not a.contains(-1, 0)
    assert a.union(ScreenGeometry(150, -20, 10, 10)) == ScreenGeometry(0, -20, 160, 70)


# ---- Desktop ------------------------------------------------------------- #


def test_desktop_bounds_and_primary():
    desktop = Desktop([RIGHT, LEFT])
    assert (desktop.left, desktop.top, desktop.width, desktop.height) == (0, 0, 3840, 1080)
    assert (desktop.right, desktop.bottom) == (3839, 1079)
    assert desktop.primary is LEFT
    assert Desktop([RIGHT]).primary is RIGHT  # ไม่มีจอที่ตั้งเป็นหลัก → จอแรก
    with pytest.raises(ValueError):
        Desktop([])


def test_desktop_with_monitor_left_of_primary_has_negative_origin():
    left_of_primary = Monitor("B", ScreenGeometry(-1280, 100, 1280, 1024))
    desktop = Desktop([LEFT, left_of_primary])
    assert (desktop.left, desktop.top, desktop.width, desktop.height) == (-1280, 0, 3200, 1124)
    assert desktop.monitor_at(-10, 500) is left_of_primary


def test_monitor_at_and_clamp_pull_gap_points_onto_nearest_monitor():
    desktop = Desktop([LEFT, LOWER_RIGHT])
    assert desktop.monitor_at(100, 100) is LEFT
    assert desktop.monitor_at(2500, 800) is LOWER_RIGHT
    # (2500, 100) อยู่ในกรอบรวมแต่ไม่มีจอ: ใกล้ขอบบนของจอขวาที่สุด
    assert desktop.monitor_at(2500, 100) is LOWER_RIGHT
    assert desktop.clamp(2500, 100) == (2500.0, 400.0)
    # (2000, 200): ห่างจอซ้าย 81px, ห่างจอขวา 200px → จอซ้าย
    assert desktop.clamp(2000, 200) == (1919.0, 200.0)
    assert desktop.clamp(-50, 5000) == (0.0, 1079.0)


def test_active_region_spans_all_monitors():
    desktop = Desktop([LEFT, RIGHT])
    assert map_to_screen(0.15, 0.10, REGION, desktop) == pytest.approx((0, 0))
    assert map_to_screen(0.85, 0.70, REGION, desktop) == pytest.approx((3839, 1079))
    x, _ = map_to_screen(0.70, 0.40, REGION, desktop)
    assert desktop.monitor_at(x, 500) is RIGHT


def test_pointer_tracker_never_leaves_the_monitors():
    tracker = PointerTracker(Desktop([LEFT, LOWER_RIGHT]))
    x, y = tracker.update((2600.0, 50.0))  # เป้าตกในช่องว่าง
    assert (x, y) == (2600.0, 400.0)
    for target in [(3000.0, 10.0), (1950.0, 200.0), (5000.0, 5000.0)]:
        x, y = tracker.update(target)
        assert any(m.geometry.contains(x, y) for m in (LEFT, LOWER_RIGHT)), (target, x, y)


def test_engine_moves_cursor_onto_second_monitor():
    engine = GestureEngine(parse_config({"pointer": {"screens": "all"}}), Desktop([LEFT, RIGHT]))
    engine.set_armed(True)
    for i in range(30):
        engine.update(make_frame(1000 + i * 33, make_hand(0.80, 0.40)))
    x, y = engine.snapshot.cursor
    assert x > 1920 and RIGHT.geometry.contains(x, y)
    # ดับเบิลคลิกยังใช้รัศมีตามความสูงจอ (ไม่ error กับ Desktop)
    assert engine.snapshot.state is State.POINTER


# ---- จับคู่จอกับ QScreen ---------------------------------------------------- #


class FakeQtScreen:
    def __init__(self, name: str, x: int, y: int) -> None:
        self._name, self._x, self._y = name, x, y

    def name(self) -> str:
        return self._name

    def geometry(self) -> QRect:
        return QRect(self._x, self._y, 100, 100)


def test_match_qt_screens_by_name_then_position():
    a, b = FakeQtScreen("\\\\.\\DISPLAY2", 1536, 0), FakeQtScreen("\\\\.\\DISPLAY1", 0, 0)
    assert match_qt_screens([LEFT, RIGHT], [a, b]) == [b, a]

    unnamed_left, unnamed_right = FakeQtScreen("X", 0, 0), FakeQtScreen("Y", 1536, 0)
    renamed = [Monitor("A", RIGHT.geometry), Monitor("B", LEFT.geometry, primary=True)]
    assert match_qt_screens(renamed, [unnamed_right, unnamed_left]) == [unnamed_right, unnamed_left]

    assert match_qt_screens([LEFT, RIGHT], [b]) == [b, None]  # QScreen น้อยกว่าจอ


def test_list_monitors_on_this_machine(qapp):
    monitors = list_monitors()
    assert monitors and sum(m.primary for m in monitors) == 1
    for monitor in monitors:
        assert monitor.geometry.width > 0 and monitor.geometry.height > 0


# ---- HUD ต่อจอ ----------------------------------------------------------- #


@pytest.fixture
def theme(qapp) -> Theme:
    return Theme.from_config(parse_config({}).hud, None)


def _overlay(theme, monitor: Monitor, *, panels: bool, sounds=None) -> HudOverlay:
    overlay = HudOverlay(
        parse_config({"pointer": {"screens": "all"}}),
        monitor.geometry,
        theme=theme,
        sounds=sounds,
        sampler=FakeSampler(),
        device_pixel_ratio=1.5,
        clock=FakeClock(),
        panels=panels,
    )
    overlay.resize(1280, 720)
    return overlay


def _run(overlay: HudOverlay, seconds: float = 0.4) -> None:
    for _ in range(20):
        overlay._clock.t += seconds / 20
        overlay._tick()


def test_secondary_overlay_has_no_panels_and_no_boot(theme):
    overlay = _overlay(theme, RIGHT, panels=False)
    assert overlay.screen_geometry == RIGHT.geometry
    # minimal (ค่าเริ่มต้น) ไม่มีอนุภาค
    assert set(overlay._elements) == {
        "boot", "streak", "collapse", "radial", "charge", "launcher", "target", "reticle"
    }  # fmt: skip
    overlay.start()
    assert not overlay._boot.active(overlay._clock.t)  # boot เล่นเฉพาะจอหลัก
    overlay.set_alert("CAMERA LOST")  # ไม่มีป้ายสถานะ: ต้องไม่ error
    overlay.set_armed(True)
    overlay.stop()


def test_reticle_shows_only_on_the_monitor_holding_the_cursor(theme):
    left, right = _overlay(theme, LEFT, panels=True), _overlay(theme, RIGHT, panels=False)
    for overlay in (left, right):
        overlay.start(boot=False)

    on_right = packet(snapshot(cursor=(1920 + 960, 540)))
    left.show_packet(on_right)
    right.show_packet(on_right)
    _run(left)
    _run(right)
    assert left._reticle.rect().isEmpty()
    position = right._reticle.position
    assert (position.x(), position.y()) == pytest.approx((640, 360))  # (960, 540) / 1.5 ในจอขวา
    assert right._reticle.alpha == 1.0
    image = render(right)
    assert image.pixelColor(640, 360).alpha() > 200
    assert opaque_pixels(render(left), QRect(560, 280, 160, 160)) == 0

    on_left = packet(snapshot(cursor=(960, 540)))
    left.show_packet(on_left)
    right.show_packet(on_left)
    _run(left)
    _run(right)
    assert right._reticle.alpha == 0.0 and left._reticle.alpha == 1.0
    for overlay in (left, right):
        overlay.stop()


def test_effects_appear_only_on_the_monitor_where_they_happen(theme):
    sounds = FakeSounds()
    left = _overlay(theme, LEFT, panels=True, sounds=sounds)
    right = _overlay(theme, RIGHT, panels=False)
    group = HudGroup([left, right])
    group.start(boot=False)
    group.show_packet(packet(snapshot(State.GESTURE, cursor=(2500, 500), swipe_ready=True)))
    group.handle_events(
        (
            GestureEvent(EventKind.CLICK, x=2500, y=500),
            GestureEvent(EventKind.ACTION, x=2500, y=500, name="swipe_left"),
            GestureEvent(EventKind.ACTION, x=2500, y=500, name="fist_hold"),
            GestureEvent(EventKind.BUTTON_DOWN, x=300, y=300),
        )
    )
    assert len(right._reticle._shockwaves) == 2  # คลิก + กำมือ บนจอขวา
    assert len(left._reticle._shockwaves) == 1  # เริ่มลากบนจอซ้าย
    assert "left" in right._radial._flash and not left._radial._flash
    assert sounds.played == ["click", "swipe", "minimize", "click"]  # เสียงเล่นครั้งเดียว (จอหลัก)
    group.stop()


def test_group_forwards_state_and_alerts_go_to_primary(theme):
    left, right = _overlay(theme, LEFT, panels=True), _overlay(theme, RIGHT, panels=False)
    group = HudGroup([left, right])
    assert group.primary is left
    group.start(boot=False)
    group.set_armed(True)
    assert left._badge._armed is True
    group.set_alert("CAMERA LOST")
    assert left._badge._alert == "CAMERA LOST"
    frame = FramePacket(
        np.zeros((2, 2, 3), np.uint8),
        FrameResult(make_frame(0, make_hand()), snapshot(cursor=(100, 100))),
        0.0,
        10.0,
        30.0,
    )
    group.show_packet(frame)
    assert left._reticle.position is not None
    group.stop()
    group.deleteLater()
    with pytest.raises(ValueError):
        HudGroup([])


# ---- ทั้งแอป --------------------------------------------------------------- #


def test_app_builds_one_overlay_per_monitor(qapp):
    config = parse_config({"pointer": {"screens": "all"}, "hud": {"sounds": False}})
    desktop = Desktop([RIGHT, LEFT])  # ลำดับไม่สำคัญ: จอหลักต้องได้แผง
    app = AetherApp(
        config,
        Options(),
        ScriptedTracker(),
        source=BlankCamera(),
        backend=RecordingBackend(),
        screen=desktop,
        hotkeys=HotkeyManager({config.hotkeys.toggle_arm: lambda: None}, api=FakeApi()),
    )
    try:
        assert app.screen is desktop
        group = app.hud
        assert isinstance(group, HudGroup) and len(group.overlays) == 2
        assert group.primary.screen_geometry == LEFT.geometry
        assert group.primary._badge is not None and group.overlays[1]._badge is None
        app.start(boot=False)
        app.toggle_armed()
        deadline = 200
        while deadline and app.latest_packet is None:
            qapp.processEvents()
            deadline -= 1
            import time

            time.sleep(0.01)
        assert app.latest_packet is not None
    finally:
        app.shutdown()


def test_primary_mode_keeps_single_overlay(qapp):
    config = parse_config({"hud": {"sounds": False}})
    app = AetherApp(
        config,
        Options(),
        ScriptedTracker(),
        source=BlankCamera(),
        backend=RecordingBackend(),
        screen=LEFT.geometry,
        hotkeys=HotkeyManager({config.hotkeys.toggle_arm: lambda: None}, api=FakeApi()),
    )
    try:
        assert isinstance(app.hud, HudOverlay)
    finally:
        app.shutdown()


def test_screens_setting_is_validated():
    assert parse_config({}).pointer.screens == "primary"
    assert parse_config({"pointer": {"screens": "all"}}).pointer.screens == "all"
    with pytest.raises(ConfigError, match="pointer.screens"):
        parse_config({"pointer": {"screens": "both"}})


def test_cursor_distance_on_wide_desktop_is_proportional():
    """ทุกจอรวมกันกว้างขึ้น: มือขยับเท่าเดิม เคอร์เซอร์ไปได้ไกลกว่า (ตามสัดส่วนความกว้าง)"""
    single = (
        map_to_screen(0.5, 0.4, REGION, LEFT.geometry)[0]
        - map_to_screen(0.4, 0.4, REGION, LEFT.geometry)[0]
    )
    wide_desktop = Desktop([LEFT, RIGHT])
    wide = (
        map_to_screen(0.5, 0.4, REGION, wide_desktop)[0]
        - map_to_screen(0.4, 0.4, REGION, wide_desktop)[0]
    )
    assert math.isclose(wide / single, 3839 / 1919, rel_tol=1e-6)
