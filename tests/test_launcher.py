"""เมนูวงกลมเปิดแอป: ค้าง 👍 → เลือกช่องด้วยทิศของมือ → จีบเพื่อเปิด / แบมือเพื่อยกเลิก"""

from __future__ import annotations

import logging
import math
import tomllib

import pytest
from handsim import OPEN, PINCHED, Driver, make_hand
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QImage, QPainter
from test_actions import RecordingBackend
from test_gestures import make_engine, of_kind, pointing
from test_hud import FakeClock, FakeSampler, FakeSounds, packet, render, snapshot

from aether.actions import ActionExecutor, DryRunBackend
from aether.config import Config, ConfigError, LauncherItem, parse_config
from aether.gestures import EventKind, GestureEvent, State, launcher_slice
from aether.hud.launcher import LauncherMenu, sector_path
from aether.hud.overlay import HudOverlay
from aether.hud.theme import Theme
from aether.mapping import ScreenGeometry

THUMB = {"gesture": "Thumb_Up"}
PALM = {"gesture": "Open_Palm"}


def actions(events):
    return [e.name for e in events if e.kind is EventKind.ACTION]


def opened(overrides: dict | None = None) -> Driver:
    """Driver ที่เมนูเปิดอยู่แล้ว (มืออยู่ที่จุดเปิดเมนู 0.5, 0.4)"""
    driver = pointing(overrides)
    driver.hold(make_hand(**THUMB), 900)
    assert driver.engine.state is State.LAUNCHER
    driver.clear()
    return driver


# ---- ทิศ → ช่อง ----------------------------------------------------------- #


@pytest.mark.parametrize(
    ("dx", "dy", "count", "expected"),
    [
        (0, -10, 4, 0),  # ขึ้น
        (10, 0, 4, 1),  # ขวา
        (0, 10, 4, 2),  # ลง
        (-10, 0, 4, 3),  # ซ้าย
        (10, -10, 8, 1),  # ขวาบน
        (-10, -10, 8, 7),  # ซ้ายบน
        (3, -10, 4, 0),  # เฉขวานิดหน่อยยังเป็นช่องบน
        (10, -9, 4, 1),  # เลย 45° ไปทางขวา
        (0, -10, 1, 0),
        (5, 5, 3, 1),
    ],
)
def test_launcher_slice(dx, dy, count, expected):
    assert launcher_slice(dx, dy, count) == expected


# ---- engine ---------------------------------------------------------------- #


def test_thumbs_up_hold_opens_menu_and_freezes_cursor():
    driver = pointing()
    cursor = driver.engine.snapshot.cursor
    driver.hold(make_hand(**THUMB), 400)
    snap = driver.engine.snapshot
    assert snap.state is State.CHARGING and snap.charge_kind == "launcher" and snap.charge < 1.0
    driver.hold(make_hand(**THUMB), 400)
    assert driver.engine.state is State.LAUNCHER
    assert actions(driver.events) == ["launcher_open"]
    assert driver.engine.snapshot.launcher_selection is None
    driver.clear()
    driver.animate(lambda p: make_hand(0.5 + 0.03 * p, 0.4, **THUMB), 200)  # ขยับนิดเดียว
    assert driver.engine.snapshot.cursor == cursor
    assert of_kind(driver.events, EventKind.MOVE) == []
    assert driver.engine.snapshot.launcher_selection is None  # ยังไม่พ้นระยะเริ่มเลือก


@pytest.mark.parametrize(
    ("dx", "dy", "slot"),
    [(0.0, -0.15, 0), (0.15, 0.0, 1), (0.0, 0.15, 2), (-0.15, 0.0, 3)],
)
def test_moving_hand_selects_slice_and_pinch_launches_it(dx, dy, slot):
    driver = opened()
    driver.animate(lambda p: make_hand(0.5 + dx * p, 0.4 + dy * p), 200)
    assert driver.engine.snapshot.launcher_selection == slot
    driver.hold(make_hand(0.5 + dx, 0.4 + dy, pinch=PINCHED), 100)
    assert actions(driver.events) == [f"launch_{slot}"]
    assert driver.engine.state is State.POINTER
    # จีบที่ใช้เลือกต้องไม่กลายเป็นคลิกหรือการลาก
    driver.hold(make_hand(0.5 + dx, 0.4 + dy, pinch=PINCHED), 500)
    driver.hold(make_hand(0.5 + dx, 0.4 + dy, pinch=OPEN), 100)
    assert of_kind(driver.events, EventKind.CLICK, EventKind.BUTTON_DOWN) == []


def test_pinch_without_selection_keeps_menu_open():
    driver = opened()
    driver.hold(make_hand(pinch=PINCHED), 100)
    assert driver.engine.state is State.LAUNCHER and actions(driver.events) == []
    driver.hold(make_hand(pinch=OPEN), 66)
    driver.animate(lambda p: make_hand(0.5 + 0.15 * p, 0.4), 150)
    driver.hold(make_hand(0.65, 0.4, pinch=PINCHED), 100)
    assert actions(driver.events) == ["launch_1"]


def test_open_palm_cancels_menu():
    driver = opened()
    driver.animate(lambda p: make_hand(0.5 + 0.15 * p, 0.4), 150)
    driver.hold(make_hand(0.65, 0.4, **PALM), 200)
    assert driver.engine.state is not State.LAUNCHER
    assert actions(driver.events) == []
    assert driver.engine.snapshot.launcher_selection is None


def test_menu_times_out():
    driver = opened({"launcher": {"timeout_ms": 1000}})
    driver.hold(make_hand(0.65, 0.4), 1200)
    assert driver.engine.state is State.POINTER and actions(driver.events) == []


def test_hand_lost_closes_menu():
    driver = opened()
    driver.hold(None, 400)
    assert driver.engine.state is State.IDLE
    assert driver.engine.snapshot.launcher_selection is None


def test_holding_thumbs_up_does_not_reopen_after_launch():
    driver = opened()
    driver.animate(lambda p: make_hand(0.5, 0.4 - 0.15 * p, **THUMB), 150)
    driver.hold(make_hand(0.5, 0.25, pinch=PINCHED, **THUMB), 100)
    assert actions(driver.events) == ["launch_0"]
    driver.hold(make_hand(0.5, 0.25, **THUMB), 2000)  # ยังค้าง 👍 อยู่
    assert actions(driver.events) == ["launch_0"]
    assert driver.engine.state is not State.LAUNCHER


def test_disabled_or_empty_launcher_ignores_thumbs_up():
    for overrides in ({"launcher": {"enabled": False}}, {"launcher": {"items": []}}):
        driver = pointing(overrides)
        driver.hold(make_hand(**THUMB), 1500)
        assert driver.engine.state is State.POINTER
        assert actions(driver.events) == []


def test_thumbs_up_does_nothing_while_disarmed():
    driver = Driver(make_engine())
    driver.hold(make_hand(**THUMB), 1500)
    assert driver.events == [] and driver.engine.state is State.DISARMED


def test_eight_items_use_eight_slices():
    items = [{"label": f"A{i}", "run": f"app{i}.exe"} for i in range(8)]
    driver = opened({"launcher": {"items": items}})
    angle = math.radians(45)
    driver.animate(
        lambda p: make_hand(
            0.5 + 0.15 * math.sin(angle) * p, 0.4 - 0.15 * math.cos(angle) * p * 640 / 480
        ),
        200,
    )
    assert driver.engine.snapshot.launcher_selection == 1


# ---- executor -------------------------------------------------------------- #


ITEMS = (
    LauncherItem("Browser", "https://example.com"),
    LauncherItem("Pad", "notepad.exe", "a.txt"),
)


def test_executor_launches_configured_item():
    backend = RecordingBackend()
    executor = ActionExecutor(backend, {}, ITEMS)
    executor.execute([GestureEvent(EventKind.ACTION, name="launch_1")])
    assert backend.calls == [("launch", "notepad.exe", "a.txt")]
    executor.execute(
        [
            GestureEvent(EventKind.ACTION, name="launch_7"),  # ไม่มีรายการที่ 7
            GestureEvent(EventKind.ACTION, name="launch_x"),
            GestureEvent(EventKind.ACTION, name="launcher_open"),
        ]
    )
    assert backend.calls == [("launch", "notepad.exe", "a.txt")]


def test_dry_run_only_logs_launch(caplog):
    executor = ActionExecutor(DryRunBackend(), {}, ITEMS)
    with caplog.at_level(logging.INFO, logger="aether.actions"):
        executor.execute([GestureEvent(EventKind.ACTION, name="launch_0")])
    assert "[dry-run] launch https://example.com" in caplog.text


def test_launch_target_uses_the_os_opener(monkeypatch):
    import subprocess
    import sys

    from aether import actions as actions_module

    calls = []
    if sys.platform == "win32":
        monkeypatch.setattr(actions_module.os, "startfile", lambda *a, **k: calls.append((a, k)))
        actions_module.launch_target("notepad.exe", "x.txt")
        assert calls == [(("notepad.exe",), {"arguments": "x.txt"})]
    else:
        monkeypatch.setattr(subprocess, "Popen", lambda args, **k: calls.append(args))
        actions_module.launch_target("https://example.com")
        assert calls == [["xdg-open", "https://example.com"]]


# ---- config ---------------------------------------------------------------- #


def test_launcher_items_parse_from_toml():
    data = tomllib.loads(
        """
[launcher]
hold_ms = 700
items = [
  { label = "Web", run = "https://example.com" },
  { label = "Code", run = "code.exe", args = "--new-window" },
]
"""
    )
    launcher = parse_config(data).launcher
    assert launcher.hold_ms == 700
    assert launcher.items == (
        LauncherItem("Web", "https://example.com"),
        LauncherItem("Code", "code.exe", "--new-window"),
    )
    assert len(Config().launcher.items) == 4


@pytest.mark.parametrize(
    ("data", "fragment"),
    [
        ({"launcher": {"items": [{"label": "", "run": "a.exe"}]}}, "label"),
        ({"launcher": {"items": [{"label": "A", "run": " "}]}}, "run"),
        ({"launcher": {"items": [{"label": "A"}]}}, "run"),
        ({"launcher": {"items": [{"label": "A", "run": "a", "icon": "x"}]}}, "icon"),
        ({"launcher": {"items": [{"label": f"A{i}", "run": "a"} for i in range(9)]}}, "8"),
        ({"launcher": {"items": "notepad"}}, "launcher.items"),
        ({"launcher": {"timeout_ms": 10}}, "timeout_ms"),
        ({"launcher": {"hold_ms": 0}}, "hold_ms"),
    ],
)
def test_invalid_launcher_config(data, fragment):
    with pytest.raises((ConfigError, TypeError)) as excinfo:
        parse_config(data)
    assert fragment in str(excinfo.value)


# ---- HUD ------------------------------------------------------------------- #


@pytest.fixture
def theme(qapp) -> Theme:
    return Theme.from_config(Config().hud, None)


def _paint(menu: LauncherMenu, size: int = 360) -> QImage:
    image = QImage(size, size, QImage.Format.Format_ARGB32)
    image.fill(Qt.GlobalColor.transparent)
    painter = QPainter(image)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    menu.paint(painter, 10.0)
    painter.end()
    return image


def test_sector_path_covers_the_right_area():
    center = QPointF(100, 100)
    top = sector_path(center, 40, 80, -45, 90)  # ช่องบน
    assert top.contains(QPointF(100, 40)) and not top.contains(QPointF(100, 160))
    assert not top.contains(QPointF(100, 100))  # รูตรงกลาง
    right = sector_path(center, 40, 80, 45, 90)
    assert right.contains(QPointF(160, 100)) and not right.contains(QPointF(40, 100))


def test_menu_shows_only_in_launcher_state_and_highlights_selection(theme):
    menu = LauncherMenu(theme, ["Browser", "Files", "Notepad", "Calc"])
    center = QPointF(180, 180)
    menu.advance(0.0)
    menu.set_state(center, snapshot(State.POINTER))
    assert menu.advance(1.0) is False and menu.rect().isEmpty()

    menu.set_state(center, snapshot(State.LAUNCHER, launcher_selection=1))
    for i in range(1, 6):
        menu.advance(1.0 + i * 0.05)
    assert menu.alpha == 1.0 and menu.selection == 1
    image = _paint(menu)
    rect = menu.rect().toAlignedRect()
    outside = sum(
        image.pixelColor(x, y).alpha() > 0
        for x in range(0, 360, 3)
        for y in range(0, 360, 3)
        if not rect.contains(x, y)
    )
    assert outside == 0
    right, left = image.pixelColor(180 + 100, 180), image.pixelColor(180 - 100, 180)
    assert right.red() > left.red()  # ช่องขวา (เลือกอยู่) เป็นสีเน้น
    assert image.pixelColor(180, 180).alpha() == 0  # กลางวงโปร่ง (เห็น reticle)

    menu.set_state(center, snapshot(State.POINTER))
    for i in range(1, 6):
        alive = menu.advance(1.25 + i * 0.05)
    assert alive is False and menu.alpha == 0.0


def test_flash_keeps_menu_visible_briefly_after_launch(theme):
    menu = LauncherMenu(theme, ["A", "B"])
    center = QPointF(180, 180)
    menu.set_state(center, snapshot(State.LAUNCHER, launcher_selection=0))
    menu.advance(0.0)
    menu.advance(1.0)
    menu.flash(0, 1.0)
    menu.flash(9, 1.0)  # ช่องที่ไม่มี: ไม่ทำอะไร
    menu.set_state(center, snapshot(State.POINTER))  # engine ปิดเมนูแล้ว
    assert menu.advance(1.1) and menu.alpha > 0.0
    menu.advance(1.5)
    assert menu.advance(2.0) is False


def test_long_labels_are_elided(theme):
    menu = LauncherMenu(theme, ["A very very long application name"] * 8)
    menu.set_state(QPointF(180, 180), snapshot(State.LAUNCHER))
    menu.advance(0.0)
    menu.advance(1.0)
    assert not _paint(menu).isNull()


def test_overlay_draws_menu_and_plays_sounds(qapp, theme):
    sounds = FakeSounds()
    overlay = HudOverlay(
        Config(),
        ScreenGeometry(0, 0, 1920, 1080),
        theme=theme,
        sounds=sounds,
        sampler=FakeSampler(),
        device_pixel_ratio=1.5,
        clock=FakeClock(),
    )
    overlay.resize(1280, 720)
    overlay.start(boot=False)
    overlay.show_packet(packet(snapshot(State.LAUNCHER, launcher_selection=2)))
    for _ in range(20):
        overlay._clock.t += 0.02
        overlay._tick()
    image = render(overlay)
    assert image.pixelColor(640, 360 + 86).alpha() > 100  # ช่องล่าง (เลือกอยู่)
    overlay.handle_events(
        (
            GestureEvent(EventKind.ACTION, x=960, y=540, name="launcher_open"),
            GestureEvent(EventKind.ACTION, x=960, y=540, name="launch_2"),
        )
    )
    assert sounds.played == ["swipe", "click"]
    assert overlay._launcher._flash is not None
    assert overlay._charge.label() in ("Minimize", "Apps", "Arm", "Disarm")
    overlay.stop()


def test_charge_ring_label_for_launcher(theme):
    from aether.hud.radial import ChargeRing

    ring = ChargeRing(theme, {})
    ring.set_state(QPointF(10, 10), snapshot(State.CHARGING, charge=0.4, charge_kind="launcher"))
    assert ring.label() == "Apps"
    holo = Theme.from_config(parse_config({"hud": {"style": "holo"}}).hud, None)
    ring = ChargeRing(holo, {})
    ring.set_state(QPointF(10, 10), snapshot(State.CHARGING, charge=0.4, charge_kind="launcher"))
    assert ring.label() == "APPS"  # holo: ตัวพิมพ์ใหญ่
