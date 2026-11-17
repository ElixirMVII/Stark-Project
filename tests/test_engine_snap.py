"""Magnetic snap ใน gesture engine (M6) — เป้ามาจาก set_target เหมือนที่ thread ของ UIA ส่งมา"""

from __future__ import annotations

import math

import pytest
from handsim import OPEN, PINCHED, Driver, make_hand

from aether.config import Config, parse_config
from aether.gestures import DEFAULT_DOUBLE_CLICK_MS, EventKind, GestureEngine, State
from aether.mapping import ScreenGeometry
from aether.snap import Target

FHD = ScreenGeometry(0, 0, 1920, 1080)
# มือที่ (0.5, 0.4) = กลางจอ (960, 540) — ปุ่มกลาง (972, 540) อยู่ห่าง 12 px (ในรัศมี 24)
BUTTON = Target(952, 520, 40, 40, "Button", "OK")


def driver(overrides: dict | None = None) -> Driver:
    config = parse_config(overrides) if overrides else Config()
    engine = GestureEngine(config, FHD, double_click_ms=DEFAULT_DOUBLE_CLICK_MS)
    d = Driver(engine)
    engine.set_armed(True)
    d.hold(make_hand(), 300)
    d.clear()
    return d


def test_cursor_is_pulled_toward_the_target_and_reported():
    d = driver()
    raw = d.engine.snapshot.cursor
    d.engine.set_target(BUTTON)
    d.hold(make_hand(0.5005, 0.4), 200)  # ขยับนิดเดียวให้ engine คำนวณตำแหน่งใหม่
    snap = d.engine.snapshot
    assert snap.target == BUTTON
    assert snap.pointer == pytest.approx(raw, abs=2)  # ตำแหน่งก่อน snap
    assert math.dist(snap.cursor, BUTTON.center) < math.dist(raw, BUTTON.center) - 1
    moves = [e for e in d.events if e.kind is EventKind.MOVE]
    assert moves and (moves[-1].x, moves[-1].y) == snap.cursor


def test_click_lands_on_the_snapped_position():
    d = driver()
    d.engine.set_target(BUTTON)
    d.hold(make_hand(0.5005, 0.4), 100)
    d.hold(make_hand(0.5005, 0.4, pinch=PINCHED), 100)
    d.hold(make_hand(0.5005, 0.4, pinch=OPEN), 100)
    clicks = [e for e in d.events if e.kind is EventKind.CLICK]
    assert len(clicks) == 1
    assert math.dist((clicks[0].x, clicks[0].y), BUTTON.center) < 12


def test_no_snap_when_targeting_is_disabled():
    d = driver({"targeting": {"enabled": False}})
    raw = d.engine.snapshot.cursor
    d.engine.set_target(BUTTON)
    d.hold(make_hand(0.5005, 0.4), 200)
    assert d.engine.snapshot.target is None
    assert d.engine.snapshot.cursor == pytest.approx(raw, abs=2)


def test_no_snap_while_scrolling_or_disarmed():
    d = driver()
    d.engine.set_target(BUTTON)
    d.hold(make_hand(gesture="Victory"), 300)
    assert d.engine.state is State.SCROLL and d.engine.snapshot.target is None
    d.engine.set_armed(False)
    d.hold(make_hand(), 200)
    assert d.engine.snapshot.target is None


def test_starting_a_drag_keeps_the_cursor_where_it_was_snapped():
    d = driver()
    d.engine.set_target(BUTTON)
    d.hold(make_hand(0.5005, 0.4), 100)
    d.hold(make_hand(0.5005, 0.4, pinch=PINCHED), 330)  # จีบค้าง → เริ่มลาก
    assert d.engine.state is State.DRAGGING
    down = next(e for e in d.events if e.kind is EventKind.BUTTON_DOWN)
    d.clear()
    d.hold(make_hand(0.5005, 0.4, pinch=PINCHED), 100)  # มือนิ่ง: เคอร์เซอร์ต้องไม่กระโดด
    assert d.engine.snapshot.cursor == pytest.approx((down.x, down.y), abs=1)
    assert d.engine.snapshot.target is None  # ลากอยู่: ไม่ดูด


def test_stale_target_far_from_the_cursor_does_nothing():
    d = driver()
    raw = d.engine.snapshot.cursor
    d.engine.set_target(Target(100, 100, 40, 40, "Button"))
    d.hold(make_hand(0.5005, 0.4), 200)
    assert d.engine.snapshot.cursor == pytest.approx(raw, abs=2)
    assert d.engine.snapshot.target is None


def test_snap_strength_is_live_tunable():
    d = driver()
    d.engine.reconfigure(parse_config({"targeting": {"snap_strength": 0.0}}))
    raw = d.engine.snapshot.cursor
    d.engine.set_target(BUTTON)
    d.hold(make_hand(0.5005, 0.4), 200)
    assert d.engine.snapshot.cursor == pytest.approx(raw, abs=2)
    assert d.engine.snapshot.target == BUTTON  # ยังล็อกเป้า (แสดงกรอบ) แต่ไม่ดูด
