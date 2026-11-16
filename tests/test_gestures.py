"""Gesture engine (M1): pointer / click / drag / right-click / scroll / safety

ทุก test ป้อนเฟรมสังเคราะห์ที่ ~30fps ผ่าน ``handsim.Driver`` — ไม่มีกล้อง ไม่แตะ OS
"""

from __future__ import annotations

import dataclasses
import math
import random

import pytest
from handsim import FRAME_MS, OPEN, PINCHED, Driver, make_frame, make_hand

from aether.config import Config, parse_config
from aether.gestures import (
    DEFAULT_DOUBLE_CLICK_MS,
    LOST_RELEASE_MS,
    EventKind,
    GestureEngine,
    GestureEvent,
    GestureVoter,
    State,
)
from aether.mapping import ScreenGeometry

FHD = ScreenGeometry(0, 0, 1920, 1080)
CENTER = (960, 540)  # anchor (0.5, 0.4) = กลาง active region = กลางจอ


def make_engine(overrides: dict | None = None, screen: ScreenGeometry = FHD) -> GestureEngine:
    config = parse_config(overrides) if overrides else Config()
    return GestureEngine(config, screen, double_click_ms=DEFAULT_DOUBLE_CLICK_MS)


def pointing(overrides: dict | None = None, **hand_kwargs) -> Driver:
    """Driver ที่ arm แล้วและมีมือกางอยู่ในสถานะ POINTER (พร้อมจีบ)"""
    driver = Driver(make_engine(overrides))
    driver.engine.set_armed(True)
    driver.hold(make_hand(**hand_kwargs), 200)
    assert driver.engine.state is State.POINTER
    driver.clear()
    return driver


def of_kind(events: list[GestureEvent], *kinds: EventKind) -> list[GestureEvent]:
    return [e for e in events if e.kind in kinds]


def click(driver: Driver, pinch_ms: int = 100, **hand_kwargs) -> None:
    driver.hold(make_hand(pinch=PINCHED, **hand_kwargs), pinch_ms)
    driver.hold(make_hand(pinch=OPEN, **hand_kwargs), 66)


INPUT_KINDS = (
    EventKind.MOVE,
    EventKind.BUTTON_DOWN,
    EventKind.BUTTON_UP,
    EventKind.CLICK,
    EventKind.SCROLL,
)


# ---- voter -------------------------------------------------------------- #


def test_voter_needs_majority_to_switch():
    voter = GestureVoter(window=5, min_score=0.6)
    assert [voter.push("None", 0.9) for _ in range(5)][-1] == "None"
    assert voter.push("Victory", 0.9) == "None"
    assert voter.push("Victory", 0.9) == "None"
    assert voter.push("Victory", 0.9) == "Victory"  # 3 ใน 5


def test_voter_ignores_low_scores_and_single_frame_glitches():
    voter = GestureVoter(window=5, min_score=0.6)
    for _ in range(5):
        voter.push("Victory", 0.9)
    assert voter.push("Open_Palm", 0.95) == "Victory"  # เฟรมเดียวไม่พอ
    for _ in range(5):
        assert voter.push("Open_Palm", 0.5) in ("Victory", "None")  # score ต่ำ = None
    assert voter.current == "None"


def test_voter_tie_keeps_current_gesture():
    voter = GestureVoter(window=4, min_score=0.6)
    for _ in range(4):
        voter.push("Victory", 0.9)
    for label in ("None", "None"):
        result = voter.push(label, 0.9)
    assert result == "Victory"  # 2:2 → คงท่าเดิม
    assert voter.push("None", 0.9) == "None"  # 1:3
    voter.reset()
    assert voter.current == "None"


def test_voter_needs_majority_even_right_after_reset():
    """เพิ่งเริ่ม/เพิ่ง reset: เฟรมเดียวตัดสินท่าไม่ได้ (เฟรมที่ยังไม่เห็นนับเป็น None)"""
    voter = GestureVoter(window=5, min_score=0.6)
    assert voter.push("Victory", 0.9) == "None"
    assert voter.push("Victory", 0.9) == "None"
    assert voter.push("Victory", 0.9) == "Victory"
    voter.reset()
    assert voter.push("Open_Palm", 0.9) == "None"


# ---- DISARMED ----------------------------------------------------------- #


def test_starts_disarmed_and_emits_nothing():
    driver = Driver(make_engine())
    assert driver.engine.state is State.DISARMED and not driver.engine.armed
    driver.hold(make_hand(), 200)
    click(driver)
    driver.hold(make_hand(pinch=PINCHED), 600)  # นานพอจะเป็นการลาก ถ้า arm อยู่
    driver.hold(make_hand(gesture="Victory", y=0.2), 300)
    assert driver.events == []
    assert set(driver.states) == {State.DISARMED}


def test_disarmed_still_tracks_ghost_cursor_for_hud():
    driver = Driver(make_engine())
    driver.hold(make_hand(0.5, 0.4), 100)
    snapshot = driver.engine.snapshot
    assert snapshot.state is State.DISARMED and not snapshot.armed
    assert snapshot.hand_present and snapshot.hands == 1
    assert snapshot.cursor == pytest.approx(CENTER, abs=1)


def test_arm_and_disarm_events():
    engine = make_engine()
    assert [e.kind for e in engine.set_armed(True)] == [EventKind.ARMED]
    assert engine.state is State.IDLE and engine.snapshot.armed
    assert engine.set_armed(True) == []  # ซ้ำ = ไม่มีอะไรเกิด
    assert [e.kind for e in engine.set_armed(False)] == [EventKind.DISARMED]
    assert engine.state is State.DISARMED and not engine.snapshot.armed
    assert engine.set_armed(False) == []


# ---- POINTER ------------------------------------------------------------ #


def test_hand_appearing_moves_cursor_to_mapped_position():
    driver = Driver(make_engine())
    driver.engine.set_armed(True)
    events = driver.step(make_hand(0.5, 0.4))
    assert driver.engine.state is State.POINTER
    assert [e.kind for e in events] == [EventKind.MOVE]
    assert (events[0].x, events[0].y) == pytest.approx(CENTER, abs=1)


@pytest.mark.parametrize(
    ("anchor", "expected"),
    [
        ((0.15, 0.10), (0, 0)),
        ((0.85, 0.70), (1919, 1079)),
        ((0.15, 0.70), (0, 1079)),
        ((0.02, 0.95), (0, 1079)),  # นอก active region → ขอบจอ
    ],
)
def test_active_region_corners_reach_screen_corners(anchor, expected):
    driver = Driver(make_engine())
    driver.engine.set_armed(True)
    driver.hold(make_hand(*anchor), 100)
    assert driver.engine.snapshot.cursor == expected


def test_cursor_follows_hand():
    driver = pointing()
    driver.animate(lambda p: make_hand(0.5 + 0.2 * p, 0.4), 500)
    driver.hold(make_hand(0.7, 0.4), 1500)  # ให้ฟิลเตอร์ลู่เข้า
    x, y = driver.engine.snapshot.cursor
    expected_x = (0.7 - 0.15) / 0.7 * 1919
    assert x == pytest.approx(expected_x, abs=2) and y == pytest.approx(540, abs=2)
    moves = of_kind(driver.events, EventKind.MOVE)
    assert [m.x for m in moves] == sorted(m.x for m in moves)  # ไม่กระตุกย้อนกลับ


def test_still_hand_emits_no_moves():
    driver = pointing()
    driver.hold(make_hand(), 1000)
    assert driver.events == []


def test_jitter_on_still_hand_is_suppressed():
    """มือนิ่งแต่ landmark สั่น ±0.75px ของกล้อง (≈ ±3.2px บนจอถ้าไม่กรอง): เคอร์เซอร์ต้องแทบไม่ขยับ"""
    driver = pointing()
    rng = random.Random(11)
    positions = []
    for _ in range(90):
        jx, jy = rng.uniform(-0.75, 0.75) / 640, rng.uniform(-0.75, 0.75) / 480
        driver.step(make_hand(0.5 + jx, 0.4 + jy))
        positions.append(driver.engine.snapshot.cursor)
    xs, ys = [p[0] for p in positions], [p[1] for p in positions]
    assert max(xs) - min(xs) <= 3 and max(ys) - min(ys) <= 3


def test_non_dominant_hand_does_not_control():
    driver = Driver(make_engine())
    driver.engine.set_armed(True)
    driver.hold(make_hand(side="left"), 300)
    click(driver, side="left")
    assert driver.events == []
    assert driver.engine.state is State.IDLE
    assert driver.engine.snapshot.hands == 1 and not driver.engine.snapshot.hand_present


def test_dominant_hand_setting_is_respected():
    driver = Driver(make_engine({"tracking": {"dominant_hand": "left"}}))
    driver.engine.set_armed(True)
    driver.hold(make_hand(side="right"), 200)
    assert driver.events == []
    driver.hold(make_hand(side="left"), 100)
    assert driver.engine.state is State.POINTER


def test_second_hand_labelled_same_side_does_not_steal_cursor():
    driver = pointing(x=0.3)
    before = driver.engine.snapshot.cursor
    intruder = make_hand(0.75, 0.4, side_score=0.99)  # มั่นใจกว่า แต่ไม่ใช่มือที่ตามอยู่
    for _ in range(10):
        driver.step(make_hand(0.3, 0.4, side_score=0.7), intruder)
        driver.step(intruder, make_hand(0.3, 0.4, side_score=0.7))  # ลำดับสลับก็ต้องไม่เปลี่ยน
    assert driver.engine.snapshot.cursor == pytest.approx(before, abs=1)


# ---- click -------------------------------------------------------------- #


def test_quick_pinch_is_one_left_click():
    driver = pointing()
    click(driver)
    clicks = of_kind(driver.events, EventKind.CLICK)
    assert len(clicks) == 1
    assert (clicks[0].button, clicks[0].count) == ("left", 1)
    assert (clicks[0].x, clicks[0].y) == pytest.approx(CENTER, abs=1)
    assert of_kind(driver.events, EventKind.BUTTON_DOWN, EventKind.BUTTON_UP) == []
    assert State.PINCH_PENDING in driver.states and driver.engine.state is State.POINTER


def test_click_fires_on_release_not_on_pinch():
    driver = pointing()
    assert of_kind(driver.hold(make_hand(pinch=PINCHED), 100), EventKind.CLICK) == []
    assert len(of_kind(driver.step(make_hand(pinch=OPEN)), EventKind.CLICK)) == 1


def test_hysteresis_between_on_and_off():
    """ratio ระหว่าง pinch.on (0.25) กับ pinch.off (0.38): ไม่เริ่มจีบ และไม่ถือว่าปล่อย"""
    driver = pointing()
    driver.hold(make_hand(pinch=0.30), 200)
    assert driver.engine.state is State.POINTER  # 0.30 > on → ยังไม่จีบ
    driver.step(make_hand(pinch=0.20))
    assert driver.engine.state is State.PINCH_PENDING
    driver.step(make_hand(pinch=0.30))
    assert driver.engine.state is State.PINCH_PENDING  # 0.30 < off → ยังไม่ปล่อย
    events = driver.step(make_hand(pinch=0.40))
    assert len(of_kind(events, EventKind.CLICK)) == 1


def test_single_frame_pinch_glitch_is_not_a_click():
    driver = pointing()
    driver.step(make_hand(pinch=PINCHED))
    driver.hold(make_hand(pinch=OPEN), 200)
    assert of_kind(driver.events, EventKind.CLICK, EventKind.BUTTON_DOWN) == []
    assert driver.engine.state is State.POINTER


def test_cursor_barely_moves_during_click():
    """เกณฑ์ M1: เคอร์เซอร์ขยับตอนคลิกไม่เกิน ~10px แม้มือจะไหล 3px ของกล้องระหว่างจีบ"""
    driver = pointing()
    start = driver.engine.snapshot.cursor
    drift = 3 / 640  # 3px ของกล้อง ≈ 13px บนจอถ้าไม่มี pinch-lock

    driver.animate(
        lambda p: make_hand(0.5 + drift * p, 0.4, pinch=OPEN - (OPEN - PINCHED) * p), 165
    )
    driver.hold(make_hand(0.5 + drift, 0.4, pinch=PINCHED), 66)
    driver.animate(lambda p: make_hand(0.5 + drift, 0.4, pinch=PINCHED + (OPEN - PINCHED) * p), 100)

    clicks = of_kind(driver.events, EventKind.CLICK)
    assert len(clicks) == 1
    assert math.hypot(clicks[0].x - start[0], clicks[0].y - start[1]) <= 10


def test_pinch_lock_reduces_gain_when_fingers_are_close():
    """ratio < pinch.on × 1.4 (= 0.35) → gain เหลือ 20%"""
    free = pointing()
    free.animate(lambda p: make_hand(0.5 + 0.1 * p, 0.4, pinch=OPEN), 400)
    locked = pointing()
    locked.hold(make_hand(pinch=0.30), 100)  # ใกล้จีบ แต่ยังไม่จีบ
    assert locked.engine.state is State.POINTER
    before = locked.engine.snapshot.cursor[0]
    locked.animate(lambda p: make_hand(0.5 + 0.1 * p, 0.4, pinch=0.30), 400)

    free_travel = free.engine.snapshot.cursor[0] - CENTER[0]
    locked_travel = locked.engine.snapshot.cursor[0] - before
    assert free_travel > 100
    assert locked_travel == pytest.approx(0.2 * free_travel, rel=0.15)


def test_fist_is_not_a_pinch():
    """กำมือ: ปลายนิ้วชี้ชิดนิ้วโป้ง แต่พับอยู่ในฝ่ามือ — ต้องไม่คลิกและไม่ลาก"""
    driver = pointing()
    driver.hold(make_hand(fist=True, gesture="Closed_Fist"), 800)
    driver.hold(make_hand(), 200)
    assert of_kind(driver.events, EventKind.CLICK, EventKind.BUTTON_DOWN) == []


def test_pinch_that_turns_into_fist_is_cancelled():
    driver = pointing()
    driver.hold(make_hand(pinch=PINCHED), 66)
    assert driver.engine.state is State.PINCH_PENDING
    driver.hold(make_hand(fist=True), 600)
    driver.hold(make_hand(), 200)
    assert of_kind(driver.events, EventKind.CLICK, EventKind.BUTTON_DOWN) == []
    assert driver.engine.state is State.POINTER


def test_hand_arriving_already_pinched_does_not_click_or_drag():
    driver = Driver(make_engine())
    driver.engine.set_armed(True)
    driver.hold(make_hand(pinch=PINCHED), 600)
    driver.hold(make_hand(), 100)
    assert of_kind(driver.events, EventKind.CLICK, EventKind.BUTTON_DOWN) == []
    click(driver)  # หลังเห็นนิ้วห่างแล้วหนึ่งครั้ง จึงจีบได้ตามปกติ
    assert len(of_kind(driver.events, EventKind.CLICK)) == 1


def test_arming_while_pinching_does_not_click_or_drag():
    driver = Driver(make_engine())
    driver.hold(make_hand(pinch=PINCHED), 300)
    driver.engine.set_armed(True)
    driver.hold(make_hand(pinch=PINCHED), 600)
    driver.hold(make_hand(), 100)
    assert of_kind(driver.events, EventKind.CLICK, EventKind.BUTTON_DOWN) == []


# ---- double click ------------------------------------------------------- #


def test_two_quick_pinches_make_a_double_click_at_the_first_position():
    driver = pointing()
    click(driver)
    moved = {"x": 0.51, "y": 0.41}  # มือขยับไป ~43px บนจอระหว่างจีบสองครั้ง
    driver.hold(make_hand(pinch=OPEN, **moved), 100)
    driver.hold(make_hand(pinch=PINCHED, **moved), 100)
    driver.step(make_hand(pinch=OPEN, **moved))
    clicks = of_kind(driver.events, EventKind.CLICK)
    assert [c.count for c in clicks] == [1, 2]
    assert (clicks[1].x, clicks[1].y) == (clicks[0].x, clicks[0].y)
    # เคอร์เซอร์ถูกดึงกลับไปที่จุดคลิกแรก (OS จึงนับเป็นดับเบิลคลิก)
    assert driver.engine.snapshot.cursor == (clicks[0].x, clicks[0].y)


def test_slow_second_pinch_is_a_separate_click():
    driver = pointing()
    click(driver)
    driver.hold(make_hand(), DEFAULT_DOUBLE_CLICK_MS + 100)
    click(driver)
    assert [c.count for c in of_kind(driver.events, EventKind.CLICK)] == [1, 1]


def test_second_pinch_far_away_is_a_separate_click():
    driver = pointing()
    click(driver)
    driver.animate(lambda p: make_hand(0.5 + 0.2 * p, 0.4), 150)
    click(driver, x=0.7)
    clicks = of_kind(driver.events, EventKind.CLICK)
    assert [c.count for c in clicks] == [1, 1]
    assert clicks[1].x > clicks[0].x + 100


def test_triple_pinch_is_double_then_single():
    driver = pointing()
    for _ in range(3):
        click(driver)
    assert [c.count for c in of_kind(driver.events, EventKind.CLICK)] == [1, 2, 1]


def test_double_click_time_comes_from_os_setting():
    config = Config()
    quick = Driver(GestureEngine(config, FHD, double_click_ms=200))
    quick.engine.set_armed(True)
    quick.hold(make_hand(), 200)
    click(quick)
    quick.hold(make_hand(), 150)
    click(quick)  # ห่างกัน > 200ms
    assert [c.count for c in of_kind(quick.events, EventKind.CLICK)] == [1, 1]


# ---- drag --------------------------------------------------------------- #


def test_held_pinch_becomes_drag():
    driver = pointing()
    driver.hold(make_hand(pinch=PINCHED), 200)
    assert driver.engine.state is State.PINCH_PENDING
    assert of_kind(driver.events, EventKind.BUTTON_DOWN) == []  # ยังไม่ถึง 250ms
    driver.hold(make_hand(pinch=PINCHED), 100)
    assert driver.engine.state is State.DRAGGING
    downs = of_kind(driver.events, EventKind.BUTTON_DOWN)
    assert len(downs) == 1 and downs[0].button == "left"
    assert (downs[0].x, downs[0].y) == pytest.approx(CENTER, abs=2)

    driver.animate(lambda p: make_hand(0.5 + 0.2 * p, 0.4 + 0.1 * p, pinch=PINCHED), 500)
    driver.hold(make_hand(0.7, 0.5, pinch=OPEN), 100)
    ups = of_kind(driver.events, EventKind.BUTTON_UP)
    assert len(ups) == 1 and ups[0].reason == "release"
    assert ups[0].x > downs[0].x + 200 and ups[0].y > downs[0].y + 50
    assert of_kind(driver.events, EventKind.CLICK) == []
    assert driver.engine.state is State.POINTER

    order = [e.kind for e in driver.events if e.kind is not EventKind.MOVE]
    assert order == [EventKind.BUTTON_DOWN, EventKind.BUTTON_UP]
    kinds = [e.kind for e in driver.events]
    down_at, up_at = kinds.index(EventKind.BUTTON_DOWN), kinds.index(EventKind.BUTTON_UP)
    assert kinds[down_at + 1 : up_at].count(EventKind.MOVE) >= 5  # ลากจริง มี move ระหว่างกด


def test_drag_moves_at_full_gain():
    driver = pointing()
    driver.hold(make_hand(pinch=PINCHED), 330)
    assert driver.engine.state is State.DRAGGING
    start = driver.engine.snapshot.cursor[0]
    driver.animate(lambda p: make_hand(0.5 + 0.2 * p, 0.4, pinch=PINCHED), 500)
    driver.hold(make_hand(0.7, 0.4, pinch=PINCHED), 1500)
    travelled = driver.engine.snapshot.cursor[0] - start
    assert travelled == pytest.approx(0.2 / 0.7 * 1919, rel=0.1)


def test_one_frame_release_flicker_does_not_drop_the_drag():
    driver = pointing()
    driver.hold(make_hand(pinch=PINCHED), 330)
    driver.step(make_hand(pinch=OPEN))  # landmark เพี้ยนเฟรมเดียว
    driver.hold(make_hand(pinch=PINCHED), 200)
    assert of_kind(driver.events, EventKind.BUTTON_UP) == []
    assert driver.engine.state is State.DRAGGING


def test_snapshot_reports_pinch_progress():
    driver = pointing()
    assert driver.engine.snapshot.pinch_level == 0.0
    assert driver.engine.snapshot.pinch_ratio == pytest.approx(OPEN)
    driver.step(make_hand(pinch=0.315))  # กึ่งกลางระหว่าง on กับ off
    assert driver.engine.snapshot.pinch_level == pytest.approx(0.5, abs=0.01)
    driver.step(make_hand(pinch=PINCHED))
    snapshot = driver.engine.snapshot
    assert snapshot.pinch_level == 1.0 and snapshot.pinch_button == "left"
    assert snapshot.hold_progress == 0.0
    driver.hold(make_hand(pinch=PINCHED), 132)
    assert 0.4 < driver.engine.snapshot.hold_progress < 0.7
    driver.hold(make_hand(pinch=PINCHED), 200)
    snapshot = driver.engine.snapshot
    assert snapshot.state is State.DRAGGING and snapshot.hold_progress == 1.0
    assert snapshot.pinch_button == "left"


# ---- safety ------------------------------------------------------------- #


def _dragging() -> Driver:
    driver = pointing()
    driver.hold(make_hand(pinch=PINCHED), 330)
    assert driver.engine.state is State.DRAGGING
    driver.clear()
    return driver


def test_hand_lost_briefly_keeps_dragging():
    driver = _dragging()
    driver.hold(None, 200)
    assert driver.events == [] and driver.engine.state is State.DRAGGING
    driver.hold(make_hand(pinch=PINCHED), 100)
    assert of_kind(driver.events, EventKind.BUTTON_UP) == []


def test_hand_lost_over_300ms_releases_button():
    driver = _dragging()
    last_seen = driver.t - FRAME_MS
    events = driver.hold(None, 400)
    ups = of_kind(events, EventKind.BUTTON_UP)
    assert len(ups) == 1 and ups[0].reason == "hand_lost" and ups[0].button == "left"
    assert driver.engine.state is State.IDLE
    # ปล่อยที่เฟรมแรกที่มือหายครบ 300ms พอดี ไม่ช้ากว่านั้น
    release_frame = next(i for i, s in enumerate(driver.states) if s is State.IDLE)
    release_time = last_seen + (release_frame + 1) * FRAME_MS
    assert LOST_RELEASE_MS <= release_time - last_seen < LOST_RELEASE_MS + FRAME_MS
    # มือกลับมาในท่าจีบ: ต้องไม่กดซ้ำเอง
    driver.clear()
    driver.hold(make_hand(pinch=PINCHED), 600)
    assert of_kind(driver.events, EventKind.BUTTON_DOWN, EventKind.CLICK) == []


def test_tick_releases_when_no_frames_arrive():
    """กล้องค้าง: watchdog เรียก tick() แทนเฟรม ต้องปล่อยปุ่มเหมือนมือหาย"""
    driver = _dragging()
    assert driver.engine.tick(driver.t + 100) == []
    events = driver.engine.tick(driver.t + 400)
    assert [e.kind for e in events] == [EventKind.BUTTON_UP]
    assert driver.engine.state is State.IDLE
    assert driver.engine.tick(driver.t + 800) == []


def test_disarm_while_dragging_releases_first():
    driver = _dragging()
    events = driver.engine.set_armed(False)
    assert [e.kind for e in events] == [EventKind.BUTTON_UP, EventKind.DISARMED]
    assert events[0].reason == "disarm"
    driver.hold(make_hand(pinch=PINCHED), 600)
    driver.hold(make_hand(0.8, 0.2), 300)
    assert driver.events == []  # disarm แล้วเงียบสนิท


def test_pending_pinch_is_cancelled_when_hand_is_lost():
    driver = pointing()
    driver.hold(make_hand(pinch=PINCHED), 100)
    driver.hold(None, 500)
    driver.hold(make_hand(), 200)
    assert of_kind(driver.events, EventKind.CLICK, EventKind.BUTTON_DOWN) == []


def test_every_button_down_is_followed_by_button_up():
    """สุ่มลำดับท่า 2,000 เฟรม: ปุ่มที่กดต้องถูกปล่อยเสมอเมื่อ disarm ตอนจบ"""
    rng = random.Random(2024)
    driver = Driver(make_engine())
    driver.engine.set_armed(True)
    poses = [
        lambda: make_hand(rng.random(), rng.random(), pinch=PINCHED),
        lambda: make_hand(rng.random(), rng.random(), pinch=OPEN),
        lambda: make_hand(rng.random(), rng.random(), middle_pinch=PINCHED),
        lambda: make_hand(rng.random(), rng.random(), gesture="Victory"),
        lambda: make_hand(fist=True),
        lambda: None,
    ]
    for _ in range(200):
        driver.hold(rng.choice(poses)(), rng.choice([33, 66, 200, 400]))
    driver.events.extend(driver.engine.set_armed(False))

    depth = 0
    for event in driver.events:
        if event.kind is EventKind.BUTTON_DOWN:
            depth += 1
            assert depth == 1  # ไม่กดซ้อน
        elif event.kind is EventKind.BUTTON_UP:
            depth -= 1
            assert depth == 0  # ไม่ปล่อยทั้งที่ไม่ได้กด
    assert depth == 0
    assert of_kind(driver.events, EventKind.BUTTON_DOWN)  # test นี้ได้ลากจริง


def test_timestamps_going_backwards_do_not_break_the_engine():
    engine = make_engine()
    engine.set_armed(True)
    engine.update(make_frame(5000, make_hand()))
    engine.update(make_frame(5033, make_hand(0.6, 0.4)))
    state, events = engine.update(make_frame(4000, make_hand(0.7, 0.4)))  # เฟรมเก่ามาทีหลัง
    assert state is State.POINTER
    assert all(e.kind is EventKind.MOVE for e in events)
    state, _ = engine.update(make_frame(5066, make_hand(0.7, 0.4)))
    assert state is State.POINTER


# ---- right click -------------------------------------------------------- #


def test_thumb_middle_pinch_is_one_right_click():
    driver = pointing()
    driver.hold(make_hand(middle_pinch=PINCHED), 600)  # ค้างนานก็ยังคลิกครั้งเดียว ไม่ลาก
    driver.hold(make_hand(), 100)
    clicks = of_kind(driver.events, EventKind.CLICK)
    assert len(clicks) == 1 and clicks[0].button == "right"
    assert of_kind(driver.events, EventKind.BUTTON_DOWN, EventKind.BUTTON_UP) == []
    assert driver.engine.state is State.POINTER


def test_right_click_fires_while_pinched_and_can_repeat_after_release():
    driver = pointing()
    events = driver.hold(make_hand(middle_pinch=PINCHED), 100)
    assert len(of_kind(events, EventKind.CLICK)) == 1  # ไม่ต้องรอปล่อย
    driver.hold(make_hand(), 100)
    driver.hold(make_hand(middle_pinch=PINCHED), 100)
    assert len(of_kind(driver.events, EventKind.CLICK)) == 2


def test_single_frame_middle_pinch_glitch_is_ignored():
    driver = pointing()
    driver.step(make_hand(middle_pinch=PINCHED))
    driver.hold(make_hand(), 200)
    assert of_kind(driver.events, EventKind.CLICK) == []


@pytest.mark.parametrize(
    ("index", "middle", "button"),
    [(0.10, 0.20, "left"), (0.20, 0.10, "right")],
)
def test_closer_finger_wins_when_both_are_near_thumb(index, middle, button):
    driver = pointing()
    driver.hold(make_hand(pinch=index, middle_pinch=middle), 100)
    assert driver.engine.snapshot.pinch_button == button
    driver.hold(make_hand(), 100)
    clicks = of_kind(driver.events, EventKind.CLICK)
    assert len(clicks) == 1 and clicks[0].button == button


def test_right_click_then_left_click_is_not_a_double_click():
    driver = pointing()
    click(driver)
    driver.hold(make_hand(middle_pinch=PINCHED), 100)
    driver.hold(make_hand(), 66)
    click(driver)
    clicks = of_kind(driver.events, EventKind.CLICK)
    assert [(c.button, c.count) for c in clicks] == [("left", 1), ("right", 1), ("left", 1)]


# ---- scroll ------------------------------------------------------------- #


def _scrolling(overrides: dict | None = None) -> Driver:
    driver = pointing(overrides)
    driver.hold(make_hand(gesture="Victory"), 200)
    assert driver.engine.state is State.SCROLL
    driver.clear()
    return driver


def _total(events: list[GestureEvent]) -> tuple[int, int]:
    scrolls = of_kind(events, EventKind.SCROLL)
    return (sum(e.dx for e in scrolls), sum(e.dy for e in scrolls))


def test_victory_enters_scroll_after_vote():
    driver = pointing()
    driver.hold(make_hand(gesture="Victory"), 66)
    assert driver.engine.state is State.POINTER  # 2 ใน 5 เฟรม ยังไม่พอ
    driver.step(make_hand(gesture="Victory"))
    assert driver.engine.state is State.SCROLL
    assert driver.engine.snapshot.gesture == "Victory"


def test_low_confidence_victory_is_ignored():
    driver = pointing()
    driver.hold(make_hand(gesture="Victory", score=0.5), 400)  # ต่ำกว่า min_gesture_score 0.6
    assert driver.engine.state is State.POINTER


def test_hand_up_scrolls_up_and_cursor_stays():
    driver = _scrolling()
    cursor = driver.engine.snapshot.cursor
    driver.animate(lambda p: make_hand(0.5, 0.4 - 0.2 * p, gesture="Victory"), 400)
    dx, dy = _total(driver.events)
    # ขยับ 0.2 ของความสูงเฟรม − ระยะฟรี 0.01 = 0.19 × 30 notch = 5.7 → 5 notch เต็ม
    assert dx == 0 and dy == 5
    assert of_kind(driver.events, EventKind.MOVE) == []
    assert driver.engine.snapshot.cursor == cursor


def test_hand_down_scrolls_down():
    driver = _scrolling()
    driver.animate(lambda p: make_hand(0.5, 0.4 + 0.2 * p, gesture="Victory"), 400)
    assert _total(driver.events) == (0, -5)


def test_horizontal_motion_scrolls_horizontally():
    driver = _scrolling()
    driver.animate(lambda p: make_hand(0.5 + 0.2 * p, 0.4, gesture="Victory"), 400)
    assert _total(driver.events) == (5, 0)
    driver.clear()
    driver.hold(make_hand(0.7, 0.4, gesture="Victory"), 400)  # หยุดพัก → ปลดล็อกแกน
    driver.animate(lambda p: make_hand(0.7 - 0.2 * p, 0.4, gesture="Victory"), 400)
    dx, dy = _total(driver.events)
    assert dx < 0 and dy == 0


def test_scroll_ignores_jitter_inside_deadzone():
    driver = _scrolling()
    rng = random.Random(5)
    for _ in range(60):
        driver.step(
            make_hand(
                0.5 + rng.uniform(-0.004, 0.004),
                0.4 + rng.uniform(-0.004, 0.004),
                gesture="Victory",
            )
        )
    assert driver.events == []


def test_scroll_axis_is_locked_while_moving():
    """สกรอลแนวตั้งอยู่ มือเฉไปด้านข้างบ้าง ต้องไม่สกรอลแนวนอนปน"""
    driver = _scrolling()
    driver.animate(lambda p: make_hand(0.5 + 0.06 * p, 0.4 - 0.25 * p, gesture="Victory"), 500)
    dx, dy = _total(driver.events)
    assert dx == 0 and dy >= 6


def test_scroll_speed_and_invert_settings():
    fast = _scrolling({"scroll": {"speed": 2.0}})
    fast.animate(lambda p: make_hand(0.5, 0.4 - 0.2 * p, gesture="Victory"), 400)
    assert _total(fast.events) == (0, 11)  # 0.19 × 30 × 2 = 11.4

    inverted = _scrolling({"scroll": {"invert": True}})
    inverted.animate(lambda p: make_hand(0.5, 0.4 - 0.2 * p, gesture="Victory"), 400)
    assert _total(inverted.events) == (0, -5)


def test_no_click_or_drag_while_scrolling():
    driver = _scrolling()
    driver.hold(make_hand(gesture="Victory", pinch=PINCHED), 600)
    assert of_kind(driver.events, EventKind.CLICK, EventKind.BUTTON_DOWN) == []


def test_leaving_scroll_does_not_make_cursor_jump():
    driver = _scrolling()
    frozen = driver.engine.snapshot.cursor
    driver.animate(lambda p: make_hand(0.5, 0.4 - 0.25 * p, gesture="Victory"), 400)
    driver.hold(make_hand(0.5, 0.15), 300)  # เลิกทำท่า ✌️ (มืออยู่สูงกว่าเดิมมาก)
    assert driver.engine.state is State.POINTER
    moves = of_kind(driver.events, EventKind.MOVE)
    assert all(math.hypot(m.x - frozen[0], m.y - frozen[1]) < 20 for m in moves)
    assert math.hypot(*(a - b for a, b in zip(driver.engine.snapshot.cursor, frozen))) < 20


def test_hand_lost_while_scrolling_returns_to_idle():
    driver = _scrolling()
    driver.hold(None, 400)
    assert driver.engine.state is State.IDLE and driver.events == []


# ---- ทั่วไป -------------------------------------------------------------- #


def test_events_are_immutable_value_objects():
    event = GestureEvent(EventKind.CLICK, x=1, y=2)
    assert event == GestureEvent(EventKind.CLICK, x=1, y=2)
    with pytest.raises(dataclasses.FrozenInstanceError):
        event.x = 5  # type: ignore[misc]


def test_only_known_input_kinds_are_emitted_while_armed():
    driver = pointing()
    click(driver)
    driver.hold(make_hand(pinch=PINCHED), 400)
    driver.hold(make_hand(), 100)
    assert {e.kind for e in driver.events} <= set(INPUT_KINDS)
