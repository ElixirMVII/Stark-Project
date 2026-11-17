"""Gesture engine (M3): แบมือปัด 4 ทิศ, กำมือค้าง, 🤟 สลับ ARM, ซูมสองมือ"""

from __future__ import annotations

import math

import pytest
from handsim import OPEN, PINCHED, Driver, make_hand
from test_gestures import CENTER, click, make_engine, of_kind, pointing

from aether.gestures import (
    SWIPE_REST_MS,
    ZOOM_SYNC_MS,
    EventKind,
    GestureEvent,
    State,
    detect_swipe,
)

PALM = {"gesture": "Open_Palm"}
FIST = {"fist": True, "gesture": "Closed_Fist"}
ARM = {"gesture": "ILoveYou"}

INPUT_KINDS = (
    EventKind.MOVE,
    EventKind.BUTTON_DOWN,
    EventKind.BUTTON_UP,
    EventKind.CLICK,
    EventKind.SCROLL,
    EventKind.ZOOM,
    EventKind.ACTION,
)


def actions(events: list[GestureEvent]) -> list[str]:
    return [e.name for e in events if e.kind is EventKind.ACTION]


def palm_ready(overrides: dict | None = None, x: float = 0.5, y: float = 0.4) -> Driver:
    """Driver ที่อยู่ในโหมด GESTURE และพร้อมรับการปัด (แบมือนิ่งมาแล้ว)"""
    driver = pointing(overrides, x=x, y=y)
    driver.hold(make_hand(x, y, **PALM), 400)
    assert driver.engine.state is State.GESTURE
    assert driver.engine.snapshot.swipe_ready
    driver.clear()
    return driver


def swipe(driver: Driver, start: tuple[float, float], end: tuple[float, float], ms: int = 200):
    (x0, y0), (x1, y1) = start, end
    return driver.animate(lambda p: make_hand(x0 + (x1 - x0) * p, y0 + (y1 - y0) * p, **PALM), ms)


# ---- detect_swipe (pure) -------------------------------------------------- #


def _history(points: list[tuple[int, float, float]]):
    return points


@pytest.mark.parametrize(
    ("end", "expected"),
    [
        ((0.80, 0.40), "right"),
        ((0.20, 0.40), "left"),
        ((0.50, 0.15), "up"),
        ((0.50, 0.65), "down"),
        ((0.70, 0.40), None),  # 0.20 < 0.25 ของความกว้าง
        ((0.50, 0.55), None),  # 0.15 < 0.20 ของความสูง
        ((0.80, 0.70), None),  # เฉียง: แกนหลักไม่ถึง 2 เท่าของแกนรอง (192px : 144px)
        ((0.80, 0.60), "right"),  # 192px : 96px = 2:1 พอดี → นับ
        ((0.80, 0.45), "right"),  # เฉไปนิดเดียว ยังนับ
    ],
)
def test_detect_swipe_thresholds(end, expected):
    history = [(0, 0.5, 0.4), (100, (0.5 + end[0]) / 2, (0.4 + end[1]) / 2), (200, *end)]
    assert detect_swipe(history, 640, 480, 0.25, 0.20, 300) == expected


def test_detect_swipe_requires_speed():
    """ขยับไกลพอแต่ช้า (เกินหน้าต่างเวลา) ไม่นับ"""
    slow = [(t, 0.5 + 0.3 * t / 1000, 0.4) for t in range(0, 1001, 33)]
    assert detect_swipe(slow, 640, 480, 0.25, 0.20, 300) is None
    fast = [(t, 0.5 + 0.3 * t / 250, 0.4) for t in range(0, 251, 33)] + [(250, 0.8, 0.4)]
    assert detect_swipe(fast, 640, 480, 0.25, 0.20, 300) == "right"


def test_detect_swipe_axis_ratio_uses_pixels_not_normalized():
    # dx = 0.25 (160px), dy = 0.15 (72px): 160 ≥ 2×72 → แนวนอน แม้ค่า normalized จะไม่ถึง 2:1
    assert detect_swipe([(0, 0.5, 0.4), (200, 0.75, 0.55)], 640, 480, 0.25, 0.20, 300) == "right"
    assert detect_swipe([(0, 0.5, 0.4)], 640, 480, 0.25, 0.20, 300) is None
    assert detect_swipe([], 640, 480, 0.25, 0.20, 300) is None


# ---- GESTURE: แบมือ + ปัด -------------------------------------------------- #


def test_open_palm_enters_gesture_mode_and_freezes_cursor():
    driver = pointing()
    cursor = driver.engine.snapshot.cursor
    driver.hold(make_hand(**PALM), 100)
    assert driver.engine.state is State.GESTURE  # โหวตครบ 3 เฟรม
    assert not driver.engine.snapshot.swipe_ready  # ยังไม่ได้นิ่งนานพอ
    driver.hold(make_hand(**PALM), SWIPE_REST_MS + 66)
    assert driver.engine.snapshot.swipe_ready
    driver.clear()
    driver.animate(lambda p: make_hand(0.5 + 0.05 * p, 0.4, **PALM), 600)  # ขยับช้าๆ
    assert driver.events == []
    assert driver.engine.snapshot.cursor == cursor


@pytest.mark.parametrize(
    ("end", "name"),
    [
        ((0.80, 0.40), "swipe_right"),
        ((0.20, 0.40), "swipe_left"),
        ((0.50, 0.15), "swipe_up"),
        ((0.50, 0.65), "swipe_down"),
    ],
)
def test_swipe_in_each_direction(end, name):
    driver = palm_ready()
    swipe(driver, (0.5, 0.4), end)
    driver.hold(make_hand(*end, **PALM), 300)
    assert actions(driver.events) == [name]
    assert of_kind(driver.events, EventKind.MOVE, EventKind.CLICK, EventKind.BUTTON_DOWN) == []


def test_swipe_event_carries_cursor_position():
    driver = palm_ready()
    swipe(driver, (0.5, 0.4), (0.8, 0.4))
    event = of_kind(driver.events, EventKind.ACTION)[0]
    assert (event.x, event.y) == pytest.approx(CENTER, abs=2)


def test_raising_an_open_hand_into_frame_is_not_a_swipe():
    """ยกมือที่แบอยู่เข้าเฟรมเร็วๆ: ยังไม่เคยนิ่ง → ไม่นับเป็นการปัดขึ้น"""
    driver = Driver(make_engine())
    driver.engine.set_armed(True)
    driver.animate(lambda p: make_hand(0.5, 0.75 - 0.45 * p, **PALM), 300)
    driver.hold(make_hand(0.5, 0.30, **PALM), 500)
    assert actions(driver.events) == []
    assert driver.engine.state is State.GESTURE and driver.engine.snapshot.swipe_ready


def test_return_stroke_does_not_trigger_opposite_swipe():
    driver = palm_ready()
    swipe(driver, (0.5, 0.4), (0.8, 0.4))
    swipe(driver, (0.8, 0.4), (0.5, 0.4))  # ดึงมือกลับทันที
    driver.hold(make_hand(0.5, 0.4, **PALM), 200)
    assert actions(driver.events) == ["swipe_right"]


def test_second_swipe_needs_cooldown_and_rest():
    driver = palm_ready()
    swipe(driver, (0.5, 0.4), (0.8, 0.4))
    swipe(driver, (0.8, 0.4), (0.5, 0.4))
    driver.hold(make_hand(0.5, 0.4, **PALM), 250)  # ยังอยู่ใน cooldown 800ms
    swipe(driver, (0.5, 0.4), (0.8, 0.4))
    assert actions(driver.events) == ["swipe_right"]

    swipe(driver, (0.8, 0.4), (0.5, 0.4), ms=600)
    driver.hold(make_hand(0.5, 0.4, **PALM), 800)  # พ้น cooldown + นิ่ง
    assert driver.engine.snapshot.swipe_ready
    swipe(driver, (0.5, 0.4), (0.2, 0.4))
    assert actions(driver.events) == ["swipe_right", "swipe_left"]


def test_swipe_survives_brief_loss_of_palm_label():
    """ปัดเร็วจนภาพเบลอ โมเดลไม่เห็นเป็นแบมือ 4 เฟรม: ยังต้องนับการปัด"""
    driver = palm_ready()
    frames = 7
    for i in range(frames):
        x = 0.5 + 0.32 * (i + 1) / frames
        blurred = 1 <= i <= 4
        driver.step(make_hand(x, 0.4, gesture="None" if blurred else "Open_Palm"))
    assert actions(driver.events) == ["swipe_right"]


def test_slow_drift_is_not_a_swipe():
    driver = palm_ready()
    driver.animate(lambda p: make_hand(0.5 + 0.33 * p, 0.4, **PALM), 1500)
    assert actions(driver.events) == []


def test_palm_outside_active_region_is_ignored():
    """มือวางพักต่ำกว่า active region (y > 0.75): ท่าพิเศษไม่เริ่ม"""
    driver = pointing(x=0.5, y=0.85)
    driver.hold(make_hand(0.5, 0.85, **PALM), 500)
    assert driver.engine.state is State.POINTER
    driver.hold(make_hand(0.5, 0.85, **FIST), 1200)
    driver.hold(make_hand(0.5, 0.85, **ARM), 1500)
    assert actions(driver.events) == []
    assert driver.engine.armed


def test_leaving_gesture_mode_does_not_jump_cursor():
    driver = palm_ready()
    frozen = driver.engine.snapshot.cursor
    driver.animate(lambda p: make_hand(0.5 + 0.2 * p, 0.4, **PALM), 1200)  # ขยับช้าๆ ไปไกล
    driver.hold(make_hand(0.7, 0.4), 500)  # เลิกแบมือ
    assert driver.engine.state is State.POINTER
    moves = of_kind(driver.events, EventKind.MOVE)
    assert all(math.hypot(m.x - frozen[0], m.y - frozen[1]) < 25 for m in moves)
    assert actions(driver.events) == []


def test_pinch_straight_after_open_palm_still_clicks():
    driver = palm_ready()
    click(driver)  # ไม่ต้องรอให้หมดช่วงผ่อนผันของโหมด GESTURE
    clicks = of_kind(driver.events, EventKind.CLICK)
    assert len(clicks) == 1 and clicks[0].button == "left"
    assert driver.engine.state is State.POINTER


def test_swipe_thresholds_come_from_config():
    easy = palm_ready({"gestures": {"swipe_dist_x": 0.10}})
    swipe(easy, (0.5, 0.4), (0.62, 0.4))
    assert actions(easy.events) == ["swipe_right"]
    default = palm_ready()
    swipe(default, (0.5, 0.4), (0.62, 0.4))
    assert actions(default.events) == []


# ---- CHARGING: กำมือค้าง --------------------------------------------------- #


def test_fist_hold_fires_once():
    driver = pointing()
    driver.hold(make_hand(**FIST), 500)
    assert driver.engine.state is State.CHARGING
    assert actions(driver.events) == []  # ยังไม่ครบ 600ms (+เวลาโหวต)
    snapshot = driver.engine.snapshot
    assert snapshot.charge_kind == "fist_hold" and 0.3 < snapshot.charge < 1.0

    driver.hold(make_hand(**FIST), 400)
    assert actions(driver.events) == ["fist_hold"]
    driver.hold(make_hand(**FIST), 3000)  # กำค้างต่อ: ไม่ยิงซ้ำ
    assert actions(driver.events) == ["fist_hold"]
    assert of_kind(driver.events, EventKind.CLICK, EventKind.BUTTON_DOWN) == []
    assert driver.engine.snapshot.charge == 0.0


def test_fist_released_early_does_nothing():
    driver = pointing()
    driver.hold(make_hand(**FIST), 400)
    driver.hold(make_hand(), 300)
    assert actions(driver.events) == []
    assert driver.engine.state is State.POINTER


def test_fist_hold_again_after_release_and_cooldown():
    driver = pointing()
    driver.hold(make_hand(**FIST), 900)
    driver.hold(make_hand(), 200)  # ปล่อยแล้ว แต่ยังอยู่ใน cooldown
    driver.hold(make_hand(**FIST), 500)
    assert actions(driver.events) == ["fist_hold"]
    driver.hold(make_hand(), 1000)
    driver.hold(make_hand(**FIST), 900)
    assert actions(driver.events) == ["fist_hold", "fist_hold"]


def test_cursor_is_frozen_while_charging():
    driver = pointing()
    driver.hold(make_hand(**FIST), 200)
    assert driver.engine.state is State.CHARGING
    cursor = driver.engine.snapshot.cursor
    driver.clear()
    driver.animate(lambda p: make_hand(0.5 + 0.1 * p, 0.4, **FIST), 300)
    assert of_kind(driver.events, EventKind.MOVE) == []
    assert driver.engine.snapshot.cursor == cursor


def test_fist_hold_duration_from_config():
    quick = pointing({"gestures": {"fist_hold_ms": 200}})
    quick.hold(make_hand(**FIST), 400)
    assert actions(quick.events) == ["fist_hold"]


# ---- 🤟 สลับ ARM ----------------------------------------------------------- #


def test_iloveyou_hold_arms_from_disarmed():
    driver = Driver(make_engine())
    driver.hold(make_hand(), 200)
    driver.hold(make_hand(**ARM), 800)
    assert not driver.engine.armed and driver.events == []
    snapshot = driver.engine.snapshot
    assert snapshot.state is State.DISARMED  # ยัง DISARMED แต่ HUD เห็นวงแหวนชาร์จ
    assert snapshot.charge_kind == "arm" and 0.5 < snapshot.charge < 1.0

    driver.hold(make_hand(**ARM), 400)
    assert driver.engine.armed
    # arm แล้วมือเริ่มคุมเคอร์เซอร์ทันที (MOVE) แต่ไม่มี input อื่น
    assert [e.kind for e in driver.events if e.kind is not EventKind.MOVE] == [EventKind.ARMED]


def test_holding_iloveyou_does_not_toggle_back():
    driver = Driver(make_engine())
    driver.hold(make_hand(), 200)
    driver.hold(make_hand(**ARM), 5000)  # ค้างยาวมาก
    assert driver.engine.armed
    assert [e.kind for e in driver.events if e.kind is not EventKind.MOVE] == [EventKind.ARMED]


def test_iloveyou_again_disarms():
    driver = Driver(make_engine())
    driver.hold(make_hand(), 200)
    driver.hold(make_hand(**ARM), 1300)
    driver.hold(make_hand(), 1000)  # เลิกทำท่า + พ้น cooldown
    assert driver.engine.state is State.POINTER
    driver.hold(make_hand(**ARM), 1300)
    assert not driver.engine.armed
    kinds = [e.kind for e in driver.events if e.kind is not EventKind.MOVE]
    assert kinds == [EventKind.ARMED, EventKind.DISARMED]
    assert driver.engine.state is State.DISARMED


def test_short_iloveyou_does_nothing():
    driver = Driver(make_engine())
    driver.hold(make_hand(), 200)
    driver.hold(make_hand(**ARM), 700)
    driver.hold(make_hand(), 300)
    assert not driver.engine.armed and driver.events == []
    assert driver.engine.snapshot.charge == 0.0


def test_other_gestures_do_nothing_while_disarmed():
    driver = Driver(make_engine())
    driver.hold(make_hand(), 200)
    driver.hold(make_hand(**PALM), 400)
    driver.animate(lambda p: make_hand(0.5 + 0.3 * p, 0.4, **PALM), 200)
    driver.hold(make_hand(**FIST), 1500)
    driver.hold(make_hand(gesture="Victory", y=0.2), 500)
    assert driver.events == []
    assert set(driver.states) == {State.DISARMED}


def test_gesture_toggle_and_hotkey_toggle_interleave():
    """arm ด้วย 🤟 แล้ว disarm ด้วย hotkey ขณะยังทำท่าค้าง: ต้องไม่ arm กลับเอง"""
    driver = Driver(make_engine())
    driver.hold(make_hand(), 200)
    driver.hold(make_hand(**ARM), 1300)
    assert driver.engine.armed
    assert [e.kind for e in driver.engine.set_armed(False)] == [EventKind.DISARMED]
    driver.hold(make_hand(**ARM), 3000)
    assert not driver.engine.armed


# ---- TWO_HAND_ZOOM --------------------------------------------------------- #


def two_hands(right: dict, left: dict) -> tuple:
    return (
        make_hand(**{"x": 0.6, "y": 0.4, **right}),
        make_hand(**{"x": 0.4, "y": 0.4, "side": "left", **left}),
    )


def zooming() -> Driver:
    driver = Driver(make_engine())
    driver.engine.set_armed(True)
    for _ in range(6):
        driver.step(*two_hands({}, {}))
    for _ in range(3):
        driver.step(*two_hands({"pinch": PINCHED}, {"pinch": PINCHED}))
    assert driver.engine.state is State.TWO_HAND_ZOOM
    driver.clear()
    return driver


def spread(driver: Driver, start: float, end: float, frames: int = 12) -> None:
    """ระยะครึ่งหนึ่งระหว่างสองมือ (normalized) จาก start → end"""
    for i in range(frames):
        half = start + (end - start) * (i + 1) / frames
        driver.step(
            make_hand(0.5 + half, 0.4, pinch=PINCHED),
            make_hand(0.5 - half, 0.4, pinch=PINCHED, side="left"),
        )


def zoom_total(events: list[GestureEvent]) -> int:
    return sum(e.dy for e in events if e.kind is EventKind.ZOOM)


def test_both_hands_pinching_enters_zoom_without_click_or_drag():
    driver = Driver(make_engine())
    driver.engine.set_armed(True)
    for _ in range(6):
        driver.step(*two_hands({}, {}))
    for _ in range(20):  # จีบค้างนานกว่า drag_hold_ms
        driver.step(*two_hands({"pinch": PINCHED}, {"pinch": PINCHED}))
    assert driver.engine.state is State.TWO_HAND_ZOOM
    assert of_kind(driver.events, EventKind.CLICK, EventKind.BUTTON_DOWN) == []


def test_spreading_hands_zooms_in_and_closing_zooms_out():
    driver = zooming()
    spread(driver, 0.10, 0.20)  # ระยะ ×2 → log(2)/log(1.06) = 11.9 → 11 notch
    assert zoom_total(driver.events) == 11
    assert all(e.dy > 0 for e in of_kind(driver.events, EventKind.ZOOM))
    driver.clear()
    spread(driver, 0.20, 0.10)
    assert zoom_total(driver.events) == -11
    assert of_kind(driver.events, EventKind.CLICK, EventKind.BUTTON_DOWN, EventKind.SCROLL) == []


def test_zoom_step_from_config():
    driver = Driver(make_engine({"gestures": {"zoom_step": 0.12}}))
    driver.engine.set_armed(True)
    for _ in range(6):
        driver.step(*two_hands({}, {}))
    for _ in range(3):
        driver.step(*two_hands({"pinch": PINCHED}, {"pinch": PINCHED}))
    driver.clear()
    spread(driver, 0.10, 0.20)
    assert zoom_total(driver.events) == 6  # log(2)/log(1.12) = 6.1


def test_small_distance_jitter_does_not_zoom():
    driver = zooming()
    for i in range(60):
        wobble = 0.004 * (1 if i % 2 else -1)  # ±2% ของระยะ
        driver.step(
            make_hand(0.6 + wobble, 0.4, pinch=PINCHED),
            make_hand(0.4 - wobble, 0.4, pinch=PINCHED, side="left"),
        )
    assert of_kind(driver.events, EventKind.ZOOM) == []


def test_cursor_moves_to_midpoint_of_hands():
    driver = zooming()
    for _ in range(40):
        driver.step(
            make_hand(0.75, 0.5, pinch=PINCHED), make_hand(0.45, 0.3, pinch=PINCHED, side="left")
        )
    # กึ่งกลาง (0.6, 0.4) → map ผ่าน active region
    expected = ((0.6 - 0.15) / 0.7 * 1919, (0.4 - 0.10) / 0.6 * 1079)
    assert driver.engine.snapshot.cursor == pytest.approx(expected, abs=3)
    zoom_events = of_kind(driver.events, EventKind.ZOOM)
    for event in zoom_events:
        assert 0 <= event.x <= 1919 and 0 <= event.y <= 1079


def test_releasing_either_hand_ends_zoom_without_click():
    for release in ("right", "left"):
        driver = zooming()
        spread(driver, 0.10, 0.14, frames=6)
        right = {"x": 0.64, "pinch": OPEN if release == "right" else PINCHED}
        left = {"x": 0.36, "pinch": OPEN if release == "left" else PINCHED}
        for _ in range(4):
            driver.step(*two_hands(right, left))
        assert driver.engine.state is State.POINTER, release
        # มือหลักที่ยังจีบค้างอยู่ต้องไม่กลายเป็นคลิก/ลาก
        for _ in range(20):
            driver.step(*two_hands(right, left))
        assert of_kind(driver.events, EventKind.CLICK, EventKind.BUTTON_DOWN) == [], release


def test_click_works_again_after_zoom_once_fingers_reopen():
    driver = zooming()
    for _ in range(4):
        driver.step(*two_hands({}, {}))
    driver.clear()
    click(driver, x=0.6)
    assert len(of_kind(driver.events, EventKind.CLICK)) == 1


def test_no_cursor_jump_when_zoom_ends():
    driver = zooming()
    spread(driver, 0.10, 0.20)
    at_midpoint = driver.engine.snapshot.cursor
    driver.clear()
    for _ in range(10):
        driver.step(make_hand(0.7, 0.4), make_hand(0.3, 0.4, side="left"))
    moves = of_kind(driver.events, EventKind.MOVE)
    assert all(math.hypot(m.x - at_midpoint[0], m.y - at_midpoint[1]) < 25 for m in moves)


def test_resting_pinched_other_hand_does_not_turn_clicks_into_zoom():
    """มืออีกข้างจีบค้างมานานแล้ว (วางพัก): การจีบของมือหลักต้องยังเป็นคลิกปกติ"""
    driver = Driver(make_engine())
    driver.engine.set_armed(True)
    for _ in range(6):
        driver.step(*two_hands({}, {}))
    for _ in range(round((ZOOM_SYNC_MS + 300) / 33)):
        driver.step(*two_hands({}, {"pinch": PINCHED}))
    for _ in range(3):
        driver.step(*two_hands({"pinch": PINCHED}, {"pinch": PINCHED}))
    for _ in range(3):
        driver.step(*two_hands({}, {"pinch": PINCHED}))
    assert len(of_kind(driver.events, EventKind.CLICK)) == 1
    assert State.TWO_HAND_ZOOM not in driver.states


def test_other_hand_pinching_slightly_later_still_zooms():
    driver = Driver(make_engine())
    driver.engine.set_armed(True)
    for _ in range(6):
        driver.step(*two_hands({}, {}))
    for _ in range(4):  # มือหลักจีบก่อน ~130ms
        driver.step(*two_hands({"pinch": PINCHED}, {}))
    assert driver.engine.state is State.PINCH_PENDING
    for _ in range(15):
        driver.step(*two_hands({"pinch": PINCHED}, {"pinch": PINCHED}))
    assert driver.engine.state is State.TWO_HAND_ZOOM
    assert of_kind(driver.events, EventKind.CLICK, EventKind.BUTTON_DOWN) == []


def test_other_hand_pinching_during_a_drag_does_not_interrupt_it():
    driver = Driver(make_engine())
    driver.engine.set_armed(True)
    for _ in range(6):
        driver.step(*two_hands({}, {}))
    for _ in range(12):
        driver.step(*two_hands({"pinch": PINCHED}, {}))
    assert driver.engine.state is State.DRAGGING
    for _ in range(10):
        driver.step(*two_hands({"pinch": PINCHED}, {"pinch": PINCHED}))
    assert driver.engine.state is State.DRAGGING
    assert of_kind(driver.events, EventKind.BUTTON_UP, EventKind.ZOOM) == []


def test_other_hand_lost_ends_zoom_after_grace():
    driver = zooming()
    for _ in range(5):  # หายแวบเดียว (~165ms): ยังซูมอยู่
        driver.step(make_hand(0.6, 0.4, pinch=PINCHED))
    assert driver.engine.state is State.TWO_HAND_ZOOM
    for _ in range(8):
        driver.step(make_hand(0.6, 0.4, pinch=PINCHED))
    assert driver.engine.state is State.POINTER
    assert of_kind(driver.events, EventKind.CLICK, EventKind.BUTTON_DOWN) == []


def test_dominant_hand_lost_during_zoom_goes_idle():
    driver = zooming()
    for _ in range(12):
        driver.step(make_hand(0.4, 0.4, pinch=PINCHED, side="left"))
    assert driver.engine.state is State.IDLE
    assert of_kind(driver.events, EventKind.CLICK, EventKind.BUTTON_DOWN) == []


def test_fist_on_other_hand_is_not_a_pinch():
    driver = Driver(make_engine())
    driver.engine.set_armed(True)
    for _ in range(6):
        driver.step(*two_hands({}, {}))
    for _ in range(3):
        driver.step(*two_hands({"pinch": PINCHED}, {"fist": True}))
    for _ in range(3):
        driver.step(*two_hands({}, {"fist": True}))
    assert State.TWO_HAND_ZOOM not in driver.states
    assert len(of_kind(driver.events, EventKind.CLICK)) == 1


# ---- ทั่วไป ---------------------------------------------------------------- #


def test_disarm_clears_special_modes():
    for prepare in (
        lambda d: d.hold(make_hand(**PALM), 400),
        lambda d: d.hold(make_hand(**FIST), 300),
    ):
        driver = pointing()
        prepare(driver)
        assert driver.engine.state in (State.GESTURE, State.CHARGING)
        driver.engine.set_armed(False)
        snapshot = driver.engine.snapshot
        assert snapshot.state is State.DISARMED
        assert snapshot.charge == 0.0 and not snapshot.swipe_ready
        driver.clear()
        driver.hold(make_hand(**FIST), 1500)
        assert driver.events == []


def test_hand_lost_in_special_modes_returns_to_idle():
    for pose in (PALM, FIST):
        driver = pointing()
        driver.hold(make_hand(**pose), 300)
        driver.hold(None, 400)
        assert driver.engine.state is State.IDLE
        assert actions(driver.events) == []


def test_only_known_event_kinds_are_emitted():
    driver = pointing()
    driver.hold(make_hand(**PALM), 400)
    swipe(driver, (0.5, 0.4), (0.8, 0.4))
    driver.hold(make_hand(0.8, 0.4), 500)
    driver.hold(make_hand(0.8, 0.4, **FIST), 900)
    assert {e.kind for e in driver.events} <= set(INPUT_KINDS)
    assert set(actions(driver.events)) == {"swipe_right", "fist_hold"}


# ---- กันท่าติดเอง (เกณฑ์ M3: < 1 ครั้งต่อการใช้งานปกติ 5 นาที) ------------------ #


def test_five_minutes_of_normal_use_with_label_flicker_triggers_nothing_special():
    """จำลองใช้งานปกติ 5 นาที: ชี้ คลิก ลาก สกรอล โดยโมเดลตอบท่าพิเศษผิดแวบๆ (1–2 เฟรม) เป็นระยะ
    และมีมืออีกข้างวางพักในท่านิ้วชิดกัน — ต้องไม่มีท่าพิเศษติดเองเลย
    """
    import random

    rng = random.Random(42)
    driver = Driver(make_engine())
    driver.engine.set_armed(True)
    resting = make_hand(0.25, 0.62, pinch=PINCHED, side="left")  # มือซ้ายวางพัก นิ้วชิดกัน
    x, y = 0.5, 0.4
    special = ("Open_Palm", "Closed_Fist", "ILoveYou")

    def frame(pinch=OPEN, gesture="None"):
        driver.step(make_hand(x, y, pinch=pinch, gesture=gesture), resting)

    total_frames = 5 * 60 * 30
    while len(driver.states) < total_frames:
        activity = rng.random()
        if activity < 0.45:  # ขยับเคอร์เซอร์ (บางครั้งเร็วมาก)
            tx, ty = rng.uniform(0.18, 0.82), rng.uniform(0.13, 0.67)
            steps = rng.randint(3, 25)
            x0, y0 = x, y
            for i in range(steps):
                x, y = x0 + (tx - x0) * (i + 1) / steps, y0 + (ty - y0) * (i + 1) / steps
                flicker = rng.random() < 0.04  # โมเดลตอบท่าพิเศษผิดแวบเดียว
                frame(gesture=rng.choice(special) if flicker else "None")
                if flicker and rng.random() < 0.5:
                    frame(gesture="None")
        elif activity < 0.65:  # คลิก
            for _ in range(rng.randint(3, 5)):
                frame(pinch=PINCHED)
            for _ in range(rng.randint(3, 8)):
                frame()
        elif activity < 0.75:  # ลาก
            for _ in range(10):
                frame(pinch=PINCHED)
            for _ in range(rng.randint(5, 20)):
                x = min(0.82, max(0.18, x + rng.uniform(-0.02, 0.02)))
                frame(pinch=PINCHED)
            for _ in range(4):
                frame()
        elif activity < 0.85:  # สกรอล
            for _ in range(rng.randint(8, 30)):
                y = min(0.67, max(0.13, y + rng.uniform(-0.02, 0.02)))
                frame(gesture="Victory")
            for _ in range(4):
                frame()
        else:  # พักมือนิ่งๆ
            for _ in range(rng.randint(10, 60)):
                frame(gesture=rng.choice(special) if rng.random() < 0.02 else "None")

    unwanted = of_kind(
        driver.events, EventKind.ACTION, EventKind.ZOOM, EventKind.ARMED, EventKind.DISARMED
    )
    assert unwanted == []
    assert driver.engine.armed
    assert State.TWO_HAND_ZOOM not in driver.states
    # การใช้งานปกติยังทำงานครบ
    assert len(of_kind(driver.events, EventKind.CLICK)) > 20
    assert len(of_kind(driver.events, EventKind.BUTTON_DOWN)) > 5
    assert len(of_kind(driver.events, EventKind.SCROLL)) > 5
