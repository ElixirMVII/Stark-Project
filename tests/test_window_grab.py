"""✊ หยิบหน้าต่าง: ✊ แล้วขยับ = ลาก, แบมือ = วาง, ปล่อยตอนเร็ว = โยน (✊ นิ่งๆ ยังย่อหน้าต่าง)"""

from __future__ import annotations

import logging

import pytest
from handsim import OPEN, Driver, make_hand
from test_actions import RecordingBackend
from test_gestures import make_engine, of_kind, pointing

from aether.actions import (
    THROW_GLIDE_S,
    THROW_KEEP_VISIBLE,
    ActionExecutor,
    DryRunBackend,
    glide_offset,
    keep_on_screen,
)
from aether.gestures import THROW_SPEED, EventKind, GestureEvent, State
from aether.recording import summarize_events

FIST = {"fist": True, "gesture": "Closed_Fist"}


def grabbing() -> Driver:
    """Driver ที่ถือหน้าต่างอยู่แล้ว (กำมือแล้วขยับไปทางขวาจนหยิบติด)"""
    driver = pointing()
    driver.hold(make_hand(**FIST), 150)
    driver.animate(lambda p: make_hand(0.5 + 0.06 * p, 0.4, **FIST), 200)
    assert driver.engine.state is State.WINDOW_DRAG
    return driver


# ---- engine ---------------------------------------------------------------- #


def test_moving_fist_grabs_window_under_cursor():
    driver = pointing()
    driver.hold(make_hand(**FIST), 150)
    grab_point = driver.engine.snapshot.cursor
    driver.animate(lambda p: make_hand(0.5 + 0.06 * p, 0.4, **FIST), 200)
    grabs = of_kind(driver.events, EventKind.WINDOW_GRAB)
    assert len(grabs) == 1 and (grabs[0].x, grabs[0].y) == grab_point
    assert [e.name for e in of_kind(driver.events, EventKind.ACTION)] == []  # ไม่ย่อหน้าต่าง
    assert driver.engine.snapshot.state is State.WINDOW_DRAG


def test_still_fist_still_minimizes():
    driver = pointing()
    driver.hold(make_hand(**FIST), 1000)
    assert [e.name for e in of_kind(driver.events, EventKind.ACTION)] == ["fist_hold"]
    assert of_kind(driver.events, EventKind.WINDOW_GRAB) == []


def test_window_and_cursor_move_together_with_the_fist():
    driver = grabbing()
    driver.clear()
    start = driver.engine.snapshot.cursor
    driver.animate(lambda p: make_hand(0.56 + 0.14 * p, 0.4 + 0.1 * p, **FIST), 500)
    driver.hold(make_hand(0.70, 0.5, **FIST), 1500)  # ให้ฟิลเตอร์ลู่เข้า
    moves = of_kind(driver.events, EventKind.WINDOW_MOVE)
    assert moves
    total = (sum(e.dx for e in moves), sum(e.dy for e in moves))
    end = driver.engine.snapshot.cursor
    assert total == (end[0] - start[0], end[1] - start[1])  # หน้าต่างขยับเท่าเคอร์เซอร์พอดี
    assert total[0] > 300 and total[1] > 100
    assert of_kind(driver.events, EventKind.CLICK, EventKind.BUTTON_DOWN) == []


def test_opening_hand_slowly_drops_the_window():
    driver = grabbing()
    driver.hold(make_hand(0.56, 0.4, **FIST), 300)
    driver.clear()
    driver.hold(make_hand(0.56, 0.4, pinch=OPEN), 200)
    releases = of_kind(driver.events, EventKind.WINDOW_RELEASE)
    assert len(releases) == 1 and (releases[0].dx, releases[0].dy) == (0, 0)
    assert driver.engine.state is State.POINTER
    driver.hold(make_hand(0.56, 0.4, **FIST), 1000)  # กำอีกครั้งทันที: ต้องรอพ้น cooldown ก่อน
    assert [e.name for e in of_kind(driver.events, EventKind.ACTION)] == []


def test_releasing_while_moving_fast_throws_the_window():
    driver = grabbing()
    driver.clear()
    # กวาดไปทางขวาเร็วๆ แล้วแบมือระหว่างที่ยังเคลื่อนอยู่
    driver.animate(lambda p: make_hand(0.56 + 0.2 * p, 0.4, **FIST), 165)
    driver.animate(lambda p: make_hand(0.76 + 0.1 * p, 0.4, pinch=OPEN), 132)
    releases = of_kind(driver.events, EventKind.WINDOW_RELEASE)
    assert len(releases) == 1
    vx, vy = releases[0].dx, releases[0].dy
    assert vx >= THROW_SPEED and abs(vy) < vx / 4
    assert summarize_events(driver.events)["window_throws"] == 1


def test_hand_lost_or_disarm_drops_the_window():
    lost = grabbing()
    lost.clear()
    lost.hold(None, 400)
    assert [e.reason for e in of_kind(lost.events, EventKind.WINDOW_RELEASE)] == ["hand_lost"]
    assert lost.engine.state is State.IDLE

    disarmed = grabbing()
    events = disarmed.engine.set_armed(False)
    assert [e.kind for e in events] == [EventKind.WINDOW_RELEASE, EventKind.DISARMED]
    assert events[0].reason == "disarm"


def test_window_grab_can_be_turned_off():
    driver = pointing({"gestures": {"window_grab": False}})
    driver.hold(make_hand(**FIST), 150)
    driver.animate(lambda p: make_hand(0.5 + 0.06 * p, 0.4, **FIST), 200)
    assert of_kind(driver.events, EventKind.WINDOW_GRAB) == []


def test_nothing_happens_while_disarmed():
    driver = Driver(make_engine())
    driver.hold(make_hand(**FIST), 150)
    driver.animate(lambda p: make_hand(0.5 + 0.1 * p, 0.4, **FIST), 300)
    assert driver.events == []


# ---- executor -------------------------------------------------------------- #


def _drag(executor: ActionExecutor, throw=(0, 0)) -> None:
    executor.execute(
        [
            GestureEvent(EventKind.WINDOW_GRAB, x=500, y=300),
            GestureEvent(EventKind.WINDOW_MOVE, x=510, y=305, dx=10, dy=5),
            GestureEvent(EventKind.WINDOW_MOVE, x=530, y=305, dx=20, dy=0),
            GestureEvent(EventKind.WINDOW_RELEASE, x=530, y=305, dx=throw[0], dy=throw[1]),
        ]
    )


def test_executor_moves_and_drops_grabbed_window():
    backend = RecordingBackend()
    executor = ActionExecutor(backend)
    _drag(executor)
    assert backend.calls == [
        ("grab", 500, 300),
        ("move_window", 4242, 10, 5),
        ("move_window", 4242, 20, 0),
    ]
    assert executor.grabbed_window is None


def test_executor_throws_on_fast_release():
    backend = RecordingBackend()
    executor = ActionExecutor(backend)
    _drag(executor, throw=(2400, -300))
    assert backend.calls[-1] == ("throw", 4242, 2400, -300)


def test_nothing_to_grab_means_nothing_moves():
    backend = RecordingBackend()
    backend.grab_result = None  # เช่น ชี้ที่ desktop หรือหน้าต่างของ AETHER เอง
    executor = ActionExecutor(backend)
    _drag(executor, throw=(3000, 0))
    assert backend.calls == [("grab", 500, 300)]


def test_release_all_drops_the_window():
    backend = RecordingBackend()
    executor = ActionExecutor(backend)
    executor.execute([GestureEvent(EventKind.WINDOW_GRAB, x=1, y=1)])
    assert executor.grabbed_window == 4242
    executor.execute([GestureEvent(EventKind.DISARMED)])
    assert executor.grabbed_window is None
    executor.execute([GestureEvent(EventKind.WINDOW_MOVE, dx=5, dy=5)])
    assert ("move_window", 4242, 5, 5) not in backend.calls


def test_dry_run_logs_grab_and_throw(caplog):
    executor = ActionExecutor(DryRunBackend())
    with caplog.at_level(logging.INFO, logger="aether.actions"):
        _drag(executor, throw=(2000, 0))
    assert "[dry-run] grab window at (500, 300)" in caplog.text
    assert "[dry-run] throw window at +2000 +0 px/s" in caplog.text


# ---- การโยน: ระยะไหลต่อ + ไม่หลุดจอ ------------------------------------------ #


def test_glide_offset_scales_with_velocity():
    assert glide_offset(2000, -1000) == (round(2000 * THROW_GLIDE_S), round(-1000 * THROW_GLIDE_S))
    assert glide_offset(0, 0) == (0, 0)


SCREEN = (0, 0, 1920, 1080)
WINDOW = (500, 300, 1300, 900)  # 800 x 600


def test_keep_on_screen_leaves_small_moves_alone():
    assert keep_on_screen(WINDOW, 100, -50, SCREEN) == (100, -50)


@pytest.mark.parametrize(
    ("dx", "dy", "expected_left", "expected_top"),
    [
        (5000, 0, 1920 - THROW_KEEP_VISIBLE, 300),  # โยนขวาสุด: เหลือขอบซ้ายของหน้าต่างบนจอ
        (-5000, 0, -800 + THROW_KEEP_VISIBLE, 300),  # โยนซ้ายสุด: เหลือขอบขวาบนจอ
        (0, -5000, 500, 0),  # ขอบบนห้ามเลยจอขึ้นไป (แถบชื่อต้องจับได้)
        (0, 5000, 500, 1080 - THROW_KEEP_VISIBLE),
    ],
)
def test_keep_on_screen_clamps_big_throws(dx, dy, expected_left, expected_top):
    mx, my = keep_on_screen(WINDOW, dx, dy, SCREEN)
    assert (WINDOW[0] + mx, WINDOW[1] + my) == (expected_left, expected_top)


def test_keep_on_screen_with_monitor_left_of_primary():
    bounds = (-1280, 0, 1920, 1080)
    mx, _ = keep_on_screen(WINDOW, -5000, 0, bounds)
    assert WINDOW[0] + mx == -1280 - 800 + THROW_KEEP_VISIBLE
