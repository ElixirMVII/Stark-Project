"""โยนหน้าต่างไปจอข้างๆ (backlog v1.2) — เรขาคณิตล้วน ไม่แตะ Windows"""

from __future__ import annotations

import pytest

from aether.actions import THROW_MONITOR_SPEED, throw_destination

LEFT = (0, 0, 1920, 1080)
RIGHT = (1920, 0, 4480, 1440)  # จอขวาใหญ่กว่าและสูงกว่า
ABOVE = (0, -1080, 1920, 0)
WINDOW = (200, 100, 1000, 700)  # บนจอซ้าย
FAST = THROW_MONITOR_SPEED + 500


def test_fast_throw_right_moves_to_the_right_monitor_keeping_relative_position():
    dx, dy = throw_destination(WINDOW, FAST, 0, [LEFT, RIGHT])
    left, top = WINDOW[0] + dx, WINDOW[1] + dy
    width, height = WINDOW[2] - WINDOW[0], WINDOW[3] - WINDOW[1]
    assert RIGHT[0] <= left and left + width <= RIGHT[2]
    assert RIGHT[1] <= top and top + height <= RIGHT[3]
    # ตำแหน่งสัมพัทธ์: เดิมอยู่ 200/(1920-800) ของช่วงที่ขยับได้ → บนจอใหม่สัดส่วนเดียวกัน
    assert (left - RIGHT[0]) / (2560 - width) == pytest.approx(200 / (1920 - width), abs=0.01)


def test_throw_back_left():
    on_right = (2500, 300, 3300, 900)
    dx, _ = throw_destination(on_right, -FAST, 100, [LEFT, RIGHT])
    assert on_right[0] + dx < 1920


def test_vertical_throw_to_a_monitor_above():
    dx, dy = throw_destination(WINDOW, 0, -FAST, [LEFT, ABOVE])
    assert WINDOW[1] + dy < 0 and dx == 0


@pytest.mark.parametrize(
    ("vx", "vy", "monitors"),
    [
        (THROW_MONITOR_SPEED - 100, 0, [LEFT, RIGHT]),  # ช้าเกิน: ไหลต่อแบบเดิม
        (FAST, 0, [LEFT]),  # จอเดียว
        (-FAST, 0, [LEFT, RIGHT]),  # ไม่มีจอทางซ้าย
        (FAST, FAST * 0.9, [LEFT, RIGHT]),  # ทิศเฉียง ไม่ชัด
        (0, FAST, [LEFT, ABOVE]),  # ลงล่าง: ไม่มีจอ
    ],
)
def test_no_jump(vx, vy, monitors):
    assert throw_destination(WINDOW, vx, vy, monitors) is None


def test_nearest_monitor_in_that_direction_wins():
    far = (4480, 0, 6400, 1080)
    dx, _ = throw_destination(WINDOW, FAST, 0, [LEFT, far, RIGHT])
    assert RIGHT[0] <= WINDOW[0] + dx < RIGHT[2]


def test_monitor_that_does_not_line_up_is_not_a_neighbour():
    diagonal = (1920, 1440, 3840, 2520)  # อยู่ขวาล่าง ไม่ซ้อนกันในแนวตั้ง
    assert throw_destination(WINDOW, FAST, 0, [LEFT, diagonal]) is None


def test_window_bigger_than_the_target_is_aligned_to_its_corner():
    big = (0, 0, 1900, 1060)
    small = (1920, 0, 3200, 720)
    dx, dy = throw_destination(big, FAST, 0, [LEFT, small])
    assert (big[0] + dx, big[1] + dy) == (1920, 0)


def test_executor_passes_the_throw_velocity_to_the_backend():
    from test_actions import RecordingBackend

    from aether.actions import ActionExecutor
    from aether.gestures import EventKind, GestureEvent

    backend = RecordingBackend()
    executor = ActionExecutor(backend)
    executor.execute([GestureEvent(EventKind.WINDOW_GRAB, x=500, y=400)])
    executor.execute([GestureEvent(EventKind.WINDOW_RELEASE, dx=int(FAST), dy=0)])
    assert ("throw", backend.grab_result, int(FAST), 0) in backend.calls
