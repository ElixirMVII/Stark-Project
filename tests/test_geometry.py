from __future__ import annotations

import dataclasses
import math

import pytest
from handsim import HEIGHT, OPEN, PINCHED, WIDTH, make_hand

from aether.geometry import (
    Finger,
    distance_3d,
    distance_px,
    finger_curled,
    finger_extended,
    hand_scale,
    palm_center,
    pinch_ratio,
    pointer_anchor,
    to_px,
)
from aether.tracker import LANDMARK_COUNT, LM, Hand, Landmark


def test_distance_px_accounts_for_non_square_frame():
    a, b = Landmark(0.0, 0.0, 0.0), Landmark(0.5, 0.5, 0.0)
    assert distance_px(a, b, 640, 480) == pytest.approx(math.hypot(320, 240))
    # ระยะ normalized เท่ากันแต่ px ไม่เท่ากัน
    horizontal = distance_px(Landmark(0, 0, 0), Landmark(0.1, 0, 0), 640, 480)
    vertical = distance_px(Landmark(0, 0, 0), Landmark(0, 0.1, 0), 640, 480)
    assert horizontal == pytest.approx(64) and vertical == pytest.approx(48)


def test_to_px_and_distance_3d():
    assert to_px(Landmark(0.25, 0.5, 0.0), 640, 480) == (160.0, 240.0)
    assert distance_3d(Landmark(0, 0, 0), Landmark(1, 2, 2)) == pytest.approx(3.0)


@pytest.mark.parametrize("scale", [60.0, 100.0, 160.0])
def test_hand_scale_is_wrist_to_middle_mcp(scale):
    assert hand_scale(make_hand(scale=scale), WIDTH, HEIGHT) == pytest.approx(scale)


@pytest.mark.parametrize("ratio", [0.1, 0.25, 0.38, 1.0])
def test_pinch_ratio_index(ratio):
    hand = make_hand(pinch=ratio)
    assert pinch_ratio(hand, WIDTH, HEIGHT) == pytest.approx(ratio)
    assert pinch_ratio(hand, WIDTH, HEIGHT, LM.INDEX_TIP) == pytest.approx(ratio)


def test_pinch_ratio_middle_is_independent_of_index():
    hand = make_hand(pinch=OPEN, middle_pinch=0.2)
    assert pinch_ratio(hand, WIDTH, HEIGHT, LM.MIDDLE_TIP) == pytest.approx(0.2)
    assert pinch_ratio(hand, WIDTH, HEIGHT, LM.INDEX_TIP) == pytest.approx(OPEN)


def test_pinch_ratio_is_scale_invariant():
    """มือใกล้/ไกลกล้อง (ขนาดในภาพต่างกัน) ต้องได้ ratio เท่ากัน"""
    near = pinch_ratio(make_hand(pinch=0.3, scale=180.0), WIDTH, HEIGHT)
    far = pinch_ratio(make_hand(pinch=0.3, scale=50.0), WIDTH, HEIGHT)
    assert near == pytest.approx(far) == pytest.approx(0.3)


def test_pinch_ratio_degenerate_hand_is_not_a_pinch():
    point = Landmark(0.5, 0.5, 0.0)
    collapsed = Hand("right", 0.9, "None", 0.0, (point,) * LANDMARK_COUNT)
    assert pinch_ratio(collapsed, WIDTH, HEIGHT) == math.inf


def test_pointer_anchor_is_midpoint_of_index_and_middle_mcp():
    hand = make_hand(0.3, 0.6)
    assert pointer_anchor(hand) == pytest.approx((0.3, 0.6))
    a, b = hand.landmarks[LM.INDEX_MCP], hand.landmarks[LM.MIDDLE_MCP]
    assert pointer_anchor(hand) == pytest.approx(((a.x + b.x) / 2, (a.y + b.y) / 2))


def test_pointer_anchor_does_not_move_when_pinching():
    """หัวใจของการกันเคอร์เซอร์กระโดดตอนคลิก: จีบแล้วจุดคุมเคอร์เซอร์ต้องอยู่ที่เดิม"""
    assert pointer_anchor(make_hand(pinch=OPEN)) == pytest.approx(
        pointer_anchor(make_hand(pinch=PINCHED))
    )


def test_palm_center_is_mean_of_five_points():
    hand = make_hand(0.5, 0.5)
    points = [hand.landmarks[i] for i in (0, 5, 9, 13, 17)]
    expected = (sum(p.x for p in points) / 5, sum(p.y for p in points) / 5)
    assert palm_center(hand) == pytest.approx(expected)


def test_finger_extended_on_open_hand_and_fist():
    open_hand = make_hand()
    for finger in (Finger.INDEX, Finger.MIDDLE, Finger.RING, Finger.PINKY):
        assert finger_extended(open_hand, finger, WIDTH, HEIGHT)
    fist = make_hand(fist=True)
    assert not finger_extended(fist, Finger.INDEX, WIDTH, HEIGHT)
    assert not finger_extended(fist, Finger.MIDDLE, WIDTH, HEIGHT)


def test_thumb_extended_when_away_from_palm():
    assert finger_extended(make_hand(), Finger.THUMB, WIDTH, HEIGHT)
    assert not finger_extended(make_hand(fist=True), Finger.THUMB, WIDTH, HEIGHT)


def test_finger_curled_separates_fist_from_pinch():
    fist = make_hand(fist=True)
    assert finger_curled(fist, Finger.INDEX, WIDTH, HEIGHT)
    assert finger_curled(fist, Finger.MIDDLE, WIDTH, HEIGHT)
    for pinch in (PINCHED, 0.25, OPEN):
        hand = make_hand(pinch=pinch, middle_pinch=pinch)
        assert not finger_curled(hand, Finger.INDEX, WIDTH, HEIGHT)
        assert not finger_curled(hand, Finger.MIDDLE, WIDTH, HEIGHT)


def test_finger_curled_prefers_world_landmarks():
    """ถ้ามี world landmark (3D) ต้องใช้แทนระยะในภาพ ซึ่งเพี้ยนตามมุมมือ"""
    hand = make_hand()  # ในภาพ: นิ้วชี้เหยียด
    world = [Landmark(0.0, 0.0, 0.0)] * LANDMARK_COUNT
    world[LM.WRIST] = Landmark(0.0, 0.0, 0.0)
    world[LM.INDEX_MCP] = Landmark(0.0, -0.09, 0.0)
    world[LM.INDEX_PIP] = Landmark(0.0, -0.10, 0.03)
    world[LM.INDEX_TIP] = Landmark(0.0, -0.04, 0.03)  # พับกลับมาใกล้ข้อมือ (เห็นได้เฉพาะใน 3D)
    curled = dataclasses.replace(hand, world_landmarks=tuple(world))
    assert finger_curled(curled, Finger.INDEX, WIDTH, HEIGHT)
    assert not finger_curled(hand, Finger.INDEX, WIDTH, HEIGHT)

    world[LM.INDEX_TIP] = Landmark(0.0, -0.11, 0.04)  # ท่าจีบ: ปลายนิ้วยังเลยโคนนิ้วออกไป
    pinching = dataclasses.replace(hand, world_landmarks=tuple(world))
    assert not finger_curled(pinching, Finger.INDEX, WIDTH, HEIGHT)


def test_finger_enum_indices():
    assert (Finger.INDEX.mcp, Finger.INDEX.pip, Finger.INDEX.tip) == (5, 6, 8)
    assert (Finger.THUMB.tip, Finger.PINKY.tip) == (4, 20)
