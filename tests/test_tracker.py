from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from aether import tracker
from aether.tracker import (
    GESTURE_NONE,
    LANDMARK_COUNT,
    LM,
    Hand,
    HandFrame,
    HandTracker,
    Landmark,
    ModelNotFoundError,
    hand_connections,
    hand_frame_from_result,
    user_handedness,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def _category(name: str, score: float) -> SimpleNamespace:
    return SimpleNamespace(category_name=name, score=score)


def _landmarks(offset: float = 0.0) -> list[SimpleNamespace]:
    return [
        SimpleNamespace(x=offset + i * 0.01, y=0.5 + i * 0.01, z=-i * 0.001)
        for i in range(LANDMARK_COUNT)
    ]


def _result(hands: list[tuple[str, str, float]]) -> SimpleNamespace:
    """ผลลัพธ์ปลอมหน้าตาเหมือน GestureRecognizerResult: (label ของโมเดล, ท่า, คะแนนท่า) ต่อมือ"""
    return SimpleNamespace(
        handedness=[[_category(label, 0.97)] for label, _, _ in hands],
        gestures=[[_category(gesture, score)] for _, gesture, score in hands],
        hand_landmarks=[_landmarks(0.1 * i) for i in range(len(hands))],
        hand_world_landmarks=[_landmarks(0.0) for _ in hands],
    )


# ---- handedness --------------------------------------------------------- #


@pytest.mark.parametrize(
    ("label", "mirrored", "expected"),
    [
        # ภาพไม่ flip: label ของโมเดลตรงกับมือจริง
        ("Right", False, "right"),
        ("Left", False, "left"),
        # ภาพ flip แบบกระจก: ต้องสลับข้าง
        ("Right", True, "left"),
        ("Left", True, "right"),
    ],
)
def test_user_handedness(label, mirrored, expected):
    assert user_handedness(label, mirrored=mirrored) == expected


def test_user_handedness_rejects_unknown_label():
    with pytest.raises(ValueError):
        user_handedness("Both", mirrored=True)


# ---- result → HandFrame ------------------------------------------------- #


def test_hand_frame_from_result_converts_everything():
    result = _result([("Left", "Victory", 0.91), ("Right", "Open_Palm", 0.8)])
    frame = hand_frame_from_result(result, width=640, height=480, timestamp_ms=1234, mirrored=True)

    assert (frame.timestamp_ms, frame.width, frame.height) == (1234, 640, 480)
    assert len(frame.hands) == 2
    first, second = frame.hands
    assert first.handedness == "right"  # สลับเพราะ mirrored
    assert second.handedness == "left"
    assert first.gesture == "Victory" and first.gesture_score == pytest.approx(0.91)
    assert first.handedness_score == pytest.approx(0.97)
    assert len(first.landmarks) == LANDMARK_COUNT
    assert len(first.world_landmarks) == LANDMARK_COUNT
    assert first.landmarks[LM.INDEX_TIP] == Landmark(0.08, 0.58, -0.008)
    assert isinstance(first.landmarks[0], Landmark)


def test_hand_frame_from_result_no_hands():
    frame = hand_frame_from_result(
        _result([]), width=640, height=480, timestamp_ms=0, mirrored=True
    )
    assert frame.hands == ()


def test_hand_frame_from_result_missing_gesture_becomes_none():
    result = _result([("Right", "Victory", 0.9)])
    result.gestures = [[]]  # โมเดลไม่ให้ท่ามา
    frame = hand_frame_from_result(result, width=640, height=480, timestamp_ms=0, mirrored=False)
    assert frame.hands[0].gesture == GESTURE_NONE
    assert frame.hands[0].gesture_score == 0.0
    assert frame.hands[0].handedness == "right"


def test_hand_frame_from_result_empty_gesture_name_becomes_none():
    result = _result([("Right", "", 0.3)])
    frame = hand_frame_from_result(result, width=640, height=480, timestamp_ms=0, mirrored=False)
    assert frame.hands[0].gesture == GESTURE_NONE


def test_hand_frame_from_result_skips_incomplete_hand():
    result = _result([("Right", "Victory", 0.9)])
    result.hand_landmarks = [_landmarks()[:10]]
    frame = hand_frame_from_result(result, width=640, height=480, timestamp_ms=0, mirrored=False)
    assert frame.hands == ()


# ---- HandFrame ---------------------------------------------------------- #


def _hand(side: str, score: float = 0.9) -> Hand:
    points = tuple(Landmark(i * 0.01, i * 0.02, 0.0) for i in range(LANDMARK_COUNT))
    return Hand(
        handedness=side,
        handedness_score=score,
        gesture="Open_Palm",
        gesture_score=0.7,
        landmarks=points,
        world_landmarks=points,
    )


def test_hand_lookup_by_side():
    frame = HandFrame(0, 640, 480, (_hand("left"), _hand("right")))
    assert frame.hand("right") is frame.hands[1]
    assert frame.hand("left") is frame.hands[0]
    assert HandFrame(0, 640, 480, (_hand("left"),)).hand("right") is None


def test_hand_lookup_prefers_higher_confidence_when_same_side_twice():
    weak, strong = _hand("right", 0.6), _hand("right", 0.95)
    assert HandFrame(0, 640, 480, (weak, strong)).hand("right") is strong


def test_json_roundtrip():
    frame = HandFrame(42, 640, 480, (_hand("left"), _hand("right")))
    restored = HandFrame.from_dict(json.loads(json.dumps(frame.to_dict())))
    assert restored == frame
    assert isinstance(restored.hands[0].landmarks[0], Landmark)


def test_from_dict_without_world_landmarks():
    data = _hand("left").to_dict()
    del data["world_landmarks"]
    assert Hand.from_dict(data).world_landmarks == ()


# ---- connections -------------------------------------------------------- #


def _normalized(connections) -> set[tuple[int, int]]:
    return {(min(a, b), max(a, b)) for a, b in connections}


def test_fallback_connections_are_a_valid_hand_skeleton():
    fallback = tracker._FALLBACK_CONNECTIONS
    assert len(fallback) == 21
    assert len(_normalized(fallback)) == 21
    assert {i for pair in fallback for i in pair} == set(range(LANDMARK_COUNT))


def test_hand_connections_match_fallback():
    """ถ้ามี mediapipe รายการของ mediapipe ต้องเป็นโครงเดียวกับที่ hardcode ไว้"""
    assert _normalized(hand_connections()) == _normalized(tracker._FALLBACK_CONNECTIONS)


# ---- HandTracker -------------------------------------------------------- #


def test_tracker_missing_model_raises_before_loading_mediapipe(tmp_path):
    with pytest.raises(ModelNotFoundError):
        HandTracker(tmp_path / "missing.task")


def test_tracker_on_blank_frame_and_monotonic_timestamps():
    """ทดสอบกับโมเดลจริงถ้าโหลดมาแล้ว: เฟรมว่างต้องไม่เจอมือ และ timestamp ซ้ำต้องไม่พัง"""
    model = REPO_ROOT / "assets" / "models" / "gesture_recognizer.task"
    if not model.is_file():
        pytest.skip("ยังไม่ได้โหลดโมเดล (python tools/download_assets.py)")
    np = pytest.importorskip("numpy")
    pytest.importorskip("mediapipe")

    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    with HandTracker(model, num_hands=2, mirrored=True) as hand_tracker:
        first = hand_tracker.process(blank, 100)
        second = hand_tracker.process(blank, 100)  # ไม่เพิ่ม → ต้องถูกขยับเป็น 101
        third = hand_tracker.process(blank, 50)  # ถอยหลัง → 102
    assert first.hands == ()
    assert (first.width, first.height) == (640, 480)
    assert [first.timestamp_ms, second.timestamp_ms, third.timestamp_ms] == [100, 101, 102]
    with pytest.raises(RuntimeError):
        hand_tracker.process(blank, 200)  # ปิดไปแล้ว


# ---- ข้อห้ามของ spec ----------------------------------------------------- #


def test_legacy_solutions_api_is_not_used_anywhere():
    """ห้ามใช้ API เก่า ``solutions`` ของ mediapipe (ถูกเอาออกตั้งแต่ 0.10.31)"""
    pattern = re.compile(r"\b(mp|mediapipe)\.solutions\b")
    offenders = [
        str(path.relative_to(REPO_ROOT))
        for folder in ("src", "tools")
        for path in (REPO_ROOT / folder).rglob("*.py")
        if pattern.search(path.read_text(encoding="utf-8"))
    ]
    assert offenders == []
