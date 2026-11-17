"""อัด → อ่าน → เล่นซ้ำ และตรวจทุกไฟล์ใน tests/fixtures/ เทียบกับ ``expect`` ใน header"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import load_tool
from handsim import OPEN, PINCHED, make_frame, make_hand
from make_synthetic_fixtures import FIXTURES_DIR, write_all

from aether.config import parse_config
from aether.gestures import EventKind, GestureEvent
from aether.recording import (
    FrameWriter,
    read_recording,
    replay,
    summarize_events,
)

FIXTURES = sorted(FIXTURES_DIR.glob("*.jsonl"))


def _matches(actual: int, expected) -> bool:
    if isinstance(expected, list):
        low, high = expected
        return low <= actual <= high
    return actual == expected


# ---- fixture ทุกไฟล์ (สังเคราะห์ + ที่อัดจากกล้องจริง) ---------------------- #


def test_fixture_directory_has_recordings():
    assert len(FIXTURES) >= 10


@pytest.mark.parametrize("path", FIXTURES, ids=lambda p: p.stem)
def test_fixture_replays_to_expected_events(path: Path):
    recording = read_recording(path)
    assert recording.frames, "ไฟล์อัดไม่มีเฟรม"
    if not recording.expect:
        pytest.skip("ไฟล์นี้ไม่มี expect ใน header")
    config = parse_config(
        {"tracking": {"dominant_hand": recording.header.get("dominant_hand", "right")}}
    )
    armed = recording.header.get("start_armed", True)
    summary = summarize_events(replay(recording.frames, config, armed=armed))
    mismatches = {
        key: (summary[key], expected)
        for key, expected in recording.expect.items()
        if not _matches(summary[key], expected)
    }
    assert not mismatches, f"{recording.header.get('note', '')}: (ได้, คาด) = {mismatches}"


def test_synthetic_fixtures_are_up_to_date(tmp_path):
    """ไฟล์ synthetic_*.jsonl ที่ commit ไว้ต้องตรงกับที่ make_synthetic_fixtures.py สร้าง"""
    for generated in write_all(tmp_path):
        committed = FIXTURES_DIR / generated.name
        assert committed.is_file(), f"รัน python tests/make_synthetic_fixtures.py ({generated.name})"
        assert committed.read_text(encoding="utf-8") == generated.read_text(encoding="utf-8"), (
            f"{generated.name} ไม่ตรงกับตัวสร้าง — รัน python tests/make_synthetic_fixtures.py"
        )


def test_fixtures_contain_only_numbers_no_images():
    """Privacy: ไฟล์อัดมีแค่ตัวเลข landmark — ไม่มี field ที่เก็บภาพ"""
    allowed_frame = {"timestamp_ms", "width", "height", "hands"}
    allowed_hand = {
        "handedness", "handedness_score", "gesture", "gesture_score",
        "landmarks", "world_landmarks",
    }  # fmt: skip
    for path in FIXTURES:
        for line in path.read_text(encoding="utf-8").splitlines()[1:]:
            data = json.loads(line)
            assert set(data) == allowed_frame
            for hand in data["hands"]:
                assert set(hand) == allowed_hand


# ---- writer / reader ---------------------------------------------------- #


def test_roundtrip_preserves_frames_and_rebases_time(tmp_path):
    frames = [
        make_frame(50_000, make_hand(0.3, 0.3, gesture="Victory", score=0.87654)),
        make_frame(50_033),
        make_frame(50_066, make_hand(side="left"), make_hand(pinch=PINCHED)),
    ]
    path = tmp_path / "sub" / "take.jsonl"
    with FrameWriter(path, note="ทดสอบ", expect={"left_clicks": 1}) as writer:
        for frame in frames:
            writer.write(frame)
    assert writer.frames == 3

    recording = read_recording(path)
    assert recording.header["note"] == "ทดสอบ"
    assert recording.expect == {"left_clicks": 1}
    assert [f.timestamp_ms for f in recording.frames] == [0, 33, 66]  # นับจากเฟรมแรก
    assert [len(f.hands) for f in recording.frames] == [1, 0, 2]
    original, restored = frames[0].hands[0], recording.frames[0].hands[0]
    assert restored.gesture == "Victory" and restored.gesture_score == pytest.approx(0.877)
    for a, b in zip(original.landmarks, restored.landmarks):
        assert b == pytest.approx(a, abs=1e-5)
    assert recording.frames[2].hands[0].handedness == "left"


def test_writer_requires_context_manager(tmp_path):
    writer = FrameWriter(tmp_path / "x.jsonl")
    with pytest.raises(RuntimeError):
        writer.write(make_frame(0))


def test_reader_reports_bad_line(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"aether_landmarks": 1}\n{"timestamp_ms": 0}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="บรรทัด 2"):
        read_recording(path)


def test_reader_rejects_unknown_version(tmp_path):
    path = tmp_path / "future.jsonl"
    path.write_text('{"aether_landmarks": 99}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="เวอร์ชัน"):
        read_recording(path)


def test_reader_accepts_file_without_header(tmp_path):
    path = tmp_path / "plain.jsonl"
    path.write_text(json.dumps(make_frame(5, make_hand()).to_dict()) + "\n\n", encoding="utf-8")
    recording = read_recording(path)
    assert recording.header == {} and recording.expect == {} and len(recording.frames) == 1


# ---- replay / summary ---------------------------------------------------- #


def test_replay_disarmed_produces_no_input():
    frames = [
        make_frame(i * 33, make_hand(pinch=PINCHED if 10 < i < 15 else OPEN)) for i in range(40)
    ]
    assert list(replay(frames, armed=False)) == []
    armed = summarize_events(replay(frames))
    assert armed["left_clicks"] == 1


def test_replay_ends_by_releasing_a_held_drag():
    frames = [make_frame(i * 33, make_hand(pinch=OPEN if i < 8 else PINCHED)) for i in range(40)]
    kinds = [e.kind for e in replay(frames) if e.kind is not EventKind.MOVE]
    assert kinds == [EventKind.BUTTON_DOWN, EventKind.BUTTON_UP]  # ไม่รายงาน disarm ของ replay เอง


def test_summarize_events():
    events = [
        GestureEvent(EventKind.CLICK, count=1),
        GestureEvent(EventKind.CLICK, count=2),
        GestureEvent(EventKind.CLICK, button="right"),
        GestureEvent(EventKind.BUTTON_DOWN),
        GestureEvent(EventKind.BUTTON_UP),
        GestureEvent(EventKind.SCROLL, dy=3),
        GestureEvent(EventKind.SCROLL, dy=-2),
        GestureEvent(EventKind.SCROLL, dx=4),
        GestureEvent(EventKind.SCROLL, dx=-1),
        GestureEvent(EventKind.MOVE, x=1, y=1),
        GestureEvent(EventKind.ACTION, name="swipe_left"),
        GestureEvent(EventKind.ACTION, name="swipe_up"),
        GestureEvent(EventKind.ACTION, name="swipe_up"),
        GestureEvent(EventKind.ACTION, name="fist_hold"),
        GestureEvent(EventKind.ACTION, name="something_else"),
        GestureEvent(EventKind.ZOOM, dy=4),
        GestureEvent(EventKind.ZOOM, dy=-1),
        GestureEvent(EventKind.ARMED),
        GestureEvent(EventKind.DISARMED),
    ]
    assert summarize_events(events) == {
        "left_clicks": 2,
        "double_clicks": 1,
        "right_clicks": 1,
        "drags": 1,
        "scroll_up": 3,
        "scroll_down": 2,
        "scroll_left": 1,
        "scroll_right": 4,
        "swipe_left": 1,
        "swipe_right": 0,
        "swipe_up": 2,
        "swipe_down": 0,
        "fist_holds": 1,
        "zoom_in": 4,
        "zoom_out": 1,
        "armed": 1,
        "disarmed": 1,
    }


# ---- tools/record_landmarks.py (ส่วนที่ไม่ต้องใช้กล้อง) --------------------- #

record_landmarks = load_tool("record_landmarks")


def test_parse_expect():
    assert record_landmarks.parse_expect(["left_clicks=3", " scroll_up = 3..20 "]) == {
        "left_clicks": 3,
        "scroll_up": [3, 20],
    }
    for bad in (["clicks=3"], ["left_clicks"], ["left_clicks=many"], ["drags=1..x"]):
        with pytest.raises(ValueError):
            record_landmarks.parse_expect(bad)


def test_list_expect_prints_known_names(capsys):
    assert record_landmarks.main(["--list-expect"]) == 0
    assert capsys.readouterr().out.split() == list(summarize_events(()))
