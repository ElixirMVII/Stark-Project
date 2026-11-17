"""M5: flight recorder, ไฟล์ incident, ป้าย, สถิติ session และรายงาน (ไม่มีกล้อง ไม่มี Qt)"""

from __future__ import annotations

import datetime
import json

import pytest
from handsim import OPEN, PINCHED, Driver, make_frame, make_hand

from aether.config import Config, parse_config
from aether.gestures import EventKind, GestureEngine, GestureEvent, State
from aether.incidents import (
    LABELS,
    FlightRecorder,
    RecordedFrame,
    SessionStats,
    build_report,
    event_from_dict,
    event_to_dict,
    percentile_from_histogram,
    read_incident,
    replay_incident,
    set_label,
    write_incident,
)
from aether.mapping import ScreenGeometry
from aether.recording import read_recording, replay, summarize_events

FHD = ScreenGeometry(0, 0, 1920, 1080)
WHEN = datetime.datetime(
    2026, 11, 18, 9, 30, 15, tzinfo=datetime.timezone(datetime.timedelta(hours=7))
)


def recorded_click(armed: bool = True) -> list[RecordedFrame]:
    """เฟรมจริงจาก engine: ชี้ → จีบ → ปล่อย (คลิกหนึ่งครั้ง)"""
    engine = GestureEngine(Config(), FHD)
    if armed:
        engine.set_armed(True)
    frames: list[RecordedFrame] = []
    for i in range(40):
        frame = make_frame(10_000 + i * 33, make_hand(pinch=PINCHED if 15 <= i < 18 else OPEN))
        state, events = engine.update(frame)
        frames.append(RecordedFrame(frame, state, tuple(events)))
    return frames


def write(tmp_path, frames=None, **kwargs):
    return write_incident(
        tmp_path,
        frames if frames is not None else recorded_click(),
        config=kwargs.pop("config", Config()),
        screen=FHD,
        double_click_ms=500,
        when=WHEN,
        **kwargs,
    )


# ---- recorder ----------------------------------------------------------- #


def test_recorder_keeps_only_the_last_five_seconds():
    recorder = FlightRecorder(5.0)
    for i in range(400):  # 400 เฟรม × 33 ms ≈ 13 วินาที
        recorder.record(make_frame(i * 33), State.IDLE, ())
    frames = recorder.snapshot()
    span = frames[-1].hands.timestamp_ms - frames[0].hands.timestamp_ms
    assert 4900 <= span <= 5000
    assert frames[-1].hands.timestamp_ms == 399 * 33


def test_recorder_snapshot_is_a_copy():
    recorder = FlightRecorder()
    recorder.record(make_frame(0), State.IDLE, ())
    copy = recorder.snapshot()
    recorder.record(make_frame(33), State.IDLE, ())
    assert len(copy) == 1 and len(recorder) == 2


def test_event_dict_round_trip():
    events = [
        GestureEvent(EventKind.CLICK, x=5, y=6, count=2),
        GestureEvent(EventKind.ACTION, name="swipe_left"),
        GestureEvent(EventKind.BUTTON_UP, reason="hand_lost"),
        GestureEvent(EventKind.MOVE, x=1, y=2),
    ]
    for event in events:
        assert event_from_dict(event_to_dict(event)) == event
    assert event_to_dict(GestureEvent(EventKind.MOVE, x=1, y=2)) == {"kind": "move", "x": 1, "y": 2}


# ---- incident files ----------------------------------------------------- #


def test_incident_file_contents(tmp_path):
    path = write(tmp_path, source="mouse")
    assert path.name == "20261118-093015.jsonl"
    incident = read_incident(path)
    header = incident.header
    assert header["incident"] == 1 and header["label"] is None and header["source"] == "mouse"
    assert header["start_armed"] is True and header["screen"] == [0, 0, 1920, 1080]
    assert len(incident.frames) == 40 and incident.frames[0].timestamp_ms == 0
    assert incident.states[0] in ("POINTER", "IDLE")
    clicks = [e for e in incident.recorded_events() if e.kind is EventKind.CLICK]
    assert len(clicks) == 1
    assert incident.config() == Config()
    # อ่านเป็นไฟล์อัดธรรมดาได้ด้วย (tools/record_landmarks, fixture test)
    assert len(read_recording(path).frames) == 40


def test_incident_has_numbers_only_and_no_images(tmp_path):
    path = write(tmp_path)
    for line in path.read_text(encoding="utf-8").splitlines()[1:]:
        data = json.loads(line)
        assert set(data) <= {"timestamp_ms", "width", "height", "hands", "state", "events"}


def test_same_second_does_not_overwrite(tmp_path):
    first, second = write(tmp_path), write(tmp_path)
    assert first != second and second.name == "20261118-093015-2.jsonl"


def test_replay_is_deterministic_and_matches_what_happened(tmp_path):
    incident = read_incident(write(tmp_path))
    runs = [
        [(t, s, [event_to_dict(e) for e in ev]) for t, s, ev in replay_incident(incident)]
        for _ in range(3)
    ]
    assert runs[0] == runs[1] == runs[2]
    replayed = [e for _, _, events in replay_incident(incident) for e in events]
    assert [e for e in replayed if e.kind is EventKind.CLICK] == [
        e for e in incident.recorded_events() if e.kind is EventKind.CLICK
    ]


def test_replay_uses_the_recorded_config(tmp_path):
    left = parse_config({"tracking": {"dominant_hand": "left"}})
    frames = [
        RecordedFrame(make_frame(i * 33, make_hand(side="left", pinch=p)), State.POINTER)
        for i, p in enumerate([OPEN] * 10 + [PINCHED] * 3 + [OPEN] * 5)
    ]
    incident = read_incident(write(tmp_path, frames, config=left))
    events = [e for _, _, ev in replay_incident(incident) for e in ev]
    assert [e.kind for e in events if e.kind is EventKind.CLICK] == [EventKind.CLICK]


def test_disarmed_incident_replays_disarmed(tmp_path):
    incident = read_incident(write(tmp_path, recorded_click(armed=False)))
    assert incident.header["start_armed"] is False
    assert all(not events for _, _, events in replay_incident(incident))


def test_set_label(tmp_path):
    path = write(tmp_path)
    set_label(path, "false_trigger", "คลิกเองตอนยกมือ")
    incident = read_incident(path)
    assert incident.label == "false_trigger" and incident.header["note"] == "คลิกเองตอนยกมือ"
    assert len(incident.frames) == 40
    with pytest.raises(ValueError):
        set_label(path, "bad_label")
    assert set(LABELS) == {"false_trigger", "missed", "jitter", "wrong_click", "other"}


def test_promoted_incident_works_as_a_fixture(tmp_path):
    """fixture test ใช้ config จาก header — incident ที่ย้ายไปเป็น fixture จึงเล่นได้ผลเดิม"""
    incident = read_incident(write(tmp_path))
    summary = summarize_events(
        replay(incident.frames, incident.config(), incident.screen(), armed=True)
    )
    assert summary["left_clicks"] == 1


# ---- session stats ------------------------------------------------------ #


def test_percentile_from_histogram():
    histogram = {10: 50, 20: 45, 80: 5}
    assert percentile_from_histogram(histogram, 50) == 10
    assert percentile_from_histogram(histogram, 95) == 20
    assert percentile_from_histogram(histogram, 99) == 80
    assert percentile_from_histogram({}, 50) is None


def test_session_stats_summary(tmp_path):
    stats = SessionStats(source="mouse", dry_run=True)
    t = 100.0
    for i in range(90):  # 3 วินาทีที่ 30 fps
        state = State.POINTER if i < 60 else State.SCROLL
        events = [GestureEvent(EventKind.MOVE, x=i, y=i)]
        if i == 30:
            events.append(GestureEvent(EventKind.CLICK))
        if i == 70:
            events.append(GestureEvent(EventKind.ACTION, name="swipe_left"))
        stats.observe(t, 20.0 if i < 85 else 70.0, state, events)
        t += 1 / 30
    stats.note_incident()
    summary = stats.summary(hud_paints=0)
    assert summary["frames"] == 90 and summary["camera_fps"] == pytest.approx(30.0, abs=0.5)
    assert summary["latency_ms"]["p50"] == 20 and summary["latency_ms"]["p95"] == 70
    assert summary["events"]["click"] == 1 and summary["events"]["move"] == 90
    assert summary["actions"] == {"swipe_left": 1}
    assert summary["state_seconds"]["POINTER"] == pytest.approx(2.0, abs=0.1)
    assert summary["state_seconds"]["SCROLL"] == pytest.approx(1.0, abs=0.1)
    assert summary["incidents"] == 1 and summary["source"] == "mouse"
    path = stats.write(tmp_path)
    assert path.name.startswith("session-") and json.loads(path.read_text())["frames"] == 90


def test_camera_stall_does_not_inflate_state_time():
    stats = SessionStats()
    stats.observe(0.0, 10.0, State.DRAGGING, ())
    stats.observe(60.0, 10.0, State.IDLE, ())  # กล้องค้าง 1 นาที
    assert stats.summary()["state_seconds"]["DRAGGING"] <= 2.0


# ---- report ------------------------------------------------------------- #


def _session(p95_ms: int, minutes: float = 10.0, fps: float = 30.0) -> dict:
    return {
        "duration_s": minutes * 60,
        "frames": int(minutes * 60 * fps),
        "camera_fps": fps,
        "hud_fps": 50.0,
        "latency_ms": {"histogram": [[20, 90], [p95_ms, 10]]},
        "events": {"click": 40},
        "actions": {"swipe_left": 6, "fist_hold": 2},
    }


def test_report_ranks_problems_and_gives_advice(tmp_path):
    def incident(label: str | None, events: list[GestureEvent]):
        frames = recorded_click()
        frames[-1] = RecordedFrame(frames[-1].hands, State.POINTER, tuple(events))
        path = write(tmp_path, frames)
        if label:
            set_label(path, label)
        return read_incident(path)

    swipe = [GestureEvent(EventKind.ACTION, name="swipe_up")]
    incidents = [
        incident("false_trigger", swipe),
        incident("false_trigger", swipe),
        incident("jitter", []),
        incident(None, []),
    ]
    report = build_report([_session(30), _session(80, fps=20.0)], incidents)
    assert report.sessions == 2 and report.incidents == 4 and report.unlabeled == 1
    assert report.latency_p95 is not None and report.latency_p95 >= 30
    assert report.problems[0] == ("false_trigger", "swipe", 2)
    text = "\n".join(report.lines())
    assert "swipe_dist" in text and "min_cutoff" in text
    assert "p95" in text and "label_incidents" in text


def test_report_flags_high_latency_and_low_fps():
    report = build_report([_session(200, fps=15.0)], [])
    assert report.latency_p95 == 200
    text = "\n".join(report.lines())
    assert "เกินเป้า 60 ms" in text and "ต่ำกว่า 25" in text
    assert any("latency" in a for a in report.advice)


def test_empty_report():
    lines = build_report([], []).lines()
    assert lines[0].startswith("session 0")


def test_drive_engine_still_works_with_driver():
    # sanity: helper ของ test อื่นยังใช้กับ engine ได้ (ไม่มีผลข้างเคียงจาก recorder)
    driver = Driver(GestureEngine(Config(), FHD))
    driver.engine.set_armed(True)
    driver.hold(make_hand(), 100)
    assert driver.engine.state is State.POINTER
