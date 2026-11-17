"""Calibration: คำนวณ threshold, state machine ของ wizard, ไฟล์ calibration.json และการทับค่า config"""

from __future__ import annotations

import json
import math
import random

import pytest
from handsim import FRAME_MS, OPEN, PINCHED, make_frame, make_hand

from aether import calibration as cal
from aether.calibration import (
    CalibrationError,
    CalibrationResult,
    CalibrationSession,
    Step,
    compute_active_region,
    compute_pinch_thresholds,
    percentile,
    save_calibration,
)
from aether.config import (
    Config,
    ConfigError,
    apply_calibration,
    load_config,
    read_calibration,
)

# ---- percentile ----------------------------------------------------------- #


def test_percentile():
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert percentile(values, 0) == 1.0
    assert percentile(values, 50) == 3.0
    assert percentile(values, 100) == 5.0
    assert percentile(values, 25) == pytest.approx(2.0)
    assert percentile([5.0, 1.0, 3.0], 50) == 3.0  # ไม่ต้องเรียงมาก่อน
    assert percentile([7.0], 90) == 7.0
    assert percentile(values, 150) == 5.0 and percentile(values, -5) == 1.0
    with pytest.raises(ValueError):
        percentile([], 50)


# ---- threshold การจีบ ------------------------------------------------------ #


def _samples(center: float, spread: float, n: int = 60, seed: int = 0) -> list[float]:
    rng = random.Random(seed)
    return [center + rng.uniform(-spread, spread) for _ in range(n)]


def test_typical_user_gets_thresholds_close_to_defaults():
    on, off = compute_pinch_thresholds(_samples(0.90, 0.08), _samples(0.12, 0.03, seed=1))
    assert 0.20 <= on <= 0.30  # ค่าเริ่มต้น 0.25
    assert 0.30 <= off <= 0.45  # ค่าเริ่มต้น 0.38
    assert off == pytest.approx(1.5 * on, abs=0.002)


def test_thresholds_always_sit_between_pinched_and_rest():
    cases = [(0.9, 0.12), (0.6, 0.30), (1.4, 0.05), (0.5, 0.20), (0.45, 0.25)]
    for rest_center, pinched_center in cases:
        rest = _samples(rest_center, 0.04)
        pinched = _samples(pinched_center, 0.02, seed=1)
        on, off = compute_pinch_thresholds(rest, pinched)
        assert max(pinched) < on < off < min(rest), (rest_center, pinched_center, on, off)
        Config.__dataclass_fields__  # noqa: B018 — ค่าต้องผ่านกติกาของ PinchConfig ด้วย
        apply_calibration(Config(), {"version": 1, "pinch": {"on": on, "off": off}})


def test_user_whose_fingertips_do_not_touch_gets_higher_thresholds():
    """ปลายนิ้วในภาพไม่ชิดกัน (ratio ตอนจีบ ~0.3): ค่าเริ่มต้น 0.25 จะคลิกไม่ติด — calibration ต้องแก้ได้"""
    on, off = compute_pinch_thresholds(_samples(0.95, 0.05), _samples(0.30, 0.02, seed=1))
    assert on > 0.32 and off > on


@pytest.mark.parametrize(
    ("rest", "pinched"),
    [
        (_samples(0.30, 0.02), _samples(0.22, 0.02, seed=1)),  # ห่างกันน้อยเกินไป
        (_samples(0.15, 0.02), _samples(0.80, 0.02, seed=1)),  # ทำสลับกัน
        (_samples(0.5, 0.3), _samples(0.4, 0.3, seed=1)),  # มั่วทั้งคู่
    ],
)
def test_overlapping_measurements_are_rejected(rest, pinched):
    with pytest.raises(CalibrationError):
        compute_pinch_thresholds(rest, pinched)


def test_too_few_samples_are_rejected():
    with pytest.raises(CalibrationError, match="น้อยเกินไป"):
        compute_pinch_thresholds([0.9] * 5, [0.1] * 60)
    with pytest.raises(CalibrationError, match="น้อยเกินไป"):
        compute_active_region([(0.5, 0.5)] * 3)


def test_outliers_do_not_break_thresholds():
    rest = [*_samples(0.9, 0.05), 0.05, 0.02]  # landmark เพี้ยนสองเฟรม
    pinched = [*_samples(0.12, 0.02, seed=1), 1.5]
    on, off = compute_pinch_thresholds(rest, pinched)
    assert 0.2 <= on <= 0.3 and off > on


# ---- active region --------------------------------------------------------- #


def _sweep(x0, y0, x1, y1, n: int = 400, seed: int = 0):
    rng = random.Random(seed)
    return [(rng.uniform(x0, x1), rng.uniform(y0, y1)) for _ in range(n)]


def test_active_region_is_inside_the_swept_area():
    region = compute_active_region(_sweep(0.2, 0.15, 0.8, 0.65))
    x0, y0, x1, y1 = region
    assert 0.2 < x0 < 0.3 and 0.7 < x1 < 0.8
    assert 0.15 < y0 < 0.25 and 0.55 < y1 < 0.65
    apply_calibration(Config(), {"version": 1, "pointer": {"active_region": list(region)}})


def test_active_region_ignores_stray_points_and_clamps_to_frame():
    points = [*_sweep(0.25, 0.2, 0.75, 0.6), (0.0, 0.0), (1.0, 1.0), (-0.2, 1.3)]
    x0, y0, x1, y1 = compute_active_region(points)
    assert x0 > 0.2 and x1 < 0.8 and y0 > 0.15 and y1 < 0.65

    x0, y0, x1, y1 = compute_active_region(_sweep(-0.1, -0.1, 1.1, 1.1))
    assert 0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0


@pytest.mark.parametrize(
    "area", [(0.4, 0.2, 0.55, 0.6), (0.2, 0.4, 0.8, 0.5), (0.5, 0.5, 0.5, 0.5)]
)
def test_small_region_is_rejected(area):
    with pytest.raises(CalibrationError, match="แคบเกินไป"):
        compute_active_region(_sweep(*area))


# ---- session (state machine ของ wizard) ------------------------------------ #


class Feeder:
    def __init__(self, session: CalibrationSession) -> None:
        self.session = session
        self.t = 0

    def feed(self, hand, ms: int) -> None:
        for _ in range(max(1, round(ms / FRAME_MS))):
            self.session.update(make_frame(self.t, *(() if hand is None else (hand,))))
            self.t += FRAME_MS

    def sweep(self, ms: int, x0=0.2, y0=0.15, x1=0.8, y1=0.65) -> None:
        frames = round(ms / FRAME_MS)
        for i in range(frames):
            angle = 2 * math.pi * 3 * i / frames  # วนสามรอบ ให้ทั่วพื้นที่
            x = (x0 + x1) / 2 + (x1 - x0) / 2 * math.cos(angle) * (0.4 + 0.6 * (i % 7) / 6)
            y = (y0 + y1) / 2 + (y1 - y0) / 2 * math.sin(angle) * (0.4 + 0.6 * (i % 5) / 4)
            self.session.update(make_frame(self.t, make_hand(x, y)))
            self.t += FRAME_MS


STEP_MS = cal.SETTLE_MS + 2200


def test_full_session_produces_result():
    feeder = Feeder(CalibrationSession("right"))
    session = feeder.session
    assert session.step is Step.REST and not session.sampling and session.progress == 0.0
    assert "1/3" in session.title and session.instruction

    feeder.feed(make_hand(pinch=0.9), 500)
    assert not session.sampling and 0 < session.countdown_ms < cal.SETTLE_MS  # ยังอยู่ช่วงจัดท่า
    feeder.feed(make_hand(pinch=0.9), cal.SETTLE_MS)
    assert session.sampling and 0.0 < session.progress < 1.0
    assert session.live_ratio == pytest.approx(0.9)

    feeder.feed(make_hand(pinch=0.9), 2000)
    assert session.step is Step.PINCH and session.progress == 0.0
    feeder.feed(make_hand(pinch=PINCHED), STEP_MS)
    assert session.step is Step.REACH and session.error is None
    feeder.sweep(cal.SETTLE_MS + cal.SAMPLE_MS["REACH"] + 300)

    assert session.done and session.step is Step.DONE and session.progress == 1.0
    result = session.result
    assert result is not None
    assert PINCHED < result.pinch_on < result.pinch_off < 0.9
    assert result.rest_ratio == pytest.approx(0.9, abs=0.01)
    assert result.pinch_ratio == pytest.approx(PINCHED, abs=0.01)
    x0, y0, x1, y1 = result.active_region
    assert 0.15 < x0 < 0.4 and 0.6 < x1 < 0.85 and y0 < 0.4 < y1
    session.update(make_frame(feeder.t, make_hand()))  # เฟรมหลังจบ: ไม่มีผล
    assert session.result is result


def test_time_only_advances_while_hand_is_visible():
    feeder = Feeder(CalibrationSession("right"))
    feeder.feed(None, 10_000)  # ไม่มีมือ 10 วินาที
    assert feeder.session.step is Step.REST and feeder.session.countdown_ms == cal.SETTLE_MS
    assert not feeder.session.hand_visible and feeder.session.live_ratio is None

    feeder.feed(make_hand(pinch=0.9), cal.SETTLE_MS + 1000)
    progress = feeder.session.progress
    assert 0.3 < progress < 0.7
    feeder.feed(None, 5000)  # เอามือออกกลางทาง: หยุดรอ ไม่เดินต่อ
    assert feeder.session.progress == progress
    feeder.feed(make_hand(pinch=0.9), 1500)
    assert feeder.session.step is Step.PINCH


def test_other_hand_is_ignored():
    feeder = Feeder(CalibrationSession("right"))
    feeder.feed(make_hand(side="left", pinch=0.9), 10_000)
    assert feeder.session.step is Step.REST and not feeder.session.hand_visible

    left = Feeder(CalibrationSession("left"))
    left.feed(make_hand(side="left", pinch=0.9), STEP_MS)
    assert left.session.step is Step.PINCH


def test_pinch_step_repeats_with_a_hint_when_user_did_not_pinch():
    feeder = Feeder(CalibrationSession("right"))
    feeder.feed(make_hand(pinch=0.9), STEP_MS)
    feeder.feed(make_hand(pinch=0.85), STEP_MS)  # ไม่ได้จีบจริง
    session = feeder.session
    assert session.step is Step.PINCH and session.error and "จีบ" in session.error
    assert session.progress == 0.0 and not session.sampling  # กลับไปช่วงจัดท่าของขั้นเดิม

    feeder.feed(make_hand(pinch=PINCHED), STEP_MS)  # ทำใหม่ถูกต้อง
    assert session.step is Step.REACH and session.error is None


def test_reach_step_repeats_when_area_is_too_small():
    feeder = Feeder(CalibrationSession("right"))
    feeder.feed(make_hand(pinch=0.9), STEP_MS)
    feeder.feed(make_hand(pinch=PINCHED), STEP_MS)
    feeder.feed(make_hand(0.5, 0.4), cal.SETTLE_MS + cal.SAMPLE_MS["REACH"] + 300)  # มือนิ่ง
    assert feeder.session.step is Step.REACH and "แคบเกินไป" in feeder.session.error
    feeder.sweep(cal.SETTLE_MS + cal.SAMPLE_MS["REACH"] + 300)
    assert feeder.session.done


def test_restart_clears_everything():
    feeder = Feeder(CalibrationSession("right"))
    feeder.feed(make_hand(pinch=0.9), STEP_MS)
    feeder.session.restart()
    assert feeder.session.step is Step.REST and feeder.session.progress == 0.0
    assert feeder.session.result is None and feeder.session.error is None


def test_long_gap_between_frames_is_capped():
    """กล้องค้าง 5 วินาทีแล้วกลับมา: ต้องไม่นับเป็นเวลาที่เก็บข้อมูลไปแล้ว"""
    session = CalibrationSession("right")
    session.update(make_frame(0, make_hand(pinch=0.9)))
    session.update(make_frame(5000, make_hand(pinch=0.9)))
    assert session.countdown_ms >= cal.SETTLE_MS - 200


# ---- calibration.json + config --------------------------------------------- #

RESULT = CalibrationResult(
    pinch_on=0.31,
    pinch_off=0.47,
    active_region=(0.2, 0.15, 0.8, 0.65),
    rest_ratio=0.9,
    pinch_ratio=0.16,
)


def test_save_and_read_roundtrip(tmp_path):
    path = tmp_path / "sub" / "calibration.json"
    save_calibration(path, RESULT, "left")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["version"] == 1 and data["dominant_hand"] == "left"
    assert data["pinch"] == {"on": 0.31, "off": 0.47}
    assert data["pointer"] == {"active_region": [0.2, 0.15, 0.8, 0.65]}
    assert data["measured"] == {"rest_ratio": 0.9, "pinch_ratio": 0.16}
    assert "T" in data["created"]  # ISO timestamp
    assert read_calibration(path) == data
    assert not list(path.parent.glob("*.tmp"))  # ไม่เหลือไฟล์ชั่วคราว


def test_apply_calibration_overrides_only_measured_values():
    config = apply_calibration(Config(), RESULT.to_dict("right"))
    assert (config.pinch.on, config.pinch.off) == (0.31, 0.47)
    assert config.pointer.active_region == (0.2, 0.15, 0.8, 0.65)
    defaults = Config()
    assert config.pinch.drag_hold_ms == defaults.pinch.drag_hold_ms
    assert config.pointer.one_euro == defaults.pointer.one_euro
    assert config.camera == defaults.camera and config.gestures == defaults.gestures

    partial = apply_calibration(Config(), {"version": 1, "pinch": {"on": 0.3, "off": 0.5}})
    assert partial.pointer.active_region == defaults.pointer.active_region


@pytest.mark.parametrize(
    "data",
    [
        {"version": 1, "pinch": {"on": 0.5, "off": 0.3}},  # on > off
        {"version": 1, "pinch": {"on": "x", "off": 0.3}},
        {"version": 1, "pinch": {"on": 0.2}},  # ขาด off
        {"version": 1, "pointer": {"active_region": [0.1, 0.1, 0.9]}},
        {"version": 1, "pointer": {"active_region": [0.9, 0.1, 0.1, 0.7]}},
        {"version": 1, "pinch": "tight"},
    ],
)
def test_invalid_calibration_values_are_rejected(data):
    with pytest.raises(ConfigError):
        apply_calibration(Config(), data)


def test_read_calibration_missing_corrupt_or_wrong_version(tmp_path):
    assert read_calibration(tmp_path / "none.json") is None
    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("{not json", encoding="utf-8")
    with pytest.raises(ConfigError, match="อ่านไม่ได้"):
        read_calibration(corrupt)
    old = tmp_path / "old.json"
    old.write_text('{"version": 99}', encoding="utf-8")
    with pytest.raises(ConfigError, match="เวอร์ชัน"):
        read_calibration(old)
    array = tmp_path / "array.json"
    array.write_text("[1, 2]", encoding="utf-8")
    with pytest.raises(ConfigError):
        read_calibration(array)


def test_load_config_applies_calibration_file(tmp_path):
    config_file = tmp_path / "config.toml"
    config_file.write_text("[pinch]\non = 0.2\noff = 0.3\ndrag_hold_ms = 400\n", encoding="utf-8")
    calibration_file = tmp_path / "calibration.json"
    save_calibration(calibration_file, RESULT, "right")

    calibrated = load_config(config_file, calibration=calibration_file)
    assert (calibrated.pinch.on, calibrated.pinch.off) == (0.31, 0.47)  # calibration ทับ config.toml
    assert calibrated.pinch.drag_hold_ms == 400  # ค่าอื่นจาก config.toml ยังอยู่

    plain = load_config(config_file, calibration=calibration_file, use_calibration=False)
    assert (plain.pinch.on, plain.pinch.off) == (0.2, 0.3)

    missing = load_config(config_file, calibration=tmp_path / "nope.json")
    assert missing.pinch.on == 0.2


def test_load_config_reports_bad_calibration_file(tmp_path):
    config_file = tmp_path / "config.toml"
    config_file.write_text("", encoding="utf-8")
    bad = tmp_path / "calibration.json"
    bad.write_text('{"version": 1, "pinch": {"on": 0.9, "off": 0.1}}', encoding="utf-8")
    with pytest.raises(ConfigError) as excinfo:
        load_config(config_file, calibration=bad)
    assert "calibration.json" in str(excinfo.value)


def test_calibrated_engine_clicks_for_user_with_loose_pinch():
    """ผู้ใช้ที่จีบแล้ว ratio ได้ ~0.3: ค่าเริ่มต้นคลิกไม่ติด แต่หลัง calibrate ต้องคลิกติด"""
    from test_gestures import click, of_kind, pointing

    from aether.gestures import EventKind

    loose = 0.30
    default_driver = pointing()
    default_driver.hold(make_hand(pinch=loose), 100)
    default_driver.hold(make_hand(pinch=OPEN), 66)
    assert of_kind(default_driver.events, EventKind.CLICK) == []

    on, off = compute_pinch_thresholds(_samples(0.95, 0.05), _samples(loose, 0.02, seed=1))
    calibrated = pointing({"pinch": {"on": on, "off": off}})
    calibrated.hold(make_hand(pinch=loose), 100)
    calibrated.hold(make_hand(pinch=OPEN), 66)
    assert len(of_kind(calibrated.events, EventKind.CLICK)) == 1
    click(calibrated)  # จีบแน่นก็ยังติดเหมือนเดิม
    assert len(of_kind(calibrated.events, EventKind.CLICK)) == 2
