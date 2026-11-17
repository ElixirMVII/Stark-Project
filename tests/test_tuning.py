"""Live tuning: ค่าที่ปรับสด, การแก้ config.toml แบบคงคอมเมนต์ (pure logic)"""

from __future__ import annotations

import json
import tomllib

import pytest
from handsim import Driver, make_hand

from aether.config import Config, ConfigError, load_config, parse_config
from aether.gestures import GestureEngine, State
from aether.mapping import ScreenGeometry
from aether.tuning import (
    TUNABLES,
    apply_overrides,
    current_values,
    save_overrides,
    set_toml_value,
)

SAMPLE = """\
# config ของฉัน
[pinch]
on = 0.25      # จีบ
off = 0.38
drag_hold_ms = 250

[pointer]
active_region = [0.15, 0.10, 0.85, 0.70]
one_euro = { min_cutoff = 1.0, beta = 0.02, d_cutoff = 1.0 }   # filter

[hud]
sounds = true
"""


def test_tunable_slider_conversion():
    for t in TUNABLES:
        assert t.from_slider(0) == pytest.approx(t.minimum)
        assert t.from_slider(t.steps) == pytest.approx(t.maximum)
        mid = t.from_slider(t.to_slider((t.minimum + t.maximum) / 2))
        assert t.minimum <= mid <= t.maximum
    cooldown = next(t for t in TUNABLES if t.key == "gestures.cooldown_ms")
    assert isinstance(cooldown.from_slider(5), int) and cooldown.text(800) == "800"
    pinch = next(t for t in TUNABLES if t.key == "pinch.on")
    assert pinch.text(0.25) == "0.25"


def test_current_values_and_apply_overrides():
    config = Config()
    values = current_values(config)
    assert values["pinch.on"] == 0.25 and values["pointer.one_euro.beta"] == 0.02
    tuned = apply_overrides(
        config,
        {
            "pinch.on": 0.3,
            "pinch.off": 0.45,
            "pointer.one_euro.beta": 0.05,
            "gestures.cooldown_ms": 1200,
        },
    )
    assert (tuned.pinch.on, tuned.pinch.off) == (0.3, 0.45)
    assert tuned.pointer.one_euro.beta == 0.05 and tuned.gestures.cooldown_ms == 1200
    assert isinstance(tuned.gestures.cooldown_ms, int)
    assert tuned.pointer.active_region == config.pointer.active_region


def test_pinch_values_change_together():
    # on 0.45 กับ off เดิม 0.38 ขัดกัน แต่ส่งมาพร้อม off 0.5 จึงใช้ได้
    tuned = apply_overrides(Config(), {"pinch.on": 0.45, "pinch.off": 0.5})
    assert tuned.pinch.on == 0.45
    with pytest.raises(ConfigError):
        apply_overrides(Config(), {"pinch.on": 0.45})


def test_unknown_override_is_rejected():
    with pytest.raises(KeyError):
        apply_overrides(Config(), {"camera.index": 1})


def test_set_toml_value_keeps_comments_and_layout():
    text = set_toml_value(SAMPLE, "pinch.on", 0.3)
    text = set_toml_value(text, "pointer.one_euro.beta", 0.05)
    assert "on = 0.3      # จีบ" in text
    assert "one_euro = { min_cutoff = 1.0, beta = 0.05, d_cutoff = 1.0 }   # filter" in text
    assert text.startswith("# config ของฉัน\n")
    data = tomllib.loads(text)
    assert data["pinch"]["on"] == 0.3 and data["pointer"]["one_euro"]["beta"] == 0.05
    assert data["pinch"]["off"] == 0.38


def test_set_toml_value_adds_missing_keys_and_tables():
    text = set_toml_value(SAMPLE, "pinch.drag_hold_ms", 300, integer=True)
    text = set_toml_value(text, "gestures.cooldown_ms", 1000, integer=True)
    text = set_toml_value(text, "gestures.swipe_dist_x", 0.3)
    data = tomllib.loads(text)
    assert data["pinch"]["drag_hold_ms"] == 300
    assert data["gestures"] == {"cooldown_ms": 1000, "swipe_dist_x": 0.3}
    no_euro = set_toml_value("[pointer]\npinch_lock_gain = 0.2\n", "pointer.one_euro.beta", 0.04)
    assert tomllib.loads(no_euro)["pointer"]["one_euro"] == {"beta": 0.04}
    empty = set_toml_value("", "pinch.on", 0.2)
    assert tomllib.loads(empty) == {"pinch": {"on": 0.2}}


def test_set_toml_value_inside_inline_table_missing_key():
    text = set_toml_value(
        "[pointer]\none_euro = { min_cutoff = 1.0 }\n", "pointer.one_euro.beta", 0.03
    )
    assert tomllib.loads(text)["pointer"]["one_euro"] == {"min_cutoff": 1.0, "beta": 0.03}


def test_save_overrides_round_trip(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text(SAMPLE, encoding="utf-8")
    overrides = {"pinch.on": 0.28, "pinch.off": 0.4, "pointer.one_euro.min_cutoff": 0.8}
    assert save_overrides(path, overrides) == [path]
    config = load_config(path, use_calibration=False)
    assert (config.pinch.on, config.pinch.off) == (0.28, 0.4)
    assert config.pointer.one_euro.min_cutoff == 0.8
    assert "# จีบ" in path.read_text(encoding="utf-8")


def test_save_without_config_starts_from_template(tmp_path):
    template = tmp_path / "bundle.toml"
    template.write_text(SAMPLE, encoding="utf-8")
    path = tmp_path / "user" / "config.toml"
    save_overrides(path, {"gestures.cooldown_ms": 900}, template=template)
    assert "# config ของฉัน" in path.read_text(encoding="utf-8")
    assert load_config(path, use_calibration=False).gestures.cooldown_ms == 900


def test_save_also_updates_calibration_pinch(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text(SAMPLE, encoding="utf-8")
    calibration = tmp_path / "calibration.json"
    calibration.write_text(
        json.dumps(
            {
                "version": 1,
                "pinch": {"on": 0.2, "off": 0.33},
                "pointer": {"active_region": [0.1, 0.1, 0.9, 0.7]},
            }
        ),
        encoding="utf-8",
    )
    written = save_overrides(path, {"pinch.on": 0.27}, calibration=calibration)
    assert written == [path, calibration]
    config = load_config(path, calibration=calibration)
    assert config.pinch.on == 0.27 and config.pinch.off == 0.33  # off จาก calibration เดิม


def test_save_refuses_inconsistent_values(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text(SAMPLE, encoding="utf-8")
    with pytest.raises(ConfigError):
        save_overrides(path, {"pinch.on": 0.5})  # off ยังเป็น 0.38
    assert path.read_text(encoding="utf-8") == SAMPLE


def test_engine_reconfigure_takes_effect_without_losing_the_hand():
    screen = ScreenGeometry(0, 0, 1920, 1080)
    engine = GestureEngine(Config(), screen)
    driver = Driver(engine)
    engine.set_armed(True)
    driver.hold(make_hand(pinch=0.30), 300)  # 0.30 > pinch.on 0.25: ไม่ใช่การจีบ
    assert engine.state is State.POINTER
    cursor = engine.snapshot.cursor
    engine.reconfigure(apply_overrides(engine.config, {"pinch.on": 0.35, "pinch.off": 0.5}))
    driver.hold(make_hand(pinch=0.6), 100)  # ปล่อยให้เห็นนิ้วห่างเกิน off ใหม่ก่อน
    driver.hold(make_hand(pinch=0.30), 100)  # ตอนนี้ 0.30 < on ใหม่ = จีบ
    assert engine.state is State.PINCH_PENDING
    assert engine.snapshot.cursor == pytest.approx(cursor, abs=3)  # ไม่กระโดด
    assert engine.config.pinch.on == 0.35
    parse_config({})  # ค่าเริ่มต้นไม่ถูกแก้ไปด้วย
    assert Config().pinch.on == 0.25
