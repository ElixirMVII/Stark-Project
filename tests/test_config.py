from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from aether.config import Config, ConfigError, load_config, parse_config

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_shipped_config_matches_builtin_defaults():
    """config.toml ที่แจกไปต้องตรงกับค่าเริ่มต้นใน dataclass (แก้ที่หนึ่งต้องแก้อีกที่)"""
    assert load_config(REPO_ROOT / "config.toml") == Config()


def test_empty_config_uses_defaults():
    config = parse_config({})
    assert config == Config()
    assert config.camera.width == 640
    assert config.pointer.active_region == (0.15, 0.10, 0.85, 0.70)
    assert config.pinch.on == 0.25 and config.pinch.off == 0.38
    assert config.hotkeys.toggle_arm == "<ctrl>+<alt>+g"


def test_partial_override_keeps_other_defaults():
    config = parse_config(tomllib.loads('[tracking]\ndominant_hand = "left"\n'))
    assert config.tracking.dominant_hand == "left"
    assert config.tracking.num_hands == 2
    assert config.camera == Config().camera


def test_nested_table_and_int_to_float():
    config = parse_config({"pointer": {"one_euro": {"min_cutoff": 2, "beta": 0.1}}})
    assert config.pointer.one_euro.min_cutoff == 2.0
    assert isinstance(config.pointer.one_euro.min_cutoff, float)
    assert config.pointer.one_euro.beta == 0.1
    assert config.pointer.one_euro.d_cutoff == 1.0


def test_active_region_becomes_tuple():
    config = parse_config({"pointer": {"active_region": [0.2, 0.2, 0.8, 0.6]}})
    assert config.pointer.active_region == (0.2, 0.2, 0.8, 0.6)


@pytest.mark.parametrize(
    ("data", "fragment"),
    [
        ({"camra": {}}, "camra"),  # section พิมพ์ผิด
        ({"pinch": {"onn": 0.2}}, "pinch.onn"),  # key พิมพ์ผิด
        ({"camera": {"width": "640"}}, "camera.width"),  # ชนิดผิด
        ({"camera": {"index": True}}, "camera.index"),  # bool ไม่ใช่ int
        ({"camera": {"mirror": 1}}, "camera.mirror"),  # int ไม่ใช่ bool
        ({"camera": {"fps": 0}}, "camera.fps"),
        ({"camera": "usb"}, "camera"),  # ต้องเป็น table
        ({"tracking": {"dominant_hand": "both"}}, "dominant_hand"),
        ({"tracking": {"num_hands": 3}}, "num_hands"),
        ({"tracking": {"min_gesture_score": 1.5}}, "min_gesture_score"),
        ({"pointer": {"active_region": [0.1, 0.1, 0.9]}}, "active_region"),  # ไม่ครบ 4 ค่า
        ({"pointer": {"active_region": [0.9, 0.1, 0.1, 0.7]}}, "active_region"),  # x0 > x1
        ({"pointer": {"active_region": [0.1, 0.1, 1.2, 0.7]}}, "active_region"),  # เกิน 1
        ({"pointer": {"pinch_lock_gain": 2.0}}, "pinch_lock_gain"),
        ({"pinch": {"on": 0.4, "off": 0.3}}, "pinch.on"),  # hysteresis กลับด้าน
        ({"gestures": {"zoom_step": 0}}, "zoom_step"),
        ({"actions": {"swipe_up": "  "}}, "actions.swipe_up"),
        ({"hotkeys": {"quit": "<ctrl>+<alt>+g"}}, "hotkeys"),  # ซ้ำกับ toggle_arm
        ({"hud": {"primary": "cyan"}}, "hud.primary"),
        ({"hud": {"opacity": 1.1}}, "hud.opacity"),
        ({"hud": {"volume": -0.1}}, "hud.volume"),
    ],
)
def test_invalid_config_is_rejected(data, fragment):
    with pytest.raises(ConfigError) as excinfo:
        parse_config(data)
    assert fragment in str(excinfo.value)


def test_load_config_explicit_missing_file(tmp_path):
    with pytest.raises(ConfigError, match="ไม่พบไฟล์ config"):
        load_config(tmp_path / "nope.toml")


def test_load_config_reports_toml_syntax_error(tmp_path):
    path = tmp_path / "bad.toml"
    path.write_text("[camera\nindex = 0\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="bad.toml"):
        load_config(path)


def test_load_config_error_names_the_file(tmp_path):
    path = tmp_path / "custom.toml"
    path.write_text('[hud]\naccent = "#12"\n', encoding="utf-8")
    with pytest.raises(ConfigError) as excinfo:
        load_config(path)
    assert "custom.toml" in str(excinfo.value)
    assert "hud.accent" in str(excinfo.value)


def test_load_config_reads_file(tmp_path):
    path = tmp_path / "custom.toml"
    path.write_text("[camera]\nindex = 2\nmirror = false\n", encoding="utf-8")
    config = load_config(path)
    assert config.camera.index == 2
    assert config.camera.mirror is False
