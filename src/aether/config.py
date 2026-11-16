"""โหลดและ validate ``config.toml`` → dataclass

ค่าเริ่มต้นทั้งหมดอยู่ใน dataclass ข้างล่าง (ตรงกับ ``config.toml`` ที่แจกไปกับโปรเจกต์)
ไฟล์ config จึงใส่เฉพาะ key ที่อยากเปลี่ยนก็ได้ ส่วน key ที่พิมพ์ผิดหรือค่าที่อยู่นอกช่วง
จะ raise :class:`ConfigError` พร้อมบอกตำแหน่ง ไม่ถูกข้ามเงียบๆ
"""

from __future__ import annotations

import dataclasses
import re
import sys
import tomllib
import typing
from dataclasses import dataclass, field
from pathlib import Path

from .hotkeys import parse_hotkey

__all__ = [
    "ActionsConfig",
    "CameraConfig",
    "Config",
    "ConfigError",
    "GesturesConfig",
    "HotkeysConfig",
    "HudConfig",
    "OneEuroConfig",
    "PinchConfig",
    "PointerConfig",
    "ScrollConfig",
    "TrackingConfig",
    "app_root",
    "assets_dir",
    "default_config_path",
    "load_config",
    "model_path",
    "parse_config",
]

GESTURE_MODEL_FILENAME = "gesture_recognizer.task"

_HEX_COLOR = re.compile(r"#[0-9A-Fa-f]{6}")


class ConfigError(ValueError):
    """config.toml มี key ที่ไม่รู้จัก ชนิดข้อมูลผิด หรือค่าอยู่นอกช่วงที่ใช้ได้"""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ConfigError(message)


# --------------------------------------------------------------------------- #
# Sections
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class CameraConfig:
    index: int = 0
    width: int = 640
    height: int = 480
    fps: int = 30
    mirror: bool = True

    def __post_init__(self) -> None:
        _require(self.index >= 0, "camera.index ต้อง ≥ 0")
        _require(self.width > 0 and self.height > 0, "camera.width/height ต้อง > 0")
        _require(0 < self.fps <= 240, "camera.fps ต้องอยู่ในช่วง 1–240")


@dataclass(frozen=True, slots=True)
class TrackingConfig:
    dominant_hand: str = "right"
    num_hands: int = 2
    min_gesture_score: float = 0.6
    vote_window: int = 5

    def __post_init__(self) -> None:
        _require(
            self.dominant_hand in ("left", "right"),
            f'tracking.dominant_hand ต้องเป็น "left" หรือ "right" (ได้ {self.dominant_hand!r})',
        )
        _require(self.num_hands in (1, 2), "tracking.num_hands ต้องเป็น 1 หรือ 2")
        _require(
            0.0 <= self.min_gesture_score <= 1.0,
            "tracking.min_gesture_score ต้องอยู่ในช่วง 0–1",
        )
        _require(self.vote_window >= 1, "tracking.vote_window ต้อง ≥ 1")


@dataclass(frozen=True, slots=True)
class OneEuroConfig:
    min_cutoff: float = 1.0
    beta: float = 0.02
    d_cutoff: float = 1.0

    def __post_init__(self) -> None:
        _require(self.min_cutoff > 0, "pointer.one_euro.min_cutoff ต้อง > 0")
        _require(self.beta >= 0, "pointer.one_euro.beta ต้อง ≥ 0")
        _require(self.d_cutoff > 0, "pointer.one_euro.d_cutoff ต้อง > 0")


@dataclass(frozen=True, slots=True)
class PointerConfig:
    # x0, y0, x1, y1 (normalized ในเฟรมกล้อง) → map เต็มจอ
    active_region: tuple[float, float, float, float] = (0.15, 0.10, 0.85, 0.70)
    one_euro: OneEuroConfig = field(default_factory=OneEuroConfig)
    pinch_lock_gain: float = 0.2

    def __post_init__(self) -> None:
        x0, y0, x1, y1 = self.active_region
        _require(
            0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0,
            "pointer.active_region ต้องเป็น [x0, y0, x1, y1] ในช่วง 0–1 โดย x0 < x1 และ y0 < y1",
        )
        _require(
            0.0 <= self.pinch_lock_gain <= 1.0,
            "pointer.pinch_lock_gain ต้องอยู่ในช่วง 0–1",
        )


@dataclass(frozen=True, slots=True)
class PinchConfig:
    on: float = 0.25
    off: float = 0.38
    drag_hold_ms: int = 250

    def __post_init__(self) -> None:
        _require(
            0.0 < self.on < self.off,
            "pinch.on ต้อง > 0 และน้อยกว่า pinch.off (hysteresis)",
        )
        _require(self.drag_hold_ms > 0, "pinch.drag_hold_ms ต้อง > 0")


@dataclass(frozen=True, slots=True)
class ScrollConfig:
    speed: float = 1.0
    deadzone: float = 0.01
    invert: bool = False  # true = กลับทิศ (มือขึ้น = เลื่อนลง แบบ "ดันหน้ากระดาษ")

    def __post_init__(self) -> None:
        _require(self.speed > 0, "scroll.speed ต้อง > 0")
        _require(0.0 <= self.deadzone < 1.0, "scroll.deadzone ต้องอยู่ในช่วง 0–1")


@dataclass(frozen=True, slots=True)
class GesturesConfig:
    swipe_dist_x: float = 0.25
    swipe_dist_y: float = 0.20
    swipe_window_ms: int = 300
    cooldown_ms: int = 800
    fist_hold_ms: int = 600
    arm_hold_ms: int = 1000
    zoom_step: float = 0.06

    def __post_init__(self) -> None:
        _require(0.0 < self.swipe_dist_x <= 1.0, "gestures.swipe_dist_x ต้องอยู่ในช่วง (0, 1]")
        _require(0.0 < self.swipe_dist_y <= 1.0, "gestures.swipe_dist_y ต้องอยู่ในช่วง (0, 1]")
        _require(self.swipe_window_ms > 0, "gestures.swipe_window_ms ต้อง > 0")
        _require(self.cooldown_ms >= 0, "gestures.cooldown_ms ต้อง ≥ 0")
        _require(self.fist_hold_ms > 0, "gestures.fist_hold_ms ต้อง > 0")
        _require(self.arm_hold_ms > 0, "gestures.arm_hold_ms ต้อง > 0")
        _require(0.0 < self.zoom_step < 1.0, "gestures.zoom_step ต้องอยู่ในช่วง (0, 1)")


@dataclass(frozen=True, slots=True)
class ActionsConfig:
    swipe_left: str = "ctrl+win+left"
    swipe_right: str = "ctrl+win+right"
    swipe_up: str = "win+tab"
    swipe_down: str = "win+d"
    fist_hold: str = "minimize_foreground"

    def __post_init__(self) -> None:
        for f in dataclasses.fields(self):
            _require(bool(getattr(self, f.name).strip()), f"actions.{f.name} ต้องไม่ว่าง")


@dataclass(frozen=True, slots=True)
class HotkeysConfig:
    toggle_arm: str = "<ctrl>+<alt>+g"
    quit: str = "<ctrl>+<alt>+q"

    def __post_init__(self) -> None:
        parsed = []
        for f in dataclasses.fields(self):
            try:
                parsed.append(parse_hotkey(getattr(self, f.name)))
            except ValueError as exc:
                raise ConfigError(f"hotkeys.{f.name}: {exc}") from exc
        _require(
            len(set(parsed)) == len(parsed),
            "hotkeys.toggle_arm กับ hotkeys.quit ต้องไม่ซ้ำกัน",
        )


@dataclass(frozen=True, slots=True)
class HudConfig:
    enabled: bool = True
    opacity: float = 0.9
    primary: str = "#00E5FF"
    accent: str = "#FFB300"
    show_telemetry: bool = True
    show_radar: bool = True
    sounds: bool = True
    volume: float = 0.5

    def __post_init__(self) -> None:
        _require(0.0 <= self.opacity <= 1.0, "hud.opacity ต้องอยู่ในช่วง 0–1")
        _require(0.0 <= self.volume <= 1.0, "hud.volume ต้องอยู่ในช่วง 0–1")
        for name in ("primary", "accent"):
            value = getattr(self, name)
            _require(
                _HEX_COLOR.fullmatch(value) is not None,
                f'hud.{name} ต้องเป็นสีแบบ "#RRGGBB" (ได้ {value!r})',
            )


@dataclass(frozen=True, slots=True)
class Config:
    camera: CameraConfig = field(default_factory=CameraConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    pointer: PointerConfig = field(default_factory=PointerConfig)
    pinch: PinchConfig = field(default_factory=PinchConfig)
    scroll: ScrollConfig = field(default_factory=ScrollConfig)
    gestures: GesturesConfig = field(default_factory=GesturesConfig)
    actions: ActionsConfig = field(default_factory=ActionsConfig)
    hotkeys: HotkeysConfig = field(default_factory=HotkeysConfig)
    hud: HudConfig = field(default_factory=HudConfig)


# --------------------------------------------------------------------------- #
# TOML → dataclass
# --------------------------------------------------------------------------- #


def _type_name(tp: object) -> str:
    return getattr(tp, "__name__", str(tp))


def _coerce(tp: object, value: object, where: str) -> object:
    """แปลงค่าจาก TOML ให้ตรงกับ type ของ field (เข้มงวด: ไม่แปลง str → ตัวเลขให้)"""
    if dataclasses.is_dataclass(tp):
        return _build(typing.cast(type, tp), value, where)

    if tp is bool:
        _require(isinstance(value, bool), f"{where} ต้องเป็น true/false (ได้ {value!r})")
        return value

    # bool เป็น subclass ของ int ใน Python จึงต้องกันออกจาก field ตัวเลข
    is_number = isinstance(value, (int, float)) and not isinstance(value, bool)

    if tp is int:
        _require(
            isinstance(value, int) and not isinstance(value, bool),
            f"{where} ต้องเป็นจำนวนเต็ม (ได้ {value!r})",
        )
        return value

    if tp is float:
        _require(is_number, f"{where} ต้องเป็นตัวเลข (ได้ {value!r})")
        return float(typing.cast(float, value))

    if tp is str:
        _require(isinstance(value, str), f"{where} ต้องเป็นข้อความ (ได้ {value!r})")
        return value

    if typing.get_origin(tp) is tuple:
        item_types = typing.get_args(tp)
        _require(
            isinstance(value, list) and len(value) == len(item_types),
            f"{where} ต้องเป็น array ที่มี {len(item_types)} ค่า (ได้ {value!r})",
        )
        items = typing.cast(list, value)
        return tuple(
            _coerce(item_tp, item, f"{where}[{i}]")
            for i, (item_tp, item) in enumerate(zip(item_types, items))
        )

    raise TypeError(f"config field {where} มี type ที่ loader ไม่รองรับ: {_type_name(tp)}")


def _build(cls: type, data: object, where: str) -> object:
    _require(isinstance(data, dict), f"[{where}] ต้องเป็น table (ได้ {data!r})")
    table = typing.cast(dict, data)
    hints = typing.get_type_hints(cls)
    known = [f.name for f in dataclasses.fields(cls)]
    unknown = sorted(set(table) - set(known))
    if unknown:
        prefix = f"{where}." if where else ""
        raise ConfigError(
            f"ไม่รู้จัก key: {', '.join(prefix + k for k in unknown)} (ที่ใช้ได้: {', '.join(known)})"
        )
    kwargs = {
        name: _coerce(hints[name], value, f"{where}.{name}" if where else name)
        for name, value in table.items()
    }
    return cls(**kwargs)


def parse_config(data: dict) -> Config:
    """สร้าง :class:`Config` จาก dict ที่ได้จาก ``tomllib`` (key ที่ไม่ใส่ใช้ค่าเริ่มต้น)"""
    return typing.cast(Config, _build(Config, data, ""))


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #


def app_root() -> Path:
    """โฟลเดอร์หลักของแอป (ที่มี ``config.toml`` และ ``assets/``)

    - รันจาก source (``pip install -e .``): root ของ repo
    - build ด้วย PyInstaller: โฟลเดอร์ที่มีไฟล์ exe
    - กรณีอื่น: โฟลเดอร์ปัจจุบัน
    """
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    repo_root = Path(__file__).resolve().parents[2]
    if (repo_root / "assets").is_dir() or (repo_root / "config.toml").is_file():
        return repo_root
    return Path.cwd()


def assets_dir() -> Path:
    return app_root() / "assets"


def model_path() -> Path:
    return assets_dir() / "models" / GESTURE_MODEL_FILENAME


def default_config_path() -> Path:
    return app_root() / "config.toml"


def load_config(path: Path | None = None) -> Config:
    """โหลด config

    ``path=None`` ใช้ ``config.toml`` ที่ root ของแอป ถ้าไม่มีไฟล์ใช้ค่าเริ่มต้นทั้งหมด
    ถ้าระบุ ``path`` เองแล้วไม่มีไฟล์นั้น ถือเป็น error
    """
    if path is None:
        path = default_config_path()
        if not path.is_file():
            return Config()
    elif not path.is_file():
        raise ConfigError(f"ไม่พบไฟล์ config: {path}")

    try:
        with path.open("rb") as fh:
            data = tomllib.load(fh)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{path}: ไวยากรณ์ TOML ผิด — {exc}") from exc

    try:
        return parse_config(data)
    except ConfigError as exc:
        raise ConfigError(f"{path}: {exc}") from exc
