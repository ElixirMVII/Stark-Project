"""โหลดและ validate ``config.toml`` → dataclass

ค่าเริ่มต้นทั้งหมดอยู่ใน dataclass ข้างล่าง (ตรงกับ ``config.toml`` ที่แจกไปกับโปรเจกต์)
ไฟล์ config จึงใส่เฉพาะ key ที่อยากเปลี่ยนก็ได้ ส่วน key ที่พิมพ์ผิดหรือค่าที่อยู่นอกช่วง
จะ raise :class:`ConfigError` พร้อมบอกตำแหน่ง ไม่ถูกข้ามเงียบๆ
"""

from __future__ import annotations

import dataclasses
import json
import os
import re
import sys
import tomllib
import typing
from dataclasses import dataclass, field
from pathlib import Path

from .hotkeys import parse_action, parse_hotkey

__all__ = [
    "THEMES",
    "ActionsConfig",
    "CameraConfig",
    "Config",
    "ConfigError",
    "GesturesConfig",
    "HotkeysConfig",
    "HudConfig",
    "LauncherConfig",
    "LauncherItem",
    "OneEuroConfig",
    "Palette",
    "PinchConfig",
    "PointerConfig",
    "PosesConfig",
    "SafetyConfig",
    "ScrollConfig",
    "TargetingConfig",
    "TrackingConfig",
    "app_root",
    "apply_calibration",
    "assets_dir",
    "bundle_root",
    "calibration_path",
    "default_config_path",
    "load_config",
    "log_dir",
    "model_path",
    "parse_config",
    "poses_path",
    "read_calibration",
]

GESTURE_MODEL_FILENAME = "gesture_recognizer.task"
CALIBRATION_FILENAME = "calibration.json"
CALIBRATION_VERSION = 1
POSES_FILENAME = "poses.json"

_HEX_COLOR = re.compile(r"#[0-9A-Fa-f]{6}")


class Palette(typing.NamedTuple):
    """สีของ HUD: หลัก, เน้น (ชาร์จ/จีบ/เตือน), อันตราย (error), สถานะ DISARMED"""

    primary: str
    accent: str
    danger: str
    disarmed: str


# ธีมสำเร็จรูป (hud.theme) — "custom" ใช้ hud.primary / hud.accent ที่ผู้ใช้ตั้งเอง
THEMES: dict[str, Palette] = {
    "aether": Palette("#00E5FF", "#FFB300", "#FF3D57", "#7A8B99"),  # ฟ้า + อำพัน (ค่าเริ่มต้น)
    "ember": Palette("#FF7A1A", "#FFE066", "#FF1744", "#8D7B70"),  # ส้ม + เหลือง
    "verdant": Palette("#3DFF8F", "#00E5FF", "#FF3D57", "#71877A"),  # เขียว + ฟ้า
    "violet": Palette("#B388FF", "#FF80CC", "#FF3D57", "#857C99"),  # ม่วง + ชมพู
    "ice": Palette("#E6F7FF", "#7FD8FF", "#FF5252", "#8A949B"),  # ขาวอมฟ้า + ฟ้า
}
THEME_NAMES = ("custom", *THEMES)


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
    low_light: int = 40  # ความสว่างเฉลี่ยของภาพ (0–255) ต่ำกว่านี้นานๆ = เตือน "แสงน้อย" (0 = ปิด)

    def __post_init__(self) -> None:
        _require(self.index >= 0, "camera.index ต้อง ≥ 0")
        _require(self.width > 0 and self.height > 0, "camera.width/height ต้อง > 0")
        _require(0 < self.fps <= 240, "camera.fps ต้องอยู่ในช่วง 1–240")
        _require(0 <= self.low_light <= 200, "camera.low_light ต้องอยู่ในช่วง 0–200")


@dataclass(frozen=True, slots=True)
class SafetyConfig:
    auto_disarm_minutes: float = 10.0  # ARMED แต่ไม่เห็นมือนานเท่านี้ = DISARM เอง (0 = ปิด)

    def __post_init__(self) -> None:
        _require(
            0.0 <= self.auto_disarm_minutes <= 240.0,
            "safety.auto_disarm_minutes ต้องอยู่ในช่วง 0–240 (0 = ปิด)",
        )


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
    screens: str = "primary"  # "primary" = จอหลักจอเดียว | "all" = ทุกจอรวมกัน

    def __post_init__(self) -> None:
        _require(
            self.screens in ("primary", "all"),
            f'pointer.screens ต้องเป็น "primary" หรือ "all" (ได้ {self.screens!r})',
        )
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
    window_grab: bool = True  # ✊ แล้วขยับ = หยิบหน้าต่างใต้เคอร์เซอร์ลาก/โยน (✊ นิ่งๆ ยังย่อหน้าต่างตามเดิม)

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
            try:
                parse_action(getattr(self, f.name))
            except ValueError as exc:
                raise ConfigError(f"actions.{f.name}: {exc}") from exc


@dataclass(frozen=True, slots=True)
class HotkeysConfig:
    toggle_arm: str = "<ctrl>+<alt>+g"
    quit: str = "<ctrl>+<alt>+q"
    mark_incident: str = "<ctrl>+<alt>+m"  # จด 5 วินาทีที่ผ่านมาลง incidents/ (แอปเพิ่งทำพลาด)

    def __post_init__(self) -> None:
        parsed = []
        for f in dataclasses.fields(self):
            try:
                parsed.append(parse_hotkey(getattr(self, f.name)))
            except ValueError as exc:
                raise ConfigError(f"hotkeys.{f.name}: {exc}") from exc
        _require(
            len(set(parsed)) == len(parsed),
            "hotkeys.toggle_arm, hotkeys.quit และ hotkeys.mark_incident ต้องไม่ซ้ำกัน",
        )


@dataclass(frozen=True, slots=True)
class HudConfig:
    enabled: bool = True
    opacity: float = 0.9
    theme: str = "custom"  # "custom" = ใช้ primary/accent ข้างล่าง | ชื่อธีมใน THEMES
    primary: str = "#00E5FF"
    accent: str = "#FFB300"
    show_telemetry: bool = True
    show_radar: bool = True
    sounds: bool = True
    volume: float = 0.5
    reduce_motion: bool = False  # ปิด particle, การหายใจ และแอนิเมชันตอนบูต (เหลือแค่ fade)
    quality: str = "high"  # "high" | "low" (low = ไม่มี glow ชั้นนอก และ particle ครึ่งเดียว)

    def __post_init__(self) -> None:
        _require(
            self.quality in ("high", "low"),
            f'hud.quality ต้องเป็น "high" หรือ "low" (ได้ {self.quality!r})',
        )
        _require(0.0 <= self.opacity <= 1.0, "hud.opacity ต้องอยู่ในช่วง 0–1")
        _require(0.0 <= self.volume <= 1.0, "hud.volume ต้องอยู่ในช่วง 0–1")
        for name in ("primary", "accent"):
            value = getattr(self, name)
            _require(
                _HEX_COLOR.fullmatch(value) is not None,
                f'hud.{name} ต้องเป็นสีแบบ "#RRGGBB" (ได้ {value!r})',
            )
        _require(
            self.theme in THEME_NAMES,
            f"hud.theme ต้องเป็นหนึ่งใน {', '.join(THEME_NAMES)} (ได้ {self.theme!r})",
        )

    @property
    def palette(self) -> Palette:
        """สีที่ใช้จริง: จากธีมสำเร็จรูป หรือ primary/accent ที่ตั้งเอง (theme = "custom")"""
        if self.theme in THEMES:
            return THEMES[self.theme]
        default = THEMES["aether"]
        return Palette(self.primary, self.accent, default.danger, default.disarmed)


LAUNCHER_MAX_ITEMS = 8


@dataclass(frozen=True, slots=True)
class LauncherItem:
    label: str  # ข้อความบนเมนู (สั้นๆ)
    run: str  # สิ่งที่จะเปิด: โปรแกรม (notepad.exe), ไฟล์/โฟลเดอร์, หรือ URL
    args: str = ""  # argument ของโปรแกรม (ไม่บังคับ)

    def __post_init__(self) -> None:
        _require(bool(self.label.strip()), "launcher.items: label ต้องไม่ว่าง")
        _require(bool(self.run.strip()), f"launcher.items ({self.label}): run ต้องไม่ว่าง")


@dataclass(frozen=True, slots=True)
class LauncherConfig:
    enabled: bool = True
    hold_ms: int = 600  # ค้าง 👍 นานเท่านี้จึงเปิดเมนู
    timeout_ms: int = 5000  # เปิดเมนูค้างไว้โดยไม่เลือกนานเท่านี้ → ปิดเอง
    items: tuple[LauncherItem, ...] = (
        LauncherItem("Browser", "https://www.google.com"),
        LauncherItem("Files", "explorer.exe"),
        LauncherItem("Notepad", "notepad.exe"),
        LauncherItem("Calc", "calc.exe"),
    )

    def __post_init__(self) -> None:
        _require(self.hold_ms > 0, "launcher.hold_ms ต้อง > 0")
        _require(self.timeout_ms >= 1000, "launcher.timeout_ms ต้อง ≥ 1000")
        _require(
            len(self.items) <= LAUNCHER_MAX_ITEMS,
            f"launcher.items มีได้ไม่เกิน {LAUNCHER_MAX_ITEMS} รายการ (มี {len(self.items)})",
        )


@dataclass(frozen=True, slots=True)
class PosesConfig:
    """ท่าที่ผู้ใช้สอนเอง (เก็บใน ``poses.json`` — สอน/ลบผ่านเมนู tray "สอนท่ามือ…")"""

    enabled: bool = True
    hold_ms: int = 600  # ค้างท่าที่สอนไว้นานเท่านี้จึงทำงาน
    sensitivity: float = 1.0  # > 1 = รับท่าง่ายขึ้น (อาจติดเองบ่อยขึ้น), < 1 = เข้มขึ้น

    def __post_init__(self) -> None:
        _require(self.hold_ms > 0, "poses.hold_ms ต้อง > 0")
        _require(0.25 <= self.sensitivity <= 4.0, "poses.sensitivity ต้องอยู่ในช่วง 0.25–4.0")


@dataclass(frozen=True, slots=True)
class TargetingConfig:
    """Target Lock + Magnetic Snap (Windows UI Automation) — spec v1.1 M6"""

    enabled: bool = True
    rate_hz: int = 15  # ถามหาปุ่มใต้เคอร์เซอร์กี่ครั้งต่อวินาที (+ ทุกครั้งที่เคอร์เซอร์ขยับเกิน 8px)
    snap_strength: float = 0.6  # แรงดูดเข้าหากลางปุ่ม (0 = ไม่ดูด แค่แสดงกรอบ)
    escape_speed: float = 2.5  # ขยับมือเร็วกว่านี้ (ความกว้างจอต่อวินาที) = หลุดจากแรงดูด
    max_size: tuple[int, int] = (480, 160)  # logical px: ปุ่มใหญ่กว่านี้ไม่ดูด (เช่นพื้นที่ข้อความ)
    show_label: bool = True  # ป้ายชนิด · ชื่อปุ่มบน HUD
    app_blocklist: tuple[str, ...] = ()  # ชื่อโปรแกรมที่ไม่ต้องหาเป้า เช่น ("game.exe",)

    def __post_init__(self) -> None:
        _require(1 <= self.rate_hz <= 60, "targeting.rate_hz ต้องอยู่ในช่วง 1–60")
        _require(0.0 <= self.snap_strength <= 1.0, "targeting.snap_strength ต้องอยู่ในช่วง 0–1")
        _require(self.escape_speed > 0, "targeting.escape_speed ต้อง > 0")
        _require(
            self.max_size[0] >= 8 and self.max_size[1] >= 8,
            "targeting.max_size ต้องเป็น [กว้าง, สูง] อย่างน้อย 8 px",
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
    launcher: LauncherConfig = field(default_factory=LauncherConfig)
    poses: PosesConfig = field(default_factory=PosesConfig)
    targeting: TargetingConfig = field(default_factory=TargetingConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)


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
        if len(item_types) == 2 and item_types[1] is Ellipsis:  # tuple[X, ...]: ยาวเท่าไรก็ได้
            _require(isinstance(value, list), f"{where} ต้องเป็น array (ได้ {value!r})")
            return tuple(
                _coerce(item_types[0], item, f"{where}[{i}]")
                for i, item in enumerate(typing.cast(list, value))
            )
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


# Paths
# --------------------------------------------------------------------------- #


def _repo_root() -> Path | None:
    root = Path(__file__).resolve().parents[2]
    if (root / "assets").is_dir() or (root / "config.toml").is_file():
        return root
    return None


def app_root() -> Path:
    """โฟลเดอร์ของไฟล์ที่ผู้ใช้แก้ได้: ``config.toml``, ``calibration.json``, ``poses.json``, ``logs/``

    - รันจาก source (``pip install -e .``): root ของ repo
    - build ด้วย PyInstaller: โฟลเดอร์ที่มีไฟล์ exe
    - กรณีอื่น: โฟลเดอร์ปัจจุบัน
    """
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return _repo_root() or Path.cwd()


def bundle_root() -> Path:
    """โฟลเดอร์ของไฟล์ที่แจกไปกับแอป (อ่านอย่างเดียว): ``assets/`` และ ``config.toml`` ต้นฉบับ

    ใน build ของ PyInstaller คือโฟลเดอร์ข้อมูลของ bundle (``sys._MEIPASS`` — ปกติคือ ``_internal``)
    """
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).resolve().parent))
    return _repo_root() or Path.cwd()


def assets_dir() -> Path:
    return bundle_root() / "assets"


def model_path() -> Path:
    return assets_dir() / "models" / GESTURE_MODEL_FILENAME


def default_config_path() -> Path:
    """``config.toml`` ข้างตัวแอป ถ้าไม่มี ใช้ต้นฉบับที่แจกไปกับ bundle"""
    path = app_root() / "config.toml"
    if not path.is_file() and (bundle_root() / "config.toml").is_file():
        return bundle_root() / "config.toml"
    return path


def calibration_path() -> Path:
    return app_root() / CALIBRATION_FILENAME


def poses_path() -> Path:
    return app_root() / POSES_FILENAME


def log_dir() -> Path:
    """โฟลเดอร์เก็บ log: ``logs/`` ข้างตัวแอป ถ้าเขียนไม่ได้ (เช่นติดตั้งใน Program Files) ใช้โฟลเดอร์ของผู้ใช้"""
    primary = app_root() / "logs"
    try:
        primary.mkdir(parents=True, exist_ok=True)
        if os.access(primary, os.W_OK):
            return primary
    except OSError:
        pass
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home()) / "AETHER"
    else:
        base = Path(os.environ.get("XDG_STATE_HOME") or Path.home() / ".local" / "state") / "aether"
    fallback = base / "logs"
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback


# --------------------------------------------------------------------------- #
# Calibration (ค่าที่วัดจากผู้ใช้ — เขียนโดย calibration wizard)
# --------------------------------------------------------------------------- #


def read_calibration(path: Path) -> dict | None:
    """อ่าน ``calibration.json`` คืน ``None`` ถ้าไม่มีไฟล์ (raise :class:`ConfigError` ถ้าไฟล์เสีย)"""
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ConfigError(f"{path}: อ่านไม่ได้ — {exc} (ลบไฟล์นี้แล้ว calibrate ใหม่ได้)") from exc
    if not isinstance(data, dict) or data.get("version") != CALIBRATION_VERSION:
        raise ConfigError(f"{path}: ไม่ใช่ไฟล์ calibration เวอร์ชัน {CALIBRATION_VERSION}")
    return data


def apply_calibration(config: Config, calibration: dict) -> Config:
    """แทนค่า ``pinch.on/off`` และ ``pointer.active_region`` ด้วยค่าที่วัดจากผู้ใช้

    ค่าอื่นของ config คงเดิม ค่าที่วัดมาถูกตรวจด้วยกติกาเดียวกับ config.toml
    """
    try:
        pinch = calibration.get("pinch")
        if pinch is not None:
            config = dataclasses.replace(
                config,
                pinch=dataclasses.replace(
                    config.pinch,
                    on=_coerce(float, pinch["on"], "calibration.pinch.on"),
                    off=_coerce(float, pinch["off"], "calibration.pinch.off"),
                ),
            )
        pointer = calibration.get("pointer")
        if pointer is not None:
            region = _coerce(
                tuple[float, float, float, float],
                pointer["active_region"],
                "calibration.pointer.active_region",
            )
            config = dataclasses.replace(
                config, pointer=dataclasses.replace(config.pointer, active_region=region)
            )
    except (KeyError, TypeError, AttributeError) as exc:
        raise ConfigError(f"calibration: รูปแบบไม่ถูกต้อง ({exc!r})") from exc
    return config


def load_config(
    path: Path | None = None,
    *,
    use_calibration: bool = True,
    calibration: Path | None = None,
) -> Config:
    """โหลด config แล้วทับด้วยค่าจาก ``calibration.json`` (ถ้ามี)

    ``path=None`` ใช้ ``config.toml`` ข้างตัวแอป ถ้าไม่มีไฟล์ใช้ค่าเริ่มต้นทั้งหมด
    ถ้าระบุ ``path`` เองแล้วไม่มีไฟล์นั้น ถือเป็น error

    ``use_calibration=False`` ไม่ใช้ค่าที่ calibrate ไว้ ``calibration`` ระบุไฟล์ calibration เอง
    (ค่าเริ่มต้น: ``calibration.json`` ข้างตัวแอป)
    """
    if path is None:
        path = default_config_path()
        config = _load_toml(path) if path.is_file() else Config()
    elif not path.is_file():
        raise ConfigError(f"ไม่พบไฟล์ config: {path}")
    else:
        config = _load_toml(path)

    if use_calibration:
        calibration_file = calibration or calibration_path()
        data = read_calibration(calibration_file)
        if data is not None:
            try:
                config = apply_calibration(config, data)
            except ConfigError as exc:
                raise ConfigError(f"{calibration_file}: {exc}") from exc
    return config


def _load_toml(path: Path) -> Config:
    try:
        with path.open("rb") as fh:
            data = tomllib.load(fh)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{path}: ไวยากรณ์ TOML ผิด — {exc}") from exc
    try:
        return parse_config(data)
    except ConfigError as exc:
        raise ConfigError(f"{path}: {exc}") from exc
