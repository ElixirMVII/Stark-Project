"""ค่าที่ปรับสดได้จากหน้าต่าง debug (spec v1.1 M5 "Live tuning panel") — pure logic

- :data:`TUNABLES` รายการ slider (key ใน config, ช่วง, ขั้น)
- :func:`apply_overrides` สร้าง :class:`~aether.config.Config` ใหม่จากค่าที่ปรับ (ตรวจด้วยกติกาเดียวกับ
  config.toml — ค่าที่ขัดกัน เช่น ``pinch.on >= pinch.off`` จะ raise ``ConfigError``)
- :func:`set_toml_value` แก้ค่าในข้อความ TOML **ทีละบรรทัด** คอมเมนต์และการจัดรูปแบบเดิมอยู่ครบ
  (stdlib มีแต่ตัวอ่าน TOML ไม่มีตัวเขียน และตัวเขียนทั่วไปจะทิ้งคอมเมนต์ของผู้ใช้)
- :func:`save_overrides` เขียนลง config.toml (และ calibration.json ถ้ามีค่า pinch ที่จะทับอยู่)
"""

from __future__ import annotations

import dataclasses
import json
import re
from dataclasses import dataclass
from pathlib import Path

from .config import Config, ConfigError

__all__ = [
    "TUNABLES",
    "Tunable",
    "apply_overrides",
    "current_values",
    "save_overrides",
    "set_toml_value",
]


@dataclass(frozen=True, slots=True)
class Tunable:
    key: str  # path ใน config เช่น "pinch.on", "pointer.one_euro.beta"
    label: str
    minimum: float
    maximum: float
    step: float
    integer: bool = False

    @property
    def section(self) -> str:
        """ชื่อ table ใน TOML ("pointer" สำหรับ pointer.one_euro.beta)"""
        return self.key.split(".")[0]

    @property
    def steps(self) -> int:
        return round((self.maximum - self.minimum) / self.step)

    def to_slider(self, value: float) -> int:
        return round((min(max(value, self.minimum), self.maximum) - self.minimum) / self.step)

    def from_slider(self, position: int) -> float:
        value = self.minimum + position * self.step
        return round(value) if self.integer else round(value, 6)

    def text(self, value: float) -> str:
        if self.integer:
            return f"{int(value)}"
        decimals = max(0, -int(f"{self.step:e}".split("e")[1]))
        return f"{value:.{decimals}f}"


TUNABLES: tuple[Tunable, ...] = (
    Tunable("pinch.on", "pinch.on (จีบ)", 0.10, 0.50, 0.01),
    Tunable("pinch.off", "pinch.off (ปล่อย)", 0.15, 0.70, 0.01),
    Tunable("pointer.one_euro.min_cutoff", "one_euro.min_cutoff", 0.05, 5.0, 0.05),
    Tunable("pointer.one_euro.beta", "one_euro.beta", 0.0, 0.2, 0.002),
    Tunable("gestures.swipe_dist_x", "swipe_dist_x", 0.05, 0.6, 0.01),
    Tunable("gestures.swipe_dist_y", "swipe_dist_y", 0.05, 0.6, 0.01),
    Tunable("gestures.cooldown_ms", "cooldown_ms", 0, 3000, 50, integer=True),
)


def _get(config: Config, key: str) -> float:
    value: object = config
    for part in key.split("."):
        value = getattr(value, part)
    return float(value)  # type: ignore[arg-type]


def current_values(config: Config) -> dict[str, float]:
    return {t.key: _get(config, t.key) for t in TUNABLES}


def _replace(obj: object, parts: list[str], value: float) -> object:
    head, *rest = parts
    current = getattr(obj, head)
    new = _replace(current, rest, value) if rest else type(current)(value)
    return dataclasses.replace(obj, **{head: new})  # type: ignore[type-var]


def apply_overrides(config: Config, overrides: dict[str, float]) -> Config:
    """config ใหม่ที่ใช้ค่าใน ``overrides`` (key ที่ไม่อยู่ใน TUNABLES = ``KeyError``)"""
    known = {t.key for t in TUNABLES}
    for key in overrides:
        if key not in known:
            raise KeyError(f"ปรับสดไม่ได้: {key}")
    # pinch.on/off ต้องเปลี่ยนพร้อมกัน ไม่งั้นค่ากลางทางอาจขัดกับ hysteresis ชั่วคราว
    pinch = {k: v for k, v in overrides.items() if k.startswith("pinch.")}
    if pinch:
        values = {"on": config.pinch.on, "off": config.pinch.off}
        values.update({k.split(".")[1]: float(v) for k, v in pinch.items()})
        config = dataclasses.replace(config, pinch=dataclasses.replace(config.pinch, **values))
    for key, value in overrides.items():
        if not key.startswith("pinch."):
            config = _replace(config, key.split("."), value)  # type: ignore[assignment]
    return config


# --------------------------------------------------------------------------- #
# TOML (แก้ทีละบรรทัด)
# --------------------------------------------------------------------------- #

_TABLE = re.compile(r"^\s*\[([A-Za-z0-9_.\-]+)\]\s*(#.*)?$")


def _format(value: float, integer: bool) -> str:
    if integer:
        return str(int(value))
    text = repr(float(round(value, 6)))
    return text if "e" not in text else f"{value:.6f}"


def set_toml_value(text: str, key: str, value: float, *, integer: bool = False) -> str:
    """ตั้ง ``key`` (เช่น ``"pinch.on"``, ``"pointer.one_euro.beta"``) ในข้อความ TOML

    - ถ้ามีอยู่แล้ว: แทนเฉพาะตัวเลข (คอมเมนต์ท้ายบรรทัดยังอยู่) รวมถึงค่าใน inline table
      ``one_euro = { min_cutoff = 1.0, beta = 0.02 }``
    - ถ้ายังไม่มี: เพิ่มบรรทัดท้าย table นั้น (หรือสร้าง table ใหม่ท้ายไฟล์)
    """
    parts = key.split(".")
    section, name = parts[0], parts[1]
    inner = parts[2] if len(parts) > 2 else None
    rendered = _format(value, integer)
    lines = text.splitlines()
    start = end = None
    for i, line in enumerate(lines):
        match = _TABLE.match(line)
        if match:
            if start is not None and end is None:
                end = i
            if match.group(1) == section:
                start = i
    if start is not None and end is None:
        end = len(lines)

    if start is not None:
        assert end is not None
        key_line = re.compile(rf"^(\s*{re.escape(name)}\s*=\s*)(.*)$")
        for i in range(start + 1, end):
            match = key_line.match(lines[i])
            if not match:
                continue
            prefix, rest = match.groups()
            if inner is None:
                lines[i] = prefix + re.sub(r"^[^\s#,]+", rendered, rest, count=1)
                return "\n".join(lines) + "\n"
            inline = re.compile(rf"(\b{re.escape(inner)}\s*=\s*)([^,}}\s]+)")
            if inline.search(rest):
                lines[i] = prefix + inline.sub(lambda m: m.group(1) + rendered, rest, count=1)
            else:  # inline table มีอยู่แต่ไม่มี key นี้: เติมก่อน }
                lines[i] = prefix + re.sub(r"\s*}", f", {inner} = {rendered} }}", rest, count=1)
            return "\n".join(lines) + "\n"
        # มี table แต่ไม่มี key: เติมหลังบรรทัดสุดท้ายที่ไม่ว่างของ table
        insert = end
        while insert > start + 1 and not lines[insert - 1].strip():
            insert -= 1
        new_line = (
            f"{name} = {{ {inner} = {rendered} }}" if inner is not None else f"{name} = {rendered}"
        )
        lines.insert(insert, new_line)
        return "\n".join(lines) + "\n"

    body = f"{name} = {{ {inner} = {rendered} }}" if inner is not None else f"{name} = {rendered}"
    tail = "\n".join(lines).rstrip("\n")
    return (tail + "\n\n" if tail else "") + f"[{section}]\n{body}\n"


def save_overrides(
    config_path: Path,
    overrides: dict[str, float],
    *,
    template: Path | None = None,
    calibration: Path | None = None,
) -> list[Path]:
    """เขียนค่าที่ปรับลง ``config_path`` (ไม่มีไฟล์: เริ่มจาก ``template`` ถ้ามี) คืนไฟล์ที่เขียน

    ถ้า ``calibration`` มีค่า pinch อยู่ (ซึ่งทับ config.toml ตอนโหลด) จะอัปเดตค่าในนั้นด้วย
    ไม่งั้นค่าที่บันทึกจะไม่มีผล
    """
    by_key = {t.key: t for t in TUNABLES}
    if config_path.is_file():
        text = config_path.read_text(encoding="utf-8")
    elif template is not None and template.is_file():
        text = template.read_text(encoding="utf-8")
    else:
        text = ""
    for key, value in overrides.items():
        text = set_toml_value(text, key, value, integer=by_key[key].integer)
    _validate(text)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = config_path.with_suffix(config_path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(config_path)
    written = [config_path]

    pinch = {k.split(".")[1]: v for k, v in overrides.items() if k.startswith("pinch.")}
    if pinch and calibration is not None and calibration.is_file():
        data = json.loads(calibration.read_text(encoding="utf-8"))
        if isinstance(data, dict) and isinstance(data.get("pinch"), dict):
            data["pinch"].update({k: round(float(v), 4) for k, v in pinch.items()})
            calibration.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", "utf-8")
            written.append(calibration)
    return written


def _validate(text: str) -> None:
    import tomllib

    from .config import parse_config

    try:
        parse_config(tomllib.loads(text))
    except (tomllib.TOMLDecodeError, ConfigError) as exc:
        raise ConfigError(f"ค่าที่จะบันทึกใช้ไม่ได้: {exc}") from exc
