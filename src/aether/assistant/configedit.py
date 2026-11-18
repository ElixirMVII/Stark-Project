"""แก้ config.toml ตามที่ผู้ช่วย Claude เสนอ — ตรวจทุกค่าก่อนเขียน (คีย์ต้องมีจริง ชนิดต้องตรง config ต้องโหลดได้)

ห้ามแก้ ``[claude]`` ผ่านข้อเสนอ (เปิด/ปิดฟีเจอร์ทำได้เฉพาะผู้ใช้กดเองในหน้าต่างผู้ช่วย)
"""

from __future__ import annotations

import dataclasses
import tomllib
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from ..config import Config, ConfigError, parse_config
from ..tuning import set_toml_literal

__all__ = [
    "apply_changes",
    "editable_keys",
    "preview_changes",
    "read_config_text",
    "set_claude_enabled",
    "write_config_text",
]

PROTECTED_SECTIONS = ("claude",)


def editable_keys(config: Config | None = None) -> dict[str, Any]:
    """``{"section.name": ค่าปัจจุบัน}`` ของทุกคีย์ที่ข้อเสนอแก้ได้ (รวม inline table เช่น pointer.one_euro.beta)"""
    config = config or Config()
    keys: dict[str, Any] = {}
    for section in dataclasses.fields(config):
        if section.name in PROTECTED_SECTIONS:
            continue
        value = getattr(config, section.name)
        if not dataclasses.is_dataclass(value):
            continue
        for field in dataclasses.fields(value):
            item = getattr(value, field.name)
            key = f"{section.name}.{field.name}"
            if dataclasses.is_dataclass(item):
                for inner in dataclasses.fields(item):
                    leaf = getattr(item, inner.name)
                    if _scalar(leaf):
                        keys[f"{key}.{inner.name}"] = leaf
            elif _scalar(item) or (isinstance(item, tuple) and all(_scalar(v) for v in item)):
                keys[key] = item
    return keys


def _scalar(value: Any) -> bool:
    return isinstance(value, bool | int | float | str)


def _kind(value: Any) -> str:
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int | float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list | tuple):
        return "array"
    return type(value).__name__


def read_config_text(path: Path, template: Path | None = None) -> str:
    if path.is_file():
        return path.read_text(encoding="utf-8")
    if template is not None and template.is_file():
        return template.read_text(encoding="utf-8")
    return ""


def _parse_literal(literal: str) -> Any:
    try:
        return tomllib.loads(f"v = {literal.strip()}")["v"]
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"ค่า {literal!r} ไม่ใช่ TOML ที่ถูกต้อง") from exc


def preview_changes(text: str, changes: Iterable[tuple[str, str]]) -> str:
    """คืนข้อความ config ใหม่หลังแก้ทุกคีย์ — ผิดแม้แต่ข้อเดียว = :class:`ConfigError` (ไม่แก้อะไรเลย)"""
    allowed = editable_keys()
    for key, literal in changes:
        key = key.strip()
        if key.split(".")[0] in PROTECTED_SECTIONS:
            raise ConfigError(f"{key}: แก้ส่วน [claude] ผ่านข้อเสนอไม่ได้")
        if key not in allowed:
            raise ConfigError(f"{key}: ไม่มีคีย์นี้ใน config (หรือแก้ผ่านผู้ช่วยไม่ได้)")
        value = _parse_literal(literal)
        expected = _kind(allowed[key])
        if _kind(value) != expected:
            raise ConfigError(f"{key}: ต้องเป็น {expected} แต่ได้ {_kind(value)} ({literal})")
        text = set_toml_literal(text, key, literal.strip())
    try:
        parse_config(tomllib.loads(text))
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"config หลังแก้อ่านไม่ได้: {exc}") from exc
    return text


def write_config_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def apply_changes(
    path: Path, changes: Iterable[tuple[str, str]], *, template: Path | None = None
) -> str:
    text = preview_changes(read_config_text(path, template), list(changes))
    write_config_text(path, text)
    return text


def set_claude_enabled(path: Path, enabled: bool, *, template: Path | None = None) -> str:
    """ผู้ใช้กดเปิด/ปิดฟีเจอร์ Claude เอง (ทางเดียวที่แก้ ``[claude]`` ได้)"""
    text = set_toml_literal(
        read_config_text(path, template), "claude.enabled", "true" if enabled else "false"
    )
    try:
        parse_config(tomllib.loads(text))
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"config อ่านไม่ได้: {exc}") from exc
    write_config_text(path, text)
    return text
