"""เก็บ API key ของ Claude ไว้ในโปรไฟล์ของผู้ใช้ (ไม่อยู่ข้างตัวแอป — copy โฟลเดอร์แอปไปให้คนอื่นแล้ว key ไม่ติดไปด้วย)

- Windows: เข้ารหัสด้วย DPAPI (``CryptProtectData``) ผูกกับบัญชี Windows ของผู้ใช้ — คนอื่น/เครื่องอื่นถอดไม่ได้
  (ใช้ ctypes ไม่ต้องเพิ่มไลบรารี)
- ระบบอื่น: ไฟล์ข้อความสิทธิ์ 600 (อ่านได้เฉพาะเจ้าของ)
"""

from __future__ import annotations

import ctypes
import logging
import os
import sys
from pathlib import Path

__all__ = ["clear_api_key", "key_path", "load_api_key", "mask", "save_api_key"]

log = logging.getLogger(__name__)

_DESCRIPTION = "AETHER Claude API key"


def key_path() -> Path:
    if sys.platform == "win32":
        base = Path(os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming")
        return base / "AETHER" / "claude.key"
    base = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
    return base / "aether" / "claude.key"


class _Blob(ctypes.Structure):
    _fields_ = (("cbData", ctypes.c_uint32), ("pbData", ctypes.POINTER(ctypes.c_char)))


def _dpapi(data: bytes, *, protect: bool) -> bytes:  # pragma: no cover — Windows เท่านั้น (ทดสอบใน CI)
    crypt32 = ctypes.windll.crypt32  # type: ignore[attr-defined]
    kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
    buffer = ctypes.create_string_buffer(data, len(data))
    source = _Blob(len(data), ctypes.cast(buffer, ctypes.POINTER(ctypes.c_char)))
    result = _Blob()
    if protect:
        ok = crypt32.CryptProtectData(
            ctypes.byref(source), _DESCRIPTION, None, None, None, 0, ctypes.byref(result)
        )
    else:
        ok = crypt32.CryptUnprotectData(
            ctypes.byref(source), None, None, None, None, 0, ctypes.byref(result)
        )
    if not ok:
        raise OSError(f"DPAPI {'protect' if protect else 'unprotect'} failed")
    try:
        return ctypes.string_at(result.pbData, result.cbData)
    finally:
        kernel32.LocalFree(result.pbData)


def save_api_key(key: str, path: Path | None = None) -> Path:
    key = key.strip()
    if not key:
        raise ValueError("API key ว่าง")
    path = path or key_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    data = key.encode("utf-8")
    if sys.platform == "win32" and path == key_path():
        path.write_bytes(_dpapi(data, protect=True))
    else:
        path.write_bytes(data)
        os.chmod(path, 0o600)
    return path


def load_api_key(path: Path | None = None) -> str | None:
    """key ที่บันทึกไว้ หรือ ``None`` (ไม่มี / ถอดรหัสไม่ได้ เช่นย้ายไฟล์มาจากบัญชีอื่น)"""
    path = path or key_path()
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        return None
    except OSError:
        log.warning("อ่านไฟล์ API key ไม่ได้", exc_info=True)
        return None
    try:
        if sys.platform == "win32" and path == key_path():
            data = _dpapi(data, protect=False)
        return data.decode("utf-8").strip() or None
    except (OSError, UnicodeDecodeError):
        log.warning("ถอดรหัส API key ไม่ได้ — บันทึกใหม่ในหน้าต่างผู้ช่วย Claude")
        return None


def clear_api_key(path: Path | None = None) -> None:
    (path or key_path()).unlink(missing_ok=True)


def mask(key: str | None) -> str:
    """แสดงแค่ต้นกับท้าย: ``sk-ant-…a1b2``"""
    if not key:
        return "—"
    return f"{key[:7]}…{key[-4:]}" if len(key) > 14 else "•" * len(key)
