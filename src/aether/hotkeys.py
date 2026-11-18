"""Global hotkey (ARM/DISARM, ปิดแอป)

บน Windows ใช้ ``RegisterHotKey`` ของ Win32 ผ่าน ctypes ไม่ใช้ ``pynput.keyboard.GlobalHotKeys``
เพราะ pynput เทียบ hotkey จาก *ตัวอักษร* ที่ปุ่มนั้นพิมพ์ออกมาใน layout ปัจจุบัน ถ้าเปิดแป้นไทยอยู่
ปุ่ม G จะกลายเป็น "เ" แล้ว ``Ctrl+Alt+G`` จะไม่ติด ซึ่งรับไม่ได้สำหรับปุ่มหยุดฉุกเฉิน
``RegisterHotKey`` เทียบจาก virtual-key code จึงไม่ขึ้นกับภาษาของแป้น (ดู docs/DECISIONS.md)

รูปแบบใน config ยังเป็นแบบ pynput ตาม spec เช่น ``<ctrl>+<alt>+g``

callback ถูกเรียกจาก thread ของ listener (ไม่ใช่ UI thread)
"""

from __future__ import annotations

import logging
import sys
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Protocol

__all__ = [
    "BUILTIN_ACTIONS",
    "MEDIA_KEYS",
    "Hotkey",
    "HotkeyManager",
    "parse_action",
    "parse_hotkey",
]

log = logging.getLogger(__name__)

_MODIFIER_ALIASES = {
    "ctrl": "ctrl",
    "control": "ctrl",
    "alt": "alt",
    "shift": "shift",
    "win": "win",
    "cmd": "win",
    "super": "win",
}

_NAMED_KEYS: dict[str, int] = {
    "esc": 0x1B,
    "space": 0x20,
    "enter": 0x0D,
    "tab": 0x09,
    "backspace": 0x08,
    "delete": 0x2E,
    "insert": 0x2D,
    "home": 0x24,
    "end": 0x23,
    "page_up": 0x21,
    "page_down": 0x22,
    "left": 0x25,
    "up": 0x26,
    "right": 0x27,
    "down": 0x28,
    "pause": 0x13,
    **{f"f{n}": 0x70 + n - 1 for n in range(1, 25)},
    # ปุ่มสื่อ/เสียง (ใช้เดี่ยวๆ ได้ เช่น actions.thumb_down_hold = "media_play_pause")
    "volume_mute": 0xAD,
    "volume_down": 0xAE,
    "volume_up": 0xAF,
    "media_next": 0xB0,
    "media_prev": 0xB1,
    "media_stop": 0xB2,
    "media_play_pause": 0xB3,
}
MEDIA_KEYS = frozenset(name for name, vk in _NAMED_KEYS.items() if 0xAD <= vk <= 0xB3)

_MOD_FLAGS = {"alt": 0x0001, "ctrl": 0x0002, "shift": 0x0004, "win": 0x0008}
_MOD_NOREPEAT = 0x4000
_WM_HOTKEY = 0x0312
_WM_QUIT = 0x0012


@dataclass(frozen=True, slots=True)
class Hotkey:
    """ปุ่มลัดที่ parse แล้ว: ชุด modifier + ปุ่มหลักหนึ่งปุ่ม"""

    modifiers: frozenset[str]  # สมาชิกจาก {"ctrl", "alt", "shift", "win"}
    key: str  # ตัวอักษร/ตัวเลขตัวเดียว (ตัวพิมพ์เล็ก) หรือชื่อปุ่ม เช่น "f9", "esc"

    @property
    def vk(self) -> int:
        """Windows virtual-key code ของปุ่มหลัก"""
        if self.key in _NAMED_KEYS:
            return _NAMED_KEYS[self.key]
        return ord(self.key.upper())

    @property
    def win32_modifiers(self) -> int:
        flags = 0
        for name in self.modifiers:
            flags |= _MOD_FLAGS[name]
        return flags

    @property
    def keys(self) -> tuple[str, ...]:
        """ลำดับการกด: modifier (ctrl, alt, shift, win) แล้วตามด้วยปุ่มหลัก"""
        order = ("ctrl", "alt", "shift", "win")
        return (*(name for name in order if name in self.modifiers), self.key)

    def __str__(self) -> str:
        order = ("ctrl", "alt", "shift", "win")
        parts = [name.capitalize() for name in order if name in self.modifiers]
        return "+".join([*parts, self.key.upper()])

    def to_pynput(self) -> str:
        """รูปแบบของ ``pynput.keyboard.GlobalHotKeys`` (ใช้นอก Windows)"""
        names = {"ctrl": "<ctrl>", "alt": "<alt>", "shift": "<shift>", "win": "<cmd>"}
        parts = [names[name] for name in ("ctrl", "alt", "shift", "win") if name in self.modifiers]
        key = self.key if len(self.key) == 1 else f"<{self.key}>"
        return "+".join([*parts, key])


def parse_hotkey(spec: str, *, require_modifier: bool = True) -> Hotkey:
    """แปลง ``"<ctrl>+<alt>+g"`` เป็น :class:`Hotkey` (raise ``ValueError`` ถ้ารูปแบบผิด)

    ต้องมีปุ่มหลักหนึ่งปุ่ม และ (ถ้า ``require_modifier``) modifier อย่างน้อยหนึ่งตัว ปุ่มหลักเป็น
    ตัวอักษร a–z, ตัวเลข 0–9 หรือชื่อปุ่ม เช่น ``<f9>``, ``<esc>``, ``<space>``, ``left``
    วงเล็บแหลมใส่หรือไม่ใส่ก็ได้
    """
    modifiers: set[str] = set()
    key: str | None = None
    parts = [part.strip().lower() for part in spec.split("+")]
    if not spec.strip() or any(not part for part in parts):
        raise ValueError(f"รูปแบบ hotkey ไม่ถูกต้อง: {spec!r}")

    for part in parts:
        name = part[1:-1] if part.startswith("<") and part.endswith(">") else part
        if name in _MODIFIER_ALIASES:
            modifier = _MODIFIER_ALIASES[name]
            if modifier in modifiers:
                raise ValueError(f"hotkey {spec!r} มี {name} ซ้ำ")
            modifiers.add(modifier)
            continue
        is_char = len(name) == 1 and name.isascii() and name.isalnum()
        if not is_char and name not in _NAMED_KEYS:
            raise ValueError(f"hotkey {spec!r}: ไม่รู้จักปุ่ม {part!r}")
        if key is not None:
            raise ValueError(f"hotkey {spec!r} มีปุ่มหลักมากกว่าหนึ่งปุ่ม")
        key = name

    if key is None:
        raise ValueError(f"hotkey {spec!r} ไม่มีปุ่มหลัก")
    if require_modifier and not modifiers:
        raise ValueError(f"hotkey {spec!r} ต้องมี modifier อย่างน้อยหนึ่งตัว (ctrl/alt/shift/win)")
    return Hotkey(frozenset(modifiers), key)


# action ใน [actions] ของ config ที่ไม่ใช่การกดปุ่ม
BUILTIN_ACTIONS = ("minimize_foreground", "none")


def parse_action(spec: str) -> Hotkey | str:
    """แปลงค่าใน ``[actions]`` ของ config: ชื่อ action สำเร็จรูป หรือปุ่มลัดที่จะกดให้

    - ``"minimize_foreground"`` — ย่อหน้าต่างที่ active, ``"none"`` — ไม่ทำอะไร (ปิดท่านั้น)
    - อย่างอื่นถือเป็นปุ่มลัด เช่น ``"ctrl+win+left"``, ``"win+tab"``, ``"alt+f4"``, ``"f11"``,
      ``"media_play_pause"``, ``"volume_up"``
    """
    name = spec.strip().lower()
    if name in BUILTIN_ACTIONS:
        return name
    return parse_hotkey(spec, require_modifier=False)


class _HotkeyApi(Protocol):
    """สิ่งที่ loop ต้องใช้จาก OS (แยกไว้ให้ test ด้วยของปลอมได้)"""

    def prepare(self) -> None: ...

    def register(self, hotkey_id: int, hotkey: Hotkey) -> bool: ...

    def unregister(self, hotkey_id: int) -> None: ...

    def wait_hotkey(self) -> int | None:
        """block จนมี hotkey คืน id หรือ ``None`` เมื่อถูกสั่งหยุด"""
        ...

    def interrupt(self) -> None: ...


class _Win32Api:
    """``RegisterHotKey`` + message loop ของ thread (ต้องเรียก prepare/register/wait จาก thread เดียวกัน)"""

    def __init__(self) -> None:
        import ctypes
        from ctypes import wintypes

        self._ctypes = ctypes
        self._wintypes = wintypes
        user32 = ctypes.WinDLL("user32", use_last_error=True)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

        user32.RegisterHotKey.argtypes = (wintypes.HWND, ctypes.c_int, wintypes.UINT, wintypes.UINT)
        user32.RegisterHotKey.restype = wintypes.BOOL
        user32.UnregisterHotKey.argtypes = (wintypes.HWND, ctypes.c_int)
        user32.UnregisterHotKey.restype = wintypes.BOOL
        message_pointer = ctypes.POINTER(wintypes.MSG)
        user32.GetMessageW.argtypes = (message_pointer, wintypes.HWND, wintypes.UINT, wintypes.UINT)
        user32.GetMessageW.restype = ctypes.c_int  # > 0 ข้อความปกติ, 0 = WM_QUIT, -1 = error
        user32.PeekMessageW.argtypes = (
            message_pointer, wintypes.HWND, wintypes.UINT, wintypes.UINT, wintypes.UINT,
        )  # fmt: skip
        user32.PeekMessageW.restype = wintypes.BOOL
        user32.PostThreadMessageW.argtypes = (
            wintypes.DWORD, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM,
        )  # fmt: skip
        user32.PostThreadMessageW.restype = wintypes.BOOL
        kernel32.GetCurrentThreadId.argtypes = ()
        kernel32.GetCurrentThreadId.restype = wintypes.DWORD

        self._user32 = user32
        self._kernel32 = kernel32
        self._thread_id = 0

    def prepare(self) -> None:
        message = self._wintypes.MSG()
        # PeekMessage ครั้งแรกทำให้ thread นี้มี message queue (จำเป็นก่อน PostThreadMessage)
        self._user32.PeekMessageW(self._ctypes.byref(message), None, 0, 0, 0)
        self._thread_id = self._kernel32.GetCurrentThreadId()

    def register(self, hotkey_id: int, hotkey: Hotkey) -> bool:
        flags = hotkey.win32_modifiers | _MOD_NOREPEAT
        return bool(self._user32.RegisterHotKey(None, hotkey_id, flags, hotkey.vk))

    def unregister(self, hotkey_id: int) -> None:
        self._user32.UnregisterHotKey(None, hotkey_id)

    def wait_hotkey(self) -> int | None:
        message = self._wintypes.MSG()
        while self._user32.GetMessageW(self._ctypes.byref(message), None, 0, 0) > 0:
            if message.message == _WM_HOTKEY:
                return int(message.wParam)
        return None

    def interrupt(self) -> None:
        if self._thread_id:
            self._user32.PostThreadMessageW(self._thread_id, _WM_QUIT, 0, 0)


class _HotkeyThread(threading.Thread):
    """ลงทะเบียน hotkey แล้ววนรอใน thread ของตัวเอง"""

    def __init__(self, api: _HotkeyApi, bindings: list[tuple[Hotkey, Callable[[], None]]]) -> None:
        super().__init__(name="aether-hotkeys", daemon=True)
        self._api = api
        self._bindings = bindings
        self._ready = threading.Event()
        self.failed: list[Hotkey] = []

    def wait_ready(self, timeout: float) -> bool:
        return self._ready.wait(timeout)

    def run(self) -> None:
        registered: list[int] = []
        try:
            self._api.prepare()
            for hotkey_id, (hotkey, _) in enumerate(self._bindings, start=1):
                if self._api.register(hotkey_id, hotkey):
                    registered.append(hotkey_id)
                else:
                    self.failed.append(hotkey)
        finally:
            self._ready.set()
        try:
            while (hotkey_id := self._api.wait_hotkey()) is not None:
                if 1 <= hotkey_id <= len(self._bindings):
                    _call_safely(self._bindings[hotkey_id - 1][1])
        finally:
            for hotkey_id in registered:
                self._api.unregister(hotkey_id)

    def stop(self) -> None:
        self._ready.wait(2.0)
        self._api.interrupt()
        self.join(2.0)


def _call_safely(callback: Callable[[], None]) -> None:
    try:
        callback()
    except Exception:
        log.exception("hotkey callback failed")


class HotkeyManager:
    """ลงทะเบียน global hotkey ตาม ``{spec: callback}``

    :meth:`start` คืนรายการ hotkey ที่ลงทะเบียน **ไม่ได้** (เช่นมีโปรแกรมอื่นใช้อยู่แล้ว)
    """

    def __init__(
        self, bindings: Mapping[str, Callable[[], None]], *, api: _HotkeyApi | None = None
    ) -> None:
        self._bindings = [(parse_hotkey(spec), callback) for spec, callback in bindings.items()]
        self._api = api
        self._thread: _HotkeyThread | None = None
        self._pynput_listener = None

    def start(self) -> list[Hotkey]:
        if self._thread is not None or self._pynput_listener is not None:
            return []
        if self._api is None and sys.platform != "win32":
            return self._start_pynput()
        thread = _HotkeyThread(self._api or _Win32Api(), self._bindings)
        thread.start()
        if not thread.wait_ready(2.0):
            log.error("hotkey thread did not start")
        self._thread = thread
        return list(thread.failed)

    def stop(self) -> None:
        if self._thread is not None:
            self._thread.stop()
            self._thread = None
        if self._pynput_listener is not None:
            self._pynput_listener.stop()
            self._pynput_listener = None

    def _start_pynput(self) -> list[Hotkey]:
        """นอก Windows (เครื่อง dev): ใช้ pynput ถ้าโหลดได้ ไม่งั้นถือว่าลงทะเบียนไม่ได้ทั้งหมด"""
        try:
            from pynput import keyboard

            listener = keyboard.GlobalHotKeys(
                {
                    hotkey.to_pynput(): (lambda cb=callback: _call_safely(cb))
                    for hotkey, callback in self._bindings
                }
            )
            listener.start()
        except Exception as exc:  # noqa: BLE001 — ไม่มีจอ / platform ไม่รองรับ
            log.warning("global hotkeys unavailable on this platform: %s", exc)
            return [hotkey for hotkey, _ in self._bindings]
        self._pynput_listener = listener
        return []
