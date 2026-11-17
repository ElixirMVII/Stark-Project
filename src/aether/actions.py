"""``GestureEvent`` → input ของ OS ผ่าน pynput + ctypes (Win32) — มีโหมด dry-run

:class:`ActionExecutor` จำว่าปุ่มเมาส์/คีย์ไหนกดค้างอยู่ และ :meth:`ActionExecutor.release_all`
ปล่อยทุกปุ่มได้เสมอ (ใช้ตอน disarm, ปิดแอป, pipeline พัง)

ท่าครั้งเดียว (``ACTION``) ถูกแปลงตาม ``[actions]`` ของ config: กดปุ่มลัด (เช่น ``ctrl+win+left``)
หรือ action สำเร็จรูป (``minimize_foreground``) — เปลี่ยน shortcut ได้โดยไม่ต้องแก้โค้ด

ไม่มีการขอสิทธิ์ admin: หน้าต่างที่รันแบบ Administrator จะไม่รับ input จากแอปนี้ (Windows UIPI)
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Protocol

from .gestures import EventKind, GestureEvent
from .hotkeys import Hotkey, parse_action

__all__ = [
    "PROTECTED_WINDOW_CLASSES",
    "ActionExecutor",
    "DryRunBackend",
    "InputBackend",
    "PynputBackend",
    "create_backend",
    "double_click_time_ms",
    "minimize_foreground_window",
    "should_minimize",
]

log = logging.getLogger(__name__)

BUTTONS = ("left", "right", "middle")

# หน้าต่างของ shell ที่ห้ามย่อ: desktop (Progman, WorkerW) และ taskbar
PROTECTED_WINDOW_CLASSES = frozenset(
    {"Progman", "WorkerW", "Shell_TrayWnd", "Shell_SecondaryTrayWnd"}
)
_SW_MINIMIZE = 6


class InputBackend(Protocol):
    """ตัวส่ง input จริง

    พิกัดเป็น physical px, scroll เป็น notch (dy > 0 = ขึ้น, dx > 0 = ขวา), ชื่อปุ่มคีย์บอร์ดเป็นแบบ
    เดียวกับ :class:`aether.hotkeys.Hotkey` (``ctrl``, ``alt``, ``shift``, ``win``, ``left``, ``d`` …)
    """

    def move(self, x: int, y: int) -> None: ...

    def press(self, button: str) -> None: ...

    def release(self, button: str) -> None: ...

    def scroll(self, dx: int, dy: int) -> None: ...

    def key_down(self, key: str) -> None: ...

    def key_up(self, key: str) -> None: ...

    def minimize_foreground(self) -> bool:
        """ย่อหน้าต่างที่ active คืน ``False`` ถ้าไม่ได้ย่อ (เป็น desktop/taskbar หรือไม่มีหน้าต่าง)"""
        ...


class DryRunBackend:
    """ไม่ส่ง input จริง แค่ log ว่าจะทำอะไร (``--dry-run``)"""

    def move(self, x: int, y: int) -> None:
        log.debug("[dry-run] move → (%d, %d)", x, y)

    def press(self, button: str) -> None:
        log.info("[dry-run] %s button DOWN", button)

    def release(self, button: str) -> None:
        log.info("[dry-run] %s button UP", button)

    def scroll(self, dx: int, dy: int) -> None:
        log.info("[dry-run] scroll dx=%+d dy=%+d", dx, dy)

    def key_down(self, key: str) -> None:
        log.info("[dry-run] key %s DOWN", key)

    def key_up(self, key: str) -> None:
        log.info("[dry-run] key %s UP", key)

    def minimize_foreground(self) -> bool:
        log.info("[dry-run] minimize foreground window")
        return False


def should_minimize(window_class: str) -> bool:
    """หน้าต่าง class นี้ย่อได้หรือไม่ (desktop และ taskbar ห้ามย่อ)"""
    return bool(window_class) and window_class not in PROTECTED_WINDOW_CLASSES


def minimize_foreground_window() -> bool:
    """``ShowWindow(GetForegroundWindow(), SW_MINIMIZE)`` — เฉพาะ Windows คืนว่าได้ย่อหรือไม่"""
    if sys.platform != "win32":
        log.info("minimize_foreground is only supported on Windows")
        return False
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    user32.GetForegroundWindow.argtypes = ()
    user32.GetForegroundWindow.restype = wintypes.HWND
    user32.GetClassNameW.argtypes = (wintypes.HWND, wintypes.LPWSTR, ctypes.c_int)
    user32.GetClassNameW.restype = ctypes.c_int
    user32.ShowWindow.argtypes = (wintypes.HWND, ctypes.c_int)
    user32.ShowWindow.restype = wintypes.BOOL

    window = user32.GetForegroundWindow()
    if not window:
        return False
    buffer = ctypes.create_unicode_buffer(256)
    user32.GetClassNameW(window, buffer, len(buffer))
    if not should_minimize(buffer.value):
        log.debug("not minimizing shell window %r", buffer.value)
        return False
    user32.ShowWindow(window, _SW_MINIMIZE)
    return True


class PynputBackend:
    """เมาส์/คีย์บอร์ดจริงผ่าน pynput (บน Windows คือ SetCursorPos + SendInput)"""

    def __init__(self) -> None:
        from pynput import keyboard, mouse

        self._mouse = mouse.Controller()
        self._buttons = {
            "left": mouse.Button.left,
            "right": mouse.Button.right,
            "middle": mouse.Button.middle,
        }
        self._keyboard_module = keyboard
        self._keyboard = keyboard.Controller()
        self._modifiers = {
            "ctrl": keyboard.Key.ctrl,
            "alt": keyboard.Key.alt,
            "shift": keyboard.Key.shift,
            "win": keyboard.Key.cmd,
        }

    def move(self, x: int, y: int) -> None:
        self._mouse.position = (x, y)

    def press(self, button: str) -> None:
        self._mouse.press(self._buttons[button])

    def release(self, button: str) -> None:
        self._mouse.release(self._buttons[button])

    def scroll(self, dx: int, dy: int) -> None:
        self._mouse.scroll(dx, dy)

    def key_down(self, key: str) -> None:
        self._keyboard.press(self._key(key))

    def key_up(self, key: str) -> None:
        self._keyboard.release(self._key(key))

    def minimize_foreground(self) -> bool:
        return minimize_foreground_window()

    def _key(self, name: str):
        """ชื่อปุ่ม → key ของ pynput

        ตัวอักษร/ตัวเลขส่งเป็น virtual-key code (ไม่ใช่ตัวอักษร) เพื่อให้ปุ่มลัดอย่าง ``Win+D`` ทำงาน
        แม้จะเปิดแป้นไทยอยู่ — ถ้าส่งเป็นตัวอักษร pynput จะพิมพ์เป็น Unicode ซึ่งไม่นับเป็นปุ่มลัด
        """
        keyboard = self._keyboard_module
        if name in self._modifiers:
            return self._modifiers[name]
        if len(name) == 1:
            if sys.platform == "win32":
                return keyboard.KeyCode.from_vk(ord(name.upper()))
            return keyboard.KeyCode.from_char(name)
        try:
            return keyboard.Key[name]  # ปุ่มที่มีชื่อ (ลูกศร, tab, esc, f1–f20 …) ได้ flag ถูกต้องจาก pynput
        except KeyError:
            return keyboard.KeyCode.from_vk(Hotkey(frozenset(), name).vk)


def create_backend(dry_run: bool) -> tuple[InputBackend, bool]:
    """สร้าง backend คืน (backend, เป็น dry-run จริงหรือไม่)

    ถ้าโหลด pynput ไม่ได้ (เช่นเครื่อง dev ที่ไม่มีจอ) จะถอยไปใช้ dry-run พร้อมเตือน
    ดีกว่าให้แอปล้มหรือส่ง input แบบครึ่งๆ กลางๆ
    """
    if dry_run:
        return DryRunBackend(), True
    try:
        return PynputBackend(), False
    except Exception as exc:  # noqa: BLE001 — pynput โยนได้หลายชนิดตาม platform
        log.error("ใช้ pynput ไม่ได้ (%s) — สลับเป็น dry-run: จะไม่มีการส่ง input ให้ OS", exc)
        return DryRunBackend(), True


def double_click_time_ms(default: int = 500) -> int:
    """เวลาดับเบิลคลิกที่ผู้ใช้ตั้งไว้ใน OS (ms) — นอก Windows ใช้ค่า ``default``"""
    try:
        import ctypes

        value = int(ctypes.windll.user32.GetDoubleClickTime())
    except (AttributeError, OSError):
        return default
    return value if value > 0 else default


class ActionExecutor:
    """ทำตาม :class:`GestureEvent` ทีละตัวผ่าน backend

    ``actions`` คือ ``{ชื่อท่า: ค่าใน config}`` เช่น ``{"swipe_left": "ctrl+win+left"}``

    backend ที่โยน exception จะถูก log แล้วข้าม event นั้น (pipeline ต้องไม่ล้มเพราะ input ตัวเดียว)
    ยกเว้นการปล่อยปุ่ม ซึ่งจะถูกลองทุกครั้งและถือว่าปุ่มไม่ค้างแล้วเสมอ
    """

    def __init__(self, backend: InputBackend, actions: Mapping[str, str] | None = None) -> None:
        self._backend = backend
        self._pressed: set[str] = set()
        self._keys_down: list[str] = []
        self._actions: dict[str, Hotkey | str] = {
            name: parse_action(spec) for name, spec in (actions or {}).items()
        }

    @property
    def pressed(self) -> frozenset[str]:
        """ปุ่มเมาส์ที่แอปนี้กดค้างอยู่"""
        return frozenset(self._pressed)

    @property
    def keys_down(self) -> tuple[str, ...]:
        """ปุ่มคีย์บอร์ดที่แอปนี้กดค้างอยู่ (ปกติว่าง — ปุ่มลัดถูกกดแล้วปล่อยทันที)"""
        return tuple(self._keys_down)

    def execute(self, events: Iterable[GestureEvent]) -> None:
        for event in events:
            try:
                self._execute_one(event)
            except Exception:
                log.exception("input backend failed on %s", event.kind.value)

    def release_all(self) -> None:
        """ปล่อยทุกปุ่มเมาส์และคีย์ที่กดค้าง (เรียกซ้ำได้ ปลอดภัยเสมอ)"""
        for button in sorted(self._pressed):
            self._release(button)
        self._release_keys()

    def _execute_one(self, event: GestureEvent) -> None:
        kind = event.kind
        if kind is EventKind.MOVE:
            self._backend.move(event.x, event.y)
        elif kind is EventKind.BUTTON_DOWN:
            if event.button not in self._pressed:
                self._backend.move(event.x, event.y)
                self._pressed.add(event.button)
                self._backend.press(event.button)
        elif kind is EventKind.BUTTON_UP:
            if event.button in self._pressed:
                self._release(event.button)
        elif kind is EventKind.CLICK:
            self._backend.move(event.x, event.y)
            self._pressed.add(event.button)  # ถ้า press พังกลางทาง release_all ยังตามปล่อยให้ได้
            self._backend.press(event.button)
            self._release(event.button)
        elif kind is EventKind.SCROLL:
            self._backend.scroll(event.dx, event.dy)
        elif kind is EventKind.ZOOM:
            # Ctrl + ล้อเมาส์ = ซูมในเบราว์เซอร์/โปรแกรมส่วนใหญ่
            self._with_keys(("ctrl",), lambda: self._backend.scroll(0, event.dy))
        elif kind is EventKind.ACTION:
            self._run_action(event.name)
        elif kind is EventKind.DISARMED:
            self.release_all()  # กันพลาดอีกชั้น: disarm แล้วต้องไม่มีปุ่มค้าง
        # ARMED: แจ้งสถานะเท่านั้น

    def _run_action(self, name: str) -> None:
        action = self._actions.get(name)
        if action is None or action == "none":
            log.debug("no action bound to %s", name)
        elif action == "minimize_foreground":
            self._backend.minimize_foreground()
        elif isinstance(action, Hotkey):
            self._with_keys(action.keys, lambda: None)  # กดครบแล้วปล่อย = กดปุ่มลัดหนึ่งครั้ง

    def _with_keys(self, keys: Sequence[str], body: Callable[[], None]) -> None:
        """กดคีย์ตามลำดับ → ทำ ``body`` → ปล่อยย้อนลำดับ (ปล่อยเสมอ แม้ ``body`` จะพัง)"""
        try:
            for key in keys:
                self._keys_down.append(key)
                self._backend.key_down(key)
            body()
        finally:
            self._release_keys()

    def _release_keys(self) -> None:
        while self._keys_down:
            key = self._keys_down.pop()
            try:
                self._backend.key_up(key)
            except Exception:
                log.exception("failed to release key %s", key)

    def _release(self, button: str) -> None:
        self._pressed.discard(button)
        try:
            self._backend.release(button)
        except Exception:
            log.exception("failed to release %s button", button)
