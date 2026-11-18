"""``GestureEvent`` → input ของ OS ผ่าน pynput + ctypes (Win32) — มีโหมด dry-run

:class:`ActionExecutor` จำว่าปุ่มเมาส์/คีย์ไหนกดค้างอยู่ และ :meth:`ActionExecutor.release_all`
ปล่อยทุกปุ่มได้เสมอ (ใช้ตอน disarm, ปิดแอป, pipeline พัง)

ท่าครั้งเดียว (``ACTION``) ถูกแปลงตาม ``[actions]`` ของ config: กดปุ่มลัด (เช่น ``ctrl+win+left``)
หรือ action สำเร็จรูป (``minimize_foreground``) — เปลี่ยน shortcut ได้โดยไม่ต้องแก้โค้ด

ไม่มีการขอสิทธิ์ admin: หน้าต่างที่รันแบบ Administrator จะไม่รับ input จากแอปนี้ (Windows UIPI)
"""

from __future__ import annotations

import logging
import math
import os
import subprocess
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Protocol

from .config import LauncherItem
from .gestures import ACTION_LAUNCH_PREFIX, EventKind, GestureEvent
from .hotkeys import Hotkey, parse_action

__all__ = [
    "PROTECTED_WINDOW_CLASSES",
    "ActionExecutor",
    "DryRunBackend",
    "InputBackend",
    "PynputBackend",
    "create_backend",
    "double_click_time_ms",
    "foreground_minimizable_rect",
    "foreground_window",
    "glide_offset",
    "keep_on_screen",
    "launch_target",
    "minimize_foreground_window",
    "should_minimize",
    "throw_destination",
]

log = logging.getLogger(__name__)

BUTTONS = ("left", "right", "middle")

# หน้าต่างของ shell ที่ห้ามย่อ: desktop (Progman, WorkerW) และ taskbar
PROTECTED_WINDOW_CLASSES = frozenset(
    {"Progman", "WorkerW", "Shell_TrayWnd", "Shell_SecondaryTrayWnd"}
)
_SW_MINIMIZE = 6
_SW_RESTORE = 9
_GA_ROOT = 2
_SWP_MOVE_ONLY = 0x0001 | 0x0004 | 0x0010  # SWP_NOSIZE | SWP_NOZORDER | SWP_NOACTIVATE

THROW_GLIDE_S = 0.25  # หน้าต่างที่ถูกโยนไหลต่อเท่ากับความเร็วตอนปล่อย × เวลานี้
THROW_KEEP_VISIBLE = 120  # หลังโยนต้องเหลือส่วนของหน้าต่างบนจออย่างน้อยเท่านี้ (px) ทั้งสองแกน
THROW_MONITOR_SPEED = 3500.0  # โยนเร็วกว่านี้ (px/s) ไปทางจอข้างๆ = ย้ายหน้าต่างไปจอนั้น
THROW_AXIS_RATIO = 1.5  # แกนหลักของการโยนต้องเร็วกว่าแกนรองอย่างน้อยเท่านี้จึงนับว่า "ไปทางจอนั้น"

Rect = tuple[int, int, int, int]  # left, top, right, bottom (physical px)


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

    def launch(self, target: str, args: str) -> None:
        """เปิดโปรแกรม / ไฟล์ / โฟลเดอร์ / URL (รายการใน [launcher])"""
        ...

    def grab_window(self, x: int, y: int) -> int | None:
        """หยิบหน้าต่างหลักที่อยู่ใต้จุด (x, y) คืน handle หรือ ``None`` ถ้าหยิบไม่ได้"""
        ...

    def move_window(self, handle: int, dx: int, dy: int) -> None: ...

    def throw_window(self, handle: int, vx: int, vy: int) -> None:
        """ปล่อยหน้าต่างที่ความเร็ว (vx, vy) px/s — หน้าต่างไหลต่อแต่ยังเหลือบนจอ"""
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

    def launch(self, target: str, args: str) -> None:
        log.info("[dry-run] launch %s %s", target, args)

    def grab_window(self, x: int, y: int) -> int | None:
        log.info("[dry-run] grab window at (%d, %d)", x, y)
        return 0  # ทำเหมือนหยิบได้ เพื่อให้เห็น log ของการลาก/โยนต่อ

    def move_window(self, handle: int, dx: int, dy: int) -> None:
        log.debug("[dry-run] move window %+d %+d", dx, dy)

    def throw_window(self, handle: int, vx: int, vy: int) -> None:
        if math.hypot(vx, vy) >= THROW_MONITOR_SPEED:
            log.info("[dry-run] throw window at %+d %+d px/s (ไปจอข้างๆ ถ้ามี)", vx, vy)
        else:
            log.info("[dry-run] throw window at %+d %+d px/s", vx, vy)


def launch_target(target: str, args: str = "") -> None:
    """เปิดสิ่งที่ระบุด้วยโปรแกรมเริ่มต้นของ OS (เหมือนพิมพ์ในกล่อง Run ของ Windows)

    Windows: ``os.startfile`` (ShellExecute — ใช้ได้ทั้ง notepad.exe, โฟลเดอร์, ไฟล์, URL)
    ระบบอื่น: ``xdg-open`` / ``open`` (args ใช้ได้เฉพาะ Windows)
    """
    if sys.platform == "win32":
        os.startfile(target, arguments=args) if args else os.startfile(target)
        return
    opener = "open" if sys.platform == "darwin" else "xdg-open"
    subprocess.Popen([opener, target], start_new_session=True)


def glide_offset(vx: float, vy: float) -> tuple[int, int]:
    """ระยะที่หน้าต่างไหลต่อหลังโยน (px)"""
    return (round(vx * THROW_GLIDE_S), round(vy * THROW_GLIDE_S))


def throw_destination(
    rect: Rect, vx: float, vy: float, monitors: Sequence[Rect]
) -> tuple[int, int] | None:
    """โยนแรงไปทางจอข้างๆ → ระยะ (dx, dy) ที่ย้ายหน้าต่างไปจอนั้น (ตำแหน่งสัมพัทธ์บนจอเท่าเดิม)

    คืน ``None`` ถ้าโยนไม่แรงพอ ทิศไม่ชัด หรือไม่มีจอทางนั้น (ใช้การไหลต่อแบบเดิมแทน)
    ``rect`` และ ``monitors`` เป็น (left, top, right, bottom) — เลือกจอที่อยู่ทางนั้นและใกล้ที่สุด
    """
    speed = math.hypot(vx, vy)
    if speed < THROW_MONITOR_SPEED or len(monitors) < 2:
        return None
    if abs(vx) >= abs(vy) * THROW_AXIS_RATIO:
        axis = 0
    elif abs(vy) >= abs(vx) * THROW_AXIS_RATIO:
        axis = 1
    else:
        return None
    sign = 1 if (vx if axis == 0 else vy) > 0 else -1
    left, top, right, bottom = rect

    def overlap(a: Rect) -> int:
        w = min(a[2], right) - max(a[0], left)
        h = min(a[3], bottom) - max(a[1], top)
        return max(w, 0) * max(h, 0)

    current = max(monitors, key=overlap)
    if overlap(current) == 0:
        return None
    ccx, ccy = (current[0] + current[2]) / 2, (current[1] + current[3]) / 2
    candidates = []
    for m in monitors:
        if m == current:
            continue
        mcx, mcy = (m[0] + m[2]) / 2, (m[1] + m[3]) / 2
        along = (mcx - ccx) if axis == 0 else (mcy - ccy)
        if along * sign <= 0:
            continue
        # จอต้องอยู่ "ข้างๆ" จริง: ช่วงของแกนรองซ้อนกับจอปัจจุบัน
        lo, hi = (1, 3) if axis == 0 else (0, 2)
        if min(m[hi], current[hi]) - max(m[lo], current[lo]) <= 0:
            continue
        candidates.append((abs(along), m))
    if not candidates:
        return None
    target = min(candidates, key=lambda c: c[0])[1]
    width, height = right - left, bottom - top
    cur_w, cur_h = current[2] - current[0], current[3] - current[1]
    t_w, t_h = target[2] - target[0], target[3] - target[1]
    rel_x = (left - current[0]) / max(cur_w - width, 1) if cur_w > width else 0.0
    rel_y = (top - current[1]) / max(cur_h - height, 1) if cur_h > height else 0.0
    rel_x, rel_y = min(max(rel_x, 0.0), 1.0), min(max(rel_y, 0.0), 1.0)
    new_left = target[0] + round(rel_x * max(t_w - width, 0))
    new_top = target[1] + round(rel_y * max(t_h - height, 0))
    return (new_left - left, new_top - top)


def keep_on_screen(
    rect: tuple[int, int, int, int],
    dx: int,
    dy: int,
    bounds: tuple[int, int, int, int],
    margin: int = THROW_KEEP_VISIBLE,
) -> tuple[int, int]:
    """ลดระยะ (dx, dy) ให้หน้าต่าง ``rect`` = (left, top, right, bottom) ยังเหลือบนจออย่างน้อย ``margin`` px

    ``bounds`` = (left, top, right, bottom) ของจอทั้งหมด ขอบบนห้ามเลยจอขึ้นไป (จะจับแถบชื่อไม่ได้)
    """
    left, top, right, _bottom = rect
    b_left, b_top, b_right, b_bottom = bounds
    new_left = min(max(left + dx, b_left - (right - left) + margin), b_right - margin)
    new_top = min(max(top + dy, b_top), b_bottom - margin)
    return (new_left - left, new_top - top)


def _win32_window_api():
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    user32.WindowFromPoint.argtypes = (wintypes.POINT,)
    user32.WindowFromPoint.restype = wintypes.HWND
    user32.GetAncestor.argtypes = (wintypes.HWND, wintypes.UINT)
    user32.GetAncestor.restype = wintypes.HWND
    user32.GetWindowThreadProcessId.argtypes = (wintypes.HWND, ctypes.POINTER(wintypes.DWORD))
    user32.GetWindowThreadProcessId.restype = wintypes.DWORD
    user32.GetClassNameW.argtypes = (wintypes.HWND, wintypes.LPWSTR, ctypes.c_int)
    user32.GetClassNameW.restype = ctypes.c_int
    user32.IsZoomed.argtypes = (wintypes.HWND,)
    user32.IsZoomed.restype = wintypes.BOOL
    user32.ShowWindow.argtypes = (wintypes.HWND, ctypes.c_int)
    user32.ShowWindow.restype = wintypes.BOOL
    user32.GetWindowRect.argtypes = (wintypes.HWND, ctypes.POINTER(wintypes.RECT))
    user32.GetWindowRect.restype = wintypes.BOOL
    user32.SetWindowPos.argtypes = (
        wintypes.HWND, wintypes.HWND, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        wintypes.UINT,
    )  # fmt: skip
    user32.SetWindowPos.restype = wintypes.BOOL
    return ctypes, wintypes, user32


def win32_root_window_at(x: int, y: int) -> int | None:
    """หน้าต่างหลัก (root) ที่อยู่บนสุดตรงจุด (x, y)"""
    if sys.platform != "win32":
        return None
    _, wintypes, user32 = _win32_window_api()
    child = user32.WindowFromPoint(wintypes.POINT(x, y))
    window = user32.GetAncestor(child, _GA_ROOT) if child else None
    return int(window) if window else None


def win32_window_info(handle: int) -> tuple[int, str]:
    """(process id เจ้าของหน้าต่าง, ชื่อ class)"""
    ctypes, wintypes, user32 = _win32_window_api()
    pid = wintypes.DWORD()
    user32.GetWindowThreadProcessId(handle, ctypes.byref(pid))
    buffer = ctypes.create_unicode_buffer(256)
    user32.GetClassNameW(handle, buffer, len(buffer))
    return pid.value, buffer.value


def win32_grab_window(x: int, y: int, *, allow_own: bool = False) -> int | None:
    """หน้าต่างหลัก (root) ใต้จุด (x, y) ที่หยิบได้ — ไม่รวม desktop/taskbar และหน้าต่างของ AETHER เอง"""
    window = win32_root_window_at(x, y)
    if window is None:
        return None
    pid, window_class = win32_window_info(window)
    if pid == os.getpid() and not allow_own:
        return None  # HUD / หน้าต่าง debug ของเราเอง
    if not should_minimize(window_class):  # desktop / taskbar
        return None
    _, _, user32 = _win32_window_api()
    if user32.IsZoomed(window):
        user32.ShowWindow(window, _SW_RESTORE)  # ขยายเต็มจออยู่: คืนขนาดปกติก่อนลาก (เหมือนลากด้วยเมาส์)
    return int(window)


def win32_window_rect(handle: int) -> tuple[int, int, int, int] | None:
    if sys.platform != "win32":
        return None
    ctypes, wintypes, user32 = _win32_window_api()
    rect = wintypes.RECT()
    if not user32.GetWindowRect(handle, ctypes.byref(rect)):
        return None
    return (rect.left, rect.top, rect.right, rect.bottom)


def win32_move_window(handle: int, dx: int, dy: int) -> bool:
    rect = win32_window_rect(handle)
    if rect is None:
        return False
    _, _, user32 = _win32_window_api()
    return bool(user32.SetWindowPos(handle, None, rect[0] + dx, rect[1] + dy, 0, 0, _SWP_MOVE_ONLY))


def win32_virtual_screen() -> tuple[int, int, int, int]:
    """(left, top, right, bottom) ของทุกจอรวมกัน (physical px)"""
    import ctypes

    metrics = ctypes.windll.user32.GetSystemMetrics
    left, top = metrics(76), metrics(77)  # SM_XVIRTUALSCREEN, SM_YVIRTUALSCREEN
    return (
        left,
        top,
        left + metrics(78),
        top + metrics(79),
    )  # SM_CXVIRTUALSCREEN, SM_CYVIRTUALSCREEN


def _monitor_rects() -> list[Rect]:
    """(left, top, right, bottom) ของทุกจอ (physical px) — ว่างถ้าอ่านไม่ได้"""
    try:
        from .mapping import list_monitors

        return [
            (g.left, g.top, g.left + g.width, g.top + g.height)
            for g in (m.geometry for m in list_monitors())
        ]
    except Exception:  # noqa: BLE001 — โยนแบบเดิม (ไหลต่อ) ได้เสมอ
        return []


def should_minimize(window_class: str) -> bool:
    """หน้าต่าง class นี้ย่อได้หรือไม่ (desktop และ taskbar ห้ามย่อ)"""
    return bool(window_class) and window_class not in PROTECTED_WINDOW_CLASSES


def _user32():
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    user32.GetForegroundWindow.argtypes = ()
    user32.GetForegroundWindow.restype = wintypes.HWND
    user32.GetClassNameW.argtypes = (wintypes.HWND, wintypes.LPWSTR, ctypes.c_int)
    user32.GetClassNameW.restype = ctypes.c_int
    user32.ShowWindow.argtypes = (wintypes.HWND, ctypes.c_int)
    user32.ShowWindow.restype = wintypes.BOOL
    return user32


def foreground_window() -> tuple[int, str] | None:
    """(handle, ชื่อ class) ของหน้าต่างที่ active อยู่ — ``None`` ถ้าไม่มี หรือไม่ใช่ Windows"""
    if sys.platform != "win32":
        return None
    import ctypes

    user32 = _user32()
    window = user32.GetForegroundWindow()
    if not window:
        return None
    buffer = ctypes.create_unicode_buffer(256)
    user32.GetClassNameW(window, buffer, len(buffer))
    return int(window), buffer.value


def minimize_foreground_window() -> bool:
    """``ShowWindow(GetForegroundWindow(), SW_MINIMIZE)`` — เฉพาะ Windows คืนว่าได้ย่อหรือไม่"""
    if sys.platform != "win32":
        log.info("minimize_foreground is only supported on Windows")
        return False
    target = foreground_window()
    if target is None:
        return False
    window, window_class = target
    if not should_minimize(window_class):
        log.debug("not minimizing shell window %r", window_class)
        return False
    _user32().ShowWindow(window, _SW_MINIMIZE)
    return True


def foreground_minimizable_rect() -> tuple[int, int, int, int] | None:
    """สี่เหลี่ยม (left, top, right, bottom) ของหน้าต่าง active ที่ย่อได้ — อ่านอย่างเดียว ไม่ย่อ

    HUD ใช้วาดกรอบยุบลง taskbar ตอนย่อหน้าต่าง (ต้องอ่านก่อนย่อ หลังย่อแล้วตำแหน่งจะเป็น -32000)
    """
    target = foreground_window()
    if target is None or not should_minimize(target[1]):
        return None
    return win32_window_rect(target[0])


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

    def launch(self, target: str, args: str) -> None:
        launch_target(target, args)

    def grab_window(self, x: int, y: int) -> int | None:
        if sys.platform != "win32":
            log.info("window grabbing is only supported on Windows")
        return win32_grab_window(x, y)

    def move_window(self, handle: int, dx: int, dy: int) -> None:
        win32_move_window(handle, dx, dy)

    def throw_window(self, handle: int, vx: int, vy: int) -> None:
        rect = win32_window_rect(handle)
        if rect is None:
            return
        monitors = _monitor_rects()
        jump = throw_destination(rect, vx, vy, monitors)
        if jump is not None:
            log.info("โยนหน้าต่างไปจอข้างๆ (%+d, %+d px)", *jump)
            win32_move_window(handle, *jump)
            return
        dx, dy = keep_on_screen(rect, *glide_offset(vx, vy), win32_virtual_screen())
        win32_move_window(handle, dx, dy)

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

    def __init__(
        self,
        backend: InputBackend,
        actions: Mapping[str, str] | None = None,
        launcher: Sequence[LauncherItem] = (),
    ) -> None:
        self._backend = backend
        self._launcher = tuple(launcher)
        self._grabbed: int | None = None  # หน้าต่างที่ถืออยู่ (✊ ลาก)
        # กรอบของหน้าต่างที่เพิ่งสั่งย่อ (อ่านก่อนย่อ) — HUD ใช้วาดกรอบยุบลง taskbar (M7)
        self.last_minimized_rect: tuple[int, int, int, int] | None = None
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

    @property
    def grabbed_window(self) -> int | None:
        return self._grabbed

    def release_all(self) -> None:
        """ปล่อยทุกปุ่มเมาส์และคีย์ที่กดค้าง และวางหน้าต่างที่ถืออยู่ (เรียกซ้ำได้ ปลอดภัยเสมอ)"""
        for button in sorted(self._pressed):
            self._release(button)
        self._release_keys()
        self._grabbed = None

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
        elif kind is EventKind.WINDOW_GRAB:
            self._grabbed = self._backend.grab_window(event.x, event.y)
            if self._grabbed is None:
                log.info("ไม่มีหน้าต่างที่หยิบได้ตรง (%d, %d)", event.x, event.y)
        elif kind is EventKind.WINDOW_MOVE:
            if self._grabbed is not None:
                self._backend.move_window(self._grabbed, event.dx, event.dy)
        elif kind is EventKind.WINDOW_RELEASE:
            handle, self._grabbed = self._grabbed, None
            if handle is not None and (event.dx or event.dy):
                self._backend.throw_window(handle, event.dx, event.dy)
        elif kind is EventKind.DISARMED:
            self.release_all()  # กันพลาดอีกชั้น: disarm แล้วต้องไม่มีปุ่มค้าง
        # ARMED: แจ้งสถานะเท่านั้น

    def _run_action(self, name: str) -> None:
        if name.startswith(ACTION_LAUNCH_PREFIX):
            index = name.removeprefix(ACTION_LAUNCH_PREFIX)
            if index.isdigit() and int(index) < len(self._launcher):
                item = self._launcher[int(index)]
                log.info("launch %s → %s %s", item.label, item.run, item.args)
                self._backend.launch(item.run, item.args)
            return
        action = self._actions.get(name)
        if action is None or action == "none":
            log.debug("no action bound to %s", name)
        elif action == "minimize_foreground":
            try:
                self.last_minimized_rect = foreground_minimizable_rect()
            except Exception:  # noqa: BLE001 — แค่ใช้วาด effect: อ่านไม่ได้ก็ย่อต่อได้
                self.last_minimized_rect = None
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
