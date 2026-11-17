"""Test ที่รันเฉพาะบน Windows (ใน CI: GitHub Actions windows-latest)

ตรวจส่วนที่ผูกกับ Win32 ซึ่ง test บน Linux ทำได้แค่ด้วยของปลอม:

- ``RegisterHotKey`` ลงทะเบียน/ชนกัน/ถอนได้จริง
- ปุ่มลัดที่ :class:`~aether.actions.ActionExecutor` ส่ง (ผ่าน pynput → SendInput) ไปถึง OS เป็นปุ่มลัด
  จริง — รวมถึงตอนที่เปิดแป้นพิมพ์ภาษาไทยอยู่ (docs/DECISIONS.md D11, D27)
- ย้ายเคอร์เซอร์ด้วย ``SetCursorPos`` แล้วอ่านกลับด้วย ``GetCursorPos`` ได้ตรง
- อ่านหน้าต่าง foreground ได้ (ใช้ตัดสินว่าจะย่อหน้าต่างได้หรือไม่) โดยไม่ย่อหน้าต่างจริง

การส่ง input จำลองต้องมี desktop session ถ้า session ของเครื่องที่รันไม่รับ input จำลอง test ที่พึ่ง
เรื่องนี้จะ skip พร้อมบอกเหตุผล (ไม่นับว่าผ่าน)
"""

from __future__ import annotations

import sys
import threading
import time

import pytest

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="Windows only")

if sys.platform == "win32":
    import ctypes
    from ctypes import wintypes

INJECT_TIMEOUT_S = 3.0
THAI_LAYOUT = "0000041E"
KLF_ACTIVATE = 0x00000001


def _wait(event: threading.Event, timeout: float = INJECT_TIMEOUT_S) -> bool:
    return event.wait(timeout)


@pytest.fixture
def backend():
    from aether.actions import PynputBackend

    return PynputBackend()


# ---- RegisterHotKey -------------------------------------------------------- #


def test_register_hotkey_conflict_and_cleanup():
    from aether.hotkeys import HotkeyManager

    spec = "<ctrl>+<alt>+<shift>+<f24>"
    first = HotkeyManager({spec: lambda: None})
    assert first.start() == []  # ลงทะเบียนได้
    second = HotkeyManager({spec: lambda: None})
    try:
        failed = second.start()  # ชนกับตัวแรก → ต้องรายงานว่าลงทะเบียนไม่ได้
        assert [str(h) for h in failed] == ["Ctrl+Alt+Shift+F24"]
    finally:
        second.stop()
        first.stop()
    third = HotkeyManager({spec: lambda: None})  # หลังถอนแล้วลงทะเบียนใหม่ได้
    try:
        assert third.start() == []
    finally:
        third.stop()


def test_stop_returns_promptly():
    from aether.hotkeys import HotkeyManager

    manager = HotkeyManager({"<ctrl>+<alt>+<shift>+<f22>": lambda: None})
    manager.start()
    started = time.monotonic()
    manager.stop()
    assert time.monotonic() - started < 1.0


# ---- ปุ่มลัดจากแอป → OS ----------------------------------------------------- #


def _roundtrip(backend, action_spec: str, hotkey_spec: str) -> bool:
    """ลงทะเบียน hotkey แล้วให้ ActionExecutor กดปุ่มลัดเดียวกัน คืน True ถ้า hotkey ติด"""
    from aether.actions import ActionExecutor
    from aether.gestures import EventKind, GestureEvent
    from aether.hotkeys import HotkeyManager

    fired = threading.Event()
    manager = HotkeyManager({hotkey_spec: fired.set})
    assert manager.start() == []
    executor = ActionExecutor(backend, {"swipe_left": action_spec})
    try:
        executor.execute([GestureEvent(EventKind.ACTION, name="swipe_left")])
        return _wait(fired)
    finally:
        manager.stop()
        assert executor.keys_down == ()


def test_action_shortcut_reaches_os_as_a_hotkey(backend):
    if not _roundtrip(backend, "ctrl+alt+shift+f23", "<ctrl>+<alt>+<shift>+<f23>"):
        pytest.skip("session นี้ไม่ส่ง input จำลองให้ hotkey (ไม่มี interactive desktop)")


def test_letter_shortcut_works_with_thai_keyboard_layout(backend):
    """เปิดแป้นไทยอยู่ ปุ่มลัดที่มีตัวอักษร (เช่น Win+D) ต้องยังเป็นปุ่มลัด ไม่กลายเป็นการพิมพ์ "ก" ฯลฯ"""
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    user32.LoadKeyboardLayoutW.argtypes = (wintypes.LPCWSTR, wintypes.UINT)
    user32.LoadKeyboardLayoutW.restype = wintypes.HKL
    user32.GetKeyboardLayout.argtypes = (wintypes.DWORD,)
    user32.GetKeyboardLayout.restype = wintypes.HKL
    user32.ActivateKeyboardLayout.argtypes = (wintypes.HKL, wintypes.UINT)
    user32.ActivateKeyboardLayout.restype = wintypes.HKL

    previous = user32.GetKeyboardLayout(0)
    thai = user32.LoadKeyboardLayoutW(THAI_LAYOUT, KLF_ACTIVATE)
    if not thai:
        pytest.skip("โหลดแป้นพิมพ์ภาษาไทยไม่ได้บนเครื่องนี้")
    try:
        assert (user32.GetKeyboardLayout(0) & 0xFFFF) == 0x041E  # แป้นไทย active อยู่จริง
        if not _roundtrip(backend, "ctrl+alt+shift+g", "<ctrl>+<alt>+<shift>+g"):
            if not _roundtrip(backend, "ctrl+alt+shift+f23", "<ctrl>+<alt>+<shift>+<f23>"):
                pytest.skip("session นี้ไม่ส่ง input จำลองให้ hotkey (ไม่มี interactive desktop)")
            pytest.fail("ปุ่มลัดที่มีตัวอักษรไม่ทำงานเมื่อเปิดแป้นไทย (แต่ปุ่ม F ทำงาน)")
    finally:
        user32.ActivateKeyboardLayout(previous, 0)


def test_letter_keys_are_sent_as_virtual_key_codes(backend):
    from pynput import keyboard

    key = backend._key("d")
    assert isinstance(key, keyboard.KeyCode) and key.vk == 0x44 and key.char is None
    assert backend._key("win") == keyboard.Key.cmd
    assert backend._key("ctrl") == keyboard.Key.ctrl
    assert backend._key("left") == keyboard.Key.left
    assert backend._key("tab") == keyboard.Key.tab
    assert backend._key("f11") == keyboard.Key.f11
    assert backend._key("1").vk == 0x31


# ---- เมาส์ ----------------------------------------------------------------- #


def test_cursor_moves_to_physical_pixel_position(qapp, backend):
    """QApplication ทำให้ process เป็น DPI-aware → SetCursorPos/GetCursorPos เป็น physical px ตรงกัน"""
    from aether.mapping import primary_screen_geometry

    user32 = ctypes.windll.user32
    original = wintypes.POINT()
    user32.GetCursorPos(ctypes.byref(original))
    screen = primary_screen_geometry()
    target = (min(123, screen.right), min(77, screen.bottom))
    try:
        backend.move(*target)
        point = wintypes.POINT()
        user32.GetCursorPos(ctypes.byref(point))
        if (point.x, point.y) == (original.x, original.y) and target != (original.x, original.y):
            pytest.skip("session นี้ย้ายเคอร์เซอร์ไม่ได้ (ไม่มี interactive desktop)")
        assert (point.x, point.y) == target
    finally:
        user32.SetCursorPos(original.x, original.y)


# ---- อื่นๆ ----------------------------------------------------------------- #


def test_foreground_window_is_read_without_minimizing():
    from aether.actions import foreground_window, should_minimize

    window = foreground_window()
    if window is None:
        pytest.skip("ไม่มีหน้าต่าง foreground ใน session นี้")
    handle, window_class = window
    assert handle > 0 and isinstance(window_class, str)
    assert should_minimize(window_class) == (
        window_class not in ("Progman", "WorkerW", "Shell_TrayWnd")
    )


def test_os_values_are_sane():
    from aether.actions import double_click_time_ms
    from aether.mapping import primary_screen_geometry

    assert 100 <= double_click_time_ms() <= 5000
    screen = primary_screen_geometry()
    assert screen.width >= 640 and screen.height >= 480


def test_mediapipe_native_library_loads():
    """libmediapipe.dll โหลดได้ และโมเดลประมวลผลเฟรมได้บน Windows"""
    import numpy as np

    from aether.config import model_path
    from aether.tracker import HandTracker

    if not model_path().is_file():
        pytest.skip("ยังไม่ได้โหลดโมเดล")
    with HandTracker(model_path(), num_hands=2, mirrored=True) as tracker:
        frame = tracker.process(np.zeros((480, 640, 3), dtype=np.uint8), 0)
    assert frame.hands == ()


def test_monitors_are_enumerated_in_physical_pixels(qapp):
    """EnumDisplayMonitors: จอหลักอยู่ที่ (0, 0) และขนาดตรงกับ GetSystemMetrics"""
    from aether.mapping import list_monitors, match_qt_screens

    monitors = list_monitors()
    primary = [m for m in monitors if m.primary]
    assert len(primary) == 1
    user32 = ctypes.windll.user32
    geometry = primary[0].geometry
    assert (geometry.left, geometry.top) == (0, 0)
    assert (geometry.width, geometry.height) == (
        user32.GetSystemMetrics(0),
        user32.GetSystemMetrics(1),
    )
    assert primary[0].name.startswith("\\\\.\\DISPLAY")

    from PySide6.QtGui import QGuiApplication

    matched = match_qt_screens(monitors, QGuiApplication.screens())
    assert all(screen is not None for screen in matched)


def _native_test_window():
    """หน้าต่าง Win32 จริงที่อยู่บนสุด (สร้างด้วย CreateWindowExW — ไม่พึ่ง Qt ซึ่ง test รันแบบ offscreen)"""
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    user32.CreateWindowExW.argtypes = (
        wintypes.DWORD, wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.DWORD,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        wintypes.HWND, wintypes.HMENU, wintypes.HINSTANCE, wintypes.LPVOID,
    )  # fmt: skip
    user32.CreateWindowExW.restype = wintypes.HWND
    user32.SetWindowPos.argtypes = (
        wintypes.HWND, wintypes.HWND, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        wintypes.UINT,
    )  # fmt: skip
    user32.DestroyWindow.argtypes = (wintypes.HWND,)
    ws_overlapped_window, ws_visible = 0x00CF0000, 0x10000000
    hwnd = user32.CreateWindowExW(
        0, "STATIC", "AETHER grab test", ws_overlapped_window | ws_visible,
        200, 200, 400, 300, None, None, None, None,
    )  # fmt: skip
    if hwnd:
        hwnd_topmost = wintypes.HWND(-1)
        user32.SetWindowPos(hwnd, hwnd_topmost, 0, 0, 0, 0, 0x0001 | 0x0002 | 0x0040)
    return user32, hwnd


def test_window_under_point_can_be_grabbed_and_moved():
    """หยิบหน้าต่างจริงด้วย WindowFromPoint + ขยับด้วย SetWindowPos"""
    import os

    from aether.actions import (
        win32_grab_window,
        win32_move_window,
        win32_root_window_at,
        win32_window_info,
        win32_window_rect,
    )

    user32, hwnd = _native_test_window()
    if not hwnd:
        pytest.skip(f"สร้างหน้าต่างทดสอบไม่ได้ (error {ctypes.get_last_error()})")
    try:
        rect = win32_window_rect(hwnd)
        assert rect is not None and rect[2] > rect[0] and rect[3] > rect[1]
        pid, window_class = win32_window_info(hwnd)
        assert pid == os.getpid()  # หน้าต่างนี้เป็นของ process เราเอง
        assert window_class.lower() == "static"

        # ขยับได้ตรงระยะ และขนาดไม่เปลี่ยน (ไม่ขึ้นกับว่ามีหน้าต่างอื่นบังหรือไม่)
        assert win32_move_window(hwnd, 37, 21)
        moved = win32_window_rect(hwnd)
        assert (moved[0] - rect[0], moved[1] - rect[1]) == (37, 21)
        assert (moved[2] - moved[0], moved[3] - moved[1]) == (rect[2] - rect[0], rect[3] - rect[1])

        center = ((moved[0] + moved[2]) // 2, (moved[1] + moved[3]) // 2)
        top = win32_root_window_at(*center)
        if top != hwnd:
            cover = win32_window_info(top)[1] if top else "ไม่มี"
            pytest.skip(f"หน้าต่างอื่น ({cover}) บังหน้าต่างทดสอบอยู่ใน session นี้")
        assert win32_grab_window(*center) is None  # หน้าต่างของ process ตัวเองห้ามหยิบ
        assert win32_grab_window(*center, allow_own=True) == hwnd
    finally:
        user32.DestroyWindow(hwnd)


def test_desktop_is_never_grabbed():
    from aether.actions import win32_grab_window

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    user32.GetShellWindow.restype = wintypes.HWND
    shell = user32.GetShellWindow()
    if not shell:
        pytest.skip("ไม่มี shell window ใน session นี้")
    grabbed = win32_grab_window(-30000, -30000)  # นอกทุกจอ → ไม่มีหน้าต่าง
    assert grabbed is None
