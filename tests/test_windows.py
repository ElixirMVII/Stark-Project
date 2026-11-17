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
