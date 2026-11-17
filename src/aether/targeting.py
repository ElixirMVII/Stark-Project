"""Target Lock (spec v1.1 M6.1) — หาปุ่มที่อยู่ใต้เคอร์เซอร์ผ่าน Windows UI Automation ใน thread แยก

- **ห้ามเรียก UIA จาก pointer thread**: :class:`TargetFinder` มี thread ของตัวเอง (CoInitializeEx ใน thread นั้น)
  ถามที่ ~``rate_hz`` และทุกครั้งที่เคอร์เซอร์ขยับเกิน ``MOVE_PX`` ผลส่งให้ engine ผ่าน callback
- :class:`TargetResolver` (pure — test ได้ด้วย backend ปลอม): ไล่จาก element ใต้จุดขึ้นไปไม่เกิน 3 ชั้นหา
  ชนิดที่ "กดได้", ต้อง enabled และไม่ offscreen, ขนาด 8 px – ``max_size`` (logical px), cache ตามสี่เหลี่ยม
  ของปุ่ม, call ที่นานเกิน 150 ms → ทิ้งผลและพักโปรแกรมนั้น 5 วินาที, ข้ามหน้าต่างที่ค้าง (Not responding),
  โปรแกรมใน ``app_blocklist`` และตอนหน้าต่างที่ active เป็นเต็มจอที่ไม่ใช่ explorer (เช่นเกม)
- :class:`UiaBackend` คุย UIA จริงด้วยไลบรารี ``uiautomation`` (Windows เท่านั้น — import ภายใน thread
  ของ finder หลังสร้าง QApplication แล้ว เพราะตอน import มันตั้ง DPI awareness ของ process)
- **Privacy**: ชื่อของ element (อาจเป็นหัวข้ออีเมล/ชื่อไฟล์) ใช้แสดงบน HUD เท่านั้น ห้าม log — log ใช้
  :meth:`aether.snap.Target.describe` (ชนิด + ขนาด)
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

from .config import TargetingConfig
from .snap import Target

__all__ = [
    "CLICKABLE",
    "ElementInfo",
    "ForegroundInfo",
    "TargetFinder",
    "TargetResolver",
    "UiaBackend",
    "WindowInfo",
]

log = logging.getLogger(__name__)

# ชนิดของ UIA ที่ "กดได้" → ป้ายบน HUD
CLICKABLE = {
    "ButtonControl": "Button",
    "MenuItemControl": "MenuItem",
    "ListItemControl": "ListItem",
    "TabItemControl": "TabItem",
    "HyperlinkControl": "Hyperlink",
    "CheckBoxControl": "CheckBox",
    "RadioButtonControl": "RadioButton",
    "TreeItemControl": "TreeItem",
    "ComboBoxControl": "ComboBox",
    "SplitButtonControl": "SplitButton",
}
MAX_ANCESTORS = 3
MIN_SIZE = 8  # px
CALL_TIMEOUT_S = 0.15
BLOCK_S = 5.0
MOVE_PX = 8.0
CACHE_S = 0.4  # ใช้ผลเดิมซ้ำได้นานเท่านี้ถ้าเคอร์เซอร์ยังอยู่ในปุ่มเดิม
POLL_S = 1.0 / 60.0
EXPLORER = "explorer.exe"


@dataclass(frozen=True, slots=True)
class ElementInfo:
    control_type: str  # ชื่อชนิดของ uiautomation เช่น "ButtonControl"
    name: str
    rect: tuple[int, int, int, int]  # left, top, right, bottom (physical px)
    enabled: bool = True
    offscreen: bool = False
    process_id: int = 0


@dataclass(frozen=True, slots=True)
class WindowInfo:
    process_id: int
    exe: str
    hung: bool = False


@dataclass(frozen=True, slots=True)
class ForegroundInfo:
    process_id: int
    exe: str
    fullscreen: bool


class Backend(Protocol):
    def window_at(self, x: int, y: int) -> WindowInfo | None: ...

    def foreground(self) -> ForegroundInfo | None: ...

    def element_chain(self, x: int, y: int, depth: int) -> list[ElementInfo]:
        """element ใต้จุด + ancestor ไม่เกิน ``depth`` ชั้น (ใกล้สุดก่อน)"""
        ...


class TargetResolver:
    """เลือกปุ่มใต้จุด (x, y) จากผลของ backend ตามกติกาของ spec M6.1 (ไม่มี state ของ OS — test ได้)"""

    def __init__(
        self,
        backend: Backend,
        config: TargetingConfig,
        *,
        scale: float = 1.0,
        own_pid: int | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._backend = backend
        self._config = config
        self._max = (config.max_size[0] * scale, config.max_size[1] * scale)
        self._own_pid = os.getpid() if own_pid is None else own_pid
        self._blocklist = {name.lower() for name in config.app_blocklist}
        self._clock = clock
        self._blocked: dict[int, float] = {}  # pid → เวลาที่พักจนถึง
        self._cache: tuple[Target, float] | None = None
        self.suspended_reason: str | None = None  # ทำไมตอนนี้ไม่หาเป้า (แสดงใน debug)

    def resolve(self, x: int, y: int) -> Target | None:
        now = self._clock()
        if self._cache is not None:
            cached, at = self._cache
            if now - at < CACHE_S and cached.contains(x, y):
                return cached
        self._cache = None
        self.suspended_reason = None

        foreground = self._backend.foreground()
        if foreground is not None and foreground.fullscreen and foreground.exe.lower() != EXPLORER:
            self.suspended_reason = "fullscreen"
            return None
        window = self._backend.window_at(x, y)
        if window is None:
            return None
        if window.process_id == self._own_pid:
            return None  # หน้าต่างของ AETHER เอง (debug, wizard)
        if window.exe.lower() in self._blocklist:
            self.suspended_reason = "blocklist"
            return None
        if window.hung:
            self.suspended_reason = "not responding"
            return None
        if self._blocked.get(window.process_id, 0.0) > now:
            self.suspended_reason = "slow"
            return None

        started = self._clock()
        try:
            chain = self._backend.element_chain(x, y, MAX_ANCESTORS)
        except Exception as exc:  # noqa: BLE001 — UIA ใช้กับหน้าต่างนี้ไม่ได้ (เช่นสิทธิ์ admin)
            log.debug("UIA ใช้กับ %s ไม่ได้: %s", window.exe, type(exc).__name__)
            self._blocked[window.process_id] = now + BLOCK_S
            self.suspended_reason = "unavailable"
            return None
        if self._clock() - started > CALL_TIMEOUT_S:
            log.info("UIA ของ %s ตอบช้า — พักการหาเป้า %.0f วินาที", window.exe, BLOCK_S)
            self._blocked[window.process_id] = now + BLOCK_S
            self.suspended_reason = "slow"
            return None

        target = self._pick(chain, now)
        if target is not None:
            self._cache = (target, now)
        return target

    def _pick(self, chain: Sequence[ElementInfo], now: float) -> Target | None:
        for element in chain[: MAX_ANCESTORS + 1]:
            label = CLICKABLE.get(element.control_type)
            if label is None:
                continue
            if element.process_id == self._own_pid:
                return None
            if not element.enabled or element.offscreen:
                return None
            left, top, right, bottom = element.rect
            width, height = right - left, bottom - top
            if min(width, height) < MIN_SIZE or width > self._max[0] or height > self._max[1]:
                return None  # เล็ก/ใหญ่เกิน (เช่นพื้นที่ข้อความทั้งหน้า) ไม่ล็อก
            return Target(left, top, width, height, label, element.name, now)
        return None


class UiaBackend:
    """Windows UI Automation ผ่าน ``uiautomation`` — สร้างและใช้ใน thread เดียวกันเท่านั้น"""

    def __init__(self) -> None:
        import ctypes
        from ctypes import wintypes

        import uiautomation

        self._auto = uiautomation
        self._init = uiautomation.UIAutomationInitializerInThread()
        self._ctypes = ctypes
        self._wintypes = wintypes
        user32 = ctypes.WinDLL("user32", use_last_error=True)
        user32.GetForegroundWindow.restype = wintypes.HWND
        user32.GetWindowRect.argtypes = (wintypes.HWND, ctypes.POINTER(wintypes.RECT))
        user32.MonitorFromWindow.restype = wintypes.HANDLE
        user32.MonitorFromWindow.argtypes = (wintypes.HWND, wintypes.DWORD)
        user32.IsHungAppWindow.argtypes = (wintypes.HWND,)
        self._user32 = user32
        self._exe_cache: dict[int, str] = {}

    def close(self) -> None:
        self._init.Uninitialize()

    def _exe(self, pid: int) -> str:
        if pid not in self._exe_cache:
            try:
                import psutil

                self._exe_cache[pid] = psutil.Process(pid).name()
            except Exception:  # noqa: BLE001 — process หายไปแล้ว / ไม่มีสิทธิ์
                self._exe_cache[pid] = ""
        return self._exe_cache[pid]

    def window_at(self, x: int, y: int) -> WindowInfo | None:
        from .actions import win32_root_window_at, win32_window_info

        window = win32_root_window_at(x, y)
        if not window:
            return None
        pid, _ = win32_window_info(window)
        return WindowInfo(pid, self._exe(pid), bool(self._user32.IsHungAppWindow(window)))

    def foreground(self) -> ForegroundInfo | None:
        from .actions import win32_window_info

        ctypes, wintypes = self._ctypes, self._wintypes
        window = self._user32.GetForegroundWindow()
        if not window:
            return None
        rect = wintypes.RECT()
        if not self._user32.GetWindowRect(window, ctypes.byref(rect)):
            return None

        class MONITORINFO(ctypes.Structure):
            _fields_ = (
                ("cbSize", wintypes.DWORD),
                ("rcMonitor", wintypes.RECT),
                ("rcWork", wintypes.RECT),
                ("dwFlags", wintypes.DWORD),
            )

        info = MONITORINFO()
        info.cbSize = ctypes.sizeof(MONITORINFO)
        monitor = self._user32.MonitorFromWindow(window, 2)  # MONITOR_DEFAULTTONEAREST
        if not monitor or not self._user32.GetMonitorInfoW(monitor, ctypes.byref(info)):
            return None
        m = info.rcMonitor
        fullscreen = (
            rect.left <= m.left
            and rect.top <= m.top
            and rect.right >= m.right
            and rect.bottom >= m.bottom
        )
        pid, _ = win32_window_info(window)
        return ForegroundInfo(pid, self._exe(pid), fullscreen)

    def element_chain(self, x: int, y: int, depth: int) -> list[ElementInfo]:
        control = self._auto.ControlFromPoint(int(x), int(y))
        chain: list[ElementInfo] = []
        while control is not None and len(chain) <= depth:
            rect = control.BoundingRectangle
            chain.append(
                ElementInfo(
                    control.ControlTypeName,
                    control.Name or "",
                    (rect.left, rect.top, rect.right, rect.bottom),
                    bool(control.IsEnabled),
                    bool(control.IsOffscreen),
                    int(control.ProcessId),
                )
            )
            control = control.GetParentControl()
        return chain


class TargetFinder:
    """thread ที่ถามหาปุ่มใต้เคอร์เซอร์เรื่อยๆ แล้วส่งผลผ่าน ``publish`` (None = ไม่มีปุ่ม)

    ``position()`` คืนตำแหน่งเคอร์เซอร์ก่อน snap (physical px) หรือ None, ``active()`` = ควรหาเป้าไหม
    (เช่นตอน ARMED) — ทั้งคู่ต้อง thread-safe
    """

    def __init__(
        self,
        config: TargetingConfig,
        *,
        position: Callable[[], tuple[int, int] | None],
        publish: Callable[[Target | None], None],
        active: Callable[[], bool] = lambda: True,
        backend_factory: Callable[[], Backend] | None = None,
        scale: float = 1.0,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._config = config
        self._position = position
        self._publish = publish
        self._active = active
        self._backend_factory = backend_factory or UiaBackend
        self._scale = scale
        self._clock = clock
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="aether-targeting", daemon=True)
        self._published: Target | None = None
        self.available: bool | None = None  # None = ยังไม่เริ่ม, False = UIA ใช้ไม่ได้
        self.queries = 0

    def start(self) -> None:
        self._thread.start()

    def stop(self, timeout: float = 1.0) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout)

    def _emit(self, target: Target | None) -> None:
        if target != self._published:
            self._published = target
            self._publish(target)

    def _run(self) -> None:
        try:
            backend = self._backend_factory()
        except Exception as exc:  # noqa: BLE001 — ไม่มี uiautomation / ไม่ใช่ Windows / COM เปิดไม่ได้
            log.warning(
                "Target Lock ใช้ไม่ได้ (%s: %s) — ทำงานต่อแบบไม่มีการล็อกเป้า", type(exc).__name__, exc
            )
            self.available = False
            return
        self.available = True
        resolver = TargetResolver(backend, self._config, scale=self._scale, clock=self._clock)
        interval = 1.0 / self._config.rate_hz
        last_at = -1e9
        last_pos: tuple[int, int] | None = None
        try:
            while not self._stop.wait(POLL_S):
                position = self._position() if self._active() else None
                if position is None:
                    self._emit(None)
                    last_pos = None
                    continue
                now = self._clock()
                moved = last_pos is None or (
                    abs(position[0] - last_pos[0]) + abs(position[1] - last_pos[1]) > MOVE_PX
                )
                if not moved and now - last_at < interval:
                    continue
                last_at, last_pos = now, position
                self.queries += 1
                try:
                    target = resolver.resolve(*position)
                except Exception:
                    log.debug("targeting query failed", exc_info=True)
                    target = None
                self._emit(target)
        finally:
            self._emit(None)
            close = getattr(backend, "close", None)
            if close is not None:
                close()
