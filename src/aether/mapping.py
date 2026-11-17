"""พิกัดกล้อง (normalized) → พิกัดจอ (physical px) และการแปลง DPI

ฝั่ง input ใช้ **physical px** เป็นหลักเสมอ (ตรงกับที่ Windows ใช้ตอนย้ายเคอร์เซอร์เมื่อ process
เป็น DPI-aware ซึ่ง Qt 6 ตั้งให้อัตโนมัติ) ส่วนการวาดของ Qt ใช้ logical px จึงต้องแปลงด้วย
``devicePixelRatio``

หลายจอ (``pointer.screens = "all"``): :class:`Desktop` รวมทุกจอเป็นพื้นที่เดียว active region ถูกยืด
เต็มกรอบที่ครอบทุกจอ และตำแหน่งที่ตกในช่องว่างระหว่างจอถูกดึงเข้าจอที่ใกล้ที่สุด
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass

__all__ = [
    "Desktop",
    "Monitor",
    "PointerTracker",
    "Region",
    "ScreenArea",
    "ScreenGeometry",
    "list_monitors",
    "logical_to_physical",
    "map_to_screen",
    "match_qt_screens",
    "normalize_in_region",
    "physical_to_logical",
    "primary_screen_geometry",
]

Region = tuple[float, float, float, float]  # x0, y0, x1, y1 (normalized ในเฟรมกล้อง)


@dataclass(frozen=True, slots=True)
class ScreenGeometry:
    """พื้นที่จอเป็น physical px"""

    left: int
    top: int
    width: int
    height: int

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("ขนาดจอต้อง > 0")

    @property
    def right(self) -> int:
        """พิกัด x ของ pixel ขวาสุด (นับรวม)"""
        return self.left + self.width - 1

    @property
    def bottom(self) -> int:
        return self.top + self.height - 1

    def clamp(self, x: float, y: float) -> tuple[float, float]:
        return (
            min(max(x, float(self.left)), float(self.right)),
            min(max(y, float(self.top)), float(self.bottom)),
        )

    def contains(self, x: float, y: float) -> bool:
        return self.left <= x < self.left + self.width and self.top <= y < self.top + self.height

    def union(self, other: ScreenGeometry) -> ScreenGeometry:
        left, top = min(self.left, other.left), min(self.top, other.top)
        right = max(self.left + self.width, other.left + other.width)
        bottom = max(self.top + self.height, other.top + other.height)
        return ScreenGeometry(left, top, right - left, bottom - top)


@dataclass(frozen=True, slots=True)
class Monitor:
    """จอหนึ่งจอ: พื้นที่เป็น physical px ในพิกัดของ virtual desktop"""

    name: str  # ชื่ออุปกรณ์ เช่น ``\\.\DISPLAY1`` (Windows) — ใช้จับคู่กับ QScreen
    geometry: ScreenGeometry
    primary: bool = False


class Desktop:
    """ทุกจอรวมกัน — ใช้แทน :class:`ScreenGeometry` ได้ (มี left/top/width/height/right/bottom/clamp)

    ``left/top/width/height`` คือกรอบที่ครอบทุกจอ (active region ถูกยืดเต็มกรอบนี้) ส่วน
    :meth:`clamp` ดึงจุดเข้าจอที่ใกล้ที่สุด จุดในช่องว่างระหว่างจอจึงไม่ทำให้เคอร์เซอร์หลุด
    """

    def __init__(self, monitors: tuple[Monitor, ...] | list[Monitor]) -> None:
        if not monitors:
            raise ValueError("ต้องมีจออย่างน้อยหนึ่งจอ")
        self.monitors = tuple(monitors)
        bounds = self.monitors[0].geometry
        for monitor in self.monitors[1:]:
            bounds = bounds.union(monitor.geometry)
        self.bounds = bounds

    @property
    def left(self) -> int:
        return self.bounds.left

    @property
    def top(self) -> int:
        return self.bounds.top

    @property
    def width(self) -> int:
        return self.bounds.width

    @property
    def height(self) -> int:
        return self.bounds.height

    @property
    def right(self) -> int:
        return self.bounds.right

    @property
    def bottom(self) -> int:
        return self.bounds.bottom

    @property
    def primary(self) -> Monitor:
        return next((m for m in self.monitors if m.primary), self.monitors[0])

    def monitor_at(self, x: float, y: float) -> Monitor:
        """จอที่มีจุดนี้ (ถ้าอยู่นอกทุกจอ: จอที่ใกล้ที่สุด)"""
        for monitor in self.monitors:
            if monitor.geometry.contains(x, y):
                return monitor
        return min(self.monitors, key=lambda m: math.dist((x, y), m.geometry.clamp(x, y)))

    def clamp(self, x: float, y: float) -> tuple[float, float]:
        return self.monitor_at(x, y).geometry.clamp(x, y)


ScreenArea = ScreenGeometry | Desktop  # พื้นที่ที่เคอร์เซอร์ไปได้: จอเดียว หรือทุกจอ


def normalize_in_region(nx: float, ny: float, region: Region) -> tuple[float, float]:
    """ตำแหน่งใน active region เป็น 0–1 (นอก region ถูกตรึงที่ขอบ)"""
    x0, y0, x1, y1 = region
    u = (nx - x0) / (x1 - x0)
    v = (ny - y0) / (y1 - y0)
    return (min(max(u, 0.0), 1.0), min(max(v, 0.0), 1.0))


def map_to_screen(nx: float, ny: float, region: Region, screen: ScreenArea) -> tuple[float, float]:
    """จุดในเฟรมกล้อง (normalized) → physical px บนจอ โดย active region ถูกยืดเต็มจอ"""
    u, v = normalize_in_region(nx, ny, region)
    return (screen.left + u * (screen.width - 1), screen.top + v * (screen.height - 1))


def physical_to_logical(x: float, y: float, dpr: float) -> tuple[float, float]:
    """physical px → logical px ของ Qt (สำหรับวาด HUD) — ``dpr`` คือ ``devicePixelRatio`` ของจอ"""
    return (x / dpr, y / dpr)


def logical_to_physical(x: float, y: float, dpr: float) -> tuple[float, float]:
    return (x * dpr, y * dpr)


def primary_screen_geometry() -> ScreenGeometry:
    """ขนาดจอหลักเป็น physical px (ต้องเรียกหลังสร้าง QApplication เพื่อให้ process เป็น DPI-aware แล้ว)"""
    if sys.platform == "win32":
        import ctypes

        user32 = ctypes.windll.user32
        width, height = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)  # SM_CXSCREEN/CY
        if width > 0 and height > 0:
            return ScreenGeometry(0, 0, width, height)

    from PySide6.QtGui import QGuiApplication

    screen = QGuiApplication.primaryScreen()
    if screen is None:
        raise RuntimeError("ไม่พบจอ (ต้องสร้าง QApplication ก่อน)")
    size, dpr = screen.size(), screen.devicePixelRatio()
    return ScreenGeometry(0, 0, round(size.width() * dpr), round(size.height() * dpr))


def list_monitors() -> list[Monitor]:
    """ทุกจอเป็น physical px (เรียกหลังสร้าง QApplication — process ต้องเป็น DPI-aware แล้ว)

    Windows ใช้ ``EnumDisplayMonitors`` + ``GetMonitorInfoW`` ระบบอื่นใช้ QScreen (geometry × dpr)
    """
    if sys.platform == "win32":
        monitors = _win32_monitors()
        if monitors:
            return monitors
    from PySide6.QtGui import QGuiApplication

    primary = QGuiApplication.primaryScreen()
    result = []
    for screen in QGuiApplication.screens():
        geometry, dpr = screen.geometry(), screen.devicePixelRatio()
        result.append(
            Monitor(
                name=screen.name(),
                geometry=ScreenGeometry(
                    round(geometry.x() * dpr),
                    round(geometry.y() * dpr),
                    round(geometry.width() * dpr),
                    round(geometry.height() * dpr),
                ),
                primary=screen is primary,
            )
        )
    if not result:
        raise RuntimeError("ไม่พบจอ (ต้องสร้าง QApplication ก่อน)")
    return result


def _win32_monitors() -> list[Monitor]:
    import ctypes
    from ctypes import wintypes

    class MONITORINFOEXW(ctypes.Structure):
        _fields_ = [
            ("cbSize", wintypes.DWORD),
            ("rcMonitor", wintypes.RECT),
            ("rcWork", wintypes.RECT),
            ("dwFlags", wintypes.DWORD),
            ("szDevice", wintypes.WCHAR * 32),
        ]

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    callback_type = ctypes.WINFUNCTYPE(
        wintypes.BOOL,
        wintypes.HMONITOR,
        wintypes.HDC,
        ctypes.POINTER(wintypes.RECT),
        wintypes.LPARAM,
    )
    user32.EnumDisplayMonitors.argtypes = (
        wintypes.HDC,
        ctypes.c_void_p,
        callback_type,
        wintypes.LPARAM,
    )
    user32.EnumDisplayMonitors.restype = wintypes.BOOL
    user32.GetMonitorInfoW.argtypes = (wintypes.HMONITOR, ctypes.POINTER(MONITORINFOEXW))
    user32.GetMonitorInfoW.restype = wintypes.BOOL

    monitors: list[Monitor] = []

    def on_monitor(handle, _hdc, _rect, _data) -> bool:
        info = MONITORINFOEXW()
        info.cbSize = ctypes.sizeof(MONITORINFOEXW)
        if user32.GetMonitorInfoW(handle, ctypes.byref(info)):
            r = info.rcMonitor
            monitors.append(
                Monitor(
                    name=info.szDevice,
                    geometry=ScreenGeometry(r.left, r.top, r.right - r.left, r.bottom - r.top),
                    primary=bool(info.dwFlags & 1),  # MONITORINFOF_PRIMARY
                )
            )
        return True

    user32.EnumDisplayMonitors(None, None, callback_type(on_monitor), 0)
    return monitors


def match_qt_screens(monitors: list[Monitor] | tuple[Monitor, ...], screens: list) -> list:
    """จับคู่จอ (physical) กับ QScreen ตามลำดับของ ``monitors`` (``None`` ถ้าจับคู่ไม่ได้)

    ใช้ชื่ออุปกรณ์ก่อน (Windows: ``QScreen.name()`` ตรงกับ ``szDevice``) ที่เหลือจับคู่ตามตำแหน่ง
    (เรียงซ้าย→ขวา บน→ล่าง เหมือนกันทั้งสองฝั่ง)
    """
    matched: list = [None] * len(monitors)
    remaining = list(screens)
    for i, monitor in enumerate(monitors):
        for screen in remaining:
            if screen.name() == monitor.name:
                matched[i] = screen
                remaining.remove(screen)
                break
    unmatched = sorted(
        (i for i in range(len(monitors)) if matched[i] is None),
        key=lambda i: (monitors[i].geometry.left, monitors[i].geometry.top),
    )
    remaining.sort(key=lambda s: (s.geometry().x(), s.geometry().y()))
    for i, screen in zip(unmatched, remaining):
        matched[i] = screen
    return matched


class PointerTracker:
    """ตำแหน่งเคอร์เซอร์ที่ตาม "เป้า" (ตำแหน่งมือที่ map แล้ว) ด้วย gain ที่ปรับได้

    - ``gain = 1``: เคอร์เซอร์อยู่ที่เป้าพอดี (absolute mapping)
    - ``gain < 1`` (pinch-lock): เคอร์เซอร์ขยับแค่บางส่วนของที่มือขยับ จึงเกิดระยะห่างจากเป้า
    - เมื่อ gain กลับเป็น 1 เคอร์เซอร์จะไม่กระโดดกลับไปที่เป้า แต่ค่อยๆ เก็บระยะห่างคืนตามที่มือขยับ
      (``RECONVERGE`` ของระยะที่ขยับแต่ละเฟรม) มืออยู่นิ่ง เคอร์เซอร์ก็นิ่ง
    """

    RECONVERGE = 0.5

    def __init__(self, screen: ScreenArea) -> None:
        self._screen = screen
        self._pos: tuple[float, float] | None = None
        self._target: tuple[float, float] | None = None

    @property
    def position(self) -> tuple[float, float] | None:
        return self._pos

    def reset(self) -> None:
        """ลืมตำแหน่งทั้งหมด — update ครั้งถัดไปจะวางเคอร์เซอร์ที่เป้าทันที"""
        self._pos = None
        self._target = None

    def rebase(self, target: tuple[float, float]) -> None:
        """ตั้งเป้าใหม่โดยไม่ขยับเคอร์เซอร์ (ใช้หลังช่วงที่ตรึงเคอร์เซอร์ไว้ เช่นตอนสกรอล)"""
        if self._pos is None:
            self._pos = self._screen.clamp(*target)
        self._target = target

    def jump_to(self, x: float, y: float) -> None:
        """ย้ายเคอร์เซอร์ไปตำแหน่งที่กำหนดโดยไม่เปลี่ยนเป้า"""
        self._pos = self._screen.clamp(x, y)

    def update(self, target: tuple[float, float], gain: float = 1.0) -> tuple[float, float]:
        if self._pos is None or self._target is None:
            self._target = target
            self._pos = self._screen.clamp(*target)
            return self._pos

        dx, dy = target[0] - self._target[0], target[1] - self._target[1]
        self._target = target
        x, y = self._pos[0] + gain * dx, self._pos[1] + gain * dy

        if gain >= 1.0:
            off_x, off_y = target[0] - x, target[1] - y
            offset = math.hypot(off_x, off_y)
            if offset > 0.0:
                step = min(offset, self.RECONVERGE * math.hypot(dx, dy))
                x += off_x / offset * step
                y += off_y / offset * step

        self._pos = self._screen.clamp(x, y)
        return self._pos
