"""สุขภาพของการใช้งานประจำวัน (spec v1.1 backlog v1.2 "ใช้ทุกวันได้") — pure logic + การอ่านค่าจาก OS

- :class:`LightMonitor` — เตือนเมื่อภาพจากกล้องมืดนานเกินไป (มี hysteresis ไม่กระพริบไปมา)
- :func:`webcam_users` — โปรแกรมไหนกำลังใช้กล้องอยู่ (Windows: อ่านจาก Privacy settings ของ Windows ใน registry
  ``CapabilityAccessManager\\ConsentStore\\webcam`` — ไม่ต้องใช้สิทธิ์ admin) ใช้บอกผู้ใช้ตอนเปิดกล้องไม่ได้
- :func:`on_battery` — เครื่องกำลังใช้แบตเตอรี่อยู่หรือไม่ (psutil)
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import PureWindowsPath

__all__ = [
    "LightMonitor",
    "consent_store_users",
    "frame_brightness",
    "on_battery",
    "webcam_users",
]

log = logging.getLogger(__name__)

LOW_SECONDS = 3.0  # มืดต่อเนื่องนานเท่านี้จึงเตือน
RECOVER_SECONDS = 2.0  # สว่างต่อเนื่องนานเท่านี้จึงเลิกเตือน
RECOVER_MARGIN = 10.0  # ต้องสว่างเกินเกณฑ์เท่านี้จึงนับว่าหายมืดแล้ว (กันกระพริบ)

_CONSENT_KEY = (
    r"Software\Microsoft\Windows\CurrentVersion\CapabilityAccessManager\ConsentStore\webcam"
)


def frame_brightness(frame) -> float:
    """ความสว่างเฉลี่ยของภาพ BGR (0–255) จากจุดตัวอย่างทุก 8 px — ถูกพอจะทำทุกเฟรม"""
    return float(frame[::8, ::8].mean())


class LightMonitor:
    """ตัดสินว่าแสงน้อยไหม: ``update`` คืน ``"low"`` / ``"ok"`` ตอนสถานะเปลี่ยน ไม่งั้น ``None``"""

    def __init__(self, threshold: float) -> None:
        self.threshold = threshold
        self.low = False
        self._since: float | None = None  # เวลาที่เริ่มเห็นสถานะตรงข้าม

    def update(self, brightness: float | None, now: float) -> str | None:
        if self.threshold <= 0 or brightness is None:
            return None
        if not self.low:
            dark = brightness < self.threshold
            if not dark:
                self._since = None
                return None
            self._since = now if self._since is None else self._since
            if now - self._since >= LOW_SECONDS:
                self.low, self._since = True, None
                return "low"
            return None
        bright = brightness >= self.threshold + RECOVER_MARGIN
        if not bright:
            self._since = None
            return None
        self._since = now if self._since is None else self._since
        if now - self._since >= RECOVER_SECONDS:
            self.low, self._since = False, None
            return "ok"
        return None


@dataclass(frozen=True, slots=True)
class ConsentEntry:
    key: str  # ชื่อ subkey: แอปจาก Store (เช่น "Microsoft.WindowsCamera_8wekyb3d8bbwe") หรือ path ที่ใช้ # แทน \
    packaged: bool
    started: int  # LastUsedTimeStart (FILETIME) 0 = ไม่เคยใช้
    stopped: int  # LastUsedTimeStop 0 = ยังใช้อยู่


def consent_store_users(
    entries: Iterable[ConsentEntry], *, exclude: str | None = None
) -> list[str]:
    """ชื่อโปรแกรมที่ "กำลังใช้กล้องอยู่" (เริ่มแล้วแต่ยังไม่หยุด) — ``exclude`` = path ของตัวเอง"""
    users: list[str] = []
    skip = exclude.lower() if exclude else None
    for entry in entries:
        if not entry.started or entry.stopped:
            continue
        if entry.packaged:
            name = entry.key.split("_")[0].split(".")[
                -1
            ]  # Microsoft.WindowsCamera_xxx → WindowsCamera
        else:
            path = entry.key.replace("#", "\\")
            if skip and path.lower() == skip:
                continue
            name = PureWindowsPath(path).name
        if name and name not in users:
            users.append(name)
    return users


def _read_consent_store() -> list[ConsentEntry]:  # pragma: no cover — Windows เท่านั้น (ทดสอบใน CI)
    import winreg

    entries: list[ConsentEntry] = []

    def values(key) -> tuple[int, int]:
        def get(name: str) -> int:
            try:
                return int(winreg.QueryValueEx(key, name)[0])
            except OSError:
                return 0

        return get("LastUsedTimeStart"), get("LastUsedTimeStop")

    def walk(root, path: str, packaged: bool) -> None:
        try:
            parent = winreg.OpenKey(root, path)
        except OSError:
            return
        with parent:
            index = 0
            while True:
                try:
                    name = winreg.EnumKey(parent, index)
                except OSError:
                    break
                index += 1
                if name == "NonPackaged":
                    continue
                try:
                    with winreg.OpenKey(parent, name) as child:
                        started, stopped = values(child)
                except OSError:
                    continue
                entries.append(ConsentEntry(name, packaged, started, stopped))

    for root in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
        walk(root, _CONSENT_KEY, packaged=True)
        walk(root, _CONSENT_KEY + r"\NonPackaged", packaged=False)
    return entries


def webcam_users() -> list[str]:
    """โปรแกรมอื่นที่กำลังใช้กล้องอยู่ (เช่น ``["Teams.exe"]``) — ไม่ใช่ Windows / อ่านไม่ได้ = ``[]``"""
    if sys.platform != "win32":
        return []
    try:
        return consent_store_users(_read_consent_store(), exclude=sys.executable)
    except Exception:
        log.debug("อ่านรายชื่อโปรแกรมที่ใช้กล้องไม่ได้", exc_info=True)
        return []


def on_battery() -> bool | None:
    """``True`` = ใช้แบตเตอรี่ (ไม่ได้เสียบสายชาร์จ), ``False`` = เสียบสาย, ``None`` = ไม่มีแบต/ไม่รู้"""
    try:
        import psutil

        battery = psutil.sensors_battery()
    except Exception:  # noqa: BLE001
        return None
    if battery is None or battery.power_plugged is None:
        return None
    return not battery.power_plugged
