"""เปิด AETHER พร้อม Windows (backlog v1.2 "ใช้ทุกวันได้") — ค่า ``AETHER`` ใน
``HKCU\\Software\\Microsoft\\Windows\\CurrentVersion\\Run``

ใช้ registry ของผู้ใช้คนนี้เท่านั้น: **ไม่ต้องใช้สิทธิ์ admin ไม่ได้ติดตั้ง service** (spec ข้อห้าม) และปิดได้จาก
tray หรือหน้า Startup apps ของ Windows เอง
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Protocol

__all__ = ["RUN_KEY", "VALUE_NAME", "AutoStart", "Registry", "WinRegistry", "launch_command"]

RUN_KEY = r"Software\Microsoft\Windows\CurrentVersion\Run"
VALUE_NAME = "AETHER"


def launch_command(*, frozen: bool | None = None, executable: str | None = None) -> str:
    """คำสั่งที่ Windows จะรันตอน login

    - build (``aether.exe``): ``"...\\aether.exe"``
    - รันจาก source: ``"...\\pythonw.exe" -m aether`` (pythonw = ไม่มีหน้าต่าง console ค้าง)
    """
    frozen = getattr(sys, "frozen", False) if frozen is None else frozen
    exe = executable or sys.executable
    if frozen:
        return f'"{exe}"'
    name = re.split(r"[\\/]", exe)[-1]
    if name.lower() == "python.exe":
        windowless = exe[: -len(name)] + "pythonw.exe"
        if executable is not None or Path(windowless).is_file():
            exe = windowless
    return f'"{exe}" -m aether'


class Registry(Protocol):
    def get(self, name: str) -> str | None: ...

    def set(self, name: str, value: str) -> None: ...

    def delete(self, name: str) -> None: ...


class WinRegistry:
    """ค่าใน ``HKCU\\...\\Run`` ผ่าน ``winreg`` (Windows เท่านั้น)"""

    def __init__(self, key: str = RUN_KEY) -> None:
        import winreg

        self._winreg = winreg
        self._key = key

    def get(self, name: str) -> str | None:
        winreg = self._winreg
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, self._key) as key:
                value, _ = winreg.QueryValueEx(key, name)
        except FileNotFoundError:
            return None
        return str(value)

    def set(self, name: str, value: str) -> None:
        winreg = self._winreg
        with winreg.CreateKey(winreg.HKEY_CURRENT_USER, self._key) as key:
            winreg.SetValueEx(key, name, 0, winreg.REG_SZ, value)

    def delete(self, name: str) -> None:
        winreg = self._winreg
        try:
            with winreg.OpenKey(
                winreg.HKEY_CURRENT_USER, self._key, 0, winreg.KEY_SET_VALUE
            ) as key:
                winreg.DeleteValue(key, name)
        except FileNotFoundError:
            pass


class AutoStart:
    """เปิด/ปิด/ตรวจการเปิดพร้อม Windows (``available`` = False บนระบบอื่น)"""

    def __init__(self, registry: Registry | None = None, *, name: str = VALUE_NAME) -> None:
        if registry is None and sys.platform == "win32":
            registry = WinRegistry()
        self._registry = registry
        self._name = name

    @property
    def available(self) -> bool:
        return self._registry is not None

    def enabled(self) -> bool:
        """เปิดไว้และชี้มาที่โปรแกรมนี้ (ถ้าชี้ไปที่อื่น เช่นย้ายโฟลเดอร์ไปแล้ว ถือว่าปิด)"""
        if self._registry is None:
            return False
        value = self._registry.get(self._name)
        return value is not None and value.strip().lower() == launch_command().lower()

    def set_enabled(self, enabled: bool) -> None:
        if self._registry is None:
            raise OSError("เปิดพร้อม Windows ใช้ได้เฉพาะบน Windows")
        if enabled:
            self._registry.set(self._name, launch_command())
        else:
            self._registry.delete(self._name)
