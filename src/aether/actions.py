"""``GestureEvent`` → input ของ OS (เมาส์) ผ่าน pynput — มีโหมด dry-run

:class:`ActionExecutor` จำว่าปุ่มไหนกดค้างอยู่ และ :meth:`ActionExecutor.release_all`
ปล่อยทุกปุ่มได้เสมอ (ใช้ตอน disarm, ปิดแอป, pipeline พัง)

ไม่มีการขอสิทธิ์ admin: หน้าต่างที่รันแบบ Administrator จะไม่รับ input จากแอปนี้ (Windows UIPI)
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Protocol

from .gestures import EventKind, GestureEvent

__all__ = [
    "ActionExecutor",
    "DryRunBackend",
    "InputBackend",
    "PynputBackend",
    "create_backend",
    "double_click_time_ms",
]

log = logging.getLogger(__name__)

BUTTONS = ("left", "right", "middle")


class InputBackend(Protocol):
    """ตัวส่ง input จริง (พิกัดเป็น physical px, scroll เป็น notch: dy > 0 = ขึ้น, dx > 0 = ขวา)"""

    def move(self, x: int, y: int) -> None: ...

    def press(self, button: str) -> None: ...

    def release(self, button: str) -> None: ...

    def scroll(self, dx: int, dy: int) -> None: ...


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


class PynputBackend:
    """เมาส์จริงผ่าน ``pynput.mouse.Controller`` (บน Windows คือ SetCursorPos + SendInput)"""

    def __init__(self) -> None:
        from pynput import mouse

        self._mouse = mouse.Controller()
        self._buttons = {
            "left": mouse.Button.left,
            "right": mouse.Button.right,
            "middle": mouse.Button.middle,
        }

    def move(self, x: int, y: int) -> None:
        self._mouse.position = (x, y)

    def press(self, button: str) -> None:
        self._mouse.press(self._buttons[button])

    def release(self, button: str) -> None:
        self._mouse.release(self._buttons[button])

    def scroll(self, dx: int, dy: int) -> None:
        self._mouse.scroll(dx, dy)


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

    backend ที่โยน exception จะถูก log แล้วข้าม event นั้น (pipeline ต้องไม่ล้มเพราะ input ตัวเดียว)
    ยกเว้นการปล่อยปุ่ม ซึ่งจะถูกลองทุกครั้งและถือว่าปุ่มไม่ค้างแล้วเสมอ
    """

    def __init__(self, backend: InputBackend) -> None:
        self._backend = backend
        self._pressed: set[str] = set()

    @property
    def pressed(self) -> frozenset[str]:
        """ปุ่มเมาส์ที่แอปนี้กดค้างอยู่"""
        return frozenset(self._pressed)

    def execute(self, events: Iterable[GestureEvent]) -> None:
        for event in events:
            try:
                self._execute_one(event)
            except Exception:
                log.exception("input backend failed on %s", event.kind.value)

    def release_all(self) -> None:
        """ปล่อยทุกปุ่มที่กดค้าง (เรียกซ้ำได้ ปลอดภัยเสมอ)"""
        for button in sorted(self._pressed):
            self._release(button)

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
        elif kind is EventKind.DISARMED:
            self.release_all()  # กันพลาดอีกชั้น: disarm แล้วต้องไม่มีปุ่มค้าง
        # ARMED: แจ้งสถานะเท่านั้น

    def _release(self, button: str) -> None:
        self._pressed.discard(button)
        try:
            self._backend.release(button)
        except Exception:
            log.exception("failed to release %s button", button)
