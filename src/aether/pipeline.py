"""ร้อย tracker → gesture engine → action executor เข้าด้วยกัน (ไม่ขึ้นกับ Qt)

:meth:`Pipeline.process` ถูกเรียกจาก worker thread ของกล้องทุกเฟรม และส่ง input ให้ OS จาก thread
นั้นเลย (latency ต่ำ ไม่รอ UI) ส่วน hotkey/UI/watchdog เรียก method อื่นจาก thread ของตัวเองได้
เพราะ engine กับ executor ถูกครอบด้วย lock เดียวกัน — สถานะ ARM มีแหล่งความจริงเดียวคือ engine
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .actions import ActionExecutor
from .gestures import LOST_RELEASE_MS, EngineSnapshot, EventKind, GestureEngine, GestureEvent
from .tracker import HandFrame

if TYPE_CHECKING:
    import numpy as np

__all__ = ["FrameResult", "Pipeline", "clock_ms"]

log = logging.getLogger(__name__)


def clock_ms() -> int:
    """นาฬิกา monotonic (ms) ที่ใช้เป็น timestamp ของเฟรมและของ watchdog — ฐานเวลาเดียวกันทั้งแอป"""
    return int(time.perf_counter() * 1000)


@dataclass(frozen=True, slots=True)
class FrameResult:
    """ผลของหนึ่งเฟรม: มือที่เห็น + สถานะ engine + event ที่ทำไปแล้ว"""

    hands: HandFrame
    snapshot: EngineSnapshot
    events: tuple[GestureEvent, ...] = ()


class Pipeline:
    def __init__(
        self,
        track: Callable[[np.ndarray, int], HandFrame],
        engine: GestureEngine,
        executor: ActionExecutor,
        *,
        on_events: Callable[[tuple[GestureEvent, ...]], None] | None = None,
        clock: Callable[[], int] = clock_ms,
    ) -> None:
        self._track = track
        self._engine = engine
        self._executor = executor
        self._on_events = on_events
        self._clock = clock
        self._lock = threading.Lock()
        self._last_frame_ms: int | None = None

    @property
    def armed(self) -> bool:
        with self._lock:
            return self._engine.armed

    @property
    def snapshot(self) -> EngineSnapshot:
        with self._lock:
            return self._engine.snapshot

    # ---- worker thread -------------------------------------------------- #

    def process(self, frame_bgr: np.ndarray, timestamp_ms: int) -> FrameResult:
        """หนึ่งเฟรม: จับมือ → state machine → ส่ง input ให้ OS"""
        hands = self._track(frame_bgr, timestamp_ms)
        try:
            with self._lock:
                _, events = self._engine.update(hands)
                self._executor.execute(events)
                snapshot = self._engine.snapshot
                self._last_frame_ms = self._clock()
        except BaseException:
            self._executor.release_all()  # logic พัง: อย่างน้อยต้องไม่ทิ้งปุ่มเมาส์ค้าง
            raise
        self._notify(events)
        return FrameResult(hands, snapshot, tuple(events))

    # ---- thread ใดก็ได้ -------------------------------------------------- #

    def set_armed(self, armed: bool) -> bool:
        """ตั้งสถานะ ARM คืนสถานะหลังตั้ง (มีผลทันที แม้กล้องจะค้างอยู่)"""
        with self._lock:
            events = self._engine.set_armed(armed)
            self._executor.execute(events)
            result = self._engine.armed
        self._notify(events)
        return result

    def toggle_armed(self) -> bool:
        with self._lock:
            events = self._engine.set_armed(not self._engine.armed)
            self._executor.execute(events)
            result = self._engine.armed
        self._notify(events)
        return result

    def check_stall(self) -> None:
        """Watchdog: ถ้าไม่มีเฟรมเข้ามาเกิน ``LOST_RELEASE_MS`` ให้ engine ทำเหมือนมือหาย

        worker thread ที่ค้างอยู่ใน driver ของกล้องปล่อยปุ่มเมาส์เองไม่ได้ จึงต้องมีคนนอกมาเรียก
        (แอปเรียกจาก timer ใน UI thread)
        """
        with self._lock:
            if self._last_frame_ms is None:
                return
            now = self._clock()
            if now - self._last_frame_ms < LOST_RELEASE_MS:
                return
            events = self._engine.tick(now)
            self._executor.execute(events)
        self._notify(events)

    def shutdown(self) -> None:
        """Disarm และปล่อยทุกปุ่ม (เรียกก่อนปิดแอป หรือเมื่อ pipeline พัง)"""
        with self._lock:
            events = self._engine.set_armed(False)
            self._executor.execute(events)
            self._executor.release_all()
        self._notify(events)

    # ---- internal ------------------------------------------------------- #

    def _notify(self, events: list[GestureEvent]) -> None:
        """แจ้ง event ที่ไม่ใช่การขยับเคอร์เซอร์ให้ผู้ฟัง (HUD/เสียง) — เรียกนอก lock"""
        if self._on_events is None:
            return
        discrete = tuple(e for e in events if e.kind is not EventKind.MOVE)
        if discrete:
            try:
                self._on_events(discrete)
            except Exception:
                log.exception("event listener failed")
