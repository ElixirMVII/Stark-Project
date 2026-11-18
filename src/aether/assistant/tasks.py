"""รันงาน Claude ใน daemon thread แล้วส่งผลกลับ UI thread ผ่าน Qt signal (ห้ามแตะ widget จาก worker)"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from typing import Any

from PySide6.QtCore import QObject, Signal

from .service import ClaudeError

__all__ = ["ClaudeBridge", "ClaudeTask", "run_task"]

log = logging.getLogger(__name__)

# ต้องเก็บ reference ไว้จนงานจบ ไม่งั้น QObject ถูกเก็บขยะก่อน signal มาถึง
_RUNNING: set[ClaudeTask] = set()


class ClaudeTask(QObject):
    text = Signal(str)  # ชิ้นข้อความระหว่าง stream
    result = Signal(object)  # ผลสุดท้าย (str หรือ pydantic model)
    failed = Signal(str)  # ข้อความไทยพร้อมแสดง
    finished = Signal()

    def __init__(self, job: Callable[[Callable[[str], None]], Any]) -> None:
        super().__init__()
        self._job = job
        self.cancelled = False

    def start(self, *, sync: bool = False) -> ClaudeTask:
        """``sync=True`` = รันใน thread ปัจจุบัน (ใช้ในเทส) — ต่อ signal ให้ครบก่อนเรียก"""
        _RUNNING.add(self)
        # ปล่อย reference ใน UI thread หลังทุก slot ได้ผลแล้ว (ต่อเป็นตัวสุดท้าย) — ถ้าปล่อยใน worker
        # QObject อาจถูกลบก่อน signal แบบ queued ถูกส่ง แล้วผลลัพธ์หายเงียบ
        self.finished.connect(self._release)
        if sync:
            self._run()
        else:
            threading.Thread(target=self._run, name="aether-claude", daemon=True).start()
        return self

    def cancel(self) -> None:
        """เลิกรับผล (คำขอที่ส่งไปแล้วยังวิ่งต่อจนจบ แต่ไม่มีอะไรแสดง)"""
        self.cancelled = True

    def _emit_text(self, piece: str) -> None:
        if not self.cancelled:
            self.text.emit(piece)

    def _run(self) -> None:
        try:
            value = self._job(self._emit_text)
        except ClaudeError as exc:
            if not self.cancelled:
                self.failed.emit(str(exc))
        except Exception as exc:
            log.exception("งาน Claude ล้มเหลว")
            if not self.cancelled:
                self.failed.emit(f"เกิดข้อผิดพลาด: {exc}")
        else:
            if not self.cancelled:
                self.result.emit(value)
        finally:
            self.finished.emit()

    def _release(self) -> None:
        _RUNNING.discard(self)


def run_task(
    job: Callable[[Callable[[str], None]], Any],
    *,
    on_text: Callable[[str], None] | None = None,
    on_result: Callable[[Any], None] | None = None,
    on_error: Callable[[str], None] | None = None,
    on_finished: Callable[[], None] | None = None,
    sync: bool = False,
) -> ClaudeTask:
    task = ClaudeTask(job)
    if on_text is not None:
        task.text.connect(on_text)
    if on_result is not None:
        task.result.connect(on_result)
    if on_error is not None:
        task.failed.connect(on_error)
    if on_finished is not None:
        task.finished.connect(on_finished)
    return task.start(sync=sync)


class ClaudeBridge:
    """ตัวกลางที่หน้า UI ใช้ถาม Claude: เช็กความพร้อมก่อน แล้วรันใน thread (``sync=True`` ในเทส)

    ``service`` คืออะไรก็ได้ที่มี ``ready_reason()`` / ``stream_text()`` / ``parse()``
    (:class:`aether.assistant.service.ClaudeService` หรือของปลอมในเทส)
    """

    def __init__(self, service: Any, *, sync: bool = False) -> None:
        self.service = service
        self.sync = sync
        self.busy = False

    def ready_reason(self) -> str | None:
        return self.service.ready_reason()

    def stream(
        self,
        system: str,
        content: Any,
        *,
        on_text: Callable[[str], None] | None = None,
        on_result: Callable[[Any], None] | None = None,
        on_error: Callable[[str], None] | None = None,
        max_tokens: int | None = None,
    ) -> ClaudeTask | None:
        def job(emit: Callable[[str], None]) -> str:
            return self.service.stream_text(system, content, emit, max_tokens=max_tokens)

        return self._start(job, on_text, on_result, on_error)

    def parse(
        self,
        system: str,
        content: Any,
        model_cls: type,
        *,
        on_result: Callable[[Any], None] | None = None,
        on_error: Callable[[str], None] | None = None,
        max_tokens: int | None = None,
    ) -> ClaudeTask | None:
        def job(_emit: Callable[[str], None]) -> Any:
            return self.service.parse(system, content, model_cls, max_tokens=max_tokens)

        return self._start(job, None, on_result, on_error)

    def _start(self, job, on_text, on_result, on_error) -> ClaudeTask | None:
        reason = self.ready_reason()
        if reason is not None:
            if on_error is not None:
                on_error(reason)
            return None
        if self.busy:
            if on_error is not None:
                on_error("Claude กำลังตอบคำถามก่อนหน้าอยู่ — รอสักครู่")
            return None
        self.busy = True

        def done() -> None:
            self.busy = False

        return run_task(
            job,
            on_text=on_text,
            on_result=on_result,
            on_error=on_error,
            on_finished=done,
            sync=self.sync,
        )
