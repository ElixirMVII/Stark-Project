"""อ่านเฟรมจากกล้องใน ``QThread`` แล้วประมวลผลต่อใน thread เดียวกัน

Data flow ใน worker thread: ``FrameSource.read()`` → ``processor(frame, timestamp_ms)``
(คือ :meth:`aether.pipeline.Pipeline.process`) → :class:`FramePacket` ฝั่ง UI ไม่ได้รับทุกเฟรมผ่าน signal queue แต่ถูกปลุกด้วย
``frame_ready`` แล้วมาหยิบ packet ล่าสุดด้วย :meth:`CameraThread.latest` ถ้า UI วาดไม่ทัน
เฟรมเก่าจะถูกทิ้ง ไม่มี queue สะสมจนภาพหน่วง

ไม่มีการเขียนภาพจากกล้องลงดิสก์ที่ใดในโมดูลนี้
"""

from __future__ import annotations

import logging
import sys
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from PySide6.QtCore import QThread, Signal

from .config import CameraConfig
from .pipeline import FrameResult
from .tracker import HandFrame

__all__ = [
    "CameraError",
    "CameraThread",
    "FpsMeter",
    "FramePacket",
    "FrameSource",
    "OpenCVCamera",
]

log = logging.getLogger(__name__)

READ_FAIL_LIMIT = 30  # อ่านเฟรมไม่ได้ติดกันกี่ครั้งถึงถือว่ากล้องหลุด
READ_RETRY_MS = 20
REOPEN_INTERVAL_S = 2.0


class CameraError(RuntimeError):
    """เปิดกล้องไม่ได้ หรือเปิดได้แต่อ่านภาพไม่ได้"""


class FrameSource(Protocol):
    """แหล่งเฟรม: กล้องจริง, แหล่งจำลองใน :mod:`aether.sources` หรือของปลอมใน test"""

    def open(self) -> str:
        """เปิดแหล่งเฟรม คืนคำอธิบายสั้นๆ เช่น ``"640x480 @ 30fps"`` (raise :class:`CameraError`)"""
        ...

    def read(self) -> np.ndarray | HandFrame | None:
        """เฟรมถัดไป (block จนกว่าจะได้): ภาพ BGR, :class:`HandFrame` สำเร็จรูป (แหล่งจำลอง —
        ไม่ต้องผ่าน tracker) หรือ ``None`` ถ้าอ่านไม่ได้"""
        ...

    def close(self) -> None: ...


class OpenCVCamera:
    """เว็บแคมผ่าน OpenCV (ใช้ DirectShow บน Windows)"""

    def __init__(self, config: CameraConfig) -> None:
        self._config = config
        self._capture = None

    def open(self) -> str:
        import cv2

        cfg = self._config
        backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
        capture = cv2.VideoCapture(cfg.index, backend)
        if not capture.isOpened():
            capture.release()
            raise CameraError(
                f"เปิดกล้อง index {cfg.index} ไม่ได้ — ตรวจว่าเสียบกล้องแล้ว "
                "ไม่มีโปรแกรมอื่นใช้กล้องอยู่ และ Windows อนุญาตให้แอป desktop ใช้กล้อง "
                "(Settings > Privacy & security > Camera) หรือลองเปลี่ยน camera.index ใน config.toml"
            )
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)
        capture.set(cv2.CAP_PROP_FPS, cfg.fps)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # ไม่ให้ driver สะสมเฟรมเก่า (ลด latency)

        ok, frame = capture.read()
        if not ok or frame is None:
            capture.release()
            raise CameraError(f"กล้อง index {cfg.index} เปิดได้แต่อ่านภาพไม่ได้ — อาจมีโปรแกรมอื่นใช้กล้องอยู่")
        height, width = frame.shape[:2]
        if (width, height) != (cfg.width, cfg.height):
            log.warning(
                "camera delivers %dx%d instead of requested %dx%d",
                width, height, cfg.width, cfg.height,
            )  # fmt: skip
        self._capture = capture
        return f"{width}x{height} @ {capture.get(cv2.CAP_PROP_FPS):.0f}fps"

    def read(self) -> np.ndarray | None:
        if self._capture is None:
            return None
        ok, frame = self._capture.read()
        if not ok or frame is None:
            return None
        if self._config.mirror:
            import cv2

            frame = cv2.flip(frame, 1)
        return frame

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None


class FpsMeter:
    """FPS เฉลี่ยจากเวลาของ ``window`` เฟรมล่าสุด"""

    def __init__(self, window: int = 30) -> None:
        if window < 2:
            raise ValueError("window ต้อง ≥ 2")
        self._times: deque[float] = deque(maxlen=window)

    def tick(self, now: float) -> float:
        self._times.append(now)
        return self.fps

    def reset(self) -> None:
        self._times.clear()

    @property
    def fps(self) -> float:
        if len(self._times) < 2:
            return 0.0
        span = self._times[-1] - self._times[0]
        return (len(self._times) - 1) / span if span > 0 else 0.0


@dataclass(frozen=True, slots=True)
class FramePacket:
    """ผลของหนึ่งเฟรมที่ worker ส่งให้ UI"""

    image: np.ndarray | None  # BGR (flip แล้วถ้าเปิด mirror) — None ถ้ามาจากแหล่งจำลอง · ผู้รับห้ามแก้ไข
    result: FrameResult
    captured_at: float  # time.perf_counter() ตอนได้เฟรมจากกล้อง
    process_ms: float  # เวลาตั้งแต่ได้เฟรมจนส่ง input ให้ OS เสร็จ (latency เฟรม→input)
    fps: float


class CameraThread(QThread):
    """Worker thread: อ่านเฟรม → ``processor`` → เผยแพร่ :class:`FramePacket`

    ถ้ากล้องเปิดไม่ได้หรือหลุดระหว่างทาง จะแจ้ง ``camera_error`` แล้วลองเปิดใหม่ทุก
    ``REOPEN_INTERVAL_S`` วินาที ถ้า ``processor`` โยน exception ถือว่าเป็น bug:
    แจ้ง ``failed`` แล้วหยุด thread
    """

    frame_ready = Signal()  # มี packet ใหม่ — เรียก latest() เพื่อหยิบ
    camera_opened = Signal(str)  # คำอธิบายกล้อง เช่น "640x480 @ 30fps"
    camera_error = Signal(str)  # เปิด/อ่านกล้องไม่ได้ (กำลังลองใหม่)
    failed = Signal(str)  # pipeline พัง thread หยุดทำงาน

    def __init__(
        self,
        source: FrameSource,
        processor: Callable[[np.ndarray | HandFrame, int], FrameResult],
        parent=None,
        *,
        observer: Callable[[FramePacket], None] | None = None,
    ) -> None:
        super().__init__(parent)
        self._source = source
        self._processor = processor
        self._observer = observer  # เรียกทุกเฟรมใน worker thread (สถิติ session)
        self._lock = threading.Lock()
        self._latest: FramePacket | None = None
        self._notified = False
        self._last_error: str | None = None

    # ---- ฝั่ง UI thread ------------------------------------------------- #

    def latest(self) -> FramePacket | None:
        """packet ล่าสุด (เรียกจาก slot ของ ``frame_ready``)"""
        with self._lock:
            self._notified = False
            return self._latest

    def stop(self, timeout_ms: int = 3000) -> bool:
        """สั่งหยุดแล้วรอ thread จบ คืน ``False`` ถ้าไม่จบภายในเวลาที่กำหนด"""
        self.requestInterruption()
        return self.wait(timeout_ms)

    # ---- ฝั่ง worker thread --------------------------------------------- #

    def run(self) -> None:
        while not self.isInterruptionRequested():
            try:
                description = self._source.open()
            except CameraError as exc:
                self._report_camera_error(str(exc))
                if self._sleep(REOPEN_INTERVAL_S):
                    break
                continue

            log.info("camera opened: %s", description)
            self._last_error = None
            self.camera_opened.emit(description)
            try:
                keep_going = self._capture_loop()
            finally:
                self._source.close()
            if not keep_going:
                break
            self._report_camera_error("กล้องหยุดส่งภาพ — กำลังลองเชื่อมต่อใหม่")
            if self._sleep(REOPEN_INTERVAL_S):
                break

    def _capture_loop(self) -> bool:
        """วนอ่านเฟรม คืน ``True`` ถ้ากล้องหลุด (ให้ลองเปิดใหม่) ``False`` ถ้าต้องหยุด thread"""
        meter = FpsMeter()
        failures = 0
        while not self.isInterruptionRequested():
            frame = self._source.read()
            captured_at = time.perf_counter()
            if frame is None:
                failures += 1
                if failures >= READ_FAIL_LIMIT:
                    return True
                self.msleep(READ_RETRY_MS)
                continue
            failures = 0

            # ฐานเวลาเดียวกับ pipeline.clock_ms() เพื่อให้ watchdog เทียบกับ timestamp ของเฟรมได้
            timestamp_ms = int(captured_at * 1000)
            try:
                result = self._processor(frame, timestamp_ms)
            except Exception as exc:
                log.exception("frame processor failed")
                self.failed.emit(f"{type(exc).__name__}: {exc}")
                return False
            done_at = time.perf_counter()

            self._publish(
                FramePacket(
                    image=None if isinstance(frame, HandFrame) else frame,
                    result=result,
                    captured_at=captured_at,
                    process_ms=(done_at - captured_at) * 1000.0,
                    fps=meter.tick(captured_at),
                )
            )
        return False

    def _publish(self, packet: FramePacket) -> None:
        with self._lock:
            self._latest = packet
            notify = not self._notified
            self._notified = True
        if notify:
            self.frame_ready.emit()
        if self._observer is not None:
            try:
                self._observer(packet)
            except Exception:
                log.exception("frame observer failed")

    def _report_camera_error(self, message: str) -> None:
        if message != self._last_error:  # ไม่ log ซ้ำทุกรอบที่ลองใหม่
            self._last_error = message
            log.error("%s", message)
            self.camera_error.emit(message)

    def _sleep(self, seconds: float) -> bool:
        """หลับแบบขัดจังหวะได้ คืน ``True`` ถ้าถูกสั่งหยุด"""
        deadline = time.perf_counter() + seconds
        while time.perf_counter() < deadline:
            if self.isInterruptionRequested():
                return True
            self.msleep(50)
        return self.isInterruptionRequested()
