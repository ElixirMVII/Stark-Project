"""ตั้งค่า logging: console (ถ้ามี) + ไฟล์แบบ rotate และจับ exception ที่หลุดมือให้ลงไฟล์

build แบบ windowed (ไม่มี console) ``sys.stdout``/``sys.stderr`` เป็น ``None`` — log ทั้งหมดจึงต้อง
ไปที่ไฟล์ และ exception ที่ไม่มีใครจับต้องถูกเขียนลง log ไม่งั้นแอปจะเงียบหายไปเฉยๆ
"""

from __future__ import annotations

import logging
import logging.handlers
import sys
import threading
from pathlib import Path

__all__ = [
    "BACKUP_COUNT",
    "LOG_FILENAME",
    "MAX_BYTES",
    "configure_logging",
    "install_excepthooks",
]

LOG_FILENAME = "aether.log"
MAX_BYTES = 1_000_000  # ~1 MB ต่อไฟล์
BACKUP_COUNT = 5  # เก็บ aether.log.1 … aether.log.5

_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
_HANDLER_TAG = "_aether_handler"


def configure_logging(debug: bool, log_directory: Path | None) -> Path | None:
    """ตั้ง root logger ใหม่ (เรียกซ้ำได้ — handler เดิมของ AETHER จะถูกแทนที่)

    คืน path ของไฟล์ log หรือ ``None`` ถ้าเขียนไฟล์ไม่ได้ / ไม่ได้ขอ
    """
    for stream in (sys.stdout, sys.stderr):
        if stream is not None and hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")  # console ที่ encode ภาษาไทยไม่ได้ต้องไม่ทำให้แอปล้ม

    root = logging.getLogger()
    for handler in list(root.handlers):
        if getattr(handler, _HANDLER_TAG, False):
            root.removeHandler(handler)
            handler.close()
    root.setLevel(logging.DEBUG if debug else logging.INFO)
    # matplotlib ถูก import โดย mediapipe (drawing_utils) และ log เรื่อง font cache ที่ระดับ INFO
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    handlers: list[logging.Handler] = []
    if sys.stderr is not None:
        console = logging.StreamHandler(sys.stderr)
        console.setFormatter(logging.Formatter(_FORMAT, datefmt="%H:%M:%S"))
        handlers.append(console)

    log_file: Path | None = None
    if log_directory is not None:
        try:
            log_directory.mkdir(parents=True, exist_ok=True)
            log_file = log_directory / LOG_FILENAME
            file_handler = logging.handlers.RotatingFileHandler(
                log_file, maxBytes=MAX_BYTES, backupCount=BACKUP_COUNT, encoding="utf-8"
            )
            file_handler.setFormatter(logging.Formatter(_FORMAT))
            handlers.append(file_handler)
        except OSError as exc:
            log_file = None
            if sys.stderr is not None:
                print(f"เขียนไฟล์ log ใน {log_directory} ไม่ได้: {exc}", file=sys.stderr)

    for handler in handlers:
        setattr(handler, _HANDLER_TAG, True)
        root.addHandler(handler)
    return log_file


def install_excepthooks() -> None:
    """exception ที่ไม่มีใครจับ (ทั้ง main thread และ thread อื่น) → เขียนลง log"""
    log = logging.getLogger("aether")

    def main_hook(exc_type, exc, tb) -> None:
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc, tb)
            return
        log.critical("unhandled exception", exc_info=(exc_type, exc, tb))

    def thread_hook(args: threading.ExceptHookArgs) -> None:
        if args.exc_type is SystemExit:
            return
        name = args.thread.name if args.thread is not None else "?"
        log.critical(
            "unhandled exception in thread %s",
            name,
            exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
        )

    sys.excepthook = main_hook
    threading.excepthook = thread_hook
