"""QApplication และการเชื่อม signal ระหว่าง worker thread กับ UI

สถานะตอนนี้ (M0): กล้อง → tracker → debug view ยังไม่มีการส่ง input ให้ OS และยังไม่มี HUD
"""

from __future__ import annotations

import logging
import signal
import sys
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QObject, QTimer
from PySide6.QtWidgets import QApplication

from . import __version__
from .camera import CameraThread, FramePacket, FrameSource, OpenCVCamera
from .config import Config, ConfigError, load_config, model_path
from .debug_view import DebugView
from .tracker import HandTracker

__all__ = ["AetherApp", "Options", "run"]

log = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_SETUP_ERROR = 2  # config ผิด หรือ asset ไม่ครบ

HEADLESS_STATS_INTERVAL_MS = 5000


@dataclass(frozen=True, slots=True)
class Options:
    """ตัวเลือกจาก command line"""

    debug: bool = False
    dry_run: bool = False
    no_hud: bool = False
    config_path: Path | None = None


class AetherApp(QObject):
    """เป็นเจ้าของ worker thread และหน้าต่างทั้งหมด"""

    def __init__(
        self,
        config: Config,
        options: Options,
        tracker: HandTracker,
        source: FrameSource | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._config = config
        self._options = options
        self._tracker = tracker
        self._packet: FramePacket | None = None

        self._thread = CameraThread(source or OpenCVCamera(config.camera), tracker.process, self)
        self._thread.frame_ready.connect(self._on_frame_ready)
        self._thread.camera_opened.connect(self._on_camera_opened)
        self._thread.camera_error.connect(self._on_camera_error)
        self._thread.failed.connect(self._on_failed)

        self._debug_view: DebugView | None = None
        if options.debug:
            self._debug_view = DebugView(config, dry_run=options.dry_run)
            # ยังไม่มี tray (M4) — ปิดหน้าต่าง debug = ปิดแอป
            self._debug_view.closed.connect(QApplication.quit)

        self._stats_timer = QTimer(self)
        self._stats_timer.setInterval(HEADLESS_STATS_INTERVAL_MS)
        self._stats_timer.timeout.connect(self._log_stats)

    @property
    def debug_view(self) -> DebugView | None:
        return self._debug_view

    @property
    def latest_packet(self) -> FramePacket | None:
        return self._packet

    def start(self) -> None:
        if self._debug_view is not None:
            self._debug_view.show()
        else:
            log.info("no --debug window; logging stats every %d s (Ctrl+C to quit)",
                     HEADLESS_STATS_INTERVAL_MS // 1000)  # fmt: skip
            self._stats_timer.start()
        self._thread.start()

    def shutdown(self) -> None:
        """หยุด worker แล้วค่อยปิด tracker (ห้ามปิด tracker ขณะ worker ยังใช้อยู่)"""
        self._stats_timer.stop()
        if self._thread.stop():
            self._tracker.close()
        else:
            log.warning("camera thread did not stop in time; skipping tracker cleanup")
        if self._debug_view is not None:
            self._debug_view.closed.disconnect(QApplication.quit)
            self._debug_view.close()

    # ---- slots (UI thread) ---------------------------------------------- #

    def _on_frame_ready(self) -> None:
        packet = self._thread.latest()
        if packet is None:
            return
        self._packet = packet
        if self._debug_view is not None and self._debug_view.isVisible():
            self._debug_view.show_packet(packet)

    def _on_camera_opened(self, description: str) -> None:
        if self._debug_view is not None:
            self._debug_view.set_camera_info(description)

    def _on_camera_error(self, message: str) -> None:
        if self._debug_view is not None:
            self._debug_view.show_message(message)

    def _on_failed(self, message: str) -> None:
        log.error("pipeline stopped: %s", message)
        if self._debug_view is not None:
            self._debug_view.show_message(f"pipeline หยุดทำงาน\n{message}\n(ดูรายละเอียดใน console)")
        else:
            QApplication.exit(1)

    def _log_stats(self) -> None:
        packet = self._packet
        if packet is None:
            log.info("waiting for camera…")
            return
        gestures = ", ".join(f"{h.handedness}:{h.gesture}" for h in packet.hands.hands) or "-"
        log.info("fps=%.1f process=%.1fms hands=%s", packet.fps, packet.process_ms, gestures)


def _setup_logging(debug: bool) -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):  # console ที่ encode ภาษาไทยไม่ได้ต้องไม่ทำให้แอปล้ม
            stream.reconfigure(errors="replace")
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def run(options: Options) -> int:
    """จุดเริ่มของแอป คืน exit code"""
    _setup_logging(options.debug)

    try:
        config = load_config(options.config_path)
    except ConfigError as exc:
        log.error("config ผิด: %s", exc)
        return EXIT_SETUP_ERROR

    model = model_path()
    if not model.is_file():
        log.error("ไม่พบโมเดล %s — รัน `python tools/download_assets.py` ก่อน", model)
        return EXIT_SETUP_ERROR

    log.info(
        "AETHER %s%s%s",
        __version__,
        " [dry-run]" if options.dry_run else "",
        " [debug]" if options.debug else "",
    )

    app = QApplication.instance() or QApplication(sys.argv[:1])
    app.setApplicationName("AETHER")
    app.setQuitOnLastWindowClosed(False)

    tracker = HandTracker(
        model,
        num_hands=config.tracking.num_hands,
        mirrored=config.camera.mirror,
    )
    aether = AetherApp(config, options, tracker)

    # ให้ Ctrl+C ใน console ปิดแอปได้: Qt event loop ต้องคืนการควบคุมให้ Python เป็นระยะ
    signal.signal(signal.SIGINT, lambda *_: app.quit())
    keepalive = QTimer()
    keepalive.setInterval(200)
    keepalive.timeout.connect(lambda: None)
    keepalive.start()

    aether.start()
    try:
        return app.exec()
    finally:
        keepalive.stop()
        aether.shutdown()
