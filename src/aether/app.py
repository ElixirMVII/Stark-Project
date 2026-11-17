"""QApplication และการเชื่อม signal ระหว่าง worker thread, hotkey และ UI

สถานะตอนนี้ (M2): กล้อง → tracker → gesture engine → input ของ OS, แสดงผลด้วย HUD overlay
(และหน้าต่าง debug ถ้าใส่ ``--debug``)

Thread ที่เกี่ยวข้อง:

- **worker** (:class:`~aether.camera.CameraThread`): อ่านกล้อง จับมือ ตัดสินท่า และส่ง input ให้ OS
- **hotkey**: สลับ ARM/DISARM ได้ทันทีโดยไม่รอ UI
- **UI** (main): วาด HUD / debug view เล่นเสียง และรัน watchdog ที่ปล่อยปุ่มเมาส์ถ้ากล้องค้าง
"""

from __future__ import annotations

import dataclasses
import logging
import signal
import sys
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QObject, QTimer, Signal
from PySide6.QtWidgets import QApplication

from . import __version__
from .actions import ActionExecutor, InputBackend, create_backend, double_click_time_ms
from .camera import CameraThread, FramePacket, FrameSource, OpenCVCamera
from .config import Config, ConfigError, assets_dir, load_config, model_path
from .debug_view import DebugView
from .gestures import EventKind, GestureEngine, GestureEvent
from .hotkeys import HotkeyManager, parse_hotkey
from .hud.overlay import HudOverlay
from .hud.theme import Theme
from .mapping import ScreenGeometry, primary_screen_geometry
from .pipeline import Pipeline
from .sound import SoundPlayer
from .tracker import HandTracker

__all__ = ["AetherApp", "Options", "run"]

log = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_SETUP_ERROR = 2  # config ผิด หรือ asset ไม่ครบ

HEADLESS_STATS_INTERVAL_MS = 5000
WATCHDOG_INTERVAL_MS = 100


@dataclass(frozen=True, slots=True)
class Options:
    """ตัวเลือกจาก command line"""

    debug: bool = False
    dry_run: bool = False
    no_hud: bool = False
    config_path: Path | None = None


class AetherApp(QObject):
    """เป็นเจ้าของ pipeline, worker thread, hotkey และหน้าต่างทั้งหมด"""

    gesture_events = Signal(object)  # tuple[GestureEvent, ...] ที่ไม่ใช่การขยับเคอร์เซอร์
    armed_changed = Signal(bool)
    quit_requested = Signal()

    def __init__(
        self,
        config: Config,
        options: Options,
        tracker: HandTracker,
        *,
        source: FrameSource | None = None,
        backend: InputBackend | None = None,
        screen: ScreenGeometry | None = None,
        hotkeys: HotkeyManager | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._config = config
        self._options = options
        self._tracker = tracker
        self._packet: FramePacket | None = None

        if backend is None:
            backend, self._dry_run = create_backend(options.dry_run)
        else:
            self._dry_run = options.dry_run
        self._executor = ActionExecutor(backend, dataclasses.asdict(config.actions))
        screen = screen or primary_screen_geometry()
        self._engine = GestureEngine(config, screen, double_click_ms=double_click_time_ms())
        # event มาจาก worker/hotkey thread → ส่งต่อเป็น Qt signal เพื่อให้ไปถึง UI thread
        self._pipeline = Pipeline(
            tracker.process, self._engine, self._executor, on_events=self.gesture_events.emit
        )
        self.gesture_events.connect(self._on_gesture_events)
        self.quit_requested.connect(QApplication.quit)

        self._thread = CameraThread(
            source or OpenCVCamera(config.camera), self._pipeline.process, self
        )
        self._thread.frame_ready.connect(self._on_frame_ready)
        self._thread.camera_opened.connect(self._on_camera_opened)
        self._thread.camera_error.connect(self._on_camera_error)
        self._thread.failed.connect(self._on_failed)

        self._hotkeys = hotkeys or HotkeyManager(
            {
                config.hotkeys.toggle_arm: self.toggle_armed,
                config.hotkeys.quit: self.quit_requested.emit,
            }
        )

        self._debug_view: DebugView | None = None
        if options.debug:
            self._debug_view = DebugView(
                config,
                dry_run=self._dry_run,
                arm_hotkey=str(parse_hotkey(config.hotkeys.toggle_arm)),
            )
            # ยังไม่มี tray (M4) — ปิดหน้าต่าง debug = ปิดแอป
            self._debug_view.closed.connect(QApplication.quit)
            self.armed_changed.connect(self._debug_view.set_armed)

        self._hud: HudOverlay | None = None
        if config.hud.enabled and not options.no_hud:
            sounds = SoundPlayer(
                assets_dir() / "sounds", enabled=config.hud.sounds, volume=config.hud.volume
            )
            self._hud = HudOverlay(
                config,
                screen,
                dry_run=self._dry_run,
                theme=Theme.from_config(config.hud, assets_dir() / "fonts"),
                sounds=sounds,
            )

        # ถ้ากล้องค้าง worker จะปล่อยปุ่มเมาส์เองไม่ได้ จึงให้ UI thread คอยเช็ก
        self._watchdog = QTimer(self)
        self._watchdog.setInterval(WATCHDOG_INTERVAL_MS)
        self._watchdog.timeout.connect(self._pipeline.check_stall)

        self._stats_timer = QTimer(self)
        self._stats_timer.setInterval(HEADLESS_STATS_INTERVAL_MS)
        self._stats_timer.timeout.connect(self._log_stats)

    @property
    def debug_view(self) -> DebugView | None:
        return self._debug_view

    @property
    def hud(self) -> HudOverlay | None:
        return self._hud

    @property
    def latest_packet(self) -> FramePacket | None:
        return self._packet

    @property
    def pipeline(self) -> Pipeline:
        return self._pipeline

    @property
    def dry_run(self) -> bool:
        return self._dry_run

    def toggle_armed(self) -> bool:
        """สลับ ARM/DISARM (เรียกจาก thread ใดก็ได้ มีผลทันที)"""
        return self._pipeline.toggle_armed()

    def start(self) -> None:
        arm_hotkey = parse_hotkey(self._config.hotkeys.toggle_arm)
        quit_hotkey = parse_hotkey(self._config.hotkeys.quit)
        failed = self._hotkeys.start()
        for hotkey in failed:
            log.error(
                "ลงทะเบียน hotkey %s ไม่ได้ (อาจมีโปรแกรมอื่นใช้อยู่) — เปลี่ยนได้ที่ [hotkeys] ใน config.toml",
                hotkey,
            )
        log.info(
            "เริ่มในสถานะ DISARMED — กด %s เพื่อ ARM/DISARM, %s เพื่อปิดแอป%s",
            arm_hotkey,
            quit_hotkey,
            " [dry-run: ไม่ส่ง input จริง]" if self._dry_run else "",
        )

        if self._hud is not None:
            self._hud.start()
        if self._debug_view is not None:
            self._debug_view.show()
        if self._hud is None and self._debug_view is None:
            self._stats_timer.start()  # ไม่มีอะไรแสดงผล: log สถานะลง console แทน
        self._watchdog.start()
        self._thread.start()

    def shutdown(self) -> None:
        """ปล่อยปุ่มเมาส์ก่อนเสมอ แล้วค่อยหยุด worker และปิด tracker"""
        self._watchdog.stop()
        self._stats_timer.stop()
        self._hotkeys.stop()
        self._pipeline.shutdown()
        if self._thread.stop():
            self._tracker.close()
        else:
            log.warning("camera thread did not stop in time; skipping tracker cleanup")
        if self._hud is not None:
            self._hud.stop()
        if self._debug_view is not None:
            self._debug_view.closed.disconnect(QApplication.quit)
            self._debug_view.close()

    # ---- slots (UI thread) ---------------------------------------------- #

    def _on_frame_ready(self) -> None:
        packet = self._thread.latest()
        if packet is None:
            return
        self._packet = packet
        if self._hud is not None:
            self._hud.show_packet(packet)
        if self._debug_view is not None and self._debug_view.isVisible():
            self._debug_view.show_packet(packet)

    def _on_gesture_events(self, events: tuple[GestureEvent, ...]) -> None:
        if self._hud is not None:
            self._hud.handle_events(events)  # shockwave + เสียง + ป้ายสถานะ
        for event in events:
            if event.kind is EventKind.ARMED:
                log.info("ARMED — มือคุมเมาส์แล้ว")
                self.armed_changed.emit(True)
            elif event.kind is EventKind.DISARMED:
                log.info("DISARMED")
                self.armed_changed.emit(False)
            else:
                log.debug("event: %s", event)

    def _on_camera_opened(self, description: str) -> None:
        if self._hud is not None:
            self._hud.set_alert(None)
        if self._debug_view is not None:
            self._debug_view.set_camera_info(description)

    def _on_camera_error(self, message: str) -> None:
        if self._hud is not None:
            self._hud.set_alert("CAMERA LOST")
        if self._debug_view is not None:
            self._debug_view.show_message(message)

    def _on_failed(self, message: str) -> None:
        log.error("pipeline stopped: %s", message)
        self._pipeline.shutdown()
        if self._hud is not None:
            self._hud.set_alert("PIPELINE ERROR")
        if self._debug_view is not None:
            self._debug_view.show_message(f"pipeline หยุดทำงาน\n{message}\n(ดูรายละเอียดใน console)")
        elif self._hud is None:
            QApplication.exit(1)

    def _log_stats(self) -> None:
        packet = self._packet
        if packet is None:
            log.info("waiting for camera…")
            return
        snapshot = packet.result.snapshot
        log.info(
            "fps=%.1f latency=%.1fms state=%s hands=%d",
            packet.fps,
            packet.process_ms,
            snapshot.state.value,
            snapshot.hands,
        )


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
