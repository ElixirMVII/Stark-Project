"""QApplication, system tray และการเชื่อม signal ระหว่าง worker thread, hotkey และ UI

- :class:`AetherApp` — หนึ่งชุดของการทำงานตาม config หนึ่งชุด: กล้อง → tracker → gesture engine →
  input ของ OS, HUD, หน้าต่าง debug, hotkey
- :class:`AppController` — อยู่ตลอดอายุของ process: tray, calibration wizard, โหลด config ใหม่
  (สร้าง :class:`AetherApp` ชุดใหม่แทนชุดเดิม)

Thread ที่เกี่ยวข้อง:

- **worker** (:class:`~aether.camera.CameraThread`): อ่านกล้อง จับมือ ตัดสินท่า และส่ง input ให้ OS
- **hotkey**: สลับ ARM/DISARM ได้ทันทีโดยไม่รอ UI
- **UI** (main): HUD / debug view / tray / wizard เล่นเสียง และรัน watchdog ที่ปล่อยปุ่มเมาส์ถ้ากล้องค้าง
"""

from __future__ import annotations

import dataclasses
import logging
import signal
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QObject, QTimer, Signal
from PySide6.QtWidgets import QApplication, QMessageBox

from . import __version__
from .actions import ActionExecutor, InputBackend, create_backend, double_click_time_ms
from .calibration_wizard import CalibrationWizard
from .camera import CameraThread, FramePacket, FrameSource, OpenCVCamera
from .config import (
    Config,
    ConfigError,
    assets_dir,
    calibration_path,
    load_config,
    log_dir,
    model_path,
)
from .debug_view import DebugView
from .gestures import EventKind, GestureEngine, GestureEvent
from .hotkeys import HotkeyManager, parse_hotkey
from .hud.overlay import HudOverlay
from .hud.theme import Theme
from .logsetup import configure_logging, install_excepthooks
from .mapping import ScreenGeometry, primary_screen_geometry
from .pipeline import Pipeline
from .sound import SOUND_NAMES, SoundPlayer
from .tracker import HandTracker
from .tray import Tray

__all__ = ["AetherApp", "AppController", "Options", "run", "run_check"]

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
    calibrate: bool = False
    no_calibration: bool = False  # ไม่ใช้ค่าใน calibration.json


class AetherApp(QObject):
    """เป็นเจ้าของ pipeline, worker thread, hotkey, HUD และหน้าต่าง debug ของ config หนึ่งชุด"""

    gesture_events = Signal(object)  # tuple[GestureEvent, ...] ที่ไม่ใช่การขยับเคอร์เซอร์
    armed_changed = Signal(bool)
    frame_packet = Signal(object)  # FramePacket ล่าสุด (UI thread) — ใช้ป้อน calibration wizard
    debug_visibility_changed = Signal(bool)
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
        quit_on_debug_close: bool = True,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._config = config
        self._options = options
        self._tracker = tracker
        self._packet: FramePacket | None = None
        self._quit_on_debug_close = quit_on_debug_close

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
        self._camera_info = ""
        self._camera_message: str | None = None

        self._hotkeys = hotkeys or HotkeyManager(
            {
                config.hotkeys.toggle_arm: self.toggle_armed,
                config.hotkeys.quit: self.quit_requested.emit,
            }
        )

        self._debug_view: DebugView | None = None
        if options.debug:
            self._ensure_debug_view()

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

    # ---- properties ----------------------------------------------------- #

    @property
    def config(self) -> Config:
        return self._config

    @property
    def debug_view(self) -> DebugView | None:
        return self._debug_view

    @property
    def debug_visible(self) -> bool:
        return self._debug_view is not None and self._debug_view.isVisible()

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

    # ---- control -------------------------------------------------------- #

    def toggle_armed(self) -> bool:
        """สลับ ARM/DISARM (เรียกจาก thread ใดก็ได้ มีผลทันที)"""
        return self._pipeline.toggle_armed()

    def set_debug_visible(self, visible: bool) -> None:
        """เปิด/ปิดหน้าต่าง debug (สร้างครั้งแรกเมื่อเปิด)"""
        if visible:
            view = self._ensure_debug_view()
            if self._packet is not None:
                view.show_packet(self._packet)
            view.show()
            view.raise_()
        elif self._debug_view is not None:
            self._debug_view.hide()
        self.debug_visibility_changed.emit(self.debug_visible)

    def start(self, *, boot: bool = True) -> None:
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
            self._hud.start(boot=boot)
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
            self._hud.deleteLater()
        if self._debug_view is not None:
            self._debug_view.closed.disconnect(self._on_debug_closed)
            self._debug_view.close()
            self._debug_view.deleteLater()

    # ---- internal ------------------------------------------------------- #

    def _ensure_debug_view(self) -> DebugView:
        if self._debug_view is None:
            view = DebugView(
                self._config,
                dry_run=self._dry_run,
                arm_hotkey=str(parse_hotkey(self._config.hotkeys.toggle_arm)),
            )
            view.closed.connect(self._on_debug_closed)
            self.armed_changed.connect(view.set_armed)
            view.set_armed(self._pipeline.armed)
            if self._camera_info:
                view.set_camera_info(self._camera_info)
            if self._camera_message:
                view.show_message(self._camera_message)
            self._debug_view = view
        return self._debug_view

    def _on_debug_closed(self) -> None:
        if self._quit_on_debug_close:
            QApplication.quit()  # ไม่มี tray: ปิดหน้าต่าง debug = ปิดแอป
        else:
            self.debug_visibility_changed.emit(False)

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
        self.frame_packet.emit(packet)

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
            elif event.kind is EventKind.ACTION:
                log.info("gesture action: %s", event.name)
            else:
                log.debug("event: %s", event)

    def _on_camera_opened(self, description: str) -> None:
        self._camera_info, self._camera_message = description, None
        if self._hud is not None:
            self._hud.set_alert(None)
        if self._debug_view is not None:
            self._debug_view.set_camera_info(description)

    def _on_camera_error(self, message: str) -> None:
        self._camera_message = message
        if self._hud is not None:
            self._hud.set_alert("CAMERA LOST")
        if self._debug_view is not None:
            self._debug_view.show_message(message)

    def _on_failed(self, message: str) -> None:
        log.error("pipeline stopped: %s", message)
        self._pipeline.shutdown()
        self._camera_message = f"pipeline หยุดทำงาน\n{message}\n(ดูรายละเอียดใน log)"
        if self._hud is not None:
            self._hud.set_alert("PIPELINE ERROR")
        if self._debug_view is not None:
            self._debug_view.show_message(self._camera_message)
        elif self._hud is None and self._quit_on_debug_close:
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


AppFactory = Callable[[Config, Options, bool], AetherApp]


def _default_app_factory(config: Config, options: Options, quit_on_debug_close: bool) -> AetherApp:
    tracker = HandTracker(
        model_path(), num_hands=config.tracking.num_hands, mirrored=config.camera.mirror
    )
    return AetherApp(config, options, tracker, quit_on_debug_close=quit_on_debug_close)


class AppController(QObject):
    """อยู่ตลอดอายุของ process: เป็นเจ้าของ tray, wizard และ :class:`AetherApp` ที่ทำงานอยู่

    โหลด config ใหม่ = ปิด :class:`AetherApp` ชุดเดิม (ปล่อยปุ่มเมาส์ก่อนเสมอ) แล้วสร้างชุดใหม่ซึ่ง
    เริ่มแบบ DISARMED — ถ้า config ใหม่ผิด ชุดเดิมทำงานต่อและแจ้งเตือนผ่าน tray
    """

    reloaded = Signal(bool)  # True = ใช้ config ใหม่แล้ว, False = config ผิด ใช้ชุดเดิมต่อ

    def __init__(
        self,
        options: Options,
        *,
        app_factory: AppFactory | None = None,
        config_loader: Callable[[], Config] | None = None,
        calibration_file: Path | None = None,
        use_tray: bool | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._options = options
        self._app_factory = app_factory or _default_app_factory
        self._config_loader = config_loader or (
            lambda: load_config(options.config_path, use_calibration=not options.no_calibration)
        )
        self._calibration_file = calibration_file or calibration_path()
        self._app: AetherApp | None = None
        self._wizard: CalibrationWizard | None = None
        self._debug_wanted = options.debug

        tray_ok = Tray.available() if use_tray is None else use_tray
        self._tray: Tray | None = Tray() if tray_ok else None
        if self._tray is not None:
            self._tray.toggle_arm_requested.connect(self._toggle_armed)
            self._tray.debug_requested.connect(self._set_debug)
            self._tray.calibrate_requested.connect(self.open_calibration)
            self._tray.reload_requested.connect(self.reload)
            self._tray.quit_requested.connect(QApplication.quit)

    # ---- properties ----------------------------------------------------- #

    @property
    def app(self) -> AetherApp | None:
        return self._app

    @property
    def tray(self) -> Tray | None:
        return self._tray

    @property
    def wizard(self) -> CalibrationWizard | None:
        return self._wizard

    # ---- lifecycle ------------------------------------------------------ #

    def start(self, config: Config) -> None:
        self._launch(config, boot=True)
        if self._tray is not None:
            self._tray.show()
        if self._options.calibrate:
            self.open_calibration()

    def shutdown(self) -> None:
        if self._wizard is not None:
            wizard, self._wizard = self._wizard, None
            wizard.finished.disconnect(self._on_calibration_finished)
            wizard.close()
        if self._app is not None:
            self._app.shutdown()
            self._app.deleteLater()
            self._app = None
        if self._tray is not None:
            self._tray.hide()

    def reload(self) -> bool:
        """โหลด config (+ calibration) ใหม่ แล้วเริ่มชุดใหม่ คืน ``True`` ถ้าสำเร็จ"""
        try:
            config = self._config_loader()
        except ConfigError as exc:
            log.error("โหลด config ใหม่ไม่ได้ — ใช้ค่าเดิมต่อ: %s", exc)
            self._notify("config ผิด — ใช้ค่าเดิมต่อ", str(exc), error=True)
            self.reloaded.emit(False)
            return False
        if self._app is not None:
            self._debug_wanted = self._app.debug_visible
            self._app.shutdown()
            self._app.deleteLater()
            self._app = None
        self._launch(config, boot=False)
        log.info("โหลด config ใหม่แล้ว (เริ่มในสถานะ DISARMED)")
        self._notify("โหลด config ใหม่แล้ว", "เริ่มในสถานะ DISARMED")
        self.reloaded.emit(True)
        return True

    def open_calibration(self) -> None:
        """เปิด calibration wizard (มือหยุดคุมเมาส์ระหว่างนั้น)"""
        if self._app is None:
            return
        if self._wizard is not None:
            self._wizard.show()
            self._wizard.raise_()
            return
        self._app.pipeline.set_armed(False)
        wizard = CalibrationWizard(self._app.config, self._calibration_file)
        wizard.finished.connect(self._on_calibration_finished)
        self._wizard = wizard
        wizard.show()
        log.info("เปิด calibration wizard")

    # ---- internal ------------------------------------------------------- #

    def _launch(self, config: Config, *, boot: bool) -> None:
        options = dataclasses.replace(self._options, debug=self._debug_wanted)
        app = self._app_factory(config, options, self._tray is None)
        app.armed_changed.connect(self._on_armed_changed)
        app.debug_visibility_changed.connect(self._on_debug_visibility)
        app.frame_packet.connect(self._on_frame_packet)
        self._app = app
        if self._tray is not None:
            self._tray.set_primary(config.hud.primary)
            self._tray.set_armed(False)
            self._tray.set_debug_checked(self._debug_wanted)
        app.start(boot=boot)

    def _toggle_armed(self) -> None:
        if self._app is not None:
            self._app.toggle_armed()

    def _set_debug(self, visible: bool) -> None:
        self._debug_wanted = visible
        if self._app is not None:
            self._app.set_debug_visible(visible)

    def _on_debug_visibility(self, visible: bool) -> None:
        self._debug_wanted = visible
        if self._tray is not None:
            self._tray.set_debug_checked(visible)

    def _on_armed_changed(self, armed: bool) -> None:
        if armed and self._wizard is not None and self._app is not None:
            log.info("กำลัง calibrate — ยังไม่เปิดการคุมเมาส์ด้วยมือ")
            self._app.pipeline.set_armed(False)
            return
        if self._tray is not None:
            self._tray.set_armed(armed)

    def _on_frame_packet(self, packet: FramePacket) -> None:
        if self._wizard is not None:
            self._wizard.show_frame(packet.result.hands)

    def _on_calibration_finished(self, saved: bool) -> None:
        self._wizard = None
        if saved:
            self.reload()

    def _notify(self, title: str, message: str, *, error: bool = False) -> None:
        if self._tray is not None:
            self._tray.notify(title, message, error=error)


# --------------------------------------------------------------------------- #
# Entry points
# --------------------------------------------------------------------------- #


def _report(title: str, text: str, *, error: bool) -> None:
    """แสดงข้อความให้ผู้ใช้เห็น: พิมพ์ลง console ถ้ามี ไม่งั้นเป็นกล่องข้อความ (build แบบ windowed)"""
    stream = sys.stderr if error else sys.stdout
    if stream is not None:
        print(text, file=stream)
        return
    _ = QApplication.instance() or QApplication(sys.argv[:1])  # QMessageBox ต้องมี QApplication
    box = QMessageBox.critical if error else QMessageBox.information
    box(None, title, text)


def run_check(options: Options) -> int:
    """``--check``: ตรวจว่า asset ครบและโมเดลโหลดได้ (ไม่ต้องใช้กล้อง) — ใช้ตรวจ build"""
    lines = [f"AETHER {__version__}"]
    ok = True
    try:
        config = load_config(options.config_path, use_calibration=not options.no_calibration)
        lines.append("✓ config.toml อ่านได้")
    except ConfigError as exc:
        config = None
        ok = False
        lines.append(f"✗ config ผิด: {exc}")

    model = model_path()
    if model.is_file():
        try:
            import numpy as np

            with HandTracker(model, num_hands=2, mirrored=True) as tracker:
                frame = tracker.process(np.zeros((480, 640, 3), dtype=np.uint8), 0)
            lines.append(f"✓ โหลดโมเดลและประมวลผลเฟรมได้ ({len(frame.hands)} มือในเฟรมว่าง)")
        except Exception as exc:  # noqa: BLE001 — รายงานทุกอย่างที่ทำให้โมเดลใช้ไม่ได้
            ok = False
            lines.append(f"✗ โหลดโมเดลไม่ได้: {type(exc).__name__}: {exc}")
    else:
        ok = False
        lines.append(f"✗ ไม่พบโมเดล {model}")

    fonts = assets_dir() / "fonts"
    missing_fonts = [
        f for f in ("Orbitron.ttf", "ShareTechMono-Regular.ttf") if not (fonts / f).is_file()
    ]
    lines.append(
        "✓ ฟอนต์ครบ"
        if not missing_fonts
        else f"! ไม่มีฟอนต์ {', '.join(missing_fonts)} (ใช้ฟอนต์ระบบแทน)"
    )
    sounds = assets_dir() / "sounds"
    present = [n for n in SOUND_NAMES if (sounds / f"{n}.wav").is_file()]
    lines.append(
        f"✓ เสียงครบ {len(present)} ไฟล์"
        if len(present) == len(SOUND_NAMES)
        else f"! มีเสียง {len(present)}/{len(SOUND_NAMES)} ไฟล์ (ไม่มีเสียงก็ใช้งานได้)"
    )
    calibration = calibration_path()
    if options.no_calibration or config is None:
        pass
    elif calibration.is_file():
        lines.append(f"✓ ใช้ค่า calibrate จาก {calibration}")
    else:
        lines.append("· ยังไม่ได้ calibrate (ใช้ค่าใน config.toml)")
    lines.append("ผลรวม: พร้อมใช้งาน" if ok else "ผลรวม: ยังใช้งานไม่ได้")
    _report("AETHER — check", "\n".join(lines), error=not ok)
    return EXIT_OK if ok else EXIT_SETUP_ERROR


def run(options: Options) -> int:
    """จุดเริ่มของแอป คืน exit code"""
    log_file = configure_logging(options.debug, log_dir())
    install_excepthooks()

    try:
        config = load_config(options.config_path, use_calibration=not options.no_calibration)
    except ConfigError as exc:
        log.error("config ผิด: %s", exc)
        _report("AETHER — config ผิด", f"config ผิด:\n{exc}", error=True)
        return EXIT_SETUP_ERROR

    model = model_path()
    if not model.is_file():
        message = f"ไม่พบโมเดล {model}\nรัน `python tools/download_assets.py` ก่อน"
        log.error("%s", message.replace("\n", " — "))
        _report("AETHER — ไม่พบโมเดล", message, error=True)
        return EXIT_SETUP_ERROR

    log.info(
        "AETHER %s%s%s — log: %s",
        __version__,
        " [dry-run]" if options.dry_run else "",
        " [debug]" if options.debug else "",
        log_file or "(console only)",
    )

    app = QApplication.instance() or QApplication(sys.argv[:1])
    app.setApplicationName("AETHER")
    app.setQuitOnLastWindowClosed(False)

    controller = AppController(options)
    if controller.tray is None:
        log.info("ไม่มี system tray บนเครื่องนี้ — ปิดแอปด้วย hotkey หรือปิดหน้าต่าง debug")

    # ให้ Ctrl+C ใน console ปิดแอปได้: Qt event loop ต้องคืนการควบคุมให้ Python เป็นระยะ
    signal.signal(signal.SIGINT, lambda *_: app.quit())
    keepalive = QTimer()
    keepalive.setInterval(200)
    keepalive.timeout.connect(lambda: None)
    keepalive.start()

    controller.start(config)
    try:
        return app.exec()
    finally:
        keepalive.stop()
        controller.shutdown()
        log.info("ปิดแอปแล้ว")
