"""QApplication, system tray และการเชื่อม signal ระหว่าง worker thread, hotkey และ UI

- :class:`AetherApp` — หนึ่งชุดของการทำงานตาม config หนึ่งชุด: กล้อง → tracker → gesture engine →
  input ของ OS, HUD, หน้าต่าง debug, hotkey
- :class:`AppController` — อยู่ตลอดอายุของ process: tray, calibration wizard, หน้าต่างสอนท่ามือ,
  โหลด config ใหม่ (สร้าง :class:`AetherApp` ชุดใหม่แทนชุดเดิม)

Thread ที่เกี่ยวข้อง:

- **worker** (:class:`~aether.camera.CameraThread`): อ่านกล้อง จับมือ ตัดสินท่า และส่ง input ให้ OS
- **hotkey**: สลับ ARM/DISARM ได้ทันทีโดยไม่รอ UI
- **UI** (main): HUD / debug view / tray / wizard เล่นเสียง และรัน watchdog ที่ปล่อยปุ่มเมาส์ถ้ากล้องค้าง
"""

from __future__ import annotations

import dataclasses
import logging
import os
import signal
import sys
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QObject, QTimer, Signal
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QApplication, QMessageBox

from . import __version__
from .actions import ActionExecutor, InputBackend, create_backend, double_click_time_ms
from .calibration_wizard import CalibrationWizard
from .camera import CameraThread, FramePacket, FrameSource
from .config import (
    Config,
    ConfigError,
    app_root,
    assets_dir,
    bundle_root,
    calibration_path,
    load_config,
    log_dir,
    model_path,
    poses_path,
)
from .debug_view import DebugView
from .gestures import EventKind, GestureEngine, GestureEvent
from .health import LightMonitor, webcam_users
from .hotkeys import HotkeyManager, parse_hotkey
from .hud.overlay import HudGroup, HudOverlay
from .hud.theme import Theme
from .incidents import FlightRecorder, SessionStats, incident_dir, write_incident
from .logsetup import configure_logging, install_excepthooks
from .mapping import (
    Desktop,
    ScreenArea,
    list_monitors,
    match_qt_screens,
    primary_screen_geometry,
)
from .pipeline import Pipeline
from .pose_wizard import PoseWizard
from .poses import PoseClassifier, PoseFileError, PoseLibrary, load_poses
from .sound import SOUND_NAMES, SoundPlayer
from .sources import CAMERA_SOURCE, NullTracker, SourceSpec, create_source
from .targeting import TargetFinder
from .tracker import HandTracker
from .tray import Tray
from .tuning import apply_overrides, current_values, save_overrides

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
    teach: bool = False  # เปิดหน้าต่างสอนท่ามือตอนเริ่ม
    source: SourceSpec = CAMERA_SOURCE  # แหล่งเฟรม: camera | mouse | replay:PATH


class AetherApp(QObject):
    """เป็นเจ้าของ pipeline, worker thread, hotkey, HUD และหน้าต่าง debug ของ config หนึ่งชุด"""

    gesture_events = Signal(object)  # tuple[GestureEvent, ...] ที่ไม่ใช่การขยับเคอร์เซอร์
    camera_paused_changed = Signal(bool)
    incident_saved = Signal(object)  # Path ของไฟล์ incident ที่เขียนแล้ว (None = เขียนไม่ได้)
    armed_changed = Signal(bool)
    frame_packet = Signal(object)  # FramePacket ล่าสุด (UI thread) — ใช้ป้อน calibration wizard
    debug_visibility_changed = Signal(bool)
    quit_requested = Signal()

    def __init__(
        self,
        config: Config,
        options: Options,
        tracker: HandTracker | NullTracker,
        *,
        source: FrameSource | None = None,
        backend: InputBackend | None = None,
        screen: ScreenArea | None = None,
        hotkeys: HotkeyManager | None = None,
        quit_on_debug_close: bool = True,
        poses: PoseLibrary | None = None,
        incident_directory: Callable[[], Path] | Path | None = None,
        session_log: Path | None = None,
        targeting_backend: Callable[[], object] | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._config = config
        self._base_config = config  # ค่าตอนเริ่ม/ที่บันทึกล่าสุด (live tuning ปรับทับค่านี้)
        self._overrides: dict[str, float] = {}
        self._incident_directory = incident_directory or incident_dir
        self._session_log = session_log  # None = ไม่เขียนสถิติ session (เช่นใน test)
        self._poses = poses if poses is not None and config.poses.enabled else PoseLibrary()
        self._options = options
        self._tracker = tracker
        self._packet: FramePacket | None = None
        self._quit_on_debug_close = quit_on_debug_close

        # --source mouse: เมาส์จริงเป็นตัวป้อนอยู่แล้ว ห้ามส่ง input ซ้ำ → dry-run เสมอ
        dry_run = options.dry_run or options.source.kind == "mouse"
        if backend is None:
            backend, self._dry_run = create_backend(dry_run)
        else:
            self._dry_run = dry_run
        actions = {**dataclasses.asdict(config.actions), **self._poses.actions()}
        self._executor = ActionExecutor(backend, actions, config.launcher.items)
        if screen is None:
            screen = (
                Desktop(list_monitors())
                if config.pointer.screens == "all"
                else primary_screen_geometry()
            )
        self._screen = screen
        self._double_click_ms = double_click_time_ms()
        self._engine = GestureEngine(
            config,
            screen,
            double_click_ms=self._double_click_ms,
            poses=PoseClassifier(self._poses, config.poses.sensitivity),
        )
        self._recorder = FlightRecorder()
        self._stats = SessionStats(source=str(options.source), dry_run=self._dry_run)
        if self._poses.poses:
            log.info(
                "ท่าที่สอนเอง: %s", ", ".join(f"{p.name} → {p.action}" for p in self._poses.poses)
            )
        # event มาจาก worker/hotkey thread → ส่งต่อเป็น Qt signal เพื่อให้ไปถึง UI thread
        self._pipeline = Pipeline(
            tracker.process,
            self._engine,
            self._executor,
            on_events=self.gesture_events.emit,
            recorder=self._recorder,
        )
        self.gesture_events.connect(self._on_gesture_events)
        self.incident_saved.connect(self._on_incident_saved)
        self.quit_requested.connect(QApplication.quit)

        self._source = source or create_source(options.source, config, screen)
        self._thread = self._make_camera_thread()
        self._camera_info = ""
        self._camera_message: str | None = None
        self._camera_paused = False
        self._light = LightMonitor(config.camera.low_light)

        # M6 Target Lock: thread แยกที่ถาม UI Automation (Windows) — ปิดได้ด้วย targeting.enabled
        self._finder: TargetFinder | None = None
        if config.targeting.enabled and (targeting_backend is not None or sys.platform == "win32"):
            screen_info = QGuiApplication.primaryScreen()
            self._finder = TargetFinder(
                config.targeting,
                position=lambda: self._pipeline.snapshot.pointer,
                publish=self._pipeline.set_target,
                active=lambda: self._pipeline.armed,
                backend_factory=targeting_backend,
                scale=screen_info.devicePixelRatio() if screen_info is not None else 1.0,
            )

        self._hotkeys = hotkeys or HotkeyManager(
            {
                config.hotkeys.toggle_arm: self.toggle_armed,
                config.hotkeys.quit: self.quit_requested.emit,
                config.hotkeys.mark_incident: self.mark_incident,
            }
        )

        self._debug_view: DebugView | None = None
        if options.debug:
            self._ensure_debug_view()

        self._hud: HudOverlay | HudGroup | None = None
        if config.hud.enabled and not options.no_hud:
            self._hud = self._build_hud(screen)
            self._hud.minimized_rect_provider = lambda: self._executor.last_minimized_rect

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
    def poses(self) -> PoseLibrary:
        """ท่าที่สอนเองที่ชุดนี้ใช้อยู่"""
        return self._poses

    @property
    def debug_view(self) -> DebugView | None:
        return self._debug_view

    @property
    def debug_visible(self) -> bool:
        return self._debug_view is not None and self._debug_view.isVisible()

    @property
    def hud(self) -> HudOverlay | HudGroup | None:
        return self._hud

    @property
    def screen(self) -> ScreenArea:
        """พื้นที่ที่เคอร์เซอร์ไปได้ (จอหลัก หรือทุกจอ)"""
        return self._screen

    @property
    def latest_packet(self) -> FramePacket | None:
        return self._packet

    @property
    def pipeline(self) -> Pipeline:
        return self._pipeline

    @property
    def dry_run(self) -> bool:
        return self._dry_run

    @property
    def camera_paused(self) -> bool:
        return self._camera_paused

    @property
    def light(self) -> LightMonitor:
        return self._light

    @property
    def finder(self) -> TargetFinder | None:
        return self._finder

    @property
    def stats(self) -> SessionStats:
        return self._stats

    @property
    def recorder(self) -> FlightRecorder:
        return self._recorder

    # ---- control -------------------------------------------------------- #

    def set_camera_paused(self, paused: bool) -> None:
        """พักกล้อง (ปล่อยกล้องให้โปรแกรมอื่น เช่นตอนเข้าประชุม) / ใช้กล้องต่อ — พักแล้วมือไม่คุมเมาส์"""
        if paused == self._camera_paused:
            return
        self._camera_paused = paused
        if paused:
            self._pipeline.set_armed(False)
            self._thread.stop()
            self._camera_message = 'พักกล้องอยู่ — เลือก "ใช้กล้องต่อ" ใน tray เพื่อกลับมาใช้'
            log.info("พักกล้องแล้ว (ปล่อยกล้องให้โปรแกรมอื่นใช้ได้)")
            if self._hud is not None:
                self._hud.set_alert("CAMERA PAUSED")
            if self._debug_view is not None:
                self._debug_view.show_message(self._camera_message)
        else:
            log.info("ใช้กล้องต่อ")
            self._camera_message = None
            if self._hud is not None:
                self._hud.set_alert(None)
            self._thread = self._make_camera_thread()
            self._thread.start()
        self.camera_paused_changed.emit(paused)

    def mark_incident(self) -> bool:
        """``Ctrl+Alt+M``: เขียน 5 วินาทีที่ผ่านมาลงไฟล์ (thread แยก — pointer ไม่สะดุด)

        เรียกได้จาก thread ใดก็ได้ (ปกติคือ hotkey thread) ผลแจ้งผ่าน ``incident_saved``
        """
        frames = self._recorder.snapshot()
        if not frames:
            log.warning("ยังไม่มีเฟรมให้จด incident")
            return False
        config, screen, poses = self._config, self._screen, self._poses
        directory = self._incident_directory

        def write() -> None:
            try:
                target = directory() if callable(directory) else directory
                path = write_incident(
                    target,
                    frames,
                    config=config,
                    screen=screen,
                    double_click_ms=self._double_click_ms,
                    poses=poses,
                    source=str(self._options.source),
                )
            except Exception:
                log.exception("เขียน incident ไม่ได้")
                self.incident_saved.emit(None)
                return
            self.incident_saved.emit(path)

        threading.Thread(target=write, name="incident-writer", daemon=True).start()
        return True

    def apply_tuning(self, overrides: dict[str, float]) -> bool:
        """ใช้ค่าจากแผง live tuning ทันที (ค่าที่ขัดกันจะไม่ถูกใช้ และแจ้งในแผง)"""
        try:
            config = apply_overrides(self._base_config, overrides)
        except (ConfigError, ValueError) as exc:
            if self._debug_view is not None:
                self._debug_view.tuning.show_status(str(exc), error=True)
            return False
        self._pipeline.reconfigure(config)
        self._config = config
        self._overrides = dict(overrides)
        if self._debug_view is not None:
            self._debug_view.set_config(config)
        log.info("live tuning: %s", overrides or "(ค่าเดิม)")
        return True

    def save_tuning(self) -> list[Path]:
        """บันทึกค่าที่ปรับไว้ลง config.toml (+ calibration.json ถ้ามีค่า pinch ที่ทับอยู่)"""
        path = self._options.config_path or app_root() / "config.toml"
        calibration = None if self._options.no_calibration else calibration_path()
        panel = self._debug_view.tuning if self._debug_view is not None else None
        try:
            written = save_overrides(
                path,
                current_values(self._config) if self._overrides else {},
                template=bundle_root() / "config.toml",
                calibration=calibration,
            )
        except (OSError, ConfigError) as exc:
            log.error("บันทึกค่าที่ปรับไม่ได้: %s", exc)
            if panel is not None:
                panel.show_status(f"บันทึกไม่ได้: {exc}", error=True)
            return []
        self._base_config = self._config
        self._overrides = {}
        log.info("บันทึกค่าที่ปรับลง %s", ", ".join(str(p) for p in written))
        if panel is not None:
            panel.set_values(current_values(self._config), base=True)
            panel.show_status("บันทึกแล้ว: " + ", ".join(p.name for p in written))
        return written

    def reset_tuning(self) -> None:
        self.apply_tuning({})
        if self._debug_view is not None:
            self._debug_view.tuning.set_values(current_values(self._base_config))
            self._debug_view.tuning.show_status("กลับเป็นค่าเดิมแล้ว")

    def toggle_armed(self) -> bool:
        """สลับ ARM/DISARM (เรียกจาก thread ใดก็ได้ มีผลทันที) — ระหว่างพักกล้อง ARM ไม่ได้"""
        if self._camera_paused and not self._pipeline.armed:
            log.info('พักกล้องอยู่ — ยังไม่ ARM (เลือก "ใช้กล้องต่อ" ใน tray ก่อน)')
            return False
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
        if self._finder is not None:
            self._finder.start()

    def shutdown(self) -> None:
        """ปล่อยปุ่มเมาส์ก่อนเสมอ แล้วค่อยหยุด worker และปิด tracker"""
        self._watchdog.stop()
        self._stats_timer.stop()
        self._hotkeys.stop()
        if self._finder is not None:
            self._finder.stop()
        self._pipeline.shutdown()
        if self._thread.stop():
            self._tracker.close()
        else:
            log.warning("camera thread did not stop in time; skipping tracker cleanup")
        if self._session_log is not None:
            try:
                paints = self._hud.paint_count if self._hud is not None else None
                path = self._stats.write(self._session_log, hud_paints=paints)
                log.info("สถิติ session: %s", path)
            except OSError as exc:
                log.warning("เขียนสถิติ session ไม่ได้: %s", exc)
        if self._hud is not None:
            self._hud.stop()
            self._hud.deleteLater()
        if self._debug_view is not None:
            self._debug_view.closed.disconnect(self._on_debug_closed)
            self._debug_view.close()
            self._debug_view.deleteLater()

    # ---- internal ------------------------------------------------------- #

    def _build_hud(self, screen: ScreenArea) -> HudOverlay | HudGroup:
        """จอเดียว: overlay เดียว / หลายจอ: หนึ่ง overlay ต่อจอ (แผงและเสียงอยู่ที่จอหลัก)"""
        config = self._config
        sounds = SoundPlayer(
            assets_dir() / "sounds", enabled=config.hud.sounds, volume=config.hud.volume
        )
        theme = Theme.from_config(config.hud, assets_dir() / "fonts")
        if not isinstance(screen, Desktop):
            return HudOverlay(config, screen, dry_run=self._dry_run, theme=theme, sounds=sounds)
        monitors = sorted(screen.monitors, key=lambda m: not m.primary)  # จอหลักก่อน
        qt_screens = match_qt_screens(monitors, QGuiApplication.screens())
        overlays = [
            HudOverlay(
                config,
                monitor.geometry,
                dry_run=self._dry_run,
                theme=theme,
                sounds=sounds if index == 0 else None,
                qt_screen=qt_screen,
                panels=index == 0,
            )
            for index, (monitor, qt_screen) in enumerate(zip(monitors, qt_screens))
        ]
        log.info(
            "HUD บน %d จอ: %s",
            len(overlays),
            ", ".join(f"{m.name or '?'} {m.geometry.width}x{m.geometry.height}" for m in monitors),
        )
        return HudGroup(overlays)

    def _make_camera_thread(self) -> CameraThread:
        thread = CameraThread(
            self._source, self._pipeline.process, self, observer=self._observe_packet
        )
        thread.frame_ready.connect(self._on_frame_ready)
        thread.camera_opened.connect(self._on_camera_opened)
        thread.camera_error.connect(self._on_camera_error)
        thread.failed.connect(self._on_failed)
        return thread

    def _ensure_debug_view(self) -> DebugView:
        if self._debug_view is None:
            view = DebugView(
                self._config,
                dry_run=self._dry_run,
                arm_hotkey=str(parse_hotkey(self._config.hotkeys.toggle_arm)),
            )
            view.closed.connect(self._on_debug_closed)
            view.tuning.changed.connect(self.apply_tuning)
            view.tuning.save_requested.connect(self.save_tuning)
            view.tuning.reset_requested.connect(self.reset_tuning)
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
        self._check_light(packet)
        if self._hud is not None:
            self._hud.show_packet(packet)
        if self._debug_view is not None and self._debug_view.isVisible():
            if self._hud is not None:
                self._debug_view.set_hud_draw_ms(self._hud.draw_ms)
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
                if event.reason == "idle":
                    minutes = self._config.safety.auto_disarm_minutes
                    log.info(
                        "DISARMED เอง — ไม่เห็นมือนาน %g นาที (safety.auto_disarm_minutes)", minutes
                    )
                    if self._hud is not None:
                        self._hud.notify("AUTO DISARM", None)
                else:
                    log.info("DISARMED")
                self.armed_changed.emit(False)
            elif event.kind is EventKind.ACTION:
                log.info("gesture action: %s", event.name)
            else:
                log.debug("event: %s", event)

    def _observe_packet(self, packet: FramePacket) -> None:
        """worker thread: สถิติของทุกเฟรม (ไม่ใช่แค่เฟรมที่ UI วาดทัน)"""
        result = packet.result
        self._stats.observe(
            packet.captured_at, packet.process_ms, result.snapshot.state, result.events
        )

    def _on_incident_saved(self, path: Path | None) -> None:
        if path is None:
            if self._hud is not None:
                self._hud.notify("INCIDENT FAILED", "error")
            return
        self._stats.note_incident()
        log.info("จด incident แล้ว: %s", path)
        if self._hud is not None:
            self._hud.notify("INCIDENT SAVED")

    def _on_camera_opened(self, description: str) -> None:
        self._camera_info, self._camera_message = description, None
        if self._hud is not None:
            self._hud.set_alert(None)
        if self._debug_view is not None:
            self._debug_view.set_camera_info(description)

    def _on_camera_error(self, message: str) -> None:
        users = (
            webcam_users()
        )  # Windows: โปรแกรมไหนถือกล้องอยู่ (เช่น Teams/Zoom) — แอปลองเปิดใหม่เรื่อยๆ เอง
        if users:
            message += f"\nตอนนี้กล้องถูกใช้โดย: {', '.join(users)} — ปิดโปรแกรมนั้นแล้วแอปจะกลับมาใช้กล้องเอง"
            log.info("กล้องถูกใช้โดย %s", ", ".join(users))
        self._camera_message = message
        if self._hud is not None:
            self._hud.set_alert("CAMERA IN USE" if users else "CAMERA LOST")
        if self._debug_view is not None:
            self._debug_view.show_message(message)

    def _check_light(self, packet: FramePacket) -> None:
        change = self._light.update(packet.brightness, packet.captured_at)
        if change == "low":
            log.info(
                "แสงน้อย (ความสว่าง %.0f/255) — เปิดไฟหรือหันหน้าเข้าหาแสง จะจับมือได้แม่นขึ้น",
                packet.brightness or 0.0,
            )
            if self._hud is not None and self._camera_message is None:
                self._hud.set_alert("LOW LIGHT")
        elif change == "ok" and self._hud is not None and self._camera_message is None:
            self._hud.set_alert(None)

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


AppFactory = Callable[[Config, Options, bool, PoseLibrary], AetherApp]
Wizard = CalibrationWizard | PoseWizard


def _default_app_factory(
    config: Config, options: Options, quit_on_debug_close: bool, poses: PoseLibrary
) -> AetherApp:
    if options.source.simulated:  # mouse / replay: เฟรมเป็น HandFrame แล้ว ไม่ต้องโหลด MediaPipe
        tracker = NullTracker()
    else:
        tracker = HandTracker(
            model_path(), num_hands=config.tracking.num_hands, mirrored=config.camera.mirror
        )
    return AetherApp(
        config,
        options,
        tracker,
        quit_on_debug_close=quit_on_debug_close,
        poses=poses,
        session_log=log_dir(),
    )


class AppController(QObject):
    """อยู่ตลอดอายุของ process: เป็นเจ้าของ tray, wizard และ :class:`AetherApp` ที่ทำงานอยู่

    wizard (calibration หรือสอนท่ามือ) เปิดได้ทีละหน้าต่าง ระหว่างเปิดมือไม่คุมเมาส์ (DISARMED)

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
        poses_file: Path | None = None,
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
        self._poses_file = poses_file or poses_path()
        self._app: AetherApp | None = None
        self._wizard: Wizard | None = None
        self._debug_wanted = options.debug

        tray_ok = Tray.available() if use_tray is None else use_tray
        self._tray: Tray | None = Tray() if tray_ok else None
        if self._tray is not None:
            self._tray.toggle_arm_requested.connect(self._toggle_armed)
            self._tray.pause_requested.connect(self._set_paused)
            self._tray.debug_requested.connect(self._set_debug)
            self._tray.calibrate_requested.connect(self.open_calibration)
            self._tray.poses_requested.connect(self.open_poses)
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
    def wizard(self) -> Wizard | None:
        """wizard ที่เปิดอยู่ (calibration หรือสอนท่ามือ)"""
        return self._wizard

    # ---- lifecycle ------------------------------------------------------ #

    def start(self, config: Config) -> None:
        self._launch(config, boot=True)
        if self._tray is not None:
            self._tray.show()
        if self._options.calibrate:
            self.open_calibration()
        elif self._options.teach:
            self.open_poses()

    def shutdown(self) -> None:
        if self._wizard is not None:
            wizard, self._wizard = self._wizard, None
            wizard.finished.disconnect(self._on_wizard_finished)
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
        if self._app is not None and not self._raise_wizard():
            self._open_wizard(CalibrationWizard(self._app.config, self._calibration_file))
            log.info("เปิด calibration wizard")

    def open_poses(self) -> None:
        """เปิดหน้าต่างสอนท่ามือ (มือหยุดคุมเมาส์ระหว่างนั้น)"""
        if self._app is not None and not self._raise_wizard():
            app = self._app
            self._open_wizard(PoseWizard(app.config, app.poses, self._poses_file))
            log.info("เปิดหน้าต่างสอนท่ามือ")

    def load_poses(self, config: Config) -> PoseLibrary:
        """ท่าที่สอนไว้ใน ``poses.json`` — ไฟล์เสียจะแจ้งเตือนแล้วใช้งานต่อแบบไม่มีท่าที่สอนเอง"""
        if not config.poses.enabled:
            return PoseLibrary()
        try:
            return load_poses(self._poses_file)
        except PoseFileError as exc:
            log.error("อ่านท่าที่สอนไว้ไม่ได้ — ใช้งานต่อโดยไม่มีท่าเหล่านี้: %s", exc)
            self._notify("ไฟล์ท่ามือเสีย", str(exc), error=True)
            return PoseLibrary()

    # ---- internal ------------------------------------------------------- #

    def _raise_wizard(self) -> bool:
        """มี wizard เปิดอยู่แล้ว: ยกขึ้นมาให้เห็น (เปิดได้ทีละหน้าต่าง)"""
        if self._wizard is None:
            return False
        self._wizard.show()
        self._wizard.raise_()
        return True

    def _open_wizard(self, wizard: Wizard) -> None:
        assert self._app is not None
        self._app.pipeline.set_armed(False)
        wizard.finished.connect(self._on_wizard_finished)
        self._wizard = wizard
        wizard.show()

    def _launch(self, config: Config, *, boot: bool) -> None:
        options = dataclasses.replace(self._options, debug=self._debug_wanted)
        app = self._app_factory(config, options, self._tray is None, self.load_poses(config))
        app.armed_changed.connect(self._on_armed_changed)
        app.camera_paused_changed.connect(self._on_paused_changed)
        app.debug_visibility_changed.connect(self._on_debug_visibility)
        app.frame_packet.connect(self._on_frame_packet)
        self._app = app
        if self._tray is not None:
            self._tray.set_primary(config.hud.palette.primary)
            self._tray.set_armed(False)
            self._tray.set_paused_checked(False)
            self._tray.set_debug_checked(self._debug_wanted)
        app.start(boot=boot)

    def _toggle_armed(self) -> None:
        if self._app is not None:
            self._app.toggle_armed()

    def _set_paused(self, paused: bool) -> None:
        if self._app is not None:
            self._app.set_camera_paused(paused)

    def _on_paused_changed(self, paused: bool) -> None:
        if self._tray is not None:
            self._tray.set_paused_checked(paused)

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
            log.info("หน้าต่าง calibrate/สอนท่ายังเปิดอยู่ — ยังไม่เปิดการคุมเมาส์ด้วยมือ")
            self._app.pipeline.set_armed(False)
            return
        if self._tray is not None:
            self._tray.set_armed(armed)

    def _on_frame_packet(self, packet: FramePacket) -> None:
        if self._wizard is not None:
            self._wizard.show_frame(packet.result.hands)

    def _on_wizard_finished(self, saved: bool) -> None:
        self._wizard = None
        if saved:  # calibration.json / poses.json เปลี่ยน → เริ่มชุดใหม่ให้ใช้ค่าใหม่
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
    if os.environ.get("AETHER_NO_DIALOG"):  # รันอัตโนมัติ (CI): ห้ามมีกล่องข้อความมาค้าง
        return
    _ = QApplication.instance() or QApplication(sys.argv[:1])  # QMessageBox ต้องมี QApplication
    box = QMessageBox.critical if error else QMessageBox.information
    box(None, title, text)


def _check_targeting() -> str:
    """Target Lock ใช้ได้ไหม (ไม่นับเป็นความล้มเหลว — ไม่มีก็ใช้งานได้ แค่ไม่ล็อกเป้า)"""
    if sys.platform != "win32":
        return "· Target Lock ใช้ได้เฉพาะบน Windows"
    try:
        from .targeting import UiaBackend

        backend = UiaBackend()
        try:
            backend.element_chain(0, 0, 0)
        finally:
            backend.close()
    except Exception as exc:  # noqa: BLE001 — รายงานทุกสาเหตุที่ทำให้ UIA ใช้ไม่ได้
        return f"! Target Lock ใช้ไม่ได้ ({type(exc).__name__}: {exc}) — แอปยังใช้งานได้"
    return "✓ UI Automation ใช้ได้ (Target Lock)"


def run_check(options: Options) -> int:
    """``--check``: ตรวจว่า asset ครบและโมเดลโหลดได้ (ไม่ต้องใช้กล้อง) — ใช้ตรวจ build

    ผลลัพธ์แสดงบน console (หรือกล่องข้อความถ้าไม่มี console) และเขียนลง log ด้วย
    """
    configure_logging(False, log_dir())
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
    if config is not None and config.targeting.enabled:
        lines.append(_check_targeting())
    if config is not None and config.poses.enabled:
        try:
            names = load_poses(poses_path()).names()
            lines.append(
                f"✓ ท่าที่สอนเอง {len(names)} ท่า: {', '.join(names)}"
                if names
                else "· ยังไม่ได้สอนท่ามือเอง"
            )
        except PoseFileError as exc:  # ไม่ทำให้ใช้งานไม่ได้: แอปทำงานต่อโดยไม่มีท่าเหล่านี้
            lines.append(f"! อ่านท่าที่สอนไว้ไม่ได้: {exc}")
    lines.append("ผลรวม: พร้อมใช้งาน" if ok else "ผลรวม: ยังใช้งานไม่ได้")
    for line in lines:
        log.info("check: %s", line)
    log.info("check result: %s", "OK" if ok else "FAIL")
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
    if not options.source.simulated and not model.is_file():
        message = f"ไม่พบโมเดล {model}\nรัน `python tools/download_assets.py` ก่อน"
        log.error("%s", message.replace("\n", " — "))
        _report("AETHER — ไม่พบโมเดล", message, error=True)
        return EXIT_SETUP_ERROR

    if options.source.kind == "mouse" and not options.dry_run:
        log.info("--source mouse บังคับ dry-run (เมาส์จริงเป็นตัวป้อนอยู่แล้ว)")
        options = dataclasses.replace(options, dry_run=True)
    log.info(
        "AETHER %s%s%s%s — log: %s",
        __version__,
        "" if options.source.kind == "camera" else f" [source {options.source}]",
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
        log.info("ปิดแอปแล้ว (shutdown complete)")
