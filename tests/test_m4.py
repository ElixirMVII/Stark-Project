"""M4: system tray, calibration wizard, โหลด config ใหม่, log แบบ rotate, ``--check`` และ CLI"""

from __future__ import annotations

import json
import logging
import logging.handlers
import sys
import threading
import time

import pytest
from handsim import FRAME_MS, PINCHED, make_frame, make_hand
from PySide6.QtGui import QColor
from test_actions import RecordingBackend
from test_app import BlankCamera, ScriptedTracker
from test_hotkeys import FakeApi

from aether import __main__ as cli
from aether import app as app_module
from aether import calibration as cal
from aether import logsetup
from aether.app import AetherApp, AppController, Options, run_check
from aether.calibration_wizard import CalibrationWizard
from aether.config import Config, ConfigError, parse_config
from aether.gestures import State
from aether.hotkeys import HotkeyManager
from aether.mapping import ScreenGeometry
from aether.tray import Tray, make_icon

FHD = ScreenGeometry(0, 0, 1920, 1080)


def wait(qapp, condition, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        qapp.processEvents()
        if condition():
            return True
        time.sleep(0.005)
    return False


# ---- tray ----------------------------------------------------------------- #


def test_tray_menu_actions_emit_signals(qapp):
    tray = Tray("#00E5FF")
    received: list[str] = []
    tray.toggle_arm_requested.connect(lambda: received.append("arm"))
    tray.debug_requested.connect(lambda on: received.append(f"debug:{on}"))
    tray.calibrate_requested.connect(lambda: received.append("calibrate"))
    tray.reload_requested.connect(lambda: received.append("reload"))
    tray.quit_requested.connect(lambda: received.append("quit"))

    actions = tray.actions()
    assert set(actions) == {"arm", "debug", "calibrate", "reload", "quit"}
    for name in ("arm", "calibrate", "reload", "quit"):
        actions[name].trigger()
    actions["debug"].trigger()  # checkable: ติ๊ก → True
    actions["debug"].trigger()  # เอาออก → False
    assert received == ["arm", "calibrate", "reload", "quit", "debug:True", "debug:False"]
    assert [a.text() for a in tray.menu.actions() if not a.isSeparator()][-1] == "ออก"


def test_tray_reflects_arm_state_and_debug_check(qapp):
    tray = Tray("#00E5FF")
    arm = tray.actions()["arm"]
    assert not tray.armed and arm.text().startswith("Arm")
    tray.set_armed(True)
    assert tray.armed and arm.text().startswith("Disarm")

    received: list[bool] = []
    tray.debug_requested.connect(received.append)
    tray.set_debug_checked(True)  # สะท้อนสถานะจากแอป: ต้องไม่ส่ง signal กลับ
    assert tray.actions()["debug"].isChecked() and received == []
    tray.notify("ทดสอบ", "ข้อความ")  # ไม่มี tray จริงบน offscreen ก็ต้องไม่ error
    tray.set_primary("#11AA33")


def test_tray_icon_is_drawn_for_both_states(qapp):
    for armed in (True, False):
        icon = make_icon(QColor("#00E5FF"), armed)
        image = icon.pixmap(64, 64).toImage()
        assert not image.isNull()
        assert image.pixelColor(32, 32).alpha() > 0  # มีพื้นหลังวงกลม
        assert image.pixelColor(1, 1).alpha() == 0  # มุมโปร่งใส
    armed_center = make_icon(QColor("#00E5FF"), True).pixmap(64, 64).toImage().pixelColor(32, 32)
    disarmed_center = (
        make_icon(QColor("#00E5FF"), False).pixmap(64, 64).toImage().pixelColor(32, 32)
    )
    assert armed_center.blue() > disarmed_center.blue()  # ARMED มีจุดกลางสีหลัก


# ---- calibration wizard --------------------------------------------------- #


def _feed(wizard: CalibrationWizard, hand, ms: int, start: int) -> int:
    t = start
    for _ in range(round(ms / FRAME_MS)):
        wizard.show_frame(make_frame(t, *(() if hand is None else (hand,))))
        t += FRAME_MS
    return t


def _sweep(wizard: CalibrationWizard, ms: int, start: int) -> int:
    import math

    t = start
    frames = round(ms / FRAME_MS)
    for i in range(frames):
        angle = 2 * math.pi * 3 * i / frames
        radius = 0.4 + 0.6 * (i % 7) / 6
        hand = make_hand(
            0.5 + 0.3 * math.cos(angle) * radius, 0.4 + 0.25 * math.sin(angle) * radius
        )
        wizard.show_frame(make_frame(t, hand))
        t += FRAME_MS
    return t


def test_wizard_runs_all_steps_and_saves(qapp, tmp_path):
    path = tmp_path / "calibration.json"
    wizard = CalibrationWizard(Config(), path)
    finished: list[bool] = []
    wizard.finished.connect(finished.append)
    wizard.show()
    assert "1/3" in wizard._title.text() and "ยกมือ" in wizard._status.text()

    step_ms = cal.SETTLE_MS + 2200
    t = _feed(wizard, make_hand(pinch=0.9), 600, 0)
    assert "เตรียมตัว" in wizard._status.text()
    t = _feed(wizard, make_hand(pinch=0.9), step_ms, t)
    assert "2/3" in wizard._title.text()
    t = _feed(wizard, make_hand(pinch=0.9), step_ms, t)  # ไม่ได้จีบจริง → ขึ้นคำแนะนำ
    assert "2/3" in wizard._title.text() and "จีบ" in wizard._hint.text()
    t = _feed(wizard, make_hand(pinch=PINCHED), step_ms, t)
    assert "3/3" in wizard._title.text() and wizard._hint.text() == ""
    _sweep(wizard, cal.SETTLE_MS + cal.SAMPLE_MS["REACH"] + 300, t)

    assert wizard.saved and path.is_file()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert PINCHED < data["pinch"]["on"] < data["pinch"]["off"] < 0.9
    assert "pinch.on" in wizard._status.text() and path.name in wizard._status.text()
    assert wizard._status.toolTip() == str(path)
    assert wizard._progress.value() == 100 and wizard._close_button.text() == "ปิด"

    wizard.close()
    assert finished == [True]
    wizard.close()
    assert finished == [True]  # แจ้งครั้งเดียว


def test_wizard_cancel_reports_not_saved(qapp, tmp_path):
    wizard = CalibrationWizard(Config(), tmp_path / "calibration.json")
    finished: list[bool] = []
    wizard.finished.connect(finished.append)
    wizard.show()
    _feed(wizard, make_hand(pinch=0.9), 1000, 0)
    wizard.restart()
    assert wizard.session.step is cal.Step.REST and wizard.session.progress == 0.0
    wizard.close()
    assert finished == [False]
    assert not (tmp_path / "calibration.json").exists()


def test_wizard_reports_unwritable_location(qapp, tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x", encoding="utf-8")
    wizard = CalibrationWizard(Config(), blocker / "calibration.json")  # parent เป็นไฟล์
    wizard.session._step = cal.Step.DONE
    wizard.session.result = cal.CalibrationResult(0.3, 0.45, (0.2, 0.2, 0.8, 0.6), 0.9, 0.15)
    wizard.show_frame(make_frame(0, make_hand()))
    assert not wizard.saved and "บันทึก" in wizard._hint.text()
    wizard.close()


def test_wizard_follows_dominant_hand(qapp, tmp_path):
    wizard = CalibrationWizard(
        parse_config({"tracking": {"dominant_hand": "left"}}), tmp_path / "c.json"
    )
    _feed(wizard, make_hand(pinch=0.9), cal.SETTLE_MS + 2500, 0)  # มือขวา: ไม่นับ
    assert wizard.session.step is cal.Step.REST
    _feed(wizard, make_hand(pinch=0.9, side="left"), cal.SETTLE_MS + 2500, 10_000)
    assert wizard.session.step is cal.Step.PINCH
    wizard.close()


# ---- controller: tray + reload + calibration ------------------------------ #


class Factory:
    """สร้าง AetherApp ด้วยของปลอมทั้งหมด และจดว่าสร้างไปกี่ชุดด้วย config อะไร"""

    def __init__(self) -> None:
        self.apps: list[AetherApp] = []
        self.configs: list[Config] = []
        self.options: list[Options] = []
        self.trackers: list[ScriptedTracker] = []
        self.backends: list[RecordingBackend] = []

    def __call__(self, config: Config, options: Options, quit_on_debug_close: bool) -> AetherApp:
        tracker, backend = ScriptedTracker(), RecordingBackend()
        app = AetherApp(
            config,
            options,
            tracker,
            source=BlankCamera(),
            backend=backend,
            screen=FHD,
            hotkeys=HotkeyManager({config.hotkeys.toggle_arm: lambda: None}, api=FakeApi()),
            quit_on_debug_close=quit_on_debug_close,
        )
        app.quit_requested.disconnect()
        self.apps.append(app)
        self.configs.append(config)
        self.options.append(options)
        self.trackers.append(tracker)
        self.backends.append(backend)
        return app


class Loader:
    def __init__(self) -> None:
        self.next: Config | Exception = parse_config({"hud": {"sounds": False}})

    def __call__(self) -> Config:
        if isinstance(self.next, Exception):
            raise self.next
        return self.next


@pytest.fixture
def controlled(qapp, tmp_path):
    factory, loader = Factory(), Loader()
    controller = AppController(
        Options(no_hud=True),
        app_factory=factory,
        config_loader=loader,
        calibration_file=tmp_path / "calibration.json",
        use_tray=True,
    )
    controller.start(loader())
    yield controller, factory, loader
    controller.shutdown()


def test_controller_starts_app_and_tray(qapp, controlled):
    controller, factory, _ = controlled
    assert controller.app is factory.apps[0] and controller.tray is not None
    assert factory.options[0].debug is False
    assert wait(qapp, lambda: controller.app.latest_packet is not None)


def test_tray_arm_toggle_and_state_follow_each_other(qapp, controlled):
    controller, _factory, _ = controlled
    tray = controller.tray
    tray.actions()["arm"].trigger()
    assert controller.app.pipeline.armed and tray.armed
    controller.app.toggle_armed()  # เช่นกด hotkey
    assert wait(qapp, lambda: not tray.armed)


def test_tray_debug_toggle_opens_and_hides_window(qapp, controlled):
    controller, _, _ = controlled
    tray = controller.tray
    tray.actions()["debug"].trigger()
    view = controller.app.debug_view
    assert view is not None and view.isVisible()
    view.close()  # ปิดหน้าต่างเอง: แอปไม่ปิด (มี tray) และเมนูเอาเครื่องหมายถูกออก
    assert wait(qapp, lambda: not tray.actions()["debug"].isChecked())
    assert controller.app is not None


def test_reload_rebuilds_app_with_new_config_and_starts_disarmed(qapp, controlled):
    controller, factory, loader = controlled
    old = controller.app
    old.toggle_armed()
    controller.tray.actions()["debug"].trigger()  # เปิด debug ไว้: ต้องคงไว้หลัง reload
    loader.next = parse_config({"hud": {"sounds": False}, "pinch": {"on": 0.3, "off": 0.45}})
    results: list[bool] = []
    controller.reloaded.connect(results.append)

    controller.tray.actions()["reload"].trigger()
    assert results == [True]
    assert len(factory.apps) == 2 and controller.app is factory.apps[1]
    assert controller.app.config.pinch.on == 0.3
    assert not controller.app.pipeline.armed and not controller.tray.armed
    assert factory.options[1].debug is True and controller.app.debug_visible
    assert factory.trackers[0].closed  # ชุดเดิมถูกปิด
    assert wait(qapp, lambda: controller.app.latest_packet is not None)


def test_reload_releases_held_button_of_old_app(qapp, controlled):
    controller, factory, _ = controlled
    app = controller.app
    app.toggle_armed()
    assert wait(qapp, lambda: app.pipeline.snapshot.state is State.POINTER)
    time.sleep(0.15)
    factory.trackers[0].hand = make_hand(pinch=PINCHED)
    assert wait(qapp, lambda: app.pipeline.snapshot.state is State.DRAGGING)
    controller.reload()
    assert ("release", "left") in factory.backends[0].calls


def test_reload_with_broken_config_keeps_running_old_app(qapp, controlled):
    controller, factory, loader = controlled
    old = controller.app
    loader.next = ConfigError("pinch.on ต้อง > 0 (test)")
    results: list[bool] = []
    controller.reloaded.connect(results.append)
    assert controller.reload() is False
    assert results == [False] and controller.app is old and len(factory.apps) == 1
    assert wait(qapp, lambda: old.latest_packet is not None)


def test_calibration_flow_disarms_feeds_frames_and_reloads(qapp, controlled):
    controller, factory, _loader = controlled
    app = controller.app
    app.toggle_armed()
    controller.tray.actions()["calibrate"].trigger()
    wizard = controller.wizard
    assert wizard is not None and wizard.isVisible()
    assert not app.pipeline.armed  # เปิด wizard = หยุดคุมเมาส์

    app.toggle_armed()  # พยายาม arm ระหว่าง calibrate → ถูกปิดทันที
    assert not app.pipeline.armed

    factory.trackers[0].hand = make_hand(pinch=0.9)
    assert wait(qapp, lambda: wizard.session.hand_visible)  # wizard ได้เฟรมจากกล้อง
    assert wait(qapp, lambda: wizard.session.live_ratio is not None)

    controller.open_calibration()  # เปิดซ้ำ: ใช้หน้าต่างเดิม
    assert controller.wizard is wizard

    wizard._saved = True  # จำลองว่าบันทึกเสร็จแล้ว
    wizard.close()
    assert controller.wizard is None
    assert len(factory.apps) == 2  # โหลด config + calibration ใหม่ทันที


def test_cancelled_calibration_does_not_reload(qapp, controlled):
    controller, factory, _ = controlled
    controller.open_calibration()
    controller.wizard.close()
    assert controller.wizard is None and len(factory.apps) == 1


def test_calibrate_option_opens_wizard_at_start(qapp, tmp_path):
    factory, loader = Factory(), Loader()
    controller = AppController(
        Options(no_hud=True, calibrate=True),
        app_factory=factory,
        config_loader=loader,
        calibration_file=tmp_path / "calibration.json",
        use_tray=False,
    )
    controller.start(loader())
    try:
        assert controller.wizard is not None and controller.tray is None
    finally:
        controller.shutdown()
    assert controller.wizard is None and controller.app is None


def test_without_tray_closing_debug_window_quits(qapp, monkeypatch):
    factory = Factory()
    controller = AppController(
        Options(no_hud=True, debug=True),
        app_factory=factory,
        config_loader=Loader(),
        use_tray=False,
    )
    quits: list[int] = []
    monkeypatch.setattr(app_module.QApplication, "quit", lambda: quits.append(1))
    controller.start(Loader()())
    try:
        controller.app.debug_view.close()
        assert quits == [1]
    finally:
        controller.shutdown()


def test_default_loader_respects_no_calibration(qapp, tmp_path, monkeypatch):
    config_file = tmp_path / "config.toml"
    config_file.write_text("[pinch]\non = 0.2\noff = 0.3\n", encoding="utf-8")
    calibration_file = tmp_path / "calibration.json"
    calibration_file.write_text(
        json.dumps({"version": 1, "pinch": {"on": 0.31, "off": 0.47}}), encoding="utf-8"
    )
    monkeypatch.setattr("aether.config.calibration_path", lambda: calibration_file)
    with_cal = AppController(Options(config_path=config_file), use_tray=False)
    without = AppController(Options(config_path=config_file, no_calibration=True), use_tray=False)
    assert with_cal._config_loader().pinch.on == 0.31
    assert without._config_loader().pinch.on == 0.2


# ---- logging -------------------------------------------------------------- #


@pytest.fixture
def clean_logging():
    root = logging.getLogger()
    saved = (list(root.handlers), root.level, sys.excepthook, threading.excepthook)
    yield
    for handler in list(root.handlers):
        if getattr(handler, "_aether_handler", False):
            root.removeHandler(handler)
            handler.close()
    root.handlers[:] = saved[0]
    root.setLevel(saved[1])
    sys.excepthook, threading.excepthook = saved[2], saved[3]


def test_configure_logging_writes_rotating_file(clean_logging, tmp_path):
    log_file = logsetup.configure_logging(False, tmp_path / "logs")
    assert log_file == tmp_path / "logs" / "aether.log"
    logging.getLogger("aether.test").info("ทดสอบภาษาไทย")
    logging.getLogger("aether.test").debug("ไม่ควรเห็นตอนไม่ใช่ debug")
    ours = [h for h in logging.getLogger().handlers if getattr(h, "_aether_handler", False)]
    rotating = [h for h in ours if isinstance(h, logging.handlers.RotatingFileHandler)]
    assert len(rotating) == 1
    assert rotating[0].maxBytes == logsetup.MAX_BYTES and rotating[0].backupCount == 5
    rotating[0].flush()
    text = log_file.read_text(encoding="utf-8")
    assert "ทดสอบภาษาไทย" in text and "ไม่ควรเห็น" not in text

    logsetup.configure_logging(True, tmp_path / "logs")  # เรียกซ้ำ: ไม่ซ้อน handler
    ours = [h for h in logging.getLogger().handlers if getattr(h, "_aether_handler", False)]
    assert len([h for h in ours if isinstance(h, logging.handlers.RotatingFileHandler)]) == 1
    assert logging.getLogger().level == logging.DEBUG


def test_log_file_rotates(clean_logging, tmp_path, monkeypatch):
    monkeypatch.setattr(logsetup, "MAX_BYTES", 2000)
    logsetup.configure_logging(False, tmp_path)
    logger = logging.getLogger("aether.rotate")
    for i in range(200):
        logger.info("บรรทัดที่ %d %s", i, "x" * 40)
    names = sorted(p.name for p in tmp_path.iterdir())
    assert "aether.log" in names and "aether.log.1" in names
    assert len([n for n in names if n.startswith("aether.log")]) <= 6  # ไฟล์หลัก + สำรอง 5


def test_logging_without_console_and_unwritable_dir(clean_logging, tmp_path, monkeypatch):
    blocker = tmp_path / "file"
    blocker.write_text("x", encoding="utf-8")
    monkeypatch.setattr(sys, "stderr", None)
    monkeypatch.setattr(sys, "stdout", None)
    assert logsetup.configure_logging(False, blocker / "logs") is None  # ไม่ล้ม
    ours = [h for h in logging.getLogger().handlers if getattr(h, "_aether_handler", False)]
    assert ours == []  # ไม่มี console และเขียนไฟล์ไม่ได้


def test_unhandled_exceptions_go_to_the_log(clean_logging, tmp_path):
    log_file = logsetup.configure_logging(False, tmp_path)
    logsetup.install_excepthooks()
    try:
        raise RuntimeError("หลุดมือ (test)")
    except RuntimeError:
        sys.excepthook(*sys.exc_info())
    thread = threading.Thread(target=lambda: 1 / 0, name="worker-test")
    thread.start()
    thread.join()
    for handler in logging.getLogger().handlers:
        handler.flush()
    text = log_file.read_text(encoding="utf-8")
    assert "หลุดมือ (test)" in text and "worker-test" in text and "ZeroDivisionError" in text


# ---- --check และ CLI ------------------------------------------------------- #


def test_check_passes_with_bundled_assets(capsys):
    from aether.config import model_path

    if not model_path().is_file():
        pytest.skip("ยังไม่ได้โหลดโมเดล")
    assert run_check(Options(no_calibration=True)) == 0
    out = capsys.readouterr().out
    assert "โหลดโมเดลและประมวลผลเฟรมได้" in out and "พร้อมใช้งาน" in out


def test_check_fails_without_model(capsys, tmp_path, monkeypatch):
    monkeypatch.setattr(app_module, "model_path", lambda: tmp_path / "missing.task")
    assert run_check(Options(no_calibration=True)) == 2
    assert "ไม่พบโมเดล" in capsys.readouterr().err


def test_check_reports_broken_config(capsys, tmp_path):
    bad = tmp_path / "bad.toml"
    bad.write_text("[pinch]\nonn = 1\n", encoding="utf-8")
    assert run_check(Options(config_path=bad)) == 2
    assert "config ผิด" in capsys.readouterr().err


def test_cli_options(monkeypatch):
    seen: dict = {}
    monkeypatch.setattr(app_module, "run", lambda options: seen.setdefault("run", options) and 0)
    monkeypatch.setattr(
        app_module, "run_check", lambda options: seen.setdefault("check", options) and 0
    )
    cli.main(["--calibrate", "--no-calibration", "--dry-run", "--debug"])
    options = seen["run"]
    assert options.calibrate and options.no_calibration and options.dry_run and options.debug
    cli.main(["--check"])
    assert "check" in seen
    with pytest.raises(SystemExit):
        cli.main(["--version"])


# ---- path ของ build (PyInstaller) ------------------------------------------ #


def test_frozen_paths_split_user_files_from_bundle(tmp_path, monkeypatch):
    from aether import config as config_module

    exe_dir = tmp_path / "dist" / "aether"
    bundle = exe_dir / "_internal"
    (bundle / "assets" / "models").mkdir(parents=True)
    (bundle / "config.toml").write_text("[camera]\nindex = 3\n", encoding="utf-8")
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(bundle), raising=False)
    monkeypatch.setattr(sys, "executable", str(exe_dir / "aether.exe"))

    assert config_module.app_root() == exe_dir
    assert config_module.bundle_root() == bundle
    assert config_module.model_path() == bundle / "assets" / "models" / "gesture_recognizer.task"
    assert config_module.calibration_path() == exe_dir / "calibration.json"
    # ไม่มี config.toml ข้าง exe → ใช้ต้นฉบับใน bundle
    assert config_module.default_config_path() == bundle / "config.toml"
    assert config_module.load_config(use_calibration=False).camera.index == 3
    # ผู้ใช้วาง config.toml ข้าง exe → ใช้ไฟล์นั้น
    (exe_dir / "config.toml").write_text("[camera]\nindex = 1\n", encoding="utf-8")
    assert config_module.default_config_path() == exe_dir / "config.toml"
    assert config_module.load_config(use_calibration=False).camera.index == 1
    assert config_module.log_dir() == exe_dir / "logs"


def test_log_dir_falls_back_to_user_folder_when_app_folder_is_read_only(tmp_path, monkeypatch):
    from aether import config as config_module

    blocker = tmp_path / "app"
    blocker.write_text("x", encoding="utf-8")  # "logs" สร้างใต้ไฟล์ไม่ได้
    monkeypatch.setattr(config_module, "app_root", lambda: blocker)
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    directory = config_module.log_dir()
    assert directory.is_dir() and directory.name == "logs"
    assert str(directory).startswith(str(tmp_path / "state")) or str(directory).startswith(
        str(tmp_path / "local")
    )


def test_build_tool_draws_an_ico_file(qapp, tmp_path):
    from conftest import load_tool

    build_exe = load_tool("build_exe")
    icon = tmp_path / "aether.ico"
    assert build_exe.make_icon(icon) is True
    assert icon.read_bytes()[:4] == b"\x00\x00\x01\x00"  # ICO header


def test_build_tool_stops_when_model_is_missing(tmp_path, monkeypatch, capsys):
    from conftest import load_tool

    build_exe = load_tool("build_exe")
    monkeypatch.setattr(build_exe, "ROOT", tmp_path)
    assert build_exe.main([]) == 1
    assert "download_assets.py" in capsys.readouterr().out


def test_spec_bundles_every_asset_the_app_reads():
    """packaging/aether.spec ต้องรวม asset ทุกชนิดที่แอปอ่าน และ libmediapipe"""
    from conftest import REPO_ROOT

    spec = (REPO_ROOT / "packaging" / "aether.spec").read_text(encoding="utf-8")
    for needle in (
        "models/*.task",
        "fonts/*.ttf",
        "fonts/OFL-*.txt",
        "sounds/*.wav",
        "LICENSES.md",
        "config.toml",
        'collect_dynamic_libs("mediapipe")',
        "mediapipe.tasks.c",
        "pynput.keyboard._win32",
        "pynput.mouse._win32",
    ):
        assert needle in spec, needle
