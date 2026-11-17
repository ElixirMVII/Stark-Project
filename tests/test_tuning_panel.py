"""แผง live tuning ในหน้าต่าง debug + ข้อความยืนยันบน HUD (Qt offscreen)"""

from __future__ import annotations

import tomllib

import pytest
from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QKeyEvent
from test_actions import RecordingBackend
from test_app import BlankCamera, ScriptedTracker
from test_hotkeys import FakeApi
from test_m4 import wait

from aether.app import AetherApp, Options
from aether.config import Config, parse_config
from aether.debug_view import DebugView
from aether.hotkeys import HotkeyManager
from aether.hud.panels import StatusBadge
from aether.hud.theme import Theme
from aether.mapping import ScreenGeometry
from aether.tuning import current_values
from aether.tuning_panel import PANEL_WIDTH, TuningPanel

FHD = ScreenGeometry(0, 0, 1920, 1080)


def test_panel_emits_overrides_and_can_be_reset(qapp):
    panel = TuningPanel(current_values(Config()))
    seen: list[dict] = []
    panel.changed.connect(seen.append)
    panel.set_value("pinch.on", 0.3)
    assert seen[-1] == {"pinch.on": 0.3}
    panel.set_value("gestures.cooldown_ms", 1200)
    assert seen[-1] == {"pinch.on": 0.3, "gestures.cooldown_ms": 1200}
    panel.set_values(current_values(Config()))
    assert panel.overrides() == {} and len(seen) == 2  # set_values ไม่ส่ง signal
    panel.show_status("ผิด", error=True)


def test_debug_view_toggles_the_panel_with_t(qapp):
    view = DebugView(Config())
    view.resize(640, 480)
    view.show()
    try:
        assert view.tuning.isHidden()
        view.keyPressEvent(
            QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_T, Qt.KeyboardModifier.NoModifier)
        )
        assert view.tuning.isVisible() and view.width() == 640 + PANEL_WIDTH
        assert view._image_rect(640, 480).right() <= view.width() - PANEL_WIDTH + 1
        assert view.tuning.geometry().left() == view.width() - PANEL_WIDTH
        view.set_tuning_visible(False)
        assert view.tuning.isHidden()
    finally:
        view.close()


@pytest.fixture
def app(qapp, tmp_path):
    config_file = tmp_path / "config.toml"
    config_file.write_text("# ของฉัน\n[hud]\nsounds = false  # เงียบ\n", encoding="utf-8")
    config = parse_config({"hud": {"sounds": False}})
    application = AetherApp(
        config,
        Options(debug=True, no_hud=True, config_path=config_file, no_calibration=True),
        ScriptedTracker(),
        source=BlankCamera(),
        backend=RecordingBackend(),
        screen=FHD,
        hotkeys=HotkeyManager({config.hotkeys.toggle_arm: lambda: None}, api=FakeApi()),
    )
    application.quit_requested.disconnect()
    application.start(boot=False)
    yield application, config_file
    application.shutdown()


def engine_config(application: AetherApp) -> Config:
    return application.pipeline._engine.config


def test_slider_changes_apply_to_the_running_engine(qapp, app):
    application, _ = app
    panel = application.debug_view.tuning
    panel.set_value("pointer.one_euro.beta", 0.05)
    panel.set_value("gestures.swipe_dist_x", 0.3)
    config = engine_config(application)
    assert config.pointer.one_euro.beta == 0.05 and config.gestures.swipe_dist_x == 0.3
    assert application.config is config
    assert wait(qapp, lambda: application.latest_packet is not None)  # ยังทำงานต่อ


def test_conflicting_values_are_refused(qapp, app):
    application, _ = app
    panel = application.debug_view.tuning
    panel.set_value("pinch.on", 0.45)  # สูงกว่า pinch.off (0.38)
    assert engine_config(application).pinch.on == 0.25
    assert "pinch.on" in panel._status.text()
    panel.set_value("pinch.off", 0.55)  # ตอนนี้ใช้ได้
    assert engine_config(application).pinch.on == 0.45
    assert engine_config(application).pinch.off == 0.55


def test_save_writes_config_and_keeps_comments(qapp, app):
    application, config_file = app
    panel = application.debug_view.tuning
    panel.set_value("pointer.one_euro.min_cutoff", 0.6)
    panel.set_value("gestures.cooldown_ms", 1000)
    written = application.save_tuning()
    assert written == [config_file]
    text = config_file.read_text(encoding="utf-8")
    assert "# ของฉัน" in text and "# เงียบ" in text
    data = tomllib.loads(text)
    assert data["pointer"]["one_euro"]["min_cutoff"] == 0.6
    assert data["gestures"]["cooldown_ms"] == 1000
    assert panel.overrides() == {}  # ค่าที่บันทึกกลายเป็นค่าตั้งต้นใหม่
    assert "บันทึกแล้ว" in panel._status.text()


def test_reset_goes_back_to_the_starting_values(qapp, app):
    application, _ = app
    panel = application.debug_view.tuning
    panel.set_value("gestures.cooldown_ms", 2000)
    application.reset_tuning()
    assert engine_config(application).gestures.cooldown_ms == 800
    assert panel.values["gestures.cooldown_ms"] == 800


def test_badge_notice_blinks_then_disappears(qapp):
    badge = StatusBadge(Theme.from_config(Config().hud))
    badge.notify("INCIDENT SAVED", 10.0)
    assert badge.notice == "INCIDENT SAVED" and badge.advance(10.5)
    assert not badge.advance(10.0 + StatusBadge.NOTICE_SECONDS + 0.01)
    assert badge.notice is None


def test_overlay_notify_shows_notice_and_counts_paints(qapp):
    from aether.hud.overlay import HudOverlay

    overlay = HudOverlay(parse_config({"hud": {"sounds": False}}), FHD)
    overlay.start(boot=False)
    try:
        overlay.notify("INCIDENT SAVED")
        assert overlay._badge.notice == "INCIDENT SAVED"
        overlay.grab()  # วาดลง pixmap (offscreen ไม่มีจอจริงให้ repaint)
        assert overlay.paint_count >= 1
    finally:
        overlay.stop()
