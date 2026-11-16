"""HUD overlay: flag ของหน้าต่าง, การแปลง DPI, การวาดเฉพาะบริเวณที่เปลี่ยน, แอนิเมชัน และเสียง

รันบน Qt platform ``offscreen`` — ไม่ต้องมีจอ กล้อง หรืออุปกรณ์เสียง
"""

from __future__ import annotations

import numpy as np
import pytest
from handsim import PINCHED, make_frame, make_hand
from PySide6.QtCore import QPointF, QRect, QRectF, Qt
from PySide6.QtGui import QColor, QImage, QPainter

from aether.camera import FramePacket
from aether.config import Config, parse_config
from aether.gestures import EngineSnapshot, EventKind, GestureEvent, State
from aether.hud import boot as boot_module
from aether.hud import reticle as reticle_module
from aether.hud.boot import BootSequence
from aether.hud.overlay import OVERLAY_FLAGS, HudOverlay
from aether.hud.panels import (
    HandRadar,
    StatusBadge,
    TelemetryPanel,
    TelemetrySampler,
    TelemetryValues,
)
from aether.hud.reticle import Reticle
from aether.hud.theme import Theme, arc_path, chamfered_rect, with_alpha
from aether.mapping import ScreenGeometry
from aether.pipeline import FrameResult
from aether.sound import SOUND_NAMES, SoundPlayer

FHD = ScreenGeometry(0, 0, 1920, 1080)


class FakeClock:
    def __init__(self) -> None:
        self.t = 50.0

    def __call__(self) -> float:
        return self.t


class FakeSounds:
    def __init__(self) -> None:
        self.played: list[str] = []

    def play(self, name: str) -> None:
        self.played.append(name)


class FakeSampler:
    def __init__(self) -> None:
        self.calls = 0

    def sample(self, fps: float, latency_ms: float) -> TelemetryValues:
        self.calls += 1
        return TelemetryValues(cpu=20.0 + self.calls, ram=50.0, fps=fps, latency_ms=latency_ms)


def snapshot(state=State.POINTER, armed=True, cursor=(960, 540), **kwargs) -> EngineSnapshot:
    defaults = {"hand_present": True, "hands": 1, "pinch_ratio": 1.0}
    return EngineSnapshot(state=state, armed=armed, cursor=cursor, **{**defaults, **kwargs})


def packet(snap: EngineSnapshot, *hands) -> FramePacket:
    hands = hands or (make_hand(),)
    return FramePacket(
        image=np.zeros((2, 2, 3), np.uint8),
        result=FrameResult(make_frame(0, *hands), snap),
        captured_at=0.0,
        process_ms=18.0,
        fps=29.5,
    )


class Spy(HudOverlay):
    """จดทุกกรอบที่ถูกสั่งวาดใหม่"""

    def __init__(self, *args, **kwargs) -> None:
        self.updates: list[QRect] = []
        super().__init__(*args, **kwargs)

    def update(self, *args) -> None:  # type: ignore[override]
        self.updates.append(args[0] if args else self.rect())
        super().update(*args)


@pytest.fixture
def theme(qapp) -> Theme:
    return Theme.from_config(Config().hud, None)


@pytest.fixture
def hud(qapp, theme):
    clock = FakeClock()
    overlay = Spy(
        Config(),
        FHD,
        theme=theme,
        sounds=FakeSounds(),
        sampler=FakeSampler(),
        device_pixel_ratio=1.5,
        clock=clock,
    )
    overlay.resize(1280, 720)
    overlay.clock = clock  # type: ignore[attr-defined]
    yield overlay
    overlay.stop()


def run(overlay, seconds: float, steps: int = 30) -> None:
    for _ in range(steps):
        overlay.clock.t += seconds / steps
        overlay._tick()


def finish_boot(overlay) -> None:
    overlay.start()
    run(overlay, boot_module.DURATION + 0.2)
    overlay.updates.clear()


def render(overlay) -> QImage:
    return overlay.grab().toImage().convertToFormat(QImage.Format.Format_ARGB32)


def opaque_pixels(image: QImage, rect: QRect) -> int:
    count = 0
    for y in range(max(rect.top(), 0), min(rect.bottom() + 1, image.height()), 2):
        for x in range(max(rect.left(), 0), min(rect.right() + 1, image.width()), 2):
            count += image.pixelColor(x, y).alpha() > 0
    return count


# ---- หน้าต่าง: คลิกทะลุ ไม่รับ focus ไม่โผล่ taskbar ------------------------- #


def test_window_is_click_through_topmost_and_focusless(hud):
    flags = hud.windowFlags()
    for flag in (
        Qt.WindowType.FramelessWindowHint,
        Qt.WindowType.WindowStaysOnTopHint,
        Qt.WindowType.WindowTransparentForInput,
        Qt.WindowType.WindowDoesNotAcceptFocus,
    ):
        assert flags & flag, flag
    assert (flags & Qt.WindowType.Tool) == Qt.WindowType.Tool  # ไม่โผล่ใน taskbar
    assert OVERLAY_FLAGS & Qt.WindowType.WindowTransparentForInput
    assert hud.testAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
    assert hud.testAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
    assert hud.testAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
    assert hud.focusPolicy() == Qt.FocusPolicy.NoFocus


def test_overlay_covers_primary_screen_minus_one_row(qapp, theme):
    overlay = HudOverlay(Config(), FHD, theme=theme, sampler=FakeSampler())
    screen = qapp.primaryScreen().geometry()
    assert overlay.geometry() == screen.adjusted(0, 0, 0, -1)


def test_background_is_fully_transparent_where_nothing_is_drawn(hud):
    finish_boot(hud)
    image = render(hud)
    assert image.pixelColor(640, 360).alpha() == 0  # กลางจอ ไม่มีอะไร
    assert image.pixelColor(300, 200).alpha() == 0


# ---- DPI ----------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("dpr", "physical", "logical"),
    [
        (1.0, (960, 540), (960, 540)),
        (1.25, (960, 540), (768, 432)),
        (1.5, (960, 540), (640, 360)),
        (1.5, (1919, 1079), (1279.33, 719.33)),
        (2.0, (0, 0), (0, 0)),
    ],
)
def test_physical_cursor_maps_to_logical_overlay_position(qapp, theme, dpr, physical, logical):
    overlay = HudOverlay(Config(), FHD, theme=theme, sampler=FakeSampler(), device_pixel_ratio=dpr)
    point = overlay.to_logical(*physical)
    assert (point.x(), point.y()) == pytest.approx(logical, abs=0.01)


def test_mapping_accounts_for_screen_origin(qapp, theme):
    second = ScreenGeometry(1920, 100, 2560, 1440)
    overlay = HudOverlay(
        Config(), second, theme=theme, sampler=FakeSampler(), device_pixel_ratio=1.25
    )
    point = overlay.to_logical(1920 + 1250, 100 + 500)
    assert (point.x(), point.y()) == pytest.approx((1000, 400))


def test_reticle_is_drawn_at_the_cursor(hud):
    finish_boot(hud)
    hud.show_packet(packet(snapshot(cursor=(960, 540))))  # → logical (640, 360) ที่ 150%
    run(hud, 0.3)
    image = render(hud)
    assert image.pixelColor(640, 360).alpha() > 200  # จุดกลาง reticle
    assert opaque_pixels(image, QRect(580, 300, 120, 120)) > 500
    assert image.pixelColor(300, 360).alpha() == 0  # ที่อื่นยังโปร่งใส


# ---- วาดเฉพาะบริเวณที่เปลี่ยน ------------------------------------------------ #


def test_nothing_is_repainted_when_idle(hud):
    finish_boot(hud)
    run(hud, 2.0, steps=120)
    assert hud.updates == []


def test_moving_reticle_only_invalidates_its_neighbourhood(hud):
    finish_boot(hud)
    hud.show_packet(packet(snapshot(cursor=(960, 540))))
    run(hud, 0.3)
    hud.updates.clear()

    hud.show_packet(packet(snapshot(cursor=(990, 540))))  # ขยับ 20 logical px
    run(hud, 0.1, steps=6)
    reticle_updates = [r for r in hud.updates if r.intersects(QRect(560, 280, 200, 160))]
    assert reticle_updates
    for rect in hud.updates:
        assert rect.width() * rect.height() < 0.05 * 1280 * 720  # ไม่เคยวาดใหม่ทั้งจอ
    union = reticle_updates[0]
    for rect in reticle_updates:
        union = union.united(rect)
    assert union.width() <= 2 * reticle_module.HALF_EXTENT + 40
    assert union.height() <= 2 * reticle_module.HALF_EXTENT + 20


def test_old_reticle_position_is_erased(hud):
    finish_boot(hud)
    hud.show_packet(packet(snapshot(cursor=(300, 300))))
    run(hud, 0.3)
    hud.show_packet(packet(snapshot(cursor=(1500, 900))))
    run(hud, 0.3)
    image = render(hud)
    assert opaque_pixels(image, QRect(200 - 70, 200 - 70, 140, 140)) == 0
    assert image.pixelColor(1000, 600).alpha() > 200


def test_badge_repaints_only_when_state_changes(hud):
    finish_boot(hud)
    hud.show_packet(packet(snapshot()))
    hud.updates.clear()
    badge = hud._badge.rect().toAlignedRect()
    for _ in range(10):
        hud.show_packet(packet(snapshot()))
    assert not [r for r in hud.updates if r.intersects(badge)]
    hud.show_packet(packet(snapshot(State.SCROLL)))
    assert [r for r in hud.updates if r.intersects(badge)]


def test_telemetry_updates_on_its_own_timer_not_every_frame(hud):
    finish_boot(hud)
    panel = hud._telemetry.rect().toAlignedRect()
    hud.show_packet(packet(snapshot()))  # เฟรมแรก: สีของแผงเปลี่ยนตามสถานะ ARM หนึ่งครั้ง
    hud.updates.clear()
    for _ in range(30):
        hud.show_packet(packet(snapshot()))
    assert not [r for r in hud.updates if r.intersects(panel)]
    hud._sample_telemetry()
    assert [r for r in hud.updates if r.intersects(panel)]
    assert hud._telemetry._values.fps == pytest.approx(29.5)
    assert hud._telemetry._values.latency_ms == pytest.approx(18.0)


# ---- reticle ------------------------------------------------------------- #


def test_reticle_fades_in_and_out_with_hand(theme):
    reticle = Reticle(theme)
    reticle.advance(0.0)
    assert reticle.alpha == 0.0 and reticle.rect().isEmpty()
    reticle.set_state(QPointF(100, 100), snapshot(), 0.0)
    reticle.advance(0.05)
    assert 0.0 < reticle.alpha < 1.0
    assert reticle.advance(0.5) and reticle.alpha == 1.0
    assert reticle.rect().contains(QPointF(100, 100))

    reticle.set_state(QPointF(100, 100), snapshot(State.IDLE, hand_present=False), 0.5)
    reticle.advance(0.6)
    assert reticle.advance(1.5) is False  # หายสนิท ไม่มีอะไรต้องวาดแล้ว
    assert reticle.alpha == 0.0 and reticle.rect().isEmpty()


def test_reticle_spins_only_when_armed(theme):
    armed, disarmed = Reticle(theme), Reticle(theme)
    armed.set_state(QPointF(0, 0), snapshot(), 0.0)
    disarmed.set_state(QPointF(0, 0), snapshot(State.DISARMED, armed=False), 0.0)
    for r in (armed, disarmed):
        r.advance(0.0)
        for i in range(1, 11):
            r.advance(i * 0.1)
    assert armed._angle == pytest.approx(reticle_module.SPIN_DEG_PER_S, rel=0.01)
    assert disarmed._angle == 0.0


def test_shockwave_expands_and_expires(theme):
    reticle = Reticle(theme)
    reticle.advance(10.0)
    reticle.add_shockwave(QPointF(400, 300), theme.primary, 10.0)
    assert reticle.advance(10.1)
    assert reticle.rect().contains(QRectF(400 - 74, 300 - 74, 148, 148))
    assert reticle.advance(10.0 + reticle_module.SHOCKWAVE_SECONDS + 0.01) is False
    assert reticle.rect().isEmpty()


def test_drag_leaves_a_fading_trail(theme):
    reticle = Reticle(theme)
    for i in range(10):
        t = i * 0.033
        reticle.set_state(QPointF(100 + i * 30, 200), snapshot(State.DRAGGING), t)
        reticle.advance(t)
    assert len(reticle._trail) == 10
    assert reticle.rect().left() < 110 and reticle.rect().right() > 370
    reticle.set_state(QPointF(370, 200), snapshot(State.POINTER), 0.4)
    reticle.advance(0.4 + reticle_module.TRAIL_SECONDS + 0.1)
    assert len(reticle._trail) == 0  # trail จางหมดหลังเลิกลาก


@pytest.mark.parametrize(
    "snap",
    [
        snapshot(),
        snapshot(State.DISARMED, armed=False),
        snapshot(State.PINCH_PENDING, pinch_level=1.0, pinch_button="left", hold_progress=0.5),
        snapshot(State.PINCH_PENDING, pinch_level=1.0, pinch_button="right"),
        snapshot(State.DRAGGING, pinch_level=1.0, pinch_button="left", hold_progress=1.0),
        snapshot(State.SCROLL, gesture="Victory"),
        snapshot(pinch_level=0.5),
    ],
    ids=lambda s: s.state.value,
)
def test_reticle_paints_every_state_inside_its_rect(theme, snap):
    reticle = Reticle(theme)
    center = QPointF(150, 150)
    reticle.set_state(center, snap, 0.0)
    reticle.advance(0.0)
    reticle.advance(1.0)
    reticle.add_shockwave(center, theme.accent, 1.0)
    reticle.advance(1.1)

    image = QImage(300, 300, QImage.Format.Format_ARGB32)
    image.fill(Qt.GlobalColor.transparent)
    painter = QPainter(image)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    reticle.paint(painter, 1.1)
    painter.end()

    rect = reticle.rect().toAlignedRect()
    assert opaque_pixels(image, rect) > 300
    outside = sum(
        image.pixelColor(x, y).alpha() > 0
        for x in range(0, 300, 3)
        for y in range(0, 300, 3)
        if not rect.contains(x, y)
    )
    assert outside == 0  # ไม่วาดล้นกรอบที่แจ้งไว้ (ไม่งั้นจะมีเศษภาพค้างบนจอ)


# ---- panels -------------------------------------------------------------- #


def test_status_badge_change_detection(theme):
    badge = StatusBadge(theme, dry_run=True)
    badge.set_viewport(QRectF(0, 0, 1280, 720))
    assert badge.rect().right() == pytest.approx(1280 - 24)
    assert badge.rect().top() == pytest.approx(24)
    assert badge.set_snapshot(snapshot()) is True
    assert badge.set_snapshot(snapshot()) is False
    assert badge.set_snapshot(snapshot(hands=2)) is True
    assert badge.set_armed(False) is True and badge._mode == "DISARMED"
    assert badge.set_armed(False) is False
    assert badge.set_alert("CAMERA LOST") is True
    assert badge.set_alert("CAMERA LOST") is False
    assert badge.set_alert(None) is True


def test_panels_sit_in_their_corners(theme):
    viewport = QRectF(0, 0, 1280, 720)
    telemetry, radar = TelemetryPanel(theme), HandRadar(theme, "right")
    telemetry.set_viewport(viewport)
    radar.set_viewport(viewport)
    assert telemetry.rect().left() == 24 and telemetry.rect().bottom() == 720 - 24
    assert radar.rect().right() == 1280 - 24 and radar.rect().bottom() == 720 - 24
    assert not telemetry.rect().intersects(radar.rect())


def test_telemetry_panel_change_detection(theme):
    panel = TelemetryPanel(theme)
    values = TelemetryValues(cpu=10, ram=20, fps=30, latency_ms=15, clock="12:00:00")
    assert panel.set_values(values) is True
    assert panel.set_values(values) is False
    assert panel.set_armed(True) is True and panel.set_armed(True) is False


def test_telemetry_sampler_reads_real_system_values():
    values = TelemetrySampler().sample(fps=28.0, latency_ms=21.0)
    assert 0.0 <= values.cpu <= 100.0 and 0.0 < values.ram <= 100.0
    assert (values.fps, values.latency_ms) == (28.0, 21.0)
    assert len(values.clock) == 8 and values.clock.count(":") == 2


def test_radar_zooms_hand_to_fit_and_prefers_dominant(theme):
    radar = HandRadar(theme, "right")
    area = QRectF(0, 0, 100, 100)
    assert radar.skeleton_points(area) == []
    assert radar.set_hands((), 640, 480, True) is True  # armed เปลี่ยน
    assert radar.set_hands((), 640, 480, True) is False

    left, right = make_hand(0.2, 0.5, side="left"), make_hand(0.7, 0.3, scale=40.0)
    assert radar.set_hands((left, right), 640, 480, True) is True
    assert radar._hand is right  # มือข้างถนัด แม้จะไม่ใช่ตัวแรก
    points = radar.skeleton_points(area)
    assert len(points) == 21
    xs, ys = [p.x() for p in points], [p.y() for p in points]
    assert min(xs) >= -0.01 and max(xs) <= 100.01 and min(ys) >= -0.01 and max(ys) <= 100.01
    assert max(max(xs) - min(xs), max(ys) - min(ys)) == pytest.approx(100.0)  # เต็มกรอบ

    # มือเล็ก/ใหญ่ในเฟรม (ไกล/ใกล้กล้อง) ได้รูปขนาดเท่ากัน
    radar.set_hands((make_hand(0.5, 0.5, scale=150.0),), 640, 480, True)
    big = radar.skeleton_points(area)

    def flat(pts):
        return [v for p in pts for v in (p.x(), p.y())]

    assert flat(big) == pytest.approx(flat(points), abs=0.5)

    radar.set_hands((left,), 640, 480, True)
    assert radar._hand is left  # ไม่เห็นมือข้างถนัด → ใช้มือที่เห็น


def test_panels_can_be_disabled(qapp, theme):
    config = parse_config({"hud": {"show_telemetry": False, "show_radar": False}})
    overlay = HudOverlay(config, FHD, theme=theme, sampler=FakeSampler())
    assert set(overlay._elements) == {"boot", "badge", "reticle"}
    overlay.start(boot=False)
    overlay.show_packet(packet(snapshot()))
    render(overlay)
    overlay.stop()


def test_opacity_setting_scales_everything(qapp):
    def badge_alpha(opacity: float) -> int:
        config = parse_config({"hud": {"opacity": opacity}})
        overlay = HudOverlay(
            config, FHD, theme=Theme.from_config(config.hud, None), sampler=FakeSampler()
        )
        overlay.resize(1280, 720)
        overlay.start(boot=False)
        image = render(overlay)
        overlay.stop()
        badge = overlay._badge.rect()
        return image.pixelColor(int(badge.center().x()), int(badge.top() + 60)).alpha()

    full, half = badge_alpha(1.0), badge_alpha(0.5)
    assert full > 0 and half == pytest.approx(full * 0.5, abs=3)


# ---- boot ---------------------------------------------------------------- #


def test_boot_sequence_runs_for_duration_then_clears(hud):
    hud.start()
    assert hud._sounds.played == ["boot"]
    run(hud, 0.8)
    image = render(hud)
    assert opaque_pixels(image, QRect(640 - 280, 360 - 280, 560, 560)) > 2000
    assert hud._boot.active(hud.clock.t)

    run(hud, 1.0)
    assert not hud._boot.active(hud.clock.t)
    image = render(hud)
    assert opaque_pixels(image, QRect(640 - 280, 360 - 280, 560, 560)) == 0  # ลบออกหมดแล้ว
    hud.updates.clear()
    run(hud, 1.0)
    assert hud.updates == []


def test_boot_sequence_object(theme):
    boot = BootSequence(theme)
    assert not boot.active(0.0) and boot.rect().isEmpty()
    boot.start(5.0, QPointF(640, 360))
    assert boot.active(5.0) and boot.advance(6.0)
    assert boot.rect().center() == QPointF(640, 360)
    assert boot.advance(5.0 + boot_module.DURATION) is False
    assert boot.rect().isEmpty()
    assert boot_module.DURATION == pytest.approx(1.5)


def test_start_without_boot(hud):
    hud.start(boot=False)
    assert hud._sounds.played == []
    assert not hud._boot.active(hud.clock.t)


# ---- events → shockwave + เสียง -------------------------------------------- #


def test_events_trigger_sounds_and_shockwaves(hud):
    finish_boot(hud)
    hud._sounds.played.clear()
    hud.handle_events(
        (
            GestureEvent(EventKind.ARMED),
            GestureEvent(EventKind.CLICK, x=300, y=300),
            GestureEvent(EventKind.CLICK, x=600, y=300, button="right"),
            GestureEvent(EventKind.BUTTON_DOWN, x=900, y=300),
            GestureEvent(EventKind.BUTTON_UP, x=1200, y=300),
            GestureEvent(EventKind.SCROLL, dy=1),
            GestureEvent(EventKind.DISARMED),
        )
    )
    assert hud._sounds.played == ["arm", "click", "click", "click", "release", "disarm"]
    waves = hud._reticle._shockwaves
    assert [(w.center.x(), w.center.y()) for w in waves] == [
        (200, 200),
        (400, 200),
        (600, 200),
        (800, 200),
    ]
    assert waves[0].color == hud._theme.primary and waves[1].color == hud._theme.accent
    assert hud._badge._armed is False

    run(hud, 0.1, steps=4)
    image = render(hud)
    assert opaque_pixels(image, QRect(200 - 80, 200 - 80, 160, 160)) > 50
    run(hud, 1.0)
    assert opaque_pixels(render(hud), QRect(0, 100, 1000, 200)) == 0  # shockwave หายหมด


def test_set_armed_updates_badge_without_a_frame(hud):
    finish_boot(hud)
    hud.set_armed(True)
    assert hud._badge._armed is True and hud._badge._mode == "IDLE"
    assert hud.updates  # สั่งวาดป้ายใหม่ทันที


def test_alert_shows_and_plays_error_once(hud):
    finish_boot(hud)
    hud._sounds.played.clear()
    hud.set_alert("CAMERA LOST")
    hud.set_alert("CAMERA LOST")
    assert hud._sounds.played == ["error"]
    hud.set_alert(None)
    assert hud._badge._alert is None and hud._sounds.played == ["error"]


def test_full_sequence_renders_without_error(hud):
    """ไล่ทุกสถานะแล้ว render จริง — จับ error ใน paint path"""
    hud.start()
    states = [
        snapshot(State.DISARMED, armed=False),
        snapshot(State.IDLE, hand_present=False, hands=0, cursor=None),
        snapshot(State.POINTER, pinch_level=0.4),
        snapshot(State.PINCH_PENDING, pinch_level=1.0, pinch_button="left", hold_progress=0.7),
        snapshot(State.DRAGGING, pinch_level=1.0, pinch_button="left", hold_progress=1.0, hands=2),
        snapshot(State.SCROLL, gesture="Victory"),
    ]
    for i, snap in enumerate(states):
        hands = (make_hand(0.3 + 0.05 * i, 0.4, pinch=PINCHED), make_hand(0.7, 0.5, side="left"))
        hud.show_packet(packet(snap, *hands[: max(snap.hands, 0)]))
        hud.handle_events((GestureEvent(EventKind.CLICK, x=100 * i, y=100 * i),))
        run(hud, 0.2, steps=6)
        assert not render(hud).isNull()


# ---- theme --------------------------------------------------------------- #


def test_theme_from_config_and_font_fallback(qapp):
    config = parse_config({"hud": {"primary": "#11AA33", "accent": "#AA5500", "opacity": 0.7}})
    theme = Theme.from_config(config.hud, None)
    assert theme.primary == QColor("#11AA33") and theme.accent == QColor("#AA5500")
    assert theme.opacity == 0.7
    assert theme.display_family and theme.mono_family  # ไม่มีไฟล์ฟอนต์ → ใช้ฟอนต์ของระบบ
    assert theme.display_is_variable is False
    assert theme.state_color(True) == theme.primary
    assert theme.state_color(False) == theme.disarmed
    assert theme.display_font(17).pixelSize() == 17
    assert theme.mono_font(13).pixelSize() == 13


def test_theme_loads_bundled_fonts_when_present(qapp, tmp_path):
    from aether.config import assets_dir

    fonts = assets_dir() / "fonts"
    if not (fonts / "Orbitron.ttf").is_file():
        pytest.skip("ยังไม่ได้โหลดฟอนต์ (python tools/download_assets.py)")
    theme = Theme.from_config(Config().hud, fonts)
    assert theme.display_family == "Orbitron" and theme.mono_family == "Share Tech Mono"
    assert theme.display_is_variable
    assert theme.display_font(20, 800).family() == "Orbitron"
    # โฟลเดอร์ที่ไม่มีฟอนต์ → ถอยไปใช้ฟอนต์ระบบโดยไม่ error
    assert Theme.from_config(Config().hud, tmp_path).display_family != ""


def test_drawing_helpers():
    assert with_alpha(QColor("#00E5FF"), 0.5).alphaF() == pytest.approx(0.5, abs=0.01)
    assert with_alpha(QColor("#00E5FF"), 3.0).alphaF() == 1.0

    arc = arc_path(100, 100, 50, 0, 90)  # จากบนสุด กวาดตามเข็ม 90°
    start, end = arc.pointAtPercent(0.0), arc.pointAtPercent(1.0)
    assert (start.x(), start.y()) == pytest.approx((100, 50), abs=0.01)
    assert (end.x(), end.y()) == pytest.approx((150, 100), abs=0.01)

    panel = chamfered_rect(QRectF(0, 0, 100, 50), 10)
    assert not panel.contains(QPointF(2, 2))  # มุมซ้ายบนถูกตัด
    assert panel.contains(QPointF(98, 2)) and panel.contains(QPointF(50, 25))


# ---- sound --------------------------------------------------------------- #


def test_sound_player_disabled_or_missing_files_is_silent(qapp, tmp_path):
    assert SoundPlayer(tmp_path, enabled=False).available == frozenset()
    player = SoundPlayer(tmp_path, enabled=True)  # ไม่มีไฟล์ .wav เลย
    assert player.available == frozenset()
    for name in (*SOUND_NAMES, "unknown"):
        player.play(name)  # ต้องไม่ raise


def test_sound_player_loads_existing_wavs(qapp, tmp_path):
    sf = pytest.importorskip("soundfile")
    tone = (0.2 * np.sin(np.linspace(0, 440 * 2 * np.pi * 0.05, 2205))).astype(np.float32)
    for name in ("click", "boot"):
        sf.write(tmp_path / f"{name}.wav", tone, 44100, subtype="PCM_16")
    player = SoundPlayer(tmp_path, enabled=True, volume=0.3)
    assert player.available == frozenset({"click", "boot"})
    player.play("click")  # ไม่มีอุปกรณ์เสียงก็ต้องไม่ raise
    player.play("arm")


def test_sound_names_match_convert_tool():
    from conftest import load_tool

    assert load_tool("convert_sounds").ROLE_NAMES == SOUND_NAMES
