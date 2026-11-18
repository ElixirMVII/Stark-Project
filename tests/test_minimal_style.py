"""สไตล์ minimal (ค่าเริ่มต้นของ hud.style): theme, เส้นบาง+เงา, reticle, แผง, boot, overlay, หน้าต่าง"""

from __future__ import annotations

import itertools

import pytest
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPainterPath

from aether.config import Config, parse_config
from aether.gestures import EngineSnapshot, EventKind, GestureEvent, State
from aether.hud import reticle as reticle_module
from aether.hud.boot import DURATION, BootSequence
from aether.hud.launcher import LauncherMenu
from aether.hud.overlay import HudOverlay
from aether.hud.panels import (
    HandRadar,
    StatusBadge,
    TelemetryPanel,
    TelemetryValues,
    _fade_out,
    mode_text,
)
from aether.hud.radial import ChargeRing, Radial
from aether.hud.reticle import MINIMAL_MODES, MODES, Reticle
from aether.hud.target_lock import TargetLock, kind_text
from aether.hud.theme import Theme, hairline_stroke
from aether.mapping import ScreenGeometry
from aether.snap import Target

FHD = ScreenGeometry(0, 0, 1920, 1080)
VIEWPORT = QRectF(0, 0, 1280, 720)


@pytest.fixture
def theme(qapp) -> Theme:
    return Theme.from_config(Config().hud, None)


@pytest.fixture
def holo(qapp) -> Theme:
    return Theme.from_config(parse_config({"hud": {"style": "holo"}}).hud, None)


def snap(state: State = State.POINTER, armed: bool = True, **kw) -> EngineSnapshot:
    return EngineSnapshot(state=state, armed=armed, hand_present=True, cursor=(400, 300), **kw)


def paint(element, now: float, size=(800, 600)) -> QImage:
    image = QImage(*size, QImage.Format.Format_ARGB32)
    image.fill(Qt.GlobalColor.transparent)
    painter = QPainter(image)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    element.paint(painter, now)
    painter.end()
    return image


def lit_bounds(image: QImage, step: int = 2) -> QRectF:
    xs, ys = [], []
    for x in range(0, image.width(), step):
        for y in range(0, image.height(), step):
            if image.pixelColor(x, y).alpha() > 0:
                xs.append(x)
                ys.append(y)
    if not xs:
        return QRectF()
    return QRectF(min(xs), min(ys), max(xs) - min(xs) + 1, max(ys) - min(ys) + 1)


def is_accent(color: QColor, accent: QColor) -> bool:
    return (
        color.alpha() > 120
        and abs(color.red() - accent.red()) < 30
        and abs(color.green() - accent.green()) < 30
        and abs(color.blue() - accent.blue()) < 30
    )


# ---- theme ------------------------------------------------------------------ #


def test_minimal_theme_is_the_default(theme, holo):
    assert theme.minimal and theme.style == "minimal" and not holo.minimal
    assert theme.primary == QColor("#F2F4F7") and theme.accent == QColor("#D6B67A")
    assert holo.primary == QColor("#00E5FF")  # holo + theme auto = ธีม aether เดิม


def test_labels_keep_their_case_in_minimal_and_shout_in_holo(theme, holo):
    assert theme.label("Camera lost") == "Camera lost"
    assert holo.label("Camera lost") == "CAMERA LOST"


def test_minimal_fonts_are_calm(theme):
    title = theme.display_font(17, 800, spacing=7.0)
    assert title.family() == theme.text_family and title.weight() <= QFont.Weight.DemiBold
    assert title.letterSpacing() == 0.0  # ระยะห่างของ holo (ตัวพิมพ์ใหญ่) ไม่ใช้
    numbers = theme.mono_font(13)
    assert numbers.pixelSize() == 13 and numbers.featureValue(QFont.Tag("tnum")) == 1
    assert theme.text_font(12, 300, spacing=3.0).letterSpacing() == 3.0


def test_hairline_has_a_soft_shadow_under_it(qapp):
    image = QImage(60, 40, QImage.Format.Format_ARGB32)
    image.fill(Qt.GlobalColor.transparent)
    painter = QPainter(image)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    path = QPainterPath()
    path.moveTo(5, 20.5)
    path.lineTo(55, 20.5)
    hairline_stroke(painter, path, QColor("#FFFFFF"), 1.0)
    painter.end()
    line = image.pixelColor(30, 20)
    shadow = image.pixelColor(30, 21)  # ขอบนอกของเส้น 1 px: มีแต่เงา
    assert line.lightness() > 200 and line.alpha() > 200
    assert 0 < shadow.alpha() < 160 and shadow.lightness() < 60
    assert image.pixelColor(30, 26).alpha() == 0  # เส้นบางจริง ไม่ฟุ้ง


def test_theme_stroke_picks_the_style(theme, holo):
    def spread(t: Theme) -> int:
        image = QImage(60, 40, QImage.Format.Format_ARGB32)
        image.fill(Qt.GlobalColor.transparent)
        painter = QPainter(image)
        path = QPainterPath()
        path.moveTo(5, 20.5)
        path.lineTo(55, 20.5)
        t.stroke(painter, path, QColor("#FFFFFF"), 1.5)
        painter.end()
        return sum(1 for y in range(40) if image.pixelColor(30, y).alpha() > 0)

    assert spread(theme) < spread(holo)  # minimal ไม่มีชั้นเรืองแสงหนาๆ


# ---- reticle ---------------------------------------------------------------- #


def test_minimal_reticle_uses_its_own_modes_and_stays_small(theme, holo):
    assert set(MINIMAL_MODES) == set(MODES)
    small, big = Reticle(theme), Reticle(holo)
    assert small._modes is MINIMAL_MODES and big._modes is MODES
    for reticle in (small, big):
        reticle.set_state(QPointF(400, 300), snap(), 0.0)
        for i in range(60):
            reticle.advance(i / 60)
    assert small.rect().width() < big.rect().width() * 0.6


@pytest.mark.parametrize(
    "state",
    [
        snap(),
        snap(State.DRAGGING),
        snap(State.SCROLL),
        snap(State.TWO_HAND_ZOOM),
        snap(State.GESTURE),
        snap(State.CHARGING, charge=0.5, charge_kind="fist_hold"),
        snap(State.PINCH_PENDING, pinch_level=1.0, pinch_button="left", hold_progress=0.6),
        snap(State.DISARMED, armed=False),
    ],
)
def test_minimal_reticle_paints_inside_its_rect(theme, state):
    reticle = Reticle(theme, reduce_motion=True)
    t = 0.0
    for _ in range(40):
        reticle.set_state(QPointF(400, 300), state, t)
        reticle.advance(t)
        t += 1 / 60
    bounds = lit_bounds(paint(reticle, t), step=1)
    assert not bounds.isEmpty()
    assert reticle.rect().adjusted(-1, -1, 1, 1).contains(bounds)


def test_pinch_progress_is_an_accent_arc_on_the_ring(theme):
    reticle = Reticle(theme, reduce_motion=True)
    t = 0.0
    for _ in range(60):
        reticle.set_state(QPointF(400, 300), snap(pinch_level=0.5), t)
        reticle.advance(t)
        t += 1 / 60
    image = paint(reticle, t)
    radius = MINIMAL_MODES["POINTER"].inner.radius * reticle.scale
    top = image.pixelColor(400, round(300 - radius))  # ส่วนโค้งเริ่มที่บนสุด วนตามเข็ม
    right = image.pixelColor(round(400 + radius), 300)
    left = image.pixelColor(round(400 - radius), 300)  # ยังไม่ถึง (จีบครึ่งเดียว)
    assert is_accent(top, theme.accent) and is_accent(right, theme.accent)
    assert not is_accent(left, theme.accent) and left.alpha() > 120  # ตรงนั้นเป็นวงแหวนสีหลัก


def test_minimal_reticle_does_not_breathe_or_flicker(theme):
    reticle = Reticle(theme)  # แม้ไม่ได้เปิด reduce_motion
    t = 0.0
    for _ in range(200):
        reticle.set_state(QPointF(400, 300), snap(), t)
        reticle.advance(t)
        t += 1 / 60
    assert reticle._flicker == {}
    first = paint(reticle, t)
    second = paint(reticle, t + 0.7)  # ครึ่งรอบหายใจของ holo
    assert first == second


def test_minimal_click_ring_is_small(theme):
    reticle = Reticle(theme)
    reticle.add_shockwave(QPointF(400, 300), theme.primary, 0.0)
    bounds = lit_bounds(paint(reticle, 0.2), step=1)
    assert bounds.width() <= 2 * reticle_module.MINIMAL_SHOCKWAVE[1] + 6


def test_morph_vector_layout_matches_between_styles():
    assert len(reticle_module._vector(MINIMAL_MODES["POINTER"])) == len(
        reticle_module._vector(MODES["POINTER"])
    )


# ---- panels ----------------------------------------------------------------- #


def test_mode_text_is_readable():
    assert mode_text("TWO_HAND_ZOOM") == "Two hand zoom"
    assert mode_text("POINTER") == "Pointer"


def test_notice_fades_instead_of_blinking():
    values = [_fade_out(t / 10, 1.5) for t in range(16)]
    assert values[0] == 1.0 and values[-1] == 0.0
    assert all(a >= b for a, b in itertools.pairwise(values))


@pytest.mark.parametrize("armed", [True, False])
def test_minimal_badge_paints_a_rounded_card(theme, armed):
    badge = StatusBadge(theme, dry_run=True)
    badge.set_viewport(VIEWPORT)
    badge.set_armed(armed)
    badge.set_alert("Camera lost")
    rect = badge.rect()
    image = paint(badge, 1.0, (1280, 720))
    assert lit_bounds(image, 1).adjusted(-1, -1, 1, 1).intersects(rect)
    corner = image.pixelColor(round(rect.left()), round(rect.top()))
    inside = image.pixelColor(round(rect.left() + 20), round(rect.top() + 50))
    assert corner.alpha() < inside.alpha()  # มุมมน: มุมสุดของกรอบโปร่ง
    assert inside.lightness() < 60  # พื้นการ์ดเข้ม


def test_minimal_telemetry_and_radar_paint(theme):
    from handsim import make_hand

    telemetry = TelemetryPanel(theme)
    telemetry.set_viewport(VIEWPORT)
    telemetry.set_values(TelemetryValues(cpu=95.0, ram=40.0, fps=30.0, latency_ms=12.0))
    radar = HandRadar(theme, "Right")
    radar.set_viewport(VIEWPORT)
    radar.set_hands((make_hand(),), 640, 480, True)
    for element in (telemetry, radar):
        bounds = lit_bounds(paint(element, 1.0, (1280, 720)), 1)
        assert element.rect().adjusted(-2, -2, 2, 2).contains(bounds)
    image = paint(telemetry, 1.0, (1280, 720))
    rect = telemetry.rect()
    # แถบ CPU เกิน 85% เป็นสีเน้น
    row_y = round(rect.top() + 36 + 9)
    assert any(
        is_accent(image.pixelColor(x, row_y), theme.accent)
        for x in range(round(rect.left() + 64), round(rect.right() - 64))
    )


def test_minimal_charge_ring_and_radial_and_launcher_paint(theme):
    charge = ChargeRing(theme, {"fist_hold": "minimize_foreground"})
    charge.set_state(QPointF(400, 300), snap(State.CHARGING, charge=0.7, charge_kind="fist_hold"))
    assert charge.label() == "Minimize"
    radial = Radial(theme, {"swipe_left": "win+tab"})
    radial.set_state(QPointF(400, 300), snap(State.GESTURE, swipe_ready=True))
    radial.advance(0.0)
    radial.advance(0.5)
    launcher = LauncherMenu(theme, ["Browser", "Files", "Notes"])
    launcher.set_state(QPointF(400, 300), snap(State.LAUNCHER, launcher_selection=1))
    launcher.advance(0.0)
    launcher.advance(0.5)
    for element in (charge, radial, launcher):
        bounds = lit_bounds(paint(element, 0.5), 1)
        assert not bounds.isEmpty()
        assert element.rect().adjusted(-2, -2, 2, 2).contains(bounds)


def test_minimal_target_lock_is_a_rounded_frame_with_a_readable_label(theme):
    assert kind_text("MenuItem") == "Menu item" and kind_text("Button") == "Button"
    lock = TargetLock(theme)
    target = Target(100, 100, 80, 30, "SplitButton", "Save")
    for i in range(90):
        lock.set_state(QRectF(100, 100, 80, 30), QPointF(140, 115), snap(target=target))
        lock.advance(i / 60)
    assert lock.label == "Split button · Save"
    bounds = lit_bounds(paint(lock, 1.5), 1)
    assert lock.rect().adjusted(-2, -2, 2, 2).contains(bounds)


# ---- boot + overlay ------------------------------------------------------- #


@pytest.mark.parametrize("reduce_motion", [False, True])
def test_minimal_boot_is_a_quiet_wordmark(theme, reduce_motion):
    boot = BootSequence(theme, reduce_motion=reduce_motion)
    boot.start(0.0, QPointF(400, 300))
    rect = boot.rect()
    assert rect.width() < 500 and rect.height() < 200  # ไม่มีวงแหวนเต็มจอ
    for t in (0.2, 0.8, 1.3):
        bounds = lit_bounds(paint(boot, t), 1)
        assert not bounds.isEmpty() and rect.contains(bounds)
    assert boot.advance(DURATION - 0.01) and not boot.advance(DURATION + 0.01)


def test_minimal_overlay_skips_particles_streak_and_slide(qapp):
    hud = HudOverlay(parse_config({"hud": {"sounds": False}}), FHD)
    hud.start()
    try:
        assert hud._particles is None and "particles" not in hud._elements
        assert hud._telemetry is not None and hud._telemetry._slide_start is None
        hud.handle_events(
            (
                GestureEvent(EventKind.CLICK, x=960, y=540),
                GestureEvent(EventKind.ACTION, x=960, y=540, name="swipe_left"),
            )
        )
        assert not hud._streak.advance(hud._clock() + 0.01)
        hud.grab()
        assert hud.paint_count >= 1
    finally:
        hud.stop()


def test_minimal_draw_time_is_measured(qapp):
    hud = HudOverlay(parse_config({"hud": {"sounds": False}}), FHD)
    hud.start(boot=False)
    try:
        for _ in range(5):
            hud.handle_events((GestureEvent(EventKind.CLICK, x=960, y=540),))
            hud.grab()
        assert 0.0 < hud.draw_ms < 1000.0
    finally:
        hud.stop()


# ---- หน้าต่าง --------------------------------------------------------------- #


def test_app_style_is_dark_and_uses_the_accent(qapp):
    from PySide6.QtGui import QPalette

    from aether import ui_style

    sheet = ui_style.stylesheet()
    assert ui_style.ACCENT in sheet and "border-radius" in sheet
    saved = (qapp.palette(), qapp.styleSheet(), qapp.font(), qapp.style().name())
    try:
        ui_style.apply_app_style(qapp)
        assert qapp.styleSheet() == sheet
        window = qapp.palette().color(QPalette.ColorRole.Window)
        assert window.lightness() < 40
        assert qapp.palette().color(QPalette.ColorRole.Highlight) == QColor(ui_style.ACCENT)
    finally:
        palette, old_sheet, font, style = saved
        qapp.setStyleSheet(old_sheet)
        qapp.setPalette(palette)
        qapp.setFont(font)
        qapp.setStyle(style)
