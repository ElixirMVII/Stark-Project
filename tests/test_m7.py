"""M7 ขัด animation: แปลงร่าง reticle, ความลึก, effect, reduce_motion / quality, เวลาวาด (Qt offscreen)

ส่วนใหญ่ทดสอบสไตล์ holo (แบบที่ M7 ออกแบบไว้) — การแปลงร่างทดสอบทั้งสองสไตล์ ส่วน minimal ดู test_minimal_style.py
"""

from __future__ import annotations

import itertools

import pytest
from handsim import Driver, make_hand
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QImage, QPainter

from aether.config import Config, parse_config
from aether.gestures import EngineSnapshot, EventKind, GestureEngine, GestureEvent, State
from aether.hud import reticle as reticle_module
from aether.hud import theme as theme_module
from aether.hud.boot import TEXT, BootSequence, decoded_text
from aether.hud.effects import MAX_PARTICLES, MinimizeCollapse, ParticlePool, SwipeStreak
from aether.hud.overlay import HudOverlay
from aether.hud.panels import TelemetryPanel
from aether.hud.reticle import MODES, Reticle, depth_scale, reticle_mode
from aether.hud.theme import Theme
from aether.mapping import ScreenGeometry

FHD = ScreenGeometry(0, 0, 1920, 1080)


HOLO = {"style": "holo"}


def style_theme(style: str) -> Theme:
    return Theme.from_config(parse_config({"hud": {"style": style}}).hud)


@pytest.fixture
def theme(qapp):
    return style_theme("holo")


def snap(state: State = State.POINTER, armed: bool = True, **kw) -> EngineSnapshot:
    return EngineSnapshot(state=state, armed=armed, hand_present=True, cursor=(100, 100), **kw)


SNAPSHOTS = {
    "POINTER": snap(),
    "DRAGGING": snap(State.DRAGGING),
    "SCROLL": snap(State.SCROLL),
    "ZOOM": snap(State.TWO_HAND_ZOOM),
    "GESTURE": snap(State.GESTURE),
    "CHARGING": snap(State.CHARGING, charge=0.5, charge_kind="fist_hold"),
    "DISARMED": snap(State.DISARMED, armed=False),
}


def test_every_state_maps_to_a_mode():
    assert {reticle_mode(s) for s in SNAPSHOTS.values()} == set(MODES)
    assert reticle_mode(snap(State.PINCH_PENDING)) == "POINTER"
    assert reticle_mode(snap(State.WINDOW_DRAG)) == "DRAGGING"


def _settle_seconds(reticle: Reticle, mode: str, t0: float) -> float:
    start = reticle_module._vector(reticle._modes[reticle.mode])
    reticle.set_state(QPointF(100, 100), SNAPSHOTS[mode], t0)
    goal = reticle_module._vector(reticle._modes[mode])
    t = t0
    while t < t0 + 1.0:
        t += 1 / 60
        reticle.advance(t)
        values = reticle._morph.value
        errors = [
            abs(v - g) / max(abs(g - s), 1e-6) if abs(g - s) > 1e-6 else abs(v - g)
            for v, g, s in zip(values, goal, start)
        ]
        if max(errors) <= 0.05:
            return t - t0
    return 99.0


@pytest.mark.parametrize("style", ["holo", "minimal"])
@pytest.mark.parametrize(("a", "b"), list(itertools.permutations(MODES, 2)))
def test_morph_between_any_two_modes_finishes_within_250_ms(qapp, style, a, b):
    reticle = Reticle(style_theme(style))
    reticle.set_state(QPointF(100, 100), SNAPSHOTS[a], 0.0)
    reticle.advance(0.0)
    for i in range(1, 90):
        reticle.advance(i / 60)  # อยู่โหมด a จนนิ่ง
    assert _settle_seconds(reticle, b, 1.5) <= 0.25


def test_morph_is_continuous_not_a_cut(theme):
    reticle = Reticle(theme)
    reticle.set_state(QPointF(100, 100), SNAPSHOTS["POINTER"], 0.0)
    for i in range(90):
        reticle.advance(i / 60)
    reticle.set_state(QPointF(100, 100), SNAPSHOTS["GESTURE"], 1.5)
    for i in range(1, 4):  # 3 เฟรม (50 ms) หลังเปลี่ยนโหมด
        reticle.advance(1.5 + i / 60)
    outer = reticle._morph.value[0]
    assert MODES["POINTER"].outer.radius < outer < MODES["GESTURE"].outer.radius  # อยู่ระหว่างทาง


def test_still_hand_reticle_does_not_jitter(theme):
    """มือนิ่ง: ตำแหน่ง ขนาด และรูปร่างของ reticle นิ่ง (ไม่สั่นจนเห็น)"""
    reticle = Reticle(theme, reduce_motion=True)
    state = snap(hand_size=0.15)
    t = 0.0
    for _ in range(120):
        reticle.set_state(QPointF(500, 400), state, t)
        reticle.advance(t)
        t += 1 / 60
    scales, outers = [], []
    for _ in range(60):
        reticle.set_state(QPointF(500, 400), state, t)
        reticle.advance(t)
        scales.append(reticle.scale)
        outers.append(reticle._morph.value[0])
        t += 1 / 60
    assert reticle.position == QPointF(500, 400)
    assert max(scales) - min(scales) < 0.002 and max(outers) - min(outers) < 0.05


def test_engine_cursor_is_still_when_the_hand_is_still():
    engine = GestureEngine(Config(), FHD)
    driver = Driver(engine)
    engine.set_armed(True)
    driver.hold(make_hand(), 500)
    cursors = set()
    for _ in range(30):
        driver.step(make_hand())
        cursors.add(engine.snapshot.cursor)
    assert len(cursors) == 1


def test_depth_scale_from_hand_size():
    assert depth_scale(None) == 1.0
    assert depth_scale(0.05) == pytest.approx(0.85)
    assert depth_scale(0.30) == pytest.approx(1.25)
    assert depth_scale(0.10) < depth_scale(0.16) < depth_scale(0.22)


def test_engine_reports_hand_size():
    engine = GestureEngine(Config(), FHD)
    Driver(engine).hold(make_hand(scale=100.0), 100)
    assert engine.snapshot.hand_size == pytest.approx(100.0 / 640, rel=0.01)
    Driver(engine).hold(None, 400)
    assert engine.snapshot.hand_size is None


def test_reticle_grows_when_the_hand_comes_closer(theme):
    reticle = Reticle(theme)
    t = 0.0
    for size in (0.10, 0.22):
        for _ in range(120):
            reticle.set_state(QPointF(100, 100), snap(hand_size=size), t)
            reticle.advance(t)
            t += 1 / 60
        if size == 0.10:
            far = reticle.scale
    assert far == pytest.approx(0.85, abs=0.02) and reticle.scale == pytest.approx(1.25, abs=0.02)


def test_click_pulses_the_reticle(theme):
    reticle = Reticle(theme)
    reticle.set_state(QPointF(100, 100), snap(), 0.0)
    reticle.advance(0.0)
    reticle.advance(0.5)
    reticle.add_shockwave(QPointF(100, 100), theme.primary, 0.5)
    assert reticle.scale == pytest.approx(0.82, abs=0.01)
    peak = max((reticle.advance(0.5 + i / 60), reticle.scale)[1] for i in range(1, 40))
    assert peak > 1.0  # เด้งเกินแล้วกลับ (bouncy)
    for i in range(40, 120):
        reticle.advance(0.5 + i / 60)
    assert reticle.scale == pytest.approx(1.0, abs=0.01)


def test_breathing_changes_alpha_unless_reduce_motion(theme):
    def lit(reduce: bool, now: float) -> int:
        reticle = Reticle(theme, reduce_motion=reduce, seed=1)
        reticle.set_state(QPointF(150, 150), snap(), 0.0)
        for i in range(60):
            reticle.advance(i / 60)
        reticle._flicker.clear()
        reticle._last_time = now - 1 / 60
        reticle.advance(now)
        reticle._flicker.clear()
        image = QImage(300, 300, QImage.Format.Format_ARGB32)
        image.fill(Qt.GlobalColor.transparent)
        painter = QPainter(image)
        reticle._paint_reticle(painter, QPointF(150, 150), 1.0, now)
        painter.end()
        return sum(image.pixelColor(x, 150).alpha() for x in range(80, 220))

    # เฟส 1/4 กับ 3/4 ของรอบหายใจ (3 วินาที): สว่างไม่เท่ากันถ้าเปิดการหายใจ
    assert lit(False, 3.75) != lit(False, 5.25)
    assert lit(True, 3.75) == lit(True, 5.25)


def test_scroll_rail_lights_the_arrow_of_the_last_direction(theme):
    reticle = Reticle(theme)
    reticle.set_state(QPointF(150, 150), SNAPSHOTS["SCROLL"], 0.0)
    for i in range(30):
        reticle.advance(i / 60)
    reticle.handle_event(EventKind.SCROLL, 2, 0.5)
    assert reticle._scroll_dir == 1
    reticle.handle_event(EventKind.SCROLL, -1, 0.6)
    assert reticle._scroll_dir == -1


@pytest.mark.parametrize("mode", list(MODES))
def test_every_mode_paints_inside_its_rect_even_when_near_and_pulsing(theme, mode):
    reticle = Reticle(theme)
    center = QPointF(200, 200)
    state = SNAPSHOTS[mode]
    t = 0.0
    for _ in range(90):
        reticle.set_state(center, EngineSnapshot(**{**_fields(state), "hand_size": 0.3}), t)
        reticle.advance(t)
        t += 1 / 60
    reticle.add_shockwave(center, theme.accent, t)
    for _ in range(8):  # ช่วงที่เด้งเกินขนาด
        t += 1 / 60
        reticle.advance(t)
    image = QImage(400, 400, QImage.Format.Format_ARGB32)
    image.fill(Qt.GlobalColor.transparent)
    painter = QPainter(image)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    reticle.paint(painter, t)
    painter.end()
    rect = reticle.rect().toAlignedRect()
    outside = sum(
        image.pixelColor(x, y).alpha() > 0
        for x in range(0, 400, 2)
        for y in range(0, 400, 2)
        if not rect.contains(x, y)
    )
    assert outside == 0


def _fields(s: EngineSnapshot) -> dict:
    import dataclasses

    return {f.name: getattr(s, f.name) for f in dataclasses.fields(s)}


# ---- effects ------------------------------------------------------------ #


def test_particle_pool_is_bounded_and_reused(theme):
    pool = ParticlePool(theme, per_burst=14)
    objects = {id(p) for p in pool._pool}
    assert pool.capacity == MAX_PARTICLES == 150
    for i in range(20):  # 20 คลิก × 14 = 280 > 150: ห้ามเกิน pool
        pool.burst(QPointF(100, 100), theme.primary, i * 0.01)
    assert pool.alive == 150
    assert {id(p) for p in pool._pool} == objects  # ไม่สร้าง object ใหม่
    assert pool.advance(0.1) and not pool.rect().isEmpty()
    assert pool.advance(0.25)
    assert not pool.advance(1.0) and pool.alive == 0 and pool.rect().isEmpty()
    pool.burst(QPointF(10, 10), theme.primary, 2.0)
    assert pool.alive == 14 and {id(p) for p in pool._pool} == objects


def test_particles_move_outward_and_fade(theme):
    pool = ParticlePool(theme, per_burst=10, seed=2)
    pool.burst(QPointF(200, 200), theme.primary, 0.0)
    pool.advance(0.0)
    pool.advance(0.2)
    assert all((p.x - 200) ** 2 + (p.y - 200) ** 2 > 100 for p in pool._pool if p.alive)
    image = QImage(400, 400, QImage.Format.Format_ARGB32)
    image.fill(0)
    painter = QPainter(image)
    pool.paint(painter, 0.2)
    painter.end()
    assert any(image.pixelColor(x, y).alpha() for x in range(0, 400, 2) for y in range(0, 400, 2))


def test_swipe_streak_crosses_the_screen(theme):
    streak = SwipeStreak(theme)
    streak.set_viewport(QRectF(0, 0, 1000, 600))
    streak.start("right", 1.0)
    assert streak.advance(1.0) and streak.rect() == QRectF(0, 0, 1000, 600)
    early, late = streak._band(0.1), streak._band(0.9)
    assert late.left() > early.left()
    assert not streak.advance(2.0) and streak.rect().isEmpty()
    streak.start("up", 3.0)
    assert streak._band(0.9).top() < streak._band(0.1).top()


def test_minimize_collapse_shrinks_toward_the_taskbar(theme):
    collapse = MinimizeCollapse(theme)
    window = QRectF(200, 100, 800, 600)
    goal = QPointF(960, 1075)
    collapse.start(window, goal, 0.0)
    assert collapse.advance(0.0) and collapse.rect().contains(window)
    mid, _alpha = collapse._current(0.2)
    assert mid.width() < window.width() and mid.center().y() > window.center().y()
    assert not collapse.advance(1.0) and collapse.rect().isEmpty()


# ---- boot v2 / telemetry ------------------------------------------------- #


def test_boot_text_decodes_into_system_online():
    early = decoded_text(TEXT, 0.25, frame=3)
    assert early != TEXT and len(early) == len(TEXT)
    assert decoded_text(TEXT, 0.25, frame=3) == early  # deterministic
    assert decoded_text(TEXT, 0.25, frame=4) != early  # ตัวสุ่มเปลี่ยนตามเวลา
    assert decoded_text(TEXT, 1.0, frame=9) == TEXT
    assert decoded_text(TEXT, 0.0, frame=0).strip() == ""


def test_reduced_motion_boot_only_fades_text(theme):
    full, reduced = BootSequence(theme), BootSequence(theme, reduce_motion=True)
    for boot in (full, reduced):
        boot.start(0.0, QPointF(400, 300))

    def lit(boot: BootSequence, now: float) -> int:
        image = QImage(800, 600, QImage.Format.Format_ARGB32)
        image.fill(0)
        painter = QPainter(image)
        boot.paint(painter, now)
        painter.end()
        return sum(
            1 for x in range(0, 800, 4) for y in range(0, 600, 4) if image.pixelColor(x, y).alpha()
        )

    assert lit(full, 0.6) > lit(reduced, 0.6) * 2  # ไม่มีวงแหวน/tick
    assert lit(reduced, 0.6) > 0


def test_telemetry_slides_in_from_the_left(theme):
    panel = TelemetryPanel(theme)
    panel.set_viewport(QRectF(0, 0, 1920, 1080))
    home = panel.rect()
    panel.slide_in(1.0)
    assert panel.advance(1.0) and panel.rect().right() <= 0  # ยังอยู่นอกจอ
    panel.advance(1.2)
    assert home.left() > panel.rect().left() > -panel.WIDTH
    assert panel.advance(2.0) and panel.rect() == home
    assert not panel.advance(2.1)


# ---- overlay: options, effects wiring, draw time ------------------------- #


def overlay(config: Config) -> HudOverlay:
    hud = HudOverlay(config, FHD)
    hud.start(boot=False)
    return hud


def test_reduce_motion_removes_particles(qapp):
    hud = overlay(parse_config({"hud": {**HOLO, "sounds": False, "reduce_motion": True}}))
    try:
        assert "particles" not in hud._elements and hud._particles is None
        hud.handle_events((GestureEvent(EventKind.CLICK, x=500, y=500),))  # ต้องไม่ error
    finally:
        hud.stop()


def test_low_quality_halves_particles_and_drops_the_outer_glow(qapp):
    hud = overlay(parse_config({"hud": {**HOLO, "sounds": False, "quality": "low"}}))
    try:
        assert hud._particles._per_burst == 7
        assert theme_module._glow_layers == theme_module._GLOW_LAYERS[1:]
    finally:
        hud.stop()
        theme_module.set_glow_quality(True)
    high = overlay(parse_config({"hud": {**HOLO, "sounds": False}}))
    try:
        assert (
            high._particles._per_burst == 14
            and theme_module._glow_layers == theme_module._GLOW_LAYERS
        )
    finally:
        high.stop()


def test_quality_must_be_high_or_low():
    from aether.config import ConfigError

    with pytest.raises(ConfigError):
        parse_config({"hud": {"quality": "ultra"}})


def test_overlay_events_start_effects(qapp):
    hud = overlay(parse_config({"hud": {**HOLO, "sounds": False}}))
    try:
        snapshot = EngineSnapshot(
            state=State.GESTURE, armed=True, hand_present=True, cursor=(960, 540)
        )

        class Packet:
            def __init__(self) -> None:
                self.result = type(
                    "R",
                    (),
                    {
                        "snapshot": snapshot,
                        "hands": type("H", (), {"hands": (), "width": 640, "height": 480})(),
                    },
                )()
                self.fps = 30.0
                self.process_ms = 10.0

        hud.show_packet(Packet())
        hud.minimized_rect_provider = lambda: (300, 200, 1100, 800)
        hud.handle_events(
            (
                GestureEvent(EventKind.CLICK, x=960, y=540),
                GestureEvent(EventKind.ACTION, x=960, y=540, name="swipe_left"),
                GestureEvent(EventKind.ACTION, x=960, y=540, name="fist_hold"),
            )
        )
        now = hud._clock()
        assert hud._particles.alive > 0
        assert hud._streak.advance(now + 0.01)
        assert hud._collapse.advance(now + 0.01)
        hud.grab()
        assert hud.paint_count >= 1 and hud.draw_ms > 0.0
    finally:
        hud.stop()


def test_draw_time_of_a_busy_frame_is_measured(qapp):
    """วาดทุกชิ้นพร้อมกันหลายรอบ แล้วรายงานเวลา (งบของ spec คือ ≤ 4 ms ต่อเฟรมบนเครื่องจริง)"""
    hud = overlay(parse_config({"hud": {**HOLO, "sounds": False}}))
    try:
        for _ in range(5):
            hud.handle_events((GestureEvent(EventKind.CLICK, x=960, y=540),))
            hud.grab()
        assert 0.0 < hud.draw_ms < 1000.0 and hud.draw_ms_max >= hud.draw_ms * 0.5
    finally:
        hud.stop()


def test_debug_view_shows_hud_draw_time(qapp):
    from aether.debug_view import DebugView

    view = DebugView(Config())
    view.set_hud_draw_ms(5.2)
    assert view._hud_draw_ms == 5.2
    view.close()
