"""SecondOrderDynamics (M7.1) และกรอบล็อกเป้าบน HUD (M6.3)"""

from __future__ import annotations

import math

import pytest
from PySide6.QtCore import QPointF, QRectF

from aether.config import Config
from aether.gestures import EngineSnapshot, State
from aether.hud.anim import DURATIONS, MAX_DT, PRESETS, SecondOrderDynamics, spring
from aether.hud.target_lock import TargetLock
from aether.hud.theme import Theme
from aether.snap import Target


def settle_time(dyn: SecondOrderDynamics, target: float, tolerance: float = 0.02) -> float:
    t = 0.0
    while t < 5.0:
        dyn.update(1 / 60, target)
        t += 1 / 60
        if dyn.settled(target, tolerance):
            return t
    return math.inf


def test_converges_to_the_target():
    for name in PRESETS:
        dyn = spring(name, 0.0)
        assert settle_time(dyn, 1.0) < 3.0, name
        assert dyn.value == pytest.approx(1.0, abs=0.02)


def test_bouncy_overshoots_and_soft_does_not():
    bouncy, soft = spring("bouncy", 0.0), spring("soft", 0.0)
    peak_b = peak_s = 0.0
    for _ in range(120):
        peak_b = max(peak_b, bouncy.update(1 / 60, 1.0))
        peak_s = max(peak_s, soft.update(1 / 60, 1.0))
    assert peak_b > 1.05  # เด้งเลยเป้า
    assert peak_s <= 1.001  # critically damped: ไม่เลย


def test_snappy_is_faster_than_soft():
    assert settle_time(spring("snappy"), 1.0) < settle_time(spring("soft"), 1.0)


def test_stable_with_huge_or_erratic_dt():
    dyn = SecondOrderDynamics(8.0, 0.2, 2.0, 0.0)  # แข็ง + damping ต่ำ: ไม่เสถียรถ้าไม่ clamp
    for dt in (1.0, 5.0, 0.0, 0.5, 2.0) * 20:
        value = dyn.update(dt, 100.0)
        assert math.isfinite(value) and abs(value) < 1000
    assert settle_time(dyn, 100.0, tolerance=0.5) < math.inf


def test_dt_is_clamped_to_one_thirtieth():
    a, b = spring("soft", 0.0), spring("soft", 0.0)
    a.update(10.0, 1.0)  # เฟรมสะดุดยาว 10 วินาที
    b.update(MAX_DT, 1.0)
    assert a.value == pytest.approx(b.value)


def test_vector_values_and_reset():
    dyn = spring("snappy", (0.0, 0.0, 10.0))
    for _ in range(200):
        dyn.update(1 / 60, (5.0, -5.0, 10.0))
    assert dyn.value == pytest.approx((5.0, -5.0, 10.0), abs=0.05)
    dyn.reset((1.0, 2.0, 3.0))
    assert dyn.value == (1.0, 2.0, 3.0) and dyn.velocity == (0.0, 0.0, 0.0)
    dyn.update(1 / 60, (1.0, 2.0))  # มิติเปลี่ยน: เริ่มใหม่ที่ค่านั้น
    assert dyn.value == (1.0, 2.0)


def test_tokens():
    assert DURATIONS == {"fast": 0.12, "base": 0.22, "slow": 0.40}
    assert (PRESETS["snappy"].f, PRESETS["snappy"].zeta) == (4.0, 0.7)
    with pytest.raises(ValueError):
        SecondOrderDynamics(0.0, 1.0)


# ---- กรอบล็อกเป้า -------------------------------------------------------- #


def snapshot(target: Target | None, pinch: float = 0.0) -> EngineSnapshot:
    return EngineSnapshot(state=State.POINTER, armed=True, target=target, pinch_level=pinch)


@pytest.fixture
def lock(qapp):
    return TargetLock(Theme.from_config(Config().hud))


def run(lock: TargetLock, seconds: float, t0: float = 0.0) -> float:
    t = t0
    while t < t0 + seconds:
        lock.advance(t)
        t += 1 / 60
    return t


def test_lock_springs_from_the_reticle_to_the_target(lock):
    target = Target(100, 100, 80, 30, "Button", "Save")
    rect = QRectF(100, 100, 80, 30)
    lock.set_state(rect, QPointF(300, 300), snapshot(target))
    lock.advance(0.0)
    lock.advance(1 / 60)
    start = lock._current()
    assert start.center().x() > 200  # เริ่มใกล้ reticle
    run(lock, 1.5, 0.02)
    frame = lock._current()
    assert frame.center().x() == pytest.approx(140, abs=1.5)
    assert frame.contains(rect) and lock.alpha == 1.0
    assert lock.label == "Button · Save" and not lock.rect().isEmpty()


def test_lock_squeezes_when_the_pinch_is_almost_done(lock):
    target = Target(100, 100, 80, 30, "Button")
    lock.set_state(QRectF(100, 100, 80, 30), QPointF(140, 115), snapshot(target))
    run(lock, 1.5)
    loose = lock._current().width()
    lock.set_state(QRectF(100, 100, 80, 30), QPointF(140, 115), snapshot(target, pinch=1.0))
    assert lock._current().width() < loose


def test_lock_fades_when_the_target_goes_away(lock):
    target = Target(100, 100, 80, 30, "MenuItem", "x" * 60)
    lock.set_state(QRectF(100, 100, 80, 30), QPointF(140, 115), snapshot(target))
    t = run(lock, 0.5)
    assert lock.label.startswith("Menu item · ")
    assert len(lock.label) <= len("Menu item · ") + 26
    lock.set_state(None, QPointF(140, 115), snapshot(None))
    run(lock, 0.5, t)
    assert lock.alpha == 0.0 and lock.rect().isEmpty()


def test_lock_paints(lock, qapp):
    from PySide6.QtGui import QImage, QPainter

    target = Target(10, 10, 80, 30, "Button", "OK")
    lock.set_state(QRectF(10, 10, 80, 30), QPointF(50, 25), snapshot(target))
    run(lock, 1.0)
    lock.flash(1.0)
    image = QImage(300, 120, QImage.Format.Format_ARGB32)
    image.fill(0)
    painter = QPainter(image)
    lock.paint(painter, 1.05)
    painter.end()
    lit = sum(
        1 for x in range(0, 300, 3) for y in range(0, 120, 3) if image.pixelColor(x, y).alpha() > 0
    )
    assert lit > 10
