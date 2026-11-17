"""Magnetic snap (M6.2) — pure function + สถานะข้ามเฟรม"""

from __future__ import annotations

import math

import pytest

from aether.snap import MIN_RADIUS, MagneticSnap, Target, snap_pull, snap_radius

BUTTON = Target(100, 100, 40, 40, "Button", "Save")  # กลาง (120, 120), r = 24
WIDE = Target(0, 0, 400, 100, "ListItem")  # r = 35


def test_radius():
    assert snap_radius(BUTTON) == MIN_RADIUS
    assert snap_radius(WIDE) == pytest.approx(35.0)
    assert snap_radius(Target(0, 0, 10, 10, "Button")) == MIN_RADIUS


def test_pull_formula():
    assert snap_pull(120, 120, BUTTON, 0.6) == (0.0, 0.0)  # อยู่กลางแล้ว
    assert snap_pull(150, 120, BUTTON, 0.6) == (0.0, 0.0)  # d = 30 ≥ r
    dx, dy = snap_pull(132, 120, BUTTON, 0.6)  # d = 12, r = 24 → s = 0.6 × 0.25 = 0.15
    assert (dx, dy) == pytest.approx((-12 * 0.15, 0.0))
    # ยิ่งใกล้ขอบรัศมี แรงยิ่งน้อย (ต่อเนื่อง ไม่กระโดดตอนเข้า/ออกรัศมี)
    edge = snap_pull(143.9, 120, BUTTON, 0.6)
    assert abs(edge[0]) < 0.01


def test_target_helpers_do_not_expose_the_name():
    assert BUTTON.center == (120.0, 120.0) and BUTTON.contains(139, 139)
    assert not BUTTON.contains(140, 120) and BUTTON.contains(141, 120, margin=2)
    assert BUTTON.describe() == "Button 40x40" and "Save" not in BUTTON.describe()


def run(snap: MagneticSnap, path, target, *, t0=0.0, dt=1 / 30, sticky=False):
    out = []
    t = t0
    for x, y in path:
        out.append(snap.update(x, y, t, target, sticky=sticky))
        t += dt
    return out, t


def test_slow_approach_is_pulled_toward_the_center():
    snap = MagneticSnap(0.6, escape_speed=2000.0)
    path = [(150 - i, 121) for i in range(20)]  # ช้า: 30 px/s
    out, _ = run(snap, path, BUTTON)
    x, y = out[-1]
    raw = path[-1]
    assert math.dist((x, y), BUTTON.center) < math.dist(raw, BUTTON.center)
    assert snap.active is BUTTON


def test_fast_motion_escapes_until_the_cursor_leaves_the_radius():
    snap = MagneticSnap(0.6, escape_speed=300.0)
    run(snap, [(118 + i * 0.2, 120) for i in range(5)], BUTTON)  # นิ่งๆ ในรัศมี: ดูดอยู่
    assert snap.active is BUTTON
    # สะบัดเร็ว (450 px/s) ข้ามปุ่มโดยยังอยู่ในรัศมี → หลุด
    _out, t = run(snap, [(105, 120), (120, 120), (135, 120)], BUTTON, t0=0.2)
    assert snap.active is None
    # กลับมาช้าๆ ในรัศมีเดิม → ยังไม่ดูด จนกว่าจะออกนอกรัศมีก่อน
    _out, t = run(snap, [(128, 120)] * 5, BUTTON, t0=t)
    assert snap.active is None
    _out, t = run(snap, [(150, 120)] * 3, BUTTON, t0=t)  # ออกนอกรัศมี
    _out, t = run(snap, [(150 - 2 * i, 120) for i in range(1, 11)], BUTTON, t0=t)  # กลับมาช้าๆ
    assert snap.active is BUTTON


def test_sticky_keeps_the_target_and_ignores_speed():
    snap = MagneticSnap(0.6, escape_speed=100.0)
    other = Target(200, 100, 40, 40, "Button")
    run(snap, [(125, 120)] * 3, BUTTON, sticky=True)
    # ระหว่าง pinch-lock: finder รายงานปุ่มอื่น + มือสั่นเร็ว → ยังยึดปุ่มเดิม
    out, _ = run(snap, [(126, 120), (124, 121), (127, 119)], other, t0=1.0, sticky=True)
    assert snap.active is BUTTON
    assert all(math.dist(p, BUTTON.center) < 7 for p in out)


def test_release_is_smooth_not_a_jump():
    snap = MagneticSnap(1.0, escape_speed=10_000.0)
    run(snap, [(132, 120)] * 3, BUTTON)
    before = snap.offset
    assert before[0] < -1
    # เป้าหายไป (finder ไม่เจอแล้ว): ระยะดึงค่อยๆ คลาย ไม่หายในเฟรมเดียว
    out, _ = run(snap, [(132, 120)] * 6, None, t0=1.0)
    first_step = abs(out[0][0] - (132 + before[0]))
    assert first_step < abs(before[0])  # ไม่ได้กระโดดกลับทั้งหมดทันที
    assert abs(out[-1][0] - 132) < 0.5  # สุดท้ายคลายเกือบหมด
    assert snap.active is None


def test_reset():
    snap = MagneticSnap(0.6, 1000.0)
    run(snap, [(130, 120)] * 3, BUTTON, sticky=True)
    snap.reset()
    assert snap.offset == (0.0, 0.0) and snap.active is None and snap.speed == 0.0
