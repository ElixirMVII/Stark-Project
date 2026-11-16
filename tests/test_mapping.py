from __future__ import annotations

import itertools

import pytest

from aether.mapping import (
    PointerTracker,
    ScreenGeometry,
    logical_to_physical,
    map_to_screen,
    normalize_in_region,
    physical_to_logical,
    primary_screen_geometry,
)

REGION = (0.15, 0.10, 0.85, 0.70)
FHD = ScreenGeometry(0, 0, 1920, 1080)


# ---- ScreenGeometry ----------------------------------------------------- #


def test_screen_edges_and_clamp():
    assert (FHD.right, FHD.bottom) == (1919, 1079)
    assert FHD.clamp(-5, 2000) == (0.0, 1079.0)
    assert FHD.clamp(100.5, 200.25) == (100.5, 200.25)
    offset = ScreenGeometry(1920, -200, 1280, 1024)  # จอที่สองทางขวา (เผื่อหลายจอ)
    assert (offset.right, offset.bottom) == (3199, 823)
    assert offset.clamp(0, 0) == (1920.0, 0.0)


@pytest.mark.parametrize("size", [(0, 100), (100, 0), (-1, 5)])
def test_screen_requires_positive_size(size):
    with pytest.raises(ValueError):
        ScreenGeometry(0, 0, *size)


# ---- camera → screen ---------------------------------------------------- #


@pytest.mark.parametrize(
    ("point", "expected"),
    [
        ((0.15, 0.10), (0.0, 0.0)),  # มุมซ้ายบนของ active region
        ((0.85, 0.70), (1.0, 1.0)),  # มุมขวาล่าง
        ((0.50, 0.40), (0.5, 0.5)),  # กลาง
        ((0.00, 0.00), (0.0, 0.0)),  # นอก region → ตรึงที่ขอบ
        ((1.00, 1.00), (1.0, 1.0)),
        ((0.50, 0.90), (0.5, 1.0)),
    ],
)
def test_normalize_in_region(point, expected):
    assert normalize_in_region(*point, REGION) == pytest.approx(expected)


def test_active_region_maps_to_full_screen():
    assert map_to_screen(0.15, 0.10, REGION, FHD) == pytest.approx((0, 0))
    assert map_to_screen(0.85, 0.70, REGION, FHD) == pytest.approx((1919, 1079))
    assert map_to_screen(0.50, 0.40, REGION, FHD) == pytest.approx((959.5, 539.5))


def test_points_outside_region_stay_on_screen():
    for point in [(-1.0, -1.0), (2.0, 2.0), (0.0, 0.99), (0.99, 0.0)]:
        x, y = map_to_screen(*point, REGION, FHD)
        assert 0 <= x <= 1919 and 0 <= y <= 1079


def test_mapping_respects_screen_origin():
    second = ScreenGeometry(1920, 100, 1280, 720)
    assert map_to_screen(0.15, 0.10, REGION, second) == pytest.approx((1920, 100))
    assert map_to_screen(0.85, 0.70, REGION, second) == pytest.approx((3199, 819))


# ---- DPI ---------------------------------------------------------------- #


@pytest.mark.parametrize("dpr", [1.0, 1.25, 1.5, 2.0])
def test_physical_logical_roundtrip(dpr):
    lx, ly = physical_to_logical(1920, 1080, dpr)
    assert (lx, ly) == pytest.approx((1920 / dpr, 1080 / dpr))
    assert logical_to_physical(lx, ly, dpr) == pytest.approx((1920, 1080))


def test_scale_125_and_150_examples():
    # จอ 1920x1080 ที่ 125% → Qt เห็น 1536x864, ที่ 150% → 1280x720
    assert physical_to_logical(1920, 1080, 1.25) == pytest.approx((1536, 864))
    assert physical_to_logical(1920, 1080, 1.5) == pytest.approx((1280, 720))
    assert physical_to_logical(960, 540, 1.5) == pytest.approx((640, 360))


def test_primary_screen_geometry_from_qt(qapp):
    """บนเครื่อง test (ไม่ใช่ Windows) ใช้ค่าจาก Qt: logical size × devicePixelRatio"""
    geometry = primary_screen_geometry()
    screen = qapp.primaryScreen()
    dpr = screen.devicePixelRatio()
    assert geometry.width == round(screen.size().width() * dpr)
    assert geometry.height == round(screen.size().height() * dpr)
    assert (geometry.left, geometry.top) == (0, 0)


# ---- PointerTracker ----------------------------------------------------- #


def test_first_update_places_cursor_on_target():
    tracker = PointerTracker(FHD)
    assert tracker.position is None
    assert tracker.update((500.0, 300.0)) == (500.0, 300.0)


def test_full_gain_follows_target_exactly():
    tracker = PointerTracker(FHD)
    tracker.update((100.0, 100.0))
    assert tracker.update((180.0, 40.0)) == pytest.approx((180.0, 40.0))
    assert tracker.update((181.0, 41.0)) == pytest.approx((181.0, 41.0))


def test_reduced_gain_moves_fraction_of_delta():
    tracker = PointerTracker(FHD)
    tracker.update((100.0, 100.0))
    assert tracker.update((150.0, 100.0), gain=0.2) == pytest.approx((110.0, 100.0))
    assert tracker.update((150.0, 150.0), gain=0.2) == pytest.approx((110.0, 110.0))


def test_zero_gain_freezes_cursor():
    tracker = PointerTracker(FHD)
    tracker.update((100.0, 100.0))
    assert tracker.update((400.0, 700.0), gain=0.0) == (100.0, 100.0)


def test_no_jump_when_gain_returns_and_hand_is_still():
    """หลัง pinch-lock เคอร์เซอร์ห่างจากเป้า — มือนิ่งต้องไม่กระโดดกลับไปหาเป้า"""
    tracker = PointerTracker(FHD)
    tracker.update((100.0, 100.0))
    tracker.update((200.0, 100.0), gain=0.2)  # cursor = 120, offset 80
    assert tracker.update((200.0, 100.0), gain=1.0) == pytest.approx((120.0, 100.0))
    assert tracker.update((200.0, 100.0), gain=1.0) == pytest.approx((120.0, 100.0))


def test_offset_bleeds_away_as_hand_moves():
    tracker = PointerTracker(FHD)
    tracker.update((100.0, 100.0))
    tracker.update((200.0, 100.0), gain=0.2)  # offset 80 px
    target_x = 200.0
    offsets = []
    for _ in range(60):
        target_x += 10.0
        x, _ = tracker.update((target_x, 100.0))
        offsets.append(target_x - x)
    assert offsets[0] < 80.0
    assert all(b <= a + 1e-9 for a, b in itertools.pairwise(offsets))  # ลดลงเรื่อยๆ ไม่แกว่ง
    assert offsets[-1] == pytest.approx(0.0, abs=1e-6)  # สุดท้ายกลับมาตรงเป้า


def test_offset_bleeds_without_overshoot_when_moving_away():
    tracker = PointerTracker(FHD)
    tracker.update((500.0, 500.0))
    tracker.update((600.0, 500.0), gain=0.0)  # cursor ค้างที่ 500, เป้า 600
    x, _ = tracker.update((590.0, 500.0))  # มือขยับกลับ 10 px
    assert 490.0 <= x <= 500.0  # ตามทิศที่มือขยับ ไม่วิ่งสวนไปหาเป้าเกินระยะที่ขยับ


def test_cursor_is_clamped_to_screen():
    tracker = PointerTracker(FHD)
    tracker.update((10.0, 10.0))
    assert tracker.update((-500.0, 5000.0)) == (0.0, 1079.0)


def test_rebase_keeps_cursor_and_restarts_delta():
    tracker = PointerTracker(FHD)
    tracker.update((100.0, 100.0))
    tracker.rebase((900.0, 900.0))  # มือย้ายไปไกลระหว่างที่เคอร์เซอร์ถูกตรึง (เช่นตอนสกรอล)
    assert tracker.position == (100.0, 100.0)
    x, y = tracker.update((910.0, 900.0))
    assert 100.0 < x <= 125.0 and y > 100.0  # ขยับต่อจากจุดเดิม ไม่กระโดดไป 910


def test_rebase_before_any_update_sets_position():
    tracker = PointerTracker(FHD)
    tracker.rebase((300.0, 200.0))
    assert tracker.position == (300.0, 200.0)


def test_jump_to_and_reset():
    tracker = PointerTracker(FHD)
    tracker.update((100.0, 100.0))
    tracker.jump_to(50.0, 60.0)
    assert tracker.position == (50.0, 60.0)
    tracker.jump_to(-10.0, 99999.0)
    assert tracker.position == (0.0, 1079.0)
    tracker.reset()
    assert tracker.position is None
    assert tracker.update((7.0, 8.0)) == (7.0, 8.0)
