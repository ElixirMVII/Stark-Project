"""Magnetic snap (spec v1.1 M6.2) — ดูดเคอร์เซอร์เข้าหากลางปุ่มที่อยู่ใต้มือ (pure logic)

- :class:`Target` สี่เหลี่ยมของปุ่ม (physical px) + ชนิด + ชื่อ (ชื่ออาจเป็นข้อมูลส่วนตัว — **ห้ามเขียนลง log/
  ไฟล์ใดๆ** ใช้แสดงบน HUD เท่านั้น)
- รัศมี ``r = max(24px, 0.35 × min(w, h))`` จากกลางปุ่ม ถ้าเคอร์เซอร์อยู่ห่าง ``d < r`` ดึงเข้าหากลาง
  ด้วยแรง ``s = snap_strength × (1 − d/r)²``
- ขยับมือเร็วกว่า ``escape_speed`` → ปิด snap ของปุ่มนั้นจนกว่าเคอร์เซอร์จะออกจากรัศมี
- ระหว่าง pinch-lock (นิ้วใกล้จีบ) ยึดปุ่มไว้ (sticky) จนคลิกเสร็จ — ไม่เปลี่ยนเป้า ไม่หลุดเพราะความเร็ว
- ตอนปล่อยการดูด (หลุด/เปลี่ยนเป้า) ระยะดึงค่อยๆ คลายใน ~``RELEASE_TAU`` วินาที เคอร์เซอร์ไม่กระโดด
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = ["MIN_RADIUS", "MagneticSnap", "Target", "snap_pull", "snap_radius"]

MIN_RADIUS = 24.0  # px
RELEASE_TAU = 0.06  # วินาที: ระยะดึงที่เหลือลดลงแบบ exponential ตอนปล่อย
SPEED_SMOOTHING = 0.5  # EMA ของความเร็วมือ (กัน noise เฟรมเดียวทำให้หลุด)


@dataclass(frozen=True, slots=True)
class Target:
    """ปุ่ม/รายการที่กดได้ใต้เคอร์เซอร์ (physical px)"""

    left: int
    top: int
    width: int
    height: int
    control_type: str  # เช่น "Button", "MenuItem"
    name: str = ""  # แสดงบน HUD เท่านั้น — ห้าม log
    timestamp: float = 0.0

    @property
    def center(self) -> tuple[float, float]:
        return (self.left + self.width / 2.0, self.top + self.height / 2.0)

    @property
    def rect(self) -> tuple[int, int, int, int]:
        return (self.left, self.top, self.width, self.height)

    def contains(self, x: float, y: float, margin: float = 0.0) -> bool:
        return (
            self.left - margin <= x < self.left + self.width + margin
            and self.top - margin <= y < self.top + self.height + margin
        )

    def describe(self) -> str:
        """สำหรับ log/สถิติ: ชนิดกับขนาดเท่านั้น (ไม่มีชื่อ)"""
        return f"{self.control_type} {self.width}x{self.height}"


def snap_radius(target: Target) -> float:
    return max(MIN_RADIUS, 0.35 * min(target.width, target.height))


def snap_pull(x: float, y: float, target: Target, strength: float) -> tuple[float, float]:
    """ระยะที่ต้องดึงเคอร์เซอร์ (dx, dy) — 0 ถ้าอยู่นอกรัศมี"""
    cx, cy = target.center
    dx, dy = cx - x, cy - y
    d = math.hypot(dx, dy)
    r = snap_radius(target)
    if d >= r:
        return (0.0, 0.0)
    s = strength * (1.0 - d / r) ** 2
    return (dx * s, dy * s)


class MagneticSnap:
    """สถานะของ snap ข้ามเฟรม: ความเร็วมือ, เป้าที่หลุดเพราะความเร็ว, เป้าที่ยึดตอน pinch-lock

    ``escape_speed`` เป็น px/s (แอปแปลงจาก "ความกว้างจอต่อวินาที" ใน config ให้)
    """

    def __init__(self, strength: float, escape_speed: float) -> None:
        self.strength = strength
        self.escape_speed = escape_speed
        self._offset = (0.0, 0.0)
        self._last: tuple[float, float, float] | None = None  # (x, y, t) ตำแหน่งก่อน snap
        self._speed = 0.0
        self._escaped: tuple[int, int, int, int] | None = None
        self._locked: Target | None = None
        self._active: Target | None = None

    @property
    def offset(self) -> tuple[float, float]:
        return self._offset

    @property
    def active(self) -> Target | None:
        """เป้าที่กำลังดูดอยู่ (None = ไม่ได้ดูด)"""
        return self._active

    @property
    def speed(self) -> float:
        return self._speed

    def reset(self) -> None:
        self._offset = (0.0, 0.0)
        self._last = None
        self._speed = 0.0
        self._escaped = None
        self._locked = None
        self._active = None

    def update(
        self, x: float, y: float, now: float, target: Target | None, *, sticky: bool = False
    ) -> tuple[float, float]:
        """ตำแหน่งเคอร์เซอร์หลัง snap จากตำแหน่งก่อน snap (``x``, ``y``) ณ เวลา ``now`` (วินาที)"""
        dt = 0.0
        if self._last is not None:
            lx, ly, lt = self._last
            dt = max(now - lt, 0.0)
            if dt > 0:
                speed = math.hypot(x - lx, y - ly) / dt
                self._speed += SPEED_SMOOTHING * (speed - self._speed)
        self._last = (x, y, now)

        if sticky:
            if self._locked is None and target is not None:
                self._locked = target
            target = self._locked or target
        else:
            self._locked = None

        pull = (0.0, 0.0)
        self._active = None
        if target is not None:
            inside = math.dist((x, y), target.center) < snap_radius(target)
            if self._escaped is not None and (self._escaped != target.rect or not inside):
                self._escaped = None  # ออกจากรัศมีแล้ว (หรือเป็นปุ่มอื่น): ดูดได้อีก
            if not sticky and self._speed > self.escape_speed and inside:
                self._escaped = target.rect
            if sticky or (inside and self._escaped is None):
                pull = snap_pull(x, y, target, self.strength)
                self._active = target
        else:
            self._escaped = None

        ox, oy = self._offset
        if math.hypot(*pull) >= math.hypot(ox, oy) or dt <= 0:
            self._offset = pull
        else:  # คลายแรงดูดที่เหลือแบบนุ่มๆ
            keep = math.exp(-dt / RELEASE_TAU)
            self._offset = (pull[0] + (ox - pull[0]) * keep, pull[1] + (oy - pull[1]) * keep)
        return (x + self._offset[0], y + self._offset[1])
