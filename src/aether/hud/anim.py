"""แกนกลางของแอนิเมชัน HUD (spec v1.1 M7.1) — spring อันดับสองแบบ ``SecondOrderDynamics(f, zeta, r)``

- ``f`` ความเร็วตอบสนอง (Hz), ``zeta`` damping (< 1 = เด้ง, 1 = ไม่เด้ง), ``r`` การพุ่งตอนเริ่ม
  (0 = ค่อยๆ เริ่ม, 1 = ตามทันที, < 0 = ถอยก่อนพุ่ง)
- ใช้ได้ทั้งตัวเลขเดียวและ tuple (ตำแหน่ง ขนาด มุม ความโปร่ง สี่เหลี่ยม)
- ``dt`` ถูกจำกัดไม่เกิน 1/30 วินาที และค่า k2 ถูกปรับให้เสถียรเสมอ (เฟรมสะดุดแล้วค่าไม่ระเบิด)

สูตรตาม "Giving Personality to Procedural Animations" (semi-implicit Euler + k2 ที่เสถียร)
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

__all__ = ["DURATIONS", "MAX_DT", "PRESETS", "Preset", "SecondOrderDynamics", "spring"]

MAX_DT = 1.0 / 30.0

Value = float | tuple[float, ...]


@dataclass(frozen=True, slots=True)
class Preset:
    f: float
    zeta: float
    r: float = 0.0


PRESETS = {
    "snappy": Preset(4.0, 0.7),  # reticle ตามเคอร์เซอร์
    "soft": Preset(2.0, 1.0),  # แผงและป้าย
    "bouncy": Preset(3.0, 0.45),  # กรอบล็อกเป้า, ตอนคลิก
}
DURATIONS = {"fast": 0.12, "base": 0.22, "slow": 0.40}  # วินาที


def _as_tuple(value: Value) -> tuple[float, ...]:
    return (float(value),) if isinstance(value, (int, float)) else tuple(float(v) for v in value)


class SecondOrderDynamics:
    def __init__(self, f: float, zeta: float, r: float = 0.0, x0: Value = 0.0) -> None:
        if f <= 0:
            raise ValueError("f ต้อง > 0")
        self._scalar = isinstance(x0, (int, float))
        self.k1 = zeta / (math.pi * f)
        self.k2 = 1.0 / ((2.0 * math.pi * f) ** 2)
        self.k3 = r * zeta / (2.0 * math.pi * f)
        start = _as_tuple(x0)
        self._xp = start
        self._y = start
        self._yd = tuple(0.0 for _ in start)

    @property
    def value(self) -> Value:
        return self._y[0] if self._scalar else self._y

    @property
    def velocity(self) -> Value:
        return self._yd[0] if self._scalar else self._yd

    def reset(self, x: Value) -> None:
        """กระโดดไปที่ ``x`` ทันทีและหยุดนิ่ง"""
        start = _as_tuple(x)
        self._xp = self._y = start
        self._yd = tuple(0.0 for _ in start)

    def settled(self, target: Value, tolerance: float = 0.5) -> bool:
        x = _as_tuple(target)
        return all(abs(a - b) <= tolerance for a, b in zip(self._y, x)) and all(
            abs(v) <= tolerance * 4 for v in self._yd
        )

    def update(self, dt: float, x: Value, xd: Sequence[float] | float | None = None) -> Value:
        """เดินเวลา ``dt`` วินาทีไปหาเป้า ``x`` คืนค่าใหม่"""
        target = _as_tuple(x)
        if len(target) != len(self._y):
            self.reset(target)
            return self.value
        dt = min(max(dt, 0.0), MAX_DT)
        if dt == 0.0:
            return self.value
        if xd is None:
            velocity = tuple((a - b) / dt for a, b in zip(target, self._xp))
        else:
            velocity = _as_tuple(xd)
        self._xp = target
        k2 = max(self.k2, dt * dt / 2 + dt * self.k1 / 2, dt * self.k1)  # เสถียรเสมอ
        y = tuple(y + dt * yd for y, yd in zip(self._y, self._yd))
        self._yd = tuple(
            yd + dt * (xt + self.k3 * xv - yv - self.k1 * yd) / k2
            for yd, xt, xv, yv in zip(self._yd, target, velocity, y)
        )
        self._y = y
        return self.value


def spring(name: str, x0: Value = 0.0) -> SecondOrderDynamics:
    """spring ตาม preset ใน :data:`PRESETS`"""
    preset = PRESETS[name]
    return SecondOrderDynamics(preset.f, preset.zeta, preset.r, x0)
