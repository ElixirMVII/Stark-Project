"""One Euro Filter — low-pass ที่ปรับ cutoff ตามความเร็ว

Casiez, Roussel, Vogel (CHI 2012) "1€ Filter: A Simple Speed-based Low-pass Filter for
Noisy Input in Interactive Systems"

- ขยับช้า → cutoff ต่ำ (= ``min_cutoff``) กรองแรง เคอร์เซอร์ไม่สั่น
- ขยับเร็ว → cutoff สูงขึ้นตาม ``beta × ความเร็ว`` กรองน้อย เคอร์เซอร์ไม่หน่วง
"""

from __future__ import annotations

import math

__all__ = ["OneEuroFilter", "OneEuroFilter2D", "smoothing_factor"]


def smoothing_factor(cutoff_hz: float, dt: float) -> float:
    """alpha ของ exponential smoothing สำหรับ cutoff และช่วงเวลา ``dt`` (วินาที)"""
    tau = 1.0 / (2.0 * math.pi * cutoff_hz)
    return 1.0 / (1.0 + tau / dt)


class OneEuroFilter:
    """กรองค่าสเกลาร์ เรียก ``filter(value, t)`` โดย ``t`` เป็นวินาทีที่เพิ่มขึ้นเรื่อยๆ"""

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.0, d_cutoff: float = 1.0) -> None:
        if min_cutoff <= 0 or d_cutoff <= 0:
            raise ValueError("min_cutoff และ d_cutoff ต้อง > 0")
        if beta < 0:
            raise ValueError("beta ต้อง ≥ 0")
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._x: float | None = None  # ค่าที่กรองแล้วล่าสุด
        self._dx = 0.0  # อนุพันธ์ที่กรองแล้วล่าสุด
        self._t = 0.0

    def reset(self) -> None:
        self._x = None
        self._dx = 0.0

    def __call__(self, value: float, t: float) -> float:
        if self._x is None:
            self._x, self._t = value, t
            return value
        dt = t - self._t
        if dt <= 0.0:  # timestamp ไม่เพิ่ม: ไม่มีข้อมูลความเร็ว คืนค่าเดิม
            return self._x
        self._t = t

        dx = (value - self._x) / dt
        self._dx += smoothing_factor(self.d_cutoff, dt) * (dx - self._dx)

        cutoff = self.min_cutoff + self.beta * abs(self._dx)
        self._x += smoothing_factor(cutoff, dt) * (value - self._x)
        return self._x


class OneEuroFilter2D:
    """One Euro Filter สองแกน (x, y) ใช้พารามิเตอร์ชุดเดียวกัน"""

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.0, d_cutoff: float = 1.0) -> None:
        self._fx = OneEuroFilter(min_cutoff, beta, d_cutoff)
        self._fy = OneEuroFilter(min_cutoff, beta, d_cutoff)

    def reset(self) -> None:
        self._fx.reset()
        self._fy.reset()

    def set_params(self, min_cutoff: float, beta: float, d_cutoff: float) -> None:
        """เปลี่ยนค่าระหว่างทำงาน (live tuning) โดยไม่ล้างค่าที่กรองไว้ — เคอร์เซอร์ไม่กระโดด"""
        for f in (self._fx, self._fy):
            f.min_cutoff, f.beta, f.d_cutoff = min_cutoff, beta, d_cutoff

    def __call__(self, x: float, y: float, t: float) -> tuple[float, float]:
        return (self._fx(x, t), self._fy(y, t))
