"""Calibration: วัดระยะจีบของผู้ใช้ + พื้นที่ที่มือเอื้อมถึง → ``calibration.json``

Pure logic ทั้งหมด (ไม่แตะ Qt/กล้อง) — หน้าต่าง wizard อยู่ใน :mod:`aether.calibration_wizard`

ขั้นตอนของ :class:`CalibrationSession` (ป้อน ``HandFrame`` ทีละเฟรม):

1. ``REST``  — กางนิ้วตามสบายในท่าที่ใช้ชี้ → วัดระยะโป้ง↔ชี้ตอน "ไม่ได้จีบ"
2. ``PINCH`` — จีบนิ้วโป้ง+ชี้ค้างไว้ → วัดระยะตอน "จีบ"
3. ``REACH`` — ขยับมือไปให้ทั่วบริเวณที่เอื้อมถึงสบายๆ → ได้ active region
4. ``DONE``  — :attr:`CalibrationSession.result` พร้อมใช้

ค่าที่ได้ถูกเขียนลง ``calibration.json`` และ :func:`aether.config.load_config` จะนำไปทับ
``pinch.on``, ``pinch.off`` และ ``pointer.active_region`` ของ config.toml
"""

from __future__ import annotations

import datetime
import enum
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from .config import CALIBRATION_VERSION
from .geometry import pinch_ratio, pointer_anchor
from .tracker import Hand, HandFrame

__all__ = [
    "MIN_GAP",
    "CalibrationError",
    "CalibrationResult",
    "CalibrationSession",
    "Step",
    "compute_active_region",
    "compute_pinch_thresholds",
    "percentile",
    "save_calibration",
]

SETTLE_MS = 1500  # เวลาให้ผู้ใช้จัดท่าก่อนเริ่มเก็บข้อมูลของแต่ละขั้น
SAMPLE_MS = {"REST": 2000, "PINCH": 2000, "REACH": 6000}
MIN_SAMPLES = 20

MIN_GAP = 0.12  # ระยะ "ไม่จีบ" กับ "จีบ" ต้องห่างกันอย่างน้อยเท่านี้ จึงตั้ง threshold ได้
MIN_REGION = (0.25, 0.20)  # active region ต้องกว้าง/สูงอย่างน้อยเท่านี้ (สัดส่วนของเฟรม)
REGION_INSET = 0.05  # หดขอบเข้ามา 5% ของช่วง เพื่อให้เอื้อมถึงมุมจอได้สบาย


class CalibrationError(ValueError):
    """ข้อมูลที่วัดได้ใช้ตั้งค่าไม่ได้ (ข้อความเป็นคำแนะนำสำหรับผู้ใช้)"""


class Step(enum.Enum):
    REST = "REST"
    PINCH = "PINCH"
    REACH = "REACH"
    DONE = "DONE"


STEP_TEXT: dict[Step, tuple[str, str]] = {
    Step.REST: (
        "1/3 · ท่าพัก",
        "ยกมือข้างถนัดขึ้นในท่าที่ใช้ชี้ตามสบาย ปล่อยนิ้วโป้งกับนิ้วชี้ห่างกันตามธรรมชาติ แล้วค้างไว้",
    ),
    Step.PINCH: (
        "2/3 · จีบนิ้ว",
        "จีบปลายนิ้วโป้งกับปลายนิ้วชี้ให้แตะกันเบาๆ แล้วค้างไว้",
    ),
    Step.REACH: (
        "3/3 · ระยะเอื้อม",
        "ขยับมือเป็นวงกว้างๆ ให้ทั่วบริเวณที่เอื้อมถึงได้สบาย (ซ้ายสุด ขวาสุด บนสุด ล่างสุด)",
    ),
    Step.DONE: ("เสร็จแล้ว", "บันทึกค่าเรียบร้อย"),
}


def percentile(values: Sequence[float], q: float) -> float:
    """เปอร์เซ็นไทล์ที่ ``q`` (0–100) แบบ interpolate เชิงเส้น"""
    if not values:
        raise ValueError("ไม่มีข้อมูล")
    ordered = sorted(values)
    position = (len(ordered) - 1) * min(max(q, 0.0), 100.0) / 100.0
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def compute_pinch_thresholds(
    rest: Sequence[float], pinched: Sequence[float]
) -> tuple[float, float]:
    """คำนวณ ``(pinch.on, pinch.off)`` จาก pinch_ratio ที่วัดตอนพักและตอนจีบ

    - ``on`` อยู่เหนือค่าตอนจีบเล็กน้อย (จีบติดง่าย) แต่ไม่เกิน 35% ของช่วงระหว่างจีบกับพัก
    - ``off`` = 1.5 × ``on`` (สัดส่วนเดียวกับค่าเริ่มต้น 0.25/0.38) แต่ไม่เกิน 70% ของช่วง
      เพื่อให้ท่าพักอยู่เหนือ ``off`` เสมอ (ปล่อยนิ้วแล้วนับว่าปล่อยแน่นอน)
    """
    if len(rest) < MIN_SAMPLES or len(pinched) < MIN_SAMPLES:
        raise CalibrationError("เก็บข้อมูลได้น้อยเกินไป — ให้มืออยู่ในกล้องตลอดแล้วลองใหม่")
    pinched_high = percentile(pinched, 90)
    rest_low = percentile(rest, 10)
    gap = rest_low - pinched_high
    if gap < MIN_GAP:
        raise CalibrationError(
            "แยกท่าพักกับท่าจีบไม่ออก — ตอนพักให้นิ้วโป้งกับนิ้วชี้ห่างกันมากขึ้น และตอนจีบให้ปลายนิ้วแตะกัน แล้วลองใหม่"
        )
    on = min(max(1.6 * pinched_high, pinched_high + 0.08), pinched_high + 0.35 * gap)
    off = min(1.5 * on, pinched_high + 0.70 * gap)
    on, off = round(on, 3), round(off, 3)
    if on <= 0.0 or off - on < 0.03:
        raise CalibrationError("ค่าที่วัดได้ใกล้กันเกินไป — ลองใหม่โดยกางนิ้วให้กว้างขึ้นตอนพัก")
    return on, off


def compute_active_region(
    points: Sequence[tuple[float, float]],
) -> tuple[float, float, float, float]:
    """active region ``(x0, y0, x1, y1)`` จากตำแหน่งมือที่เก็บตอนขยับให้ทั่วระยะเอื้อม

    ใช้เปอร์เซ็นไทล์ 3–97 (ตัดจุดที่หลุดออกไป) แล้วหดขอบเข้า 5% ของช่วง
    """
    if len(points) < MIN_SAMPLES:
        raise CalibrationError("เก็บข้อมูลได้น้อยเกินไป — ให้มืออยู่ในกล้องตลอดแล้วลองใหม่")
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x0, x1 = percentile(xs, 3), percentile(xs, 97)
    y0, y1 = percentile(ys, 3), percentile(ys, 97)
    inset_x, inset_y = (x1 - x0) * REGION_INSET, (y1 - y0) * REGION_INSET
    x0, x1 = max(0.0, x0 + inset_x), min(1.0, x1 - inset_x)
    y0, y1 = max(0.0, y0 + inset_y), min(1.0, y1 - inset_y)
    if x1 - x0 < MIN_REGION[0] or y1 - y0 < MIN_REGION[1]:
        raise CalibrationError("พื้นที่แคบเกินไป — ขยับมือให้กว้างขึ้นทั้งแนวนอนและแนวตั้ง แล้วลองใหม่")
    return (round(x0, 3), round(y0, 3), round(x1, 3), round(y1, 3))


@dataclass(frozen=True, slots=True)
class CalibrationResult:
    pinch_on: float
    pinch_off: float
    active_region: tuple[float, float, float, float]
    rest_ratio: float  # ค่ากลางของ pinch_ratio ตอนพัก (เก็บไว้ดูประกอบ)
    pinch_ratio: float  # ค่ากลางตอนจีบ

    def to_dict(self, dominant_hand: str) -> dict:
        return {
            "version": CALIBRATION_VERSION,
            "created": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
            "dominant_hand": dominant_hand,
            "pinch": {"on": self.pinch_on, "off": self.pinch_off},
            "pointer": {"active_region": list(self.active_region)},
            "measured": {"rest_ratio": self.rest_ratio, "pinch_ratio": self.pinch_ratio},
        }


def save_calibration(path: Path, result: CalibrationResult, dominant_hand: str) -> None:
    """เขียน ``calibration.json`` (เขียนลงไฟล์ชั่วคราวแล้วย้าย จะได้ไม่เหลือไฟล์ครึ่งๆ กลางๆ)"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    text = json.dumps(result.to_dict(dominant_hand), ensure_ascii=False, indent=2)
    temporary.write_text(text + "\n", encoding="utf-8")
    temporary.replace(path)


class CalibrationSession:
    """State machine ของการ calibrate — เรียก :meth:`update` ทุกเฟรมที่ได้จาก tracker

    เวลาเดินตาม ``timestamp_ms`` ของเฟรม และนับเฉพาะช่วงที่เห็นมือข้างถนัด
    (เอามือออกจากกล้อง = หยุดรอ ไม่เก็บข้อมูลผิดๆ)
    """

    def __init__(self, dominant_hand: str) -> None:
        self._dominant = dominant_hand
        self._rest: list[float] = []
        self._pinched: list[float] = []
        self._points: list[tuple[float, float]] = []
        self._thresholds: tuple[float, float] | None = None
        self.result: CalibrationResult | None = None
        self.error: str | None = None  # เหตุผลที่ต้องทำขั้นล่าสุดใหม่ (แสดงให้ผู้ใช้เห็น)
        self.hand_visible = False
        self.live_ratio: float | None = None
        self._step = Step.REST
        self._elapsed = 0  # ms ที่เห็นมือในขั้นนี้ (รวมช่วงจัดท่า)
        self._last_ms: int | None = None

    @property
    def step(self) -> Step:
        return self._step

    @property
    def done(self) -> bool:
        return self._step is Step.DONE

    @property
    def title(self) -> str:
        return STEP_TEXT[self._step][0]

    @property
    def instruction(self) -> str:
        return STEP_TEXT[self._step][1]

    @property
    def sampling(self) -> bool:
        """กำลังเก็บข้อมูลอยู่ (พ้นช่วงจัดท่าแล้ว)"""
        return not self.done and self._elapsed >= SETTLE_MS

    @property
    def progress(self) -> float:
        """ความคืบหน้าของขั้นปัจจุบัน 0–1 (0 ระหว่างช่วงจัดท่า)"""
        if self.done:
            return 1.0
        duration = SAMPLE_MS[self._step.value]
        return min(max((self._elapsed - SETTLE_MS) / duration, 0.0), 1.0)

    @property
    def countdown_ms(self) -> int:
        """เวลาที่เหลือก่อนเริ่มเก็บข้อมูลของขั้นนี้"""
        return max(0, SETTLE_MS - self._elapsed)

    def restart(self) -> None:
        """เริ่มใหม่ตั้งแต่ขั้นแรก"""
        self.__init__(self._dominant)  # type: ignore[misc]

    def update(self, frame: HandFrame) -> None:
        if self.done:
            return
        hand = frame.hand(self._dominant)
        now = frame.timestamp_ms
        dt = 0 if self._last_ms is None else max(0, min(now - self._last_ms, 200))
        self._last_ms = now
        self.hand_visible = hand is not None
        if hand is None:
            self.live_ratio = None
            return  # ไม่เห็นมือ: เวลาไม่เดิน

        ratio = pinch_ratio(hand, frame.width, frame.height)
        self.live_ratio = ratio
        was_sampling = self.sampling
        self._elapsed += dt
        if self.sampling:
            if not was_sampling:
                self.error = None  # เริ่มเก็บรอบใหม่: ล้างคำเตือนของรอบก่อน
            self._collect(hand, ratio)
            if self.progress >= 1.0:
                self._finish_step()

    # ---- internal ------------------------------------------------------- #

    def _collect(self, hand: Hand, ratio: float) -> None:
        if self._step is Step.REST:
            self._rest.append(ratio)
        elif self._step is Step.PINCH:
            self._pinched.append(ratio)
        elif self._step is Step.REACH:
            self._points.append(pointer_anchor(hand))

    def _retry(self, message: str, samples: list) -> None:
        samples.clear()
        self.error = message
        self._elapsed = 0

    def _finish_step(self) -> None:
        if self._step is Step.REST:
            if len(self._rest) < MIN_SAMPLES:
                self._retry("เก็บข้อมูลได้น้อยเกินไป — ลองอีกครั้ง", self._rest)
                return
            self._advance(Step.PINCH)
        elif self._step is Step.PINCH:
            try:
                self._thresholds = compute_pinch_thresholds(self._rest, self._pinched)
            except CalibrationError as exc:
                self._retry(str(exc), self._pinched)
                return
            self._advance(Step.REACH)
        elif self._step is Step.REACH:
            try:
                region = compute_active_region(self._points)
            except CalibrationError as exc:
                self._retry(str(exc), self._points)
                return
            assert self._thresholds is not None
            self.result = CalibrationResult(
                pinch_on=self._thresholds[0],
                pinch_off=self._thresholds[1],
                active_region=region,
                rest_ratio=round(percentile(self._rest, 50), 3),
                pinch_ratio=round(percentile(self._pinched, 50), 3),
            )
            self._advance(Step.DONE)

    def _advance(self, step: Step) -> None:
        self._step = step
        self._elapsed = 0
        self.error = None
