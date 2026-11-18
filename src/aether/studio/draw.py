"""วาดรูปในอากาศ — โมเดลภาพวาด (pure) + ตัวควบคุมที่แปลงมือเป็นการวาด

- จีบ = วาด (เริ่มจีบบนจานสีมุมซ้ายบน = เลือกสี) · ปล่อย = จบเส้น
- ✊ ค้าง 0.8 วินาที = ย้อนกลับ (undo) · ✌ ค้าง = เปลี่ยนสี · 👎 ค้าง 1.2 วินาที = ล้างทั้งหมด
- 👍 ค้าง 1 วินาที = บันทึกเป็น PNG (หน้าต่างเป็นคนบันทึกจริง)

พิกัดทั้งหมดเป็น 0–1 ของผืนผ้าใบ จึงวาดซ้ำได้ทุกขนาดหน้าต่าง และบันทึกเป็นภาพความละเอียดเท่าไรก็ได้
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .input import HoldDetector, StudioHand

__all__ = [
    "BRUSH_SIZES",
    "PALETTE",
    "SWATCH_RADIUS",
    "DrawController",
    "DrawEvent",
    "Drawing",
    "Stroke",
    "swatch_centers",
]

PALETTE = ("#F2F4F7", "#D6B67A", "#E5484D", "#5B9BD5", "#7CC4A0", "#B392F0", "#1B1E23")
BRUSH_SIZES = (2.0, 4.0, 8.0, 16.0)  # px ที่ความกว้างผืนผ้าใบ 1000 px (ย่อ/ขยายตามขนาดจริง)
MIN_STEP = 0.0025  # จุดที่ใกล้กว่านี้ (หน่วยผืนผ้าใบ) ไม่เก็บ — กันเส้นหนาเป็นก้อนตอนมือนิ่ง
MAX_UNDO = 200

SWATCH_RADIUS = 0.016  # หน่วย = ความกว้างผืนผ้าใบ
SWATCH_GAP = 0.045
SWATCH_TOP = 0.035  # ระยะจากขอบบน (หน่วย = ความกว้างผืนผ้าใบ)


def swatch_centers(aspect: float = 16 / 9) -> list[tuple[float, float]]:
    """ตำแหน่งจานสี (มุมซ้ายบน) — ``aspect`` = กว้าง ÷ สูง ของผืนผ้าใบ (วงกลมกลมจริงทุกขนาด)"""
    return [(0.035 + i * SWATCH_GAP, SWATCH_TOP * aspect) for i in range(len(PALETTE))]


@dataclass(slots=True)
class Stroke:
    color: str
    width: float
    points: list[tuple[float, float]] = field(default_factory=list)

    def length(self) -> float:
        return sum(math.dist(a, b) for a, b in zip(self.points, self.points[1:], strict=False))


class Drawing:
    def __init__(self) -> None:
        self.strokes: list[Stroke] = []
        self.color_index = 1  # ทองแชมเปญ
        self.size_index = 1
        self._current: Stroke | None = None

    @property
    def color(self) -> str:
        return PALETTE[self.color_index]

    @property
    def width(self) -> float:
        return BRUSH_SIZES[self.size_index]

    @property
    def drawing(self) -> bool:
        return self._current is not None

    def begin(self, x: float, y: float) -> None:
        self._current = Stroke(self.color, self.width, [(x, y)])
        self.strokes.append(self._current)
        if len(self.strokes) > MAX_UNDO:
            del self.strokes[0]

    def extend(self, x: float, y: float) -> bool:
        """เพิ่มจุดให้เส้นปัจจุบัน (คืน ``False`` ถ้าใกล้จุดก่อนหน้าเกินไป)"""
        if self._current is None:
            self.begin(x, y)
            return True
        if math.dist(self._current.points[-1], (x, y)) < MIN_STEP:
            return False
        self._current.points.append((x, y))
        return True

    def end(self) -> None:
        self._current = None

    def undo(self) -> bool:
        self._current = None
        if not self.strokes:
            return False
        self.strokes.pop()
        return True

    def clear(self) -> None:
        self._current = None
        self.strokes.clear()

    def next_color(self) -> str:
        self.color_index = (self.color_index + 1) % len(PALETTE)
        return self.color

    def select_color(self, index: int) -> None:
        self.color_index = index % len(PALETTE)

    def cycle_size(self) -> float:
        self.size_index = (self.size_index + 1) % len(BRUSH_SIZES)
        return self.width


@dataclass(frozen=True, slots=True)
class DrawEvent:
    kind: str  # "undo" | "clear" | "color" | "save"
    text: str


class DrawController:
    """มือหลักหนึ่งข้าง → การวาด: เรียก :meth:`update` ทุกเฟรม คืน event ที่ควรแจ้งผู้ใช้"""

    def __init__(self, drawing: Drawing, *, aspect: float = 16 / 9) -> None:
        self.drawing = drawing
        self.aspect = aspect
        self.undo_hold = HoldDetector(0.8)
        self.color_hold = HoldDetector(0.8)
        self.clear_hold = HoldDetector(1.2)
        self.save_hold = HoldDetector(1.0)
        self._was_pinched = False
        self._picking = False  # จีบเริ่มบนจานสี: ไม่วาดจนกว่าจะปล่อย

    def swatch_at(self, x: float, y: float) -> int | None:
        for index, (sx, sy) in enumerate(swatch_centers(self.aspect)):
            dy = (y - sy) / self.aspect  # y ของผืนผ้าใบถูกบีบตามสัดส่วน: แปลงเป็นหน่วยเดียวกับ x
            if math.hypot(x - sx, dy) <= SWATCH_RADIUS * 1.6:
                return index
        return None

    def update(self, hand: StudioHand | None, now: float) -> list[DrawEvent]:
        events: list[DrawEvent] = []
        if hand is None:
            self.drawing.end()
            self._was_pinched = self._picking = False
            for hold in (self.undo_hold, self.color_hold, self.clear_hold, self.save_hold):
                hold.reset()
            return events

        if hand.pinch and not self._was_pinched:
            swatch = self.swatch_at(hand.x, hand.y)
            if swatch is not None:
                self.drawing.select_color(swatch)
                self._picking = True
                events.append(DrawEvent("color", "Color"))
            else:
                self.drawing.begin(hand.x, hand.y)
        elif hand.pinch and not self._picking:
            self.drawing.extend(hand.x, hand.y)
        elif not hand.pinch and self._was_pinched:
            self.drawing.end()
            self._picking = False
        self._was_pinched = hand.pinch

        idle = not hand.pinch
        undo = self.undo_hold.update(idle and hand.gesture == "Closed_Fist", now)
        if undo and self.drawing.undo():
            events.append(DrawEvent("undo", "Undo"))
        if self.color_hold.update(idle and hand.gesture == "Victory", now):
            self.drawing.next_color()
            events.append(DrawEvent("color", "Color"))
        if self.clear_hold.update(idle and hand.gesture == "Thumb_Down", now):
            self.drawing.clear()
            events.append(DrawEvent("clear", "Cleared"))
        if self.save_hold.update(idle and hand.gesture == "Thumb_Up", now):
            events.append(DrawEvent("save", "Saved"))
        return events

    @property
    def hold_progress(self) -> float:
        """ความคืบหน้าของท่าที่กำลังค้าง (วาดเป็นวงแหวนรอบเคอร์เซอร์)"""
        return max(
            self.undo_hold.progress,
            self.color_hold.progress,
            self.clear_hold.progress,
            self.save_hold.progress,
        )
