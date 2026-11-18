"""มือสังเคราะห์สำหรับ test — สร้าง ``Hand``/``HandFrame`` ที่ควบคุมค่าที่ engine วัดได้ตรงๆ

โครงมือ (หน่วยเป็นเท่าของ ``scale`` = ระยะ wrist → middle MCP เป็น px, มือตั้งตรง นิ้วชี้ขึ้น)
วางรอบ "anchor" (กึ่งกลางโคนนิ้วชี้กับนิ้วกลาง) ซึ่งเป็นจุดที่ engine ใช้คุมเคอร์เซอร์:

- ``pinch`` / ``middle_pinch`` คือ pinch_ratio ของนิ้วชี้/นิ้วกลางที่จะวัดได้ **พอดี**
- ``fist=True`` พับนิ้วชี้กับนิ้วกลางเข้าฝ่ามือ (ปลายนิ้วชิดโป้ง แต่ต้องไม่นับเป็นการจีบ)
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

from aether.gestures import GestureEngine, GestureEvent, State
from aether.tracker import LANDMARK_COUNT, LM, Hand, HandFrame, Landmark

WIDTH, HEIGHT = 640, 480
FRAME_MS = 33  # ~30 fps

OPEN = 1.0  # pinch_ratio ของมือที่กางอยู่ (ไกลกว่า pinch.off มาก)
PINCHED = 0.15  # จีบแน่น (ต่ำกว่า pinch.on = 0.25)

_INDEX_DIR = (0.6, -0.8)  # ทิศจากปลายโป้งไปปลายนิ้วชี้ (เวกเตอร์หนึ่งหน่วย)
_MIDDLE_DIR = (0.8, -0.6)
_RING_DIR = (0.92, -0.39)
_PINKY_DIR = (0.97, -0.24)


def make_hand(
    x: float = 0.5,
    y: float = 0.4,
    *,
    pinch: float = OPEN,
    middle_pinch: float = 1.2,
    ring_pinch: float | None = None,
    pinky_pinch: float | None = None,
    fist: bool = False,
    gesture: str = "None",
    score: float = 0.9,
    side: str = "right",
    scale: float = 100.0,
    side_score: float = 0.95,
) -> Hand:
    """มือที่ anchor อยู่ที่ (x, y) normalized ของเฟรม 640x480"""
    ax, ay = x * WIDTH, y * HEIGHT
    s = scale

    def at(dx: float, dy: float) -> tuple[float, float]:
        return (ax + dx * s, ay + dy * s)

    px: dict[int, tuple[float, float]] = {
        LM.INDEX_MCP: at(-0.11, 0.0),
        LM.MIDDLE_MCP: at(0.11, 0.0),
        LM.RING_MCP: at(0.31, 0.05),
        LM.PINKY_MCP: at(0.50, 0.15),
        LM.WRIST: at(0.11, 1.0),  # ตรงใต้ middle MCP พอดี → hand_scale = s
        LM.THUMB_CMC: at(-0.15, 0.80),
        LM.THUMB_MCP: at(-0.35, 0.55),
        LM.THUMB_IP: at(-0.40, 0.15),
        LM.RING_PIP: at(0.31, -0.30),
        LM.RING_DIP: at(0.31, -0.55),
        LM.RING_TIP: at(0.31, -0.75),
        LM.PINKY_PIP: at(0.50, -0.15),
        LM.PINKY_DIP: at(0.50, -0.35),
        LM.PINKY_TIP: at(0.50, -0.50),
    }
    if fist:
        thumb = at(0.0, 0.30)
        index_tip = at(0.0, 0.45)  # ห่างโป้ง 0.15 แต่พับอยู่ในฝ่ามือ
        middle_tip = at(0.10, 0.50)
    else:
        thumb = at(-0.35, -0.25)
        index_tip = (thumb[0] + pinch * s * _INDEX_DIR[0], thumb[1] + pinch * s * _INDEX_DIR[1])
        middle_tip = (
            thumb[0] + middle_pinch * s * _MIDDLE_DIR[0],
            thumb[1] + middle_pinch * s * _MIDDLE_DIR[1],
        )
    px[LM.THUMB_TIP] = thumb
    px[LM.INDEX_TIP] = index_tip
    px[LM.MIDDLE_TIP] = middle_tip
    for ratio, direction, joints in (
        (ring_pinch, _RING_DIR, (LM.RING_MCP, LM.RING_PIP, LM.RING_DIP, LM.RING_TIP)),
        (pinky_pinch, _PINKY_DIR, (LM.PINKY_MCP, LM.PINKY_PIP, LM.PINKY_DIP, LM.PINKY_TIP)),
    ):
        if ratio is None or fist:
            continue
        mcp, pip, dip, tip = joints  # ปลายนิ้วมาแตะโป้งที่ระยะ ratio (ข้อนิ้วเรียงตามเส้นตรง)
        px[tip] = (thumb[0] + ratio * s * direction[0], thumb[1] + ratio * s * direction[1])
        (x0, y0), (x1, y1) = px[mcp], px[tip]
        px[pip] = (x0 + (x1 - x0) * 0.45, y0 + (y1 - y0) * 0.45)
        px[dip] = (x0 + (x1 - x0) * 0.75, y0 + (y1 - y0) * 0.75)
    for mcp, pip, dip, tip in (
        (LM.INDEX_MCP, LM.INDEX_PIP, LM.INDEX_DIP, LM.INDEX_TIP),
        (LM.MIDDLE_MCP, LM.MIDDLE_PIP, LM.MIDDLE_DIP, LM.MIDDLE_TIP),
    ):
        (x0, y0), (x1, y1) = px[mcp], px[tip]
        px[pip] = (x0 + (x1 - x0) * 0.45, y0 + (y1 - y0) * 0.45)
        px[dip] = (x0 + (x1 - x0) * 0.75, y0 + (y1 - y0) * 0.75)

    assert len(px) == LANDMARK_COUNT
    landmarks = tuple(Landmark(px[i][0] / WIDTH, px[i][1] / HEIGHT, 0.0) for i in range(21))
    return Hand(
        handedness=side,
        handedness_score=side_score,
        gesture=gesture,
        gesture_score=score,
        landmarks=landmarks,
    )


def make_frame(t: int, *hands: Hand) -> HandFrame:
    return HandFrame(timestamp_ms=t, width=WIDTH, height=HEIGHT, hands=tuple(hands))


class Driver:
    """ป้อนเฟรมเข้า engine ทีละเฟรมที่ ~30fps แล้วเก็บ event ทั้งหมด"""

    def __init__(self, engine: GestureEngine, start_ms: int = 1000) -> None:
        self.engine = engine
        self.t = start_ms
        self.events: list[GestureEvent] = []
        self.states: list[State] = []

    def step(self, *hands: Hand) -> list[GestureEvent]:
        """หนึ่งเฟรม (ไม่ส่งมือ = เฟรมที่ไม่เห็นมือ)"""
        state, events = self.engine.update(make_frame(self.t, *hands))
        self.t += FRAME_MS
        self.events.extend(events)
        self.states.append(state)
        return events

    def hold(self, hand: Hand | None, ms: int) -> list[GestureEvent]:
        """ป้อนมือท่าเดิมต่อเนื่อง ``ms`` มิลลิวินาที"""
        return self.play([hand] * max(1, round(ms / FRAME_MS)))

    def play(self, hands: Iterable[Hand | None]) -> list[GestureEvent]:
        collected: list[GestureEvent] = []
        for hand in hands:
            collected.extend(self.step(*(() if hand is None else (hand,))))
        return collected

    def animate(self, make: Callable[[float], Hand], ms: int) -> list[GestureEvent]:
        """ป้อนมือที่เปลี่ยนตามเวลา: ``make(progress)`` โดย progress วิ่ง 0 → 1"""
        frames = max(2, round(ms / FRAME_MS))
        return self.play(make(i / (frames - 1)) for i in range(frames))

    def clear(self) -> None:
        self.events.clear()
        self.states.clear()
