"""เรขาคณิตของมือ — pure function ทั้งหมด (ไม่แตะกล้อง ไม่แตะ OS)

พิกัด ``landmarks`` เป็น normalized 0–1 ของเฟรมซึ่งไม่ใช่สี่เหลี่ยมจัตุรัส การวัดระยะในภาพ
จึงต้องคูณ width/height ของเฟรมก่อนเสมอ (ฟังก์ชันที่ลงท้าย ``_px`` ทำให้แล้ว)
"""

from __future__ import annotations

import enum
import math

from .tracker import LM, Hand, Landmark

__all__ = [
    "CURL_RATIO",
    "Finger",
    "Point",
    "distance_3d",
    "distance_px",
    "finger_curled",
    "finger_extended",
    "hand_scale",
    "palm_center",
    "pinch_ratio",
    "pointer_anchor",
    "to_px",
]

Point = tuple[float, float]

# นิ้วถือว่า "งอเข้าฝ่ามือ" เมื่อปลายนิ้วอยู่ใกล้ข้อมือกว่าโคนนิ้ว (MCP) ด้วยอัตราส่วนนี้
# กำมือ ≈ 0.5–0.65, จีบนิ้ว ≈ 0.9–1.25, เหยียดนิ้ว ≈ 1.8
CURL_RATIO = 0.8

_PALM = (LM.WRIST, LM.INDEX_MCP, LM.MIDDLE_MCP, LM.RING_MCP, LM.PINKY_MCP)


class Finger(enum.Enum):
    """index ของข้อนิ้ว (โคน, ข้อกลาง, ข้อปลาย, ปลายนิ้ว)"""

    THUMB = (LM.THUMB_CMC, LM.THUMB_MCP, LM.THUMB_IP, LM.THUMB_TIP)
    INDEX = (LM.INDEX_MCP, LM.INDEX_PIP, LM.INDEX_DIP, LM.INDEX_TIP)
    MIDDLE = (LM.MIDDLE_MCP, LM.MIDDLE_PIP, LM.MIDDLE_DIP, LM.MIDDLE_TIP)
    RING = (LM.RING_MCP, LM.RING_PIP, LM.RING_DIP, LM.RING_TIP)
    PINKY = (LM.PINKY_MCP, LM.PINKY_PIP, LM.PINKY_DIP, LM.PINKY_TIP)

    @property
    def mcp(self) -> int:
        return self.value[0]

    @property
    def pip(self) -> int:
        return self.value[1]

    @property
    def tip(self) -> int:
        return self.value[3]


def to_px(point: Landmark, width: int, height: int) -> Point:
    return (point.x * width, point.y * height)


def distance_px(a: Landmark, b: Landmark, width: int, height: int) -> float:
    """ระยะในภาพ (px ของเฟรม) ระหว่าง landmark สองจุด"""
    return math.hypot((a.x - b.x) * width, (a.y - b.y) * height)


def distance_3d(a: Landmark, b: Landmark) -> float:
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)


def _distance(hand: Hand, a: int, b: int, width: int, height: int) -> float:
    """ระยะระหว่างสองจุดของมือ: ใช้ world landmark (3D, ไม่เพี้ยนตามมุมมือ) ถ้ามี ไม่งั้นใช้ระยะในภาพ"""
    if hand.world_landmarks:
        return distance_3d(hand.world_landmarks[a], hand.world_landmarks[b])
    return distance_px(hand.landmarks[a], hand.landmarks[b], width, height)


def hand_scale(hand: Hand, width: int, height: int) -> float:
    """ขนาดมือในภาพ = ระยะ wrist(0) → middle MCP(9) เป็น px ใช้ normalize ระยะอื่นๆ"""
    return distance_px(hand.landmarks[LM.WRIST], hand.landmarks[LM.MIDDLE_MCP], width, height)


def pinch_ratio(hand: Hand, width: int, height: int, tip: int = LM.INDEX_TIP) -> float:
    """ระยะปลายนิ้วโป้ง ↔ ปลายนิ้ว ``tip`` หารด้วย :func:`hand_scale`

    ค่าน้อย = นิ้วชิดกัน ถ้ามือเล็กจนวัดไม่ได้คืน ``inf`` (ถือว่าไม่ได้จีบ)
    """
    scale = hand_scale(hand, width, height)
    if scale < 1e-6:
        return math.inf
    return distance_px(hand.landmarks[LM.THUMB_TIP], hand.landmarks[tip], width, height) / scale


def palm_center(hand: Hand) -> Point:
    """กลางฝ่ามือ (normalized) = ค่าเฉลี่ยของ landmark 0, 5, 9, 13, 17"""
    xs = [hand.landmarks[i].x for i in _PALM]
    ys = [hand.landmarks[i].y for i in _PALM]
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def pointer_anchor(hand: Hand) -> Point:
    """จุดคุมเคอร์เซอร์ (normalized) = ค่าเฉลี่ยของ index MCP(5) กับ middle MCP(9)

    ใช้โคนนิ้วแทนปลายนิ้ว เพราะตอนจีบปลายนิ้วขยับแต่โคนนิ้วแทบไม่ขยับ
    """
    a, b = hand.landmarks[LM.INDEX_MCP], hand.landmarks[LM.MIDDLE_MCP]
    return ((a.x + b.x) / 2.0, (a.y + b.y) / 2.0)


def finger_extended(hand: Hand, finger: Finger, width: int, height: int) -> bool:
    """นิ้วเหยียดอยู่หรือไม่

    นิ้วชี้–ก้อย: ปลายนิ้วอยู่ไกลข้อมือกว่าข้อกลางนิ้ว
    นิ้วโป้ง: ปลายนิ้วอยู่ไกลโคนนิ้วก้อยกว่าข้อปลายของนิ้วโป้ง (กางออกจากฝ่ามือ)
    """
    if finger is Finger.THUMB:
        return _distance(hand, LM.THUMB_TIP, LM.PINKY_MCP, width, height) > _distance(
            hand, LM.THUMB_IP, LM.PINKY_MCP, width, height
        )
    return _distance(hand, finger.tip, LM.WRIST, width, height) > _distance(
        hand, finger.pip, LM.WRIST, width, height
    )


def finger_curled(hand: Hand, finger: Finger, width: int, height: int) -> bool:
    """นิ้วงอเข้าฝ่ามือ (แบบตอนกำมือ) หรือไม่ — ใช้แยก "กำมือ" ออกจาก "จีบนิ้ว"

    ตอนจีบ ปลายนิ้วยังอยู่เลยโคนนิ้วออกไป ตอนกำมือ ปลายนิ้วพับกลับมาใกล้ข้อมือ
    """
    return (
        _distance(hand, finger.tip, LM.WRIST, width, height)
        < _distance(hand, finger.mcp, LM.WRIST, width, height) * CURL_RATIO
    )
