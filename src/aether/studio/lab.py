"""ห้องทดลองมือ — วัดค่าต่างๆ ของมือแบบสด และเก็บตัวอย่างท่ามือที่ติดป้ายไว้ (pure)

- :func:`hand_metrics`: นิ้วไหนเหยียด/งอ, จำนวนนิ้ว, มุมงอของแต่ละนิ้ว, ระยะจีบของแต่ละนิ้ว, มุมเอียงของมือ,
  หันฝ่ามือหรือหลังมือเข้ากล้อง
- :class:`SampleRecorder`: ต่อท้ายไฟล์ ``hand_samples.jsonl`` ทีละบรรทัด — **มีแต่ตัวเลข landmark ไม่มีภาพ**
  ใช้เป็นข้อมูลสอน/ทดสอบตัวจำแนกท่ามือ (``tools/train_poses.py``)
"""

from __future__ import annotations

import json
import math
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from ..geometry import Finger, finger_curled, finger_extended, hand_scale, pinch_ratio
from ..tracker import LM, Hand, Landmark

__all__ = [
    "FINGER_NAMES",
    "HandMetrics",
    "SampleRecorder",
    "hand_metrics",
    "joint_angle",
    "read_samples",
    "sample_hand",
]

FINGER_NAMES = ("Thumb", "Index", "Middle", "Ring", "Pinky")
_FINGERS = (Finger.THUMB, Finger.INDEX, Finger.MIDDLE, Finger.RING, Finger.PINKY)
_CHAINS = (  # จุดที่ใช้วัดมุมงอ: ข้อมือ/โคน → ข้อ → ข้อ → ปลาย
    (LM.WRIST, LM.THUMB_CMC, LM.THUMB_MCP, LM.THUMB_IP, LM.THUMB_TIP),
    (LM.WRIST, LM.INDEX_MCP, LM.INDEX_PIP, LM.INDEX_DIP, LM.INDEX_TIP),
    (LM.WRIST, LM.MIDDLE_MCP, LM.MIDDLE_PIP, LM.MIDDLE_DIP, LM.MIDDLE_TIP),
    (LM.WRIST, LM.RING_MCP, LM.RING_PIP, LM.RING_DIP, LM.RING_TIP),
    (LM.WRIST, LM.PINKY_MCP, LM.PINKY_PIP, LM.PINKY_DIP, LM.PINKY_TIP),
)
SAMPLE_VERSION = 1


def joint_angle(a: Landmark, b: Landmark, c: Landmark) -> float:
    """มุมงอที่ข้อ ``b`` (องศา): 0 = ตรง, 90 = งอตั้งฉาก (ใช้ 3 มิติ)"""
    v1 = (a.x - b.x, a.y - b.y, a.z - b.z)
    v2 = (c.x - b.x, c.y - b.y, c.z - b.z)
    n1, n2 = math.sqrt(sum(v * v for v in v1)), math.sqrt(sum(v * v for v in v2))
    if n1 < 1e-9 or n2 < 1e-9:
        return 0.0
    cosine = max(-1.0, min(1.0, sum(p * q for p, q in zip(v1, v2, strict=True)) / (n1 * n2)))
    return 180.0 - math.degrees(math.acos(cosine))


@dataclass(frozen=True, slots=True)
class HandMetrics:
    handedness: str
    extended: tuple[bool, ...]  # โป้ง → ก้อย
    curled: tuple[bool, ...]
    bend: tuple[float, ...]  # มุมงอรวม (องศา) ของแต่ละนิ้ว: ผลรวมมุมที่ 3 ข้อ
    pinch: tuple[float, ...]  # ระยะโป้ง ↔ ชี้/กลาง/นาง/ก้อย ÷ ขนาดมือ
    roll: float  # องศา: 0 = นิ้วชี้ขึ้น, บวก = เอียงตามเข็ม (ในภาพ)
    facing: str  # "palm" (หันฝ่ามือเข้ากล้อง) / "back"
    size: float  # ขนาดมือ ÷ ความกว้างเฟรม
    gesture: str
    gesture_score: float

    @property
    def count(self) -> int:
        return sum(self.extended)


def hand_metrics(hand: Hand, width: int, height: int) -> HandMetrics:
    points = hand.world_landmarks or hand.landmarks
    bends = tuple(
        sum(joint_angle(points[c[i - 1]], points[c[i]], points[c[i + 1]]) for i in (1, 2, 3))
        for c in _CHAINS
    )
    pinches = tuple(
        pinch_ratio(hand, width, height, tip)
        for tip in (LM.INDEX_TIP, LM.MIDDLE_TIP, LM.RING_TIP, LM.PINKY_TIP)
    )
    lm = hand.landmarks
    wrist, middle = lm[LM.WRIST], lm[LM.MIDDLE_MCP]
    roll = math.degrees(math.atan2((middle.x - wrist.x) * width, -(middle.y - wrist.y) * height))
    # หันฝ่ามือ: เครื่องหมายของ cross product (ข้อมือ→โคนชี้) × (ข้อมือ→โคนก้อย) กลับกันตามข้างมือ
    index, pinky = lm[LM.INDEX_MCP], lm[LM.PINKY_MCP]
    cross = (index.x - wrist.x) * (pinky.y - wrist.y) - (index.y - wrist.y) * (pinky.x - wrist.x)
    palm = cross < 0 if hand.handedness == "right" else cross > 0
    return HandMetrics(
        hand.handedness,
        tuple(finger_extended(hand, f, width, height) for f in _FINGERS),
        tuple(finger_curled(hand, f, width, height) for f in _FINGERS),
        bends,
        pinches,
        roll,
        "palm" if palm else "back",
        hand_scale(hand, width, height) / max(width, 1),
        hand.gesture,
        hand.gesture_score,
    )


def _rounded(points: tuple[Landmark, ...]) -> list[list[float]]:
    return [[round(p.x, 5), round(p.y, 5), round(p.z, 5)] for p in points]


class SampleRecorder:
    """เก็บตัวอย่างท่ามือที่ติดป้ายลงไฟล์ JSON Lines (ต่อท้าย ไม่เขียนทับ)"""

    def __init__(self, path: Path, *, clock=time.time) -> None:
        self.path = Path(path)
        self._clock = clock
        self.counts: Counter[str] = Counter(sample["label"] for sample in read_samples(self.path))

    def add(self, label: str, hand: Hand, width: int, height: int) -> None:
        label = label.strip()
        if not label:
            raise ValueError("ต้องตั้งชื่อป้ายของท่าก่อนเก็บตัวอย่าง")
        record = {
            "v": SAMPLE_VERSION,
            "label": label,
            "handedness": hand.handedness,
            "width": width,
            "height": height,
            "time": round(self._clock(), 3),
            "landmarks": _rounded(hand.landmarks),
            "world": _rounded(hand.world_landmarks),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.counts[label] += 1


def read_samples(path: Path) -> list[dict]:
    """ตัวอย่างทั้งหมดในไฟล์ (บรรทัดที่เสีย/คนละเวอร์ชันข้ามไป) — ไม่มีไฟล์ = ``[]``"""
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return []
    samples = []
    for line in lines:
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (
            isinstance(record, dict)
            and record.get("v") == SAMPLE_VERSION
            and isinstance(record.get("label"), str)
            and len(record.get("landmarks", ())) == 21
        ):
            samples.append(record)
    return samples


def sample_hand(record: dict) -> Hand:
    """แปลงตัวอย่างที่เก็บไว้กลับเป็น :class:`Hand` (ใช้ตอนสอน/ทดสอบตัวจำแนก)"""
    world = record.get("world") or []
    return Hand(
        record.get("handedness", "right"),
        1.0,
        "None",
        0.0,
        tuple(Landmark(*p) for p in record["landmarks"]),
        tuple(Landmark(*p) for p in world) if len(world) == 21 else (),
    )
