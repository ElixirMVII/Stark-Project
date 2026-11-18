"""มือสังเคราะห์แบบ 3 มิติ (kinematic) — สร้าง landmark 21 จุดแบบที่ MediaPipe จะเห็น จากท่าและมุมกล้อง

ใช้สร้างข้อมูลสอน/ทดสอบตัวจำแนกท่ามือ (:mod:`aether.training`) โดยไม่ต้องมีกล้องหรือภาพ:

- โครงกระดูกมือขวาของผู้ใหญ่ (เมตร) — ฝ่ามือหันเข้ากล้อง นิ้วชี้ขึ้น นิ้วโป้งอยู่ซ้ายของภาพ (ภาพ mirror แบบ selfie
  เหมือนที่ AETHER ใช้) ความยาวกระดูกแต่ละท่อนตามสัดส่วนเฉลี่ยของมือคน
- แต่ละนิ้วงอด้วยค่า ``curl`` 0 (เหยียด) … 1 (กำ) แจกเป็นมุมของข้อ MCP/PIP/DIP + กางนิ้วออกด้านข้าง
- นิ้วโป้งผสมระหว่าง "กางออก" กับ "พับพาดฝ่ามือ"
- หมุนทั้งมือ (roll ในระนาบภาพ, pitch/yaw ออกนอกระนาบ) วางห่างกล้องตามระยะ แล้วฉายภาพแบบ perspective
  ด้วยกล้องมุมกว้าง ~60° ได้ ``landmarks`` (normalized + z สัมพัทธ์กับข้อมือ) และ ``world_landmarks`` (เมตร)

ทุกอย่าง deterministic เมื่อให้ ``random.Random`` เดียวกัน
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .tracker import LM, Hand, Landmark

__all__ = [
    "FINGERS",
    "HandShape",
    "HandView",
    "build_hand",
    "skeleton",
]

FINGERS = ("thumb", "index", "middle", "ring", "pinky")

# ตำแหน่งโคนนิ้ว (MCP) จากข้อมือ (เมตร): x = ไปทางนิ้วก้อย, y = ไปทางนิ้ว
_MCP = {
    "index": (-0.024, 0.088),
    "middle": (-0.003, 0.091),
    "ring": (0.016, 0.086),
    "pinky": (0.032, 0.077),
}
# ความยาวกระดูกนิ้ว (proximal, middle, distal)
_PHALANGES = {
    "index": (0.040, 0.024, 0.020),
    "middle": (0.045, 0.028, 0.021),
    "ring": (0.042, 0.027, 0.021),
    "pinky": (0.033, 0.019, 0.018),
}
_SPLAY = {"index": -0.10, "middle": 0.0, "ring": 0.08, "pinky": 0.18}  # เรเดียน ตอนกางนิ้วสุด
_THUMB_CMC = (-0.024, 0.026, 0.008)
_THUMB_BONES = (0.040, 0.031, 0.025)  # metacarpal, proximal, distal
# ทิศของกระดูกนิ้วโป้งแต่ละท่อน: กางออก vs พับพาดฝ่ามือ (x, y, z — z = ไปทางกล้อง)
_THUMB_OPEN = ((-0.62, 0.70, 0.35), (-0.66, 0.70, 0.27), (-0.66, 0.72, 0.20))
_THUMB_FOLDED = ((-0.15, 0.75, 0.64), (0.62, 0.45, 0.64), (0.90, 0.02, 0.44))
# มุมงอสูงสุด (องศา) ของ MCP, PIP, DIP ตอน curl = 1
_FLEX = (85.0, 105.0, 70.0)

HORIZONTAL_FOV = 60.0  # องศา (เว็บแคมทั่วไป)


@dataclass(frozen=True, slots=True)
class HandShape:
    """ท่ามือ: ระดับการงอของแต่ละนิ้ว (0 = เหยียด, 1 = กำ) + การกางนิ้ว + สัดส่วนมือ"""

    curl: tuple[float, float, float, float, float] = (0.0, 0.0, 0.0, 0.0, 0.0)  # โป้ง → ก้อย
    spread: float = 0.5  # 0 = นิ้วชิดกัน, 1 = กางสุด
    size: float = 1.0  # ขนาดมือเทียบค่าเฉลี่ย
    finger_length: float = 1.0  # นิ้วยาว/สั้นเทียบฝ่ามือ
    jitter: tuple[float, ...] = ()  # มุมเพิ่ม/ลด (องศา) ต่อข้อ: 4 นิ้ว × 3 ข้อ (ว่าง = 0)
    # ตัวคูณมุมงอสูงสุดของ MCP, PIP, DIP — คนละคน/คนละท่างอไม่เท่ากัน (เช่นนิ้วที่อยู่ข้างนิ้วเหยียด
    # งอโคนนิ้วได้ไม่สุดเพราะเส้นเอ็นยึดกัน)
    flex_scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    thumb_bias: tuple[float, float, float] = (0.0, 0.0, 0.0)  # เลื่อนทิศของนิ้วโป้งที่พับ (x, y, z)
    thumb_angle: float = 42.0  # องศาจากแนวตั้งของนิ้วโป้งที่กาง (ท่า L / 🤙 กางเกือบแนวนอน ~70–85°)


@dataclass(frozen=True, slots=True)
class HandView:
    """มือถูกถือไว้ที่ไหน หันทางไหน ในภาพของกล้อง"""

    x: float = 0.5  # ตำแหน่งข้อมือในภาพ (normalized)
    y: float = 0.65
    distance: float = 0.55  # เมตร จากกล้อง
    roll: float = 0.0  # องศา: หมุนในระนาบภาพ (บวก = ตามเข็มบนจอ)
    pitch: float = 0.0  # องศา: ปลายนิ้วเอนเข้าหากล้อง (+) / ออก (−)
    yaw: float = 0.0  # องศา: หมุนรอบแกนตั้ง (เห็นข้างมือ)
    side: str = "right"
    width: int = 640
    height: int = 480
    noise: float = 0.0  # σ ของสัญญาณรบกวนต่อพิกัด (normalized) แบบ landmark จริง
    extra: dict = field(default_factory=dict)


def _normalize(v: tuple[float, float, float]) -> tuple[float, float, float]:
    n = math.sqrt(sum(c * c for c in v)) or 1.0
    return (v[0] / n, v[1] / n, v[2] / n)


def _lerp(a, b, t: float):
    return tuple(x + (y - x) * t for x, y in zip(a, b, strict=True))


def _add(a, b, scale: float = 1.0):
    return tuple(x + y * scale for x, y in zip(a, b, strict=True))


def _thumb_open(direction, angle: float):
    """หมุนทิศของนิ้วโป้งที่กางในระนาบฝ่ามือให้ทำมุม ``angle`` กับแนวตั้ง (คง z ไว้)"""
    x, y, z = direction
    planar = math.hypot(x, y)
    base = math.degrees(math.atan2(-x, y))  # ทิศเดิม (~42°)
    a = math.radians(angle + (math.degrees(math.atan2(-x, y)) - base))
    return (-math.sin(a) * planar, math.cos(a) * planar, z)


def skeleton(shape: HandShape) -> list[tuple[float, float, float]]:
    """ตำแหน่ง 21 จุดของมือขวาในพิกัดมือ (เมตร, ข้อมือ = จุดกำเนิด, y ขึ้น, z ไปทางกล้อง)"""
    s = shape.size
    lengths = shape.finger_length
    points: list[tuple[float, float, float]] = [(0.0, 0.0, 0.0)] * 21
    jitter = shape.jitter or (0.0,) * 12

    # นิ้วโป้ง
    t = min(max(shape.curl[0], 0.0), 1.0)
    cmc = tuple(c * s for c in _THUMB_CMC)
    points[LM.THUMB_CMC] = cmc
    joint = cmc
    for bone, (open_dir, folded_dir), index in zip(
        _THUMB_BONES,
        zip(_THUMB_OPEN, _THUMB_FOLDED, strict=True),
        (LM.THUMB_MCP, LM.THUMB_IP, LM.THUMB_TIP),
        strict=True,
    ):
        folded = _add(folded_dir, shape.thumb_bias)
        direction = _normalize(_lerp(_thumb_open(open_dir, shape.thumb_angle), folded, t))
        joint = _add(joint, direction, bone * s * (lengths if index != LM.THUMB_MCP else 1.0))
        points[index] = joint

    # นิ้วชี้ → ก้อย
    chains = {
        "index": (LM.INDEX_MCP, LM.INDEX_PIP, LM.INDEX_DIP, LM.INDEX_TIP),
        "middle": (LM.MIDDLE_MCP, LM.MIDDLE_PIP, LM.MIDDLE_DIP, LM.MIDDLE_TIP),
        "ring": (LM.RING_MCP, LM.RING_PIP, LM.RING_DIP, LM.RING_TIP),
        "pinky": (LM.PINKY_MCP, LM.PINKY_PIP, LM.PINKY_DIP, LM.PINKY_TIP),
    }
    for f, name in enumerate(("index", "middle", "ring", "pinky")):
        curl = min(max(shape.curl[f + 1], 0.0), 1.0)
        mcp = (_MCP[name][0] * s, _MCP[name][1] * s, 0.0)
        # นิ้วที่งอมากกางน้อยลง (ตอนกำ นิ้วชิดกัน)
        splay = _SPLAY[name] * (0.3 + shape.spread) * (1.0 - 0.8 * curl)
        indices = chains[name]
        points[indices[0]] = mcp
        joint, theta = mcp, 0.0
        for k, bone in enumerate(_PHALANGES[name]):
            theta += math.radians(_FLEX[k] * shape.flex_scale[k] * curl + jitter[f * 3 + k])
            direction = (
                math.sin(splay) * math.cos(theta),
                math.cos(splay) * math.cos(theta),
                math.sin(theta),
            )
            joint = _add(joint, direction, bone * s * lengths)
            points[indices[k + 1]] = joint
    return points


def _rotate(point, roll: float, pitch: float, yaw: float):
    """หมุนรอบข้อมือ: yaw (แกน y) → pitch (แกน x) → roll (แกน z, ตามเข็มบนจอ)"""
    x, y, z = point
    cy, sy = math.cos(yaw), math.sin(yaw)
    x, z = x * cy + z * sy, -x * sy + z * cy
    cp, sp = math.cos(pitch), math.sin(pitch)
    y, z = y * cp - z * sp, y * sp + z * cp
    cr, sr = math.cos(-roll), math.sin(-roll)  # roll บวก = ตามเข็มบนจอ (y ขึ้น → หมุนลบ)
    x, y = x * cr - y * sr, x * sr + y * cr
    return (x, y, z)


def build_hand(
    shape: HandShape,
    view: HandView,
    *,
    rng=None,
    gesture: str = "None",
    gesture_score: float = 0.0,
) -> Hand:
    """มือหนึ่งข้างตามท่า ``shape`` ที่ถืออยู่แบบ ``view`` → :class:`~aether.tracker.Hand`"""
    local = skeleton(shape)
    if view.side == "left":  # มือซ้าย = กลับซ้ายขวาของมือขวา
        local = [(-x, y, z) for x, y, z in local]
    roll, pitch, yaw = (math.radians(a) for a in (view.roll, view.pitch, view.yaw))
    rotated = [_rotate(p, roll, pitch, yaw) for p in local]

    focal = (view.width / 2.0) / math.tan(math.radians(HORIZONTAL_FOV / 2.0))  # px
    # ตำแหน่งข้อมือในพื้นที่กล้อง (เมตร): ให้ฉายไปตกที่ (view.x, view.y) พอดี
    wx = (view.x - 0.5) * view.width * view.distance / focal
    wy = -(view.y - 0.5) * view.height * view.distance / focal
    landmarks = []
    for x, y, z in rotated:
        depth = view.distance - z  # z ไปทางกล้อง = ใกล้ขึ้น
        u = 0.5 + focal * (wx + x) / depth / view.width
        v = 0.5 - focal * (wy + y) / depth / view.height
        rel = -z * focal / (view.distance * view.width)  # แบบ MediaPipe: ลบ = ใกล้กล้องกว่าข้อมือ
        landmarks.append([u, v, rel])
    if rng is not None and view.noise > 0.0:
        for point in landmarks:
            point[0] += rng.gauss(0.0, view.noise)
            point[1] += rng.gauss(0.0, view.noise * view.width / view.height)
            point[2] += rng.gauss(0.0, view.noise * 1.5)
    cx = sum(p[0] for p in rotated) / 21
    cy = sum(p[1] for p in rotated) / 21
    cz = sum(p[2] for p in rotated) / 21
    world = tuple(Landmark(x - cx, -(y - cy), -(z - cz)) for x, y, z in rotated)
    return Hand(
        view.side,
        0.97,
        gesture,
        gesture_score,
        tuple(Landmark(*p) for p in landmarks),
        world,
    )
