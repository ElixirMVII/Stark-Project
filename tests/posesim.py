"""มือสังเคราะห์ที่กำหนดได้ว่านิ้วไหนเหยียด/งอ — ใช้ test ท่าที่ผู้ใช้สอนเอง (``aether.poses``)

``fingers`` เป็นสตริง 5 ตัว (โป้ง ชี้ กลาง นาง ก้อย) ``"1"`` = เหยียด ``"0"`` = งอ เช่น
``"11111"`` แบมือ, ``"01001"`` ชี้+ก้อย, ``"10000"`` ยกนิ้วโป้ง
หน่วยของโครงมือคือระยะข้อมือ→โคนนิ้วกลาง (``scale`` px) มือขวาตั้งตรง นิ้วโป้งอยู่ซ้ายของภาพ
(ตรงกับ ``handsim``)
"""

from __future__ import annotations

import math
import random

from aether.tracker import LM, Hand, Landmark

WIDTH, HEIGHT = 640, 480

_MCP = {
    "index": (-0.22, -0.95),
    "middle": (0.0, -1.0),
    "ring": (0.20, -0.95),
    "pinky": (0.38, -0.85),
}
_SPREAD = {"index": -0.10, "middle": 0.0, "ring": 0.08, "pinky": 0.18}
_SEGMENTS = {
    "index": (0.40, 0.25, 0.22),
    "middle": (0.44, 0.28, 0.24),
    "ring": (0.40, 0.26, 0.22),
    "pinky": (0.32, 0.20, 0.18),
}
_JOINTS = {
    "index": (LM.INDEX_PIP, LM.INDEX_DIP, LM.INDEX_TIP),
    "middle": (LM.MIDDLE_PIP, LM.MIDDLE_DIP, LM.MIDDLE_TIP),
    "ring": (LM.RING_PIP, LM.RING_DIP, LM.RING_TIP),
    "pinky": (LM.PINKY_PIP, LM.PINKY_DIP, LM.PINKY_TIP),
}
_MCP_INDEX = {
    "index": LM.INDEX_MCP,
    "middle": LM.MIDDLE_MCP,
    "ring": LM.RING_MCP,
    "pinky": LM.PINKY_MCP,
}

# เพิ่มกำหนดท่าที่ใช้บ่อยใน test
POSES = {
    "open": "11111",
    "fist": "00000",
    "rock": "01001",  # ชี้ + ก้อย
    "gun": "11000",  # โป้ง + ชี้ (รูปตัว L)
    "shaka": "10001",  # โป้ง + ก้อย
    "three": "01110",
    "thumb": "10000",
    "pinky": "00001",
}


def local_points(fingers: str) -> dict[int, tuple[float, float, float]]:
    """โครงมือขวาตั้งตรงในหน่วยของมือ: {landmark: (x, y, z)} (y ชี้ลง)"""
    assert len(fingers) == 5 and set(fingers) <= {"0", "1"}
    thumb, *rest = (c == "1" for c in fingers)
    points: dict[int, tuple[float, float, float]] = {LM.WRIST: (0.0, 0.0, 0.0)}
    for name, (x, y) in _MCP.items():
        points[_MCP_INDEX[name]] = (x, y, 0.0)
    for name, extended in zip(("index", "middle", "ring", "pinky"), rest):
        mx, my = _MCP[name]
        pip, dip, tip = _JOINTS[name]
        if extended:
            dx, dy = _SPREAD[name], -1.0
            norm = math.hypot(dx, dy)
            dx, dy = dx / norm, dy / norm
            total = 0.0
            for joint, length in zip((pip, dip, tip), _SEGMENTS[name]):
                total += length
                points[joint] = (mx + dx * total, my + dy * total, 0.0)
        else:  # พับลงมาในฝ่ามือ เข้าหากล้อง (ปลายนิ้วใกล้ข้อมือกว่าโคนนิ้ว = finger_curled)
            points[pip] = (mx, my - 0.28, -0.10)
            points[dip] = (mx, my + 0.02, -0.28)
            points[tip] = (mx, my + 0.35, -0.22)
    points[LM.THUMB_CMC] = (-0.20, -0.25, 0.0)
    points[LM.THUMB_MCP] = (-0.38, -0.45, 0.0)
    if thumb:
        points[LM.THUMB_IP] = (-0.55, -0.62, 0.0)
        points[LM.THUMB_TIP] = (-0.68, -0.78, 0.0)
    else:  # พาดฝ่ามือ
        points[LM.THUMB_IP] = (-0.25, -0.62, -0.10)
        points[LM.THUMB_TIP] = (-0.05, -0.68, -0.15)
    assert len(points) == 21
    return points


def make_pose_hand(
    fingers: str = "11111",
    x: float = 0.5,
    y: float = 0.6,
    *,
    angle: float = 0.0,
    tilt: float = 0.0,
    scale: float = 110.0,
    side: str = "right",
    noise: float = 0.0,
    rng: random.Random | None = None,
    gesture: str = "None",
    score: float = 0.9,
) -> Hand:
    """มือที่ข้อมืออยู่ที่ (x, y) normalized ของเฟรม 640x480

    ``angle`` หมุนในระนาบภาพ (องศา ตามเข็มนาฬิกาบนจอ), ``tilt`` เอียงฝ่ามือเข้าหา/ออกจากกล้อง
    (องศา), ``noise`` คือ σ ของสัญญาณรบกวนต่อพิกัด (หน่วยของมือ)
    """
    rng = rng or random.Random(0)
    flip = -1.0 if side == "left" else 1.0
    cos_a, sin_a = math.cos(math.radians(angle)), math.sin(math.radians(angle))
    cos_t, sin_t = math.cos(math.radians(tilt)), math.sin(math.radians(tilt))
    landmarks = []
    for index in range(21):
        lx, ly, lz = local_points(fingers)[index]
        if noise:
            lx += rng.gauss(0.0, noise)
            ly += rng.gauss(0.0, noise)
            lz += rng.gauss(0.0, noise)
        ly, lz = ly * cos_t, lz + ly * sin_t
        lx *= flip
        rx = lx * cos_a - ly * sin_a
        ry = lx * sin_a + ly * cos_a
        px, py = x * WIDTH + rx * scale, y * HEIGHT + ry * scale
        landmarks.append(Landmark(px / WIDTH, py / HEIGHT, lz * scale / WIDTH))
    return Hand(
        handedness=side,
        handedness_score=0.95,
        gesture=gesture,
        gesture_score=score,
        landmarks=tuple(landmarks),
    )


def samples(fingers: str, count: int = 40, *, seed: int = 1, **kwargs) -> list[Hand]:
    """ตัวอย่างหลายเฟรมแบบที่ผู้ใช้ทำตอนสอน: ขยับ/หมุน/เอียงเล็กน้อย + noise"""
    rng = random.Random(seed)
    base_angle = kwargs.pop("angle", 0.0)
    noise = kwargs.pop("noise", 0.02)
    hands = []
    for _ in range(count):
        hands.append(
            make_pose_hand(
                fingers,
                0.5 + rng.uniform(-0.05, 0.05),
                0.6 + rng.uniform(-0.05, 0.05),
                angle=base_angle + rng.uniform(-12.0, 12.0),
                tilt=rng.uniform(-15.0, 15.0),
                scale=rng.uniform(95.0, 125.0),
                noise=noise,
                rng=rng,
                **kwargs,
            )
        )
    return hands
