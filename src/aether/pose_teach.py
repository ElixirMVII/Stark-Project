"""ขั้นตอนสอนท่าใหม่ (pure logic — หน้าต่างอยู่ใน :mod:`aether.pose_wizard`)

:class:`TeachSession` รับ ``HandFrame`` ทีละเฟรม (เหมือน :class:`aether.calibration.CalibrationSession`):

1. ``NEUTRAL`` — ใช้มือตามปกติ (ชี้ ขยับ จีบเบาๆ) → ตัวอย่าง "มือปกติ" ไว้แยกท่าออกจากการใช้งานปกติ
2. ``POSE``    — ทำท่าใหม่ค้างไว้ (เอียง/ขยับเล็กน้อย) → ตัวอย่างของท่า แล้วตรวจว่าใช้ได้:
   ไม่ใช่ท่าที่แอปใช้อยู่แล้ว (✌️ 🖐 ✊ 🤟 👍), นิ้วโป้งไม่ชิดนิ้วชี้/กลางจนชนกับการคลิก,
   ไม่คล้ายมือปกติ และไม่คล้ายท่าที่สอนไว้แล้ว — ไม่ผ่านให้ทำขั้นนี้ใหม่พร้อมบอกเหตุผล
3. ``DONE``    — :attr:`TeachSession.result` คือ library ใหม่ที่มีท่านี้แล้ว (ยังไม่ได้บันทึกลงไฟล์)
"""

from __future__ import annotations

import enum
from collections import Counter
from dataclasses import dataclass

from .config import Config
from .geometry import Finger, finger_curled, pinch_ratio
from .gestures import PINCH_LOCK_FACTOR, reserved_gestures
from .hotkeys import parse_action
from .poses import (
    MAX_NEGATIVES,
    MAX_POSES,
    ORIENTATION_TOLERANCE,
    PoseLibrary,
    PoseTemplate,
    Vector,
    angle_difference,
    knn_distance,
    make_template,
    pose_features,
    spread,
    validate_pose_name,
)
from .tracker import LM, Hand, HandFrame

__all__ = [
    "GESTURE_ROLES",
    "TeachError",
    "TeachSession",
    "TeachStep",
    "TeachSummary",
    "check_pose",
]

SETTLE_MS = 1500  # เวลาจัดท่าก่อนเริ่มเก็บของแต่ละขั้น
SAMPLE_MS = {"NEUTRAL": 3000, "POSE": 3000}
MIN_SAMPLES = 20
NEUTRAL_SAMPLES = 60  # ตัวอย่างมือปกติที่เก็บเพิ่มต่อการสอนหนึ่งครั้ง

RESERVED_LIMIT = 0.5  # โมเดลเห็นเป็นท่าที่แอปใช้อยู่แล้วเกินสัดส่วนนี้ของเฟรม = ใช้ไม่ได้
PINCH_LIMIT = 0.5  # นิ้วใกล้จีบเกินสัดส่วนนี้ของเฟรม = ชนกับการคลิก
NEUTRAL_LIMIT = 0.15  # มือปกติเข้าเกณฑ์ของท่านี้เกินสัดส่วนนี้ = คล้ายมือปกติเกินไป
SIMILAR_LIMIT = 0.3  # ตัวอย่างเข้าเกณฑ์ของอีกท่าเกินสัดส่วนนี้ = คล้ายท่านั้นเกินไป

# ท่าสำเร็จรูปที่แอปใช้อยู่แล้ว → คำอธิบายในข้อความเตือน
GESTURE_ROLES = {
    "Victory": "✌️ (สกรอล)",
    "Open_Palm": "🖐 (โหมดปัด)",
    "Closed_Fist": "✊ (ย่อ/หยิบหน้าต่าง)",
    "ILoveYou": "🤟 (ARM/DISARM)",
    "Thumb_Up": "👍 (เมนูเปิดแอป)",
}


class TeachError(ValueError):
    """ท่าที่อัดมาใช้ไม่ได้ (ข้อความเป็นคำแนะนำสำหรับผู้ใช้)"""


class TeachStep(enum.Enum):
    NEUTRAL = "NEUTRAL"
    POSE = "POSE"
    DONE = "DONE"


STEP_TEXT: dict[TeachStep, tuple[str, str]] = {
    TeachStep.NEUTRAL: (
        "1/2 · มือปกติ",
        "ใช้มือข้างถนัดแบบที่ใช้ชี้ตามปกติ ขยับไปมาช้าๆ และลองจีบนิ้วเบาๆ สักสองสามครั้ง",
    ),
    TeachStep.POSE: (
        "2/2 · ท่าใหม่",
        "ทำท่าที่จะสอนค้างไว้ แล้วเอียง/ขยับมือเล็กน้อยระหว่างวัด (ให้จำได้หลายมุม)",
    ),
    TeachStep.DONE: ("เสร็จแล้ว", 'จำท่าได้แล้ว — กด "บันทึก" เพื่อเริ่มใช้'),
}


@dataclass(frozen=True, slots=True)
class _Frame:
    features: Vector
    orientation: float
    canned: str | None  # ท่าสำเร็จรูปที่ score ผ่าน (None = ไม่มี)
    near_pinch: bool


@dataclass(frozen=True, slots=True)
class TeachSummary:
    samples: int
    threshold: float
    neutral_margin: float  # ระยะจากมือปกติที่ใกล้ที่สุด ÷ เกณฑ์ (มาก = แยกจากมือปกติได้ชัด)


def _near_pinch(hand: Hand, width: int, height: int, pinch_on: float) -> bool:
    """นิ้วชี้หรือนิ้วกลางใกล้จะจีบกับโป้ง (นิ้วที่งอเข้าฝ่ามือไม่นับ เหมือนใน engine)"""
    limit = pinch_on * PINCH_LOCK_FACTOR
    for finger, tip in ((Finger.INDEX, LM.INDEX_TIP), (Finger.MIDDLE, LM.MIDDLE_TIP)):
        if finger_curled(hand, finger, width, height):
            continue
        if pinch_ratio(hand, width, height, tip) < limit:
            return True
    return False


def check_pose(
    template: PoseTemplate,
    frames_canned: list[str | None],
    frames_near_pinch: list[bool],
    library: PoseLibrary,
    negatives: tuple[Vector, ...],
) -> TeachSummary:
    """ตรวจว่าท่าที่อัดมาใช้ได้ (raise :class:`TeachError` พร้อมเหตุผลถ้าไม่ได้)"""
    total = max(len(frames_canned), 1)
    counts = Counter(c for c in frames_canned if c is not None)
    if counts:
        gesture, seen = counts.most_common(1)[0]
        if seen / total >= RESERVED_LIMIT:
            role = GESTURE_ROLES.get(gesture, gesture)
            raise TeachError(f"โมเดลเห็นท่านี้เป็น {role} ซึ่งแอปใช้อยู่แล้ว — เลือกท่าอื่น")
    if sum(frames_near_pinch) / max(len(frames_near_pinch), 1) >= PINCH_LIMIT:
        raise TeachError("นิ้วโป้งชิดนิ้วชี้หรือนิ้วกลางเกินไป จะชนกับการคลิก — เลือกท่าที่นิ้วโป้งไม่แตะสองนิ้วนี้")

    neutral = [knn_distance(template.samples, n) for n in negatives]
    if neutral and sum(d <= template.threshold for d in neutral) / len(neutral) > NEUTRAL_LIMIT:
        raise TeachError("ท่านี้คล้ายมือตอนชี้ปกติเกินไป (จะติดเองระหว่างใช้งาน) — เลือกท่าที่ต่างกว่านี้")

    for other in library.poses:
        if other.name == template.name:
            continue  # สอนท่าเดิมใหม่ = แทนที่
        if angle_difference(other.orientation, template.orientation) > ORIENTATION_TOLERANCE:
            continue  # หันคนละทิศ แยกกันได้อยู่แล้ว
        ours = [knn_distance(other.samples, s) <= other.threshold for s in template.samples]
        theirs = [knn_distance(template.samples, s) <= template.threshold for s in other.samples]
        if max(sum(ours) / len(ours), sum(theirs) / len(theirs)) > SIMILAR_LIMIT:
            raise TeachError(f'ท่านี้คล้ายท่า "{other.name}" ที่สอนไว้แล้วเกินไป — เลือกท่าอื่น หรือลบท่านั้นก่อน')

    margin = min(neutral) / template.threshold if neutral else float("inf")
    return TeachSummary(len(template.samples), template.threshold, margin)


class TeachSession:
    """State machine ของการสอนหนึ่งท่า — เรียก :meth:`update` ทุกเฟรมที่ได้จาก tracker

    เวลาเดินเฉพาะตอนเห็นมือข้างถนัด (เอามือออก = หยุดรอ) เหมือน calibration
    """

    def __init__(self, config: Config, library: PoseLibrary, name: str, action: str) -> None:
        self._config = config
        self._library = library
        self._name = validate_pose_name(name)
        parse_action(action)  # ปุ่มลัดผิด = ValueError ตั้งแต่ก่อนเริ่มอัด
        self._action = action.strip()
        if library.get(self._name) is None and len(library) >= MAX_POSES:
            raise ValueError(f"สอนได้ไม่เกิน {MAX_POSES} ท่า — ลบท่าเก่าก่อน")
        self._reserved = reserved_gestures(config)
        self._dominant = config.tracking.dominant_hand
        self._neutral: list[Vector] = []
        self._frames: list[_Frame] = []
        self._step = TeachStep.NEUTRAL
        self._elapsed = 0
        self._last_ms: int | None = None
        self.error: str | None = None
        self.hand_visible = False
        self.result: PoseLibrary | None = None
        self.summary: TeachSummary | None = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def action(self) -> str:
        return self._action

    @property
    def step(self) -> TeachStep:
        return self._step

    @property
    def done(self) -> bool:
        return self._step is TeachStep.DONE

    @property
    def title(self) -> str:
        return STEP_TEXT[self._step][0]

    @property
    def instruction(self) -> str:
        return STEP_TEXT[self._step][1]

    @property
    def sampling(self) -> bool:
        return not self.done and self._elapsed >= SETTLE_MS

    @property
    def progress(self) -> float:
        if self.done:
            return 1.0
        duration = SAMPLE_MS[self._step.value]
        return min(max((self._elapsed - SETTLE_MS) / duration, 0.0), 1.0)

    @property
    def countdown_ms(self) -> int:
        return max(0, SETTLE_MS - self._elapsed)

    def restart(self) -> None:
        self.__init__(self._config, self._library, self._name, self._action)  # type: ignore[misc]

    def update(self, frame: HandFrame) -> None:
        if self.done:
            return
        hand = frame.hand(self._dominant)
        now = frame.timestamp_ms
        dt = 0 if self._last_ms is None else max(0, min(now - self._last_ms, 200))
        self._last_ms = now
        measured = None if hand is None else pose_features(hand, frame.width, frame.height)
        self.hand_visible = measured is not None
        if hand is None or measured is None:
            return
        was_sampling = self.sampling
        self._elapsed += dt
        if not self.sampling:
            return
        if not was_sampling:
            self.error = None
        if self._step is TeachStep.NEUTRAL:
            self._neutral.append(measured[0])
        else:
            tracking = self._config.tracking
            canned = (
                hand.gesture
                if hand.gesture in self._reserved
                and hand.gesture_score >= tracking.min_gesture_score
                else None
            )
            near = _near_pinch(hand, frame.width, frame.height, self._config.pinch.on)
            self._frames.append(_Frame(measured[0], measured[1], canned, near))
        if self.progress >= 1.0:
            self._finish_step()

    # ---- internal ------------------------------------------------------- #

    def _retry(self, message: str) -> None:
        self.error = message
        self._elapsed = 0
        if self._step is TeachStep.NEUTRAL:
            self._neutral.clear()
        else:
            self._frames.clear()

    def _finish_step(self) -> None:
        if self._step is TeachStep.NEUTRAL:
            if len(self._neutral) < MIN_SAMPLES:
                self._retry("เห็นมือน้อยเกินไป — ลองอีกครั้ง")
                return
            self._step = TeachStep.POSE
            self._elapsed = 0
            return
        if len(self._frames) < MIN_SAMPLES:
            self._retry("เห็นมือน้อยเกินไป — ลองอีกครั้ง")
            return
        template = make_template(
            self._name, self._action, [(f.features, f.orientation) for f in self._frames]
        )
        negatives = (*self._library.negatives, *spread(self._neutral, NEUTRAL_SAMPLES))[
            -MAX_NEGATIVES:
        ]
        try:
            self.summary = check_pose(
                template,
                [f.canned for f in self._frames],
                [f.near_pinch for f in self._frames],
                self._library,
                tuple(negatives),
            )
        except TeachError as exc:
            self._retry(str(exc))
            return
        self.result = self._library.with_pose(template).with_negatives(
            spread(self._neutral, NEUTRAL_SAMPLES)
        )
        self._step = TeachStep.DONE
        self._elapsed = 0
