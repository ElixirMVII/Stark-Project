"""ครอบ MediaPipe ``GestureRecognizer`` (Tasks API) → :class:`HandFrame`

ส่วนอื่นของแอปเห็นแค่ dataclass ในไฟล์นี้ ไม่มี type ของ mediapipe หลุดออกไป
และ mediapipe ถูก import ตอนสร้าง :class:`HandTracker` เท่านั้น ส่วนที่เป็น pure logic
(dataclass, การแปลงผลลัพธ์, serialization) จึง test ได้โดยไม่ต้องมีกล้องหรือโมเดล

เรื่อง handedness (ทดสอบกับ mediapipe 1.2.3 ดู ``docs/DECISIONS.md``):
โมเดลให้ label ตรงกับมือจริงเมื่อภาพ **ไม่ถูก flip** ถ้าเรา flip ภาพแบบกระจกก่อนส่งเข้าโมเดล
label จะสลับข้าง จึงต้องสลับกลับใน :func:`user_handedness`
"""

from __future__ import annotations

import enum
import functools
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple, Self

if TYPE_CHECKING:
    import numpy as np

__all__ = [
    "CANNED_GESTURES",
    "GESTURE_NONE",
    "LANDMARK_COUNT",
    "LM",
    "Hand",
    "HandFrame",
    "HandTracker",
    "Landmark",
    "ModelNotFoundError",
    "hand_connections",
    "hand_frame_from_result",
    "user_handedness",
]

LANDMARK_COUNT = 21
GESTURE_NONE = "None"

# ท่าสำเร็จรูปของโมเดล gesture_recognizer.task
CANNED_GESTURES = (
    "None",
    "Closed_Fist",
    "Open_Palm",
    "Pointing_Up",
    "Thumb_Down",
    "Thumb_Up",
    "Victory",
    "ILoveYou",
)


class LM(enum.IntEnum):
    """index ของ landmark ทั้ง 21 จุด"""

    WRIST = 0
    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4
    INDEX_MCP = 5
    INDEX_PIP = 6
    INDEX_DIP = 7
    INDEX_TIP = 8
    MIDDLE_MCP = 9
    MIDDLE_PIP = 10
    MIDDLE_DIP = 11
    MIDDLE_TIP = 12
    RING_MCP = 13
    RING_PIP = 14
    RING_DIP = 15
    RING_TIP = 16
    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20


# ใช้เมื่อ mediapipe ไม่มี HandLandmarksConnections ให้
_FALLBACK_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1), (0, 5), (9, 13), (13, 17), (5, 9), (0, 17),  # ฝ่ามือ
    (1, 2), (2, 3), (3, 4),  # โป้ง
    (5, 6), (6, 7), (7, 8),  # ชี้
    (9, 10), (10, 11), (11, 12),  # กลาง
    (13, 14), (14, 15), (15, 16),  # นาง
    (17, 18), (18, 19), (19, 20),  # ก้อย
)  # fmt: skip


class ModelNotFoundError(FileNotFoundError):
    """ไม่พบไฟล์โมเดล ``gesture_recognizer.task``"""


class Landmark(NamedTuple):
    x: float
    y: float
    z: float


@dataclass(frozen=True, slots=True)
class Hand:
    """มือหนึ่งข้างในหนึ่งเฟรม

    - ``handedness``: ``"left"`` | ``"right"`` คือมือจริงของผู้ใช้
    - ``landmarks``: 21 จุด, x/y เป็น normalized 0–1 ของเฟรม (ไม่ใช่สี่เหลี่ยมจัตุรัส
      ต้องคูณ width/height ก่อนวัดระยะ), z เป็นความลึกสัมพัทธ์กับข้อมือ
    - ``world_landmarks``: 21 จุดในหน่วยเมตร จุดกำเนิดอยู่ราวกลางมือ
    """

    handedness: str
    handedness_score: float
    gesture: str
    gesture_score: float
    landmarks: tuple[Landmark, ...]
    world_landmarks: tuple[Landmark, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "handedness": self.handedness,
            "handedness_score": self.handedness_score,
            "gesture": self.gesture,
            "gesture_score": self.gesture_score,
            "landmarks": [list(p) for p in self.landmarks],
            "world_landmarks": [list(p) for p in self.world_landmarks],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Hand:
        return cls(
            handedness=data["handedness"],
            handedness_score=float(data["handedness_score"]),
            gesture=data["gesture"],
            gesture_score=float(data["gesture_score"]),
            landmarks=tuple(Landmark(*p) for p in data["landmarks"]),
            world_landmarks=tuple(Landmark(*p) for p in data.get("world_landmarks", ())),
        )


@dataclass(frozen=True, slots=True)
class HandFrame:
    """ผลการจับมือของหนึ่งเฟรม (``width``/``height`` คือขนาดเฟรมกล้องเป็น px)"""

    timestamp_ms: int
    width: int
    height: int
    hands: tuple[Hand, ...] = ()

    def hand(self, handedness: str) -> Hand | None:
        """มือข้างที่ระบุ (ถ้าเจอข้างเดียวกันสองมือ เอาอันที่มั่นใจกว่า)"""
        matches = [h for h in self.hands if h.handedness == handedness]
        if not matches:
            return None
        return max(matches, key=lambda h: h.handedness_score)

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp_ms": self.timestamp_ms,
            "width": self.width,
            "height": self.height,
            "hands": [h.to_dict() for h in self.hands],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HandFrame:
        return cls(
            timestamp_ms=int(data["timestamp_ms"]),
            width=int(data["width"]),
            height=int(data["height"]),
            hands=tuple(Hand.from_dict(h) for h in data.get("hands", ())),
        )


def user_handedness(model_label: str, *, mirrored: bool) -> str:
    """แปลง label ของโมเดล (``"Left"``/``"Right"``) เป็นมือจริงของผู้ใช้

    โมเดลตอบถูกข้างเมื่อภาพไม่ถูก flip ถ้าภาพถูก flip แบบกระจก (``mirrored=True``)
    ต้องสลับข้าง
    """
    label = model_label.strip().lower()
    if label not in ("left", "right"):
        raise ValueError(f"handedness ที่ไม่รู้จัก: {model_label!r}")
    if mirrored:
        return "left" if label == "right" else "right"
    return label


def _points(landmarks: Any) -> tuple[Landmark, ...]:
    return tuple(Landmark(float(p.x), float(p.y), float(p.z)) for p in landmarks)


def hand_frame_from_result(
    result: Any, *, width: int, height: int, timestamp_ms: int, mirrored: bool
) -> HandFrame:
    """แปลง ``GestureRecognizerResult`` เป็น :class:`HandFrame`

    รับ object อะไรก็ได้ที่มี ``handedness``, ``gestures``, ``hand_landmarks``,
    ``hand_world_landmarks`` (list ต่อมือ) จึง test ด้วย object ปลอมได้
    """
    hands: list[Hand] = []
    world = getattr(result, "hand_world_landmarks", None) or []
    gestures = getattr(result, "gestures", None) or []
    for i, landmarks in enumerate(result.hand_landmarks):
        if len(landmarks) != LANDMARK_COUNT:
            continue
        side = result.handedness[i][0]
        top = gestures[i][0] if i < len(gestures) and gestures[i] else None
        hands.append(
            Hand(
                handedness=user_handedness(side.category_name, mirrored=mirrored),
                handedness_score=float(side.score),
                gesture=(top.category_name or GESTURE_NONE) if top else GESTURE_NONE,
                gesture_score=float(top.score) if top else 0.0,
                landmarks=_points(landmarks),
                world_landmarks=_points(world[i]) if i < len(world) else (),
            )
        )
    return HandFrame(timestamp_ms=timestamp_ms, width=width, height=height, hands=tuple(hands))


@functools.cache
def hand_connections() -> tuple[tuple[int, int], ...]:
    """คู่ index ของ landmark ที่ต้องลากเส้นเชื่อม (ใช้ของ mediapipe ถ้ามี)"""
    try:
        from mediapipe.tasks.python.vision import HandLandmarksConnections

        return tuple((c.start, c.end) for c in HandLandmarksConnections.HAND_CONNECTIONS)
    except (ImportError, AttributeError):  # ไม่มี mediapipe หรือ API เปลี่ยน → ใช้ที่ hardcode
        return _FALLBACK_CONNECTIONS


class HandTracker:
    """GestureRecognizer ใน running mode VIDEO

    เรียก :meth:`process` จาก thread เดียวเท่านั้น (worker thread ของกล้อง)

    ``mirrored`` บอกว่าเฟรมที่ส่งเข้ามาถูก flip แนวนอนมาแล้วหรือไม่ (tracker ไม่ flip เอง)
    """

    def __init__(self, model_path: Path, *, num_hands: int = 2, mirrored: bool = True) -> None:
        model_path = Path(model_path)
        if not model_path.is_file():
            raise ModelNotFoundError(str(model_path))

        import cv2
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        self._cv2 = cv2
        self._mp = mp
        self._mirrored = mirrored
        self._last_timestamp_ms = -1
        # ส่งโมเดลเป็น bytes แทน path เพราะ path ที่มีอักษรไทย (เช่นชื่อ user) อาจเปิดไม่ได้
        # จากฝั่ง C ของ mediapipe บน Windows
        options = vision.GestureRecognizerOptions(
            base_options=BaseOptions(model_asset_buffer=model_path.read_bytes()),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=num_hands,
        )
        self._recognizer = vision.GestureRecognizer.create_from_options(options)

    def process(self, frame_bgr: np.ndarray, timestamp_ms: int) -> HandFrame:
        """จับมือในเฟรม BGR หนึ่งเฟรม

        MediaPipe บังคับให้ timestamp เพิ่มขึ้นเสมอ ถ้าได้ค่าที่ไม่เพิ่ม จะขยับให้ +1ms
        """
        if self._recognizer is None:
            raise RuntimeError("HandTracker ถูกปิดไปแล้ว")
        timestamp_ms = max(int(timestamp_ms), self._last_timestamp_ms + 1)
        self._last_timestamp_ms = timestamp_ms

        height, width = frame_bgr.shape[:2]
        rgb = self._cv2.cvtColor(frame_bgr, self._cv2.COLOR_BGR2RGB)
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._recognizer.recognize_for_video(image, timestamp_ms)
        return hand_frame_from_result(
            result,
            width=width,
            height=height,
            timestamp_ms=timestamp_ms,
            mirrored=self._mirrored,
        )

    def close(self) -> None:
        if self._recognizer is not None:
            self._recognizer.close()
            self._recognizer = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
