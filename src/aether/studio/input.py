"""แปลง :class:`HandFrame` เป็นข้อมูลที่ Studio ใช้ (pure — ไม่มี Qt)

- ตำแหน่ง = จุดโคนนิ้ว (:func:`~aether.geometry.pointer_anchor`) ในกรอบ ``region`` ของเฟรมกล้อง → 0–1 ของผืนผ้าใบ
  กรองสั่นด้วย One Euro และมีความเร็ว (หน่วยผืนผ้าใบต่อวินาที) ไว้ใช้โยนของ / ปัดลูกบอล
- จีบ = ระยะโป้ง↔ชี้ หาร ขนาดมือ มี hysteresis (``pinch_on`` / ``pinch_off`` เดียวกับ config)
- :class:`HoldDetector` = "ค้างท่านี้ N วินาที" ยิงครั้งเดียวต่อการค้างหนึ่งครั้ง พร้อม progress ไว้วาดวงแหวน
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..filters import OneEuroFilter2D
from ..geometry import Finger, finger_extended, hand_scale, palm_center, pinch_ratio, pointer_anchor
from ..tracker import Hand, HandFrame

__all__ = [
    "DEFAULT_REGION",
    "HoldDetector",
    "StudioHand",
    "StudioInput",
    "SwipeDetector",
    "mouse_hand",
]

DEFAULT_REGION = (0.15, 0.10, 0.85, 0.70)  # เท่ากับ pointer.active_region ค่าเริ่มต้น
FINGERS = (Finger.THUMB, Finger.INDEX, Finger.MIDDLE, Finger.RING, Finger.PINKY)
VELOCITY_SMOOTHING = 0.35  # สัดส่วนของค่าใหม่ใน EMA ของความเร็ว
MAX_GAP = 0.25  # มือหายไปนานกว่านี้ (วินาที) = เริ่มใหม่ (ไม่คำนวณความเร็วข้ามช่วงที่หาย)


@dataclass(frozen=True, slots=True)
class StudioHand:
    handedness: str  # "left" / "right" / "mouse"
    x: float  # 0–1 ของผืนผ้าใบ (ซ้าย → ขวา)
    y: float  # 0–1 (บน → ล่าง)
    pinch: bool
    pinch_level: float = 0.0  # 0 = นิ้วห่าง … 1 = ถึงเกณฑ์จีบ
    gesture: str = "None"  # ท่าสำเร็จรูปของ MediaPipe ("Closed_Fist", "Victory", …)
    fingers: tuple[bool, ...] = (False,) * 5  # เหยียดอยู่: โป้ง ชี้ กลาง นาง ก้อย
    vx: float = 0.0  # หน่วยผืนผ้าใบต่อวินาที
    vy: float = 0.0
    palm: tuple[float, float] = (0.5, 0.5)  # กลางฝ่ามือบนผืนผ้าใบ
    size: float = 0.0  # ขนาดมือ ÷ ความกว้างเฟรม (ใกล้กล้อง = ใหญ่)

    @property
    def finger_count(self) -> int:
        return sum(self.fingers)

    @property
    def open_palm(self) -> bool:
        """แบมือ (ใช้ผลักของ): MediaPipe บอกว่าแบ หรือเหยียดครบ 4 นิ้วและไม่ได้จีบ"""
        return not self.pinch and (self.gesture == "Open_Palm" or all(self.fingers[1:]))


def mouse_hand(x: float, y: float, pressed: bool, vx: float = 0.0, vy: float = 0.0) -> StudioHand:
    """มือจำลองจากเมาส์จริง (ใช้ Studio ได้แม้ไม่มีกล้อง): กดปุ่มซ้าย = จีบ"""
    return StudioHand(
        "mouse",
        x,
        y,
        pressed,
        1.0 if pressed else 0.0,
        fingers=(True,) * 5,
        vx=vx,
        vy=vy,
        palm=(x, y),
    )


@dataclass(slots=True)
class _Track:
    filter: OneEuroFilter2D
    pinched: bool = False
    last: tuple[float, float, float] | None = None  # x, y, t
    vx: float = 0.0
    vy: float = 0.0


class StudioInput:
    def __init__(
        self,
        region: tuple[float, float, float, float] = DEFAULT_REGION,
        *,
        pinch_on: float = 0.25,
        pinch_off: float = 0.38,
        min_cutoff: float = 1.2,
        beta: float = 0.03,
    ) -> None:
        self.region = region
        self.pinch_on = pinch_on
        self.pinch_off = pinch_off
        self._filter_params = (min_cutoff, beta)
        self._tracks: dict[str, _Track] = {}
        self.hands: tuple[StudioHand, ...] = ()

    def to_canvas(self, x: float, y: float) -> tuple[float, float]:
        x0, y0, x1, y1 = self.region
        return (
            min(max((x - x0) / (x1 - x0), 0.0), 1.0),
            min(max((y - y0) / (y1 - y0), 0.0), 1.0),
        )

    def update(self, frame: HandFrame) -> tuple[StudioHand, ...]:
        """มือทั้งหมดในเฟรม (มือขวาก่อน) — มือที่หายไปถูกลืม (จีบค้างไม่ได้)"""
        t = frame.timestamp_ms / 1000.0
        seen: set[str] = set()
        hands: list[StudioHand] = []
        for hand in sorted(
            frame.hands, key=lambda h: (h.handedness != "right", -h.handedness_score)
        ):
            if hand.handedness in seen:
                continue
            seen.add(hand.handedness)
            hands.append(self._hand(hand, frame.width, frame.height, t))
        for name in list(self._tracks):
            if name not in seen:
                del self._tracks[name]
        self.hands = tuple(hands)
        return self.hands

    def _hand(self, hand: Hand, width: int, height: int, t: float) -> StudioHand:
        track = self._tracks.get(hand.handedness)
        if track is None or (track.last is not None and t - track.last[2] > MAX_GAP):
            track = _Track(OneEuroFilter2D(*self._filter_params))
            self._tracks[hand.handedness] = track
        x, y = track.filter(*self.to_canvas(*pointer_anchor(hand)), t)
        if track.last is not None and t > track.last[2]:
            dt = t - track.last[2]
            vx, vy = (x - track.last[0]) / dt, (y - track.last[1]) / dt
            track.vx += VELOCITY_SMOOTHING * (vx - track.vx)
            track.vy += VELOCITY_SMOOTHING * (vy - track.vy)
        track.last = (x, y, t)

        ratio = pinch_ratio(hand, width, height)
        if track.pinched:
            track.pinched = ratio < self.pinch_off
        else:
            track.pinched = ratio < self.pinch_on
        span = self.pinch_off * 1.6 - self.pinch_on
        level = 1.0 if track.pinched else min(max((self.pinch_off * 1.6 - ratio) / span, 0.0), 1.0)
        fingers = tuple(finger_extended(hand, f, width, height) for f in FINGERS)
        size = hand_scale(hand, width, height) / max(width, 1)
        return StudioHand(
            hand.handedness,
            x,
            y,
            track.pinched,
            level if math.isfinite(level) else 0.0,
            hand.gesture,
            fingers,
            track.vx,
            track.vy,
            self.to_canvas(*palm_center(hand)),
            size,
        )


@dataclass(slots=True)
class HoldDetector:
    """ค้างเงื่อนไขไว้ ``seconds`` วินาทีแล้ว :meth:`update` คืน ``True`` หนึ่งครั้ง (ต้องปล่อยก่อนจึงยิงอีกได้)"""

    seconds: float
    _since: float | None = field(default=None, init=False)
    _fired: bool = field(default=False, init=False)
    progress: float = field(default=0.0, init=False)

    def update(self, active: bool, now: float) -> bool:
        if not active:
            self._since, self._fired, self.progress = None, False, 0.0
            return False
        if self._since is None:
            self._since = now
        if self._fired:
            self.progress = 0.0
            return False
        self.progress = min((now - self._since) / self.seconds, 1.0)
        if self.progress >= 1.0:
            self._fired = True
            self.progress = 0.0
            return True
        return False

    def reset(self) -> None:
        self._since, self._fired, self.progress = None, False, 0.0


class SwipeDetector:
    """แบมือแล้วสะบัดซ้าย/ขวา → ``-1`` (ซ้าย) / ``+1`` (ขวา) หนึ่งครั้งต่อการสะบัด (มี cooldown)

    ไม่นับมือจำลองจากเมาส์ (ขยับเมาส์เร็วๆ ไม่ควรเปลี่ยนหน้า — เมาส์ใช้ปุ่ม/ลูกศรแทน) และไม่นับตอนจีบ
    """

    def __init__(
        self, speed: float = 1.6, *, cooldown: float = 0.7, axis_ratio: float = 1.5
    ) -> None:
        self.speed = speed  # ความกว้างผืนผ้าใบต่อวินาที
        self.cooldown = cooldown
        self.axis_ratio = axis_ratio
        self._until = 0.0

    def update(self, hand: StudioHand | None, now: float) -> int:
        if hand is None or hand.handedness == "mouse" or not hand.open_palm or now < self._until:
            return 0
        if abs(hand.vx) < self.speed or abs(hand.vx) < self.axis_ratio * abs(hand.vy):
            return 0
        self._until = now + self.cooldown
        return 1 if hand.vx > 0 else -1
