"""Gesture state machine: ``HandFrame`` → ``(State, list[GestureEvent])``

Pure logic ล้วน — ไม่แตะ OS ไม่แตะเวลาจริง (ใช้ ``timestamp_ms`` ของเฟรม) จึง test ได้ด้วยเฟรมสังเคราะห์
ผู้ที่ทำตาม event คือ :mod:`aether.actions`

สถานะ (M1)::

    DISARMED ──(arm)──► IDLE ◄──► POINTER ──► PINCH_PENDING ──► DRAGGING
                                     │  ▲          │ ปล่อยก่อน drag_hold_ms = CLICK
                                     ▼  │
                                   SCROLL (✌️)

กติกาความปลอดภัยที่ engine รับผิดชอบ:

- ``DISARMED``: ไม่ออก event ที่เป็น input ของ OS เลย
- มือหายเกิน ``LOST_RELEASE_MS`` ระหว่างลาก → ออก ``BUTTON_UP`` ทันที
- disarm ระหว่างลาก → ออก ``BUTTON_UP`` ก่อน
"""

from __future__ import annotations

import dataclasses
import enum
import math
from collections import Counter, deque
from dataclasses import dataclass

from .config import Config
from .filters import OneEuroFilter2D
from .geometry import Finger, finger_curled, palm_center, pinch_ratio, pointer_anchor
from .mapping import PointerTracker, ScreenGeometry, map_to_screen
from .tracker import GESTURE_NONE, LM, Hand, HandFrame

__all__ = [
    "DEFAULT_DOUBLE_CLICK_MS",
    "DRAG_RELEASE_CONFIRM_MS",
    "LOST_RELEASE_MS",
    "MIN_PINCH_MS",
    "EngineSnapshot",
    "EventKind",
    "GestureEngine",
    "GestureEvent",
    "GestureVoter",
    "State",
]

LOST_RELEASE_MS = 300  # มือหายนานเท่านี้ถือว่าหลุด: ปล่อยปุ่มเมาส์ กลับ IDLE
MIN_PINCH_MS = 40  # การจีบที่สั้นกว่านี้ (เห็นแค่เฟรมเดียว) ถือเป็น noise ไม่นับเป็นคลิก
DRAG_RELEASE_CONFIRM_MS = 30  # ระหว่างลาก ต้องเห็นนิ้วห่างต่อเนื่องเท่านี้จึงปล่อย (กันของหลุดมือ)
PINCH_LOCK_FACTOR = 1.4  # pinch_ratio < pinch.on × ค่านี้ = "ใกล้จะจีบ" → ลด gain
SCROLL_NOTCHES_PER_UNIT = 30.0  # ขยับมือเต็มความสูงเฟรม = กี่ notch (ที่ scroll.speed = 1)
SCROLL_AXIS_UNLOCK_MS = 300  # หยุดสกรอลนานเท่านี้จึงเลือกแกนใหม่ได้
DEFAULT_DOUBLE_CLICK_MS = 500  # ค่าเริ่มต้นของ Windows (แอปจะส่งค่าจริงของ OS เข้ามา)

GESTURE_SCROLL = "Victory"


class State(enum.Enum):
    DISARMED = "DISARMED"
    IDLE = "IDLE"  # arm แล้วแต่ไม่เห็นมือข้างถนัด
    POINTER = "POINTER"
    PINCH_PENDING = "PINCH_PENDING"  # จีบอยู่ ยังไม่รู้ว่าจะเป็นคลิกหรือลาก
    DRAGGING = "DRAGGING"
    SCROLL = "SCROLL"


class EventKind(enum.Enum):
    MOVE = "move"
    BUTTON_DOWN = "button_down"
    BUTTON_UP = "button_up"
    CLICK = "click"
    SCROLL = "scroll"
    ARMED = "armed"  # แจ้งสถานะเท่านั้น ไม่ใช่ input ของ OS
    DISARMED = "disarmed"


@dataclass(frozen=True, slots=True)
class GestureEvent:
    """สิ่งที่ต้องทำกับ OS (พิกัดเป็น physical px)

    - ``CLICK`` ที่ ``count == 2`` คือคลิกที่สองของดับเบิลคลิก ``x, y`` เป็นตำแหน่งของคลิกแรก
      เพื่อให้ OS นับเป็นดับเบิลคลิก
    - ``SCROLL``: ``dy > 0`` = เลื่อนขึ้น, ``dx > 0`` = เลื่อนขวา หน่วยเป็น notch ของล้อเมาส์
    """

    kind: EventKind
    x: int = 0
    y: int = 0
    button: str = "left"
    count: int = 1
    dx: int = 0
    dy: int = 0
    reason: str = ""


@dataclass(frozen=True, slots=True)
class EngineSnapshot:
    """สถานะล่าสุดสำหรับ HUD / debug view (immutable ส่งข้าม thread ได้)"""

    state: State = State.DISARMED
    armed: bool = False
    hand_present: bool = False
    hands: int = 0
    cursor: tuple[int, int] | None = None  # physical px (ตอน DISARMED คือจุดที่เคอร์เซอร์ *จะ* อยู่)
    gesture: str = GESTURE_NONE  # ท่าสำเร็จรูปหลัง majority vote
    pinch_ratio: float | None = None  # โป้ง↔ชี้ (ค่าดิบ)
    pinch_level: float = 0.0  # 0 = นิ้วห่าง (≥ pinch.off) … 1 = จีบแล้ว (≤ pinch.on)
    pinch_button: str | None = None  # ปุ่มของการจีบที่กำลังเกิด
    hold_progress: float = 0.0  # 0→1 ระหว่างรอให้การจีบกลายเป็นการลาก


class GestureVoter:
    """Majority vote ของท่าสำเร็จรูปใน ``window`` เฟรมล่าสุด (ท่าที่ score ต่ำนับเป็น ``None``)

    ช่วงที่ยังเห็นไม่ครบ ``window`` เฟรม (เพิ่งเริ่ม/เพิ่ง reset) เฟรมที่ขาดนับเป็น ``None``
    ท่าใหม่จึงต้องถูกเห็นเกินครึ่งของ window เสมอ เฟรมเดียวตัดสินไม่ได้
    """

    def __init__(self, window: int, min_score: float) -> None:
        self._history: deque[str] = deque(maxlen=window)
        self._min_score = min_score
        self._current = GESTURE_NONE

    @property
    def current(self) -> str:
        return self._current

    def reset(self) -> None:
        self._history.clear()
        self._current = GESTURE_NONE

    def push(self, gesture: str, score: float) -> str:
        self._history.append(gesture if score >= self._min_score else GESTURE_NONE)
        counts = Counter(self._history)
        counts[GESTURE_NONE] += (self._history.maxlen or 0) - len(self._history)
        top = max(counts.values())
        leaders = [label for label, n in counts.items() if n == top]
        if self._current not in leaders:  # เสมอกัน: คงท่าเดิมไว้ถ้ายังเป็นผู้นำ ไม่งั้นเอาท่าที่เห็นล่าสุด
            self._current = next(
                (label for label in reversed(self._history) if label in leaders), GESTURE_NONE
            )
        return self._current


@dataclass(slots=True)
class _Metrics:
    gesture: str
    index_raw: float
    middle_raw: float
    index: float  # inf ถ้านิ้วงอเข้าฝ่ามือ (กำมือ ไม่ใช่จีบ)
    middle: float
    target: tuple[float, float]  # ตำแหน่งบนจอที่กรองแล้ว


class GestureEngine:
    """เรียก :meth:`update` ทีละเฟรมจาก thread เดียว (หรือครอบด้วย lock เดียวกับ :meth:`set_armed`)"""

    def __init__(
        self,
        config: Config,
        screen: ScreenGeometry,
        *,
        double_click_ms: int = DEFAULT_DOUBLE_CLICK_MS,
    ) -> None:
        self._cfg = config
        self._screen = screen
        self._double_click_ms = double_click_ms
        self._double_click_radius = max(32.0, 0.03 * screen.height)

        one_euro = config.pointer.one_euro
        self._filter = OneEuroFilter2D(one_euro.min_cutoff, one_euro.beta, one_euro.d_cutoff)
        self._pointer = PointerTracker(screen)
        self._voter = GestureVoter(config.tracking.vote_window, config.tracking.min_gesture_score)

        self._armed = False
        self._state = State.DISARMED
        self._snapshot = EngineSnapshot()
        self._now = 0  # เวลาล่าสุดที่ engine เห็น (ms) — ไม่ถอยหลังแม้เฟรมจะมาสลับลำดับกับ tick

        self._last_seen: int | None = None  # เวลาของเฟรมล่าสุดที่เห็นมือข้างถนัด
        self._last_anchor: tuple[float, float] | None = None
        self._cursor: tuple[int, int] | None = None
        self._emitted: tuple[int, int] | None = None  # ตำแหน่งล่าสุดที่สั่ง MOVE ไป
        self._last_click: tuple[int, int, int] | None = None  # (เวลา, x, y) ของคลิกซ้ายเดี่ยวล่าสุด

        # นิ้วต้อง "ห่างเกิน pinch.off" ให้เห็นก่อนหนึ่งครั้ง จึงจะเริ่มจีบได้
        # (กันคลิกเองตอน arm หรือตอนมือเพิ่งเข้าเฟรมในท่านิ้วชิดกันอยู่แล้ว)
        self._ready = {"left": False, "right": False}
        self._pinch_button: str | None = None
        self._pinch_t0 = 0
        self._pinch_fired = False
        self._release_since: int | None = None

        self._scroll_ref = (0.0, 0.0)
        self._scroll_axis: str | None = None
        self._scroll_last_output = 0
        self._scroll_remainder = 0.0

    # ---- public --------------------------------------------------------- #

    @property
    def armed(self) -> bool:
        return self._armed

    @property
    def state(self) -> State:
        return self._state

    @property
    def snapshot(self) -> EngineSnapshot:
        return self._snapshot

    def set_armed(self, armed: bool) -> list[GestureEvent]:
        """เปลี่ยนสถานะ ARM/DISARM คืน event ที่ต้องทำ (เช่นปล่อยปุ่มที่กดค้าง)"""
        if armed == self._armed:
            return []
        events: list[GestureEvent] = []
        if armed:
            self._armed = True
            self._state = State.IDLE
            events.append(GestureEvent(EventKind.ARMED))
        else:
            if self._state is State.DRAGGING:
                events.append(self._button_up("disarm"))
            self._armed = False
            self._state = State.DISARMED
            events.append(GestureEvent(EventKind.DISARMED))
        self._reset_tracking()
        self._snapshot = dataclasses.replace(
            self._snapshot,
            state=self._state,
            armed=self._armed,
            pinch_button=None,
            pinch_level=0.0,
            hold_progress=0.0,
        )
        return events

    def update(self, frame: HandFrame) -> tuple[State, list[GestureEvent]]:
        """ประมวลผลหนึ่งเฟรม คืน (สถานะใหม่, event ที่ต้องทำตามลำดับ)"""
        now = self._now = max(frame.timestamp_ms, self._now)
        events: list[GestureEvent] = []
        hand = self._select_hand(frame)
        metrics: _Metrics | None = None
        if hand is None:
            self._hand_missing(now, events)
        else:
            metrics = self._measure(hand, frame, now)
            self._hand_present(hand, metrics, now, events)
        self._snapshot = self._make_snapshot(frame, metrics, now)
        return self._state, events

    def tick(self, now_ms: int) -> list[GestureEvent]:
        """ไม่มีเฟรมเข้ามา (กล้องค้าง/หลุด): เดินเวลาเหมือนได้เฟรมที่ไม่มีมือ เพื่อให้กติกา "มือหาย" ทำงาน"""
        return self.update(HandFrame(now_ms, 1, 1))[1]

    # ---- hand tracking -------------------------------------------------- #

    def _select_hand(self, frame: HandFrame) -> Hand | None:
        """มือข้างถนัด ถ้าเจอข้างเดียวกันสองมือ เลือกมือที่อยู่ใกล้ตำแหน่งเดิม (ไม่ให้เคอร์เซอร์กระโดด)"""
        side = self._cfg.tracking.dominant_hand
        candidates = [h for h in frame.hands if h.handedness == side]
        if not candidates:
            return None
        if len(candidates) == 1 or self._last_anchor is None:
            return max(candidates, key=lambda h: h.handedness_score)
        last = self._last_anchor
        return min(candidates, key=lambda h: math.dist(pointer_anchor(h), last))

    def _measure(self, hand: Hand, frame: HandFrame, now: int) -> _Metrics:
        width, height = frame.width, frame.height
        index_raw = pinch_ratio(hand, width, height, LM.INDEX_TIP)
        middle_raw = pinch_ratio(hand, width, height, LM.MIDDLE_TIP)
        index_curled = finger_curled(hand, Finger.INDEX, width, height)
        middle_curled = finger_curled(hand, Finger.MIDDLE, width, height)
        anchor = pointer_anchor(hand)
        self._last_anchor = anchor
        mapped = map_to_screen(*anchor, self._cfg.pointer.active_region, self._screen)
        return _Metrics(
            gesture=self._voter.push(hand.gesture, hand.gesture_score),
            index_raw=index_raw,
            middle_raw=middle_raw,
            index=math.inf if index_curled else index_raw,
            middle=math.inf if middle_curled else middle_raw,
            target=self._filter(*mapped, now / 1000.0),
        )

    def _hand_missing(self, now: int, events: list[GestureEvent]) -> None:
        if self._last_seen is None:  # ไม่ได้ตามมืออยู่
            return
        if now - self._last_seen < LOST_RELEASE_MS:
            return  # ช่วงผ่อนผัน: มือหายแวบเดียว คงสถานะเดิมไว้ ไม่ขยับ ไม่ปล่อยปุ่ม
        if self._state is State.DRAGGING:
            events.append(self._button_up("hand_lost"))
        self._reset_tracking()
        if self._armed:
            self._state = State.IDLE

    def _reset_tracking(self) -> None:
        """ลืมทุกอย่างที่ผูกกับมือที่ตามอยู่ (ตำแหน่ง ฟิลเตอร์ ท่า การจีบค้าง)"""
        self._last_seen = None
        self._last_anchor = None
        self._filter.reset()
        self._pointer.reset()
        self._voter.reset()
        self._ready = {"left": False, "right": False}
        self._end_pinch()
        self._scroll_axis = None
        self._scroll_remainder = 0.0

    # ---- state machine -------------------------------------------------- #

    def _hand_present(self, hand: Hand, m: _Metrics, now: int, events: list[GestureEvent]) -> None:
        self._last_seen = now
        pinch = self._cfg.pinch
        if m.index_raw > pinch.off:
            self._ready["left"] = True
        if m.middle_raw > pinch.off:
            self._ready["right"] = True

        if not self._armed:
            # ไม่ออก event ใดๆ แค่ตามตำแหน่งไว้ให้ HUD แสดงว่าเคอร์เซอร์จะอยู่ตรงไหน
            x, y = self._pointer.update(m.target)
            self._cursor = (round(x), round(y))
            return

        if self._state is State.IDLE:
            self._state = State.POINTER

        if self._state is State.DRAGGING:
            self._while_dragging(m, now, events)
        elif self._state is State.PINCH_PENDING:
            self._while_pinch_pending(m, now, events)
        elif self._state is State.SCROLL:
            if m.gesture == GESTURE_SCROLL:
                self._scroll(hand, now, events)
            else:
                self._state = State.POINTER
                self._pointer.rebase(m.target)  # เคอร์เซอร์ไปต่อจากจุดที่ตรึงไว้ ไม่กระโดด
        else:
            self._while_pointing(hand, m, now, events)

    def _while_pointing(
        self, hand: Hand, m: _Metrics, now: int, events: list[GestureEvent]
    ) -> None:
        if m.gesture == GESTURE_SCROLL:
            self._state = State.SCROLL
            self._scroll_ref = palm_center(hand)
            self._scroll_axis = None
            self._scroll_last_output = now
            self._scroll_remainder = 0.0
            return

        pinch = self._cfg.pinch
        near_pinch = min(m.index, m.middle) < pinch.on * PINCH_LOCK_FACTOR
        self._move(m.target, self._cfg.pointer.pinch_lock_gain if near_pinch else 1.0, events)

        left = self._ready["left"] and m.index < pinch.on
        right = self._ready["right"] and m.middle < pinch.on
        if left and right:  # สองนิ้วชิดโป้งพร้อมกัน: นิ้วที่ชิดกว่าชนะ
            left = m.index <= m.middle
            right = not left
        if left or right:
            self._pinch_button = "left" if left else "right"
            self._ready[self._pinch_button] = False
            self._pinch_t0 = now
            self._pinch_fired = False
            self._state = State.PINCH_PENDING

    def _while_pinch_pending(self, m: _Metrics, now: int, events: list[GestureEvent]) -> None:
        pinch = self._cfg.pinch
        button = self._pinch_button
        raw = m.index_raw if button == "left" else m.middle_raw
        effective = m.index if button == "left" else m.middle
        held = now - self._pinch_t0
        self._move(m.target, self._cfg.pointer.pinch_lock_gain, events)

        if raw > pinch.off:  # ปล่อยนิ้วแล้ว
            if button == "left" and held >= MIN_PINCH_MS:
                self._click(now, events)
            self._end_pinch()
            self._state = State.POINTER
        elif math.isinf(effective):  # นิ้วพับเข้าฝ่ามือ = กำลังกำมือ ไม่ใช่จีบ → ยกเลิก
            self._end_pinch()
            self._state = State.POINTER
        elif button == "left" and held >= pinch.drag_hold_ms:
            x, y = self._cursor_or_origin()
            events.append(GestureEvent(EventKind.BUTTON_DOWN, x=x, y=y, button="left"))
            self._release_since = None
            self._last_click = None
            self._state = State.DRAGGING
        elif button == "right" and not self._pinch_fired and held >= MIN_PINCH_MS:
            x, y = self._cursor_or_origin()
            events.append(GestureEvent(EventKind.CLICK, x=x, y=y, button="right"))
            self._pinch_fired = True
            self._last_click = None

    def _while_dragging(self, m: _Metrics, now: int, events: list[GestureEvent]) -> None:
        self._move(m.target, 1.0, events)
        if m.index_raw > self._cfg.pinch.off:
            if self._release_since is None:
                self._release_since = now
            if now - self._release_since >= DRAG_RELEASE_CONFIRM_MS:
                events.append(self._button_up("release"))
                self._end_pinch()
                self._state = State.POINTER
        else:
            self._release_since = None

    def _end_pinch(self) -> None:
        self._pinch_button = None
        self._pinch_fired = False
        self._release_since = None

    # ---- outputs -------------------------------------------------------- #

    def _cursor_or_origin(self) -> tuple[int, int]:
        return self._cursor if self._cursor is not None else (self._screen.left, self._screen.top)

    def _move(self, target: tuple[float, float], gain: float, events: list[GestureEvent]) -> None:
        x, y = self._pointer.update(target, gain)
        position = (round(x), round(y))
        self._cursor = position
        if position != self._emitted:  # มือนิ่ง = ไม่สั่งย้ายซ้ำ (เมาส์จริงยังใช้ได้)
            self._emitted = position
            events.append(GestureEvent(EventKind.MOVE, x=position[0], y=position[1]))

    def _button_up(self, reason: str) -> GestureEvent:
        x, y = self._cursor_or_origin()
        return GestureEvent(EventKind.BUTTON_UP, x=x, y=y, button="left", reason=reason)

    def _click(self, now: int, events: list[GestureEvent]) -> None:
        x, y = self._cursor_or_origin()
        last = self._last_click
        if (
            last is not None
            and now - last[0] <= self._double_click_ms
            and math.hypot(x - last[1], y - last[2]) <= self._double_click_radius
        ):
            # คลิกที่สองภายในเวลาดับเบิลคลิกของ OS: คลิกซ้ำที่จุดเดิมเป๊ะ เพื่อให้ OS นับเป็นดับเบิลคลิก
            # แม้มือจะขยับไปเล็กน้อยระหว่างจีบสองครั้ง
            first = (last[1], last[2])
            self._pointer.jump_to(*first)
            self._cursor = self._emitted = first
            events.append(GestureEvent(EventKind.CLICK, x=first[0], y=first[1], count=2))
            self._last_click = None
        else:
            events.append(GestureEvent(EventKind.CLICK, x=x, y=y, count=1))
            self._last_click = (now, x, y)

    def _scroll(self, hand: Hand, now: int, events: list[GestureEvent]) -> None:
        """สกรอลตามระยะที่มือขยับ โดยมี "ระยะฟรี" เท่า scroll.deadzone กันมือสั่น และล็อกแกนเดียว"""
        cfg = self._cfg.scroll
        px, py = palm_center(hand)
        ref_x, ref_y = self._scroll_ref
        move_x = _beyond(px - ref_x, cfg.deadzone)
        move_y = _beyond(py - ref_y, cfg.deadzone)

        if self._scroll_axis is None:
            if move_x == 0.0 and move_y == 0.0:
                return
            self._scroll_axis = "y" if abs(move_y) >= abs(move_x) else "x"
            self._scroll_remainder = 0.0

        if self._scroll_axis == "y":
            move = move_y
            self._scroll_ref = (px, ref_y + move_y)  # แกนที่ไม่ได้ล็อกตามมือไปเลย ไม่สะสมระยะ
        else:
            move = move_x
            self._scroll_ref = (ref_x + move_x, py)

        if move != 0.0:
            self._scroll_last_output = now
            direction = -1.0 if cfg.invert else 1.0
            # แกน y ของภาพชี้ลง: มือขึ้น (y ลด) = เลื่อนขึ้น (dy > 0), มือไปขวา = เลื่อนขวา (dx > 0)
            sign = -1.0 if self._scroll_axis == "y" else 1.0
            self._scroll_remainder += sign * direction * move * SCROLL_NOTCHES_PER_UNIT * cfg.speed
            notches = math.trunc(self._scroll_remainder)
            if notches:
                self._scroll_remainder -= notches
                if self._scroll_axis == "y":
                    events.append(GestureEvent(EventKind.SCROLL, dy=notches))
                else:
                    events.append(GestureEvent(EventKind.SCROLL, dx=notches))
        elif now - self._scroll_last_output > SCROLL_AXIS_UNLOCK_MS:
            self._scroll_axis = None
            self._scroll_ref = (px, py)

    def _make_snapshot(self, frame: HandFrame, m: _Metrics | None, now: int) -> EngineSnapshot:
        pinch = self._cfg.pinch
        level = 0.0
        hold = 0.0
        if m is not None:
            closest = min(m.index, m.middle)
            level = min(max((pinch.off - closest) / (pinch.off - pinch.on), 0.0), 1.0)
        if self._state is State.PINCH_PENDING and self._pinch_button == "left":
            hold = min(max((now - self._pinch_t0) / pinch.drag_hold_ms, 0.0), 1.0)
        elif self._state is State.DRAGGING:
            hold = 1.0
        return EngineSnapshot(
            state=self._state,
            armed=self._armed,
            hand_present=m is not None,
            hands=len(frame.hands),
            cursor=self._cursor,
            gesture=m.gesture if m is not None else GESTURE_NONE,
            pinch_ratio=m.index_raw if m is not None else None,
            pinch_level=level,
            pinch_button="left" if self._state is State.DRAGGING else self._pinch_button,
            hold_progress=hold,
        )


def _beyond(delta: float, deadband: float) -> float:
    """ส่วนของ ``delta`` ที่เลยระยะฟรี ``deadband`` ออกไป (มีเครื่องหมาย)"""
    if delta > deadband:
        return delta - deadband
    if delta < -deadband:
        return delta + deadband
    return 0.0
