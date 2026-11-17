"""แหล่งเฟรม (input source) — ให้พัฒนาและทดสอบต่อได้แม้ยังไม่มีกล้อง (spec หัวข้อ 4.1)

ทุกอย่างหลัง tracker รับแค่ :class:`~aether.tracker.HandFrame` ไม่สนว่ามาจากไหน:

- ``camera`` — :class:`CameraSource` เว็บแคมจริง (ภาพ → MediaPipe) ค่าเริ่มต้น
- ``mouse`` — :class:`MouseSimSource` **จำลองมือด้วยเมาส์**: ตำแหน่งเมาส์จริงแทนจุดคุมเคอร์เซอร์
  กดซ้ายค้าง = จีบโป้ง+ชี้, กดขวาค้าง = จีบโป้ง+กลาง, ล้อเมาส์ = ✌️ สกรอล,
  ``1``–``5`` = None / Open_Palm / Closed_Fist / Victory / ILoveYou, ลูกศร = ปัด (ผ่าน global
  listener ของ pynput) — **บังคับ dry-run เสมอ** เพราะเมาส์จริงเป็นตัวป้อนอยู่แล้ว
- ``replay:PATH`` — :class:`ReplaySource` เล่นไฟล์ JSONL ที่อัดด้วย ``tools/record_landmarks.py``
  ตามจังหวะเวลาเดิม

แหล่งจำลองคืน ``HandFrame`` ตรงๆ จาก :meth:`read` (ไม่มีภาพ) ซึ่ง :class:`aether.pipeline.Pipeline`
ส่งเข้า gesture engine โดยไม่ผ่าน tracker
"""

from __future__ import annotations

import dataclasses
import logging
import math
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .camera import CameraError, OpenCVCamera
from .config import Config
from .gestures import LOST_RELEASE_MS, SWIPE_REST_MS
from .mapping import ScreenArea, screen_to_camera
from .tracker import GESTURE_NONE, LANDMARK_COUNT, LM, Hand, HandFrame, Landmark

__all__ = [
    "CAMERA_SOURCE",
    "KEY_GESTURES",
    "CameraSource",
    "MouseHandModel",
    "MouseSimSource",
    "NullTracker",
    "ReplaySource",
    "SourceSpec",
    "create_source",
    "key_name",
    "parse_source",
    "synthetic_hand",
]

log = logging.getLogger(__name__)

CameraSource = OpenCVCamera  # เว็บแคมจริง: ให้ภาพ แล้ว tracker (MediaPipe) แปลงเป็น HandFrame

# ปุ่ม 1–5 → ท่าสำเร็จรูป (ตาม spec 4.1)
KEY_GESTURES = {
    "1": GESTURE_NONE,
    "2": "Open_Palm",
    "3": "Closed_Fist",
    "4": "Victory",
    "5": "ILoveYou",
}
ARROWS = {"left": (-1, 0), "right": (1, 0), "up": (0, -1), "down": (0, 1)}

HAND_SCALE = 110.0  # px ของเฟรม: ระยะข้อมือ→โคนนิ้วกลางของมือจำลอง (นั่งห่างกล้องราว 60 ซม.)
OPEN_RATIO = 1.0  # pinch_ratio ตอนไม่ได้จีบ
PINCH_RATIO = 0.12  # pinch_ratio ตอนจีบ (ต่ำกว่า pinch.on)

VOTE_MS = 150  # รอให้ majority vote ยอมรับท่าใหม่ก่อนเริ่มขยับ (5 เฟรมที่ 30fps ≈ 3 เฟรมพอ)
SCROLL_STEP = 1.0 / 30.0  # ล้อ 1 notch = มือขยับเท่านี้ของความสูงเฟรม ≈ สกรอล 1 notch ที่ scroll.speed = 1
SCROLL_IDLE_MS = 400  # หยุดหมุนล้อนานเท่านี้ = เลิกท่า ✌️
SWIPE_MOVE_MS = 120
SWIPE_HOLD_MS = 120
SWIPE_RETURN_MS = 250
SWIPE_MARGIN = 0.08  # ปัดไกลกว่าเกณฑ์ใน config เท่านี้ (สัดส่วนของเฟรม)

_INDEX_DIR = (0.6, -0.8)
_MIDDLE_DIR = (0.8, -0.6)


# --------------------------------------------------------------------------- #
# Source spec
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class SourceSpec:
    kind: str  # "camera" | "mouse" | "replay"
    path: Path | None = None

    def __str__(self) -> str:
        return f"replay:{self.path}" if self.kind == "replay" else self.kind

    @property
    def simulated(self) -> bool:
        """ไม่ใช้กล้อง/MediaPipe (ไม่ต้องมีไฟล์โมเดล)"""
        return self.kind != "camera"


CAMERA_SOURCE = SourceSpec("camera")


def parse_source(text: str) -> SourceSpec:
    """``"camera"`` | ``"mouse"`` | ``"replay:PATH"`` → :class:`SourceSpec` (raise ``ValueError``)"""
    value = text.strip()
    lowered = value.lower()
    if lowered in ("camera", "mouse"):
        return SourceSpec(lowered)
    if lowered.startswith("replay:"):
        path = value[len("replay:") :].strip().strip('"')
        if not path:
            raise ValueError("replay: ต้องตามด้วย path ของไฟล์ เช่น replay:tests/fixtures/x.jsonl")
        return SourceSpec("replay", Path(path))
    raise ValueError(f"source ต้องเป็น camera, mouse หรือ replay:PATH (ได้ {text!r})")


class NullTracker:
    """แทน HandTracker เมื่อใช้แหล่งจำลอง — เฟรมเป็น HandFrame อยู่แล้ว ไม่ต้องโหลด MediaPipe"""

    def process(self, frame: Any, timestamp_ms: int) -> HandFrame:
        raise TypeError("แหล่งจำลองต้องส่ง HandFrame ไม่ใช่ภาพ")

    def close(self) -> None:
        pass


def create_source(spec: SourceSpec, config: Config, screen: ScreenArea):
    if spec.kind == "mouse":
        return MouseSimSource(config, screen)
    if spec.kind == "replay":
        assert spec.path is not None
        return ReplaySource(spec.path)
    return CameraSource(config.camera)


# --------------------------------------------------------------------------- #
# มือสังเคราะห์
# --------------------------------------------------------------------------- #


def synthetic_hand(
    anchor: tuple[float, float],
    width: int,
    height: int,
    *,
    side: str = "right",
    gesture: str = GESTURE_NONE,
    score: float = 0.95,
    pinch_index: float = OPEN_RATIO,
    pinch_middle: float = 1.2,
    fist: bool = False,
    scale: float = HAND_SCALE,
) -> Hand:
    """มือที่จุดคุมเคอร์เซอร์ (กึ่งกลางโคนนิ้วชี้กับนิ้วกลาง) อยู่ที่ ``anchor`` (normalized)

    ``pinch_index`` / ``pinch_middle`` คือ pinch_ratio ที่ engine จะวัดได้พอดี
    ``fist=True`` พับนิ้วเข้าฝ่ามือ (engine ไม่นับเป็นการจีบ) — มือซ้ายกลับซ้าย/ขวาให้เอง
    """
    ax, ay = anchor[0] * width, anchor[1] * height
    flip = -1.0 if side == "left" else 1.0

    def at(dx: float, dy: float) -> tuple[float, float]:
        return (ax + dx * flip * scale, ay + dy * scale)

    px: dict[int, tuple[float, float]] = {
        LM.INDEX_MCP: at(-0.11, 0.0),
        LM.MIDDLE_MCP: at(0.11, 0.0),
        LM.RING_MCP: at(0.31, 0.05),
        LM.PINKY_MCP: at(0.50, 0.15),
        LM.WRIST: at(0.11, 1.0),  # ใต้โคนนิ้วกลางพอดี → hand_scale = scale
        LM.THUMB_CMC: at(-0.15, 0.80),
        LM.THUMB_MCP: at(-0.35, 0.55),
        LM.THUMB_IP: at(-0.40, 0.15),
    }
    if fist:
        thumb = at(0.0, 0.30)
        index_tip, middle_tip = at(0.0, 0.45), at(0.10, 0.50)
        ring_tip, pinky_tip = at(0.31, 0.50), at(0.50, 0.50)
        ring_mid, pinky_mid = at(0.31, -0.10), at(0.50, 0.0)
    else:
        thumb = at(-0.35, -0.25)
        index_tip = (
            thumb[0] + pinch_index * scale * _INDEX_DIR[0] * flip,
            thumb[1] + pinch_index * scale * _INDEX_DIR[1],
        )
        middle_tip = (
            thumb[0] + pinch_middle * scale * _MIDDLE_DIR[0] * flip,
            thumb[1] + pinch_middle * scale * _MIDDLE_DIR[1],
        )
        ring_tip, pinky_tip = at(0.31, -0.75), at(0.50, -0.50)
        ring_mid, pinky_mid = at(0.31, -0.40), at(0.50, -0.25)
    px[LM.THUMB_TIP] = thumb
    px[LM.INDEX_TIP] = index_tip
    px[LM.MIDDLE_TIP] = middle_tip
    px[LM.RING_TIP], px[LM.PINKY_TIP] = ring_tip, pinky_tip
    px[LM.RING_PIP], px[LM.RING_DIP] = ring_mid, _between(ring_mid, ring_tip)
    px[LM.PINKY_PIP], px[LM.PINKY_DIP] = pinky_mid, _between(pinky_mid, pinky_tip)
    for mcp, pip, dip, tip in (
        (LM.INDEX_MCP, LM.INDEX_PIP, LM.INDEX_DIP, LM.INDEX_TIP),
        (LM.MIDDLE_MCP, LM.MIDDLE_PIP, LM.MIDDLE_DIP, LM.MIDDLE_TIP),
    ):
        if fist:  # ข้อกลางนิ้วยกขึ้นเหนือโคนนิ้ว ปลายนิ้วพับลงมาในฝ่ามือ
            px[pip] = (px[mcp][0], px[mcp][1] - 0.25 * scale)
            px[dip] = _between(px[pip], px[tip])
        else:
            px[pip] = _between(px[mcp], px[tip], 0.45)
            px[dip] = _between(px[mcp], px[tip], 0.75)
    assert len(px) == LANDMARK_COUNT
    landmarks = tuple(Landmark(px[i][0] / width, px[i][1] / height, 0.0) for i in range(21))
    return Hand(side, 0.95, gesture, score if gesture != GESTURE_NONE else 0.9, landmarks)


def _between(a: tuple[float, float], b: tuple[float, float], t: float = 0.5) -> tuple[float, float]:
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)


def _smoothstep(t: float) -> float:
    t = min(max(t, 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


# --------------------------------------------------------------------------- #
# มือจำลองจากเมาส์ (pure — ป้อน input แล้วขอเฟรม)
# --------------------------------------------------------------------------- #


class MouseHandModel:
    """แปลงสถานะเมาส์/คีย์บอร์ดเป็น :class:`HandFrame` (ไม่แตะ OS — test ได้ตรงๆ)

    - ตำแหน่งเมาส์ถูกแปลงกลับเป็นจุดในเฟรมกล้อง (:func:`screen_to_camera`) เคอร์เซอร์ของ engine
      จึงตรงกับเมาส์จริง
    - ล้อเมาส์: เปลี่ยนเป็นท่า ✌️ แล้วขยับมือขึ้น/ลงทีละ ``SCROLL_STEP`` ต่อ notch หยุดหมุนล้อ
      แล้วมือ "หายไป" ครู่หนึ่ง (เกิน 300ms) ก่อนกลับมาที่ตำแหน่งเมาส์ — ให้ engine เริ่มตามมือใหม่
      ตรงตำแหน่งเมาส์ ไม่ใช่ตำแหน่งที่เลื่อนไประหว่างสกรอล
    - ลูกศร: แบมือให้นิ่ง แล้วสะบัดไปทิศนั้นไกลกว่าเกณฑ์ใน config เล็กน้อย แล้วดึงกลับ
    """

    def __init__(self, config: Config, screen: ScreenArea) -> None:
        self._config = config
        self._screen = screen
        self._side = config.tracking.dominant_hand
        self._width, self._height = config.camera.width, config.camera.height
        self._buttons: set[str] = set()
        self._gesture = GESTURE_NONE
        # ล้อเมาส์
        self._pending = 0  # notch ที่ยังไม่ได้ขยับมือ (+ = ขึ้น)
        self._scroll_since: int | None = None
        self._last_wheel = 0
        self._scroll_offset = 0.0
        self._hide_until = 0
        # ปัด
        self._swipe: tuple[int, int, int] | None = None  # (dx, dy, เวลาเริ่ม)
        self._swipe_rest_ms = SWIPE_REST_MS + VOTE_MS + 150

    @property
    def gesture(self) -> str:
        return self._gesture

    def press(self, button: str) -> None:
        if button in ("left", "right"):
            self._buttons.add(button)

    def release(self, button: str) -> None:
        self._buttons.discard(button)

    def wheel(self, notches: int, now_ms: int) -> None:
        """หมุนล้อ ``notches`` (บวก = ขึ้น)"""
        if notches == 0 or self._swipe is not None:
            return
        if self._scroll_since is None:
            self._scroll_since = now_ms
        self._pending += notches
        self._last_wheel = now_ms

    def key(self, name: str, now_ms: int) -> None:
        """ปุ่ม ``"1"``–``"5"`` เปลี่ยนท่า, ``"left"/"right"/"up"/"down"`` ปัด"""
        if name in KEY_GESTURES:
            self._gesture = KEY_GESTURES[name]
        elif name in ARROWS and self._swipe is None and self._scroll_since is None:
            dx, dy = ARROWS[name]
            self._swipe = (dx, dy, now_ms)

    def frame(self, now_ms: int, mouse: tuple[float, float]) -> HandFrame:
        cfg = self._config
        anchor = screen_to_camera(mouse[0], mouse[1], cfg.pointer.active_region, self._screen)
        if now_ms < self._hide_until:
            return HandFrame(now_ms, self._width, self._height, ())
        gesture = self._gesture
        offset = (0.0, 0.0)
        if self._swipe is not None:
            gesture, offset = self._swipe_frame(now_ms)
        elif self._scroll_since is not None:
            scrolled = self._scroll_frame(now_ms)
            if scrolled is None:
                return HandFrame(now_ms, self._width, self._height, ())
            gesture, offset = scrolled

        fist = gesture == "Closed_Fist"
        pinch_index = PINCH_RATIO if "left" in self._buttons else OPEN_RATIO
        pinch_middle = PINCH_RATIO if "right" in self._buttons else 1.2
        if "left" in self._buttons and "right" in self._buttons:
            pinch_middle = PINCH_RATIO + 0.05  # สองปุ่มพร้อมกัน: นิ้วชี้ชิดกว่า = คลิกซ้าย
        hand = synthetic_hand(
            (anchor[0] + offset[0], anchor[1] + offset[1]),
            self._width,
            self._height,
            side=self._side,
            gesture=gesture,
            pinch_index=pinch_index,
            pinch_middle=pinch_middle,
            fist=fist,
        )
        return HandFrame(now_ms, self._width, self._height, (hand,))

    # ---- internal ------------------------------------------------------- #

    def _scroll_frame(self, now: int) -> tuple[str, tuple[float, float]] | None:
        """``None`` = จบการสกรอลแล้ว (เฟรมนี้ไม่มีมือ)"""
        assert self._scroll_since is not None
        if not self._pending and now - self._last_wheel >= SCROLL_IDLE_MS:
            # จบ: ให้มือหายเกิน LOST_RELEASE_MS (engine ลืมมือนี้) แล้วกลับมาที่ตำแหน่งเมาส์
            # — ถ้าดึงมือกลับที่เดิมขณะยังเป็น ✌️ จะกลายเป็นสกรอลย้อนกลับ
            self._scroll_since = None
            self._scroll_offset = 0.0
            self._hide_until = now + LOST_RELEASE_MS + 70
            return None
        if now - self._scroll_since >= VOTE_MS and self._pending:
            step = 1 if self._pending > 0 else -1
            distance = SCROLL_STEP
            if self._scroll_offset == 0.0:
                distance += self._config.scroll.deadzone  # ก้าวแรกต้องพ้นระยะฟรีของการสกรอลด้วย
            self._pending -= step
            self._scroll_offset -= step * distance  # ล้อขึ้น = มือขึ้น (y ของภาพลดลง)
        return "Victory", (0.0, self._scroll_offset)

    def _swipe_frame(self, now: int) -> tuple[str, tuple[float, float]]:
        assert self._swipe is not None
        dx, dy, start = self._swipe
        elapsed = now - start
        gestures = self._config.gestures
        distance = (gestures.swipe_dist_x if dx else gestures.swipe_dist_y) + SWIPE_MARGIN
        rest = self._swipe_rest_ms
        if elapsed < rest:
            amount = 0.0
        elif elapsed < rest + SWIPE_MOVE_MS:
            amount = _smoothstep((elapsed - rest) / SWIPE_MOVE_MS)
        elif elapsed < rest + SWIPE_MOVE_MS + SWIPE_HOLD_MS:
            amount = 1.0
        else:
            back = (elapsed - rest - SWIPE_MOVE_MS - SWIPE_HOLD_MS) / SWIPE_RETURN_MS
            amount = 1.0 - _smoothstep(back)
            if back >= 1.0:
                self._swipe = None
                return self._gesture, (0.0, 0.0)
        return "Open_Palm", (dx * distance * amount, dy * distance * amount)


def key_name(key: Any) -> str | None:
    """ปุ่มจาก pynput → ``"1"``–``"5"`` หรือชื่อลูกศร (None = ไม่สนใจ)

    ใช้รหัสปุ่ม (virtual-key) เป็นหลัก ปุ่มตัวเลขจึงใช้ได้แม้เปิดแป้นภาษาไทย (ซึ่งพิมพ์เป็น ๅ / - ภ ถ)
    """
    name = getattr(key, "name", None)
    if name in ARROWS:
        return name
    vk = getattr(key, "vk", None)
    if isinstance(vk, int):
        if 0x31 <= vk <= 0x35:  # แถวตัวเลข 1–5
            return str(vk - 0x30)
        if 0x61 <= vk <= 0x65:  # numpad 1–5
            return str(vk - 0x60)
    char = getattr(key, "char", None)
    if isinstance(char, str) and char in KEY_GESTURES:
        return char
    return None


class MouseSimSource:
    """``--source mouse``: อ่านเมาส์/คีย์บอร์ดด้วย global listener ของ pynput แล้วคืน HandFrame

    เมาส์และปุ่มที่กดยังไปถึงโปรแกรมอื่นตามปกติ (listener แค่ฟัง ไม่ดักไว้) — แอปจึงบังคับ dry-run
    """

    def __init__(
        self,
        config: Config,
        screen: ScreenArea,
        *,
        position: Callable[[], tuple[float, float]] | None = None,
        clock: Callable[[], float] = time.perf_counter,
        sleep: Callable[[float], None] = time.sleep,
        listen: bool = True,
    ) -> None:
        self._model = MouseHandModel(config, screen)
        self._interval = 1.0 / config.camera.fps
        self._position = position
        self._clock = clock
        self._sleep = sleep
        self._listen = listen
        self._lock = threading.Lock()
        self._listeners: list[Any] = []
        self._next = 0.0

    @property
    def model(self) -> MouseHandModel:
        return self._model

    def open(self) -> str:
        if self._listen:
            try:
                from pynput import keyboard, mouse
            except Exception as exc:
                raise CameraError(f"เริ่มฟังเมาส์/คีย์บอร์ดไม่ได้: {exc}") from exc
            controller = mouse.Controller()
            if self._position is None:
                self._position = lambda: tuple(controller.position)  # type: ignore[assignment]
            self._listeners = [
                mouse.Listener(on_click=self._on_click, on_scroll=self._on_scroll),
                keyboard.Listener(on_press=self._on_key),
            ]
            for listener in self._listeners:
                listener.daemon = True
                listener.start()
        self._next = self._clock()
        log.info("ใช้เมาส์จำลองมือ (dry-run): ปุ่ม 1–5 เปลี่ยนท่า, ลูกศร = ปัด, ล้อ = ✌️ สกรอล")
        return "mouse simulation"

    def read(self) -> HandFrame | None:
        now = self._clock()
        if self._next > now:
            self._sleep(self._next - now)
            now = self._clock()
        self._next = max(self._next + self._interval, now)
        position = self._position() if self._position is not None else (0.0, 0.0)
        with self._lock:
            return self._model.frame(int(now * 1000), position)

    def close(self) -> None:
        for listener in self._listeners:
            listener.stop()
        self._listeners = []

    # ---- pynput callbacks (thread ของ listener) ------------------------- #

    def _now_ms(self) -> int:
        return int(self._clock() * 1000)

    def _on_click(self, x: float, y: float, button: Any, pressed: bool) -> None:
        name = getattr(button, "name", str(button))
        with self._lock:
            if pressed:
                self._model.press(name)
            else:
                self._model.release(name)

    def _on_scroll(self, x: float, y: float, dx: float, dy: float) -> None:
        notches = int(math.copysign(max(1, round(abs(dy))), dy)) if dy else 0
        with self._lock:
            self._model.wheel(notches, self._now_ms())

    def _on_key(self, key: Any) -> None:
        name = key_name(key)
        if name is not None:
            with self._lock:
                self._model.key(name, self._now_ms())


# --------------------------------------------------------------------------- #
# Replay
# --------------------------------------------------------------------------- #


class ReplaySource:
    """``--source replay:PATH``: เล่นไฟล์ที่อัดไว้ตามจังหวะเวลาเดิม

    เล่นจบแล้วส่งเฟรมที่ไม่มีมือต่อ (engine จะปล่อยปุ่มที่ค้างและกลับ IDLE เหมือนมือออกจากกล้อง)
    """

    def __init__(
        self,
        path: Path,
        *,
        clock: Callable[[], float] = time.perf_counter,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._path = path
        self._clock = clock
        self._sleep = sleep
        self._frames: tuple[HandFrame, ...] = ()
        self._index = 0
        self._start = 0.0
        self._finished = False
        self._interval = 1.0 / 30.0

    @property
    def finished(self) -> bool:
        return self._finished

    def open(self) -> str:
        from .recording import read_recording

        try:
            recording = read_recording(self._path)
        except (OSError, ValueError, KeyError) as exc:
            raise CameraError(f"เปิดไฟล์ replay {self._path} ไม่ได้: {exc}") from exc
        if not recording.frames:
            raise CameraError(f"ไฟล์ replay {self._path} ไม่มีเฟรม")
        self._frames = recording.frames
        first, last = self._frames[0].timestamp_ms, self._frames[-1].timestamp_ms
        if len(self._frames) > 1 and last > first:
            self._interval = (last - first) / 1000.0 / (len(self._frames) - 1)
        self._index = 0
        self._finished = False
        self._start = self._clock()
        return f"replay {self._path.name} ({len(self._frames)} เฟรม, {(last - first) / 1000:.1f}s)"

    def read(self) -> HandFrame | None:
        if self._index >= len(self._frames):
            if not self._finished:
                self._finished = True
                log.info("replay จบแล้ว: %s", self._path)
            self._sleep(self._interval)
            last = self._frames[-1] if self._frames else HandFrame(0, 640, 480)
            return dataclasses.replace(last, hands=())
        frame = self._frames[self._index]
        due = self._start + (frame.timestamp_ms - self._frames[0].timestamp_ms) / 1000.0
        wait = due - self._clock()
        if wait > 0:
            self._sleep(wait)
        self._index += 1
        return frame

    def close(self) -> None:
        pass
