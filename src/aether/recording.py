"""อัด/เล่นซ้ำสตรีม landmark เป็นไฟล์ JSONL — เก็บ **เฉพาะตัวเลข** ไม่มีภาพจากกล้อง

รูปแบบไฟล์: บรรทัดแรกเป็น header (``{"aether_landmarks": 1, ...}``) บรรทัดต่อไปคือ
:meth:`HandFrame.to_dict` ทีละเฟรม ``timestamp_ms`` นับจากเฟรมแรก

header มี ``expect`` (ไม่บังคับ) บอกผลที่ควรได้เมื่อป้อนไฟล์นี้เข้า gesture engine เช่น
``{"left_clicks": 2, "drags": 0}`` — ``tests/test_replay.py`` จะตรวจทุกไฟล์ใน ``tests/fixtures/``
ค่าเป็นจำนวนเต็ม (ต้องเท่าเป๊ะ) หรือ ``[ต่ำสุด, สูงสุด]`` และ ``start_armed`` (ค่าเริ่มต้น ``true``)
บอกว่าตอนเล่นซ้ำให้เริ่มในสถานะ ARMED หรือไม่ (ใช้ ``false`` กับไฟล์ที่ทดสอบการ arm ด้วยท่า 🤟)
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any, Self

from .config import Config
from .gestures import EventKind, GestureEngine, GestureEvent
from .mapping import ScreenGeometry
from .tracker import HandFrame

__all__ = [
    "FORMAT_VERSION",
    "FrameWriter",
    "Recording",
    "read_recording",
    "replay",
    "summarize_events",
]

FORMAT_VERSION = 1
_COORD_DIGITS = 5
_SCORE_DIGITS = 3


@dataclass(frozen=True, slots=True)
class Recording:
    header: dict[str, Any]
    frames: tuple[HandFrame, ...] = field(default_factory=tuple)

    @property
    def expect(self) -> dict[str, Any]:
        return dict(self.header.get("expect", {}))


def _compact(frame: HandFrame, origin_ms: int) -> dict[str, Any]:
    """dict ของเฟรมที่ปัดทศนิยมแล้ว (ไฟล์เล็กลงมาก ความละเอียดยังเกินพอ: 1e-5 ≈ 0.006px)"""
    data = frame.to_dict()
    data["timestamp_ms"] = frame.timestamp_ms - origin_ms
    for hand in data["hands"]:
        hand["handedness_score"] = round(hand["handedness_score"], _SCORE_DIGITS)
        hand["gesture_score"] = round(hand["gesture_score"], _SCORE_DIGITS)
        for key in ("landmarks", "world_landmarks"):
            hand[key] = [[round(v, _COORD_DIGITS) for v in point] for point in hand[key]]
    return data


class FrameWriter:
    """เขียน JSONL ทีละเฟรม (ใช้เป็น context manager)"""

    def __init__(self, path: Path, **header: Any) -> None:
        self._path = Path(path)
        self._header = {"aether_landmarks": FORMAT_VERSION, **header}
        self._file: IO[str] | None = None
        self._origin_ms: int | None = None
        self.frames = 0

    def __enter__(self) -> Self:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self._path.open("w", encoding="utf-8", newline="\n")
        self._file.write(json.dumps(self._header, ensure_ascii=False) + "\n")
        return self

    def __exit__(self, *exc: object) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def write(self, frame: HandFrame) -> None:
        if self._file is None:
            raise RuntimeError("FrameWriter ยังไม่ได้เปิด (ใช้กับ with)")
        if self._origin_ms is None:
            self._origin_ms = frame.timestamp_ms
        line = json.dumps(_compact(frame, self._origin_ms), separators=(",", ":"))
        self._file.write(line + "\n")
        self.frames += 1


def read_recording(path: Path) -> Recording:
    header: dict[str, Any] = {}
    frames: list[HandFrame] = []
    with Path(path).open(encoding="utf-8") as fh:
        for number, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                if "aether_landmarks" in data:
                    header = data
                else:
                    frames.append(HandFrame.from_dict(data))
            except (ValueError, KeyError, TypeError) as exc:
                raise ValueError(f"{path}: บรรทัด {number} อ่านไม่ได้ ({exc})") from exc
    version = header.get("aether_landmarks", FORMAT_VERSION)
    if version != FORMAT_VERSION:
        raise ValueError(f"{path}: ไม่รองรับรูปแบบเวอร์ชัน {version}")
    return Recording(header, tuple(frames))


def replay(
    frames: Iterable[HandFrame],
    config: Config | None = None,
    screen: ScreenGeometry | None = None,
    *,
    armed: bool = True,
) -> Iterator[GestureEvent]:
    """ป้อนเฟรมเข้า gesture engine ตัวใหม่ แล้วคืน event ตามลำดับ (ไม่แตะ OS)

    ``armed`` คือสถานะตอนเริ่ม ตอนจบจะปล่อยปุ่มที่ยังกดค้าง (``BUTTON_UP``) ให้ แต่ไม่รายงาน
    การ arm/disarm ที่ตัว replay ทำเอง — event ``ARMED``/``DISARMED`` ที่ได้จึงมาจากท่ามือล้วนๆ
    """
    engine = GestureEngine(config or Config(), screen or ScreenGeometry(0, 0, 1920, 1080))
    if armed:
        engine.set_armed(True)
    for frame in frames:
        yield from engine.update(frame)[1]
    for event in engine.set_armed(False):
        if event.kind is not EventKind.DISARMED:
            yield event


def summarize_events(events: Iterable[GestureEvent]) -> dict[str, int]:
    """นับสิ่งที่เกิดขึ้นจาก event (ใช้เทียบกับ ``expect`` ใน header ของไฟล์อัด)"""
    summary = {
        "left_clicks": 0,  # คลิกซ้ายทั้งหมด (ดับเบิลคลิกนับเป็น 2)
        "double_clicks": 0,
        "right_clicks": 0,
        "drags": 0,
        "scroll_up": 0,  # จำนวน notch
        "scroll_down": 0,
        "scroll_left": 0,
        "scroll_right": 0,
        "swipe_left": 0,
        "swipe_right": 0,
        "swipe_up": 0,
        "swipe_down": 0,
        "fist_holds": 0,
        "zoom_in": 0,  # จำนวน notch ของ Ctrl+ล้อเมาส์
        "zoom_out": 0,
        "armed": 0,  # จำนวนครั้งที่ถูก arm / disarm ด้วยท่า 🤟
        "disarmed": 0,
        "launches": 0,  # เปิดรายการจากเมนูวงกลม
    }
    for event in events:
        if event.kind is EventKind.CLICK:
            if event.button == "right":
                summary["right_clicks"] += 1
            else:
                summary["left_clicks"] += 1
                summary["double_clicks"] += event.count == 2
        elif event.kind is EventKind.BUTTON_DOWN:
            summary["drags"] += 1
        elif event.kind is EventKind.SCROLL:
            summary["scroll_up"] += max(event.dy, 0)
            summary["scroll_down"] += max(-event.dy, 0)
            summary["scroll_right"] += max(event.dx, 0)
            summary["scroll_left"] += max(-event.dx, 0)
        elif event.kind is EventKind.ZOOM:
            summary["zoom_in"] += max(event.dy, 0)
            summary["zoom_out"] += max(-event.dy, 0)
        elif event.kind is EventKind.ACTION:
            key = "fist_holds" if event.name == "fist_hold" else event.name
            if event.name.startswith("launch_") and event.name.removeprefix("launch_").isdigit():
                key = "launches"
            if key in summary:
                summary[key] += 1
        elif event.kind is EventKind.ARMED:
            summary["armed"] += 1
        elif event.kind is EventKind.DISARMED:
            summary["disarmed"] += 1
    return summary
