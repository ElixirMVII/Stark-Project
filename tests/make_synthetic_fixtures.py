"""สร้าง fixture JSONL สังเคราะห์ใน ``tests/fixtures/`` จากมือจำลอง (``handsim``)

    python tests/make_synthetic_fixtures.py

ไฟล์ ``synthetic_*.jsonl`` ใช้ทดสอบเส้นทาง อัด → อ่าน → เล่นซ้ำ ตอนที่ยังไม่มีไฟล์อัดจากกล้องจริง
ไฟล์ที่อัดจริงด้วย ``tools/record_landmarks.py`` วางในโฟลเดอร์เดียวกันได้เลย (อย่าใช้ชื่อขึ้นต้นด้วย
``synthetic_``) ``tests/test_replay.py`` จะตรวจทุกไฟล์ที่มี ``expect`` ใน header
"""

from __future__ import annotations

import random
import sys
from collections.abc import Iterator
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from handsim import FRAME_MS, OPEN, PINCHED, make_frame, make_hand

from aether.recording import FrameWriter
from aether.tracker import Hand, HandFrame

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
JITTER_PX = 0.4  # landmark สั่นเล็กน้อยเหมือนของจริง


class Script:
    """ลำดับท่าที่เขียนเป็นช่วงๆ → เฟรมที่ ~30fps"""

    def __init__(self, seed: int) -> None:
        self._rng = random.Random(seed)
        self._frames: list[HandFrame] = []
        self._t = 0

    def _emit(self, hand: Hand | None) -> None:
        self._frames.append(make_frame(self._t, *(() if hand is None else (hand,))))
        self._t += FRAME_MS

    def hold(self, ms: int, x: float = 0.5, y: float = 0.4, **pose) -> Script:
        return self.move(ms, (x, y), (x, y), **pose)

    def move(self, ms: int, start: tuple[float, float], end: tuple[float, float], **pose) -> Script:
        frames = max(1, round(ms / FRAME_MS))
        for i in range(frames):
            p = i / max(frames - 1, 1)
            jx = self._rng.uniform(-JITTER_PX, JITTER_PX) / 640
            jy = self._rng.uniform(-JITTER_PX, JITTER_PX) / 480
            x = start[0] + (end[0] - start[0]) * p + jx
            y = start[1] + (end[1] - start[1]) * p + jy
            self._emit(make_hand(x, y, **pose))
        return self

    def no_hand(self, ms: int) -> Script:
        for _ in range(max(1, round(ms / FRAME_MS))):
            self._emit(None)
        return self

    def click(self, x: float = 0.5, y: float = 0.4, pinch_ms: int = 130) -> Script:
        return self.hold(pinch_ms, x, y, pinch=PINCHED).hold(100, x, y, pinch=OPEN)

    def frames(self) -> Iterator[HandFrame]:
        return iter(self._frames)


def build() -> dict[str, tuple[str, dict, Script]]:
    """ชื่อไฟล์ → (คำอธิบาย, ผลที่คาด, สคริปต์)"""
    click_x3 = Script(1).hold(500)
    for _ in range(3):
        click_x3.click().hold(900)

    double_click = Script(2).hold(500).click().click().hold(500)

    drag = (
        Script(3)
        .hold(500, 0.35, 0.35)
        .hold(400, 0.35, 0.35, pinch=PINCHED)
        .move(700, (0.35, 0.35), (0.65, 0.55), pinch=PINCHED)
        .hold(200, 0.65, 0.55, pinch=PINCHED)
        .hold(400, 0.65, 0.55, pinch=OPEN)
    )

    right_click = Script(4).hold(500).hold(300, middle_pinch=PINCHED).hold(500)

    scroll = (
        Script(5)
        .hold(500)
        .hold(300, gesture="Victory")
        .move(500, (0.5, 0.4), (0.5, 0.2), gesture="Victory")
        .hold(500, 0.5, 0.2, gesture="Victory")
        .move(500, (0.5, 0.2), (0.5, 0.4), gesture="Victory")
        .hold(300, 0.5, 0.4, gesture="Victory")
        .hold(400)
    )

    fist = (
        Script(6)
        .hold(500)
        .hold(900, fist=True, gesture="Closed_Fist")
        .hold(400)
        .no_hand(600)
        .hold(400, pinch=PINCHED)  # มือกลับเข้าเฟรมในท่าจีบอยู่แล้ว
        .hold(300)
    )

    return {
        "synthetic_click_x3.jsonl": (
            "จีบคลิก 3 ครั้ง ห่างกัน ~1 วินาที",
            {"left_clicks": 3, "double_clicks": 0, "right_clicks": 0, "drags": 0},
            click_x3,
        ),
        "synthetic_double_click.jsonl": (
            "จีบสองครั้งติดกัน",
            {"left_clicks": 2, "double_clicks": 1, "drags": 0},
            double_click,
        ),
        "synthetic_drag.jsonl": (
            "จีบค้างแล้วลากจากซ้ายบนไปขวาล่าง แล้วปล่อย",
            {"left_clicks": 0, "drags": 1},
            drag,
        ),
        "synthetic_right_click.jsonl": (
            "จีบนิ้วโป้ง+กลาง หนึ่งครั้ง",
            {"left_clicks": 0, "right_clicks": 1, "drags": 0},
            right_click,
        ),
        "synthetic_scroll.jsonl": (
            "✌️ เลื่อนขึ้นแล้วเลื่อนลง",
            {
                "scroll_up": [4, 6],
                "scroll_down": [4, 6],
                "scroll_left": 0,
                "scroll_right": 0,
                "left_clicks": 0,
                "drags": 0,
            },
            scroll,
        ),
        "synthetic_no_false_clicks.jsonl": (
            "กำมือค้าง → มือออกจากเฟรม → กลับเข้ามาในท่าจีบ: ต้องไม่มีคลิกหรือลาก",
            {"left_clicks": 0, "right_clicks": 0, "drags": 0, "scroll_up": 0, "scroll_down": 0},
            fist,
        ),
    }


def write_all(directory: Path = FIXTURES_DIR) -> list[Path]:
    paths = []
    for name, (note, expect, script) in build().items():
        path = directory / name
        header = {
            "note": note,
            "synthetic": True,
            "camera": "640x480 @ 30fps (handsim)",
            "mirror": True,
            "dominant_hand": "right",
            "expect": expect,
        }
        with FrameWriter(path, **header) as writer:
            for frame in script.frames():
                writer.write(frame)
        paths.append(path)
    return paths


if __name__ == "__main__":
    for written in write_all():
        print(written.relative_to(Path.cwd()) if written.is_relative_to(Path.cwd()) else written)
