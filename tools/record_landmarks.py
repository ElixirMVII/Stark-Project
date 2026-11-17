#!/usr/bin/env python3
"""อัดสตรีม landmark จากกล้องเป็น JSONL ไว้ใช้ test/replay

เก็บ **เฉพาะตัวเลข landmark** (ตำแหน่งข้อนิ้ว 21 จุด, ท่า, ซ้าย/ขวา) — ไม่มีการบันทึกภาพจากกล้อง

ใช้งาน (จาก root ของโปรเจกต์ หลัง ``pip install -e ".[dev]"``)::

    python tools/record_landmarks.py tests/fixtures/click_x3.jsonl --seconds 6 \\
        --note "จีบคลิก 3 ครั้ง ห่างกันครั้งละ 1 วินาที" --expect left_clicks=3 --expect drags=0

ไฟล์ที่วางใน ``tests/fixtures/`` พร้อม ``--expect`` จะถูก ``pytest`` ตรวจอัตโนมัติ
(ป้อนเข้า gesture engine แล้วนับผลเทียบกับที่คาดไว้) ค่าของ ``--expect`` เป็นจำนวนเต็ม หรือช่วง
``ต่ำสุด..สูงสุด`` เช่น ``scroll_up=3..20`` ดูชื่อที่ใช้ได้ด้วย ``--list-expect``
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from aether.camera import CameraError, OpenCVCamera
from aether.config import ConfigError, load_config, model_path
from aether.pipeline import clock_ms
from aether.recording import (
    FrameWriter,
    parse_expect,
    read_recording,
    replay,
    summarize_events,
)
from aether.tracker import HandTracker


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="อัด landmark จากกล้องเป็น JSONL (ไม่บันทึกภาพ)")
    parser.add_argument("output", nargs="?", type=Path, help="ไฟล์ .jsonl ที่จะเขียน")
    parser.add_argument("--seconds", type=float, default=5.0, help="ความยาวที่อัด (ค่าเริ่มต้น 5)")
    parser.add_argument("--countdown", type=int, default=3, help="นับถอยหลังก่อนเริ่ม (วินาที)")
    parser.add_argument("--note", default="", help="คำอธิบายว่าทำท่าอะไร (เก็บใน header)")
    parser.add_argument("--expect", action="append", default=[], metavar="NAME=N")
    parser.add_argument("--list-expect", action="store_true", help="แสดงชื่อที่ใช้กับ --expect")
    parser.add_argument(
        "--start-disarmed",
        action="store_true",
        help="ตอนเล่นซ้ำให้เริ่มแบบ DISARMED (ใช้กับไฟล์ที่ทดสอบการ arm ด้วยท่า 🤟)",
    )
    parser.add_argument("--config", type=Path, help="ไฟล์ config (ค่าเริ่มต้น: config.toml)")
    args = parser.parse_args(argv)

    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")

    if args.list_expect:
        print("\n".join(summarize_events(())))
        return 0
    if args.output is None:
        parser.error("ต้องระบุไฟล์ output")
    try:
        expect = parse_expect(args.expect)
        config = load_config(args.config)
    except (ValueError, ConfigError) as exc:
        parser.error(str(exc))

    model = model_path()
    if not model.is_file():
        print(f"ไม่พบโมเดล {model} — รัน `python tools/download_assets.py` ก่อน")
        return 1

    camera = OpenCVCamera(config.camera)
    try:
        description = camera.open()
    except CameraError as exc:
        print(exc)
        return 1

    header = {
        "note": args.note,
        "camera": description,
        "mirror": config.camera.mirror,
        "dominant_hand": config.tracking.dominant_hand,
        "start_armed": not args.start_disarmed,
        "expect": expect,
    }
    try:
        with (
            HandTracker(
                model, num_hands=config.tracking.num_hands, mirrored=config.camera.mirror
            ) as tracker,
            FrameWriter(args.output, **header) as writer,
        ):
            for remaining in range(args.countdown, 0, -1):
                print(f"เริ่มอัดใน {remaining}…", flush=True)
                deadline = time.perf_counter() + 1.0
                while time.perf_counter() < deadline:
                    camera.read()  # อ่านทิ้งเพื่อไม่ให้เฟรมค้างใน buffer
            print(f"● กำลังอัด {args.seconds:g} วินาที", flush=True)

            started = time.perf_counter()
            next_report = started + 1.0
            with_hand = 0
            while (now := time.perf_counter()) - started < args.seconds:
                image = camera.read()
                if image is None:
                    continue
                frame = tracker.process(image, clock_ms())
                writer.write(frame)
                with_hand += bool(frame.hands)
                if now >= next_report:
                    next_report += 1.0
                    seen = (
                        ", ".join(f"{h.handedness}:{h.gesture}" for h in frame.hands) or "ไม่เห็นมือ"
                    )
                    print(f"  {now - started:4.1f}s  {writer.frames:4d} เฟรม  {seen}", flush=True)
            elapsed = time.perf_counter() - started
            frames = writer.frames
    finally:
        camera.close()

    if frames == 0:
        print("ไม่ได้เฟรมจากกล้องเลย")
        return 1
    print(
        f"\nเขียน {args.output} — {frames} เฟรม ({frames / elapsed:.1f} fps), "
        f"เห็นมือ {with_hand * 100 // frames}% ของเฟรม"
    )

    recording = read_recording(args.output)
    summary = summarize_events(replay(recording.frames, config, armed=not args.start_disarmed))
    happened = ", ".join(f"{k}={v}" for k, v in summary.items() if v) or "ไม่มี event"
    print(f"ผลเมื่อป้อนเข้า gesture engine: {happened}")
    if not expect:
        print("(ใส่ --expect NAME=N เพื่อให้ pytest ตรวจไฟล์นี้อัตโนมัติ)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
