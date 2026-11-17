#!/usr/bin/env python3
"""เล่น incident (หรือไฟล์อัดใดๆ) ผ่าน gesture engine แบบไม่มีหน้าจอ แล้วพิมพ์ timeline ของ state/event

ใช้ config / จอ / ท่าที่สอนเอง ตอนเกิดเหตุ (จาก header) — เล่นกี่รอบก็ได้ผลเดิม

ใช้งาน::

    python tools/replay.py incidents/20261118-093015.jsonl
    python tools/replay.py incidents/20261118-093015.jsonl --moves        # แสดงการขยับเคอร์เซอร์ด้วย

    # ย้ายไปเป็น test fixture (pytest จะตรวจทุกครั้งว่าได้ผลตาม --expect)
    python tools/replay.py incidents/20261118-093015.jsonl --promote \\
        --expect left_clicks=0 --expect swipe_up=0 --name no_swipe_when_scratching

``--expect`` คือสิ่งที่ **ควร** เกิด (ไม่ใช่สิ่งที่เกิดตอนพลาด) ถ้า engine ตอนนี้ยังให้ผลไม่ตรง ไฟล์ก็ยังถูก
ย้ายไปได้ แต่ test จะ fail จนกว่าจะแก้ engine/ค่าเริ่มต้น — ใช้ ``--check`` ถ้าอยากให้ย้ายเฉพาะตอนผ่านแล้ว
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

from aether.gestures import EventKind, GestureEvent, State
from aether.incidents import read_incident, replay_incident
from aether.recording import parse_expect, summarize_events

REPO_FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures"


def format_event(event: GestureEvent) -> str:
    kind = event.kind
    if kind is EventKind.ACTION:
        return f"ACTION {event.name}"
    if kind is EventKind.CLICK:
        extra = " ×2" if event.count == 2 else ""
        return f"CLICK {event.button}{extra} ({event.x}, {event.y})"
    if kind in (EventKind.BUTTON_DOWN, EventKind.BUTTON_UP):
        reason = f" [{event.reason}]" if event.reason else ""
        return f"{kind.value.upper()} {event.button} ({event.x}, {event.y}){reason}"
    if kind in (EventKind.SCROLL, EventKind.ZOOM):
        return f"{kind.value.upper()} dx={event.dx:+d} dy={event.dy:+d}"
    if kind is EventKind.MOVE:
        return f"move ({event.x}, {event.y})"
    if kind in (EventKind.WINDOW_MOVE, EventKind.WINDOW_RELEASE):
        return f"{kind.value.upper()} {event.dx:+d} {event.dy:+d}"
    return kind.value.upper() + (f" [{event.reason}]" if event.reason else "")


def timeline_lines(timeline: list[tuple[int, State, list]], *, moves: bool = False) -> list[str]:
    lines = []
    previous: State | None = None
    for t, state, events in timeline:
        shown = [e for e in events if moves or e.kind is not EventKind.MOVE]
        if state is previous and not shown:
            continue
        marker = state.value if state is not previous else ""
        detail = " · ".join(format_event(e) for e in shown)
        lines.append(f"+{t / 1000:6.3f}s  {marker:<14} {detail}".rstrip())
        previous = state
    return lines


def promote(
    source: Path,
    expect: dict,
    *,
    name: str | None = None,
    note: str = "",
    fixtures: Path = REPO_FIXTURES,
) -> Path:
    """คัดลอกไปเป็น fixture พร้อม ``expect`` ใน header (test_replay ตรวจไฟล์ใน fixtures อัตโนมัติ)"""
    stem = re.sub(r"[^a-z0-9_-]+", "_", (name or f"incident_{source.stem}").lower()).strip("_")
    target = fixtures / f"{stem}.jsonl"
    if target.exists():
        raise FileExistsError(f"มี {target.name} อยู่แล้ว — ตั้งชื่ออื่นด้วย --name")
    fixtures.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    lines = target.read_text(encoding="utf-8").splitlines()
    header = json.loads(lines[0])
    header["expect"] = expect
    header["note"] = note or header.get("note") or f"จาก incident {source.name}"
    lines[0] = json.dumps(header, ensure_ascii=False)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="เล่น incident ผ่าน gesture engine (ไม่มีหน้าจอ)")
    parser.add_argument("file", type=Path)
    parser.add_argument("--moves", action="store_true", help="แสดงการขยับเคอร์เซอร์ด้วย")
    parser.add_argument("--promote", action="store_true", help="ย้ายไปเป็น test fixture")
    parser.add_argument("--expect", action="append", default=[], metavar="NAME=N")
    parser.add_argument("--name", help="ชื่อไฟล์ fixture (ไม่ต้องใส่ .jsonl)")
    parser.add_argument("--note", default="", help="คำอธิบายใน header ของ fixture")
    parser.add_argument("--check", action="store_true", help="ย้ายเฉพาะเมื่อผลตอนนี้ตรง --expect แล้ว")
    parser.add_argument("--fixtures", type=Path, default=REPO_FIXTURES, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")

    try:
        incident = read_incident(args.file)
        expect = parse_expect(args.expect)
    except (OSError, ValueError) as exc:
        print(f"อ่านไม่ได้: {exc}", file=sys.stderr)
        return 1
    if args.promote and not expect:
        parser.error("--promote ต้องมี --expect อย่างน้อยหนึ่งค่า (สิ่งที่ควรเกิด)")

    header = incident.header
    print(
        f"{args.file.name} · {len(incident.frames)} เฟรม · ป้าย {header.get('label') or '—'}"
        f" · เริ่ม {'ARMED' if header.get('start_armed', True) else 'DISARMED'}"
    )
    timeline = replay_incident(incident)
    print("\n".join(timeline_lines(timeline, moves=args.moves)) or "(ไม่มี event)")
    summary = summarize_events(e for _, _, events in timeline for e in events)
    happened = ", ".join(f"{k}={v}" for k, v in summary.items() if v) or "ไม่มี"
    print(f"สรุป: {happened}")

    if not args.promote:
        return 0
    mismatches = {
        key: (summary[key], wanted)
        for key, wanted in expect.items()
        if not (
            wanted[0] <= summary[key] <= wanted[1]
            if isinstance(wanted, list)
            else summary[key] == wanted
        )
    }
    if mismatches:
        print(f"ตอนนี้ยังไม่ตรง --expect: (ได้, ควร) = {mismatches}")
        if args.check:
            print("ไม่ได้ย้าย (--check)")
            return 2
    try:
        target = promote(args.file, expect, name=args.name, note=args.note, fixtures=args.fixtures)
    except OSError as exc:
        print(f"ย้ายไม่ได้: {exc}", file=sys.stderr)
        return 1
    print(f"ย้ายเป็น fixture แล้ว: {target} — pytest จะตรวจไฟล์นี้ทุกครั้ง")
    return 0


if __name__ == "__main__":
    sys.exit(main())
