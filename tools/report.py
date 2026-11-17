#!/usr/bin/env python3
"""สรุปสถิติหลาย session (``logs/session-*.json``) + incident (``incidents/*.jsonl``)

บอก latency p50/p95, FPS, ท่าที่ใช้บ่อย, ปัญหาที่เจอบ่อย (จากป้ายของ incident) และค่าที่ควรปรับ

ใช้งาน::

    python tools/report.py
    python tools/report.py --logs logs --incidents incidents --last 5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from aether.config import log_dir
from aether.incidents import build_report, incident_dir, list_incidents, read_incident


def load_sessions(directory: Path, last: int | None = None) -> list[dict]:
    sessions = []
    paths = sorted(directory.glob("session-*.json")) if directory.is_dir() else []
    if last:
        paths = paths[-last:]
    for path in paths:
        try:
            sessions.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError) as exc:
            print(f"ข้าม {path.name}: {exc}", file=sys.stderr)
    return sessions


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="สรุปสถิติการใช้งานและปัญหาที่จดไว้")
    parser.add_argument("--logs", type=Path, help="โฟลเดอร์ session-*.json (ค่าเริ่มต้น: logs/)")
    parser.add_argument("--incidents", type=Path, help="โฟลเดอร์ incident (ค่าเริ่มต้น: incidents/)")
    parser.add_argument("--last", type=int, help="ใช้เฉพาะ N session ล่าสุด")
    args = parser.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")

    sessions = load_sessions(args.logs or log_dir(), args.last)
    incidents = []
    for path in list_incidents(args.incidents or incident_dir()):
        try:
            incidents.append(read_incident(path))
        except (OSError, ValueError) as exc:
            print(f"ข้าม {path.name}: {exc}", file=sys.stderr)
    if not sessions and not incidents:
        print("ยังไม่มีข้อมูล — ใช้แอปสักพัก (สถิติเขียนตอนปิดแอป) และกด Ctrl+Alt+M เมื่อแอปทำพลาด")
        return 0
    print("\n".join(build_report(sessions, incidents).lines()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
