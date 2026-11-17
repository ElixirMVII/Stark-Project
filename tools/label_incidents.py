#!/usr/bin/env python3
"""ใส่ป้ายให้ incident ที่จดไว้ด้วย Ctrl+Alt+M (``incidents/*.jsonl``)

ป้าย: false_trigger (ติดเองทั้งที่ไม่ได้ตั้งใจ) · missed (ทำแล้วไม่ติด) · jitter (เคอร์เซอร์สั่น) ·
wrong_click (คลิกผิดที่) · other

ใช้งาน::

    python tools/label_incidents.py                 # ถามทีละไฟล์ที่ยังไม่มีป้าย
    python tools/label_incidents.py --list          # แสดงทุกไฟล์พร้อมป้าย
    python tools/label_incidents.py --set incidents/20261118-093015.jsonl false_trigger --note "ปัดเองตอนเกาหัว"
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

from aether.incidents import LABELS, incident_dir, list_incidents, read_incident, set_label

_SHORTCUTS = {str(i): label for i, label in enumerate(LABELS, start=1)}


def describe(path: Path) -> str:
    incident = read_incident(path)
    header = incident.header
    events = [e for e in incident.recorded_events() if e.kind.value != "move"]
    last = ", ".join(e.name or e.kind.value for e in events[-3:]) or "ไม่มี event"
    label = header.get("label") or "—"
    return (
        f"{path.name}  [{label}]  {header.get('seconds', 0):.1f}s  "
        f"state สุดท้าย {incident.states[-1] if incident.states else '?'}  · event ท้าย: {last}"
    )


def interactive(paths: list[Path], ask: Callable[[str], str] = input) -> int:
    pending = [p for p in paths if not read_incident(p).label]
    if not pending:
        print("ทุกไฟล์มีป้ายแล้ว")
        return 0
    choices = "  ".join(f"{k}={v}" for k, v in _SHORTCUTS.items())
    print(f"ยังไม่มีป้าย {len(pending)} ไฟล์ — เลือก: {choices}  (Enter = ข้าม, q = ออก)")
    done = 0
    for path in pending:
        print(describe(path))
        answer = ask("ป้าย> ").strip().lower()
        if answer == "q":
            break
        if not answer:
            continue
        label = _SHORTCUTS.get(answer, answer)
        if label not in LABELS:
            print(f"  ไม่รู้จัก {answer!r} — ข้าม")
            continue
        note = ask("หมายเหตุ (Enter = ไม่ใส่)> ").strip()
        set_label(path, label, note or None)
        done += 1
    print(f"ใส่ป้ายแล้ว {done} ไฟล์")
    return 0


def main(argv: list[str] | None = None, ask: Callable[[str], str] = input) -> int:
    parser = argparse.ArgumentParser(description="ใส่ป้ายให้ไฟล์ incident")
    parser.add_argument("--dir", type=Path, help="โฟลเดอร์ incident (ค่าเริ่มต้น: incidents/)")
    parser.add_argument("--list", action="store_true", help="แสดงทุกไฟล์พร้อมป้าย")
    parser.add_argument("--set", nargs=2, metavar=("FILE", "LABEL"), help="ใส่ป้ายให้ไฟล์เดียว")
    parser.add_argument("--note", help="หมายเหตุ (ใช้กับ --set)")
    args = parser.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")

    if args.set:
        path, label = Path(args.set[0]), args.set[1]
        try:
            set_label(path, label, args.note)
        except (OSError, ValueError) as exc:
            print(f"ใส่ป้ายไม่ได้: {exc}", file=sys.stderr)
            return 1
        print(describe(path))
        return 0

    directory = args.dir or incident_dir()
    paths = list_incidents(directory)
    if not paths:
        print(f"ไม่มี incident ใน {directory} (กด Ctrl+Alt+M ตอนแอปทำพลาดเพื่อจด)")
        return 0
    if args.list:
        for path in paths:
            print(describe(path))
        return 0
    return interactive(paths, ask)


if __name__ == "__main__":
    sys.exit(main())
