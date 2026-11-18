"""CLI: ``python -m aether [--source camera|mouse|replay:PATH] [--debug] [--dry-run] [--no-hud]
[--config PATH] [--calibrate] [--teach] [--studio] [--check]``"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m aether",
        description="AETHER — ควบคุม Windows ด้วยมือผ่านเว็บแคม",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="เปิดหน้าต่าง debug (ภาพกล้อง + landmark + ชื่อท่า + FPS)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="ไม่ส่ง input ให้ OS จริง แค่ log ว่าจะทำอะไร",
    )
    parser.add_argument("--no-hud", action="store_true", help="ไม่แสดง HUD overlay")
    parser.add_argument(
        "--source",
        type=_source,
        default="camera",
        metavar="camera|mouse|replay:PATH",
        help="แหล่งเฟรม: กล้องจริง (ค่าเริ่มต้น), mouse = จำลองมือด้วยเมาส์ (บังคับ dry-run), "
        "replay:PATH = เล่นไฟล์ JSONL ที่อัดไว้",
    )
    parser.add_argument(
        "--config",
        type=Path,
        metavar="PATH",
        help="ไฟล์ config (ค่าเริ่มต้น: config.toml ที่ root ของโปรเจกต์)",
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="เปิด calibration wizard ตอนเริ่ม (วัดระยะจีบ + พื้นที่เอื้อม แล้วบันทึก calibration.json)",
    )
    parser.add_argument(
        "--teach",
        action="store_true",
        help="เปิดหน้าต่างสอนท่ามือตอนเริ่ม (สอนท่าใหม่ ผูกปุ่มลัด แล้วบันทึก poses.json)",
    )
    parser.add_argument(
        "--studio",
        action="store_true",
        help="เปิด AETHER Studio ตอนเริ่ม (วาดรูปในอากาศ, กระบะทรายฟิสิกส์, เครื่องดนตรีอากาศ, ห้องทดลองมือ)",
    )
    parser.add_argument(
        "--no-calibration",
        action="store_true",
        help="ไม่ใช้ค่าที่ calibrate ไว้ (ใช้ค่าใน config.toml อย่างเดียว)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="ตรวจว่า config, โมเดล, ฟอนต์ และเสียงพร้อมใช้ แล้วออก (ไม่ใช้กล้อง)",
    )
    parser.add_argument("--version", action="version", version=f"AETHER {__version__}")
    return parser


def _source(text: str):
    from .sources import parse_source

    try:
        return parse_source(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    # import หลัง parse เพื่อให้ --help / --version ไม่ต้องโหลด Qt
    from .app import Options, run, run_check

    options = Options(
        debug=args.debug,
        dry_run=args.dry_run,
        no_hud=args.no_hud,
        config_path=args.config,
        calibrate=args.calibrate,
        no_calibration=args.no_calibration,
        teach=args.teach,
        studio=args.studio,
        source=args.source,
    )
    return run_check(options) if args.check else run(options)


if __name__ == "__main__":
    sys.exit(main())
