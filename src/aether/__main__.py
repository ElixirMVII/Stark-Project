"""CLI: ``python -m aether [--debug] [--dry-run] [--no-hud] [--config PATH]``"""

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
        "--config",
        type=Path,
        metavar="PATH",
        help="ไฟล์ config (ค่าเริ่มต้น: config.toml ที่ root ของโปรเจกต์)",
    )
    parser.add_argument("--version", action="version", version=f"AETHER {__version__}")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    # import หลัง parse เพื่อให้ --help / --version ไม่ต้องโหลด Qt
    from .app import Options, run

    return run(
        Options(
            debug=args.debug,
            dry_run=args.dry_run,
            no_hud=args.no_hud,
            config_path=args.config,
        )
    )


if __name__ == "__main__":
    sys.exit(main())
