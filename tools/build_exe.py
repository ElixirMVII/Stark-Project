#!/usr/bin/env python3
"""Build AETHER เป็นโปรแกรมที่รันได้โดยไม่ต้องมี Python (PyInstaller, one-folder)

ใช้งาน (จาก root ของโปรเจกต์ หลัง ``pip install -e ".[dev,build]"``)::

    python tools/build_exe.py             # → dist/aether/aether.exe
    python tools/build_exe.py --console   # build แบบมีหน้าต่าง console (ไว้ debug)

ขั้นตอน: ตรวจ asset → สร้างไอคอน (build/aether.ico) → รัน PyInstaller ด้วย packaging/aether.spec
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPEC = ROOT / "packaging" / "aether.spec"
ICON = ROOT / "build" / "aether.ico"
REQUIRED = ("assets/models/gesture_recognizer.task", "assets/fonts/Orbitron.ttf")


def make_icon(path: Path) -> bool:
    """วาดไอคอนเดียวกับใน tray แล้วบันทึกเป็น .ico หลายขนาด (ใช้ Qt แบบไม่มีจอ)"""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    sys.path.insert(0, str(ROOT / "src"))
    from PySide6.QtGui import QColor, QGuiApplication

    from aether.tray import make_icon as draw

    _app = QGuiApplication.instance() or QGuiApplication([])
    icon = draw(QColor("#00E5FF"), True, size=256)
    path.parent.mkdir(parents=True, exist_ok=True)
    return icon.pixmap(256, 256).save(str(path), "ICO")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="build AETHER ด้วย PyInstaller")
    parser.add_argument("--console", action="store_true", help="build แบบมีหน้าต่าง console")
    args = parser.parse_args(argv)

    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")

    missing = [name for name in REQUIRED if not (ROOT / name).is_file()]
    if missing:
        print(f"ขาดไฟล์ {', '.join(missing)} — รัน python tools/download_assets.py ก่อน")
        return 1
    if not list((ROOT / "assets" / "sounds").glob("*.wav")):
        print("หมายเหตุ: ยังไม่มีไฟล์เสียง (assets/sounds/*.wav) — build ได้ แต่แอปจะไม่มีเสียง")

    if make_icon(ICON):
        print(f"สร้างไอคอน {ICON.relative_to(ROOT)}")
    else:
        print("สร้างไอคอนไม่ได้ (Qt ไม่มี plugin ICO) — build ต่อโดยใช้ไอคอนค่าเริ่มต้น")

    env = dict(os.environ, AETHER_CONSOLE="1" if args.console else "0")
    command = [sys.executable, "-m", "PyInstaller", str(SPEC), "--noconfirm", "--clean"]
    print("รัน:", " ".join(command))
    result = subprocess.run(command, cwd=ROOT, env=env, check=False)
    if result.returncode != 0:
        print("build ไม่สำเร็จ — ดูรายละเอียดด้านบน")
        return result.returncode

    exe = ROOT / "dist" / "aether" / ("aether.exe" if sys.platform == "win32" else "aether")
    print(f"\nเสร็จแล้ว: {exe}")
    print(
        "ตรวจ build ด้วย:  dist\\aether\\aether.exe --check"
        if sys.platform == "win32"
        else f"ตรวจ build ด้วย:  {exe} --check"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
