# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec ของ AETHER — build แบบ one-folder

    pyinstaller packaging/aether.spec --noconfirm --clean

(หรือใช้ ``python tools/build_exe.py`` ซึ่งตรวจ asset และสร้างไอคอนให้ก่อน)

ผลลัพธ์: ``dist/aether/aether.exe`` + ``dist/aether/_internal/`` (ไลบรารีและ assets)
ไฟล์ที่ผู้ใช้แก้ได้ (``config.toml``) และเอกสารถูกคัดลอกไปไว้ข้าง exe
ตั้ง environment variable ``AETHER_CONSOLE=1`` เพื่อ build แบบมีหน้าต่าง console (ไว้ debug)
"""

import os
import shutil
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

ROOT = Path(SPECPATH).resolve().parent  # noqa: F821 — SPECPATH มาจาก PyInstaller
ASSETS = ROOT / "assets"
CONSOLE = os.environ.get("AETHER_CONSOLE", "") not in ("", "0")


def _assets(pattern: str, dest: str) -> list[tuple[str, str]]:
    return [(str(path), dest) for path in sorted(ASSETS.glob(pattern))]


models = _assets("models/*.task", "assets/models")
if not any(Path(src).name == "gesture_recognizer.task" for src, _ in models):
    raise SystemExit("ไม่พบ assets/models/gesture_recognizer.task — รัน python tools/download_assets.py ก่อน")

datas = [
    *models,
    *_assets("fonts/*.ttf", "assets/fonts"),
    *_assets("fonts/OFL-*.txt", "assets/fonts"),  # SIL OFL: ต้องแนบ license ไปกับฟอนต์
    *_assets("sounds/*.wav", "assets/sounds"),  # ไม่บังคับ — ไม่มีเสียงแอปก็ทำงานได้
    *_assets("poses/*.json", "assets/poses"),  # ชุดท่าสำเร็จรูป (tools/train_poses.py)
    (str(ASSETS / "LICENSES.md"), "assets"),
    (str(ROOT / "config.toml"), "."),  # ต้นฉบับใน bundle (ใช้เมื่อไม่มี config.toml ข้าง exe)
    # mediapipe: ไฟล์ข้อมูลที่ไม่ใช่ .py (metadata ของ tasks ฯลฯ)
    *collect_data_files("mediapipe", excludes=["**/*.py", "**/__pycache__"]),
]

# mediapipe 1.x โหลด libmediapipe.{dll,so} จาก package mediapipe.tasks.c ด้วย ctypes
binaries = collect_dynamic_libs("mediapipe")

if sys.platform == "win32":
    pynput_backends = ["pynput._util.win32", "pynput.keyboard._win32", "pynput.mouse._win32"]
    # Target Lock: uiautomation (+ DLL ใน uiautomation/bin) และ comtypes ที่มัน import ตอนใช้งาน
    pynput_backends += ["uiautomation", "comtypes", "comtypes.client"]
    binaries += collect_dynamic_libs("uiautomation")
else:
    pynput_backends = ["pynput._util.xorg", "pynput.keyboard._xorg", "pynput.mouse._xorg"]

a = Analysis(  # noqa: F821
    [str(ROOT / "packaging" / "aether_entry.py")],
    pathex=[str(ROOT / "src")],
    binaries=binaries,
    datas=datas,
    # anthropic: import ตอนใช้งานใน aether.assistant.service (ฟีเจอร์ Claude) — ระบุไว้ให้ชัด
    hiddenimports=["mediapipe.tasks.c", *pynput_backends, "anthropic", "httpx2", "jiter"],
    excludes=[
        "tkinter",
        "PySide6.QtWebEngineCore",
        "PySide6.QtWebEngineWidgets",
        "PySide6.QtWebEngineQuick",
        "PySide6.QtQuick",
        "PySide6.QtQml",
        "PySide6.QtPdf",
        "PySide6.Qt3DCore",
        "PySide6.QtCharts",
        "PySide6.QtDataVisualization",
        "PySide6.QtSql",
        "PySide6.QtTest",
        "PySide6.QtBluetooth",
        "pytest",
        "soundfile",
    ],
    noarchive=False,
)
pyz = PYZ(a.pure)  # noqa: F821

icon = ROOT / "build" / "aether.ico"
exe = EXE(  # noqa: F821
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="aether",
    console=CONSOLE,
    icon=str(icon) if icon.is_file() else None,
    upx=False,
    disable_windowed_traceback=False,
)
coll = COLLECT(  # noqa: F821
    exe,
    a.binaries,
    a.datas,
    name="aether",
    upx=False,
)

# ไฟล์ที่ผู้ใช้เปิดอ่าน/แก้ได้ ไว้ข้าง exe (ไม่ทับ config.toml ที่ผู้ใช้แก้ไว้แล้ว)
DIST = Path(DISTPATH) / "aether"  # noqa: F821
if not (DIST / "config.toml").exists():
    shutil.copy2(ROOT / "config.toml", DIST / "config.toml")
shutil.copy2(ROOT / "README.md", DIST / "README.md")
shutil.copy2(ASSETS / "LICENSES.md", DIST / "LICENSES.md")
