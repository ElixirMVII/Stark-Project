#!/usr/bin/env python3
"""โหลด asset ทั้งหมดของ AETHER ลง ``assets/`` แล้วตรวจว่าไฟล์ครบ

ใช้งาน (จาก root ของโปรเจกต์)::

    python tools/download_assets.py            # โหลดเฉพาะไฟล์ที่ยังไม่มี
    python tools/download_assets.py --force    # โหลดใหม่ทั้งหมด
    python tools/download_assets.py --check    # ตรวจอย่างเดียว ไม่ต่อ network

ใช้แค่ standard library ถ้าไฟล์ไหนโหลดอัตโนมัติไม่ได้ สคริปต์จะบอกว่าต้องโหลดเอง
จากที่ไหนและวางไว้ตรงไหน ที่มาและ license ของทุกไฟล์อยู่ใน ``assets/LICENSES.md``

หมายเหตุ: ชุดเสียงของ Kenney ต้องโหลดเองผ่านเบราว์เซอร์เสมอ เพราะหน้าเว็บให้กดผ่าน
หน้าต่างบริจาคแล้วออกลิงก์แบบ signed ให้ทีละครั้ง ไม่มีลิงก์ตรงให้สคริปต์โหลด
(ดู ``docs/DECISIONS.md``)
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
USER_AGENT = "aether-asset-downloader/1.0 (python-urllib)"
TIMEOUT_S = 30
MAX_BYTES = 200 * 1024 * 1024  # กันไฟล์ใหญ่ผิดปกติ
CHUNK = 64 * 1024

_GCS = "https://storage.googleapis.com/mediapipe-models"
_GFONTS = "https://raw.githubusercontent.com/google/fonts/main/ofl"
_GESTURE_DOC = "https://ai.google.dev/edge/mediapipe/solutions/vision/gesture_recognizer"

# font ขึ้นต้นด้วย byte อะไร (ใช้ตรวจว่าไม่ได้โหลดหน้า error ของเว็บมาแทน)
_FONT_MAGIC = (b"\x00\x01\x00\x00", b"OTTO", b"true")
KINDS = ("zip", "ttf", "text")  # .task ของ MediaPipe เป็น zip (มี byte นำหน้า 2 ตัว)


@dataclass(frozen=True)
class Asset:
    dest: str  # path ใต้ assets/
    kind: str  # หนึ่งใน KINDS
    min_size: int  # ขนาดขั้นต่ำ (bytes) ที่ถือว่าไฟล์สมบูรณ์
    url: str | None = None  # ลิงก์ตรง (None = ต้องโหลดเองผ่านเบราว์เซอร์)
    alt_name: str | None = None  # regex (search) ของชื่อไฟล์อื่นในโฟลเดอร์เดียวกันที่ใช้แทนได้
    required: bool = True
    manual: str = ""  # คำแนะนำถ้าต้องโหลดเอง


ASSETS: tuple[Asset, ...] = (
    Asset(
        dest="models/gesture_recognizer.task",
        kind="zip",
        min_size=1_000_000,
        url=f"{_GCS}/gesture_recognizer/gesture_recognizer/float16/latest/gesture_recognizer.task",
        manual=f"หาลิงก์โมเดลล่าสุดได้ที่หน้า Gesture Recognizer: {_GESTURE_DOC}",
    ),
    Asset(
        dest="models/hand_landmarker.task",
        kind="zip",
        min_size=1_000_000,
        url=f"{_GCS}/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
        required=False,
        manual="ไฟล์สำรอง ไม่บังคับ",
    ),
    Asset(
        dest="fonts/Orbitron.ttf",
        kind="ttf",
        min_size=10_000,
        url=f"{_GFONTS}/orbitron/Orbitron%5Bwght%5D.ttf",
        manual="โหลดจาก https://fonts.google.com/specimen/Orbitron แล้วเปลี่ยนชื่อไฟล์ .ttf",
    ),
    Asset(
        dest="fonts/OFL-Orbitron.txt",
        kind="text",
        min_size=1_000,
        url=f"{_GFONTS}/orbitron/OFL.txt",
        manual="ไฟล์ license (OFL.txt) อยู่ใน zip ที่โหลดจาก Google Fonts",
    ),
    Asset(
        dest="fonts/ShareTechMono-Regular.ttf",
        kind="ttf",
        min_size=10_000,
        url=f"{_GFONTS}/sharetechmono/ShareTechMono-Regular.ttf",
        manual="โหลดจาก https://fonts.google.com/specimen/Share+Tech+Mono",
    ),
    Asset(
        dest="fonts/OFL-ShareTechMono.txt",
        kind="text",
        min_size=1_000,
        url=f"{_GFONTS}/sharetechmono/OFL.txt",
        manual="ไฟล์ license (OFL.txt) อยู่ใน zip ที่โหลดจาก Google Fonts",
    ),
    Asset(
        dest="_downloads/kenney_interface-sounds.zip",
        kind="zip",
        min_size=100_000,
        alt_name=r"interface[-_ ]?sounds.*\.zip$",
        manual=(
            "เปิด https://kenney.nl/assets/interface-sounds ในเบราว์เซอร์ กด Download "
            "(ข้ามหน้าบริจาคได้) แล้ววางไฟล์ zip โดยไม่ต้องแตกไฟล์"
        ),
    ),
    Asset(
        dest="_downloads/kenney_sci-fi-sounds.zip",
        kind="zip",
        min_size=100_000,
        alt_name=r"sci[-_ ]?fi[-_ ]?sounds.*\.zip$",
        manual=(
            "เปิด https://kenney.nl/assets/sci-fi-sounds ในเบราว์เซอร์ กด Download "
            "(ข้ามหน้าบริจาคได้) แล้ววางไฟล์ zip โดยไม่ต้องแตกไฟล์"
        ),
    ),
)


class DownloadError(Exception):
    pass


# --------------------------------------------------------------------------- #
# Pure helpers (มี unit test)
# --------------------------------------------------------------------------- #


def verify_file(path: Path, kind: str, min_size: int) -> str | None:
    """คืน ``None`` ถ้าไฟล์ใช้ได้ ไม่งั้นคืนเหตุผลที่ใช้ไม่ได้"""
    if kind not in KINDS:
        raise ValueError(f"ไม่รู้จักชนิดไฟล์: {kind}")
    if not path.is_file():
        return "ไม่พบไฟล์"
    size = path.stat().st_size
    if size < min_size:
        return f"ไฟล์เล็กผิดปกติ ({size:,} bytes)"
    bad_format = "รูปแบบไฟล์ไม่ถูกต้อง (อาจเป็นหน้า error ของเว็บ)"
    if kind == "zip" and not zipfile.is_zipfile(path):
        return bad_format
    if kind == "ttf":
        with path.open("rb") as fh:
            if not fh.read(4).startswith(_FONT_MAGIC):
                return bad_format
    return None


def locate(asset: Asset, assets_dir: Path) -> tuple[Path, str | None]:
    """หาไฟล์ของ asset (ชื่อมาตรฐาน หรือชื่ออื่นที่ตรง ``alt_name``)

    คืน (path, ปัญหา) โดยปัญหาเป็น ``None`` ถ้าไฟล์ใช้ได้
    """
    dest = assets_dir / asset.dest
    problem = verify_file(dest, asset.kind, asset.min_size)
    if problem is None or asset.alt_name is None or not dest.parent.is_dir():
        return dest, problem
    pattern = re.compile(asset.alt_name, flags=re.IGNORECASE)
    for candidate in sorted(dest.parent.iterdir()):
        if pattern.search(candidate.name) and (
            verify_file(candidate, asset.kind, asset.min_size) is None
        ):
            return candidate, None
    return dest, problem


def human_size(n: int) -> str:
    if n >= 1024 * 1024:
        return f"{n / (1024 * 1024):.1f} MB"
    return f"{n / 1024:.0f} KB"


# --------------------------------------------------------------------------- #
# Network
# --------------------------------------------------------------------------- #


def _open(url: str):
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        return urllib.request.urlopen(request, timeout=TIMEOUT_S)
    except urllib.error.HTTPError as exc:
        raise DownloadError(f"HTTP {exc.code} จาก {url}") from exc
    except (urllib.error.URLError, OSError) as exc:
        reason = getattr(exc, "reason", exc)
        raise DownloadError(f"ต่อ {url} ไม่ได้ ({reason})") from exc


def fetch_file(url: str, dest: Path) -> None:
    """โหลดลงไฟล์ชั่วคราว (.part) แล้วค่อยย้ายเป็นชื่อจริง จะได้ไม่เหลือไฟล์ครึ่งๆ กลางๆ"""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    try:
        with _open(url) as response, part.open("wb") as out:
            total = 0
            while chunk := response.read(CHUNK):
                total += len(chunk)
                if total > MAX_BYTES:
                    raise DownloadError(f"ไฟล์ใหญ่เกิน {human_size(MAX_BYTES)}: {url}")
                out.write(chunk)
        os.replace(part, dest)
    except OSError as exc:
        raise DownloadError(f"โหลด {url} ไม่สำเร็จ ({exc})") from exc
    finally:
        part.unlink(missing_ok=True)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def process(asset: Asset, assets_dir: Path, *, force: bool, check_only: bool) -> tuple[str, str]:
    """คืน (สถานะ, รายละเอียด) โดยสถานะเป็น ``ok`` | ``downloaded`` | ``missing``"""
    found, problem = locate(asset, assets_dir)
    if asset.url is None:  # โหลดเองผ่านเบราว์เซอร์เท่านั้น
        if problem is None:
            return "ok", f"{found.name}, {human_size(found.stat().st_size)}"
        return "missing", f"{problem} — ต้องโหลดเองผ่านเบราว์เซอร์"
    if problem is None and not force:
        return "ok", human_size(found.stat().st_size)
    if check_only:
        return "missing", problem or "ไม่ได้ตรวจ"

    dest = assets_dir / asset.dest
    try:
        fetch_file(asset.url, dest)
    except DownloadError as exc:
        return "missing", str(exc)

    problem = verify_file(dest, asset.kind, asset.min_size)
    if problem is not None:
        dest.unlink(missing_ok=True)
        return "missing", problem
    return "downloaded", human_size(dest.stat().st_size)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="โหลด asset ของ AETHER ลง assets/")
    parser.add_argument("--force", action="store_true", help="โหลดใหม่แม้มีไฟล์อยู่แล้ว")
    parser.add_argument("--check", action="store_true", help="ตรวจไฟล์อย่างเดียว ไม่ต่อ network")
    parser.add_argument("--assets-dir", type=Path, default=ASSETS_DIR, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")

    labels = {"ok": "[มีแล้ว]", "downloaded": "[โหลดแล้ว]", "missing": "[ขาด]"}
    missing: list[tuple[Asset, str]] = []

    print(f"asset → {args.assets_dir}\n")
    for asset in ASSETS:
        status, detail = process(asset, args.assets_dir, force=args.force, check_only=args.check)
        optional = "" if asset.required else " (ไม่บังคับ)"
        print(f"  {labels[status]:<11} {asset.dest}{optional} — {detail}")
        if status == "missing":
            missing.append((asset, detail))

    required_missing = [(a, d) for a, d in missing if a.required]
    if missing:
        print("\nไฟล์ที่ต้องโหลดเอง:")
        for asset, _ in missing:
            optional = "" if asset.required else " (ไม่บังคับ)"
            print(f"\n  • {asset.dest}{optional}")
            print(f"    {asset.manual}")
            print(f"    วางไว้ที่: {args.assets_dir / asset.dest}")
        print("\nวางไฟล์แล้วรัน `python tools/download_assets.py --check` เพื่อตรวจอีกครั้ง")

    if required_missing:
        print(f"\nยังขาดไฟล์ที่จำเป็น {len(required_missing)} ไฟล์")
        if all(asset.dest.startswith("_downloads/") for asset, _ in required_missing):
            print("(ที่ขาดคือชุดเสียงเอฟเฟกต์เท่านั้น — แอปรันได้แล้ว แค่ยังไม่มีเสียง)")
        return 1

    print("\nasset ที่จำเป็นครบแล้ว ขั้นต่อไป: python tools/convert_sounds.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
