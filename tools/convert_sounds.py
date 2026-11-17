#!/usr/bin/env python3
"""เลือกเสียงจากชุด Kenney (OGG ใน zip) แล้วแปลงเป็น WAV ตั้งชื่อตามหน้าที่ → ``assets/sounds/``

``QSoundEffect`` ของ Qt เล่นได้แค่ WAV จึงต้องแปลงก่อน ใช้งาน (จาก root ของโปรเจกต์)::

    python tools/convert_sounds.py                      # แปลงตามตารางค่าเริ่มต้น
    python tools/convert_sounds.py --list               # ดูชื่อไฟล์ .ogg ทั้งหมดในชุดที่โหลดมา
    python tools/convert_sounds.py --map click=select_003.ogg --map boot=forceField_001.ogg

สคริปต์อ่านไฟล์ zip ทุกไฟล์ใน ``assets/_downloads/`` โดยตรง (ไม่ต้องแตกไฟล์) แต่ละเสียงจะถูก
ตัดช่วงเงียบตอนต้น จำกัดความยาว (~200ms ยกเว้น boot) ใส่ fade-out และปรับความดังให้เท่ากัน

ต้องมี ``soundfile`` (dev dependency): ``pip install -e ".[dev]"``
"""

from __future__ import annotations

import argparse
import fnmatch
import io
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
DOWNLOADS_DIR = ASSETS_DIR / "_downloads"
SOUNDS_DIR = ASSETS_DIR / "sounds"

SHORT_MS = 200  # ความยาวสูงสุดของเสียงทั่วไป
BOOT_MS = 1500
PEAK = 0.7  # ≈ -3 dBFS
MAX_MEMBER_BYTES = 20 * 1024 * 1024


@dataclass(frozen=True)
class Role:
    name: str  # ชื่อไฟล์ผลลัพธ์ (<name>.wav)
    candidates: tuple[str, ...]  # pattern ชื่อไฟล์ในชุดเสียง เรียงตามลำดับที่อยากได้
    max_ms: int = SHORT_MS
    # เสียงสลับแบบสุ่ม (M7.4): ไฟล์ในชุดเดียวกันที่โทนใกล้กัน → <name>_2.wav, <name>_3.wav (ไม่มีก็ข้าม)
    variants: tuple[str, ...] = ()


# ตัวแรกของแต่ละรายการคือไฟล์ที่ตั้งใจเลือก ที่เหลือเป็นตัวสำรองถ้าชุดเสียงเปลี่ยนชื่อไฟล์
# (Interface Sounds: click/drop/confirmation/close/scroll/minimize/error,
#  Sci-Fi Sounds: computerNoise/forceField/doorOpen)
ROLES: tuple[Role, ...] = (
    Role(
        "boot",
        ("computerNoise_000.ogg", "computerNoise_*.ogg", "forceField_*.ogg", "doorOpen_*.ogg"),
        max_ms=BOOT_MS,
    ),
    Role(
        "click",
        ("click_002.ogg", "click_*.ogg", "select_*.ogg", "tick_*.ogg"),
        variants=("click_001.ogg", "click_003.ogg"),
    ),
    Role(
        "release",
        ("drop_002.ogg", "drop_*.ogg", "tick_*.ogg", "click_*.ogg"),
        variants=("drop_001.ogg", "drop_003.ogg"),
    ),
    Role("arm", ("confirmation_001.ogg", "confirmation_*.ogg", "maximize_*.ogg", "open_*.ogg")),
    Role("disarm", ("close_002.ogg", "close_*.ogg", "back_*.ogg")),
    Role(
        "swipe",
        ("scroll_003.ogg", "scroll_*.ogg", "switch_*.ogg"),
        variants=("scroll_001.ogg", "scroll_005.ogg"),
    ),
    Role(
        "minimize",
        ("minimize_004.ogg", "minimize_*.ogg", "close_*.ogg"),
        variants=("minimize_002.ogg",),
    ),
    Role("error", ("error_004.ogg", "error_*.ogg", "glitch_*.ogg")),
)
ROLE_NAMES = tuple(role.name for role in ROLES)


# --------------------------------------------------------------------------- #
# Audio processing (pure numpy — มี unit test)
# --------------------------------------------------------------------------- #


def to_mono(samples: np.ndarray) -> np.ndarray:
    """(frames,) หรือ (frames, channels) → (frames,) float32"""
    samples = np.asarray(samples, dtype=np.float32)
    if samples.ndim == 2:
        samples = samples.mean(axis=1)
    return samples


def trim_leading_silence(
    samples: np.ndarray, sample_rate: int, threshold: float = 0.02, keep_ms: float = 2.0
) -> np.ndarray:
    """ตัดช่วงเงียบตอนต้น (เสียง UI ต้องดังทันทีที่ trigger)

    ``threshold`` เทียบกับ peak ของไฟล์ และเหลือ ``keep_ms`` ก่อนจุดเริ่มเสียงไว้กันเสียงสะดุด
    """
    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    if peak <= 0.0:
        return samples
    loud = np.flatnonzero(np.abs(samples) >= threshold * peak)
    start = max(0, int(loud[0]) - int(sample_rate * keep_ms / 1000.0))
    return samples[start:]


def limit_duration(
    samples: np.ndarray, sample_rate: int, max_ms: float, fade_ms: float | None = None
) -> np.ndarray:
    """ตัดให้ยาวไม่เกิน ``max_ms`` แล้ว fade-out ช่วงท้าย (เฉพาะเมื่อมีการตัด)"""
    max_len = int(sample_rate * max_ms / 1000.0)
    if len(samples) <= max_len:
        return samples
    if fade_ms is None:
        fade_ms = max_ms * 0.15
    out = samples[:max_len].copy()
    fade_len = min(max_len, int(sample_rate * fade_ms / 1000.0))
    if fade_len > 0:
        out[-fade_len:] *= np.linspace(1.0, 0.0, fade_len, dtype=np.float32)
    return out


def normalize_peak(samples: np.ndarray, peak: float = PEAK) -> np.ndarray:
    """ปรับให้ค่าสูงสุดเท่ากับ ``peak`` ทุกไฟล์ จะได้ดังพอๆ กัน"""
    current = float(np.max(np.abs(samples))) if samples.size else 0.0
    if current <= 0.0:
        return samples
    return samples * (peak / current)


def prepare(samples: np.ndarray, sample_rate: int, max_ms: float) -> np.ndarray:
    """mono → ตัดเงียบต้นไฟล์ → จำกัดความยาว → ปรับความดัง"""
    mono = to_mono(samples)
    mono = trim_leading_silence(mono, sample_rate)
    mono = limit_duration(mono, sample_rate, max_ms)
    return normalize_peak(mono)


# --------------------------------------------------------------------------- #
# Sound packs
# --------------------------------------------------------------------------- #


def index_packs(downloads_dir: Path) -> dict[str, tuple[Path, str]]:
    """ชื่อไฟล์ .ogg (ตัวพิมพ์เล็ก) → (zip, ชื่อ member) จาก zip ทุกไฟล์ในโฟลเดอร์"""
    index: dict[str, tuple[Path, str]] = {}
    if not downloads_dir.is_dir():
        return index
    for pack in sorted(downloads_dir.glob("*.zip")):
        try:
            with zipfile.ZipFile(pack) as archive:
                for info in archive.infolist():
                    name = info.filename.rsplit("/", 1)[-1].lower()
                    if (
                        name.endswith(".ogg")
                        and not info.is_dir()
                        and info.file_size <= MAX_MEMBER_BYTES
                    ):
                        index.setdefault(name, (pack, info.filename))
        except zipfile.BadZipFile:
            print(f"ข้าม {pack.name}: ไม่ใช่ไฟล์ zip ที่ถูกต้อง", file=sys.stderr)
    return index


def choose(role: Role, names: list[str]) -> str | None:
    """ชื่อไฟล์แรกที่ตรงกับ pattern ของ role (ไล่ตามลำดับ pattern แล้วตามชื่อ)"""
    for pattern in role.candidates:
        matches = sorted(n for n in names if fnmatch.fnmatchcase(n, pattern.lower()))
        if matches:
            return matches[0]
    return None


def parse_overrides(items: list[str]) -> dict[str, str]:
    """``["click=select_003.ogg"]`` → ``{"click": "select_003.ogg"}``"""
    overrides: dict[str, str] = {}
    for item in items:
        role, sep, filename = item.partition("=")
        role, filename = role.strip(), filename.strip()
        if not sep or not filename:
            raise ValueError(f"--map ต้องอยู่ในรูป ROLE=FILE (ได้ {item!r})")
        if role not in ROLE_NAMES:
            raise ValueError(f"ไม่รู้จัก role {role!r} (ที่ใช้ได้: {', '.join(ROLE_NAMES)})")
        overrides[role] = filename
    return overrides


def read_member(pack: Path, member: str) -> bytes:
    with zipfile.ZipFile(pack) as archive:
        return archive.read(member)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def convert(
    downloads_dir: Path, sounds_dir: Path, overrides: dict[str, str]
) -> tuple[list[str], list[str]]:
    """แปลงทุก role คืน (บรรทัดรายงาน, role ที่หาไฟล์ไม่ได้)"""
    import soundfile as sf

    index = index_packs(downloads_dir)
    names = sorted(index)
    report: list[str] = []
    missing: list[str] = []
    sounds_dir.mkdir(parents=True, exist_ok=True)

    for role in ROLES:
        wanted = overrides.get(role.name)
        name = wanted.lower() if wanted else choose(role, names)
        if name is None or name not in index:
            missing.append(role.name)
            tried = wanted or ", ".join(role.candidates)
            report.append(f"  [ขาด]    {role.name}.wav — ไม่พบ {tried}")
            continue

        pack, member = index[name]
        samples, sample_rate = sf.read(io.BytesIO(read_member(pack, member)), dtype="float32")
        out = prepare(samples, sample_rate, role.max_ms)
        sf.write(sounds_dir / f"{role.name}.wav", out, sample_rate, subtype="PCM_16")
        duration_ms = len(out) * 1000.0 / sample_rate
        note = "" if wanted or name == role.candidates[0].lower() else " (ตัวสำรอง)"
        report.append(
            f"  [แปลงแล้ว] {role.name}.wav ← {name} [{pack.name}] {duration_ms:.0f} ms{note}"
        )
        if wanted:
            continue  # เลือกไฟล์เองด้วย --map: ไม่ใส่เสียงสลับ
        for number, variant in enumerate(role.variants, start=2):
            key = variant.lower()
            if key not in index or key == name:
                continue
            pack, member = index[key]
            samples, sample_rate = sf.read(io.BytesIO(read_member(pack, member)), dtype="float32")
            out = prepare(samples, sample_rate, role.max_ms)
            sf.write(sounds_dir / f"{role.name}_{number}.wav", out, sample_rate, subtype="PCM_16")
            report.append(f"  [แปลงแล้ว] {role.name}_{number}.wav ← {key} (เสียงสลับ)")
    return report, missing


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="แปลงเสียง Kenney (OGG) เป็น WAV ตามหน้าที่")
    parser.add_argument("--list", action="store_true", help="แสดงชื่อไฟล์ .ogg ทั้งหมดที่มีให้เลือก")
    parser.add_argument(
        "--map",
        action="append",
        default=[],
        metavar="ROLE=FILE",
        help=f"เลือกไฟล์เองสำหรับบางเสียง (ROLE: {', '.join(ROLE_NAMES)})",
    )
    parser.add_argument("--downloads-dir", type=Path, default=DOWNLOADS_DIR, help=argparse.SUPPRESS)
    parser.add_argument("--sounds-dir", type=Path, default=SOUNDS_DIR, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")

    try:
        overrides = parse_overrides(args.map)
    except ValueError as exc:
        parser.error(str(exc))

    index = index_packs(args.downloads_dir)
    if not index:
        print(
            f"ไม่พบไฟล์เสียง .ogg ใน zip ใดๆ ที่ {args.downloads_dir}\n"
            "รัน `python tools/download_assets.py` เพื่อดูวิธีโหลดชุดเสียง Kenney ก่อน"
        )
        return 1

    if args.list:
        by_pack: dict[str, list[str]] = {}
        for name, (pack, _) in index.items():
            by_pack.setdefault(pack.name, []).append(name)
        for pack_name, pack_names in sorted(by_pack.items()):
            print(f"{pack_name} ({len(pack_names)} ไฟล์)")
            for name in sorted(pack_names):
                print(f"  {name}")
        return 0

    try:
        report, missing = convert(args.downloads_dir, args.sounds_dir, overrides)
    except ImportError:
        print('ต้องติดตั้ง soundfile ก่อน: pip install -e ".[dev]"')
        return 1

    print(f"เสียง → {args.sounds_dir}\n")
    print("\n".join(report))
    if missing:
        print(
            f"\nยังขาด {len(missing)} เสียง: {', '.join(missing)}\n"
            "ดูชื่อไฟล์ที่มีด้วย `python tools/convert_sounds.py --list` "
            "แล้วเลือกเองด้วย `--map ROLE=FILE`"
        )
        return 1
    print("\nเสียงครบแล้ว ลองเปิดฟังใน assets/sounds/ ถ้าอยากเปลี่ยนเสียงไหนใช้ `--map ROLE=FILE`")
    return 0


if __name__ == "__main__":
    sys.exit(main())
