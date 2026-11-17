"""เสียงเอฟเฟกต์ของ HUD ผ่าน ``QSoundEffect`` (เล่นได้เฉพาะ WAV — ดู ``tools/convert_sounds.py``)

ทุกอย่างในนี้ "พังเงียบ": ไม่มีไฟล์เสียง ไม่มีอุปกรณ์เสียง หรือโหลด QtMultimedia ไม่ได้
แอปต้องทำงานต่อได้ตามปกติ แค่ไม่มีเสียง
"""

from __future__ import annotations

import logging
import random
from pathlib import Path

from PySide6.QtCore import QUrl

__all__ = ["MAX_VARIANTS", "SOUND_NAMES", "SoundPlayer", "variant_paths"]

log = logging.getLogger(__name__)

# ชื่อไฟล์ใน assets/sounds/ (ไม่รวม .wav) — ตรงกับ ROLES ใน tools/convert_sounds.py
SOUND_NAMES = ("boot", "click", "release", "arm", "disarm", "swipe", "minimize", "error")
MAX_VARIANTS = 3  # <name>.wav, <name>_2.wav, <name>_3.wav — เล่นสลับแบบสุ่ม (M7.4)


class SoundPlayer:
    """โหลดเสียงทั้งหมดครั้งเดียวตอนสร้าง แล้ว :meth:`play` ตามชื่อ (เรียกจาก UI thread)"""

    def __init__(self, sounds_dir: Path, *, enabled: bool = True, volume: float = 0.5) -> None:
        self._effects: dict[str, list[object]] = {}
        self._last: dict[str, int] = {}
        self._rng = random.Random()
        if not enabled:
            return
        try:
            from PySide6.QtMultimedia import QSoundEffect
        except ImportError as exc:
            log.warning("QtMultimedia unavailable (%s) — sounds disabled", exc)
            return

        missing = []
        for name in SOUND_NAMES:
            path = sounds_dir / f"{name}.wav"
            if not path.is_file():
                missing.append(name)
                continue
            effects = []
            for variant in variant_paths(sounds_dir, name):
                effect = QSoundEffect()
                effect.setSource(QUrl.fromLocalFile(str(variant)))
                effect.setVolume(volume)
                effects.append(effect)
            self._effects[name] = effects
        if missing:
            log.info(
                "ไม่พบไฟล์เสียง %s ใน %s — รัน `python tools/convert_sounds.py` (แอปใช้งานได้ตามปกติ แค่ไม่มีเสียงเหล่านี้)",
                ", ".join(missing),
                sounds_dir,
            )

    @property
    def available(self) -> frozenset[str]:
        return frozenset(self._effects)

    def variants(self, name: str) -> int:
        return len(self._effects.get(name, ()))

    def play(self, name: str) -> None:
        """เล่นทันที (ภาพกับเสียงใช้ event เดียวกัน จึงตรงจังหวะกัน) — มีหลายแบบจะสุ่มไม่ซ้ำแบบเดิมติดกัน"""
        effects = self._effects.get(name)
        if not effects:
            return
        choice = self._rng.randrange(len(effects))
        if len(effects) > 1 and choice == self._last.get(name):
            choice = (choice + 1) % len(effects)
        self._last[name] = choice
        effects[choice].play()  # type: ignore[attr-defined]


def variant_paths(sounds_dir: Path, name: str) -> list[Path]:
    """ไฟล์ของเสียง ``name``: ตัวหลัก + เสียงสลับที่มีอยู่ (เรียงตามเลข)"""
    paths = [sounds_dir / f"{name}.wav"]
    paths += [sounds_dir / f"{name}_{n}.wav" for n in range(2, MAX_VARIANTS + 1)]
    return [p for p in paths if p.is_file()]
