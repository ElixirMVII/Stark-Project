"""เสียงเอฟเฟกต์ของ HUD ผ่าน ``QSoundEffect`` (เล่นได้เฉพาะ WAV — ดู ``tools/convert_sounds.py``)

ทุกอย่างในนี้ "พังเงียบ": ไม่มีไฟล์เสียง ไม่มีอุปกรณ์เสียง หรือโหลด QtMultimedia ไม่ได้
แอปต้องทำงานต่อได้ตามปกติ แค่ไม่มีเสียง
"""

from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import QUrl

__all__ = ["SOUND_NAMES", "SoundPlayer"]

log = logging.getLogger(__name__)

# ชื่อไฟล์ใน assets/sounds/ (ไม่รวม .wav) — ตรงกับ ROLES ใน tools/convert_sounds.py
SOUND_NAMES = ("boot", "click", "release", "arm", "disarm", "swipe", "minimize", "error")


class SoundPlayer:
    """โหลดเสียงทั้งหมดครั้งเดียวตอนสร้าง แล้ว :meth:`play` ตามชื่อ (เรียกจาก UI thread)"""

    def __init__(self, sounds_dir: Path, *, enabled: bool = True, volume: float = 0.5) -> None:
        self._effects: dict[str, object] = {}
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
            effect = QSoundEffect()
            effect.setSource(QUrl.fromLocalFile(str(path)))
            effect.setVolume(volume)
            self._effects[name] = effect
        if missing:
            log.info(
                "ไม่พบไฟล์เสียง %s ใน %s — รัน `python tools/convert_sounds.py` (แอปใช้งานได้ตามปกติ แค่ไม่มีเสียงเหล่านี้)",
                ", ".join(missing),
                sounds_dir,
            )

    @property
    def available(self) -> frozenset[str]:
        return frozenset(self._effects)

    def play(self, name: str) -> None:
        effect = self._effects.get(name)
        if effect is not None:
            effect.play()  # type: ignore[attr-defined]
