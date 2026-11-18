"""เครื่องดนตรีอากาศ (theremin) — ความสูงของมือ = ระดับเสียง (pure numpy, ไม่มี Qt)

- จีบ = เล่นโน้ต (ปล่อย = เงียบ แบบกดคีย์) · มือสูง = เสียงสูง ช่วง 2 octave (A3 220 Hz → A5 880 Hz)
- มือซ้าย-ขวาในแนวนอน = ความลึกของ vibrato · ถ้ามีมือที่สอง ความสูงของมือนั้น = ความดัง
- ``scale`` = ``"pentatonic"`` (ค่าเริ่มต้น เล่นยังไงก็ไม่เพี้ยน) / ``"major"`` / ``"minor"`` / ``"free"`` (ไล่เสียงต่อเนื่อง)

:class:`Voice` สร้างคลื่นเสียงแบบต่อเฟสระหว่างบล็อก (ไม่มีเสียงแตก "ป๊อก") เลื่อนความถี่/ความดังแบบนุ่ม
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .input import StudioHand

__all__ = [
    "NOTE_NAMES",
    "SCALES",
    "ThereminController",
    "Voice",
    "frequency_for",
    "note_name",
]

SAMPLE_RATE = 48_000
LOW_HZ = 220.0  # A3
OCTAVES = 2.0
SCALES: dict[str, tuple[int, ...] | None] = {
    "pentatonic": (0, 2, 4, 7, 9),  # A C# E … (major pentatonic เริ่มที่ A)
    "major": (0, 2, 4, 5, 7, 9, 11),
    "minor": (0, 2, 3, 5, 7, 8, 10),
    "free": None,
}
NOTE_NAMES = ("A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#")
HARMONICS = ((1.0, 1.0), (2.0, 0.18), (3.0, 0.06))  # (ตัวคูณความถี่, ความดัง) — อุ่นกว่า sine ล้วน
GLIDE_SECONDS = 0.045  # เลื่อนความถี่ไปหาโน้ตใหม่
FADE_SECONDS = 0.03  # ขึ้น/ลงของความดัง
VIBRATO_HZ = 5.5
MAX_VIBRATO = 0.012  # สัดส่วนของความถี่ (≈ 20 cent)


def frequency_for(height: float, scale: str = "pentatonic", low: float = LOW_HZ) -> float:
    """``height`` 0 = ล่างสุด … 1 = บนสุด → ความถี่ (Hz) ปัดเข้าโน้ตใน ``scale``"""
    height = min(max(height, 0.0), 1.0)
    semitones = height * 12.0 * OCTAVES
    degrees = SCALES[scale]
    if degrees is not None:
        candidates = [octave * 12 + d for octave in range(int(OCTAVES) + 1) for d in degrees]
        candidates = [c for c in candidates if c <= 12 * OCTAVES]
        semitones = min(candidates, key=lambda c: abs(c - semitones))
    return low * 2.0 ** (semitones / 12.0)


def note_name(frequency: float) -> str:
    """ชื่อโน้ตที่ใกล้ที่สุด เช่น 440 → ``"A4"``"""
    if frequency <= 0:
        return "—"
    semitones = round(12.0 * math.log2(frequency / 440.0))
    name = NOTE_NAMES[semitones % 12]
    octave = 4 + (semitones + 9) // 12  # อ็อกเทฟเปลี่ยนที่โน้ต C
    return f"{name}{octave}"


class Voice:
    """เสียงหนึ่งเสียง: ตั้ง ``frequency`` / ``amplitude`` / ``vibrato`` เป้าหมาย แล้ว :meth:`render` เป็นบล็อก"""

    def __init__(self, sample_rate: int = SAMPLE_RATE) -> None:
        self.sample_rate = sample_rate
        self.frequency = LOW_HZ  # เป้าหมาย
        self.amplitude = 0.0
        self.vibrato = 0.0  # 0–1
        self._freq = LOW_HZ  # ค่าปัจจุบัน (เลื่อนเข้าหาเป้าหมาย)
        self._amp = 0.0
        self._phase = 0.0  # รอบ (0–1) ของความถี่หลัก
        self._vib_phase = 0.0

    @property
    def current_frequency(self) -> float:
        return self._freq

    @property
    def current_amplitude(self) -> float:
        return self._amp

    def render(self, frames: int) -> np.ndarray:
        """``frames`` ตัวอย่างเสียง float32 ช่วง −1…1 (mono)"""
        if frames <= 0:
            return np.zeros(0, dtype=np.float32)
        sr = self.sample_rate
        n = np.arange(1, frames + 1, dtype=np.float64)
        glide = 1.0 - math.exp(-1.0 / (GLIDE_SECONDS * sr))
        fade = 1.0 - math.exp(-1.0 / (FADE_SECONDS * sr))
        freq = self.frequency + (self._freq - self.frequency) * (1.0 - glide) ** n
        amp = self.amplitude + (self._amp - self.amplitude) * (1.0 - fade) ** n
        vib_phase = self._vib_phase + n * VIBRATO_HZ / sr
        freq = freq * (1.0 + MAX_VIBRATO * self.vibrato * np.sin(math.tau * vib_phase))
        phase = self._phase + np.cumsum(freq) / sr
        wave = np.zeros(frames, dtype=np.float64)
        for multiple, gain in HARMONICS:
            wave += gain * np.sin(math.tau * multiple * phase)
        wave *= amp / sum(gain for _, gain in HARMONICS)
        self._phase = float(phase[-1] % 1.0)
        self._vib_phase = float(vib_phase[-1] % 1.0)
        self._freq = float(self.frequency + (self._freq - self.frequency) * (1.0 - glide) ** frames)
        self._amp = float(amp[-1])
        return wave.astype(np.float32)


@dataclass(frozen=True, slots=True)
class ThereminState:
    playing: bool
    frequency: float
    amplitude: float
    note: str
    height: float  # 0–1 (ล่าง → บน) ของมือที่เล่น


class ThereminController:
    def __init__(self, voice: Voice, *, scale: str = "pentatonic", volume: float = 0.6) -> None:
        if scale not in SCALES:
            raise ValueError(f"scale ต้องเป็นหนึ่งใน {', '.join(SCALES)}")
        self.voice = voice
        self.scale = scale
        self.volume = volume
        self.state = ThereminState(False, voice.frequency, 0.0, "—", 0.0)

    def update(self, hands: tuple[StudioHand, ...], margin: float = 0.0) -> ThereminState:
        """``margin`` = ขอบบน/ล่างของผืนผ้าใบ (สัดส่วน) ที่ไม่ได้ใช้เป็นช่วงเสียง (หน้าจอเว้นที่ให้ป้ายโน้ต)"""
        player = hands[0] if hands else None
        if player is None:
            self.voice.amplitude = 0.0
            self.state = ThereminState(False, self.voice.frequency, 0.0, "—", self.state.height)
            return self.state
        height = min(max((1.0 - player.y - margin) / max(1.0 - 2.0 * margin, 1e-6), 0.0), 1.0)
        frequency = frequency_for(height, self.scale)
        loudness = self.volume
        if len(hands) > 1:  # มือที่สองคุมความดัง (สูง = ดัง)
            loudness *= min(max(1.0 - hands[1].y, 0.0), 1.0)
        playing = player.pinch
        self.voice.frequency = frequency
        self.voice.vibrato = min(max(player.x, 0.0), 1.0)
        self.voice.amplitude = loudness if playing else 0.0
        self.state = ThereminState(
            playing, frequency, self.voice.amplitude, note_name(frequency), height
        )
        return self.state
