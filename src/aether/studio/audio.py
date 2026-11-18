"""ส่งเสียงจาก :class:`~aether.studio.theremin.Voice` ออกลำโพงด้วย ``QAudioSink`` (push mode)

"พังเงียบ" แบบเดียวกับ :mod:`aether.sound`: ไม่มี QtMultimedia / ไม่มีอุปกรณ์เสียง / เปิดไม่ได้ → ``available``
เป็น ``False`` และ Studio ใช้งานต่อได้ (แค่ไม่มีเสียง)
"""

from __future__ import annotations

import logging

import numpy as np

from .theremin import Voice

__all__ = ["AudioOut"]

log = logging.getLogger(__name__)

BUFFER_SECONDS = 0.08  # บัฟเฟอร์สั้น = เสียงตามมือเร็ว แต่ต้องเติมทันทุก ~10 ms
MAX_BLOCK_SECONDS = 0.05


class AudioOut:
    def __init__(self, voice: Voice) -> None:
        self.voice = voice
        self.available = False
        self._sink = None
        self._io = None
        self._int16 = False
        try:
            from PySide6.QtMultimedia import QAudioFormat, QAudioSink, QMediaDevices
        except ImportError as exc:
            log.info("QtMultimedia unavailable (%s) — theremin is silent", exc)
            return
        try:
            device = QMediaDevices.defaultAudioOutput()
            if device.isNull():
                log.info("no audio output device — theremin is silent")
                return
            fmt = QAudioFormat()
            fmt.setChannelCount(1)
            fmt.setSampleRate(voice.sample_rate)
            fmt.setSampleFormat(QAudioFormat.SampleFormat.Float)
            if not device.isFormatSupported(fmt):
                preferred = device.preferredFormat()
                fmt.setSampleRate(preferred.sampleRate())
                fmt.setSampleFormat(QAudioFormat.SampleFormat.Int16)
                self._int16 = True
                voice.sample_rate = preferred.sampleRate()
            self._bytes_per_frame = 2 if self._int16 else 4
            self._sink = QAudioSink(device, fmt)
            self._sink.setBufferSize(
                int(voice.sample_rate * BUFFER_SECONDS) * self._bytes_per_frame
            )
            self.available = True
        except Exception:  # เสียงเป็นของเสริม
            log.warning("cannot open audio output — theremin is silent", exc_info=True)

    @property
    def running(self) -> bool:
        return self._io is not None

    def start(self) -> None:
        if self.available and self._io is None:
            self._io = self._sink.start()

    def pump(self) -> int:
        """เติมเสียงเท่าที่บัฟเฟอร์ว่าง (เรียกทุก ~10–16 ms) คืนจำนวน frame ที่เขียน"""
        if self._io is None:
            return 0
        free = self._sink.bytesFree() // self._bytes_per_frame
        frames = min(free, int(self.voice.sample_rate * MAX_BLOCK_SECONDS))
        if frames <= 0:
            return 0
        samples = self.voice.render(frames)
        if self._int16:
            data = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
        else:
            data = samples.astype(np.float32).tobytes()
        self._io.write(data)
        return frames

    def stop(self) -> None:
        if self._sink is not None and self._io is not None:
            self._sink.stop()
        self._io = None
