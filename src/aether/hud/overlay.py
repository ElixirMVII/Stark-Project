"""หน้าต่าง overlay ของ HUD: โปร่งใส เต็มจอหลัก อยู่บนสุด และ **คลิกทะลุได้ 100%**

- ไม่รับ input ใดๆ (``WindowTransparentForInput``) ไม่แย่ง focus ไม่โผล่ใน taskbar
- พิกัดจาก engine เป็น physical px → แปลงเป็น logical px ด้วย ``devicePixelRatio`` ก่อนวาด
- วาดด้วย QPainter และสั่ง ``update()`` เฉพาะกรอบของชิ้นที่เปลี่ยน (รอบ reticle และแผงต่างๆ)
  ไม่วาดใหม่ทั้งจอทุกเฟรม ถ้าไม่มีอะไรขยับก็ไม่วาดเลย
"""

from __future__ import annotations

import dataclasses
import logging
import time
from typing import Protocol

from PySide6.QtCore import QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import QGuiApplication, QPainter
from PySide6.QtWidgets import QWidget

from ..camera import FramePacket
from ..config import Config, assets_dir
from ..gestures import EventKind, GestureEvent
from ..mapping import ScreenGeometry, physical_to_logical
from ..sound import SoundPlayer
from .boot import BootSequence
from .panels import HandRadar, StatusBadge, TelemetryPanel, TelemetrySampler
from .radial import ChargeRing, Radial
from .reticle import Reticle
from .theme import Theme

__all__ = ["OVERLAY_FLAGS", "HudOverlay"]

log = logging.getLogger(__name__)

OVERLAY_FLAGS = (
    Qt.WindowType.FramelessWindowHint
    | Qt.WindowType.WindowStaysOnTopHint
    | Qt.WindowType.Tool  # ไม่โผล่ใน taskbar / Alt+Tab
    | Qt.WindowType.WindowTransparentForInput  # คลิกทะลุ
    | Qt.WindowType.WindowDoesNotAcceptFocus
    | Qt.WindowType.NoDropShadowWindowHint
)

FRAME_INTERVAL_MS = 16  # ~60 fps
TELEMETRY_INTERVAL_MS = 500
_PAD = 3.0  # เผื่อขอบ anti-aliasing รอบกรอบที่สั่งวาดใหม่


class _Element(Protocol):
    def rect(self) -> QRectF: ...

    def paint(self, painter: QPainter, now: float) -> None: ...


class HudOverlay(QWidget):
    """เรียก method ทั้งหมดจาก UI thread"""

    def __init__(
        self,
        config: Config,
        screen: ScreenGeometry,
        *,
        dry_run: bool = False,
        theme: Theme | None = None,
        sounds: SoundPlayer | None = None,
        sampler: TelemetrySampler | None = None,
        device_pixel_ratio: float | None = None,
        clock=time.perf_counter,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent, OVERLAY_FLAGS)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setWindowTitle("AETHER HUD")

        self._config = config
        self._screen = screen
        self._clock = clock
        self._theme = theme or Theme.from_config(config.hud, assets_dir() / "fonts")
        self._sounds = sounds
        self._sampler = sampler

        qt_screen = QGuiApplication.primaryScreen()
        self._dpr = device_pixel_ratio or (qt_screen.devicePixelRatio() if qt_screen else 1.0)
        if qt_screen is not None:
            # เตี้ยกว่าจอ 1px: หน้าต่างที่คลุมจอพอดีอาจถูก Windows มองเป็นแอป fullscreen
            self.setGeometry(qt_screen.geometry().adjusted(0, 0, 0, -1))

        self._reticle = Reticle(self._theme)
        self._boot = BootSequence(self._theme)
        actions = dataclasses.asdict(config.actions)
        self._radial = Radial(self._theme, actions)
        self._charge = ChargeRing(self._theme, actions)
        self._badge = StatusBadge(self._theme, dry_run=dry_run)
        self._telemetry = TelemetryPanel(self._theme) if config.hud.show_telemetry else None
        self._radar = (
            HandRadar(self._theme, config.tracking.dominant_hand) if config.hud.show_radar else None
        )
        # ลำดับการวาด: ล่าง → บน
        self._elements: dict[str, _Element] = {"boot": self._boot, "badge": self._badge}
        if self._telemetry is not None:
            self._elements["telemetry"] = self._telemetry
        if self._radar is not None:
            self._elements["radar"] = self._radar
        self._elements["radial"] = self._radial
        self._elements["charge"] = self._charge
        self._elements["reticle"] = self._reticle
        self._painted: dict[str, QRectF] = {}

        self._fps = 0.0
        self._latency_ms = 0.0
        self._layout_panels()

        self._frame_timer = QTimer(self)
        self._frame_timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._frame_timer.setInterval(FRAME_INTERVAL_MS)
        self._frame_timer.timeout.connect(self._tick)

        self._telemetry_timer = QTimer(self)
        self._telemetry_timer.setInterval(TELEMETRY_INTERVAL_MS)
        self._telemetry_timer.timeout.connect(self._sample_telemetry)

    # ---- lifecycle ------------------------------------------------------ #

    def start(self, *, boot: bool = True) -> None:
        """แสดง overlay และเล่น boot sequence"""
        self.show()
        now = self._clock()
        if boot:
            self._boot.start(now, QPointF(self.width() / 2.0, self.height() / 2.0))
            self._play("boot")
        self._sample_telemetry()
        self._frame_timer.start()
        if self._telemetry is not None:
            self._telemetry_timer.start()
        self.update()

    def stop(self) -> None:
        self._frame_timer.stop()
        self._telemetry_timer.stop()
        self.hide()

    # ---- inputs --------------------------------------------------------- #

    def to_logical(self, x: float, y: float) -> QPointF:
        """physical px ของจอ → logical px ภายใน overlay"""
        lx, ly = physical_to_logical(x - self._screen.left, y - self._screen.top, self._dpr)
        return QPointF(lx, ly)

    def show_packet(self, packet: FramePacket) -> None:
        """สถานะจากเฟรมล่าสุด (เรียกทุกเฟรมของกล้อง)"""
        snapshot = packet.result.snapshot
        hands = packet.result.hands
        self._fps = packet.fps
        self._latency_ms = packet.process_ms

        position = self.to_logical(*snapshot.cursor) if snapshot.cursor is not None else None
        self._reticle.set_state(position, snapshot, self._clock())
        self._refresh("reticle")
        self._radial.set_state(position, snapshot)
        self._refresh("radial")
        if self._charge.set_state(position, snapshot):
            self._refresh("charge")

        if self._badge.set_snapshot(snapshot):
            self._refresh("badge")
        if self._telemetry is not None and self._telemetry.set_armed(snapshot.armed):
            self._refresh("telemetry")
        if self._radar is not None and self._radar.set_hands(
            hands.hands, hands.width, hands.height, snapshot.armed
        ):
            self._refresh("radar")

    def handle_events(self, events: tuple[GestureEvent, ...]) -> None:
        """event ที่ไม่ใช่การขยับ: shockwave + เสียง + สถานะ ARM"""
        now = self._clock()
        theme = self._theme
        for event in events:
            kind = event.kind
            if kind is EventKind.CLICK:
                color = theme.accent if event.button == "right" else theme.primary
                self._reticle.add_shockwave(self.to_logical(event.x, event.y), color, now)
                self._play("click")
            elif kind is EventKind.BUTTON_DOWN:
                self._reticle.add_shockwave(self.to_logical(event.x, event.y), theme.accent, now)
                self._play("click")
            elif kind is EventKind.BUTTON_UP:
                self._reticle.add_shockwave(self.to_logical(event.x, event.y), theme.primary, now)
                self._play("release")
            elif kind is EventKind.ACTION:
                if event.name.startswith("swipe_"):
                    self._radial.flash(event.name.removeprefix("swipe_"), now)
                    self._refresh("radial")
                    self._play("swipe")
                else:  # กำมือค้าง (ค่าเริ่มต้น: ย่อหน้าต่าง)
                    self._reticle.add_shockwave(
                        self.to_logical(event.x, event.y), theme.accent, now
                    )
                    self._play("minimize")
            elif kind is EventKind.ARMED:
                self.set_armed(True)
                self._play("arm")
            elif kind is EventKind.DISARMED:
                self.set_armed(False)
                self._play("disarm")
        self._refresh("reticle")

    def set_armed(self, armed: bool) -> None:
        """อัปเดตป้ายสถานะทันที (ไม่ต้องรอเฟรมถัดไป เผื่อกล้องค้างอยู่)"""
        if self._badge.set_armed(armed):
            self._refresh("badge")
        if self._telemetry is not None and self._telemetry.set_armed(armed):
            self._refresh("telemetry")

    def set_alert(self, text: str | None) -> None:
        """ข้อความเตือนใต้ป้ายสถานะ (เช่น กล้องหลุด) — ``None`` = ล้าง"""
        if self._badge.set_alert(text):
            self._refresh("badge")
            if text:
                self._play("error")

    # ---- animation / repaint -------------------------------------------- #

    def _tick(self) -> None:
        now = self._clock()
        # วาดใหม่ขณะยังมีแอนิเมชัน และอีกหนึ่งครั้งหลังจบ (เพื่อลบภาพสุดท้ายออก)
        animated = (("boot", self._boot), ("radial", self._radial), ("reticle", self._reticle))
        for key, element in animated:
            was_painted = not self._painted.get(key, QRectF()).isEmpty()
            if element.advance(now) or was_painted:
                self._refresh(key)

    def _sample_telemetry(self) -> None:
        if self._telemetry is None:
            return
        if self._sampler is None:
            try:
                self._sampler = TelemetrySampler()
            except ImportError:
                log.warning("psutil unavailable — telemetry panel disabled")
                self._elements.pop("telemetry", None)
                self._telemetry = None
                return
        if self._telemetry.set_values(self._sampler.sample(self._fps, self._latency_ms)):
            self._refresh("telemetry")

    def _refresh(self, key: str) -> None:
        """สั่งวาดใหม่เฉพาะบริเวณของชิ้น ``key`` (กรอบเดิม ∪ กรอบใหม่)"""
        element = self._elements.get(key)
        if element is None:
            return
        new = element.rect()
        old = self._painted.get(key, QRectF())
        dirty = old.united(new)
        self._painted[key] = new
        if not dirty.isEmpty():
            self.update(dirty.adjusted(-_PAD, -_PAD, _PAD, _PAD).toAlignedRect())

    def _play(self, name: str) -> None:
        if self._sounds is not None:
            self._sounds.play(name)

    def _layout_panels(self) -> None:
        viewport = QRectF(0.0, 0.0, float(self.width()), float(self.height()))
        self._badge.set_viewport(viewport)
        if self._telemetry is not None:
            self._telemetry.set_viewport(viewport)
        if self._radar is not None:
            self._radar.set_viewport(viewport)

    # ---- Qt events ------------------------------------------------------ #

    def resizeEvent(self, event) -> None:
        self._layout_panels()
        self._painted.clear()
        self.update()
        super().resizeEvent(event)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setClipRegion(event.region())
        # ล้างบริเวณที่จะวาดให้โปร่งใสก่อน (หน้าต่างไม่มีพื้นหลัง)
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Source)
        painter.fillRect(event.rect(), Qt.GlobalColor.transparent)
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        painter.setOpacity(self._theme.opacity)

        now = self._clock()
        exposed = QRectF(event.rect())
        for element in self._elements.values():
            rect = element.rect()
            if rect.isEmpty() or not rect.adjusted(-_PAD, -_PAD, _PAD, _PAD).intersects(exposed):
                continue
            painter.save()
            element.paint(painter, now)
            painter.restore()
        painter.end()
