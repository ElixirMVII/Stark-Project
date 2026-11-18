"""หน้าต่าง overlay ของ HUD: โปร่งใส เต็มจอหลัก อยู่บนสุด และ **คลิกทะลุได้ 100%**

- ไม่รับ input ใดๆ (``WindowTransparentForInput``) ไม่แย่ง focus ไม่โผล่ใน taskbar
- หลายจอ: หนึ่ง overlay ต่อจอ (:class:`HudGroup`) แผงต่างๆ และ boot อยู่บนจอหลัก ส่วน reticle /
  เข็มทิศ / วงแหวนชาร์จ / คลื่นคลิก ขึ้นบนจอที่เคอร์เซอร์อยู่
- พิกัดจาก engine เป็น physical px → แปลงเป็น logical px ด้วย ``devicePixelRatio`` ก่อนวาด
- วาดด้วย QPainter และสั่ง ``update()`` เฉพาะกรอบของชิ้นที่เปลี่ยน (รอบ reticle และแผงต่างๆ)
  ไม่วาดใหม่ทั้งจอทุกเฟรม ถ้าไม่มีอะไรขยับก็ไม่วาดเลย
"""

from __future__ import annotations

import dataclasses
import logging
import time
from typing import Protocol

from PySide6.QtCore import QElapsedTimer, QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import QGuiApplication, QPainter
from PySide6.QtWidgets import QWidget

from ..camera import FramePacket
from ..config import Config, assets_dir
from ..gestures import ACTION_LAUNCH_PREFIX, ACTION_LAUNCHER_OPEN, EventKind, GestureEvent
from ..mapping import ScreenGeometry, physical_to_logical
from ..poses import POSE_PREFIX
from ..sound import SoundPlayer
from .boot import BootSequence
from .effects import PARTICLES_PER_BURST, MinimizeCollapse, ParticlePool, SwipeStreak
from .launcher import LauncherMenu
from .panels import HandRadar, StatusBadge, TelemetryPanel, TelemetrySampler
from .radial import ChargeRing, Radial
from .reticle import Reticle
from .target_lock import TargetLock
from .theme import Theme, set_glow_quality

__all__ = ["OVERLAY_FLAGS", "HudGroup", "HudOverlay"]

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
        qt_screen=None,
        panels: bool = True,
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

        qt_screen = qt_screen or QGuiApplication.primaryScreen()
        self._dpr = device_pixel_ratio or (qt_screen.devicePixelRatio() if qt_screen else 1.0)
        if qt_screen is not None:
            self.setScreen(qt_screen)
            # เตี้ยกว่าจอ 1px: หน้าต่างที่คลุมจอพอดีอาจถูก Windows มองเป็นแอป fullscreen
            self.setGeometry(qt_screen.geometry().adjusted(0, 0, 0, -1))
        self._panels = panels
        self._cursor_inside = False

        hud = config.hud
        set_glow_quality(hud.quality == "high")
        self._reduce_motion = hud.reduce_motion
        self._reticle = Reticle(self._theme, reduce_motion=hud.reduce_motion)
        burst = PARTICLES_PER_BURST if hud.quality == "high" else PARTICLES_PER_BURST // 2
        # minimal: ไม่มีอนุภาคตอนคลิก และไม่มีแถบแสงตอนปัด (เหลือวงคลิกบางๆ กับลูกศรที่สว่าง)
        self._minimal = self._theme.minimal
        quiet = hud.reduce_motion or self._minimal
        self._particles = None if quiet else ParticlePool(self._theme, per_burst=burst)
        self._streak = SwipeStreak(self._theme)
        self._collapse = MinimizeCollapse(self._theme)
        # คืนกรอบ (left, top, right, bottom) ของหน้าต่างที่เพิ่งย่อ — แอปต่อเข้ากับ ActionExecutor
        self.minimized_rect_provider = lambda: None
        self.draw_ms = 0.0  # เวลาวาดเฉลี่ยต่อครั้ง (EMA) — งบ ≤ 4 ms (spec M7.6)
        self.draw_ms_max = 0.0
        self._draw_timer = QElapsedTimer()
        self._boot = BootSequence(self._theme, reduce_motion=config.hud.reduce_motion)
        actions = dataclasses.asdict(config.actions)
        self._radial = Radial(self._theme, actions)
        self._charge = ChargeRing(self._theme, actions)
        self._launcher = LauncherMenu(self._theme, [item.label for item in config.launcher.items])
        self._target_lock = TargetLock(self._theme, show_label=config.targeting.show_label)
        self._badge = StatusBadge(self._theme, dry_run=dry_run) if panels else None
        self._telemetry = (
            TelemetryPanel(self._theme) if panels and config.hud.show_telemetry else None
        )
        self._radar = (
            HandRadar(self._theme, config.tracking.dominant_hand)
            if panels and config.hud.show_radar
            else None
        )
        # ลำดับการวาด: ล่าง → บน
        self._elements: dict[str, _Element] = {"boot": self._boot}
        self._elements["streak"] = self._streak
        self._elements["collapse"] = self._collapse
        if self._badge is not None:
            self._elements["badge"] = self._badge
        if self._telemetry is not None:
            self._elements["telemetry"] = self._telemetry
        if self._radar is not None:
            self._elements["radar"] = self._radar
        self._elements["radial"] = self._radial
        self._elements["charge"] = self._charge
        self._elements["launcher"] = self._launcher
        self._elements["target"] = self._target_lock
        if self._particles is not None:
            self._elements["particles"] = self._particles
        self._elements["reticle"] = self._reticle
        self._painted: dict[str, QRectF] = {}

        self._fps = 0.0
        self._latency_ms = 0.0
        self._notice_shown = False
        self.paint_count = 0  # จำนวนครั้งที่วาด (สถิติ FPS ของ HUD ต่อ session)
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
        if boot and self._panels:
            self._boot.start(now, QPointF(self.width() / 2.0, self.height() / 2.0))
            if self._telemetry is not None and not (self._reduce_motion or self._minimal):
                self._telemetry.slide_in(now + 0.35)  # Boot v2: แผงสไลด์เข้าหลังวงแหวนเริ่มขยาย
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

    @property
    def screen_geometry(self) -> ScreenGeometry:
        """พื้นที่ของจอที่ overlay นี้คลุม (physical px)"""
        return self._screen

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

        cursor = snapshot.cursor
        self._cursor_inside = cursor is not None and self._screen.contains(*cursor)
        position = self.to_logical(*cursor) if self._cursor_inside else None  # อยู่จออื่น: ซ่อน
        self._reticle.set_state(position, snapshot, self._clock())
        self._refresh("reticle")
        self._radial.set_state(position, snapshot)
        self._refresh("radial")
        if self._charge.set_state(position, snapshot):
            self._refresh("charge")
        self._launcher.set_state(position, snapshot)
        self._refresh("launcher")
        self._target_lock.set_state(self._target_rect(snapshot), position, snapshot)
        self._refresh("target")

        if self._badge is not None and self._badge.set_snapshot(snapshot):
            self._refresh("badge")
        if self._telemetry is not None and self._telemetry.set_armed(snapshot.armed):
            self._refresh("telemetry")
        if self._radar is not None and self._radar.set_hands(
            hands.hands, hands.width, hands.height, snapshot.armed
        ):
            self._refresh("radar")

    def handle_events(self, events: tuple[GestureEvent, ...]) -> None:
        """event ที่ไม่ใช่การขยับ: shockwave + เสียง + สถานะ ARM

        คลื่นคลิก/ไฮไลต์ของเข็มทิศขึ้นเฉพาะบนจอที่เกิดเหตุ ส่วนเสียงเล่นโดย overlay ที่มี sounds
        (จอหลัก) เสมอ ไม่ว่าเหตุจะเกิดบนจอไหน
        """
        now = self._clock()
        theme = self._theme
        shockwaves = {
            EventKind.BUTTON_DOWN: theme.accent,
            EventKind.BUTTON_UP: theme.primary,
            EventKind.WINDOW_GRAB: theme.accent,
            EventKind.WINDOW_RELEASE: theme.primary,
        }
        sounds = {
            EventKind.CLICK: "click",
            EventKind.BUTTON_DOWN: "click",
            EventKind.BUTTON_UP: "release",
            EventKind.WINDOW_GRAB: "click",
            EventKind.WINDOW_RELEASE: "release",
            EventKind.ARMED: "arm",
            EventKind.DISARMED: "disarm",
        }
        for event in events:
            kind = event.kind
            here = self._screen.contains(event.x, event.y)
            if kind is EventKind.CLICK and here:
                color = theme.accent if event.button == "right" else theme.primary
                point = self.to_logical(event.x, event.y)
                self._reticle.add_shockwave(point, color, now)
                if self._particles is not None:
                    self._particles.burst(point, color, now)
                self._target_lock.flash(now)
            elif kind is EventKind.SCROLL:
                self._reticle.handle_event(kind, event.dy, now)
            elif kind in shockwaves and here:
                self._reticle.add_shockwave(
                    self.to_logical(event.x, event.y), shockwaves[kind], now
                )
            elif kind is EventKind.ACTION and event.name == ACTION_LAUNCHER_OPEN:
                self._play("swipe")
            elif kind is EventKind.ACTION and event.name.startswith(ACTION_LAUNCH_PREFIX):
                index = event.name.removeprefix(ACTION_LAUNCH_PREFIX)
                if self._cursor_inside and index.isdigit():
                    self._launcher.flash(int(index), now)
                    self._refresh("launcher")
                self._play("click")
            elif kind is EventKind.ACTION:
                if event.name.startswith("swipe_"):
                    if self._cursor_inside:
                        direction = event.name.removeprefix("swipe_")
                        self._radial.flash(direction, now)
                        self._refresh("radial")
                        if not self._minimal:
                            self._streak.start(direction, now)
                            self._refresh("streak")
                    self._play("swipe")
                elif event.name.startswith(POSE_PREFIX):  # ท่าที่สอนเอง
                    if here:
                        point = self.to_logical(event.x, event.y)
                        self._reticle.add_shockwave(point, theme.accent, now)
                    self._play("click")
                else:  # กำมือค้าง (ค่าเริ่มต้น: ย่อหน้าต่าง)
                    if here:
                        point = self.to_logical(event.x, event.y)
                        self._reticle.add_shockwave(point, theme.accent, now)
                    self._start_collapse(now)
                    self._play("minimize")
            elif kind is EventKind.ARMED:
                self.set_armed(True)
            elif kind is EventKind.DISARMED:
                self.set_armed(False)
            if kind in sounds:
                self._play(sounds[kind])
        self._refresh("reticle")

    def set_armed(self, armed: bool) -> None:
        """อัปเดตป้ายสถานะทันที (ไม่ต้องรอเฟรมถัดไป เผื่อกล้องค้างอยู่)"""
        if self._badge is not None and self._badge.set_armed(armed):
            self._refresh("badge")
        if self._telemetry is not None and self._telemetry.set_armed(armed):
            self._refresh("telemetry")

    def _start_collapse(self, now: float) -> None:
        """กรอบของหน้าต่างที่เพิ่งย่อ ยุบลงไปทาง taskbar (ล่างกลางของจอนี้)"""
        try:
            rect = self.minimized_rect_provider()
        except Exception:  # noqa: BLE001 — effect เท่านั้น
            rect = None
        if rect is None:
            return
        left, top, right, bottom = rect
        if not self._screen.contains((left + right) / 2, (top + bottom) / 2):
            return
        a, b = self.to_logical(left, top), self.to_logical(right, bottom)
        goal = QPointF(self.width() / 2.0, self.height() - 6.0)
        self._collapse.start(QRectF(a, b), goal, now)
        self._refresh("collapse")

    def _target_rect(self, snapshot) -> QRectF | None:
        """สี่เหลี่ยมของปุ่มที่ล็อกเป้า เป็น logical px ของ overlay นี้ (None = ไม่มี / อยู่จออื่น)"""
        target = snapshot.target
        if target is None or not self._screen.contains(*target.center):
            return None
        top_left = self.to_logical(target.left, target.top)
        return QRectF(
            top_left.x(), top_left.y(), target.width / self._dpr, target.height / self._dpr
        )

    def notify(self, text: str, sound: str | None = "click") -> None:
        """ข้อความยืนยันชั่วครู่ใต้ป้ายสถานะ + เสียงสั้นๆ (เช่น จดเหตุการณ์แล้ว)"""
        if self._badge is not None:
            self._badge.notify(text, self._clock())
            self._notice_shown = True
            self._refresh("badge")
        if sound:
            self._play(sound)

    def set_alert(self, text: str | None) -> None:
        """ข้อความเตือนใต้ป้ายสถานะ (เช่น กล้องหลุด) — ``None`` = ล้าง"""
        if self._badge is not None and self._badge.set_alert(text):
            self._refresh("badge")
            if text:
                self._play("error")

    # ---- animation / repaint -------------------------------------------- #

    def _tick(self) -> None:
        now = self._clock()
        # วาดใหม่ขณะยังมีแอนิเมชัน และอีกหนึ่งครั้งหลังจบ (เพื่อลบภาพสุดท้ายออก)
        animated = (
            ("boot", self._boot),
            ("radial", self._radial),
            ("launcher", self._launcher),
            ("target", self._target_lock),
            ("reticle", self._reticle),
            ("streak", self._streak),
            ("collapse", self._collapse),
            *((("particles", self._particles),) if self._particles is not None else ()),
        )
        for key, element in animated:
            was_painted = not self._painted.get(key, QRectF()).isEmpty()
            if element.advance(now) or was_painted:
                self._refresh(key)
        if self._telemetry is not None and self._telemetry.advance(now):
            self._refresh("telemetry")
        if self._badge is not None and (self._badge.advance(now) or self._notice_shown):
            self._notice_shown = self._badge.notice is not None
            self._refresh("badge")  # ข้อความยืนยันกระพริบ (และวาดอีกครั้งตอนหายไป)

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
        self._streak.set_viewport(viewport)
        if self._badge is not None:
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
        self.paint_count += 1
        self._draw_timer.start()
        self._paint(event)
        elapsed = self._draw_timer.nsecsElapsed() / 1e6
        self.draw_ms += 0.1 * (elapsed - self.draw_ms)
        self.draw_ms_max = max(self.draw_ms_max * 0.995, elapsed)

    def _paint(self, event) -> None:
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


class HudGroup:
    """HUD หลายจอ: หนึ่ง :class:`HudOverlay` ต่อจอ ใช้แทน :class:`HudOverlay` ตัวเดียวได้

    ``overlays[0]`` คือจอหลัก (มีแผง สถานะ boot และเล่นเสียง) จออื่นแสดงเฉพาะสิ่งที่ตามเคอร์เซอร์
    """

    def __init__(self, overlays: list[HudOverlay]) -> None:
        if not overlays:
            raise ValueError("ต้องมี overlay อย่างน้อยหนึ่งจอ")
        self.overlays = list(overlays)

    @property
    def primary(self) -> HudOverlay:
        return self.overlays[0]

    def start(self, *, boot: bool = True) -> None:
        for overlay in self.overlays:
            overlay.start(boot=boot)

    def stop(self) -> None:
        for overlay in self.overlays:
            overlay.stop()

    def deleteLater(self) -> None:
        for overlay in self.overlays:
            overlay.deleteLater()

    def show_packet(self, packet: FramePacket) -> None:
        for overlay in self.overlays:
            overlay.show_packet(packet)

    def handle_events(self, events: tuple[GestureEvent, ...]) -> None:
        for overlay in self.overlays:
            overlay.handle_events(events)

    def set_armed(self, armed: bool) -> None:
        for overlay in self.overlays:
            overlay.set_armed(armed)

    def set_alert(self, text: str | None) -> None:
        self.primary.set_alert(text)

    def notify(self, text: str, sound: str | None = "click") -> None:
        self.primary.notify(text, sound)

    @property
    def paint_count(self) -> int:
        return sum(overlay.paint_count for overlay in self.overlays)

    @property
    def draw_ms(self) -> float:
        return max(overlay.draw_ms for overlay in self.overlays)

    @property
    def minimized_rect_provider(self):
        return self.primary.minimized_rect_provider

    @minimized_rect_provider.setter
    def minimized_rect_provider(self, provider) -> None:
        for overlay in self.overlays:
            overlay.minimized_rect_provider = provider
