"""แผงของ HUD: ป้ายสถานะ (ขวาบน), telemetry (ซ้ายล่าง), radar มือ (ขวาล่าง)

แต่ละแผงมี ``rect`` คงที่ และ method ``set_*`` ที่คืน ``True`` เมื่อสิ่งที่แสดงเปลี่ยนจริง
overlay จะสั่งวาดใหม่เฉพาะแผงที่เปลี่ยน — minimal = การ์ดมุมมนพื้นเข้ม ขอบบางมาก ตัวหนังสือปกติ,
holo = กรอบตัดมุมเรืองแสง ตัวพิมพ์ใหญ่
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QBrush, QColor, QPainter, QPainterPath, QPen

from ..gestures import EngineSnapshot, State
from ..tracker import Hand, hand_connections
from .theme import LayerCache, Theme, chamfered_rect, glow_stroke, with_alpha

__all__ = [
    "HandRadar",
    "StatusBadge",
    "TelemetryPanel",
    "TelemetrySampler",
    "TelemetryValues",
    "mode_text",
]

MARGIN = 24.0
_FILL = QColor(4, 10, 16)
_FILL_ALPHA = 0.55  # พื้นหลังแผง: ให้อ่านออกบนพื้นหลังสว่าง


_FRAME_PAD = 6.0  # ขอบเรืองแสงล้นกรอบออกมาเท่านี้
_frames = LayerCache()

CARD_FILL = QColor(16, 18, 22)  # minimal: พื้นการ์ด
CARD_FILL_ALPHA = 0.80
CARD_BORDER_ALPHA = 0.10  # ขอบบาง (สีหลักจางมาก)
CARD_RADIUS = 12.0


def _card(painter: QPainter, rect: QRectF, color: QColor) -> None:
    """กรอบแผงแบบ minimal: การ์ดมุมมนพื้นเข้มโปร่งแสง + ขอบ 1 px จางๆ — วาดครั้งเดียวแล้ว cache"""

    def render(layer: QPainter) -> None:
        local = QRectF(_FRAME_PAD, _FRAME_PAD, rect.width(), rect.height())
        layer.setPen(QPen(with_alpha(color, CARD_BORDER_ALPHA), 1.0))
        layer.setBrush(QBrush(with_alpha(CARD_FILL, CARD_FILL_ALPHA)))
        layer.drawRoundedRect(local.adjusted(0.5, 0.5, -0.5, -0.5), CARD_RADIUS, CARD_RADIUS)

    _frames.draw(
        painter,
        ("card", rect.width(), rect.height(), color.rgba()),
        QPointF(rect.left() - _FRAME_PAD, rect.top() - _FRAME_PAD),
        rect.width() + 2 * _FRAME_PAD,
        rect.height() + 2 * _FRAME_PAD,
        render,
    )


def mode_text(mode: str) -> str:
    """ชื่อ state ของ engine แบบอ่านง่าย: ``TWO_HAND_ZOOM`` → ``Two hand zoom``"""
    return mode.replace("_", " ").capitalize()


def _fade_out(elapsed: float, duration: float) -> float:
    """1 → 0 ช่วงท้าย (minimal ใช้แทนการกระพริบ)"""
    t = min(max(elapsed / duration, 0.0), 1.0)
    return 1.0 - t**3


def _panel(
    painter: QPainter, rect: QRectF, color: QColor, *, scanlines: bool = False, cut: float = 10.0
) -> None:
    """กรอบแผงแบบ holo: พื้นเข้มโปร่งแสง + ขอบตัดมุมเรืองแสง (+ เส้นสแกน) — วาดครั้งเดียวแล้ว cache"""

    def render(layer: QPainter) -> None:
        local = QRectF(_FRAME_PAD, _FRAME_PAD, rect.width(), rect.height())
        path = chamfered_rect(local, cut)
        layer.setPen(Qt.PenStyle.NoPen)
        layer.setBrush(QBrush(with_alpha(_FILL, _FILL_ALPHA)))
        layer.drawPath(path)
        if scanlines:  # เส้นแนวนอนจางๆ ทุก 3px
            layer.setPen(QPen(with_alpha(color, 0.06), 1.0))
            y = local.top() + 2.0
            while y < local.bottom():
                layer.drawLine(QPointF(local.left() + 2.0, y), QPointF(local.right() - 2.0, y))
                y += 3.0
        glow_stroke(layer, path, color, 1.2, strength=0.6, alpha=0.9)

    _frames.draw(
        painter,
        ("panel", rect.width(), rect.height(), color.rgba(), scanlines, cut),
        QPointF(rect.left() - _FRAME_PAD, rect.top() - _FRAME_PAD),
        rect.width() + 2 * _FRAME_PAD,
        rect.height() + 2 * _FRAME_PAD,
        render,
    )


class StatusBadge:
    """มุมขวาบน: ARMED/DISARMED, MODE ปัจจุบัน, จำนวนมือ (+ ข้อความเตือน เช่นกล้องหลุด)"""

    WIDTH = 252.0
    HEIGHT = 64.0
    ALERT_HEIGHT = 22.0
    NOTICE_SECONDS = 1.5

    def __init__(self, theme: Theme, *, dry_run: bool = False) -> None:
        self._theme = theme
        self._dry_run = dry_run
        self._armed = False
        self._mode = State.DISARMED.value
        self._hands = 0
        self._alert: str | None = None
        self._notice: tuple[str, float] | None = None  # (ข้อความ, เวลาเริ่ม) ข้อความยืนยันชั่วครู่
        self._viewport = QRectF()

    def set_viewport(self, viewport: QRectF) -> None:
        self._viewport = QRectF(viewport)

    def notify(self, text: str, now: float) -> None:
        """ข้อความยืนยันที่กระพริบใต้ป้ายราว ``NOTICE_SECONDS`` วินาที (เช่น จดเหตุการณ์แล้ว)"""
        self._notice = (text, now)

    @property
    def notice(self) -> str | None:
        return self._notice[0] if self._notice is not None else None

    def advance(self, now: float) -> bool:
        """คืน ``True`` ระหว่างที่ข้อความยืนยันยังแสดงอยู่ (ต้องวาดใหม่)"""
        if self._notice is not None and now - self._notice[1] >= self.NOTICE_SECONDS:
            self._notice = None
        return self._notice is not None

    def rect(self) -> QRectF:
        height = self.HEIGHT + self.ALERT_HEIGHT  # จองที่ของบรรทัดเตือนไว้เสมอ กรอบจะได้ไม่เปลี่ยนขนาด
        return QRectF(
            self._viewport.right() - MARGIN - self.WIDTH,
            self._viewport.top() + MARGIN,
            self.WIDTH,
            height,
        )

    def set_snapshot(self, snapshot: EngineSnapshot) -> bool:
        return self._set(snapshot.armed, snapshot.state.value, snapshot.hands)

    def set_armed(self, armed: bool) -> bool:
        mode = State.IDLE.value if armed else State.DISARMED.value
        return self._set(armed, mode if armed != self._armed else self._mode, self._hands)

    def set_alert(self, text: str | None) -> bool:
        if text == self._alert:
            return False
        self._alert = text
        return True

    def _set(self, armed: bool, mode: str, hands: int) -> bool:
        if (armed, mode, hands) == (self._armed, self._mode, self._hands):
            return False
        self._armed, self._mode, self._hands = armed, mode, hands
        return True

    def paint(self, painter: QPainter, now: float) -> None:
        if self._theme.minimal:
            self._paint_minimal(painter, now)
            return
        theme = self._theme
        color = theme.state_color(self._armed)
        outer = self.rect()
        body = QRectF(outer.left(), outer.top(), outer.width(), self.HEIGHT)
        _panel(painter, body, color)

        # ไฟสถานะ: ทึบเมื่อ ARMED, วงกลวงเมื่อ DISARMED
        dot = QPointF(body.left() + 22.0, body.top() + 23.0)
        if self._armed:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(with_alpha(color, 0.25)))
            painter.drawEllipse(dot, 9.0, 9.0)
            painter.setBrush(QBrush(color))
            painter.drawEllipse(dot, 5.0, 5.0)
        else:
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(color, 1.5))
            painter.drawEllipse(dot, 5.0, 5.0)

        painter.setPen(color)
        painter.setFont(theme.display_font(17, 700, spacing=2.0))
        title = QRectF(body.left() + 40.0, body.top() + 8.0, body.width() - 48.0, 30.0)
        flags = Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        painter.drawText(title, flags, "ARMED" if self._armed else "DISARMED")

        if self._dry_run:
            painter.setPen(theme.accent)
            painter.setFont(theme.mono_font(11))
            tag = QRectF(body.right() - 78.0, body.top() + 8.0, 68.0, 30.0)
            painter.drawText(
                tag, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, "DRY-RUN"
            )

        painter.setPen(with_alpha(color, 0.85))
        painter.setFont(theme.mono_font(13))
        detail = QRectF(body.left() + 16.0, body.top() + 38.0, body.width() - 28.0, 20.0)
        painter.drawText(detail, flags, f"MODE {self._mode}")
        painter.drawText(
            detail,
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
            f"HANDS {self._hands}",
        )

        line = QRectF(outer.left(), body.bottom() + 2.0, outer.width(), self.ALERT_HEIGHT - 2.0)
        right = Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        if self._alert:
            painter.setPen(theme.danger)
            painter.setFont(theme.mono_font(13))
            painter.drawText(line, right, theme.label(self._alert))
        elif self._notice is not None:
            text, start = self._notice
            blink = 0.55 + 0.45 * math.cos((now - start) * math.tau * 3.0)  # กระพริบ 3 ครั้ง/วินาที
            painter.setPen(with_alpha(theme.accent, blink))
            painter.setFont(theme.mono_font(13))
            painter.drawText(line, right, theme.label(text))

    def _paint_minimal(self, painter: QPainter, now: float) -> None:
        """การ์ดมน: จุดสถานะ + "Armed" / "Disarmed", บรรทัดรอง "Pointer · 1 hand", ข้อความเตือนใต้การ์ด"""
        theme = self._theme
        outer = self.rect()
        body = QRectF(outer.left(), outer.top(), outer.width(), self.HEIGHT)
        _card(painter, body, theme.primary)
        left = Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        right = Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter

        dot = QPointF(body.left() + 22.0, body.top() + 23.0)
        if self._armed:  # จุดทึบสีเน้น = พร้อมใช้
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(theme.accent))
            painter.drawEllipse(dot, 4.0, 4.0)
        else:
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(theme.disarmed, 1.4))
            painter.drawEllipse(dot, 3.6, 3.6)

        painter.setPen(theme.primary if self._armed else with_alpha(theme.primary, 0.75))
        painter.setFont(theme.text_font(15, 600))
        title = QRectF(body.left() + 36.0, body.top() + 9.0, body.width() - 48.0, 28.0)
        painter.drawText(title, left, "Armed" if self._armed else "Disarmed")

        if self._dry_run:
            painter.setPen(theme.accent)
            painter.setFont(theme.text_font(11, 500))
            painter.drawText(
                QRectF(body.right() - 92.0, body.top() + 9.0, 78.0, 28.0), right, "Dry run"
            )

        hands = f"{self._hands} hand" + ("" if self._hands == 1 else "s")
        painter.setPen(theme.secondary)
        painter.setFont(theme.text_font(12, 400))
        detail = QRectF(body.left() + 36.0, body.top() + 34.0, body.width() - 50.0, 20.0)
        painter.drawText(detail, left, f"{mode_text(self._mode)}  ·  {hands}")

        line = QRectF(
            outer.left(), body.bottom() + 3.0, outer.width() - 4.0, self.ALERT_HEIGHT - 3.0
        )
        painter.setFont(theme.text_font(12, 500))
        if self._alert:
            painter.setPen(theme.danger)
            painter.drawText(line, right, self._alert)
        elif self._notice is not None:
            text, start = self._notice
            painter.setPen(with_alpha(theme.accent, _fade_out(now - start, self.NOTICE_SECONDS)))
            painter.drawText(line, right, text)


@dataclass(frozen=True, slots=True)
class TelemetryValues:
    cpu: float = 0.0  # % ของทั้งเครื่อง
    ram: float = 0.0  # %
    fps: float = 0.0  # FPS ของกล้อง
    latency_ms: float = 0.0  # เฟรม → input
    clock: str = "--:--:--"


class TelemetrySampler:
    """อ่าน CPU/RAM ของเครื่องด้วย psutil (เรียก :meth:`sample` ไม่ถี่กว่าวินาทีละครั้งก็พอ)"""

    def __init__(self) -> None:
        import psutil

        self._psutil = psutil
        psutil.cpu_percent(interval=None)  # ครั้งแรกคืน 0 เสมอ: เรียกทิ้งเพื่อเริ่มช่วงวัด

    def sample(self, fps: float, latency_ms: float) -> TelemetryValues:
        return TelemetryValues(
            cpu=self._psutil.cpu_percent(interval=None),
            ram=self._psutil.virtual_memory().percent,
            fps=fps,
            latency_ms=latency_ms,
            clock=time.strftime("%H:%M:%S"),
        )


class TelemetryPanel:
    """มุมซ้ายล่าง: CPU %, RAM %, FPS กล้อง, latency, เวลา — ฟอนต์ mono + เส้นสแกนจางๆ"""

    WIDTH = 232.0
    HEIGHT = 140.0
    SLIDE_SECONDS = 0.45

    def __init__(self, theme: Theme) -> None:
        self._theme = theme
        self._values = TelemetryValues()
        self._armed = False
        self._viewport = QRectF()
        self._slide_start: float | None = None
        self._offset = 0.0  # px ที่ยังเลื่อนไม่ถึงที่ (ติดลบ = อยู่นอกจอด้านซ้าย)

    def set_viewport(self, viewport: QRectF) -> None:
        self._viewport = QRectF(viewport)

    def slide_in(self, start: float) -> None:
        """Boot v2: สไลด์เข้าจากขอบซ้ายตั้งแต่เวลา ``start``"""
        self._slide_start = start
        self._offset = -(self.WIDTH + MARGIN)

    def advance(self, now: float) -> bool:
        """คืน ``True`` ระหว่างที่ยังสไลด์อยู่"""
        if self._slide_start is None:
            return False
        t = (now - self._slide_start) / self.SLIDE_SECONDS
        if t >= 1.0:
            self._slide_start = None
            self._offset = 0.0
            return True  # วาดอีกครั้งที่ตำแหน่งสุดท้าย
        eased = 1.0 - (1.0 - max(t, 0.0)) ** 3
        self._offset = -(self.WIDTH + MARGIN) * (1.0 - eased)
        return True

    def rect(self) -> QRectF:
        return QRectF(
            self._viewport.left() + MARGIN + self._offset,
            self._viewport.bottom() - MARGIN - self.HEIGHT,
            self.WIDTH,
            self.HEIGHT,
        )

    def set_values(self, values: TelemetryValues) -> bool:
        if values == self._values:
            return False
        self._values = values
        return True

    def set_armed(self, armed: bool) -> bool:
        if armed == self._armed:
            return False
        self._armed = armed
        return True

    def paint(self, painter: QPainter, now: float) -> None:
        if self._theme.minimal:
            self._paint_minimal(painter)
            return
        theme = self._theme
        color = theme.state_color(self._armed)
        rect = self.rect()
        _panel(painter, rect, color, scanlines=True)
        values = self._values
        left = rect.left() + 16.0
        right = rect.right() - 16.0
        align_left = Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        align_right = Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter

        painter.setPen(with_alpha(color, 0.9))
        painter.setFont(theme.display_font(11, 600, spacing=2.5))
        painter.drawText(QRectF(left, rect.top() + 8.0, 120.0, 18.0), align_left, "TELEMETRY")
        painter.setFont(theme.mono_font(13))
        painter.drawText(
            QRectF(left, rect.top() + 8.0, right - left, 18.0), align_right, values.clock
        )

        rows = (
            ("CPU", f"{values.cpu:3.0f}%", values.cpu / 100.0),
            ("RAM", f"{values.ram:3.0f}%", values.ram / 100.0),
            ("FPS", f"{values.fps:4.1f}", min(values.fps / 30.0, 1.0)),
            ("LAT", f"{values.latency_ms:3.0f} ms", min(values.latency_ms / 60.0, 1.0)),
        )
        y = rect.top() + 34.0
        for label, text, level in rows:
            row = QRectF(left, y, right - left, 20.0)
            painter.setPen(with_alpha(color, 0.75))
            painter.drawText(row, align_left, label)
            painter.setPen(color)
            painter.drawText(row, align_right, text)

            bar = QRectF(left + 40.0, y + 8.0, 84.0, 4.0)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(with_alpha(color, 0.18)))
            painter.drawRect(bar)
            warn = (label in ("CPU", "RAM") and level > 0.85) or (label == "LAT" and level >= 1.0)
            warn = warn or (label == "FPS" and 0.0 < values.fps < 25.0)
            painter.setBrush(QBrush(theme.accent if warn else color))
            painter.drawRect(
                QRectF(bar.left(), bar.top(), bar.width() * max(0.0, min(level, 1.0)), 4.0)
            )
            y += 24.0

    def _paint_minimal(self, painter: QPainter) -> None:
        theme = self._theme
        rect = self.rect()
        _card(painter, rect, theme.primary)
        values = self._values
        left = rect.left() + 16.0
        right = rect.right() - 16.0
        align_left = Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        align_right = Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        text = theme.primary if self._armed else with_alpha(theme.primary, 0.75)

        painter.setPen(text)
        painter.setFont(theme.text_font(12, 600))
        painter.drawText(QRectF(left, rect.top() + 10.0, 120.0, 18.0), align_left, "System")
        painter.setPen(theme.secondary)
        painter.setFont(theme.mono_font(12))
        painter.drawText(
            QRectF(left, rect.top() + 10.0, right - left, 18.0), align_right, values.clock
        )

        rows = (
            ("CPU", f"{values.cpu:.0f}%", values.cpu / 100.0),
            ("Memory", f"{values.ram:.0f}%", values.ram / 100.0),
            ("Camera", f"{values.fps:.0f} fps", min(values.fps / 30.0, 1.0)),
            ("Latency", f"{values.latency_ms:.0f} ms", min(values.latency_ms / 60.0, 1.0)),
        )
        label_font = theme.text_font(12, 400)
        value_font = theme.mono_font(12)
        y = rect.top() + 36.0
        for label, value, level in rows:
            row = QRectF(left, y, right - left, 20.0)
            painter.setFont(label_font)
            painter.setPen(theme.secondary)
            painter.drawText(row, align_left, label)
            painter.setFont(value_font)
            painter.setPen(text)
            painter.drawText(row, align_right, value)

            warn = (label in ("CPU", "Memory") and level > 0.85) or (
                label == "Latency" and level >= 1.0
            )
            warn = warn or (label == "Camera" and 0.0 < values.fps < 25.0)
            bar = QRectF(left + 64.0, y + 9.0, right - left - 128.0, 2.0)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(with_alpha(theme.primary, 0.10)))
            painter.drawRoundedRect(bar, 1.0, 1.0)
            fill = QRectF(bar.left(), bar.top(), bar.width() * max(0.0, min(level, 1.0)), 2.0)
            painter.setBrush(QBrush(theme.accent if warn else with_alpha(theme.primary, 0.55)))
            painter.drawRoundedRect(fill, 1.0, 1.0)
            y += 24.0


class HandRadar:
    """มุมขวาล่าง: skeleton 21 จุดของมือ แบบเส้นเรืองแสงขนาดเล็ก

    ซูมเข้าที่มือข้างถนัด (ถ้าไม่เห็น ใช้มือแรกที่เห็น) ให้เต็มกรอบ จึงเห็นการจีบนิ้วชัด
    ไม่ว่ามือจะอยู่ใกล้หรือไกลกล้อง
    """

    SIZE = 148.0
    PADDING = 20.0

    def __init__(self, theme: Theme, dominant_hand: str) -> None:
        self._theme = theme
        self._dominant = dominant_hand
        self._hand: Hand | None = None
        self._aspect = 4.0 / 3.0
        self._armed = False
        self._connections = hand_connections()
        self._viewport = QRectF()

    def set_viewport(self, viewport: QRectF) -> None:
        self._viewport = QRectF(viewport)

    def rect(self) -> QRectF:
        return QRectF(
            self._viewport.right() - MARGIN - self.SIZE,
            self._viewport.bottom() - MARGIN - self.SIZE,
            self.SIZE,
            self.SIZE,
        )

    def set_hands(self, hands: tuple[Hand, ...], width: int, height: int, armed: bool) -> bool:
        hand = next((h for h in hands if h.handedness == self._dominant), None)
        if hand is None and hands:
            hand = hands[0]
        changed = hand is not None or self._hand is not None or armed != self._armed
        self._hand = hand
        self._armed = armed
        if height > 0:
            self._aspect = width / height
        return changed

    def skeleton_points(self, area: QRectF) -> list[QPointF]:
        """ตำแหน่ง landmark 21 จุดที่ย่อ/ขยายให้พอดี ``area`` โดยคงสัดส่วนของมือ"""
        if self._hand is None:
            return []
        xs = [p.x * self._aspect for p in self._hand.landmarks]  # ชดเชยเฟรมที่ไม่จัตุรัส
        ys = [p.y for p in self._hand.landmarks]
        span = max(max(xs) - min(xs), max(ys) - min(ys), 1e-6)
        scale = min(area.width(), area.height()) / span
        mid_x, mid_y = (max(xs) + min(xs)) / 2.0, (max(ys) + min(ys)) / 2.0
        center = area.center()
        return [
            QPointF(center.x() + (x - mid_x) * scale, center.y() + (y - mid_y) * scale)
            for x, y in zip(xs, ys)
        ]

    def paint(self, painter: QPainter, now: float) -> None:
        if self._theme.minimal:
            self._paint_minimal(painter)
            return
        theme = self._theme
        color = theme.state_color(self._armed)
        rect = self.rect()
        _panel(painter, rect, color)

        painter.setPen(with_alpha(color, 0.9))
        painter.setFont(theme.display_font(10, 600, spacing=2.0))
        title = QRectF(rect.left() + 12.0, rect.top() + 5.0, rect.width() - 24.0, 14.0)
        align = Qt.AlignmentFlag.AlignVCenter
        painter.drawText(title, Qt.AlignmentFlag.AlignLeft | align, "HAND")

        hand = self._hand
        if hand is None:
            painter.setPen(with_alpha(color, 0.45))
            painter.setFont(theme.mono_font(12))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, "NO SIGNAL")
            return

        painter.setFont(theme.mono_font(10))
        painter.drawText(title, Qt.AlignmentFlag.AlignRight | align, hand.handedness.upper())

        hand_color = color if hand.handedness == self._dominant else theme.accent
        if not self._armed:
            hand_color = theme.disarmed
        area = rect.adjusted(self.PADDING, self.PADDING + 6.0, -self.PADDING, -self.PADDING + 2.0)
        points = self.skeleton_points(area)
        path = QPainterPath()
        for start, end in self._connections:
            path.moveTo(points[start])
            path.lineTo(points[end])
        glow_stroke(painter, path, hand_color, 1.4, strength=0.5)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(hand_color))
        for point in points:
            painter.drawEllipse(point, 1.8, 1.8)

    def _paint_minimal(self, painter: QPainter) -> None:
        theme = self._theme
        rect = self.rect()
        _card(painter, rect, theme.primary)
        align = Qt.AlignmentFlag.AlignVCenter
        title = QRectF(rect.left() + 14.0, rect.top() + 8.0, rect.width() - 28.0, 16.0)
        painter.setPen(theme.secondary)
        painter.setFont(theme.text_font(11, 500))
        painter.drawText(title, Qt.AlignmentFlag.AlignLeft | align, "Hand")

        hand = self._hand
        if hand is None:
            painter.setFont(theme.text_font(12, 400))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, "No hand")
            return
        painter.drawText(title, Qt.AlignmentFlag.AlignRight | align, hand.handedness.capitalize())

        if not self._armed:
            hand_color = theme.disarmed
        elif hand.handedness == self._dominant:
            hand_color = theme.primary
        else:
            hand_color = theme.accent
        area = rect.adjusted(self.PADDING, self.PADDING + 8.0, -self.PADDING, -self.PADDING + 2.0)
        points = self.skeleton_points(area)
        path = QPainterPath()
        for start, end in self._connections:
            path.moveTo(points[start])
            path.lineTo(points[end])
        painter.setBrush(Qt.BrushStyle.NoBrush)
        pen = QPen(with_alpha(hand_color, 0.85), 1.2)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        painter.drawPath(path)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(hand_color))
        for point in points:
            painter.drawEllipse(point, 1.6, 1.6)
