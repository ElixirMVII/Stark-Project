"""หน้าต่าง debug: ภาพกล้อง + landmark + ชื่อท่า + FPS / เวลาประมวลผล

วาดด้วย ``QPainter`` ทั้งหมด (ไม่ใช้หน้าต่างของ OpenCV) ภาพอยู่ในหน่วยความจำเท่านั้น
ไม่มีการบันทึกลงดิสก์
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QFontDatabase,
    QImage,
    QPainter,
    QPen,
)
from PySide6.QtWidgets import QWidget

from .camera import FramePacket
from .config import Config
from .tracker import GESTURE_NONE, LM, Hand, hand_connections

__all__ = ["DebugView"]

_BACKGROUND = QColor("#0B0F14")
_TEXT = QColor("#E6F1F5")
_DIM = QColor("#7A8B99")
_PANEL = QColor(0, 0, 0, 170)
_WARN = QColor("#FFB300")
_FINGERTIPS = (LM.THUMB_TIP, LM.INDEX_TIP, LM.MIDDLE_TIP, LM.RING_TIP, LM.PINKY_TIP)


class DebugView(QWidget):
    """แสดง :class:`FramePacket` ล่าสุด (เรียก method ทั้งหมดจาก UI thread)"""

    closed = Signal()

    def __init__(self, config: Config, *, dry_run: bool = False, parent=None) -> None:
        super().__init__(parent)
        self._config = config
        self._dry_run = dry_run
        self._packet: FramePacket | None = None
        self._camera_info = ""
        self._message: str | None = "กำลังเปิดกล้อง…"
        self._connections = hand_connections()
        self._primary = QColor(config.hud.primary)
        self._accent = QColor(config.hud.accent)

        self._mono = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
        self._mono.setPointSize(10)
        self._label_font = QFont(self.font())
        self._label_font.setPointSize(10)
        self._label_font.setBold(True)

        self.setWindowTitle("AETHER — Debug")
        self.setMinimumSize(320, 240)
        self.resize(config.camera.width, config.camera.height)

    # ---- API ------------------------------------------------------------ #

    def show_packet(self, packet: FramePacket) -> None:
        self._packet = packet
        self._message = None
        self.update()

    def set_camera_info(self, description: str) -> None:
        self._camera_info = description
        self._message = None
        self.update()

    def show_message(self, message: str) -> None:
        """แสดงข้อความแจ้ง (เช่น กล้องเปิดไม่ได้) ทับภาพ"""
        self._message = message
        self.update()

    # ---- Qt events ------------------------------------------------------ #

    def closeEvent(self, event) -> None:
        self.closed.emit()
        super().closeEvent(event)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), _BACKGROUND)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        packet = self._packet
        if packet is not None:
            target = self._image_rect(packet.hands.width, packet.hands.height)
            self._draw_image(painter, target, packet.image)
            self._draw_active_region(painter, target)
            for hand in packet.hands.hands:
                self._draw_hand(painter, target, hand)
            self._draw_stats(painter, packet)
        if self._message:
            self._draw_message(painter, self._message)
        painter.end()

    # ---- drawing -------------------------------------------------------- #

    def _image_rect(self, width: int, height: int) -> QRectF:
        """พื้นที่วาดภาพกล้อง: ใหญ่สุดที่พอดีหน้าต่างโดยคงสัดส่วน"""
        scale = min(self.width() / width, self.height() / height)
        w, h = width * scale, height * scale
        return QRectF((self.width() - w) / 2, (self.height() - h) / 2, w, h)

    def _draw_image(self, painter: QPainter, target: QRectF, frame: np.ndarray) -> None:
        frame = np.ascontiguousarray(frame)
        height, width = frame.shape[:2]
        image = QImage(frame.data, width, height, frame.strides[0], QImage.Format.Format_BGR888)
        painter.drawImage(target, image)  # frame ยังถูกถือโดย packet ระหว่างวาด

    def _draw_active_region(self, painter: QPainter, target: QRectF) -> None:
        x0, y0, x1, y1 = self._config.pointer.active_region
        region = QRectF(
            target.left() + x0 * target.width(),
            target.top() + y0 * target.height(),
            (x1 - x0) * target.width(),
            (y1 - y0) * target.height(),
        )
        pen = QPen(_DIM, 1, Qt.PenStyle.DashLine)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(region)

    def _draw_hand(self, painter: QPainter, target: QRectF, hand: Hand) -> None:
        dominant = hand.handedness == self._config.tracking.dominant_hand
        color = self._primary if dominant else self._accent
        points = [
            QPointF(target.left() + p.x * target.width(), target.top() + p.y * target.height())
            for p in hand.landmarks
        ]

        painter.setPen(QPen(color, 2))
        for start, end in self._connections:
            painter.drawLine(points[start], points[end])

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(color))
        for point in points:
            painter.drawEllipse(point, 3, 3)
        painter.setBrush(QBrush(_TEXT))
        for tip in _FINGERTIPS:
            painter.drawEllipse(points[tip], 3.5, 3.5)

        if hand.gesture == GESTURE_NONE:
            gesture = "—"
        else:
            gesture = f"{hand.gesture} {hand.gesture_score * 100:.0f}%"
        label = f"{hand.handedness.upper()} · {gesture}"
        painter.setFont(self._label_font)
        metrics = painter.fontMetrics()
        text_w = metrics.horizontalAdvance(label)
        top = min(p.y() for p in points)
        center_x = sum(p.x() for p in points) / len(points)
        box = QRectF(0, top - metrics.height() - 14, text_w + 12, metrics.height() + 6)
        box.moveLeft(center_x - box.width() / 2)
        box.moveLeft(max(target.left(), min(box.left(), target.right() - box.width())))
        box.moveTop(max(target.top(), box.top()))
        painter.setBrush(QBrush(_PANEL))
        painter.drawRoundedRect(box, 4, 4)
        painter.setPen(color)
        painter.drawText(box, Qt.AlignmentFlag.AlignCenter, label)

    def _draw_stats(self, painter: QPainter, packet: FramePacket) -> None:
        cfg = self._config
        fps_ok = packet.fps >= 25.0
        line1 = (
            f"FPS {packet.fps:5.1f}   process {packet.process_ms:5.1f} ms   "
            f"hands {len(packet.hands.hands)}"
        )
        details = [self._camera_info or f"{packet.hands.width}x{packet.hands.height}"]
        details.append("mirror on" if cfg.camera.mirror else "mirror off")
        details.append(f"dominant {cfg.tracking.dominant_hand}")
        if self._dry_run:
            details.append("DRY-RUN")
        line2 = " · ".join(details)

        painter.setFont(self._mono)
        metrics = painter.fontMetrics()
        width = max(metrics.horizontalAdvance(line1), metrics.horizontalAdvance(line2)) + 16
        line_h = metrics.height()
        # มุมซ้ายล่าง: มืออยู่ครึ่งบนของเฟรมเป็นส่วนใหญ่ จะได้ไม่บังป้ายชื่อท่า
        box = QRectF(8, self.height() - 8 - (line_h * 2 + 12), width, line_h * 2 + 12)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(_PANEL))
        painter.drawRoundedRect(box, 4, 4)
        baseline = box.top() + 6 + metrics.ascent()
        painter.setPen(_TEXT if fps_ok else _WARN)  # FPS ต่ำกว่าเกณฑ์ 25 → สีเตือน
        painter.drawText(QPointF(box.left() + 8, baseline), line1)
        painter.setPen(_DIM)
        painter.drawText(QPointF(box.left() + 8, baseline + line_h), line2)

    def _draw_message(self, painter: QPainter, message: str) -> None:
        painter.setFont(self._label_font)
        margin = 24
        area = QRectF(self.rect()).adjusted(margin, margin, -margin, -margin)
        flags = Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap
        text_rect = painter.boundingRect(area, flags, message)
        box = text_rect.adjusted(-12, -8, 12, 8)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(_PANEL))
        painter.drawRoundedRect(box, 6, 6)
        painter.setPen(_WARN)
        painter.drawText(area, flags, message)
