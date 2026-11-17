"""หน้าต่าง debug: ภาพกล้อง + landmark + ชื่อท่า + สถานะ engine + FPS / latency

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
from PySide6.QtWidgets import QPushButton, QWidget

from .camera import FramePacket
from .config import Config
from .geometry import pointer_anchor
from .gestures import EngineSnapshot, State
from .tracker import GESTURE_NONE, LM, Hand, hand_connections
from .tuning import current_values
from .tuning_panel import PANEL_WIDTH, TuningPanel

__all__ = ["DebugView"]

_BACKGROUND = QColor("#0B0F14")
_TEXT = QColor("#E6F1F5")
_DIM = QColor("#7A8B99")
_SIM_BACKGROUND = QColor("#101820")  # พื้นของเฟรมจำลอง (ไม่มีภาพกล้อง)
_PANEL = QColor(0, 0, 0, 170)
_WARN = QColor("#FFB300")
_DISARMED = QColor("#7A8B99")
_FINGERTIPS = (LM.THUMB_TIP, LM.INDEX_TIP, LM.MIDDLE_TIP, LM.RING_TIP, LM.PINKY_TIP)
_MARGIN = 8


class DebugView(QWidget):
    """แสดง :class:`FramePacket` ล่าสุด (เรียก method ทั้งหมดจาก UI thread)

    มีแผง live tuning (:class:`~aether.tuning_panel.TuningPanel`) ด้านขวา เปิด/ปิดด้วยปุ่ม TUNE หรือกด T
    """

    closed = Signal()

    def __init__(
        self,
        config: Config,
        *,
        dry_run: bool = False,
        arm_hotkey: str = "",
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._config = config
        self._dry_run = dry_run
        self._arm_hotkey = arm_hotkey
        self._packet: FramePacket | None = None
        self._armed = False
        self._camera_info = ""
        self._message: str | None = "กำลังเปิดกล้อง…"
        self._connections = hand_connections()
        self._primary = QColor(config.hud.palette.primary)
        self._accent = QColor(config.hud.palette.accent)

        self._mono = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
        self._mono.setPointSize(10)
        self._label_font = QFont(self.font())
        self._label_font.setPointSize(10)
        self._label_font.setBold(True)

        self.setWindowTitle("AETHER — Debug")
        self.setMinimumSize(320, 240)
        self.resize(config.camera.width, config.camera.height)

        self._tuning = TuningPanel(current_values(config), self)
        self._tuning.hide()
        self._tune_button = QPushButton("TUNE", self)
        self._tune_button.setCheckable(True)
        self._tune_button.setFixedSize(58, 24)
        self._tune_button.move(_MARGIN, _MARGIN)
        self._tune_button.toggled.connect(self.set_tuning_visible)

    # ---- API ------------------------------------------------------------ #

    @property
    def tuning(self) -> TuningPanel:
        return self._tuning

    def set_tuning_visible(self, visible: bool) -> None:
        if visible and self._tuning.isHidden():
            self.resize(self.width() + PANEL_WIDTH, self.height())  # ภาพกล้องไม่เล็กลง
        self._tuning.setVisible(visible)
        blocked = self._tune_button.blockSignals(True)
        self._tune_button.setChecked(visible)
        self._tune_button.blockSignals(blocked)
        self._place_tuning()
        self.update()

    def set_config(self, config: Config) -> None:
        """config ที่ปรับสดแล้ว (กรอบ active region ฯลฯ วาดตามค่าใหม่)"""
        self._config = config
        self.update()

    def show_packet(self, packet: FramePacket) -> None:
        self._packet = packet
        self._armed = packet.result.snapshot.armed
        self._message = None
        self.update()

    def set_armed(self, armed: bool) -> None:
        """อัปเดตป้าย ARM ทันที (ไม่ต้องรอเฟรมถัดไป เผื่อกล้องค้างอยู่)"""
        self._armed = armed
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

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_T:
            self.set_tuning_visible(self._tuning.isHidden())
            return
        super().keyPressEvent(event)

    def resizeEvent(self, event) -> None:
        self._place_tuning()
        super().resizeEvent(event)

    def _place_tuning(self) -> None:
        self._tuning.setGeometry(self.width() - PANEL_WIDTH, 0, PANEL_WIDTH, self.height())

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), _BACKGROUND)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        packet = self._packet
        if packet is not None:
            hands = packet.result.hands
            snapshot = packet.result.snapshot
            target = self._image_rect(hands.width, hands.height)
            self._draw_image(painter, target, packet.image)
            self._draw_active_region(painter, target)
            for hand in hands.hands:
                self._draw_hand(painter, target, hand)
            self._draw_stats(painter, packet)
            self._draw_pinch_meter(painter, snapshot)
        self._draw_arm_badge(painter, packet.result.snapshot if packet else None)
        if self._message:
            self._draw_message(painter, self._message)
        painter.end()

    # ---- drawing -------------------------------------------------------- #

    def _image_rect(self, width: int, height: int) -> QRectF:
        """พื้นที่วาดภาพกล้อง: ใหญ่สุดที่พอดีหน้าต่าง (ไม่รวมแผง tuning) โดยคงสัดส่วน"""
        area = self._view_width()
        scale = min(area / width, self.height() / height)
        w, h = width * scale, height * scale
        return QRectF((area - w) / 2, (self.height() - h) / 2, w, h)

    def _view_width(self) -> float:
        return float(self.width() - (PANEL_WIDTH if self._tuning.isVisible() else 0))

    def _draw_image(self, painter: QPainter, target: QRectF, frame: np.ndarray | None) -> None:
        if frame is None:  # แหล่งจำลอง (mouse / replay): ไม่มีภาพ วาดพื้นทึบแทน
            painter.fillRect(target, _SIM_BACKGROUND)
            painter.setPen(_DIM)
            painter.drawText(
                target.adjusted(10, 8, -10, -8),
                Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignRight,
                f"SIMULATED · {self._camera_info or 'no camera image'}",
            )
            return
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

        if dominant:  # จุดที่ใช้คุมเคอร์เซอร์ (กึ่งกลางโคนนิ้วชี้กับนิ้วกลาง)
            ax, ay = pointer_anchor(hand)
            anchor = QPointF(
                target.left() + ax * target.width(), target.top() + ay * target.height()
            )
            painter.setPen(QPen(_TEXT, 1.5))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(anchor, 7, 7)
            painter.drawLine(anchor + QPointF(-11, 0), anchor + QPointF(11, 0))
            painter.drawLine(anchor + QPointF(0, -11), anchor + QPointF(0, 11))

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
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(_PANEL))
        painter.drawRoundedRect(box, 4, 4)
        painter.setPen(color)
        painter.drawText(box, Qt.AlignmentFlag.AlignCenter, label)

    def _draw_stats(self, painter: QPainter, packet: FramePacket) -> None:
        """แถบมุมซ้ายล่าง (มืออยู่ครึ่งบนของเฟรมเป็นส่วนใหญ่ จะได้ไม่บังป้ายชื่อท่า)"""
        cfg = self._config
        hands = packet.result.hands
        snapshot = packet.result.snapshot
        fps_ok = packet.fps >= 25.0
        line1 = (
            f"FPS {packet.fps:5.1f}   latency {packet.process_ms:5.1f} ms   "
            f"hands {len(hands.hands)}"
        )
        details = [self._camera_info or f"{hands.width}x{hands.height}"]
        details.append("mirror on" if cfg.camera.mirror else "mirror off")
        details.append(f"dominant {cfg.tracking.dominant_hand}")
        if self._dry_run:
            details.append("DRY-RUN")
        line2 = " · ".join(details)

        state = [snapshot.state.value]
        if snapshot.gesture != GESTURE_NONE:
            state.append(snapshot.gesture)
        if snapshot.pose is not None:  # ท่าที่สอนเองที่ใกล้ที่สุด: ระยะ/เกณฑ์ (ไว้ปรับ poses.sensitivity)
            pose = snapshot.pose
            verdict = "ok" if pose.accepted else pose.reason
            state.append(f"pose {pose.name} {pose.distance:.2f}/{pose.threshold:.2f} {verdict}")
        if snapshot.pinch_ratio is not None:
            state.append(f"pinch {snapshot.pinch_ratio:4.2f}")
        if snapshot.charge_kind:
            state.append(f"{snapshot.charge_kind} {snapshot.charge * 100:3.0f}%")
        if snapshot.state is State.GESTURE:
            state.append("swipe ready" if snapshot.swipe_ready else "hold still")
        if snapshot.state is State.LAUNCHER:
            selection = snapshot.launcher_selection
            state.append("slice —" if selection is None else f"slice {selection}")
        if snapshot.cursor is not None:
            state.append(f"cursor ({snapshot.cursor[0]}, {snapshot.cursor[1]})")
        line3 = " · ".join(state)

        painter.setFont(self._mono)
        metrics = painter.fontMetrics()
        lines = (line1, line2, line3)
        width = max(metrics.horizontalAdvance(line) for line in lines) + 16
        line_h = metrics.height()
        height = line_h * len(lines) + 12
        box = QRectF(_MARGIN, self.height() - _MARGIN - height, width, height)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(_PANEL))
        painter.drawRoundedRect(box, 4, 4)
        baseline = box.top() + 6 + metrics.ascent()
        colors = (_TEXT if fps_ok else _WARN, _DIM, _TEXT)  # FPS ต่ำกว่าเกณฑ์ 25 → สีเตือน
        for i, (line, color) in enumerate(zip(lines, colors)):
            painter.setPen(color)
            painter.drawText(QPointF(box.left() + 8, baseline + i * line_h), line)

    def _draw_pinch_meter(self, painter: QPainter, snapshot: EngineSnapshot) -> None:
        """แถบมุมขวาล่าง: ใกล้จีบแค่ไหน (เต็ม = ถึง pinch.on) และ % ของการค้างเพื่อเริ่มลาก"""
        if not snapshot.hand_present:
            return
        width, height = 120.0, 10.0
        box = QRectF(
            self._view_width() - _MARGIN - width - 12,
            self.height() - _MARGIN - height - 28,
            width + 12,
            height + 28,
        )
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(_PANEL))
        painter.drawRoundedRect(box, 4, 4)

        painter.setFont(self._mono)
        painter.setPen(_DIM)
        label = "PINCH"
        if snapshot.pinch_button:
            label += f" {snapshot.pinch_button.upper()}"
        if snapshot.state is State.PINCH_PENDING and snapshot.pinch_button == "left":
            label += f" {snapshot.hold_progress * 100:3.0f}%"
        painter.drawText(
            QPointF(box.left() + 6, box.top() + 6 + painter.fontMetrics().ascent()), label
        )

        bar = QRectF(box.left() + 6, box.bottom() - height - 6, width, height)
        painter.setPen(QPen(_DIM, 1))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(bar)
        pinched = snapshot.pinch_level >= 1.0
        fill = QRectF(bar.left(), bar.top(), bar.width() * snapshot.pinch_level, bar.height())
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(self._accent if pinched else self._primary))
        painter.drawRect(fill)

    def _draw_arm_badge(self, painter: QPainter, snapshot: EngineSnapshot | None) -> None:
        """ป้ายมุมขวาบน: ARMED / DISARMED"""
        if self._armed:
            text = "ARMED"
            if snapshot is not None and snapshot.armed:
                text += f" · {snapshot.state.value}"
            color = self._primary
        else:
            text = "DISARMED"
            if self._arm_hotkey:
                text += f" · {self._arm_hotkey}"
            color = _DISARMED
        painter.setFont(self._label_font)
        metrics = painter.fontMetrics()
        width = metrics.horizontalAdvance(text) + 16
        box = QRectF(self._view_width() - _MARGIN - width, _MARGIN, width, metrics.height() + 8)
        painter.setPen(QPen(color, 1.5))
        painter.setBrush(QBrush(_PANEL))
        painter.drawRoundedRect(box, 4, 4)
        painter.setPen(color)
        painter.drawText(box, Qt.AlignmentFlag.AlignCenter, text)

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
