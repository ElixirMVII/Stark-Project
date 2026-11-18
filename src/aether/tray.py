"""System tray: Arm/Disarm, หน้าต่าง Debug, Calibrate, สอนท่ามือ, Studio, โหลด config ใหม่, ออก

ไอคอนวาดด้วย QPainter (ไม่มีไฟล์รูป): วงแหวนสีหลักเมื่อ ARMED สีเทาเมื่อ DISARMED
"""

from __future__ import annotations

from PySide6.QtCore import QObject, QPointF, Qt, Signal
from PySide6.QtGui import QAction, QColor, QIcon, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QMenu, QSystemTrayIcon

from .hud.theme import DISARMED

__all__ = ["Tray", "make_icon"]


def make_icon(color: QColor, armed: bool, size: int = 64) -> QIcon:
    """ไอคอน reticle: วงนอก + ส่วนโค้ง + จุดกลาง (ทึบเมื่อ ARMED)"""
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    center = QPointF(size / 2, size / 2)
    unit = size / 64.0

    painter.setBrush(QColor(8, 14, 20, 230))
    painter.setPen(Qt.PenStyle.NoPen)
    painter.drawEllipse(center, 30 * unit, 30 * unit)

    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.setPen(QPen(color, 5 * unit))
    painter.drawEllipse(center, 24 * unit, 24 * unit)
    pen = QPen(color, 4 * unit)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    painter.setPen(pen)
    arc = int(13 * unit)
    for start in (30, 150, 270):  # ส่วนโค้ง 3 ช่องเหมือน reticle ของ HUD
        painter.drawArc(
            int(center.x()) - arc, int(center.y()) - arc, arc * 2, arc * 2, start * 16, 70 * 16
        )
    if armed:
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(color)
        painter.drawEllipse(center, 5 * unit, 5 * unit)
    painter.end()
    return QIcon(pixmap)


class Tray(QObject):
    """เมนู tray — ส่ง signal ให้ตัวควบคุมแอป ไม่ทำงานเองโดยตรง"""

    toggle_arm_requested = Signal()
    debug_requested = Signal(bool)
    calibrate_requested = Signal()
    pause_requested = Signal(bool)  # True = พักกล้อง
    autostart_requested = Signal(bool)  # True = เปิดพร้อม Windows
    poses_requested = Signal()
    studio_requested = Signal()
    reload_requested = Signal()
    quit_requested = Signal()

    def __init__(self, primary: str = "#00E5FF", parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._primary = QColor(primary)
        self._armed = False

        self._menu = QMenu()
        self._arm_action = QAction(self._menu)
        self._arm_action.triggered.connect(self.toggle_arm_requested)
        self._pause_action = QAction("พักกล้อง (ให้โปรแกรมอื่นใช้)", self._menu)
        self._pause_action.setCheckable(True)
        self._pause_action.toggled.connect(self.pause_requested)
        self._debug_action = QAction("หน้าต่าง Debug", self._menu)
        self._debug_action.setCheckable(True)
        self._debug_action.toggled.connect(self.debug_requested)
        self._calibrate_action = QAction("Calibrate…", self._menu)
        self._calibrate_action.triggered.connect(self.calibrate_requested)
        self._poses_action = QAction("สอนท่ามือ…", self._menu)
        self._poses_action.triggered.connect(self.poses_requested)
        self._studio_action = QAction("Studio — วาดรูป ฟิสิกส์ ดนตรี…", self._menu)
        self._studio_action.triggered.connect(self.studio_requested)
        self._autostart_action = QAction("เปิดพร้อม Windows", self._menu)
        self._autostart_action.setCheckable(True)
        self._autostart_action.setVisible(False)  # แสดงเมื่อ set_autostart(available=True)
        self._autostart_action.toggled.connect(self.autostart_requested)
        self._reload_action = QAction("โหลด config ใหม่", self._menu)
        self._reload_action.triggered.connect(self.reload_requested)
        self._quit_action = QAction("ออก", self._menu)
        self._quit_action.triggered.connect(self.quit_requested)

        self._menu.addAction(self._arm_action)
        self._menu.addAction(self._pause_action)
        self._menu.addSeparator()
        self._menu.addAction(self._debug_action)
        self._menu.addAction(self._calibrate_action)
        self._menu.addAction(self._poses_action)
        self._menu.addAction(self._studio_action)
        self._menu.addAction(self._autostart_action)
        self._menu.addAction(self._reload_action)
        self._menu.addSeparator()
        self._menu.addAction(self._quit_action)

        self._icon = QSystemTrayIcon(self)
        self._icon.setContextMenu(self._menu)
        self.set_armed(False)

    @staticmethod
    def available() -> bool:
        return QSystemTrayIcon.isSystemTrayAvailable()

    @property
    def menu(self) -> QMenu:
        return self._menu

    @property
    def armed(self) -> bool:
        return self._armed

    def actions(self) -> dict[str, QAction]:
        return {
            "arm": self._arm_action,
            "pause": self._pause_action,
            "debug": self._debug_action,
            "calibrate": self._calibrate_action,
            "poses": self._poses_action,
            "studio": self._studio_action,
            "autostart": self._autostart_action,
            "reload": self._reload_action,
            "quit": self._quit_action,
        }

    def show(self) -> None:
        self._icon.show()

    def hide(self) -> None:
        self._icon.hide()

    def set_primary(self, color: str) -> None:
        self._primary = QColor(color)
        self.set_armed(self._armed)

    def set_armed(self, armed: bool) -> None:
        self._armed = armed
        color = self._primary if armed else QColor(DISARMED)
        self._icon.setIcon(make_icon(color, armed))
        self._icon.setToolTip(f"AETHER — {'ARMED' if armed else 'DISARMED'}")
        self._arm_action.setText("Disarm (หยุดคุมด้วยมือ)" if armed else "Arm (เริ่มคุมด้วยมือ)")

    def set_autostart(self, available: bool, checked: bool) -> None:
        """แสดงเมนู "เปิดพร้อม Windows" (เฉพาะ Windows) และตั้งเครื่องหมายถูกโดยไม่ส่ง signal กลับ"""
        self._autostart_action.setVisible(available)
        blocked = self._autostart_action.blockSignals(True)
        self._autostart_action.setChecked(checked)
        self._autostart_action.blockSignals(blocked)

    def set_paused_checked(self, paused: bool) -> None:
        """ตั้งเครื่องหมายถูกของเมนูพักกล้องโดยไม่ส่ง signal กลับ"""
        blocked = self._pause_action.blockSignals(True)
        self._pause_action.setChecked(paused)
        self._pause_action.blockSignals(blocked)

    def set_debug_checked(self, checked: bool) -> None:
        """ตั้งเครื่องหมายถูกของเมนู Debug โดยไม่ส่ง signal กลับ"""
        blocked = self._debug_action.blockSignals(True)
        self._debug_action.setChecked(checked)
        self._debug_action.blockSignals(blocked)

    def notify(self, title: str, message: str, *, error: bool = False) -> None:
        """ข้อความแจ้งเตือนจาก tray (เช่น config ผิด)"""
        icon = (
            QSystemTrayIcon.MessageIcon.Critical
            if error
            else QSystemTrayIcon.MessageIcon.Information
        )
        self._icon.showMessage(title, message, icon, 6000)
