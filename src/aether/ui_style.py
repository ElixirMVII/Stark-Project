"""หน้าตาของหน้าต่างทั้งหมดของ AETHER (debug, wizard, Studio) — โทนเข้ม เรียบ เข้าชุดกับ HUD แบบ minimal

ใช้ style "Fusion" ของ Qt (หน้าตาเหมือนกันทุกเครื่อง) + palette เข้ม + stylesheet สั้นๆ ให้ปุ่ม/ช่องกรอก
มุมมน ขอบบาง สีเน้นทองแชมเปญ — เรียก :func:`apply_app_style` ครั้งเดียวหลังสร้าง ``QApplication``
"""

from __future__ import annotations

from PySide6.QtGui import QColor, QFont, QPalette
from PySide6.QtWidgets import QApplication

__all__ = [
    "ACCENT",
    "BACKGROUND",
    "DANGER",
    "OK",
    "SURFACE",
    "TEXT",
    "TEXT_DIM",
    "WARN",
    "apply_app_style",
    "stylesheet",
]

BACKGROUND = "#121417"  # พื้นหน้าต่าง
SURFACE = "#1A1D21"  # ช่องกรอก / รายการ / แผง
SURFACE_HOVER = "#22262B"
BORDER = "#2B3036"
TEXT = "#E8EAED"
TEXT_DIM = "#8B9098"
ACCENT = "#D6B67A"  # ทองแชมเปญ (ตรงกับธีม pearl)
ACCENT_TEXT = "#16130D"  # ตัวหนังสือบนพื้นสีเน้น
OK = "#7CC4A0"
WARN = "#D9A85B"
DANGER = "#E5484D"


def stylesheet() -> str:
    return f"""
QWidget {{ color: {TEXT}; }}
QToolTip {{
    color: {TEXT}; background: {SURFACE}; border: 1px solid {BORDER};
    border-radius: 6px; padding: 4px 8px;
}}
QPushButton {{
    background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 8px;
    padding: 6px 14px; min-height: 18px;
}}
QPushButton:hover {{ background: {SURFACE_HOVER}; }}
QPushButton:pressed {{ background: {BORDER}; }}
QPushButton:checked {{ background: {ACCENT}; color: {ACCENT_TEXT}; border-color: {ACCENT}; }}
QPushButton:disabled {{ color: {TEXT_DIM}; background: {BACKGROUND}; }}
QPushButton:default {{ border-color: {ACCENT}; }}
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QListWidget, QPlainTextEdit, QTextEdit {{
    background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 8px;
    padding: 4px 8px; selection-background-color: {ACCENT}; selection-color: {ACCENT_TEXT};
}}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
    border-color: {ACCENT};
}}
QComboBox::drop-down {{ border: none; width: 20px; }}
QListWidget::item {{ padding: 4px 6px; border-radius: 6px; }}
QListWidget::item:selected {{ background: {SURFACE_HOVER}; color: {ACCENT}; }}
QProgressBar {{
    background: {SURFACE}; border: none; border-radius: 3px; max-height: 6px;
}}
QProgressBar::chunk {{ background: {ACCENT}; border-radius: 3px; }}
QSlider::groove:horizontal {{ height: 4px; background: {BORDER}; border-radius: 2px; }}
QSlider::sub-page:horizontal {{ background: {ACCENT}; border-radius: 2px; }}
QSlider::handle:horizontal {{
    width: 14px; height: 14px; margin: -5px 0; border-radius: 7px; background: {TEXT};
}}
QTabBar::tab {{
    background: transparent; color: {TEXT_DIM}; padding: 8px 14px;
    border: none; border-bottom: 2px solid transparent;
}}
QTabBar::tab:selected {{ color: {TEXT}; border-bottom-color: {ACCENT}; }}
QTabWidget::pane {{ border: none; }}
QGroupBox {{
    border: 1px solid {BORDER}; border-radius: 10px; margin-top: 14px; padding: 10px 8px 8px;
}}
QGroupBox::title {{ subcontrol-origin: margin; left: 12px; padding: 0 4px; color: {TEXT_DIM}; }}
QMenu {{ background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 8px; padding: 4px; }}
QMenu::item {{ padding: 6px 18px; border-radius: 6px; }}
QMenu::item:selected {{ background: {SURFACE_HOVER}; }}
QMenu::separator {{ height: 1px; background: {BORDER}; margin: 4px 8px; }}
"""


def _palette() -> QPalette:
    palette = QPalette()
    roles = {
        QPalette.ColorRole.Window: BACKGROUND,
        QPalette.ColorRole.WindowText: TEXT,
        QPalette.ColorRole.Base: SURFACE,
        QPalette.ColorRole.AlternateBase: SURFACE_HOVER,
        QPalette.ColorRole.Text: TEXT,
        QPalette.ColorRole.Button: SURFACE,
        QPalette.ColorRole.ButtonText: TEXT,
        QPalette.ColorRole.Highlight: ACCENT,
        QPalette.ColorRole.HighlightedText: ACCENT_TEXT,
        QPalette.ColorRole.ToolTipBase: SURFACE,
        QPalette.ColorRole.ToolTipText: TEXT,
        QPalette.ColorRole.PlaceholderText: TEXT_DIM,
        QPalette.ColorRole.Link: ACCENT,
    }
    for role, color in roles.items():
        palette.setColor(role, QColor(color))
    for role in (
        QPalette.ColorRole.WindowText,
        QPalette.ColorRole.Text,
        QPalette.ColorRole.ButtonText,
    ):
        palette.setColor(QPalette.ColorGroup.Disabled, role, QColor(TEXT_DIM))
    return palette


def apply_app_style(app: QApplication) -> None:
    """ตั้งหน้าตาของทุกหน้าต่างใน process (ไม่มีผลกับ HUD ซึ่งวาดเองทั้งหมด)"""
    app.setStyle("Fusion")
    app.setPalette(_palette())
    font = QFont(app.font())
    for family in ("Segoe UI Variable Text", "Segoe UI"):
        if family in _families():
            font.setFamily(family)
            break
    app.setFont(font)
    app.setStyleSheet(stylesheet())


def _families() -> set[str]:
    from PySide6.QtGui import QFontDatabase

    return set(QFontDatabase.families())
