"""สี ฟอนต์ และตัวช่วยวาดเส้นเรืองแสงของ HUD

Glow ทำด้วยการวาดเส้นหนาโปร่งแสงหลายชั้นไว้ใต้เส้นบางที่สว่าง (ไม่ใช้ blur filter เพราะช้า)
ขนาดทั้งหมดเป็น logical px — Qt คูณ devicePixelRatio ให้ตอนวาด ภาพจึงคมทุก DPI
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Hashable
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QFontDatabase, QPainter, QPainterPath, QPen, QPixmap

from ..config import HudConfig

__all__ = [
    "DANGER",
    "DISARMED",
    "LayerCache",
    "Theme",
    "arc_path",
    "chamfered_rect",
    "draw_label",
    "glow_stroke",
    "set_glow_quality",
    "with_alpha",
]

log = logging.getLogger(__name__)

DANGER = "#FF3D57"
DISARMED = "#7A8B99"

DISPLAY_FONT_FILE = "Orbitron.ttf"  # หัวข้อ / ตัวเลขใหญ่
MONO_FONT_FILE = "ShareTechMono-Regular.ttf"  # ตัวเลข telemetry

# (ตัวคูณความหนา, alpha) ของชั้นเรืองแสง จากนอกเข้าใน — เส้นหนาโปร่งแสงเป็นส่วนที่แพงที่สุดของการวาด
# จึงใช้แค่สองชั้น (ชั้นที่สามแทบมองไม่เห็นแต่กินเวลาเพิ่มราว 40%)
_GLOW_LAYERS = ((4.5, 0.09), (2.2, 0.22))
_glow_layers = _GLOW_LAYERS  # hud.quality = "low" ตัดชั้นนอกออก (set_glow_quality)


def set_glow_quality(high: bool) -> None:
    """``False`` = ไม่วาด glow ชั้นนอก (ชั้นที่แพงที่สุด) — ใช้กับทุก overlay ใน process"""
    global _glow_layers
    _glow_layers = _GLOW_LAYERS if high else _GLOW_LAYERS[1:]


_loaded_families: dict[Path, str | None] = {}


def _load_family(path: Path) -> str | None:
    """โหลดฟอนต์จากไฟล์ (ครั้งเดียวต่อ process) คืนชื่อ family หรือ ``None`` ถ้าโหลดไม่ได้"""
    if path not in _loaded_families:
        family = None
        if path.is_file():
            font_id = QFontDatabase.addApplicationFont(str(path))
            families = QFontDatabase.applicationFontFamilies(font_id) if font_id >= 0 else []
            family = families[0] if families else None
        if family is None:
            log.info("font %s not available — using a system font instead", path.name)
        _loaded_families[path] = family
    return _loaded_families[path]


def with_alpha(color: QColor, alpha: float) -> QColor:
    result = QColor(color)
    result.setAlphaF(max(0.0, min(1.0, alpha)) * color.alphaF())
    return result


@dataclass(frozen=True)
class Theme:
    primary: QColor
    accent: QColor  # charge / warning / กำลังจีบ
    danger: QColor
    disarmed: QColor
    opacity: float
    display_family: str
    mono_family: str
    display_is_variable: bool = False

    @classmethod
    def from_config(cls, hud: HudConfig, fonts_dir: Path | None = None) -> Theme:
        """ต้องเรียกหลังสร้าง QApplication (ต้องใช้ font database)"""
        display = _load_family(fonts_dir / DISPLAY_FONT_FILE) if fonts_dir else None
        mono = _load_family(fonts_dir / MONO_FONT_FILE) if fonts_dir else None
        fixed = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont).family()
        general = QFontDatabase.systemFont(QFontDatabase.SystemFont.GeneralFont).family()
        palette = hud.palette
        return cls(
            primary=QColor(palette.primary),
            accent=QColor(palette.accent),
            danger=QColor(palette.danger),
            disarmed=QColor(palette.disarmed),
            opacity=hud.opacity,
            display_family=display or general,
            mono_family=mono or fixed,
            display_is_variable=display is not None,
        )

    def display_font(self, pixel_size: float, weight: int = 700, spacing: float = 0.0) -> QFont:
        font = QFont(self.display_family)
        font.setPixelSize(max(1, round(pixel_size)))
        font.setWeight(QFont.Weight(weight))
        if self.display_is_variable:  # Orbitron เป็น variable font: เลือกน้ำหนักผ่านแกน wght
            font.setVariableAxis(QFont.Tag("wght"), float(weight))
        if spacing:
            font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, spacing)
        return font

    def mono_font(self, pixel_size: float) -> QFont:
        font = QFont(self.mono_family)
        font.setPixelSize(max(1, round(pixel_size)))
        font.setStyleHint(QFont.StyleHint.Monospace)
        return font

    def state_color(self, armed: bool) -> QColor:
        return self.primary if armed else self.disarmed


def glow_stroke(
    painter: QPainter,
    path: QPainterPath,
    color: QColor,
    width: float = 1.5,
    strength: float = 1.0,
    alpha: float = 1.0,
) -> None:
    """วาดเส้น ``path`` แบบเรืองแสง: ชั้นหนาโปร่งแสงอยู่ใต้ เส้นบางสว่างอยู่บน"""
    if alpha <= 0.0:
        return
    painter.setBrush(Qt.BrushStyle.NoBrush)
    if strength > 0.0:
        for scale, layer_alpha in _glow_layers:
            pen = QPen(with_alpha(color, layer_alpha * strength * alpha), width * scale)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
            painter.setPen(pen)
            painter.drawPath(path)
    pen = QPen(with_alpha(color, alpha), width)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    painter.setPen(pen)
    painter.drawPath(path)


class LayerCache:
    """เก็บ "ชั้น" ที่วาดครั้งเดียวแล้วใช้ซ้ำทุกเฟรม (กรอบแผง, ส่วนที่ไม่ขยับของ reticle)

    ชั้นถูกวาดลง ``QPixmap`` ที่ความละเอียดของจอจริง (คูณ devicePixelRatio) จึงคมเท่ากับวาดสด
    key ต้องรวมทุกอย่างที่ทำให้ภาพต่างกัน (ขนาด สี) ส่วน devicePixelRatio ถูกรวมให้อัตโนมัติ
    """

    def __init__(self) -> None:
        self._layers: dict[tuple[Hashable, int], QPixmap] = {}

    def __len__(self) -> int:
        return len(self._layers)

    def draw(
        self,
        painter: QPainter,
        key: Hashable,
        top_left: QPointF,
        width: float,
        height: float,
        render: Callable[[QPainter], None],
    ) -> None:
        """วาดชั้น ``key`` ที่ ``top_left`` (logical px) — เรียก ``render`` เฉพาะครั้งแรก"""
        dpr = painter.device().devicePixelRatioF()
        cache_key = (key, round(dpr * 1000))
        pixmap = self._layers.get(cache_key)
        if pixmap is None:
            pixmap = QPixmap(math.ceil(width * dpr), math.ceil(height * dpr))
            pixmap.setDevicePixelRatio(dpr)
            pixmap.fill(Qt.GlobalColor.transparent)
            layer = QPainter(pixmap)
            layer.setRenderHint(QPainter.RenderHint.Antialiasing)
            layer.setRenderHint(QPainter.RenderHint.TextAntialiasing)
            render(layer)
            layer.end()
            self._layers[cache_key] = pixmap
        painter.drawPixmap(top_left, pixmap)


def arc_path(
    cx: float, cy: float, radius: float, start_deg: float, sweep_deg: float
) -> QPainterPath:
    """ส่วนโค้งของวงกลม — มุมแบบเข็มนาฬิกา: 0° = บนสุด, ค่าบวก = ตามเข็ม"""
    rect = QRectF(cx - radius, cy - radius, radius * 2, radius * 2)
    qt_start = 90.0 - start_deg  # Qt: 0° = ขวา, ค่าบวก = ทวนเข็ม
    path = QPainterPath()
    path.arcMoveTo(rect, qt_start)
    path.arcTo(rect, qt_start, -sweep_deg)
    return path


def draw_label(
    painter: QPainter, box: QRectF, flags: Qt.AlignmentFlag, text: str, color: QColor, alpha: float
) -> None:
    """ข้อความลอย (ไม่ได้อยู่ในแผง) พร้อมแถบเข้มโปร่งแสงรองหลัง ให้อ่านออกบนพื้นหลังสว่าง

    ใช้ฟอนต์ที่ตั้งไว้ใน painter แล้ว แถบรองหลังพอดีกับข้อความและไม่ล้น ``box``
    """
    if alpha <= 0.0 or not text:
        return
    bounds = painter.boundingRect(box, flags, text).adjusted(-6.0, -2.0, 6.0, 2.0).intersected(box)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QColor(2, 8, 14, round(255 * 0.5 * min(alpha, 1.0))))
    painter.drawRoundedRect(bounds, 4.0, 4.0)
    painter.setPen(with_alpha(color, alpha))
    painter.drawText(box, flags, text)


def chamfered_rect(rect: QRectF, cut: float) -> QPainterPath:
    """สี่เหลี่ยมตัดมุมซ้ายบนกับขวาล่าง (ทรงแผงของ HUD)"""
    path = QPainterPath()
    path.moveTo(rect.left() + cut, rect.top())
    path.lineTo(rect.right(), rect.top())
    path.lineTo(rect.right(), rect.bottom() - cut)
    path.lineTo(rect.right() - cut, rect.bottom())
    path.lineTo(rect.left(), rect.bottom())
    path.lineTo(rect.left(), rect.top() + cut)
    path.closeSubpath()
    return path
