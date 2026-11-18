"""สี ฟอนต์ และตัวช่วยวาดเส้นของ HUD — สองสไตล์ (``hud.style``)

- ``minimal`` (ค่าเริ่มต้น): เส้นบางทึบ มีเงาเข้มบางๆ ใต้เส้นให้อ่านออกบนพื้นสว่าง ฟอนต์ระบบ
  (Segoe UI Variable บน Windows 11) ข้อความตัวพิมพ์ปกติ ตัวเลขกว้างเท่ากัน (tabular)
- ``holo``: เส้นเรืองแสง ฟอนต์ Orbitron / Share Tech Mono ตัวพิมพ์ใหญ่ทั้งหมด

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
    "hairline_stroke",
    "set_glow_quality",
    "with_alpha",
]

log = logging.getLogger(__name__)

DANGER = "#FF3D57"
DISARMED = "#7A8B99"

DISPLAY_FONT_FILE = "Orbitron.ttf"  # holo: หัวข้อ / ตัวเลขใหญ่
MONO_FONT_FILE = "ShareTechMono-Regular.ttf"  # holo: ตัวเลข telemetry

# minimal: ฟอนต์ของระบบ เลือกตัวแรกที่มีในเครื่อง (ไม่มีเลย = ฟอนต์ทั่วไปของระบบ)
SANS_FAMILIES = (
    "Segoe UI Variable Text",  # Windows 11
    "Segoe UI",  # Windows 10
    "Inter",
    "SF Pro Text",
    "Helvetica Neue",
    "Noto Sans",
)
MINIMAL_MAX_WEIGHT = 600  # minimal ไม่ใช้ตัวหนาจัด
SHADOW = QColor(6, 8, 11)  # เงาใต้เส้นของ minimal
SHADOW_ALPHA = 0.38
SHADOW_EXTRA = 2.0  # เงากว้างกว่าเส้นเท่านี้ (logical px)

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


def sans_family() -> str:
    """ฟอนต์ sans ของระบบที่ใช้กับ minimal (ต้องมี QApplication แล้ว)"""
    available = set(QFontDatabase.families())
    for family in SANS_FAMILIES:
        if family in available:
            return family
    return QFontDatabase.systemFont(QFontDatabase.SystemFont.GeneralFont).family()


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
    style: str = "holo"
    text_family: str = ""  # ฟอนต์ข้อความของ minimal ("" = ฟอนต์ทั่วไปของระบบ)

    @classmethod
    def from_config(cls, hud: HudConfig, fonts_dir: Path | None = None) -> Theme:
        """ต้องเรียกหลังสร้าง QApplication (ต้องใช้ font database)"""
        palette = hud.palette
        general = QFontDatabase.systemFont(QFontDatabase.SystemFont.GeneralFont).family()
        colors = {
            "primary": QColor(palette.primary),
            "accent": QColor(palette.accent),
            "danger": QColor(palette.danger),
            "disarmed": QColor(palette.disarmed),
        }
        if hud.style == "minimal":
            sans = sans_family()
            return cls(
                **colors,
                opacity=hud.opacity,
                display_family=sans,
                mono_family=sans,
                style="minimal",
                text_family=sans,
            )
        display = _load_family(fonts_dir / DISPLAY_FONT_FILE) if fonts_dir else None
        mono = _load_family(fonts_dir / MONO_FONT_FILE) if fonts_dir else None
        fixed = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont).family()
        return cls(
            **colors,
            opacity=hud.opacity,
            display_family=display or general,
            mono_family=mono or fixed,
            display_is_variable=display is not None,
            text_family=general,
        )

    @property
    def minimal(self) -> bool:
        return self.style == "minimal"

    def label(self, text: str) -> str:
        """ข้อความบน HUD เขียนแบบตัวพิมพ์ปกติในโค้ด — holo แสดงเป็นตัวพิมพ์ใหญ่ทั้งหมด"""
        return text if self.minimal else text.upper()

    @property
    def secondary(self) -> QColor:
        """สีข้อความรอง (ป้ายกำกับ ค่าที่ไม่สำคัญ)"""
        return with_alpha(self.primary, 0.58)

    def text_font(self, pixel_size: float, weight: int = 400, spacing: float = 0.0) -> QFont:
        """ฟอนต์ข้อความ sans (minimal ใช้ทุกที่ holo ใช้กับข้อความยาว)"""
        font = QFont(self.text_family) if self.text_family else QFont()
        font.setPixelSize(max(1, round(pixel_size)))
        font.setWeight(QFont.Weight(weight))
        font.setHintingPreference(QFont.HintingPreference.PreferNoHinting)
        if spacing:
            font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, spacing)
        return font

    def display_font(self, pixel_size: float, weight: int = 700, spacing: float = 0.0) -> QFont:
        if self.minimal:  # ระยะห่างตัวอักษรของ holo ตั้งไว้สำหรับตัวพิมพ์ใหญ่ — minimal ไม่ใช้
            return self.text_font(pixel_size, min(weight, MINIMAL_MAX_WEIGHT))
        font = QFont(self.display_family)
        font.setPixelSize(max(1, round(pixel_size)))
        font.setWeight(QFont.Weight(weight))
        if self.display_is_variable:  # Orbitron เป็น variable font: เลือกน้ำหนักผ่านแกน wght
            font.setVariableAxis(QFont.Tag("wght"), float(weight))
        if spacing:
            font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, spacing)
        return font

    def mono_font(self, pixel_size: float) -> QFont:
        if self.minimal:  # ตัวเลขกว้างเท่ากัน (tabular figures) แทนฟอนต์ mono
            font = self.text_font(pixel_size, 500)
            font.setFeature(QFont.Tag("tnum"), 1)
            return font
        font = QFont(self.mono_family)
        font.setPixelSize(max(1, round(pixel_size)))
        font.setStyleHint(QFont.StyleHint.Monospace)
        return font

    def state_color(self, armed: bool) -> QColor:
        return self.primary if armed else self.disarmed

    def stroke(
        self,
        painter: QPainter,
        path: QPainterPath,
        color: QColor,
        width: float = 1.5,
        strength: float = 1.0,
        alpha: float = 1.0,
    ) -> None:
        """เส้นตามสไตล์: holo = เรืองแสง (``strength`` = ความแรงของ glow), minimal = เส้นบาง + เงา"""
        if self.minimal:
            hairline_stroke(painter, path, color, width, alpha=alpha)
        else:
            glow_stroke(painter, path, color, width, strength, alpha)


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


def hairline_stroke(
    painter: QPainter, path: QPainterPath, color: QColor, width: float = 1.2, alpha: float = 1.0
) -> None:
    """เส้นทึบบางของ minimal + เงาเข้มโปร่งแสงที่กว้างกว่าเล็กน้อยอยู่ใต้ (อ่านออกทั้งบนพื้นมืดและสว่าง)"""
    if alpha <= 0.0:
        return
    painter.setBrush(Qt.BrushStyle.NoBrush)
    for pen_color, pen_width in (
        (with_alpha(SHADOW, SHADOW_ALPHA * alpha), width + SHADOW_EXTRA),
        (with_alpha(color, alpha), width),
    ):
        pen = QPen(pen_color, pen_width)
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
    painter: QPainter,
    box: QRectF,
    flags: Qt.AlignmentFlag,
    text: str,
    color: QColor,
    alpha: float,
    *,
    pill: bool = False,
) -> None:
    """ข้อความลอย (ไม่ได้อยู่ในแผง) พร้อมแถบเข้มโปร่งแสงรองหลัง ให้อ่านออกบนพื้นหลังสว่าง

    ใช้ฟอนต์ที่ตั้งไว้ใน painter แล้ว แถบรองหลังพอดีกับข้อความและไม่ล้น ``box``
    ``pill=True`` (minimal) = แถบมนเต็มความสูง เข้มกว่าเล็กน้อย
    """
    if alpha <= 0.0 or not text:
        return
    if pill:  # ข้อความอยู่ในขอบของแคปซูล (ไม่ชิดขอบ box)
        text_box = box.adjusted(9.0, 0.0, -9.0, 0.0)
        bounds = painter.boundingRect(text_box, flags, text).adjusted(-9.0, -3.0, 9.0, 3.0)
        bounds = bounds.intersected(box)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(14, 16, 20, round(255 * 0.8 * min(alpha, 1.0))))
        painter.drawRoundedRect(bounds, bounds.height() / 2.0, bounds.height() / 2.0)
        painter.setPen(with_alpha(color, alpha))
        painter.drawText(text_box, flags, text)
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
