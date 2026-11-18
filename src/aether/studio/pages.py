"""หน้าทั้งสี่ของ Studio — แต่ละหน้าเป็น QWidget ที่วาดเองด้วย QPainter (หน้าตา minimal ชุดเดียวกับ HUD)

ทุกหน้ามี ``set_hands(hands, now)`` (เรียกทุกเฟรมของกล้อง) และ ``tick(now)`` (ทุก ~16 ms จาก timer ของหน้าต่าง)
เมาส์จริงบนผืนผ้าใบ = มือจำลอง (กดซ้าย = จีบ) ใช้ได้เมื่อไม่เห็นมือจากกล้อง
"""

from __future__ import annotations

import itertools
import math
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QImage,
    QKeySequence,
    QPainter,
    QPainterPath,
    QPen,
    QRadialGradient,
    QShortcut,
)
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSlider,
    QWidget,
)

from .. import ui_style
from ..assistant import prompts
from ..hud.theme import arc_path, hairline_stroke, sans_family, with_alpha
from ..tracker import Hand, HandFrame, hand_connections
from .claude_ui import StudioAnswer, png_bytes
from .draw import PALETTE, SWATCH_RADIUS, DrawController, Drawing, swatch_centers
from .input import StudioHand, mouse_hand
from .lab import FINGER_NAMES, HandMetrics, SampleRecorder, hand_metrics
from .sandbox import SandboxController, World
from .theremin import SCALES, ThereminController, Voice, frequency_for, note_name

__all__ = ["DrawPage", "LabPage", "SandboxPage", "StudioPage", "ThereminPage", "default_save_dir"]

CANVAS = QColor("#15171B")
GRID = QColor(255, 255, 255, 12)
PRIMARY = QColor("#F2F4F7")
ACCENT = QColor(ui_style.ACCENT)
DIM = QColor(ui_style.TEXT_DIM)
TOAST_SECONDS = 1.4
MOUSE_IDLE_SECONDS = 1.5  # เมาส์ไม่ขยับนานเท่านี้ = เลิกใช้เป็นมือจำลอง


def default_save_dir() -> Path:
    return Path.home() / "Pictures" / "AETHER"


def _font(pixel_size: float, weight: int = 400, *, tabular: bool = False) -> QFont:
    font = QFont(sans_family())
    font.setPixelSize(max(1, round(pixel_size)))
    font.setWeight(QFont.Weight(weight))
    if tabular:
        font.setFeature(QFont.Tag("tnum"), 1)
    return font


class StudioPage(QWidget):
    """ฐานของทุกหน้า: รับมือ, มือจำลองจากเมาส์, วาดเคอร์เซอร์, ข้อความแจ้งสั้นๆ (toast)"""

    title = ""
    hint = ""
    toast_shown = Signal(str)

    def __init__(
        self, parent: QWidget | None = None, *, clock: Callable[[], float] = time.perf_counter
    ) -> None:
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMinimumSize(480, 320)
        self._clock = clock
        self.hands: tuple[StudioHand, ...] = ()
        self._camera_hands: tuple[StudioHand, ...] = ()
        self._mouse: tuple[float, float, float] | None = None  # x, y (0–1), เวลาที่ขยับล่าสุด
        self._mouse_down = False
        self._mouse_velocity = (0.0, 0.0)
        self._toast: tuple[str, float] | None = None

    # ---- input ---------------------------------------------------------- #

    def set_hands(self, hands: tuple[StudioHand, ...], now: float) -> None:
        self._camera_hands = hands
        self._refresh_hands(now)
        self.on_hands(now)
        self.update()

    def _refresh_hands(self, now: float) -> None:
        if self._camera_hands:
            self.hands = self._camera_hands
        elif self._mouse is not None and (
            self._mouse_down or now - self._mouse[2] < MOUSE_IDLE_SECONDS
        ):
            x, y, _ = self._mouse
            self.hands = (mouse_hand(x, y, self._mouse_down, *self._mouse_velocity),)
        else:
            self.hands = ()

    def on_hands(self, now: float) -> None:
        """เรียกหลังอัปเดตมือ (หน้าลูกเอาไปใช้)"""

    def tick(self, now: float) -> None:
        if self._toast is not None and now - self._toast[1] > TOAST_SECONDS:
            self._toast = None
            self.update()
        elif self._toast is not None:
            self.update()

    def toast(self, text: str) -> None:
        self._toast = (text, self._clock())
        self.toast_shown.emit(text)
        self.update()

    def _mouse_event(self, event, pressed: bool | None = None) -> None:
        now = self._clock()
        x = min(max(event.position().x() / max(self.width(), 1), 0.0), 1.0)
        y = min(max(event.position().y() / max(self.height(), 1), 0.0), 1.0)
        if self._mouse is not None and now > self._mouse[2]:
            dt = now - self._mouse[2]
            vx, vy = (x - self._mouse[0]) / dt, (y - self._mouse[1]) / dt
            self._mouse_velocity = (
                0.6 * self._mouse_velocity[0] + 0.4 * vx,
                0.6 * self._mouse_velocity[1] + 0.4 * vy,
            )
        self._mouse = (x, y, now)
        if pressed is not None:
            self._mouse_down = pressed
        if not self._camera_hands:
            self._refresh_hands(now)
            self.on_hands(now)
            self.update()

    def mouseMoveEvent(self, event) -> None:
        self._mouse_event(event)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._mouse_event(event, True)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._mouse_event(event, False)

    def leaveEvent(self, event) -> None:
        if not self._mouse_down:
            self._mouse = None
            self._refresh_hands(self._clock())
            self.on_hands(self._clock())
            self.update()
        super().leaveEvent(event)

    # ---- paint helpers ---------------------------------------------------- #

    def to_px(self, x: float, y: float) -> QPointF:
        return QPointF(x * self.width(), y * self.height())

    def paint_background(self, painter: QPainter, *, grid: bool = True) -> None:
        painter.fillRect(self.rect(), CANVAS)
        if not grid:
            return
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(GRID)
        step = 32
        for x in range(step // 2, self.width(), step):
            for y in range(step // 2, self.height(), step):
                painter.drawEllipse(QPointF(x, y), 0.9, 0.9)

    def paint_cursors(self, painter: QPainter, hold: float = 0.0) -> None:
        for index, hand in enumerate(self.hands):
            center = self.to_px(hand.x, hand.y)
            ring = QPainterPath()
            ring.addEllipse(center, 12.0, 12.0)
            color = PRIMARY if index == 0 else with_alpha(PRIMARY, 0.6)
            hairline_stroke(painter, ring, color, 1.3, alpha=0.9)
            if hand.pinch_level > 0.02:
                arc = arc_path(center.x(), center.y(), 12.0, 0.0, 359.99 * hand.pinch_level)
                hairline_stroke(painter, arc, ACCENT, 2.0)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(ACCENT if hand.pinch else color)
            painter.drawEllipse(center, 3.0 if hand.pinch else 1.8, 3.0 if hand.pinch else 1.8)
            if index == 0 and hold > 0.0:
                hairline_stroke(
                    painter, arc_path(center.x(), center.y(), 19.0, 0.0, 359.99 * hold), ACCENT, 1.5
                )

    def paint_toast(self, painter: QPainter) -> None:
        if self._toast is None:
            return
        text, start = self._toast
        age = self._clock() - start
        alpha = min(1.0, age / 0.12) * min(1.0, max(0.0, (TOAST_SECONDS - age) / 0.35))
        if alpha <= 0.0:
            return
        painter.setFont(_font(13, 500))
        width = painter.fontMetrics().horizontalAdvance(text) + 32
        box = QRectF((self.width() - width) / 2, self.height() - 64, width, 32)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(28, 31, 36, round(235 * alpha)))
        painter.drawRoundedRect(box, 16, 16)
        painter.setPen(with_alpha(PRIMARY, alpha))
        painter.drawText(box, Qt.AlignmentFlag.AlignCenter, text)

    def paint_empty_hint(self, painter: QPainter, text: str) -> None:
        if self.hands:
            return
        painter.setFont(_font(14, 400))
        painter.setPen(DIM)
        painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, text)


# --------------------------------------------------------------------------- #
# วาดรูปในอากาศ
# --------------------------------------------------------------------------- #


def stroke_path(points: list[tuple[float, float]], width: float, height: float) -> QPainterPath:
    """เส้นนุ่ม: โค้ง quadratic ผ่านจุดกึ่งกลางระหว่างจุดที่เก็บไว้"""
    path = QPainterPath()
    pts = [QPointF(x * width, y * height) for x, y in points]
    if not pts:
        return path
    path.moveTo(pts[0])
    if len(pts) == 1:
        path.lineTo(pts[0] + QPointF(0.01, 0.0))
        return path
    for a, b in itertools.pairwise(pts[1:]):
        path.quadTo(a, (a + b) / 2.0)
    path.lineTo(pts[-1])
    return path


def render_drawing(painter: QPainter, drawing: Drawing, width: float, height: float) -> None:
    scale = width / 1000.0
    for stroke in drawing.strokes:
        pen = QPen(QColor(stroke.color), max(stroke.width * scale, 1.0))
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPath(stroke_path(stroke.points, width, height))


class DrawPage(StudioPage):
    title = "Draw"
    hint = "Pinch to draw  ·  pinch a color to pick it  ·  hold ✊ undo  ·  hold ✌ next color  ·  hold 👍 save  ·  hold 👎 clear"

    def __init__(
        self, parent=None, *, clock=time.perf_counter, save_dir: Path | None = None
    ) -> None:
        super().__init__(parent, clock=clock)
        self.drawing = Drawing()
        self.controller = DrawController(self.drawing)
        self.save_dir = save_dir or default_save_dir()
        self.last_saved: Path | None = None
        # Claude ทายภาพ/ติชม: ส่งเฉพาะภาพที่วาด (ไม่ใช่ภาพจากกล้อง) — ตั้งโดย StudioWindow
        self.claude = None
        self.on_answer: Callable[[str, str], None] | None = None
        self.answer = StudioAnswer(clock)
        self._answer_strokes = 0

    def resizeEvent(self, event) -> None:
        self.controller.aspect = self.width() / max(self.height(), 1)
        super().resizeEvent(event)

    def on_hands(self, now: float) -> None:
        for event in self.controller.update(self.hands[0] if self.hands else None, now):
            if event.kind == "save":
                self.save_png()
            else:
                self.toast(event.text)
        # เริ่มวาดเส้นใหม่หลังอ่านคำตอบแล้ว = ปิดการ์ดคำตอบ
        if not self.answer.busy and len(self.drawing.strokes) > self._answer_strokes:
            self.answer.dismiss()

    # ---- Claude --------------------------------------------------------- #

    def ask_claude(self, question: str | None = None) -> bool:
        """ส่งภาพที่วาด (PNG กว้าง 768 px) ให้ Claude ทายและติชม — คำตอบขึ้นเป็นการ์ดมุมขวาบน"""
        if not self.drawing.strokes:
            self.toast("Draw something first")
            return False
        if self.claude is None:
            self.toast("Claude is not available")
            return False
        if self.answer.busy:
            self.toast("Claude is still answering")
            return False
        system, content = prompts.drawing_request(png_bytes(self.image(768)), question)
        asked = question.strip() if question else "ทายภาพที่วาด"
        self.answer.begin("Claude · your drawing")
        self._answer_strokes = len(self.drawing.strokes)

        def done(text: str) -> None:
            self.answer.finish(text)
            if self.on_answer is not None:
                self.on_answer(f"[Draw] {asked}", text)
            self.update()

        def failed(message: str) -> None:
            self.answer.fail(message)
            self.update()

        def piece(text: str) -> None:
            self.answer.append(text)
            self.update()

        self.claude.stream(system, content, on_text=piece, on_result=done, on_error=failed)
        self.update()
        return True

    def answer_rect(self) -> QRectF:
        width = min(420.0, max(260.0, self.width() * 0.42))
        height = min(self.height() * 0.6, self.answer.height_for(width))
        return QRectF(self.width() - width - 16, 16, width, height)

    def tick(self, now: float) -> None:
        super().tick(now)
        if self.answer.busy:
            self.update()

    def undo(self) -> None:
        if self.drawing.undo():
            self.toast("Undo")
            self.update()

    def clear(self) -> None:
        self.drawing.clear()
        self.answer.dismiss()
        self._answer_strokes = 0
        self.toast("Cleared")
        self.update()

    def cycle_size(self) -> None:
        self.drawing.cycle_size()
        self.toast(f"Brush {self.drawing.width:.0f}")
        self.update()

    def image(self, width: int = 1920, height: int | None = None) -> QImage:
        """ภาพวาดบนพื้นผืนผ้าใบ (สัดส่วนเดียวกับหน้าต่าง)"""
        height = height or round(width * self.height() / max(self.width(), 1))
        image = QImage(width, height, QImage.Format.Format_ARGB32)
        image.fill(CANVAS)
        painter = QPainter(image)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        render_drawing(painter, self.drawing, width, height)
        painter.end()
        return image

    def save_png(self, path: Path | None = None) -> Path | None:
        if path is None:
            stamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
            path = self.save_dir / f"drawing-{stamp}.png"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            if not self.image().save(str(path)):
                raise OSError(f"save failed: {path}")
        except OSError:
            self.toast("Could not save the drawing")
            return None
        self.last_saved = path
        self.toast(f"Saved {path.name}")
        return path

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.paint_background(painter)
        render_drawing(painter, self.drawing, self.width(), self.height())
        self._paint_swatches(painter)
        self.paint_empty_hint(painter, "Show your hand to the camera — or draw with the mouse")
        self.answer.paint(painter, self.answer_rect())
        self.paint_cursors(painter, self.controller.hold_progress)
        self.paint_toast(painter)
        painter.end()

    def _paint_swatches(self, painter: QPainter) -> None:
        width, height = self.width(), self.height()
        radius = SWATCH_RADIUS * width
        for index, (sx, sy) in enumerate(swatch_centers(width / max(height, 1))):
            center = QPointF(sx * width, sy * height)
            painter.setPen(QPen(QColor(255, 255, 255, 40), 1.0))
            painter.setBrush(QColor(PALETTE[index]))
            painter.drawEllipse(center, radius, radius)
            if index == self.drawing.color_index:
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.setPen(QPen(PRIMARY, 1.4))
                painter.drawEllipse(center, radius + 4.0, radius + 4.0)


# --------------------------------------------------------------------------- #
# กระบะทรายฟิสิกส์
# --------------------------------------------------------------------------- #


class SandboxPage(StudioPage):
    title = "Sandbox"
    hint = "Pinch a ball to grab · let go to throw  ·  open palm to push  ·  hold ✌ new ball  ·  hold 👎 reset"

    def __init__(self, parent=None, *, clock=time.perf_counter) -> None:
        super().__init__(parent, clock=clock)
        self.world = World(800.0, 500.0)
        self.world.reset()
        self.controller = SandboxController(self.world)
        self._last_tick: float | None = None

    def resizeEvent(self, event) -> None:
        self.world.resize(float(self.width()), float(self.height()))
        super().resizeEvent(event)

    def on_hands(self, now: float) -> None:
        for text in self.controller.update(self.hands, now):
            self.toast(text)

    def add_ball(self) -> None:
        self.world.spawn(self.width() * 0.5, self.height() * 0.2)
        self.update()

    def reset(self) -> None:
        self.world.reset()
        self.toast("Reset")
        self.update()

    def tick(self, now: float) -> None:
        dt = 0.0 if self._last_tick is None else min(now - self._last_tick, 0.1)
        self._last_tick = now
        self.world.step(dt)
        super().tick(now)
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.paint_background(painter, grid=False)
        painter.setPen(QPen(QColor(255, 255, 255, 18), 1.0))
        painter.drawLine(
            QPointF(0, self.height() - 0.5), QPointF(self.width(), self.height() - 0.5)
        )
        for ball in self.world.balls:
            center = QPointF(ball.x, ball.y)
            # เงาบนพื้นจางๆ (ไกลพื้น = จางและเล็ก)
            lift = min((self.height() - ball.y - ball.r) / max(self.height(), 1), 1.0)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(0, 0, 0, round(70 * (1.0 - lift))))
            painter.drawEllipse(
                QPointF(ball.x, self.height() - 3), ball.r * (1.0 - 0.5 * lift), 3.0
            )
            base = QColor(ball.color)
            shade = QRadialGradient(center + QPointF(-ball.r * 0.35, -ball.r * 0.4), ball.r * 1.4)
            shade.setColorAt(0.0, base.lighter(118))
            shade.setColorAt(1.0, base.darker(135))
            painter.setBrush(QBrush(shade))
            painter.drawEllipse(center, ball.r, ball.r)
            if ball.held_by is not None:
                ring = QPainterPath()
                ring.addEllipse(center, ball.r + 5.0, ball.r + 5.0)
                hairline_stroke(painter, ring, ACCENT, 1.4)
        self.paint_cursors(painter, self.controller.hold_progress)
        self.paint_toast(painter)
        painter.end()


# --------------------------------------------------------------------------- #
# เครื่องดนตรีอากาศ
# --------------------------------------------------------------------------- #


class ThereminPage(StudioPage):
    title = "Theremin"
    hint = "Pinch to play  ·  raise your hand for higher notes  ·  move right for vibrato  ·  second hand = volume"

    def __init__(self, parent=None, *, clock=time.perf_counter, audio: bool = True) -> None:
        super().__init__(parent, clock=clock)
        self.voice = Voice()
        self.controller = ThereminController(self.voice)
        self._audio_wanted = audio
        self.audio = None
        self._scope: list[float] = []

    def ensure_audio(self) -> bool:
        if self.audio is None and self._audio_wanted:
            from .audio import AudioOut

            self.audio = AudioOut(self.voice)
        return bool(self.audio is not None and self.audio.available)

    def start_audio(self) -> None:
        if self.ensure_audio():
            self.audio.start()

    def stop_audio(self) -> None:
        if self.audio is not None:
            self.audio.stop()
        self.voice.amplitude = 0.0

    def set_scale(self, scale: str) -> None:
        self.controller.scale = scale
        self.update()

    PAD = 28.0  # px เว้นบน/ล่างให้ป้ายโน้ตไม่ถูกตัด

    def _pitch_y(self, height: float) -> float:
        return self.PAD + (1.0 - height) * (self.height() - 2 * self.PAD)

    def on_hands(self, now: float) -> None:
        self.controller.update(self.hands, self.PAD / max(self.height(), 1))

    def tick(self, now: float) -> None:
        if self.audio is not None and self.audio.running:
            self.audio.pump()
        super().tick(now)
        self.update()

    def scope_samples(self) -> list[float]:
        """คลื่นตัวอย่างสั้นๆ สำหรับวาด (ไม่เกี่ยวกับเสียงจริงที่ส่งออกลำโพง)"""
        state = self.controller.state
        if state.amplitude <= 0.0 and self.voice.current_amplitude < 0.005:
            return [0.0] * 160
        cycles = 3.0
        return [
            state.amplitude
            * math.sin(math.tau * cycles * i / 160)
            * (0.82 + 0.18 * math.sin(math.tau * 2 * cycles * i / 160))
            for i in range(160)
        ]

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.paint_background(painter, grid=False)
        width, height = self.width(), self.height()
        state = self.controller.state
        degrees = SCALES[self.controller.scale]
        # เส้นของโน้ตในบันได (หรือทุกครึ่งเสียงเมื่อ free)
        painter.setFont(_font(11, 500))
        steps = range(25)
        for semitone in steps:
            if degrees is not None and semitone % 12 not in degrees:
                continue
            y = self._pitch_y(semitone / 24.0)
            octave_line = semitone % 12 == 0
            painter.setPen(QPen(QColor(255, 255, 255, 34 if octave_line else 14), 1.0))
            painter.drawLine(QPointF(64, y), QPointF(width - 24, y))
            painter.setPen(DIM if not octave_line else PRIMARY)
            freq = 220.0 * 2 ** (semitone / 12)
            painter.drawText(
                QRectF(12, y - 9, 44, 18),
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                note_name(freq),
            )

        if self.hands:
            y = self._pitch_y(state.height)
            snapped = frequency_for(state.height, self.controller.scale)
            sy = self._pitch_y(12.0 * math.log2(snapped / 220.0) / 24.0)
            pen = QPen(ACCENT if state.playing else with_alpha(PRIMARY, 0.4), 1.4)
            painter.setPen(pen)
            painter.drawLine(QPointF(64, sy), QPointF(width - 24, sy))
            painter.setPen(QPen(with_alpha(PRIMARY, 0.25), 1.0, Qt.PenStyle.DashLine))
            painter.drawLine(QPointF(64, y), QPointF(width - 24, y))

        # ชื่อโน้ตตัวใหญ่ + ความถี่
        box = QRectF(width - 280, 24, 250, 90)
        painter.setPen(ACCENT if state.playing else PRIMARY)
        painter.setFont(_font(54, 300))
        painter.drawText(
            box,
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop,
            state.note if self.hands else "—",
        )
        painter.setPen(DIM)
        painter.setFont(_font(13, 400, tabular=True))
        detail = f"{state.frequency:.1f} Hz" if self.hands else "Show a hand, then pinch to play"
        if self.audio is not None and not self.audio.available:
            detail += "  ·  no audio output"
        painter.drawText(QRectF(width - 380, 112, 350, 20), Qt.AlignmentFlag.AlignRight, detail)

        # scope
        samples = self.scope_samples()
        scope = QRectF(width - 280, height - 110, 250, 70)
        path = QPainterPath()
        for i, value in enumerate(samples):
            point = QPointF(
                scope.left() + scope.width() * i / (len(samples) - 1),
                scope.center().y() - value * scope.height() / 2,
            )
            if i == 0:
                path.moveTo(point)
            else:
                path.lineTo(point)
        painter.setPen(QPen(ACCENT if state.playing else with_alpha(PRIMARY, 0.3), 1.4))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPath(path)

        self.paint_cursors(painter)
        self.paint_toast(painter)
        painter.end()


# --------------------------------------------------------------------------- #
# ห้องทดลองมือ
# --------------------------------------------------------------------------- #


class LabPage(StudioPage):
    title = "Hand lab"
    hint = "Live finger states and angles  ·  type a label and press Record to save samples (numbers only, no images)"

    BURST_SECONDS = 3.0
    BURST_EVERY = 0.1

    def __init__(
        self, parent=None, *, clock=time.perf_counter, samples_path: Path | None = None
    ) -> None:
        super().__init__(parent, clock=clock)
        self.recorder = SampleRecorder(samples_path) if samples_path else None
        self._frame: HandFrame | None = None
        self.metrics: HandMetrics | None = None
        self._connections = hand_connections()
        self.label = ""
        self._burst_until: float | None = None
        self._next_sample = 0.0

    def show_hand_frame(self, frame: HandFrame) -> None:
        self._frame = frame
        hand = self._hand()
        self.metrics = hand_metrics(hand, frame.width, frame.height) if hand else None
        now = self._clock()
        if self._burst_until is not None:
            if now >= self._burst_until:
                self._burst_until = None
                self.toast(
                    f"Saved · {self.label} × {self.recorder.counts[self.label] if self.recorder else 0}"
                )
            elif now >= self._next_sample:
                self._next_sample = now + self.BURST_EVERY
                self.record_once(quiet=True)
        self.update()

    def _hand(self) -> Hand | None:
        if self._frame is None or not self._frame.hands:
            return None
        return max(self._frame.hands, key=lambda h: (h.handedness == "right", h.handedness_score))

    def record_once(self, *, quiet: bool = False) -> bool:
        hand = self._hand()
        if self.recorder is None or hand is None or not self.label.strip():
            if not quiet:
                self.toast("Show a hand and type a label first")
            return False
        self.recorder.add(self.label, hand, self._frame.width, self._frame.height)
        if not quiet:
            self.toast(f"Saved · {self.label} × {self.recorder.counts[self.label.strip()]}")
        return True

    def start_burst(self) -> None:
        """เก็บตัวอย่างทุก 0.1 วินาทีเป็นเวลา 3 วินาที (ขยับมือเล็กน้อยระหว่างนั้น ได้ข้อมูลหลากหลายขึ้น)"""
        if self._hand() is None or not self.label.strip() or self.recorder is None:
            self.toast("Show a hand and type a label first")
            return
        self._burst_until = self._clock() + self.BURST_SECONDS
        self._next_sample = 0.0

    @property
    def bursting(self) -> bool:
        return self._burst_until is not None

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.paint_background(painter, grid=False)
        width, height = self.width(), self.height()
        hand = self._hand()
        stage = QRectF(24, 24, width * 0.5 - 36, height - 48)
        painter.setPen(QPen(QColor(255, 255, 255, 16), 1.0))
        painter.setBrush(QColor(255, 255, 255, 5))
        painter.drawRoundedRect(stage, 14, 14)
        if hand is None or self._frame is None:
            painter.setPen(DIM)
            painter.setFont(_font(14))
            painter.drawText(stage, Qt.AlignmentFlag.AlignCenter, "Show your hand to the camera")
        else:
            self._paint_skeleton(painter, hand, stage.adjusted(30, 30, -30, -30))
        self._paint_metrics(painter, QRectF(width * 0.5 + 12, 24, width * 0.5 - 36, height - 48))
        if self.bursting:
            left = max(0.0, (self._burst_until or 0) - self._clock())
            painter.setPen(ACCENT)
            painter.setFont(_font(13, 500))
            painter.drawText(
                stage.adjusted(16, 12, -16, -12),
                Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignRight,
                f"Recording {left:.1f}s",
            )
        self.paint_toast(painter)
        painter.end()

    def _paint_skeleton(self, painter: QPainter, hand: Hand, area: QRectF) -> None:
        frame = self._frame
        aspect = frame.width / max(frame.height, 1)
        xs = [p.x * aspect for p in hand.landmarks]
        ys = [p.y for p in hand.landmarks]
        span = max(max(xs) - min(xs), max(ys) - min(ys), 1e-6)
        scale = min(area.width(), area.height()) / span
        mx, my = (max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2
        points = [
            QPointF(area.center().x() + (x - mx) * scale, area.center().y() + (y - my) * scale)
            for x, y in zip(xs, ys, strict=True)
        ]
        metrics = self.metrics
        finger_of = {}
        for finger, chain in enumerate(
            ((1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20))
        ):
            for index in chain:
                finger_of[index] = finger
        for start, end in self._connections:
            finger = finger_of.get(end)
            lit = metrics is not None and finger is not None and metrics.extended[finger]
            pen = QPen(PRIMARY if lit else with_alpha(PRIMARY, 0.35), 2.0)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(pen)
            painter.drawLine(points[start], points[end])
        painter.setPen(Qt.PenStyle.NoPen)
        for index, point in enumerate(points):
            tip = index in (4, 8, 12, 16, 20)
            painter.setBrush(ACCENT if tip else PRIMARY)
            painter.drawEllipse(point, 3.6 if tip else 2.6, 3.6 if tip else 2.6)

    def _paint_metrics(self, painter: QPainter, area: QRectF) -> None:
        metrics = self.metrics
        left = Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        right = Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        y = area.top()
        painter.setPen(DIM)
        painter.setFont(_font(12, 500))
        painter.drawText(QRectF(area.left(), y, area.width(), 18), left, "Fingers up")
        painter.setPen(PRIMARY)
        painter.setFont(_font(64, 300, tabular=True))
        painter.drawText(
            QRectF(area.left(), y + 14, 120, 76), left, str(metrics.count) if metrics else "–"
        )
        if metrics is not None:
            painter.setFont(_font(13, 400))
            painter.setPen(DIM)
            side = f"{metrics.handedness.capitalize()} hand · {metrics.facing} · roll {metrics.roll:+.0f}°"
            painter.drawText(QRectF(area.left() + 120, y + 26, area.width() - 120, 20), left, side)
            gesture = (
                metrics.gesture.replace("_", " ") if metrics.gesture != "None" else "no gesture"
            )
            painter.setPen(PRIMARY)
            painter.drawText(
                QRectF(area.left() + 120, y + 50, area.width() - 120, 20),
                left,
                f"{gesture}  {metrics.gesture_score:.0%}" if metrics.gesture != "None" else gesture,
            )
        y += 110
        painter.setFont(_font(12, 400))
        for i, name in enumerate(FINGER_NAMES):
            row = QRectF(area.left(), y, area.width(), 22)
            painter.setPen(DIM)
            painter.drawText(row, left, name)
            bend = metrics.bend[i] if metrics else 0.0
            bar = QRectF(area.left() + 70, y + 10, max(area.width() - 200, 40), 3)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(255, 255, 255, 22))
            painter.drawRoundedRect(bar, 1.5, 1.5)
            level = min(bend / 270.0, 1.0)
            painter.setBrush(ACCENT if metrics and metrics.curled[i] else with_alpha(PRIMARY, 0.7))
            painter.drawRoundedRect(QRectF(bar.left(), bar.top(), bar.width() * level, 3), 1.5, 1.5)
            painter.setPen(PRIMARY)
            painter.setFont(_font(12, 500, tabular=True))
            state = (
                "–"
                if metrics is None
                else ("up" if metrics.extended[i] else ("curled" if metrics.curled[i] else "bent"))
            )
            painter.drawText(row, right, f"{bend:4.0f}°  {state}" if metrics else "–")
            painter.setFont(_font(12, 400))
            y += 28
        y += 8
        painter.setPen(DIM)
        painter.drawText(
            QRectF(area.left(), y, area.width(), 20), left, "Thumb to finger (÷ hand size)"
        )
        y += 22
        if metrics is not None:
            painter.setPen(PRIMARY)
            painter.setFont(_font(12, 500, tabular=True))
            text = "   ".join(
                f"{n[0]} {d:.2f}" for n, d in zip(FINGER_NAMES[1:], metrics.pinch, strict=True)
            )
            painter.drawText(QRectF(area.left(), y, area.width(), 20), left, text)
        y += 32
        if self.recorder is not None and self.recorder.counts:
            painter.setPen(DIM)
            painter.setFont(_font(12, 400))
            counts = "  ·  ".join(
                f"{label} {count}" for label, count in sorted(self.recorder.counts.items())
            )
            painter.drawText(
                QRectF(area.left(), y, area.width(), 40),
                Qt.AlignmentFlag.AlignLeft | Qt.TextFlag.TextWordWrap,
                f"Saved samples: {counts}",
            )


def lab_controls(page: LabPage) -> QWidget:
    """แถบควบคุมใต้หน้า Hand lab: ช่องป้าย + ปุ่มเก็บตัวอย่าง"""
    bar = QWidget()
    layout = QHBoxLayout(bar)
    layout.setContentsMargins(0, 0, 0, 0)
    label = QLineEdit()
    label.setPlaceholderText("Label, e.g. peace, rock, ok")
    label.setMaxLength(32)
    label.textChanged.connect(lambda text: setattr(page, "label", text))
    once = QPushButton("Record")
    once.clicked.connect(lambda: page.record_once())
    burst = QPushButton("Record 3 s")
    burst.clicked.connect(page.start_burst)
    layout.addWidget(QLabel("Label"))
    layout.addWidget(label, 1)
    layout.addWidget(once)
    layout.addWidget(burst)
    bar.label_edit = label  # type: ignore[attr-defined]
    return bar


def draw_controls(page: DrawPage) -> QWidget:
    bar = QWidget()
    layout = QHBoxLayout(bar)
    layout.setContentsMargins(0, 0, 0, 0)
    for text, slot in (
        ("Undo", page.undo),
        ("Brush size", page.cycle_size),
        ("Clear", page.clear),
        ("Save PNG", page.save_png),
    ):
        button = QPushButton(text)
        button.clicked.connect(lambda _=False, s=slot: s())
        layout.addWidget(button)
    layout.addStretch(1)
    question = QLineEdit()
    question.setPlaceholderText("Ask Claude about the drawing (optional)")
    question.setMinimumWidth(220)
    ask = QPushButton("Ask Claude")

    def send() -> None:
        if page.ask_claude(question.text() or None):
            question.clear()

    ask.clicked.connect(send)
    question.returnPressed.connect(send)
    layout.addWidget(question)
    layout.addWidget(ask)
    bar.claude_question = question  # type: ignore[attr-defined]
    bar.claude_button = ask  # type: ignore[attr-defined]
    return bar


def sandbox_controls(page: SandboxPage) -> QWidget:
    bar = QWidget()
    layout = QHBoxLayout(bar)
    layout.setContentsMargins(0, 0, 0, 0)
    add = QPushButton("Add ball")
    add.clicked.connect(page.add_ball)
    reset = QPushButton("Reset")
    reset.clicked.connect(page.reset)
    gravity = QSlider(Qt.Orientation.Horizontal)
    gravity.setRange(0, 3000)
    gravity.setValue(int(page.world.gravity))
    gravity.setFixedWidth(160)
    gravity.valueChanged.connect(lambda value: setattr(page.world, "gravity", float(value)))
    layout.addWidget(add)
    layout.addWidget(reset)
    layout.addSpacing(16)
    layout.addWidget(QLabel("Gravity"))
    layout.addWidget(gravity)
    layout.addStretch(1)
    return bar


def theremin_controls(page: ThereminPage) -> QWidget:
    bar = QWidget()
    layout = QHBoxLayout(bar)
    layout.setContentsMargins(0, 0, 0, 0)
    scale = QComboBox()
    scale.addItems(list(SCALES))
    scale.currentTextChanged.connect(page.set_scale)
    volume = QSlider(Qt.Orientation.Horizontal)
    volume.setRange(0, 100)
    volume.setValue(int(page.controller.volume * 100))
    volume.setFixedWidth(160)
    volume.valueChanged.connect(lambda value: setattr(page.controller, "volume", value / 100))
    layout.addWidget(QLabel("Scale"))
    layout.addWidget(scale)
    layout.addSpacing(16)
    layout.addWidget(QLabel("Volume"))
    layout.addWidget(volume)
    layout.addStretch(1)
    return bar


def page_shortcuts(widget: QWidget, bindings: dict[str, Callable[[], None]]) -> None:
    for key, slot in bindings.items():
        shortcut = QShortcut(QKeySequence(key), widget)
        shortcut.activated.connect(slot)


__all__ += [
    "draw_controls",
    "lab_controls",
    "page_shortcuts",
    "render_drawing",
    "sandbox_controls",
    "stroke_path",
    "theremin_controls",
]
