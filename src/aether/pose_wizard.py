"""หน้าต่างสอนท่ามือ — สอนท่าใหม่, ลบท่า, และลองท่าที่สอนไว้ (โดยไม่ต้อง ARM)

ขั้นตอนสอนอยู่ใน :class:`aether.pose_teach.TeachSession` (pure) หน้าต่างนี้แค่รับเฟรม
(:meth:`PoseWizard.show_frame`) แสดงผล และเขียน ``poses.json`` ตอนกดบันทึก/ลบ
ปิดหน้าต่างแล้วส่ง ``finished(True)`` ถ้าไฟล์เปลี่ยน ให้แอปโหลดชุดใหม่ (ระหว่างเปิด แอปอยู่ในสถานะ DISARMED)
"""

from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from . import ui_style
from .calibration_wizard import HandPreview
from .config import Config
from .hotkeys import parse_action
from .pose_teach import TeachSession
from .poses import MAX_NEGATIVES, PoseClassifier, PoseFileError, PoseLibrary, save_poses
from .tracker import HandFrame
from .training import load_builtin_pack

__all__ = ["ACTION_PRESETS", "PoseWizard"]

log = logging.getLogger(__name__)

_WARN = ui_style.WARN
_OK = ui_style.OK

# ปุ่มลัดที่ให้เลือกง่ายๆ (พิมพ์เองได้ทุกแบบที่ [actions] ใน config.toml รับ)
ACTION_PRESETS = (
    "ctrl+c",
    "ctrl+v",
    "ctrl+z",
    "space",
    "alt+tab",
    "ctrl+w",
    "ctrl+t",
    "win+e",
    "win+shift+s",
    "f5",
    "media_play_pause",
    "minimize_foreground",
    "ask_claude",
)

_REASONS = {"far": "ไม่ตรงท่าไหน", "neutral": "มือปกติ", "ambiguous": "ก้ำกึ่งหลายท่า"}


class PoseWizard(QWidget):
    """เรียก :meth:`show_frame` จาก UI thread ทุกเฟรม"""

    finished = Signal(bool)  # True = poses.json เปลี่ยน (สอน/ลบท่าแล้ว), False = ไม่มีอะไรเปลี่ยน

    def __init__(
        self,
        config: Config,
        library: PoseLibrary,
        path: Path,
        parent: QWidget | None = None,
        *,
        pack: tuple[PoseLibrary, dict] | None = None,
    ) -> None:
        super().__init__(parent)
        self._config = config
        self._library = library
        if pack is None:
            try:
                pack = load_builtin_pack()
            except PoseFileError as exc:
                log.warning("ชุดท่าสำเร็จรูปเสีย — ไม่แสดง: %s", exc)
                pack = (PoseLibrary(), {})
        self._pack, self._pack_meta = pack
        self._classifier = PoseClassifier(library, config.poses.sensitivity)
        self._path = path
        self._session: TeachSession | None = None
        self._changed = False
        self._reported = False

        self.setWindowTitle("AETHER — สอนท่ามือ")
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint)
        self.setMinimumWidth(520)

        title_font = QFont(self.font())
        title_font.setPointSize(title_font.pointSize() + 4)
        title_font.setBold(True)

        # ---- ท่าที่สอนไว้ ----
        self._list = QListWidget()
        self._list.setMaximumHeight(110)
        self._list.currentRowChanged.connect(lambda _row: self._refresh())
        self._delete_button = QPushButton("ลบท่าที่เลือก")
        self._delete_button.clicked.connect(self.delete_selected)
        self._live = QLabel()  # ผลการลองท่า (ตอนไม่ได้สอน)

        # ---- ฟอร์มสอนท่าใหม่ ----
        self._name = QLineEdit()
        self._name.setPlaceholderText("เช่น rock (a-z, 0-9, _ หรือ -)")
        self._name.setMaxLength(24)
        self._action = QComboBox()
        self._action.setEditable(True)
        self._action.addItems(ACTION_PRESETS)
        self._start_button = QPushButton("เริ่มสอน")
        self._start_button.clicked.connect(self._start_clicked)
        form = QFormLayout()
        form.addRow("ชื่อท่า", self._name)
        form.addRow("ปุ่มลัดที่จะกด", self._action)

        # ---- ขั้นตอนสอน ----
        self._title = QLabel()
        self._title.setFont(title_font)
        self._instruction = QLabel()
        self._instruction.setWordWrap(True)
        self._preview = HandPreview(QColor(config.hud.palette.primary))
        self._status = QLabel()
        self._status.setWordWrap(True)
        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setTextVisible(False)
        self._hint = QLabel()
        self._hint.setWordWrap(True)
        self._hint.setStyleSheet(f"color: {_WARN};")
        line = self.fontMetrics().lineSpacing()
        self._instruction.setMinimumHeight(line * 2 + 4)
        self._status.setMinimumHeight(line * 2 + 4)
        self._hint.setMinimumHeight(line * 2 + 4)

        self._save_button = QPushButton("บันทึก")
        self._save_button.clicked.connect(self.save)
        self._cancel_button = QPushButton("ยกเลิกการสอน")
        self._cancel_button.clicked.connect(self.cancel_teaching)
        self._close_button = QPushButton("ปิด")
        self._close_button.clicked.connect(self.close)

        # ---- ท่าสำเร็จรูป (สอนล่วงหน้าด้วยมือสังเคราะห์ — ไม่ต้องสอนเอง) ----
        descriptions = self._pack_meta.get("descriptions", {})
        self._builtin = QComboBox()
        for pose in self._pack.poses:
            self._builtin.addItem(f"{pose.name} — {descriptions.get(pose.name, '')}", pose.name)
        self._builtin_action = QComboBox()
        self._builtin_action.setEditable(True)
        self._builtin_action.addItems(ACTION_PRESETS)
        self._builtin_button = QPushButton("เพิ่มท่านี้")
        self._builtin_button.clicked.connect(
            lambda: self.add_builtin(
                self._builtin.currentData(), self._builtin_action.currentText()
            )
        )

        list_row = QHBoxLayout()
        list_row.addWidget(self._live, 1)
        list_row.addWidget(self._delete_button)
        teach_row = QHBoxLayout()
        teach_row.addLayout(form, 1)
        teach_row.addWidget(self._start_button, 0, Qt.AlignmentFlag.AlignBottom)
        buttons = QHBoxLayout()
        buttons.addWidget(self._cancel_button)
        buttons.addStretch(1)
        buttons.addWidget(self._save_button)
        buttons.addWidget(self._close_button)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("ท่าที่สอนไว้ (ทำท่าให้กล้องเห็นเพื่อลองได้เลย — ตอนนี้มือไม่คุมเมาส์)"))
        layout.addWidget(self._list)
        layout.addLayout(list_row)
        layout.addSpacing(8)
        layout.addLayout(teach_row)
        if self._pack.poses:
            builtin_row = QHBoxLayout()
            builtin_row.addWidget(QLabel("ท่าสำเร็จรูป"))
            builtin_row.addWidget(self._builtin, 2)
            builtin_row.addWidget(self._builtin_action, 1)
            builtin_row.addWidget(self._builtin_button)
            layout.addLayout(builtin_row)
        layout.addSpacing(8)
        layout.addWidget(self._title)
        layout.addWidget(self._instruction)
        layout.addWidget(self._preview, 1)
        layout.addWidget(self._status)
        layout.addWidget(self._progress)
        layout.addWidget(self._hint)
        layout.addLayout(buttons)
        self._fill_list()
        self._refresh()

    # ---- public --------------------------------------------------------- #

    @property
    def library(self) -> PoseLibrary:
        return self._library

    @property
    def session(self) -> TeachSession | None:
        return self._session

    @property
    def changed(self) -> bool:
        return self._changed

    def start_teaching(self, name: str, action: str) -> bool:
        """เริ่มสอนท่า ``name`` ที่จะกด ``action`` คืน ``False`` (พร้อมแสดงเหตุผล) ถ้าชื่อ/ปุ่มลัดผิด"""
        try:
            self._session = TeachSession(self._config, self._library, name, action)
        except ValueError as exc:
            self._session = None
            self._hint.setText(str(exc))
            self._refresh()
            return False
        if self._library.get(self._session.name) is not None:
            log.info("สอนท่า %s ใหม่ (แทนท่าเดิม)", self._session.name)
        self._hint.setText("")
        self._refresh()
        return True

    def cancel_teaching(self) -> None:
        self._session = None
        self._hint.setText("")
        self._refresh()

    def save(self) -> bool:
        """บันทึกท่าที่สอนเสร็จแล้วลง poses.json"""
        session = self._session
        if session is None or session.result is None:
            return False
        if not self._write(session.result):
            return False
        log.info("บันทึกท่า %s → %s ลง %s", session.name, session.action, self._path)
        self._session = None
        self._hint.setText("")
        self._fill_list(select=session.name)
        self._refresh()
        return True

    @property
    def pack(self) -> PoseLibrary:
        return self._pack

    def add_builtin(self, name: str | None, action: str) -> bool:
        """เพิ่มท่าสำเร็จรูป ``name`` ผูกกับ ``action`` (ไม่ต้องสอน) — ติดยากเมื่อใช้จริง: สอนทับด้วยชื่อเดิม"""
        template = self._pack.get(name or "")
        if template is None or self._session is not None:
            return False
        try:
            parse_action(action)
            library = self._library.with_pose(replace(template, action=action.strip()))
        except ValueError as exc:
            self._hint.setText(str(exc))
            self._refresh()
            return False
        # มือปกติ: ของผู้ใช้เองสำคัญกว่า เติมของชุดสำเร็จรูปเฉพาะที่ยังว่าง
        own = library.negatives
        room = max(0, MAX_NEGATIVES - len(own))
        library = replace(library, negatives=(*self._pack.negatives[:room], *own))
        if not self._write(library):
            return False
        log.info("เพิ่มท่าสำเร็จรูป %s → %s", template.name, action)
        self._hint.setText(f"เพิ่มท่า {template.name} แล้ว — ลองทำท่าดู ถ้าติดยาก ให้สอนทับด้วยชื่อเดียวกัน")
        self._fill_list(select=template.name)
        self._refresh()
        return True

    def delete_selected(self) -> bool:
        item = self._list.currentItem()
        if item is None or self._session is not None:
            return False
        name = item.data(Qt.ItemDataRole.UserRole)
        if not self._write(self._library.without(name)):
            return False
        log.info("ลบท่า %s", name)
        self._fill_list()
        self._refresh()
        return True

    def show_frame(self, frame: HandFrame) -> None:
        """ป้อนเฟรมล่าสุดจากกล้อง"""
        hand = frame.hand(self._config.tracking.dominant_hand)
        self._preview.set_hand(hand, frame.width, frame.height)
        if self._session is not None:
            self._session.update(frame)
        else:
            self._show_live(hand, frame)
        self._refresh()

    def closeEvent(self, event) -> None:
        if not self._reported:
            self._reported = True
            self.finished.emit(self._changed)
        super().closeEvent(event)

    # ---- internal ------------------------------------------------------- #

    def _write(self, library: PoseLibrary) -> bool:
        try:
            save_poses(self._path, library)
        except OSError as exc:
            log.error("cannot write %s: %s", self._path, exc)
            self._hint.setText(f"บันทึก {self._path} ไม่ได้: {exc}")
            return False
        self._library = library
        self._classifier = PoseClassifier(library, self._config.poses.sensitivity)
        self._changed = True
        return True

    def _start_clicked(self) -> None:
        self.start_teaching(self._name.text(), self._action.currentText())

    def _fill_list(self, select: str | None = None) -> None:
        self._list.clear()
        for pose in self._library.poses:
            item = QListWidgetItem(f"{pose.name}  →  {pose.action}")
            item.setData(Qt.ItemDataRole.UserRole, pose.name)
            self._list.addItem(item)
            if pose.name == select:
                self._list.setCurrentItem(item)

    def _show_live(self, hand, frame: HandFrame) -> None:
        """ลองท่า: บอกว่ามือในกล้องตรงกับท่าไหน (ใช้เกณฑ์เดียวกับตอนใช้งานจริง)"""
        if not self._library.poses:
            self._live.setText("ยังไม่มีท่าที่สอนไว้")
            self._live.setStyleSheet("")
            return
        if hand is None:
            self._live.setText("ลองท่า: ไม่เห็นมือข้างถนัด")
            self._live.setStyleSheet("")
            return
        match = self._classifier.classify(hand, frame.width, frame.height)
        if match is None:
            self._live.setText("ลองท่า: —")
            self._live.setStyleSheet("")
        elif match.accepted:
            self._live.setText(
                f"ลองท่า: ✓ {match.name}  ({match.distance:.2f} / เกณฑ์ {match.threshold:.2f})"
            )
            self._live.setStyleSheet(f"color: {_OK};")
        else:
            reason = _REASONS.get(match.reason, match.reason)
            self._live.setText(
                f"ลองท่า: ✗ {reason} — ใกล้ {match.name} ที่สุด "
                f"({match.distance:.2f} / เกณฑ์ {match.threshold:.2f})"
            )
            self._live.setStyleSheet("")

    def _refresh(self) -> None:
        session = self._session
        teaching = session is not None
        self._name.setEnabled(not teaching)
        self._action.setEnabled(not teaching)
        self._start_button.setEnabled(not teaching)
        self._builtin_button.setEnabled(not teaching)
        self._list.setEnabled(not teaching)
        self._delete_button.setEnabled(not teaching and self._list.currentItem() is not None)
        self._cancel_button.setEnabled(teaching)
        self._save_button.setEnabled(teaching and session.done)
        if teaching:
            self._live.setText("(หยุดลองท่าระหว่างสอน)")
            self._live.setStyleSheet("")

        if session is None:
            self._title.setText("สอนท่าใหม่")
            self._instruction.setText(
                'ตั้งชื่อท่าและเลือกปุ่มลัด แล้วกด "เริ่มสอน" — ใช้เวลาราว 10 วินาที '
                "(ใช้ท่าเดิมได้ถ้าจะสอนใหม่หรือเปลี่ยนปุ่มลัด)"
            )
            self._status.setText("")
            self._progress.setValue(0)
            return

        self._title.setText(f"{session.title} — {session.name}")
        self._instruction.setText(session.instruction)
        self._progress.setValue(round(session.progress * 100))
        if session.done and session.summary is not None:
            summary = session.summary
            self._status.setText(
                f"จำได้ {summary.samples} ตัวอย่าง · เกณฑ์ {summary.threshold:.2f} · "
                f"ห่างจากมือปกติ ×{summary.neutral_margin:.1f}\n"
                f'กด "บันทึก" แล้วค้างท่านี้ {self._config.poses.hold_ms} ms เพื่อกด {session.action}'
            )
            self._hint.setText("")
            return
        self._hint.setText(session.error or "")
        if not session.hand_visible:
            self._status.setText("ยกมือข้างถนัดให้กล้องเห็น")
        elif session.sampling:
            self._status.setText("กำลังวัด… ค้างไว้")
        else:
            self._status.setText(f"เตรียมตัว… เริ่มวัดใน {session.countdown_ms / 1000:.1f} วินาที")
