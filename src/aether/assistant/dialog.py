"""หน้าต่างผู้ช่วย Claude (tray → "ผู้ช่วย Claude…")

- บน: เปิด/ปิดฟีเจอร์ (เขียน ``[claude] enabled``) + API key (บันทึก/ลบ/ทดสอบ) + คำอธิบายว่าส่งอะไรบ้าง
- แท็บ Ask: ถามอะไรก็ได้ · Logs: ดูข้อมูลที่จะส่งก่อน แล้วให้ Claude วิเคราะห์ ·
  Tune: บอกปัญหาเป็นภาษาคน → Claude เสนอค่าที่ควรแก้ → ติ๊กเลือก → Apply (ตรวจทุกค่าก่อนเขียน แล้วโหลด config ใหม่)
  History: คำตอบย้อนหลังจากท่ามือ/Studio
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .. import ui_style
from ..config import ConfigError
from . import configedit, diagnostics, prompts, secret
from .schemas import ConfigProposal
from .tasks import ClaudeBridge
from .thought_view import ThoughtView

__all__ = ["AssistantWindow", "KeyStore"]

log = logging.getLogger(__name__)

PRIVACY = (
    "Claude ทำงานผ่านอินเทอร์เน็ต และส่งเฉพาะสิ่งที่คุณเลือกส่ง: ข้อความที่พิมพ์/copy ไว้, ข้อมูลธาตุ, "
    "ภาพที่วาดใน Studio, config.toml และ log/สถิติ (เห็นก่อนส่งในแท็บ Logs) — ไม่ส่งภาพจากกล้องเด็ดขาด"
)


class KeyStore:
    """ที่เก็บ API key (ค่าเริ่มต้น: :mod:`aether.assistant.secret`) — เทสส่ง path ชั่วคราวมาแทน"""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path

    def load(self) -> str | None:
        return secret.load_api_key(self.path)

    def save(self, key: str) -> None:
        secret.save_api_key(key, self.path)

    def clear(self) -> None:
        secret.clear_api_key(self.path)


def _answer_box() -> QPlainTextEdit:
    box = QPlainTextEdit()
    box.setReadOnly(True)
    box.setPlaceholderText("คำตอบของ Claude จะขึ้นที่นี่")
    return box


def _dim(text: str = "") -> QLabel:
    label = QLabel(text)
    label.setWordWrap(True)
    label.setStyleSheet(f"color: {ui_style.TEXT_DIM};")
    return label


class AssistantWindow(QWidget):
    def __init__(
        self,
        bridge: ClaudeBridge,
        *,
        config_path: Path,
        template: Path | None = None,
        log_directory: Path,
        incident_directory: Path,
        on_config_changed: Callable[[], Any] | None = None,
        history: Sequence[tuple[str, str]] = (),
        keys: KeyStore | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("AETHER — Claude")
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.resize(900, 720)
        self.bridge = bridge
        self._config_path = config_path
        self._template = template
        self._log_directory = log_directory
        self._incident_directory = incident_directory
        self._on_config_changed = on_config_changed
        self._keys = keys or KeyStore()
        self._proposal: ConfigProposal | None = None
        self._collected: tuple[str, list[str]] = ("", [])

        title = QLabel("Claude")
        title.setStyleSheet("font-size: 18px; font-weight: 600;")
        self.state_label = QLabel()
        header = QHBoxLayout()
        header.addWidget(title)
        header.addStretch(1)
        header.addWidget(self.state_label)

        # ---- เปิด/ปิด + key ----
        self.enabled_box = QCheckBox("ใช้ Claude (ต้องต่ออินเทอร์เน็ต)")
        self.enabled_box.setChecked(bridge.service.enabled)
        self.enabled_box.toggled.connect(self.set_enabled)
        self.key_edit = QLineEdit()
        self.key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.key_edit.setPlaceholderText("วาง API key (sk-ant-…) จาก platform.claude.com")
        self.key_save = QPushButton("บันทึก key")
        self.key_save.clicked.connect(self.save_key)
        self.key_clear = QPushButton("ลบ key")
        self.key_clear.clicked.connect(self.clear_key)
        self.key_test = QPushButton("ทดสอบ")
        self.key_test.clicked.connect(self.test_connection)
        self.key_label = _dim()
        key_row = QHBoxLayout()
        key_row.addWidget(self.key_edit, 1)
        key_row.addWidget(self.key_save)
        key_row.addWidget(self.key_clear)
        key_row.addWidget(self.key_test)

        # ---- แท็บ ----
        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_ask(), "Ask")
        self.tabs.addTab(self._build_logs(), "Logs")
        self.tabs.addTab(self._build_tune(), "Tune settings")
        self.tabs.addTab(self._build_history(history), "History")

        self.status = _dim()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(10)
        layout.addLayout(header)
        layout.addWidget(self.enabled_box)
        layout.addLayout(key_row)
        layout.addWidget(self.key_label)
        layout.addWidget(_dim(PRIVACY))
        layout.addWidget(self.tabs, 1)
        layout.addWidget(self.status)
        QShortcut(QKeySequence("Escape"), self, activated=self.close)
        self._refresh_state()

    # ---- แท็บ ------------------------------------------------------------ #

    def _build_ask(self) -> QWidget:
        page = QWidget()
        self.ask_edit = QPlainTextEdit()
        self.ask_edit.setPlaceholderText("ถามอะไรก็ได้ เช่น ท่ามือไหนเหมาะกับการดูวิดีโอ? (Ctrl+Enter = ส่ง)")
        self.ask_edit.setMaximumHeight(110)
        self.ask_button = QPushButton("ถาม Claude")
        self.ask_button.setDefault(True)
        self.ask_button.clicked.connect(self.ask)
        QShortcut(QKeySequence("Ctrl+Return"), self.ask_edit, activated=self.ask)
        self.ask_answer = _answer_box()
        self.ask_thoughts = ThoughtView()  # โครงข่ายความคิดที่งอกระหว่าง Claude คิด
        layout = QVBoxLayout(page)
        layout.addWidget(self.ask_edit)
        row = QHBoxLayout()
        row.addStretch(1)
        row.addWidget(self.ask_button)
        layout.addLayout(row)
        answers = QHBoxLayout()
        answers.addWidget(self.ask_answer, 2)
        answers.addWidget(self.ask_thoughts, 3)
        layout.addLayout(answers, 1)
        return page

    def _build_logs(self) -> QWidget:
        page = QWidget()
        self.logs_collect = QPushButton("รวบรวม log ล่าสุด")
        self.logs_collect.clicked.connect(self.collect_logs)
        self.logs_question = QLineEdit()
        self.logs_question.setPlaceholderText("(ไม่บังคับ) อาการที่เจอ เช่น คลิกไม่ค่อยติดตอนเย็น")
        self.logs_send = QPushButton("ให้ Claude วิเคราะห์")
        self.logs_send.clicked.connect(self.analyze_logs)
        self.logs_preview = QPlainTextEdit()
        self.logs_preview.setReadOnly(True)
        self.logs_preview.setPlaceholderText("กด 'รวบรวม log ล่าสุด' เพื่อดูสิ่งที่จะส่งก่อน")
        self.logs_answer = _answer_box()
        layout = QVBoxLayout(page)
        row = QHBoxLayout()
        row.addWidget(self.logs_collect)
        row.addWidget(self.logs_question, 1)
        row.addWidget(self.logs_send)
        layout.addLayout(row)
        layout.addWidget(_dim("สิ่งที่จะส่ง (path โฟลเดอร์ผู้ใช้ถูกแทนด้วย ~):"))
        layout.addWidget(self.logs_preview, 1)
        layout.addWidget(self.logs_answer, 1)
        return page

    def _build_tune(self) -> QWidget:
        page = QWidget()
        self.tune_edit = QLineEdit()
        self.tune_edit.setPlaceholderText(
            "บอกปัญหา/สิ่งที่อยากได้ เช่น เคอร์เซอร์สั่นเกินไป, อยากให้ ✊ ค้างนานขึ้นถึงจะทำงาน"
        )
        self.tune_edit.returnPressed.connect(self.suggest_config)
        self.tune_button = QPushButton("ขอคำแนะนำ")
        self.tune_button.clicked.connect(self.suggest_config)
        self.tune_summary = _dim()
        self.tune_list = QListWidget()
        self.tune_apply = QPushButton("Apply ที่เลือก + โหลด config ใหม่")
        self.tune_apply.setEnabled(False)
        self.tune_apply.clicked.connect(self.apply_selected)
        layout = QVBoxLayout(page)
        row = QHBoxLayout()
        row.addWidget(self.tune_edit, 1)
        row.addWidget(self.tune_button)
        layout.addLayout(row)
        layout.addWidget(self.tune_summary)
        layout.addWidget(self.tune_list, 1)
        bottom = QHBoxLayout()
        bottom.addWidget(_dim(f"เขียนลง {self._config_path.name} (ตรวจชนิด/ช่วงค่าก่อนเขียนทุกครั้ง)"), 1)
        bottom.addWidget(self.tune_apply)
        layout.addLayout(bottom)
        return page

    def _build_history(self, history: Sequence[tuple[str, str]]) -> QWidget:
        page = QWidget()
        self.history_list = QListWidget()
        self.history_view = _answer_box()
        self.history_list.currentRowChanged.connect(self._show_history)
        layout = QHBoxLayout(page)
        layout.addWidget(self.history_list, 2)
        layout.addWidget(self.history_view, 3)
        for question, answer in history:
            self.add_history(question, answer)
        return page

    # ---- ตั้งค่า --------------------------------------------------------- #

    def set_enabled(self, enabled: bool) -> None:
        try:
            configedit.set_claude_enabled(self._config_path, enabled, template=self._template)
        except (OSError, ConfigError) as exc:
            self._status(f"บันทึกไม่ได้: {exc}", error=True)
            self.enabled_box.blockSignals(True)
            self.enabled_box.setChecked(not enabled)
            self.enabled_box.blockSignals(False)
            return
        service = self.bridge.service
        service.update_config(dataclasses.replace(service.config, enabled=enabled))
        log.info("ฟีเจอร์ Claude: %s", "เปิด" if enabled else "ปิด")
        self._status("เปิด Claude แล้ว" if enabled else "ปิด Claude แล้ว — ไม่มีการต่ออินเทอร์เน็ตจากฟีเจอร์นี้")
        self._refresh_state()

    def save_key(self) -> None:
        key = self.key_edit.text().strip()
        if not key:
            self._status("วาง API key ก่อน", error=True)
            return
        try:
            self._keys.save(key)
        except (OSError, ValueError) as exc:
            self._status(f"บันทึก key ไม่ได้: {exc}", error=True)
            return
        self.key_edit.clear()
        self.bridge.service.reset_client()
        self._status("บันทึก key แล้ว (เข้ารหัสด้วยบัญชี Windows ของคุณ)")
        self._refresh_state()

    def clear_key(self) -> None:
        try:
            self._keys.clear()
        except OSError as exc:
            self._status(f"ลบ key ไม่ได้: {exc}", error=True)
            return
        self.bridge.service.reset_client()
        self._status("ลบ key แล้ว")
        self._refresh_state()

    def test_connection(self) -> None:
        self._status("กำลังทดสอบ…")
        self.bridge.stream(
            prompts.BASE_SYSTEM,
            "Connection test from the AETHER settings window. Reply with one short Thai sentence.",
            on_result=lambda text: self._status(f"ต่อได้แล้ว ✓  {text}"),
            on_error=lambda message: self._status(message, error=True),
            max_tokens=1024,
        )

    # ---- งาน ------------------------------------------------------------- #

    def ask(self) -> None:
        question = self.ask_edit.toPlainText().strip()
        if not question:
            return
        system, content = prompts.free_question(question)
        self._stream_into(self.ask_answer, system, content, question, thoughts=self.ask_thoughts)

    def collect_logs(self) -> None:
        tail, lines = diagnostics.collect(self._log_directory, self._incident_directory)
        report = "\n".join(lines) or "(ยังไม่มีสถิติ session — ใช้แอปสักพักแล้วปิด/เปิดใหม่)"
        self.logs_preview.setPlainText(f"[สรุปสถิติ]\n{report}\n\n[aether.log ท้ายไฟล์]\n{tail}")
        self._collected = (tail, lines)

    def analyze_logs(self) -> None:
        if not self.logs_preview.toPlainText():
            self.collect_logs()
        tail, lines = self._collected
        question = self.logs_question.text().strip() or None
        system, content = prompts.log_request(tail, lines, question)
        self._stream_into(self.logs_answer, system, content, question or "วิเคราะห์ log")

    def suggest_config(self) -> None:
        request = self.tune_edit.text().strip()
        if not request:
            self._status("บอกก่อนว่าอยากปรับอะไร", error=True)
            return
        text = configedit.read_config_text(self._config_path, self._template)
        notes = diagnostics.report_lines(self._log_directory, self._incident_directory)
        system, content = prompts.config_request(text, request, notes)
        self.tune_list.clear()
        self.tune_summary.setText("Claude กำลังคิด…")
        self.tune_apply.setEnabled(False)
        self.bridge.parse(
            system,
            content,
            ConfigProposal,
            on_result=self.show_proposal,
            on_error=self._tune_failed,
        )

    def show_proposal(self, proposal: ConfigProposal) -> None:
        """แสดงข้อเสนอ — ข้อที่ตรวจไม่ผ่านแสดงเหตุผลและติ๊กไม่ได้"""
        self._proposal = proposal
        self.tune_list.clear()
        self.tune_summary.setText(proposal.summary)
        text = configedit.read_config_text(self._config_path, self._template)
        current = configedit.editable_keys()
        valid = 0
        for change in proposal.changes:
            try:
                configedit.preview_changes(text, [(change.key, change.value)])
                error = None
            except ConfigError as exc:
                error = str(exc)
            old = current.get(change.key, "—")
            label = f"{change.key}:  {old}  →  {change.value}\n    {change.reason}"
            item = QListWidgetItem(label + (f"\n    ✕ ใช้ไม่ได้: {error}" if error else ""))
            item.setData(Qt.ItemDataRole.UserRole, (change.key, change.value))
            if error:
                item.setFlags(
                    item.flags() & ~Qt.ItemFlag.ItemIsUserCheckable & ~Qt.ItemFlag.ItemIsEnabled
                )
            else:
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(Qt.CheckState.Checked)
                valid += 1
            self.tune_list.addItem(item)
        self.tune_apply.setEnabled(valid > 0)
        if not proposal.changes:
            self._status("Claude ไม่แนะนำให้แก้อะไร")

    def selected_changes(self) -> list[tuple[str, str]]:
        changes = []
        for row in range(self.tune_list.count()):
            item = self.tune_list.item(row)
            if (
                item.flags() & Qt.ItemFlag.ItemIsEnabled
                and item.checkState() == Qt.CheckState.Checked
            ):
                changes.append(item.data(Qt.ItemDataRole.UserRole))
        return changes

    def apply_selected(self) -> bool:
        changes = self.selected_changes()
        if not changes:
            self._status("ยังไม่ได้เลือกข้อไหน", error=True)
            return False
        try:
            configedit.apply_changes(self._config_path, changes, template=self._template)
        except (OSError, ConfigError) as exc:
            self._status(f"ไม่ได้แก้อะไร: {exc}", error=True)
            return False
        keys = ", ".join(key for key, _ in changes)
        log.info("ผู้ช่วย Claude แก้ config: %s", keys)
        self._status(f"แก้แล้ว: {keys} — โหลด config ใหม่")
        self.tune_apply.setEnabled(False)
        if self._on_config_changed is not None:
            self._on_config_changed()
        return True

    def add_history(self, question: str, answer: str) -> None:
        first = question.strip().splitlines()[0] if question.strip() else "(ไม่มีข้อความ)"
        item = QListWidgetItem(first[:80])
        item.setData(Qt.ItemDataRole.UserRole, (question, answer))
        self.history_list.insertItem(0, item)

    # ---- internal ------------------------------------------------------- #

    def _show_history(self, row: int) -> None:
        item = self.history_list.item(row)
        if item is None:
            return
        question, answer = item.data(Qt.ItemDataRole.UserRole)
        self.history_view.setPlainText(f"{question}\n\n— Claude —\n{answer}")

    def _stream_into(
        self,
        box: QPlainTextEdit,
        system: str,
        content: Any,
        question: str,
        *,
        thoughts: ThoughtView | None = None,
    ) -> None:
        box.clear()
        self._status("Claude กำลังตอบ…")
        if thoughts is not None:
            thoughts.begin(question)

        def piece(text: str) -> None:
            cursor = box.textCursor()
            cursor.movePosition(cursor.MoveOperation.End)
            cursor.insertText(text)

        def done(answer: str) -> None:
            box.setPlainText(answer)
            self._status("")
            self.add_history(question, answer)
            if thoughts is not None:
                thoughts.finish(answer)

        self.bridge.stream(
            system,
            content,
            on_text=piece,
            on_thinking=thoughts.add_thinking if thoughts is not None else None,
            on_result=done,
            on_error=lambda message: self._status(message, error=True),
        )

    def _tune_failed(self, message: str) -> None:
        self.tune_summary.setText("")
        self._status(message, error=True)

    def _status(self, text: str, *, error: bool = False) -> None:
        self.status.setText(text)
        self.status.setStyleSheet(f"color: {ui_style.DANGER if error else ui_style.TEXT_DIM};")

    def _refresh_state(self) -> None:
        service = self.bridge.service
        stored = self._keys.load()
        if stored:
            self.key_label.setText(f"key ที่บันทึกไว้: {secret.mask(stored)}")
        elif service.has_key():
            self.key_label.setText("ใช้ key จากตัวแปร ANTHROPIC_API_KEY")
        else:
            self.key_label.setText("ยังไม่มี API key")
        reason = service.ready_reason()
        if reason is None:
            self.state_label.setText(
                f"พร้อม · {service.config.model} · effort {service.config.effort}"
            )
            self.state_label.setStyleSheet(f"color: {ui_style.OK};")
        else:
            self.state_label.setText("ปิดอยู่" if not service.enabled else "ยังไม่มี key")
            self.state_label.setStyleSheet(f"color: {ui_style.TEXT_DIM};")
        for widget in (self.ask_button, self.logs_send, self.tune_button, self.key_test):
            widget.setEnabled(reason is None)
