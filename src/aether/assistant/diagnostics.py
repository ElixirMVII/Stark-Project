"""รวบรวมข้อมูลให้ Claude วิเคราะห์: ท้าย ``aether.log`` + สรุปสถิติ session/incident (แบบ tools/report.py)

ก่อนส่ง: แทน path โฟลเดอร์ผู้ใช้ด้วย ``~`` และชื่อผู้ใช้ด้วย ``<user>`` — ผู้ใช้เห็นข้อความทั้งหมดก่อนกดส่งเสมอ
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from ..incidents import build_report, list_incidents, read_incident
from ..logsetup import LOG_FILENAME

__all__ = ["collect", "load_sessions", "log_tail", "report_lines", "scrub"]

MAX_LINES = 160
MAX_CHARS = 16000


def scrub(text: str) -> str:
    home = str(Path.home())
    if home and home not in ("/", "\\"):
        text = text.replace(home, "~")
        text = text.replace(home.replace("\\", "/"), "~")
    user = os.environ.get("USERNAME") or os.environ.get("USER")
    if user and len(user) >= 3:
        text = text.replace(user, "<user>")
    return text


def log_tail(directory: Path, max_lines: int = MAX_LINES, max_chars: int = MAX_CHARS) -> str:
    path = directory / LOG_FILENAME
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return "(ยังไม่มี aether.log)"
    tail = "\n".join(lines[-max_lines:])
    return scrub(tail[-max_chars:])


def load_sessions(directory: Path, last: int = 20) -> list[dict]:
    paths = sorted(directory.glob("session-*.json"))[-last:] if directory.is_dir() else []
    sessions = []
    for path in paths:
        try:
            sessions.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            continue
    return sessions


def report_lines(log_directory: Path, incident_directory: Path) -> list[str]:
    sessions = load_sessions(log_directory)
    incidents = []
    for path in list_incidents(incident_directory) if incident_directory.is_dir() else []:
        try:
            incidents.append(read_incident(path))
        except (OSError, ValueError):
            continue
    if not sessions and not incidents:
        return []
    return [scrub(line) for line in build_report(sessions, incidents).lines()]


def collect(log_directory: Path, incident_directory: Path) -> tuple[str, list[str]]:
    return log_tail(log_directory), report_lines(log_directory, incident_directory)
