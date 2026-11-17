"""จดปัญหาและสถิติการใช้งาน (spec v1.1 M5) — pure logic + ไฟล์ ไม่แตะ Qt

- :class:`FlightRecorder` เก็บเฟรม + state + event ย้อนหลัง ``INCIDENT_SECONDS`` วินาทีไว้ในหน่วยความจำ
  กด ``Ctrl+Alt+M`` แล้วแอปเขียนลง ``incidents/YYYYMMDD-HHMMSS.jsonl`` (:func:`write_incident`)
- ไฟล์ incident เป็นรูปแบบเดียวกับไฟล์อัดของ ``tools/record_landmarks.py`` (header + เฟรมละบรรทัด)
  เพิ่ม ``state``/``events`` ในแต่ละเฟรม และ config/จอ/ท่าที่สอนเองใน header เพื่อให้ replay ได้ผลเดิม
  — **มีแค่ตัวเลข landmark ไม่มีภาพ**
- ใส่ป้ายทีหลังด้วย :func:`set_label` (``tools/label_incidents.py``)
- :class:`SessionStats` สรุปต่อ session ลง ``logs/session-*.json`` และ :func:`build_report`
  รวมหลาย session + incident เป็นคำแนะนำว่าควรปรับค่าไหน (``tools/report.py``)
"""

from __future__ import annotations

import dataclasses
import datetime
import json
import os
import sys
import threading
import time
from collections import Counter, deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import Config, app_root, parse_config
from .gestures import EventKind, GestureEvent, State
from .mapping import ScreenGeometry
from .poses import PoseLibrary
from .recording import FORMAT_VERSION, compact_frame
from .tracker import HandFrame

__all__ = [
    "INCIDENT_SECONDS",
    "LABELS",
    "FlightRecorder",
    "Incident",
    "RecordedFrame",
    "SessionStats",
    "build_report",
    "event_from_dict",
    "event_to_dict",
    "incident_dir",
    "incident_name",
    "list_incidents",
    "percentile_from_histogram",
    "read_incident",
    "replay_incident",
    "set_label",
    "write_incident",
]

INCIDENT_SECONDS = 5.0
LABELS = ("false_trigger", "missed", "jitter", "wrong_click", "other")
LATENCY_BUCKETS = 500  # histogram ละ 1 ms (เกินนี้นับรวมช่องสุดท้าย)
STATE_GAP_CAP_S = 2.0  # ช่วงห่างระหว่างเฟรมที่นานกว่านี้ (กล้องค้าง) นับเวลาให้ state แค่เท่านี้


# --------------------------------------------------------------------------- #
# Events ↔ dict
# --------------------------------------------------------------------------- #


def event_to_dict(event: GestureEvent) -> dict[str, Any]:
    """เฉพาะ field ที่ไม่ใช่ค่าเริ่มต้น (ไฟล์เล็ก อ่านง่าย)"""
    data: dict[str, Any] = {"kind": event.kind.value}
    defaults = GestureEvent(event.kind)
    for f in dataclasses.fields(GestureEvent):
        if f.name != "kind" and getattr(event, f.name) != getattr(defaults, f.name):
            data[f.name] = getattr(event, f.name)
    return data


def event_from_dict(data: dict[str, Any]) -> GestureEvent:
    values = {k: v for k, v in data.items() if k != "kind"}
    return GestureEvent(EventKind(data["kind"]), **values)


# --------------------------------------------------------------------------- #
# Flight recorder
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class RecordedFrame:
    hands: HandFrame
    state: State
    events: tuple[GestureEvent, ...] = ()


class FlightRecorder:
    """ring buffer ของเฟรมย้อนหลัง ``seconds`` วินาที (thread-safe: worker เขียน, hotkey อ่าน)"""

    def __init__(self, seconds: float = INCIDENT_SECONDS) -> None:
        self._window_ms = int(seconds * 1000)
        self._frames: deque[RecordedFrame] = deque()
        self._lock = threading.Lock()

    def record(self, hands: HandFrame, state: State, events: Sequence[GestureEvent]) -> None:
        entry = RecordedFrame(hands, state, tuple(events))
        with self._lock:
            self._frames.append(entry)
            limit = hands.timestamp_ms - self._window_ms
            while self._frames and self._frames[0].hands.timestamp_ms < limit:
                self._frames.popleft()

    def snapshot(self) -> list[RecordedFrame]:
        """สำเนาของเฟรมที่เก็บไว้ (เร็ว — ใช้เวลาใน lock แค่คัดลอก list)"""
        with self._lock:
            return list(self._frames)

    def __len__(self) -> int:
        with self._lock:
            return len(self._frames)


# --------------------------------------------------------------------------- #
# Incident files
# --------------------------------------------------------------------------- #


def incident_dir() -> Path:
    """``incidents/`` ข้างตัวแอป ถ้าเขียนไม่ได้ใช้โฟลเดอร์ของผู้ใช้ (เหมือน log)"""
    primary = app_root() / "incidents"
    try:
        primary.mkdir(parents=True, exist_ok=True)
        if os.access(primary, os.W_OK):
            return primary
    except OSError:
        pass
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home()) / "AETHER"
    else:
        base = Path(os.environ.get("XDG_STATE_HOME") or Path.home() / ".local" / "state") / "aether"
    fallback = base / "incidents"
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback


def incident_name(when: datetime.datetime) -> str:
    return when.strftime("%Y%m%d-%H%M%S") + ".jsonl"


def write_incident(
    directory: Path,
    frames: Sequence[RecordedFrame],
    *,
    config: Config,
    screen: ScreenGeometry | Any,
    double_click_ms: int,
    poses: PoseLibrary | None = None,
    source: str = "camera",
    when: datetime.datetime | None = None,
) -> Path:
    """เขียนไฟล์ incident (ไม่ทับไฟล์เดิม — ชื่อซ้ำในวินาทีเดียวกันจะเติม -2, -3, …)"""
    when = when or datetime.datetime.now().astimezone()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / incident_name(when)
    counter = 2
    while path.exists():
        path = directory / incident_name(when).replace(".jsonl", f"-{counter}.jsonl")
        counter += 1
    origin = frames[0].hands.timestamp_ms if frames else 0
    header: dict[str, Any] = {
        "aether_landmarks": FORMAT_VERSION,
        "incident": 1,
        "created": when.isoformat(timespec="seconds"),
        "label": None,
        "note": "",
        "source": source,
        "seconds": round((frames[-1].hands.timestamp_ms - origin) / 1000.0, 3) if frames else 0,
        "start_armed": bool(frames) and frames[0].state is not State.DISARMED,
        "dominant_hand": config.tracking.dominant_hand,
        "double_click_ms": double_click_ms,
        "screen": [screen.left, screen.top, screen.width, screen.height],
        "config": _jsonable(dataclasses.asdict(config)),
    }
    if poses is not None and poses.poses:
        header["poses"] = poses.to_dict()
    lines = [json.dumps(header, ensure_ascii=False)]
    for entry in frames:
        data = compact_frame(entry.hands, origin)
        data["state"] = entry.state.value
        if entry.events:
            data["events"] = [event_to_dict(e) for e in entry.events]
        lines.append(json.dumps(data, ensure_ascii=False, separators=(",", ":")))
    temporary = path.with_suffix(".tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(path)
    return path


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


@dataclass(frozen=True, slots=True)
class Incident:
    path: Path
    header: dict[str, Any]
    frames: tuple[HandFrame, ...] = ()
    states: tuple[str, ...] = ()
    events: tuple[tuple[GestureEvent, ...], ...] = ()

    @property
    def label(self) -> str | None:
        return self.header.get("label")

    def config(self) -> Config:
        """config ตอนเกิดเหตุ (ไฟล์อัดทั่วไปที่ไม่มี config ใช้ค่าเริ่มต้น + มือข้างถนัดใน header)"""
        if "config" in self.header:
            return parse_config(self.header["config"])
        return parse_config(
            {"tracking": {"dominant_hand": self.header.get("dominant_hand", "right")}}
        )

    def screen(self) -> ScreenGeometry:
        left, top, width, height = self.header.get("screen", [0, 0, 1920, 1080])
        return ScreenGeometry(int(left), int(top), int(width), int(height))

    def poses(self) -> PoseLibrary | None:
        data = self.header.get("poses")
        return PoseLibrary.from_dict(data) if data else None

    def recorded_events(self) -> list[GestureEvent]:
        return [e for batch in self.events for e in batch]


def read_incident(path: Path) -> Incident:
    """อ่านไฟล์ incident หรือไฟล์อัดทั่วไป (raise ``ValueError`` ถ้าอ่านไม่ได้)"""
    header: dict[str, Any] = {}
    frames: list[HandFrame] = []
    states: list[str] = []
    events: list[tuple[GestureEvent, ...]] = []
    with Path(path).open(encoding="utf-8") as fh:
        for number, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                if "aether_landmarks" in data:
                    header = data
                    continue
                frames.append(HandFrame.from_dict(data))
                states.append(str(data.get("state", "")))
                events.append(tuple(event_from_dict(e) for e in data.get("events", ())))
            except (ValueError, KeyError, TypeError) as exc:
                raise ValueError(f"{path}: บรรทัด {number} อ่านไม่ได้ ({exc})") from exc
    return Incident(Path(path), header, tuple(frames), tuple(states), tuple(events))


def replay_incident(
    incident: Incident, *, armed: bool | None = None
) -> list[tuple[int, State, list]]:
    """เล่น incident ผ่าน gesture engine ด้วย config/จอ/ท่าที่สอนเองตอนเกิดเหตุ (ไม่แตะ OS)

    คืน timeline ``[(เวลา ms นับจากเฟรมแรก, state, event ของเฟรมนั้น), ...]`` — เล่นกี่รอบก็ได้ผลเดิม
    (engine เริ่มจากศูนย์ที่เฟรมแรกของไฟล์ จึงอาจต่างจากตอนเกิดเหตุจริงถ้าตอนนั้นกำลังทำท่าค้างอยู่)
    """
    from .gestures import GestureEngine
    from .poses import PoseClassifier

    config = incident.config()
    library = incident.poses()
    engine = GestureEngine(
        config,
        incident.screen(),
        double_click_ms=int(incident.header.get("double_click_ms", 500)),
        poses=PoseClassifier(library, config.poses.sensitivity) if library else None,
    )
    start_armed = incident.header.get("start_armed", True) if armed is None else armed
    if start_armed:
        engine.set_armed(True)
    origin = incident.frames[0].timestamp_ms if incident.frames else 0
    timeline: list[tuple[int, State, list]] = []
    for frame in incident.frames:
        state, events = engine.update(frame)
        timeline.append((frame.timestamp_ms - origin, state, events))
    return timeline


def set_label(path: Path, label: str, note: str | None = None) -> None:
    """ใส่ป้ายให้ incident (เขียนทับเฉพาะบรรทัด header)"""
    if label not in LABELS:
        raise ValueError(f"ป้ายต้องเป็นหนึ่งใน {', '.join(LABELS)} (ได้ {label!r})")
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError(f"{path}: ไฟล์ว่าง")
    header = json.loads(lines[0])
    if "aether_landmarks" not in header:
        raise ValueError(f"{path}: ไม่มี header")
    header["label"] = label
    if note is not None:
        header["note"] = note
    lines[0] = json.dumps(header, ensure_ascii=False)
    temporary = Path(path).with_suffix(".tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(path)


def list_incidents(directory: Path) -> list[Path]:
    """เรียงตามเวลา (ชื่อไฟล์ — ``…-2`` ที่จดในวินาทีเดียวกันตามหลังไฟล์แรก)"""
    return sorted(directory.glob("*.jsonl"), key=lambda p: p.stem) if directory.is_dir() else []


# --------------------------------------------------------------------------- #
# Session statistics
# --------------------------------------------------------------------------- #


def percentile_from_histogram(histogram: dict[int, int], q: float) -> float | None:
    """เปอร์เซ็นไทล์ ``q`` (0–100) จาก histogram ``{ms: จำนวน}`` (ความละเอียด 1 ms)"""
    total = sum(histogram.values())
    if total == 0:
        return None
    rank = q / 100.0 * total
    seen = 0
    for ms in sorted(histogram):
        seen += histogram[ms]
        if seen >= rank:
            return float(ms)
    return float(max(histogram))


class SessionStats:
    """สถิติของหนึ่ง session — :meth:`observe` จาก worker thread ทุกเฟรม (thread-safe)"""

    def __init__(self, *, source: str = "camera", dry_run: bool = False) -> None:
        self._lock = threading.Lock()
        self._started = datetime.datetime.now().astimezone()
        self._started_perf = time.perf_counter()
        self._source = source
        self._dry_run = dry_run
        self._frames = 0
        self._first: float | None = None
        self._last: float | None = None
        self._last_state: State | None = None
        self._state_seconds: Counter[str] = Counter()
        self._events: Counter[str] = Counter()
        self._actions: Counter[str] = Counter()
        self._latency: Counter[int] = Counter()
        self._latency_max = 0.0
        self.incidents = 0

    def observe(
        self, captured_at: float, latency_ms: float, state: State, events: Iterable[GestureEvent]
    ) -> None:
        with self._lock:
            self._frames += 1
            if self._first is None:
                self._first = captured_at
            if self._last is not None and self._last_state is not None:
                gap = min(max(captured_at - self._last, 0.0), STATE_GAP_CAP_S)
                self._state_seconds[self._last_state.value] += gap
            self._last, self._last_state = captured_at, state
            self._latency[min(int(latency_ms), LATENCY_BUCKETS)] += 1
            self._latency_max = max(self._latency_max, latency_ms)
            for event in events:
                self._events[event.kind.value] += 1
                if event.kind is EventKind.ACTION:
                    self._actions[event.name] += 1

    def note_incident(self) -> None:
        with self._lock:
            self.incidents += 1

    def summary(self, *, hud_paints: int | None = None) -> dict[str, Any]:
        with self._lock:
            ended = datetime.datetime.now().astimezone()
            duration = time.perf_counter() - self._started_perf
            span = (self._last - self._first) if self._first is not None and self._last else 0.0
            histogram = dict(self._latency)
            return {
                "aether_session": 1,
                "started": self._started.isoformat(timespec="seconds"),
                "ended": ended.isoformat(timespec="seconds"),
                "duration_s": round(duration, 1),
                "source": self._source,
                "dry_run": self._dry_run,
                "frames": self._frames,
                "camera_fps": round((self._frames - 1) / span, 1) if span > 0 else 0.0,
                "hud_fps": round(hud_paints / duration, 1) if hud_paints and duration > 0 else None,
                "latency_ms": {
                    "p50": percentile_from_histogram(histogram, 50),
                    "p95": percentile_from_histogram(histogram, 95),
                    "max": round(self._latency_max, 1),
                    "histogram": [[ms, histogram[ms]] for ms in sorted(histogram)],
                },
                "events": dict(sorted(self._events.items())),
                "actions": dict(sorted(self._actions.items())),
                "state_seconds": {k: round(v, 1) for k, v in sorted(self._state_seconds.items())},
                "incidents": self.incidents,
            }

    def write(self, directory: Path, *, hud_paints: int | None = None) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        name = "session-" + self._started.strftime("%Y%m%d-%H%M%S")
        path = directory / f"{name}.json"
        counter = 2
        while path.exists():
            path = directory / f"{name}-{counter}.json"
            counter += 1
        path.write_text(
            json.dumps(self.summary(hud_paints=hud_paints), ensure_ascii=False, indent=1) + "\n",
            encoding="utf-8",
        )
        return path


# --------------------------------------------------------------------------- #
# Report (หลาย session + incident)
# --------------------------------------------------------------------------- #

# (ป้าย, ท่าที่เกี่ยวข้อง) → คำแนะนำ — ท่าที่เกี่ยวข้องดูจาก event สุดท้ายใน incident
_ADVICE: dict[tuple[str, str], str] = {
    ("false_trigger", "swipe"): "ปัดติดเอง → เพิ่ม gestures.swipe_dist_x / swipe_dist_y",
    ("false_trigger", "fist_hold"): "✊ ติดเอง → เพิ่ม gestures.fist_hold_ms",
    ("false_trigger", "launcher_open"): "เมนู 👍 เปิดเอง → เพิ่ม launcher.hold_ms หรือปิด launcher",
    ("false_trigger", "pose"): "ท่าที่สอนเองติดเอง → เพิ่ม poses.hold_ms หรือลด poses.sensitivity",
    ("false_trigger", "click"): "คลิกเอง → ลด pinch.on (หรือ calibrate ใหม่)",
    ("false_trigger", "drag"): "ลากเอง → เพิ่ม pinch.drag_hold_ms",
    ("false_trigger", "scroll"): "สกรอลเอง → เพิ่ม scroll.deadzone หรือ tracking.min_gesture_score",
    ("false_trigger", "armed"): "🤟 สลับ ARM เอง → เพิ่ม gestures.arm_hold_ms",
    ("false_trigger", "window"): "หยิบหน้าต่างเอง → ปิด gestures.window_grab ถ้าไม่ได้ใช้",
    ("missed", "click"): "คลิกไม่ติด → เพิ่ม pinch.on / pinch.off (หรือ calibrate ใหม่)",
    ("missed", "swipe"): "ปัดไม่ติด → ลด gestures.swipe_dist_* หรือเพิ่ม swipe_window_ms",
    ("missed", "none"): "ท่าไม่ติดเลย → ดูชื่อท่าในหน้าต่าง debug, เพิ่มแสง, ลด tracking.min_gesture_score",
    ("jitter", "any"): "เคอร์เซอร์สั่น → ลด pointer.one_euro.min_cutoff",
    ("wrong_click", "any"): "คลิกผิดที่ → ลด pointer.pinch_lock_gain (เคอร์เซอร์นิ่งขึ้นตอนจีบ)",
}


def _gesture_of(incident: Incident) -> str:
    """ท่าที่น่าจะเกี่ยวกับเหตุ: event สำคัญตัวสุดท้ายในไฟล์ ("none" = ไม่มี event)"""
    for event in reversed(incident.recorded_events()):
        kind = event.kind
        if kind is EventKind.ACTION:
            name = event.name
            if name.startswith("swipe_"):
                return "swipe"
            if name.startswith("pose:"):
                return "pose"
            if name.startswith("launch_"):
                return "launcher_open"
            return name
        if kind is EventKind.CLICK:
            return "click"
        if kind in (EventKind.BUTTON_DOWN, EventKind.BUTTON_UP):
            return "drag"
        if kind is EventKind.SCROLL:
            return "scroll"
        if kind in (EventKind.ARMED, EventKind.DISARMED):
            return "armed"
        if kind in (EventKind.WINDOW_GRAB, EventKind.WINDOW_RELEASE):
            return "window"
    return "none"


@dataclass
class Report:
    sessions: int = 0
    duration_s: float = 0.0
    frames: int = 0
    latency_p50: float | None = None
    latency_p95: float | None = None
    camera_fps: float | None = None
    hud_fps: float | None = None
    events: Counter[str] = field(default_factory=Counter)
    actions: Counter[str] = field(default_factory=Counter)
    incidents: int = 0
    unlabeled: int = 0
    problems: list[tuple[str, str, int]] = field(default_factory=list)  # (ป้าย, ท่า, จำนวน)
    advice: list[str] = field(default_factory=list)

    def lines(self) -> list[str]:
        hours = self.duration_s / 3600.0
        out = [
            f"session {self.sessions} ครั้ง · รวม {self.duration_s / 60:.1f} นาที · {self.frames} เฟรม",
        ]
        if self.latency_p95 is not None:
            out.append(
                f"latency เฟรม→input: p50 {self.latency_p50:.0f} ms · p95 {self.latency_p95:.0f} ms"
                + ("  ⚠ เกินเป้า 60 ms" if self.latency_p95 > 60 else "")
            )
        if self.camera_fps is not None:
            hud = f" · HUD {self.hud_fps:.0f} fps" if self.hud_fps else ""
            warn = "  ⚠ ต่ำกว่า 25" if self.camera_fps < 25 else ""
            out.append(f"FPS กล้องเฉลี่ย {self.camera_fps:.1f}{warn}{hud}")
        if self.actions:
            per_hour = ", ".join(
                f"{name} {count}" + (f" ({count / hours:.0f}/ชม.)" if hours > 0.05 else "")
                for name, count in self.actions.most_common(8)
            )
            out.append(f"ท่าที่ใช้: {per_hour}")
        out.append(
            f"incident {self.incidents} ไฟล์"
            + (
                f" (ยังไม่ใส่ป้าย {self.unlabeled} — รัน tools/label_incidents.py)"
                if self.unlabeled
                else ""
            )
        )
        if self.problems:
            out.append("ปัญหาที่เจอบ่อย:")
            out.extend(
                f"  {i}. {label} · {gesture} — {count} ครั้ง"
                for i, (label, gesture, count) in enumerate(self.problems, start=1)
            )
        if self.advice:
            out.append("ควรปรับ:")
            out.extend(f"  - {line}" for line in self.advice)
        return out


def build_report(sessions: Sequence[dict[str, Any]], incidents: Sequence[Incident]) -> Report:
    report = Report(sessions=len(sessions))
    histogram: Counter[int] = Counter()
    fps_weighted = 0.0
    hud_weighted = 0.0
    hud_time = 0.0
    for session in sessions:
        duration = float(session.get("duration_s", 0.0))
        report.duration_s += duration
        report.frames += int(session.get("frames", 0))
        for ms, count in session.get("latency_ms", {}).get("histogram", []):
            histogram[int(ms)] += int(count)
        fps_weighted += float(session.get("camera_fps", 0.0)) * duration
        if session.get("hud_fps"):
            hud_weighted += float(session["hud_fps"]) * duration
            hud_time += duration
        report.events.update(session.get("events", {}))
        report.actions.update(session.get("actions", {}))
    if histogram:
        report.latency_p50 = percentile_from_histogram(dict(histogram), 50)
        report.latency_p95 = percentile_from_histogram(dict(histogram), 95)
    if report.duration_s > 0:
        report.camera_fps = fps_weighted / report.duration_s
    if hud_time > 0:
        report.hud_fps = hud_weighted / hud_time

    report.incidents = len(incidents)
    problems: Counter[tuple[str, str]] = Counter()
    for incident in incidents:
        label = incident.label
        if not label:
            report.unlabeled += 1
            continue
        problems[(label, _gesture_of(incident))] += 1
    report.problems = [(label, gesture, n) for (label, gesture), n in problems.most_common()]
    for (label, gesture), _n in problems.most_common():
        advice = _ADVICE.get((label, gesture)) or _ADVICE.get((label, "any"))
        if advice and advice not in report.advice:
            report.advice.append(advice)
    if report.latency_p95 is not None and report.latency_p95 > 60:
        report.advice.append("latency สูง → ปิดโปรแกรมที่กิน CPU, ใช้กล้อง 640×480, ปิด hud.show_radar")
    if report.camera_fps is not None and 0 < report.camera_fps < 25:
        report.advice.append("FPS กล้องต่ำ → เพิ่มแสง (กล้องลด FPS เองตอนมืด), ตรวจ camera.width/height")
    return report
