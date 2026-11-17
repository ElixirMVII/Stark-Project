"""M5: tools/label_incidents.py, tools/report.py, tools/replay.py และการจด incident จากแอปจริง"""

from __future__ import annotations

import json

from conftest import load_tool
from handsim import OPEN, PINCHED, make_hand
from test_incidents import WHEN, recorded_click

from aether.config import Config, parse_config
from aether.incidents import SessionStats, read_incident, write_incident
from aether.mapping import ScreenGeometry

label_tool = load_tool("label_incidents")
report_tool = load_tool("report")
replay_tool = load_tool("replay")

FHD = ScreenGeometry(0, 0, 1920, 1080)


def incident(tmp_path, name: str = "a"):
    directory = tmp_path / name
    return write_incident(
        directory, recorded_click(), config=Config(), screen=FHD, double_click_ms=500, when=WHEN
    )


# ---- label_incidents ---------------------------------------------------- #


def test_label_tool_set_and_list(tmp_path, capsys):
    path = incident(tmp_path)
    assert label_tool.main(["--set", str(path), "missed", "--note", "จีบแล้วไม่ติด"]) == 0
    assert read_incident(path).label == "missed"
    assert label_tool.main(["--dir", str(path.parent), "--list"]) == 0
    assert "[missed]" in capsys.readouterr().out
    assert label_tool.main(["--set", str(path), "nonsense"]) == 1


def test_label_tool_interactive(tmp_path, capsys):
    first = incident(tmp_path)
    second = write_incident(
        first.parent, recorded_click(), config=Config(), screen=FHD, double_click_ms=500, when=WHEN
    )
    answers = iter(["1", "ยกมือแล้วคลิกเอง", ""])  # ไฟล์แรก: false_trigger + หมายเหตุ, ไฟล์สอง: ข้าม
    assert label_tool.main(["--dir", str(first.parent)], ask=lambda prompt: next(answers)) == 0
    assert read_incident(first).label == "false_trigger"
    assert read_incident(first).header["note"] == "ยกมือแล้วคลิกเอง"
    assert read_incident(second).label is None
    assert "ใส่ป้ายแล้ว 1 ไฟล์" in capsys.readouterr().out


def test_label_tool_with_no_incidents(tmp_path, capsys):
    assert label_tool.main(["--dir", str(tmp_path / "none")]) == 0
    assert "Ctrl+Alt+M" in capsys.readouterr().out


# ---- report ------------------------------------------------------------- #


def test_report_tool(tmp_path, capsys):
    logs = tmp_path / "logs"
    stats = SessionStats()
    t = 0.0
    for _ in range(60):
        stats.observe(t, 25.0, recorded_click()[0].state, [])
        t += 1 / 30
    stats.write(logs)
    path = incident(tmp_path, "inc")
    label_tool.main(["--set", str(path), "jitter"])
    assert report_tool.main(["--logs", str(logs), "--incidents", str(path.parent)]) == 0
    out = capsys.readouterr().out
    assert "session 1 ครั้ง" in out and "p95 25 ms" in out
    assert "jitter" in out and "min_cutoff" in out


def test_report_tool_without_data(tmp_path, capsys):
    assert (
        report_tool.main(["--logs", str(tmp_path / "x"), "--incidents", str(tmp_path / "y")]) == 0
    )
    assert "ยังไม่มีข้อมูล" in capsys.readouterr().out


# ---- replay ------------------------------------------------------------- #


def test_replay_tool_prints_a_deterministic_timeline(tmp_path, capsys):
    path = incident(tmp_path)
    assert replay_tool.main([str(path)]) == 0
    first = capsys.readouterr().out
    assert replay_tool.main([str(path)]) == 0
    assert capsys.readouterr().out == first
    assert "CLICK left" in first and "สรุป: left_clicks=1" in first
    assert "move (" not in first
    replay_tool.main([str(path), "--moves"])
    assert "move (" in capsys.readouterr().out


def test_replay_tool_promotes_to_a_fixture(tmp_path, capsys):
    path = incident(tmp_path)
    fixtures = tmp_path / "fixtures"
    args = [str(path), "--promote", "--expect", "left_clicks=1", "--fixtures", str(fixtures)]
    assert replay_tool.main([*args, "--name", "Click Once"]) == 0
    target = fixtures / "click_once.jsonl"
    header = json.loads(target.read_text(encoding="utf-8").splitlines()[0])
    assert header["expect"] == {"left_clicks": 1} and header["incident"] == 1
    assert "ย้ายเป็น fixture แล้ว" in capsys.readouterr().out
    # ชื่อซ้ำ → ไม่ทับ
    assert replay_tool.main([*args, "--name", "Click Once"]) == 1


def test_replay_tool_check_refuses_when_not_yet_fixed(tmp_path, capsys):
    path = incident(tmp_path)
    fixtures = tmp_path / "fixtures"
    code = replay_tool.main(
        [
            str(path),
            "--promote",
            "--check",
            "--expect",
            "left_clicks=0",
            "--fixtures",
            str(fixtures),
        ]
    )
    assert code == 2 and not fixtures.exists()
    assert "ยังไม่ตรง" in capsys.readouterr().out


def test_replay_tool_promote_needs_expect(tmp_path):
    import pytest

    with pytest.raises(SystemExit):
        replay_tool.main([str(incident(tmp_path)), "--promote"])


# ---- จาก app จริง: Ctrl+Alt+M → ไฟล์ incident, ปิดแอป → session stats -------- #


def test_app_marks_incident_and_writes_session_stats(qapp, tmp_path):
    from test_actions import RecordingBackend
    from test_app import BlankCamera, ScriptedTracker
    from test_hotkeys import FakeApi
    from test_m4 import wait

    from aether.app import AetherApp, Options
    from aether.hotkeys import HotkeyManager

    config = parse_config({"hud": {"sounds": False}})
    tracker = ScriptedTracker()
    app = AetherApp(
        config,
        Options(no_hud=True),
        tracker,
        source=BlankCamera(),
        backend=RecordingBackend(),
        screen=FHD,
        hotkeys=HotkeyManager({config.hotkeys.toggle_arm: lambda: None}, api=FakeApi()),
        incident_directory=tmp_path / "incidents",
        session_log=tmp_path / "logs",
    )
    app.quit_requested.disconnect()
    saved: list = []
    app.incident_saved.connect(saved.append)
    app.start(boot=False)
    try:
        app.toggle_armed()
        tracker.hand = make_hand(pinch=OPEN)
        assert wait(qapp, lambda: len(app.recorder) > 20)
        tracker.hand = make_hand(pinch=PINCHED)
        assert wait(qapp, lambda: app.pipeline.snapshot.hold_progress > 0.3)  # จีบนานพอ (> 40 ms)
        tracker.hand = make_hand(pinch=OPEN)
        assert wait(qapp, lambda: app.stats.summary()["events"].get("click", 0) >= 1)
        assert app.mark_incident()
        assert wait(qapp, lambda: saved)
        path = saved[0]
        assert path.parent == tmp_path / "incidents"
        recorded = read_incident(path)
        assert recorded.header["start_armed"] and recorded.header["seconds"] <= 5.1
        assert any(e.kind.value == "click" for e in recorded.recorded_events())
        assert app.stats.incidents == 1
    finally:
        app.shutdown()
    sessions = list((tmp_path / "logs").glob("session-*.json"))
    assert len(sessions) == 1
    data = json.loads(sessions[0].read_text(encoding="utf-8"))
    assert data["incidents"] == 1 and data["frames"] > 20 and data["latency_ms"]["p95"] is not None


def test_mark_incident_hotkey_is_registered_by_default():
    hotkeys = Config().hotkeys
    assert hotkeys.mark_incident == "<ctrl>+<alt>+m"
    import pytest

    from aether.config import ConfigError

    with pytest.raises(ConfigError):
        parse_config({"hotkeys": {"mark_incident": "<ctrl>+<alt>+g"}})  # ชนกับ toggle_arm
