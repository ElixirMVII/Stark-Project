"""หน้าต่างสอนท่ามือ + การเชื่อมกับ AppController (Qt แบบ offscreen, ไม่มีกล้องจริง)"""

from __future__ import annotations

import json
import logging

import pytest
from posesim import HEIGHT, WIDTH, make_pose_hand, samples
from test_m4 import Factory, Loader, wait

from aether import __main__ as cli
from aether.app import AppController, Options
from aether.config import Config, parse_config
from aether.gestures import EventKind
from aether.pose_teach import TeachStep
from aether.pose_wizard import PoseWizard
from aether.poses import PoseLibrary, load_poses, make_template, pose_features, save_poses
from aether.tracker import HandFrame

ROCK = "01001"


class Frames:
    def __init__(self) -> None:
        self.t = 1000

    def frame(self, hand) -> HandFrame:
        self.t += 33
        return HandFrame(self.t, WIDTH, HEIGHT, () if hand is None else (hand,))


def rock_library() -> PoseLibrary:
    data = [pose_features(h, WIDTH, HEIGHT) for h in samples(ROCK, seed=3)]
    neutral = [pose_features(h, WIDTH, HEIGHT)[0] for h in samples("11111", 60, seed=4)]
    return PoseLibrary().with_pose(make_template("rock", "ctrl+c", data)).with_negatives(neutral)


def run_teaching(wizard: PoseWizard, frames: Frames, fingers: str = ROCK) -> None:
    session = wizard.session
    assert session is not None
    for hand in samples("11111", 200, seed=frames.t):
        if session.step is not TeachStep.NEUTRAL:
            break
        wizard.show_frame(frames.frame(hand))
    for hand in samples(fingers, 200, seed=frames.t + 1):
        if session.done or session.error:
            break
        wizard.show_frame(frames.frame(hand))


@pytest.fixture
def wizard(qapp, tmp_path):
    w = PoseWizard(Config(), PoseLibrary(), tmp_path / "poses.json")
    yield w
    w.close()


def test_teach_and_save_writes_poses_json(qapp, wizard, tmp_path):
    results: list[bool] = []
    wizard.finished.connect(results.append)
    assert wizard.start_teaching("rock", "ctrl+c")
    assert not wizard._save_button.isEnabled()  # ยังสอนไม่เสร็จ
    run_teaching(wizard, Frames())
    assert wizard.session.done and wizard._save_button.isEnabled()
    assert "ตัวอย่าง" in wizard._status.text()
    assert wizard.save()
    assert wizard.session is None and wizard.changed
    saved = load_poses(tmp_path / "poses.json")
    assert saved.names() == ("rock",) and saved.get("rock").action == "ctrl+c"
    assert wizard._list.count() == 1 and "rock" in wizard._list.item(0).text()
    wizard.close()
    assert results == [True]


def test_bad_name_is_reported_and_nothing_starts(qapp, wizard):
    assert not wizard.start_teaching("not ok!", "ctrl+c")
    assert wizard.session is None and "ชื่อท่า" in wizard._hint.text()
    assert not wizard.start_teaching("ok", "ctrl+nosuchkey")
    assert wizard.session is None


def test_cancel_teaching_writes_nothing(qapp, wizard, tmp_path):
    results: list[bool] = []
    wizard.finished.connect(results.append)
    wizard.start_teaching("rock", "ctrl+c")
    run_teaching(wizard, Frames())
    wizard.cancel_teaching()
    assert wizard.session is None and not (tmp_path / "poses.json").exists()
    wizard.close()
    assert results == [False]


def test_refused_pose_shows_the_reason(qapp, wizard):
    wizard.start_teaching("palm", "ctrl+c")
    frames = Frames()
    for hand in samples("11111", 400, seed=1):  # ทั้งสองขั้นด้วยแบมือ (~9 วินาที)
        wizard.show_frame(frames.frame(hand))
        if wizard.session.error:
            break
    assert "มือตอนชี้ปกติ" in wizard._hint.text()
    assert not wizard._save_button.isEnabled()


def test_live_test_shows_which_pose_the_hand_matches(qapp, tmp_path):
    wizard = PoseWizard(Config(), rock_library(), tmp_path / "poses.json")
    try:
        frames = Frames()
        wizard.show_frame(frames.frame(make_pose_hand(ROCK)))
        assert "✓ rock" in wizard._live.text()
        wizard.show_frame(frames.frame(make_pose_hand("11111")))
        assert "✗" in wizard._live.text()
        wizard.show_frame(frames.frame(None))
        assert "ไม่เห็นมือ" in wizard._live.text()
    finally:
        wizard.close()


def test_delete_selected_pose(qapp, tmp_path):
    path = tmp_path / "poses.json"
    save_poses(path, rock_library())
    wizard = PoseWizard(Config(), load_poses(path), path)
    results: list[bool] = []
    wizard.finished.connect(results.append)
    assert not wizard._delete_button.isEnabled()  # ยังไม่ได้เลือก
    wizard._list.setCurrentRow(0)
    assert wizard._delete_button.isEnabled()
    assert wizard.delete_selected()
    assert load_poses(path).names() == () and wizard._list.count() == 0
    wizard.close()
    assert results == [True]


def test_unwritable_location_is_reported(qapp, tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    wizard = PoseWizard(Config(), PoseLibrary(), blocker / "poses.json")
    try:
        wizard.start_teaching("rock", "ctrl+c")
        run_teaching(wizard, Frames())
        assert not wizard.save()
        assert "บันทึก" in wizard._hint.text() and not wizard.changed
    finally:
        wizard.close()


# ---- AppController ------------------------------------------------------ #


@pytest.fixture
def controlled(qapp, tmp_path):
    factory, loader = Factory(), Loader()
    controller = AppController(
        Options(no_hud=True),
        app_factory=factory,
        config_loader=loader,
        calibration_file=tmp_path / "calibration.json",
        poses_file=tmp_path / "poses.json",
        use_tray=True,
    )
    yield controller, factory, loader, tmp_path / "poses.json"
    controller.shutdown()


def test_saved_poses_are_given_to_the_app_at_start(qapp, controlled):
    controller, factory, loader, path = controlled
    save_poses(path, rock_library())
    controller.start(loader())
    assert factory.poses[0].names() == ("rock",)
    app = controller.app
    assert app.poses.names() == ("rock",)
    assert app._executor._actions["pose:rock"].keys == ("ctrl", "c")
    # engine รู้จักท่านี้: ค้างท่า (ตอน ARM) → ACTION pose:rock → กด ctrl+c
    app.toggle_armed()
    events: list = []
    app.gesture_events.connect(lambda batch: events.extend(batch))
    factory.trackers[0].hand = make_pose_hand(ROCK)
    assert wait(qapp, lambda: any(e.kind is EventKind.ACTION for e in events), timeout=3.0)
    assert [e.name for e in events if e.kind is EventKind.ACTION] == ["pose:rock"]
    calls = factory.backends[0].calls
    assert ("key_down", "ctrl") in calls and ("key_down", "c") in calls


def test_disabled_poses_are_not_loaded(qapp, controlled):
    controller, factory, _loader, path = controlled
    save_poses(path, rock_library())
    controller.start(parse_config({"hud": {"sounds": False}, "poses": {"enabled": False}}))
    assert factory.poses[0].names() == ()
    assert controller.app.poses.names() == ()


def test_broken_poses_file_starts_without_poses(qapp, controlled, caplog):
    controller, factory, loader, path = controlled
    path.write_text("{broken", encoding="utf-8")
    with caplog.at_level(logging.ERROR):
        controller.start(loader())
    assert factory.poses[0].names() == ()
    assert "poses.json" in caplog.text
    assert controller.app is not None


def test_pose_wizard_flow_disarms_feeds_frames_and_reloads(qapp, controlled):
    controller, factory, loader, path = controlled
    controller.start(loader())
    app = controller.app
    app.toggle_armed()
    controller.tray.actions()["poses"].trigger()
    wizard = controller.wizard
    assert isinstance(wizard, PoseWizard) and wizard.isVisible()
    assert not app.pipeline.armed
    app.toggle_armed()  # arm ระหว่างเปิดหน้าต่างสอน → ถูกปิดทันที
    assert not app.pipeline.armed

    controller.open_calibration()  # เปิดได้ทีละ wizard: ได้หน้าต่างเดิม
    assert controller.wizard is wizard

    factory.trackers[0].hand = make_pose_hand(ROCK)
    assert wait(qapp, lambda: "ยังไม่มีท่า" in wizard._live.text())  # ได้เฟรมจากกล้อง

    save_poses(path, rock_library())  # จำลองว่าสอนแล้วบันทึก
    wizard._changed = True
    wizard.close()
    assert controller.wizard is None
    assert len(factory.apps) == 2 and factory.poses[1].names() == ("rock",)


def test_closing_pose_wizard_without_changes_does_not_reload(qapp, controlled):
    controller, factory, loader, _path = controlled
    controller.start(loader())
    controller.open_poses()
    controller.wizard.close()
    assert controller.wizard is None and len(factory.apps) == 1


def test_teach_option_opens_pose_wizard_at_start(qapp, tmp_path):
    factory, loader = Factory(), Loader()
    controller = AppController(
        Options(no_hud=True, teach=True),
        app_factory=factory,
        config_loader=loader,
        poses_file=tmp_path / "poses.json",
        use_tray=False,
    )
    controller.start(loader())
    try:
        assert isinstance(controller.wizard, PoseWizard)
    finally:
        controller.shutdown()


def test_cli_teach_flag(monkeypatch):
    captured = {}
    monkeypatch.setattr("aether.app.run", lambda options: captured.setdefault("o", options) and 0)
    cli.main(["--teach", "--dry-run"])
    assert captured["o"].teach and captured["o"].dry_run


def test_poses_file_contains_no_images(qapp, controlled):
    """privacy: poses.json เก็บแค่ชื่อ ปุ่มลัด และตัวเลข"""
    _controller, _factory, _loader, path = controlled
    save_poses(path, rock_library())
    text = path.read_text(encoding="utf-8")
    data = json.loads(text)
    assert set(data) == {"version", "poses", "negatives"}
    assert len(text) < 200_000
