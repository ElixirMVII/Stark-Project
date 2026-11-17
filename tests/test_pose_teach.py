"""ขั้นตอนสอนท่าใหม่ (TeachSession) — ป้อนเฟรมสังเคราะห์ ไม่มีกล้อง ไม่มี Qt"""

from __future__ import annotations

import pytest
from handsim import make_hand
from posesim import HEIGHT, WIDTH, make_pose_hand, samples

from aether.config import Config, parse_config
from aether.pose_teach import SETTLE_MS, TeachSession, TeachStep
from aether.poses import MAX_POSES, PoseClassifier, PoseLibrary, make_template, pose_features
from aether.tracker import Hand, HandFrame

FRAME_MS = 33


class Feeder:
    def __init__(self, session: TeachSession) -> None:
        self.session = session
        self.t = 1000

    def feed(self, hands: list[Hand | None] | Hand | None, ms: int = 0) -> None:
        if not isinstance(hands, list):
            hands = [hands] * max(1, round(ms / FRAME_MS))
        for hand in hands:
            frame = HandFrame(self.t, WIDTH, HEIGHT, () if hand is None else (hand,))
            self.session.update(frame)
            self.t += FRAME_MS

    def step(self, fingers: str | None = None, *, hands: list[Hand] | None = None, **kw) -> None:
        """ผ่านหนึ่งขั้น (จัดท่า + เก็บข้อมูล) ด้วยมือท่า ``fingers`` ที่ขยับ/เอียงเล็กน้อย"""
        assert fingers is not None or hands is not None
        frames = hands if hands is not None else samples(fingers, 160, seed=self.t, **kw)
        step, error = self.session.step, self.session.error
        for hand in frames:
            self.feed([hand])
            if self.session.step is not step:
                return  # ผ่านขั้นนี้แล้ว
            if self.session.error is not None and self.session.error != error:
                return  # ขั้นนี้ไม่ผ่าน (ข้อความใหม่)
            if self.session.error is None:
                error = None


def session(
    library: PoseLibrary | None = None,
    name: str = "rock",
    action: str = "ctrl+c",
    config: Config | None = None,
) -> Feeder:
    return Feeder(TeachSession(config or Config(), library or PoseLibrary(), name, action))


def teach(feeder: Feeder, fingers: str, neutral: str = "11111", **kw) -> Feeder:
    feeder.step(neutral, **({"side": kw["side"]} if "side" in kw else {}))
    assert feeder.session.step is TeachStep.POSE, feeder.session.error
    feeder.step(fingers, **kw)
    return feeder


def test_happy_path_produces_a_library_with_the_new_pose():
    f = teach(session(), "01001")
    s = f.session
    assert s.done and s.error is None
    assert s.result is not None and s.result.names() == ("rock",)
    assert s.result.get("rock").action == "ctrl+c"
    assert len(s.result.negatives) > 0
    assert s.summary is not None and s.summary.neutral_margin > 1.0
    # library ที่ได้จำท่านี้ได้จริง และไม่จำมือปกติ
    classifier = PoseClassifier(s.result)
    assert classifier.classify(make_pose_hand("01001"), WIDTH, HEIGHT).accepted
    neutral = classifier.classify(make_pose_hand("11111"), WIDTH, HEIGHT)
    assert neutral is None or not neutral.accepted


def test_settle_time_and_progress():
    f = session()
    s = f.session
    assert s.step is TeachStep.NEUTRAL and not s.sampling and s.progress == 0.0
    f.feed(make_pose_hand("11111"), SETTLE_MS - 200)
    assert not s.sampling and 0 < s.countdown_ms <= 250
    f.feed(make_pose_hand("11111"), 1000)
    assert s.sampling and 0.0 < s.progress < 1.0


def test_time_only_runs_while_the_hand_is_visible():
    f = session()
    f.feed(None, 5000)
    assert f.session.countdown_ms == SETTLE_MS and not f.session.hand_visible
    f.feed(make_pose_hand("11111"), 100)
    assert f.session.hand_visible


def test_only_the_dominant_hand_counts():
    f = session()
    f.feed(make_pose_hand("11111", side="left"), 6000)
    assert f.session.step is TeachStep.NEUTRAL and f.session.countdown_ms == SETTLE_MS


def test_left_handed_user_teaches_with_the_left_hand():
    f = session(config=parse_config({"tracking": {"dominant_hand": "left"}}))
    teach(f, "01001", side="left")
    assert f.session.done
    # ท่าที่สอนด้วยมือซ้ายจำมือขวาท่าเดียวกันได้ด้วย (feature กลับซ้าย/ขวาแล้ว)
    match = PoseClassifier(f.session.result).classify(make_pose_hand("01001"), WIDTH, HEIGHT)
    assert match is not None and match.accepted


def test_builtin_gesture_is_refused_and_the_step_can_be_retried():
    f = session()
    f.step("11111")
    f.step("00000", gesture="Closed_Fist", score=0.9)
    s = f.session
    assert s.step is TeachStep.POSE and not s.done
    assert s.error is not None and "✊" in s.error
    assert s.countdown_ms == SETTLE_MS  # เริ่มขั้นนี้ใหม่
    f.step("01001")  # ลองท่าอื่น
    assert s.done and s.error is None


def test_low_confidence_builtin_label_is_not_a_conflict():
    f = teach(session(), "01001", gesture="Victory", score=0.3)
    assert f.session.done


def test_thumb_up_is_allowed_when_the_launcher_is_off():
    refused = teach(session(), "10000", gesture="Thumb_Up", score=0.9)
    assert "👍" in (refused.session.error or "")
    config = parse_config({"launcher": {"enabled": False}})
    allowed = teach(session(config=config), "10000", gesture="Thumb_Up", score=0.9)
    assert allowed.session.done


def test_pose_that_touches_the_thumb_is_refused():
    f = session()
    f.step("11111")
    f.step(hands=[make_hand(pinch=0.30, x=0.5 + i * 0.001) for i in range(160)])
    assert f.session.error is not None and "คลิก" in f.session.error


def test_pose_that_looks_like_the_normal_hand_is_refused():
    f = teach(session(), "11111")
    assert f.session.error is not None and "มือตอนชี้ปกติ" in f.session.error


def test_pose_similar_to_an_existing_one_is_refused():
    first = teach(session(), "01001")
    library = first.session.result
    f = teach(session(library, name="rock2"), "01001")
    assert f.session.error is not None and '"rock"' in f.session.error


def test_re_teaching_the_same_name_replaces_the_pose():
    first = teach(session(action="ctrl+c"), "01001")
    again = teach(session(first.session.result, action="ctrl+v"), "01001")
    result = again.session.result
    assert again.session.done and result.names() == ("rock",)
    assert result.get("rock").action == "ctrl+v"


def test_same_shape_facing_another_way_is_a_different_pose():
    config = parse_config({"launcher": {"enabled": False}})
    up = teach(session(name="up", config=config), "10000")
    down = teach(session(up.session.result, name="down", config=config), "10000", angle=180.0)
    assert down.session.done, down.session.error
    assert down.session.result.names() == ("up", "down")


def test_different_poses_can_be_taught_one_after_another():
    library = PoseLibrary()
    for name, fingers in (("rock", "01001"), ("gun", "11000"), ("shaka", "10001")):
        f = teach(session(library, name=name), fingers)
        assert f.session.done, (name, f.session.error)
        library = f.session.result
    assert library.names() == ("rock", "gun", "shaka")


def test_restart_goes_back_to_the_first_step():
    f = session()
    f.step("11111")
    f.session.restart()
    assert f.session.step is TeachStep.NEUTRAL and f.session.countdown_ms == SETTLE_MS


@pytest.mark.parametrize(("name", "action"), [("bad name", "f5"), ("ok", "ctrl+nosuchkey")])
def test_bad_name_or_action_is_rejected_up_front(name, action):
    with pytest.raises(ValueError):
        TeachSession(Config(), PoseLibrary(), name, action)


def test_pose_limit_is_checked_up_front():
    shape = pose_features(make_pose_hand("01001"), WIDTH, HEIGHT)
    library = PoseLibrary()
    for i in range(MAX_POSES):
        library = library.with_pose(make_template(f"p{i}", "f5", [shape]))
    with pytest.raises(ValueError):
        TeachSession(Config(), library, "extra", "f5")
    TeachSession(Config(), library, "p3", "f5")  # สอนท่าเดิมซ้ำได้เสมอ
