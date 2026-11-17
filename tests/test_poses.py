"""ท่าที่ผู้ใช้สอนเอง: feature, การจำแนก, ไฟล์ poses.json (pure logic — ไม่มีกล้อง)"""

from __future__ import annotations

import json
import math
import time

import pytest
from posesim import HEIGHT, POSES, WIDTH, make_pose_hand, samples

from aether.poses import (
    FEATURE_SIZE,
    MAX_NEGATIVES,
    MAX_POSES,
    MAX_SAMPLES,
    POSE_PREFIX,
    PoseClassifier,
    PoseFileError,
    PoseLibrary,
    angle_difference,
    feature_distance,
    load_poses,
    make_template,
    pose_features,
    save_poses,
    spread,
    validate_pose_name,
)


def features(hand):
    result = pose_features(hand, WIDTH, HEIGHT)
    assert result is not None
    return result


def template(name: str, fingers: str, *, seed: int = 1, action: str = "f5", **kwargs):
    return make_template(name, action, [features(h) for h in samples(fingers, seed=seed, **kwargs)])


def library(*names: str, neutral: str | None = "11111") -> PoseLibrary:
    lib = PoseLibrary()
    for i, name in enumerate(names):
        lib = lib.with_pose(template(name, POSES[name], seed=i + 1))
    if neutral is not None:
        lib = lib.with_negatives(features(h)[0] for h in samples(neutral, 60, seed=99))
    return lib


# ---- features ----------------------------------------------------------- #


def test_feature_vector_size_and_upright_orientation():
    vector, angle = features(make_pose_hand("01001"))
    assert len(vector) == FEATURE_SIZE
    assert abs(angle) < 1.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"x": 0.2, "y": 0.4},  # ตำแหน่งในเฟรม
        {"scale": 60.0},  # ขนาดมือ (นั่งไกล/ใกล้กล้อง)
        {"angle": 35.0},  # หมุนในระนาบภาพ
        {"side": "left"},  # มือซ้ายท่าเดียวกัน
    ],
)
def test_features_ignore_position_size_rotation_and_side(kwargs):
    base, _ = features(make_pose_hand("01001"))
    moved, _ = features(make_pose_hand("01001", **kwargs))
    assert feature_distance(base, moved) < 1e-6


def test_orientation_follows_in_plane_rotation():
    _, right = features(make_pose_hand("10000", angle=90.0))
    _, down = features(make_pose_hand("10000", angle=180.0))
    assert right == pytest.approx(90.0, abs=0.5)
    assert angle_difference(down, 180.0) < 0.5


def test_mirrored_left_hand_reports_mirrored_orientation():
    _, right = features(make_pose_hand("10000", angle=30.0))
    _, left = features(make_pose_hand("10000", angle=-30.0, side="left"))
    assert left == pytest.approx(right, abs=0.5)


def test_one_finger_changes_the_shape_clearly():
    a, _ = features(make_pose_hand("10000"))
    b, _ = features(make_pose_hand("00000"))
    c, _ = features(make_pose_hand("10001"))
    assert feature_distance(a, b) > 0.15
    assert feature_distance(a, c) > 0.15


def test_degenerate_hand_has_no_features():
    hand = make_pose_hand(scale=0.0)
    assert pose_features(hand, WIDTH, HEIGHT) is None


def test_angle_difference_wraps():
    assert angle_difference(170.0, -170.0) == pytest.approx(20.0)
    assert angle_difference(0.0, 360.0) == pytest.approx(0.0)


def test_spread_picks_evenly():
    assert spread(list(range(10)), 3) == [0, 4, 9]
    assert spread([1, 2], 5) == [1, 2]
    assert spread(list(range(5)), 1) == [2]


# ---- templates ---------------------------------------------------------- #


def test_template_caps_samples_and_learns_a_bounded_threshold():
    t = make_template("rock", "ctrl+c", [features(h) for h in samples("01001", 90)])
    assert len(t.samples) == MAX_SAMPLES
    assert 0.10 <= t.threshold <= 0.25
    assert abs(t.orientation) < 5.0
    assert t.label == POSE_PREFIX + "rock"


def test_template_validates_name_and_action():
    data = [features(make_pose_hand("01001"))]
    with pytest.raises(ValueError):
        make_template("Rock Star!", "f5", data)
    with pytest.raises(ValueError):
        make_template("rock", "ctrl+nosuchkey", data)
    with pytest.raises(ValueError):
        make_template("rock", "f5", [])


@pytest.mark.parametrize("name", ["rock", "  Rock  ", "two_fingers", "a-1", "9"])
def test_valid_pose_names(name):
    assert validate_pose_name(name) == name.strip().lower()


@pytest.mark.parametrize("name", ["", "_x", "มือ", "a b", "x" * 25])
def test_invalid_pose_names(name):
    with pytest.raises(ValueError):
        validate_pose_name(name)


def test_library_add_replace_remove_and_actions():
    lib = library("rock", "gun", neutral=None)
    assert lib.names() == ("rock", "gun")
    replaced = lib.with_pose(template("rock", "01001", action="ctrl+c"))
    assert replaced.names() == ("rock", "gun")
    assert replaced.get("rock").action == "ctrl+c"
    assert replaced.actions() == {"pose:rock": "ctrl+c", "pose:gun": "f5"}
    assert replaced.without("rock").names() == ("gun",)
    assert replaced.with_action("gun", "alt+tab").get("gun").action == "alt+tab"
    with pytest.raises(ValueError):
        replaced.with_action("gun", "ctrl+nosuchkey")


def test_library_limits():
    lib = PoseLibrary()
    shape = template("p0", "01001")
    for i in range(MAX_POSES):
        lib = lib.with_pose(make_template(f"p{i}", "f5", [(shape.samples[0], 0.0)]))
    with pytest.raises(ValueError):
        lib.with_pose(make_template("extra", "f5", [(shape.samples[0], 0.0)]))
    lots = lib.with_negatives([shape.samples[0]] * (MAX_NEGATIVES + 30))
    assert len(lots.negatives) == MAX_NEGATIVES


# ---- classifier --------------------------------------------------------- #


@pytest.mark.parametrize("name", ["rock", "gun", "shaka", "three"])
def test_taught_pose_is_recognised_on_new_frames(name):
    classifier = PoseClassifier(library("rock", "gun", "shaka", "three"))
    hits = 0
    for hand in samples(POSES[name], 30, seed=500):
        match = classifier.classify(hand, WIDTH, HEIGHT)
        if match is not None and match.accepted:
            assert match.name == name
            hits += 1
    assert hits >= 27  # ≥ 90% ของเฟรม


def test_left_hand_matches_pose_taught_with_right_hand():
    classifier = PoseClassifier(library("rock"))
    match = classifier.classify(make_pose_hand("01001", side="left"), WIDTH, HEIGHT)
    assert match is not None and match.accepted and match.name == "rock"


@pytest.mark.parametrize("other", ["open", "fist", "pinky", "thumb"])
def test_untaught_shapes_are_not_accepted(other):
    classifier = PoseClassifier(library("rock", "gun", "shaka", "three"))
    for hand in samples(POSES[other], 30, seed=700):
        match = classifier.classify(hand, WIDTH, HEIGHT)
        assert match is None or not match.accepted, (other, match)


def test_neutral_hand_is_rejected_even_when_close_to_a_pose():
    # "three" ใกล้แบมือ (มือปกติ) — ต้องไม่ติดเพราะมือปกติที่อัดไว้ใกล้กว่า
    lib = library("three", neutral="11111")
    loose = PoseClassifier(lib, sensitivity=4.0)  # เกณฑ์กว้างมากจนเรื่องระยะไม่กั้นแล้ว
    reasons = {loose.classify(h, WIDTH, HEIGHT).reason for h in samples("11111", 20, seed=800)}
    assert reasons == {"neutral"}


def test_orientation_separates_thumb_up_from_thumb_down():
    lib = PoseLibrary().with_pose(template("up", "10000"))
    classifier = PoseClassifier(lib)
    up = classifier.classify(make_pose_hand("10000", angle=5.0), WIDTH, HEIGHT)
    assert up is not None and up.accepted
    assert classifier.classify(make_pose_hand("10000", angle=180.0), WIDTH, HEIGHT) is None


def test_two_near_identical_poses_are_ambiguous():
    lib = PoseLibrary()
    lib = lib.with_pose(template("a", "01001", seed=1))
    lib = lib.with_pose(template("b", "01001", seed=2))
    match = PoseClassifier(lib).classify(make_pose_hand("01001"), WIDTH, HEIGHT)
    assert match is not None and not match.accepted and match.reason == "ambiguous"


def test_sensitivity_scales_the_threshold():
    lib = library("rock", neutral=None)
    hand = make_pose_hand("01101")  # rock + นิ้วกลาง: ใกล้ rock แต่ไม่ใช่
    strict = PoseClassifier(lib, sensitivity=1.0).classify(hand, WIDTH, HEIGHT)
    loose = PoseClassifier(lib, sensitivity=4.0).classify(hand, WIDTH, HEIGHT)
    assert strict is not None and not strict.accepted and strict.reason == "far"
    assert loose is not None and loose.accepted
    assert loose.threshold == pytest.approx(strict.threshold * 4.0)


def test_empty_library_classifies_nothing():
    classifier = PoseClassifier(PoseLibrary())
    assert not classifier
    assert classifier.classify(make_pose_hand("01001"), WIDTH, HEIGHT) is None


def test_classification_is_fast_enough_for_every_frame():
    lib = library("rock", "gun", "shaka", "three", "thumb", "pinky")
    shape = lib.poses[0].samples[0]
    for i in range(6, MAX_POSES):  # เต็มจำนวนท่า
        lib = lib.with_pose(
            make_template(f"p{i}", "f5", [(tuple(v + 0.3 * i for v in shape), 0.0)] * 40)
        )
    lib = lib.with_negatives([shape] * MAX_NEGATIVES)
    classifier = PoseClassifier(lib)
    hands = samples("01001", 60, seed=3)
    start = time.perf_counter()
    for hand in hands:
        classifier.classify(hand, WIDTH, HEIGHT)
    per_frame = (time.perf_counter() - start) / len(hands)
    assert per_frame < 0.005  # < 5 ms ต่อเฟรม (งบของทั้ง pipeline คือ ~33 ms)


# ---- poses.json --------------------------------------------------------- #


def test_save_and_load_round_trip(tmp_path):
    lib = library("rock", "gun")
    path = tmp_path / "poses.json"
    save_poses(path, lib)
    loaded = load_poses(path)
    assert loaded.names() == lib.names()
    assert loaded.actions() == lib.actions()
    assert len(loaded.negatives) == len(lib.negatives)
    assert loaded.poses[0].samples[0] == pytest.approx(lib.poses[0].samples[0])
    hand = make_pose_hand("01001")
    a = PoseClassifier(lib).classify(hand, WIDTH, HEIGHT)
    b = PoseClassifier(loaded).classify(hand, WIDTH, HEIGHT)
    assert (a.name, a.accepted) == (b.name, b.accepted)
    assert a.distance == pytest.approx(b.distance, abs=1e-3)
    assert not (tmp_path / "poses.json.tmp").exists()


def test_missing_file_is_an_empty_library(tmp_path):
    assert len(load_poses(tmp_path / "nope.json")) == 0


@pytest.mark.parametrize(
    "content",
    [
        "{not json",
        json.dumps({"version": 99, "poses": []}),
        json.dumps({"version": 1, "poses": [{"name": "x"}]}),
        json.dumps(
            {
                "version": 1,
                "poses": [
                    {"name": "x", "action": "f5", "orientation": 0, "threshold": 0.2,
                     "samples": [[0.0] * 5]}
                ],
            }
        ),
        json.dumps(
            {
                "version": 1,
                "poses": [
                    {"name": "x", "action": "ctrl+nosuchkey", "orientation": 0,
                     "threshold": 0.2, "samples": [[0.0] * FEATURE_SIZE]}
                ],
            }
        ),
    ],
)  # fmt: skip
def test_broken_file_is_reported(tmp_path, content):
    path = tmp_path / "poses.json"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(PoseFileError, match="poses.json"):
        load_poses(path)


def test_file_holds_numbers_only(tmp_path):
    """privacy: ไฟล์มีแค่ตัวเลข landmark — ไม่มีภาพ"""
    path = tmp_path / "poses.json"
    save_poses(path, library("rock"))
    data = json.loads(path.read_text(encoding="utf-8"))
    assert set(data) == {"version", "poses", "negatives"}
    assert set(data["poses"][0]) == {"name", "action", "orientation", "threshold", "samples"}
    values = [v for s in data["poses"][0]["samples"] for v in s]
    assert all(isinstance(v, float | int) and math.isfinite(v) for v in values)
