"""มือสังเคราะห์ 3D, การสอนชุดท่าสำเร็จรูป, การวัดผล, ไฟล์ชุดท่าที่แถมมา และการเพิ่มท่าในหน้าต่างสอนท่า"""

from __future__ import annotations

import json
import math
import random
import subprocess
import sys
from pathlib import Path

import pytest
from posesim import HEIGHT, WIDTH, make_pose_hand, samples

from aether.config import Config
from aether.geometry import Finger, finger_extended, pointer_anchor
from aether.handmodel import HandShape, HandView, build_hand, skeleton
from aether.poses import (
    MAX_NEGATIVES,
    MAX_SAMPLES,
    PoseClassifier,
    PoseLibrary,
    load_poses,
    make_template,
    pose_features,
)
from aether.studio.lab import SampleRecorder, read_samples
from aether.tracker import LM
from aether.training import (
    NEGATIVE_CLASSES,
    PACK_POSES,
    POSE_SPECS,
    THRESHOLD_LIMITS,
    evaluate,
    farthest_points,
    generate,
    load_builtin_pack,
    sample_hand,
    train_from_samples,
    train_pack,
)

ROOT = Path(__file__).resolve().parents[1]
FINGERS = (Finger.THUMB, Finger.INDEX, Finger.MIDDLE, Finger.RING, Finger.PINKY)


# ---- มือสังเคราะห์ 3D ------------------------------------------------------- #


def test_skeleton_has_human_proportions():
    points = skeleton(HandShape())
    middle_length = math.dist(points[LM.MIDDLE_MCP], points[LM.MIDDLE_TIP])
    palm_length = math.dist(points[LM.WRIST], points[LM.MIDDLE_MCP])
    assert 0.085 < middle_length < 0.10 and 0.085 < palm_length < 0.095  # เมตร (มือผู้ใหญ่)
    assert points[LM.THUMB_TIP][0] < points[LM.INDEX_TIP][0] < points[LM.PINKY_TIP][0]
    assert all(p[2] == pytest.approx(0.0) for p in points[LM.INDEX_MCP : LM.INDEX_TIP + 1])
    fist = skeleton(HandShape((1.0,) * 5))
    # ตอนกำ ข้อนิ้วยื่นเข้าหากล้อง (ฝ่ามือหันเข้ากล้อง) แล้วปลายนิ้วพับกลับลงมาต่ำกว่าข้อ
    assert fist[LM.INDEX_PIP][2] > 0.03
    assert fist[LM.INDEX_TIP][1] < fist[LM.INDEX_PIP][1]


@pytest.mark.parametrize("name", list(POSE_SPECS))
def test_every_spec_shows_the_right_fingers(name):
    spec = POSE_SPECS[name]
    if name in ("relaxed", "pinch"):  # ท่ากึ่งงอ: ไม่มีคำตอบตายตัว
        return
    rng = random.Random(3)
    expected = tuple(high <= 0.2 for _, high in spec.curl)
    for side in ("right", "left"):
        hand = sample_hand(spec, rng, side=side)
        assert tuple(finger_extended(hand, f, WIDTH, HEIGHT) for f in FINGERS) == expected


def test_projection_places_the_wrist_and_is_deterministic():
    view = HandView(x=0.3, y=0.7, distance=0.6, roll=20, pitch=10, yaw=-15)
    hand = build_hand(HandShape(), view)
    wrist = hand.landmarks[LM.WRIST]
    assert (wrist.x, wrist.y) == pytest.approx((0.3, 0.7), abs=1e-9)
    assert hand.landmarks[LM.WRIST].z == pytest.approx(0.0, abs=1e-9)
    span = math.dist(hand.world_landmarks[LM.WRIST], hand.world_landmarks[LM.MIDDLE_TIP])
    assert 0.15 < span < 0.21  # world landmark เป็นเมตร
    near = build_hand(HandShape(), HandView(distance=0.35))
    far = build_hand(HandShape(), HandView(distance=0.8))

    def size(h):
        return math.dist((h.landmarks[0].x, h.landmarks[0].y), (h.landmarks[9].x, h.landmarks[9].y))

    assert size(near) > size(far) * 2
    noisy_a = build_hand(HandShape(), HandView(noise=0.003), rng=random.Random(1))
    noisy_b = build_hand(HandShape(), HandView(noise=0.003), rng=random.Random(1))
    assert noisy_a == noisy_b


def test_left_hand_is_the_mirror_of_the_right_one():
    shape = HandShape((0.0, 0.0, 1.0, 1.0, 0.0))
    right = pose_features(build_hand(shape, HandView(side="right")), WIDTH, HEIGHT)
    left = pose_features(build_hand(shape, HandView(side="left")), WIDTH, HEIGHT)
    assert right is not None and left is not None
    assert max(abs(a - b) for a, b in zip(right[0], left[0], strict=True)) < 1e-6
    assert pointer_anchor(build_hand(shape, HandView(side="left")))[0] > 0.5  # โป้งอยู่ขวาของภาพ


def test_roll_is_removed_by_the_pose_features():
    a = pose_features(build_hand(HandShape(), HandView(roll=0)), WIDTH, HEIGHT)
    b = pose_features(build_hand(HandShape(), HandView(roll=30)), WIDTH, HEIGHT)
    assert max(abs(x - y) for x, y in zip(a[0], b[0], strict=True)) < 0.06
    assert b[1] - a[1] == pytest.approx(30.0, abs=3.0)


# ---- การสอน / วัดผล -------------------------------------------------------- #


def test_farthest_points_cover_the_spread():
    vectors = [(float(i), 0.0) for i in range(100)]
    chosen = farthest_points(vectors, 5)
    assert len(set(chosen)) == 5 and 0 in chosen and 99 in chosen
    assert farthest_points(vectors[:3], 5) == [0, 1, 2]


@pytest.fixture(scope="module")
def small_pack():
    names = [*PACK_POSES, *NEGATIVE_CLASSES]
    library = train_pack(generate(names, 150, 1), generate(names, 80, 2))
    return library, generate(names, 80, 3)


def test_trained_pack_recognises_held_out_hands(small_pack):
    library, test = small_pack
    assert library.names() == PACK_POSES
    assert all(len(p.samples) == MAX_SAMPLES for p in library.poses)
    assert all(THRESHOLD_LIMITS[0] <= p.threshold <= THRESHOLD_LIMITS[1] for p in library.poses)
    assert 0 < len(library.negatives) <= MAX_NEGATIVES
    report = evaluate(library, test)
    assert report.pose_recall > 0.85
    assert report.wrong_accept_rate < 0.02
    lines = report.lines()
    assert lines[0].split()[: len(PACK_POSES)] == list(PACK_POSES) and "recall" in lines[0]


def test_trained_pack_never_mistakes_engine_gestures_from_another_hand_model(small_pack):
    """มือสังเคราะห์อีกแบบ (posesim): ท่าที่ engine ใช้ (แบมือ กำมือ ชี้ ✌ 🤟) ต้องไม่ถูกรับเป็นท่าในชุด"""
    library, _ = small_pack
    classifier = PoseClassifier(library)
    for code in ("11111", "00000", "01000", "01100", "11001"):
        for hand in samples(code, 30, seed=4):
            match = classifier.classify(hand, WIDTH, HEIGHT)
            assert match is None or not match.accepted, code


# ---- ไฟล์ชุดท่าที่แถมมากับแอป ---------------------------------------------------- #


def test_builtin_pack_ships_with_the_app():
    library, meta = load_builtin_pack()
    assert library.names() == PACK_POSES
    assert all(p.action == "none" for p in library.poses)
    assert set(meta["descriptions"]) == set(PACK_POSES)
    assert meta["evaluation"]["pose_recall"] >= 0.9
    assert meta["evaluation"]["wrong_accept_rate"] <= 0.01
    assert "synthetic" in meta["source"]
    assert load_builtin_pack(Path("/nonexistent/pack.json")) == (PoseLibrary(), {})


def test_builtin_pack_rejects_common_hands_from_another_hand_model():
    """มือสังเคราะห์อีกแบบ (ไม่ได้ใช้สอน): แบมือ กำมือ ชี้ ✌ 🤟 สี่นิ้ว ต้องไม่ถูกรับเป็นท่าใดในชุด"""
    classifier = PoseClassifier(load_builtin_pack()[0])
    for code in ("11111", "00000", "01000", "01100", "11001", "01111"):
        for hand in samples(code, 40, seed=8) + samples(code, 20, seed=9, side="left"):
            match = classifier.classify(hand, WIDTH, HEIGHT)
            assert match is None or not match.accepted, code


def test_builtin_pack_recognises_a_clean_synthetic_rock():
    library, _ = load_builtin_pack()
    hand = build_hand(HandShape((0.9, 0.05, 0.95, 0.95, 0.05)), HandView(yaw=10))
    match = PoseClassifier(library).classify(hand, WIDTH, HEIGHT)
    assert match is not None and match.accepted and match.name == "rock"


# ---- สอนจากตัวอย่างจริงของ Hand lab ----------------------------------------------- #


def record_samples(path: Path) -> None:
    recorder = SampleRecorder(path, clock=lambda: 1.0)
    for label, code in (("rock", "01001"), ("peace", "01100"), ("none", "11111")):
        for hand in samples(code, 24, seed=hash(label) % 100):
            recorder.add(label, hand, WIDTH, HEIGHT)


def test_train_from_hand_lab_samples(tmp_path):
    path = tmp_path / "hand_samples.jsonl"
    record_samples(path)
    library, report = train_from_samples(read_samples(path))
    assert set(library.names()) == {"rock", "peace"}
    assert library.negatives  # ป้าย none = มือปกติ
    assert report is not None and report.recall("rock") > 0.8 and report.recall("peace") > 0.8
    with pytest.raises(ValueError):
        train_from_samples([r for r in read_samples(path) if r["label"] == "none"])


def test_cli_builds_a_pack_and_trains_from_samples(tmp_path):
    out, report = tmp_path / "pack.json", tmp_path / "PACK.md"
    tool = ROOT / "tools" / "train_poses.py"
    result = subprocess.run(
        [sys.executable, str(tool), "--count", "60", "--out", str(out), "--report", str(report)],
        capture_output=True,
        encoding="utf-8",
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "recall" in result.stdout
    library, meta = load_builtin_pack(out)
    assert library.names() == PACK_POSES and meta["samples_per_pose"] == 60
    assert "rock" in report.read_text(encoding="utf-8")

    samples_path = tmp_path / "hand_samples.jsonl"
    record_samples(samples_path)
    trained = tmp_path / "trained.json"
    result = subprocess.run(
        [sys.executable, str(tool), "--samples", str(samples_path), "--out", str(trained)],
        capture_output=True,
        encoding="utf-8",
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert set(load_poses(trained).names()) == {"rock", "peace"}
    assert json.loads(trained.read_text(encoding="utf-8"))["version"] == 1


# ---- หน้าต่างสอนท่า: เพิ่มท่าสำเร็จรูป ---------------------------------------------- #


def test_wizard_adds_a_builtin_pose_with_an_action(qapp, tmp_path):
    from aether.pose_wizard import PoseWizard

    own_negatives = tuple(
        pose_features(make_pose_hand("11111", noise=0.02), WIDTH, HEIGHT)[0] for _ in range(3)
    )
    taught = make_template(
        "mine", "ctrl+c", [pose_features(make_pose_hand("01110"), WIDTH, HEIGHT)]
    )
    library = PoseLibrary((taught,), own_negatives)
    path = tmp_path / "poses.json"
    wizard = PoseWizard(Config(), library, path)
    try:
        assert wizard.pack.names() == PACK_POSES
        assert not wizard.add_builtin("rock", "ctrl+nope")  # ปุ่มลัดผิด
        assert not wizard.add_builtin("unknown", "ctrl+c")
        assert wizard.add_builtin("rock", "media_next")
        saved = load_poses(path)
        assert saved.names() == ("mine", "rock")
        assert saved.get("rock").action == "media_next"
        assert saved.negatives[-3:] == own_negatives  # มือปกติของผู้ใช้อยู่ครบ
        assert len(saved.negatives) == MAX_NEGATIVES
        assert wizard.changed and "rock" in wizard._hint.text()
        wizard.start_teaching("other", "ctrl+v")
        assert not wizard.add_builtin("shaka", "ctrl+c")  # ระหว่างสอน: เพิ่มไม่ได้
    finally:
        wizard.close()
