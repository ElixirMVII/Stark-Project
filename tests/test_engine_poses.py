"""ท่าที่สอนเองใน gesture engine: ค้างท่า → ACTION ``pose:<ชื่อ>`` (เฟรมสังเคราะห์ ไม่มีกล้อง)"""

from __future__ import annotations

import pytest
from handsim import HEIGHT, WIDTH, Driver, make_hand
from posesim import make_pose_hand, samples

from aether.config import Config, parse_config
from aether.gestures import (
    DEFAULT_DOUBLE_CLICK_MS,
    EventKind,
    GestureEngine,
    State,
    reserved_gestures,
)
from aether.mapping import ScreenGeometry
from aether.poses import PoseClassifier, PoseLibrary, make_template, pose_features

FHD = ScreenGeometry(0, 0, 1920, 1080)
ROCK = "01001"


def taught(*shapes: tuple[str, str]) -> PoseLibrary:
    lib = PoseLibrary()
    for seed, (name, fingers) in enumerate(shapes, start=1):
        data = [pose_features(h, WIDTH, HEIGHT) for h in samples(fingers, seed=seed)]
        lib = lib.with_pose(make_template(name, "ctrl+c", data))
    return lib


def driver_for(
    lib: PoseLibrary | None = None, overrides: dict | None = None, *, armed: bool = True
) -> Driver:
    config = parse_config(overrides) if overrides else Config()
    classifier = PoseClassifier(lib if lib is not None else taught(("rock", ROCK)))
    engine = GestureEngine(config, FHD, double_click_ms=DEFAULT_DOUBLE_CLICK_MS, poses=classifier)
    driver = Driver(engine)
    if armed:
        engine.set_armed(True)
    driver.hold(make_pose_hand("11111", gesture="None"), 200)  # มือปกติก่อน
    driver.clear()
    return driver


def actions(driver: Driver) -> list[str]:
    return [e.name for e in driver.events if e.kind is EventKind.ACTION]


def test_holding_a_taught_pose_fires_its_action_once():
    driver = driver_for()
    driver.hold(make_pose_hand(ROCK), 300)
    snap = driver.engine.snapshot
    assert driver.engine.state is State.CHARGING
    assert snap.charge_kind == "pose:rock" and 0.3 < snap.charge < 0.8
    assert snap.pose is not None and snap.pose.accepted and snap.pose.name == "rock"
    driver.hold(make_pose_hand(ROCK), 1500)  # ค้างต่อไปนานๆ ต้องไม่ติดซ้ำ
    assert actions(driver) == ["pose:rock"]
    assert driver.engine.state is State.POINTER


def test_cursor_stays_still_while_charging_a_pose():
    driver = driver_for()
    driver.hold(make_pose_hand(ROCK), 100)
    before = driver.engine.snapshot.cursor
    driver.clear()
    driver.animate(lambda p: make_pose_hand(ROCK, 0.5 + 0.02 * p), 300)
    assert [e for e in driver.events if e.kind is EventKind.MOVE] == []
    assert driver.engine.snapshot.cursor == before


def test_pose_must_be_released_before_it_fires_again():
    driver = driver_for()
    driver.hold(make_pose_hand(ROCK), 800)
    driver.hold(make_pose_hand("11111"), 900)  # เลิกทำท่า + พ้น cooldown
    driver.hold(make_pose_hand(ROCK), 800)
    assert actions(driver) == ["pose:rock", "pose:rock"]


def test_hold_time_comes_from_config():
    driver = driver_for(overrides={"poses": {"hold_ms": 1200}})
    driver.hold(make_pose_hand(ROCK), 900)
    assert actions(driver) == []
    driver.hold(make_pose_hand(ROCK), 500)  # + ~3 เฟรมที่ vote ใช้ยืนยันท่า
    assert actions(driver) == ["pose:rock"]


def test_short_flash_of_the_pose_does_nothing():
    driver = driver_for()
    driver.hold(make_pose_hand(ROCK), 300)
    driver.hold(make_pose_hand("11111"), 500)
    assert actions(driver) == []
    assert driver.engine.state is State.POINTER


def test_confident_builtin_gesture_wins_over_a_taught_pose():
    driver = driver_for()
    driver.hold(make_pose_hand(ROCK, gesture="Victory", score=0.9), 800)
    assert driver.engine.state is State.SCROLL
    assert actions(driver) == []
    assert driver.engine.snapshot.pose is None  # ไม่ได้จำแนกด้วยซ้ำ


def test_low_confidence_builtin_gesture_does_not_block_the_pose():
    driver = driver_for()
    driver.hold(make_pose_hand(ROCK, gesture="Victory", score=0.3), 800)
    assert actions(driver) == ["pose:rock"]


def test_unused_builtin_gesture_can_be_overridden():
    # ปิดปุ่มหมุนแล้ว Pointing_Up ไม่ได้ใช้ทำอะไรใน engine — ท่าที่สอนเองทับได้
    driver = driver_for(overrides={"gestures": {"knob": False}})
    driver.hold(make_pose_hand(ROCK, gesture="Pointing_Up", score=0.95), 800)
    assert actions(driver) == ["pose:rock"]


def test_knob_and_thumb_down_reserve_their_gestures():
    reserved = reserved_gestures(Config())
    assert {"Pointing_Up", "Thumb_Down"} <= reserved
    off = parse_config(
        {
            "gestures": {"knob": False},
            "actions": {"thumb_down_hold": "none"},
        }
    )
    assert not {"Pointing_Up", "Thumb_Down"} & reserved_gestures(off)
    knob_unbound = parse_config({"actions": {"knob_right": "none", "knob_left": "none"}})
    assert "Pointing_Up" not in reserved_gestures(knob_unbound)


def test_thumb_up_is_reserved_only_while_the_launcher_is_on():
    assert "Thumb_Up" in reserved_gestures(Config())
    off = parse_config({"launcher": {"enabled": False}})
    assert "Thumb_Up" not in reserved_gestures(off)
    assert {"Victory", "Open_Palm", "Closed_Fist", "ILoveYou"} <= reserved_gestures(off)


def test_poses_do_nothing_while_disarmed():
    driver = driver_for(armed=False)
    driver.hold(make_pose_hand(ROCK), 1500)
    assert driver.events == []
    assert driver.engine.state is State.DISARMED
    # แต่ยังเห็นว่าจำท่าได้ (ใช้ลองท่าแบบปลอดภัยในหน้าต่าง debug)
    assert driver.engine.snapshot.gesture == "pose:rock"


def test_pose_outside_the_active_region_does_not_start():
    driver = driver_for()
    driver.hold(make_pose_hand(ROCK, 0.5, 0.98), 1000)  # มือต่ำจนโคนนิ้วอยู่ใต้ active region
    assert actions(driver) == []


def test_pose_does_not_start_while_fingers_are_about_to_pinch():
    # ท่าที่นิ้วโป้งเกือบแตะนิ้วชี้: การคลิกต้องมาก่อน ท่านี้จึงเริ่มนับไม่ได้
    near = make_hand(pinch=0.30)
    data = [pose_features(near, WIDTH, HEIGHT)] * 20
    lib = PoseLibrary().with_pose(make_template("ok", "f5", data))
    driver = driver_for(lib)
    driver.hold(near, 1000)
    assert driver.engine.state is State.POINTER
    assert actions(driver) == []


def test_two_taught_poses_fire_their_own_actions():
    lib = taught(("rock", ROCK), ("shaka", "10001"))
    driver = driver_for(lib)
    driver.hold(make_pose_hand("10001"), 800)
    driver.hold(make_pose_hand("11111"), 900)
    driver.hold(make_pose_hand(ROCK), 800)
    assert actions(driver) == ["pose:shaka", "pose:rock"]


def test_engine_without_poses_never_reports_one():
    engine = GestureEngine(Config(), FHD)
    driver = Driver(engine)
    engine.set_armed(True)
    driver.hold(make_pose_hand(ROCK), 1000)
    assert engine.snapshot.pose is None
    assert actions(driver) == []


def test_empty_library_is_the_same_as_no_poses():
    engine = GestureEngine(Config(), FHD, poses=PoseClassifier(PoseLibrary()))
    Driver(engine).hold(make_pose_hand(ROCK), 300)
    assert engine.snapshot.pose is None


@pytest.mark.parametrize("fingers", ["11111", "00000", "10000"])
def test_other_shapes_never_fire_a_taught_pose(fingers):
    driver = driver_for()
    driver.hold(make_pose_hand(fingers), 1500)
    assert [a for a in actions(driver) if a.startswith("pose:")] == []
