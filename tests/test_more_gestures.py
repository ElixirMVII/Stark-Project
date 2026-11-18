"""ท่ามือชุดใหม่: จีบโป้ง+นาง (คลิกกลาง), จีบโป้ง+ก้อย, 👎 ค้าง, ☝️ ปุ่มหมุน + ปุ่มสื่อ/เสียง และ HUD"""

from __future__ import annotations

import dataclasses
import types

import pytest
from handsim import PINCHED, Driver, make_hand
from posesim import make_pose_hand
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QImage, QPainter
from test_actions import RecordingBackend

from aether.actions import _PYNPUT_KEY_NAMES, ActionExecutor, PynputBackend
from aether.config import Config, parse_config
from aether.gestures import (
    ACTION_KNOB_LEFT,
    ACTION_KNOB_RIGHT,
    ACTION_PINKY,
    CHARGE_KNOB,
    CHARGE_THUMB,
    EngineSnapshot,
    EventKind,
    GestureEngine,
    GestureEvent,
    State,
)
from aether.hotkeys import MEDIA_KEYS, Hotkey, parse_action
from aether.hud.radial import ChargeRing, KnobDial, action_label, knob_label
from aether.hud.reticle import reticle_mode
from aether.hud.theme import Theme
from aether.mapping import ScreenGeometry
from aether.sources import KEY_GESTURES, MouseHandModel, key_name

FHD = ScreenGeometry(0, 0, 1920, 1080)


def armed(overrides: dict | None = None, hand=None) -> Driver:
    config = parse_config(overrides) if overrides else Config()
    driver = Driver(GestureEngine(config, FHD))
    driver.engine.set_armed(True)
    driver.hold(hand or make_hand(), 200)
    assert driver.engine.state is State.POINTER
    driver.clear()
    return driver


def of_kind(events, kind: EventKind) -> list[GestureEvent]:
    return [e for e in events if e.kind is kind]


def action_names(events) -> list[str]:
    return [e.name for e in of_kind(events, EventKind.ACTION)]


# ---- จีบโป้ง+นาง = คลิกปุ่มกลาง ------------------------------------------- #


def test_thumb_ring_pinch_is_a_middle_click():
    driver = armed()
    events = driver.hold(make_hand(ring_pinch=PINCHED), 100)
    clicks = of_kind(events, EventKind.CLICK)
    assert [(c.button, c.count) for c in clicks] == [("middle", 1)]
    driver.hold(make_hand(ring_pinch=PINCHED), 300)  # ค้างต่อ: ไม่คลิกซ้ำ ไม่กลายเป็นลาก
    assert len(of_kind(driver.events, EventKind.CLICK)) == 1
    assert not of_kind(driver.events, EventKind.BUTTON_DOWN)
    driver.hold(make_hand(), 100)
    driver.hold(make_hand(ring_pinch=PINCHED), 100)
    assert len(of_kind(driver.events, EventKind.CLICK)) == 2


def test_middle_click_can_be_turned_off():
    driver = armed({"gestures": {"middle_click": False}})
    driver.hold(make_hand(ring_pinch=PINCHED), 300)
    assert not of_kind(driver.events, EventKind.CLICK)


def test_closest_finger_wins_when_two_touch_the_thumb():
    driver = armed()
    driver.hold(make_hand(middle_pinch=0.2, ring_pinch=0.1), 100)
    assert [c.button for c in of_kind(driver.events, EventKind.CLICK)] == ["middle"]


def test_a_fist_is_never_a_ring_or_pinky_pinch():
    driver = armed()
    driver.hold(make_hand(fist=True, ring_pinch=PINCHED, pinky_pinch=PINCHED), 300)
    assert not of_kind(driver.events, EventKind.CLICK)
    assert ACTION_PINKY not in action_names(driver.events)


# ---- จีบโป้ง+ก้อย = action ---------------------------------------------- #


def test_thumb_pinky_pinch_runs_its_action_once_per_pinch():
    driver = armed()
    events = driver.hold(make_hand(pinky_pinch=PINCHED), 300)
    assert action_names(events) == [ACTION_PINKY]
    driver.hold(make_hand(), 100)
    driver.hold(make_hand(pinky_pinch=PINCHED), 100)
    assert action_names(driver.events) == [ACTION_PINKY, ACTION_PINKY]


def test_unbound_pinky_pinch_does_nothing():
    driver = armed({"actions": {"pinky_pinch": "none"}})
    driver.hold(make_hand(pinky_pinch=PINCHED), 300)
    assert driver.events == [] or not action_names(driver.events)
    assert not of_kind(driver.events, EventKind.CLICK)


# ---- 👎 ค้าง -------------------------------------------------------------- #


def test_thumb_down_hold_runs_its_action_once():
    driver = armed()
    hand = make_hand(gesture="Thumb_Down", score=0.9)
    driver.hold(hand, 300)
    assert driver.engine.state is State.CHARGING
    assert driver.engine.snapshot.charge_kind == CHARGE_THUMB
    driver.hold(hand, 800)
    assert action_names(driver.events) == [CHARGE_THUMB]
    driver.hold(hand, 1500)  # ค้างต่อไม่ติดซ้ำ
    assert action_names(driver.events) == [CHARGE_THUMB]


def test_thumb_down_does_nothing_when_unbound():
    driver = armed({"actions": {"thumb_down_hold": "none"}})
    driver.hold(make_hand(gesture="Thumb_Down", score=0.9), 1500)
    assert action_names(driver.events) == [] and driver.engine.state is State.POINTER


# ---- ☝️ ปุ่มหมุน ------------------------------------------------------------ #


def point(angle: float = 0.0, x: float = 0.5, y: float = 0.62):
    return make_pose_hand("01000", x, y, angle=angle, gesture="Pointing_Up", score=0.9)


def knob_driver(overrides: dict | None = None) -> Driver:
    driver = armed(overrides, hand=make_pose_hand("11111", 0.5, 0.62))
    driver.hold(point(), 700)
    return driver


def test_still_pointing_up_becomes_a_knob_and_turning_steps():
    driver = knob_driver()
    assert driver.engine.state is State.KNOB
    assert driver.engine.snapshot.knob == pytest.approx(0.0, abs=1.0)
    driver.animate(lambda p: point(40.0 * p), 400)  # หมุนตามเข็ม 40° = 2 step (ทีละ 18°)
    assert action_names(driver.events) == [ACTION_KNOB_RIGHT, ACTION_KNOB_RIGHT]
    assert driver.engine.snapshot.knob == pytest.approx(40.0, abs=1.5)
    driver.clear()
    driver.animate(lambda p: point(40.0 - 80.0 * p), 600)  # ย้อนไป −40°
    assert action_names(driver.events) == [ACTION_KNOB_LEFT] * 4
    assert not of_kind(driver.events, EventKind.MOVE)  # เคอร์เซอร์หยุดระหว่างหมุน


def test_knob_releases_when_the_hand_stops_pointing():
    driver = knob_driver()
    driver.hold(make_pose_hand("11111", 0.5, 0.62), 300)
    assert driver.engine.state is not State.KNOB
    assert driver.engine.snapshot.knob is None


def test_pointing_up_while_moving_keeps_the_cursor_moving():
    driver = armed(hand=make_pose_hand("11111", 0.3, 0.62))
    driver.animate(lambda p: point(x=0.3 + 0.3 * p), 900)
    assert State.KNOB not in driver.states
    assert len(of_kind(driver.events, EventKind.MOVE)) > 10


def test_knob_charge_shows_on_the_hud_before_it_engages():
    driver = armed(hand=make_pose_hand("11111", 0.5, 0.62))
    driver.hold(point(), 330)
    snapshot = driver.engine.snapshot
    assert snapshot.state is State.POINTER
    assert snapshot.charge_kind == CHARGE_KNOB and 0.0 < snapshot.charge < 1.0


def test_knob_can_be_turned_off():
    driver = knob_driver({"gestures": {"knob": False}})
    assert State.KNOB not in driver.states


def test_reticle_shows_the_spread_ring_while_turning():
    assert reticle_mode(EngineSnapshot(state=State.KNOB, armed=True)) == "GESTURE"


# ---- ปุ่มสื่อ/เสียง และการกดจริง --------------------------------------------- #


@pytest.mark.parametrize(
    ("name", "vk"),
    [
        ("volume_mute", 0xAD),
        ("volume_down", 0xAE),
        ("volume_up", 0xAF),
        ("media_next", 0xB0),
        ("media_prev", 0xB1),
        ("media_stop", 0xB2),
        ("media_play_pause", 0xB3),
    ],
)
def test_media_keys_parse_as_single_key_actions(name, vk):
    action = parse_action(name)
    assert isinstance(action, Hotkey) and action.keys == (name,) and action.vk == vk
    assert name in MEDIA_KEYS


def test_executor_presses_media_keys_and_middle_button():
    backend = RecordingBackend()
    config = Config()
    executor = ActionExecutor(backend, actions=dataclasses.asdict(config.actions))
    executor.execute(
        (
            GestureEvent(EventKind.ACTION, name=CHARGE_THUMB),
            GestureEvent(EventKind.ACTION, name=ACTION_KNOB_RIGHT),
            GestureEvent(EventKind.ACTION, name=ACTION_PINKY),
            GestureEvent(EventKind.CLICK, x=5, y=6, button="middle"),
        )
    )
    assert backend.calls[:4] == [
        ("key_down", "media_play_pause"),
        ("key_up", "media_play_pause"),
        ("key_down", "volume_up"),
        ("key_up", "volume_up"),
    ]
    assert ("key_down", "win") in backend.calls and ("key_down", "s") in backend.calls
    assert backend.calls[-3:] == [("move", 5, 6), ("press", "middle"), ("release", "middle")]


def test_pynput_backend_maps_media_key_names():
    keys = types.SimpleNamespace(
        media_volume_up="VOL+", media_play_pause="PLAY", media_previous="PREV", media_next="NEXT"
    )

    class Key:
        def __class_getitem__(cls, name):
            try:
                return getattr(keys, name)
            except AttributeError:
                raise KeyError(name) from None

    module = types.SimpleNamespace(
        Key=Key, KeyCode=types.SimpleNamespace(from_vk=lambda vk: ("vk", vk))
    )
    backend = PynputBackend.__new__(PynputBackend)
    backend._keyboard_module = module
    backend._modifiers = {}
    assert backend._key("volume_up") == "VOL+"
    assert backend._key("media_prev") == "PREV"
    assert backend._key("media_play_pause") == "PLAY"
    assert backend._key("media_stop") == ("vk", 0xB2)  # pynput ไม่มีชื่อนี้: ส่ง virtual-key code
    assert set(_PYNPUT_KEY_NAMES) <= MEDIA_KEYS


# ---- เมาส์จำลอง --------------------------------------------------------------- #


def test_mouse_sim_middle_button_and_new_gesture_keys():
    config = Config()
    model = MouseHandModel(config, FHD)
    model.press("middle")
    engine = GestureEngine(config, FHD)
    engine.set_armed(True)
    events: list[GestureEvent] = []
    for i in range(12):
        _, batch = engine.update(model.frame(1000 + i * 33, (960, 540)))
        events += batch
    assert [e.button for e in events if e.kind is EventKind.CLICK] == []  # ยังไม่เคยเห็นนิ้วห่าง
    model.release("middle")
    for i in range(6):
        engine.update(model.frame(1500 + i * 33, (960, 540)))
    model.press("middle")
    for i in range(6):
        _, batch = engine.update(model.frame(1800 + i * 33, (960, 540)))
        events += batch
    assert [e.button for e in events if e.kind is EventKind.CLICK] == ["middle"]
    assert KEY_GESTURES["6"] == "Thumb_Down" and KEY_GESTURES["7"] == "Pointing_Up"
    assert key_name(types.SimpleNamespace(vk=0x37, name=None, char=None)) == "7"


# ---- HUD -------------------------------------------------------------------- #


@pytest.fixture
def theme(qapp) -> Theme:
    return Theme.from_config(Config().hud, None)


def test_action_labels_for_new_defaults():
    assert action_label("media_play_pause") == "Play/Pause"
    assert action_label("win+shift+s") == "Snip"
    assert action_label("volume_up") == "Volume up"
    assert action_label("media_stop") == "Stop"
    assert knob_label(dataclasses.asdict(Config().actions)) == "Volume"
    assert knob_label({"knob_right": "ctrl+tab", "knob_left": "ctrl+shift+tab"}) == (
        "Ctrl+Shift+Tab / Ctrl+Tab"
    )


def test_charge_ring_labels_thumb_down_and_knob(theme):
    ring = ChargeRing(theme, dataclasses.asdict(Config().actions))
    ring.set_state(QPointF(100, 100), EngineSnapshot(charge=0.5, charge_kind=CHARGE_THUMB))
    assert ring.label() == "Play/Pause"
    ring.set_state(QPointF(100, 100), EngineSnapshot(charge=0.5, charge_kind=CHARGE_KNOB))
    assert ring.label() == "Volume"


def test_knob_dial_follows_the_angle_and_flashes_steps(theme):
    dial = KnobDial(theme, dataclasses.asdict(Config().actions))
    center = QPointF(300, 200)
    assert dial.set_state(center, EngineSnapshot(state=State.KNOB, armed=True, knob=30.0))
    dial.step(1, 0.0)
    for i in range(12):
        dial.advance(i / 60)
    assert dial.alpha == 1.0 and not dial.rect().isEmpty()
    image = QImage(600, 400, QImage.Format.Format_ARGB32)
    image.fill(Qt.GlobalColor.transparent)
    painter = QPainter(image)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    dial.paint(painter, 0.1)
    painter.end()
    rect = dial.rect().toAlignedRect()
    lit_outside = sum(
        1
        for x in range(0, 600, 3)
        for y in range(0, 400, 3)
        if image.pixelColor(x, y).alpha() and not rect.adjusted(-2, -2, 2, 2).contains(x, y)
    )
    assert lit_outside == 0
    assert dial.set_state(center, EngineSnapshot(state=State.POINTER, armed=True))
    for i in range(30):
        dial.advance(1.0 + i / 60)
    assert dial.alpha == 0.0 and dial.rect().isEmpty()
