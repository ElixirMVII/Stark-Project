from __future__ import annotations

import dataclasses
import logging
import sys

import pytest

from aether import actions
from aether.actions import (
    PROTECTED_WINDOW_CLASSES,
    ActionExecutor,
    DryRunBackend,
    create_backend,
    double_click_time_ms,
    foreground_window,
    minimize_foreground_window,
    should_minimize,
)
from aether.config import Config
from aether.gestures import EventKind, GestureEvent


class RecordingBackend:
    """จด input ที่ถูกส่ง แทนการส่งจริง"""

    def __init__(self) -> None:
        self.calls: list[tuple] = []
        self.fail_on: set[str] = set()

    def _record(self, name: str, *args) -> None:
        if name in self.fail_on:
            raise OSError(f"{name} failed (test)")
        self.calls.append((name, *args))

    def move(self, x: int, y: int) -> None:
        self._record("move", x, y)

    def press(self, button: str) -> None:
        self._record("press", button)

    def release(self, button: str) -> None:
        self._record("release", button)

    def scroll(self, dx: int, dy: int) -> None:
        self._record("scroll", dx, dy)

    def key_down(self, key: str) -> None:
        self._record("key_down", key)

    def key_up(self, key: str) -> None:
        self._record("key_up", key)

    def minimize_foreground(self) -> bool:
        self._record("minimize")
        return True

    def launch(self, target: str, args: str) -> None:
        self._record("launch", target, args)


DEFAULT_ACTIONS = dataclasses.asdict(Config().actions)


@pytest.fixture
def backend() -> RecordingBackend:
    return RecordingBackend()


@pytest.fixture
def executor(backend) -> ActionExecutor:
    return ActionExecutor(backend, DEFAULT_ACTIONS)


def test_move(executor, backend):
    executor.execute([GestureEvent(EventKind.MOVE, x=10, y=20)])
    assert backend.calls == [("move", 10, 20)]


def test_click_moves_then_presses_and_releases(executor, backend):
    executor.execute([GestureEvent(EventKind.CLICK, x=5, y=6, button="right")])
    assert backend.calls == [("move", 5, 6), ("press", "right"), ("release", "right")]
    assert executor.pressed == frozenset()


def test_double_click_event_is_a_second_click_at_given_position(executor, backend):
    executor.execute(
        [
            GestureEvent(EventKind.CLICK, x=100, y=100, count=1),
            GestureEvent(EventKind.MOVE, x=104, y=101),
            GestureEvent(EventKind.CLICK, x=100, y=100, count=2),
        ]
    )
    presses = [c for c in backend.calls if c[0] == "press"]
    assert len(presses) == 2  # OS เห็นสองคลิกที่จุดเดียวกัน → ดับเบิลคลิก
    assert backend.calls[-3:] == [("move", 100, 100), ("press", "left"), ("release", "left")]


def test_drag_sequence_tracks_pressed_button(executor, backend):
    executor.execute([GestureEvent(EventKind.BUTTON_DOWN, x=1, y=2)])
    assert executor.pressed == frozenset({"left"})
    executor.execute([GestureEvent(EventKind.MOVE, x=50, y=60)])
    executor.execute([GestureEvent(EventKind.BUTTON_UP, x=50, y=60)])
    assert executor.pressed == frozenset()
    assert backend.calls == [
        ("move", 1, 2),
        ("press", "left"),
        ("move", 50, 60),
        ("release", "left"),
    ]


def test_duplicate_down_and_stray_up_are_ignored(executor, backend):
    executor.execute([GestureEvent(EventKind.BUTTON_UP)])  # ไม่ได้กดอยู่
    assert backend.calls == []
    executor.execute([GestureEvent(EventKind.BUTTON_DOWN), GestureEvent(EventKind.BUTTON_DOWN)])
    assert [c for c in backend.calls if c[0] == "press"] == [("press", "left")]


def test_scroll(executor, backend):
    executor.execute([GestureEvent(EventKind.SCROLL, dy=3), GestureEvent(EventKind.SCROLL, dx=-2)])
    assert backend.calls == [("scroll", 0, 3), ("scroll", -2, 0)]


def test_release_all_releases_everything_and_is_idempotent(executor, backend):
    executor.execute([GestureEvent(EventKind.BUTTON_DOWN, button="left")])
    executor.execute([GestureEvent(EventKind.BUTTON_DOWN, button="right")])
    executor.release_all()
    assert executor.pressed == frozenset()
    assert sorted(c for c in backend.calls if c[0] == "release") == [
        ("release", "left"),
        ("release", "right"),
    ]
    count = len(backend.calls)
    executor.release_all()
    assert len(backend.calls) == count


def test_disarmed_event_releases_held_buttons(executor, backend):
    executor.execute([GestureEvent(EventKind.BUTTON_DOWN)])
    executor.execute([GestureEvent(EventKind.DISARMED)])
    assert executor.pressed == frozenset()
    assert backend.calls[-1] == ("release", "left")


def test_armed_event_sends_no_input(executor, backend):
    executor.execute([GestureEvent(EventKind.ARMED)])
    assert backend.calls == []


def test_backend_error_does_not_stop_following_events(executor, backend, caplog):
    backend.fail_on = {"scroll"}
    with caplog.at_level(logging.ERROR):
        executor.execute(
            [GestureEvent(EventKind.SCROLL, dy=1), GestureEvent(EventKind.MOVE, x=1, y=1)]
        )
    assert backend.calls == [("move", 1, 1)]
    assert "input backend failed" in caplog.text


def test_failed_release_still_clears_pressed_state(executor, backend):
    executor.execute([GestureEvent(EventKind.BUTTON_DOWN)])
    backend.fail_on = {"release"}
    executor.release_all()  # ต้องไม่ raise และไม่ถือว่ายังกดค้าง
    assert executor.pressed == frozenset()


def test_press_failure_during_click_can_be_cleaned_up(executor, backend):
    backend.fail_on = {"press"}
    executor.execute([GestureEvent(EventKind.CLICK, x=1, y=1)])
    backend.fail_on = set()
    executor.release_all()  # เผื่อ OS รับ press ไปแล้วครึ่งทาง: ต้องส่ง release ตาม
    assert ("release", "left") in backend.calls
    assert executor.pressed == frozenset()


def test_dry_run_backend_only_logs(caplog):
    executor = ActionExecutor(DryRunBackend())
    with caplog.at_level(logging.DEBUG, logger="aether.actions"):
        executor.execute(
            [
                GestureEvent(EventKind.MOVE, x=3, y=4),
                GestureEvent(EventKind.CLICK, x=3, y=4),
                GestureEvent(EventKind.SCROLL, dy=-2),
            ]
        )
    assert "[dry-run] move → (3, 4)" in caplog.text
    assert "[dry-run] left button DOWN" in caplog.text
    assert "[dry-run] left button UP" in caplog.text
    assert "[dry-run] scroll dx=+0 dy=-2" in caplog.text


def test_create_backend_dry_run():
    backend, dry = create_backend(dry_run=True)
    assert isinstance(backend, DryRunBackend) and dry is True


def test_create_backend_falls_back_to_dry_run_when_pynput_is_unusable(monkeypatch, caplog):
    def broken():
        raise ImportError("no display (test)")

    monkeypatch.setattr(actions, "PynputBackend", broken)
    with caplog.at_level(logging.ERROR):
        backend, dry = create_backend(dry_run=False)
    assert isinstance(backend, DryRunBackend) and dry is True
    assert "dry-run" in caplog.text


def test_double_click_time_has_sane_value():
    assert double_click_time_ms(default=432) > 0  # นอก Windows = ค่า default


# ---- M3: ท่าพิเศษ → ปุ่มลัด / ย่อหน้าต่าง / ซูม ------------------------------ #


def _action(name: str) -> GestureEvent:
    return GestureEvent(EventKind.ACTION, name=name)


@pytest.mark.parametrize(
    ("name", "keys"),
    [
        ("swipe_left", ["ctrl", "win", "left"]),  # สลับ virtual desktop
        ("swipe_right", ["ctrl", "win", "right"]),
        ("swipe_up", ["win", "tab"]),  # Task View
        ("swipe_down", ["win", "d"]),  # โชว์ desktop
    ],
)
def test_default_swipe_actions_press_and_release_shortcut(executor, backend, name, keys):
    executor.execute([_action(name)])
    downs = [(c[0], c[1]) for c in backend.calls]
    assert downs == [("key_down", k) for k in keys] + [("key_up", k) for k in reversed(keys)]
    assert executor.keys_down == ()


def test_fist_hold_minimizes_foreground_window(executor, backend):
    executor.execute([_action("fist_hold")])
    assert backend.calls == [("minimize",)]


def test_actions_are_remappable_from_config(backend):
    executor = ActionExecutor(
        backend,
        {"swipe_left": "alt+tab", "swipe_up": "f11", "fist_hold": "none", "swipe_down": "win+D"},
    )
    executor.execute([_action("swipe_left")])
    assert backend.calls == [
        ("key_down", "alt"), ("key_down", "tab"), ("key_up", "tab"), ("key_up", "alt"),
    ]  # fmt: skip
    backend.calls.clear()
    executor.execute([_action("swipe_up")])
    assert backend.calls == [("key_down", "f11"), ("key_up", "f11")]
    backend.calls.clear()
    executor.execute([_action("fist_hold"), _action("unknown_gesture")])  # "none" / ไม่ได้ผูก
    assert backend.calls == []
    executor.execute([_action("swipe_down")])
    assert ("key_down", "d") in backend.calls


def test_invalid_action_spec_is_rejected_up_front(backend):
    with pytest.raises(ValueError):
        ActionExecutor(backend, {"swipe_left": "ctrl+nope"})


def test_zoom_is_ctrl_plus_wheel(executor, backend):
    executor.execute([GestureEvent(EventKind.ZOOM, x=5, y=5, dy=2)])
    executor.execute([GestureEvent(EventKind.ZOOM, x=5, y=5, dy=-1)])
    assert backend.calls == [
        ("key_down", "ctrl"), ("scroll", 0, 2), ("key_up", "ctrl"),
        ("key_down", "ctrl"), ("scroll", 0, -1), ("key_up", "ctrl"),
    ]  # fmt: skip


def test_modifier_is_released_even_if_scroll_fails(executor, backend):
    backend.fail_on = {"scroll"}
    executor.execute([GestureEvent(EventKind.ZOOM, dy=1)])
    assert backend.calls == [("key_down", "ctrl"), ("key_up", "ctrl")]
    assert executor.keys_down == ()


def test_keys_pressed_before_a_failure_are_released(executor, backend):
    """กดปุ่มลัดไปครึ่งทางแล้ว backend พัง: ปุ่มที่กดไปแล้วต้องถูกปล่อย (ไม่งั้น Ctrl/Win จะค้าง)"""

    class FailOnThirdKey(RecordingBackend):
        def key_down(self, key: str) -> None:
            if key == "left":
                raise OSError("key failed (test)")
            super().key_down(key)

    flaky = FailOnThirdKey()
    executor = ActionExecutor(flaky, DEFAULT_ACTIONS)
    executor.execute([_action("swipe_left")])
    ups = [c[1] for c in flaky.calls if c[0] == "key_up"]
    assert set(ups) >= {"ctrl", "win"}
    assert executor.keys_down == ()


def test_release_all_releases_held_keys_too(executor, backend):
    executor._keys_down.extend(["ctrl", "win"])  # จำลองปุ่มค้างจาก error กลางทาง
    executor.release_all()
    assert [c for c in backend.calls if c[0] == "key_up"] == [("key_up", "win"), ("key_up", "ctrl")]
    assert executor.keys_down == ()


def test_dry_run_backend_logs_keys_and_minimize(caplog):
    executor = ActionExecutor(DryRunBackend(), DEFAULT_ACTIONS)
    with caplog.at_level(logging.INFO, logger="aether.actions"):
        executor.execute([_action("swipe_up"), _action("fist_hold")])
    assert "[dry-run] key win DOWN" in caplog.text and "[dry-run] key tab UP" in caplog.text
    assert "[dry-run] minimize foreground window" in caplog.text


def test_desktop_and_taskbar_are_never_minimized():
    for protected in ("Progman", "WorkerW", "Shell_TrayWnd"):
        assert protected in PROTECTED_WINDOW_CLASSES
        assert not should_minimize(protected)
    assert should_minimize("Notepad") and should_minimize("Chrome_WidgetWin_1")
    assert not should_minimize("")  # อ่าน class ไม่ได้ → ไม่ย่อ


@pytest.mark.skipif(sys.platform == "win32", reason="นอก Windows เท่านั้น")
def test_minimize_is_a_no_op_outside_windows():
    assert minimize_foreground_window() is False
    assert foreground_window() is None
