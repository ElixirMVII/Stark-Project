from __future__ import annotations

import logging

import pytest

from aether import actions
from aether.actions import ActionExecutor, DryRunBackend, create_backend, double_click_time_ms
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


@pytest.fixture
def backend() -> RecordingBackend:
    return RecordingBackend()


@pytest.fixture
def executor(backend) -> ActionExecutor:
    return ActionExecutor(backend)


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
