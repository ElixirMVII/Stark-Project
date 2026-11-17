"""Target Lock (M6.1) — กติกาการเลือกปุ่มและ thread ของ finder ด้วย backend ปลอม (ไม่ต้องมี Windows)"""

from __future__ import annotations

import threading
import time

import pytest

from aether.config import TargetingConfig
from aether.snap import Target
from aether.targeting import (
    ElementInfo,
    ForegroundInfo,
    TargetFinder,
    TargetResolver,
    WindowInfo,
)

OWN = 1000
APP = 2000


class FakeBackend:
    def __init__(self) -> None:
        self.chain: list[ElementInfo] = [
            ElementInfo("TextControl", "label", (110, 110, 130, 130), process_id=APP),
            ElementInfo("ButtonControl", "Save", (100, 100, 180, 140), process_id=APP),
            ElementInfo("PaneControl", "", (0, 0, 800, 600), process_id=APP),
        ]
        self.window = WindowInfo(APP, "notepad.exe")
        self.fg: ForegroundInfo | None = ForegroundInfo(APP, "notepad.exe", fullscreen=False)
        self.calls = 0
        self.delay = 0.0
        self.error: Exception | None = None

    def window_at(self, x, y):
        return self.window

    def foreground(self):
        return self.fg

    def element_chain(self, x, y, depth):
        self.calls += 1
        if self.error:
            raise self.error
        if self.delay:
            clock[0] += self.delay  # เวลาผ่านไประหว่าง call (นาฬิกาปลอม)
        return self.chain[: depth + 1]


clock = [0.0]


@pytest.fixture(autouse=True)
def _reset_clock():
    clock[0] = 0.0


def resolver(backend: FakeBackend, **config) -> TargetResolver:
    return TargetResolver(backend, TargetingConfig(**config), own_pid=OWN, clock=lambda: clock[0])


def test_walks_up_to_the_clickable_ancestor():
    target = resolver(FakeBackend()).resolve(120, 120)
    assert target == Target(100, 100, 80, 40, "Button", "Save", 0.0)


def test_nothing_clickable_within_three_levels():
    backend = FakeBackend()
    backend.chain = [ElementInfo("TextControl", "", (0, 0, 10, 10), process_id=APP)] * 4 + [
        ElementInfo("ButtonControl", "deep", (0, 0, 50, 50), process_id=APP)
    ]
    assert resolver(backend).resolve(5, 5) is None


@pytest.mark.parametrize(
    "element",
    [
        ElementInfo("ButtonControl", "", (100, 100, 180, 140), enabled=False, process_id=APP),
        ElementInfo("ButtonControl", "", (100, 100, 180, 140), offscreen=True, process_id=APP),
        ElementInfo("ButtonControl", "", (100, 100, 105, 140), process_id=APP),  # แคบเกิน 8 px
        ElementInfo("ListItemControl", "", (0, 0, 900, 100), process_id=APP),  # กว้างเกิน 480
        ElementInfo("ListItemControl", "", (0, 0, 300, 200), process_id=APP),  # สูงเกิน 160
        ElementInfo("ButtonControl", "", (100, 100, 180, 140), process_id=OWN),  # ของ AETHER เอง
    ],
)
def test_filtered_elements(element):
    backend = FakeBackend()
    backend.chain = [element]
    assert resolver(backend).resolve(120, 120) is None


def test_max_size_is_logical_px_scaled_by_dpi():
    backend = FakeBackend()
    backend.chain = [ElementInfo("ListItemControl", "", (0, 0, 600, 100), process_id=APP)]
    assert resolver(backend).resolve(10, 10) is None  # 600 > 480 ที่ 100%
    scaled = TargetResolver(
        backend, TargetingConfig(), scale=1.5, own_pid=OWN, clock=lambda: clock[0]
    )
    assert scaled.resolve(10, 10) is not None  # 600 ≤ 720 physical px ที่ 150%


def test_all_clickable_types_are_recognised():
    backend = FakeBackend()
    for ui_type, label in (
        ("MenuItemControl", "MenuItem"),
        ("TabItemControl", "TabItem"),
        ("HyperlinkControl", "Hyperlink"),
        ("CheckBoxControl", "CheckBox"),
        ("RadioButtonControl", "RadioButton"),
        ("TreeItemControl", "TreeItem"),
        ("ComboBoxControl", "ComboBox"),
        ("SplitButtonControl", "SplitButton"),
    ):
        backend.chain = [ElementInfo(ui_type, "x", (0, 0, 40, 20), process_id=APP)]
        clock[0] += 1.0  # ไม่ใช้ cache
        target = resolver(backend).resolve(5, 5)
        assert target is not None and target.control_type == label


def test_cache_reuses_the_target_while_inside_it():
    backend = FakeBackend()
    r = resolver(backend)
    first = r.resolve(120, 120)
    clock[0] += 0.1
    assert r.resolve(150, 130) is first and backend.calls == 1  # ยังอยู่ในปุ่มเดิม
    clock[0] += 0.5  # cache หมดอายุ
    r.resolve(150, 130)
    assert backend.calls == 2
    r.resolve(500, 500)  # ออกนอกปุ่ม → ถามใหม่
    assert backend.calls == 3


def test_slow_call_is_discarded_and_the_app_is_paused():
    backend = FakeBackend()
    backend.delay = 0.3
    r = resolver(backend)
    assert r.resolve(120, 120) is None and r.suspended_reason == "slow"
    backend.delay = 0.0
    clock[0] += 1.0
    assert r.resolve(120, 120) is None and backend.calls == 1  # พักอยู่ ไม่ถามเลย
    clock[0] += 5.0
    assert r.resolve(120, 120) is not None


def test_uia_error_pauses_the_app():
    backend = FakeBackend()
    backend.error = PermissionError("access denied")  # เช่นหน้าต่างที่รันแบบ admin
    r = resolver(backend)
    assert r.resolve(120, 120) is None and r.suspended_reason == "unavailable"
    backend.error = None
    clock[0] += 1.0
    assert r.resolve(120, 120) is None


def test_hung_window_is_skipped_without_calling_uia():
    backend = FakeBackend()
    backend.window = WindowInfo(APP, "frozen.exe", hung=True)
    r = resolver(backend)
    assert r.resolve(120, 120) is None and backend.calls == 0
    assert r.suspended_reason == "not responding"


def test_blocklisted_app_is_skipped():
    backend = FakeBackend()
    backend.window = WindowInfo(APP, "Game.EXE")
    r = resolver(backend, app_blocklist=("game.exe",))
    assert r.resolve(120, 120) is None and backend.calls == 0


def test_fullscreen_foreground_suspends_except_explorer():
    backend = FakeBackend()
    backend.fg = ForegroundInfo(APP, "game.exe", fullscreen=True)
    r = resolver(backend)
    assert r.resolve(120, 120) is None and r.suspended_reason == "fullscreen"
    backend.fg = ForegroundInfo(APP, "explorer.exe", fullscreen=True)  # desktop เต็มจอ = ปกติ
    assert r.resolve(120, 120) is not None


def test_own_windows_are_ignored():
    backend = FakeBackend()
    backend.window = WindowInfo(OWN, "aether.exe")
    assert resolver(backend).resolve(120, 120) is None and backend.calls == 0


# ---- finder thread ------------------------------------------------------ #


def wait_until(condition, timeout=2.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.005)
    return False


def test_finder_publishes_targets_and_clears_when_inactive():
    published: list[Target | None] = []
    position = [(120, 120)]
    active = threading.Event()
    active.set()
    finder = TargetFinder(
        TargetingConfig(rate_hz=30),
        position=lambda: position[0],
        publish=published.append,
        active=active.is_set,
        backend_factory=FakeBackend,
    )
    finder.start()
    try:
        assert wait_until(lambda: published and published[-1] is not None)
        assert published[-1].control_type == "Button" and finder.available
        count = len(published)
        time.sleep(0.1)
        assert len(published) == count  # ผลเดิม: ไม่ส่งซ้ำ
        active.clear()  # DISARM
        assert wait_until(lambda: published[-1] is None)
    finally:
        finder.stop()


def test_finder_without_uia_disables_itself():
    def broken():
        raise ImportError("No module named 'uiautomation'")

    published: list = []
    finder = TargetFinder(
        TargetingConfig(), position=lambda: (0, 0), publish=published.append, backend_factory=broken
    )
    finder.start()
    finder.stop()
    assert finder.available is False and published == []


def test_finder_survives_backend_exceptions():
    class Flaky(FakeBackend):
        def foreground(self):
            raise OSError("boom")

    published: list = []
    finder = TargetFinder(
        TargetingConfig(rate_hz=60),
        position=lambda: (120, 120),
        publish=published.append,
        backend_factory=Flaky,
    )
    finder.start()
    try:
        assert wait_until(lambda: finder.queries >= 3)  # ยังวนต่อ ไม่ตาย
    finally:
        finder.stop()
    assert all(p is None for p in published)


# ---- app: finder → pipeline → engine → HUD -------------------------------- #


def test_app_wires_the_finder_into_the_engine(qapp):
    from handsim import make_hand
    from test_actions import RecordingBackend
    from test_app import BlankCamera, ScriptedTracker
    from test_hotkeys import FakeApi
    from test_m4 import wait

    from aether.app import AetherApp, Options
    from aether.config import parse_config
    from aether.hotkeys import HotkeyManager
    from aether.mapping import ScreenGeometry

    class CenterButton(FakeBackend):
        def __init__(self) -> None:
            super().__init__()
            self.chain = [ElementInfo("ButtonControl", "OK", (945, 525, 985, 555), process_id=APP)]

    config = parse_config({"hud": {"sounds": False}, "targeting": {"rate_hz": 60}})
    tracker = ScriptedTracker()
    app = AetherApp(
        config,
        Options(no_hud=True),
        tracker,
        source=BlankCamera(),
        backend=RecordingBackend(),
        screen=ScreenGeometry(0, 0, 1920, 1080),
        hotkeys=HotkeyManager({config.hotkeys.toggle_arm: lambda: None}, api=FakeApi()),
        targeting_backend=CenterButton,
    )
    app.quit_requested.disconnect()
    app.start(boot=False)
    try:
        assert app.finder is not None
        tracker.hand = make_hand()  # กลางจอ (960, 540) อยู่ในปุ่ม
        assert wait(qapp, lambda: app.latest_packet is not None)
        assert app.pipeline.snapshot.target is None  # DISARMED: ไม่หาเป้า
        app.toggle_armed()
        assert wait(qapp, lambda: app.pipeline.snapshot.target is not None)
        assert app.pipeline.snapshot.target.control_type == "Button"
        app.toggle_armed()
        assert wait(qapp, lambda: app.pipeline.snapshot.target is None)
    finally:
        app.shutdown()


def test_disabled_targeting_starts_no_finder(qapp):
    from test_actions import RecordingBackend
    from test_app import BlankCamera, ScriptedTracker
    from test_hotkeys import FakeApi

    from aether.app import AetherApp, Options
    from aether.config import parse_config
    from aether.hotkeys import HotkeyManager
    from aether.mapping import ScreenGeometry

    config = parse_config({"hud": {"sounds": False}, "targeting": {"enabled": False}})
    app = AetherApp(
        config,
        Options(no_hud=True),
        ScriptedTracker(),
        source=BlankCamera(),
        backend=RecordingBackend(),
        screen=ScreenGeometry(0, 0, 1920, 1080),
        hotkeys=HotkeyManager({config.hotkeys.toggle_arm: lambda: None}, api=FakeApi()),
        targeting_backend=FakeBackend,
    )
    try:
        assert app.finder is None
    finally:
        app.shutdown()
