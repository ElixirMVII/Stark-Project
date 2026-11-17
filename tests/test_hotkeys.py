from __future__ import annotations

import queue
import threading

import pytest

from aether.hotkeys import Hotkey, HotkeyManager, parse_action, parse_hotkey

# ---- parse -------------------------------------------------------------- #


def test_parse_default_hotkeys():
    toggle = parse_hotkey("<ctrl>+<alt>+g")
    assert toggle == Hotkey(frozenset({"ctrl", "alt"}), "g")
    assert str(toggle) == "Ctrl+Alt+G"
    # virtual-key code ของตัวอักษร = รหัสตัวพิมพ์ใหญ่ ไม่ขึ้นกับ layout แป้นพิมพ์ (ไทย/อังกฤษ)
    assert toggle.vk == 0x47
    assert toggle.win32_modifiers == 0x0001 | 0x0002  # MOD_ALT | MOD_CONTROL
    assert parse_hotkey("<ctrl>+<alt>+q").vk == 0x51


@pytest.mark.parametrize(
    ("spec", "modifiers", "key", "vk"),
    [
        ("<Ctrl>+<Shift>+<F9>", {"ctrl", "shift"}, "f9", 0x78),
        ("ctrl+alt+g", {"ctrl", "alt"}, "g", 0x47),  # ไม่ใส่วงเล็บแหลมก็ได้
        ("<cmd>+<alt>+1", {"win", "alt"}, "1", 0x31),
        ("<win>+<esc>", {"win"}, "esc", 0x1B),
        (" <ctrl> + <alt> + <space> ", {"ctrl", "alt"}, "space", 0x20),
        ("<alt>+<f24>", {"alt"}, "f24", 0x87),
    ],
)
def test_parse_variants(spec, modifiers, key, vk):
    hotkey = parse_hotkey(spec)
    assert hotkey.modifiers == frozenset(modifiers)
    assert hotkey.key == key and hotkey.vk == vk


@pytest.mark.parametrize(
    "spec",
    [
        "",
        "g",  # ไม่มี modifier
        "<ctrl>+<alt>",  # ไม่มีปุ่มหลัก
        "<ctrl>+g+h",  # ปุ่มหลักสองปุ่ม
        "<ctrl>+<ctrl>+g",  # modifier ซ้ำ
        "<ctrl>+<nope>",
        "<ctrl>++g",
        "<ctrl>+เ",  # ไม่ใช่ ASCII
        "<ctrl>+gg",
    ],
)
def test_parse_rejects_invalid(spec):
    with pytest.raises(ValueError):
        parse_hotkey(spec)


def test_to_pynput_format():
    assert parse_hotkey("ctrl+alt+g").to_pynput() == "<ctrl>+<alt>+g"
    assert parse_hotkey("<win>+<shift>+<f5>").to_pynput() == "<shift>+<cmd>+<f5>"


# ---- listener loop (API ของ OS ปลอม) -------------------------------------- #


class FakeApi:
    """เลียนแบบ RegisterHotKey + message loop: test ป้อน id ของ hotkey ผ่าน ``fire``"""

    def __init__(self, reject: set[str] | None = None) -> None:
        self.reject = reject or set()
        self.registered: dict[int, Hotkey] = {}
        self.unregistered: list[int] = []
        self.threads: set[int] = set()
        self._messages: queue.Queue[int | None] = queue.Queue()

    def prepare(self) -> None:
        self.threads.add(threading.get_ident())

    def register(self, hotkey_id: int, hotkey: Hotkey) -> bool:
        self.threads.add(threading.get_ident())
        if str(hotkey) in self.reject:
            return False
        self.registered[hotkey_id] = hotkey
        return True

    def unregister(self, hotkey_id: int) -> None:
        self.unregistered.append(hotkey_id)

    def wait_hotkey(self) -> int | None:
        self.threads.add(threading.get_ident())
        return self._messages.get()

    def interrupt(self) -> None:
        self._messages.put(None)

    def fire(self, hotkey_id: int) -> None:
        self._messages.put(hotkey_id)


def _wait(event: threading.Event) -> bool:
    return event.wait(2.0)


def test_manager_registers_and_dispatches():
    api = FakeApi()
    toggled, quit_ = threading.Event(), threading.Event()
    callback_threads: list[int] = []

    def on_toggle():
        callback_threads.append(threading.get_ident())
        toggled.set()

    manager = HotkeyManager({"<ctrl>+<alt>+g": on_toggle, "<ctrl>+<alt>+q": quit_.set}, api=api)
    assert manager.start() == []
    assert {str(h) for h in api.registered.values()} == {"Ctrl+Alt+G", "Ctrl+Alt+Q"}

    api.fire(1)
    assert _wait(toggled) and not quit_.is_set()
    api.fire(2)
    assert _wait(quit_)

    manager.stop()
    assert sorted(api.unregistered) == [1, 2]
    # ลงทะเบียน รอข้อความ และเรียก callback ใน thread เดียวกัน ซึ่งไม่ใช่ thread ที่เรียก start()
    assert len(api.threads) == 1 and threading.get_ident() not in api.threads
    assert callback_threads == list(api.threads)


def test_manager_reports_hotkeys_that_cannot_be_registered():
    api = FakeApi(reject={"Ctrl+Alt+G"})
    fired = threading.Event()
    manager = HotkeyManager({"<ctrl>+<alt>+g": lambda: None, "<ctrl>+<alt>+q": fired.set}, api=api)
    failed = manager.start()
    assert [str(h) for h in failed] == ["Ctrl+Alt+G"]
    api.fire(2)  # ตัวที่ลงทะเบียนได้ยังใช้งานได้
    assert _wait(fired)
    manager.stop()
    assert api.unregistered == [2]


def test_callback_exception_does_not_kill_listener():
    api = FakeApi()
    second = threading.Event()

    def boom():
        raise RuntimeError("callback failed (test)")

    manager = HotkeyManager({"<ctrl>+<alt>+g": boom, "<ctrl>+<alt>+q": second.set}, api=api)
    manager.start()
    api.fire(1)
    api.fire(2)
    assert _wait(second)
    manager.stop()


def test_unknown_message_id_is_ignored():
    api = FakeApi()
    fired = threading.Event()
    manager = HotkeyManager({"<ctrl>+<alt>+g": fired.set}, api=api)
    manager.start()
    api.fire(99)
    api.fire(0)
    api.fire(1)
    assert _wait(fired)
    manager.stop()


def test_start_twice_and_stop_twice_are_safe():
    api = FakeApi()
    manager = HotkeyManager({"<ctrl>+<alt>+g": lambda: None}, api=api)
    assert manager.start() == []
    assert manager.start() == []
    manager.stop()
    manager.stop()
    assert api.unregistered == [1]


def test_manager_rejects_bad_spec_up_front():
    with pytest.raises(ValueError):
        HotkeyManager({"g": lambda: None}, api=FakeApi())


# ---- parse_action ([actions] ใน config) ----------------------------------- #


def test_parse_action_builtins_and_shortcuts():
    assert parse_action("minimize_foreground") == "minimize_foreground"
    assert parse_action(" None ") == "none"
    combo = parse_action("ctrl+win+left")
    assert isinstance(combo, Hotkey)
    assert combo.keys == ("ctrl", "win", "left")
    assert parse_action("win+tab").keys == ("win", "tab")
    assert parse_action("Win+D").keys == ("win", "d")
    assert parse_action("<ctrl>+<shift>+<esc>").keys == ("ctrl", "shift", "esc")
    assert parse_action("f11").keys == ("f11",)  # ปุ่มเดียวไม่มี modifier ก็ได้ (ต่างจาก hotkey)


def test_keys_are_ordered_modifiers_first():
    assert parse_hotkey("g+<alt>+<ctrl>").keys == ("ctrl", "alt", "g")
    assert parse_hotkey("<win>+<shift>+<alt>+<ctrl>+x").keys == ("ctrl", "alt", "shift", "win", "x")


@pytest.mark.parametrize(
    "spec", ["", "   ", "ctrl+", "ctrl+nope", "ctrl+a+b", "minimize", "ctrl+alt"]
)
def test_parse_action_rejects_invalid(spec):
    with pytest.raises(ValueError):
        parse_action(spec)
