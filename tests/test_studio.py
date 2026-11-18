"""AETHER Studio: input, วาดรูป, ฟิสิกส์, theremin, ห้องทดลองมือ, หน้าต่าง และการเปิดจาก tray"""

from __future__ import annotations

import itertools
import json
import math

import numpy as np
import pytest
from handsim import make_hand
from posesim import HEIGHT, WIDTH, make_pose_hand
from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest
from test_m4 import Factory, Loader, wait

from aether import __main__ as cli
from aether.app import AppController, Options
from aether.config import Config
from aether.studio.draw import (
    BRUSH_SIZES,
    MIN_STEP,
    PALETTE,
    DrawController,
    Drawing,
    swatch_centers,
)
from aether.studio.input import HoldDetector, StudioHand, StudioInput, mouse_hand
from aether.studio.lab import SampleRecorder, hand_metrics, joint_angle, read_samples, sample_hand
from aether.studio.sandbox import MAX_BALLS, MAX_THROW, SandboxController, World
from aether.studio.theremin import (
    SCALES,
    ThereminController,
    Voice,
    frequency_for,
    note_name,
)
from aether.studio.window import StudioWindow
from aether.tracker import HandFrame, Landmark


def frame(t: float, *hands) -> HandFrame:
    return HandFrame(int(t * 1000), 640, 480, tuple(hands))


def hand(x=0.5, y=0.5, *, pinch=False, gesture="None", side="right", vx=0.0, vy=0.0) -> StudioHand:
    return StudioHand(
        side, x, y, pinch, 1.0 if pinch else 0.0, gesture, (True,) * 5, vx, vy, (x, y)
    )


# ---- input ------------------------------------------------------------------ #


def test_input_maps_the_active_region_to_the_canvas():
    studio = StudioInput((0.2, 0.1, 0.8, 0.7))
    assert studio.to_canvas(0.2, 0.1) == (0.0, 0.0)
    assert studio.to_canvas(0.8, 0.7) == (1.0, 1.0)
    assert studio.to_canvas(0.5, 0.4) == pytest.approx((0.5, 0.5))
    assert studio.to_canvas(0.0, 1.0) == (0.0, 1.0)  # นอกกรอบ = ชิดขอบ


def test_input_pinch_has_hysteresis_and_velocity():
    studio = StudioInput()
    (open_hand,) = studio.update(frame(0.0, make_hand(0.4, 0.4)))
    assert not open_hand.pinch and open_hand.pinch_level < 1.0
    (pinched,) = studio.update(frame(0.033, make_hand(0.4, 0.4, pinch=0.1)))
    assert pinched.pinch and pinched.pinch_level == 1.0
    (between,) = studio.update(frame(0.066, make_hand(0.4, 0.4, pinch=0.3)))
    assert between.pinch  # 0.3 อยู่ระหว่าง on (0.25) กับ off (0.38): ยังจีบอยู่
    for i in range(10):  # ขยับไปทางขวาเร็วๆ
        (moving,) = studio.update(frame(0.1 + i * 0.033, make_hand(0.4 + i * 0.02, 0.4)))
    assert moving.vx > 0.5 and abs(moving.vy) < 0.2
    assert moving.fingers == (True,) * 5 and moving.finger_count == 5


def test_input_orders_hands_and_forgets_missing_ones():
    studio = StudioInput()
    hands = studio.update(
        frame(0.0, make_hand(0.3, 0.4, side="left"), make_hand(0.6, 0.4, side="right"))
    )
    assert [h.handedness for h in hands] == ["right", "left"]
    studio.update(frame(0.033, make_hand(0.6, 0.4, pinch=0.1)))
    assert set(studio._tracks) == {"right"}
    assert studio.update(frame(0.066)) == ()


def test_open_palm_and_mouse_hand():
    assert hand(gesture="Open_Palm").open_palm
    assert not hand(pinch=True, gesture="Open_Palm").open_palm
    mouse = mouse_hand(0.2, 0.3, True, 1.0, 0.0)
    assert mouse.handedness == "mouse" and mouse.pinch and mouse.vx == 1.0


def test_hold_detector_fires_once_per_hold():
    hold = HoldDetector(0.5)
    assert not hold.update(True, 0.0)
    assert not hold.update(True, 0.3) and hold.progress == pytest.approx(0.6)
    assert hold.update(True, 0.5)
    assert not hold.update(True, 2.0) and hold.progress == 0.0  # ค้างต่อ: ไม่ยิงซ้ำ
    assert not hold.update(False, 2.1)
    assert not hold.update(True, 2.2) and hold.update(True, 2.8)


# ---- วาดรูป ------------------------------------------------------------------ #


def test_drawing_strokes_undo_and_clear():
    drawing = Drawing()
    drawing.begin(0.1, 0.1)
    assert not drawing.extend(0.1 + MIN_STEP / 2, 0.1)  # ใกล้เกินไป
    assert drawing.extend(0.2, 0.2)
    drawing.end()
    assert len(drawing.strokes) == 1 and drawing.strokes[0].points == [(0.1, 0.1), (0.2, 0.2)]
    assert drawing.strokes[0].length() == pytest.approx(math.dist((0.1, 0.1), (0.2, 0.2)))
    color = drawing.color
    assert drawing.next_color() != color
    assert drawing.cycle_size() == BRUSH_SIZES[2]
    assert drawing.undo() and not drawing.undo()
    drawing.begin(0.5, 0.5)
    drawing.clear()
    assert drawing.strokes == [] and not drawing.drawing


def test_draw_controller_draws_picks_colors_and_runs_hold_commands():
    drawing = Drawing()
    control = DrawController(drawing)
    t = 0.0
    for i in range(10):
        control.update(hand(0.3 + i * 0.02, 0.5, pinch=True), t)
        t += 0.033
    control.update(hand(0.5, 0.5), t)
    assert len(drawing.strokes) == 1 and len(drawing.strokes[0].points) == 10

    sx, sy = swatch_centers(control.aspect)[4]
    events = control.update(hand(sx, sy, pinch=True), t + 0.1)
    assert drawing.color == PALETTE[4] and [e.kind for e in events] == ["color"]
    control.update(hand(sx + 0.1, sy + 0.1, pinch=True), t + 0.15)  # จีบต่อ: ไม่วาด
    control.update(hand(0.5, 0.5), t + 0.2)
    assert len(drawing.strokes) == 1

    def hold(gesture: str, start: float, seconds: float = 1.5) -> list[str]:
        kinds = []
        now = start
        while now < start + seconds:
            kinds += [e.kind for e in control.update(hand(0.5, 0.5, gesture=gesture), now)]
            now += 0.05
        control.update(hand(0.5, 0.5), now)
        return kinds

    assert hold("Victory", 1.0) == ["color"]
    assert hold("Thumb_Up", 3.0) == ["save"]
    assert hold("Closed_Fist", 5.0) == ["undo"] and drawing.strokes == []
    drawing.begin(0.1, 0.1)
    drawing.end()
    assert hold("Thumb_Down", 7.0) == ["clear"] and drawing.strokes == []
    control.update(None, 9.0)  # มือหาย: จบเส้น ไม่ค้าง
    assert not drawing.drawing


# ---- ฟิสิกส์ ----------------------------------------------------------------- #


def test_balls_fall_bounce_and_come_to_rest():
    world = World(800, 500)
    ball = world.spawn(400, 100, r=20)
    ball.vx = 0.0
    energies = []
    for _ in range(600):
        world.step(1 / 60)
        energies.append(world.energy())
    assert ball.y == pytest.approx(480, abs=0.5) and abs(ball.vy) < 1.0
    assert energies[-1] < energies[0]
    assert all(b <= a * 1.02 + 1.0 for a, b in itertools.pairwise(energies))


def test_walls_and_collisions_keep_balls_apart_and_inside():
    world = World(400, 300)
    a = world.spawn(100, 150, r=30)
    b = world.spawn(110, 150, r=30)  # ซ้อนกันตั้งแต่เริ่ม
    a.vx, b.vx = -900.0, 900.0
    world.gravity = 0.0
    for _ in range(120):
        world.step(1 / 60)
    assert math.hypot(a.x - b.x, a.y - b.y) >= a.r + b.r - 0.5
    for ball in world.balls:
        assert ball.r - 0.01 <= ball.x <= 400 - ball.r + 0.01
        assert ball.r - 0.01 <= ball.y <= 300 - ball.r + 0.01


def test_grab_hold_and_throw():
    world = World(800, 500)
    ball = world.spawn(400, 300, r=25)
    assert world.grab("right", 400 + 25 + 40, 300) is None  # ไกลเกิน
    assert world.grab("right", 420, 300) is ball and ball.held_by == "right"
    world.hold("right", 200, 150)
    for _ in range(60):
        world.step(1 / 60)
    assert ball.x == pytest.approx(200, abs=3) and ball.y == pytest.approx(150, abs=3)  # ไม่ตก
    world.release("right", 1e6, 0.0)
    assert ball.held_by is None and ball.vx == pytest.approx(MAX_THROW)


def test_palm_push_and_ball_limit():
    world = World(800, 500)
    ball = world.spawn(400, 475, r=25)
    ball.vx = 0.0
    assert world.push(380, 470, 1200.0, 0.0) == 1 and ball.vx > 300
    assert world.push(50, 50, 1200.0, 0.0) == 0
    for _ in range(MAX_BALLS + 5):
        world.spawn(100, 100)
    assert len(world.balls) == MAX_BALLS


def test_sandbox_controller_grabs_throws_and_spawns():
    world = World(1000, 500)
    world.balls.clear()
    ball = world.spawn(500, 250, r=30)
    control = SandboxController(world)
    control.update((hand(0.5, 0.5, pinch=True),), 0.0)
    assert ball.held_by == "right"
    control.update((hand(0.6, 0.4, pinch=True),), 0.05)
    control.update((hand(0.6, 0.4, vx=1.0, vy=-0.5),), 0.1)  # ปล่อย: โยนด้วยความเร็วมือ
    assert (
        ball.held_by is None and ball.vx == pytest.approx(1000) and ball.vy == pytest.approx(-250)
    )

    control.update((hand(0.5, 0.5, pinch=True),), 0.2)
    control.update((), 0.25)  # มือหายระหว่างถือ: ปล่อยเฉยๆ
    assert ball.held_by is None

    count = len(world.balls)
    now = 1.0
    events = []
    while now < 2.0:
        events += control.update((hand(0.3, 0.3, gesture="Victory"),), now)
        now += 0.05
    assert events == ["Ball"] and len(world.balls) == count + 1


# ---- theremin --------------------------------------------------------------- #


def test_frequency_follows_height_and_snaps_to_the_scale():
    assert frequency_for(0.0) == pytest.approx(220.0)
    assert frequency_for(1.0) == pytest.approx(880.0)
    assert frequency_for(0.5, "free") == pytest.approx(440.0)
    allowed = {d for d in SCALES["pentatonic"]}
    for step in range(50):
        semis = round(12 * math.log2(frequency_for(step / 49) / 220.0))
        assert semis % 12 in allowed
    assert note_name(440.0) == "A4" and note_name(261.63) == "C4" and note_name(0) == "—"


def test_voice_is_continuous_and_reaches_the_target_pitch():
    voice = Voice(sample_rate=48_000)
    voice.frequency, voice.amplitude = 440.0, 0.5
    first = voice.render(4800)
    second = voice.render(48_000)
    assert first.dtype == np.float32 and np.abs(second).max() <= 0.5 + 1e-6
    assert abs(float(second[0] - first[-1])) < 0.05  # ไม่มีรอยต่อ (ไม่แตก)
    crossings = np.count_nonzero(np.diff(np.signbit(second[24_000:])))
    assert crossings / 2 / 0.5 == pytest.approx(440, rel=0.03)
    voice.amplitude = 0.0
    tail = voice.render(9600)
    assert np.abs(tail[-500:]).max() < 0.01  # fade ลงจนเงียบ


def test_theremin_controller_plays_only_while_pinching():
    voice = Voice()
    control = ThereminController(voice, volume=0.8)
    state = control.update((hand(0.2, 0.0, pinch=True),))
    assert state.playing and state.frequency == pytest.approx(880.0) and voice.amplitude == 0.8
    assert control.update((hand(0.2, 0.0),)).playing is False and voice.amplitude == 0.0
    both = control.update((hand(0.2, 1.0, pinch=True), hand(0.8, 0.5, side="left")))
    assert both.frequency == pytest.approx(220.0) and voice.amplitude == pytest.approx(0.4)
    margin = control.update((hand(0.2, 0.05, pinch=True),), margin=0.05)
    assert margin.height == pytest.approx(1.0)
    assert control.update(()).playing is False
    with pytest.raises(ValueError):
        ThereminController(voice, scale="blues")


# ---- ห้องทดลองมือ ------------------------------------------------------------ #


def test_joint_angle():
    a, b = Landmark(0, 0, 0), Landmark(1, 0, 0)
    assert joint_angle(a, b, Landmark(2, 0, 0)) == pytest.approx(0.0)
    assert joint_angle(a, b, Landmark(1, 1, 0)) == pytest.approx(90.0)


@pytest.mark.parametrize(("fingers", "count"), [("11111", 5), ("01100", 2), ("01000", 1)])
def test_hand_metrics_counts_fingers(fingers, count):
    metrics = hand_metrics(make_pose_hand(fingers), WIDTH, HEIGHT)
    assert metrics.count == count
    assert metrics.extended == tuple(c == "1" for c in fingers)
    assert len(metrics.bend) == 5 and len(metrics.pinch) == 4
    assert abs(metrics.roll) < 10 and metrics.facing in ("palm", "back")


def test_hand_metrics_bend_grows_when_curled():
    straight = hand_metrics(make_pose_hand("11111"), WIDTH, HEIGHT)
    fist = hand_metrics(make_pose_hand("00000"), WIDTH, HEIGHT)
    assert all(f > s for f, s in zip(fist.bend[1:], straight.bend[1:], strict=True))
    assert sum(fist.curled[1:]) >= 3


def test_sample_recorder_appends_numbers_only(tmp_path):
    path = tmp_path / "samples.jsonl"
    recorder = SampleRecorder(path, clock=lambda: 12.5)
    recorder.add("peace", make_pose_hand("01100"), WIDTH, HEIGHT)
    recorder.add("peace", make_pose_hand("01100", x=0.4), WIDTH, HEIGHT)
    with pytest.raises(ValueError):
        recorder.add("  ", make_pose_hand(), WIDTH, HEIGHT)
    with path.open("a", encoding="utf-8") as stream:
        stream.write("{broken\n")
    lines = path.read_text(encoding="utf-8").splitlines()
    record = json.loads(lines[0])
    assert set(record) == {
        "v",
        "label",
        "handedness",
        "width",
        "height",
        "time",
        "landmarks",
        "world",
    }
    assert len(record["landmarks"]) == 21 and all(len(p) == 3 for p in record["landmarks"])
    assert recorder.counts["peace"] == 2
    assert SampleRecorder(path).counts == {"peace": 2}  # เปิดใหม่: นับจากไฟล์ (ข้ามบรรทัดเสีย)
    restored = sample_hand(read_samples(path)[0])
    assert hand_metrics(restored, WIDTH, HEIGHT).count == 2


# ---- หน้าต่าง ------------------------------------------------------------------ #


@pytest.fixture
def studio(qapp, tmp_path):
    clock = {"t": 0.0}
    window = StudioWindow(
        Config(),
        samples_path=tmp_path / "samples.jsonl",
        save_dir=tmp_path / "pictures",
        audio=False,
        clock=lambda: clock["t"],
    )
    window.resize(1000, 700)
    window.show()
    yield window, clock
    window.close()


def feed(window: StudioWindow, clock: dict, hands_at, frames: int = 20) -> None:
    for i in range(frames):
        clock["t"] += 1 / 30
        window.show_frame(frame(clock["t"], *hands_at(i)))
        window._tick()


def test_window_draws_with_a_pinched_hand_and_saves_png(studio):
    window, clock = studio
    feed(window, clock, lambda i: [make_hand(0.4 + i * 0.01, 0.4, pinch=0.1)])
    feed(window, clock, lambda i: [make_hand(0.6, 0.4)], frames=2)
    assert len(window.draw.drawing.strokes) == 1
    assert "Right" in window.status.text()
    path = window.draw.save_png()
    assert path is not None and path.is_file() and path.suffix == ".png"
    window.grab()


def test_window_switches_pages_and_drops_held_things(studio):
    window, clock = studio
    window.set_page(1)
    assert window.current is window.sandbox and "Pinch a ball" in window.hint.text()
    window.sandbox.world.balls.clear()
    ball = window.sandbox.world.spawn(window.sandbox.width() * 0.5, window.sandbox.height() * 0.5)
    feed(window, clock, lambda i: [make_hand(0.5, 0.4, pinch=0.1)], frames=5)
    held = [b for b in window.sandbox.world.balls if b.held_by]
    window.set_page(2)
    assert all(b.held_by is None for b in window.sandbox.world.balls)
    assert window.current is window.theremin
    feed(window, clock, lambda i: [make_hand(0.5, 0.2, pinch=0.1)], frames=5)
    assert window.theremin.controller.state.playing
    window.grab()
    window.set_page(0)
    assert window.theremin.voice.amplitude == 0.0  # ออกจากหน้า theremin = เงียบ
    assert ball is not None and isinstance(held, list)


def test_window_lab_records_samples(studio, tmp_path):
    window, clock = studio
    window.set_page(window.pages.index(window.lab))
    window.lab_label.setText("open")
    feed(window, clock, lambda i: [make_hand(0.5, 0.4)], frames=3)
    assert window.lab.metrics is not None and window.lab.metrics.count >= 4
    assert window.lab.record_once()
    window.lab.start_burst()
    feed(window, clock, lambda i: [make_hand(0.5, 0.4)], frames=100)  # 3.3 วินาที
    assert not window.lab.bursting
    assert window.lab.recorder.counts["open"] >= 20
    assert len(read_samples(tmp_path / "samples.jsonl")) == window.lab.recorder.counts["open"]
    window.grab()


def test_mouse_draws_when_no_camera_hand(studio):
    window, clock = studio
    page = window.draw
    QTest.mousePress(page, Qt.MouseButton.LeftButton, pos=QPoint(400, 300))
    for x in range(400, 600, 20):
        clock["t"] += 0.02
        QTest.mouseMove(page, QPoint(x, 320))
    QTest.mouseRelease(page, Qt.MouseButton.LeftButton, pos=QPoint(600, 320))
    assert len(page.drawing.strokes) == 1 and len(page.drawing.strokes[0].points) > 3


def test_close_emits_finished_once(studio):
    window, _ = studio
    results = []
    window.finished.connect(results.append)
    window.close()
    window.close()
    assert results == [False]


# ---- เปิดจาก tray / command line ------------------------------------------------- #


def test_tray_opens_studio_disarms_and_feeds_frames(qapp, tmp_path):
    factory, loader = Factory(), Loader()
    controller = AppController(
        Options(no_hud=True),
        app_factory=factory,
        config_loader=loader,
        calibration_file=tmp_path / "calibration.json",
        poses_file=tmp_path / "poses.json",
        samples_file=tmp_path / "samples.jsonl",
        use_tray=True,
    )
    try:
        controller.start(loader())
        app = controller.app
        app.toggle_armed()
        controller.tray.actions()["studio"].trigger()
        studio = controller.wizard
        assert isinstance(studio, StudioWindow) and studio.isVisible()
        assert not app.pipeline.armed
        app.toggle_armed()
        assert not app.pipeline.armed  # เปิด Studio อยู่: มือไม่คุมเมาส์
        factory.trackers[0].hand = make_hand(0.5, 0.4)
        assert wait(qapp, lambda: bool(studio.input.hands))
        studio.close()
        assert controller.wizard is None and len(factory.apps) == 1  # ไม่ reload
    finally:
        controller.shutdown()


def test_studio_cli_flag(monkeypatch):
    captured = {}
    monkeypatch.setattr("aether.app.run", lambda options: captured.setdefault("o", options) and 0)
    cli.main(["--studio", "--source", "mouse"])
    assert captured["o"].studio and captured["o"].source.kind == "mouse"
