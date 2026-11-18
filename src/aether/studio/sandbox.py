"""กระบะทรายฟิสิกส์ 2D — ลูกบอลมีแรงโน้มถ่วง เด้งผนัง ชนกันเอง (pure, หน่วย = px ของผืนผ้าใบ)

มือ: จีบใกล้ลูกบอล = หยิบ · ขยับ = ลูกบอลตามแบบสปริง · ปล่อย = โยนด้วยความเร็วของมือ ·
แบมือแล้วปัด = ผลักลูกบอลรอบฝ่ามือ · ✌ ค้าง = เพิ่มลูกบอล · 👎 ค้าง = เริ่มใหม่

ใช้ timestep คงที่ (``STEP``) + เก็บเศษเวลา ผลจึงเหมือนเดิมทุกเครื่องไม่ว่า FPS เท่าไร
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from .input import HoldDetector, StudioHand

__all__ = ["Ball", "SandboxController", "World"]

STEP = 1.0 / 120.0
MAX_STEPS = 12  # ต่อการเรียก step หนึ่งครั้ง (หน้าต่างค้างนาน: ไม่ไล่จำลองย้อนหลังทั้งหมด)
GRAVITY = 1500.0  # px/s²
RESTITUTION = 0.78  # ความเด้งตอนชนผนัง/ชนกัน
AIR_DRAG = 0.15  # ต่อวินาที
FLOOR_FRICTION = 2.5  # ต่อวินาที ตอนกลิ้งบนพื้น
GRAB_MARGIN = 28.0  # หยิบได้แม้นิ้วอยู่นอกลูกบอลเล็กน้อย
GRAB_STIFFNESS = 900.0  # สปริงดึงลูกที่ถืออยู่เข้าหานิ้ว
GRAB_DAMPING = 45.0
MAX_THROW = 4000.0  # px/s
PUSH_RADIUS = 140.0
PUSH_GAIN = 0.9  # สัดส่วนของความเร็วฝ่ามือที่ถ่ายให้ลูกบอล
MAX_BALLS = 40
COLORS = ("#F2F4F7", "#D6B67A", "#E5484D", "#5B9BD5", "#7CC4A0", "#B392F0")


@dataclass(slots=True)
class Ball:
    x: float
    y: float
    r: float
    color: str
    vx: float = 0.0
    vy: float = 0.0
    held_by: str | None = None
    target: tuple[float, float] | None = None

    @property
    def mass(self) -> float:
        return self.r * self.r


class World:
    def __init__(self, width: float, height: float, *, seed: int = 5) -> None:
        self.width = width
        self.height = height
        self.balls: list[Ball] = []
        self.gravity = GRAVITY
        self._rng = random.Random(seed)
        self._carry = 0.0

    def resize(self, width: float, height: float) -> None:
        self.width, self.height = max(width, 1.0), max(height, 1.0)
        for ball in self.balls:
            self._contain(ball)

    def spawn(self, x: float, y: float, r: float | None = None, color: str | None = None) -> Ball:
        if len(self.balls) >= MAX_BALLS:
            self.balls.pop(0)
        ball = Ball(
            x,
            y,
            r if r is not None else self._rng.uniform(18.0, 34.0),
            color or self._rng.choice(COLORS),
            vx=self._rng.uniform(-60.0, 60.0),
        )
        self._contain(ball)
        self.balls.append(ball)
        return ball

    def reset(self, count: int = 6) -> None:
        self.balls.clear()
        for i in range(count):
            self.spawn(self.width * (i + 1) / (count + 1), self.height * 0.3)

    # ---- มือ ------------------------------------------------------------ #

    def grab(self, hand: str, x: float, y: float) -> Ball | None:
        """หยิบลูกบอลที่ใกล้ที่สุดในระยะ (ลูกที่มือนี้ถืออยู่แล้วถือว่าหยิบอยู่)"""
        best, best_gap = None, math.inf
        for ball in self.balls:
            if ball.held_by not in (None, hand):
                continue
            gap = math.hypot(ball.x - x, ball.y - y) - ball.r
            if gap <= GRAB_MARGIN and gap < best_gap:
                best, best_gap = ball, gap
        if best is not None:
            best.held_by, best.target = hand, (x, y)
        return best

    def hold(self, hand: str, x: float, y: float) -> None:
        for ball in self.balls:
            if ball.held_by == hand:
                ball.target = (x, y)

    def release(self, hand: str, vx: float, vy: float) -> Ball | None:
        released = None
        for ball in self.balls:
            if ball.held_by == hand:
                speed = math.hypot(vx, vy)
                scale = min(1.0, MAX_THROW / speed) if speed > 0 else 1.0
                ball.vx, ball.vy = vx * scale, vy * scale
                ball.held_by, ball.target = None, None
                released = ball
        return released

    def push(self, x: float, y: float, vx: float, vy: float, radius: float = PUSH_RADIUS) -> int:
        """ฝ่ามือที่ขยับผ่าน: ลูกบอลในรัศมีได้ความเร็วตามฝ่ามือ (ใกล้ = แรงกว่า) คืนจำนวนลูกที่โดน"""
        hit = 0
        for ball in self.balls:
            if ball.held_by is not None:
                continue
            distance = math.hypot(ball.x - x, ball.y - y)
            if distance > radius + ball.r:
                continue
            falloff = 1.0 - min(distance / (radius + ball.r), 1.0)
            ball.vx += (vx - ball.vx) * PUSH_GAIN * falloff
            ball.vy += (vy - ball.vy) * PUSH_GAIN * falloff
            hit += 1
        return hit

    # ---- จำลอง ---------------------------------------------------------- #

    def step(self, dt: float) -> int:
        """เดินเวลา ``dt`` วินาที (แบ่งเป็นช่วงละ ``STEP``) คืนจำนวนช่วงที่จำลอง"""
        self._carry += max(dt, 0.0)
        steps = 0
        while self._carry >= STEP and steps < MAX_STEPS:
            self._substep(STEP)
            self._carry -= STEP
            steps += 1
        if steps == MAX_STEPS:
            self._carry = 0.0
        return steps

    def _substep(self, dt: float) -> None:
        drag = math.exp(-AIR_DRAG * dt)
        for ball in self.balls:
            if ball.held_by is not None and ball.target is not None:
                tx, ty = ball.target
                ax = GRAB_STIFFNESS * (tx - ball.x) - GRAB_DAMPING * ball.vx
                ay = GRAB_STIFFNESS * (ty - ball.y) - GRAB_DAMPING * ball.vy
                ball.vx += ax * dt
                ball.vy += ay * dt
            else:
                ball.vy += self.gravity * dt
                ball.vx *= drag
                ball.vy *= drag
            ball.x += ball.vx * dt
            ball.y += ball.vy * dt
            self._walls(ball, dt)
        self._collide()

    def _walls(self, ball: Ball, dt: float) -> None:
        if ball.x - ball.r < 0.0:
            ball.x, ball.vx = ball.r, abs(ball.vx) * RESTITUTION
        elif ball.x + ball.r > self.width:
            ball.x, ball.vx = self.width - ball.r, -abs(ball.vx) * RESTITUTION
        if ball.y - ball.r < 0.0:
            ball.y, ball.vy = ball.r, abs(ball.vy) * RESTITUTION
        elif ball.y + ball.r > self.height:
            ball.y = self.height - ball.r
            ball.vy = -abs(ball.vy) * RESTITUTION
            if abs(ball.vy) < 40.0:  # หยุดเด้งเล็กๆ ไม่จบ
                ball.vy = 0.0
            ball.vx *= math.exp(-FLOOR_FRICTION * dt)

    def _contain(self, ball: Ball) -> None:
        ball.x = min(max(ball.x, ball.r), max(self.width - ball.r, ball.r))
        ball.y = min(max(ball.y, ball.r), max(self.height - ball.r, ball.r))

    def _collide(self) -> None:
        balls = self.balls
        for i, a in enumerate(balls):
            for b in balls[i + 1 :]:
                dx, dy = b.x - a.x, b.y - a.y
                distance = math.hypot(dx, dy)
                overlap = a.r + b.r - distance
                if overlap <= 0.0:
                    continue
                if distance < 1e-6:
                    dx, dy, distance = 1.0, 0.0, 1.0
                nx, ny = dx / distance, dy / distance
                # มวลตามพื้นที่ ลูกที่ถืออยู่ถือว่าหนักมาก (ไม่ถูกดันออกจากมือ)
                inv_a = 0.0 if a.held_by else 1.0 / a.mass
                inv_b = 0.0 if b.held_by else 1.0 / b.mass
                total = inv_a + inv_b
                if total == 0.0:
                    continue
                a.x -= nx * overlap * inv_a / total
                a.y -= ny * overlap * inv_a / total
                b.x += nx * overlap * inv_b / total
                b.y += ny * overlap * inv_b / total
                closing = (b.vx - a.vx) * nx + (b.vy - a.vy) * ny
                if closing >= 0.0:
                    continue
                impulse = -(1.0 + RESTITUTION) * closing / total
                a.vx -= impulse * inv_a * nx
                a.vy -= impulse * inv_a * ny
                b.vx += impulse * inv_b * nx
                b.vy += impulse * inv_b * ny

    def energy(self) -> float:
        """พลังงานจลน์ + ศักย์รวม (ไว้ทดสอบว่าไม่ "ระเบิด")"""
        return sum(
            0.5 * b.mass * (b.vx * b.vx + b.vy * b.vy) + b.mass * self.gravity * (self.height - b.y)
            for b in self.balls
        )


class SandboxController:
    """มือ (หน่วยผืนผ้าใบ 0–1) → การกระทำกับ :class:`World` (หน่วย px)"""

    def __init__(self, world: World) -> None:
        self.world = world
        self.spawn_hold = HoldDetector(0.7)
        self.reset_hold = HoldDetector(1.2)
        self._pinched: dict[str, bool] = {}

    def update(self, hands: tuple[StudioHand, ...], now: float) -> list[str]:
        """คืนข้อความสั้นๆ ที่ควรแจ้งผู้ใช้ (เช่น "Ball", "Reset")"""
        world = self.world
        events: list[str] = []
        present = set()
        for hand in hands:
            name = hand.handedness
            present.add(name)
            x, y = hand.x * world.width, hand.y * world.height
            vx, vy = hand.vx * world.width, hand.vy * world.height
            was = self._pinched.get(name, False)
            if hand.pinch and not was:
                world.grab(name, x, y)
            elif hand.pinch:
                world.hold(name, x, y)
            elif was:
                world.release(name, vx, vy)
            self._pinched[name] = hand.pinch
            if hand.open_palm:
                px, py = hand.palm[0] * world.width, hand.palm[1] * world.height
                world.push(px, py, vx, vy)
        for name in list(self._pinched):
            if name not in present:  # มือหายไประหว่างถือ: ปล่อยลูกบอล (ไม่มีแรงโยน)
                world.release(name, 0.0, 0.0)
                del self._pinched[name]

        main = hands[0] if hands else None
        idle = main is not None and not main.pinch
        if self.spawn_hold.update(idle and main.gesture == "Victory", now) and main:
            world.spawn(main.x * world.width, main.y * world.height)
            events.append("Ball")
        if self.reset_hold.update(idle and main.gesture == "Thumb_Down", now):
            world.reset()
            events.append("Reset")
        return events

    @property
    def hold_progress(self) -> float:
        return max(self.spawn_hold.progress, self.reset_hold.progress)
