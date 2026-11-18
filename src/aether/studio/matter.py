"""จำลองสถานะของสสารด้วยอนุภาค (pure — ไม่มี Qt) สำหรับเครื่องมือทดลอง "Matter"

สถานะมาจากจุดหลอมเหลว/จุดเดือดจริงของสาร (:func:`aether.studio.science.phase_at`) ส่วนการเคลื่อนที่เป็นภาพประกอบ:

- ของแข็ง: อนุภาคอยู่ประจำตำแหน่งในโครงผลึก สั่นมากขึ้นเมื่อร้อนขึ้น
- ของเหลว: ไหลรวมกันอยู่ก้นภาชนะ ชนกันไปมา (ปริมาตรคงที่ รูปร่างตามภาชนะ)
- แก๊ส: วิ่งเร็วเต็มภาชนะ ความเร็วเพิ่มตาม √อุณหภูมิ (เคลวิน)

พิกัดเป็น 0–1 ของภาชนะ (y = 0 บนสุด)
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

__all__ = ["MatterSim", "Particle", "slider_from_temperature", "temperature_from_slider"]

T_MIN_K = 1.0
T_MAX_K = 6000.0
RADIUS = 0.022  # รัศมีอนุภาค (สัดส่วนความกว้างภาชนะ)


def temperature_from_slider(value: float) -> float:
    """สไลเดอร์ 0–1 → องศาเซลเซียส แบบลอการิทึม (1 K … 6000 K) — ให้ช่วงน้ำ 0–100 °C ไม่เบียดกันจนใช้ไม่ได้"""
    value = min(max(value, 0.0), 1.0)
    return T_MIN_K * (T_MAX_K / T_MIN_K) ** value - 273.15


def slider_from_temperature(celsius: float) -> float:
    kelvin = min(max(celsius + 273.15, T_MIN_K), T_MAX_K)
    return math.log(kelvin / T_MIN_K) / math.log(T_MAX_K / T_MIN_K)


@dataclass
class Particle:
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    home: tuple[float, float] = (0.5, 0.5)


@dataclass
class MatterSim:
    count: int = 64
    seed: int = 11
    phase: str = "solid"
    celsius: float = 25.0
    particles: list[Particle] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)
        self._sites = self._lattice()
        if not self.particles:
            self.particles = [Particle(x, y, home=(x, y)) for x, y in self._sites]

    # ---- ตั้งค่า ------------------------------------------------------------- #

    def _lattice(self) -> list[tuple[float, float]]:
        """ตำแหน่งในผลึกแบบหกเหลี่ยม ชิดก้นภาชนะ"""
        spacing = RADIUS * 2.15
        per_row = int((1.0 - 2 * RADIUS) / spacing)
        sites = []
        row = 0
        while len(sites) < self.count:
            offset = spacing / 2 if row % 2 else 0.0
            y = 1.0 - RADIUS - row * spacing * 0.87
            for column in range(per_row):
                x = 0.5 - (per_row - 1) * spacing / 2 + column * spacing + offset
                if len(sites) < self.count and RADIUS <= x <= 1 - RADIUS:
                    sites.append((x, y))
            row += 1
        return sites

    def set_state(self, phase: str | None, celsius: float) -> None:
        """``phase=None`` (ไม่มีข้อมูล) คงสถานะเดิมไว้"""
        self.celsius = celsius
        if phase is None or phase == self.phase:
            return
        if phase == "solid":  # แข็งตัว: จับคู่อนุภาคกับตำแหน่งในผลึกที่ใกล้ (เรียงตามความสูงก่อน)
            ordered = sorted(self.particles, key=lambda p: (-p.y, p.x))
            sites = sorted(self._sites, key=lambda s: (-s[1], s[0]))
            for particle, site in zip(ordered, sites, strict=False):
                particle.home = site
        self.phase = phase

    @property
    def speed(self) -> float:
        """ความเร็วเฉลี่ยของอนุภาคแก๊ส (หน่วยภาชนะ/วินาที) ∝ √T"""
        return 0.35 * math.sqrt(max(self.celsius + 273.15, 1.0) / 300.0)

    # ---- จำลอง -------------------------------------------------------------- #

    def step(self, dt: float) -> None:
        dt = min(max(dt, 0.0), 0.05)
        if dt == 0.0:
            return
        if self.phase == "solid":
            self._step_solid(dt)
        elif self.phase == "liquid":
            self._step_liquid(dt)
        else:
            self._step_gas(dt)
        for p in self.particles:
            self._walls(p)

    def _step_solid(self, dt: float) -> None:
        amplitude = 0.0008 + 0.003 * min(self.speed / 0.5, 2.0)
        for p in self.particles:
            hx, hy = p.home
            p.vx += (-(p.x - hx) * 120.0 + self._rng.gauss(0, 1) * amplitude * 400) * dt
            p.vy += (-(p.y - hy) * 120.0 + self._rng.gauss(0, 1) * amplitude * 400) * dt
            p.vx *= 0.8
            p.vy *= 0.8
            p.x += p.vx * dt
            p.y += p.vy * dt

    def _step_liquid(self, dt: float) -> None:
        jitter = 0.4 * self.speed
        for p in self.particles:
            p.vy += 1.6 * dt  # แรงโน้มถ่วง: ของเหลวไหลลงก้น
            p.vx += self._rng.gauss(0, 1) * jitter * dt * 6
            p.vy += self._rng.gauss(0, 1) * jitter * dt * 6
            p.vx *= 0.92
            p.vy *= 0.92
            p.x += p.vx * dt
            p.y += p.vy * dt
        self._separate(RADIUS * 2.05)

    def _separate(self, distance: float, rounds: int = 3) -> None:
        """ดันอนุภาคที่ซ้อนกันออก (ของเหลวอัดตัวไม่ได้ — ปริมาตรใกล้ของแข็ง)"""
        for _ in range(rounds):
            cells: dict[tuple[int, int], list[Particle]] = {}
            for p in self.particles:
                cells.setdefault((int(p.x / distance), int(p.y / distance)), []).append(p)
            for (cx, cy), members in cells.items():
                for dx in (0, 1):
                    for dy in (-1, 0, 1):
                        if dx == 0 and dy == -1:
                            continue
                        others = (
                            members if (dx, dy) == (0, 0) else cells.get((cx + dx, cy + dy), ())
                        )
                        for i, p in enumerate(members):
                            start = i + 1 if (dx, dy) == (0, 0) else 0
                            for q in others[start:]:
                                ddx, ddy = q.x - p.x, q.y - p.y
                                dist = math.hypot(ddx, ddy)
                                if dist >= distance:
                                    continue
                                if dist < 1e-6:
                                    ddx, ddy, dist = 1e-3, 0.0, 1e-3
                                shift = (distance - dist) / 2
                                ux, uy = ddx / dist, ddy / dist
                                p.x -= ux * shift
                                p.y -= uy * shift
                                q.x += ux * shift
                                q.y += uy * shift
            for p in self.particles:
                self._walls(p)

    def _step_gas(self, dt: float) -> None:
        target = self.speed
        for p in self.particles:
            speed = math.hypot(p.vx, p.vy)
            if speed < 1e-6:
                angle = self._rng.uniform(0, 2 * math.pi)
                p.vx, p.vy = math.cos(angle) * target, math.sin(angle) * target
            else:  # ค่อยๆ ปรับความเร็วเข้าหาค่าตามอุณหภูมิ
                scale = 1.0 + (target / speed - 1.0) * min(1.0, dt * 4)
                p.vx *= scale
                p.vy *= scale
            p.vy += 0.05 * dt  # แรงโน้มถ่วงน้อยมากเมื่อเทียบกับความเร็วโมเลกุล
            p.x += p.vx * dt
            p.y += p.vy * dt

    @staticmethod
    def _walls(p: Particle) -> None:
        if p.x < RADIUS:
            p.x, p.vx = RADIUS, abs(p.vx)
        elif p.x > 1 - RADIUS:
            p.x, p.vx = 1 - RADIUS, -abs(p.vx)
        if p.y < RADIUS:
            p.y, p.vy = RADIUS, abs(p.vy)
        elif p.y > 1 - RADIUS:
            p.y, p.vy = 1 - RADIUS, -abs(p.vy)

    # ---- สถิติ -------------------------------------------------------------- #

    def spread(self) -> float:
        """ความสูงที่อนุภาคกระจาย (0 = อัดแน่นก้นภาชนะ … 1 = เต็มภาชนะ)"""
        ys = [p.y for p in self.particles]
        return max(ys) - min(ys)

    def mean_offset(self) -> float:
        """ระยะเฉลี่ยจากตำแหน่งในผลึก (ของแข็งควรน้อย)"""
        return sum(math.hypot(p.x - p.home[0], p.y - p.home[1]) for p in self.particles) / len(
            self.particles
        )
