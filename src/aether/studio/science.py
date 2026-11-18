"""ข้อมูลและสูตรของเครื่องมือทดลอง (pure — ไม่มี Qt): "สาร" ที่ส่งมาจากหน้า Elements, จุดหลอมเหลว/จุดเดือด/ความหนาแน่น,
อากาศ (องค์ประกอบ + ลูกโป่งลอย/จม) และแรงโน้มถ่วงของดาวต่างๆ

ค่าทางกายภาพ (ที่ความดัน 1 atm, อุณหภูมิห้อง) ใส่เฉพาะสารที่ค่าเป็นที่รู้กันแน่นอน — สารที่ไม่มีข้อมูลจะบอกว่า
"ไม่มีข้อมูล" แทนการเดา (ตาม D54) ค่าปัดให้อ่านง่าย ใช้เพื่อการเรียนรู้ ไม่ใช่งานวิศวกรรม
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass

from .atoms import atom_info
from .chemistry import Compound, Element, by_symbol, formula_text, hill_formula, parse_formula

__all__ = [
    "AIR",
    "AIR_MOLAR_MASS",
    "PLANETS",
    "Balloon",
    "PhaseData",
    "Subject",
    "balloon_for",
    "fall_time",
    "kelvin",
    "phase_at",
    "phase_data",
]

ROOM_C = 25.0
ABSOLUTE_ZERO = -273.15
MOLAR_VOLUME_L = 22.414  # ลิตรต่อโมลของแก๊สที่ 0 °C, 1 atm


def kelvin(celsius: float) -> float:
    return celsius - ABSOLUTE_ZERO


# --------------------------------------------------------------------------- #
# สาร (subject) ที่เครื่องมือทดลองใช้
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Subject:
    kind: str  # "element" | "compound" | "mix" (ของในขวดที่ยังไม่ตรงสูตรใด)
    formula: str
    name: str
    thai: str
    counts: Counter[str]
    element: Element | None = None
    compound: Compound | None = None

    @classmethod
    def from_element(cls, element: Element) -> Subject:
        return cls(
            "element",
            element.symbol,
            element.name,
            element.thai,
            Counter({element.symbol: 1}),
            element,
        )

    @classmethod
    def from_compound(cls, compound: Compound) -> Subject:
        return cls(
            "compound",
            compound.formula,
            compound.name,
            compound.thai,
            compound.counts,
            compound=compound,
        )

    @classmethod
    def from_counts(cls, counts: Mapping[str, int]) -> Subject:
        counts = Counter({s: n for s, n in counts.items() if n > 0})
        formula = hill_formula(counts)
        return cls("mix", formula, "Flask mix", "ของในขวด", counts)

    @classmethod
    def from_formula(cls, formula: str, name: str = "", thai: str = "") -> Subject:
        counts = parse_formula(formula)
        if len(counts) == 1 and sum(counts.values()) == 1:
            return cls.from_element(by_symbol(next(iter(counts))))
        return cls("compound", formula, name or formula, thai, counts)

    @property
    def key(self) -> str:
        """ใช้เทียบ/ค้นข้อมูล: สูตรแบบ Hill (ธาตุ = สัญลักษณ์)"""
        return self.element.symbol if self.element is not None else hill_formula(self.counts)

    @property
    def display(self) -> str:
        return formula_text(self.formula)

    @property
    def molar_mass(self) -> float:
        """g/mol ของหนึ่งหน่วยสูตร — ธาตุที่เป็นโมเลกุลคู่ในธรรมชาติ (H₂, O₂ …) ใช้มวลของโมเลกุลนั้น"""
        mass = sum(by_symbol(s).mass * n for s, n in self.counts.items())
        if self.element is not None and self.element.symbol in DIATOMIC:
            return mass * 2
        return mass

    @property
    def chip(self) -> str:
        """ป้ายสั้นบนชิป: ปฏิกิริยาใช้ชื่อ (สูตรรวมอะตอมของสารตั้งต้นอ่านไม่รู้เรื่อง)"""
        if self.is_reaction and self.compound is not None:
            return (
                self.compound.name
                if len(self.compound.name) <= 22
                else self.compound.name[:21] + "…"
            )
        return self.display

    @property
    def main_symbol(self) -> str:
        """ธาตุที่เป็นตัวแทนสี: ธาตุที่หนักที่สุดในสูตร (น้ำ = O, เกลือ = Cl)"""
        return max(self.counts, key=lambda s: by_symbol(s).number) if self.counts else "H"

    @property
    def is_reaction(self) -> bool:
        return self.compound is not None and self.compound.is_reaction


DIATOMIC = frozenset({"H", "N", "O", "F", "Cl", "Br", "I"})


# --------------------------------------------------------------------------- #
# จุดหลอมเหลว / จุดเดือด / ความหนาแน่น
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PhaseData:
    melt: float | None  # °C — None = ไม่มีข้อมูล
    boil: float | None  # °C
    density: float | None  # g/cm³ ของของแข็ง/ของเหลวที่อุณหภูมิห้อง (แก๊สคำนวณจากมวลโมเลกุล)
    sublimes: bool = False  # ระเหิด (ไม่มีสถานะของเหลวที่ 1 atm) — melt = boil = จุดระเหิด
    never_freezes: bool = False  # ฮีเลียม: ไม่แข็งตัวที่ 1 atm แม้ใกล้ศูนย์สัมบูรณ์
    note: str = ""


def _p(melt, boil, density=None, *, sublimes=False, never_freezes=False, note=""):
    return PhaseData(melt, boil, density, sublimes, never_freezes, note)


# fmt: off
_ELEMENT_DATA: dict[str, PhaseData] = {
    "H": _p(-259.1, -252.9), "He": _p(None, -268.9, never_freezes=True,
                                      note="ฮีเลียมไม่แข็งตัวที่ความดันปกติ"),
    "Li": _p(180.5, 1342, 0.534), "Be": _p(1287, 2469, 1.85), "B": _p(2076, 3927, 2.34),
    "C": _p(3642, 3642, 2.27, sublimes=True, note="แกรไฟต์ระเหิดที่ความดันปกติ"),
    "N": _p(-210.0, -195.8), "O": _p(-218.8, -183.0), "F": _p(-219.7, -188.1),
    "Ne": _p(-248.6, -246.0), "Na": _p(97.8, 883, 0.971), "Mg": _p(650, 1091, 1.74),
    "Al": _p(660.3, 2519, 2.70), "Si": _p(1414, 3265, 2.33), "P": _p(44.2, 280.5, 1.82,
                                                                     note="ฟอสฟอรัสขาว"),
    "S": _p(115.2, 444.6, 2.07), "Cl": _p(-101.5, -34.0), "Ar": _p(-189.3, -185.8),
    "K": _p(63.5, 759, 0.862), "Ca": _p(842, 1484, 1.55), "Ti": _p(1668, 3287, 4.51),
    "Cr": _p(1907, 2671, 7.19), "Mn": _p(1246, 2061, 7.21), "Fe": _p(1538, 2861, 7.87),
    "Co": _p(1495, 2927, 8.90), "Ni": _p(1455, 2913, 8.91), "Cu": _p(1084.6, 2562, 8.96),
    "Zn": _p(419.5, 907, 7.14), "Ga": _p(29.8, 2400, 5.91), "Br": _p(-7.2, 58.8, 3.10),
    "Kr": _p(-157.4, -153.4), "Rb": _p(39.3, 688, 1.53), "Sr": _p(777, 1377, 2.64),
    "Ag": _p(961.8, 2162, 10.49), "Sn": _p(231.9, 2602, 7.29), "I": _p(113.7, 184.3, 4.93),
    "Xe": _p(-111.8, -108.1), "Cs": _p(28.4, 671, 1.93), "W": _p(3422, None, 19.25),
    "Pt": _p(1768.3, 3825, 21.45), "Au": _p(1064.2, None, 19.32), "Hg": _p(-38.8, 356.7, 13.53),
    "Pb": _p(327.5, 1749, 11.34), "U": _p(1132.2, 4131, 19.1), "Os": _p(3033, None, 22.59),
}

_COMPOUND_DATA: dict[str, PhaseData] = {
    "H2O": _p(0.0, 100.0, 1.00),
    "CO2": _p(-78.5, -78.5, sublimes=True, note="น้ำแข็งแห้ง: ระเหิดที่ความดันปกติ"),
    "NaCl": _p(801, 1465, 2.17), "KCl": _p(770, 1420, 1.98), "CH4": _p(-182.5, -161.5),
    "NH3": _p(-77.7, -33.3), "C2H6O": _p(-114.1, 78.4, 0.789), "H2O2": _p(-0.4, 150.2, 1.45),
    "HCl": _p(-114.2, -85.1), "CO": _p(-205.0, -191.5), "SO2": _p(-72.0, -10.0),
    "H2S": _p(-82.0, -60.0), "O3": _p(-192.2, -111.9), "C2H4O2": _p(16.6, 118.1, 1.05),
    "C6H6": _p(5.5, 80.1, 0.876), "NaOH": _p(318, 1388, 2.13), "N2O": _p(-90.9, -88.5),
    "HF": _p(-83.6, 19.5), "CH4O": _p(-97.6, 64.7, 0.792), "C3H8": _p(-187.7, -42.1),
    "H2SO4": _p(10.3, 337, 1.83), "HNO3": _p(-42.0, 83.0, 1.51), "CaO": _p(2613, 2850, 3.34),
    "MgO": _p(2852, 3600, 3.58), "Al2O3": _p(2072, 2977, 3.95), "C3H6O": _p(-94.7, 56.1, 0.784),
    "C2H4": _p(-169.2, -103.7), "C2H2": _p(-84.0, -84.0, sublimes=True,
                                         note="อะเซทิลีนระเหิดที่ความดันปกติ"),
    "CCl4": _p(-22.9, 76.7, 1.59), "CHCl3": _p(-63.5, 61.2, 1.49), "CH2O": _p(-92.0, -19.0),
    "HCN": _p(-13.3, 26.0, 0.69), "PH3": _p(-132.8, -87.7), "NO2": _p(-11.2, 21.2),
    "SF6": _p(-63.8, -63.8, sublimes=True, note="SF₆ ระเหิดที่ความดันปกติ"),
    "N2H4": _p(2.0, 114.0, 1.02), "C6H12O6": _p(146, None, 1.54, note="สลายตัวก่อนเดือด"),
    "SiO2": _p(1713, 2950, 2.65), "Fe2O3": _p(1566, None, 5.24, note="สลายตัวก่อนเดือด"),
    "CaCO3": _p(None, None, 2.71, note="สลายตัวเป็น CaO + CO₂ เมื่อร้อน (~825 °C)"),
    "TiO2": _p(1843, 2972, 4.23), "ZnO": _p(1975, None, 5.61), "AgCl": _p(455, 1547, 5.56),
    "CuO": _p(1326, None, 6.31), "UF6": _p(56.5, 56.5, 5.09, sublimes=True,
                                        note="UF₆ ระเหิดที่ความดันปกติ"),
}
# fmt: on

_BY_KEY: dict[str, PhaseData] = {}
for _formula, _data in _COMPOUND_DATA.items():
    _BY_KEY[hill_formula(parse_formula(_formula))] = _data


def phase_data(subject: Subject) -> PhaseData | None:
    if subject.element is not None:
        return _ELEMENT_DATA.get(subject.element.symbol)
    data = _BY_KEY.get(subject.key)
    if data is None and len(subject.counts) == 1:  # H₂ O₂ N₂ Cl₂ จากขวด = รูปปกติของธาตุนั้น
        symbol, count = next(iter(subject.counts.items()))
        if count == 2 and symbol in DIATOMIC:
            return _ELEMENT_DATA.get(symbol)
    return data


def room_state(subject: Subject) -> str:
    """สถานะที่ 25 °C: ``solid`` | ``liquid`` | ``gas`` | ``unknown``"""
    phase = phase_at(subject, ROOM_C)
    if phase is not None:
        return phase
    if subject.element is not None:
        return atom_info(subject.element).state
    return "unknown"


def phase_at(subject: Subject, celsius: float) -> str | None:
    """สถานะที่อุณหภูมินี้ (1 atm) — ``None`` = ข้อมูลไม่พอจะบอก"""
    data = phase_data(subject)
    if data is None:
        return None
    if data.never_freezes:
        return "gas" if data.boil is not None and celsius >= data.boil else "liquid"
    if data.sublimes and data.melt is not None:
        return "solid" if celsius < data.melt else "gas"
    if data.melt is None:
        return None
    if celsius < data.melt:
        return "solid"
    if data.boil is None:
        return "liquid"  # ไม่รู้จุดเดือด: บอกได้แค่ว่าเหลว (และหน้าจอแจ้งว่าไม่มีข้อมูลจุดเดือด)
    return "liquid" if celsius < data.boil else "gas"


def density(subject: Subject, celsius: float = ROOM_C) -> float | None:
    """g/cm³ — แก๊สคำนวณจากมวลโมเลกุล (กฎแก๊สอุดมคติ) ของแข็ง/ของเหลวจากตาราง"""
    phase = phase_at(subject, celsius) or room_state(subject)
    if phase == "gas":
        litres_per_mol = MOLAR_VOLUME_L * kelvin(celsius) / 273.15
        return subject.molar_mass / litres_per_mol / 1000.0
    data = phase_data(subject)
    return data.density if data is not None else None


# --------------------------------------------------------------------------- #
# อากาศ
# --------------------------------------------------------------------------- #

# (สูตร, % โดยปริมาตรในอากาศแห้ง, สี)
AIR: tuple[tuple[str, float, str], ...] = (
    ("N2", 78.08, "#5B8DEF"),
    ("O2", 20.95, "#E5484D"),
    ("Ar", 0.93, "#C98AC9"),
    ("CO2", 0.04, "#8C939D"),
)
AIR_MOLAR_MASS = sum(
    Subject.from_formula(formula).molar_mass * percent for formula, percent, _ in AIR
) / sum(percent for _formula, percent, _ in AIR)


@dataclass(frozen=True)
class Balloon:
    gas: bool  # เติมลูกโป่งได้ไหม (เป็นแก๊สที่อุณหภูมิห้อง)
    lift: float  # > 0 ลอย, < 0 จม (สัดส่วนต่างของความหนาแน่นเทียบอากาศ)
    ratio: float  # อากาศ ÷ แก๊สนี้ (> 1 = เบากว่าอากาศ)
    verdict: str  # "rises" | "sinks" | "floats" | "not_gas"
    text: str


def balloon_for(subject: Subject) -> Balloon:
    """ลูกโป่งที่เติมสารนี้ลอยหรือจมในอากาศ — เทียบมวลโมเลกุลกับอากาศ (28.96 g/mol) ที่อุณหภูมิ/ความดันเท่ากัน"""
    if room_state(subject) != "gas":
        state = {"solid": "ของแข็ง", "liquid": "ของเหลว"}.get(room_state(subject), "ไม่ใช่แก๊ส")
        return Balloon(
            False,
            -1.0,
            0.0,
            "not_gas",
            f"{subject.name} เป็น{state}ที่อุณหภูมิห้อง — เติมลูกโป่งไม่ได้ (ถ้าใส่ไปก็กองอยู่ก้นลูกโป่ง)",
        )
    mass = subject.molar_mass
    ratio = AIR_MOLAR_MASS / mass
    lift = (AIR_MOLAR_MASS - mass) / AIR_MOLAR_MASS
    if abs(lift) < 0.04:
        verdict, text = (
            "floats",
            f"หนักใกล้เคียงอากาศ ({mass:.1f} vs {AIR_MOLAR_MASS:.1f} g/mol) — ลอยนิ่งๆ",
        )
    elif lift > 0:
        verdict, text = "rises", f"เบากว่าอากาศ {ratio:.1f} เท่า ({mass:.1f} g/mol) — ลูกโป่งลอยขึ้น"
    else:
        verdict = "sinks"
        text = f"หนักกว่าอากาศ {1 / ratio:.1f} เท่า ({mass:.1f} g/mol) — ลูกโป่งจมลงพื้น"
    return Balloon(True, lift, ratio, verdict, text)


def diffusion_speed(subject: Subject) -> float:
    """ความเร็วแพร่เทียบอากาศ (กฎของเกรแฮม: ∝ 1/√M)"""
    return math.sqrt(AIR_MOLAR_MASS / subject.molar_mass)


# --------------------------------------------------------------------------- #
# แรงโน้มถ่วง
# --------------------------------------------------------------------------- #

# (ชื่อ, ชื่อไทย, g เป็น m/s²)
PLANETS: tuple[tuple[str, str, float], ...] = (
    ("Pluto", "ดาวพลูโต", 0.62),
    ("Moon", "ดวงจันทร์", 1.62),
    ("Mars", "ดาวอังคาร", 3.71),
    ("Venus", "ดาวศุกร์", 8.87),
    ("Earth", "โลก", 9.81),
    ("Saturn", "ดาวเสาร์", 10.44),
    ("Jupiter", "ดาวพฤหัสบดี", 24.79),
    ("Sun", "ดวงอาทิตย์ (พื้นผิว)", 274.0),
)
EARTH_G = 9.81
CUBE_CM = 10.0  # ก้อนทดลอง 10 × 10 × 10 ซม. = 1 ลิตร


def cube_mass_kg(subject: Subject) -> float | None:
    """มวลของก้อน 1 ลิตร (แก๊ส = กล่อง 1 ลิตรที่บรรจุแก๊สนี้)"""
    rho = density(subject)
    return None if rho is None else rho * CUBE_CM**3 / 1000.0


def weight_newton(mass_kg: float, g: float) -> float:
    return mass_kg * g


def fall_time(height_m: float, g: float) -> float:
    """เวลาตกอิสระในสุญญากาศ — ไม่ขึ้นกับมวล (กาลิเลโอ)"""
    return math.sqrt(2.0 * height_m / g)
