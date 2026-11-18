"""ห้องทดลองเคมี — ตารางธาตุ 118 ธาตุ, สารประกอบที่รู้จัก, ขวดผสม และตัวควบคุมด้วยมือ (pure, ไม่มี Qt)

- :data:`ELEMENTS` — เลขอะตอม สัญลักษณ์ ชื่อ มวลอะตอม หมวด และตำแหน่งในตาราง (คาบ/หมู่ คำนวณจากเลขอะตอม
  จึงไม่มีพิมพ์ผิดทีละช่อง) มวลในวงเล็บเหลี่ยม = เลขมวลของไอโซโทปที่เสถียรที่สุด (ธาตุกัมมันตรังสี)
- :data:`COMPOUNDS` — สารที่ผสมได้ เขียนเป็นสูตรเคมีปกติ (เช่น ``"Ca(OH)2"``) แล้วแยกนับอะตอมด้วย
  :func:`parse_formula` — ผสมอะตอมในขวดให้ครบพอดี (หรือเป็นจำนวนเท่า) = เกิดสารนั้น
- :class:`Flask` — ขวดผสม: เพิ่ม/เอาออก/เท นับอะตอม หาสารที่ตรง และคำใบ้ว่าขาดอะไร
- :class:`ChemLab` — มือ → หยิบธาตุจากตาราง ลากไปปล่อยในขวด + ท่าค้าง (✌ เอาออก, ✊ เท, 👍 เปิด/ปิดตาราง)
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from .input import HoldDetector, StudioHand

__all__ = [
    "CATEGORIES",
    "COMPOUNDS",
    "ELEMENTS",
    "MAX_ATOMS",
    "ChemEvent",
    "ChemLab",
    "Compound",
    "Element",
    "Flask",
    "by_symbol",
    "formula_text",
    "hill_formula",
    "parse_formula",
]

MAX_ATOMS = 24  # อะตอมในขวดได้สูงสุด

# หมวด: (ชื่ออังกฤษ, ชื่อไทย, สี)
CATEGORIES: dict[str, tuple[str, str, str]] = {
    "alkali": ("Alkali metal", "โลหะแอลคาไล", "#E5795E"),
    "alkaline": ("Alkaline earth metal", "โลหะแอลคาไลน์เอิร์ท", "#E5A65E"),
    "transition": ("Transition metal", "โลหะทรานซิชัน", "#6C8EBF"),
    "post": ("Post-transition metal", "โลหะหลังทรานซิชัน", "#7FA7A0"),
    "metalloid": ("Metalloid", "กึ่งโลหะ", "#9C8FD6"),
    "nonmetal": ("Nonmetal", "อโลหะ", "#7CC4A0"),
    "halogen": ("Halogen", "แฮโลเจน", "#5DB3C9"),
    "noble": ("Noble gas", "แก๊สมีตระกูล", "#C98AC9"),
    "lanthanide": ("Lanthanide", "แลนทาไนด์", "#C9B37A"),
    "actinide": ("Actinide", "แอกทิไนด์", "#C98F6B"),
}

_CATEGORY_OF: dict[int, str] = {}
for _numbers, _category in (
    ((3, 11, 19, 37, 55, 87), "alkali"),
    ((4, 12, 20, 38, 56, 88), "alkaline"),
    ((5, 14, 32, 33, 51, 52), "metalloid"),
    ((1, 6, 7, 8, 15, 16, 34), "nonmetal"),
    ((9, 17, 35, 53, 85, 117), "halogen"),
    ((2, 10, 18, 36, 54, 86, 118), "noble"),
    ((13, 31, 49, 50, 81, 82, 83, 84, 113, 114, 115, 116), "post"),
    (tuple(range(57, 72)), "lanthanide"),
    (tuple(range(89, 104)), "actinide"),
):
    for _n in _numbers:
        _CATEGORY_OF[_n] = _category

# เลขอะตอม สัญลักษณ์ ชื่อ มวล (* = เลขมวลของไอโซโทปที่เสถียรที่สุด)
_TABLE = """
1 H Hydrogen 1.008|2 He Helium 4.0026|3 Li Lithium 6.94|4 Be Beryllium 9.0122|5 B Boron 10.81
6 C Carbon 12.011|7 N Nitrogen 14.007|8 O Oxygen 15.999|9 F Fluorine 18.998|10 Ne Neon 20.180
11 Na Sodium 22.990|12 Mg Magnesium 24.305|13 Al Aluminium 26.982|14 Si Silicon 28.085
15 P Phosphorus 30.974|16 S Sulfur 32.06|17 Cl Chlorine 35.45|18 Ar Argon 39.948
19 K Potassium 39.098|20 Ca Calcium 40.078|21 Sc Scandium 44.956|22 Ti Titanium 47.867
23 V Vanadium 50.942|24 Cr Chromium 51.996|25 Mn Manganese 54.938|26 Fe Iron 55.845
27 Co Cobalt 58.933|28 Ni Nickel 58.693|29 Cu Copper 63.546|30 Zn Zinc 65.38
31 Ga Gallium 69.723|32 Ge Germanium 72.630|33 As Arsenic 74.922|34 Se Selenium 78.971
35 Br Bromine 79.904|36 Kr Krypton 83.798|37 Rb Rubidium 85.468|38 Sr Strontium 87.62
39 Y Yttrium 88.906|40 Zr Zirconium 91.224|41 Nb Niobium 92.906|42 Mo Molybdenum 95.95
43 Tc Technetium 98*|44 Ru Ruthenium 101.07|45 Rh Rhodium 102.91|46 Pd Palladium 106.42
47 Ag Silver 107.87|48 Cd Cadmium 112.41|49 In Indium 114.82|50 Sn Tin 118.71
51 Sb Antimony 121.76|52 Te Tellurium 127.60|53 I Iodine 126.90|54 Xe Xenon 131.29
55 Cs Caesium 132.91|56 Ba Barium 137.33|57 La Lanthanum 138.91|58 Ce Cerium 140.12
59 Pr Praseodymium 140.91|60 Nd Neodymium 144.24|61 Pm Promethium 145*|62 Sm Samarium 150.36
63 Eu Europium 151.96|64 Gd Gadolinium 157.25|65 Tb Terbium 158.93|66 Dy Dysprosium 162.50
67 Ho Holmium 164.93|68 Er Erbium 167.26|69 Tm Thulium 168.93|70 Yb Ytterbium 173.05
71 Lu Lutetium 174.97|72 Hf Hafnium 178.49|73 Ta Tantalum 180.95|74 W Tungsten 183.84
75 Re Rhenium 186.21|76 Os Osmium 190.23|77 Ir Iridium 192.22|78 Pt Platinum 195.08
79 Au Gold 196.97|80 Hg Mercury 200.59|81 Tl Thallium 204.38|82 Pb Lead 207.2
83 Bi Bismuth 208.98|84 Po Polonium 209*|85 At Astatine 210*|86 Rn Radon 222*
87 Fr Francium 223*|88 Ra Radium 226*|89 Ac Actinium 227*|90 Th Thorium 232.04
91 Pa Protactinium 231.04|92 U Uranium 238.03|93 Np Neptunium 237*|94 Pu Plutonium 244*
95 Am Americium 243*|96 Cm Curium 247*|97 Bk Berkelium 247*|98 Cf Californium 251*
99 Es Einsteinium 252*|100 Fm Fermium 257*|101 Md Mendelevium 258*|102 No Nobelium 259*
103 Lr Lawrencium 266*|104 Rf Rutherfordium 267*|105 Db Dubnium 268*|106 Sg Seaborgium 269*
107 Bh Bohrium 270*|108 Hs Hassium 269*|109 Mt Meitnerium 278*|110 Ds Darmstadtium 281*
111 Rg Roentgenium 282*|112 Cn Copernicium 285*|113 Nh Nihonium 286*|114 Fl Flerovium 289*
115 Mc Moscovium 290*|116 Lv Livermorium 293*|117 Ts Tennessine 294*|118 Og Oganesson 294*
"""

# ชื่อไทยของธาตุที่พบบ่อย (ธาตุอื่นแสดงชื่ออังกฤษ)
THAI_NAMES = {
    "H": "ไฮโดรเจน",
    "He": "ฮีเลียม",
    "Li": "ลิเทียม",
    "C": "คาร์บอน",
    "N": "ไนโตรเจน",
    "O": "ออกซิเจน",
    "F": "ฟลูออรีน",
    "Ne": "นีออน",
    "Na": "โซเดียม",
    "Mg": "แมกนีเซียม",
    "Al": "อะลูมิเนียม",
    "Si": "ซิลิคอน",
    "P": "ฟอสฟอรัส",
    "S": "กำมะถัน",
    "Cl": "คลอรีน",
    "Ar": "อาร์กอน",
    "K": "โพแทสเซียม",
    "Ca": "แคลเซียม",
    "Ti": "ไทเทเนียม",
    "Fe": "เหล็ก",
    "Cu": "ทองแดง",
    "Zn": "สังกะสี",
    "Br": "โบรมีน",
    "Ag": "เงิน",
    "Sn": "ดีบุก",
    "I": "ไอโอดีน",
    "Xe": "ซีนอน",
    "Pt": "แพลทินัม",
    "Au": "ทองคำ",
    "Hg": "ปรอท",
    "Pb": "ตะกั่ว",
    "U": "ยูเรเนียม",
}

# สีของอะตอมในขวด (แนว CPK ปรับให้เห็นบนพื้นมืด) — ธาตุอื่นใช้สีของหมวด
ATOM_COLORS = {
    "H": "#F2F4F7",
    "C": "#8C939D",
    "N": "#5B8DEF",
    "O": "#E5484D",
    "F": "#7CC4A0",
    "Cl": "#7CC4A0",
    "Br": "#B5651D",
    "I": "#9370DB",
    "S": "#E6C84F",
    "P": "#F0A050",
    "Fe": "#D9824B",
    "Cu": "#C87533",
    "Ag": "#C0C6CF",
    "Au": "#D6B67A",
}

_PERIOD_STARTS = (1, 3, 11, 19, 37, 55, 87, 119)


@dataclass(frozen=True, slots=True)
class Element:
    number: int
    symbol: str
    name: str
    mass: float
    stable_mass: bool  # False = มวลเป็นเลขมวลของไอโซโทปที่เสถียรที่สุด (แสดงในวงเล็บเหลี่ยม)
    category: str
    period: int
    group: int | None  # None = แถว f (แลนทาไนด์/แอกทิไนด์)
    row: int  # แถวในผัง 0–6 = คาบ 1–7, 7 = แลนทาไนด์, 8 = แอกทิไนด์
    column: int  # 0–17

    @property
    def thai(self) -> str:
        return THAI_NAMES.get(self.symbol, "")

    @property
    def color(self) -> str:
        return ATOM_COLORS.get(self.symbol, CATEGORIES[self.category][2])

    @property
    def mass_text(self) -> str:
        return f"{self.mass:g}" if self.stable_mass else f"[{self.mass:.0f}]"


def _position(number: int) -> tuple[int, int | None, int, int]:
    """(คาบ, หมู่, แถวในผัง, คอลัมน์) จากเลขอะตอม"""
    period = next(p for p in range(1, 8) if number < _PERIOD_STARTS[p])
    offset = number - _PERIOD_STARTS[period - 1]
    if period == 1:
        group = 1 if number == 1 else 18
    elif period in (2, 3):
        group = offset + 1 if offset < 2 else offset + 11
    elif period in (4, 5) or offset < 2:  # คาบ 4–5 ทั้งแถว / คาบ 6–7 หมู่ 1–2
        group = offset + 1
    elif offset < 17:  # แถว f (57–71, 89–103)
        row = 7 if period == 6 else 8
        return period, None, row, offset - 2 + 2  # เริ่มคอลัมน์ 2 (ใต้หมู่ 3)
    else:  # คาบ 6–7 หลังแถว f: Hf/Rf = หมู่ 4
        group = offset - 13
    return period, group, period - 1, group - 1


def _category(number: int, group: int | None) -> str:
    if number in _CATEGORY_OF:
        return _CATEGORY_OF[number]
    return "transition" if group is not None and 3 <= group <= 12 else "post"


def _parse_table() -> tuple[Element, ...]:
    elements = []
    for entry in re.split(r"[|\n]", _TABLE.strip()):
        number, symbol, name, mass = entry.split()
        stable = not mass.endswith("*")
        period, group, row, column = _position(int(number))
        elements.append(
            Element(
                int(number),
                symbol,
                name,
                float(mass.rstrip("*")),
                stable,
                _category(int(number), group),
                period,
                group,
                row,
                column,
            )
        )
    return tuple(elements)


ELEMENTS: tuple[Element, ...] = _parse_table()
_BY_SYMBOL = {e.symbol: e for e in ELEMENTS}


def by_symbol(symbol: str) -> Element:
    return _BY_SYMBOL[symbol]


# --------------------------------------------------------------------------- #
# สูตรเคมี
# --------------------------------------------------------------------------- #

_TOKEN = re.compile(r"([A-Z][a-z]?)(\d*)|(\()|(\))(\d*)")
_SUBSCRIPT = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")


def parse_formula(formula: str) -> Counter[str]:
    """``"Ca(OH)2"`` → ``{"Ca": 1, "O": 2, "H": 2}`` (ValueError ถ้ามีธาตุที่ไม่รู้จักหรือวงเล็บไม่ครบ)"""
    stack: list[Counter[str]] = [Counter()]
    position = 0
    while position < len(formula):
        match = _TOKEN.match(formula, position)
        if match is None:
            raise ValueError(f"อ่านสูตร {formula!r} ไม่ได้ที่ตำแหน่ง {position}")
        symbol, count, opening, closing, multiplier = match.groups()
        if symbol:
            if symbol not in _BY_SYMBOL:
                raise ValueError(f"ไม่รู้จักธาตุ {symbol!r} ใน {formula!r}")
            stack[-1][symbol] += int(count or 1)
        elif opening:
            stack.append(Counter())
        elif closing:
            if len(stack) == 1:
                raise ValueError(f"วงเล็บปิดเกินใน {formula!r}")
            group = stack.pop()
            for key, value in group.items():
                stack[-1][key] += value * int(multiplier or 1)
        position = match.end()
    if len(stack) != 1:
        raise ValueError(f"วงเล็บไม่ครบใน {formula!r}")
    return stack[0]


def formula_text(formula: str) -> str:
    """ตัวเลขห้อยแบบ unicode: ``"H2O"`` → ``"H₂O"``"""
    return formula.translate(_SUBSCRIPT)


def hill_formula(counts: Mapping[str, int]) -> str:
    """สูตรแบบ Hill (C, H แล้วตามตัวอักษร ถ้าไม่มี C เรียงตามตัวอักษรทั้งหมด) สำหรับของในขวดที่ยังไม่ใช่สารใด"""
    present = {k: v for k, v in counts.items() if v > 0}
    if "C" in present:
        order = ["C", *(["H"] if "H" in present else []), *sorted(set(present) - {"C", "H"})]
    else:
        order = sorted(present)
    return "".join(f"{s}{present[s] if present[s] > 1 else ''}" for s in order)


def molar_mass(counts: Mapping[str, int]) -> float:
    return sum(by_symbol(s).mass * n for s, n in counts.items())


@dataclass(frozen=True, slots=True)
class Compound:
    formula: str  # เขียนแบบที่คนคุ้น (CH3COOH, Ca(OH)2)
    name: str
    thai: str
    fact: str
    color: str
    energetic: bool = False  # ปฏิกิริยารุนแรง/ปล่อยพลังงานมาก (HUD วาบ)

    @property
    def counts(self) -> Counter[str]:
        return parse_formula(self.formula)

    @property
    def display(self) -> str:
        return formula_text(self.formula)

    @property
    def molar_mass(self) -> float:
        return molar_mass(self.counts)

    @property
    def atoms(self) -> int:
        return sum(self.counts.values())


COMPOUNDS: tuple[Compound, ...] = (
    Compound(
        "H2",
        "Hydrogen gas",
        "ก๊าซไฮโดรเจน",
        "เบาที่สุดในจักรวาล ติดไฟง่าย ใช้เป็นเชื้อเพลิงจรวด",
        "#F2F4F7",
        True,
    ),
    Compound("O2", "Oxygen gas", "ก๊าซออกซิเจน", "ราว 21% ของอากาศที่เราหายใจ", "#9FD3FF"),
    Compound("N2", "Nitrogen gas", "ก๊าซไนโตรเจน", "ราว 78% ของอากาศ เฉื่อยมาก", "#C9D1DB"),
    Compound(
        "Cl2", "Chlorine gas", "ก๊าซคลอรีน", "ก๊าซสีเหลืองแกมเขียว เป็นพิษ ใช้ฆ่าเชื้อในน้ำประปา", "#C8E07A"
    ),
    Compound("O3", "Ozone", "โอโซน", "ชั้นโอโซนช่วยกรองรังสียูวีจากดวงอาทิตย์", "#8EC8FF"),
    Compound("H2O", "Water", "น้ำ", "ไฮโดรเจนกับออกซิเจนรวมกันแล้วปล่อยพลังงานสูงมาก", "#5B9BD5", True),
    Compound(
        "H2O2",
        "Hydrogen peroxide",
        "ไฮโดรเจนเปอร์ออกไซด์",
        "ใช้ล้างแผลและฟอกสี สลายตัวเป็นน้ำกับออกซิเจน",
        "#A9D4F5",
    ),
    Compound(
        "CO2", "Carbon dioxide", "คาร์บอนไดออกไซด์", "ฟองในน้ำอัดลม และก๊าซเรือนกระจกตัวหลัก", "#C9D1DB"
    ),
    Compound(
        "CO", "Carbon monoxide", "คาร์บอนมอนอกไซด์", "ก๊าซพิษไร้สีไร้กลิ่น เกิดจากการเผาไหม้ไม่สมบูรณ์", "#AAB2BC"
    ),
    Compound("CH4", "Methane", "มีเทน", "ส่วนประกอบหลักของก๊าซธรรมชาติ", "#B7E3C8", True),
    Compound("C2H2", "Acetylene", "อะเซทิลีน", "ก๊าซเชื่อมโลหะ เปลวไฟร้อนกว่า 3,000 °C", "#E8D9A8", True),
    Compound("C2H4", "Ethylene", "เอทิลีน", "ฮอร์โมนพืชที่ทำให้ผลไม้สุก", "#D8EFC0"),
    Compound("C3H8", "Propane", "โพรเพน", "ส่วนหนึ่งของก๊าซหุงต้ม (LPG)", "#E0E8F0", True),
    Compound("C6H6", "Benzene", "เบนซีน", "โมเลกุลรูปวงแหวนหกเหลี่ยม เป็นสารก่อมะเร็ง", "#EDE3C8"),
    Compound("CH3OH", "Methanol", "เมทานอล", "แอลกอฮอล์ที่เป็นพิษ ดื่มแล้วอาจตาบอดได้", "#D5E6F5"),
    Compound("C2H5OH", "Ethanol", "เอทานอล", "แอลกอฮอล์ในเครื่องดื่มและเจลล้างมือ", "#E6F0FA"),
    Compound("CH3COOH", "Acetic acid", "กรดแอซีติก", "ทำให้น้ำส้มสายชูมีรสเปรี้ยว", "#F5EBD0"),
    Compound("CH2O", "Formaldehyde", "ฟอร์มาลดีไฮด์", "ละลายน้ำแล้วคือฟอร์มาลิน ใช้ดองตัวอย่าง", "#E6E6E6"),
    Compound("NH3", "Ammonia", "แอมโมเนีย", "กลิ่นฉุน เป็นวัตถุดิบหลักของปุ๋ย", "#D9ECFF"),
    Compound("N2O", "Nitrous oxide", "ไนตรัสออกไซด์", "ก๊าซหัวเราะ ใช้เป็นยาสลบอ่อนๆ", "#E0E7F2"),
    Compound("NO2", "Nitrogen dioxide", "ไนโตรเจนไดออกไซด์", "ก๊าซสีน้ำตาลแดงในหมอกควันพิษ", "#B5651D"),
    Compound("HNO3", "Nitric acid", "กรดไนตริก", "กรดแก่ ใช้ผลิตปุ๋ยและวัตถุระเบิด", "#F2E6B0"),
    Compound("HCl", "Hydrochloric acid", "กรดไฮโดรคลอริก", "กรดในกระเพาะอาหารของเรา", "#E9F2C9"),
    Compound("HF", "Hydrofluoric acid", "กรดไฮโดรฟลูออริก", "กัดแก้วได้ ใช้แกะลายบนแก้ว", "#E2F5E9"),
    Compound("H2S", "Hydrogen sulfide", "ไฮโดรเจนซัลไฟด์", "ก๊าซกลิ่นไข่เน่า เป็นพิษ", "#EFE7A0"),
    Compound(
        "SO2",
        "Sulfur dioxide",
        "ซัลเฟอร์ไดออกไซด์",
        "ออกมาจากภูเขาไฟและการเผาถ่านหิน ทำให้เกิดฝนกรด",
        "#E6DFA0",
    ),
    Compound("H2SO4", "Sulfuric acid", "กรดซัลฟิวริก", "น้ำกรดในแบตเตอรี่รถยนต์", "#F0E8C0"),
    Compound(
        "NaCl",
        "Sodium chloride",
        "เกลือแกง",
        "โลหะที่ระเบิดเมื่อโดนน้ำ + ก๊าซพิษ = เกลือที่เรากิน",
        "#F2F4F7",
        True,
    ),
    Compound("KCl", "Potassium chloride", "โพแทสเซียมคลอไรด์", "ใช้แทนเกลือในเกลือโซเดียมต่ำ", "#F0F0F5"),
    Compound("NaF", "Sodium fluoride", "โซเดียมฟลูออไรด์", "ฟลูออไรด์ในยาสีฟันช่วยป้องกันฟันผุ", "#EEF6F0"),
    Compound("NaOH", "Sodium hydroxide", "โซดาไฟ", "ด่างแก่ ใช้ทำสบู่และล้างท่อตัน", "#F5F5F5"),
    Compound(
        "NaClO", "Sodium hypochlorite", "โซเดียมไฮโปคลอไรต์", "สารฟอกขาวในน้ำยาซักผ้าขาว", "#F2F7D9"
    ),
    Compound("NaHCO3", "Sodium bicarbonate", "โซเดียมไบคาร์บอเนต", "เบกกิ้งโซดา ทำให้ขนมฟู", "#FAFAFA"),
    Compound("CaCO3", "Calcium carbonate", "แคลเซียมคาร์บอเนต", "หินปูน ชอล์ก และเปลือกหอย", "#F2EEE6"),
    Compound("CaO", "Calcium oxide", "ปูนดิบ", "เติมน้ำแล้วร้อนจัด กลายเป็นปูนสุก", "#F5F2EA", True),
    Compound("Ca(OH)2", "Calcium hydroxide", "ปูนสุก", "ใช้ปรับดินเปรี้ยวและทำปูนฉาบ", "#F7F5EF"),
    Compound(
        "CaCl2", "Calcium chloride", "แคลเซียมคลอไรด์", "สารดูดความชื้นในกล่องกันชื้นบางแบบ", "#F5F5F5"
    ),
    Compound(
        "MgO",
        "Magnesium oxide",
        "แมกนีเซียมออกไซด์",
        "เกิดตอนเผาแมกนีเซียม ให้แสงขาวจ้าแบบพลุ",
        "#FFFFFF",
        True,
    ),
    Compound(
        "Al2O3",
        "Aluminium oxide",
        "อะลูมิเนียมออกไซด์",
        "ผลึกของมันคือทับทิมและไพลิน (เมื่อมีโลหะอื่นเจือ)",
        "#E8C7D0",
    ),
    Compound("SiO2", "Silicon dioxide", "ซิลิคอนไดออกไซด์", "ทราย ควอตซ์ และวัตถุดิบทำแก้ว", "#EDE6D6"),
    Compound(
        "Fe2O3", "Iron(III) oxide", "สนิมเหล็ก", "สนิมสีน้ำตาลแดงที่เกิดเมื่อเหล็กเจอน้ำและอากาศ", "#A0522D"
    ),
    Compound("Fe3O4", "Magnetite", "แมกนีไทต์", "แร่เหล็กที่เป็นแม่เหล็กได้เองตามธรรมชาติ", "#5A5A5A"),
    Compound(
        "CuO", "Copper(II) oxide", "คอปเปอร์(II) ออกไซด์", "ผงสีดำที่เกิดเมื่อเผาทองแดงในอากาศ", "#4A4A4A"
    ),
    Compound("CuSO4", "Copper(II) sulfate", "จุนสี", "ผลึกสีฟ้าสด (เมื่อมีน้ำผลึก)", "#2F80ED"),
    Compound("ZnO", "Zinc oxide", "ซิงค์ออกไซด์", "ใช้ในครีมกันแดดและแป้งเด็ก", "#FFFFFF"),
    Compound("TiO2", "Titanium dioxide", "ไทเทเนียมไดออกไซด์", "สีขาวในสีทาบ้านและครีมกันแดด", "#FFFFFF"),
    Compound(
        "AgCl",
        "Silver chloride",
        "ซิลเวอร์คลอไรด์",
        "ตะกอนขาวที่คล้ำลงเมื่อโดนแสง ใช้ในฟิล์มถ่ายรูปยุคแรก",
        "#E8E8E8",
    ),
    Compound("KNO3", "Potassium nitrate", "ดินประสิว", "ส่วนผสมหลักของดินปืน", "#F5F5F0", True),
    Compound(
        "NH4Cl", "Ammonium chloride", "แอมโมเนียมคลอไรด์", "ใช้ในถ่านไฟฉายแบบเก่าและงานบัดกรี", "#F5F5F5"
    ),
    Compound(
        "XeF4", "Xenon tetrafluoride", "ซีนอนเตตระฟลูออไรด์", "พิสูจน์ว่าแก๊สมีตระกูลก็ทำปฏิกิริยาได้", "#E8F4FF"
    ),
)


# --------------------------------------------------------------------------- #
# ขวดผสม
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class Hint:
    compound: Compound
    missing: Counter[str]

    @property
    def text(self) -> str:
        parts = [f"{s}{'×' + str(n) if n > 1 else ''}" for s, n in sorted(self.missing.items())]
        return f"+{' +'.join(parts)} → {self.compound.display}"


class Flask:
    def __init__(self, capacity: int = MAX_ATOMS) -> None:
        self.capacity = capacity
        self.atoms: list[str] = []  # ตามลำดับที่ใส่ (ไว้เอาออกทีละตัว)

    @property
    def counts(self) -> Counter[str]:
        return Counter(self.atoms)

    @property
    def full(self) -> bool:
        return len(self.atoms) >= self.capacity

    def add(self, symbol: str) -> bool:
        if symbol not in _BY_SYMBOL or self.full:
            return False
        self.atoms.append(symbol)
        return True

    def undo(self) -> str | None:
        return self.atoms.pop() if self.atoms else None

    def clear(self) -> None:
        self.atoms.clear()

    def formula(self) -> str:
        return formula_text(hill_formula(self.counts))

    def match(self) -> tuple[Compound, int] | None:
        """สารที่ของในขวดตรงพอดี (หรือเป็นจำนวนเท่า เช่น H×4 O×2 = 2 H₂O)"""
        counts = self.counts
        if not counts:
            return None
        for compound in COMPOUNDS:
            target = compound.counts
            if set(target) != set(counts):
                continue
            multiples = {counts[s] / target[s] for s in target}
            if len(multiples) == 1:
                multiple = multiples.pop()
                if multiple == int(multiple):
                    return compound, int(multiple)
        return None

    def hints(self, limit: int = 3) -> list[Hint]:
        """สารที่ยังขาดอีกไม่กี่อะตอม (ของในขวดเป็นส่วนหนึ่งของสูตร) เรียงจากขาดน้อยสุด"""
        counts = self.counts
        if not counts:
            return []
        hints = []
        for compound in COMPOUNDS:
            target = compound.counts
            if all(counts[s] <= target.get(s, 0) for s in counts) and counts != target:
                hints.append(Hint(compound, target - counts))
        hints.sort(key=lambda h: (sum(h.missing.values()), h.compound.atoms, h.compound.name))
        return hints[:limit]


# --------------------------------------------------------------------------- #
# มือ → หยิบ ลาก ปล่อย
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class ChemEvent:
    kind: str  # "pick" | "add" | "drop" | "full" | "match" | "undo" | "empty" | "table"
    symbol: str = ""
    compound: Compound | None = None
    multiple: int = 1


class ChemLab:
    """มือหลายข้าง (หน่วยผืนผ้าใบ 0–1) → หยิบธาตุจากตาราง ลากไปปล่อยในขวด

    ตำแหน่งของตาราง/ขวดมาจากหน้าจอผ่าน ``cell_at(x, y)`` และ ``in_flask(x, y)`` (logic จึงไม่รู้จัก Qt)
    """

    def __init__(self) -> None:
        self.flask = Flask()
        self.held: dict[str, str] = {}  # มือ → สัญลักษณ์ธาตุที่ถืออยู่
        self.hover: Element | None = None
        self.undo_hold = HoldDetector(0.7)
        self.empty_hold = HoldDetector(1.0)
        self.table_hold = HoldDetector(0.8)
        self.last_match: tuple[Compound, int] | None = None
        self._pinched: dict[str, bool] = {}

    def update(
        self,
        hands: tuple[StudioHand, ...],
        now: float,
        *,
        cell_at: Callable[[float, float], Element | None],
        in_flask: Callable[[float, float], bool],
    ) -> list[ChemEvent]:
        events: list[ChemEvent] = []
        before = list(self.flask.atoms)
        present = set()
        self.hover = None
        for hand in hands:
            name = hand.handedness
            present.add(name)
            was = self._pinched.get(name, False)
            under = cell_at(hand.x, hand.y)
            if self.hover is None and name not in self.held:
                self.hover = under
            if hand.pinch and not was and under is not None:
                self.held[name] = under.symbol
                events.append(ChemEvent("pick", under.symbol))
            elif not hand.pinch and was and name in self.held:
                symbol = self.held.pop(name)
                if in_flask(hand.x, hand.y):
                    added = self.flask.add(symbol)
                    events.append(ChemEvent("add" if added else "full", symbol))
                else:
                    events.append(ChemEvent("drop", symbol))
            self._pinched[name] = hand.pinch
        for name in list(self._pinched):
            if name not in present:  # มือหายระหว่างถือ: ทิ้งอะตอม
                self._pinched.pop(name)
                if name in self.held:
                    events.append(ChemEvent("drop", self.held.pop(name)))

        main = hands[0] if hands else None
        idle = main is not None and not main.pinch and not self.held
        if self.undo_hold.update(idle and main.gesture == "Victory", now):
            symbol = self.flask.undo()
            if symbol:
                events.append(ChemEvent("undo", symbol))
        if self.empty_hold.update(idle and main.gesture == "Closed_Fist", now) and self.flask.atoms:
            self.flask.clear()
            events.append(ChemEvent("empty"))
        if self.table_hold.update(idle and main.gesture == "Thumb_Up", now):
            events.append(ChemEvent("table"))
        if self.flask.atoms != before:
            events.extend(self.check())
        return events

    def check(self) -> list[ChemEvent]:
        """หลังของในขวดเปลี่ยน: เกิดสารใหม่หรือไม่"""
        found = self.flask.match()
        changed = found != self.last_match
        self.last_match = found
        if found is not None and changed:
            return [ChemEvent("match", compound=found[0], multiple=found[1])]
        return []

    @property
    def hold_progress(self) -> float:
        return max(self.undo_hold.progress, self.empty_hold.progress, self.table_hold.progress)
